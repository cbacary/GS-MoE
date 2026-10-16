from collections import OrderedDict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

dtype_p = torch.float64
d1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import time

import math


def _free_bytes(dev):
    if dev.type != "cuda":
        return 8 << 30
    free, _ = torch.cuda.mem_get_info(dev)
    # blocks the caching allocator already holds are free to us
    return free + torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)


@torch.no_grad()
def get_shared_experts_fast(
    adapter,
    n_shared_projs,
    layer_idx,
    cov_x,
    cov_mlp_fact,
    cov_mlp_nys,
    leverage_scores,
    rank,
    beta=1.0,
    n_iter=3,
    fixed_size=True,
    r_min=8,
    gamma=1.0,
    floor_frac=0.35,
    ceil_frac=2.5,
    vram_gb=None,
    cache_gb=None,
    reserve_frac=0.85,
    nys_ranks=None,
    experts_mlp=None,
    disable_down=False,
    log_iters=False,
    dtype_gain=None,
    probe_rank=None,
):

    print(f"gamma = {gamma}, beta = {beta}")
    include_down = not disable_down
    d, E, n = adapter.d_model, adapter.n_experts_in_layer(layer_idx), n_shared_projs
    lo = 2
    hi = int(min(E, 1.5 * -(-E // n)))
    BIG = 1e9
    toks = torch.as_tensor(adapter.metrics["tokens_per_exp"][layer_idx], dtype=dtype_p, device=d1)
    toks_h = [float(t) for t in toks]

    # w = torch.as_tensor([t / sum(toks) for t in toks], dtype=dtype_p, device=d1)
    w = toks / toks.sum()
    gw = w * (E / w.sum())

    if n <= 1:
        raise AttributeError("1 < n_shared_projs < E")
    elif n >= E:
        raise AttributeError("1 < n_shared_projs < E")

    dtype_gain = dtype_p if dtype_gain is None else dtype_gain
    rk = int(rank if probe_rank is None else max(1, min(int(probe_rank), int(rank))))
    eye_d = torch.eye(d, dtype=dtype_p, device=d1)

    tim = time.time()

    Sig = torch.empty(E, d, d, dtype=dtype_p, device=d1)
    # Sig_ridged[e] = Sig[e] + ridge[e] * I is only needed by solve/_chol; it is
    # built per slice by sig_r() instead of being stored as a second (E, d, d).
    ridge = torch.empty(E, dtype=dtype_p, device=d1)

    k_max = max(experts_mlp[e].up_proj.shape[0] for e in range(E))
    print(k_max)
    if k_max < d:
        # weights are the compact representation; the grams W^T W are built per
        # slice by gram() on demand instead of being stored as two (E, d, d).
        Wu = torch.zeros(E, k_max, d, dtype=dtype_gain, device=d1)
        Wg = torch.zeros(E, k_max, d, dtype=dtype_gain, device=d1)
        S_ups = S_gates = None
    else:
        Wu = Wg = None
        S_ups = torch.empty(E, d, d, dtype=dtype_p, device=d1)
        S_gates = torch.empty(E, d, d, dtype=dtype_p, device=d1)

    A = torch.empty(E, d, d, dtype=dtype_p, device=d1) if include_down else None

    en_comp = torch.zeros(E, dtype=dtype_p, device=d1)
    en_orig = torch.zeros(E, dtype=dtype_p, device=d1)
    up_e = torch.zeros(E, dtype=dtype_p, device=d1)
    gate_e = torch.zeros(E, dtype=dtype_p, device=d1)

    # traces = []
    for e in range(E):
        if experts_mlp and experts_mlp[e]:
            m = experts_mlp[e]
        else:
            m = adapter.get_mlp_tensors(layer_idx=layer_idx, expert_idx=e)
        W_up = m.up_proj.to(device=d1, dtype=dtype_p)
        W_g = m.gate_proj.to(device=d1, dtype=dtype_p)

        if S_ups is not None:
            S_ups[e] = W_up.T @ W_up
            S_gates[e] = W_g.T @ W_g
        else:
            Wu[e, : W_up.shape[0]] = W_up.to(dtype=dtype_gain)
            Wg[e, : W_g.shape[0]] = W_g.to(dtype=dtype_gain)

        Sig[e] = cov_x[layer_idx][e]
        ridge[e] = adapter.config.Ex_ridge * torch.trace(Sig[e]) / d
        if include_down:
            A[e] = cov_mlp_fact[layer_idx][e].to(dtype=dtype_p, device=d1)
        # traces.append(torch.trace(cov_x[layer_idx][e]))

        # (W^T W * Sig).sum() == tr(W Sig W^T): same energy without the d x d gram
        up_e[e] = ((W_up @ Sig[e]) * W_up).sum()
        gate_e[e] = ((W_g @ Sig[e]) * W_g).sum()

        del W_g, W_up

        if experts_mlp:
            en_comp[e] = up_e[e] + gate_e[e]
            m0 = adapter.get_mlp_tensors(layer_idx=layer_idx, expert_idx=e)
            W0u = m0.up_proj.to(device=d1, dtype=dtype_p)
            W0g = m0.gate_proj.to(device=d1, dtype=dtype_p)
            en_orig[e] = ((W0u @ Sig[e]) * W0u).sum() + ((W0g @ Sig[e]) * W0g).sum()
            del W0u, W0g, m0

    bpm = d * d * torch.finfo(dtype_p).bits // 8
    _bud = (int(vram_gb) << 30) if vram_gb else int(reserve_frac * _free_bytes(d1))
    # +3 d x d per expert in the chunk: gram() and sig_r() are temporaries now, not views
    cb = max(1, min(E, _bud // (15 * bpm)))
    # cb = max(1, min(E, _bud // (bpm)))
    if experts_mlp:
        _tw = torch.tensor([float(x) for x in toks], dtype=dtype_p, device=d1)
        rho = float((_tw * (en_comp / en_orig.clamp_min(1e-30))).sum() / _tw.sum().clamp_min(1e-30))
        rho = min(max(rho, 1e-6), 1.0)
        if include_down and rho < 1.0:
            A.mul_(rho)
            print(f"L{layer_idx} down rescale: rho={rho:.4f}")

    print(f"initilaize took {time.time() - tim}s")

    def gram(sl):
        if S_ups is None:
            U = Wu[sl].to(dtype_p)
            G_ = Wg[sl].to(dtype_p)
            return U.mT @ U, G_.mT @ G_
        return S_ups[sl], S_gates[sl]

    def sig_r(sl):
        return Sig[sl] + ridge[sl, None, None] * eye_d

    def accumulate(labels):
        acc_x = torch.zeros(n, d, d, dtype=dtype_p, device=d1)
        acc_u = torch.zeros_like(acc_x)
        acc_g = torch.zeros_like(acc_x)
        acc_a = torch.zeros_like(acc_x) if include_down else None
        for e in range(E):
            j = labels[e]
            Su_e, Sg_e = gram(e)
            acc_x[j].add_(sig_r(e), alpha=toks[e])
            acc_u[j].add_(Su_e, alpha=toks[e])
            acc_g[j].add_(Sg_e, alpha=toks[e])
            if include_down:
                acc_a[j].add_(A[e], alpha=toks[e])
                # acc_a[j].add_(A[e])
        return acc_x, acc_u, acc_g, acc_a

    def downdate(acc, sl, gmem, labels):
        """We build stats like this to keep fast"""
        es = list(range(sl.start, sl.stop))
        gi = torch.as_tensor([labels[e] for e in es], device=d1)
        keep = [0.0 if len(gmem[labels[e]]) <= 1 else 1.0 for e in es]
        a = torch.tensor([toks_h[e] * k for e, k in zip(es, keep)], dtype=dtype_p, device=d1)[
            :, None, None
        ]
        # a1 = torch.tensor(keep, dtype=dtype_p, device=d1)[:, None, None]
        # Sx = acc[0][gi].sub_(Sig_ridged[sl] * a)
        # S_u = acc[1][gi].sub_(S_ups[sl] * a)
        # S_g = acc[2][gi].sub_(S_gates[sl] * a)
        # # Sa = acc[3][gi].sub_(A[sl] * a1) if include_down else None
        # Sa = acc[3][gi].sub_(A[sl] * a) if include_down else None  # was a1
        Su_sl, Sg_sl = gram(sl)
        Sx = acc[0][gi] - (sig_r(sl) * a)
        S_u = acc[1][gi] - (Su_sl * a)
        S_g = acc[2][gi] - (Sg_sl * a)
        Sa = acc[3][gi] - (A[sl] * a) if include_down else None  # was a1

        return Sx, S_u, S_g, Sa

    def _chol(Sx):
        L, info = torch.linalg.cholesky_ex(Sx)
        bad = info.ne(0)
        if not bool(bad.any()):
            return L
        Sxj = Sx.clone()
        tr = Sx.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30) / d
        eps = 1e-10
        for _ in range(12):
            Sxj[bad] = Sx[bad] + (eps * tr[bad])[:, None, None] * eye_d
            L, info = torch.linalg.cholesky_ex(Sxj)
            bad = info.ne(0)
            if not bool(bad.any()):
                return L
            eps *= 10.0
        Sxj[bad] = tr[bad][:, None, None] * eye_d
        L, _ = torch.linalg.cholesky_ex(Sxj)
        return L

    def solve(Sx, S_u, S_g, Sa, _rk=None):
        if _rk is None:
            _rk = rk
        m = Sx.shape[0]
        L = _chol(Sx)
        Hc = torch.cat([L.mT @ S_u @ L, L.mT @ S_g @ L], 0)
        _, Vc = torch.linalg.eigh(Hc)
        del Hc
        Vc = Vc.flip(-1)[..., :_rk]
        P_up = torch.linalg.solve_triangular(L.mT, Vc[:m], upper=True)
        P_gate = torch.linalg.solve_triangular(L.mT, Vc[m:], upper=True)
        del Vc, L
        Pa = None
        if include_down and Sa is not None:
            _, Va = torch.linalg.eigh(Sa)
            Pa = Va.flip(-1)[..., :_rk]
        return P_up, P_gate, Pa

    @torch.no_grad()
    def prep(sl):
        """Everything that depends only on the expert slice. Cast once per slice."""
        blk = {"H": Sig[sl].to(dtype_gain), "gw": gw[sl].to(dtype_gain)}
        if Wu is not None:
            blk["Wu"] = Wu[sl].to(dtype_gain)
            blk["Wg"] = Wg[sl].to(dtype_gain)
        else:
            Su, Sg = gram(sl)
            blk["Su"] = Su.to(dtype_gain)
            blk["Sg"] = Sg.to(dtype_gain)
        if include_down and A is not None:
            blk["A"] = A[sl].to(dtype_gain)
        return blk

    def gain(sl, P_up, P_gate, Pa, blk=None):

        if blk is None:
            blk = prep(sl)

        H = blk["H"]
        # H = Sig[sl].to(dtype_gain)
        Pu = P_up.to(dtype_gain)
        Pg = P_gate.to(dtype_gain)

        B_up = H @ Pu
        B_gate = H @ Pg
        if "Wu" in blk:
            Yu = blk["Wu"] @ B_up
            Yg = blk["Wg"] @ B_gate

            Ou = Yu
            Og = Yg

        # if Wu is not None:
        # Yu = Wu[sl] @ B_up
        # Yg = Wg[sl] @ B_gate

        else:

            Yu = blk["Su"] @ B_up
            Yg = blk["Sg"] @ B_gate
            # Yu = S_ups[sl].to(dtype_gain) @ B_up
            # Yg = S_gates[sl].to(dtype_gain) @ B_gate

            Ou = B_up
            Og = B_gate

        term_up = (
            # torch.linalg.solve(Pu.mT @ B_up, B_up.mT @ (S_ups[sl].to(dtype_gain) @ B_up))
            torch.linalg.solve(Pu.mT @ B_up, Ou.mT @ Yu)
            .diagonal(dim1=-2, dim2=-1)
            .sum(-1)
        )
        term_gate = (
            torch.linalg.solve(Pg.mT @ B_gate, Og.mT @ Yg).diagonal(dim1=-2, dim2=-1).sum(-1)
        )
        out = term_up + term_gate
        if Pa is not None and "A" in blk:
            Pa_ = Pa.to(dtype_gain)
            out = out + beta * (Pa_ * (blk["A"] @ Pa_)).sum((-2, -1))

        return out * blk["gw"]
        # if Pa is not None:
        #     Pa_ = Pa.to(dtype_gain)
        #     out = out + beta * (Pa_ * (A[sl].to(dtype_gain) @ Pa_)).sum((-2, -1))
        # return out * gw[sl]
        # return out

    # only use the fast path for qwen because the main results were already made for others
    # if you're using this outside of reproduction you can set fast_gain true for all models
    fast_gain = "qwen2" in adapter.arch
    kF = Wu.shape[1] if Wu is not None else d
    tol_w = torch.finfo(dtype_gain).eps ** (1.0 / 3.0)

    def _tr_inv(Gm, Y):
        C, info = torch.linalg.cholesky_ex(Gm)
        t = torch.linalg.solve_triangular(C.mT, Y, upper=True, left=False).square().sum((-2, -1))
        del C
        bad = info.ne(0)
        if bool(bad.any()):
            # not numerically PD: the original LU solve for those entries. The solve holds
            # ~3 more (nb, r, r) copies, so drop C and Y[bad] before it.
            Yb = Y[bad]
            YtY = Yb.mT @ Yb
            del Yb
            t[bad] = torch.linalg.solve(Gm[bad], YtY).diagonal(dim1=-2, dim2=-1).sum(-1)
        return t

    def _factor(e, i):
        """F with F^T F == the up (i=0) / gate (i=1) gram of expert e."""
        if Wu is not None:
            return (Wu, Wg)[i][e]  # zero-padded rows add nothing
        S = (S_ups, S_gates)[i][e].to(dtype_gain)
        C, info = torch.linalg.cholesky_ex(S)
        if int(info) == 0:
            return C.mT
        ev, V = torch.linalg.eigh(S)
        return ev.clamp_min(0.0).sqrt()[:, None] * V.mT

    def _blk_tr(Z, Y, w, P=None):
        """Per width-w column block j of Z, Y: tr((P_j^T Z_j)^-1 Y_j^T Y_j), P_j = Z_j by default."""
        Zb = Z.unflatten(-1, (-1, w)).transpose(0, 1)
        Yb = Y.unflatten(-1, (-1, w)).transpose(0, 1)
        return _tr_inv((Zb if P is None else P).mT @ Zb, Yb)

    @torch.no_grad()
    def gain_matrix(P_up, P_gate, Pa=None):
        """out[e, j] == gain(e, P_up[j], P_gate[j], Pa[j]) for every expert e and every j.

        Same quantity through exact identities (no approximation):
          up/gate: t = tr((P^T Sig P)^-1 P^T Sig S Sig P), S = F^T F. With Sig = L L^T,
            P^T Sig P = Z^T Z and F Sig P = (F L) Z for Z = L^T P = (L^-1)^-T P, a triangular
            solve (half the flops of Sig @ P); the trace is ||Y C^-T||^2 with C C^T = Z^T Z.
            If r > d/2 the complement is cheaper: for U spanning span(P)^perp, span(L^T P) is
            the orthogonal complement of span(L^-1 U), so
            t = tr(S Sig) - tr((U^T Sig^-1 U)^-1 U^T S U) and everything scales with d - r.
          down: tr(Pa^T A Pa) = <A, Pa Pa^T>_F, so the whole E x m block is one
            (E, d^2) @ (d^2, m) GEMM instead of E * m products A_e @ Pa_j.
        """
        m, _, r = P_up.shape
        Ps = (P_up.to(dtype_gain), P_gate.to(dtype_gain))
        comp = d < 2 * r < 2 * d
        Xs, w = Ps, r
        if comp:
            w = d - r
            Xs = tuple(torch.empty(m, d, w, dtype=dtype_gain, device=d1) for _ in Ps)
            for P, U in zip(Ps, Xs):
                for a in range(0, m, cb):
                    U[a : a + cb] = torch.linalg.qr(P[a : a + cb], mode="complete").Q[..., r:]
        # whitening costs ~d^3/3 + 2 kF d^2 per expert and saves m r d^2 / 2
        white = comp or m * r > d // 3 + 2 * kF
        wm = max(w, r)
        per_j = (torch.finfo(dtype_gain).bits // 8) * ((3 * d + 3 * kF) * wm + 3 * wm * wm)
        jb = max(1, min(m, int(reserve_frac * _free_bytes(d1)) // per_j))
        eye_g = eye_d.to(dtype_gain)

        def cat(X, a, b):  # (b - a, d, w) -> (d, (b - a) * w), column block j = X[a + j]
            return X[a:b].transpose(0, 1).reshape(d, -1)

        def chunked(body):
            # body(a, b) over column chunks of width jb
            nonlocal jb
            a = 0
            while a < m:
                b = min(a + jb, m)
                try:
                    body(a, b)
                except torch.OutOfMemoryError:
                    if jb == 1:
                        raise
                    b = None
                if b is None:
                    # outside the except block, so the failed chunk's tensors are freed by now
                    jb = max(1, jb // 2)
                    torch.cuda.empty_cache()
                    print(f"gain_matrix OOM: retrying with {jb} columns per chunk")
                else:
                    a = b

        out = torch.zeros(E, m, dtype=dtype_gain, device=d1)
        for e in range(E):
            H = Sig[e].to(dtype_gain)
            L = Linv = None
            if white:
                L, info = torch.linalg.cholesky_ex(H)
                Ld = L.diagonal()
                if int(info) != 0 or not bool(Ld.min() > tol_w * Ld.max()):
                    L = None  # Sig[e] (near) singular: plain formula for this row
                elif not comp:
                    Linv = torch.linalg.solve_triangular(L, eye_g, upper=False)
            for i in range(2):
                F = _factor(e, i)
                if L is not None and comp:
                    tot = ((F @ H) * F).sum()  # tr(S Sig)

                    def body(a, b):
                        U = cat(Xs[i], a, b)
                        Z = torch.linalg.solve_triangular(L, U, upper=False)  # L^-1 U
                        out[e, a:b] += tot - _blk_tr(Z, F @ U, w)

                elif L is not None:
                    N = F @ L

                    def body(a, b):
                        # Z = L^T P
                        Z = torch.linalg.solve_triangular(Linv.mT, cat(Ps[i], a, b), upper=True)
                        out[e, a:b] += _blk_tr(Z, N @ Z, r)

                else:

                    def body(a, b):
                        B = H @ cat(Ps[i], a, b)
                        out[e, a:b] += _blk_tr(B, F @ B, r, P=Ps[i][a:b])

                chunked(body)

        if Pa is not None and A is not None:
            Pa_ = Pa.to(dtype_gain)
            A_flat = A.reshape(E, -1).to(dtype_gain)
            for a in range(0, m, cb):
                Pj = Pa_[a : a + cb]
                out[:, a : a + cb] += beta * (A_flat @ (Pj @ Pj.mT).flatten(1).mT)
        return out * gw.to(dtype_gain)[:, None]

    def score(labels, use_loo=True):
        gmem = [[e for e in range(E) if labels[e] == j] for j in range(n)]
        acc = accumulate(labels)
        P_up, P_gate, Pa = solve(*acc)

        if fast_gain:
            G = gain_matrix(P_up, P_gate, Pa)
        else:
            G = torch.empty(E, n, dtype=dtype_gain, device=d1)
            for j in range(n):
                Pa_j = None if Pa is None else Pa[j]
                for s in range(0, E, cb):
                    sl = slice(s, min(s + cb, E))
                    G[sl, j] = gain(sl, P_up[j], P_gate[j], Pa_j)

        lab = torch.as_tensor(labels, device=d1, dtype=torch.long)

        tot_in = float(G.gather(1, lab[:, None]).sum())  # sync

        if use_loo:
            del P_up, P_gate, Pa
            for s in range(0, E, cb):
                sl = slice(s, min(s + cb, E))
                Pu2, Pg2, Pa2 = solve(*downdate(acc, sl, gmem, labels))
                v = gain(sl, Pu2, Pg2, Pa2)
                G[torch.arange(sl.start, sl.stop, device=d1), lab[sl]] = v.to(
                    dtype=dtype_gain, device=d1
                )
                del Pu2, Pg2, Pa2, v
        tot_loo = float(G.gather(1, lab[:, None]).sum())  # sync
        return G, tot_in, tot_loo

    def swap_pass(Gn, labels, max_swaps):
        lab = list(labels)
        cands = []
        for e in range(len(lab)):
            for f in range(e + 1, len(lab)):
                if lab[e] == lab[f]:
                    continue
                g = (Gn[e, lab[f]] + Gn[f, lab[e]]) - (Gn[e, lab[e]] + Gn[f, lab[f]])
                if g > 0.0:
                    cands.append((g, e, f))
        cands.sort(reverse=True)
        used, k = set(), 0
        for g, e, f in cands:
            if e in used or f in used:
                continue
            lab[e], lab[f] = lab[f], lab[e]
            used.add(e)
            used.add(f)
            k += 1
            if k >= max_swaps:
                break
        return lab, k

    def in_sample_total(labels):
        return score(labels, use_loo=False)[1]

    print(fixed_size)

    # up_e / gate_e were accumulated per expert in the init loop
    up_tot = float(up_e.sum())
    gate_tot = float(gate_e.sum())

    down_e = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)
    down_tot = float(down_e.sum())

    del up_e, gate_e, down_e

    gate_mult = gate_tot / up_tot
    down_mult = down_tot / up_tot

    print(f"ups 1x, gate {gate_mult:.4f} down {down_mult:.4f}")

    if fixed_size:
        cols = [j for j in range(n) for _ in range(E // n + (j < E % n))]
    else:
        cols = np.array([j for j in range(n) for _ in range(hi)])
        mand = np.array([k < lo for _ in range(n) for k in range(hi)])
        pad = len(cols) - E
    # labels = [e % n for e in range(E)]
    if E // n != E / n:
        raise AttributeError("Uneven grouping.")
    # labels = []
    # for i in range(n):
    #     labels.extend([i] * (E // n))

    l_r, singleton_self = init_labels(
        adapter=adapter,
        layer_idx=layer_idx,
        solve=solve,
        gain=gain,
        sig_r=sig_r,
        gram=gram,
        A=A,
        E=E,
        n=n,
        cb=cb,
        dtype_gain=dtype_gain,
        device=d1,
        fixed_size=True,
        prep=prep,
        gain_matrix=gain_matrix if fast_gain else None,
    )
    labels, R = l_r
    print(labels)

    # labels = [e for e in range(i, i + n) for i in range(0, E, E // n)]
    # labels = [e % n for e in range(E)]
    best, seen = (-float("inf"), None), set()
    og_score = None

    start = time.time()
    PEN = float(E + 1)
    # G, tot_in, tot_loo = score(labels)
    # og_score = tot_loo
    # best = (tot_loo, list(labels))
    max_swaps = max(1, E // n)
    n_iter = -1
    for it in range(n_iter + 1):
        # if max_swaps < 1:
        #     break
        # Gn = G.double().cpu().numpy()
        # cand, k = swap_pass(Gn, labels, max_swaps)
        # if k == 0:
        #     if log_iters:
        #         print(f"  L{layer_idx} it{it}: no improving swap, local opt")
        #     break

        # G2, tot_in2, tot_loo2 = score(cand)
        # if tot_loo2 > best[0]:
        #     del G
        #     labels, G, best = cand, G2, (tot_loo2, list(cand))
        #     if log_iters:
        #         print(
        #             f"  L{layer_idx} it{it}: +{k} swaps ACCEPT "
        #             f"loo={tot_loo2:.6e} in={tot_in2:.6e}"
        #         )
        # else:
        #     del G2
        #     max_swaps //= 2
        #     if log_iters:
        #         print(
        #             f"  L{layer_idx} it{it}: {k} swaps REJECT "
        #             f"loo={tot_loo2:.6e} -> max_swaps={max_swaps}"
        #         )
        if tuple(labels) in seen:
            break
        seen.add(tuple(labels))

        G, tot_in, tot_loo = score(labels)  # gains AND in-sample total, one pass
        print(f"tot_in = {tot_in}, tot_loo = {tot_loo}")
        tot = tot_loo
        if og_score is None:
            og_score = tot
        if tot > best[0]:
            best = (tot, list(labels))
        if log_iters:
            t = time.time()
            print(f"  L{layer_idx} it{it}: tot={tot:.6e} | iters/s={(t - start) / (max(1, it))}")
        if it == n_iter:
            break

        Gn = G.double().cpu().numpy()
        if not np.isfinite(Gn).all():
            raise FloatingPointError(
                f"L{layer_idx} it{it}: non-finite gains "
                f"({np.count_nonzero(~np.isfinite(Gn))} entries) -- "
                f"degenerate group, check sizes"
            )
        self_gain = np.array([singleton_self[e] for e in range(E)])
        Gn = Gn / np.maximum(self_gain[:, None], 1e-30)

        if fixed_size:
            _, c = linear_sum_assignment(Gn[:, cols], maximize=True)
            labels = [cols[ci] for ci in c]
        else:
            # C = np.empty((len(cols), len(cols)))
            # C[:E] = Gn[:, cols] + BIG * mand
            span = float(Gn.max() - Gn.min())
            Gs = (Gn - Gn.min()) / (span if span > 0.0 else 1.0)
            C = np.zeros((len(cols), len(cols)))
            C[:E] = Gs[:, cols]  # no BIG here, it was a no-op
            if pad:
                C[E:] = np.where(mand, -PEN, 0.0)
            # C = np.empty((len(cols), len(cols)))
            # C[:E] = Gs[:, cols] + BIG * mand
            # if pad:
            # C[E:] = np.where(mand, BIG, 0.0)
            _, c = linear_sum_assignment(C, maximize=True)
            labels = [int(cols[ci]) for ci in c[:E]]
        del G
    # labels = best[1]

    groups = [sorted(e for e in range(E) if labels[e] == j) for j in range(n)]

    print(f"L{layer_idx} experts groups len: {[len(g) for g in groups]}")
    print(f"total time: {(time.time() - start)} | s/it={(time.time() - start) / n_iter}")
    print(f"og_score = {og_score}, best_score = {best[0]}")

    for g in groups:
        if len(g) == 0:
            raise AttributeError("len(g) == 0 not allowed")

    start = time.time()

    # rank alloc
    k_e = [adapter.d_int] * E if nys_ranks is None else [int(v) for v in nys_ranks]
    cost = [float(d + sum(k_e[e] for e in g)) for g in groups]
    budget = float(rank) * sum(cost)

    def group_curve(g):

        g_t = sum(toks[e] for e in g)
        inv_t = 1.0 / max(g_t, 1e-12)

        Sx = torch.zeros(d, d, dtype=dtype_p, device=d1)
        Su = torch.zeros_like(Sx)
        Sg = torch.zeros_like(Sx)
        # Sa = sum(A[e].to(dtype=dtype_p) for e in g) if include_down else None
        # A[e] are normalized by token count
        Sa = torch.zeros_like(Sx) if include_down else None
        for e in g:
            c = toks[e] * inv_t
            Su_e, Sg_e = gram(e)
            Sx += c * Sig[e]
            Su += c * Su_e
            Sg += c * Sg_e
            if include_down:
                Sa += c * A[e]

        ev, U = torch.linalg.eigh((Sx + Sx.T) * 0.5)
        ev = ev.clamp_min(ev.max().clamp_min(1e-30) * 1e-8)
        Mh = (U * ev.sqrt()) @ U.T

        scale = len(g)
        # scale = 1.0
        lu = scale * torch.linalg.eigvalsh(Mh @ Su @ Mh).flip(0).clamp_min(0)
        lg = scale * torch.linalg.eigvalsh(Mh @ Sg @ Mh).flip(0).clamp_min(0)
        la = (
            scale * torch.linalg.eigvalsh(Sa).flip(0).clamp_min(0)
            if Sa is not None
            else torch.zeros_like(lu)
        )
        # la = (
        #     torch.linalg.eigvalsh(Sa).flip(0).clamp_min(0)
        #     if Sa is not None
        #     else torch.zeros_like(lu)
        # )
        return (lu + lg + beta * la).double().cpu()

    dec_raw = [group_curve(g) for g in groups]

    tot = [max(float(dec_raw[j].sum()) / len(groups[j]), 1e-30) for j in range(n)]
    dec = [dec_raw[j] * (tot[j] ** (gamma - 1.0)) for j in range(n)]

    c_bar = sum(cost) / n
    r_lo = [min(max(r_min, int(round(floor_frac * rank))), int(rank)) for _ in range(n)]
    # r_hi = [min(d, max(int(round(ceil_frac * rank)), r_lo[j])) for j in range(n)]
    r_hi = [min(d, max(int(round(ceil_frac * rank * c_bar / cost[j])), r_lo[j])) for j in range(n)]

    def ranks_at(lam):
        return [min(max(int((dec[j] >= lam * cost[j]).sum()), r_lo[j]), r_hi[j]) for j in range(n)]

    def spend(rs):
        return sum(r * c for r, c in zip(rs, cost))

    if spend(ranks_at(0.0)) <= budget:
        rs = ranks_at(0.0)
    else:
        lam_lo, lam_hi = 0.0, max(float(dec[j][0]) / cost[j] for j in range(n))
        for _ in range(100):
            mid = 0.5 * (lam_lo + lam_hi)
            if spend(ranks_at(mid)) > budget:
                lam_lo = mid
            else:
                lam_hi = mid
        rs = ranks_at(lam_hi)

    used = spend(rs)
    while True:
        cand = [
            (float(dec[j][rs[j]]) / cost[j], j)
            for j in range(n)
            if rs[j] < r_hi[j] and used + cost[j] <= budget
        ]
        if not cand:
            break
        _, j = max(cand)
        rs[j] += 1
        used += cost[j]

    keep = [rs[j] * cost[j] / (d * sum(k_e[e] for e in groups[j])) for j in range(n)]
    print(
        f"L{layer_idx} rank={rank} -> {rs} | keep={['%.2f' % k for k in keep]} "
        f"| budget={used/budget:.3f}"
    )

    return groups, rs


import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def singleton_projections(solve, sig_r, gram, A, E, cb):
    pu, pg, pa = [], [], []
    for s in range(0, E, cb):
        sl = slice(s, min(s + cb, E))
        Su, Sg = gram(sl)
        a, b, c = solve(sig_r(sl), Su, Sg, None if A is None else A[sl])
        del Su, Sg
        pu.append(a)
        pg.append(b)
        if c is not None:
            pa.append(c)
    return torch.cat(pu), torch.cat(pg), (torch.cat(pa) if pa else None)


@torch.no_grad()
def affinity_matrix(gain, prep, P_up, P_gate, Pa, E, cb, dtype_gain, device, gain_matrix=None):
    if gain_matrix is not None:
        M = gain_matrix(P_up, P_gate, Pa)
    else:
        M = torch.empty(E, E, dtype=dtype_gain, device=device)
        for s in range(0, E, cb):
            sl = slice(s, min(s + cb, E))
            blk = prep(sl)
            for f in range(E):
                M[sl, f] = gain(sl, P_up[f], P_gate[f], None if Pa is None else Pa[f], blk)
            del blk
    R = M / M.diagonal().clamp_min(1e-30)[:, None]
    return R.clamp_(0.0, 1.0).double().cpu().numpy(), M.diagonal().double().cpu().numpy()


# @torch.no_grad()
# def affinity_matrix(gain, P_up, P_gate, Pa, E, cb, dtype_gain, device):
#     M = torch.empty(E, E, dtype=dtype_gain, device=device)
#     for f in range(E):
#         Pa_f = None if Pa is None else Pa[f]
#         for s in range(0, E, cb):
#             sl = slice(s, min(s + cb, E))
#             M[sl, f] = gain(sl, P_up[f], P_gate[f], Pa_f)
#     R = M / M.diagonal().clamp_min(1e-30)[:, None]
#     return R.clamp_(0.0, 1.0).double().cpu().numpy()


def balanced_kmedoids(D, n, fixed_size, n_restarts=8, n_iter=25, seed=1234, lo=None, hi=None):

    E = D.shape[0]
    if fixed_size == True:
        sizes = [E // n + (j < E % n) for j in range(n)]
        cols, mand, pad = np.repeat(np.arange(n), sizes), None, 0
    else:
        # doesnt work dont use this
        lo = 5 if lo is None else lo
        hi = hi = int(min(E, 1.5 * -(-E // n))) if hi is None else hi

        cols = np.repeat(np.arange(n), hi)
        mand = np.tile(np.arange(hi) < lo, n)
        pad = len(cols) - E

    BIG = 1e3 * max(1.0, float(D.max()))
    S = len(cols)

    # assert sum(sizes) == E, "sizes must partition all E experts"
    rng = np.random.default_rng(seed)
    best = (np.inf, None)

    def assign(med):
        C = np.empty((S, S))
        C[:E] = D[:, med][:, cols]
        if mand is not None:
            C[:E] -= BIG * mand  # reward real experts in mandatory seats
            if pad:
                C[E:] = np.where(mand, BIG, 0.0)
        _, c = linear_sum_assignment(C)
        lab = cols[c[:E]]
        # true cost, free of the BIG offsets
        return lab, float(D[np.arange(E), med[lab]].sum())

    best = (np.inf, None)
    for _ in range(n_restarts):
        med = np.empty(n, dtype=np.int64)
        med[0] = rng.integers(E)
        for k in range(1, n):
            p = D[:, med[:k]].min(1) ** 2
            t = p.sum()
            med[k] = rng.choice(E, p=p / t) if t > 0 else rng.integers(E)

        for _ in range(n_iter):
            lab, _ = assign(med)
            new = med.copy()
            for j in range(n):
                mem = np.flatnonzero(lab == j)
                if len(mem) == 0:
                    continue
                new[j] = mem[D[np.ix_(mem, mem)].sum(0).argmin()]
            if np.array_equal(new, med):
                break
            med = new

        lab, cost = assign(med)
        if cost < best[0]:
            best = (cost, lab)

    return [int(j) for j in best[1]]


@torch.no_grad()
def init_labels(
    adapter,
    layer_idx,
    solve,
    gain,
    sig_r,
    gram,
    A,
    E,
    n,
    cb,
    dtype_gain,
    device,
    prep,
    fixed_size=True,
    solve_chunk=None,
    verbose=True,
    gain_matrix=None,
):
    if solve_chunk is None:
        solve_chunk = max(1, min(E, cb))
    t = time.time()
    P_up, P_gate, Pa = singleton_projections(solve, sig_r, gram, A, E, solve_chunk)
    print(f"singleton in {time.time() - t}s")
    t = time.time()
    R, M_diag = affinity_matrix(
        gain, prep, P_up, P_gate, Pa, E, cb, dtype_gain, device, gain_matrix=gain_matrix
    )
    print(f"affinity in {time.time() - t}s")
    del P_up, P_gate, Pa

    if verbose:
        off = R[~np.eye(E, dtype=bool)]
        q = np.quantile(off, [0.1, 0.5, 0.9])
        print(f"  affinity R off-diag: p10={q[0]:.4f} med={q[1]:.4f} p90={q[2]:.4f}")
        if q[1] > 0.97:
            print(
                "  WARNING: every subspace already explains every expert at this "
                "probe rank -- no grouping can matter. Lower probe_rank."
            )

    # Dm = 1.0 - 0.5 * (R + R.T) # worked very good initally
    Dm = 1.0 - R
    np.fill_diagonal(Dm, 0.0)
    np.clip(Dm, 0.0, None, out=Dm)

    t = time.time()
    b_kmed = (
        balanced_kmedoids(
            Dm,
            n,
            fixed_size,
            n_restarts=adapter.config.grouped_restarts,
            n_iter=adapter.config.grouped_iters,
        ),
        R,
    )
    print(f"b_kmed in {time.time() - t}s")

    return b_kmed, M_diag
