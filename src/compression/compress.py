import gc
import time

import numpy as np

import json

from torch import Tensor
import torch
from src.utils.model import dtype_p, d1, calib_device, parallel
from src.adapters.model_adapter import MLPComponents
from src.adapters.moe_adapter import MoEAdapter


from src.compression.get_shared_experts import get_shared_experts_fast
from src.adapters.model_adapter import MLPTensors
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import os

from src.utils.logger import logger

factorization_scores = {}

router_reconstruction_cost = 0.0

get_size = lambda t: t.element_size() * t.nelement()

spectrals = []

tmpdir = os.environ.get("TMPDIR")


@torch.no_grad()
def compress_mlp_fact_nys(
    adapter: MoEAdapter,
    cov_x: list[Tensor],
    cov_mlp_fact: list[Tensor],
    cov_mlp_nys: list[list[Tensor]],
    fact_keep_ratios: list[float],
    nys_keep_ratios: list[float],
    target_layers: list[int],
    taylor_scores: list[Tensor],  # [n_layers, n_exps, d_int]
):

    global router_reconstruction_cost

    arch = adapter.arch

    import gc

    blocks = adapter.get_transformer_blocks()

    n_shared_projs = adapter.config.n_shared_projs

    for layer_idx in target_layers:

        shared_cost = adapter.d_model / (adapter.n_experts * adapter.d_int)
        total_keep = fact_keep_ratios[layer_idx] * (nys_keep_ratios[layer_idx] + shared_cost)
        print(
            f"Layer {layer_idx} keep_ratio={total_keep}, nys_keep_ratio = {nys_keep_ratios[layer_idx]}, fact_keep_ratio = {fact_keep_ratios[layer_idx]}"
        )

        adapter.metrics["latent_moe_metrics"][f"layer_{layer_idx}"] = {}
        gc.collect()
        if arch == "deepseek" and layer_idx == 0:
            continue
        leverage_scores = []
        if adapter.config.nys_to_fact_ratio != 0.0:
            for expert_idx in range(adapter.n_experts):
                leverage = get_leverage(
                    adapter,
                    cov_mlp_nys[layer_idx][expert_idx],
                    nys_keep_ratios[layer_idx],
                    ridge_lambda=adapter.config.nystrom_ridge,
                )
                leverage_scores.append(leverage)

        ts = None if taylor_scores is None else taylor_scores[layer_idx]

        ranks, unsorted_scores = allocate_dff_ranks(
            adapter,
            leverage_scores,
            ts,
            nys_keep_ratios[layer_idx],
            uniformity=adapter.config.uniformity,
            layer_idx=layer_idx,
        )

        adapter.metrics["latent_moe_metrics"][f"layer_{layer_idx}"]["int_ranks"] = ranks

        if fact_keep_ratios[layer_idx] == 1.0:

            experts_mlp = [
                adapter.get_mlp_tensors(layer_idx=layer_idx, expert_idx=expert_idx)
                for expert_idx in range(adapter.n_experts_in_layer(layer_idx))
            ]

            if nys_keep_ratios[layer_idx] != 1.0:
                compress_coefficients_dff(
                    adapter,
                    experts_mlp,
                    cov_mlp_nys[layer_idx],
                    leverage_scores=leverage_scores,
                    ranks=ranks,
                    unsorted_scores=unsorted_scores,
                    layer_idx=layer_idx,
                )

            for expert_idx, exp_coeff in enumerate(experts_mlp):
                exp_coeff.group_number = 0
                exp_coeff.expert_idx = expert_idx

            shared_projs = [MLPTensors(up_proj=None, down_proj=None, gate_proj=None)]
            expert_coefficients = experts_mlp
        else:

            experts_mlp = None
            experts_mlp = [
                adapter.get_mlp_tensors(layer_idx=layer_idx, expert_idx=expert_idx)
                for expert_idx in range(adapter.n_experts_in_layer(layer_idx))
            ]

            if nys_keep_ratios[layer_idx] != 1.0:
                compress_coefficients_dff(
                    adapter,
                    experts_mlp,
                    cov_mlp_nys[layer_idx],
                    leverage_scores=leverage_scores,
                    ranks=ranks,
                    unsorted_scores=unsorted_scores,
                    layer_idx=layer_idx,
                    ups=True,
                    down=False,
                )

            if n_shared_projs == 1:
                groups = [
                    [expert_idx for expert_idx in range(adapter.n_experts_in_layer(layer_idx))]
                ]
                fact_ranks = [
                    int(fact_keep_ratios[layer_idx] * adapter.d_model)
                ] * adapter.n_experts_in_layer(layer_idx)
            else:
                # groups, fact_ranks = get_shared_experts(
                groups, fact_ranks = get_shared_experts_fast(
                    adapter,
                    n_shared_projs,
                    layer_idx,
                    cov_x,
                    cov_mlp_fact,
                    cov_mlp_nys,
                    leverage_scores,
                    rank=int(fact_keep_ratios[layer_idx] * adapter.d_model),
                    fixed_size=False,
                    experts_mlp=experts_mlp,
                    nys_ranks=ranks,
                    disable_down=adapter.config.disable_down,
                    dtype_gain=torch.float32,
                    probe_rank=int(fact_keep_ratios[layer_idx] * adapter.d_model),
                )

            shared_projs = []
            expert_coefficients = []
            for group_number, group in enumerate(groups):
                shared_proj, expert_coefficients_in_group = get_shared_and_expert_coeffs(
                    adapter=adapter,
                    layer_idx=layer_idx,
                    cov_mlp_fact=cov_mlp_fact,
                    cov_x=cov_x,
                    cov_mlp_nys=cov_mlp_nys,
                    leverage_scores=leverage_scores,
                    fact_keep_ratios=fact_keep_ratios,
                    group=group,
                    nys_keep_ratios=nys_keep_ratios,
                    nys_ranks=ranks,
                    fact_rank=fact_ranks[group_number],
                    expert_coefficients=experts_mlp,
                )

                for expert_idx, exp_coeff in zip(group, expert_coefficients_in_group):
                    exp_coeff.group_number = group_number
                    exp_coeff.expert_idx = expert_idx

                expert_coefficients.extend(expert_coefficients_in_group)
                shared_projs.append(shared_proj)

            expert_coefficients.sort(key=lambda x: x.expert_idx)

        for shared_proj in shared_projs:
            shared_proj.to(dtype=torch.bfloat16, device=d1)
        for expert_coefficient in expert_coefficients:
            expert_coefficient.to(dtype=torch.bfloat16, device=d1)

        adapter.save_layer(
            os.path.join(tmpdir, "compressed_layers/"),
            {
                "shared_proj": [
                    {
                        "up": shared_proj.up_proj,
                        "down": shared_proj.down_proj,
                        "gate": shared_proj.gate_proj,
                    }
                    for shared_proj in shared_projs
                ],
                "expert_coefficients": [
                    {
                        "up": expert.up_proj,
                        "down": expert.down_proj,
                        "gate": expert.gate_proj,
                        "group_number": expert.group_number,
                        "expert_idx": expert.expert_idx,
                    }
                    for expert in expert_coefficients
                ],
            },
            layer_idx,
            factorized=True,
        )

        del expert_coefficients
        del shared_proj


def get_shared_and_expert_coeffs(
    adapter: MoEAdapter,
    layer_idx: int,
    cov_mlp_fact: list[Tensor],
    cov_x: list[Tensor],
    cov_mlp_nys: list[list[Tensor]],
    leverage_scores: list[Tensor],
    fact_keep_ratios: list[float],
    group: list[int],
    nys_keep_ratios: list[float],
    nys_ranks: list[int],
    fact_rank: int,
    expert_coefficients: list[MLPTensors] | None = None,
):

    if expert_coefficients is None:
        expert_coefficients = [None] * adapter.n_experts_in_layer(layer_idx)
        for expert_idx in group:
            expert_coefficients[expert_idx] = adapter.get_mlp_tensors(
                layer_idx=layer_idx, expert_idx=expert_idx
            )

        if nys_keep_ratios[layer_idx] != 1.0:
            compress_coefficients_dff(
                adapter,
                expert_coefficients,
                cov_mlp_nys[layer_idx],
                leverage_scores=leverage_scores,
                ranks=nys_ranks,
                unsorted_scores=nys_ranks,
                layer_idx=layer_idx,
                down=False,
                ups=True,
                # nys_keep_ratios[layer],
            )
    else:
        for mlp in expert_coefficients:
            mlp.to(dtype=dtype_p, device=d1)

    tokens_per_exp = adapter.metrics["tokens_per_exp"][layer_idx]

    tokens_per_exp = [t if count in group else 0 for count, t in enumerate(tokens_per_exp)]
    shared_C_x = torch.zeros(adapter.d_model, adapter.d_model, dtype=dtype_p, device=d1)
    shared_C_a = torch.zeros(adapter.d_model, adapter.d_model, dtype=dtype_p, device=d1)
    total_tokens = sum(tokens_per_exp)
    # for expert_idx, C_x in enumerate(cov_x[layer_idx]):
    for expert_idx in group:
        C_x = cov_x[layer_idx][expert_idx]
        C_a = cov_mlp_fact[layer_idx][expert_idx]

        t = torch.trace(C_x) / adapter.d_model
        ridge = adapter.config.Ex_ridge * t
        I_r = ridge * torch.eye(adapter.d_model, adapter.d_model, dtype=dtype_p, device=d1)

        if tokens_per_exp[expert_idx] != 0:
            shared_C_x += (tokens_per_exp[expert_idx] / total_tokens) * (I_r + C_x)
            # shared_C_x += (tokens_per_exp[expert_idx] / total_tokens) * (I_r)
            shared_C_a += (tokens_per_exp[expert_idx] / total_tokens) * (C_a)

    # shared_C_x = torch.eye(adapter.d_model, adapter.d_model, dtype=dtype_p, device=d1)

    shared_proj = find_shared_proj(
        adapter=adapter,
        layer_idx=layer_idx,
        C_x=shared_C_x,
        # cov_mlp_fact=cov_mlp_fact[layer_idx],
        cov_mlp_fact=shared_C_a,
        cov_mlp_nys=cov_mlp_nys[layer_idx],
        keep_ratio=fact_keep_ratios,
        leverage_scores=leverage_scores,
        weights=expert_coefficients,
        nys_ranks=nys_ranks,
        group=group,
        fact_rank=fact_rank,
    )

    expert_coefficients = accumulate_expert_coefficients(
        adapter=adapter,
        layer_idx=layer_idx,
        cov_x=cov_x[layer_idx],
        P_shared=shared_proj,
        shared_C_x=shared_C_x,
        group=group,
        weights=expert_coefficients,
    )

    s = [None] * adapter.n_experts_in_layer(layer_idx)
    for count, expert_idx in enumerate(group):
        s[expert_idx] = expert_coefficients[count]

    if adapter.config.nys_to_fact_ratio != 0.0:
        compress_coefficients_dff(
            adapter,
            # expert_coefficients,
            s,
            cov_mlp_nys[layer_idx],
            leverage_scores=leverage_scores,
            ranks=nys_ranks,
            unsorted_scores=nys_ranks,
            layer_idx=layer_idx,
            down=True,
            ups=False,
            # nys_keep_ratios[layer],
        )

    expert_coefficients = s
    expert_coefficients = [e for e in expert_coefficients if e is not None]

    return shared_proj, expert_coefficients


@torch.no_grad()
def find_shared_proj(
    adapter: MoEAdapter,
    layer_idx: int,
    C_x: Tensor,
    cov_mlp_fact: Tensor,  # [d, d]
    cov_mlp_nys: list[Tensor],  # [n_experts, 1 activations]
    keep_ratio: list[float],
    leverage_scores: list[Tensor],  # [n_experts, d_ff]
    weights: list[MLPTensors] = None,
    nys_ranks: list[int] = None,
    group: list[int] = None,
    fact_rank: int = None,
) -> MLPTensors:

    disable_up, disable_down = adapter.config.disable_up, adapter.config.disable_down

    # layer_metrics = adapter.metrics["latent_moe_metrics"][f"layer_{layer_idx}"]

    C_x = C_x.to(device=d1, dtype=dtype_p)

    E_up = torch.zeros(adapter.d_model, adapter.d_model, device=d1, dtype=dtype_p)
    E_gate = torch.zeros(adapter.d_model, adapter.d_model, device=d1, dtype=dtype_p)

    # if cov_mlp_fact is a single d x d tensor then E_down is just cov_mlp_fact
    # E_down = torch.zeros_like(cov_mlp_fact).to(dtype=dtype_p, device=d1)
    E_down = cov_mlp_fact

    # if cov_mlp_fact is n_experts [d x d] tensors then we'll sum
    # E_down = torch.zeros(adapter.d_model, adapter.d_model, device=d1, dtype=dtype_p)

    tokens_per_exp = adapter.metrics["tokens_per_exp"][layer_idx]
    layer_toks = [t if count in group else 0 for count, t in enumerate(tokens_per_exp)]
    total_toks = sum(layer_toks)
    for expert_idx in group:
        if weights is None:
            comps = adapter.get_mlp_tensors(layer_idx=layer_idx, expert_idx=expert_idx)
        else:
            comps = weights[expert_idx]
        comps = comps.to(dtype=dtype_p, device=d1)
        W_up, W_down, W_gate = comps.up_proj, comps.down_proj, comps.gate_proj

        # topk = expert_topks[expert_idx]
        # W_up = W_up[topk, :]
        # W_gate = W_gate[topk, :]

        # E_gate += W_gate.T @ torch.diag(expert_ridge_scores[expert_idx]) @ W_gate

        toks = layer_toks[expert_idx]
        # # T_up = W_down @ W_up
        # # T_gate = W_down @ W_gate

        # topk = torch.topk(
        #     leverage_scores[expert_idx], k=nys_ranks[expert_idx], largest=True, dim=0
        # ).indices
        # topk, _ = torch.sort(topk)

        # topk = topk.to(device=d1)

        # d_up = (
        #     adapter.act_sens_up[layer_idx][expert_idx]
        #     .to(dtype=dtype_p, device=d1)[topk]
        #     .clamp_min(0)
        #     .sqrt()
        # )
        # d_gate = (
        #     adapter.act_sens_gate[layer_idx][expert_idx]
        #     .to(dtype=dtype_p, device=d1)[topk]
        #     .clamp_min(0)
        #     .sqrt()
        # )

        # T_up = (W_down * d_up.unsqueeze(0)) @ W_up
        # T_gate = (W_down * d_gate.unsqueeze(0)) @ W_gate

        # E_up += (toks / total_toks) * (T_up.T @ T_up)
        # E_gate += (toks / total_toks) * (T_gate.T @ T_gate)

        # E_gate += (toks / total_toks) * (T_gate.T @ T_gate)
        # E_up += (
        #     W_up.T @ torch.diag(expert_ridge_scores[expert_idx]) @ W_up
        # )  # [d, d_ff] [d_ff, d_ff], [d_ff, d]

        E_up += (toks / total_toks) * (W_up.T @ W_up)
        E_gate += (toks / total_toks) * (W_gate.T @ W_gate)

        # E_up += (toks / total_toks) * (T_up.T @ T_up)

        # C = cov_mlp_nys[expert_idx]

        # C_reduced = C.to(device=d1)[topk][:, topk]

        # E_down += (toks / total_toks) * (W_down @ C_reduced @ W_down.T)  # [d, d]
        # del C_reduced
        # del T_up, T_gate

    C_x_ridged = (C_x + C_x.T) / 2.0
    if adapter.arch == "qwen2_moe":
        # THIS MAY BE UNNECCARY!
        C_x_ridged += 1e-4 * torch.eye(C_x.shape[0], dtype=dtype_p, device=d1)
    eigval_Cx, eigvec_Cx = torch.linalg.eigh(C_x_ridged)
    # eigval_Cx, eigvec_Cx = torch.linalg.eigh((C_x + C_x.T) / 2.0)
    sqrt_Cx_eigvals = torch.sqrt(eigval_Cx.clamp(min=0.0))

    sqrt_Cx = eigvec_Cx @ torch.diag(sqrt_Cx_eigvals) @ eigvec_Cx.T

    # inv_sqrt_eigvals = 1.0 / (sqrt_Cx_eigvals + ridge_Cx)
    # inv_sqrt_Cx = eigvec_Cx @ torch.diag(inv_sqrt_eigvals) @ eigvec_Cx.T

    S_up = sqrt_Cx @ E_up @ sqrt_Cx
    S_gate = sqrt_Cx @ E_gate @ sqrt_Cx

    eigval_up, eigvec_up = torch.linalg.eigh(S_up)
    eigval_gate, eigvec_gate = torch.linalg.eigh(S_gate)
    eigval_down, eigvec_down = torch.linalg.eigh(E_down)

    up_rank, down_rank = fact_rank, fact_rank

    P_up, P_gate, P_down = None, None, None
    if not disable_up:

        if adapter.arch == "qwen2_moe":
            inv_sqrt = eigvec_Cx @ torch.diag(eigval_Cx.rsqrt()) @ eigvec_Cx.T
            P_up = inv_sqrt @ eigvec_up[:, -up_rank:].flip(dims=(1,))
            P_gate = inv_sqrt @ eigvec_gate[:, -up_rank:].flip(dims=(1,))

            P_up, _ = torch.linalg.qr(P_up)
            P_gate, _ = torch.linalg.qr(P_gate)
        else:
            P_up = torch.linalg.solve(sqrt_Cx, eigvec_up[:, -up_rank:].flip(dims=(1,)))
            P_gate = torch.linalg.solve(sqrt_Cx, eigvec_gate[:, -up_rank:].flip(dims=(1,)))

            P_up, _ = torch.linalg.qr(P_up)
            P_gate, _ = torch.linalg.qr(P_gate)

    if not disable_down:
        P_down = eigvec_down[:, -down_rank:].flip(dims=(1,))

    shared = MLPTensors(up_proj=P_up, down_proj=P_down, gate_proj=P_gate)

    return shared


def accumulate_expert_coefficients(
    adapter: MoEAdapter,
    layer_idx: int,
    # C_x: Tensor,
    cov_x: list[Tensor],  # [n_experts, Tensor[d, d]]
    P_shared: MLPTensors,
    shared_C_x: Tensor = None,
    group: list[int] | None = None,
    weights: list[MLPTensors] = None,
) -> list[MLPTensors]:

    if adapter.arch == "deepseek" and layer_idx == 0:
        # deepseek has dense 0th layer
        # figure out what to do here later
        raise AttributeError("Not viable MoE layer for Deepseek")

    disable_down = adapter.config.disable_down
    disable_up = adapter.config.disable_up

    if group is None:
        group = [expert_idx for expert_idx in range(adapter.n_experts_in_layer(layer_idx))]

    # C_x = C_x.to(device=d1, dtype=dtype_p)
    P_shared.to(dtype=dtype_p, device=d1)

    M_down = None
    if not disable_down:
        # M_down = C_a @ P_shared.down_proj @ G_down
        M_down = P_shared.down_proj.T

    # only if C is not regularized
    # M_up = C_x @ P_shared.up_proj
    # M_gate = C_x @ P_shared.gate_proj
    # M_down = C_a @ P_shared.down_proj

    up_weights = []
    gate_weights = []
    down_weights = []
    for exp_idx in group:
        # stack experts then compute coefficients
        if weights is None:
            c = adapter.get_mlp_components(layer_idx, exp_idx)
            up_weights.append(c.up_proj.weight.to(dtype=dtype_p, device=d1))
            gate_weights.append(c.gate_proj.weight.to(dtype=dtype_p, device=d1))
            down_weights.append(c.down_proj.weight.to(dtype=dtype_p, device=d1))
        else:
            c = weights[exp_idx]
            up_weights.append(c.up_proj.to(dtype=dtype_p, device=d1))
            gate_weights.append(c.gate_proj.to(dtype=dtype_p, device=d1))
            down_weights.append(c.down_proj.to(dtype=dtype_p, device=d1))

    expert_coefficients = []
    group_covariances = [cov_x[expert_idx] for expert_idx in group]
    for W_up, W_gate, W_down, C_x, expert_idx in zip(
        up_weights, gate_weights, down_weights, group_covariances, group
    ):

        U_up, U_gate, U_down = None, None, None

        if not disable_up:
            if adapter.arch == "qwen2_moe":
                try:

                    gram_up = P_shared.up_proj.T @ C_x @ P_shared.up_proj
                    gram_up = (gram_up + gram_up.T) / 2
                    gram_gate = P_shared.gate_proj.T @ C_x @ P_shared.gate_proj
                    gram_gate = (gram_gate + gram_gate.T) / 2

                    L_up = torch.linalg.cholesky(gram_up)
                    L_gate = torch.linalg.cholesky(gram_gate)

                    M_up = torch.cholesky_solve((C_x @ P_shared.up_proj).T.contiguous(), L_up).T
                    M_gate = torch.cholesky_solve(
                        (C_x @ P_shared.gate_proj).T.contiguous(), L_gate
                    ).T
                    U_up = W_up @ M_up
                    U_gate = W_gate @ M_gate

                except:
                    # if per-expert Cx is singular we fall back to shared_C_x

                    Cx_ridged = (shared_C_x + shared_C_x.T) / 2.0
                    G_up = torch.linalg.inv(P_shared.up_proj.T @ Cx_ridged @ P_shared.up_proj)
                    G_gate = torch.linalg.inv(P_shared.gate_proj.T @ Cx_ridged @ P_shared.gate_proj)

                    M_up = Cx_ridged @ P_shared.up_proj @ G_up
                    M_gate = Cx_ridged @ P_shared.gate_proj @ G_gate
                    U_up = W_up @ M_up
                    U_gate = W_gate @ M_gate

            else:
                gram_up = P_shared.up_proj.T @ C_x @ P_shared.up_proj
                gram_up = (gram_up + gram_up.T) / 2
                gram_gate = P_shared.gate_proj.T @ C_x @ P_shared.gate_proj
                gram_gate = (gram_gate + gram_gate.T) / 2

                L_up = torch.linalg.cholesky(gram_up)
                L_gate = torch.linalg.cholesky(gram_gate)

                M_up = torch.cholesky_solve((C_x @ P_shared.up_proj).T.contiguous(), L_up).T
                M_gate = torch.cholesky_solve((C_x @ P_shared.gate_proj).T.contiguous(), L_gate).T
                U_up = W_up @ M_up
                U_gate = W_gate @ M_gate
        if not disable_down:
            U_down = M_down @ W_down
        else:
            U_down = W_down

        mlp = MLPTensors(up_proj=U_up, down_proj=U_down, gate_proj=U_gate)
        expert_coefficients.append(mlp)

    return expert_coefficients


def allocate_ranks(
    adapter: MoEAdapter,
    eig_up: Tensor,
    eig_gate: Tensor,
    eig_down: Tensor,
    keep_ratio: float,
    layer_idx: int,
):

    global factorization_scores

    disable_up = adapter.config.disable_up
    disable_down = adapter.config.disable_down

    d_model, d_int = adapter.d_model, adapter.get_n_inner(layer_idx=layer_idx)  # type: ignore

    if disable_up:
        return d_model, int(keep_ratio * d_model)

    if disable_down:
        return int(d_model * keep_ratio), d_model

    return int(d_model * keep_ratio), int(d_model * keep_ratio)


def get_leverage(adapter: MoEAdapter, C: Tensor, keep_ratio: float, ridge_lambda=1e-4) -> Tensor:
    """
    C is a particular expert's activations

    inv_diag is shape [d_ff]
    """

    C = C.to(dtype=dtype_p, device=d1)  # [D_int, D_int]
    D_int = C.shape[0]

    # nys_keep_ratio = keep_ratio**adapter.config.nys_to_fact_ratio
    # rank = int(D_int * nys_keep_ratio)
    rank = int(D_int * keep_ratio)

    if not C.any() or not torch.isfinite(C).all():
        print(f"C_a still ZERO!")
    # ridge_lambda = ridge_lambda * torch.trace(C) / D_int
    # print(torch.trace(C))
    # print(f"yielded_ridged = {ridge_lambda}")
    # print(f"ridge_lambda = {ridge_lambda}")
    C_ridge = C + (ridge_lambda * torch.eye(D_int, device=C.device))

    L = torch.linalg.cholesky(C_ridge)
    C_ridge_inv = torch.cholesky_inverse(L)

    inv_diag = torch.diag(C_ridge_inv)

    leverage = 1 - (ridge_lambda * inv_diag)
    leverage = leverage.clamp(min=0.0)

    # topk = torch.topk(inv_diag, k=rank, largest=False, dim=0).indices
    # topk, _ = torch.sort(topk)

    return leverage


def compress_coefficients_dff(
    adapter: MoEAdapter,
    expert_coeffs: list[MLPTensors],  # [n_experts, Tensor[r, d_int | d_int, r]]
    expert_covs: list[Tensor],
    leverage_scores: list[Tensor],  # [n_experts, d_int]
    ranks: list[int],
    unsorted_scores: list[Tensor],  # [n_experts, d_int]
    layer_idx: int,
    group: list[int] | None = None,
    down=True,
    ups=True,
):

    for expert_comps, C, leverage, rank, expert_neuron_scores, expert_idx in zip(
        expert_coeffs,
        expert_covs,
        leverage_scores,
        ranks,
        unsorted_scores,
        range(adapter.n_experts_in_layer(layer_idx)),
    ):

        if expert_comps is None:
            continue

        C = C.to(dtype=dtype_p, device=d1)

        topk = torch.topk(leverage, k=rank, largest=True, dim=0).indices
        topk, _ = torch.sort(topk)

        topk = topk.to(device=d1)

        C_reduced = C[topk][:, topk]
        C_reduced = (C_reduced + C_reduced.T) / 2

        if down:
            U_d = expert_comps.down_proj.to(device=d1, dtype=dtype_p)
            down_cross_term = C[topk, :] @ U_d.T
            for jitter in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
                try:
                    # L_red = torch.linalg.cholesky(C_reduced + 1e-6 * torch.eye(rank, device=d1, dtype=dtype_p))
                    L_red = torch.linalg.cholesky(
                        C_reduced + jitter * torch.eye(rank, device=d1, dtype=dtype_p)
                    )
                    U_d_proj = torch.cholesky_solve(down_cross_term, L_red).T
                    break
                except:
                    # This hopefully never fires
                    print(f"Increasing jitter on cholesky")
                    continue

            expert_comps.down_proj = U_d_proj

        if ups:
            expert_comps.up_proj = expert_comps.up_proj[topk, :]
            expert_comps.gate_proj = expert_comps.gate_proj[topk, :]


import os

evo_path1 = "./expert_sparsity_evo_search/search1_Aug25/"

evo_ranks = {}
evo_ranks_path1 = os.path.join(evo_path1, "favorite_ranks.pt")
all_ranks = []
all_ranks_path1 = os.path.join(evo_path1, "all_ranks_scored.pt")


@torch.no_grad()
def allocate_dff_ranks(
    adapter: MoEAdapter,
    leverage_scores: list[Tensor],
    taylor_scores: Tensor,  # [n_exps, d_int]
    layer_keep_ratio: float,
    expert_min_keep: float = 0.05,
    uniformity: str = "taylor",
    layer_idx: int = None,
    # expert_scale: list | None = None,
):
    """
    Allocates ranks by leverage meaned with taylor
    """

    if adapter.config.nys_to_fact_ratio == 0.0:
        return [adapter.d_int] * adapter.n_experts, [adapter.d_int] * adapter.n_experts

    global evo_ranks, all_ranks

    if uniformity == "evolutionary":
        import os

        max_iters = 100

        if not os.path.exists(evo_ranks_path1):
            raise AttributeError(f"{evo_ranks_path1}")

        if not evo_ranks:
            e1 = torch.load(evo_ranks_path1)
            # e2 = torch.load(evo_ranks_path2)
            # e3 = torch.load(evo_ranks_path3)
            # e4 = torch.load(evo_ranks_path4)
            # evo_ranks = {**e1, **e2, **e3, **e4}
            evo_ranks = {**e1}
        # if not all_ranks:
        #     a1 = torch.load(all_ranks_path1)
        #     a2 = torch.load(all_ranks_path2)
        #     all_ranks = [*a1[0:4], *a2[4:8]]

        # from src.experiments.evo_search import allocate_dff_ranks_evolutionary, ranks_to_theta

        # ranks = evo_ranks[layer_idx]

        # thetas = ranks_to_theta(ranks, min_rank=int(0.1 * adapter.d_int))
        # print(thetas)
        # ranks = allocate_dff_ranks_evolutionary(thetas, layer_keep_ratio, adapter.d_int, adapter.n_experts)

        # return ranks, ranks
        return evo_ranks[layer_idx], evo_ranks[layer_idx]

    n_experts = adapter.n_experts
    d_int = leverage_scores[0].shape[0]

    min_rank_per_expert = int(expert_min_keep * d_int)
    max_rank_alloc = int(layer_keep_ratio * d_int * n_experts)

    sorted_scores = []
    unsorted_scores = []

    for expert_idx in range(n_experts):

        l_exp = leverage_scores[expert_idx]

        l_score = l_exp

        if uniformity == "taylor":
            pass
            # score = t_score
        elif uniformity == "leverage":
            score = l_score
        elif uniformity == "mixed":
            pass
            # score = (l_score.to(device=d1) + t_score.to(device=d1)) / 2.0

        l = torch.sort(score).values.to(device=d1)
        sorted_scores.append(l)
        unsorted_scores.append(score)

    ranks = [min_rank_per_expert] * n_experts
    remaining_rank = max_rank_alloc - min_rank_per_expert * n_experts

    cand_vals, cand_exp = [], []
    for expert_idx in range(n_experts):
        # change this if descending True
        vals = sorted_scores[expert_idx][: len(sorted_scores[expert_idx]) - min_rank_per_expert]
        cand_vals.append(vals)
        cand_exp.append(torch.full((vals.numel(),), expert_idx, dtype=torch.long))

    cand_vals = torch.cat(cand_vals).to(device=d1)
    cand_exp = torch.cat(cand_exp).to(device=d1)

    chosen_ranks = torch.topk(cand_vals, k=remaining_rank, largest=True).indices.to(device=d1)
    chosen_exp = cand_exp[chosen_ranks]

    for expert_idx in range(n_experts):
        ranks[expert_idx] += int((chosen_exp == expert_idx).sum().item())

        if adapter.metrics["tokens_per_exp"][layer_idx][expert_idx] == 0:
            print(
                f"int rank of dead expert: {ranks[expert_idx]}. k/d_int = {ranks[expert_idx]}/{d_int}. eff_rank: {torch.sum(leverage_scores[expert_idx]).item()}"
            )

    return ranks, unsorted_scores
