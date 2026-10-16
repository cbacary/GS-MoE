import types

import torch
from torch.types import Tensor

from src.utils.logger import logger

from src.utils.model import d1, d2, dtype_p
from src.adapters.moe_adapter import MoEAdapter

from torch.nn.functional import softmax


def allocate_taylor_sparsity(
    adapter: MoEAdapter,
    taylor_scores: Tensor,  # [n_layers, n_experts]
    target_layer_keeps: Tensor,  # [n_layers]
    compression_ratio: float,
    beta: float = 1.5,  # basically smoothing for power transform
):
    """
    Allocate sparsity per-expert within layer using target_layer_sparsities
    """

    n_layers, n_experts = adapter.n_layers, adapter.n_experts

    taylor_scores = taylor_scores.to(device="cpu")
    target_layer_keeps = target_layer_keeps.to(device="cpu")

    powered = taylor_scores**beta
    # powered_mean = (taylor_scores**beta).mean(dim=1, keepdim=True)  # [n_layers, 1]
    # power_taylor = (taylor_scores**beta) / powered_mean

    keep_min, keep_max = 0.01, 0.98
    expert_keeps = torch.empty_like(powered)
    for layer_idx in range(n_layers):
        w = powered[layer_idx]
        budget = target_layer_keeps[layer_idx].item() * n_experts
        keep = torch.full((n_experts,), keep_min, dtype=w.dtype)
        rem = budget - keep_min * n_experts
        active = w > 0
        if target_layer_keeps[layer_idx] > keep_max:
            print(
                f"layer {layer_idx}: target keep {target_layer_keeps[layer_idx]:.3f} > cap {keep_max}; saturating"
            )

        while rem > 0 and active.any():
            idx = active.nonzero(as_tuple=True)[0]
            w_act = w[idx]
            alloc = rem * w_act / w_act.sum()
            over = alloc >= (keep_max - keep_min)
            if not over.any():
                keep[idx] = keep_min + alloc
                break
            keep[idx[over]] = keep_max
            rem -= (keep_max - keep_min) * int(over.sum())
            active[idx[over]] = False
        expert_keeps[layer_idx] = keep

    # expert_keeps = power_taylor * target_layer_keeps.unsqueeze(dim=1)

    return expert_keeps


def allocate_global_sparsity(
    bi_scores: list[float],
    compression_ratio: float,
    smoothing: float = 0.015,
    adapter=None,
    invert=False,
    null_layers=[],
):
    if adapter:
        adapter.metrics["smoothing"] = smoothing

    if null_layers:
        bi_scores = [bi_scores[i] for i in range(len(bi_scores)) if i not in null_layers]

    n_layers = len(bi_scores)

    L = n_layers
    phi_avg = compression_ratio
    epsilon = smoothing

    s = torch.tensor(bi_scores).to(dtype_p)
    if invert:
        s = -s
    # phi = L * phi_avg * softmax(-s / epsilon, dim=0)
    # the -s flips when using the CKA (higher score more compression)
    softmaxxed = softmax(-s / epsilon, dim=0)
    ratios = L * phi_avg * softmaxxed
    # phi = L * phi_avg * softmax(-s / epsilon, dim=0)

    min_keep, max_keep = 0.15, 0.98

    if compression_ratio >= 0.40:
        min_keep = 0.05
    min_prune, max_prune = 1 - max_keep, 1 - min_keep
    for _ in range(30):
        under = ratios < min_prune
        over = ratios > max_prune
        clamped = under | over
        if not clamped.any():
            break

        # positive when clamping adds prune budget, negative when it frees some
        excess = (min_prune - ratios[under]).sum() + (max_prune - ratios[over]).sum()
        ratios[under] = min_prune
        ratios[over] = max_prune

        free = ~clamped
        if not free.any():
            break

        free_weights = ratios[free] / ratios[free].sum()
        ratios[free] -= excess * free_weights

    ratios = ratios.clamp(min_prune, max_prune)

    return 1 - ratios
