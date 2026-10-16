import torch

from src.adapters.moe_adapter import MoEAdapter
from src.eval import load_calibration_texts
from src.utils.compression import allocate_global_sparsity

from src.compression.layer_scores import collect_importance

import os


def allocate_budgets(adapter: MoEAdapter, compression_ratio: float):
    if adapter.nystrom_scores is None:
        # p = f"/tmp/layer_scores_raw-{adapter.arch}.pt"
        # if os.path.exists(p):
        # adapter.nystrom_scores = torch.load(p)
        # else:
        clbs = load_calibration_texts(
            adapter.config.calib_size,
            adapter.model,
            adapter.tokenizer,
            adapter.config.calibs_batch_size,
            dataset="c4",
            split="train",
        )

        fisher_batches = None
        print("Collecting layer importance scores")
        s = collect_importance(adapter, clbs, fisher_batches=fisher_batches)
        adapter.nystrom_scores = s.layer_scores_raw
        # torch.save(s.layer_scores_raw, p)
    nystrom_scores = adapter.nystrom_scores

    if adapter.arch == "deepseek":
        nystrom_scores = nystrom_scores[1:]

    nystrom_scores = torch.log10(nystrom_scores)

    print(f"nys_scores = ", nystrom_scores)

    disable_down = adapter.config.disable_down
    disable_up = adapter.config.disable_up
    n_shared_mats = 3.0
    if disable_down:
        n_shared_mats -= 1.0
    if disable_up:
        n_shared_mats -= 2.0
    n_shared_mats /= 3.0

    d_int, d_model = adapter.d_int, adapter.d_model
    n_experts = adapter.n_experts

    shared_cost = adapter.config.n_shared_projs * n_shared_mats * (d_model / (n_experts * d_int))
    nys_keep = (1 - compression_ratio) ** adapter.config.nys_to_fact_ratio
    fact_keep = (1 - compression_ratio) / (nys_keep + shared_cost)

    nys_compression_ratio = 1 - nys_keep
    fact_compression_ratio = 1 - fact_keep

    print(f"fact compression ratio = {fact_compression_ratio}")
    print(f"nys compression ratio = {nys_compression_ratio}")

    nys_keep_ratios = allocate_global_sparsity(
        nystrom_scores,
        compression_ratio=nys_compression_ratio,
        adapter=adapter,
        smoothing=adapter.config.nystrom_layer_smoothing,
    )

    fact_keep_ratios = allocate_global_sparsity(
        nystrom_scores,
        compression_ratio=fact_compression_ratio,
        adapter=adapter,
        smoothing=adapter.config.factorization_layer_smoothing,
    )

    # restore budget so model gets compressed exactly to 0.6
    if adapter.config.nys_to_fact_ratio != 1.0 and adapter.config.nys_to_fact_ratio != 0.0:
        # just realloc fact
        A = (fact_keep_ratios * nys_keep_ratios).mean()
        B = fact_keep_ratios.mean()
        beta = (1 - compression_ratio) / (A + shared_cost * B)
        fact_keep_ratios = fact_keep_ratios * beta

        # if you  want to dist between fact AND nys
        # b = shared_cost * B
        # beta = (-b + torch.sqrt(b * b + 4 * A * (1 - compression_ratio))) / (2 * A)
        # fact_keep_ratios = fact_keep_ratios * beta
        # nys_keep_ratios = nys_keep_ratios * beta
    elif adapter.config.nys_to_fact_ratio == 0.0:
        nys_keep_ratios = torch.as_tensor([1.0] * len(nys_keep_ratios))
    else:
        fact_keep_ratios = torch.as_tensor([1.0] * len(fact_keep_ratios))
        shared_cost = 0.0

    achieved = (fact_keep_ratios * (nys_keep_ratios + shared_cost)).mean()

    if adapter.arch == "deepseek":
        nys_keep_ratios = torch.as_tensor([0.0] + nys_keep_ratios.tolist())
        fact_keep_ratios = torch.as_tensor([0.0] + fact_keep_ratios.tolist())

    print(f"fact_keep_ratios = {fact_keep_ratios}")
    print(f"nys_keep_ratios = {nys_keep_ratios}")
    print(f"Actual compression = {achieved}")

    adapter.metrics["nystrom_scores"] = nystrom_scores
    adapter.metrics["fact_keep_ratios"] = fact_keep_ratios.tolist()
    adapter.metrics["nys_keep_ratios"] = nys_keep_ratios.tolist()

    return nys_keep_ratios, fact_keep_ratios, nystrom_scores
