import os
import gc
import json

import torch
import optuna
import math
from src.utils.model import reload_compressed_model, save_compressed_model
from src.adapters.moe_adapter import MoEAdapter

from src.utils.logger import logger

from src.experiments.layer_shards import is_shard_worker
from src.calibration import calibrate_combined
from src.utils.compression import dtype_p, d1
from src.compression.compress import compress_mlp_fact_nys

from src.compression.allocator import allocate_budgets

ALPACA = False


@torch.no_grad()
def combined_compression(
    adapter: MoEAdapter,
    compression_ratio: float,
    trial: optuna.Trial,
    save_dir: str,
    start_layer=0,
    end_layer=None,
):
    global decay_scores

    config = adapter.config

    adapter.config = config

    n_layers = adapter.n_layers
    end_layer = n_layers if end_layer is None or end_layer < 0 else min(end_layer, n_layers)
    layers_per_step = adapter.config.layers_per_compression_step
    if start_layer == 0 and adapter.arch == "deepseek":
        start_layer = 1

    from src.eval import load_calibration_texts

    if adapter.calibs is None:
        adapter.calibs = load_calibration_texts(
            calib_size=config.calib_size,
            model=adapter.model,
            tokenizer=adapter.tokenizer,
            batch_size=config.calibs_batch_size,
            dataset=config.dataset,
        )

    nys_keep_ratios, fact_keep_ratios, nystrom_scores = allocate_budgets(adapter, compression_ratio)

    for start_idx in range(start_layer, end_layer, layers_per_step):
        end_idx = min(end_layer, start_idx + layers_per_step)
        target_layers = list(range(start_idx, end_idx))

        cov_mlp_fact, cov_mlp_nys, cov_x_list, nystrom_scores, taylor_scores = get_activations(
            adapter, start_idx, end_idx
        )

        """
        for deepseek models:
        nystrom_scores shape: n_layers
        bi_scores shape: n_layers
        """

        bi_scores = adapter.bi_scores

        adapter.metrics["bi_scores"] = bi_scores

        torch.cuda.empty_cache()
        gc.collect()
        adapter.model.cuda()
        adapter.model.eval()
        logger.info("Beginning compression...")

        compress_mlp_fact_nys(
            adapter=adapter,
            cov_x=cov_x_list,
            cov_mlp_fact=cov_mlp_fact,
            cov_mlp_nys=cov_mlp_nys,
            fact_keep_ratios=fact_keep_ratios,
            nys_keep_ratios=nys_keep_ratios,
            target_layers=target_layers,
            taylor_scores=taylor_scores,
        )

        logger.info(f"Finished compressing layers {target_layers}.")
        # print(f"Router reconstruction cost = {router_reconstruction_cost / (1000**3)}GB")

        del cov_x_list, cov_mlp_fact, cov_mlp_nys
        torch.cuda.empty_cache()
        gc.collect()

        if start_idx + layers_per_step >= n_layers:
            break

    adapter.save_metrics()

    from src.compression.compress import spectrals

    if not is_shard_worker(config):
        adapter.convert_model(os.path.expandvars("$TMPDIR/compressed_layers/"))
        adapter.patch(nystrom=False, flipped_down=adapter.config.flipped_down)

        if adapter.config.save:
            save_compressed_model(
                adapter,
                None,
                save_dir=save_dir,
                source_model_name=config.model,
            )
        else:
            from src.utils.model import rebuild_in_memory_meta

            rebuild_in_memory_meta(adapter, adapter.get_ModelForCausalLM())

    # del adapter.model
    # del adapter.tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # adapter.save_metrics()

    return adapter


"""
Save activations as FP32 and only keep upper triangle since symmetric.
FP32 acts had near-identical performance when restored versus using og fp64 acts.
"""


def _unpack1(t, device):
    n = (math.isqrt(8 * t.numel() + 1) - 1) // 2
    assert n * (n + 1) // 2 == t.numel(), "not a packed upper triangle"
    C = torch.zeros(n, n, dtype=torch.float32, device=device)
    C[torch.ones(n, n, dtype=torch.bool, device=device).triu()] = t.to(device, torch.float32)
    C = C + C.T
    C.diagonal().mul_(0.5)
    del t
    return C.to(dtype=dtype_p)


def pack(covs):
    out = []
    for layer in covs:
        if layer is None:
            continue
        n = layer[0].shape[0]
        r, c = torch.triu_indices(n, n, device=layer[0].device)
        out.append([C[r, c].float() for C in layer])
    return out


def unpack(packed, device, n_layers, start_idx, end_idx):
    out = [None] * n_layers
    for count, l_idx in enumerate(range(start_idx, end_idx)):
        out[l_idx] = [_unpack1(t, device) for t in packed[count]]
    del packed
    return out


def load_compressed_acts(adapter: MoEAdapter, acts_pth: str, start_idx: int, end_idx: int):
    acts = torch.load(acts_pth, map_location=d1)
    cov_mlp_fact, cov_mlp_nys, cov_x_list = (
        unpack(a, d1, adapter.n_layers, start_idx, end_idx) for a in acts[:3]
    )
    nystrom_scores, taylor_scores, toks_p_exp, f_p_exp = acts[3:]
    adapter.metrics["tokens_per_exp"] = toks_p_exp
    adapter.metrics["fake_toks"] = f_p_exp

    del acts

    torch.cuda.empty_cache()
    gc.collect()

    return cov_mlp_fact, cov_mlp_nys, cov_x_list, nystrom_scores, taylor_scores


def save_compressed_acts(
    adapter: MoEAdapter,
    acts_pth: str,
    cov_mlp_fact,
    cov_mlp_nys,
    cov_x_list,
    nystrom_scores,
    taylor_scores,
):
    os.makedirs(os.path.dirname(acts_pth), exist_ok=True)
    to_save = (
        pack(cov_mlp_fact),
        pack(cov_mlp_nys),
        pack(cov_x_list),
        nystrom_scores,
        taylor_scores,
        adapter.metrics["tokens_per_exp"],
        adapter.metrics["fake_toks"],
    )
    torch.save(to_save, acts_pth + ".tmp")
    os.replace(acts_pth + ".tmp", acts_pth)
    del to_save


def get_activations(adapter: MoEAdapter, start_idx: int, end_idx: int):
    config = adapter.config
    c_sv_pth = config.compressed_acts_save_path
    acts_pth = os.path.join(c_sv_pth, f"layer{start_idx}_{end_idx}.pt")

    if os.path.exists(acts_pth):
        return load_compressed_acts(adapter, acts_pth, start_idx, end_idx)

    cov_mlp_fact, cov_mlp_nys, cov_x_list, nystrom_scores, taylor_scores = calibrate_combined(
        adapter,
        n_samples=config.calib_size,
        batch_size=int(config.calibs_batch_size),
        target_layers=list(range(start_idx, end_idx)),
        dataset=config.dataset,
    )
    if c_sv_pth:
        save_compressed_acts(
            adapter,
            acts_pth,
            cov_mlp_fact,
            cov_mlp_nys,
            cov_x_list,
            nystrom_scores,
            taylor_scores,
        )

    return cov_mlp_fact, cov_mlp_nys, cov_x_list, nystrom_scores, taylor_scores
