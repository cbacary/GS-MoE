import numpy as np
import random

import types

import os

import gc

import torch
from torch.types import Tensor

from src.eval import load_calibration_texts
from src.utils.model import dtype_p, calib_device, d1, d2

from src.adapters.moe_adapter import MoEAdapter
from typing import Callable

from src.utils.delay_interrupt import delay_keyboard_interrupt

from src.utils.logger import logger


class _StopForward(Exception):
    pass


@torch.no_grad()
def calibrate_combined(
    adapter: MoEAdapter,
    n_samples: int,
    batch_size: int,
    target_layers: list[int] = [],
    dataset: str = "wikitext",
    dki: delay_keyboard_interrupt = None,
):
    for layer_idx in target_layers:
        if adapter.dense(layer_idx):
            print(f"removing {layer_idx} from target layers")
            target_layers.remove(layer_idx)

    model, tokenizer = adapter.model, adapter.tokenizer
    n_layers = adapter.n_layers
    n_experts = adapter.n_experts
    arch = adapter.arch
    adapter.fake_toks = False

    compute_nystrom = adapter.config.nys_to_fact_ratio != 0
    # compute_nystrom = True

    assert tokenizer is not None

    if not target_layers:
        target_layers = [i for i in range(n_layers)]
    logger.info(f"Calibrating on target_layers: {target_layers}")

    if adapter.calibs is None:
        adapter.calibs = load_calibration_texts(
            calib_size=n_samples,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            dataset=dataset,
        )

    if "tokens_per_exp" not in adapter.metrics.keys():
        adapter.metrics["fake_toks"] = [[0 for _ in range(n_experts)] for _ in range(n_layers)]
        adapter.metrics["tokens_per_exp"] = [[0 for _ in range(n_experts)] for _ in range(n_layers)]

    if "exp_routing_frequency" not in adapter.metrics.keys():
        adapter.metrics["exp_routing_frequency"] = [
            [0.0 for _ in range(n_experts)] for _ in range(n_layers)
        ]

    adapter.taylor_scores = [None] * adapter.n_layers

    calc_bi_scores = adapter.bi_scores is None
    if calc_bi_scores:
        adapter.bi_scores = [0.0 for _ in range(n_layers)]

    if not getattr(model.config, "output_hidden_states", False):
        model.config.output_hidden_states = True

    cov_mlp_list_fact = [None for _ in range(n_layers)]
    cov_mlp_list_nys = [None for _ in range(n_layers)]
    cov_x_list = [None for _ in range(n_layers)]
    for layer_idx in target_layers:

        # cov_x_list[i] = torch.zeros(adapter.d_model, adapter.d_model, dtype=dtype_p, device=d1)

        current_n_exps = adapter.n_experts_in_layer(layer_idx)
        if not adapter.dense(layer_idx):
            cov_x_list[layer_idx] = [
                torch.zeros(adapter.d_model, adapter.d_model, dtype=dtype_p, device=d1)
                for _ in range(current_n_exps)
            ]

            cov_mlp_list_fact[layer_idx] = [
                torch.zeros(
                    adapter.d_model,
                    adapter.d_model,
                    dtype=dtype_p,
                    device=d1,
                )
                for _ in range(current_n_exps)
            ]

        cov_mlp_list_nys[layer_idx] = [
            torch.zeros(
                (
                    1
                    if not compute_nystrom
                    else adapter.get_mlp_int_rank(layer_idx=layer_idx, expert_idx=expert_idx)
                ),
                (
                    1
                    if not compute_nystrom
                    else adapter.get_mlp_int_rank(layer_idx=layer_idx, expert_idx=expert_idx)
                ),
                dtype=dtype_p,
                device=calib_device,
            )
            for expert_idx in range(current_n_exps)
        ]

    unregister_callbacks: list[Callable] = []
    hidden_state_cache = [[] for _ in range(adapter.n_layers)]
    c1 = register_hooks(
        adapter,
        target_layers,
        cov_mlp_list_nys,
        cov_mlp_list_fact,
        cov_x_list,
        calc_bi_scores,
        compute_nystrom,
        hidden_state_cache=hidden_state_cache,
    )
    unregister_callbacks.append(c1)

    handles = []

    model_kwargs = {
        "output_hidden_states": False,
    }
    # if adapter.arch == "mixtral":
    #     model_kwargs["output_router_logits"] = False

    n_texts = 0
    model.eval()
    adapter.fake_toks = False
    for count, batch in enumerate(adapter.calibs):
        n_texts += len(batch)
        # batch_idx_box["i"] = count
        model.zero_grad()
        try:
            outputs = model.model(
                batch,
                use_cache=False,
                **model_kwargs,
                # output_hidden_states=False,
                # output_router_logits=False,
            )
            # hidden_states = outputs.hidden_states
            del outputs
        except Exception:
            # only needed if in a notebook, otherwise you'll get oom on the 2nd run
            for c in unregister_callbacks:
                c()
            for h in handles:
                h.remove()
            # for layer_idx in skipped_layers:
            #     blocks[layer_idx].forward = blocks[layer_idx].pre_skip_forward
            #     del blocks[layer_idx].pre_skip_forward
            del (
                cov_mlp_list_nys,
                cov_mlp_list_fact,
                cov_x_list,
                adapter.nystrom_scores,
                adapter.taylor_scores,
            )
            raise

        if dki and dki.interrupted:
            for c in unregister_callbacks:
                c()
            for h in handles:
                h.remove()
            # for layer_idx in skipped_layers:
            #     blocks[layer_idx].forward = blocks[layer_idx].pre_skip_forward
            #     del blocks[layer_idx].pre_skip_forward
            del (
                cov_mlp_list_nys,
                cov_mlp_list_fact,
                cov_x_list,
                adapter.nystrom_scores,
                adapter.taylor_scores,
            )
            raise KeyboardInterrupt

        if count % 4 == 0:
            logger.info(f"Completed {count} of {len(adapter.calibs)} batches")

    import random

    fake_toks = adapter.metrics["fake_toks"]
    toks = adapter.metrics["tokens_per_exp"]
    adapter.fake_toks = True
    if adapter.arch == "qwen2_moe":
        logger.info(f"Saturating empty experts with fake text")
        for layer_idx in target_layers:
            experts = adapter.get_experts(layer_idx)
            hs = hidden_state_cache[layer_idx]
            if not hs:
                continue
            random.seed(adapter.config.seed)
            sample = random.choice(hs)
            for expert_idx, expert in enumerate(experts):
                min_toks = ((adapter.get_mlp_int_rank(layer_idx, expert_idx) * 2) // 2048) * 2048
                if toks[layer_idx][expert_idx] < min_toks:

                    d_model = sample.shape[-1]
                    batch = sample.reshape(sample.shape[0] * sample.shape[1], -1)[:min_toks, :]
                    batch = batch.reshape(batch.shape[0] // 2048, -1, d_model)

                    r = batch.reshape(-1, d_model).to(
                        dtype=dtype_p, device=cov_x_list[layer_idx][expert_idx].device
                    )
                    cov_x_list[layer_idx][expert_idx] += r.T @ r
                    del r

                    for hs in batch:
                        expert(hs)

                    fake_toks[layer_idx][expert_idx] += min_toks
    adapter.fake_toks = False

    del hidden_state_cache

    for h in handles:
        h.remove()

    for unregister_callback in unregister_callbacks:
        unregister_callback()
    if calc_bi_scores:
        for layer_idx in range(n_layers):
            # adapter.nystrom_scores[layer_idx] /= len(adapter.calibs)
            adapter.bi_scores[layer_idx] /= len(adapter.calibs)

    for layer_idx in target_layers:
        for expert_idx in range(adapter.n_experts_in_layer(layer_idx)):
            layer_toks = adapter.metrics["tokens_per_exp"][layer_idx]
            exp_toks = layer_toks[expert_idx]

            yielded_toks = exp_toks + adapter.metrics["fake_toks"][layer_idx][expert_idx]
            if yielded_toks == 0:
                print(f"Big problem fake toks is zero")
            cov_mlp_list_nys[layer_idx][expert_idx] /= yielded_toks

            if exp_toks == 0:
                print(f"l {layer_idx} exp {expert_idx} has 0 tokens")
                continue

            cov_x_list[layer_idx][expert_idx] /= yielded_toks
            cov_x_list[layer_idx][expert_idx] = cov_x_list[layer_idx][expert_idx].to(
                device=calib_device
            )

            cov_mlp_list_fact[layer_idx][expert_idx] /= yielded_toks
            cov_mlp_list_fact[layer_idx][expert_idx] = cov_mlp_list_fact[layer_idx][expert_idx].to(
                device=calib_device
            )

    gc.collect()
    torch.cuda.empty_cache()

    adapter.taylor_scores = [
        (
            [score.to(device=calib_device, dtype=dtype_p) for score in layer_scores]
            if layer_scores is not None
            else None
        )
        for layer_scores in adapter.taylor_scores
    ]

    return (
        cov_mlp_list_fact,
        cov_mlp_list_nys,
        cov_x_list,
        adapter.nystrom_scores,
        adapter.taylor_scores,
    )


def register_hooks(
    adapter: MoEAdapter,
    target_layers: list[int],
    cov_mlp_list_nys: list,
    cov_mlp_list_fact: list,
    cov_x_list: list[Tensor | None],
    calc_bi_scores: bool,
    compute_nystrom: bool,
    hidden_state_cache,
) -> Callable[[], None]:

    patched_moe_forward = adapter.get_patched_moe_forward()
    patched_decoder_layer_forward = adapter.get_patched_decoder_layer_forward()

    handles = []

    blocks = adapter.get_transformer_blocks()
    patched_moes = []
    for layer_idx in target_layers:

        block = blocks[layer_idx]

        moe = adapter.get_moe_block(layer_idx)
        experts = moe.experts
        moe.old_forward = moe.forward

        moe.forward = types.MethodType(
            patched_moe_forward(
                layer_idx=layer_idx, cov_x=cov_x_list, hidden_state_cache=hidden_state_cache
            ),
            moe,
        )
        patched_moes.append(moe)

        for expert_idx, expert in enumerate(experts):
            comps = adapter.get_mlp_components(layer_idx, expert_idx)
            down_proj = comps.down_proj
            handle = down_proj.register_forward_hook(
                post_forward_down_hook(adapter, layer_idx, cov_mlp_list_fact, expert_idx, expert)
            )
            handles.append(handle)
            handle = down_proj.register_forward_pre_hook(
                pre_forward_down_hook(
                    adapter,
                    layer_idx,
                    cov_mlp_list_nys,
                    expert_idx=expert_idx,
                    expert_module=expert,
                    compute_nystrom=compute_nystrom,
                )
            )
            handles.append(handle)

    for layer_idx in range(adapter.n_layers):
        block = blocks[layer_idx]
        block.old_forward = block.forward

        block.forward = types.MethodType(
            # patched_mixtral_decoder_layer_forward(
            patched_decoder_layer_forward(
                bi_scores=adapter.bi_scores if calc_bi_scores else None,
                # cov_x=cov_x_list,
                cov_x=None,
                layer_idx=layer_idx,
            ),
            block,
        )

    def unregister_hooks_patches():
        for layer_idx, block in enumerate(adapter.get_transformer_blocks()):
            if hasattr(block, "old_forward"):
                block.forward = block.old_forward
                del block.old_forward
        for handle in handles:
            handle.remove()
        for moe in patched_moes:
            moe.forward = moe.old_forward
            del moe.old_forward
        patched_moes.clear()

    return unregister_hooks_patches


@torch.no_grad()
def pre_forward_down_hook(
    adapter, layer_idx, cov_mlp_list_nys, expert_idx=None, expert_module=None, compute_nystrom=True
):
    @torch.no_grad()
    def hook(module, input: tuple[torch.Tensor]):
        x_input = input[0]
        w = 1.0
        fake_tok = getattr(adapter, "fake_toks", False)
        if expert_module is not None and hasattr(expert_module, "routing_weights") and not fake_tok:
            # w: [batch_size, 1]
            w = expert_module.routing_weights.view(-1, 1).sqrt()
            x_input = x_input * w

        H = (
            x_input.detach()
            .to(device=calib_device, non_blocking=True)
            .to(dtype=dtype_p)
            .view(-1, x_input.size(-1))
        )

        if not fake_tok:
            adapter.metrics["tokens_per_exp"][layer_idx][expert_idx] += H.shape[0]
        else:
            print(f"Getting fake data")

        if compute_nystrom:
            cov_mlp_list_nys[layer_idx][expert_idx] += H.T @ H

        return None

    return hook


@torch.no_grad()
def post_forward_down_hook(adapter, layer_idx, cov_mlp_list_fact, expert_idx, expert_module=None):
    @torch.no_grad()
    def hook(module, args, output: tuple[torch.Tensor]):
        """
        Computes [d, d] activation for each expert. Expects cov_mlp_list_fact of list[n_layers, n_experts, Tensor[d, d]]
        """

        if getattr(adapter, "fake_toks", False):
            return

        x_input = output
        w = 1.0
        if expert_module is not None and hasattr(expert_module, "routing_weights"):
            w = expert_module.routing_weights.view(-1, 1).sqrt()
            x_input = x_input * w
        H = (
            x_input.detach()
            .to(device=cov_mlp_list_fact[layer_idx][expert_idx].device)
            .to(dtype=dtype_p)
            .view(-1, x_input.size(-1))
        )

        # self.metrics["tokens_per_exp"][layer_idx][expert_idx] += H.shape[0]
        cov_mlp_list_fact[layer_idx][expert_idx] += H.T @ H
        return None

    return hook
