from __future__ import annotations

import gc
import types
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from torch import Tensor, nn

from src.adapters.moe_adapter import MoEAdapter
from src.utils.delay_interrupt import delay_keyboard_interrupt

from src.utils.logger import logger

ACC = torch.float64


@dataclass
class RoutingImportance:
    raw: Tensor  # (n_layers, n_experts)  routed experts
    shared: Tensor  # (n_layers,)           always-on experts
    res_energy: Tensor  # (n_layers,)
    gate_mass: Tensor  # (n_layers, n_experts)
    load: Tensor  # (n_layers, n_experts)
    n_tokens: Tensor  # (n_layers,)
    moe_layers: list[int]
    used_fisher: bool

    @property
    def layer_mask(self) -> Tensor:
        mask = torch.zeros(self.raw.shape[0], dtype=torch.bool)
        mask[self.moe_layers] = True
        return mask

    @property
    def layer_scores_raw(self) -> Tensor:
        # return (self.raw.sum(dim=1) + self.shared).to(torch.float32)
        return (self.raw.sum(dim=1)).to(torch.float32)

    @property
    def layer_scores(self) -> Tensor:
        b = self.raw.sum(dim=1) + self.shared
        mask = self.layer_mask
        v = b[mask].clamp_min(1e-30).log()
        sd = v.std().nan_to_num(1.0).clamp_min(1e-8)
        out = torch.zeros_like(b)
        out[mask] = (v - v.mean()) / sd
        return out.to(torch.float32)

    @property
    def expert_scores(self) -> Tensor:
        s = self.raw.clamp_min(0)
        return (s / s.sum(dim=1, keepdim=True).clamp_min(1e-30)).to(torch.float32)

    @property
    def routed_frac(self) -> Tensor:
        m = self.gate_mass
        return (m / m.sum(dim=1, keepdim=True).clamp_min(1e-30)).to(torch.float32)

    def as_list(self) -> list[float]:
        return [float(v) for v in self.layer_scores]

    def summary(self, top: int = 5) -> str:
        scores, raw, shares = self.layer_scores, self.layer_scores_raw, self.expert_scores
        lines = [f"{'layer':>5} {'score':>8} {'B_l':>11} {'shared':>8} {'tokens':>10}  top experts"]
        for l in self.moe_layers:
            values, indices = torch.topk(shares[l], min(top, shares.shape[1]))
            row = " ".join(f"e{int(i)}:{float(v):.3f}" for v, i in zip(values, indices))
            share = float(self.shared[l]) / max(float(raw[l]), 1e-30)
            lines.append(
                f"{l:>5} {float(scores[l]):>8.3f} {float(raw[l]):>11.3e} {share:>8.3f} "
                f"{float(self.n_tokens[l]):>10.0f}  {row}"
            )
        return "\n".join(lines)


@dataclass
class _LayerModules:
    block: nn.Module
    moe: nn.Module
    routed: list[nn.Module]
    always_on: list[nn.Module]


class _Restore:
    def __init__(self) -> None:
        self._handles: list[Any] = []
        self._attrs: list[tuple[Any, str, Any, bool]] = []

    def hook(self, handle) -> None:
        self._handles.append(handle)

    def attr(self, obj, name: str, value) -> None:
        if obj is None or not hasattr(obj, name):
            return
        owned = name in getattr(obj, "__dict__", {})
        self._attrs.append((obj, name, getattr(obj, name) if owned else None, owned))
        setattr(obj, name, value)

    def __call__(self) -> None:
        for handle in self._handles:
            handle.remove()
        for obj, name, value, owned in reversed(self._attrs):
            setattr(obj, name, value) if owned else delattr(obj, name)
        self._handles.clear()
        self._attrs.clear()


def _moe_layers(adapter: MoEAdapter, target_layers: Optional[list[int]]) -> list[int]:
    candidates = target_layers if target_layers else range(adapter.n_layers)
    layers = [l for l in candidates if not adapter.dense(l)]
    if not layers:
        raise RuntimeError("No MoE layers found.")
    return layers


def _parent_map(model: nn.Module) -> dict[int, nn.Module]:
    parents: dict[int, nn.Module] = {}
    for module in model.modules():
        for child in module.children():
            parents[id(child)] = module
    return parents


def _child_of(block: nn.Module, descendant: nn.Module, parents: dict[int, nn.Module]) -> nn.Module:
    module = descendant
    while True:
        parent = parents.get(id(module))
        if parent is None:
            raise RuntimeError(f"{type(descendant).__name__} is not inside the decoder block.")
        if parent is block:
            return module
        module = parent


def _layout(adapter: MoEAdapter, moe_layers: list[int]) -> dict[int, _LayerModules]:
    parents = _parent_map(adapter.model)
    blocks = adapter.get_transformer_blocks()
    layout: dict[int, _LayerModules] = {}
    for layer_idx in moe_layers:
        block = blocks[layer_idx]
        routed = [
            parents[id(adapter.get_mlp_components(layer_idx, expert_idx).down_proj)]
            for expert_idx in range(adapter.n_experts_in_layer(layer_idx))
        ]
        moe = _child_of(block, routed[0], parents)
        seen = {id(m) for m in routed}
        always_on = [m for m in moe.children() if type(m) is type(routed[0]) and id(m) not in seen]
        layout[layer_idx] = _LayerModules(block, moe, routed, always_on)
    return layout


def _tensor(out):
    return out[0] if isinstance(out, tuple) else out


def _split(batch, size: int) -> list:
    if size <= 0:
        return [batch]
    if isinstance(batch, dict):
        n = len(next(v for v in batch.values() if torch.is_tensor(v)))
        return [
            {k: (v[i : i + size] if torch.is_tensor(v) else v) for k, v in batch.items()}
            for i in range(0, n, size)
        ]
    return list(torch.split(batch, size))


def _forward_activations(model, batch) -> None:
    base = getattr(model, "model", model)
    if isinstance(batch, dict):
        kwargs = {k: v for k, v in batch.items() if k != "labels"}
        base(**kwargs, use_cache=False)
    else:
        base(batch, use_cache=False)


def _forward_loss(model, batch):
    if isinstance(batch, dict):
        kwargs = dict(batch)
        kwargs.setdefault("labels", kwargs["input_ids"])
        return model(**kwargs, use_cache=False)
    return model(batch, labels=batch, use_cache=False)


def _patch_moe_forwards(
    adapter: MoEAdapter, layout: dict[int, _LayerModules], restore: _Restore
) -> None:
    patched_moe_forward = adapter.get_patched_moe_forward()
    for layer_idx, mods in layout.items():
        if "forward" not in mods.moe.__dict__:
            restore.attr(
                mods.moe,
                "forward",
                types.MethodType(patched_moe_forward(layer_idx=layer_idx, cov_x=None), mods.moe),
            )
        for expert in mods.routed + mods.always_on:
            restore.attr(expert, "routing_weights", None)


def _fisher_hook(layer_idx: int, fisher: dict[int, Tensor], hits: dict[int, int], chunk: int):
    def forward(module, inputs, output):
        y = _tensor(output)
        if not torch.is_tensor(y) or not y.requires_grad:
            return

        def backward(grad: Tensor) -> None:
            g = grad.detach()
            if g.dim() == 2:
                g = g.unsqueeze(0)
            dims = tuple(range(g.dim() - 1))
            total = None
            for s in range(0, g.shape[0], chunk):
                part = g[s : s + chunk].float().pow(2).sum(dim=dims, dtype=ACC)
                total = part if total is None else total + part
            fisher[layer_idx] += total.cpu()
            hits[layer_idx] += 1

        y.register_hook(backward)

    return forward


def _collect_fisher(
    adapter: MoEAdapter,
    batches: Iterable[Any],
    layout: dict[int, _LayerModules],
    dki: Optional[delay_keyboard_interrupt] = None,
    grad_chunk: int = 1,
    micro_batch: int = 1,
) -> dict[int, Tensor]:
    model = adapter.model
    blocks = adapter.get_transformer_blocks()
    fisher = {l: torch.zeros(adapter.d_model, dtype=ACC) for l in layout}
    hits = {l: 0 for l in layout}

    restore = _Restore()
    for layer_idx, mods in layout.items():
        restore.hook(
            mods.block.register_forward_hook(_fisher_hook(layer_idx, fisher, hits, grad_chunk))
        )

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    model.config.router_aux_loss_coef = 0.0
    model.config.router_jitter_noise = 0.0
    for block in blocks:
        block.training = True
    for mods in layout.values():
        if hasattr(mods.moe, "jitter_noise"):
            mods.moe.jitter_noise = 0
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    def cleanup():
        for block in blocks:
            block.training = False
        model.gradient_checkpointing_disable()
        model.disable_input_require_grads()
        restore()
        gc.collect()
        torch.cuda.empty_cache()

    n_seen = 0
    try:
        with torch.enable_grad():
            for batch in batches:
                for micro in _split(batch, micro_batch):
                    model.zero_grad()
                    outputs = _forward_loss(model, micro)
                    outputs.loss.backward()
                    del outputs
                n_seen += 1
                if dki and dki.interrupted:
                    raise KeyboardInterrupt
    finally:
        cleanup()

    dead = [l for l, h in hits.items() if h == 0]
    if dead:
        raise RuntimeError(f"No gradient reached layers {dead} over {n_seen} batches.")
    logger.info("Fisher collected over %d batches", n_seen)

    return {l: g / g.mean().clamp_min(1e-30) for l, g in fisher.items()}


def collect_importance(
    adapter: MoEAdapter,
    batches: Iterable[Any],
    fisher_batches: Optional[Iterable[Any]] = None,
    target_layers: Optional[list[int]] = None,
    gate_power: float = 2.0,
    fisher_micro_batch: int = 1,
    dki: Optional[delay_keyboard_interrupt] = None,
    verbose: bool = False,
) -> RoutingImportance:
    model = adapter.model
    moe_layers = _moe_layers(adapter, target_layers)
    layout = _layout(adapter, moe_layers)
    n_layers = adapter.n_layers
    n_experts = max(len(mods.routed) for mods in layout.values())

    fisher: dict[int, Tensor] = {}
    if fisher_batches is not None:
        logger.info("Collecting diagonal Fisher metric over %d MoE layers", len(moe_layers))
        fisher = _collect_fisher(
            adapter, fisher_batches, layout, dki=dki, micro_batch=fisher_micro_batch
        )

    importance = torch.zeros(n_layers, n_experts, dtype=ACC)
    shared = torch.zeros(n_layers, dtype=ACC)
    gate_mass = torch.zeros(n_layers, n_experts, dtype=ACC)
    load = torch.zeros(n_layers, n_experts, dtype=ACC)
    res_energy = torch.zeros(n_layers, dtype=ACC)
    n_tokens = torch.zeros(n_layers, dtype=ACC)

    metric_cache: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}
    dropped = 0

    def metric(layer_idx: int, device, dtype) -> Optional[Tensor]:
        g = fisher.get(layer_idx)
        if g is None:
            return None
        key = (layer_idx, device, dtype)
        if key not in metric_cache:
            metric_cache[key] = g.to(device=device, dtype=dtype)
        return metric_cache[key]

    def residual_hook(layer_idx: int):
        def hook(module, inputs):
            r = inputs[0].detach()
            res_energy[layer_idx] += r.pow(2).sum(dtype=ACC).cpu()
            n_tokens[layer_idx] += r.numel() // r.shape[-1]

        return hook

    def expert_hook(layer_idx: int, expert_idx: Optional[int]):
        def hook(module, inputs, output):
            nonlocal dropped
            y = _tensor(output).detach()
            y = y.reshape(-1, y.shape[-1])
            if y.shape[0] == 0:
                return

            w = getattr(module, "routing_weights", None)
            if w is None:
                g = torch.ones(y.shape[0], device=y.device, dtype=ACC)
            else:
                w = w.detach().reshape(-1)
                if w.numel() != y.shape[0]:
                    dropped += 1
                    return
                g = w.to(device=y.device, dtype=ACC)

            energy = y.pow(2)
            G = metric(layer_idx, y.device, y.dtype)
            if G is not None:
                energy = energy * G
            score = (energy.sum(dim=-1, dtype=ACC) * g.pow(gate_power)).sum().cpu()

            if expert_idx is None:
                shared[layer_idx] += score
            else:
                importance[layer_idx, expert_idx] += score
                gate_mass[layer_idx, expert_idx] += g.sum().cpu()
                load[layer_idx, expert_idx] += y.shape[0]

        return hook

    restore = _Restore()
    _patch_moe_forwards(adapter, layout, restore)
    for layer_idx, mods in layout.items():
        restore.hook(
            mods.block.post_attention_layernorm.register_forward_pre_hook(residual_hook(layer_idx))
        )
        for expert_idx, expert in enumerate(mods.routed):
            restore.hook(expert.register_forward_hook(expert_hook(layer_idx, expert_idx)))
        for expert in mods.always_on:
            restore.hook(expert.register_forward_hook(expert_hook(layer_idx, None)))

    model.eval()
    try:
        with torch.no_grad():
            for count, batch in enumerate(batches):
                _forward_activations(model, batch)
                if dki and dki.interrupted:
                    raise KeyboardInterrupt
                if verbose:
                    logger.info("importance batch %d", count + 1)
    finally:
        restore()
        gc.collect()
        torch.cuda.empty_cache()

    silent = [l for l in moe_layers if float(load[l].sum()) == 0.0]
    if silent:
        raise RuntimeError(f"No routed expert activations recorded for layers {silent}.")
    if dropped:
        logger.warning("Dropped %d expert calls whose routing weights did not match", dropped)

    denom = res_energy.clamp_min(1e-30)
    result = RoutingImportance(
        raw=importance / denom.unsqueeze(1),
        shared=shared / denom,
        res_energy=res_energy,
        gate_mass=gate_mass,
        load=load,
        n_tokens=n_tokens,
        moe_layers=moe_layers,
        used_fisher=bool(fisher),
    )
    if verbose:
        logger.info("\n%s", result.summary())
    return result
