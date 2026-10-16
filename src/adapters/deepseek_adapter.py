from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .llama_adapter import LlamaAdapter
from .model_adapter import AttentionComponents, MLPComponents, MLPTensors
from .moe_adapter import MoEAdapter
from src.utils.model import dtype_p

from src.utils.logger import logger


class DeepseekAdapter(MoEAdapter):
    """Adapter for DeepSeek models with a mixture of dense and routed-MoE layers."""

    @property
    def arch(self) -> str:
        return "deepseek"

    @property
    def n_experts(self) -> int:
        return getattr(self.model_config, "n_routed_experts", 0)

    @property
    def d_int(self) -> int:
        return getattr(self.model_config, "moe_intermediate_size", None) or getattr(
            self.model_config, "intermediate_size", None
        )

    def get_transformer_blocks(self) -> nn.ModuleList:
        return self.model.model.layers

    def dense(self, layer_idx: int) -> bool:
        return not hasattr(self.get_transformer_blocks()[layer_idx].mlp, "experts")

    def n_experts_in_layer(self, layer_idx: int) -> int:
        return 1 if self.dense(layer_idx) else self.n_experts

    def dense(self, layer_idx: int) -> bool:
        # deepseek moe 16b base:
        return layer_idx == 0

    def get_moe_block(self, layer_idx: int) -> nn.Module:
        # if self.dense(layer_idx):
        #     raise ValueError("Dense DeepSeek layers do not have a routed MoE block")
        return self.get_transformer_blocks()[layer_idx].mlp

    def get_experts(self, layer_idx):
        return self.get_moe_block(layer_idx).experts

    def get_n_inner(
        self,
        block: nn.Module | None = None,
        expert_idx: int | None = None,
        layer_idx: int | None = None,
    ) -> int:
        if block is None and layer_idx is not None:
            block = self.get_moe_block(layer_idx)
        if self.dense(layer_idx):
            return block.up_proj.out_features

        return block.experts[expert_idx or 0].up_proj.out_features

    def get_mlp_rank(self, layer_idx: int, expert_idx: int | None = None) -> int:
        return self.get_n_inner(layer_idx=layer_idx, expert_idx=expert_idx)

    def get_mlp_d_rank(self, layer_idx: int, expert_idx: int | None = None) -> int:
        return self.get_mlp_components(layer_idx, expert_idx).down_proj.out_features

    def get_mlp_int_rank(self, layer_idx: int, expert_idx: int | None = None) -> int:
        return self.get_mlp_components(layer_idx, expert_idx).up_proj.out_features

    def get_mlp_components(self, layer_idx: int, expert_idx: int | None = None) -> MLPComponents:
        block = self.get_transformer_blocks()[layer_idx]
        if self.dense(layer_idx):
            mlp = block.mlp
        else:
            if expert_idx is None:
                raise ValueError("expert_idx required for DeepSeek MoE layer")
            mlp = block.mlp.experts[expert_idx]

        return MLPComponents(
            block=block,
            up_proj=mlp.up_proj,
            down_proj=mlp.down_proj,
            gate_proj=mlp.gate_proj,
        )

    def get_mlp_tensors(self, layer_idx: int, expert_idx: int | None = None) -> MLPTensors:
        components = self.get_mlp_components(layer_idx, expert_idx)
        return MLPTensors(
            up_proj=components.up_proj.weight.data,
            down_proj=components.down_proj.weight.data,
            gate_proj=components.gate_proj.weight.data,
        )

    def replace_mlp_layers(
        self,
        layer_idx: int,
        new_up: nn.Module,
        new_down: nn.Module,
        new_gate: nn.Module | None = None,
        expert_idx: int | None = None,
    ) -> None:
        if self.dense(layer_idx):
            mlp = self.get_transformer_blocks()[layer_idx].mlp
        else:
            if expert_idx is None:
                raise ValueError("expert_idx required for DeepSeek MoE layer")
            mlp = self.get_transformer_blocks()[layer_idx].mlp.experts[expert_idx]

        mlp.up_proj = new_up
        mlp.down_proj = new_down
        if new_gate is not None:
            mlp.gate_proj = new_gate

    def get_attn_components(self, layer_idx: int) -> AttentionComponents:
        block = self.get_transformer_blocks()[layer_idx]
        return AttentionComponents(
            block=block,
            q_proj=block.self_attn.q_proj,
            k_proj=block.self_attn.k_proj,
            v_proj=block.self_attn.v_proj,
            o_proj=block.self_attn.o_proj,
        )

    def replace_attn_layers(
        self,
        layer_idx: int,
        new_q: Optional[nn.Module],
        new_k: Optional[nn.Module],
        new_v: Optional[nn.Module],
        new_o: Optional[nn.Module],
    ) -> None:
        attention = self.get_transformer_blocks()[layer_idx].self_attn
        if new_q is not None:
            attention.q_proj = new_q
        if new_k is not None:
            attention.k_proj = new_k
        if new_v is not None:
            attention.v_proj = new_v
        if new_o is not None:
            attention.o_proj = new_o

    def get_qk_weights(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        attention = self.get_transformer_blocks()[layer_idx].self_attn
        return attention.q_proj.weight, attention.k_proj.weight

    def get_vo_weights(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        attention = self.get_transformer_blocks()[layer_idx].self_attn
        return attention.v_proj.weight, attention.o_proj.weight

    def get_ModelForCausalLM(self):
        from src.patchers.deepseek_rebuild import DeepseekForCausalLM

        return DeepseekForCausalLM

    def get_patched_decoder_layer_forward(self):
        return patched_deepseek_decoder_layer_forward

    def get_patched_moe_forward(self):
        return patched_deepseek_moe_forward


class DeepseekV2Adapter(DeepseekAdapter):
    def get_ModelForCausalLM(self):
        from src.patchers.deepseekv2_rebuild import DeepseekV2ForCausalLM

        return DeepseekV2ForCausalLM

    def get_patched_moe_forward(self):
        return patched_deepseekv2_moe_forward

    def get_qk_weights(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        attention = self.get_transformer_blocks()[layer_idx].self_attn
        query = attention.q_proj if attention.q_lora_rank is None else attention.q_b_proj
        return query.weight, attention.kv_a_proj_with_mqa.weight

    def get_vo_weights(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        attention = self.get_transformer_blocks()[layer_idx].self_attn
        return attention.kv_b_proj.weight, attention.o_proj.weight


def patched_deepseek_decoder_layer_forward(
    bi_scores: list[float] | None,
    cov_x: list[Tensor] | None,
    layer_idx: int,
):
    """DeepSeekDecoderLayer forward preserving its legacy cache and attention contract."""

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if bi_scores is not None:
            x_in = residual.detach().to(dtype=dtype_p)
            x_out = hidden_states.detach().to(dtype=dtype_p)
            score = (1 - torch.cosine_similarity(x_in, x_out, dim=-1)).mean().item()
            bi_scores[layer_idx] += score

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    return forward


def patched_deepseek_moe_forward(
    layer_idx: int,
    cov_x: list[Tensor] | None = None,
    taylor: bool = False,
    hidden_state_cache: list[Tensor] = None,  # len == n_calibs // batch_size
):
    """DeepSeek MoE forward preserving the native routing and expert execution path."""
    from src.patchers.deepseek_rebuild import AddAuxiliaryLoss

    def forward(self, hidden_states: Tensor) -> Tensor:
        identity = hidden_states
        original_shape = hidden_states.shape
        topk_indices, topk_weights, auxiliary_loss = self.gate(hidden_states)
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_indices = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)

        if cov_x is not None:
            repeated_hidden_states = flat_hidden_states.repeat_interleave(
                self.num_experts_per_tok,
                dim=0,
            )
            for expert_idx in range(len(self.experts)):
                selected = flat_indices == expert_idx
                expert_input = repeated_hidden_states[selected]
                H = expert_input.detach().to(
                    device=cov_x[layer_idx][expert_idx].device,
                    dtype=dtype_p,
                )
                w = (
                    flat_weights[selected]
                    .detach()
                    .to(
                        device=cov_x[layer_idx][expert_idx].device,
                        dtype=dtype_p,
                    )
                )
                cov_x[layer_idx][expert_idx] += (H * w.unsqueeze(-1)).T @ H

        if self.training:
            repeated_hidden_states = flat_hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            expert_outputs = torch.empty_like(repeated_hidden_states)

            for expert_idx, expert in enumerate(self.experts):
                selected = flat_indices == expert_idx
                expert_input = repeated_hidden_states[selected]
                # Pass weights to expert layer for calibration hooks: same rows,
                # same order as `expert_input`, hence as the down_proj input.
                expert.routing_weights = flat_weights[selected].detach().view(-1, 1)
                expert_outputs[selected] = expert(expert_input)

            routed_output = (
                expert_outputs.view(*topk_weights.shape, -1) * topk_weights.unsqueeze(-1)
            ).sum(dim=1)
            routed_output = routed_output.view(*original_shape)
            routed_output = AddAuxiliaryLoss.apply(routed_output, auxiliary_loss)
        else:
            # moe_infer feeds expert i the slice `argsort(flat_indices)[start_i:end_i]`,
            # so regrouping the weights the same way lines them up with the rows each
            # expert -- and so each down_proj calibration hook -- actually sees.
            order = flat_indices.argsort()
            counts = flat_indices.bincount(minlength=len(self.experts)).tolist()
            for expert, w in zip(self.experts, flat_weights[order].detach().split(counts)):
                expert.routing_weights = w.view(-1, 1)

            routed_output = self.moe_infer(
                flat_hidden_states,
                flat_indices,
                topk_weights.view(-1, 1),
            ).view(*original_shape)

        if self.config.n_shared_experts is not None:
            routed_output = routed_output + self.shared_experts(identity)
        return routed_output

    return forward


def patched_deepseekv2_moe_forward(
    layer_idx: int,
    cov_x: list[Tensor] | None = None,
    taylor: bool = False,
):
    """DeepSeek MoE forward preserving the native routing and expert execution path."""
    from src.patchers.deepseekv2_rebuild import AddAuxiliaryLoss

    def forward(self, hidden_states: Tensor) -> Tensor:
        identity = hidden_states
        original_shape = hidden_states.shape
        topk_indices, topk_weights, auxiliary_loss = self.gate(hidden_states)
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_indices = topk_indices.reshape(-1)

        if cov_x is not None:
            repeated_hidden_states = flat_hidden_states.repeat_interleave(
                self.num_experts_per_tok,
                dim=0,
            )
            for expert_idx in range(len(self.experts)):
                expert_input = repeated_hidden_states[flat_indices == expert_idx]
                H = expert_input.detach().to(
                    device=cov_x[layer_idx][expert_idx].device,
                    dtype=dtype_p,
                )
                cov_x[layer_idx][expert_idx] += H.T @ H

        if self.training:
            repeated_hidden_states = flat_hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            expert_outputs = torch.empty_like(repeated_hidden_states)

            for expert_idx, expert in enumerate(self.experts):
                selected = flat_indices == expert_idx
                expert_input = repeated_hidden_states[selected]
                expert_outputs[selected] = expert(expert_input)

            routed_output = (
                expert_outputs.view(*topk_weights.shape, -1) * topk_weights.unsqueeze(-1)
            ).sum(dim=1)
            routed_output = routed_output.view(*original_shape)
            routed_output = AddAuxiliaryLoss.apply(routed_output, auxiliary_loss)
            routed_output = routed_output.to(flat_hidden_states.dtype).view(*original_shape)

        else:
            routed_output = self.moe_infer(
                flat_hidden_states,
                topk_indices,
                topk_weights,
            ).view(*original_shape)

        if self.config.n_shared_experts is not None:
            routed_output = routed_output + self.shared_experts(identity)
        return routed_output

    return forward
