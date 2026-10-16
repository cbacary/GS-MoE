import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from transformers.cache_utils import Cache

from .deepseek_adapter import DeepseekAdapter
from src.utils.model import dtype_p

from src.utils.logger import logger


class Qwen2MoeAdapter(DeepseekAdapter):
    """Adapter for Qwen2MoE models.

    Qwen2MoE shares DeepSeek's module layout -- `block.mlp` holding an `experts` ModuleList
    of `up_proj`/`gate_proj`/`down_proj` MLPs, plus a standard q/k/v/o attention -- so every
    MLP and attention accessor is inherited unchanged. Only the config names, the dense
    layer rule, and the two calibration forwards differ.
    """

    @property
    def arch(self) -> str:
        return "qwen2_moe"

    @property
    def n_experts(self) -> int:
        return getattr(self.model.config, "num_experts", 0)

    def dense(self, layer_idx: int) -> bool:
        return False

    def get_ModelForCausalLM(self):
        from src.patchers.qwen2_moe_rebuild import Qwen2MoeForCausalLM

        return Qwen2MoeForCausalLM

    def get_experts(self, layer_idx):
        moe_block = self.get_moe_block(layer_idx)
        return moe_block.experts

    def get_patched_decoder_layer_forward(self):
        return patched_qwen2_moe_decoder_layer_forward

    def get_patched_moe_forward(self):
        return patched_qwen2_moe_forward


def patched_qwen2_moe_decoder_layer_forward(
    bi_scores: list[float] | None,
    cov_x: list[Tensor] | None,
    layer_idx: int,
):
    """Qwen2MoeDecoderLayer forward preserving its cache and router-logits contract."""

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        if bi_scores is not None:
            x_in: Tensor = residual.detach().to(dtype=dtype_p)
            x_out: Tensor = hidden_states.detach().to(dtype=dtype_p)
            score = (1 - torch.cosine_similarity(x_in, x_out, dim=-1)).mean().item()
            bi_scores[layer_idx] += score
            del x_in, x_out

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

    return forward


def patched_qwen2_moe_forward(
    layer_idx: int,
    cov_x: list[Tensor] | None = None,  # [layer_idx, n_experts, Tensor[d, d]]
    taylor: bool = False,
    hidden_state_cache: list[
        list[Tensor]
    ] = None,  # [n_layers, n_calibs // calibs_size, Tensor[B, T, d]]
):
    """Qwen2MoE sparse block forward preserving the native routing and shared expert."""

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if hidden_state_cache is not None:
            hidden_state_cache[layer_idx].append(hidden_states.detach())
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Pass weights to expert layer for calibration hook
            expert_layer.routing_weights = routing_weights[top_x, idx, None]

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            if cov_x:
                H = current_state.detach().to(
                    device=cov_x[layer_idx][expert_idx].device, dtype=dtype_p
                )
                r = routing_weights[top_x, idx, None].sqrt()
                H *= r
                cov_x[layer_idx][expert_idx] += H.T @ H
                del H

            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    return forward
