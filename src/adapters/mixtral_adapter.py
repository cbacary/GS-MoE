from .model_adapter import MLPComponents, AttentionComponents, MLPTensors
from .moe_adapter import MoEAdapter

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union, Optional
from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig

from src.utils.model import dtype_p, calib_device

import gc

from src.patchers.latent_linear import LatentLinear

from src.utils.logger import logger


class MixtralAdapter(MoEAdapter):
    @property
    def arch(self) -> str:
        return "mixtral"

    @property
    def n_experts(self) -> int:
        return getattr(self.model_config, "num_local_experts", 8)

    @property
    def d_int(self) -> int:
        return getattr(self.model_config, "intermediate_size", None)

    def dense(self, layer_idx: int) -> bool:
        # Mixtral-8x7B: all decoder layers are MoE, no dense layers
        return False

    def n_experts_in_layer(self, layer_idx: int) -> int:
        return self.n_experts

    def get_n_inner(self, block=None, expert_idx=None, layer_idx=None) -> int:
        if block is None and layer_idx is not None:
            b = self.get_transformer_blocks()
            block = b[layer_idx]
        return block.block_sparse_moe.experts[expert_idx or 0].w1.out_features

    def get_mlp_rank(self, layer_idx, expert_idx=None):
        return (
            self.get_transformer_blocks()[layer_idx]
            .block_sparse_moe.experts[expert_idx or 0]
            .w1.out_features
        )

    def get_mlp_d_rank(self, layer_idx: int, expert_idx: int = None) -> int:
        block = self.get_transformer_blocks()[layer_idx]
        return block.block_sparse_moe.experts[expert_idx or 0].w2.out_features

    def get_mlp_int_rank(self, layer_idx: int, expert_idx: int = None) -> int:
        block = self.get_transformer_blocks()[layer_idx]
        return block.block_sparse_moe.experts[expert_idx or 0].w1.out_features

    def get_transformer_blocks(
        self,
    ) -> nn.ModuleList:
        return self.model.model.layers

    def get_moe_block(self, layer_idx: int) -> nn.Module:
        return self.get_transformer_blocks()[layer_idx].block_sparse_moe

    def get_experts(self, layer_idx: int) -> nn.ModuleList:
        return self.get_moe_block(layer_idx).experts

    def get_mlp_components(self, layer_idx: int, expert_idx: int = None) -> MLPComponents:
        if expert_idx is None:
            raise ValueError("expert_idx required for Mixtral MoE layer")
        block = self.get_transformer_blocks()[layer_idx]
        mlp = block.block_sparse_moe.experts[expert_idx]
        # Mixtral naming: w1 = gate_proj, w2 = down_proj, w3 = up_proj
        return MLPComponents(block=block, up_proj=mlp.w3, down_proj=mlp.w2, gate_proj=mlp.w1)

    def get_mlp_tensors(self, layer_idx: int, expert_idx: int = None) -> MLPTensors:
        if expert_idx is None:
            raise ValueError("expert_idx required for Mixtral MoE layer")
        block = self.get_transformer_blocks()[layer_idx]
        mlp = block.block_sparse_moe.experts[expert_idx]
        return MLPTensors(
            up_proj=mlp.w3.weight.data, down_proj=mlp.w2.weight.data, gate_proj=mlp.w1.weight.data
        )

    def get_ModelForCausalLM(self):
        from src.patchers.mixtral_rebuild import MixtralForCausalLM

        return MixtralForCausalLM

    def get_patched_decoder_layer_forward(self):
        return patched_mixtral_decoder_layer_forward

    def get_patched_moe_forward(self):
        return patched_moe_forward

    def replace_mlp_layers(
        self,
        layer_idx: int,
        new_up: nn.Module,
        new_down: nn.Module,
        new_gate: nn.Module | None = None,
        expert_idx: int = None,
    ) -> None:
        if expert_idx is None:
            raise Exception("expert_idx cannot be None")
        block = self.get_transformer_blocks()[layer_idx]
        mlp = block.block_sparse_moe.experts[expert_idx]
        mlp.w1 = new_gate
        mlp.w2 = new_down
        mlp.w3 = new_up

    def load_target_layers_mlp(self, input_dir: str, target_layers: list[int]):
        import os

        input_dir = os.path.expandvars(f"$TMPDIR/compressed_layers/")

        blocks = self.get_transformer_blocks()

        def tensor_to_linear(expert: nn.Linear | LatentLinear, weight: Tensor):
            if isinstance(expert, LatentLinear):
                l = LatentLinear(
                    in_features=weight.shape[1],
                    out_features=weight.shape[0],
                    device="cuda",
                    dtype=torch.bfloat16,
                    bias=False,
                    shared_weight_getter=expert.get_shared_weight,
                    flipped_down=expert.flipped_down,
                )
            else:
                l = nn.Linear(
                    in_features=weight.shape[1],
                    out_features=weight.shape[0],
                    device="cuda",
                    dtype=torch.bfloat16,
                    bias=False,
                )
            l.weight.data.copy_(weight.to(torch.bfloat16))
            return l

        for layer_idx in target_layers:

            gc.collect()
            torch.cuda.empty_cache()

            load_path = os.path.join(input_dir, f"layer_{layer_idx}_nys")
            if not os.path.exists(load_path):
                print(f"File {load_path} not found. Skipping layer {layer_idx}")
                continue

            experts_dict = torch.load(load_path)
            block = blocks[layer_idx]

            for expert_idx, expert in enumerate(block.block_sparse_moe.experts):
                expert_data = None
                # I keep this in here to laugh at chatbots when you ask them to do something:
                if isinstance(experts_dict, list) and expert_idx < len(experts_dict):
                    expert_data = experts_dict[expert_idx]
                elif isinstance(experts_dict, dict) and expert_idx in experts_dict:
                    expert_data = experts_dict[expert_idx]

                if expert_data is not None:
                    expert.w1 = tensor_to_linear(expert.w1, expert_data["w1"])
                    expert.w2 = tensor_to_linear(expert.w2, expert_data["w2"])
                    expert.w3 = tensor_to_linear(expert.w3, expert_data["w3"])

        print(f"All layers succesfully initialized.")

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
        block = self.get_transformer_blocks()[layer_idx]
        if new_q is not None:
            block.self_attn.q_proj = new_q
        if new_k is not None:
            block.self_attn.k_proj = new_k
        if new_v is not None:
            block.self_attn.v_proj = new_v
        if new_o is not None:
            block.self_attn.o_proj = new_o

    def get_qk_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        block = self.get_transformer_blocks()[layer_idx]
        W_q = block.self_attn.q_proj.weight
        W_k = block.self_attn.k_proj.weight
        return W_q, W_k

    def get_vo_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        block = self.get_transformer_blocks()[layer_idx]
        W_v = block.self_attn.v_proj.weight
        W_o = block.self_attn.o_proj.weight
        return W_v, W_o


from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack


def patched_mixtral_decoder_layer_forward(
    bi_scores: list[float],
    cov_x: list[Tensor] | None,
    layer_idx: int,
):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, _ = self.block_sparse_moe(hidden_states)

        hidden_states = residual + hidden_states

        if bi_scores is not None:
            x_in: Tensor = residual.detach().to(dtype=dtype_p)
            # We will use x_out for both fact_scores and nys_scores
            x_out: Tensor = hidden_states.detach().to(dtype=dtype_p)

            score = (1 - torch.cosine_similarity(x_in, x_out, dim=2)).mean().item()
            bi_scores[layer_idx] += score
            del x_in, x_out

        # torch.cuda.empty_cache()

        return hidden_states

    return forward


def patched_moe_forward(
    layer_idx,
    cov_x: list[Tensor] | None = None,  # [layer_idx, n_experts, Tensor[d, d]]
    taylor: bool = False,
    hidden_state_cache: list[Tensor] = None,  # len == n_calibs // batch_size
):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        This is for patching the routing weights into the expert's forward.
        Not used because it doesn't actually help calibration.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # if self.training and self.jitter_noise > 0:
        #     hidden_states *= torch.empty_like(hidden_states).uniform_(
        #         1.0 - self.jitter_noise, 1.0 + self.jitter_noise
        #     )
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
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
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    return forward
