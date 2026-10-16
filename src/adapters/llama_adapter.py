from .model_adapter import ModelAdapter, MLPComponents, AttentionComponents

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union, Optional
from dataclasses import dataclass
import logging

from transformers.configuration_utils import PretrainedConfig

from src.utils.model import dtype_p, calib_device


class LlamaAdapter(ModelAdapter):
    @property
    def arch(self) -> str:
        return "llama"

    def get_transformer_blocks(self) -> nn.ModuleList:
        return self.model.model.layers

    # def get_n_inner(self, block) -> int:
    #     return block.mlp.gate_proj.out_features

    def register_hooks(
        self, layer_idx, block, cov_mlp_list, cov_q_list, cov_k_list, cov_x_list, handles, logger
    ):
        handles.append(
            block.mlp.down_proj.register_forward_pre_hook(
                self._llama_pre_gate_hook(layer_idx, cov_mlp_list)
            )
        )
        handles.append(
            block.input_layernorm.register_forward_hook(self._input_hook(layer_idx, cov_x_list))
        )
        handles.append(
            block.self_attn.k_proj.register_forward_hook(
                self._make_proj_hook(
                    layer_idx, cov_k_list, self.n_kv_heads, self.head_dim, self.d_model
                )
            )
        )
        handles.append(
            block.self_attn.q_proj.register_forward_hook(
                self._make_proj_hook(
                    layer_idx, cov_q_list, self.n_heads, self.head_dim, self.d_model
                )
            )
        )

    @staticmethod
    def _llama_pre_gate_hook(layer_idx, cov_mlp_list):
        @torch.no_grad()
        def hook(module, input: Tuple[torch.Tensor]):
            x_input = input[0]
            H = x_input.detach().to(dtype=dtype_p).view(-1, x_input.size(-1))
            cov_mlp_list[layer_idx] += (H.T @ H).to(device=calib_device)
            return None

        return hook

    @staticmethod
    def _input_hook(layer_idx, cov_list):
        @torch.no_grad()
        def hook(module, inp, out):
            proj_out = out.detach().to(dtype=dtype_p, device="cuda")
            cov_list[layer_idx] += torch.sum(proj_out.mT @ proj_out, dim=0).to(device=calib_device)

        return hook

    def get_mlp_components(self, layer_idx: int) -> MLPComponents:
        block = self.get_transformer_blocks()[layer_idx]
        return MLPComponents(
            block=block,
            up_proj=block.mlp.up_proj,
            down_proj=block.mlp.down_proj,
            gate_proj=block.mlp.gate_proj,
        )

    def replace_mlp_layers(
        self,
        layer_idx: int,
        new_up: nn.Module,
        new_down: nn.Module,
        new_gate: Optional[nn.Module] = None,
        expert_idx: Optional[int] = None,
    ) -> None:
        block = self.get_transformer_blocks()[layer_idx]
        block.mlp.up_proj = new_up
        block.mlp.down_proj = new_down
        if new_gate is not None:
            block.mlp.gate_proj = new_gate

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
