from math import e
from src.adapters.llama_adapter import LlamaAdapter
from .model_adapter import MLPComponents, AttentionComponents, MLPTensors
from .llama_adapter import LlamaAdapter

import os

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union, Optional
from dataclasses import dataclass
from tqdm import tqdm

from transformers.configuration_utils import PretrainedConfig

from src.utils.model import dtype_p, calib_device

from src.patchers.latent_linear import LatentLinear

import gc

from src.eval import load_calibration_texts

from src.utils.logger import logger

from src.adapters.compression_config import CompressionConfig


class MoEAdapter(LlamaAdapter):

    nystrom_scores: list[Tensor] | None = None
    bi_scores: list[Tensor] | None = None

    @staticmethod
    def from_model(model: nn.Module, tokenizer, config: CompressionConfig) -> "MoEAdapter":

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Could be Llama or Mixtral or others sharing this structure
            # Check config type or architecture string if available
            arch = getattr(model.config, "model_type", None)
            if arch == "mixtral":
                from .mixtral_adapter import MixtralAdapter

                m = MixtralAdapter(model, tokenizer=tokenizer)
            elif arch == "deepseek_v2":
                from .deepseek_adapter import DeepseekV2Adapter

                m = DeepseekV2Adapter(model, tokenizer=tokenizer)
            elif arch == "deepseek":
                from .deepseek_adapter import DeepseekAdapter

                m = DeepseekAdapter(model, tokenizer=tokenizer)
            elif arch == "qwen2_moe":
                from .qwen2_moe_adapter import Qwen2MoeAdapter

                m = Qwen2MoeAdapter(model, tokenizer=tokenizer)
            else:
                m = LlamaAdapter(model, tokenizer=tokenizer)
            m.config = config
            return m
        else:
            raise RuntimeError("Unsupported model architecture")

    @property
    def n_experts(self) -> int:
        pass

    def save_layer(self, output_dir: str, expert_weight_cache, layer_idx, factorized=False):
        import os

        output_dir = os.path.expandvars(f"$TMPDIR/compressed_layers/")

        os.makedirs(output_dir, exist_ok=True)
        p = os.path.join(output_dir, f"layer_{layer_idx}")
        torch.save(expert_weight_cache, p)

    @abstractmethod
    def dense(self, layer_idx: int) -> bool:
        pass

    @abstractmethod
    def get_mlp_d_rank(self, layer_idx: int, expert_idx: int = None) -> int:
        raise NotImplementedError("Must implement for class")

    @abstractmethod
    def get_mlp_int_rank(self, layer_idx: int, expert_idx: int = None) -> int:
        raise NotImplementedError("Must implement for class")

    @abstractmethod
    def n_experts_in_layer(self, layer_idx: int) -> int:
        raise NotImplementedError("Must implement for class")

    @abstractmethod
    def dense(self, layer_idx: int) -> bool:
        raise NotImplementedError("Must implement for class")

    def get_mlp_components(self, layer_idx: int, expert_idx: int) -> MLPComponents:
        raise NotImplementedError("Must implement for class")

    @abstractmethod
    def get_mlp_tensors(self, layer_idx: int, expert_idx: int) -> MLPTensors:
        raise NotImplementedError("Must implement for class")

    @abstractmethod
    def get_moe_block(self, layer_idx: int):
        raise NotImplementedError("Must implement for class")

    @abstractmethod
    def get_ModelForCausalLM(self):
        raise NotImplementedError("Must implement for class")

    def get_experts(self, layer_idx):
        raise NotImplementedError("Must implement for class")

    def patch(self, nystrom=False, flipped_down=True):
        """
        base_ranks should look like this:
        {
            "1": {
                "up_base": (in_features, out_features),
                "gate_base": (in_features, out_features),
                "down_base": (in_features, out_features)
            },
            ....
        }
        """

        n_layers, arch = self.n_layers, self.arch
        # self.model.config.base_ranks = base_ranks

        q_ranks = []
        k_ranks = []
        v_ranks = []
        o_ranks = []
        int_ranks = []
        base_ranks_up = []
        base_ranks_gate = []
        base_ranks_down = []
        damage_deltas_ranks = []
        factorized_map = []
        shared_proj_ranks = []

        blocks = self.get_transformer_blocks()

        for layer in range(n_layers):
            query_weight, key_weight = self.get_qk_weights(layer)
            value_weight, output_weight = self.get_vo_weights(layer)

            q_rank = query_weight.shape[0]
            k_rank = key_weight.shape[0]

            v_rank = value_weight.shape[0]
            o_rank = output_weight.shape[1]

            q_ranks.append(q_rank)
            k_ranks.append(k_rank)
            v_ranks.append(v_rank)
            o_ranks.append(o_rank)

            int_rank = []
            base_rank_up = []
            base_rank_gate = []
            base_rank_down = []

            comps = self.get_mlp_components(layer, 0)
            layer_factorized_map = {
                "up": isinstance(comps.up_proj, LatentLinear)
                or hasattr(comps.up_proj, "get_shared_weight"),
                "gate": isinstance(comps.gate_proj, LatentLinear)
                or hasattr(comps.gate_proj, "get_shared_weight"),
                "down": isinstance(comps.down_proj, LatentLinear)
                or hasattr(comps.down_proj, "get_shared_weight"),
            }
            factorized_map.append(layer_factorized_map)

            layer_shared_proj_ranks = []
            for shared_proj in getattr(blocks[layer], "shared_weights", ()) or ():
                shared_rank = next(
                    (weight.shape[1] for weight in shared_proj.values() if weight is not None),
                    None,
                )

                layer_shared_proj_ranks.append(int(shared_rank) if shared_rank else None)
            shared_proj_ranks.append(layer_shared_proj_ranks)

            """
            WARNING: Every layer MUST have a base_rank_up/gate/down
            That is, base_ranks_up/gate/down must be of length n_layers 
            EVEN if no compression occured.
            

            In other words, you CANNOT skip the following code allocating base_ranks_up/gate/down
            """
            layer_damage_delta_ranks = []
            for expert_idx in range(self.n_experts_in_layer(layer)):
                # print(
                #     f"down_proj.weight.shape = {self.get_mlp_components(layer, expert_idx).down_proj.weight.shape}"
                # )
                # print(
                #     f"up_proj.weight.shape = {self.get_mlp_components(layer, expert_idx).up_proj.weight.shape}"
                # )
                int_rank.append(self.get_mlp_components(layer, expert_idx).up_proj.weight.shape[0])
                # base_rank_up.append(
                #     self.get_mlp_components(layer, expert_idx).up_proj.weight.shape[1]
                # )
                # base_rank_gate.append(
                #     self.get_mlp_components(layer, expert_idx).gate_proj.weight.shape[1]
                # )
                # base_rank_down.append(
                #     self.get_mlp_components(layer, expert_idx).down_proj.weight.shape[
                #         0 if self.config.flipped_down else 1
                #     ]
                # )
                base_rank_up.append(
                    self.get_mlp_components(layer, expert_idx).up_proj.weight.shape[1]
                )
                base_rank_gate.append(
                    self.get_mlp_components(layer, expert_idx).gate_proj.weight.shape[1]
                )
                base_rank_down.append(
                    self.get_mlp_components(layer, expert_idx).down_proj.weight.shape[
                        0 if self.config.flipped_down else 1
                    ]
                )
                layer_damage_delta_ranks.append(1)
            # layer_damage_delta_ranks.append(
            #     int(getattr(moe_block, f"damage_u_{expert_idx}").shape[1])
            # )

            int_ranks.append(int_rank)
            base_ranks_up.append(base_rank_up)
            base_ranks_gate.append(base_rank_gate)
            base_ranks_down.append(base_rank_down)
            damage_deltas_ranks.append(layer_damage_delta_ranks)

        config = self.model.config
        config.factorized_map = factorized_map
        config.ffn_dim = -1
        config.flipped_down = self.config.flipped_down
        config.q_ranks = q_ranks
        config.k_ranks = k_ranks
        config.v_ranks = v_ranks
        config.o_ranks = o_ranks
        config.int_ranks = int_ranks
        config.base_ranks_up = base_ranks_up
        config.base_ranks_gate = base_ranks_gate
        config.base_ranks_down = base_ranks_down
        config.damage_deltas_ranks = damage_deltas_ranks
        config.n_shared_projs = self.config.n_shared_projs
        config.shared_proj_ranks = shared_proj_ranks

        if arch == "mixtral":
            config.auto_map = {"AutoModelForCausalLM": "mixtral_rebuild.MixtralForCausalLM"}
        if arch == "qwen2_moe":
            config.auto_map = {"AutoModelForCausalLM": "qwen2_moe_rebuild.Qwen2MoeForCausalLM"}
        if arch == "deepseek":
            config.auto_map = {}
            if config.model_type == "deepseek_v2":
                config.auto_map["AutoModelForCausalLM"] = "deepseekv2_rebuild.DeepseekV2ForCausalLM"
                config.auto_map["AutoConfig"] = "deepseekv2_rebuild.DeepseekV2Config"
                config.auto_map["AutoModel"] = "deepseekv2_rebuild.DeepseekV2Model"

            else:
                config.auto_map["AutoModelForCausalLM"] = "deepseek_rebuild.DeepseekForCausalLM"
                config.auto_map["AutoConfig"] = "deepseek_rebuild.DeepseekConfig"
                config.auto_map["AutoModel"] = "deepseek_rebuild.DeepseekModel"

    @torch.no_grad()
    def convert_model(self, saved_layers_dir: str = "./compressed_output/compressed_layers/"):

        saved_layers_dir = os.path.expandvars(saved_layers_dir)

        projection_names = {
            "mixtral": {"gate": "w1", "up": "w3", "down": "w2"},
            "deepseek": {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"},
            "qwen2_moe": {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"},
        }

        if self.arch not in projection_names:
            raise ValueError(f"Unsupported architecture: {self.arch}")

        first_parameter = next(self.model.parameters())
        weight_dtype = first_parameter.dtype
        weight_device = first_parameter.device
        projection_name = projection_names[self.arch]

        def replace_proj(expert: nn.Module, proj: nn.Module, key: str) -> None:
            setattr(expert, projection_name[key], proj)

        expert_to_shared_proj = [
            [0] * self.n_experts_in_layer(layer_idx) for layer_idx in range(self.n_layers)
        ]

        for layer_idx in range(self.n_layers):
            path = os.path.join(saved_layers_dir, f"layer_{layer_idx}")
            if not os.path.exists(path):
                print(f"Path {path} does not exist")
                continue

            projs = torch.load(path, map_location=weight_device)
            """

            grouped projs looks like:
            {
                "shared_proj": [{
                    "up": Tensor,
                    ....
                }, .....],
                "expert_coefficients": list[
                    {
                        ## This would be expert 0
                        "up": Tensor,
                        ....
                    },
                ]
            }
            
            projs looks like:
            {
                "shared_proj": {
                    "up": Tensor,
                    ....
                },
                "expert_coefficients": list[
                    {
                        ## This would be expert 0
                        "up": Tensor,
                        ....
                    },
                ]
            }
            """

            try:
                expert_coefficients = projs["expert_coefficients"]
                shared_projs = projs["shared_proj"]
            except (KeyError, TypeError):
                logger.info(
                    f"Saved layer is not in expected format {layer_idx}",
                )
                continue

            if not isinstance(shared_projs, list):
                raise ValueError(
                    f"Layer {layer_idx} must store grouped shared projections as a list."
                )

            block = self.get_transformer_blocks()[layer_idx]
            if self.arch in ("deepseek", "qwen2_moe"):
                if self.dense(layer_idx):
                    experts = [block.mlp]
                else:
                    experts = block.mlp.experts
            else:
                experts = block.block_sparse_moe.experts

            if len(expert_coefficients) != len(experts):
                raise ValueError(
                    f"Layer {layer_idx} has {len(experts)} experts but its artifact contains "
                    f"{len(expert_coefficients)} coefficient sets."
                )

            expert_to_shared_proj[layer_idx] = [0] * len(experts)
            for exp in expert_coefficients:
                expert_to_shared_proj[layer_idx][int(exp["expert_idx"])] = int(exp["group_number"])

            shared_module_list = nn.ModuleList()
            for shared_group in shared_projs:
                shared_dict = {}
                for key, weight in shared_group.items():
                    if weight is not None:
                        shared_dict[key] = nn.Parameter(
                            weight.to(dtype=weight_dtype, device=weight_device)
                        )
                shared_module_list.append(nn.ParameterDict(shared_dict))

            if shared_module_list:
                block.shared_weights = shared_module_list

            for expert_idx, expert in enumerate(experts):
                exp = expert_coefficients[expert_idx]
                group_number = int(exp["group_number"])

                for key, weight in shared_projs[group_number].items():
                    if weight is None:
                        continue
                    latent_proj = LatentLinear(
                        in_features=exp[key].shape[1],
                        out_features=exp[key].shape[0],
                        shared_weight_getter=lambda b=block, k=key, group_number=group_number: b.shared_weights[
                            group_number
                        ][
                            k
                        ],
                        bias=False,
                        dtype=weight_dtype,
                        device=weight_device,
                        flipped_down=self.config.flipped_down,
                        is_down=key == "down",
                    )
                    latent_proj.weight.data.copy_(
                        exp[key].to(device=weight_device, dtype=weight_dtype)
                    )

                    replace_proj(expert, latent_proj, key)

                for key, weight in shared_projs[group_number].items():
                    if weight is not None:
                        continue
                    if exp[key] is None:
                        continue
                    linear_proj = nn.Linear(
                        in_features=exp[key].shape[1],
                        out_features=exp[key].shape[0],
                        bias=False,
                        dtype=weight_dtype,
                        device=weight_device,
                    )

                    linear_proj.weight.data.copy_(
                        exp[key].to(device=weight_device, dtype=weight_dtype)
                    )

                    replace_proj(expert, linear_proj, key)

            logger.info(f"Converted layer {layer_idx} to a Latent Linear Layer")
        self.model.config.expert_to_shared_proj = expert_to_shared_proj
        ##

    @torch.no_grad()
    def compute_layer_energy(
        self,
        layer_idx: int,
        Ca: list[Tensor] | Tensor | None = None,
        up_activations: list[Tensor] | None = None,
        flipped_down=True,
    ) -> MLPTensors:

        # up_proj: [ d_int x d ] [ d x d_int]

        d_int = self.get_mlp_int_rank(layer_idx)

        def sum_energy(
            S: Tensor,
            weight: Tensor,
            expert_idx: int,
            activations: Tensor | None = None,
            up_activations: Tensor | None = None,
        ) -> Tensor:
            weight = weight.to(dtype=dtype_p, device=S.device)

            if activations is not None:
                # (down): [d, d_ff] [d_ff, d_ff] [d_ff, d]
                # [d, d]
                S += weight @ activations @ weight.T
            elif up_activations is not None:
                S += weight @ up_activations @ weight.T
            else:
                cumulative_routing_frequency = (
                    self.metrics["exp_routing_frequency"][layer_idx][expert_idx]
                    / self.metrics["tokens_per_exp"][layer_idx][expert_idx]
                )
                # S += cumulative_routing_frequency * weight.T @ weight
                S += weight.T @ weight
            return S

        E_up = torch.zeros(self.d_model, self.d_model, device=calib_device, dtype=dtype_p)
        E_gate = torch.zeros(self.d_model, self.d_model, device=calib_device, dtype=dtype_p)

        if not flipped_down:
            down_dim = d_int
        else:
            down_dim = d_int if Ca is None else self.d_model
        E_down = torch.zeros(down_dim, down_dim, device=calib_device, dtype=dtype_p)

        for expert_idx in range(self.n_experts):
            comps = self.get_mlp_components(layer_idx=layer_idx, expert_idx=expert_idx)
            u, d, g = comps.up_proj.weight, comps.down_proj.weight, comps.gate_proj.weight

            E_up = sum_energy(E_up, u, expert_idx, up_activations[expert_idx])
            E_gate = sum_energy(E_gate, g, expert_idx)

            if flipped_down:
                if not isinstance(Ca, list):
                    # Original Flipped down implementation
                    E_down = sum_energy(
                        E_down,
                        d,
                        expert_idx,
                        Ca,
                    )
                else:
                    # Summing y_e y_e.T  over all experts
                    E_down += Ca[expert_idx]
            else:
                E_down = sum_energy(E_down, d, expert_idx)

        E_down.to(device="cuda", dtype=dtype_p)

        layer_energy = MLPTensors(up_proj=E_up, gate_proj=E_gate, down_proj=E_down)

        return layer_energy

    @torch.no_grad()
    def compute_layer_energy_for_combined(
        self,
        layer_idx: int,
        down_activations: list[Tensor],
        up_activations: list[Tensor],
        flipped_down=True,
    ) -> MLPTensors:
        # assumes flipped_down True

        # up_proj: [ d_int x d ] [ d x d_int]

        d_int = self.get_mlp_int_rank(layer_idx)

        def sum_energy(
            S: Tensor,
            weight: Tensor,
            expert_idx: int,
            down_activations: list[Tensor] | None = None,
            up_activations: Tensor | None = None,
        ) -> Tensor:
            """
            down_activations: list of each expert's activations
            up_activations: list of each expert's activations  (up)
            """
            cumulative_routing_frequency = (
                self.metrics["exp_routing_frequency"][layer_idx][expert_idx]
                / self.metrics["tokens_per_exp"][layer_idx][expert_idx]
            )

            weight = weight.to(dtype=dtype_p, device=S.device)

            S += weight @ up_activations @ weight.T

            return S

        E_up = torch.zeros(self.d_model, self.d_model, device=calib_device, dtype=dtype_p)
        E_gate = torch.zeros(self.d_model, self.d_model, device=calib_device, dtype=dtype_p)
        E_down = torch.zeros(self.d_model, self.d_model, device=calib_device, dtype=dtype_p)

        for expert_idx in range(self.n_experts):
            comps = self.get_mlp_components(layer_idx=layer_idx, expert_idx=expert_idx)
            u, d, g = comps.up_proj.weight, comps.down_proj.weight, comps.gate_proj.weight

            E_up = sum_energy(E_up, u, expert_idx, up_activations[expert_idx])
            E_gate = sum_energy(E_gate, g, expert_idx)

            E_down += down_activations[expert_idx]

            E_down.to(device="cuda", dtype=dtype_p)

        layer_energy = MLPTensors(up_proj=E_up, gate_proj=E_gate, down_proj=E_down)

        return layer_energy

    @abstractmethod
    def get_patched_decoder_layer_forward(self):
        raise NotImplementedError

    @abstractmethod
    def get_patched_moe_forward(self):
        raise NotImplementedError
