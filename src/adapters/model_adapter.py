import torch
import numpy as np
import os

from torch import Tensor
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union, Optional
from dataclasses import dataclass
import logging
import json

import pandas as pd

from transformers.configuration_utils import PretrainedConfig

from src.adapters.compression_config import CompressionConfig
from src.utils.model import dtype_p, calib_device

from datetime import datetime

import lm_eval
from lm_eval.models.huggingface import HFLM

from src.utils.metrics import MetricsManager

import datasets

tmpdir = os.environ.get("TMPDIR")


@dataclass
class MLPComponents:
    block: nn.Module
    up_proj: nn.Module
    down_proj: nn.Module
    gate_proj: Optional[nn.Module] = None

    def __iter__(self):
        """
        returns up, gate, down
        """
        yield self.up_proj
        yield self.gate_proj
        yield self.down_proj


@dataclass
class MLPTensors:
    up_proj: Tensor | None
    down_proj: Tensor | None
    gate_proj: Tensor | None

    def to(self, dtype, device=None):
        if self.up_proj is not None:
            self.up_proj = self.up_proj.to(dtype=dtype, device=device)
        if self.down_proj is not None:
            self.down_proj = self.down_proj.to(dtype=dtype, device=device)
        if self.gate_proj is not None:
            self.gate_proj = self.gate_proj.to(dtype=dtype, device=device)
        return self

    def __iter__(self):
        """
        returns up, gate, down
        """
        yield self.up_proj
        yield self.gate_proj
        yield self.down_proj


@dataclass
class AttentionComponents:
    block: nn.Module
    q_proj: nn.Module
    k_proj: nn.Module
    v_proj: Optional[nn.Module] = None
    o_proj: Optional[nn.Module] = None


class ModelAdapter(ABC):
    def __init__(self, model: nn.Module, tokenizer=None):
        self.model = model
        self.model_config: PretrainedConfig = model.config
        self.config: CompressionConfig = CompressionConfig()
        self.tokenizer = tokenizer
        self.calibs = None

        MetricsManager.load_metrics()
        self.metrics = MetricsManager.build_metrics()

        # self.metrics = {}
        # run_name = datetime.now().strftime("%Y_%m_%d--%H_%M_%S")

        # ModelAdapter._metrics[run_name] = self.metrics
        # self.metrics["RunDate"] = datetime.now().strftime("%b %d, %Y %I:%M %p")

    @staticmethod
    def from_model(model: nn.Module, tokenizer) -> "ModelAdapter":
        from .llama_adapter import LlamaAdapter
        from .opt_adapter import OptAdapter
        from .mixtral_adapter import MixtralAdapter
        from .deepseek_adapter import DeepseekAdapter

        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            return OptAdapter(model, tokenizer=tokenizer)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            # Could be Llama or Mixtral or others sharing this structure
            # Check config type or architecture string if available
            arch = getattr(model.config, "model_type", None)
            if arch == "mixtral":
                return MixtralAdapter(model, tokenizer=tokenizer)
            if arch == "deepseek_v2":  # Deepseek often uses this, or just "deepseek"
                return DeepseekAdapter(model, tokenizer=tokenizer)
            if arch == "deepseek":
                return DeepseekAdapter(model, tokenizer=tokenizer)

            return LlamaAdapter(model, tokenizer=tokenizer)
        else:
            raise RuntimeError("Unsupported model architecture")

    def save_metrics_self(self, directory: str):
        os.makedirs(directory, exist_ok=True)

        run_metrics_path = os.path.join(directory, f"compression-metrics.json")

        with open(run_metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def save_metrics(
        self,
        path=None,
        backup_dir: str = "./metrics/backups/",
    ):

        class MetricsEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, torch.Tensor):
                    return o.detach().cpu().tolist()
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if isinstance(o, np.generic):
                    return o.item()
                if isinstance(o, (set, frozenset)):
                    return sorted(o)
                if hasattr(o, "item"):  # anything else 0-d
                    return o.item()
                return repr(o)

        def dumps(obj, indent=2, level=0):
            pad = " " * (indent * level)
            inner = " " * (indent * (level + 1))

            if isinstance(obj, dict) and obj:
                body = ",\n".join(
                    f"{inner}{json.dumps(str(k))}: {dumps(v, indent, level + 1)}"
                    for k, v in obj.items()
                )
                return "{\n" + body + "\n" + pad + "}"

            if isinstance(obj, list) and obj:
                if any(isinstance(x, (dict, list)) for x in obj):
                    body = ",\n".join(inner + dumps(x, indent, level + 1) for x in obj)
                    return "[\n" + body + "\n" + pad + "]"
                return json.dumps(obj, cls=MetricsEncoder)  # flat list -> single line

            return json.dumps(obj, cls=MetricsEncoder)

        if path is None:
            return
        with open(path, "w") as f:
            # json.dump(self.metrics, f, indent=4, cls=MetricsEncoder)
            f.write(dumps(self.metrics))
        # MetricsManager.save_metrics_static(
        #     path=path, backup_dir=backup_dir, run_metrics=self.metrics
        # )

    def benchmark_and_save(self, kl_loss: float, save_path=None):

        save_path = self.config.output_dir if save_path is None else save_path

        # tasks = ["openbookqa", "hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande", "mathqa"]
        tasks = [
            "openbookqa",
            "hellaswag",
            "arc_challenge",
            "arc_easy",
            "piqa",
            "winogrande",
            "mathqa",
            "boolq",
        ]
        timestamp = self.metrics["RunName"]
        uniformity = self.config.uniformity
        order = self.config.order
        method = f"{self.config.order} ({uniformity})"
        if order == "combined" and self.config.nys_to_fact_ratio == 0.0:
            method = f"pure factorize"
        compression_ratio = self.config.compression_ratio

        from src.eval import compute_perplexity

        wikitext_ppl = compute_perplexity(
            self.model, self.tokenizer, 2, dataset="wikitext", adapter=self
        )
        c4_ppl = compute_perplexity(self.model, self.tokenizer, 2, dataset="c4", adapter=self)

        # alpaca_ppl = compute_perplexity(
        # self.model, self.tokenizer, 2, dataset="alpaca", adapter=self
        # )

        from pathlib import Path

        def get_dir_size_pathlib(path="."):
            root = Path(path)
            # rglob('*') recursively finds all files and subdirectories
            return sum(
                f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink()
            )

        columns = [
            "model",
            "compression_ratio",
            "method",
            "nys_to_fact_ratio",
            "actual_size-GB",
            "timestamp",
            "wikitext-ppl",
            "c4-ppl",
            # "alpaca-ppl"
        ]
        for task in tasks:
            columns.append(f"{task}_norm")
            columns.append(task)
        row = {
            "model": self.config.model,
            "compression_ratio": compression_ratio,
            "actual_size-GB": f"{get_dir_size_pathlib(os.path.join(tmpdir, "compressed_output", "model")) / (1000**3):.2f}",
            "method": method,
            "timestamp": timestamp,
            "wikitext-ppl": wikitext_ppl,
            "c4-ppl": c4_ppl,
            # f"alpaca-ppl": alpaca_ppl,
            "kl-loss": kl_loss,
            "nys_to_fact_ratio": (
                self.config.nys_to_fact_ratio
                if not self.config.dynamic_fact_nys_ratio
                else "dynamic per-layer"
            ),
        }
        if self.config.benchmark:

            datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            lm = HFLM(
                pretrained=self.model,
                tokenizer=self.tokenizer,
                device="cuda",
                batch_size=128,
                # batch_size="auto",
                # batch_size=32,
            )

            results = lm_eval.simple_evaluate(
                model=lm, tasks=tasks, num_fewshot=0, batch_size=256, model_args="dtype=bfloat16"
            )

            for task in tasks:
                task_results = results["results"].get(task, {})
                row[task] = task_results.get("acc,none")
                row[f"{task}_norm"] = task_results.get("acc_norm,none")

        df = pd.DataFrame([row], columns=columns)
        print(df.to_string())

        csv_path = "./benchmark_results.csv"
        df.to_csv(
            csv_path,
            mode="a",
            header=not os.path.exists(csv_path),
            index=False,
        )
        return df

    @abstractmethod
    def compute_layer_energy(
        self, layer_idx: int, Ca: Tensor | None = None, flipped_down=True
    ) -> MLPTensors:
        pass

    @property
    def arch(self) -> str:
        return self.model_config.model_type

    @property
    def n_layers(self) -> int:
        return (
            getattr(self.model_config, "n_layer", None)
            or getattr(self.model_config, "num_hidden_layers", None)
            or getattr(self.model_config, "num_layers", None)
        )

    @property
    def n_heads(self) -> int:
        return getattr(self.model_config, "n_head", None) or getattr(
            self.model_config, "num_attention_heads", None
        )

    @property
    def d_model(self) -> int:
        return getattr(self.model_config, "hidden_size", None) or getattr(
            self.model_config, "dim", None
        )

    @property
    def d_int(self) -> int:
        return getattr(self.model_config, "intermediate_size", None)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_experts(self) -> int:
        if self.arch == "deepseek":
            return getattr(self.model_config, "n_routed_experts", 0)
        return getattr(self.model_config, "num_local_experts", 0)

    @property
    def n_kv_heads(self) -> int:
        return getattr(self.model_config, "num_key_value_heads", self.n_heads)

    @abstractmethod
    def get_transformer_blocks(self) -> nn.ModuleList:
        pass

    def get_n_inner(self) -> int:
        return self.model_config.intermediate_size

    @abstractmethod
    def register_hooks(
        self,
        layer_idx: int,
        block: nn.Module,
        cov_mlp_list: List[torch.Tensor],
        cov_q_list: List[List[torch.Tensor]],
        cov_k_list: List[List[torch.Tensor]],
        cov_x_list: List[torch.Tensor],
        handles: List[Any],
        logger: logging.Logger,
    ):
        pass

    def on_batch_end_step(self, layer_idx: int, x_in: torch.Tensor, cov_x_list: List[torch.Tensor]):
        pass

    @abstractmethod
    def get_mlp_components(self, layer_idx: int) -> MLPComponents:
        pass

    @abstractmethod
    def get_mlp_components(self, layer_idx: int, expert_idx: int) -> MLPComponents:
        pass

    @abstractmethod
    def replace_mlp_layers(
        self,
        layer_idx: int,
        new_up: nn.Module,
        new_down: nn.Module,
        new_gate: Optional[nn.Module] = None,
        expert_idx: Optional[int] = None,
    ) -> None:
        pass

    @abstractmethod
    def get_attn_components(self, layer_idx: int) -> AttentionComponents:
        pass

    @abstractmethod
    def replace_attn_layers(
        self,
        layer_idx: int,
        new_q: Optional[nn.Module],
        new_k: Optional[nn.Module],
        new_v: Optional[nn.Module],
        new_o: Optional[nn.Module],
    ) -> None:
        pass

    @abstractmethod
    def get_qk_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q and K projection weights for a layer.

        Returns:
            Tuple of (W_q, W_k) weight tensors
        """
        pass

    @abstractmethod
    def get_vo_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get V and O projection weights for a layer.

        Returns:
            Tuple of (W_v, W_o) weight tensors
        """
        pass

    def slice_gate_dims(
        self,
        layer_idx: int,
        up_weights: torch.Tensor,
        down_weights: torch.Tensor,
        gate_weights: Optional[torch.Tensor],
        new_bias_u: Optional[torch.Tensor],
        new_bias_g: Optional[torch.Tensor],
        bias: bool = True,
        expert_idx: Optional[int] = None,
    ):
        comps = self.get_mlp_components(layer_idx, expert_idx=expert_idx)

        # up_weights: [out, in]
        new_layer_U = nn.Linear(
            in_features=up_weights.shape[1],
            out_features=up_weights.shape[0],
            device="cuda",
            dtype=torch.float16,
            bias=True if comps.up_proj.bias is not None and bias else False,
        )
        new_layer_U.weight.data.copy_(up_weights.to(torch.float16))
        if bias and new_bias_u is not None:
            if new_layer_U.bias is not None:
                new_layer_U.bias.data.copy_(new_bias_u)

        new_layer_G = None
        if gate_weights is not None:
            # If we have gate weights, we expect the component to support it (like llama)
            # or we are adding it? Original code only did this if arch == llama.
            # But gate_weights is None for other arches in caller usually.

            gate_has_bias = False
            if comps.gate_proj is not None and comps.gate_proj.bias is not None:
                gate_has_bias = True

            new_layer_G = nn.Linear(
                in_features=gate_weights.shape[1],
                out_features=gate_weights.shape[0],
                device="cuda",
                dtype=torch.float16,
                bias=True if gate_has_bias and bias else False,
            )
            new_layer_G.weight.data.copy_(gate_weights.to(torch.float16))
            if bias and new_bias_g is not None:
                if new_layer_G.bias is not None:
                    new_layer_G.bias.data.copy_(new_bias_g)

        new_layer_D = nn.Linear(
            in_features=down_weights.shape[1],
            out_features=down_weights.shape[0],
            device="cuda",
            dtype=torch.float16,
            bias=True if comps.down_proj.bias is not None and bias else False,
        )
        new_layer_D.weight.data.copy_(down_weights.to(torch.float16))
        if comps.down_proj.bias is not None and bias:
            if new_layer_D.bias is not None:
                new_layer_D.bias.data.copy_(comps.down_proj.bias.data)

        self.replace_mlp_layers(
            layer_idx, new_layer_U, new_layer_D, new_layer_G, expert_idx=expert_idx
        )

    def slice_qk_dims(
        self,
        layer_idx: int,
        new_heads_Q: List[torch.Tensor],
        new_heads_K: List[torch.Tensor],
        new_bias_Q: List[torch.Tensor] = [],
        new_bias_K: List[torch.Tensor] = [],
        bias: bool = True,
    ):
        comps = self.get_attn_components(layer_idx)

        Q_heads = torch.cat(new_heads_Q, dim=0).to(device="cuda", dtype=torch.float16)
        K_heads = torch.cat(new_heads_K, dim=0).to(device="cuda", dtype=torch.float16)

        bias_Q = None
        if len(new_bias_Q) > 0:
            bias_Q = torch.cat(new_bias_Q, dim=0).to(device="cuda", dtype=torch.float16)

        bias_K = None
        if len(new_bias_K) > 0:
            bias_K = torch.cat(new_bias_K, dim=0).to(device="cuda", dtype=torch.float16)

        new_layer_Q = nn.Linear(
            in_features=Q_heads.shape[1],
            out_features=Q_heads.shape[0],
            device="cuda",
            dtype=torch.float16,
            bias=comps.q_proj.bias is not None and bias and (bias_Q is not None),
        )
        new_layer_Q.weight.data.copy_(Q_heads)
        if new_layer_Q.bias is not None and bias_Q is not None:
            new_layer_Q.bias.data.copy_(bias_Q)

        new_layer_K = nn.Linear(
            in_features=K_heads.shape[1],
            out_features=K_heads.shape[0],
            device="cuda",
            dtype=torch.float16,
            bias=comps.k_proj.bias is not None and bias and (bias_K is not None),
        )
        new_layer_K.weight.data.copy_(K_heads)
        if new_layer_K.bias is not None and bias_K is not None:
            new_layer_K.bias.data.copy_(bias_K)

        self.replace_attn_layers(
            layer_idx, new_q=new_layer_Q, new_k=new_layer_K, new_v=None, new_o=None
        )

    def slice_vo_dims(
        self,
        layer_idx: int,
        new_heads_V: List[torch.Tensor],
        new_heads_O: List[torch.Tensor],
        bias: bool,
    ):
        comps = self.get_attn_components(layer_idx)
        original_o = comps.o_proj
        original_v = comps.v_proj

        V_heads = torch.cat(new_heads_V, dim=0).to(device="cuda", dtype=torch.float16)
        O_heads = torch.cat(new_heads_O, dim=1).to(device="cuda", dtype=torch.float16)

        new_layer_V = nn.Linear(
            in_features=V_heads.shape[1],
            out_features=V_heads.shape[0],
            device="cuda",
            dtype=torch.float16,
            bias=False,  # As per original implementation, bias=False for V
        )
        new_layer_V.weight.data.copy_(V_heads)

        new_layer_O = nn.Linear(
            in_features=O_heads.shape[1],
            out_features=O_heads.shape[0],
            device="cuda",
            dtype=torch.float16,
            bias=True if original_o.bias is not None and bias else False,
        )
        new_layer_O.weight.data.copy_(O_heads)
        if original_o.bias is not None and bias and new_layer_O.bias is not None:
            new_layer_O.bias.data.copy_(original_o.bias.data)

        self.replace_attn_layers(
            layer_idx, new_q=None, new_k=None, new_v=new_layer_V, new_o=new_layer_O
        )

    # -- Helper hooks --

    @staticmethod
    def _make_fc_hook(layer_idx, cov_mlp_list):
        @torch.no_grad()
        def hook(module, inp, out):
            act = torch.nn.functional.relu(out.to(dtype=dtype_p, device=calib_device))
            H = act.detach().to(dtype=dtype_p).view(-1, act.size(-1))
            cov_mlp_list[layer_idx] += (H.T @ H).to(device=calib_device)

        return hook

    @staticmethod
    def _make_proj_hook(layer_idx, cov_list, n_heads, head_dim, d_model):
        @torch.no_grad()
        def hook(module, inp, out):
            proj_out = out.detach().to(dtype=dtype_p, device="cuda")
            proj = proj_out.view(-1, n_heads, head_dim)
            proj = proj.permute(1, 0, 2)  # [n_heads, B*T, head_dim]
            C_proj = torch.bmm(proj.transpose(1, 2), proj).to(calib_device)
            for h in range(n_heads):
                cov_list[layer_idx][h] += C_proj[h, :, :]

        return hook
