# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch DeepSeek model."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .latent_linear import LatentLinear

logger = logging.get_logger(__name__)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}
import transformers.modeling_utils as tm

# _orig = tm.set_module_tensor_to_device


# def _debug_set(*args, **kwargs):
#     try:
#         return _orig(*args, **kwargs)
#     except ValueError as e:
#         print(f"Error on param: {args[1]}")
#         raise


# tm.set_module_tensor_to_device = _debug_set


class DeepseekConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekModel`]. It is used to instantiate an DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeepSeek-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DeepseekModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        base_rank_up = config.base_ranks_up[layer_idx][0]
        base_rank_gate = config.base_ranks_gate[layer_idx][0]
        base_rank_down = config.base_ranks_down[layer_idx][0]
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts, None means dense model.
        n_routed_experts (`int`, *optional*, defaults to None):
            Number of routed experts, None means dense model.
        num_experts_per_tok (`int`, *optional*, defaults to None):
            Number of selected experts, None means dense model.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the weights of the routed experts.
        scoring_func (`str`, *optional*, defaults to 'softmax'):
            Method of computing expert weights.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary loss weight coefficient.
        seq_aux = (`bool`, *optional*, defaults to True):
            Whether to compute the auxiliary loss for each individual sample.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    ```python
    >>> from transformers import DeepseekModel, DeepseekConfig
    >>> # Initializing a Deepseek deepseek-7b style configuration
    >>> configuration = DeepseekConfig()
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=11008,
        moe_intermediate_size=1407,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=None,
        n_routed_experts=None,
        num_experts_per_tok=None,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        norm_topk_prob=False,
        scoring_func="softmax",
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, float)
            or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}"
            )


# configuration_deepseek import DeepseekConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.Deepseek.modeling_Deepseek._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    warnings.warn(
        "Calling `transformers.models.Deepseek.modeling_Deepseek._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.Deepseek.modeling_Deepseek.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape,
        dtype=dtype,
        device=device,
        past_key_values_length=past_key_values_length,
    )


class DeepseekRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(DeepseekRMSNorm)


class DeepseekRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->Deepseek
class DeepseekLinearScalingRotaryEmbedding(DeepseekRotaryEmbedding):
    """DeepseekRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->Deepseek
class DeepseekDynamicNTKScalingRotaryEmbedding(DeepseekRotaryEmbedding):
    """DeepseekRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1, rotary_mask=None):
    """Applies Rotary Position Embedding to the query and key tensors.

    rotary_mask of shape [n_kv_heads, compressed_head_dims]
     - layer_head_dims varies by layer

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        rotary_mask (`torch.Tensor | None`):
            Has shape [1, n_heads, 1, layer_head_dims], where the last dimension specifies which "columns" to choose
            from the cos, sin tensors
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids]
    sin = sin[position_ids]

    if rotary_mask is not None:
        n_heads, n_kv_heads = q.shape[1], k.shape[1]
        batch_size, seq_len = cos.shape[0], cos.shape[1]

        cos_q = cos.unsqueeze(unsqueeze_dim).expand(-1, n_heads, -1, cos.shape[-1])
        sin_q = sin.unsqueeze(unsqueeze_dim).expand(-1, n_heads, -1, sin.shape[-1])

        cos_k = cos.unsqueeze(unsqueeze_dim).expand(-1, n_kv_heads, -1, cos.shape[-1])
        sin_k = sin.unsqueeze(unsqueeze_dim).expand(-1, n_kv_heads, -1, sin.shape[-1])

        rotary_mask = rotary_mask.unsqueeze(0).unsqueeze(2)  # [1, n_kv_heads, 1, compressed_hd]
        rotary_mask = rotary_mask.expand(batch_size, -1, seq_len, -1)
        rotary_mask_q = torch.repeat_interleave(rotary_mask, n_heads // n_kv_heads, dim=1)
        # rotary_mask:   [1, n_kv_heads, seq_len, compressed_head_dim]
        # rotary_mask_q: [1, n_heads, seq_len, head_dim]

        cos_q = torch.gather(cos_q, 3, rotary_mask_q.to(cos.device))
        sin_q = torch.gather(sin_q, 3, rotary_mask_q.to(sin.device))

        cos_k = torch.gather(cos_k, 3, rotary_mask.to(cos.device))
        sin_k = torch.gather(sin_k, 3, rotary_mask.to(sin.device))

        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)

        return q_embed, k_embed

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def layer_rank(config, name: str, layer_idx: int, expert_idx: int | None, default: int) -> int:
    ranks = getattr(config, name, None)
    if ranks is None or layer_idx is None or layer_idx >= len(ranks):
        return default

    layer_ranks = ranks[layer_idx]
    if layer_ranks is None:
        return default
    if isinstance(layer_ranks, int):
        return layer_ranks

    idx = 0 if expert_idx is None or expert_idx >= len(layer_ranks) else expert_idx
    return layer_ranks[idx]


def factorized_projections(config, layer_idx: int) -> dict:
    factorized_map = getattr(config, "factorized_map", None)
    if factorized_map is None or layer_idx is None or layer_idx >= len(factorized_map):
        return {"up": False, "gate": False, "down": False}

    layer_map = factorized_map[layer_idx]
    return {key: bool(layer_map.get(key, False)) for key in ("up", "gate", "down")}


def shared_projection_rank(config, layer_idx: int, group_idx: int) -> int:
    return int(config.shared_proj_ranks[layer_idx][group_idx])


class DeepseekMLP(nn.Module):
    def __init__(
        self,
        config,
        hidden_size=None,
        intermediate_size=None,
        layer_idx=None,
        expert_idx=None,
        shared_weights: nn.ParameterDict | None = None,
        shared_expert=False,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.act_fn = ACT2FN[config.hidden_act]

        self.latent_gate, self.latent_up, self.latent_down = False, False, False

        if shared_expert:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            return

        compressed_int_size = self.intermediate_size
        if config.int_ranks[layer_idx] is not None:
            compressed_int_size = config.int_ranks[layer_idx][expert_idx or 0]

        if shared_weights is None:
            self.up_proj = nn.Linear(self.hidden_size, compressed_int_size, bias=False)
            self.gate_proj = nn.Linear(self.hidden_size, compressed_int_size, bias=False)
            self.down_proj = nn.Linear(compressed_int_size, self.hidden_size, bias=False)

            return

        def is_shared(key: str) -> bool:
            return key in shared_weights

        flipped_down = config.flipped_down

        # base_rank_up = config.base_ranks_up[layer_idx][expert_idx]
        # base_rank_gate = config.base_ranks_gate[layer_idx][expert_idx]
        # base_rank_down = config.base_ranks_down[layer_idx][expert_idx]

        group_idx = 0
        group_indices = getattr(config, "expert_to_shared_proj", None)
        if expert_idx is not None and group_indices is not None:
            group_idx = group_indices[layer_idx][expert_idx]

        shared_rank = shared_projection_rank(config, layer_idx, group_idx)
        base_rank_up = shared_rank
        base_rank_gate = shared_rank
        base_rank_down = shared_rank

        if flipped_down:
            down_in_features = compressed_int_size
            down_out_features = base_rank_down
        else:
            down_in_features = base_rank_down
            down_out_features = self.hidden_size

        if is_shared("gate"):
            self.gate_proj = LatentLinear(
                in_features=base_rank_gate,
                out_features=compressed_int_size,
                shared_weight_getter=lambda shared=shared_weights: (
                    shared["gate"] if shared is not None and "gate" in shared else None
                ),
                bias=False,
                flipped_down=flipped_down,
                is_down=False,
            )
            self.latent_gate = True
        else:
            self.gate_proj = nn.Linear(self.hidden_size, compressed_int_size, bias=False)

        if is_shared("up"):
            self.up_proj = LatentLinear(
                in_features=base_rank_up,
                out_features=compressed_int_size,
                shared_weight_getter=lambda shared=shared_weights: (
                    shared["up"] if shared is not None and "up" in shared else None
                ),
                bias=False,
                flipped_down=flipped_down,
                is_down=False,
            )
            self.latent_up = True
        else:
            self.up_proj = nn.Linear(self.hidden_size, compressed_int_size, bias=False)

        if is_shared("down"):
            self.down_proj = LatentLinear(
                in_features=down_in_features,
                out_features=down_out_features,
                shared_weight_getter=lambda shared=shared_weights: (
                    shared["down"] if shared is not None and "down" in shared else None
                ),
                bias=False,
                flipped_down=flipped_down,
                is_down=True,
            )
            self.latent_down = True
        else:
            self.down_proj = nn.Linear(compressed_int_size, self.hidden_size, bias=False)

    def forward(self, x, hidden_up=None, hidden_gate=None, project_down=True):
        if not (self.latent_gate or self.latent_up or self.latent_down):
            return self.dense_forward(x)

        if self.latent_gate:
            h_gate = hidden_gate if hidden_gate is not None else x
            gate_states = self.gate_proj(h_gate, use_shared=hidden_gate is None)
        else:
            gate_states = self.gate_proj(x)

        if self.latent_up:
            h_up = hidden_up if hidden_up is not None else x
            up_states = self.up_proj(h_up, use_shared=hidden_up is None)
        else:
            up_states = self.up_proj(x)

        if self.latent_down:
            return self.down_proj(self.act_fn(gate_states) * up_states, use_shared=project_down)
        return self.down_proj(self.act_fn(gate_states) * up_states)

    def dense_forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config, layer_idx=None, shared_weights: nn.ModuleList | None = None):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        # The decoder layer owns this ModuleList. Keep only a tuple here to
        # avoid registering duplicate state-dict paths for shared parameters.
        self.shared_projs = tuple(shared_weights) if shared_weights is not None else ()
        self.expert_group_indices = config.expert_to_shared_proj[layer_idx]
        self._expert_to_group_cache = None

        self.experts = nn.ModuleList(
            [
                DeepseekMLP(
                    config,
                    intermediate_size=config.moe_intermediate_size,
                    layer_idx=layer_idx,
                    expert_idx=i,
                    shared_weights=(
                        self.shared_projs[self.expert_group_indices[i]]
                        if self.shared_projs
                        else None
                    ),
                )
                for i in range(config.n_routed_experts)
            ]
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(
                config=config,
                intermediate_size=intermediate_size,
                layer_idx=layer_idx,
                shared_expert=True,
            )

    """Fast. one sync per layer"""

    def _precompute_shared(self) -> bool:
        # x @ P_up / x @ P_gate is hoisted out of the experts only in the
        # flipped_down layout; otherwise each LatentLinear applies P itself.
        return bool(self.shared_projs) and getattr(self.config, "flipped_down", True)

    def _expert_to_group(self, device):
        cached = self._expert_to_group_cache
        if cached is None or cached.device != device:
            cached = torch.as_tensor(self.expert_group_indices, dtype=torch.long, device=device)
            self._expert_to_group_cache = cached
        return cached

    def _group_membership(self, flat_expert_indices, num_tokens):
        # in_group[t, g] = token t is routed to at least one expert of group g.  [T, G]
        # Built with scatter_, so no host sync.
        slot_group = self._expert_to_group(flat_expert_indices.device)[flat_expert_indices]
        in_group = torch.zeros(
            num_tokens, len(self.shared_projs), dtype=torch.bool, device=flat_expert_indices.device
        )
        in_group.scatter_(1, slot_group.view(num_tokens, self.num_experts_per_tok), True)
        return in_group, in_group.sum(dim=0)

    @staticmethod
    def _group_plan(in_group, group_counts):
        # group_counts: per-group token counts, already on the host.
        # A stable sort that puts each group's members first leaves them in ascending
        # token order, so group g's tokens are the first group_counts[g] entries of column g.
        order = torch.argsort((~in_group).to(torch.uint8), dim=0, stable=True)  # [T, G]
        group_tokens = [order[:count, g] for g, count in enumerate(group_counts)]
        # token_rank[t, g] = row of token t in group g's projected activations
        # (valid where in_group[t, g]); matches the ascending order above.
        token_rank = in_group.cumsum(dim=0) - 1  # [T, G]
        return group_tokens, token_rank

    def _experts_forward(self, x, token_idx_per_expert, weight_per_expert, group_plan=None):
        out = torch.zeros_like(x)

        if group_plan is None:
            # No hoisted projection: each expert applies its shared P itself
            # (use_shared=True inside LatentLinear) or has no shared P at all.
            for expert_idx, token_idx in token_idx_per_expert.items():
                expert_out = self.experts[expert_idx](x[token_idx]) * weight_per_expert[expert_idx]
                # Same sum as scatter_reduce_(..., reduce="sum") without the [n, d] index tensor.
                out.index_add_(0, token_idx, expert_out)
            return out

        group_tokens, rows_per_expert = group_plan

        experts_by_group: dict[int, list[int]] = {}
        for expert_idx in token_idx_per_expert:
            experts_by_group.setdefault(self.expert_group_indices[expert_idx], []).append(
                expert_idx
            )

        for group_idx, group_experts in experts_by_group.items():
            shared = self.shared_projs[group_idx]

            # Tokens routed to at least one expert of this group. A token routed to two
            # experts of the same group appears once, so it is projected once.
            # P_up / P_gate: [d, r]  ->  [T_g, r]
            x_group = x[group_tokens[group_idx]]
            group_up = x_group @ shared["up"] if "up" in shared else None
            group_gate = x_group @ shared["gate"] if "gate" in shared else None
            group_latent = (
                x.new_zeros(x_group.shape[0], shared["down"].shape[1]) if "down" in shared else None
            )
            needs_x = group_up is None or group_gate is None

            for expert_idx in group_experts:
                token_idx = token_idx_per_expert[expert_idx]
                rows = rows_per_expert[expert_idx]
                expert_out = (
                    self.experts[expert_idx](
                        x[token_idx] if needs_x else None,
                        hidden_up=group_up[rows] if group_up is not None else None,
                        hidden_gate=group_gate[rows] if group_gate is not None else None,
                        project_down=group_latent is None,
                    )
                    * weight_per_expert[expert_idx]
                )
                if group_latent is None:
                    out.index_add_(0, token_idx, expert_out)
                else:
                    group_latent.index_add_(0, rows, expert_out)

            if group_latent is not None:
                out.index_add_(0, group_tokens[group_idx], group_latent @ shared["down"].T)
        return out

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # Slot s of flat_topk_idx is the (s % k)-th expert choice of token s // k.
            slots_per_expert = {
                i: (flat_topk_idx == i).nonzero().squeeze(-1) for i in range(len(self.experts))
            }
            token_idx_per_expert = {
                i: slots // self.num_experts_per_tok for i, slots in slots_per_expert.items()
            }
            group_plan = None
            if self._precompute_shared():
                in_group, group_counts = self._group_membership(
                    flat_topk_idx, hidden_states.shape[0]
                )
                group_tokens, token_rank = self._group_plan(in_group, group_counts.tolist())
                rows_per_expert = {
                    i: token_rank[token_idx, self.expert_group_indices[i]]
                    for i, token_idx in token_idx_per_expert.items()
                }
                group_plan = (group_tokens, rows_per_expert)

            flat_topk_weight = topk_weight.view(-1, 1)
            y = self._experts_forward(
                hidden_states,
                token_idx_per_expert,
                {i: flat_topk_weight[slots] for i, slots in slots_per_expert.items()},
                group_plan,
            )
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(
                hidden_states,
                flat_topk_idx,
                topk_weight.view(-1, 1),
            ).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        num_experts = len(self.experts)
        sorted_experts, idxs = flat_expert_indices.sort()
        token_idxs = idxs // self.num_experts_per_tok
        sorted_weights = flat_expert_weights[idxs]

        # Tokens per expert via scatter_add_ instead of bincount: CUDA bincount reads the
        # input's max and min back to the host to size its output, which syncs.
        expert_counts = torch.zeros(num_experts, dtype=torch.long, device=x.device).scatter_add_(
            0, flat_expert_indices, torch.ones_like(flat_expert_indices)
        )

        sorted_rows = None
        if self._precompute_shared():
            in_group, group_counts = self._group_membership(flat_expert_indices, x.shape[0])
            # The one host sync in this layer: expert and group counts in a single transfer.
            counts = torch.cat([expert_counts, group_counts]).tolist()
            expert_counts_host = counts[:num_experts]
            group_tokens, token_rank = self._group_plan(in_group, counts[num_experts:])
            sorted_rows = token_rank[token_idxs, self._expert_to_group(x.device)[sorted_experts]]
        else:
            expert_counts_host = expert_counts.tolist()

        # Firing experts only, sliced with host ints (no further syncs).
        token_idx_per_expert, weight_per_expert, rows_per_expert = {}, {}, {}
        start = 0
        for i, count in enumerate(expert_counts_host):
            if count:
                token_idx_per_expert[i] = token_idxs[start : start + count]
                weight_per_expert[i] = sorted_weights[start : start + count]
                if sorted_rows is not None:
                    rows_per_expert[i] = sorted_rows[start : start + count]
            start += count

        group_plan = (group_tokens, rows_per_expert) if sorted_rows is not None else None
        return self._experts_forward(x, token_idx_per_expert, weight_per_expert, group_plan)

    # def _precompute_shared(self) -> bool:
    #     # x @ P_up / x @ P_gate is hoisted out of the experts only in the
    #     # flipped_down layout
    #     return bool(self.shared_projs)

    # def _iter_experts(self, hidden_states, token_idx_per_expert):
    #     if not self._precompute_shared():
    #         # (use_shared=True inside LatentLinear) or has no shared P at all.
    #         for expert_idx in token_idx_per_expert:
    #             yield expert_idx, None, None
    #         return

    #     experts_by_group: dict[int, list[int]] = {}
    #     for expert_idx in token_idx_per_expert:
    #         experts_by_group.setdefault(self.expert_group_indices[expert_idx], []).append(
    #             expert_idx
    #         )

    #     row_in_group = torch.empty(
    #         hidden_states.shape[0], dtype=torch.long, device=hidden_states.device
    #     )

    #     for group_idx, group_experts in experts_by_group.items():
    #         shared = self.shared_projs[group_idx]

    #         # Tokens routed to at least one expert of this group, in ascending order.
    #         # A token routed to two experts of the same group appears once, so it is
    #         # projected once and the result is shared by both experts.
    #         group_tokens = torch.cat(
    #             [token_idx_per_expert[expert_idx] for expert_idx in group_experts]
    #         ).unique()

    #         # P_up / P_gate: [d, r]  ->  [T_g, r], with T_g = number of tokens routed to the group
    #         x_group = hidden_states[group_tokens]
    #         group_up = x_group @ shared["up"] if "up" in shared else None
    #         group_gate = x_group @ shared["gate"] if "gate" in shared else None

    #         row_in_group[group_tokens] = torch.arange(
    #             group_tokens.numel(), device=group_tokens.device
    #         )

    #         for expert_idx in group_experts:
    #             rows = row_in_group[token_idx_per_expert[expert_idx]]
    #             yield (
    #                 expert_idx,
    #                 group_up[rows] if group_up is not None else None,
    #                 group_gate[rows] if group_gate is not None else None,
    #             )

    # def forward(self, hidden_states):
    #     identity = hidden_states
    #     orig_shape = hidden_states.shape
    #     topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
    #     hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    #     flat_topk_idx = topk_idx.view(-1)

    #     if self.training:
    #         # Slot s of flat_topk_idx is the (s % k)-th expert choice of token s // k.
    #         slots_per_expert = {
    #             i: (flat_topk_idx == i).nonzero().squeeze(-1) for i in range(len(self.experts))
    #         }
    #         token_idx_per_expert = {
    #             i: slots // self.num_experts_per_tok for i, slots in slots_per_expert.items()
    #         }
    #         y = hidden_states.new_empty(flat_topk_idx.shape[0], hidden_states.shape[-1])
    #         for i, hidden_up, hidden_gate in self._iter_experts(
    #             hidden_states, token_idx_per_expert
    #         ):
    #             y[slots_per_expert[i]] = self.experts[i](
    #                 hidden_states[token_idx_per_expert[i]],
    #                 hidden_up=hidden_up,
    #                 hidden_gate=hidden_gate,
    #             )
    #         y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    #         y = y.view(*orig_shape)
    #         y = AddAuxiliaryLoss.apply(y, aux_loss)
    #     else:
    #         y = self.moe_infer(
    #             hidden_states,
    #             flat_topk_idx,
    #             topk_weight.view(-1, 1),
    #         ).view(*orig_shape)
    #     if self.config.n_shared_experts is not None:
    #         y = y + self.shared_experts(identity)
    #     return y

    # @torch.no_grad()
    # def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
    #     expert_cache = torch.zeros_like(x)
    #     idxs = flat_expert_indices.argsort()
    #     tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
    #     token_idxs = idxs // self.num_experts_per_tok
    #     # Firing experts only: expert_idx -> its [start_idx, end_idx) range in idxs / token_idxs.
    #     expert_ranges = {}
    #     for i, end_idx in enumerate(tokens_per_expert):
    #         start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
    #         if start_idx == end_idx:
    #             continue
    #         expert_ranges[i] = slice(start_idx, end_idx)

    #     token_idx_per_expert = {i: token_idxs[r] for i, r in expert_ranges.items()}
    #     for i, hidden_up, hidden_gate in self._iter_experts(x, token_idx_per_expert):
    #         expert = self.experts[i]
    #         exp_token_idx = token_idx_per_expert[i]
    #         expert_tokens = x[exp_token_idx]
    #         expert_out = expert(
    #             expert_tokens,
    #             hidden_up=hidden_up,
    #             hidden_gate=hidden_gate,
    #         )
    #         expert_out.mul_(flat_expert_weights[idxs[expert_ranges[i]]])
    #         expert_cache.scatter_reduce_(
    #             0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce="sum"
    #         )
    #     return expert_cache


# def project_shared(self, hidden_states, flat_topk_idx=None):
#     if not self.shared_projs or not getattr(self.config, "flipped_down", True):
#         return None

#     # With several shared projections per layer only a subset of the groups is
#     # usually routed to, so skip the P_up / P_gate matmuls of the groups whose
#     # experts do not fire for this batch. Groups reported as inactive fall back
#     # to the per-expert `use_shared=True` path, which is slower but equivalent.
#     active_groups = None
#     if flat_topk_idx is not None and len(self.shared_projs) > 1:
#         active_groups = {
#             self.expert_group_indices[expert_idx]
#             for expert_idx in torch.unique(flat_topk_idx).tolist()
#         }

#     return [
#         (
#             (
#                 hidden_states @ shared["up"] if "up" in shared else None,
#                 hidden_states @ shared["gate"] if "gate" in shared else None,
#             )
#             if active_groups is None or group_idx in active_groups
#             else (None, None)
#         )
#         for group_idx, shared in enumerate(self.shared_projs)
#     ]

# def forward(self, hidden_states):
#     identity = hidden_states
#     orig_shape = hidden_states.shape
#     topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
#     hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
#     flat_topk_idx = topk_idx.view(-1)

#     shared_activations = self.project_shared(hidden_states, flat_topk_idx=flat_topk_idx)

#     if self.training:
#         hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
#         if shared_activations is not None:
#             shared_activations = [
#                 (
#                     (
#                         None
#                         if hidden_up is None
#                         else hidden_up.repeat_interleave(self.num_experts_per_tok, dim=0)
#                     ),
#                     (
#                         None
#                         if hidden_gate is None
#                         else hidden_gate.repeat_interleave(self.num_experts_per_tok, dim=0)
#                     ),
#                 )
#                 for hidden_up, hidden_gate in shared_activations
#             ]
#         y = torch.empty_like(hidden_states)
#         for i, expert in enumerate(self.experts):
#             selected = flat_topk_idx == i
#             hidden_up, hidden_gate = (
#                 (None, None)
#                 if shared_activations is None
#                 else shared_activations[self.expert_group_indices[i]]
#             )
#             y[selected] = expert(
#                 hidden_states[selected],
#                 hidden_up=None if hidden_up is None else hidden_up[selected],
#                 hidden_gate=None if hidden_gate is None else hidden_gate[selected],
#             )
#         y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
#         y = y.view(*orig_shape)
#         y = AddAuxiliaryLoss.apply(y, aux_loss)
#     else:
#         y = self.moe_infer(
#             hidden_states,
#             flat_topk_idx,
#             topk_weight.view(-1, 1),
#             shared_activations=shared_activations,
#         ).view(*orig_shape)
#     if self.config.n_shared_experts is not None:
#         y = y + self.shared_experts(identity)
#     return y

# @torch.no_grad()
# def moe_infer(self, x, flat_expert_indices, flat_expert_weights, shared_activations=None):
#     expert_cache = torch.zeros_like(x)
#     idxs = flat_expert_indices.argsort()
#     tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
#     token_idxs = idxs // self.num_experts_per_tok
#     for i, end_idx in enumerate(tokens_per_expert):
#         start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
#         if start_idx == end_idx:
#             continue
#         expert = self.experts[i]
#         exp_token_idx = token_idxs[start_idx:end_idx]
#         expert_tokens = x[exp_token_idx]
#         hidden_up, hidden_gate = (
#             (None, None)
#             if shared_activations is None
#             else shared_activations[self.expert_group_indices[i]]
#         )
#         expert_out = expert(
#             expert_tokens,
#             hidden_up=None if hidden_up is None else hidden_up[exp_token_idx],
#             hidden_gate=None if hidden_gate is None else hidden_gate[exp_token_idx],
#         )
#         expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
#         expert_cache.scatter_reduce_(
#             0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce="sum"
#         )
#     return expert_cache


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->Deepseek
class DeepseekAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: DeepseekConfig,
        layer_idx: Optional[int] = None,
        layer_rotary_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.layer_rotary_mask = layer_rotary_mask

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rope_head_dim = self.head_dim

        if layer_idx is not None and getattr(config, "q_ranks", None) is not None:
            self.compressed_q_dims = config.q_ranks[layer_idx]
            self.compressed_k_dims = config.k_ranks[layer_idx]
            self.compressed_v_dims = config.v_ranks[layer_idx]
            self.compressed_o_dims = config.o_ranks[layer_idx]
        else:
            self.compressed_q_dims = self.num_heads * self.head_dim
            self.compressed_k_dims = self.num_key_value_heads * self.head_dim
            self.compressed_v_dims = self.num_key_value_heads * self.head_dim
            self.compressed_o_dims = self.num_heads * self.head_dim

        self.head_dim = self.compressed_q_dims // self.num_heads
        self.v_head_dim = self.compressed_v_dims // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.compressed_q_dims, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.compressed_k_dims, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.compressed_v_dims, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.compressed_o_dims, self.hidden_size, bias=config.attention_bias
        )
        self._init_rope()

    def _init_rope(self):
        rope_dim = self.head_dim if self.layer_rotary_mask is None else self.rope_head_dim
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekRotaryEmbedding(
                rope_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekLinearScalingRotaryEmbedding(
                    rope_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekDynamicNTKScalingRotaryEmbedding(
                    rope_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.v_head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, rotary_mask=self.layer_rotary_mask
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.compressed_o_dims)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->Deepseek
class DeepseekFlashAttention2(DeepseekAttention):
    """
    Deepseek flash attention module. This module inherits from `DeepseekAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # DeepseekFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.v_head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, rotary_mask=self.layer_rotary_mask
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (DeepseekRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.compressed_o_dims).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in DeepseekFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
                self._upad_input(
                    query_states, key_states, value_states, attention_mask, query_length
                )
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->Deepseek
class DeepseekSdpaAttention(DeepseekAttention):
    """
    Deepseek attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `DeepseekAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from DeepseekAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "DeepseekModel is using DeepseekSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.v_head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, rotary_mask=self.layer_rotary_mask
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.compressed_o_dims)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


Deepseek_ATTENTION_CLASSES = {
    "eager": DeepseekAttention,
    "flash_attention_2": DeepseekFlashAttention2,
    "sdpa": DeepseekSdpaAttention,
}


class DeepseekDecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepseekConfig,
        layer_idx: int,
        layer_rotary_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = Deepseek_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx, layer_rotary_mask=layer_rotary_mask
        )

        is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )

        flipped_down = config.flipped_down

        default_int_size = config.moe_intermediate_size if is_moe else config.intermediate_size

        int_rank = config.int_ranks[layer_idx][0]

        down_in_features = self.hidden_size if flipped_down else int_rank

        """
        shared projs dims: (d = hidden, D = int)
        Up/Gate: [d, r], Down: [D, r]

        coefficients:
        Up/Gate: [D, r], Down: [d, r]
        """

        features_factorized = factorized_projections(config, layer_idx)
        shared_weights = None
        if any(features_factorized.values()):
            group_indices = config.expert_to_shared_proj[layer_idx] if is_moe else [0]
            n_shared_groups = max(group_indices) + 1
            shared_weights = nn.ModuleList()
            for group_idx in range(n_shared_groups):
                base_rank = shared_projection_rank(config, layer_idx, group_idx)
                shared_dict = {}
                if features_factorized["up"]:
                    shared_dict["up"] = nn.Parameter(torch.empty(self.hidden_size, base_rank))

                if features_factorized["gate"]:
                    shared_dict["gate"] = nn.Parameter(torch.empty(self.hidden_size, base_rank))

                if features_factorized["down"]:
                    shared_dict["down"] = nn.Parameter(torch.empty(down_in_features, base_rank))

                shared_weights.append(nn.ParameterDict(shared_dict))

        self.shared_weights = shared_weights

        self.mlp = (
            DeepseekMoE(config, layer_idx=layer_idx, shared_weights=self.shared_weights)
            if is_moe
            else DeepseekMLP(
                config,
                layer_idx=layer_idx,
                shared_weights=None if self.shared_weights is None else self.shared_weights[0],
            )
        )
        self.input_layernorm = DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


Deepseek_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`DeepseekConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Deepseek Model outputting raw hidden-states without any specific head on top.",
    Deepseek_START_DOCSTRING,
)
class DeepseekPreTrainedModel(PreTrainedModel):
    config_class = DeepseekConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


Deepseek_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Deepseek Model outputting raw hidden-states without any specific head on top.",
    Deepseek_START_DOCSTRING,
)
class DeepseekModel(DeepseekPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekDecoderLayer`]
    Args:
        config: DeepseekConfig
    """

    def __init__(self, config: DeepseekConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        mask_path = getattr(config, "mask_path", None)
        rotary_masks = None
        if mask_path is not None:
            print(f"mask_path = {mask_path}")
            rotary_masks = torch.load(mask_path, map_location="cuda")

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                DeepseekDecoderLayer(
                    config,
                    layer_idx,
                    None if rotary_masks is None else rotary_masks[layer_idx],
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(Deepseek_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            )
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DeepseekForCausalLM(DeepseekPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(Deepseek_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, DeepseekForCausalLM
        >>> model = DeepseekForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = cache_length
                max_cache_length = None
            # if isinstance(past_key_values, Cache):
            #     cache_length = past_key_values.get_seq_length()
            #     past_length = past_key_values.seen_tokens
            #     max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


@add_start_docstrings(
    """
    The Deepseek Model transformer with a sequence classification head on top (linear layer).
    [`DeepseekForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.
    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    Deepseek_START_DOCSTRING,
)
class DeepseekForSequenceClassification(DeepseekPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeepseekModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(Deepseek_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
