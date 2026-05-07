# pyright: reportPrivateImportUsage=false, reportOptionalSubscript=false
# pyright: reportOptionalMemberAccess=false, reportOptionalOperand=false
# pyright: reportOperatorIssue=false, reportArgumentType=false
# pyright: reportAssignmentType=false, reportReturnType=false
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false
# pyright: reportGeneralTypeIssues=false, reportOptionalIterable=false
# ruff: noqa
#
# Adapted from `transformers.models.gemma4.modeling_gemma4` (HF transformers 5.7.0).
# Reduced to the text-generation path: `Gemma4ForCausalLM` and the layers it
# needs. Audio, vision, multimodal embedder, conditional generation, and the
# multimodal `Gemma4Model` were removed wholesale. Helper utilities (RoPE
# init, masking, attention dispatch, …) remain imported from `transformers`;
# they can be brought in-tree as the engine matures.
#
# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.integrations import use_experts_implementation
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_accelerate_available,
)
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Gemma4 outputs, with hidden states and attentions.
    """
)
class Gemma4ModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
    """

    image_hidden_states: torch.FloatTensor | None = None

    audio_hidden_states: torch.FloatTensor | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Gemma4 causal language model (or autoregressive) outputs.
    """
)
class Gemma4CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None

    audio_hidden_states: torch.FloatTensor | None = None


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        # `torch.pow(..., -0.5)` instead of `rsqrt` keeps Torch / JAX bit-equivalent.
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float | int = 0.0,
        scaling: float | None = None,
        softcap: float | None = None,
        **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim ** -0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Gemma4TextMLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma4TextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma4TextConfig, device=None, layer_type=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.layer_types = set(config.layer_types)
        self.rope_init_fns: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}
        self.rope_type: dict[str, str] = {}

        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            if (rope_type := rope_params["rope_type"]) != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = self.compute_default_rope_parameters

            self.rope_init_fns[layer_type] = rope_init_fn
            self.rope_type[layer_type] = rope_type

            rope_init_fn_kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, **rope_init_fn_kwargs)
            self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
            config: Gemma4TextConfig | None = None,
            device: Optional["torch.device"] = None,
            seq_len: int | None = None,
            layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Gemma4TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
        num_key_value_heads = (
            config.num_global_key_value_heads if self.use_alternative_attention else config.num_key_value_heads
        )
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = config.use_bidirectional_attention != "all"

        # Shared kv cache
        first_kv_shared_layer_idx = self.config.num_hidden_layers - getattr(self.config, "num_kv_shared_layers", 0)
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            # For shared layers, find the last non-shared layer of the same type before sharing starts
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            # For non-shared layers, store full-length kv if this is the last non-shared layer of its type
            self.store_full_length_kv = layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(
                config.layer_types[layer_idx]
            )

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

        # Layers sharing kv states don't need any weight matrices
        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

            self.k_proj = nn.Linear(
                config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.v_proj = (
                nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
                if not self.use_alternative_attention
                else None
            )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor,
            attention_mask: torch.Tensor | None,
            shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
            past_key_values: Cache | None = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer.
        # We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember the full states in their Cache
        # once we are past the sliding window - so we always use `shared_kv_states` instead, even when past_key_values is not None
        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
            # Device of past layer may be different from current one
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None and not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_idx] = key_states, value_states

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_experts_implementation
class Gemma4TextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Gemma4TextRouter(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size ** -0.5
        self.eps = config.rms_norm_eps

        self.norm = Gemma4RMSNorm(self.hidden_size, eps=self.eps, with_scale=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)  # [B*S, E]
        router_probabilities = nn.functional.softmax(expert_scores, dim=-1)

        # topk returns both values (probabilities) and indices directly
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )  # both [B*S, K]

        # Normalize the top-k weights so they sum to 1 per token
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

        # Apply per-expert scale directly to the weights
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return router_probabilities, top_k_weights, top_k_index


class Gemma4TextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = ACT2FN[config.hidden_activation]
            self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = Gemma4TextRouter(config)
            self.experts = Gemma4TextExperts(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            per_layer_input: torch.Tensor = None,
            shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
            position_embeddings: torch.Tensor = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            # Take hidden states before MLP here
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            # Combine mlp and moe outputs
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


class Gemma4TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


@auto_docstring
class Gemma4PreTrainedModel(PreTrainedModel):
    config: Gemma4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]
    _skip_keys_device_placement = ["past_key_values", "shared_kv_states"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = None  # override
    input_modalities = ("image", "text", "video", "audio")

    @torch.no_grad()
    def _init_weights(self, module):
        # Vision/audio/clipped-linear branches removed with the multimodal
        # classes; only text-stack initializers remain.
        super()._init_weights(module)
        if isinstance(module, Gemma4TextRotaryEmbedding):
            for layer_type, rope_init_fn in module.rope_init_fns.items():
                rope_init_fn_kwargs = {"layer_type": layer_type}
                if layer_type == "full_attention" and module.rope_type[layer_type] == "proportional":
                    rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

                curr_inv_freq, _ = rope_init_fn(module.config, **rope_init_fn_kwargs)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, Gemma4TextScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        elif isinstance(module, Gemma4TextRouter):
            init.ones_(module.scale)
            init.ones_(module.per_expert_scale)
        elif isinstance(module, Gemma4TextExperts):
            std = self.config.initializer_range
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, Gemma4TextDecoderLayer):
            init.ones_(module.layer_scalar)

    def get_per_layer_input_embeddings(self):
        return self.base_model.embed_tokens_per_layer

    def set_per_layer_input_embeddings(self, value):
        self.base_model.embed_tokens_per_layer = value

    def resize_token_embeddings(
            self,
            new_num_tokens: int | None = None,
            pad_to_multiple_of: int | None = None,
            mean_resizing: bool = True,
    ) -> nn.Embedding:
        inputs_embeds = super().resize_token_embeddings(
            new_num_tokens=new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=mean_resizing,
        )
        self._resize_per_layer_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        return inputs_embeds

    def _resize_per_layer_embeddings(
            self,
            new_num_tokens: int | None = None,
            pad_to_multiple_of: int | None = None,
            mean_resizing: bool = True,
    ):
        self.config.get_text_config().vocab_size_per_layer_input = self.vocab_size
        if self.config.get_text_config().hidden_size_per_layer_input:
            embed_tokens_per_layer = self.get_per_layer_input_embeddings()
            new_embeddings_per_layer = self._get_resized_embeddings(
                embed_tokens_per_layer, new_num_tokens, pad_to_multiple_of, mean_resizing
            )
            if hasattr(embed_tokens_per_layer, "_hf_hook"):
                hook = embed_tokens_per_layer._hf_hook
                add_hook_to_module(new_embeddings_per_layer, hook)
            new_embeddings_per_layer.requires_grad_(embed_tokens_per_layer.weight.requires_grad)
            self.set_per_layer_input_embeddings(new_embeddings_per_layer)


@auto_docstring(custom_intro="The base Gemma 4 language model without a language modeling head.")
class Gemma4TextModel(Gemma4PreTrainedModel):
    config: Gemma4TextConfig
    input_modalities = ("text",)
    _can_record_outputs = {
        "router_logits": OutputRecorder(Gemma4TextRouter, index=0),
        "hidden_states": Gemma4TextDecoderLayer,
        "attentions": Gemma4TextAttention,
    }

    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma4 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size ** 0.5
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.unique_layer_types = set(self.config.layer_types)

        # Per-Layer Embeddings (PLE): auxiliary embedding that feeds a residual signal
        # into each decoder layer. See `get_per_layer_inputs()` and `project_per_layer_inputs()`
        # for the full pipeline. The embedding is packed: total dim = num_layers * per_layer_dim.
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=config.hidden_size_per_layer_input ** 0.5,
            )
            self.per_layer_input_scale = 2.0 ** -0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = config.hidden_size ** -0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

        # Update `_keys_to_ignore_on_load_unexpected` to drop all k/v proj and norms for the shared layers
        self._keys_to_ignore_on_load_unexpected = []
        for i, layer in enumerate(self.layers):
            if layer.self_attn.is_kv_shared_layer:
                self._keys_to_ignore_on_load_unexpected.extend(
                    [f"layers.{i}.self_attn.{name}" for name in ("k_proj", "v_proj", "k_norm", "v_norm")]
                )

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            per_layer_inputs: torch.Tensor | None = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        per_layer_inputs (`torch.Tensor` of shape `(batch_size, sequence_length, num_hidden_layers, hidden_size_per_layer_input)`, *optional*):
            Pre-computed per-layer input embeddings. When provided, these are used directly instead of being
            computed from `input_ids` via `get_per_layer_inputs()`. This is primarily used by the multimodal
            model (`Gemma4Model`) which pre-computes per-layer inputs from the original `input_ids` *before*
            merging multimodal soft tokens into `inputs_embeds` — at which point the original token ids are
            no longer recoverable.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # Initialize as empty dict - it will be filled in the right layers
        shared_kv_states = {}

        # decoder layers
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def get_per_layer_inputs(self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None) -> torch.Tensor:
        """Compute the token-identity component of Per-Layer Embeddings (PLE).

        Looks up `input_ids` in `embed_tokens_per_layer` (a scaled embedding that multiplies
        by `sqrt(hidden_size_per_layer_input)`) and reshapes the packed output from
        `[batch, seq, num_hidden_layers * hidden_size_per_layer_input]` to
        `[batch, seq, num_hidden_layers, hidden_size_per_layer_input]`.

        If only `inputs_embeds` is provided (no `input_ids`), reverses the main embedding
        to recover `input_ids` for the PLE lookup.
        """
        if not self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Attempting to call get_per_layer_inputs() from a model initialized with a config that does not support"
                f" per-layer embeddings. {self.config}"
            )

        # If only inputs_embeds are provided, reverse main embedding to find the input_ids - this allows to `generate`
        # from `inputs_embeds` only as other models (otherwise it would need the value from both embeddings)
        if input_ids is None:
            with torch.no_grad():
                input_ids = (
                    (
                            inputs_embeds[:, :, None, :]
                            == self.embed_tokens.weight[None, None, :, :] * self.config.hidden_size ** 0.5
                    )
                    .all(dim=3)
                    .nonzero()[:, 2]
                )
                try:
                    input_ids = input_ids.view(inputs_embeds.shape[:2])
                except RuntimeError:
                    raise RuntimeError(
                        "It seems like you tried to call `forward` from `inputs_embeds` without providing `input_ids`, and that "
                        "the `inputs_embeds` you provided do not exactly match the embedding weights. Since Gemma4 needs to reverse "
                        "the embedding to compute another embedding, make sure you provide exact `inputs_embeds`"
                    )

        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
            self,
            inputs_embeds: torch.Tensor,
            per_layer_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the context-aware component of PLE and combine with token-identity.

        Projects `inputs_embeds` through `per_layer_model_projection` (Linear), scales by
        `1/sqrt(hidden_size)`, reshapes to `[batch, seq, num_layers, ple_dim]`, and normalizes
        with `per_layer_projection_norm` (RMSNorm).

        If `per_layer_inputs` (the token-identity component from `get_per_layer_inputs()`)
        is provided, combines both: `(context_projection + token_identity) * (1/sqrt(2))`.
        If `per_layer_inputs` is None (e.g. for multimodal inputs where input_ids are not
        available), returns just the context projection.
        """
        if not self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Attempting to call project_per_layer_inputs() from a model initialized with a config that does not"
                f" support per-layer embeddings. {self.config}"
            )

        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale


@auto_docstring(custom_intro="The base Gemma 4 language model with a language modeling head.")
class Gemma4ForCausalLM(Gemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma4TextConfig
    base_model_prefix = "model"

    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)
        self.model = Gemma4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Grab the ones from the child
        self._keys_to_ignore_on_load_unexpected = [
            f"model.{name}" for name in self.model._keys_to_ignore_on_load_unexpected
        ]

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma4ForCausalLM

        >>> model = Gemma4ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def sliding_window_mask_function(sliding_window: tuple[int, int]) -> Callable:
    """
    This creates uni/bidirectional attention mask with sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        left_window_size, right_window_size = sliding_window

        dist = q_idx - kv_idx
        left_mask = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask

    return inner_mask


__all__ = [
    "Gemma4ForCausalLM",
    "Gemma4PreTrainedModel",
    "Gemma4TextModel",
]
