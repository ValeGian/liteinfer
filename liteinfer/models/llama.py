# pyright: reportPrivateImportUsage=false, reportOptionalSubscript=false
# pyright: reportOptionalOperand=false, reportOperatorIssue=false
# pyright: reportArgumentType=false, reportAssignmentType=false
"""Llama generation path. Adapted from `transformers.models.llama.modeling_llama`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig

from liteinfer.models.attention import UNIVERSAL_IMPLEMENTATION, DenseKV, resolve


@dataclass
class CausalLMOutput:
    """Minimal forward output. Mirrors the fields liteinfer reads."""

    logits: torch.Tensor
    past_key_values: Cache | None = None


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class LlamaRMSNorm(nn.Module):
    """RMSNorm in float32 to match HF training precision exactly."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    """Position-dependent (cos, sin) tables for RoPE.

    The init function is delegated to HF's ROPE registry so frequency
    interpolation schemes (e.g. `llama3`, `linear`, `dynamic`, `yarn`)
    keep working without re-implementing each one.
    """

    inv_freq: torch.Tensor

    def __init__(self, config: LlamaConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        rope_params: dict = config.rope_parameters  # always a dict once config is built
        self.rope_type = rope_params["rope_type"]
        if self.rope_type == "default":
            # Standard RoPE: no frequency scaling, custom base theta.
            theta = rope_params.get("rope_theta", config.default_theta)
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
            self.attention_scaling = 1.0
        else:
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        # RoPE math is sensitive to fp16/bf16 rounding; force fp32.
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.attention_scaling).to(x.dtype)
        sin = (emb.sin() * self.attention_scaling).to(x.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)  # broadcast over heads
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Grouped-query causal self-attention.

    The kernel that computes the attention itself comes from
    `models/attention.py`. `models/loader.py` picks it and records the name on
    the config; a model built directly, without the loader, gets the kernel that
    runs on any device, since nothing has checked what this one can do.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = resolve(config._attn_implementation or UNIVERSAL_IMPLEMENTATION)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # The cache decides what the kernel reads: K and V as tensors, or the
        # pool plus the addresses to walk it (see `cache/continuous_kv_cache.py`).
        # `is not None` rather than truthiness: an empty Cache is falsy.
        if past_key_values is None:
            kv = DenseKV(k, v)
        else:
            kv = past_key_values.update(k, v, self.layer_idx)

        attn_output = self.attention(q, kv, attention_mask, self.scaling, self.num_kv_groups)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# ---------------------------------------------------------------------------
# Top-level models
# ---------------------------------------------------------------------------


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        past_key_values: Cache,
        attention_mask: torch.Tensor,
    ) -> CausalLMOutput:
        inputs_embeds = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        return CausalLMOutput(logits=hidden_states, past_key_values=past_key_values)


class LlamaForCausalLM(nn.Module):
    """Llama with a tied LM head — the entry point liteinfer dispatches to."""

    # Read by `liteinfer.models.loader` to resolve checkpoints that store
    # only the embedding copy of a tied weight.
    _tied_weights_keys: ClassVar[dict[str, str]] = {
        "lm_head.weight": "model.embed_tokens.weight"
    }

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        past_key_values: Cache,
        attention_mask: torch.Tensor,
    ) -> CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(outputs.logits)
        return CausalLMOutput(logits=logits, past_key_values=outputs.past_key_values)
