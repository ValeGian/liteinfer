"""ContinuousModelRunner — forward-pass execution for continuous batching.

* ``prefill(seqs)`` — full prompt pass for newly admitted sequences.
* ``decode(seqs)`` — single-token pass for sequences already past prefill.

A step where new and running sequences coexist issues both, rather than one
mixed pass, which would need a flash-attention-style kernel. See roadmap §1.3.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from liteinfer.cache.block_pool import BlockPool
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache, KVPayload
from liteinfer.config import EngineConfig
from liteinfer.engine.attention_mask import builders_for
from liteinfer.engine.sequence import Sequence
from liteinfer.hub import resolve_model_path
from liteinfer.models.attention import reads_paged_kv
from liteinfer.models.loader import load_hf_model
from liteinfer.tokenizer import Tokenizer

if TYPE_CHECKING:
    from transformers import PretrainedConfig

_LOGGER = logging.getLogger(__name__)
_GIB = 1 << 30


def _head_dim(hf_config: PretrainedConfig) -> int:
    """Head dimension, which most configs state and the rest imply."""
    return getattr(
        hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
    )


class ContinuousModelRunner:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = config.resolved_device()
        self.model: torch.nn.Module | None = None
        self.hf_config = None
        self.tokenizer: Tokenizer | None = None
        self._cache: ContinuousKVCache | None = None

    def load_model(self) -> None:
        model_path = resolve_model_path(self.config.model)
        self.model, self.hf_config = load_hf_model(self.config, model_path)
        self.tokenizer = Tokenizer(model_path)
        self._cache = ContinuousKVCache(self._create_block_pool())

    @property
    def attn_implementation(self) -> str:
        """The kernel this engine resolved to, which may not be the one requested.

        `EngineConfig.attn_implementation` can be `None` for "choose for me";
        `load_hf_model` makes the choice and records it, because that is where
        the device and the model's head dimension are both known. Reading it
        back from there keeps one answer rather than two.
        """
        resolved = getattr(self.hf_config, "_attn_implementation", None)
        assert isinstance(resolved, str), "load_model() records the resolved kernel"
        return resolved

    def deregister_sequence(self, seq: Sequence) -> None:
        """Free paged KV blocks allocated for a finished sequence."""
        assert self._cache is not None
        self._cache.deregister(seq.request_id)

    @torch.inference_mode()
    def prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        """Prefill pass for a batch of newly admitted sequences.

        Allocates KV cache blocks for each sequence and runs one full
        forward pass over their (left-padded) prompts. Returns logits
        ``[B, vocab_size]`` at each sequence's last real token position.
        """
        assert self._cache is not None and self.model is not None

        request_ids = [s.request_id for s in seqs]
        prompt_lens = [len(s.prompt_token_ids) for s in seqs]
        max_prompt_len = max(prompt_lens)

        for seq, prompt_len in zip(seqs, prompt_lens, strict=True):
            self._cache.register(seq.request_id, prompt_len)

        input_ids, position_ids = self._build_prefill_inputs(seqs, prompt_lens, max_prompt_len)
        build_prefill, _ = builders_for(type(self.model).__name__)
        attention_mask = build_prefill(prompt_lens, self.config.dtype, self.device)
        payload = self._cache.make_prefill_payload(request_ids, prompt_lens)
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=payload,
            attention_mask=attention_mask,
        )
        return out.logits[:, -1, :]  # left-padded, so the last column is the last real token

    @torch.inference_mode()
    def decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """Decode pass for sequences already past their prefill step.

        Feeds one token per sequence (the last sampled token) and returns
        logits ``[B, vocab_size]`` for sampling the next token.
        """
        assert self._cache is not None and self.model is not None

        request_ids = [s.request_id for s in seqs]
        input_ids, position_ids = self._build_decode_inputs(seqs)

        # Account for this step's token before addressing it, then build the slot
        # table and mask here rather than on the first layer: the forward pass
        # must contain no host-side work.
        self._cache.advance(request_ids)
        slots = self._cache.slot_table_for(request_ids)
        payload, attention_mask = self._build_decode_kv(request_ids, slots)
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=payload,
            attention_mask=attention_mask,
        )
        return out.logits[:, -1, :]

    # ------------------------------------------------------------------
    # Input builders
    # ------------------------------------------------------------------

    def _build_decode_kv(
        self, request_ids: list[str], slots: torch.Tensor
    ) -> tuple[KVPayload, torch.Tensor | None]:
        """Pair this step's KV payload with the mask its attention kernel needs.

        The paged kernel stops each sequence at its own context length, so there
        is no padding to hide and no mask to build. The dense kernels read a
        gather padded out to the longest sequence in the batch, so there is.
        """
        assert self._cache is not None
        if reads_paged_kv(self.attn_implementation):
            context_lens = self._cache.context_lens_for(request_ids)
            payload = self._cache.make_paged_decode_payload(
                slots, context_lens, self.config.paged_decode_splits
            )
            return payload, None

        # The cache's token counts already include this step's token, and they are
        # what addressed the slots above — so the mask is built from the same source.
        seq_total_lens = [self._cache.seq_total_len(rid) for rid in request_ids]
        _, build_decode = builders_for(type(self.model).__name__)
        return (
            self._cache.make_decode_payload(slots),
            build_decode(seq_total_lens, self.config.dtype, self.device),
        )

    def _build_prefill_inputs(
        self,
        seqs: list[Sequence],
        prompt_lens: list[int],
        max_prompt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (len(seqs), max_prompt_len)
        input_ids = torch.zeros(shape, dtype=torch.long, device=self.device)
        position_ids = torch.zeros(shape, dtype=torch.long, device=self.device)
        for i, (seq, prompt_len) in enumerate(zip(seqs, prompt_lens, strict=True)):
            offset = max_prompt_len - prompt_len
            input_ids[i, offset:] = torch.tensor(
                seq.prompt_token_ids, dtype=torch.long, device=self.device
            )
            position_ids[i, offset:] = torch.arange(prompt_len, device=self.device)
        return input_ids, position_ids

    def _build_decode_inputs(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        last_tokens = [s.output_token_ids[-1] for s in seqs]
        positions = [len(s.prompt_token_ids) + len(s.output_token_ids) - 1 for s in seqs]
        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        position_ids = torch.tensor(positions, dtype=torch.long, device=self.device).unsqueeze(1)
        return input_ids, position_ids

    # ------------------------------------------------------------------
    # Block pool
    # ------------------------------------------------------------------

    def _create_block_pool(self) -> BlockPool:
        num_layers: int = self.hf_config.num_hidden_layers
        num_kv_heads: int = self.hf_config.num_key_value_heads
        head_dim = _head_dim(self.hf_config)
        num_blocks = self._compute_num_blocks(num_layers, num_kv_heads, head_dim)
        return BlockPool(
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=self.config.dtype,
            device=self.device,
        )

    def _compute_num_blocks(self, num_layers: int, num_kv_heads: int, head_dim: int) -> int:
        """Size the pool to the smaller of what memory allows and what the engine can reach.

        `max_num_seqs` sequences of `max_model_len` tokens is the most KV that can
        ever exist, so blocks beyond that are memory the engine is structurally
        unable to use — and that surplus is what the forward pass needs for
        activations.
        """
        dtype_bytes = torch.finfo(self.config.dtype).bits // 8
        bytes_per_block = (
            self.config.block_size * num_kv_heads * head_dim * dtype_bytes * 2 * num_layers
        )
        if self.config.num_gpu_blocks is not None:
            self._log_pool(self.config.num_gpu_blocks, bytes_per_block, "set by num_gpu_blocks")
            return self.config.num_gpu_blocks

        budget = (
            torch.cuda.mem_get_info(self.device)[0] if self.device.type == "cuda" else 1 << 30
        )
        affordable = int(budget * self.config.kv_cache_memory_fraction) // bytes_per_block
        reachable = math.ceil(
            self.config.max_num_seqs * self.config.max_model_len / self.config.block_size
        )

        if affordable < reachable:
            _LOGGER.warning(
                "KV pool holds %d blocks but this config could need %d: "
                "%d concurrent sequences of %d tokens may exhaust it. "
                "Lower max_num_seqs or max_model_len, or raise kv_cache_memory_fraction.",
                affordable, reachable, self.config.max_num_seqs, self.config.max_model_len,
            )
            num_blocks = max(1, affordable)
            reason = "limited by free memory"
        else:
            num_blocks = max(1, reachable)
            reason = f"sized for {self.config.max_num_seqs} x {self.config.max_model_len} tokens"
        self._log_pool(num_blocks, bytes_per_block, reason)
        return num_blocks

    def _log_pool(self, num_blocks: int, bytes_per_block: int, reason: str) -> None:
        _LOGGER.info(
            "KV pool: %d blocks x %d tokens = %.2f GiB (%s)",
            num_blocks, self.config.block_size, num_blocks * bytes_per_block / _GIB, reason,
        )
