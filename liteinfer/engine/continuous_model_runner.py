"""ContinuousModelRunner — forward-pass execution for continuous batching.

* ``prefill(seqs)`` — full prompt pass for newly admitted sequences.
* ``decode(seqs)`` — single-token pass for sequences already past prefill.

A step where new and running sequences coexist issues both, rather than one
mixed pass, which would need a flash-attention-style kernel. See roadmap §1.3.
"""

from __future__ import annotations

import logging
import math

import torch

from liteinfer.cache.block_pool import BlockPool
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache
from liteinfer.config import EngineConfig
from liteinfer.engine.attention_mask import builders_for
from liteinfer.engine.sequence import Sequence
from liteinfer.hub import resolve_model_path
from liteinfer.models.loader import load_hf_model
from liteinfer.tokenizer import Tokenizer

_LOGGER = logging.getLogger(__name__)
_GIB = 1 << 30


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

        # seq_total_len includes the current decode token (prompt + all output tokens).
        seq_total_lens = [len(s.prompt_token_ids) + len(s.output_token_ids) for s in seqs]
        _, build_decode = builders_for(type(self.model).__name__)
        attention_mask = build_decode(seq_total_lens, self.config.dtype, self.device)
        payload = self._cache.make_decode_payload(slots)
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
        head_dim: int = getattr(
            self.hf_config,
            "head_dim",
            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
        )
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
