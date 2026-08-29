"""ContinuousModelRunner — forward-pass execution for continuous batching.

Exposes two explicit methods instead of the single ``execute()`` of
``ModelRunner``:

* ``prefill(seqs)`` — full prompt pass for newly admitted sequences.
* ``decode(seqs)`` — single-token pass for sequences already past prefill.

Two separate forward passes are issued when both new and running sequences
coexist in the same engine step. This keeps the implementation simple and
avoids the mixed-prefill/decode attention complexity that would otherwise
require a flash-attention-style kernel.

See roadmap §1.3 for the single-pass chunked-prefill upgrade path.
"""

from __future__ import annotations

import torch

from liteinfer.cache.block_pool import BlockPool
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache
from liteinfer.config import EngineConfig
from liteinfer.engine.attention_mask import (
    build_continuous_decode_for_model,
    build_prefill_for_model,
)
from liteinfer.engine.sequence import Sequence
from liteinfer.hub import resolve_model_path
from liteinfer.models.loader import load_hf_model
from liteinfer.tokenizer import Tokenizer


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

        for seq, pl in zip(seqs, prompt_lens, strict=False):
            self._cache.register(seq.request_id, pl)

        input_ids, position_ids = self._build_prefill_inputs(seqs, prompt_lens, max_prompt_len)
        attention_mask = build_prefill_for_model(
            type(self.model).__name__,
            prompt_lens=prompt_lens,
            dtype=self.config.dtype,
            device=self.device,
        )
        payload = self._cache.make_prefill_payload(request_ids, prompt_lens)
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=payload,
            attention_mask=attention_mask,
        )
        return self._select_prefill_logits(out.logits, prompt_lens, max_prompt_len)

    @torch.inference_mode()
    def decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """Decode pass for sequences already past their prefill step.

        Feeds one token per sequence (the last sampled token) and returns
        logits ``[B, vocab_size]`` for sampling the next token.
        """
        assert self._cache is not None and self.model is not None

        request_ids = [s.request_id for s in seqs]
        input_ids, position_ids = self._build_decode_inputs(seqs)

        # seq_total_len includes the current decode token (prompt + all output tokens).
        seq_total_lens = [len(s.prompt_token_ids) + len(s.output_token_ids) for s in seqs]
        attention_mask = build_continuous_decode_for_model(
            type(self.model).__name__,
            seq_total_lens=seq_total_lens,
            dtype=self.config.dtype,
            device=self.device,
        )
        payload = self._cache.make_decode_payload(request_ids)
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
        B = len(seqs)
        input_ids = torch.full((B, max_prompt_len), 0, dtype=torch.long, device=self.device)
        position_ids = torch.zeros((B, max_prompt_len), dtype=torch.long, device=self.device)
        for i, (seq, pl) in enumerate(zip(seqs, prompt_lens, strict=False)):
            offset = max_prompt_len - pl
            input_ids[i, offset:] = torch.tensor(
                seq.prompt_token_ids, dtype=torch.long, device=self.device
            )
            position_ids[i, offset:] = torch.arange(pl, device=self.device)
        return input_ids, position_ids

    def _build_decode_inputs(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        last_tokens = [s.output_token_ids[-1] for s in seqs]
        positions = [len(s.prompt_token_ids) + len(s.output_token_ids) - 1 for s in seqs]
        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        position_ids = torch.tensor(positions, dtype=torch.long, device=self.device).unsqueeze(1)
        return input_ids, position_ids

    def _select_prefill_logits(
        self,
        logits: torch.Tensor,
        prompt_lens: list[int],
        max_prompt_len: int,
    ) -> torch.Tensor:
        """Pick logits at each sequence's last real (non-padded) token."""
        if logits.shape[1] == 1:
            return logits[:, -1, :]
        last_indices = torch.tensor(
            [max_prompt_len - 1] * len(prompt_lens),
            dtype=torch.long,
            device=logits.device,
        )
        return logits[torch.arange(len(prompt_lens), device=logits.device), last_indices]

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
        if self.config.num_gpu_blocks is not None:
            return self.config.num_gpu_blocks
        dtype_bytes = torch.finfo(self.config.dtype).bits // 8
        bytes_per_block = (
            self.config.block_size * num_kv_heads * head_dim * dtype_bytes * 2 * num_layers
        )
        if self.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            usable_bytes = int(free_bytes * 0.85)
        else:
            usable_bytes = 1 << 30
        return max(1, usable_bytes // bytes_per_block)
