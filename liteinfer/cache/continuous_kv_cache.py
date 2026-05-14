"""Per-sequence paged KV cache for continuous batching.

Unlike ``PagedKVCache`` (which is batch-level and fixed for the lifetime of
a static batch), ``ContinuousKVCache`` manages individual sequences identified
by ``request_id``. Sequences can register and deregister at any time, which is
required by the continuous scheduler's slot-filling policy.

Payload protocol
----------------
``make_prefill_payload`` and ``make_decode_payload`` return lightweight objects
that implement the same ``update(k, v, layer_idx)`` interface understood by the
model's attention layers. Payloads hold a reference to this cache; they are
ephemeral (created per forward pass) and must not outlive the forward call.

Prefill payload
    Stores the real (non-padded) prompt K/V into pre-allocated blocks and
    returns the original (left-padded) tensors unchanged so the prefill
    attention computation is unmodified.

Decode payload
    Appends one new token per sequence, then gathers and left-pads the full
    accumulated K/V to ``[B, num_kv_heads, max_total_len, head_dim]`` — the
    shape expected by the continuous-decode attention mask.
"""

from __future__ import annotations

import math

import torch

from liteinfer.cache.block_pool import BlockPool


class ContinuousKVCache:
    """Per-sequence block-allocated KV cache for continuous batching."""

    def __init__(self, pool: BlockPool) -> None:
        self._pool = pool
        self._block_tables: dict[str, list[int]] = {}
        self._token_counts: dict[str, int] = {}
        self._prompt_lens: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def register(self, request_id: str, prompt_len: int) -> None:
        """Pre-allocate blocks for a new sequence's prompt before prefill."""
        n_blocks = math.ceil(prompt_len / self._pool.block_size)
        self._block_tables[request_id] = [self._pool.allocate() for _ in range(n_blocks)]
        self._token_counts[request_id] = 0
        self._prompt_lens[request_id] = prompt_len

    def deregister(self, request_id: str) -> None:
        """Free all blocks belonging to a finished sequence."""
        for block_idx in self._block_tables.pop(request_id, []):
            self._pool.free(block_idx)
        self._token_counts.pop(request_id, None)
        self._prompt_lens.pop(request_id, None)

    def seq_total_len(self, request_id: str) -> int:
        """Cached token count for a registered sequence."""
        return self._token_counts[request_id]

    # ------------------------------------------------------------------
    # Payload factory
    # ------------------------------------------------------------------

    def make_prefill_payload(
        self,
        request_ids: list[str],
        prompt_lens: list[int],
    ) -> _PrefillPayload:
        """Return a payload for one prefill forward pass over ``request_ids``."""
        return _PrefillPayload(self, request_ids, prompt_lens)

    def make_decode_payload(self, request_ids: list[str]) -> _DecodePayload:
        """Return a payload for one decode forward pass over ``request_ids``."""
        return _DecodePayload(self, request_ids)

    # ------------------------------------------------------------------
    # Internal helpers shared by payloads
    # ------------------------------------------------------------------

    def _write_tokens(
        self,
        layer_idx: int,
        request_id: str,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ) -> None:
        """Write k/v ``[num_kv_heads, n_tokens, head_dim]`` starting at ``start_pos``."""
        n_tokens = k.shape[1]
        written = 0
        pos = start_pos
        while written < n_tokens:
            block_in_table = pos // self._pool.block_size
            slot_in_block = pos % self._pool.block_size
            n_in_block = min(n_tokens - written, self._pool.block_size - slot_in_block)
            block_idx = self._block_tables[request_id][block_in_table]
            self._pool.write_tokens(
                layer_idx,
                block_idx,
                slot_in_block,
                k[:, written : written + n_in_block, :],
                v[:, written : written + n_in_block, :],
            )
            written += n_in_block
            pos += n_in_block

    def _read_seq_kv(
        self,
        layer_idx: int,
        request_id: str,
        total_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather all cached K/V for one sequence from its block table.

        Returns ``([num_kv_heads, total_tokens, head_dim], same)`` pair.
        """
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        remaining = total_tokens
        for block_idx in self._block_tables[request_id]:
            n = min(remaining, self._pool.block_size)
            k_parts.append(self._pool.get_key_block(layer_idx, block_idx)[:, :n, :])
            v_parts.append(self._pool.get_value_block(layer_idx, block_idx)[:, :n, :])
            remaining -= n
            if remaining == 0:
                break
        return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)


class _PrefillPayload:
    """Prefill-pass payload for a batch of newly admitted sequences."""

    def __init__(
        self,
        cache: ContinuousKVCache,
        request_ids: list[str],
        prompt_lens: list[int],
    ) -> None:
        self._cache = cache
        self._request_ids = request_ids
        self._prompt_lens = prompt_lens
        self._max_prompt_len = max(prompt_lens) if prompt_lens else 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store real prompt tokens in blocks; return original tensors unchanged."""
        if layer_idx == 0:
            for req_id, pl in zip(self._request_ids, self._prompt_lens, strict=False):
                self._cache._token_counts[req_id] = pl

        for i, (req_id, pl) in enumerate(zip(self._request_ids, self._prompt_lens, strict=False)):
            real_start = self._max_prompt_len - pl
            k_real = key_states[i, :, real_start:, :]
            v_real = value_states[i, :, real_start:, :]
            self._cache._write_tokens(layer_idx, req_id, k_real, v_real, start_pos=0)

        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0


class _DecodePayload:
    """Decode-pass payload for the currently running sequences."""

    def __init__(self, cache: ContinuousKVCache, request_ids: list[str]) -> None:
        self._cache = cache
        self._request_ids = request_ids

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append decode token for each sequence; return gathered left-padded K/V."""
        if layer_idx == 0:
            self._advance_slots()

        for i, req_id in enumerate(self._request_ids):
            write_pos = self._cache._token_counts[req_id] - 1
            self._cache._write_tokens(
                layer_idx,
                req_id,
                key_states[i, :, :, :],
                value_states[i, :, :, :],
                start_pos=write_pos,
            )

        return self._gather_all(layer_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self._request_ids:
            return 0
        return max(self._cache._token_counts[rid] for rid in self._request_ids)

    def _advance_slots(self) -> None:
        for req_id in self._request_ids:
            current = self._cache._token_counts[req_id]
            if current % self._cache._pool.block_size == 0:
                self._cache._block_tables[req_id].append(self._cache._pool.allocate())
            self._cache._token_counts[req_id] += 1

    def _gather_all(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_counts = [self._cache._token_counts[rid] for rid in self._request_ids]
        max_total = max(token_counts)
        pool = self._cache._pool
        k_batch: list[torch.Tensor] = []
        v_batch: list[torch.Tensor] = []
        for req_id, total in zip(self._request_ids, token_counts, strict=False):
            k_seq, v_seq = self._cache._read_seq_kv(layer_idx, req_id, total)
            pad_len = max_total - total
            if pad_len > 0:
                zeros = k_seq.new_zeros(pool.num_kv_heads, pad_len, pool.head_dim)
                k_seq = torch.cat([zeros, k_seq], dim=1)
                v_seq = torch.cat([zeros, v_seq], dim=1)
            k_batch.append(k_seq)
            v_batch.append(v_seq)
        return torch.stack(k_batch), torch.stack(v_batch)
