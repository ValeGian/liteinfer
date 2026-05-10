"""Paged KV cache — block-allocated tensors backed by a pre-allocated BlockPool.

Design overview
---------------
Tokens are stored in fixed-size blocks drawn from a shared ``BlockPool``.
Each sequence owns a *block table*: an ordered list of physical block indices
that hold its cached K/V tokens.  Because the pool is shared across batches,
blocks can be returned and re-used after each static batch completes.

Attention transparency
----------------------
``_PagedCachePayload.update()`` follows the same contract as
``_NativeCachePayload``:

* Prefill (``k.shape[2] > 1``): stores only the *real* (non-padded) prompt
  tokens in blocks, then returns the original padded K/V unchanged so the
  attention layer can compute the prefill step without any modification.

* Decode (``k.shape[2] == 1``): appends the single new token for each
  sequence into the next available block slot, then gathers all cached K/V
  from the block tables and left-pads them to
  ``[B, num_kv_heads, max_total_len, head_dim]`` — the same shape the
  existing attention-mask logic already expects.

No changes are required in the model layers or the attention-mask builder.

Block allocation strategy
-------------------------
Block tables are per-sequence and shared across layers: allocating block B
gives access to ``pool.keys[layer_idx, B, ...]`` for every layer_idx.  A
single allocation therefore services all transformer layers at a given token
position, exactly as in vLLM's PagedAttention.

Block allocation for a given decode step is triggered once, on the
``layer_idx == 0`` call of ``update()``.  Subsequent layer calls in the same
step write to the already-allocated slot without touching the block table.
This assumption — that layer 0 is always called first in each step — holds
for all sequential transformer architectures supported by liteinfer.
"""

from __future__ import annotations

import math

import torch

from liteinfer.cache.block_pool import BlockPool
from liteinfer.cache.kv_cache import KVCache
from liteinfer.config import EngineConfig


class _PagedCachePayload:
    """Per-batch KV store backed by a pre-allocated BlockPool.

    Implements the same ``update(k, v, layer_idx)`` interface as
    ``_NativeCachePayload``, making it transparent to attention layers.
    """

    def __init__(
            self,
            pool: BlockPool,
            num_seqs: int,
            prompt_lens: list[int],
    ) -> None:
        self._pool = pool
        self._num_seqs = num_seqs
        self._prompt_lens = prompt_lens
        self._max_prompt_len = max(prompt_lens) if prompt_lens else 0

        # block_tables[seq_idx]: physical block indices in order, shared across layers.
        self._block_tables: list[list[int]] = [[] for _ in range(num_seqs)]
        # token_counts[seq_idx]: total tokens currently cached for this sequence.
        self._token_counts: list[int] = [0] * num_seqs

    # ------------------------------------------------------------------
    # Public interface (mirrors _NativeCachePayload)
    # ------------------------------------------------------------------

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store new K/V in paged blocks and return the full accumulated K/V.

        During prefill (``key_states.shape[2] > 1``), only real (non-padded)
        tokens are stored; the original padded tensors are returned so the
        prefill attention computation is unchanged.

        During decode (``key_states.shape[2] == 1``), the single new token is
        appended for each sequence and the gathered K/V (left-padded to
        ``max_total_len``) is returned.
        """
        if key_states.shape[2] > 1:
            return self._update_prefill(key_states, value_states, layer_idx)
        return self._update_decode(key_states, value_states, layer_idx)

    def get_seq_length(self) -> int:
        """Return the maximum cached token count across all sequences."""
        return max(self._token_counts) if self._token_counts else 0

    # ------------------------------------------------------------------
    # Prefill path
    # ------------------------------------------------------------------

    def _update_prefill(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._allocate_prefill_blocks()

        for seq_idx in range(self._num_seqs):
            prompt_len = self._prompt_lens[seq_idx]
            real_start = self._max_prompt_len - prompt_len
            k_real = key_states[seq_idx, :, real_start:, :]  # [H, prompt_len, D]
            v_real = value_states[seq_idx, :, real_start:, :]
            self._write_tokens(layer_idx, seq_idx, k_real, v_real, start_pos=0)

        return key_states, value_states

    def _allocate_prefill_blocks(self) -> None:
        for seq_idx, prompt_len in enumerate(self._prompt_lens):
            n_blocks = math.ceil(prompt_len / self._pool.block_size)
            for _ in range(n_blocks):
                self._block_tables[seq_idx].append(self._pool.allocate())
            self._token_counts[seq_idx] = prompt_len

    # ------------------------------------------------------------------
    # Decode path
    # ------------------------------------------------------------------

    def _update_decode(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._advance_decode_slots()

        for seq_idx in range(self._num_seqs):
            write_pos = self._token_counts[seq_idx] - 1  # slot set by layer 0
            k_tok = key_states[seq_idx, :, :, :]  # [H, 1, D]
            v_tok = value_states[seq_idx, :, :, :]
            self._write_tokens(layer_idx, seq_idx, k_tok, v_tok, start_pos=write_pos)

        max_total = max(self._token_counts)
        return self._gather_all(layer_idx, max_total)

    def _advance_decode_slots(self) -> None:
        for seq_idx in range(self._num_seqs):
            current = self._token_counts[seq_idx]
            if current % self._pool.block_size == 0:
                self._block_tables[seq_idx].append(self._pool.allocate())
            self._token_counts[seq_idx] += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_tokens(
            self,
            layer_idx: int,
            seq_idx: int,
            k: torch.Tensor,
            v: torch.Tensor,
            start_pos: int,
    ) -> None:
        """Write k/v (shape [num_kv_heads, n_tokens, head_dim]) to blocks
        starting at logical token position start_pos."""
        n_tokens = k.shape[1]
        written = 0
        pos = start_pos
        while written < n_tokens:
            block_in_table = pos // self._pool.block_size
            slot_in_block = pos % self._pool.block_size
            n_in_block = min(n_tokens - written, self._pool.block_size - slot_in_block)
            block_idx = self._block_tables[seq_idx][block_in_table]
            self._pool.write_tokens(
                layer_idx,
                block_idx,
                slot_in_block,
                k[:, written: written + n_in_block, :],
                v[:, written: written + n_in_block, :],
            )
            written += n_in_block
            pos += n_in_block

    def _gather_all(
            self,
            layer_idx: int,
            max_total: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for all sequences from their block tables.

        Returns ``[B, num_kv_heads, max_total, head_dim]`` tensors,
        left-padded with zeros for shorter sequences. The attention mask
        (built with the original ``prompt_lens``) handles masking these zeros.
        """
        k_batch: list[torch.Tensor] = []
        v_batch: list[torch.Tensor] = []

        for seq_idx in range(self._num_seqs):
            total_tokens = self._token_counts[seq_idx]
            k_parts: list[torch.Tensor] = []
            v_parts: list[torch.Tensor] = []
            remaining = total_tokens

            for block_idx in self._block_tables[seq_idx]:
                n = min(remaining, self._pool.block_size)
                k_parts.append(self._pool.get_key_block(layer_idx, block_idx)[:, :n, :])
                v_parts.append(self._pool.get_value_block(layer_idx, block_idx)[:, :n, :])
                remaining -= n
                if remaining == 0:
                    break

            k_seq = torch.cat(k_parts, dim=1)  # [H, total_tokens, D]
            v_seq = torch.cat(v_parts, dim=1)

            pad_len = max_total - total_tokens
            if pad_len > 0:
                k_seq = torch.cat(
                    [k_seq.new_zeros(self._pool.num_kv_heads, pad_len, self._pool.head_dim), k_seq],
                    dim=1,
                )
                v_seq = torch.cat(
                    [v_seq.new_zeros(self._pool.num_kv_heads, pad_len, self._pool.head_dim), v_seq],
                    dim=1,
                )

            k_batch.append(k_seq)
            v_batch.append(v_seq)

        return torch.stack(k_batch), torch.stack(v_batch)


class PagedKVCache(KVCache):
    """Paged KV cache backed by a pre-allocated BlockPool.

    The pool is shared across batches and passed in at construction time.
    One PagedKVCache is created per static batch by ModelRunner.start_batch()
    and destroyed via reset() + ModelRunner.end_batch().

    Args:
        config: engine configuration (dtype, device, block_size, etc.).
        block_pool: shared pre-allocated pool; outlives this object.
        prompt_lens: real (non-padded) prompt length for each sequence in the batch.
    """

    def __init__(
            self,
            config: EngineConfig,
            block_pool: BlockPool,
            prompt_lens: list[int],
    ) -> None:
        super().__init__(config)
        self._block_pool = block_pool
        self._payload = _PagedCachePayload(block_pool, len(prompt_lens), prompt_lens)

    def reset(self) -> None:
        """Return all allocated blocks to the pool and clear per-batch state."""
        for block_list in self._payload._block_tables:
            for block_idx in block_list:
                self._block_pool.free(block_idx)
        self._payload._block_tables = [[] for _ in range(self._payload._num_seqs)]
        self._payload._token_counts = [0] * self._payload._num_seqs

    def get_seq_length(self) -> int:
        return self._payload.get_seq_length()

    @property
    def payload(self) -> _PagedCachePayload:
        return self._payload
