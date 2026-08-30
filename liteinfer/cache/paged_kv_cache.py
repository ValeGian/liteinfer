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
  existing attention-mask logic already expects. Both the write and the
  gather are single indexing ops over the pool's flat slot storage; the
  logical-to-physical slot table is rebuilt once per step, not per layer.

No changes are required in the model layers or the attention-mask builder.

Block allocation strategy
-------------------------
Block tables are per-sequence and shared across layers: one allocation
services every transformer layer at a given token position, exactly as in
vLLM's PagedAttention.

Block allocation for a given decode step is triggered once, on the
``layer_idx == 0`` call of ``update()``.  Subsequent layer calls in the same
step write to the already-allocated slot without touching the block table.
This assumption — that layer 0 is always called first in each step — holds
for all sequential transformer architectures supported by liteinfer.
"""

from __future__ import annotations

import math

import torch

from liteinfer.cache.block_pool import BlockPool, slot_table
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
        # Physical slot of every cached token; rebuilt whenever the counts change.
        self._slots = torch.empty(0, dtype=torch.long, device=pool.device)

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
            self._rebuild_slots()

        # Prompts arrive left-padded and the slot table is right-aligned, so the
        # two line up column for column; padding lands in the null block.
        self._scatter(layer_idx, key_states, value_states)
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
            self._rebuild_slots()

        keys, values = self._pool.slots(layer_idx)
        newest = self._slots[:, -1]  # right-aligned, so the new token is last
        keys[newest] = key_states[:, :, 0, :]
        values[newest] = value_states[:, :, 0, :]

        # [B, T, H, D] -> [B, H, T, D]
        return keys[self._slots].permute(0, 2, 1, 3), values[self._slots].permute(0, 2, 1, 3)

    def _advance_decode_slots(self) -> None:
        for seq_idx in range(self._num_seqs):
            current = self._token_counts[seq_idx]
            if current % self._pool.block_size == 0:
                self._block_tables[seq_idx].append(self._pool.allocate())
            self._token_counts[seq_idx] += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_slots(self) -> None:
        self._slots = slot_table(
            self._block_tables, self._token_counts, self._pool.block_size, self._pool.device
        )

    def _scatter(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store one K/V column per slot: ``[B, H, T, D]`` -> ``[B, T, H, D]``."""
        keys, values = self._pool.slots(layer_idx)
        keys[self._slots] = k.permute(0, 2, 1, 3)
        values[self._slots] = v.permute(0, 2, 1, 3)

    def reset(self) -> None:
        """Return every allocated block to the pool and clear per-batch state."""
        for block_list in self._block_tables:
            for block_idx in block_list:
                self._pool.free(block_idx)
        self._block_tables = [[] for _ in range(self._num_seqs)]
        self._token_counts = [0] * self._num_seqs


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
        self._payload.reset()

    def get_seq_length(self) -> int:
        return self._payload.get_seq_length()

    @property
    def payload(self) -> _PagedCachePayload:
        return self._payload
