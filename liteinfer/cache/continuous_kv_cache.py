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

from liteinfer.cache.block_pool import BlockPool, slot_table


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

    def slot_table_for(self, request_ids: list[str]) -> torch.Tensor:
        """Physical slot of every cached token for these sequences."""
        return slot_table(
            [self._block_tables[rid] for rid in request_ids],
            [self._token_counts[rid] for rid in request_ids],
            self._pool.block_size,
            self._pool.device,
        )

    def scatter(self, layer_idx: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store one K/V column per slot: ``[B, H, T, D]`` -> ``[B, T, H, D]``."""
        keys, values = self._pool.slots(layer_idx)
        keys[slots] = k.permute(0, 2, 1, 3)
        values[slots] = v.permute(0, 2, 1, 3)

    def gather(self, layer_idx: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the cached K/V at ``slots`` as ``[B, H, T, D]``."""
        keys, values = self._pool.slots(layer_idx)
        return keys[slots].permute(0, 2, 1, 3), values[slots].permute(0, 2, 1, 3)


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
        self._slots = torch.empty(0, dtype=torch.long)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store real prompt tokens in blocks; return original tensors unchanged."""
        if layer_idx == 0:
            for req_id, prompt_len in zip(self._request_ids, self._prompt_lens, strict=True):
                self._cache._token_counts[req_id] = prompt_len
            self._slots = self._cache.slot_table_for(self._request_ids)

        # Prompts arrive left-padded and the slot table is right-aligned, so the
        # two line up column for column; padding lands in the null block.
        self._cache.scatter(layer_idx, self._slots, key_states, value_states)
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0


class _DecodePayload:
    """Decode-pass payload for the currently running sequences."""

    def __init__(self, cache: ContinuousKVCache, request_ids: list[str]) -> None:
        self._cache = cache
        self._request_ids = request_ids
        self._slots = torch.empty(0, dtype=torch.long)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append each sequence's decode token; return gathered left-padded K/V."""
        if layer_idx == 0:
            self._advance_slots()
            self._slots = self._cache.slot_table_for(self._request_ids)

        self._cache.scatter(layer_idx, self._slots[:, -1:], key_states, value_states)
        return self._cache.gather(layer_idx, self._slots)

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
