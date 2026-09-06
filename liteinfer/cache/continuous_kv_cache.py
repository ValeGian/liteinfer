"""Per-sequence paged KV cache for continuous batching.

Sequences are keyed by ``request_id`` and can register or deregister at any
time, which is what the scheduler's slot-filling policy requires.

Payload protocol
----------------
The payload factories return lightweight objects that implement the same
``update(k, v, layer_idx)`` interface understood by the model's attention
layers: store this pass's K/V, then hand back the K/V the attention kernel
should read, as a ``DenseKV`` or a ``PagedKV``. Payloads hold a reference to
this cache; they are ephemeral (created per forward pass) and must not outlive
the forward call.

Prefill payload
    Stores the real (non-padded) prompt K/V into pre-allocated blocks and
    returns the original (left-padded) tensors unchanged, so the prefill
    attention computation is unmodified.

Decode payload
    Appends one new token per sequence, then gathers and left-pads the full
    accumulated K/V to ``[B, num_kv_heads, max_total_len, head_dim]`` — the
    shape the dense kernels and the continuous-decode attention mask expect.

Paged decode payload
    Appends the same token and then returns nothing but addresses: the layer's
    flat pool storage plus the slot table and context lengths the fused paged
    kernel walks. The gather never happens, which is the whole point — see
    ``models/paged_decode.py``.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch

from liteinfer.cache.block_pool import BlockPool, slot_table
from liteinfer.models.attention import DenseKV, PagedKV


class KVPayload(Protocol):
    """What one forward pass is handed in place of a KV cache.

    The model calls ``update`` once per layer and passes the result to its
    attention kernel, so this one method is the whole contract between the cache
    and the model.
    """

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> DenseKV | PagedKV:
        ...


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

    def make_decode_payload(self, slots: torch.Tensor) -> _DecodePayload:
        """Return a payload for one decode forward pass reading ``slots``.

        The slot table is computed by the caller rather than on the first layer,
        so the forward pass contains no host-side work — which is what lets it
        be captured into a CUDA graph, and what keeps the GPU from stalling
        mid-pass otherwise.
        """
        return _DecodePayload(self, slots)

    def make_paged_decode_payload(
        self, slots: torch.Tensor, context_lens: torch.Tensor
    ) -> _PagedDecodePayload:
        """Return a payload that hands the pool's addresses to the paged kernel.

        Same slot table as ``make_decode_payload``; the difference is that the
        K/V never leave the pool, so the kernel also needs to know where each
        sequence's history ends.
        """
        return _PagedDecodePayload(self, slots, context_lens)

    def advance(self, request_ids: list[str]) -> None:
        """Account for the token about to be decoded, allocating a block if needed."""
        for request_id in request_ids:
            if self._token_counts[request_id] % self._pool.block_size == 0:
                self._block_tables[request_id].append(self._pool.allocate())
            self._token_counts[request_id] += 1

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

    def context_lens_for(self, request_ids: list[str]) -> torch.Tensor:
        """Cached-token count per sequence, as ``[B]`` on the pool's device."""
        return torch.tensor(
            [self._token_counts[rid] for rid in request_ids],
            dtype=torch.int32,
            device=self._pool.device,
        )

    def layer_storage(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """This layer's flat key/value stores, for a kernel that addresses them itself."""
        return self._pool.slots(layer_idx)

    def scatter(
        self, layer_idx: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> None:
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
    ) -> DenseKV:
        """Store real prompt tokens in blocks; return the original tensors unchanged."""
        if layer_idx == 0:
            for req_id, prompt_len in zip(self._request_ids, self._prompt_lens, strict=True):
                self._cache._token_counts[req_id] = prompt_len
            self._slots = self._cache.slot_table_for(self._request_ids)

        # Prompts arrive left-padded and the slot table is right-aligned, so the
        # two line up column for column; padding lands in the null block.
        self._cache.scatter(layer_idx, self._slots, key_states, value_states)
        return DenseKV(key_states, value_states)


class _DecodePayload:
    """Decode-pass payload reading a slot table the caller already built.

    Every operation here is a tensor op on fixed pool storage, which is what
    makes the pass capturable: nothing decides anything on the host.
    """

    def __init__(self, cache: ContinuousKVCache, slots: torch.Tensor) -> None:
        self._cache = cache
        self._slots = slots

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> DenseKV:
        """Store each sequence's decode token; return the gathered left-padded K/V."""
        self._cache.scatter(layer_idx, self._slots[:, -1:], key_states, value_states)
        return DenseKV(*self._cache.gather(layer_idx, self._slots))


class _PagedDecodePayload:
    """Decode-pass payload that stores the new token and then only points at the pool.

    The gathering payload above copies ``[B, H, max_total, D]`` of K and V out of
    the pool on every layer of every step. This one returns the pool itself plus
    the addresses, and the fused kernel walks them — so the copy that dominated
    decode does not exist on this path.
    """

    def __init__(
        self,
        cache: ContinuousKVCache,
        slots: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> None:
        self._cache = cache
        self._slots = slots
        self._context_lens = context_lens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> PagedKV:
        """Store each sequence's decode token; return where the whole history lives."""
        self._cache.scatter(layer_idx, self._slots[:, -1:], key_states, value_states)
        key_pool, value_pool = self._cache.layer_storage(layer_idx)
        return PagedKV(key_pool, value_pool, self._slots, self._context_lens)
