"""Per-sequence paged KV cache for continuous batching.

Sequences are keyed by ``request_id`` and can register or deregister at any
time, which is what the scheduler's slot-filling policy requires. A pass first
`advance`s the sequences it computes by the tokens it will write, then asks for
a payload; every payload addresses each sequence's *newest* tokens, so a whole
prompt, a chunk continuing one, and a decode step are the same request with
different counts.

Payload protocol
----------------
The payload factories return lightweight objects that implement the same
``update(k, v, layer_idx)`` interface understood by the model's attention
layers: store this pass's K/V, then hand back the K/V the attention kernel
should read, as a ``DenseKV``, a ``VarlenKV`` or a ``PagedKV``. Payloads hold a
reference to this cache; they are ephemeral (created per forward pass) and must
not outlive the forward call. Every address is computed when the payload is
made rather than on the first layer, so the forward contains no host-side work.

Prefill payloads
    Store the tokens this pass computed. When no sequence had anything cached
    before the pass, those K/V are the whole context and are returned as they
    are. When one did — a prompt chunked across steps — attention must also see
    the prefix a previous pass wrote. The packed payload then hands the paged
    kernel the pool and every token's address, as decode does; the padded one,
    serving the dense kernels, reads the context back out of the pool
    left-padded.

Decode payload
    Appends one new token per sequence, then gathers and left-pads the full
    accumulated K/V to ``[B, num_kv_heads, max_total_len, head_dim]`` — the
    shape the dense kernels and the continuous-decode attention mask expect.

Paged decode payload
    Appends the same token and then returns nothing but addresses: the layer's
    flat pool storage plus the slot table and context lengths the fused paged
    kernel walks. The gather never happens, which is the whole point — see
    ``models/paged_decode.py``.

Both decode payloads write through a flat mapping — one slot per sequence,
`slot_mapping` over a batch where every count is 1 — and read through the
right-aligned slot table, which only the reads still need.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import torch

from liteinfer.cache.block_pool import BlockPool, slot_mapping, slot_table
from liteinfer.models.attention import DenseKV, PagedKV, VarlenKV


class KVPayload(Protocol):
    """What one forward pass is handed in place of a KV cache.

    The model calls ``update`` once per layer and passes the result to its
    attention kernel, so this one method is the whole contract between the cache
    and the model.
    """

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> DenseKV | VarlenKV | PagedKV:
        ...


class ProfilePayload:
    """What a forward is handed when it is being measured rather than served.

    Returns this pass's K/V untouched, which is exactly what the prefill payload
    returns *after* storing them — so the forward has the same shapes and the same
    activation peak while needing no pool to write into. That is what lets the
    measurement happen before the pool is sized, which is the whole point: the
    pool gets what the forward turns out not to need.

    Correct only for prefill, where attention reads the K/V the pass just
    computed. A decode pass reads history it did not compute, so measuring one
    means giving it a real cache.
    """

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> DenseKV:
        return DenseKV(key_states, value_states)


class ContinuousKVCache:
    """Per-sequence block-allocated KV cache for continuous batching."""

    def __init__(self, pool: BlockPool) -> None:
        self._pool = pool
        self._block_tables: dict[str, list[int]] = {}
        self._token_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def register(self, request_id: str) -> None:
        """Start tracking a sequence that holds nothing yet; `advance` allocates for it."""
        self._block_tables[request_id] = []
        self._token_counts[request_id] = 0

    def deregister(self, request_id: str) -> None:
        """Free all blocks belonging to a finished sequence."""
        for block_idx in self._block_tables.pop(request_id, []):
            self._pool.free(block_idx)
        self._token_counts.pop(request_id, None)

    def is_registered(self, request_id: str) -> bool:
        return request_id in self._token_counts

    def seq_total_len(self, request_id: str) -> int:
        """Cached token count for a registered sequence."""
        return self._token_counts[request_id]

    def advance(self, request_ids: list[str], counts: list[int]) -> None:
        """Account for the tokens a pass is about to write, allocating blocks to hold them.

        Called before the payload is made, because the payload addresses each
        sequence's newest tokens and those must already have somewhere to live.
        Allocation is host-side bookkeeping, and it stays out of the forward.
        """
        block_size = self._pool.block_size
        for request_id, count in zip(request_ids, counts, strict=True):
            total = self._token_counts[request_id] + count
            table = self._block_tables[request_id]
            while len(table) * block_size < total:
                table.append(self._pool.allocate())
            self._token_counts[request_id] = total

    # ------------------------------------------------------------------
    # Payload factory
    # ------------------------------------------------------------------

    def make_prefill_payload(self, request_ids: list[str], counts: list[int]) -> _PrefillPayload:
        """Return a payload for a left-padded pass over each sequence's newest `counts` tokens."""
        write_slots = self.slot_table_for(request_ids, counts)
        context_slots = (
            self.slot_table_for(request_ids) if self._has_history(request_ids, counts) else None
        )
        return _PrefillPayload(self, write_slots, context_slots)

    def make_packed_prefill_payload(
        self, request_ids: list[str], counts: list[int]
    ) -> _PackedPrefillPayload | _PackedChunkPayload:
        """Return a payload for the same tokens laid end to end rather than left-padded.

        Same writes as `make_prefill_payload` — every token lands in the slot its
        block table names — addressed by a flat mapping instead of a padded,
        right-aligned table. What changes for the kernel is what comes back:
        boundaries to respect rather than padding to mask. With nothing cached
        before the pass those are the K/V it computed; with a prefix cached, the
        pool itself and the addresses of every token each sequence holds.
        """
        write_slots = self.slot_mapping_for(request_ids, counts)
        queries = _Boundaries.of(counts, self._pool.device)
        if not self._has_history(request_ids, counts):
            return _PackedPrefillPayload(self, write_slots, queries)
        return _PackedChunkPayload(
            self,
            write_slots,
            queries,
            self.slot_table_for(request_ids),
            self.context_lens_for(request_ids),
        )

    def make_decode_payload(self, write_slots: torch.Tensor, slots: torch.Tensor) -> _DecodePayload:
        """Return a payload for one decode forward pass writing `write_slots`, reading `slots`.

        Both are computed by the caller rather than on the first layer, so the
        forward pass contains no host-side work — which is what lets it be
        captured into a CUDA graph, and what keeps the GPU from stalling
        mid-pass otherwise.
        """
        return _DecodePayload(self, write_slots, slots)

    def make_paged_decode_payload(
        self,
        write_slots: torch.Tensor,
        slots: torch.Tensor,
        context_lens: torch.Tensor,
        num_splits: int | None = None,
    ) -> _PagedDecodePayload:
        """Return a payload that hands the pool's addresses to the paged kernel.

        Same addresses as ``make_decode_payload``; the difference is that the
        K/V never leave the pool, so the kernel also needs to know where each
        sequence's history ends — and how many programs to cut that history
        across, where the caller has an opinion.
        """
        return _PagedDecodePayload(self, write_slots, slots, context_lens, num_splits)

    # ------------------------------------------------------------------
    # Internal helpers shared by payloads
    # ------------------------------------------------------------------

    def slot_mapping_for(self, request_ids: list[str], counts: list[int]) -> torch.Tensor:
        """Each sequence's newest `counts` tokens, end to end. See `block_pool.slot_mapping`.

        A decode step's write is the case where every count is 1.
        """
        return slot_mapping(
            [self._block_tables[r] for r in request_ids],
            counts,
            self._pool.block_size,
            self._pool.device,
            starts=self._window_starts(request_ids, counts),
        )

    def slot_table_for(
        self, request_ids: list[str], counts: list[int] | None = None
    ) -> torch.Tensor:
        """Each sequence's newest `counts` tokens, right-aligned; every cached token by default."""
        tables = [self._block_tables[rid] for rid in request_ids]
        block_size, device = self._pool.block_size, self._pool.device
        if counts is None:
            totals = [self._token_counts[rid] for rid in request_ids]
            return slot_table(tables, totals, block_size, device)
        starts = self._window_starts(request_ids, counts)
        return slot_table(tables, counts, block_size, device, starts=starts)

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
        """Store one K/V column per slot, in the order the pass holds its tokens.

        `slots` has one entry per token of ``[B, H, T, D]`` taken batch-major:
        ``[B, T]`` for a padded pass, ``[1, T]`` for a packed one, and ``[1, B]``
        for a decode step, whose one token per sequence is a packed run of B.
        Flattening both sides is what lets one write serve all three layouts.
        """
        keys, values = self._pool.slots(layer_idx)
        flat_slots = slots.reshape(-1)
        keys[flat_slots] = _tokens_first(k)
        values[flat_slots] = _tokens_first(v)

    def gather(self, layer_idx: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the cached K/V at ``slots`` as ``[B, H, T, D]``."""
        keys, values = self._pool.slots(layer_idx)
        return keys[slots].permute(0, 2, 1, 3), values[slots].permute(0, 2, 1, 3)

    def _window_starts(self, request_ids: list[str], counts: list[int]) -> list[int]:
        """Where each sequence's newest `counts` tokens begin."""
        return [
            self._token_counts[rid] - count for rid, count in zip(request_ids, counts, strict=True)
        ]

    def _has_history(self, request_ids: list[str], counts: list[int]) -> bool:
        """Whether any sequence held tokens before this pass's `counts` were added."""
        return any(
            self._token_counts[rid] > count for rid, count in zip(request_ids, counts, strict=True)
        )


def _tokens_first(states: torch.Tensor) -> torch.Tensor:
    """``[B, H, T, D]`` -> ``[B * T, H, D]``, batch-major, which is the order slots are in."""
    batch, heads, tokens, head_dim = states.shape
    return states.permute(0, 2, 1, 3).reshape(batch * tokens, heads, head_dim)


class _PrefillPayload:
    """Prefill-pass payload for a left-padded batch.

    Prompts arrive left-padded and the write table is right-aligned, so the two
    line up column for column and padding lands in the null block.
    """

    def __init__(
        self,
        cache: ContinuousKVCache,
        write_slots: torch.Tensor,
        context_slots: torch.Tensor | None,
    ) -> None:
        self._cache = cache
        self._write_slots = write_slots
        self._context_slots = context_slots

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> DenseKV:
        """Store this pass's tokens; return every token each sequence now holds."""
        self._cache.scatter(layer_idx, self._write_slots, key_states, value_states)
        if self._context_slots is None:
            return DenseKV(key_states, value_states)
        return DenseKV(*self._cache.gather(layer_idx, self._context_slots))


class _Boundaries(NamedTuple):
    """Where each sequence starts in a packed run, in the two forms the flash kernel reads."""

    cu_seqlens: torch.Tensor
    max_seqlen: int

    @classmethod
    def of(cls, lengths: list[int], device: torch.device) -> _Boundaries:
        return cls(_cumulative_lengths(lengths, device), max(lengths))


class _PackedPrefillPayload:
    """Prefill payload for whole prompts packed end to end, with no padding anywhere.

    The padded sibling above leans on two alignments cancelling: prompts arrive
    left-padded, the slot table is right-aligned, so the two line up column for
    column. Here there is nothing to cancel — token `i` of the flat run belongs
    to whichever sequence `cu_seqlens` says, and writes go to the slot the
    mapping names.
    """

    def __init__(
        self, cache: ContinuousKVCache, write_slots: torch.Tensor, boundaries: _Boundaries
    ) -> None:
        self._cache = cache
        self._write_slots = write_slots
        self._boundaries = boundaries

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> VarlenKV:
        """Store every prompt token; return the K/V this pass computed, plus the boundaries."""
        self._cache.scatter(layer_idx, self._write_slots, key_states, value_states)
        return VarlenKV(
            key_states, value_states, self._boundaries.cu_seqlens, self._boundaries.max_seqlen
        )


class _PackedChunkPayload:
    """Prefill payload for a packed batch where some prompt continues one already cached.

    The chunk's keys are not all in this pass: the prefix is in the pool, written
    by earlier ones. So once the chunk is stored the payload hands the paged
    kernel the pool and the addresses of everything each sequence holds, as it
    does for a decode step — here with several queries per sequence, at the end
    of its context, bounded by `query_start_loc`. Nothing is copied out of the pool.
    """

    def __init__(
        self,
        cache: ContinuousKVCache,
        write_slots: torch.Tensor,
        queries: _Boundaries,
        slots: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> None:
        self._cache = cache
        self._write_slots = write_slots
        self._queries = queries
        self._slots = slots
        self._context_lens = context_lens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> PagedKV:
        """Store this pass's tokens; return where every token each sequence holds lives."""
        self._cache.scatter(layer_idx, self._write_slots, key_states, value_states)
        key_pool, value_pool = self._cache.layer_storage(layer_idx)
        return PagedKV(
            key_pool,
            value_pool,
            self._slots,
            self._context_lens,
            query_start_loc=self._queries.cu_seqlens,
            max_query_len=self._queries.max_seqlen,
        )


def _cumulative_lengths(lengths: list[int], device: torch.device) -> torch.Tensor:
    """`[0, l0, l0 + l1, ...]` as int32, which is the layout the flash kernel reads."""
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return torch.tensor(boundaries, dtype=torch.int32, device=device)


class _DecodePayload:
    """Decode-pass payload over addresses the caller already built.

    Every operation here is a tensor op on fixed pool storage, which is what
    makes the pass capturable: nothing decides anything on the host.
    """

    def __init__(
        self, cache: ContinuousKVCache, write_slots: torch.Tensor, slots: torch.Tensor
    ) -> None:
        self._cache = cache
        self._write_slots = write_slots
        self._slots = slots

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> DenseKV:
        """Store each sequence's decode token; return the gathered left-padded K/V."""
        self._cache.scatter(layer_idx, self._write_slots, key_states, value_states)
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
        write_slots: torch.Tensor,
        slots: torch.Tensor,
        context_lens: torch.Tensor,
        num_splits: int | None = None,
    ) -> None:
        self._cache = cache
        self._write_slots = write_slots
        self._slots = slots
        self._context_lens = context_lens
        self._num_splits = num_splits

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> PagedKV:
        """Store each sequence's decode token; return where the whole history lives."""
        self._cache.scatter(layer_idx, self._write_slots, key_states, value_states)
        key_pool, value_pool = self._cache.layer_storage(layer_idx)
        return PagedKV(key_pool, value_pool, self._slots, self._context_lens, self._num_splits)
