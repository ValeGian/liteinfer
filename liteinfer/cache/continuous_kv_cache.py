# pyright: reportPrivateImportUsage=false
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
layers: store this pass's K/V, then hand back a ``PagedKV`` — the layer's pool
plus the address of every token each sequence holds. Payloads hold a reference
to this cache; they are ephemeral (created per forward pass) and must not
outlive the forward call. Every address is computed when the payload is made
rather than on the first layer, so the forward contains no host-side work.

Packed payload
    A pass laid end to end: whole prompts, chunks continuing a cached prompt
    and sampled tokens side by side, bounded by `query_start_loc`. Each
    sequence's context is the chunk alone for a whole prompt, or reaches back to
    a prefix earlier passes wrote.

Decode payload
    One token per sequence, one per batch row, over addresses the caller built —
    which is what lets a captured graph refill them in place.
"""

from __future__ import annotations

from typing import Protocol

import torch

from liteinfer.cache.block_pool import (
    BlockPool,
    BlockPoolExhaustedError,
    PackedAddresses,
    newest_slots,
    packed_addresses,
    slot_table,
)
from liteinfer.models.attention import PagedKV


class KVPayload(Protocol):
    """What one forward pass is handed in place of a KV cache.

    The model calls ``update`` once per layer and passes the result to its
    attention kernel, so this one method is the whole contract between the cache
    and the model.
    """

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> PagedKV:
        ...


class ProfilePayload:
    """What a packed forward is handed when it is being measured rather than served.

    Each sequence is a whole prompt, so its context is exactly the K/V the pass
    computes. Handing those back as the "pool", addressed by where each sequence
    sits in the run, gives attention the same reads a served pass makes — through
    whichever kernel the engine runs — while needing no pool to write into. That
    is what lets the measurement happen before the pool is sized, which is the
    whole point: the pool gets what the forward turns out not to need.
    """

    def __init__(self, lengths: list[int], device: torch.device) -> None:
        self._lengths = tuple(lengths)
        lens = torch.tensor(lengths, dtype=torch.long, device=device)
        run_ends = torch.cumsum(lens, 0)
        self.query_start_loc = torch.nn.functional.pad(run_ends, (1, 0)).to(torch.int32)
        self.positions = torch.cat([torch.arange(length, device=device) for length in lengths]).unsqueeze(0)
        # Columns past a sequence's length address the next one's tokens, and no
        # answer depends on them: every read is bounded by the context length.
        # Clamped so the last sequence's stay inside the pass.
        slots = (run_ends - lens).unsqueeze(1) + torch.arange(max(lengths), device=device)
        self._slots = slots.clamp_(max=sum(lengths) - 1)
        self._context_lens = lens.to(torch.int32)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> PagedKV:
        return PagedKV(
            _tokens_first(key_states),
            _tokens_first(value_states),
            self._slots,
            self._context_lens,
            query_start_loc=self.query_start_loc,
            max_query_len=max(self._lengths),
            host_context_lens=self._lengths,
            host_query_lens=self._lengths,
        )


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

        All or nothing: when the pool cannot hold every sequence's tokens, it
        raises `BlockPoolExhaustedError` before anything changes. A step carries
        running sequences and newly admitted ones in one pass, so a caller can
        then drop the newcomers and keep the rest rather than lose the whole step,
        and a cache left half-advanced would account for tokens no pass wrote.
        """
        block_size = self._pool.block_size
        windows = list(zip(request_ids, counts, strict=True))
        needed = sum(self._blocks_short_of(request_id, count) for request_id, count in windows)
        if needed > self._pool.num_free_blocks:
            raise BlockPoolExhaustedError(
                f"KV block pool exhausted: this step needs {needed} more blocks and "
                f"{self._pool.num_free_blocks} of {self._pool.num_blocks} are free. "
                "Increase num_gpu_blocks or reduce max_num_seqs / max_model_len."
            )
        for request_id, count in windows:
            total = self._token_counts[request_id] + count
            table = self._block_tables[request_id]
            while len(table) * block_size < total:
                table.append(self._pool.allocate())
            self._token_counts[request_id] = total

    def _blocks_short_of(self, request_id: str, count: int) -> int:
        """Blocks a sequence must be given before it can hold `count` more tokens."""
        total = self._token_counts[request_id] + count
        held = len(self._block_tables[request_id])
        return max(0, -(-total // self._pool.block_size) - held)

    # ------------------------------------------------------------------
    # Payload factory
    # ------------------------------------------------------------------

    def make_packed_payload(self, request_ids: list[str], counts: list[int]) -> _PackedPayload:
        """Return a payload for each sequence's newest `counts` tokens, laid end to end.

        The payload carries the run's `PackedAddresses`, whose positions and
        boundaries the caller also needs for RoPE and for picking each
        sequence's last token, so they are built once per pass.
        """
        totals = [self._token_counts[r] for r in request_ids]
        addresses = packed_addresses(
            [self._block_tables[r] for r in request_ids],
            [total - count for total, count in zip(totals, counts, strict=True)],
            counts,
            self._pool.block_size,
            self._pool.device,
        )
        return _PackedPayload(self, addresses, tuple(totals), tuple(counts))

    def make_decode_payload(
        self,
        write_slots: torch.Tensor,
        slots: torch.Tensor,
        context_lens: torch.Tensor,
        num_splits: int | None = None,
    ) -> _DecodePayload:
        """Return a payload for one decode forward pass writing `write_slots`, reading `slots`.

        The addresses are computed by the caller rather than on the first layer,
        so the forward pass contains no host-side work — which is what lets it be
        captured into a CUDA graph, and what keeps the GPU from stalling
        mid-pass otherwise.
        """
        return _DecodePayload(self, write_slots, slots, context_lens, num_splits)

    # ------------------------------------------------------------------
    # Addresses a decode step builds
    # ------------------------------------------------------------------

    def newest_slots_for(self, request_ids: list[str]) -> torch.Tensor:
        """Each sequence's newest token's slot, ``[1, B]`` — where a decode step writes."""
        return newest_slots(
            [self._block_tables[r] for r in request_ids],
            [self._token_counts[r] - 1 for r in request_ids],
            self._pool.block_size,
            self._pool.device,
        )

    def slot_table_for(self, request_ids: list[str]) -> torch.Tensor:
        """Every cached token of each sequence, left-aligned. See `block_pool.slot_table`."""
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

    # ------------------------------------------------------------------
    # Pool access shared by payloads
    # ------------------------------------------------------------------

    def layer_storage(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """This layer's flat key/value stores, for a kernel that addresses them itself."""
        return self._pool.slots(layer_idx)

    def scatter(self, layer_idx: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store one K/V column per slot, in the order the pass holds its tokens.

        `slots` has one entry per token of ``[B, H, T, D]`` taken batch-major:
        ``[1, T]`` for a packed pass, and ``[1, B]`` for a decode step, whose one
        token per sequence is a packed run of B. Flattening both sides is what
        lets one write serve both layouts.
        """
        keys, values = self._pool.slots(layer_idx)
        flat_slots = slots.reshape(-1)
        keys[flat_slots] = _tokens_first(k)
        values[flat_slots] = _tokens_first(v)


def _tokens_first(states: torch.Tensor) -> torch.Tensor:
    """``[B, H, T, D]`` -> ``[B * T, H, D]``, batch-major, which is the order slots are in."""
    batch, heads, tokens, head_dim = states.shape
    return states.permute(0, 2, 1, 3).reshape(batch * tokens, heads, head_dim)


class _PackedPayload:
    """Payload for a pass packed end to end: whole prompts, chunks and sampled tokens.

    Each sequence's queries are the newest tokens of its context, and that
    context may reach back past this pass to a prefix earlier ones wrote. So once
    the pass is stored the payload hands attention the pool and the addresses of
    everything each sequence holds, as it does for a decode step — here with any
    number of queries per sequence, bounded by `query_start_loc`. A whole prompt
    is the case where the context is the chunk itself.
    """

    def __init__(
        self,
        cache: ContinuousKVCache,
        addresses: PackedAddresses,
        host_context_lens: tuple[int, ...],
        host_query_lens: tuple[int, ...],
    ) -> None:
        self._cache = cache
        self.addresses = addresses
        self._host_context_lens = host_context_lens
        self._host_query_lens = host_query_lens

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> PagedKV:
        """Store this pass's tokens; return where every token each sequence holds lives."""
        self._cache.scatter(layer_idx, self.addresses.slots, key_states, value_states)
        key_pool, value_pool = self._cache.layer_storage(layer_idx)
        return PagedKV(
            key_pool,
            value_pool,
            self.addresses.slot_table,
            self.addresses.context_lens,
            query_start_loc=self.addresses.query_start_loc,
            max_query_len=max(self._host_query_lens),
            host_context_lens=self._host_context_lens,
            host_query_lens=self._host_query_lens,
        )


class _DecodePayload:
    """Decode-pass payload that stores the new token and then only points at the pool.

    Every operation here is a tensor op on fixed pool storage, which is what
    makes the pass capturable: nothing decides anything on the host.
    """

    def __init__(
        self,
        cache: ContinuousKVCache,
        write_slots: torch.Tensor,
        slots: torch.Tensor,
        context_lens: torch.Tensor,
        num_splits: int | None,
    ) -> None:
        self._cache = cache
        self._write_slots = write_slots
        self._slots = slots
        self._context_lens = context_lens
        self._num_splits = num_splits

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> PagedKV:
        """Store each sequence's decode token; return where the whole history lives."""
        self._cache.scatter(layer_idx, self._write_slots, key_states, value_states)
        key_pool, value_pool = self._cache.layer_storage(layer_idx)
        return PagedKV(key_pool, value_pool, self._slots, self._context_lens, self._num_splits)
