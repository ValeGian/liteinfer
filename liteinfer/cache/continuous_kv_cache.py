# pyright: reportPrivateImportUsage=false
"""Per-sequence paged KV cache for continuous batching.

The cache owns each sequence's blocks and nothing else: how many tokens a
sequence holds is `Sequence.num_computed_tokens`, and every method that needs it
is handed it. A pass first `reserve`s room for what each sequence will hold once
it is written, then asks for a payload; every payload addresses each sequence's
*newest* tokens, so a whole prompt, a chunk continuing one, and a decode step are
the same request with different counts. A sequence's block table exists from its
first reservation to its `release`.

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
        self._max_query_len = max(lengths)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> PagedKV:
        return PagedKV(
            _tokens_first(key_states),
            _tokens_first(value_states),
            self._slots,
            self._context_lens,
            query_start_loc=self.query_start_loc,
            max_query_len=self._max_query_len,
            host_context_lens=self._lengths,
            host_query_lens=self._lengths,
        )


class DecodeProfilePayload:
    """A decode step being measured: `batch` rows, each reading a `width`-token context.

    A kernel that copies each row's context out of the pool copies ``batch x
    width`` tokens of K and V in a decode step, however small the token budget —
    so for those kernels the widest packed pass does not bound the widest step.
    The stand-in pool holds only `width` slots and every row reads all of them:
    the copy's size follows from how many slots each row reads, not which.
    Nothing is written, as with `ProfilePayload`.
    """

    def __init__(
        self,
        batch: int,
        width: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._key_pool = torch.zeros((width, num_kv_heads, head_dim), dtype=dtype, device=device)
        self._value_pool = torch.zeros_like(self._key_pool)
        self._slots = torch.arange(width, device=device).expand(batch, width)
        self._context_lens = torch.full((batch,), width, dtype=torch.int32, device=device)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> PagedKV:
        return PagedKV(self._key_pool, self._value_pool, self._slots, self._context_lens)


class ContinuousKVCache:
    """Per-sequence block-allocated KV cache for continuous batching."""

    def __init__(self, pool: BlockPool) -> None:
        self._pool = pool
        self._block_tables: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def release(self, request_id: str) -> None:
        """Free all blocks belonging to a finished sequence."""
        for block_idx in self._block_tables.pop(request_id, []):
            self._pool.free(block_idx)

    def reserve(self, request_ids: list[str], totals: list[int]) -> None:
        """Give each sequence the blocks to hold `totals[i]` tokens, before a pass writes them.

        Called before the payload is made, because the payload addresses each
        sequence's newest tokens and those must already have somewhere to live.
        Allocation is host-side bookkeeping, and it stays out of the forward.

        All or nothing: when the pool cannot hold every sequence's tokens, it
        raises `BlockPoolExhaustedError` before anything changes. A step carries
        running sequences and newly admitted ones in one pass, so a caller can
        then drop the newcomers and keep the rest rather than lose the whole step.
        """
        block_size = self._pool.block_size
        wanted = list(zip(request_ids, totals, strict=True))
        needed = sum(self._blocks_short_of(request_id, total) for request_id, total in wanted)
        if needed > self._pool.num_free_blocks:
            raise BlockPoolExhaustedError(
                f"KV block pool exhausted: this step needs {needed} more blocks and "
                f"{self._pool.num_free_blocks} of {self._pool.num_blocks} are free. "
                "Increase num_gpu_blocks or reduce max_num_seqs / max_model_len."
            )
        for request_id, total in wanted:
            table = self._block_tables.setdefault(request_id, [])
            while len(table) * block_size < total:
                table.append(self._pool.allocate())

    def _blocks_short_of(self, request_id: str, total: int) -> int:
        """Blocks a sequence must be given before it can hold `total` tokens."""
        held = len(self._block_tables.get(request_id, ()))
        return max(0, -(-total // self._pool.block_size) - held)

    # ------------------------------------------------------------------
    # Payload factory
    # ------------------------------------------------------------------

    def make_packed_payload(
        self, request_ids: list[str], starts: list[int], counts: list[int]
    ) -> _PackedPayload:
        """Return a payload for each sequence's `counts` tokens from `starts`, laid end to end.

        The payload carries the run's `PackedAddresses`, whose positions and
        boundaries the caller also needs for RoPE and for picking each
        sequence's last token, so they are built once per pass.
        """
        addresses = packed_addresses(
            self._tables(request_ids), starts, counts, self._pool.block_size, self._pool.device
        )
        totals = tuple(start + count for start, count in zip(starts, counts, strict=True))
        return _PackedPayload(self, addresses, totals, tuple(counts))

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

    def newest_slots_for(self, request_ids: list[str], positions: list[int]) -> torch.Tensor:
        """The slot of each sequence's token at `positions[i]`, ``[1, B]`` — where a decode step writes."""
        return newest_slots(
            self._tables(request_ids), positions, self._pool.block_size, self._pool.device
        )

    def slot_table_for(self, request_ids: list[str], totals: list[int]) -> torch.Tensor:
        """Each sequence's first `totals[i]` tokens, left-aligned. See `block_pool.slot_table`."""
        return slot_table(self._tables(request_ids), totals, self._pool.block_size, self._pool.device)

    def _tables(self, request_ids: list[str]) -> list[list[int]]:
        return [self._block_tables[request_id] for request_id in request_ids]

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
        self._max_query_len = max(host_query_lens)

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
            max_query_len=self._max_query_len,
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
