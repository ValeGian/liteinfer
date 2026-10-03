# pyright: reportPrivateImportUsage=false
"""Pre-allocated physical KV storage divided into fixed-size blocks.

One block index services all transformer layers: allocating block B gives
access to K/V memory at pool.keys[layer_idx, B, ...] for every layer_idx.
This lets a single free-list serve the whole model, and enables future
cross-layer sharing (e.g. prefix caching) where the same block can be
reused by multiple sequences at the same logical position.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch


class BlockPoolExhaustedError(RuntimeError):
    """The pool cannot hold what was asked of it.

    Raised by `allocate` with no free block left, and by
    `ContinuousKVCache.advance` before it allocates anything, when a step's
    tokens need more blocks than are free.
    """


class BlockPool:
    """Fixed-size pool of KV storage blocks, shared across all transformer layers.

    Physical layout::

        keys  : [num_layers, num_blocks * block_size, num_kv_heads, head_dim]
        values: same shape as keys

    Storage is a flat run of token *slots*; a block is just ``block_size``
    consecutive slots, so the slot holding token ``t`` of a block is
    ``block_idx * block_size + t``. Addressing a token by a single integer is
    what lets the caches read and write a whole batch with one indexing op
    instead of a Python loop over blocks (see ``slot_table``).

    A single free-list tracks which block indices are available. Allocating
    block B means all layers can write K/V into that block's slots. Freeing
    block B returns it to the free-list regardless of which layers wrote to it.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        shape = (num_layers, num_blocks * block_size, num_kv_heads, head_dim)
        self._keys = torch.zeros(shape, dtype=dtype, device=device)
        self._values = torch.zeros(shape, dtype=dtype, device=device)
        self._free_blocks: list[int] = list(range(num_blocks))

    @property
    def nbytes(self) -> int:
        """Bytes of GPU memory this pool holds, keys and values together."""
        return self._keys.numel() * self._keys.element_size() * 2

    @property
    def device(self) -> torch.device:
        return self._keys.device

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def allocate(self) -> int:
        """Pop and return a free block index.

        Raises:
            BlockPoolExhaustedError: if no free blocks remain.
        """
        if not self._free_blocks:
            raise BlockPoolExhaustedError(
                f"KV block pool exhausted: all {self.num_blocks} blocks are in use. "
                "Increase num_gpu_blocks or reduce max_num_seqs / max_model_len."
            )
        return self._free_blocks.pop()

    def free(self, block_idx: int) -> None:
        """Return block_idx to the free-list."""
        self._free_blocks.append(block_idx)

    def slots(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this layer's flat key/value stores: ``[num_slots, num_kv_heads, head_dim]``."""
        return self._keys[layer_idx], self._values[layer_idx]


def _padded_blocks(block_tables: Sequence[Sequence[int]], device: torch.device) -> torch.Tensor:
    """Every sequence's block table as one ``[B, max_blocks]`` tensor, in one transfer.

    Padded on the host and moved once: a row-by-row copy costs one
    host-to-device transfer per sequence — and those are pageable, so each one
    blocks — where this costs one for the batch. The padding is block 0, which
    is a real block; no answer depends on it, because every read is bounded by
    its sequence's context length.
    """
    width = max(len(table) for table in block_tables)
    padded = [list(table) + [0] * (width - len(table)) for table in block_tables]
    return torch.tensor(padded, dtype=torch.long, device=device)


def _slots_of(blocks: torch.Tensor, max_total: int, block_size: int) -> torch.Tensor:
    """``[B, max_total]`` slot per logical position, read off padded block tables.

    Left-aligned: column ``p`` of row ``i`` is position ``p`` of sequence ``i``,
    so a kernel reads its sequence's first ``context_len`` columns and nothing
    past them. One gather for the whole batch.
    """
    columns = torch.arange(max_total, device=blocks.device)
    return blocks[:, columns // block_size] * block_size + columns % block_size


def slot_table(
    block_tables: Sequence[Sequence[int]],
    totals: Sequence[int],
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Map each sequence's logical positions ``[0, totals[i])`` to physical pool slots.

    Returns ``[B, max(totals)]``, left-aligned: a row's columns past its total
    hold whatever the padding addresses, and are never read. ``max(totals)``
    comes from the Python counts, because reading it off the device would sync
    the whole queue to learn something already known.
    """
    return _slots_of(_padded_blocks(block_tables, device), max(totals), block_size)


def newest_slots(
    block_tables: Sequence[Sequence[int]],
    positions: Sequence[int],
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """The slot of one position per sequence, as ``[1, B]`` — where a decode step writes.

    Computed on the host: the device path is about ten kernel launches and
    three transfers whatever the batch holds, which measured 0.28 ms at batch 1
    — about 5% of a captured decode step, paid on every one. For one token per
    sequence the answer is B integers the host can compute from the block
    tables it already holds, in one transfer: 0.021 ms at batch 1 and 0.048 ms
    at 128. The leading axis of 1 is the packed batch axis the layers carry.
    """
    slots = [
        table[position // block_size] * block_size + position % block_size
        for table, position in zip(block_tables, positions, strict=True)
    ]
    return torch.tensor([slots], dtype=torch.long, device=device)


class PackedAddresses(NamedTuple):
    """Where every token of a packed run lives, what each sequence reads, and where it begins."""

    slots: torch.Tensor
    """``[1, total]`` physical slot per token of the run — where the pass writes."""
    positions: torch.Tensor
    """``[1, total]`` logical position per token, which is what RoPE reads."""
    query_start_loc: torch.Tensor
    """``[B + 1]`` int32 prefix sums of the counts: sequence ``i`` owns
    ``[query_start_loc[i], query_start_loc[i + 1])``, so its last token is one
    before the next sequence's first."""
    slot_table: torch.Tensor
    """``[B, max_context]`` left-aligned slot per position of each sequence's whole
    context, the run's tokens included — what attention reads."""
    context_lens: torch.Tensor
    """``[B]`` int32 tokens each sequence holds once the run is written."""


def packed_addresses(
    block_tables: Sequence[Sequence[int]],
    starts: Sequence[int],
    counts: Sequence[int],
    block_size: int,
    device: torch.device,
) -> PackedAddresses:
    """Every address a packed pass needs, from one transfer of the block tables.

    Sequence ``i`` contributes its logical positions ``[starts[i], starts[i] +
    counts[i])``, the newest tokens of a context of ``starts[i] + counts[i]``.
    The read table covers that whole context, and the write slots are a gather
    out of it: token ``j`` of the run is at its owner's row, its position's
    column. The owner comes from `repeat_interleave`, the position from a
    running index minus where the owner's run starts — `cumsum` gives that for
    every token at once — plus the owner's start. The total and the widest
    context come from the Python counts, so nothing here waits on the device.
    """
    blocks = _padded_blocks(block_tables, device)
    count, start = torch.tensor([list(counts), list(starts)], dtype=torch.long, device=device)
    total = sum(counts)
    owner = torch.repeat_interleave(torch.arange(len(counts), device=device), count, output_size=total)
    run_ends = torch.cumsum(count, 0)
    positions = torch.arange(total, device=device) - (run_ends - count)[owner] + start[owner]

    table = _slots_of(blocks, max(s + c for s, c in zip(starts, counts, strict=True)), block_size)
    return PackedAddresses(
        slots=table[owner, positions].unsqueeze(0),
        positions=positions.unsqueeze(0),
        query_start_loc=torch.nn.functional.pad(run_ends, (1, 0)).to(torch.int32),
        slot_table=table,
        context_lens=(start + count).to(torch.int32),
    )
