# pyright: reportPrivateImportUsage=false
"""Pre-allocated physical KV storage divided into fixed-size blocks.

One block index services all transformer layers: block B is K/V memory at
pool.keys[layer_idx, B, ...] for every layer_idx. That is what lets one set of
host-side block tables (`cache/block_tables.py`) serve the whole model, and
enables future cross-layer sharing (e.g. prefix caching) where the same block
can be reused by multiple sequences at the same logical position.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch


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

    Which blocks are free and which sequence holds which is `BlockTables`, on
    the host; this is only the storage they index, for every layer at once.
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

    @property
    def nbytes(self) -> int:
        """Bytes of GPU memory this pool holds, keys and values together."""
        return self._keys.numel() * self._keys.element_size() * 2

    @property
    def device(self) -> torch.device:
        return self._keys.device

    def slots(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this layer's flat key/value stores: ``[num_slots, num_kv_heads, head_dim]``."""
        return self._keys[layer_idx], self._values[layer_idx]


def _padded_blocks(
    block_tables: Sequence[Sequence[int]], device: torch.device, *leading: Sequence[int]
) -> torch.Tensor:
    """Every sequence's block table as one ``[B, len(leading) + max_blocks]`` tensor, in one transfer.

    Padded on the host and moved once: a row-by-row copy costs one
    host-to-device transfer per sequence — and those are pageable, so each one
    blocks — where this costs one for the batch. Each of `leading` is one more
    integer per sequence riding in the same transfer, as the first columns, for
    the same reason. The padding is block 0, which is a real block; no answer
    depends on it, because every read is bounded by its sequence's context length.
    """
    width = max(len(table) for table in block_tables)
    fronts = list(zip(*leading, strict=True)) if leading else [()] * len(block_tables)
    rows = [
        [*front, *table, *[0] * (width - len(table))]
        for front, table in zip(fronts, block_tables, strict=True)
    ]
    return torch.tensor(rows, dtype=torch.long, device=device)


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
    moved = _padded_blocks(block_tables, device, counts, starts)
    count, start, blocks = moved[:, 0], moved[:, 1], moved[:, 2:]
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
