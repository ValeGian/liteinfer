"""Pre-allocated physical KV storage divided into fixed-size blocks.

One block index services all transformer layers: allocating block B gives
access to K/V memory at pool.keys[layer_idx, B, ...] for every layer_idx.
This lets a single free-list serve the whole model, and enables future
cross-layer sharing (e.g. prefix caching) where the same block can be
reused by multiple sequences at the same logical position.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class BlockPoolExhaustedError(RuntimeError):
    """Raised when BlockPool.allocate() is called with no free blocks remaining."""


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

    Block 0 is a *null block*: never allocated, it absorbs the reads and writes
    that padded batch positions generate, so neither caller needs to mask them.

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

        # One extra block backs the null block, so num_blocks stays the usable count.
        shape = (num_layers, (num_blocks + 1) * block_size, num_kv_heads, head_dim)
        self._keys = torch.zeros(shape, dtype=dtype, device=device)
        self._values = torch.zeros(shape, dtype=dtype, device=device)
        self._free_blocks: list[int] = list(range(1, num_blocks + 1))

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

    def _block(self, store: torch.Tensor, layer_idx: int, block_idx: int) -> torch.Tensor:
        start = block_idx * self.block_size
        return store[layer_idx, start : start + self.block_size].transpose(0, 1)

    def get_key_block(self, layer_idx: int, block_idx: int) -> torch.Tensor:
        """Return a view of the key block: ``[num_kv_heads, block_size, head_dim]``."""
        return self._block(self._keys, layer_idx, block_idx)

    def get_value_block(self, layer_idx: int, block_idx: int) -> torch.Tensor:
        """Return a view of the value block: ``[num_kv_heads, block_size, head_dim]``."""
        return self._block(self._values, layer_idx, block_idx)

    def slots(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this layer's flat key/value stores: ``[num_slots, num_kv_heads, head_dim]``."""
        return self._keys[layer_idx], self._values[layer_idx]


def _window_blocks(
    block_tables: Sequence[Sequence[int]],
    starts: Sequence[int],
    counts: Sequence[int],
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The block rows each window of tokens touches, plus its count and where it begins.

    Sequence ``i``'s window is its logical positions ``[starts[i], starts[i] +
    counts[i])``, and ``counts[i]`` must be at least 1. Only the blocks that
    window covers are padded on the host and moved: a window that is one
    sequence's newest few tokens needs one or two blocks, not the whole table,
    and the whole table is what makes the read table cost 3.4 ms to build at 128
    sequences of 4,000 tokens.

    Returns ``[B, width]`` block indices padded with the null block, then ``[B]``
    counts and ``[B]`` offsets of each window's first token from the start of its
    first block. The two vectors travel as one tensor — each transfer from
    pageable memory blocks, so two small ones cost twice what one does.
    """
    first_blocks = [start // block_size for start in starts]
    rows = [
        table[first : (start + count - 1) // block_size + 1]
        for table, first, start, count in zip(block_tables, first_blocks, starts, counts, strict=True)
    ]
    width = max(len(row) for row in rows)
    padded = [list(row) + [0] * (width - len(row)) for row in rows]
    offsets = [start - first * block_size for start, first in zip(starts, first_blocks, strict=True)]
    blocks = torch.tensor(padded, dtype=torch.long, device=device)
    count, offset = torch.tensor([list(counts), offsets], dtype=torch.long, device=device)
    return blocks, count, offset


def slot_mapping(
    block_tables: Sequence[Sequence[int]],
    counts: Sequence[int],
    block_size: int,
    device: torch.device,
    starts: Sequence[int] | None = None,
) -> torch.Tensor:
    """One physical slot per token, the sequences laid end to end.

    Sequence ``i`` contributes its logical positions ``[starts[i], starts[i] +
    counts[i])``, from 0 when ``starts`` is omitted. That one shape covers every
    packed address the engine needs: a whole prompt (start 0), a chunk of one
    that continues where the cache left off, and a decode step, which is the
    batch where every count is 1.

    Returns ``[1, sum(counts)]``, so it indexes a packed pass the way
    `slot_table` indexes a padded one — same arithmetic, no padded columns and no
    right-alignment, because a packed batch has nothing to align to. The leading
    axis of 1 is the packed batch axis the layers still carry.

    Built on the device from three vectors rather than a Python loop per
    sequence: the request each token belongs to, its position within that
    request's window, and the block row to read. That is the same shape of
    computation `slot_table` does, and it is why both live here. The total comes
    from the Python counts, so nothing here waits on the device.

    A batch of single-token windows — every decode step — is answered on the
    host instead; see `_single_token_slots`.
    """
    starts = [0] * len(counts) if starts is None else starts
    if all(count == 1 for count in counts):
        return _single_token_slots(block_tables, starts, block_size, device)
    total = sum(counts)
    blocks, count, offsets = _window_blocks(block_tables, starts, counts, block_size, device)

    owner = torch.repeat_interleave(
        torch.arange(len(counts), device=device), count, output_size=total
    )
    # Position within the owning window: a running index minus where that
    # window starts in the run, which `cumsum` gives for every token at once.
    run_starts = torch.cumsum(count, 0) - count
    local = torch.arange(total, device=device) - run_starts[owner] + offsets[owner]

    slots = blocks[owner].gather(1, (local // block_size).unsqueeze(1)).squeeze(1)
    return (slots * block_size + local % block_size).unsqueeze(0)


def _single_token_slots(
    block_tables: Sequence[Sequence[int]],
    positions: Sequence[int],
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """`slot_mapping` for one token per sequence, as ``[1, B]``, computed in Python.

    The device path above is about ten kernel launches and three transfers
    whatever the batch holds, which measured 0.28 ms at batch 1 — about 5% of a
    captured decode step, paid on every one. For one token per sequence the
    answer is B integers the host can compute from the block tables it already
    holds, in one transfer: 0.021 ms at batch 1 and 0.048 ms at 128.
    """
    slots = [
        table[position // block_size] * block_size + position % block_size
        for table, position in zip(block_tables, positions, strict=True)
    ]
    return torch.tensor([slots], dtype=torch.long, device=device)


def slot_table(
    block_tables: Sequence[Sequence[int]],
    counts: Sequence[int],
    block_size: int,
    device: torch.device,
    starts: Sequence[int] | None = None,
) -> torch.Tensor:
    """Map each sequence's logical token positions to physical pool slots.

    Row ``i`` holds positions ``[starts[i], starts[i] + counts[i])``, from 0 when
    ``starts`` is omitted. Returns ``[B, max(counts)]``, right-aligned so it
    matches the left-padding the attention masks expect; each window's newest
    token is therefore the last column, and padded columns point into the null
    block.

    The block tables are padded on the host and moved in a single transfer. A
    row-by-row copy costs one host-to-device transfer per sequence — and those
    are pageable, so each one blocks — where this costs one for the batch, plus
    one for the counts and offsets.
    ``max(counts)`` comes from the Python counts for the same reason: reading it
    off the device would sync the whole queue to learn something already known.
    """
    is_windowed = starts is not None
    starts = starts if is_windowed else [0] * len(counts)
    max_count = max(counts)
    blocks, count, offsets = _window_blocks(block_tables, starts, counts, block_size, device)

    local = torch.arange(max_count, device=device) - (max_count - count).unsqueeze(1)
    is_real = local >= 0
    local = local.clamp(min=0)
    if is_windowed:
        # Skipped for the whole history — every decode step's read — where every
        # offset is zero and the add is a launch that changes nothing.
        local = local + offsets.unsqueeze(1)
    slots = blocks.gather(1, local // block_size) * block_size + local % block_size
    return torch.where(is_real, slots, 0)
