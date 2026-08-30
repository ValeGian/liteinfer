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


def slot_table(
    block_tables: Sequence[Sequence[int]],
    counts: Sequence[int],
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Map each sequence's logical token positions to physical pool slots.

    Returns ``[B, max_total]``, right-aligned so it matches the left-padding the
    attention masks expect; the newest token is therefore the last column, and
    padded columns point into the null block.

    Both caches use this for every read and write, so block addressing lives in
    exactly one place.
    """
    count = torch.tensor(counts, device=device)
    max_total = int(count.max())

    blocks = torch.zeros(
        (len(block_tables), max(len(t) for t in block_tables)), dtype=torch.long, device=device
    )
    for row, table in enumerate(block_tables):
        blocks[row, : len(table)] = torch.tensor(table, device=device)

    logical = torch.arange(max_total, device=device) - (max_total - count).unsqueeze(1)
    is_real = logical >= 0
    logical = logical.clamp(min=0)
    slots = blocks.gather(1, logical // block_size) * block_size + logical % block_size
    return torch.where(is_real, slots, 0)
