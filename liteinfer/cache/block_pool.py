"""Pre-allocated physical KV storage divided into fixed-size blocks.

One block index services all transformer layers: allocating block B gives
access to K/V memory at pool.keys[layer_idx, B, ...] for every layer_idx.
This lets a single free-list serve the whole model, and enables future
cross-layer sharing (e.g. prefix caching) where the same block can be
reused by multiple sequences at the same logical position.
"""

from __future__ import annotations

import torch


class BlockPoolExhaustedError(RuntimeError):
    """Raised when BlockPool.allocate() is called with no free blocks remaining."""


class BlockPool:
    """Fixed-size pool of KV storage blocks, shared across all transformer layers.

    Physical layout::

        keys  : [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
        values: same shape as keys

    A single free-list tracks which block indices are available. Allocating
    block B means all layers can write K/V into ``keys[layer_idx, B, ...]``
    and ``values[layer_idx, B, ...]``.  Freeing block B returns it to the
    free-list regardless of which layers wrote to it.
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

        self._keys = torch.zeros(
            num_layers,
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._values = torch.zeros(
            num_layers,
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._free_blocks: list[int] = list(range(num_blocks))

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

    def write_tokens(
            self,
            layer_idx: int,
            block_idx: int,
            slot_offset: int,
            k: torch.Tensor,
            v: torch.Tensor,
    ) -> None:
        """Write KV tokens into slots of a block.

        Args:
            layer_idx: which transformer layer's K/V to write.
            block_idx: physical block index returned by allocate().
            slot_offset: first slot within the block to write (0-indexed).
            k: key states to store, shape ``[num_kv_heads, n_tokens, head_dim]``.
            v: value states to store, same shape as k.
        """
        n = k.shape[1]
        self._keys[layer_idx, block_idx, :, slot_offset: slot_offset + n, :] = k
        self._values[layer_idx, block_idx, :, slot_offset: slot_offset + n, :] = v

    def get_key_block(self, layer_idx: int, block_idx: int) -> torch.Tensor:
        """Return a view of the key block: ``[num_kv_heads, block_size, head_dim]``."""
        return self._keys[layer_idx, block_idx]

    def get_value_block(self, layer_idx: int, block_idx: int) -> torch.Tensor:
        """Return a view of the value block: ``[num_kv_heads, block_size, head_dim]``."""
        return self._values[layer_idx, block_idx]
