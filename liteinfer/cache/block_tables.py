"""Which KV blocks each sequence holds, and which are free — host-side bookkeeping only.

Kept apart from the device storage (`BlockPool`) because two components need it
and only one needs the tensors: the scheduler decides what a step can admit by
how many blocks are free, and the runner allocates what the step was granted
before addressing it. Neither is a tensor operation, and the scheduler is built
without the model, so this holds integers and nothing else.

How many tokens a sequence holds is `Sequence.num_computed_tokens`; every method
that needs a count is handed one.
"""

from __future__ import annotations

from collections.abc import Sequence


class BlockPoolExhaustedError(RuntimeError):
    """The pool cannot hold what was asked of it.

    Raised by `BlockTables.reserve` before it allocates anything, and handed to
    a request the scheduler refuses for want of blocks.
    """


class BlockTables:
    """Per-sequence block tables over one free list of `num_blocks` blocks."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free_blocks: list[int] = list(range(num_blocks))
        self._tables: dict[str, list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def blocks_for(self, total: int) -> int:
        """Blocks it takes to hold `total` tokens from nothing."""
        return -(-total // self.block_size)

    def blocks_short_of(self, request_id: str, total: int) -> int:
        """Blocks a sequence must be given before it can hold `total` tokens."""
        return max(0, self.blocks_for(total) - len(self._tables.get(request_id, ())))

    def capacity(self, request_id: str) -> int:
        """Most tokens a sequence can hold in the blocks it has; 0 before its first reservation."""
        return len(self._tables.get(request_id, ())) * self.block_size

    def table(self, request_id: str) -> list[int]:
        """The blocks a sequence holds, in logical order. It must have been given room."""
        return self._tables[request_id]

    def reserve(self, request_ids: Sequence[str], totals: Sequence[int]) -> None:
        """Give each sequence the blocks to hold `totals[i]` tokens, before a pass writes them.

        All or nothing: when the free blocks cannot cover every sequence, it raises
        `BlockPoolExhaustedError` before anything changes, so a refused step leaves
        no sequence holding blocks for tokens no pass wrote. A sequence's table
        exists from its first reservation to its `release`.
        """
        wanted = list(zip(request_ids, totals, strict=True))
        needed = sum(self.blocks_short_of(request_id, total) for request_id, total in wanted)
        if needed > self.num_free_blocks:
            raise BlockPoolExhaustedError(
                f"KV block pool exhausted: this step needs {needed} more blocks and "
                f"{self.num_free_blocks} of {self.num_blocks} are free. "
                "Increase num_gpu_blocks or reduce max_num_seqs / max_model_len."
            )
        for request_id, total in wanted:
            table = self._tables.setdefault(request_id, [])
            while len(table) * self.block_size < total:
                table.append(self._free_blocks.pop())

    def release(self, request_id: str) -> None:
        """Return every block a sequence holds to the free list."""
        self._free_blocks.extend(self._tables.pop(request_id, ()))
