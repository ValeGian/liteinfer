"""Logical token positions map to physical pool slots."""

from __future__ import annotations

import torch

from liteinfer.cache.block_pool import BlockPool, slot_table
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache

CPU = torch.device("cpu")
BLOCK_SIZE = 4


def _table(block_tables, counts):
    return slot_table(block_tables, counts, BLOCK_SIZE, CPU)


def test_slots_follow_block_index_times_block_size() -> None:
    # Block 2, three tokens -> slots 8, 9, 10.
    assert _table([[2]], [3]).tolist() == [[8, 9, 10]]


def test_second_block_continues_the_sequence() -> None:
    # Blocks 0 then 3, five tokens -> 0..3 then 12.
    assert _table([[0, 3]], [5]).tolist() == [[0, 1, 2, 3, 12]]


def test_newest_token_is_the_last_column() -> None:
    # The decode write path relies on this.
    assert _table([[1]], [2])[0, -1].item() == 5


def test_shorter_sequences_are_right_aligned() -> None:
    rows = _table([[0], [1]], [1, 3])
    assert rows[0].tolist()[-1] == 0


def test_padding_columns_point_at_slot_zero() -> None:
    # Padded columns are discarded by the attention mask.
    rows = _table([[2], [1]], [1, 3])
    assert rows[0].tolist()[:2] == [0, 0]


def test_every_sequence_gets_max_total_columns() -> None:
    assert _table([[0], [1]], [1, 3]).shape == (2, 3)


def test_unequal_block_table_lengths_are_handled() -> None:
    rows = _table([[0, 1], [2]], [5, 2])
    assert rows[1].tolist()[-2:] == [8, 9]


def test_ragged_block_tables_pad_to_the_null_block() -> None:
    """Rows are padded on the host before the transfer; the padding must be block 0."""
    table = slot_table([[3, 7], [5]], counts=[20, 4], block_size=16, device=torch.device("cpu"))

    assert table[1, -4:].tolist() == [5 * 16 + i for i in range(4)]


def test_max_total_comes_from_the_counts_not_the_device() -> None:
    """Width is a property of the Python counts, so it must not need a sync to learn."""
    table = slot_table([[1], [1]], counts=[3, 9], block_size=16, device=torch.device("cpu"))

    assert table.shape[1] == 9


def test_advance_allocates_a_block_only_when_the_last_one_fills() -> None:
    """Block allocation is host-side bookkeeping, and must stay out of the forward."""
    pool = BlockPool(
        num_blocks=8, block_size=4, num_layers=1, num_kv_heads=1, head_dim=2,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    cache = ContinuousKVCache(pool)
    cache.register("r0", prompt_len=4)          # one block, exactly full
    blocks_after_prompt = len(cache._block_tables["r0"])

    cache.advance(["r0"])                        # token 5 needs a second block

    assert len(cache._block_tables["r0"]) == blocks_after_prompt + 1
