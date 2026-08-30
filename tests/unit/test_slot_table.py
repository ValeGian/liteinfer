"""Logical token positions map to physical pool slots."""

from __future__ import annotations

import torch

from liteinfer.cache.block_pool import slot_table

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
