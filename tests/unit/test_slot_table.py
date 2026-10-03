# pyright: reportPrivateImportUsage=false
"""Logical token positions map to physical pool slots."""

from __future__ import annotations

import torch

from liteinfer.cache.block_pool import BlockPool, newest_slots, packed_addresses, slot_table
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache

CPU = torch.device("cpu")
BLOCK_SIZE = 4


def _table(block_tables, totals):
    return slot_table(block_tables, totals, BLOCK_SIZE, CPU)


def _cache_with(counts: list[int]) -> ContinuousKVCache:
    pool = BlockPool(
        num_blocks=16, block_size=BLOCK_SIZE, num_layers=1, num_kv_heads=1, head_dim=2,
        dtype=torch.float32, device=CPU,
    )
    cache = ContinuousKVCache(pool)
    for i, count in enumerate(counts):
        cache.register(f"r{i}")
        cache.advance([f"r{i}"], [count])
    return cache


def test_slots_follow_block_index_times_block_size() -> None:
    # Block 2, three tokens -> slots 8, 9, 10.
    assert _table([[2]], [3]).tolist() == [[8, 9, 10]]


def test_second_block_continues_the_sequence() -> None:
    # Blocks 0 then 3, five tokens -> 0..3 then 12.
    assert _table([[0, 3]], [5]).tolist() == [[0, 1, 2, 3, 12]]


def test_shorter_sequences_are_left_aligned() -> None:
    """Column p is position p, so a kernel reads a row's first `context_len` columns."""
    rows = _table([[0], [1]], [1, 3])
    assert rows[1].tolist() == [4, 5, 6]


def test_every_sequence_gets_max_total_columns() -> None:
    assert _table([[0], [1]], [1, 3]).shape == (2, 3)


def test_unequal_block_table_lengths_are_handled() -> None:
    rows = _table([[0, 1], [2]], [5, 2])
    assert rows[1].tolist()[:2] == [8, 9]


def test_max_total_comes_from_the_counts_not_the_device() -> None:
    """Width is a property of the Python counts, so it must not need a sync to learn."""
    table = slot_table([[1], [1]], [3, 9], block_size=16, device=CPU)

    assert table.shape[1] == 9


def test_advance_allocates_a_block_only_when_the_last_one_fills() -> None:
    """Block allocation is host-side bookkeeping, and must stay out of the forward."""
    pool = BlockPool(
        num_blocks=8, block_size=4, num_layers=1, num_kv_heads=1, head_dim=2,
        dtype=torch.float32, device=CPU,
    )
    cache = ContinuousKVCache(pool)
    cache.register("r0")
    cache.advance(["r0"], [4])                   # one block, exactly full
    blocks_after_prompt = len(cache._block_tables["r0"])

    cache.advance(["r0"], [1])                   # token 5 needs a second block

    assert len(cache._block_tables["r0"]) == blocks_after_prompt + 1


def test_advance_allocates_every_block_a_chunk_needs_at_once() -> None:
    """A prompt chunk can span several blocks; a decode step never more than one."""
    cache = _cache_with([9])

    assert len(cache._block_tables["r0"]) == 3


# --- where a decode step writes ----------------------------------------------


def test_a_decode_write_is_each_sequence_s_newest_position() -> None:
    """One slot per sequence, as ``[1, B]``: the table's column at its last position."""
    block_tables, totals = [[2, 7], [5, 1, 3]], [6, 9]
    table = _table(block_tables, totals)
    newest = [table[row, total - 1].item() for row, total in enumerate(totals)]

    assert newest_slots(block_tables, [t - 1 for t in totals], BLOCK_SIZE, CPU).tolist() == [newest]


def test_the_cache_writes_a_decode_token_where_it_just_advanced() -> None:
    """What a pass writes is what it just `advance`d, wherever the sequence already was."""
    cache = _cache_with([3])
    cache.advance(["r0"], [1])
    everything = cache.slot_table_for(["r0"])[0].tolist()

    assert cache.newest_slots_for(["r0"]).tolist() == [[everything[-1]]]


# --- a packed pass: where it writes and what it reads, from one transfer -------


def _addresses(starts, counts):
    # Three sequences, two blocks each, so windows straddle block boundaries.
    return packed_addresses([[2, 5], [1, 7], [3, 6]], starts, counts, BLOCK_SIZE, CPU)


def test_packed_positions_count_from_where_each_window_starts() -> None:
    assert _addresses([5, 0, 3], [3, 2, 1]).positions.tolist() == [[5, 6, 7, 0, 1, 3]]


def test_packed_slots_follow_each_window_into_its_own_blocks() -> None:
    # Positions 5-7 in block 5, 0-1 in block 1, 3 in block 3: 5*4+1.., 1*4+0.., 3*4+3.
    assert _addresses([5, 0, 3], [3, 2, 1]).slots.tolist() == [[21, 22, 23, 4, 5, 15]]


def test_packed_slots_hold_one_slot_per_real_token() -> None:
    """No padded columns, which is the whole difference from the read table."""
    assert _addresses([5, 0, 3], [3, 2, 1]).slots.shape == (1, 6)


def test_packed_query_starts_are_the_running_sum_of_the_counts() -> None:
    assert _addresses([5, 0, 3], [3, 2, 1]).query_start_loc.tolist() == [0, 3, 5, 6]


def test_packed_query_starts_are_int32_as_the_kernels_read_them() -> None:
    assert _addresses([0, 0, 0], [2, 2, 2]).query_start_loc.dtype == torch.int32


def test_a_packed_pass_reads_each_sequence_s_whole_context() -> None:
    """The read table is the slot table of everything each sequence holds, the window included."""
    block_tables, starts, counts = [[2, 5], [1, 7], [3, 6]], [5, 0, 3], [3, 2, 1]
    totals = [s + c for s, c in zip(starts, counts, strict=True)]

    addresses = packed_addresses(block_tables, starts, counts, BLOCK_SIZE, CPU)

    torch.testing.assert_close(addresses.slot_table, _table(block_tables, totals))


def test_a_packed_pass_s_context_ends_after_its_window() -> None:
    assert _addresses([5, 0, 3], [3, 2, 1]).context_lens.tolist() == [8, 2, 4]


def test_a_packed_pass_writes_where_its_read_table_says_its_window_lives() -> None:
    """Write and read addresses come from one transfer, so they cannot disagree."""
    addresses = _addresses([5, 0, 3], [3, 2, 1])
    windows = [(0, 5, 8), (1, 0, 2), (2, 3, 4)]
    expected = [slot for row, first, end in windows for slot in addresses.slot_table[row, first:end].tolist()]

    assert addresses.slots.tolist() == [expected]
