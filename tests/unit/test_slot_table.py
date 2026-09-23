"""Logical token positions map to physical pool slots."""

from __future__ import annotations

import torch

from liteinfer.cache.block_pool import BlockPool, slot_mapping, slot_table
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache

CPU = torch.device("cpu")
BLOCK_SIZE = 4


def _table(block_tables, counts):
    return slot_table(block_tables, counts, BLOCK_SIZE, CPU)


def _mapping(block_tables, counts, starts=None):
    return slot_mapping(block_tables, counts, BLOCK_SIZE, CPU, starts=starts)


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
    cache.register("r0")
    cache.advance(["r0"], [4])                   # one block, exactly full
    blocks_after_prompt = len(cache._block_tables["r0"])

    cache.advance(["r0"], [1])                   # token 5 needs a second block

    assert len(cache._block_tables["r0"]) == blocks_after_prompt + 1


def test_advance_allocates_every_block_a_chunk_needs_at_once() -> None:
    """A prompt chunk can span several blocks; a decode step never more than one."""
    cache = _cache_with([9])

    assert len(cache._block_tables["r0"]) == 3


def test_a_window_of_the_table_starts_where_it_is_told() -> None:
    # Blocks 2 then 5, positions 3..5 -> 11 in the first block, then 20, 21.
    assert slot_table([[2, 5]], [3], BLOCK_SIZE, CPU, starts=[3]).tolist() == [[11, 20, 21]]


def test_a_shorter_window_is_right_aligned_behind_the_null_block() -> None:
    table = slot_table([[2, 5], [1]], [3, 1], BLOCK_SIZE, CPU, starts=[3, 2])

    assert table[1].tolist() == [0, 0, 6]


# --- packed: the same addresses, laid end to end -----------------------------


def test_packed_mapping_lays_sequences_end_to_end() -> None:
    # Block 2 for three tokens, block 5 for two -> 8, 9, 10 then 20, 21.
    assert _mapping([[2], [5]], [3, 2]).tolist() == [[8, 9, 10, 20, 21]]


def test_packed_mapping_holds_one_slot_per_real_token() -> None:
    """No padded columns, which is the whole difference from `slot_table`."""
    assert _mapping([[1], [3, 6], [0]], [2, 7, 1]).shape == (1, 10)


def test_packed_mapping_crosses_blocks_within_a_sequence() -> None:
    # Blocks 0 then 3, five tokens -> 0..3 then 12, same as the padded table.
    assert _mapping([[0, 3]], [5]).tolist() == [[0, 1, 2, 3, 12]]


def test_packed_mapping_addresses_the_same_slots_as_the_padded_table() -> None:
    """Two layouts of one answer: the padded table's real columns, concatenated."""
    block_tables, counts = [[2], [5, 1]], [3, 6]
    padded = _table(block_tables, counts)
    real = torch.cat([row[-count:] for row, count in zip(padded, counts, strict=True)])

    assert _mapping(block_tables, counts).tolist() == [real.tolist()]


def test_packed_mapping_starts_each_window_where_it_is_told() -> None:
    """A chunk continuing a prompt starts part-way into its block table."""
    # Blocks 2 then 5 from position 3 -> 11, 20, 21; block 1 from position 2 -> 6.
    assert _mapping([[2, 5], [1]], [3, 1], starts=[3, 2]).tolist() == [[11, 20, 21, 6]]


def test_a_decode_write_is_the_padded_table_s_newest_column() -> None:
    """Every count 1, starting at the newest token: the right-aligned table's last column."""
    block_tables, counts = [[2, 7], [5, 1, 3]], [6, 9]
    newest_column = _table(block_tables, counts)[:, -1]

    decode_write = _mapping(block_tables, [1, 1], starts=[count - 1 for count in counts])

    assert decode_write.tolist() == [newest_column.tolist()]


def test_the_cache_addresses_each_sequence_s_newest_tokens() -> None:
    """What a pass writes is what it just `advance`d, wherever the sequence already was."""
    cache = _cache_with([3])
    cache.advance(["r0"], [2])
    everything = cache.slot_table_for(["r0"])[0].tolist()

    assert cache.slot_mapping_for(["r0"], [2]).tolist() == [everything[-2:]]


def test_single_token_windows_are_addressed_as_a_longer_window_would_address_them() -> None:
    """The host path for decode must agree with the device path it stands in for."""
    block_tables, starts = [[2, 7], [5, 1, 3]], [5, 10]
    as_pairs = _mapping(block_tables, [2, 2], starts=[start - 1 for start in starts])[0, 1::2]

    assert _mapping(block_tables, [1, 1], starts=starts).tolist() == [as_pairs.tolist()]
