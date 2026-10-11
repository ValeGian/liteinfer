"""Which blocks each sequence holds, and which are free: host-side, no tensors."""

from __future__ import annotations

import pytest

from liteinfer.cache.block_tables import BlockPoolExhaustedError, BlockTables

BLOCK_SIZE = 4


def _holding(num_blocks: int, totals: dict[str, int]) -> BlockTables:
    blocks = BlockTables(num_blocks, BLOCK_SIZE)
    for request_id, total in totals.items():
        blocks.reserve([request_id], [total])
    return blocks


# --- the free list -------------------------------------------------------------


def test_every_block_starts_free() -> None:
    assert BlockTables(num_blocks=8, block_size=BLOCK_SIZE).num_free_blocks == 8


def test_every_block_is_handed_out_before_the_pool_runs_dry() -> None:
    """No block is held back: every read stops at its sequence's context, so none absorbs padding."""
    blocks = _holding(3, {"r0": 3 * BLOCK_SIZE})

    assert sorted(blocks.table("r0")) == [0, 1, 2]


def test_released_blocks_return_to_the_free_list() -> None:
    blocks = _holding(8, {"r0": 9})

    blocks.release("r0")

    assert blocks.num_free_blocks == 8


def test_releasing_a_sequence_that_holds_nothing_is_a_no_op() -> None:
    """A request refused before it was given room is still released when it finishes."""
    blocks = BlockTables(num_blocks=2, block_size=BLOCK_SIZE)

    blocks.release("never-admitted")

    assert blocks.num_free_blocks == 2


# --- reserving room for a pass --------------------------------------------------


def test_reserve_allocates_a_block_only_when_the_last_one_fills() -> None:
    blocks = _holding(8, {"r0": 4})  # one block, exactly full

    blocks.reserve(["r0"], [5])  # token 5 needs a second block

    assert len(blocks.table("r0")) == 2


def test_reserve_allocates_every_block_a_chunk_needs_at_once() -> None:
    """A prompt chunk can span several blocks; a decode step never more than one."""
    assert len(_holding(8, {"r0": 9}).table("r0")) == 3


def test_a_step_the_pool_cannot_hold_is_refused() -> None:
    blocks = _holding(3, {"running": 4})

    with pytest.raises(BlockPoolExhaustedError):
        blocks.reserve(["running", "new"], [5, 8])  # 1 + 2 blocks against 2 free


def test_a_refused_step_allocates_no_blocks() -> None:
    blocks = _holding(3, {"running": 4})

    with pytest.raises(BlockPoolExhaustedError):
        blocks.reserve(["running", "new"], [5, 8])

    assert blocks.num_free_blocks == 2


def test_a_refused_step_gives_a_newcomer_no_table() -> None:
    blocks = _holding(3, {"running": 4})

    with pytest.raises(BlockPoolExhaustedError):
        blocks.reserve(["running", "new"], [5, 8])

    assert blocks.blocks_short_of("new", 8) == 2


def test_a_step_that_exactly_fills_the_pool_is_accepted() -> None:
    blocks = _holding(3, {"running": 4})

    blocks.reserve(["running", "new"], [5, 4])  # one block each, two free

    assert blocks.num_free_blocks == 0


# --- what the scheduler asks ---------------------------------------------------


def test_blocks_short_of_counts_only_what_a_sequence_does_not_hold() -> None:
    assert _holding(8, {"r0": 5}).blocks_short_of("r0", 13) == 2  # holds 2, needs 4


def test_a_sequence_s_capacity_is_every_slot_of_the_blocks_it_holds() -> None:
    assert _holding(8, {"r0": 5}).capacity("r0") == 2 * BLOCK_SIZE


def test_a_sequence_never_given_room_has_no_capacity() -> None:
    assert BlockTables(num_blocks=8, block_size=BLOCK_SIZE).capacity("new") == 0
