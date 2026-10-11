"""Unit tests for Scheduler — no model, no GPU."""

from __future__ import annotations

import pytest

from liteinfer.cache.block_tables import BlockTables
from liteinfer.config import EngineConfig
from liteinfer.engine.scheduler import Scheduler, SchedulerOutput
from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.sampling.params import SamplingParams


def _make_config(max_num_seqs: int = 4, max_num_batched_tokens: int | None = None) -> EngineConfig:
    return EngineConfig(
        model="unused", max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
    )


def _plenty_of_blocks() -> BlockTables:
    """More blocks than any test here asks for, so only the slot and token caps bind."""
    return BlockTables(num_blocks=1 << 12, block_size=16)


def _scheduler(max_num_seqs: int = 4, max_num_batched_tokens: int | None = None) -> Scheduler:
    return Scheduler(_make_config(max_num_seqs, max_num_batched_tokens), _plenty_of_blocks())


def _make_seq(request_id: str, prompt_len: int = 3) -> Sequence:
    return Sequence(
        request_id=request_id,
        prompt="hello",
        prompt_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=SamplingParams(max_tokens=5),
    )


def _run(out: SchedulerOutput) -> None:
    """Do what the engine does after a step: record the computed tokens, sample where due."""
    for seq in out.seqs:
        seq.num_computed_tokens += out.num_scheduled_tokens[seq.request_id]
        if seq.is_prompt_computed:
            seq.output_token_ids.append(42)


def _ids(out: SchedulerOutput) -> list[str]:
    return [seq.request_id for seq in out.seqs]


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def test_schedule_admits_up_to_max_num_seqs() -> None:
    scheduler = _scheduler(max_num_seqs=2)
    for i in range(3):
        scheduler.add(_make_seq(f"req-{i}"))

    out = scheduler.schedule()

    assert _ids(out) == ["req-0", "req-1"]


def test_schedule_leaves_what_it_did_not_admit_waiting() -> None:
    scheduler = _scheduler(max_num_seqs=2)
    for i in range(3):
        scheduler.add(_make_seq(f"req-{i}"))

    scheduler.schedule()

    assert [seq.request_id for seq in scheduler.waiting] == ["req-2"]


def test_schedule_no_waiting_returns_empty_output() -> None:
    scheduler = _scheduler(max_num_seqs=4)

    assert scheduler.schedule().is_empty


# ---------------------------------------------------------------------------
# What each sequence is granted
# ---------------------------------------------------------------------------


def test_a_new_sequence_is_granted_its_whole_prompt() -> None:
    scheduler = _scheduler(max_num_seqs=4)
    scheduler.add(_make_seq("req-0", prompt_len=7))

    assert scheduler.schedule().num_scheduled_tokens == {"req-0": 7}


def test_a_decoding_sequence_is_granted_one_token() -> None:
    scheduler = _scheduler(max_num_seqs=4)
    scheduler.add(_make_seq("req-0", prompt_len=7))
    _run(scheduler.schedule())

    assert scheduler.schedule().num_scheduled_tokens == {"req-0": 1}


def test_running_sequences_are_scheduled_before_newly_admitted_ones() -> None:
    """One decoding sequence and one new one in the same step: the running one comes first."""
    scheduler = _scheduler(max_num_seqs=4)
    scheduler.add(_make_seq("req-a"))
    _run(scheduler.schedule())
    scheduler.add(_make_seq("req-b"))

    assert _ids(scheduler.schedule()) == ["req-a", "req-b"]


def test_default_budget_never_chunks_a_full_batch_of_the_longest_prompts() -> None:
    """None caps nothing `max_num_seqs x max_model_len` does not, so admission is by slot alone."""
    config = _make_config(max_num_seqs=2)
    scheduler = Scheduler(config, _plenty_of_blocks())
    for i in range(2):
        scheduler.add(_make_seq(f"req-{i}", prompt_len=config.max_model_len - 1))

    out = scheduler.schedule()

    assert list(out.num_scheduled_tokens.values()) == [config.max_model_len - 1] * 2


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def test_a_prompt_longer_than_the_budget_is_granted_the_budget() -> None:
    scheduler = _scheduler(max_num_seqs=2, max_num_batched_tokens=4)
    scheduler.add(_make_seq("req-0", prompt_len=10))

    assert scheduler.schedule().num_scheduled_tokens == {"req-0": 4}


def test_a_chunked_prompt_continues_where_it_stopped() -> None:
    scheduler = _scheduler(max_num_seqs=2, max_num_batched_tokens=4)
    scheduler.add(_make_seq("req-0", prompt_len=10))
    _run(scheduler.schedule())
    _run(scheduler.schedule())

    assert scheduler.schedule().num_scheduled_tokens == {"req-0": 2}


def test_a_chunked_prompt_samples_nothing_until_its_last_chunk() -> None:
    scheduler = _scheduler(max_num_seqs=2, max_num_batched_tokens=4)
    seq = _make_seq("req-0", prompt_len=10)
    scheduler.add(seq)
    _run(scheduler.schedule())

    assert seq.output_token_ids == []


def test_nothing_is_admitted_behind_a_prompt_that_took_the_rest_of_the_budget() -> None:
    """Both caps bind: a free slot is not enough without tokens to spend in it."""
    scheduler = _scheduler(max_num_seqs=4, max_num_batched_tokens=4)
    scheduler.add(_make_seq("req-long", prompt_len=10))
    scheduler.add(_make_seq("req-short", prompt_len=2))

    assert _ids(scheduler.schedule()) == ["req-long"]


def test_the_budget_left_by_decodes_is_what_a_new_prompt_gets() -> None:
    scheduler = _scheduler(max_num_seqs=4, max_num_batched_tokens=4)
    scheduler.add(_make_seq("req-a", prompt_len=2))
    scheduler.add(_make_seq("req-b", prompt_len=2))
    _run(scheduler.schedule())
    scheduler.add(_make_seq("req-c", prompt_len=10))

    out = scheduler.schedule()

    assert out.num_scheduled_tokens == {"req-a": 1, "req-b": 1, "req-c": 2}


def test_what_is_left_of_a_prompt_is_admitted_alongside_others() -> None:
    """Once a chunked prompt's remainder fits, the rest of the budget goes to the next one."""
    scheduler = _scheduler(max_num_seqs=4, max_num_batched_tokens=4)
    scheduler.add(_make_seq("req-long", prompt_len=6))
    scheduler.add(_make_seq("req-short", prompt_len=2))
    _run(scheduler.schedule())

    assert scheduler.schedule().num_scheduled_tokens == {"req-long": 2, "req-short": 2}


def test_a_budget_below_the_slot_count_is_refused() -> None:
    """Below it, a full batch of decodes could not all run in one step."""
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        _make_config(max_num_seqs=4, max_num_batched_tokens=3)


# ---------------------------------------------------------------------------
# Continuous slot-filling
# ---------------------------------------------------------------------------


def test_freed_slot_filled_by_waiting_seq_on_next_schedule() -> None:
    """After one seq finishes, a waiting seq is admitted immediately."""
    scheduler = _scheduler(max_num_seqs=2)
    seq_a = _make_seq("req-a")
    seq_b = _make_seq("req-b")
    seq_c = _make_seq("req-c")
    for s in (seq_a, seq_b, seq_c):
        scheduler.add(s)
    _run(scheduler.schedule())  # admits a, b; c still waiting

    seq_a.status = SequenceStatus.FINISHED_LENGTH
    scheduler.remove_finished()  # frees slot

    assert _ids(scheduler.schedule()) == ["req-b", "req-c"]


def test_multiple_finished_seqs_allow_multiple_admissions() -> None:
    scheduler = _scheduler(max_num_seqs=2)
    seqs = [_make_seq(f"req-{i}") for i in range(4)]
    for s in seqs:
        scheduler.add(s)
    _run(scheduler.schedule())  # admits 0, 1

    seqs[0].status = SequenceStatus.FINISHED_STOPPED
    seqs[1].status = SequenceStatus.FINISHED_STOPPED
    scheduler.remove_finished()

    assert _ids(scheduler.schedule()) == ["req-2", "req-3"]


# ---------------------------------------------------------------------------
# remove_finished — individual eviction
# ---------------------------------------------------------------------------


def test_remove_finished_evicts_only_finished_seqs() -> None:
    scheduler = _scheduler(max_num_seqs=4)
    seq_a = _make_seq("req-a")
    seq_b = _make_seq("req-b")
    scheduler.add(seq_a)
    scheduler.add(seq_b)
    scheduler.schedule()

    seq_a.status = SequenceStatus.FINISHED_STOPPED
    finished = scheduler.remove_finished()

    assert len(finished) == 1
    assert finished[0].request_id == "req-a"
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == "req-b"


def test_remove_finished_returns_empty_when_none_done() -> None:
    scheduler = _scheduler(max_num_seqs=4)
    scheduler.add(_make_seq("req-0"))
    scheduler.schedule()

    assert scheduler.remove_finished() == []


# ---------------------------------------------------------------------------
# has_unfinished
# ---------------------------------------------------------------------------


def test_has_unfinished_true_when_waiting() -> None:
    scheduler = _scheduler()
    scheduler.add(_make_seq("req-0"))
    assert scheduler.has_unfinished()


def test_has_unfinished_true_when_running() -> None:
    scheduler = _scheduler()
    scheduler.add(_make_seq("req-0"))
    scheduler.schedule()
    assert scheduler.has_unfinished()


def test_has_unfinished_false_when_empty() -> None:
    scheduler = _scheduler()
    assert not scheduler.has_unfinished()


def test_has_unfinished_false_after_all_finished_and_removed() -> None:
    scheduler = _scheduler()
    seq = _make_seq("req-0")
    scheduler.add(seq)
    scheduler.schedule()
    seq.status = SequenceStatus.FINISHED_STOPPED
    scheduler.remove_finished()
    assert not scheduler.has_unfinished()


# ---------------------------------------------------------------------------
# KV blocks
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 4


def _tight(num_blocks: int, max_num_seqs: int = 4) -> tuple[Scheduler, BlockTables]:
    """A scheduler over a pool of `num_blocks` blocks of `_BLOCK_SIZE` tokens."""
    blocks = BlockTables(num_blocks, _BLOCK_SIZE)
    return Scheduler(_make_config(max_num_seqs=max_num_seqs), blocks), blocks


def _running(
    scheduler: Scheduler,
    blocks: BlockTables,
    request_id: str,
    prompt_len: int,
    outputs: int = 0,
    computed: int | None = None,
) -> Sequence:
    """A sequence already running, holding blocks for the `computed` tokens it has cached.

    By default that is every token but the newest sampled one, as after its last
    pass, so a sequence with `outputs` owes one token; `computed` short of the
    prompt makes it a prompt part-way through.
    """
    seq = _make_seq(request_id, prompt_len)
    seq.output_token_ids = [42] * outputs
    seq.num_computed_tokens = len(seq) - (1 if outputs else 0) if computed is None else computed
    seq.status = SequenceStatus.RUNNING
    blocks.reserve([request_id], [seq.num_computed_tokens])
    scheduler.running.append(seq)
    return seq


def _refused(out: SchedulerOutput) -> list[str]:
    return [refusal.seq.request_id for refusal in out.refused]


def test_a_prompt_whose_blocks_are_not_free_waits() -> None:
    scheduler, _ = _tight(num_blocks=3)
    scheduler.add(_make_seq("a", prompt_len=5))  # two blocks
    scheduler.add(_make_seq("b", prompt_len=6))  # two more, one left

    assert _ids(scheduler.schedule()) == ["a"]


def test_a_waiting_prompt_is_admitted_once_a_finished_sequence_frees_its_blocks() -> None:
    scheduler, blocks = _tight(num_blocks=3)
    first = _running(scheduler, blocks, "first", prompt_len=5, outputs=1)
    scheduler.add(_make_seq("second", prompt_len=6))
    first.status = SequenceStatus.FINISHED_STOPPED
    scheduler.remove_finished()

    assert _ids(scheduler.schedule()) == ["second"]


def test_nothing_is_admitted_past_a_prompt_waiting_for_blocks() -> None:
    """A short prompt behind it would fit, and taking it would starve the long one."""
    scheduler, _ = _tight(num_blocks=3)
    for request_id, prompt_len in (("a", 5), ("long", 6), ("short", 2)):
        scheduler.add(_make_seq(request_id, prompt_len))

    scheduler.schedule()

    assert [seq.request_id for seq in scheduler.waiting] == ["long", "short"]


def test_a_prompt_no_empty_pool_could_hold_is_refused() -> None:
    scheduler, _ = _tight(num_blocks=2)
    scheduler.add(_make_seq("big", prompt_len=9))  # three blocks of 4

    assert _refused(scheduler.schedule()) == ["big"]


def test_a_refused_prompt_does_not_hold_up_the_prompts_behind_it() -> None:
    scheduler, _ = _tight(num_blocks=2)
    scheduler.add(_make_seq("big", prompt_len=9))
    scheduler.add(_make_seq("small", prompt_len=3))

    assert _ids(scheduler.schedule()) == ["small"]


def test_a_part_way_prompt_takes_a_smaller_chunk_when_blocks_run_short() -> None:
    scheduler, blocks = _tight(num_blocks=3)
    _running(scheduler, blocks, "decoding", prompt_len=3, outputs=1)  # its block has room
    _running(scheduler, blocks, "part-way", prompt_len=12, computed=2)  # one block held, one free

    assert scheduler.schedule().num_scheduled_tokens["part-way"] == 6  # up to the end of block two


def test_a_decode_whose_block_is_full_sits_out_while_others_go_on() -> None:
    scheduler, blocks = _tight(num_blocks=2)
    _running(scheduler, blocks, "full", prompt_len=4, outputs=1)  # its fifth token needs a block
    _running(scheduler, blocks, "room", prompt_len=3, outputs=1)

    assert _ids(scheduler.schedule()) == ["room"]


def test_running_sequences_get_blocks_before_newcomers() -> None:
    scheduler, blocks = _tight(num_blocks=2)
    _running(scheduler, blocks, "decoding", prompt_len=4, outputs=1)  # needs the free block
    scheduler.add(_make_seq("newcomer", prompt_len=3))  # would fit in it too

    assert _ids(scheduler.schedule()) == ["decoding"]


def test_when_no_running_sequence_can_grow_the_newest_is_refused() -> None:
    """Nothing would ever free a block; refusing the least invested lets the rest go on."""
    scheduler, blocks = _tight(num_blocks=2)
    _running(scheduler, blocks, "older", prompt_len=4, outputs=1)
    _running(scheduler, blocks, "newer", prompt_len=4, outputs=1)

    assert _refused(scheduler.schedule()) == ["newer"]


def test_a_sequence_that_sits_out_is_not_refused_while_another_goes_on() -> None:
    scheduler, blocks = _tight(num_blocks=2)
    _running(scheduler, blocks, "full", prompt_len=4, outputs=1)
    _running(scheduler, blocks, "room", prompt_len=3, outputs=1)

    assert _refused(scheduler.schedule()) == []


def test_remove_finished_frees_the_blocks_of_what_it_evicts() -> None:
    scheduler, blocks = _tight(num_blocks=3)
    done = _running(scheduler, blocks, "done", prompt_len=5, outputs=1)
    done.status = SequenceStatus.FINISHED_STOPPED

    scheduler.remove_finished()

    assert blocks.num_free_blocks == 3

