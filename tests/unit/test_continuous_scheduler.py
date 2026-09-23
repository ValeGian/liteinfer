"""Unit tests for ContinuousScheduler — no model, no GPU."""

from __future__ import annotations

import pytest

from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_scheduler import ContinuousScheduler, ContinuousSchedulerOutput
from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.sampling.params import SamplingParams


def _make_config(max_num_seqs: int = 4, max_num_batched_tokens: int | None = None) -> EngineConfig:
    return EngineConfig(
        model="unused", max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
    )


def _make_seq(request_id: str, prompt_len: int = 3) -> Sequence:
    return Sequence(
        request_id=request_id,
        prompt="hello",
        prompt_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=SamplingParams(max_tokens=5),
    )


def _run(out: ContinuousSchedulerOutput) -> None:
    """Do what the engine does after a step: record the computed tokens, sample where due."""
    for seq in out.seqs:
        seq.num_computed_tokens += out.num_scheduled_tokens[seq.request_id]
        if seq.is_prompt_computed:
            seq.output_token_ids.append(42)


def _ids(out: ContinuousSchedulerOutput) -> list[str]:
    return [seq.request_id for seq in out.seqs]


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def test_schedule_admits_up_to_max_num_seqs() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=2))
    for i in range(3):
        scheduler.add(_make_seq(f"req-{i}"))

    out = scheduler.schedule()

    assert _ids(out) == ["req-0", "req-1"]


def test_schedule_leaves_what_it_did_not_admit_waiting() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=2))
    for i in range(3):
        scheduler.add(_make_seq(f"req-{i}"))

    scheduler.schedule()

    assert [seq.request_id for seq in scheduler.waiting] == ["req-2"]


def test_schedule_no_waiting_returns_empty_output() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4))

    assert scheduler.schedule().is_empty


# ---------------------------------------------------------------------------
# What each sequence is granted
# ---------------------------------------------------------------------------


def test_a_new_sequence_is_granted_its_whole_prompt() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4))
    scheduler.add(_make_seq("req-0", prompt_len=7))

    assert scheduler.schedule().num_scheduled_tokens == {"req-0": 7}


def test_a_decoding_sequence_is_granted_one_token() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4))
    scheduler.add(_make_seq("req-0", prompt_len=7))
    _run(scheduler.schedule())

    assert scheduler.schedule().num_scheduled_tokens == {"req-0": 1}


def test_running_sequences_are_scheduled_before_newly_admitted_ones() -> None:
    """One decoding sequence and one new one in the same step: the running one comes first."""
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4))
    scheduler.add(_make_seq("req-a"))
    _run(scheduler.schedule())
    scheduler.add(_make_seq("req-b"))

    assert _ids(scheduler.schedule()) == ["req-a", "req-b"]


def test_default_budget_never_chunks_a_full_batch_of_the_longest_prompts() -> None:
    """None caps nothing `max_num_seqs x max_model_len` does not, so admission is by slot alone."""
    config = _make_config(max_num_seqs=2)
    scheduler = ContinuousScheduler(config)
    for i in range(2):
        scheduler.add(_make_seq(f"req-{i}", prompt_len=config.max_model_len - 1))

    out = scheduler.schedule()

    assert list(out.num_scheduled_tokens.values()) == [config.max_model_len - 1] * 2


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def test_a_prompt_longer_than_the_budget_is_granted_the_budget() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=2, max_num_batched_tokens=4))
    scheduler.add(_make_seq("req-0", prompt_len=10))

    assert scheduler.schedule().num_scheduled_tokens == {"req-0": 4}


def test_a_chunked_prompt_continues_where_it_stopped() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=2, max_num_batched_tokens=4))
    scheduler.add(_make_seq("req-0", prompt_len=10))
    _run(scheduler.schedule())
    _run(scheduler.schedule())

    assert scheduler.schedule().num_scheduled_tokens == {"req-0": 2}


def test_a_chunked_prompt_samples_nothing_until_its_last_chunk() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=2, max_num_batched_tokens=4))
    seq = _make_seq("req-0", prompt_len=10)
    scheduler.add(seq)
    _run(scheduler.schedule())

    assert seq.output_token_ids == []


def test_nothing_is_admitted_behind_a_prompt_that_took_the_rest_of_the_budget() -> None:
    """Both caps bind: a free slot is not enough without tokens to spend in it."""
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4, max_num_batched_tokens=4))
    scheduler.add(_make_seq("req-long", prompt_len=10))
    scheduler.add(_make_seq("req-short", prompt_len=2))

    assert _ids(scheduler.schedule()) == ["req-long"]


def test_the_budget_left_by_decodes_is_what_a_new_prompt_gets() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4, max_num_batched_tokens=4))
    scheduler.add(_make_seq("req-a", prompt_len=2))
    scheduler.add(_make_seq("req-b", prompt_len=2))
    _run(scheduler.schedule())
    scheduler.add(_make_seq("req-c", prompt_len=10))

    out = scheduler.schedule()

    assert out.num_scheduled_tokens == {"req-a": 1, "req-b": 1, "req-c": 2}


def test_what_is_left_of_a_prompt_is_admitted_alongside_others() -> None:
    """Once a chunked prompt's remainder fits, the rest of the budget goes to the next one."""
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4, max_num_batched_tokens=4))
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
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=2))
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
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=2))
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
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4))
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
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4))
    scheduler.add(_make_seq("req-0"))
    scheduler.schedule()

    assert scheduler.remove_finished() == []


# ---------------------------------------------------------------------------
# has_unfinished
# ---------------------------------------------------------------------------


def test_has_unfinished_true_when_waiting() -> None:
    scheduler = ContinuousScheduler(_make_config())
    scheduler.add(_make_seq("req-0"))
    assert scheduler.has_unfinished()


def test_has_unfinished_true_when_running() -> None:
    scheduler = ContinuousScheduler(_make_config())
    scheduler.add(_make_seq("req-0"))
    scheduler.schedule()
    assert scheduler.has_unfinished()


def test_has_unfinished_false_when_empty() -> None:
    scheduler = ContinuousScheduler(_make_config())
    assert not scheduler.has_unfinished()


def test_has_unfinished_false_after_all_finished_and_removed() -> None:
    scheduler = ContinuousScheduler(_make_config())
    seq = _make_seq("req-0")
    scheduler.add(seq)
    scheduler.schedule()
    seq.status = SequenceStatus.FINISHED_STOPPED
    scheduler.remove_finished()
    assert not scheduler.has_unfinished()
