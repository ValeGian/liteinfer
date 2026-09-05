"""Unit tests for ContinuousScheduler — no model, no GPU."""

from __future__ import annotations

from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_scheduler import ContinuousScheduler
from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.sampling.params import SamplingParams


def _make_config(max_num_seqs: int = 4) -> EngineConfig:
    return EngineConfig(model="unused", max_num_seqs=max_num_seqs)


def _make_seq(request_id: str) -> Sequence:
    return Sequence(
        request_id=request_id,
        prompt="hello",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=5),
    )


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def test_schedule_admits_up_to_max_num_seqs() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=2))
    for i in range(3):
        scheduler.add(_make_seq(f"req-{i}"))

    out = scheduler.schedule()

    assert len(out.prefill_seqs) == 2
    assert len(scheduler.waiting) == 1


def test_schedule_does_not_over_admit_beyond_max_num_seqs() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=2))
    for i in range(5):
        scheduler.add(_make_seq(f"req-{i}"))

    out = scheduler.schedule()

    assert len(out.prefill_seqs) + len(out.decode_seqs) <= 2


def test_schedule_no_waiting_returns_empty_prefill() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4))

    out = scheduler.schedule()

    assert out.prefill_seqs == []
    assert out.decode_seqs == []


# ---------------------------------------------------------------------------
# Prefill vs decode classification
# ---------------------------------------------------------------------------


def test_newly_added_seqs_appear_in_prefill_seqs() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4))
    scheduler.add(_make_seq("req-0"))

    out = scheduler.schedule()

    assert len(out.prefill_seqs) == 1
    assert out.prefill_seqs[0].request_id == "req-0"
    assert out.decode_seqs == []


def test_seqs_with_output_tokens_appear_in_decode_seqs() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4))
    seq = _make_seq("req-0")
    scheduler.add(seq)
    scheduler.schedule()

    # Simulate prefill having produced one token.
    seq.output_token_ids.append(42)

    out = scheduler.schedule()

    assert out.prefill_seqs == []
    assert len(out.decode_seqs) == 1
    assert out.decode_seqs[0].request_id == "req-0"


def test_mixed_batch_splits_correctly() -> None:
    """One new seq (prefill) and one already-decoded seq (decode) in same step."""
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=4))
    seq_a = _make_seq("req-a")
    scheduler.add(seq_a)
    scheduler.schedule()
    seq_a.output_token_ids.append(10)  # seq_a is now in decode phase

    seq_b = _make_seq("req-b")
    scheduler.add(seq_b)

    out = scheduler.schedule()

    assert len(out.prefill_seqs) == 1
    assert out.prefill_seqs[0].request_id == "req-b"
    assert len(out.decode_seqs) == 1
    assert out.decode_seqs[0].request_id == "req-a"


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

    scheduler.schedule()  # admits a, b; c still waiting
    assert len(scheduler.waiting) == 1

    # Simulate prefill having produced one token for both a and b.
    seq_a.output_token_ids.append(10)
    seq_b.output_token_ids.append(11)

    seq_a.status = SequenceStatus.FINISHED_LENGTH
    scheduler.remove_finished()  # frees slot

    out = scheduler.schedule()  # should admit c into prefill, b stays in decode

    assert len(out.prefill_seqs) == 1
    assert out.prefill_seqs[0].request_id == "req-c"
    assert len(out.decode_seqs) == 1
    assert out.decode_seqs[0].request_id == "req-b"


def test_multiple_finished_seqs_allow_multiple_admissions() -> None:
    scheduler = ContinuousScheduler(_make_config(max_num_seqs=2))
    seqs = [_make_seq(f"req-{i}") for i in range(4)]
    for s in seqs:
        scheduler.add(s)

    scheduler.schedule()  # admits 0, 1

    seqs[0].status = SequenceStatus.FINISHED_STOPPED
    seqs[1].status = SequenceStatus.FINISHED_STOPPED
    scheduler.remove_finished()

    out = scheduler.schedule()  # should admit 2 and 3

    assert len(out.prefill_seqs) == 2


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
