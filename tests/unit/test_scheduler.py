"""Unit tests for the scheduler under the static-batching policy."""

from __future__ import annotations

from liteinfer.config import EngineConfig
from liteinfer.engine.scheduler import Scheduler
from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.sampling.params import SamplingParams


def _make_sequence(request_id: str) -> Sequence:
    return Sequence(
        request_id=request_id,
        prompt="",
        prompt_token_ids=[0],
        sampling_params=SamplingParams(),
    )


def _make_scheduler(max_num_seqs: int = 1) -> Scheduler:
    cfg = EngineConfig(model="dummy", max_num_seqs=max_num_seqs, device="cpu")
    return Scheduler(cfg)


def test_schedule_returns_empty_when_no_requests() -> None:
    sched = _make_scheduler()
    out = sched.schedule()
    assert out.scheduled == []
    assert not out.is_new_batch


def test_schedule_promotes_one_waiting_to_running() -> None:
    sched = _make_scheduler()
    sched.add(_make_sequence("req-1"))
    out = sched.schedule()
    assert len(out.scheduled) == 1
    assert out.is_new_batch
    assert out.scheduled[0].status == SequenceStatus.RUNNING


def test_schedule_does_not_pull_new_batch_while_running_busy() -> None:
    sched = _make_scheduler()
    sched.add(_make_sequence("req-1"))
    sched.add(_make_sequence("req-2"))
    out_a = sched.schedule()
    out_b = sched.schedule()
    assert out_a.is_new_batch
    assert not out_b.is_new_batch
    assert out_a.scheduled == out_b.scheduled
    assert sched.waiting  # second one still waiting


def test_remove_finished_drains_running() -> None:
    sched = _make_scheduler()
    sched.add(_make_sequence("req-1"))
    out = sched.schedule()
    out.scheduled[0].status = SequenceStatus.FINISHED_STOPPED
    finished = sched.remove_finished()
    assert len(finished) == 1
    assert sched.running == []


def test_next_batch_starts_after_drain() -> None:
    sched = _make_scheduler()
    sched.add(_make_sequence("req-1"))
    sched.add(_make_sequence("req-2"))
    sched.schedule()
    sched.running[0].status = SequenceStatus.FINISHED_STOPPED
    sched.remove_finished()
    out = sched.schedule()
    assert out.is_new_batch
    assert out.scheduled[0].request_id == "req-2"
