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


def test_schedule_pulls_up_to_max_num_seqs() -> None:
    """With max_num_seqs=3 and 5 waiting, first batch has 3 and 2 remain waiting."""
    sched = _make_scheduler(max_num_seqs=3)
    for i in range(5):
        sched.add(_make_sequence(f"req-{i}"))
    out = sched.schedule()
    assert out.is_new_batch
    assert len(out.scheduled) == 3
    assert [s.request_id for s in out.scheduled] == ["req-0", "req-1", "req-2"]
    assert [s.request_id for s in sched.waiting] == ["req-3", "req-4"]


def test_schedule_does_not_refill_partial_running_batch() -> None:
    """Strict static policy: no new seqs join a batch while it is still running."""
    sched = _make_scheduler(max_num_seqs=4)
    for i in range(2):
        sched.add(_make_sequence(f"req-{i}"))
    sched.schedule()  # 2 running, capacity for 2 more

    sched.add(_make_sequence("req-2"))
    sched.add(_make_sequence("req-3"))

    out = sched.schedule()
    assert not out.is_new_batch
    assert [s.request_id for s in out.scheduled] == ["req-0", "req-1"]
    assert [s.request_id for s in sched.waiting] == ["req-2", "req-3"]


def test_schedule_drains_all_then_starts_new_batch() -> None:
    """After every running seq finishes, scheduler picks a fresh batch from waiting."""
    sched = _make_scheduler(max_num_seqs=2)
    for i in range(4):
        sched.add(_make_sequence(f"req-{i}"))
    sched.schedule()  # batch [0,1]
    for s in sched.running:
        s.status = SequenceStatus.FINISHED_STOPPED
    sched.remove_finished()
    out = sched.schedule()
    assert out.is_new_batch
    assert [s.request_id for s in out.scheduled] == ["req-2", "req-3"]
