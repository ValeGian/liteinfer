"""Unit tests for `engine.metrics`."""

from __future__ import annotations

from liteinfer.engine.metrics import EngineStats, Phase, StepMetrics


def test_step_throughput_handles_zero_wall_time() -> None:
    s = StepMetrics(
        step_idx=0,
        phase=Phase.PREFILL,
        num_seqs=1,
        input_tokens=10,
        new_tokens=1,
        wall_time_s=0.0,
    )
    assert s.throughput_tokens_per_s == 0.0
    assert s.prefill_throughput_tokens_per_s == 0.0


def test_engine_stats_accumulates_phases() -> None:
    stats = EngineStats()
    stats.record(StepMetrics(0, Phase.PREFILL, 1, input_tokens=20, new_tokens=1, wall_time_s=0.5))
    stats.record(StepMetrics(1, Phase.DECODE, 1, input_tokens=1, new_tokens=1, wall_time_s=0.1))
    stats.record(StepMetrics(2, Phase.DECODE, 1, input_tokens=1, new_tokens=1, wall_time_s=0.1))

    assert stats.total_input_tokens == 22
    assert stats.total_new_tokens == 3
    assert stats.total_prefill_input_tokens == 20
    assert stats.total_prefill_wall_s == 0.5
    assert stats.total_decode_wall_s == 0.2
    assert stats.avg_prefill_throughput_tokens_per_s == 40.0
    assert stats.avg_decode_throughput_tokens_per_s == 10.0


def test_engine_stats_listeners_fire_synchronously() -> None:
    stats = EngineStats()
    seen: list[int] = []
    stats.on_step(lambda s: seen.append(s.step_idx))
    stats.record(StepMetrics(0, Phase.PREFILL, 1, 10, 1, 0.1))
    stats.record(StepMetrics(1, Phase.DECODE, 1, 1, 1, 0.05))
    assert seen == [0, 1]


def test_recompute_phase_does_not_inflate_prefill_avg() -> None:
    stats = EngineStats()
    stats.record(StepMetrics(0, Phase.RECOMPUTE, 1, input_tokens=50, new_tokens=1, wall_time_s=0.5))
    assert stats.total_prefill_input_tokens == 0
    assert stats.avg_prefill_throughput_tokens_per_s == 0.0
