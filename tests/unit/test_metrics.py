"""Unit tests for `engine.metrics`."""

from __future__ import annotations

import pytest

from liteinfer.engine.metrics import EngineStats, Phase, StepMetrics


def _step(step_idx: int, input_tokens: int, prompt_tokens: int, wall_time_s: float) -> StepMetrics:
    return StepMetrics(
        step_idx=step_idx,
        num_seqs=1,
        input_tokens=input_tokens,
        prompt_tokens=prompt_tokens,
        new_tokens=1,
        wall_time_s=wall_time_s,
    )


def test_step_throughput_handles_zero_wall_time() -> None:
    s = _step(0, input_tokens=10, prompt_tokens=10, wall_time_s=0.0)
    assert s.throughput_tokens_per_s == 0.0
    assert s.prefill_throughput_tokens_per_s == 0.0


@pytest.mark.parametrize(
    ("input_tokens", "prompt_tokens", "phase"),
    [(20, 20, Phase.PREFILL), (4, 0, Phase.DECODE), (24, 20, Phase.MIXED)],
)
def test_a_step_is_named_by_the_tokens_it_carried(
    input_tokens: int, prompt_tokens: int, phase: Phase
) -> None:
    assert _step(0, input_tokens, prompt_tokens, 0.1).phase is phase


def test_engine_stats_accumulates_phases() -> None:
    stats = EngineStats()
    stats.record(_step(0, input_tokens=20, prompt_tokens=20, wall_time_s=0.5))
    stats.record(_step(1, input_tokens=1, prompt_tokens=0, wall_time_s=0.1))
    stats.record(_step(2, input_tokens=1, prompt_tokens=0, wall_time_s=0.1))

    assert stats.total_input_tokens == 22
    assert stats.total_new_tokens == 3
    assert stats.total_prefill_input_tokens == 20
    assert stats.total_prefill_wall_s == 0.5
    assert stats.total_decode_wall_s == 0.2
    assert stats.avg_prefill_throughput_tokens_per_s == 40.0
    assert stats.avg_decode_throughput_tokens_per_s == 10.0


def test_a_mixed_step_counts_toward_neither_phase_total() -> None:
    """Its wall time belongs to both, and splitting it would invent a number."""
    stats = EngineStats()
    stats.record(_step(0, input_tokens=24, prompt_tokens=20, wall_time_s=0.5))

    assert (stats.total_prefill_wall_s, stats.total_decode_wall_s) == (0.0, 0.0)


def test_engine_stats_listeners_fire_synchronously() -> None:
    stats = EngineStats()
    seen: list[int] = []
    stats.on_step(lambda s: seen.append(s.step_idx))
    stats.record(_step(0, input_tokens=10, prompt_tokens=10, wall_time_s=0.1))
    stats.record(_step(1, input_tokens=1, prompt_tokens=0, wall_time_s=0.05))
    assert seen == [0, 1]
