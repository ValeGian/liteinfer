"""Summary statistics."""

from __future__ import annotations

import pytest

from benchmarks.stats import latency_summary, percentile, throughput_summary


def test_percentile_interpolates_between_neighbours() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_percentile_of_a_single_value_is_that_value() -> None:
    assert percentile([7.0], 99) == 7.0


def test_percentile_rejects_an_empty_series() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


def test_throughput_divides_output_tokens_by_wall_time() -> None:
    assert (
        throughput_summary(output_tokens=1000, num_requests=10, wall_time_s=2.0).output_tokens_per_s
        == 500.0
    )


def test_throughput_divides_requests_by_wall_time() -> None:
    assert (
        throughput_summary(output_tokens=1000, num_requests=10, wall_time_s=2.0).requests_per_s
        == 5.0
    )


def test_itl_is_the_decode_time_spread_over_the_decode_steps() -> None:
    # 1 s prefill, 2 s total, 11 tokens -> 1 s spread over 10 decode steps.
    summary = latency_summary(ttft_s=[1.0], e2e_s=[2.0], osl=11)
    assert summary.itl_p50_ms == pytest.approx(100.0)


def test_ttft_is_reported_in_milliseconds() -> None:
    assert latency_summary(ttft_s=[0.25], e2e_s=[1.0], osl=2).ttft_p50_ms == pytest.approx(250.0)


def test_latency_needs_at_least_two_output_tokens() -> None:
    with pytest.raises(ValueError, match="target_osl >= 2"):
        latency_summary(ttft_s=[1.0], e2e_s=[2.0], osl=1)


def test_latency_rejects_mismatched_series() -> None:
    with pytest.raises(ValueError):
        latency_summary(ttft_s=[1.0, 2.0], e2e_s=[2.0], osl=4)
