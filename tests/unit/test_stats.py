"""Tests for benchmarks.stats module."""

from __future__ import annotations

from benchmarks.adapters.base import RequestMeasurement
from benchmarks.stats import compute_summary


def _make_measurement(
    sample_index: int = 0,
    input_token_count: int = 128,
    output_token_count: int = 256,
    ttft_s: float = 0.1,
    token_timestamps_s: list[float] | None = None,
    e2e_s: float = 2.0,
) -> RequestMeasurement:
    return RequestMeasurement(
        sample_index=sample_index,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
        ttft_s=ttft_s,
        token_timestamps_s=token_timestamps_s,
        e2e_s=e2e_s,
    )


def test_compute_summary_throughput_stats():
    # 10 measurements, each with output_token_count=100, wall_time_s=10 → 100 tok/s, 1 req/s
    measurements = [
        _make_measurement(sample_index=i, output_token_count=100) for i in range(10)
    ]
    summary = compute_summary(measurements, "throughput", wall_time_s=10.0)

    assert summary.throughput is not None
    assert abs(summary.throughput.tokens_per_second - 100.0) < 1e-6
    assert abs(summary.throughput.requests_per_second - 1.0) < 1e-6


def test_compute_summary_latency_percentiles():
    # TTFT values: 0.1, 0.2, ..., 1.0 seconds
    ttft_values = [0.1 * (i + 1) for i in range(10)]
    measurements = [
        _make_measurement(sample_index=i, ttft_s=ttft_values[i], e2e_s=ttft_values[i] + 1.0)
        for i in range(10)
    ]
    summary = compute_summary(measurements, "latency", wall_time_s=20.0)

    assert summary.latency is not None
    # p50 of 10 values [0.1..1.0] → median at index 4.5 = (0.5 + 0.6) / 2 = 0.55 s = 550 ms
    assert abs(summary.latency.ttft_p50_ms - 550.0) < 1.0
    # p95 → index 8.55 = 0.9 + 0.55*(1.0-0.9) = 0.955 s = 955 ms
    assert summary.latency.ttft_p95_ms > 900.0
    # p99 → very close to max (1.0 s = 1000 ms)
    assert summary.latency.ttft_p99_ms > 950.0


def test_itl_derived_from_e2e_when_no_token_timestamps():
    # output_token_count=11, ttft=0.1s, e2e=1.1s → ITL = (1.1 - 0.1) / 10 = 0.1s = 100ms
    m = _make_measurement(output_token_count=11, ttft_s=0.1, e2e_s=1.1, token_timestamps_s=None)
    summary = compute_summary([m], "latency", wall_time_s=1.1)

    assert summary.latency is not None
    assert abs(summary.latency.itl_p50_ms - 100.0) < 1e-6


def test_itl_from_token_timestamps_when_available():
    # 5 tokens at t=0.1, 0.2, 0.3, 0.4, 0.5 → deltas = 0.1s each → mean ITL = 0.1s = 100ms
    timestamps = [0.1, 0.2, 0.3, 0.4, 0.5]
    m = _make_measurement(
        output_token_count=5,
        ttft_s=0.1,
        e2e_s=0.5,
        token_timestamps_s=timestamps,
    )
    summary = compute_summary([m], "latency", wall_time_s=0.5)

    assert summary.latency is not None
    assert abs(summary.latency.itl_p50_ms - 100.0) < 1e-6


def test_stats_independent_of_sample_count():
    """compute_summary is deterministic: same inputs → same outputs."""
    measurements = [
        _make_measurement(sample_index=i, ttft_s=0.05 * (i + 1), output_token_count=100)
        for i in range(10)
    ]
    summary_a = compute_summary(measurements, "throughput", wall_time_s=5.0)
    summary_b = compute_summary(measurements, "throughput", wall_time_s=5.0)

    assert summary_a.throughput is not None
    assert summary_b.throughput is not None
    assert summary_a.throughput.tokens_per_second == summary_b.throughput.tokens_per_second
    assert summary_a.latency is not None
    assert summary_b.latency is not None
    assert summary_a.latency.ttft_p50_ms == summary_b.latency.ttft_p50_ms
    assert summary_a.latency.itl_p50_ms == summary_b.latency.itl_p50_ms
