"""Compute BenchmarkSummary from raw RequestMeasurement objects.

Wall time is used only to compute rates (tok/s, req/s) and is never stored
in the summary directly.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal

from benchmarks.adapters.base import RequestMeasurement


@dataclass
class ThroughputStats:
    tokens_per_second: float
    requests_per_second: float


@dataclass
class LatencyStats:
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    e2e_p50_ms: float


@dataclass
class BenchmarkSummary:
    throughput: ThroughputStats | None
    latency: LatencyStats | None


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a list of values."""
    if not values:
        raise ValueError("Cannot compute percentile of empty list")
    sorted_vals = sorted(values)
    idx = (p / 100) * (len(sorted_vals) - 1)
    lower = int(idx)
    upper = lower + 1
    if upper >= len(sorted_vals):
        return sorted_vals[lower]
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def _compute_itl_values(measurements: list[RequestMeasurement]) -> list[float]:
    """Return per-request mean ITL in seconds.

    If token_timestamps_s is available, derive ITL from per-token deltas.
    Otherwise use the mean approximation: (e2e - ttft) / (output_token_count - 1).
    """
    itl_values: list[float] = []
    for m in measurements:
        if m.output_token_count <= 1:
            itl_values.append(0.0)
            continue
        if m.token_timestamps_s is not None and len(m.token_timestamps_s) >= 2:
            deltas = [
                m.token_timestamps_s[i] - m.token_timestamps_s[i - 1]
                for i in range(1, len(m.token_timestamps_s))
            ]
            itl_values.append(statistics.mean(deltas))
        else:
            itl_values.append((m.e2e_s - m.ttft_s) / (m.output_token_count - 1))
    return itl_values


def compute_summary(
    measurements: list[RequestMeasurement],
    benchmark_type: Literal["throughput", "latency"],
    wall_time_s: float,
) -> BenchmarkSummary:
    """Compute BenchmarkSummary from raw measurements.

    Returns a BenchmarkSummary with ThroughputStats populated for throughput
    runs and LatencyStats populated for latency runs. Both sub-structs are
    always populated regardless of benchmark_type so the dashboard can show
    latency percentiles even for throughput runs (with the documented caveat
    that TTFT in throughput mode includes scheduling contention).
    """
    ttft_values_s = [m.ttft_s for m in measurements]
    itl_values_s = _compute_itl_values(measurements)
    e2e_values_s = [m.e2e_s for m in measurements]

    latency = LatencyStats(
        ttft_p50_ms=_percentile(ttft_values_s, 50) * 1000,
        ttft_p95_ms=_percentile(ttft_values_s, 95) * 1000,
        ttft_p99_ms=_percentile(ttft_values_s, 99) * 1000,
        itl_p50_ms=_percentile(itl_values_s, 50) * 1000,
        itl_p95_ms=_percentile(itl_values_s, 95) * 1000,
        itl_p99_ms=_percentile(itl_values_s, 99) * 1000,
        e2e_p50_ms=_percentile(e2e_values_s, 50) * 1000,
    )

    throughput: ThroughputStats | None = None
    if benchmark_type == "throughput":
        total_output_tokens = sum(m.output_token_count for m in measurements)
        throughput = ThroughputStats(
            tokens_per_second=total_output_tokens / wall_time_s,
            requests_per_second=len(measurements) / wall_time_s,
        )

    return BenchmarkSummary(throughput=throughput, latency=latency)
