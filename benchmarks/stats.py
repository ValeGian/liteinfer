"""Summary statistics.

The two modes report disjoint metric sets, on purpose:

* ``latency`` runs one request at a time, so TTFT / ITL / E2E are properties of
  the engine and nothing else.
* ``throughput`` offers every request at once, so tok/s and req/s measure
  capacity. Per-request latencies under that regime are dominated by queue
  position rather than by the engine, so they are not reported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (p / 100) * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


@dataclass(frozen=True)
class ThroughputSummary:
    output_tokens_per_s: float
    requests_per_s: float
    wall_time_s: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LatencySummary:
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    e2e_p50_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def throughput_summary(
    output_tokens: int, num_requests: int, wall_time_s: float
) -> ThroughputSummary:
    return ThroughputSummary(
        output_tokens_per_s=output_tokens / wall_time_s,
        requests_per_s=num_requests / wall_time_s,
        wall_time_s=wall_time_s,
    )


def latency_summary(ttft_s: list[float], e2e_s: list[float], osl: int) -> LatencySummary:
    """Build latency percentiles.

    Inter-token latency is derived rather than measured: with exactly one
    request in flight and a forced output length, ``(e2e - ttft) / (osl - 1)``
    is the mean decode-step cost, and needs no per-token instrumentation that
    the two engines would implement differently.
    """
    if osl < 2:
        raise ValueError("latency mode needs target_osl >= 2 to derive ITL")
    itl_s = [(e2e - ttft) / (osl - 1) for ttft, e2e in zip(ttft_s, e2e_s, strict=True)]
    return LatencySummary(
        ttft_p50_ms=percentile(ttft_s, 50) * 1000,
        ttft_p95_ms=percentile(ttft_s, 95) * 1000,
        ttft_p99_ms=percentile(ttft_s, 99) * 1000,
        itl_p50_ms=percentile(itl_s, 50) * 1000,
        itl_p95_ms=percentile(itl_s, 95) * 1000,
        itl_p99_ms=percentile(itl_s, 99) * 1000,
        e2e_p50_ms=percentile(e2e_s, 50) * 1000,
    )
