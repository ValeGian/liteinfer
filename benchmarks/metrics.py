"""Aggregate `GenerationResult`s into headline metrics.

Engine-agnostic: any runner implementing `EngineRunner` can be summarized
with `summarize()`.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from benchmarks.runners.base import GenerationResult


@dataclass
class BenchmarkMetrics:
    engine: str
    batch_size: int
    num_requests: int
    output_tokens: int
    wall_time_s: float

    requests_per_second: float
    output_tokens_per_second: float

    ttft_p50_s: float
    ttft_p99_s: float

    e2e_latency_p50_s: float
    e2e_latency_p99_s: float

    peak_memory_bytes: int | None

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(
    engine: str,
    results: list[GenerationResult],
    wall_time_s: float,
    peak_memory_bytes: int | None = None,
    batch_size: int = 1,
) -> BenchmarkMetrics:
    output_tokens = sum(len(r.output_token_ids) for r in results)
    ttfts = sorted(r.ttft_s for r in results)
    e2e = sorted(r.total_time_s for r in results)
    return BenchmarkMetrics(
        engine=engine,
        batch_size=batch_size,
        num_requests=len(results),
        output_tokens=output_tokens,
        wall_time_s=wall_time_s,
        requests_per_second=len(results) / wall_time_s if wall_time_s > 0 else 0.0,
        output_tokens_per_second=output_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        ttft_p50_s=statistics.median(ttfts) if ttfts else 0.0,
        ttft_p99_s=_percentile(ttfts, 0.99),
        e2e_latency_p50_s=statistics.median(e2e) if e2e else 0.0,
        e2e_latency_p99_s=_percentile(e2e, 0.99),
        peak_memory_bytes=peak_memory_bytes,
    )


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1)))
    return sorted_values[idx]
