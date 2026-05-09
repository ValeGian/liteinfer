"""Unit tests for benchmarks/metrics.py."""

from __future__ import annotations

import pytest

from benchmarks.metrics import summarize
from benchmarks.runners.base import GenerationResult


def _make_result(ttft: float, total: float, num_tokens: int = 10) -> GenerationResult:
    return GenerationResult(
        prompt="p",
        output_text="o",
        output_token_ids=list(range(num_tokens)),
        ttft_s=ttft,
        total_time_s=total,
    )


def test_summarize_throughput_metrics() -> None:
    results = [_make_result(0.1, 1.0), _make_result(0.2, 2.0)]
    m = summarize("engine", results, wall_time_s=2.0)
    assert m.num_requests == 2
    assert m.output_tokens == 20
    assert m.requests_per_second == pytest_approx(1.0)
    assert m.output_tokens_per_second == pytest_approx(10.0)


def test_summarize_ttft_percentiles() -> None:
    results = [_make_result(float(i) / 100, 1.0) for i in range(1, 101)]
    m = summarize("engine", results, wall_time_s=10.0)
    assert m.ttft_p50_s == pytest_approx(0.5, abs=0.01)
    assert m.ttft_p99_s == pytest_approx(0.99, abs=0.01)  # round(0.99 * 99) = 98 → index 98


def test_summarize_e2e_latency_populated() -> None:
    results = [_make_result(0.05, 0.5), _make_result(0.1, 1.5)]
    m = summarize("engine", results, wall_time_s=2.0)
    assert m.e2e_latency_p50_s == pytest_approx(1.0)  # median of [0.5, 1.5]
    assert m.e2e_latency_p99_s == pytest_approx(1.5, abs=0.01)


def test_summarize_empty_results_returns_zeros() -> None:
    m = summarize("engine", [], wall_time_s=1.0)
    assert m.ttft_p50_s == 0.0
    assert m.ttft_p99_s == 0.0
    assert m.e2e_latency_p50_s == 0.0
    assert m.e2e_latency_p99_s == 0.0
    assert m.requests_per_second == 0.0


def test_summarize_zero_wall_time_does_not_divide_by_zero() -> None:
    results = [_make_result(0.1, 0.5)]
    m = summarize("engine", results, wall_time_s=0.0)
    assert m.requests_per_second == 0.0
    assert m.output_tokens_per_second == 0.0


def test_summarize_peak_memory_passed_through() -> None:
    results = [_make_result(0.1, 0.5)]
    m = summarize("engine", results, wall_time_s=1.0, peak_memory_bytes=1024)
    assert m.peak_memory_bytes == 1024


def test_as_dict_includes_e2e_fields() -> None:
    results = [_make_result(0.05, 0.5)]
    m = summarize("engine", results, wall_time_s=1.0)
    d = m.as_dict()
    assert "e2e_latency_p50_s" in d
    assert "e2e_latency_p99_s" in d


def test_summarize_batch_size_defaults_to_one() -> None:
    m = summarize("engine", [_make_result(0.1, 0.5)], wall_time_s=1.0)
    assert m.batch_size == 1


def test_summarize_batch_size_passed_through() -> None:
    m = summarize("engine", [_make_result(0.1, 0.5)], wall_time_s=1.0, batch_size=4)
    assert m.batch_size == 4
    assert m.as_dict()["batch_size"] == 4


pytest_approx = pytest.approx
