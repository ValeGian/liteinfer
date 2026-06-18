"""Tests for benchmarks.dashboard.builder module."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.dashboard.builder import build_comparison_dashboard, build_main_dashboard

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    run_id: str,
    engine: str,
    benchmark_type: str,
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    isl: int = 128,
    osl: int = 256,
    tag: str = "",
    tokens_per_second: float = 400.0,
    ttft_p50_ms: float = 100.0,
    ttft_p99_ms: float = 150.0,
) -> dict:
    summary = {
        "ttft_p50_ms": ttft_p50_ms,
        "ttft_p95_ms": ttft_p50_ms * 1.2,
        "ttft_p99_ms": ttft_p99_ms,
        "itl_p50_ms": 8.0,
        "itl_p95_ms": 10.0,
        "itl_p99_ms": 12.0,
        "e2e_p50_ms": 2000.0,
    }
    if benchmark_type == "throughput":
        summary["tokens_per_second"] = tokens_per_second
        summary["requests_per_second"] = tokens_per_second / 256

    return {
        "run_id": run_id,
        "timestamp": "2026-05-18T14:30:00",
        "tag": tag,
        "engine": engine,
        "benchmark_type": benchmark_type,
        "model": model,
        "dataset": {
            "path": "benchmarks/datasets/test.jsonl",
            "num_samples": 100,
            "target_isl": isl,
            "target_osl": osl,
            "prompt_batch_sha256": "abc123",
        },
        "summary": summary,
        "raw": [],
    }


def _write_results(results_dir: Path, results: list[dict]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        path = results_dir / f"{r['run_id']}.json"
        path.write_text(json.dumps(r))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_comparison_dashboard_groups_by_benchmark_type(tmp_path):
    results = [
        _make_result("run1", "liteinfer", "throughput"),
        _make_result("run2", "vllm", "throughput"),
        _make_result("run3", "liteinfer", "latency"),
    ]
    results_dir = tmp_path / "results"
    _write_results(results_dir, results)

    output = tmp_path / "out.html"
    build_comparison_dashboard(["run1", "run2", "run3"], results_dir, output)

    html = output.read_text()
    # Two distinct tab groups: one throughput, one latency
    assert html.count("throughput") >= 1
    assert html.count("latency") >= 1
    # Both engines appear in output
    assert "liteinfer" in html
    assert "vllm" in html


def test_build_comparison_dashboard_marks_best_cell(tmp_path):
    # liteinfer has lower TTFT (better in latency mode)
    results = [
        _make_result("run1", "liteinfer", "latency", ttft_p50_ms=80.0),
        _make_result("run2", "vllm", "latency", ttft_p50_ms=120.0),
    ]
    results_dir = tmp_path / "results"
    _write_results(results_dir, results)

    output = tmp_path / "out.html"
    build_comparison_dashboard(["run1", "run2"], results_dir, output)

    html = output.read_text()
    # The best cell should have the "best" CSS class
    assert 'class="best"' in html


def test_build_main_dashboard_includes_only_pinned(tmp_path):
    results = [
        _make_result("run1", "liteinfer", "throughput"),
        _make_result("run2", "vllm", "throughput"),
        _make_result("run3", "trtllm", "throughput"),
    ]
    results_dir = tmp_path / "results"
    _write_results(results_dir, results)

    # Pin only 2 results
    main_json = results_dir / "main.json"
    main_json.write_text(json.dumps({"pinned": ["run1", "run2"]}))

    output = tmp_path / "index.html"
    build_main_dashboard(results_dir=results_dir, output_path=output)

    html = output.read_text()
    # Pinned engines appear
    assert "liteinfer" in html
    assert "vllm" in html
    # Unpinned engines do NOT appear
    assert "trtllm" not in html


def test_throughput_tab_contains_ttft_warning(tmp_path):
    results = [_make_result("run1", "liteinfer", "throughput")]
    results_dir = tmp_path / "results"
    _write_results(results_dir, results)

    output = tmp_path / "out.html"
    build_comparison_dashboard(["run1"], results_dir, output)

    html = output.read_text()
    assert "scheduling contention" in html
    assert "not comparable" in html


def test_build_main_dashboard_creates_output_file(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "main.json").write_text(json.dumps({"pinned": []}))

    output = tmp_path / "docs" / "index.html"
    build_main_dashboard(results_dir=results_dir, output_path=output)

    assert output.exists()
    html = output.read_text()
    assert "<!DOCTYPE html>" in html


def test_html_is_self_contained(tmp_path):
    """Generated HTML must contain no CDN links or external resource fetches."""
    results = [_make_result("run1", "liteinfer", "throughput")]
    results_dir = tmp_path / "results"
    _write_results(results_dir, results)

    output = tmp_path / "out.html"
    build_comparison_dashboard(["run1"], results_dir, output)

    html = output.read_text()
    # No CDN links
    assert "cdn.jsdelivr.net" not in html
    assert "unpkg.com" not in html
    assert "googleapis.com" not in html
    assert 'src="http' not in html
    assert 'href="http' not in html
