"""Unit tests for benchmarks/dashboard.py."""

from __future__ import annotations

import json

from benchmarks.dashboard import _delta_class, _fmt, build_html, load_history

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_metrics(**overrides) -> dict:
    base = {
        "engine": "liteinfer",
        "num_requests": 10,
        "output_tokens": 100,
        "wall_time_s": 5.0,
        "requests_per_second": 2.0,
        "output_tokens_per_second": 20.0,
        "ttft_p50_s": 0.05,
        "ttft_p99_s": 0.10,
        "e2e_latency_p50_s": 0.5,
        "e2e_latency_p99_s": 1.0,
        "peak_memory_bytes": None,
    }
    base.update(overrides)
    return base


def _run(tag: str = "run", engine: str = "liteinfer", **metric_overrides) -> dict:
    return {
        "timestamp": "2026-05-03T12:00:00",
        "tag": tag,
        "workload": "throughput",
        "model": "org/model-1b",
        "results": [_engine_metrics(engine=engine, **metric_overrides)],
    }


# ---------------------------------------------------------------------------
# load_history
# ---------------------------------------------------------------------------

def test_load_history_parses_jsonl(tmp_path) -> None:
    history_file = tmp_path / "history.jsonl"
    runs = [_run("a"), _run("b")]
    history_file.write_text("\n".join(json.dumps(r) for r in runs) + "\n")

    loaded = load_history(history_file)
    assert len(loaded) == 2
    assert loaded[0]["tag"] == "a"
    assert loaded[1]["tag"] == "b"


def test_load_history_skips_blank_lines(tmp_path) -> None:
    history_file = tmp_path / "history.jsonl"
    history_file.write_text(json.dumps(_run()) + "\n\n" + json.dumps(_run()) + "\n")
    assert len(load_history(history_file)) == 2


# ---------------------------------------------------------------------------
# _fmt
# ---------------------------------------------------------------------------

def test_fmt_latency_converts_to_ms() -> None:
    assert _fmt("ttft_p50_s", 0.05) == "50.0 ms"
    assert _fmt("e2e_latency_p99_s", 1.234) == "1234.0 ms"


def test_fmt_throughput_uses_two_decimals() -> None:
    assert _fmt("requests_per_second", 6.123) == "6.12"
    assert _fmt("output_tokens_per_second", 390.1) == "390.10"


# ---------------------------------------------------------------------------
# _delta_class
# ---------------------------------------------------------------------------

def test_delta_class_improvement_higher_is_better() -> None:
    assert _delta_class(1.1, 1.0, higher_is_better=True) == "ok"


def test_delta_class_regression_higher_is_better() -> None:
    assert _delta_class(0.9, 1.0, higher_is_better=True) == "bad"


def test_delta_class_improvement_lower_is_better() -> None:
    assert _delta_class(0.9, 1.0, higher_is_better=False) == "ok"


def test_delta_class_regression_lower_is_better() -> None:
    assert _delta_class(1.1, 1.0, higher_is_better=False) == "bad"


def test_delta_class_within_threshold_is_neutral() -> None:
    assert _delta_class(1.02, 1.0, higher_is_better=True) == ""
    assert _delta_class(0.98, 1.0, higher_is_better=False) == ""


def test_delta_class_no_previous_is_neutral() -> None:
    assert _delta_class(1.0, None, higher_is_better=True) == ""
    assert _delta_class(1.0, 0.0, higher_is_better=True) == ""


# ---------------------------------------------------------------------------
# build_html
# ---------------------------------------------------------------------------

def test_build_html_empty_runs_returns_no_table() -> None:
    html = build_html([])
    assert "<table>" not in html


def test_build_html_contains_workload_heading() -> None:
    html = build_html([_run()])
    assert "throughput" in html


def test_build_html_contains_engine_header() -> None:
    html = build_html([_run(engine="liteinfer")])
    assert "liteinfer" in html


def test_build_html_tag_rendered() -> None:
    html = build_html([_run(tag="my-tag")])
    assert "my-tag" in html


def test_build_html_two_engines_both_appear() -> None:
    run = {
        "timestamp": "2026-05-03T12:00:00",
        "tag": "both",
        "workload": "throughput",
        "model": "org/model",
        "results": [
            _engine_metrics(engine="vllm"),
            _engine_metrics(engine="liteinfer"),
        ],
    }
    html = build_html([run])
    assert "vllm" in html
    assert "liteinfer" in html


def test_build_html_color_codes_regression() -> None:
    run1 = _run("baseline", requests_per_second=2.0)
    run2 = _run("optimized", requests_per_second=1.0)  # worse throughput
    html = build_html([run1, run2])
    assert 'class="bad"' in html


def test_build_html_color_codes_improvement() -> None:
    run1 = _run("baseline", requests_per_second=2.0)
    run2 = _run("optimized", requests_per_second=3.0)  # better throughput
    html = build_html([run1, run2])
    assert 'class="ok"' in html
