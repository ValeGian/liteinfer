"""Tests for benchmarks.harness module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.adapters.base import RequestMeasurement
from benchmarks.harness import BenchmarkError, run_benchmark

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, num_samples: int, forced_osl: int = 256) -> None:
    with open(path, "w") as f:
        for i in range(num_samples):
            record = {
                "prompt": f"prompt number {i}",
                "input_token_count": 128,
                "forced_output_token_count": forced_osl,
            }
            f.write(json.dumps(record) + "\n")


def _make_measurements(
    count: int,
    output_token_count: int = 256,
    ttft_s: float = 0.1,
    e2e_s: float = 2.0,
) -> list[RequestMeasurement]:
    return [
        RequestMeasurement(
            sample_index=i,
            input_token_count=128,
            output_token_count=output_token_count,
            ttft_s=ttft_s,
            token_timestamps_s=None,
            e2e_s=e2e_s,
        )
        for i in range(count)
    ]


def _make_mock_adapter(measurements: list[RequestMeasurement], wall_time_s: float = 10.0):
    adapter = MagicMock()
    adapter.__enter__ = MagicMock(return_value=adapter)
    adapter.__exit__ = MagicMock(return_value=None)
    adapter.run.return_value = (measurements, wall_time_s)
    adapter_cls = MagicMock(return_value=adapter)
    return adapter_cls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_benchmark_saves_result_json(tmp_path):
    dataset_file = tmp_path / "data.jsonl"
    _write_jsonl(dataset_file, num_samples=10)

    measurements = _make_measurements(10)
    adapter_cls = _make_mock_adapter(measurements)

    with patch("benchmarks.harness.ADAPTER_REGISTRY", {"liteinfer": adapter_cls}):
        result = run_benchmark(
            engine_name="liteinfer",
            benchmark_type="throughput",
            model="test/model",
            dataset_path=dataset_file,
            num_samples=10,
            tag="test-tag",
            results_dir=tmp_path / "results",
        )

    assert result["engine"] == "liteinfer"
    assert result["benchmark_type"] == "throughput"
    assert result["model"] == "test/model"
    assert result["tag"] == "test-tag"

    result_files = list((tmp_path / "results").glob("*.json"))
    assert len(result_files) == 1

    saved = json.loads(result_files[0].read_text())
    assert saved["run_id"] == result["run_id"]
    assert saved["dataset"]["num_samples"] == 10


def test_run_benchmark_result_contains_sha256(tmp_path):
    dataset_file = tmp_path / "data.jsonl"
    _write_jsonl(dataset_file, num_samples=5)

    measurements = _make_measurements(5)
    adapter_cls = _make_mock_adapter(measurements)

    with patch("benchmarks.harness.ADAPTER_REGISTRY", {"engine_a": adapter_cls}):
        result = run_benchmark(
            engine_name="engine_a",
            benchmark_type="throughput",
            model="m",
            dataset_path=dataset_file,
            num_samples=5,
            tag="",
            results_dir=tmp_path / "results",
        )

    assert "prompt_batch_sha256" in result["dataset"]
    assert len(result["dataset"]["prompt_batch_sha256"]) == 64  # sha256 hex length


def test_run_benchmark_uses_num_samples_subset(tmp_path):
    dataset_file = tmp_path / "data.jsonl"
    _write_jsonl(dataset_file, num_samples=50)

    measurements = _make_measurements(10)
    adapter_cls = _make_mock_adapter(measurements)

    with patch("benchmarks.harness.ADAPTER_REGISTRY", {"eng": adapter_cls}):
        result = run_benchmark(
            engine_name="eng",
            benchmark_type="latency",
            model="m",
            dataset_path=dataset_file,
            num_samples=10,
            tag="",
            results_dir=tmp_path / "results",
        )

    # The adapter should have been called with exactly 10 samples
    call_args = adapter_cls.return_value.run.call_args
    assert len(call_args[0][0]) == 10
    assert result["dataset"]["num_samples"] == 10


def test_strict_osl_raises_when_mismatch_exceeds_threshold(tmp_path):
    dataset_file = tmp_path / "data.jsonl"
    # forced_osl = 256
    _write_jsonl(dataset_file, num_samples=10, forced_osl=256)

    # 10% of measurements have wrong output_token_count (1 out of 10)
    measurements = _make_measurements(9, output_token_count=256)
    measurements.append(
        RequestMeasurement(
            sample_index=9,
            input_token_count=128,
            output_token_count=100,  # wrong
            ttft_s=0.1,
            token_timestamps_s=None,
            e2e_s=2.0,
        )
    )
    adapter_cls = _make_mock_adapter(measurements)

    with patch("benchmarks.harness.ADAPTER_REGISTRY", {"eng": adapter_cls}), pytest.raises(BenchmarkError, match="strict-osl"):
        run_benchmark(
                engine_name="eng",
                benchmark_type="throughput",
                model="m",
                dataset_path=dataset_file,
                num_samples=10,
                tag="",
                results_dir=tmp_path / "results",
                strict_osl=True,
            )


def test_strict_osl_passes_when_mismatch_within_threshold(tmp_path):
    dataset_file = tmp_path / "data.jsonl"
    # 25 samples, forced_osl = 256
    _write_jsonl(dataset_file, num_samples=25, forced_osl=256)

    # 1 out of 25 = 4% mismatch, under the 5% threshold
    measurements = _make_measurements(24, output_token_count=256)
    measurements.append(
        RequestMeasurement(
            sample_index=24,
            input_token_count=128,
            output_token_count=250,  # wrong but only 4%
            ttft_s=0.1,
            token_timestamps_s=None,
            e2e_s=2.0,
        )
    )
    adapter_cls = _make_mock_adapter(measurements)

    with patch("benchmarks.harness.ADAPTER_REGISTRY", {"eng": adapter_cls}):
        # Should not raise
        result = run_benchmark(
            engine_name="eng",
            benchmark_type="throughput",
            model="m",
            dataset_path=dataset_file,
            num_samples=25,
            tag="",
            results_dir=tmp_path / "results",
            strict_osl=True,
        )
    assert result["dataset"]["num_samples"] == 25
