"""Integration test: cross-engine sample identity.

Verifies that two result files from different (mock) engines carry identical
SHA-256 hashes when run against the same dataset. This catches any regression
where the harness accidentally re-generates the dataset per adapter instead
of loading the canonical file.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from benchmarks.adapters.base import RequestMeasurement
from benchmarks.dataset import generate_dataset
from benchmarks.harness import run_benchmark


def _make_registry_mock(measurements: list[RequestMeasurement], wall_time_s: float = 5.0):
    """Create a fake ADAPTER_REGISTRY entry that echoes the given measurements."""
    adapter = MagicMock()
    adapter.__enter__ = MagicMock(return_value=adapter)
    adapter.__exit__ = MagicMock(return_value=None)
    adapter.run.return_value = (measurements, wall_time_s)
    adapter_cls = MagicMock(return_value=adapter)
    return adapter_cls


def generate_test_dataset(tmp_path: Path, num_samples: int, target_isl: int, target_osl: int) -> Path:
    """Generate a tiny test dataset using a mock tokenizer."""
    from unittest.mock import MagicMock, patch

    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=True: list(range(len(text)))
    tokenizer.decode.side_effect = lambda ids, skip_special_tokens=True: "a" * len(ids)

    mock_corpus = "some realistic instruction text here " * 300
    with patch("benchmarks.dataset.AutoTokenizer") as mock_auto, \
            patch("benchmarks.dataset._get_corpus", return_value=mock_corpus):
        mock_auto.from_pretrained.return_value = tokenizer
        dataset_path = generate_dataset(
            model_id="test/model",
            target_isl=target_isl,
            target_osl=target_osl,
            num_samples=num_samples,
            output_path=tmp_path,
            seed=42,
        )
    return dataset_path


def run_benchmark_with_mock_adapter(
    engine_name: str,
    dataset_path: Path,
    results_dir: Path,
) -> dict:
    """Run benchmark with a mock adapter that returns trivial measurements."""
    from unittest.mock import patch

    from benchmarks.dataset import load_dataset

    samples = load_dataset(dataset_path)
    measurements = [
        RequestMeasurement(
            sample_index=i,
            input_token_count=s.input_token_count,
            output_token_count=s.forced_output_token_count,
            ttft_s=0.05,
            token_timestamps_s=None,
            e2e_s=0.5,
        )
        for i, s in enumerate(samples)
    ]

    adapter_cls = _make_registry_mock(measurements)
    registry = {engine_name: adapter_cls}

    with patch("benchmarks.harness.ADAPTER_REGISTRY", registry):
        result = run_benchmark(
            engine_name=engine_name,
            benchmark_type="throughput",
            model="test/model",
            dataset_path=dataset_path,
            num_samples=None,
            tag="",
            results_dir=results_dir,
        )
    return result


def test_cross_engine_sample_identity(tmp_path):
    """Results from different engines on the same dataset must carry identical SHA-256."""
    dataset = generate_test_dataset(tmp_path, num_samples=5, target_isl=10, target_osl=5)

    results_dir = tmp_path / "results"
    result_a = run_benchmark_with_mock_adapter("engine_a", dataset_path=dataset, results_dir=results_dir)
    result_b = run_benchmark_with_mock_adapter("engine_b", dataset_path=dataset, results_dir=results_dir)

    assert result_a["dataset"]["prompt_batch_sha256"] == result_b["dataset"]["prompt_batch_sha256"]
    assert result_a["dataset"]["num_samples"] == result_b["dataset"]["num_samples"]
    assert (
        [s["input_token_count"] for s in result_a["raw"]]
        == [s["input_token_count"] for s in result_b["raw"]]
    )
