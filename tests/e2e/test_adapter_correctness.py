"""E2E adapter correctness tests.

All tests require a CUDA GPU and load a real model. They are skipped
automatically when the required hardware or software is not available.

Run with: pytest -m "e2e and gpu"
Never run with -n auto (GPU tests must be sequential).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.adapters.base import RequestMeasurement
from benchmarks.dataset import generate_dataset

_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_FORCED_OSL = 16
_TARGET_ISL = 32
_NUM_SAMPLES = 5


# ---------------------------------------------------------------------------
# Shared fixture: generate dataset once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_dataset(tmp_path_factory):
    from unittest.mock import patch

    tmp_path = tmp_path_factory.mktemp("bench_dataset")
    # Mock the corpus so the fixture does not trigger a ShareGPT download.
    # Adapter correctness tests verify token counts and timing, not corpus content.
    mock_corpus = "Some realistic instruction text about many different topics. " * 500
    with patch("benchmarks.dataset._get_corpus", return_value=mock_corpus):
        dataset_path = generate_dataset(
            model_id=_MODEL,
            target_isl=_TARGET_ISL,
            target_osl=_FORCED_OSL,
            num_samples=_NUM_SAMPLES,
            output_path=tmp_path,
        )
    return dataset_path


# ---------------------------------------------------------------------------
# Shared correctness assertion
# ---------------------------------------------------------------------------


def _assert_measurement_correctness(
    measurements: list[RequestMeasurement],
    wall_time_s: float,
    forced_osl: int,
) -> None:
    assert len(measurements) > 0
    assert wall_time_s > 0
    for m in measurements:
        assert m.output_token_count == forced_osl, (
            f"sample {m.sample_index}: expected {forced_osl} output tokens, "
            f"got {m.output_token_count}"
        )
        assert m.ttft_s > 0
        assert m.e2e_s >= m.ttft_s
        assert m.input_token_count > 0


# ---------------------------------------------------------------------------
# liteinfer
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
def test_liteinfer_adapter_throughput_correctness(test_dataset, tmp_path):
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")

    from benchmarks.adapters.liteinfer import LiteInferAdapter

    adapter = LiteInferAdapter()
    from benchmarks.dataset import load_dataset

    samples = load_dataset(test_dataset, num_samples=_NUM_SAMPLES)
    with adapter:
        measurements, wall_time_s = adapter.run(samples, _MODEL, "throughput")

    _assert_measurement_correctness(measurements, wall_time_s, _FORCED_OSL)


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
def test_liteinfer_adapter_latency_correctness(test_dataset, tmp_path):
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")

    from benchmarks.adapters.liteinfer import LiteInferAdapter
    from benchmarks.dataset import load_dataset

    adapter = LiteInferAdapter()
    samples = load_dataset(test_dataset, num_samples=_NUM_SAMPLES)
    with adapter:
        measurements, wall_time_s = adapter.run(samples, _MODEL, "latency")

    _assert_measurement_correctness(measurements, wall_time_s, _FORCED_OSL)
    assert len(measurements) == _NUM_SAMPLES


_VLLM_PYTHON = Path("benchmarks/envs/vllm/bin/python")
_TRTLLM_PYTHON = Path("benchmarks/envs/trtllm/bin/python")


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
def test_vllm_adapter_throughput_correctness(test_dataset, tmp_path):
    if not _VLLM_PYTHON.exists():
        pytest.skip("vLLM venv not found at benchmarks/envs/vllm/bin/python")
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")

    from benchmarks.adapters.vllm.adapter import VLLMAdapter
    from benchmarks.dataset import load_dataset

    adapter = VLLMAdapter()
    samples = load_dataset(test_dataset, num_samples=_NUM_SAMPLES)
    with adapter:
        measurements, wall_time_s = adapter.run(samples, _MODEL, "throughput")

    _assert_measurement_correctness(measurements, wall_time_s, _FORCED_OSL)


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
def test_vllm_adapter_latency_correctness(test_dataset, tmp_path):
    if not _VLLM_PYTHON.exists():
        pytest.skip("vLLM venv not found at benchmarks/envs/vllm/bin/python")
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")

    from benchmarks.adapters.vllm.adapter import VLLMAdapter
    from benchmarks.dataset import load_dataset

    adapter = VLLMAdapter()
    samples = load_dataset(test_dataset, num_samples=_NUM_SAMPLES)
    with adapter:
        measurements, wall_time_s = adapter.run(samples, _MODEL, "latency")

    _assert_measurement_correctness(measurements, wall_time_s, _FORCED_OSL)


# ---------------------------------------------------------------------------
# TRT-LLM
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
def test_trtllm_adapter_throughput_correctness(test_dataset, tmp_path):
    if not _TRTLLM_PYTHON.exists():
        pytest.skip("TRT-LLM venv not found at benchmarks/envs/trtllm/bin/python")

    from benchmarks.adapters.trtllm.adapter import TRTLLMAdapter
    from benchmarks.dataset import load_dataset

    adapter = TRTLLMAdapter()
    samples = load_dataset(test_dataset, num_samples=_NUM_SAMPLES)
    with adapter:
        measurements, wall_time_s = adapter.run(samples, _MODEL, "throughput")

    _assert_measurement_correctness(measurements, wall_time_s, _FORCED_OSL)


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
def test_trtllm_adapter_latency_correctness(test_dataset, tmp_path):
    if not _TRTLLM_PYTHON.exists():
        pytest.skip("TRT-LLM venv not found at benchmarks/envs/trtllm/bin/python")

    from benchmarks.adapters.trtllm.adapter import TRTLLMAdapter
    from benchmarks.dataset import load_dataset

    adapter = TRTLLMAdapter()
    samples = load_dataset(test_dataset, num_samples=_NUM_SAMPLES)
    with adapter:
        measurements, wall_time_s = adapter.run(samples, _MODEL, "latency")

    _assert_measurement_correctness(measurements, wall_time_s, _FORCED_OSL)
