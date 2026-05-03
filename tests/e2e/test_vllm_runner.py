"""End-to-end tests for the vLLM benchmark runner against vLLM 0.20.0.

Downloads facebook/opt-125m (~250 MB) on first run; the model is cached
by HuggingFace in ~/.cache/huggingface/hub and removed by the session
fixture below so it does not accumulate between CI runs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchmarks.runners.base import SamplingSpec
from benchmarks.runners.vllm_runner import VLLMRunner, _extract_timing

_MODEL = "facebook/opt-125m"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner():
    r = VLLMRunner()
    r.setup(_MODEL)
    yield r
    r.teardown()


@pytest.fixture(scope="session", autouse=True)
def _cleanup_model_cache():
    """Remove the downloaded model from the HF cache after the session."""
    yield
    _delete_hf_cache(_MODEL)


def _delete_hf_cache(repo_id: str) -> None:
    slug = repo_id.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    for candidate in cache_root.glob(f"models--{slug}"):
        shutil.rmtree(candidate, ignore_errors=True)


# ---------------------------------------------------------------------------
# Generation shape
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
def test_single_prompt_returns_one_result(runner: VLLMRunner) -> None:
    results = runner.generate(["Hello, my name is"], SamplingSpec(temperature=0.0, max_tokens=16))
    assert len(results) == 1


@pytest.mark.gpu
@pytest.mark.e2e
def test_result_has_non_empty_output(runner: VLLMRunner) -> None:
    results = runner.generate(["The capital of France is"], SamplingSpec(temperature=0.0, max_tokens=8))
    assert results[0].output_text != ""
    assert len(results[0].output_token_ids) > 0


@pytest.mark.gpu
@pytest.mark.e2e
def test_batch_returns_correct_count(runner: VLLMRunner) -> None:
    prompts = [f"Write a sentence about the number {i}." for i in range(4)]
    results = runner.generate(prompts, SamplingSpec(temperature=0.0, max_tokens=20))
    assert len(results) == len(prompts)


@pytest.mark.gpu
@pytest.mark.e2e
def test_prompt_field_populated(runner: VLLMRunner) -> None:
    prompt = "Once upon a time"
    results = runner.generate([prompt], SamplingSpec(temperature=0.0, max_tokens=8))
    assert results[0].prompt == prompt


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
def test_ttft_is_positive(runner: VLLMRunner) -> None:
    results = runner.generate(["Hello world"], SamplingSpec(temperature=0.0, max_tokens=16))
    assert results[0].ttft_s > 0.0


@pytest.mark.gpu
@pytest.mark.e2e
def test_total_time_gte_ttft(runner: VLLMRunner) -> None:
    results = runner.generate(["Hello world"], SamplingSpec(temperature=0.0, max_tokens=32))
    assert results[0].total_time_s >= results[0].ttft_s


@pytest.mark.gpu
@pytest.mark.e2e
def test_total_time_positive_for_all_results(runner: VLLMRunner) -> None:
    prompts = [f"Sentence {i}" for i in range(3)]
    results = runner.generate(prompts, SamplingSpec(temperature=0.0, max_tokens=16))
    for r in results:
        assert r.total_time_s > 0.0


# ---------------------------------------------------------------------------
# _extract_timing unit tests (no GPU needed)
# ---------------------------------------------------------------------------


class _MockMetricsV020:
    arrival_time = 100.0
    first_token_ts = 100.5
    last_token_ts = 101.5
    first_token_latency = 0.5


class _MockMetricsLegacy:
    arrival_time = 100.0
    first_token_time = 100.5
    finished_time = 101.5


class _MockMetricsEmpty:
    pass


def test_extract_timing_uses_v020_fields() -> None:
    ttft, total = _extract_timing(_MockMetricsV020(), wall=5.0, num_tokens=10)
    assert ttft == pytest.approx(0.5)   # first_token_latency
    assert total == pytest.approx(1.5)  # last_token_ts - arrival_time


def test_extract_timing_falls_back_to_legacy_fields() -> None:
    ttft, total = _extract_timing(_MockMetricsLegacy(), wall=5.0, num_tokens=10)
    assert ttft == pytest.approx(0.5)   # first_token_time - arrival_time
    assert total == pytest.approx(1.5)  # finished_time - arrival_time


def test_extract_timing_wall_fallback_when_no_metrics() -> None:
    ttft, total = _extract_timing(_MockMetricsEmpty(), wall=2.0, num_tokens=20)
    assert ttft == pytest.approx(0.1)   # wall / num_tokens
    assert total == pytest.approx(2.0)


def test_extract_timing_v020_preferred_over_legacy_when_both_present() -> None:
    class Both(_MockMetricsV020, _MockMetricsLegacy):
        pass

    ttft, total = _extract_timing(Both(), wall=5.0, num_tokens=10)
    # v0.20.0 path wins because first_token_ts > 0
    assert ttft == pytest.approx(0.5)
    assert total == pytest.approx(1.5)
