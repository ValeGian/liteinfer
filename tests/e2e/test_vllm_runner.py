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
    # All _ts fields are monotonic; arrival_time is Unix — different clocks.
    arrival_time = 1_700_000_000.0  # Unix, must NOT be mixed with _ts fields
    queued_ts = 1000.0
    first_token_ts = 1000.5
    last_token_ts = 1001.5
    first_token_latency = 99.0  # unreliable cross-clock value — must be ignored


class _MockMetricsLegacy:
    arrival_time = 100.0
    first_token_time = 100.5
    finished_time = 101.5


class _MockMetricsEmpty:
    pass


def test_extract_timing_uses_v020_monotonic_fields() -> None:
    ttft, total = _extract_timing(_MockMetricsV020())
    assert ttft == pytest.approx(0.5)   # first_token_ts - queued_ts
    assert total == pytest.approx(1.5)  # last_token_ts - queued_ts


def test_extract_timing_ignores_first_token_latency() -> None:
    # first_token_latency crosses clock sources and must not be used for TTFT.
    ttft, _ = _extract_timing(_MockMetricsV020())
    assert ttft != pytest.approx(99.0)


def test_extract_timing_v020_preferred_over_legacy_when_both_present() -> None:
    class Both(_MockMetricsV020, _MockMetricsLegacy):
        pass

    ttft, total = _extract_timing(Both())
    # v0.20.0 monotonic path wins (queued_ts / first_token_ts / last_token_ts > 0)
    assert ttft == pytest.approx(0.5)   # first_token_ts - queued_ts, not legacy
    assert total == pytest.approx(1.5)  # last_token_ts - queued_ts
