"""Harness orchestration, using a stub engine so nothing touches a GPU."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from benchmarks import harness
from benchmarks.configs import get
from benchmarks.dataset import Dataset, Sample

OSL = 8


def _dataset(num_samples: int = 3) -> Dataset:
    samples = [Sample(prompt=f"prompt {i}", input_tokens=16) for i in range(num_samples)]
    return Dataset(model="test/model", target_isl=16, target_osl=OSL, samples=samples)


class StubAdapter:
    """Records every call; returns the requested output length by default."""

    def __init__(self, short_by: int = 0) -> None:
        self.calls: list[tuple[int, int]] = []  # (num_prompts, max_tokens)
        self._short_by = short_by

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def generate(self, prompts: list[str], max_tokens: int) -> list[int]:
        self.calls.append((len(prompts), max_tokens))
        return [max_tokens - self._short_by] * len(prompts)


@contextmanager
def _stubbed(adapter: StubAdapter):
    with patch.object(harness.adapters, "build", return_value=adapter):
        yield


def _run(
    mode: str, tmp_path, adapter: StubAdapter, data: Dataset | None = None, config="liteinfer-paged"
):
    with _stubbed(adapter):
        return harness.run(get(config), data or _dataset(), "d.json", mode, tmp_path)


def _warmup_calls(mode: str, width: int) -> list[tuple[int, int]]:
    round_ = (
        [(width, 1), (width, harness.WARMUP_TOKENS)]
        if mode == "latency"
        else [(width, harness.WARMUP_TOKENS)]
    )
    return round_ * harness.WARMUP_ROUNDS


def test_throughput_submits_every_prompt_in_one_call(tmp_path) -> None:
    adapter = StubAdapter()
    _run("throughput", tmp_path, adapter)
    assert adapter.calls[-1] == (3, OSL)


def test_throughput_reports_tokens_per_second(tmp_path) -> None:
    result = _run("throughput", tmp_path, StubAdapter())
    assert result.summary["output_tokens_per_s"] > 0


def test_latency_times_each_sample_with_a_prefill_only_pass(tmp_path) -> None:
    adapter = StubAdapter()
    _run("latency", tmp_path, adapter)
    assert adapter.calls[len(_warmup_calls("latency", 1)) :] == [(1, 1), (1, OSL)] * 3


def test_latency_reports_itl_percentiles(tmp_path) -> None:
    result = _run("latency", tmp_path, StubAdapter())
    assert "itl_p50_ms" in result.summary


def test_warmup_precedes_the_measured_work(tmp_path) -> None:
    adapter = StubAdapter()
    _run("throughput", tmp_path, adapter)
    assert adapter.calls[:-1] == _warmup_calls("throughput", width=1)


def test_warmup_uses_the_batch_width_it_is_about_to_measure(tmp_path) -> None:
    adapter = StubAdapter()
    _run("throughput", tmp_path, adapter, config="liteinfer-paged-b4")
    assert adapter.calls[0] == (4, harness.WARMUP_TOKENS)


def test_latency_warmup_includes_a_prefill_only_pass(tmp_path) -> None:
    adapter = StubAdapter()
    _run("latency", tmp_path, adapter)
    assert adapter.calls[0] == (1, 1)


def test_warmup_uses_real_prompts_rather_than_a_placeholder(tmp_path) -> None:
    seen: list[list[str]] = []
    adapter = StubAdapter()
    original = adapter.generate
    adapter.generate = lambda prompts, max_tokens: (
        seen.append(list(prompts)),
        original(prompts, max_tokens),
    )[1]
    _run("throughput", tmp_path, adapter)
    assert seen[0] == [_dataset().samples[0].prompt]


def test_a_wrong_output_length_fails_the_run(tmp_path) -> None:
    with pytest.raises(harness.BenchmarkError, match="wrong output length"):
        _run("throughput", tmp_path, StubAdapter(short_by=1))


def test_an_empty_dataset_fails_the_run(tmp_path) -> None:
    with pytest.raises(harness.BenchmarkError, match="no samples"):
        _run("throughput", tmp_path, StubAdapter(), data=_dataset(0))


def test_result_filename_identifies_config_mode_and_shape(tmp_path) -> None:
    result = _run("throughput", tmp_path, StubAdapter())
    assert result.filename == "liteinfer-paged__throughput__isl16_osl8.json"


def test_result_file_records_the_prompt_set_digest(tmp_path) -> None:
    data = _dataset()
    result = _run("throughput", tmp_path, StubAdapter(), data=data)
    written = json.loads((tmp_path / result.filename).read_text())
    assert written["dataset"]["sha256"] == data.sha256


def test_rerunning_overwrites_rather_than_accumulates(tmp_path) -> None:
    _run("throughput", tmp_path, StubAdapter())
    _run("throughput", tmp_path, StubAdapter())
    assert len(list(tmp_path.glob("*.json"))) == 1
