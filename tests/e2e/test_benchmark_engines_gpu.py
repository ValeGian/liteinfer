"""Every adapter must honour the forced output length on real hardware.

The whole benchmark rests on one invariant: each request emits exactly the
requested number of tokens, so output-length variance can never be mistaken for
an engine difference. These tests check that invariant against real models
rather than stubs.
"""

from __future__ import annotations

import pytest

from benchmarks import harness
from benchmarks.configs import get
from benchmarks.dataset import Dataset, Sample

MODEL = "unsloth/Llama-3.2-1B-Instruct"
OSL = 8
CONFIGS = [
    "liteinfer-nocache",
    "liteinfer-eager",
    "liteinfer-native-eager",
    "liteinfer-paged",
    "liteinfer-paged-b4",
    "liteinfer-continuous",
    "vllm",
]

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    prompts = [
        "The capital city of France is",
        "In a single sentence, explain what a compiler does:",
    ]
    return Dataset(
        model=MODEL,
        target_isl=8,
        target_osl=OSL,
        samples=[Sample(prompt=p, input_tokens=8) for p in prompts],
    )


@pytest.fixture(autouse=True)
def _require_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.mark.parametrize("config_name", CONFIGS)
def test_throughput_run_forces_the_requested_output_length(config_name, dataset, tmp_path) -> None:
    # harness.run raises BenchmarkError if any request returns the wrong length.
    result = harness.run(get(config_name), dataset, "memory", "throughput", tmp_path)
    assert result.raw["output_tokens"] == OSL * len(dataset.samples)


@pytest.mark.parametrize("config_name", ["liteinfer-paged", "liteinfer-continuous", "vllm"])
def test_latency_run_produces_positive_percentiles(config_name, dataset, tmp_path) -> None:
    result = harness.run(get(config_name), dataset, "memory", "latency", tmp_path)
    assert result.summary["itl_p50_ms"] > 0


def test_async_engine_generates_past_eos_when_ignore_eos_is_set(dataset, tmp_path) -> None:
    # The async engine once had its own copy of the stop rule that ignored
    # ignore_eos, silently truncating every liteinfer throughput run.
    long_osl = 64
    forced = Dataset(dataset.model, dataset.target_isl, long_osl, dataset.samples[:1])
    result = harness.run(get("liteinfer-continuous"), forced, "memory", "throughput", tmp_path)
    assert result.raw["output_tokens"] == long_osl
