# pyright: reportPrivateImportUsage=false
"""GPU parity and throughput tests for continuous batching.

Run:
    pytest tests/e2e/test_continuous_gpu.py -m "gpu and e2e" -v
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from liteinfer import AsyncLLM
from liteinfer.hub import resolve_model_path
from liteinfer.sampling.params import SamplingParams

_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
_DEVICE = "cuda:0"
_DTYPE = torch.bfloat16
_MAX_TOKENS = 20

_PARITY_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time in a land far away,",
]


@pytest.fixture(scope="module")
def model_dir():
    return resolve_model_path(_MODEL_ID)


# ---------------------------------------------------------------------------


@pytest.mark.gpu
# ---------------------------------------------------------------------------
# Functional: multi-prompt throughput run (correctness only, not perf)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
def test_continuous_gpu_all_prompts_complete(model_dir) -> None:
    """All requests must complete without error under continuous batching."""
    prompts = _PARITY_PROMPTS * 3  # 9 requests
    params = SamplingParams(max_tokens=_MAX_TOKENS, temperature=0.0)

    async def _run_async():
        async with AsyncLLM(
            str(model_dir),
            device=_DEVICE,
            dtype=_DTYPE,  # type: ignore[arg-type]
            max_num_seqs=4,
        ) as llm:
            return await llm.generate(prompts, params)

    outputs = asyncio.run(_run_async())
    assert len(outputs) == len(prompts)
    for out in outputs:
        assert 0 < len(out.token_ids) <= _MAX_TOKENS
        assert out.finish_reason in ("stop", "length")


# ---------------------------------------------------------------------------
# Streaming API on GPU
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
def test_continuous_gpu_streaming_yields_tokens(model_dir) -> None:
    token_counts = []

    async def _run_async():
        async with AsyncLLM(
            str(model_dir),
            device=_DEVICE,
            dtype=_DTYPE,  # type: ignore[arg-type]
        ) as llm:
            async for event in llm.stream(
                "The quick brown fox", SamplingParams(max_tokens=10, temperature=0.0)
            ):
                token_counts.append(len(event.output_token_ids))

    asyncio.run(_run_async())
    assert len(token_counts) >= 1
    assert token_counts == sorted(token_counts)
    assert token_counts[-1] >= 1
