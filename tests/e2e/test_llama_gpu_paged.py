# pyright: reportPrivateImportUsage=false
"""GPU parity tests for the paged KV cache.

Tests operate exclusively through `LLM.generate()` — no engine internals.
This ensures the test contract survives any internal reimplementation of the
paged cache, as long as the user-facing API remains unchanged.

Run:
    pytest tests/e2e/test_llama_gpu_paged.py -m "gpu and e2e" -v
"""

from __future__ import annotations

import gc

import pytest
import torch

from liteinfer import LLM
from liteinfer.hub import resolve_model_path
from liteinfer.sampling.params import SamplingParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
_DEVICE = "cuda:0"
_DTYPE = torch.bfloat16
_PARITY_MAX_TOKENS = 20

_PARITY_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time in a land far away,",
]

_BATCH_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time in a land far away,",
    "In modern computer architecture,",
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_dir():
    return resolve_model_path(_MODEL_ID)


_PAGED_NUM_BLOCKS = 512  # limits pool to ~256 MB — ample for max_tokens=20 tests


def _make_llm(model_dir, *, cache_mode: str, max_num_seqs: int = 4) -> LLM:
    extra: dict = {}
    if cache_mode == "paged":
        extra["num_gpu_blocks"] = _PAGED_NUM_BLOCKS
    return LLM(
        str(model_dir),
        device=_DEVICE,
        dtype=_DTYPE,
        cache_mode=cache_mode,
        max_num_seqs=max_num_seqs,
        **extra,
    )


def _teardown_llm(llm: LLM) -> None:
    if llm.engine.model_runner.model is not None:
        llm.engine.model_runner.model.cpu()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def llm_paged(model_dir):
    llm = _make_llm(model_dir, cache_mode="paged", max_num_seqs=4)
    yield llm
    _teardown_llm(llm)


@pytest.fixture(scope="module")
def llm_native_eager(model_dir):
    llm = _make_llm(model_dir, cache_mode="native_eager", max_num_seqs=4)
    yield llm
    _teardown_llm(llm)


# ---------------------------------------------------------------------------
# Parity: paged cache vs native_eager (single sequence)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("prompt", _PARITY_PROMPTS)
def test_paged_cache_single_seq_matches_native_eager(
    llm_paged: LLM, llm_native_eager: LLM, prompt: str
) -> None:
    """Paged KV cache must produce token-for-token identical output to native_eager
    under greedy decoding for a single prompt."""
    params = SamplingParams(max_tokens=_PARITY_MAX_TOKENS, temperature=0.0)
    paged_ids = llm_paged.generate(prompt, params)[0].token_ids
    native_ids = llm_native_eager.generate(prompt, params)[0].token_ids
    assert paged_ids == native_ids, (
        f"prompt={prompt!r}\n"
        f"  paged       : {paged_ids}\n"
        f"  native_eager: {native_ids}"
    )


# ---------------------------------------------------------------------------
# Parity: paged cache vs native_eager (batched)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_paged_cache_batched_matches_native_eager(
    llm_paged: LLM, llm_native_eager: LLM
) -> None:
    """B=4 paged cache must match B=4 native_eager token-for-token per prompt."""
    params = SamplingParams(max_tokens=_PARITY_MAX_TOKENS, temperature=0.0)
    paged_outputs = llm_paged.generate(_BATCH_PROMPTS, params)
    native_outputs = llm_native_eager.generate(_BATCH_PROMPTS, params)

    assert len(paged_outputs) == len(native_outputs) == len(_BATCH_PROMPTS)
    for prompt, paged_out, native_out in zip(_BATCH_PROMPTS, paged_outputs, native_outputs, strict=True):
        assert paged_out.token_ids == native_out.token_ids, (
            f"prompt={prompt!r}\n"
            f"  paged       : {paged_out.token_ids}\n"
            f"  native_eager: {native_out.token_ids}"
        )


# ---------------------------------------------------------------------------
# Parity: paged cache across multiple batches (blocks freed and reused)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_paged_cache_multiple_batches_matches_native_eager(model_dir) -> None:
    """8 prompts with max_num_seqs=2 must all match native_eager.

    This drives multiple static batches through the paged cache so that
    blocks are freed and re-allocated between batches. A corruption
    bug in free/reuse would surface here.
    """
    prompts = [
        "Apples are",
        "Bears live in",
        "Cats sometimes",
        "Dogs prefer",
        "Eagles fly",
        "Foxes hunt",
        "Geese migrate",
        "Hippos weigh",
    ]
    params = SamplingParams(max_tokens=8, temperature=0.0)
    llm_paged = _make_llm(model_dir, cache_mode="paged", max_num_seqs=2)
    llm_native = _make_llm(model_dir, cache_mode="native_eager", max_num_seqs=2)
    try:
        paged_outs = llm_paged.generate(prompts, params)
        native_outs = llm_native.generate(prompts, params)
        assert len(paged_outs) == len(native_outs) == len(prompts)
        for prompt, po, no in zip(prompts, paged_outs, native_outs, strict=True):
            assert po.token_ids == no.token_ids, (
                f"prompt={prompt!r}\n"
                f"  paged       : {po.token_ids}\n"
                f"  native_eager: {no.token_ids}"
            )
    finally:
        _teardown_llm(llm_paged)
        _teardown_llm(llm_native)


# ---------------------------------------------------------------------------
# Structural correctness
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_paged_cache_respects_max_tokens(llm_paged: LLM) -> None:
    """Every output must honour its per-request max_tokens cap."""
    cap = 12
    outputs = llm_paged.generate(
        _BATCH_PROMPTS, SamplingParams(max_tokens=cap, temperature=0.0)
    )
    for out in outputs:
        assert 0 < len(out.token_ids) <= cap
        assert out.finish_reason in ("stop", "length")


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_paged_cache_output_fields_populated(llm_paged: LLM) -> None:
    """RequestOutput must be fully populated: text, token_ids, finish_reason."""
    out = llm_paged.generate(
        "Tell me a fun fact about penguins.",
        SamplingParams(max_tokens=15, temperature=0.0),
    )[0]
    assert isinstance(out.text, str) and len(out.text) > 0
    assert isinstance(out.token_ids, list) and len(out.token_ids) > 0
    assert out.finish_reason in ("stop", "length")
