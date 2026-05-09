# pyright: reportPrivateImportUsage=false
"""GPU correctness tests for static batching with B > 1.

These tests are intentionally written against observable behavior of
`LLM.generate(...)` only. They do not reach into the engine, scheduler,
or model runner internals, so the same suite remains valid across any
implementation that satisfies the static-batching contract.
"""

from __future__ import annotations

import gc

import pytest
import torch
from transformers import AutoModelForCausalLM

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

_SHORT_PROMPT = "The capital of France is"
_MEDIUM_PROMPT = "Once upon a time in a land far away, there lived a"
_LONG_PROMPT = (
    "In modern computer architecture, a hierarchy of caches sits between the "
    "processor and main memory. Each level trades off capacity for latency. "
    "The first level cache, called L1, is closest to the cores and"
)
_TINY_PROMPT = "Hi,"

_PARITY_PROMPTS = [_SHORT_PROMPT, _MEDIUM_PROMPT, _LONG_PROMPT, _TINY_PROMPT]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_dir():
    return resolve_model_path(_MODEL_ID)


@pytest.fixture(scope="module")
def hf_model(model_dir):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=_DTYPE,
        attn_implementation="eager",
        device_map=_DEVICE,
    )
    model.eval()
    yield model
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _make_llm(model_dir, *, cache_mode: str, max_num_seqs: int) -> LLM:
    return LLM(
        str(model_dir),
        device=_DEVICE,
        dtype=_DTYPE,
        cache_mode=cache_mode,
        max_num_seqs=max_num_seqs,
    )


def _teardown_llm(llm: LLM) -> None:
    if llm.engine.model_runner.model is not None:
        llm.engine.model_runner.model.cpu()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def llm_b4_no_cache(model_dir):
    llm = _make_llm(model_dir, cache_mode="none", max_num_seqs=4)
    yield llm
    _teardown_llm(llm)


@pytest.fixture(scope="module")
def llm_b4_eager(model_dir):
    llm = _make_llm(model_dir, cache_mode="eager", max_num_seqs=4)
    yield llm
    _teardown_llm(llm)


@pytest.fixture(scope="module")
def llm_b1_eager(model_dir):
    llm = _make_llm(model_dir, cache_mode="eager", max_num_seqs=1)
    yield llm
    _teardown_llm(llm)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_greedy(hf_model, prompt_token_ids: list[int], max_new_tokens: int) -> list[int]:
    input_ids = torch.tensor([prompt_token_ids], device=_DEVICE)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = hf_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    return out[0, len(prompt_token_ids) :].tolist()


def _by_request_id(outputs):
    return {out.request_id: out for out in outputs}


# ---------------------------------------------------------------------------
# Parity: batched eager-cache vs transformers
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_batched_greedy_eager_cache_matches_transformers(
    llm_b4_eager: LLM, hf_model
) -> None:
    """B=4 batched greedy must match transformers token-for-token per prompt."""
    params = SamplingParams(max_tokens=_PARITY_MAX_TOKENS, temperature=0.0)
    outputs = llm_b4_eager.generate(_PARITY_PROMPTS, params)
    assert len(outputs) == len(_PARITY_PROMPTS)

    for prompt, out in zip(_PARITY_PROMPTS, outputs, strict=True):
        assert out.prompt == prompt
        prompt_ids = llm_b4_eager.tokenizer.encode(prompt)
        expected = _hf_greedy(hf_model, prompt_ids, _PARITY_MAX_TOKENS)
        assert out.token_ids == expected, (
            f"prompt={prompt!r}\n"
            f"  liteinfer (B=4): {out.token_ids}\n"
            f"  transformers   : {expected}"
        )


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_batched_greedy_no_cache_matches_transformers(
    llm_b4_no_cache: LLM, hf_model
) -> None:
    """B=4 batched greedy with cache_mode=none must match transformers per prompt."""
    params = SamplingParams(max_tokens=_PARITY_MAX_TOKENS, temperature=0.0)
    outputs = llm_b4_no_cache.generate(_PARITY_PROMPTS, params)
    assert len(outputs) == len(_PARITY_PROMPTS)

    for prompt, out in zip(_PARITY_PROMPTS, outputs, strict=True):
        prompt_ids = llm_b4_no_cache.tokenizer.encode(prompt)
        expected = _hf_greedy(hf_model, prompt_ids, _PARITY_MAX_TOKENS)
        assert out.token_ids == expected, (
            f"prompt={prompt!r}\n"
            f"  liteinfer (B=4, no cache): {out.token_ids}\n"
            f"  transformers              : {expected}"
        )


# ---------------------------------------------------------------------------
# Parity: B=1 vs B=N (internal consistency)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_batched_b1_vs_b4_parity_eager_cache(
    llm_b1_eager: LLM, llm_b4_eager: LLM
) -> None:
    """Same prompts must produce identical token sequences regardless of batch size."""
    params = SamplingParams(max_tokens=_PARITY_MAX_TOKENS, temperature=0.0)
    out_b1 = llm_b1_eager.generate(_PARITY_PROMPTS, params)
    out_b4 = llm_b4_eager.generate(_PARITY_PROMPTS, params)

    assert len(out_b1) == len(out_b4) == len(_PARITY_PROMPTS)
    for prompt, o1, o4 in zip(_PARITY_PROMPTS, out_b1, out_b4, strict=True):
        assert o1.prompt == prompt and o4.prompt == prompt
        assert o1.token_ids == o4.token_ids, (
            f"prompt={prompt!r}\n"
            f"  B=1: {o1.token_ids}\n"
            f"  B=4: {o4.token_ids}"
        )


# ---------------------------------------------------------------------------
# Per-request max_tokens within a batch
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_batched_respects_max_tokens(llm_b4_eager: LLM) -> None:
    """Every output respects its own max_tokens cap inside a static batch."""
    cap = 12
    outputs = llm_b4_eager.generate(
        _PARITY_PROMPTS, SamplingParams(max_tokens=cap, temperature=0.0)
    )
    for out in outputs:
        assert 0 < len(out.token_ids) <= cap
        assert out.finish_reason in ("stop", "length")


# ---------------------------------------------------------------------------
# Many prompts, smaller batch — multiple static batches must complete
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_more_prompts_than_max_num_seqs(model_dir, hf_model) -> None:
    """8 prompts with max_num_seqs=2 must all complete and match transformers."""
    llm = _make_llm(model_dir, cache_mode="eager", max_num_seqs=2)
    try:
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
        outputs = llm.generate(prompts, params)
        assert len(outputs) == len(prompts)

        by_req = _by_request_id(outputs)
        assert len(by_req) == len(prompts)
        for prompt, out in zip(prompts, outputs, strict=True):
            assert out.prompt == prompt
            prompt_ids = llm.tokenizer.encode(prompt)
            expected = _hf_greedy(hf_model, prompt_ids, 8)
            assert out.token_ids == expected, (
                f"prompt={prompt!r}\n"
                f"  liteinfer (max_num_seqs=2): {out.token_ids}\n"
                f"  transformers              : {expected}"
            )
    finally:
        _teardown_llm(llm)
