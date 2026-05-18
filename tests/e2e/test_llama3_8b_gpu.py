# pyright: reportPrivateImportUsage=false
"""GPU parity tests against meta-llama/Meta-Llama-3-8B-Instruct.

Parity strategy: encode the prompt with liteinfer's tokenizer
(add_special_tokens=False), feed those exact token IDs to transformers
generate(), and compare the resulting token sequences.

Run:
    pytest tests/e2e/test_llama3_8b_gpu.py -m "gpu and e2e" -v
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

_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
_DEVICE = "cuda:0"
_DTYPE = torch.bfloat16
_PARITY_MAX_TOKENS = 20

_PARITY_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time in a land far away,",
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_dir():
    return resolve_model_path(_MODEL_ID)


@pytest.fixture()
def hf_model(model_dir):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=_DTYPE,
        attn_implementation="eager",
        device_map=_DEVICE,
    )
    model.eval()
    yield model
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture()
def llm_no_cache(model_dir):
    llm = LLM(str(model_dir), device=_DEVICE, dtype=_DTYPE, cache_mode="none")
    yield llm
    if llm.engine.model_runner.model is not None:
        llm.engine.model_runner.model.cpu()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture()
def llm_eager_cache(model_dir):
    llm = LLM(str(model_dir), device=_DEVICE, dtype=_DTYPE, cache_mode="eager")
    yield llm
    if llm.engine.model_runner.model is not None:
        llm.engine.model_runner.model.cpu()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture()
def llm_native_eager_cache(model_dir):
    llm = LLM(str(model_dir), device=_DEVICE, dtype=_DTYPE, cache_mode="native_eager")
    yield llm
    if llm.engine.model_runner.model is not None:
        llm.engine.model_runner.model.cpu()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


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
    return out[0, len(prompt_token_ids):].tolist()


def _liteinfer_greedy(llm: LLM, prompt: str, max_tokens: int) -> list[int]:
    return llm.generate(prompt, SamplingParams(max_tokens=max_tokens, temperature=0.0))[0].token_ids


# ---------------------------------------------------------------------------
# Parity: no-cache vs transformers
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("prompt", _PARITY_PROMPTS)
def test_greedy_no_cache_matches_transformers(llm_no_cache: LLM, hf_model, prompt: str) -> None:
    """liteinfer (cache_mode=none) greedy output must match transformers token-for-token."""
    prompt_ids = llm_no_cache.tokenizer.encode(prompt)
    expected = _hf_greedy(hf_model, prompt_ids, _PARITY_MAX_TOKENS)
    actual = _liteinfer_greedy(llm_no_cache, prompt, _PARITY_MAX_TOKENS)
    assert actual == expected, (
        f"prompt={prompt!r}\n"
        f"  liteinfer   : {actual}\n"
        f"  transformers: {expected}"
    )


# ---------------------------------------------------------------------------
# Parity: eager cache vs transformers
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("prompt", _PARITY_PROMPTS)
def test_greedy_eager_cache_matches_transformers(llm_eager_cache: LLM, hf_model, prompt: str) -> None:
    """liteinfer (cache_mode=eager) greedy output must match transformers token-for-token."""
    prompt_ids = llm_eager_cache.tokenizer.encode(prompt)
    expected = _hf_greedy(hf_model, prompt_ids, _PARITY_MAX_TOKENS)
    actual = _liteinfer_greedy(llm_eager_cache, prompt, _PARITY_MAX_TOKENS)
    assert actual == expected, (
        f"prompt={prompt!r}\n"
        f"  liteinfer   : {actual}\n"
        f"  transformers: {expected}"
    )


# ---------------------------------------------------------------------------
# Parity: native_eager cache vs transformers
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("prompt", _PARITY_PROMPTS)
def test_greedy_native_eager_cache_matches_transformers(llm_native_eager_cache: LLM, hf_model, prompt: str) -> None:
    """liteinfer (cache_mode=native_eager) greedy output must match transformers token-for-token."""
    prompt_ids = llm_native_eager_cache.tokenizer.encode(prompt)
    expected = _hf_greedy(hf_model, prompt_ids, _PARITY_MAX_TOKENS)
    actual = _liteinfer_greedy(llm_native_eager_cache, prompt, _PARITY_MAX_TOKENS)
    assert actual == expected, (
        f"prompt={prompt!r}\n"
        f"  liteinfer   : {actual}\n"
        f"  transformers: {expected}"
    )


# ---------------------------------------------------------------------------
# Internal consistency
# ---------------------------------------------------------------------------
# These tests compare two liteinfer cache modes without HF. Loading both
# simultaneously exceeds single-GPU VRAM, so each mode is loaded, sampled,
# and unloaded before the next one is loaded.


def _liteinfer_greedy_ephemeral(model_dir, cache_mode: str, prompt: str, max_tokens: int) -> list[int]:
    """Load a single liteinfer instance, generate, then fully release VRAM."""
    llm = LLM(str(model_dir), device=_DEVICE, dtype=_DTYPE, cache_mode=cache_mode)
    try:
        return _liteinfer_greedy(llm, prompt, max_tokens)
    finally:
        if llm.engine.model_runner.model is not None:
            llm.engine.model_runner.model.cpu()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("prompt", _PARITY_PROMPTS)
def test_greedy_no_cache_matches_eager_cache(model_dir, prompt: str) -> None:
    """Both cache modes must produce the same token sequence under greedy decoding."""
    tokens_no_cache = _liteinfer_greedy_ephemeral(model_dir, "none", prompt, _PARITY_MAX_TOKENS)
    tokens_eager = _liteinfer_greedy_ephemeral(model_dir, "eager", prompt, _PARITY_MAX_TOKENS)
    assert tokens_no_cache == tokens_eager, (
        f"prompt={prompt!r}\n"
        f"  no_cache   : {tokens_no_cache}\n"
        f"  eager_cache: {tokens_eager}"
    )


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("prompt", _PARITY_PROMPTS)
def test_greedy_eager_cache_matches_native_eager_cache(model_dir, prompt: str) -> None:
    """eager (DynamicCache) and native_eager must produce identical token sequences."""
    tokens_eager = _liteinfer_greedy_ephemeral(model_dir, "eager", prompt, _PARITY_MAX_TOKENS)
    tokens_native = _liteinfer_greedy_ephemeral(model_dir, "native_eager", prompt, _PARITY_MAX_TOKENS)
    assert tokens_eager == tokens_native, (
        f"prompt={prompt!r}\n"
        f"  eager       : {tokens_eager}\n"
        f"  native_eager: {tokens_native}"
    )
