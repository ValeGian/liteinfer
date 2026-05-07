# pyright: reportPrivateImportUsage=false
"""GPU correctness and e2e generation tests against meta-llama/Llama-3.2-1B-Instruct.

Parity strategy: encode the prompt with liteinfer's tokenizer
(add_special_tokens=False), feed those exact token IDs to transformers
generate(), and compare the resulting token sequences.  This keeps the
test independent of any special-token handling differences.

Run:
    pytest tests/e2e/test_llama_gpu.py -m "gpu and e2e" -v
"""

from __future__ import annotations

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
_PARITY_MAX_TOKENS = 20  # enough to catch divergence, fast enough for CI

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
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def llm_no_cache(model_dir):
    llm = LLM(str(model_dir), device=_DEVICE, dtype=_DTYPE, cache_mode="none")
    yield llm
    del llm
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def llm_eager_cache(model_dir):
    llm = LLM(str(model_dir), device=_DEVICE, dtype=_DTYPE, cache_mode="eager")
    yield llm
    del llm
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_greedy(hf_model, prompt_token_ids: list[int], max_new_tokens: int) -> list[int]:
    """Run transformers greedy generation from the given token IDs."""
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
        f"  liteinfer : {actual}\n"
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
        f"  liteinfer : {actual}\n"
        f"  transformers: {expected}"
    )


# ---------------------------------------------------------------------------
# Parity: no-cache vs eager-cache (internal consistency)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("prompt", _PARITY_PROMPTS)
def test_greedy_no_cache_matches_eager_cache(
    llm_no_cache: LLM, llm_eager_cache: LLM, prompt: str
) -> None:
    """Both cache modes must produce the same token sequence under greedy decoding."""
    tokens_no_cache = _liteinfer_greedy(llm_no_cache, prompt, _PARITY_MAX_TOKENS)
    tokens_eager = _liteinfer_greedy(llm_eager_cache, prompt, _PARITY_MAX_TOKENS)
    assert tokens_no_cache == tokens_eager, (
        f"prompt={prompt!r}\n"
        f"  no_cache   : {tokens_no_cache}\n"
        f"  eager_cache: {tokens_eager}"
    )


# ---------------------------------------------------------------------------
# E2E generation: structural correctness
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_generation_output_fields_populated(llm_no_cache: LLM) -> None:
    outputs = llm_no_cache.generate(
        "Tell me a fun fact about penguins.",
        SamplingParams(max_tokens=15, temperature=0.0),
    )
    assert len(outputs) == 1
    out = outputs[0]
    assert out.prompt == "Tell me a fun fact about penguins."
    assert isinstance(out.text, str) and len(out.text) > 0
    assert isinstance(out.token_ids, list) and len(out.token_ids) > 0
    assert out.finish_reason in ("stop", "length")


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_generation_respects_max_tokens(llm_no_cache: LLM) -> None:
    for max_tokens in (1, 5, 15):
        out = llm_no_cache.generate(
            "Hello, my name is",
            SamplingParams(max_tokens=max_tokens, temperature=0.0),
        )[0]
        assert len(out.token_ids) <= max_tokens, (
            f"max_tokens={max_tokens} but got {len(out.token_ids)} tokens"
        )


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_generation_multiple_prompts(llm_no_cache: LLM) -> None:
    prompts = [
        "The sky is",
        "Water boils at",
        "The fastest animal on earth is",
    ]
    outputs = llm_no_cache.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0))
    assert len(outputs) == len(prompts)
    for i, (out, prompt) in enumerate(zip(outputs, prompts)):
        assert out.prompt == prompt, f"prompt mismatch at index {i}"
        assert len(out.token_ids) > 0
        assert len(out.token_ids) <= 8


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_generation_token_ids_in_vocab_range(llm_no_cache: LLM) -> None:
    out = llm_no_cache.generate(
        "Once upon a time",
        SamplingParams(max_tokens=20, temperature=0.0),
    )[0]
    vocab_size = llm_no_cache.tokenizer.vocab_size
    for token_id in out.token_ids:
        assert 0 <= token_id < vocab_size, (
            f"token_id {token_id} out of range [0, {vocab_size})"
        )


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
def test_generation_eager_cache_output_fields_populated(llm_eager_cache: LLM) -> None:
    out = llm_eager_cache.generate(
        "The meaning of life is",
        SamplingParams(max_tokens=10, temperature=0.0),
    )[0]
    assert isinstance(out.text, str) and len(out.text) > 0
    assert 0 < len(out.token_ids) <= 10
    assert out.finish_reason in ("stop", "length")
