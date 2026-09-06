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
def llm(model_dir):
    """liteinfer with the same attention kernel `hf_model` uses.

    The kernel has to be pinned on both sides, for the reason
    `tests/e2e/test_llama_gpu.py::eager_llm` documents: this test isolates
    liteinfer's *model* against the reference, and comparing two different
    kernels in bf16 measures their summation order instead. Left unpinned it
    would get whichever kernel the engine chooses for the device, which is how
    it started failing — `paged` rounds two candidate logits onto the same bf16
    value at token 11 of one prompt (20.125 against sdpa's 20.25 vs 20.125, one
    ULP apart at that magnitude) and the argmax tie-break then takes the lower
    token id. That the kernels agree is pinned separately, in fp32, in the 1B
    file.
    """
    engine = LLM(str(model_dir), device=_DEVICE, dtype=_DTYPE, attn_implementation="eager")
    yield engine
    engine.close()
    del engine
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
def test_greedy_matches_transformers(llm: LLM, hf_model, prompt: str) -> None:
    """liteinfer greedy output must match transformers token-for-token."""
    prompt_ids = llm.tokenizer.encode(prompt)
    expected = _hf_greedy(hf_model, prompt_ids, _PARITY_MAX_TOKENS)
    actual = _liteinfer_greedy(llm, prompt, _PARITY_MAX_TOKENS)
    assert actual == expected, (
        f"prompt={prompt!r}\n"
        f"  liteinfer   : {actual}\n"
        f"  transformers: {expected}"
    )


# ---------------------------------------------------------------------------
# Parity: eager cache vs transformers
# ---------------------------------------------------------------------------


