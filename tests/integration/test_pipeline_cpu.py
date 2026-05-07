# pyright: reportPrivateImportUsage=false, reportCallIssue=false
"""Integration tests for the full LLM pipeline on CPU with a tiny fake Llama model.

Uses randomly-initialised weights — tests structure and correctness of the
inference pipeline, not output quality.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.llama.configuration_llama import LlamaConfig

from liteinfer import LLM
from liteinfer.sampling.params import SamplingParams

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 256
_EOS_ID = 1


def _build_tiny_llama_dir(model_dir: Path) -> None:
    """Populate *model_dir* with a tiny Llama model and a minimal tokenizer."""
    cfg = LlamaConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        # "default" rope_type is not in ROPE_INIT_FUNCTIONS for this
        # transformers version; use "linear" with factor=1 (identity).
        rope_scaling={"rope_type": "linear", "factor": 1.0},
        architectures=["LlamaForCausalLM"],
        tie_word_embeddings=True,
    )
    cfg.save_pretrained(str(model_dir))

    # Build model on CPU with a fixed seed for reproducibility.
    from liteinfer.models.llama import LlamaForCausalLM

    torch.manual_seed(0)
    with torch.device("cpu"):
        model = LlamaForCausalLM(cfg)
    model = model.to(dtype=torch.float32)

    # lm_head.weight is tied to embed_tokens.weight — omit it from the
    # checkpoint so the loader can exercise the tied-weight resolution path.
    state = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, str(model_dir / "model.safetensors"))

    # Minimal word-level tokenizer: vocab entries are "tok<i>" for i in
    # [2, 256), plus <unk>=0 and <eos>=1.  Unknown words map to 0, which
    # is a valid embedding index.
    vocab: dict[str, int] = {"<unk>": 0, "<eos>": _EOS_ID}
    for i in range(2, _VOCAB_SIZE):
        vocab[f"tok{i}"] = i
    hf_tok = HFTokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    hf_tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=hf_tok,
        eos_token="<eos>",
        unk_token="<unk>",
    )
    fast_tok.save_pretrained(str(model_dir))


@pytest.fixture(scope="module")
def tiny_llama_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model_dir = tmp_path_factory.mktemp("tiny_llama")
    _build_tiny_llama_dir(model_dir)
    return model_dir


def _make_llm(model_dir: Path, cache_mode: str = "none") -> LLM:
    return LLM(str(model_dir), device="cpu", dtype=torch.float32, cache_mode=cache_mode)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_no_cache_returns_request_output(tiny_llama_dir: Path) -> None:
    llm = _make_llm(tiny_llama_dir, cache_mode="none")
    outputs = llm.generate("hello world", SamplingParams(max_tokens=4, temperature=0.0))
    assert len(outputs) == 1
    out = outputs[0]
    assert out.request_id == "req-0"
    assert out.prompt == "hello world"
    assert isinstance(out.text, str)
    assert isinstance(out.token_ids, list)


def test_pipeline_no_cache_respects_max_tokens(tiny_llama_dir: Path) -> None:
    llm = _make_llm(tiny_llama_dir, cache_mode="none")
    for max_tokens in (1, 3, 7):
        out = llm.generate("hello", SamplingParams(max_tokens=max_tokens, temperature=0.0))[0]
        assert len(out.token_ids) <= max_tokens


def test_pipeline_no_cache_finish_reason_length(tiny_llama_dir: Path) -> None:
    llm = _make_llm(tiny_llama_dir, cache_mode="none")
    out = llm.generate("hello", SamplingParams(max_tokens=3, temperature=0.0))[0]
    # Random weights almost certainly won't hit EOS in 3 steps.
    assert out.finish_reason == "length"


def test_pipeline_no_cache_token_ids_in_vocab_range(tiny_llama_dir: Path) -> None:
    llm = _make_llm(tiny_llama_dir, cache_mode="none")
    out = llm.generate("hello world", SamplingParams(max_tokens=8, temperature=0.0))[0]
    for token_id in out.token_ids:
        assert 0 <= token_id < _VOCAB_SIZE


def test_pipeline_eager_cache_returns_request_output(tiny_llama_dir: Path) -> None:
    llm = _make_llm(tiny_llama_dir, cache_mode="eager")
    outputs = llm.generate("hello world", SamplingParams(max_tokens=4, temperature=0.0))
    assert len(outputs) == 1
    out = outputs[0]
    assert isinstance(out.token_ids, list)
    assert len(out.token_ids) <= 4


def test_pipeline_eager_cache_token_ids_in_vocab_range(tiny_llama_dir: Path) -> None:
    llm = _make_llm(tiny_llama_dir, cache_mode="eager")
    out = llm.generate("hello world", SamplingParams(max_tokens=8, temperature=0.0))[0]
    for token_id in out.token_ids:
        assert 0 <= token_id < _VOCAB_SIZE


def test_pipeline_greedy_parity_no_cache_vs_eager_cache(tiny_llama_dir: Path) -> None:
    """Greedy decoding must produce identical token sequences regardless of cache mode."""
    llm_no_cache = _make_llm(tiny_llama_dir, cache_mode="none")
    llm_eager = _make_llm(tiny_llama_dir, cache_mode="eager")
    params = SamplingParams(max_tokens=10, temperature=0.0)
    prompt = "tok2 tok3 tok4"
    out_nc = llm_no_cache.generate(prompt, params)[0]
    out_ec = llm_eager.generate(prompt, params)[0]
    assert out_nc.token_ids == out_ec.token_ids, (
        f"cache-mode mismatch: no_cache={out_nc.token_ids} eager={out_ec.token_ids}"
    )


def test_pipeline_multiple_prompts(tiny_llama_dir: Path) -> None:
    llm = _make_llm(tiny_llama_dir, cache_mode="none")
    prompts = ["hello", "world", "foo bar"]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=3, temperature=0.0))
    assert len(outputs) == len(prompts)
    for out in outputs:
        assert len(out.token_ids) <= 3


def test_pipeline_stops_on_eos(tiny_llama_dir: Path) -> None:
    """When the model emits EOS, generation stops before max_tokens."""
    llm = _make_llm(tiny_llama_dir, cache_mode="none")
    original_execute = llm.engine.model_runner.execute

    def _execute_forcing_eos(scheduled, is_new_batch):
        logits, n_tokens = original_execute(scheduled, is_new_batch)
        if not is_new_batch:
            # Override logits so EOS has the highest score on every decode step.
            forced = torch.full((1, _VOCAB_SIZE), float("-inf"))
            forced[0, _EOS_ID] = 0.0
            logits = forced
        return logits, n_tokens

    llm.engine.model_runner.execute = _execute_forcing_eos  # type: ignore[method-assign]
    out = llm.generate("hello", SamplingParams(max_tokens=20, temperature=0.0))[0]

    assert out.finish_reason == "stop"
    assert len(out.token_ids) < 20
