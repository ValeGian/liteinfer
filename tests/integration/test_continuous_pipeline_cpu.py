# pyright: reportPrivateImportUsage=false
"""Integration tests for the async continuous-batching pipeline on CPU.

Uses randomly-initialised tiny Llama weights — validates pipeline structure
and correctness, not output quality.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.llama.configuration_llama import LlamaConfig

from liteinfer import LLM, AsyncLLM
from liteinfer.sampling.params import SamplingParams

# ---------------------------------------------------------------------------
# Shared fixture (reuse tiny model from test_pipeline_cpu.py conventions)
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 256
_EOS_ID = 1


def _build_tiny_llama_dir(model_dir: Path) -> None:
    cfg = LlamaConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_scaling={"rope_type": "linear", "factor": 1.0},
        architectures=["LlamaForCausalLM"],
        tie_word_embeddings=True,
    )
    cfg.save_pretrained(str(model_dir))

    from liteinfer.models.llama import LlamaForCausalLM

    torch.manual_seed(0)
    with torch.device("cpu"):
        model = LlamaForCausalLM(cfg)
    model = model.to(dtype=torch.float32)

    state = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, str(model_dir / "model.safetensors"))

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
    model_dir = tmp_path_factory.mktemp("tiny_llama_async")
    _build_tiny_llama_dir(model_dir)
    return model_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously — avoids pytest-asyncio dependency."""
    return asyncio.run(coro)


def _async_llm(model_dir: Path, max_num_seqs: int = 4) -> AsyncLLM:
    return AsyncLLM(
        str(model_dir),
        device="cpu",
        dtype=torch.float32,  # type: ignore[arg-type]
        max_num_seqs=max_num_seqs,
    )


# ---------------------------------------------------------------------------
# Basic output shape and correctness
# ---------------------------------------------------------------------------


def test_continuous_pipeline_returns_one_output_per_prompt(tiny_llama_dir: Path) -> None:
    async def _run_test():
        async with _async_llm(tiny_llama_dir) as llm:
            return await llm.generate(
                ["tok2", "tok3 tok4", "tok5 tok6 tok7"],
                SamplingParams(max_tokens=4, temperature=0.0),
            )

    outputs = _run(_run_test())
    assert len(outputs) == 3


def test_continuous_pipeline_output_order_matches_input(tiny_llama_dir: Path) -> None:
    prompts = ["tok2", "tok3 tok4 tok5", "tok6"]

    async def _run_test():
        async with _async_llm(tiny_llama_dir) as llm:
            return await llm.generate(prompts, SamplingParams(max_tokens=3, temperature=0.0))

    outputs = _run(_run_test())
    for prompt, out in zip(prompts, outputs, strict=True):
        assert out.prompt == prompt


def test_continuous_pipeline_respects_max_tokens(tiny_llama_dir: Path) -> None:
    async def _run_test():
        async with _async_llm(tiny_llama_dir) as llm:
            return await llm.generate(
                ["tok2 tok3", "tok4"],
                SamplingParams(max_tokens=3, temperature=0.0),
            )

    for out in _run(_run_test()):
        assert 0 < len(out.token_ids) <= 3


def test_continuous_pipeline_tokens_in_vocab_range(tiny_llama_dir: Path) -> None:
    async def _run_test():
        async with _async_llm(tiny_llama_dir) as llm:
            return await llm.generate("tok2 tok3", SamplingParams(max_tokens=8, temperature=0.0))

    for out in _run(_run_test()):
        for tid in out.token_ids:
            assert 0 <= tid < _VOCAB_SIZE


def test_continuous_pipeline_finish_reason_length(tiny_llama_dir: Path) -> None:
    async def _run_test():
        async with _async_llm(tiny_llama_dir) as llm:
            return await llm.generate("tok2", SamplingParams(max_tokens=3, temperature=0.0))

    out = _run(_run_test())[0]
    # Random weights almost never hit EOS in 3 steps.
    assert out.finish_reason == "length"


# ---------------------------------------------------------------------------
# Parity: continuous batching must produce same greedy tokens as static B=1
# ---------------------------------------------------------------------------


def test_continuous_greedy_parity_vs_static_b1(tiny_llama_dir: Path) -> None:
    """Greedy output from AsyncLLM must match LLM (static, B=1, paged cache)."""
    prompts = ["tok2 tok3 tok4", "tok5", "tok6 tok7 tok8 tok9"]
    params = SamplingParams(max_tokens=8, temperature=0.0)

    # Static reference (B=1, paged cache for fair comparison)
    llm_static = LLM(
        str(tiny_llama_dir),
        device="cpu",
        dtype=torch.float32,  # type: ignore[arg-type]
        cache_mode="paged",
        max_num_seqs=1,
        num_gpu_blocks=256,
    )
    static_by_prompt = {
        out.prompt: out.token_ids for out in llm_static.generate(prompts, params)
    }

    async def _run_async():
        async with _async_llm(tiny_llama_dir, max_num_seqs=1) as llm:
            return await llm.generate(prompts, params)

    for out in _run(_run_async()):
        expected = static_by_prompt[out.prompt]
        assert out.token_ids == expected, (
            f"prompt={out.prompt!r}\n"
            f"  continuous: {out.token_ids}\n"
            f"  static:     {expected}"
        )


# ---------------------------------------------------------------------------
# Continuous slot-filling: early-finishers don't block others
# ---------------------------------------------------------------------------


def test_continuous_pipeline_processes_more_seqs_than_max_num_seqs(tiny_llama_dir: Path) -> None:
    """With max_num_seqs=1 and 5 prompts, all 5 complete (slot freed continuously)."""
    prompts = [f"tok{i + 2}" for i in range(5)]

    async def _run_test():
        async with _async_llm(tiny_llama_dir, max_num_seqs=1) as llm:
            return await llm.generate(prompts, SamplingParams(max_tokens=4, temperature=0.0))

    outputs = _run(_run_test())
    assert len(outputs) == 5
    for out in outputs:
        assert 0 < len(out.token_ids) <= 4


# ---------------------------------------------------------------------------
# Streaming API
# ---------------------------------------------------------------------------


def test_stream_yields_events_before_completion(tiny_llama_dir: Path) -> None:
    """stream() must yield at least one intermediate event before is_finished."""
    intermediate_seen = False

    async def _run_test():
        nonlocal intermediate_seen
        async with _async_llm(tiny_llama_dir) as llm:
            async for event in llm.stream(
                "tok2 tok3 tok4 tok5", SamplingParams(max_tokens=6, temperature=0.0)
            ):
                if not event.is_finished:
                    intermediate_seen = True

    _run(_run_test())
    assert intermediate_seen, "stream() never yielded an intermediate (non-finished) event"


def test_stream_final_event_is_finished(tiny_llama_dir: Path) -> None:
    last_event = None

    async def _run_test():
        nonlocal last_event
        async with _async_llm(tiny_llama_dir) as llm:
            async for event in llm.stream("tok2", SamplingParams(max_tokens=3, temperature=0.0)):
                last_event = event

    _run(_run_test())
    assert last_event is not None
    assert last_event.is_finished
    assert last_event.finish_reason in ("stop", "length")


def test_stream_cumulative_tokens_grow_monotonically(tiny_llama_dir: Path) -> None:
    token_counts = []

    async def _run_test():
        async with _async_llm(tiny_llama_dir) as llm:
            async for event in llm.stream("tok2 tok3", SamplingParams(max_tokens=5, temperature=0.0)):
                token_counts.append(len(event.output_token_ids))

    _run(_run_test())
    assert token_counts == sorted(token_counts), "token counts must be non-decreasing"
    assert token_counts[-1] >= 1


def test_stream_concurrent_requests_both_complete(tiny_llama_dir: Path) -> None:
    """Two concurrent stream() calls must both complete."""
    async def _collect(llm, prompt):
        events = []
        async for event in llm.stream(prompt, SamplingParams(max_tokens=4, temperature=0.0)):
            events.append(event)
        return events

    async def _run_test():
        async with _async_llm(tiny_llama_dir, max_num_seqs=4) as llm:
            return await asyncio.gather(
                _collect(llm, "tok2 tok3"),
                _collect(llm, "tok4 tok5 tok6"),
            )

    results_a, results_b = _run(_run_test())
    assert results_a[-1].is_finished
    assert results_b[-1].is_finished


# ---------------------------------------------------------------------------
# EOS stops generation
# ---------------------------------------------------------------------------


def test_continuous_pipeline_stops_on_eos(tiny_llama_dir: Path) -> None:
    """When the model emits EOS, generation stops before max_tokens."""
    async def _run_test():
        async with _async_llm(tiny_llama_dir) as llm:
            original_prefill = llm.engine.model_runner.prefill
            original_decode = llm.engine.model_runner.decode

            def _prefill_forcing_eos(seqs):
                logits = original_prefill(seqs)
                forced = torch.full((logits.shape[0], _VOCAB_SIZE), float("-inf"))
                forced[:, _EOS_ID] = 0.0
                return forced

            def _decode_forcing_eos(seqs):
                logits = original_decode(seqs)
                forced = torch.full((logits.shape[0], _VOCAB_SIZE), float("-inf"))
                forced[:, _EOS_ID] = 0.0
                return forced

            llm.engine.model_runner.prefill = _prefill_forcing_eos  # type: ignore[method-assign]
            llm.engine.model_runner.decode = _decode_forcing_eos  # type: ignore[method-assign]

            return await llm.generate("tok2 tok3", SamplingParams(max_tokens=20, temperature=0.0))

    outputs = _run(_run_test())
    assert outputs[0].finish_reason == "stop"
    assert len(outputs[0].token_ids) < 20
