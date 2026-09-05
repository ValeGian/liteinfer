"""The synchronous LLM facade over the continuous-batching engine."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from liteinfer import LLM, Phase
from liteinfer.sampling.params import SamplingParams
from tests.integration import tiny_llama


def _llm(model_dir: Path, **kwargs) -> LLM:
    return LLM(str(model_dir), device="cpu", dtype=torch.float32, **kwargs)  # type: ignore[arg-type]


def test_generate_returns_one_output_per_prompt(tiny_llama_dir: Path) -> None:
    with _llm(tiny_llama_dir) as llm:
        outputs = llm.generate(["tok5 tok6", "tok7", "tok8 tok9"], SamplingParams(max_tokens=3))
    assert len(outputs) == 3


def test_output_order_matches_input_order(tiny_llama_dir: Path) -> None:
    prompts = ["tok5", "tok6 tok7 tok8", "tok9"]
    with _llm(tiny_llama_dir) as llm:
        outputs = llm.generate(prompts, SamplingParams(max_tokens=3))
    assert [o.prompt for o in outputs] == prompts


def test_a_single_prompt_may_be_passed_as_a_string(tiny_llama_dir: Path) -> None:
    with _llm(tiny_llama_dir) as llm:
        outputs = llm.generate("tok5 tok6", SamplingParams(max_tokens=2))
    assert len(outputs) == 1


def test_generate_respects_max_tokens(tiny_llama_dir: Path) -> None:
    with _llm(tiny_llama_dir) as llm:
        outputs = llm.generate(["tok5 tok6"], SamplingParams(max_tokens=4, ignore_eos=True))
    assert len(outputs[0].token_ids) == 4


def test_hitting_max_tokens_reports_length(tiny_llama_dir: Path) -> None:
    with _llm(tiny_llama_dir) as llm:
        outputs = llm.generate(["tok5 tok6"], SamplingParams(max_tokens=3, ignore_eos=True))
    assert outputs[0].finish_reason == "length"


def test_generated_tokens_are_in_vocab_range(tiny_llama_dir: Path) -> None:
    with _llm(tiny_llama_dir) as llm:
        outputs = llm.generate(["tok5 tok6"], SamplingParams(max_tokens=4))
    assert all(0 <= t < tiny_llama.VOCAB_SIZE for t in outputs[0].token_ids)


def test_more_prompts_than_slots_all_complete(tiny_llama_dir: Path) -> None:
    prompts = [f"tok{i}" for i in range(10, 20)]
    with _llm(tiny_llama_dir, max_num_seqs=2) as llm:
        outputs = llm.generate(prompts, SamplingParams(max_tokens=2))
    assert len(outputs) == len(prompts)


def test_tokenizer_is_exposed(tiny_llama_dir: Path) -> None:
    with _llm(tiny_llama_dir) as llm:
        assert llm.tokenizer.decode(llm.tokenizer.encode("tok5")) is not None


# --- metrics ---------------------------------------------------------------


def test_stats_record_a_prefill_pass(tiny_llama_dir: Path) -> None:
    with _llm(tiny_llama_dir) as llm:
        llm.generate(["tok5 tok6"], SamplingParams(max_tokens=3, ignore_eos=True))
        phases = [step.phase for step in llm.stats.steps]
    assert Phase.PREFILL in phases


def test_stats_record_a_decode_pass_per_token(tiny_llama_dir: Path) -> None:
    # Prefill emits the first token, so max_tokens=3 leaves two decode passes.
    with _llm(tiny_llama_dir) as llm:
        llm.generate(["tok5 tok6"], SamplingParams(max_tokens=3, ignore_eos=True))
        decodes = [s for s in llm.stats.steps if s.phase is Phase.DECODE]
    assert len(decodes) == 2


def test_collect_stats_false_records_nothing(tiny_llama_dir: Path) -> None:
    with _llm(tiny_llama_dir, collect_stats=False) as llm:
        llm.generate(["tok5 tok6"], SamplingParams(max_tokens=3))
        assert llm.stats.steps == []


# --- lifecycle -------------------------------------------------------------


def test_close_is_idempotent(tiny_llama_dir: Path) -> None:
    llm = _llm(tiny_llama_dir)
    llm.close()
    llm.close()  # must not raise


def test_building_inside_a_running_loop_is_refused(tiny_llama_dir: Path) -> None:
    # LLM owns a loop; nesting one would deadlock, so it refuses loudly.
    async def build() -> None:
        _llm(tiny_llama_dir)

    with pytest.raises(RuntimeError, match="AsyncLLM"):
        asyncio.run(build())
