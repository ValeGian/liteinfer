"""A token budget must not change what a CUDA engine generates.

On CUDA the engine packs prefill and replays decode from captured graphs, so a
chunk continuing a prompt goes through a path the CPU tests never reach:
FlashAttention's varlen entry with fewer queries than keys, reading its prefix
back out of the pool. Every test here runs the same prompts with and without a
budget small enough to split them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from liteinfer import AsyncLLM
from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_model_runner import ContinuousModelRunner
from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.sampling.params import SamplingParams

pytestmark = pytest.mark.gpu

# Lengths on both sides of the budget, and more prompts than slots, so sequences
# are admitted mid-generation and several prompts need more than one chunk.
_PROMPT_LENS = (37, 5, 21, 3, 50, 12)
_SLOTS = 3
_BUDGET = 8


def _prompt(length: int, offset: int) -> str:
    return " ".join(f"tok{2 + (offset + i) % 200}" for i in range(length))


def _generate(model_dir: Path, max_num_batched_tokens: int | None) -> list[list[int]]:
    prompts = [_prompt(length, 11 * i) for i, length in enumerate(_PROMPT_LENS)]

    async def _run():
        llm = AsyncLLM(
            str(model_dir), device="cuda", dtype=torch.bfloat16,  # type: ignore[arg-type]
            max_num_seqs=_SLOTS, max_model_len=128,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        async with llm:
            assert llm.engine.model_runner._packs_prefill, "the packed path is what this tests"
            outputs = await llm.generate(
                prompts, SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
            )
            return [list(o.token_ids) for o in outputs]

    return asyncio.run(_run())


def test_a_budgeted_cuda_engine_generates_what_an_unbudgeted_one_does(tiny_llama_dir: Path):
    """The plumbing end to end: admission mid-generation, chunking, graph-replayed decode.

    Not the parity check it looks like. The tiny random model mostly repeats its
    input token whatever attention returns, so its greedy output cannot tell a
    chunk that ignores its prefix. The logit tests below are the ones that see
    attention.
    """
    assert _generate(tiny_llama_dir, _BUDGET) == _generate(tiny_llama_dir, None)


_CHUNKED_PROMPT_LEN = 30
_CHUNKS = [7, 16, 7]
_DECODE_STEPS = 3
# How much more a chunked bf16 run may drift from single precision than the
# whole-prompt bf16 run already does. The logits reach ~60, where one bf16 ulp is
# 0.25, so any fixed tolerance is either looser than a real bug or tighter than
# the rounding; a ratio to the path being matched is neither. Measured on the
# tiny model: both runs sit 0.10-0.14 from the reference, a chunk that ignores
# its prefix 3.19 on prefill and 0.26-0.34 on decode, a context one token short
# 0.39 on prefill — so twice the whole run's error separates them.
_DRIFT_ALLOWANCE = 2.0


def _runner(model_dir: Path, dtype: torch.dtype = torch.bfloat16) -> ContinuousModelRunner:
    config = EngineConfig(
        model=str(model_dir), device="cuda", dtype=dtype,  # type: ignore[arg-type]
        max_num_seqs=1, max_model_len=64,
    )
    runner = ContinuousModelRunner(config)
    runner.load_model()
    return runner


def _sequence() -> Sequence:
    return Sequence(
        request_id="seq",
        prompt="",
        prompt_token_ids=list(range(2, 2 + _CHUNKED_PROMPT_LEN)),
        sampling_params=SamplingParams(temperature=0.0),
        status=SequenceStatus.RUNNING,
    )


def _run(model_dir: Path, chunks: list[int], dtype: torch.dtype = torch.bfloat16) -> list[torch.Tensor]:
    """Logits at the prompt's end, then after each captured decode step over fixed tokens."""
    runner, seq = _runner(model_dir, dtype), _sequence()
    assert runner._packs_prefill == (dtype == torch.bfloat16), "bf16 packs; fp32 is the padded reference"
    for count in chunks:
        logits = runner.prefill([seq], [count])
    steps = [logits.float()]
    for step in range(_DECODE_STEPS):
        seq.output_token_ids.append(3 + step)
        steps.append(runner.decode([seq]).float())
    return steps


def _drift(run: list[torch.Tensor], reference: list[torch.Tensor], step: int) -> float:
    return (run[step] - reference[step]).abs().max().item()


@pytest.fixture(scope="module")
def runs(tiny_llama_dir: Path) -> dict[str, list[torch.Tensor]]:
    """The single-precision reference, and the bf16 engine with the prompt whole and chunked.

    In bf16 on CUDA the chunked run's later chunks go through the paged kernel
    and the whole prompt through FlashAttention; in fp32 packing is off and the
    prompt goes through `sdpa`, which is what makes it a reference for both.
    """
    return {
        "reference": _run(tiny_llama_dir, [_CHUNKED_PROMPT_LEN], torch.float32),
        "whole": _run(tiny_llama_dir, [_CHUNKED_PROMPT_LEN]),
        "chunked": _run(tiny_llama_dir, _CHUNKS),
    }


def test_the_last_chunk_of_a_prompt_predicts_what_the_whole_prompt_does(runs):
    reference = runs["reference"]

    assert _drift(runs["chunked"], reference, 0) <= _DRIFT_ALLOWANCE * _drift(runs["whole"], reference, 0)


@pytest.mark.parametrize("step", range(1, _DECODE_STEPS + 1))
def test_a_captured_decode_reads_what_the_chunks_wrote(runs, step: int):
    """Graph-replayed decode walks the pool the packed chunks filled, so a misplaced token shows."""
    reference = runs["reference"]

    assert _drift(runs["chunked"], reference, step) <= _DRIFT_ALLOWANCE * _drift(runs["whole"], reference, step)
