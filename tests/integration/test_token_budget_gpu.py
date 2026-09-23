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
_DECODE_STEPS = 3
# bf16 logits of a model whose attention outputs are small: two paths that sum
# the same keys in a different order land within a few ulps at this magnitude.
_TOLERANCE = {"rtol": 0, "atol": 2**-5}


def _runner(model_dir: Path) -> ContinuousModelRunner:
    config = EngineConfig(
        model=str(model_dir), device="cuda", dtype=torch.bfloat16,  # type: ignore[arg-type]
        max_num_seqs=1, max_model_len=64,
    )
    runner = ContinuousModelRunner(config)
    runner.load_model()
    assert runner._packs_prefill and runner.captured_decode_widths == [], "packed, graphs pending"
    return runner


def _sequence() -> Sequence:
    return Sequence(
        request_id="seq",
        prompt="",
        prompt_token_ids=list(range(2, 2 + _CHUNKED_PROMPT_LEN)),
        sampling_params=SamplingParams(temperature=0.0),
        status=SequenceStatus.RUNNING,
    )


def _prefill_logits(runner: ContinuousModelRunner, seq: Sequence, chunks: list[int]) -> torch.Tensor:
    """Logits at the end of the prompt, prefilled in `chunks` packed passes."""
    for count in chunks:
        logits = runner.prefill([seq], [count])
    return logits


def _decode_logits(model_dir: Path, chunks: list[int]) -> list[torch.Tensor]:
    """A few captured decode steps over fixed tokens, after prefilling in `chunks`."""
    runner, seq = _runner(model_dir), _sequence()
    _prefill_logits(runner, seq, chunks)
    steps = []
    for step in range(_DECODE_STEPS):
        seq.output_token_ids.append(3 + step)
        steps.append(runner.decode([seq]))
    return steps


def test_the_last_chunk_of_a_prompt_predicts_what_the_whole_prompt_does(tiny_llama_dir: Path):
    whole = _prefill_logits(_runner(tiny_llama_dir), _sequence(), [_CHUNKED_PROMPT_LEN])
    chunked = _prefill_logits(_runner(tiny_llama_dir), _sequence(), [7, 16, 7])

    torch.testing.assert_close(chunked, whole, **_TOLERANCE)


@pytest.mark.parametrize("step", range(_DECODE_STEPS))
def test_a_captured_decode_reads_what_the_chunks_wrote(tiny_llama_dir: Path, step: int):
    """Graph-replayed decode walks the pool the packed chunks filled, so a misplaced token shows."""
    whole = _decode_logits(tiny_llama_dir, [_CHUNKED_PROMPT_LEN])
    chunked = _decode_logits(tiny_llama_dir, [7, 16, 7])

    torch.testing.assert_close(chunked[step], whole[step], **_TOLERANCE)
