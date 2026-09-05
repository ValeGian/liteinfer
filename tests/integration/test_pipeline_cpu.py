# pyright: reportPrivateImportUsage=false
"""Integration tests for the async continuous-batching pipeline on CPU.

Uses randomly-initialised tiny Llama weights — validates pipeline structure
and correctness, not output quality.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest
import torch

from liteinfer import AsyncLLM
from liteinfer.sampling.params import SamplingParams
from tests.integration import tiny_llama

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
            assert 0 <= tid < tiny_llama.VOCAB_SIZE


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
            async for event in llm.stream(
                "tok2 tok3", SamplingParams(max_tokens=5, temperature=0.0)
            ):
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
                forced = torch.full((logits.shape[0], tiny_llama.VOCAB_SIZE), float("-inf"))
                forced[:, tiny_llama.EOS_ID] = 0.0
                return forced

            def _decode_forcing_eos(seqs):
                logits = original_decode(seqs)
                forced = torch.full((logits.shape[0], tiny_llama.VOCAB_SIZE), float("-inf"))
                forced[:, tiny_llama.EOS_ID] = 0.0
                return forced

            llm.engine.model_runner.prefill = _prefill_forcing_eos  # type: ignore[method-assign]
            llm.engine.model_runner.decode = _decode_forcing_eos  # type: ignore[method-assign]

            return await llm.generate("tok2 tok3", SamplingParams(max_tokens=20, temperature=0.0))

    outputs = _run(_run_test())
    assert outputs[0].finish_reason == "stop"
    assert len(outputs[0].token_ids) < 20


# ---------------------------------------------------------------------------
# Failure handling: a bad request must not take the engine down with it
# ---------------------------------------------------------------------------


def test_over_long_prompt_raises_to_its_own_caller(tiny_llama_dir: Path) -> None:
    async def run() -> None:
        async with AsyncLLM(
            str(tiny_llama_dir), device="cpu", dtype=torch.float32, max_model_len=8
        ) as llm:
            long_prompt = " ".join(f"tok{i}" for i in range(10, 40))
            with pytest.raises(ValueError, match="max_model_len"):
                await asyncio.wait_for(
                    llm.generate([long_prompt], SamplingParams(max_tokens=2)), timeout=20
                )

    _run(run())


def test_engine_still_serves_requests_after_a_bad_one(tiny_llama_dir: Path) -> None:
    async def run() -> list:
        async with AsyncLLM(
            str(tiny_llama_dir), device="cpu", dtype=torch.float32, max_model_len=8
        ) as llm:
            long_prompt = " ".join(f"tok{i}" for i in range(10, 40))
            with contextlib.suppress(ValueError):
                await llm.generate([long_prompt], SamplingParams(max_tokens=2))
            return await asyncio.wait_for(
                llm.generate(["tok5"], SamplingParams(max_tokens=2)), timeout=20
            )

    assert len(_run(run())) == 1


def test_a_failing_forward_pass_raises_to_the_caller(tiny_llama_dir: Path) -> None:
    async def run() -> None:
        async with AsyncLLM(str(tiny_llama_dir), device="cpu", dtype=torch.float32) as llm:
            llm.engine.model_runner.prefill = _raise_boom
            with pytest.raises(RuntimeError, match="boom"):
                await asyncio.wait_for(
                    llm.generate(["tok5"], SamplingParams(max_tokens=2)), timeout=20
                )

    _run(run())


def test_streaming_before_start_is_refused(tiny_llama_dir: Path) -> None:
    async def run() -> None:
        llm = AsyncLLM(str(tiny_llama_dir), device="cpu", dtype=torch.float32)
        with pytest.raises(RuntimeError, match="not running"):
            async for _ in llm.stream("tok5", SamplingParams(max_tokens=1)):
                pass

    _run(run())


def _raise_boom(*_args, **_kwargs):
    raise RuntimeError("boom")
