"""Benchmark runner for liteinfer's async continuous-batching engine."""

from __future__ import annotations

import asyncio
import time

from benchmarks.runners.base import GenerationResult, SamplingSpec
from liteinfer import AsyncLLM
from liteinfer.sampling.params import SamplingParams


class LiteInferContinuousRunner:
    """liteinfer with async continuous batching.

    Requests are submitted concurrently via ``AsyncLLM.stream()``. Each
    stream() call is an independent coroutine, so the engine processes them
    in parallel under its continuous-batching policy.

    TTFT is measured per-request from submission time to first token event.
    """

    name = "liteinfer-continuous"
    batch_size = 4  # default max_num_seqs

    def __init__(self) -> None:
        self._model: str = ""
        self._kwargs: dict = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._llm: AsyncLLM | None = None

    def setup(self, model: str, **kwargs) -> None:
        kwargs.setdefault("max_num_seqs", 4)
        self._model = model
        self._kwargs = kwargs
        self._loop = asyncio.new_event_loop()
        self._llm = AsyncLLM(model, **kwargs)
        self._loop.run_until_complete(self._llm.start())

    def generate(self, prompts: list[str], sampling: SamplingSpec) -> list[GenerationResult]:
        assert self._loop is not None and self._llm is not None
        params = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            seed=sampling.seed,
        )
        return self._loop.run_until_complete(self._generate_async(prompts, params))

    async def _generate_async(self, prompts: list[str], params: SamplingParams) -> list[GenerationResult]:
        assert self._llm is not None
        t0 = time.perf_counter()

        async def _collect(prompt: str) -> GenerationResult:
            ttft: float | None = None
            final = None
            async for event in self._llm.stream(prompt, params):  # type: ignore[union-attr]
                if ttft is None:
                    ttft = time.perf_counter() - t0
                final = event
            assert final is not None
            return GenerationResult(
                prompt=prompt,
                output_text=final.text,
                output_token_ids=final.output_token_ids,
                ttft_s=ttft if ttft is not None else 0.0,
                total_time_s=time.perf_counter() - t0,
            )

        return list(await asyncio.gather(*(_collect(p) for p in prompts)))

    def teardown(self) -> None:
        import gc

        import torch

        if self._loop is not None and self._llm is not None:
            self._loop.run_until_complete(self._llm.stop())
        if self._loop is not None:
            self._loop.close()
        self._llm = None
        self._loop = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
