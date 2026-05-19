"""In-process liteinfer adapter.

Uses AsyncLLM for throughput benchmarks (asyncio.Semaphore to limit concurrency)
and LLM for latency benchmarks (sequential, batch_size=1).
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from benchmarks.adapters.base import BenchmarkSample, RequestMeasurement

_WARMUP_PROMPT = "Hello"
_WARMUP_MAX_TOKENS = 16
_WARMUP_COUNT = 4
_THROUGHPUT_MAX_SEQS = 32


class LiteInferAdapter:
    name = "liteinfer"

    def __enter__(self) -> LiteInferAdapter:
        return self

    def __exit__(self, *_) -> None:
        pass

    def run(
        self,
        samples: list[BenchmarkSample],
        model: str,
        benchmark_type: Literal["throughput", "latency"],
    ) -> tuple[list[RequestMeasurement], float]:
        if benchmark_type == "throughput":
            return asyncio.run(self._run_throughput(samples, model))
        return self._run_latency(samples, model)

    # ------------------------------------------------------------------
    # Throughput mode
    # ------------------------------------------------------------------

    async def _run_throughput(
        self,
        samples: list[BenchmarkSample],
        model: str,
    ) -> tuple[list[RequestMeasurement], float]:
        from liteinfer import AsyncLLM
        from liteinfer.sampling import SamplingParams

        llm = AsyncLLM(model=model, cache_mode="paged", max_num_seqs=_THROUGHPUT_MAX_SEQS)

        # Warmup: 4 requests before starting the wall clock
        warmup_params = SamplingParams(max_tokens=_WARMUP_MAX_TOKENS, min_tokens=_WARMUP_MAX_TOKENS)
        warmup_tasks = [
            asyncio.create_task(self._collect_stream(llm, _WARMUP_PROMPT, warmup_params))
            for _ in range(_WARMUP_COUNT)
        ]
        await asyncio.gather(*warmup_tasks)

        # Real benchmark: all coroutines created, semaphore limits concurrency
        semaphore = asyncio.Semaphore(_THROUGHPUT_MAX_SEQS)
        wall_start = time.perf_counter()

        async def _run_with_semaphore(
            sample: BenchmarkSample, idx: int
        ) -> RequestMeasurement:
            params = SamplingParams(
                max_tokens=sample.forced_output_token_count,
                min_tokens=sample.forced_output_token_count,
                ignore_eos=True,
            )
            async with semaphore:
                submit_time = time.perf_counter()
                first_token_time: float | None = None
                token_times: list[float] = []

                async for _event in llm.stream(sample.prompt, params):
                    now = time.perf_counter()
                    token_times.append(now)
                    if first_token_time is None:
                        first_token_time = now

                end_time = time.perf_counter()

                return RequestMeasurement(
                    sample_index=idx,
                    input_token_count=sample.input_token_count,
                    output_token_count=len(token_times),
                    ttft_s=(first_token_time - submit_time) if first_token_time else 0.0,
                    token_timestamps_s=token_times if token_times else None,
                    e2e_s=end_time - submit_time,
                )

        tasks = [
            asyncio.create_task(_run_with_semaphore(sample, i))
            for i, sample in enumerate(samples)
        ]
        measurements = await asyncio.gather(*tasks)
        wall_time_s = time.perf_counter() - wall_start

        return list(measurements), wall_time_s

    async def _collect_stream(self, llm, prompt: str, params) -> None:
        async for _ in llm.stream(prompt, params):
            pass

    # ------------------------------------------------------------------
    # Latency mode
    # ------------------------------------------------------------------

    def _run_latency(
        self,
        samples: list[BenchmarkSample],
        model: str,
    ) -> tuple[list[RequestMeasurement], float]:
        from liteinfer import LLM
        from liteinfer.sampling import SamplingParams

        llm = LLM(model=model, cache_mode="paged", max_num_seqs=1)

        # Warmup
        warmup_params = SamplingParams(max_tokens=_WARMUP_MAX_TOKENS, min_tokens=_WARMUP_MAX_TOKENS)
        for _ in range(_WARMUP_COUNT):
            llm.generate([_WARMUP_PROMPT], warmup_params)

        measurements: list[RequestMeasurement] = []
        wall_start = time.perf_counter()

        for idx, sample in enumerate(samples):
            params = SamplingParams(
                max_tokens=sample.forced_output_token_count,
                min_tokens=sample.forced_output_token_count,
                ignore_eos=True,
            )

            # Drive the engine step loop manually to capture per-step timestamps.
            req_id = f"bench-{idx}"
            llm.engine.add_request(req_id, sample.prompt, params)

            step_times: list[float] = []
            start = time.perf_counter()
            output_token_ids: list[int] = []

            while llm.engine.has_unfinished_requests():
                finished = llm.engine.step()
                step_times.append(time.perf_counter())
                for seq in finished:
                    if seq.request_id == req_id:
                        output_token_ids = list(seq.output_token_ids)

            end = time.perf_counter()
            ttft_s = (step_times[0] - start) if step_times else (end - start)

            measurements.append(
                RequestMeasurement(
                    sample_index=idx,
                    input_token_count=sample.input_token_count,
                    output_token_count=len(output_token_ids),
                    ttft_s=ttft_s,
                    token_timestamps_s=step_times if step_times else None,
                    e2e_s=end - start,
                )
            )

        wall_time_s = time.perf_counter() - wall_start
        return measurements, wall_time_s
