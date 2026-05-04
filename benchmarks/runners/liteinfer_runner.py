"""Run benchmarks against `liteinfer`."""

from __future__ import annotations

import time

from benchmarks.runners.base import GenerationResult, SamplingSpec
from liteinfer import LLM, SamplingParams
from liteinfer.engine.metrics import StepMetrics


class LiteInferRunner:
    name = "liteinfer"

    def __init__(self) -> None:
        self.llm: LLM | None = None

    def setup(self, model: str, **kwargs) -> None:
        self.llm = LLM(model=model, **kwargs)

    def generate(self, prompts: list[str], sampling: SamplingSpec) -> list[GenerationResult]:
        if self.llm is None:
            raise RuntimeError("call setup() before generate()")
        params = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            seed=sampling.seed,
        )

        results: list[GenerationResult] = []
        # liteinfer v0 processes one prompt per static batch. Time each
        # prompt individually so per-request TTFT/total are accurate.
        for prompt in prompts:
            first_step_wall: list[float] = []

            def _capture_first(step: StepMetrics, sink: list[float] = first_step_wall) -> None:
                if not sink:
                    sink.append(step.wall_time_s)

            self.llm.stats.on_step(_capture_first)

            t0 = time.perf_counter()
            outs = self.llm.generate(prompt, params)
            total = time.perf_counter() - t0

            self.llm.stats.listeners.remove(_capture_first)

            ttft = first_step_wall[0] if first_step_wall else total
            results.append(
                GenerationResult(
                    prompt=prompt,
                    output_text=outs[0].text,
                    output_token_ids=outs[0].token_ids,
                    ttft_s=ttft,
                    total_time_s=total,
                )
            )
        return results

    def teardown(self) -> None:
        self.llm = None
