"""Run benchmarks against `liteinfer`."""

from __future__ import annotations

import time
from itertools import count

from benchmarks.runners.base import GenerationResult, SamplingSpec
from liteinfer import LLM, SamplingParams
from liteinfer.engine.metrics import Phase, StepMetrics


class LiteInferRunner:
    name = "liteinfer"

    def __init__(self) -> None:
        self.llm: LLM | None = None
        self._req_id_counter = count(0)

    def setup(self, model: str, **kwargs) -> None:
        # max_num_seqs=1 keeps the scheduler from batching waiting requests together.
        # All submitted prompts queue in engine.waiting; the scheduler picks one at a
        # time, which is the only mode supported by the v0 model runner.
        kwargs.setdefault("max_num_seqs", 1)
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

        # Submit all prompts to the engine queue at once (t0 is the batch arrival time).
        # With B=1 the engine processes them sequentially; for a single-prompt latency
        # call there is no queue, and for a multi-prompt throughput call later requests
        # queue behind earlier ones — both are the correct semantics for their workload.
        request_ids: list[str] = []
        t0 = time.perf_counter()
        for prompt in prompts:
            req_id = f"bench-{next(self._req_id_counter)}"
            self.llm.engine.add_request(req_id, prompt, params)
            request_ids.append(req_id)

        # TTFT is the wall-clock time from t0 to when the first token for each request
        # is ready. We capture time.perf_counter() inside the step-listener, which fires
        # after the CUDA-synced forward pass + sampling + _apply_sampled — the full path
        # a token takes before it becomes observable. step.wall_time_s would only capture
        # the GPU compute portion and would miss Python/scheduling overhead.
        #
        # With B=1 FIFO, first steps happen in submission order, so first_token_times[i]
        # corresponds to request_ids[i].
        #
        # liteinfer v0 runs in no-cache (RECOMPUTE) mode by default: every step feeds
        # the full sequence and all steps have Phase.RECOMPUTE. In RECOMPUTE mode,
        # input_tokens grows monotonically within a request (prompt + generated tokens),
        # so a drop in input_tokens signals the first step of the next request. In KV-cache
        # mode the first step has Phase.PREFILL, which we detect directly.
        first_token_times: list[float] = []
        last_input_tokens: float = float("inf")  # float so first step always satisfies ≤

        def _on_step(step: StepMetrics) -> None:
            nonlocal last_input_tokens
            is_first_step_of_request = step.phase == Phase.PREFILL or (
                step.phase == Phase.RECOMPUTE and step.input_tokens <= last_input_tokens
            )
            if is_first_step_of_request:
                first_token_times.append(time.perf_counter())
            last_input_tokens = step.input_tokens

        self.llm.stats.on_step(_on_step)
        finished_data: dict[str, tuple[str, list[int], float]] = {}
        try:
            while self.llm.engine.has_unfinished_requests():
                for seq in self.llm.engine.step():
                    finished_data[seq.request_id] = (
                        self.llm.tokenizer.decode(seq.output_token_ids),
                        list(seq.output_token_ids),
                        time.perf_counter() - t0,
                    )
        finally:
            self.llm.stats.listeners.remove(_on_step)

        results: list[GenerationResult] = []
        for i, req_id in enumerate(request_ids):
            output_text, output_token_ids, e2e_s = finished_data[req_id]
            ttft_s = (first_token_times[i] - t0) if i < len(first_token_times) else e2e_s
            results.append(
                GenerationResult(
                    prompt=prompts[i],
                    output_text=output_text,
                    output_token_ids=output_token_ids,
                    ttft_s=ttft_s,
                    total_time_s=e2e_s,
                )
            )
        return results

    def teardown(self) -> None:
        import gc

        import torch

        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
