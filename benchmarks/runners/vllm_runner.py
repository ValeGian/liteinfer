"""Run benchmarks against vLLM (the reference engine)."""

from __future__ import annotations

import os
import time

from benchmarks.runners.base import GenerationResult, SamplingSpec


class VLLMRunner:
    name = "vllm"

    def setup(self, model: str, **kwargs) -> None:
        from vllm import LLM  # deferred: vllm is an optional dependency

        # vLLM 0.20.0 defaults to fork on Linux. CUDA cannot be re-initialised
        # in a forked child when the parent process has already called into CUDA
        # (e.g. torch.cuda.is_available() in a test runner). Spawn avoids this.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self._llm = LLM(model=model, dtype="bfloat16", **kwargs)
        self.peak_memory_bytes: int | None = None

    def generate(
        self, prompts: list[str], sampling: SamplingSpec
    ) -> list[GenerationResult]:
        from vllm import SamplingParams as VLLMSamplingParams

        _reset_peak_cuda_memory()

        params = VLLMSamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            seed=sampling.seed,
        )

        wall_start = time.perf_counter()
        outputs = self._llm.generate(prompts, params)
        wall = time.perf_counter() - wall_start

        self.peak_memory_bytes = _peak_cuda_memory()

        results = []
        for output in outputs:
            best = output.outputs[0]
            ttft, total = _extract_timing(output.metrics, wall, len(best.token_ids))
            results.append(
                GenerationResult(
                    prompt=output.prompt or "",
                    output_text=best.text,
                    output_token_ids=list(best.token_ids),
                    ttft_s=ttft,
                    total_time_s=total,
                )
            )

        return results

    def teardown(self) -> None:
        self._llm = None


def _extract_timing(metrics: object, wall: float, num_tokens: int) -> tuple[float, float]:
    """Return (ttft_s, total_s) from vLLM request metrics, or fall back to wall-time estimates.

    vLLM 0.20.0 uses RequestStateStats with first_token_ts / last_token_ts.
    Older releases used first_token_time / finished_time.  We probe both.
    """
    arrival = getattr(metrics, "arrival_time", 0.0) or 0.0

    # vLLM >= 0.20.0: RequestStateStats
    first_ts = getattr(metrics, "first_token_ts", 0.0) or 0.0
    last_ts = getattr(metrics, "last_token_ts", 0.0) or 0.0
    if first_ts > 0.0 and last_ts > 0.0 and arrival > 0.0:
        # first_token_latency is pre-computed by vLLM; use it when populated.
        precomputed = getattr(metrics, "first_token_latency", 0.0) or 0.0
        ttft = precomputed if precomputed > 0.0 else max(0.0, first_ts - arrival)
        return ttft, max(0.0, last_ts - arrival)

    # vLLM < 0.20.0: first_token_time / finished_time
    first_token = getattr(metrics, "first_token_time", None)
    finished = getattr(metrics, "finished_time", None)
    if first_token is not None and finished is not None and arrival > 0.0:
        return max(0.0, first_token - arrival), max(0.0, finished - arrival)

    # Fallback: estimate TTFT as one inter-token step, total as full wall time.
    return wall / max(1, num_tokens), wall


def _reset_peak_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_cuda_memory() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    return None
