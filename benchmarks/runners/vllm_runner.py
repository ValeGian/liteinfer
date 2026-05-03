"""Run benchmarks against vLLM (the reference engine)."""

from __future__ import annotations

import os

from benchmarks.runners.base import GenerationResult, SamplingSpec


class VLLMRunner:
    name = "vllm"

    def setup(self, model: str, **kwargs) -> None:
        from vllm import LLM  # deferred: vllm is an optional dependency

        # vLLM 0.20.0 defaults to fork on Linux. CUDA cannot be re-initialised
        # in a forked child when the parent process has already called into CUDA
        # (e.g. torch.cuda.is_available() in a test runner). Spawn avoids this.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        # disable_log_stats=False is required: LLM() overrides it to True by
        # default, which leaves RequestOutput.metrics=None and forces us into
        # the wall-time fallback path.
        kwargs.setdefault("disable_log_stats", False)
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

        outputs = self._llm.generate(prompts, params)

        self.peak_memory_bytes = _peak_cuda_memory()

        results = []
        for output in outputs:
            best = output.outputs[0]
            ttft, total = _extract_timing(output.metrics)
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


def _extract_timing(metrics: object) -> tuple[float, float]:
    """Return (ttft_s, total_s) from vLLM request metrics, or fall back to wall-time estimates.

    vLLM 0.20.0 uses RequestStateStats with monotonic-clock timestamps:
      queued_ts / first_token_ts / last_token_ts

    arrival_time is Unix time; it must NOT be mixed with the _ts fields because
    they come from different clocks.  first_token_latency crosses those clocks
    and is therefore unreliable — we ignore it.
    """
    # vLLM >= 0.20.0: all three fields from the same monotonic clock
    queued_ts = getattr(metrics, "queued_ts", 0.0) or 0.0
    first_ts = getattr(metrics, "first_token_ts", 0.0) or 0.0
    last_ts = getattr(metrics, "last_token_ts", 0.0) or 0.0
    if queued_ts > 0.0 and first_ts > 0.0 and last_ts > 0.0:
        return max(0.0, first_ts - queued_ts), max(0.0, last_ts - queued_ts)

    raise Exception("Not possible")


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
