"""Engine adapters.

Every engine exposes one primitive: submit a list of prompts, force each to emit
exactly ``max_tokens`` tokens, return the realised output lengths. Timing lives
in the harness, so all engines are timed by the same clock in the same way.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from benchmarks.configs import BenchmarkConfig

GPU_MEMORY_FRACTION = 0.90


class Adapter(Protocol):
    def __enter__(self) -> Adapter: ...
    def __exit__(self, *exc) -> None: ...

    def generate(self, prompts: list[str], max_tokens: int) -> list[int]:
        """Run all prompts to completion; return per-prompt output token counts."""
        ...


def _release_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _liteinfer_params(max_tokens: int):
    from liteinfer import SamplingParams

    return SamplingParams(
        temperature=0.0, max_tokens=max_tokens, min_tokens=max_tokens, ignore_eos=True
    )


class LiteInferAdapter:
    """Synchronous engine: no batching at ``max_num_seqs=1``, static above it."""

    def __init__(self, config: BenchmarkConfig, model: str) -> None:
        self._config = config
        self._model = model

    def __enter__(self) -> LiteInferAdapter:
        from liteinfer import LLM

        self._llm = LLM(
            model=self._model,
            cache_mode=self._config.cache_mode,
            max_num_seqs=self._config.max_num_seqs,
        )
        return self

    def __exit__(self, *exc) -> None:
        del self._llm
        _release_gpu()

    def generate(self, prompts: list[str], max_tokens: int) -> list[int]:
        outputs = self._llm.generate(prompts, _liteinfer_params(max_tokens))
        return [len(o.token_ids) for o in outputs]


class LiteInferContinuousAdapter:
    """Continuous batching. Owns an event loop so the engine stays up across calls."""

    def __init__(self, config: BenchmarkConfig, model: str) -> None:
        self._config = config
        self._model = model

    def __enter__(self) -> LiteInferContinuousAdapter:
        from liteinfer import AsyncLLM

        self._loop = asyncio.new_event_loop()
        self._llm = AsyncLLM(model=self._model, max_num_seqs=self._config.max_num_seqs)
        self._loop.run_until_complete(self._llm.start())
        return self

    def __exit__(self, *exc) -> None:
        self._loop.run_until_complete(self._llm.stop())
        self._loop.close()
        del self._llm
        _release_gpu()

    def generate(self, prompts: list[str], max_tokens: int) -> list[int]:
        outputs = self._loop.run_until_complete(
            self._llm.generate(prompts, _liteinfer_params(max_tokens))
        )
        return [len(o.token_ids) for o in outputs]


class VLLMAdapter:
    """vLLM at its best: its own scheduler and CUDA graphs both left enabled."""

    def __init__(self, config: BenchmarkConfig, model: str) -> None:
        self._config = config
        self._model = model

    def __enter__(self) -> VLLMAdapter:
        from vllm import LLM

        self._llm = LLM(
            model=self._model,
            dtype="bfloat16",
            max_num_seqs=self._config.max_num_seqs,
            gpu_memory_utilization=GPU_MEMORY_FRACTION,
            disable_log_stats=True,
        )
        return self

    def __exit__(self, *exc) -> None:
        del self._llm
        _release_gpu()

    def generate(self, prompts: list[str], max_tokens: int) -> list[int]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, min_tokens=max_tokens, ignore_eos=True
        )
        outputs = self._llm.generate(prompts, params, use_tqdm=False)
        return [len(o.outputs[0].token_ids) for o in outputs]


def build(config: BenchmarkConfig, model: str) -> Adapter:
    if config.engine == "vllm":
        return VLLMAdapter(config, model)
    if config.continuous:
        return LiteInferContinuousAdapter(config, model)
    return LiteInferAdapter(config, model)
