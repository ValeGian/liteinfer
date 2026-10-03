"""Engine adapters.

Every engine exposes one primitive: submit a list of prompts, force each to emit
exactly ``max_tokens`` tokens, return the realised output lengths. Timing lives
in the harness, so all engines are timed by the same clock in the same way.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from benchmarks.configs import BenchmarkConfig

GPU_MEMORY_FRACTION = 0.90


class Adapter(Protocol):
    def __enter__(self) -> Adapter: ...
    def __exit__(self, *exc) -> None: ...

    def generate(self, prompts: list[str], max_tokens: int) -> list[int]:
        """Run all prompts to completion; return per-prompt output token counts."""
        ...

    def step_phases(self) -> dict[str, int] | None:
        """Steps run so far, by phase, or None for an engine that does not report them."""
        ...


def _release_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LiteInferAdapter:
    """Continuous batching through the synchronous facade."""

    def __init__(self, config: BenchmarkConfig, model: str) -> None:
        self._config = config
        self._model = model

    def __enter__(self) -> LiteInferAdapter:
        from liteinfer import LLM

        self._llm = LLM(
            model=self._model,
            max_num_seqs=self._config.max_num_seqs,
            max_model_len=self._config.max_model_len,
            attn_implementation=self._config.attn_implementation,
            enable_cuda_graphs=self._config.enable_cuda_graphs,
            enable_packed_prefill=self._config.enable_packed_prefill,
            paged_decode_splits=self._config.paged_decode_splits,
            max_num_batched_tokens=self._config.max_num_batched_tokens,
        )
        return self

    def __exit__(self, *exc) -> None:
        self._llm.close()
        del self._llm
        _release_gpu()

    def generate(self, prompts: list[str], max_tokens: int) -> list[int]:
        from liteinfer import SamplingParams

        params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, min_tokens=max_tokens, ignore_eos=True
        )
        return [len(o.token_ids) for o in self._llm.generate(prompts, params)]

    def step_phases(self) -> dict[str, int]:
        return dict(Counter(step.phase.value for step in self._llm.stats.steps))


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
            max_model_len=self._config.max_model_len,
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

    def step_phases(self) -> None:
        return None


def build(config: BenchmarkConfig, model: str) -> Adapter:
    return VLLMAdapter(config, model) if config.engine == "vllm" else LiteInferAdapter(config, model)
