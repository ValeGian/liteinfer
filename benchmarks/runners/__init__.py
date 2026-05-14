"""Engine runners for the benchmark harness.

Each engine implements `EngineRunner` (see `base.py`) and registers
itself in `RUNNERS` so the CLI can look it up by name.
"""

from __future__ import annotations

from benchmarks.runners.base import EngineRunner, GenerationResult, SamplingSpec
from benchmarks.runners.liteinfer_b4_runner import LiteInferB4Runner
from benchmarks.runners.liteinfer_continuous_runner import LiteInferContinuousRunner
from benchmarks.runners.liteinfer_eager_runner import LiteInferEagerCacheRunner
from benchmarks.runners.liteinfer_native_eager_b4_runner import LiteInferNativeEagerB4Runner
from benchmarks.runners.liteinfer_native_eager_runner import LiteInferNativeEagerCacheRunner
from benchmarks.runners.liteinfer_paged_b4_runner import LiteInferPagedB4Runner
from benchmarks.runners.liteinfer_paged_runner import LiteInferPagedCacheRunner
from benchmarks.runners.liteinfer_runner import LiteInferRunner
from benchmarks.runners.vllm_b4_runner import VLLMB4Runner
from benchmarks.runners.vllm_continuous_runner import VLLMContinuousRunner
from benchmarks.runners.vllm_runner import VLLMRunner

RUNNERS: dict[str, type[EngineRunner]] = {
    "liteinfer": LiteInferRunner,
    "liteinfer-kvcache": LiteInferEagerCacheRunner,
    "liteinfer-native-kvcache": LiteInferNativeEagerCacheRunner,
    "liteinfer-paged-kvcache": LiteInferPagedCacheRunner,
    "liteinfer-b4": LiteInferB4Runner,
    "liteinfer-native-kvcache-b4": LiteInferNativeEagerB4Runner,
    "liteinfer-paged-b4": LiteInferPagedB4Runner,
    "liteinfer-continuous": LiteInferContinuousRunner,
    "vllm": VLLMRunner,
    "vllm-b4": VLLMB4Runner,
    "vllm-continuous": VLLMContinuousRunner,
}

__all__ = ["RUNNERS", "EngineRunner", "GenerationResult", "SamplingSpec"]
