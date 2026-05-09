"""Engine runners for the benchmark harness.

Each engine implements `EngineRunner` (see `base.py`) and registers
itself in `RUNNERS` so the CLI can look it up by name.
"""

from __future__ import annotations

from benchmarks.runners.base import EngineRunner, GenerationResult, SamplingSpec
from benchmarks.runners.liteinfer_b4_runner import LiteInferB4Runner
from benchmarks.runners.liteinfer_eager_runner import LiteInferEagerCacheRunner
from benchmarks.runners.liteinfer_runner import LiteInferRunner
from benchmarks.runners.vllm_b4_runner import VLLMB4Runner
from benchmarks.runners.vllm_runner import VLLMRunner

RUNNERS: dict[str, type[EngineRunner]] = {
    "liteinfer": LiteInferRunner,
    "liteinfer-kvcache": LiteInferEagerCacheRunner,
    "liteinfer-b4": LiteInferB4Runner,
    "vllm": VLLMRunner,
    "vllm-b4": VLLMB4Runner,
}

__all__ = ["RUNNERS", "EngineRunner", "GenerationResult", "SamplingSpec"]
