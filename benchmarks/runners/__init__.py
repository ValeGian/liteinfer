"""Engine runners for the benchmark harness.

Each engine implements `EngineRunner` (see `base.py`) and registers
itself in `RUNNERS` so the CLI can look it up by name.
"""

from __future__ import annotations

from benchmarks.runners.base import EngineRunner, GenerationResult, SamplingSpec
from benchmarks.runners.liteinfer_runner import LiteInferRunner
from benchmarks.runners.vllm_runner import VLLMRunner

RUNNERS: dict[str, type[EngineRunner]] = {
    "liteinfer": LiteInferRunner,
    "vllm": VLLMRunner,
}

__all__ = ["RUNNERS", "EngineRunner", "GenerationResult", "SamplingSpec"]
