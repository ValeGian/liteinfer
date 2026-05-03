"""User-facing entry point: the `LLM` class."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from liteinfer.config import EngineConfig
from liteinfer.engine.llm_engine import LLMEngine
from liteinfer.sampling.params import SamplingParams


@dataclass
class RequestOutput:
    """Result for a single generation request."""

    request_id: str
    prompt: str
    text: str
    token_ids: list[int]
    finish_reason: str  # "stop" | "length" | "abort"


class LLM:
    """High-level offline inference API.

    A thin facade over `LLMEngine`. The intent is for this public surface
    to stay stable while engine internals evolve.
    """

    def __init__(self, model: str, **engine_kwargs) -> None:
        self.config = EngineConfig(model=model, **engine_kwargs)
        self.engine = LLMEngine(self.config)

    def generate(
        self,
        prompts: str | Sequence[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Generate completions for one or more prompts."""
        raise NotImplementedError
