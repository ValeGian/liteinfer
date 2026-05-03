"""Common interface that every benchmarked engine must implement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SamplingSpec:
    """Engine-agnostic sampling parameters used by the harness."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 128
    seed: int | None = 0


@dataclass
class GenerationResult:
    """One completed request."""

    prompt: str
    output_text: str
    output_token_ids: list[int]
    ttft_s: float
    """Wall time from `generate()` start to the first emitted token."""
    total_time_s: float
    """Wall time for the full generation."""


class EngineRunner(Protocol):
    name: str

    def setup(self, model: str, **kwargs) -> None: ...
    def generate(
        self, prompts: list[str], sampling: SamplingSpec
    ) -> list[GenerationResult]: ...
    def teardown(self) -> None: ...
