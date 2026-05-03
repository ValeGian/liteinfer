"""Sampling parameters — user-controlled knobs for token generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingParams:
    n: int = 1
    """Number of output sequences to return per prompt."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1   # -1 disables top-k

    max_tokens: int = 16
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None

    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0
