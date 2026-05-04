"""Sampling parameters — user-controlled knobs for token generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingParams:
    n: int = 1
    """Number of output sequences to return per prompt."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 disables top-k

    max_tokens: int = 64
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None

    seed: int | None = None

    def __post_init__(self) -> None:
        assert self.n >= 1, "n must be >= 1"
        assert self.temperature >= 0, "temperature must be >= 0"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.top_k == -1 or self.top_k >= 1, "top_k must be -1 (disabled) or >= 1"
        assert self.max_tokens >= 1, "max_tokens must be >= 1"

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0
