"""Sampling parameters — user-controlled knobs for token generation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 disables top-k

    max_tokens: int = 64
    min_tokens: int = 0
    ignore_eos: bool = False
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None

    seed: int | None = None

    # Stable unique ID so Sampler can key per-request RNG state without
    # relying on id(), which reuses addresses after GC.
    _id: str = field(default_factory=lambda: uuid.uuid4().hex, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.min_tokens < 0:
            raise ValueError("min_tokens must be >= 0")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be <= max_tokens")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def id(self):
        return self._id
