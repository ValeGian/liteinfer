# pyright: reportPrivateImportUsage=false
"""Per-step engine metrics. Wall time uses CUDA sync so it reflects executed work."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch


class Phase(str, Enum):
    """What a step's forward carried: prompt tokens, sampled tokens, or both."""

    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


@dataclass(frozen=True)
class StepMetrics:
    """Snapshot of one engine step, which is one forward wherever prefill is packed."""

    step_idx: int

    num_seqs: int
    input_tokens: int   # Total tokens in the forward-pass input across the batch.
    prompt_tokens: int  # How many of `input_tokens` are prompt; the rest are sampled tokens.
    new_tokens: int     # Total tokens sampled (and appended to outputs) this step.

    wall_time_s: float
    peak_gpu_mem_bytes: int | None = None

    @property
    def phase(self) -> Phase:
        """Read off the token counts rather than stored beside them, so the two cannot disagree."""
        if self.prompt_tokens == 0:
            return Phase.DECODE
        if self.prompt_tokens == self.input_tokens:
            return Phase.PREFILL
        return Phase.MIXED

    @property
    def throughput_tokens_per_s(self) -> float:
        return (self.input_tokens + self.new_tokens) / self.wall_time_s if self.wall_time_s > 0 else 0.0


@dataclass
class TimeBreakdown:
    """Where the engine loop spent its wall time, in seconds.

    The forward pass is not the whole story: sampling, detokenising an event
    for each sequence, and scheduling all happen between passes, and their
    share grows with output length. `unattributed` is the loop's own overhead —
    the asyncio round trip and the queue puts.
    """

    forward: float = 0.0
    sample: float = 0.0
    deliver: float = 0.0   # build one StreamEvent per sequence, which detokenises
    schedule: float = 0.0
    loop: float = 0.0      # the whole step, everything above included

    @property
    def unattributed(self) -> float:
        return max(0.0, self.loop - (self.forward + self.sample + self.deliver + self.schedule))

    def shares(self) -> dict[str, float]:
        """Each stage as a fraction of loop time. Empty before the first step."""
        if self.loop <= 0:
            return {}
        stages = {
            "forward": self.forward, "sample": self.sample, "deliver": self.deliver,
            "schedule": self.schedule, "unattributed": self.unattributed,
        }
        return {name: seconds / self.loop for name, seconds in stages.items()}

    def add(self, stage: str, seconds: float) -> None:
        setattr(self, stage, getattr(self, stage) + seconds)


@dataclass
class EngineStats:
    """Cumulative stats + per-step log. Subscribe via `on_step`.

    Totals are over every step and nothing is split by phase: a step that admits
    beside running sequences computes prompt and sampled tokens in one forward,
    and its wall time cannot be divided between them. Each step's
    `prompt_tokens` and `new_tokens` are exact, and are what to sum for either
    kind of work.
    """

    steps: list[StepMetrics] = field(default_factory=list)
    total_input_tokens: int = 0
    total_new_tokens: int = 0
    total_wall_s: float = 0.0
    num_requests_finished: int = 0
    time: TimeBreakdown = field(default_factory=TimeBreakdown)
    listeners: list[Callable[[StepMetrics], None]] = field(default_factory=list)

    def record(self, step: StepMetrics) -> None:
        self.steps.append(step)
        self.total_input_tokens += step.input_tokens
        self.total_new_tokens += step.new_tokens
        self.total_wall_s += step.wall_time_s
        for listener in self.listeners:
            listener(step)

    def on_step(self, listener: Callable[[StepMetrics], None]) -> None:
        self.listeners.append(listener)

    @property
    def avg_throughput_tokens_per_s(self) -> float:
        return (self.total_input_tokens + self.total_new_tokens) / self.total_wall_s if self.total_wall_s > 0 else 0.0


class StepTimer:
    """Times a block, syncing CUDA first and last so it reflects executed work.

    Pass `stats_time` and `stage` to fold the result into a `TimeBreakdown` on
    exit; pass `sync=False` for a stage that issues no GPU work of its own, so
    it is not charged for the previous stage's queue.
    """

    def __init__(
        self,
        device: torch.device,
        stats_time: TimeBreakdown | None = None,
        stage: str = "",
        sync: bool = True,
    ) -> None:
        self.device = device if sync else torch.device("cpu")
        self._stats_time = stats_time
        self._stage = stage
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> StepTimer:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self._start
        if self._stats_time is not None:
            self._stats_time.add(self._stage, self.elapsed)


def peak_gpu_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))
