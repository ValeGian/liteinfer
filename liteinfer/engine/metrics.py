# pyright: reportPrivateImportUsage=false
"""Per-step engine metrics. Wall time uses CUDA sync so it reflects executed work."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch


class Phase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class StepMetrics:
    """Snapshot of one engine step."""

    step_idx: int
    phase: Phase

    num_seqs: int
    input_tokens: int  # Total tokens in the forward-pass input across the batch.
    new_tokens: int    # Total tokens sampled (and appended to outputs) this step.

    wall_time_s: float
    peak_gpu_mem_bytes: int | None = None

    @property
    def throughput_tokens_per_s(self) -> float:
        return (self.input_tokens + self.new_tokens) / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def decode_throughput_tokens_per_s(self) -> float:
        return self.new_tokens / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def prefill_throughput_tokens_per_s(self) -> float:
        if self.phase != Phase.PREFILL:
            return 0.0
        return self.input_tokens / self.wall_time_s if self.wall_time_s > 0 else 0.0


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
    """Cumulative stats + per-step log. Subscribe via `on_step`."""

    steps: list[StepMetrics] = field(default_factory=list)
    total_input_tokens: int = 0
    total_new_tokens: int = 0
    total_prefill_input_tokens: int = 0
    total_prefill_wall_s: float = 0.0
    total_decode_new_tokens: int = 0
    total_decode_wall_s: float = 0.0
    total_wall_s: float = 0.0
    num_requests_finished: int = 0
    time: TimeBreakdown = field(default_factory=TimeBreakdown)
    listeners: list[Callable[[StepMetrics], None]] = field(default_factory=list)

    def record(self, step: StepMetrics) -> None:
        self.steps.append(step)
        self.total_input_tokens += step.input_tokens
        self.total_new_tokens += step.new_tokens
        self.total_wall_s += step.wall_time_s
        if step.phase == Phase.PREFILL:
            self.total_prefill_input_tokens += step.input_tokens
            self.total_prefill_wall_s += step.wall_time_s
        elif step.phase == Phase.DECODE:
            self.total_decode_new_tokens += step.new_tokens
            self.total_decode_wall_s += step.wall_time_s
        for listener in self.listeners:
            listener(step)

    def on_step(self, listener: Callable[[StepMetrics], None]) -> None:
        self.listeners.append(listener)

    @property
    def avg_throughput_tokens_per_s(self) -> float:
        return (self.total_input_tokens + self.total_new_tokens) / self.total_wall_s if self.total_wall_s > 0 else 0.0

    @property
    def avg_decode_throughput_tokens_per_s(self) -> float:
        if self.total_decode_wall_s <= 0:
            return 0.0
        return self.total_decode_new_tokens / self.total_decode_wall_s

    @property
    def avg_prefill_throughput_tokens_per_s(self) -> float:
        if self.total_prefill_wall_s <= 0:
            return 0.0
        return self.total_prefill_input_tokens / self.total_prefill_wall_s


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
