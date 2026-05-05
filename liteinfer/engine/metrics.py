# pyright: reportPrivateImportUsage=false
"""Per-step engine metrics.

Designed for a teaching engine: emit one ``StepMetrics`` snapshot per
``LLMEngine.step()`` so a UI, dashboard, or log can show *what just
happened* without touching internals. ``EngineStats`` accumulates them
and exposes derived rates (decode tok/s, prefill tok/s, request
throughput).

Cost model: a step's ``wall_time_s`` is wall-clock around the forward
pass plus sampling. CUDA work is synchronized before reading the clock
so the number reflects executed time, not enqueue latency.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch


class Phase(str, Enum):
    PREFILL = "prefill"
    """First step of an eager-cache batch: model consumes the prompt."""
    DECODE = "decode"
    """Subsequent steps of an eager-cache batch: one new token per seq."""
    RECOMPUTE = "recompute"
    """Cache-disabled step: model re-processes the full sequence."""


@dataclass(frozen=True)
class StepMetrics:
    """Snapshot of one engine step.

    Distinguishes ``input_tokens`` (tokens fed to the forward pass) from
    ``new_tokens`` (tokens sampled this step) so callers can compute
    prefill vs. decode throughput consistently across cache modes.
    """

    step_idx: int
    phase: Phase

    num_seqs: int
    input_tokens: int
    """Total tokens in the forward-pass input across the batch."""
    new_tokens: int
    """Total tokens sampled (and appended to outputs) this step."""

    wall_time_s: float
    peak_gpu_mem_bytes: int | None = None

    @property
    def throughput_tokens_per_s(self) -> float:
        denom = self.wall_time_s
        return (self.input_tokens + self.new_tokens) / denom if denom > 0 else 0.0

    @property
    def decode_throughput_tokens_per_s(self) -> float:
        return self.new_tokens / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def prefill_throughput_tokens_per_s(self) -> float:
        if self.phase != Phase.PREFILL:
            return 0.0
        return self.input_tokens / self.wall_time_s if self.wall_time_s > 0 else 0.0


@dataclass
class EngineStats:
    """Cumulative engine stats and a per-step log.

    The engine appends to ``steps`` after every forward pass and updates
    the running totals. Subscribers registered via ``on_step`` receive
    each ``StepMetrics`` synchronously — useful for live dashboards.
    """

    steps: list[StepMetrics] = field(default_factory=list)
    total_input_tokens: int = 0
    total_new_tokens: int = 0
    total_prefill_input_tokens: int = 0
    total_prefill_wall_s: float = 0.0
    total_decode_new_tokens: int = 0
    total_decode_wall_s: float = 0.0
    total_wall_s: float = 0.0
    num_requests_finished: int = 0
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
        """Subscribe to per-step metrics."""
        self.listeners.append(listener)

    @property
    def avg_throughput_tokens_per_s(self) -> float:
        denom = self.total_wall_s
        return (self.total_input_tokens + self.total_new_tokens) / denom if denom > 0 else 0.0

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
    """Context manager that times a forward pass with CUDA sync.

    Use as ``with StepTimer(device) as t: ...`` then read ``t.elapsed``.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
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


def peak_gpu_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))
