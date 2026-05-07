"""Standard benchmark workloads — pure data, engine-agnostic."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from benchmarks.runners.base import SamplingSpec


@dataclass
class Workload:
    name: str
    prompts: list[str]
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    sequential: bool = False
    """If True, the harness submits prompts one at a time (latency semantics).
    If False, all prompts are submitted at once (throughput semantics)."""


def throughput_workload(num_prompts: int = 32) -> Workload:
    """Many short independent prompts submitted all at once.
    The engine queues them and processes at its own pace (B=1 → sequential).
    Stresses req/s, tok/s, and per-request E2E latency under load."""
    base = "Write one sentence about the number "
    return Workload(
        name="throughput",
        prompts=[f"{base}{i}." for i in range(num_prompts)],
        sampling=SamplingSpec(temperature=0.0, max_tokens=64),
        sequential=False,
    )


def latency_workload(num_prompts: int = 20) -> Workload:
    """Single prompt repeated num_prompts times, submitted one at a time.
    No request queues: each prompt is sent only after the previous has finished.
    Measures pure engine latency: TTFT and E2E without queueing effects."""
    prompt = "Explain paged attention in one paragraph."
    return Workload(
        name="latency",
        prompts=[prompt] * num_prompts,
        sampling=SamplingSpec(temperature=0.0, max_tokens=128),
        sequential=True,
    )


def prefix_share_workload(num_prompts: int = 32) -> Workload:
    """Prompts share a long prefix. Stresses prefix caching."""
    prefix = "You are a helpful assistant. " * 50
    return Workload(
        name="prefix_share",
        prompts=[f"{prefix}Question {i}: what is {i} + {i}?" for i in range(num_prompts)],
        sampling=SamplingSpec(temperature=0.0, max_tokens=32),
        sequential=False,
    )


WORKLOADS: dict[str, Callable[[], Workload]] = {
    "throughput": throughput_workload,
    "latency": latency_workload,
    "prefix_share": prefix_share_workload,
}
