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


def throughput_workload(num_prompts: int = 64) -> Workload:
    """Many short, independent prompts. Stresses batching / throughput."""
    base = "Write one sentence about the number "
    return Workload(
        name="throughput",
        prompts=[f"{base}{i}." for i in range(num_prompts)],
        sampling=SamplingSpec(temperature=0.0, max_tokens=64),
    )


def latency_workload() -> Workload:
    """Single prompt. Measures TTFT and inter-token latency."""
    return Workload(
        name="latency",
        prompts=["Explain paged attention in one paragraph."],
        sampling=SamplingSpec(temperature=0.0, max_tokens=128),
    )


def prefix_share_workload(num_prompts: int = 32) -> Workload:
    """Prompts share a long prefix. Stresses prefix caching."""
    prefix = "You are a helpful assistant. " * 50
    return Workload(
        name="prefix_share",
        prompts=[f"{prefix}Question {i}: what is {i} + {i}?" for i in range(num_prompts)],
        sampling=SamplingSpec(temperature=0.0, max_tokens=32),
    )


WORKLOADS: dict[str, Callable[[], Workload]] = {
    "throughput": throughput_workload,
    "latency": latency_workload,
    "prefix_share": prefix_share_workload,
}
