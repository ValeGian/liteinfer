"""The benchmark matrix.

Every liteinfer milestone is one entry. ``baseline`` names the config an entry
is meant to improve on; the report turns each such pair into a 1:1 delta, which
is how a change is judged. Entries without a baseline are reference points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Engine = Literal["liteinfer", "vllm"]
CacheMode = Literal["none", "eager", "native_eager", "paged"]


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    engine: Engine
    description: str
    max_num_seqs: int = 1
    cache_mode: CacheMode = "paged"  # liteinfer only
    continuous: bool = False  # liteinfer only: AsyncLLM instead of LLM
    baseline: str | None = None


_ENTRIES: tuple[BenchmarkConfig, ...] = (
    # --- liteinfer: KV cache lineage, one sequence at a time (§1.2, §2.1) ---
    BenchmarkConfig(
        name="liteinfer-nocache",
        engine="liteinfer",
        cache_mode="none",
        description="No KV cache: every step re-feeds the whole sequence",
    ),
    BenchmarkConfig(
        name="liteinfer-eager",
        engine="liteinfer",
        cache_mode="eager",
        baseline="liteinfer-nocache",
        description="KV cache via transformers DynamicCache",
    ),
    BenchmarkConfig(
        name="liteinfer-native-eager",
        engine="liteinfer",
        cache_mode="native_eager",
        baseline="liteinfer-eager",
        description="KV cache as plain tensors, no DynamicCache",
    ),
    BenchmarkConfig(
        name="liteinfer-paged",
        engine="liteinfer",
        cache_mode="paged",
        baseline="liteinfer-native-eager",
        description="Paged KV cache: fixed-size blocks from a pool",
    ),
    # --- liteinfer: static batching at B=4 (§1.1) ---
    BenchmarkConfig(
        name="liteinfer-eager-b4",
        engine="liteinfer",
        cache_mode="eager",
        max_num_seqs=4,
        baseline="liteinfer-eager",
        description="Static batching, B=4, eager cache",
    ),
    BenchmarkConfig(
        name="liteinfer-native-eager-b4",
        engine="liteinfer",
        cache_mode="native_eager",
        max_num_seqs=4,
        baseline="liteinfer-native-eager",
        description="Static batching, B=4, native eager cache",
    ),
    BenchmarkConfig(
        name="liteinfer-paged-b4",
        engine="liteinfer",
        cache_mode="paged",
        max_num_seqs=4,
        baseline="liteinfer-paged",
        description="Static batching, B=4, paged cache",
    ),
    # --- liteinfer: continuous batching (§1.2) ---
    BenchmarkConfig(
        name="liteinfer-continuous",
        engine="liteinfer",
        cache_mode="paged",
        max_num_seqs=32,
        continuous=True,
        baseline="liteinfer-paged-b4",
        description="Continuous batching, up to 32 concurrent sequences",
    ),
    # --- vLLM reference points, matched on batch size ---
    BenchmarkConfig(
        name="vllm",
        engine="vllm",
        description="vLLM, one sequence at a time",
    ),
    BenchmarkConfig(
        name="vllm-b4",
        engine="vllm",
        max_num_seqs=4,
        description="vLLM, up to 4 concurrent sequences",
    ),
    BenchmarkConfig(
        name="vllm-continuous",
        engine="vllm",
        max_num_seqs=32,
        description="vLLM, up to 32 concurrent sequences",
    ),
)

CONFIGS: dict[str, BenchmarkConfig] = {c.name: c for c in _ENTRIES}


def get(name: str) -> BenchmarkConfig:
    if name not in CONFIGS:
        raise KeyError(f"Unknown config {name!r}. Known: {', '.join(CONFIGS)}")
    return CONFIGS[name]
