"""The benchmark matrix.

Every liteinfer milestone is one entry. ``baseline`` names the config an entry
is meant to improve on; the report turns each such pair into a 1:1 delta, which
is how a change is judged. Entries without a baseline are reference points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Engine = Literal["liteinfer", "vllm"]


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    engine: Engine
    description: str
    max_num_seqs: int = 1
    baseline: str | None = None
    historical: bool = False
    """Measured before the code was removed. Kept so the report still shows the
    progression; `bench run --all` skips it because it can no longer run."""


_ENTRIES: tuple[BenchmarkConfig, ...] = (
    # --- liteinfer: KV cache lineage, one sequence at a time (§1.2, §2.1) ---
    BenchmarkConfig(
        name="liteinfer-nocache",
        historical=True,
        engine="liteinfer",
        description="No KV cache: every step re-feeds the whole sequence",
    ),
    BenchmarkConfig(
        name="liteinfer-eager",
        historical=True,
        engine="liteinfer",
        baseline="liteinfer-nocache",
        description="KV cache via transformers DynamicCache",
    ),
    BenchmarkConfig(
        name="liteinfer-native-eager",
        historical=True,
        engine="liteinfer",
        baseline="liteinfer-eager",
        description="KV cache as plain tensors, no DynamicCache",
    ),
    BenchmarkConfig(
        name="liteinfer-paged",
        historical=True,
        engine="liteinfer",
        baseline="liteinfer-native-eager",
        description="Paged KV cache: fixed-size blocks from a pool",
    ),
    # --- liteinfer: static batching at B=4 (§1.1) ---
    BenchmarkConfig(
        name="liteinfer-eager-b4",
        historical=True,
        engine="liteinfer",
        max_num_seqs=4,
        baseline="liteinfer-eager",
        description="Static batching, B=4, eager cache",
    ),
    BenchmarkConfig(
        name="liteinfer-native-eager-b4",
        historical=True,
        engine="liteinfer",
        max_num_seqs=4,
        baseline="liteinfer-native-eager",
        description="Static batching, B=4, native eager cache",
    ),
    BenchmarkConfig(
        name="liteinfer-paged-b4",
        historical=True,
        engine="liteinfer",
        max_num_seqs=4,
        baseline="liteinfer-paged",
        description="Static batching, B=4, paged cache",
    ),
    # --- liteinfer: continuous batching (§1.2) ---
    BenchmarkConfig(
        name="liteinfer-continuous",
        engine="liteinfer",
        max_num_seqs=32,
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
