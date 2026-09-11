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
    max_model_len: int = 4096
    """Context budget the row claims. It sizes the KV pool's ceiling and the
    worst-case forward the engine profiles at load, so a wide row has to state one
    it can actually serve — 128 sequences x 4,096 tokens does not fit an A40."""
    attn_implementation: str = "eager"
    """Pinned per row rather than inherited from `EngineConfig`: a stored result
    has to keep meaning the same thing after the engine default moves on."""
    enable_cuda_graphs: bool = False
    """Off by default for the same reason: every row stored before §3.2 was
    measured launching the decode forward kernel by kernel, and must go on
    meaning that."""
    paged_decode_splits: int | None = 1
    """Programs sharing one sequence's decode key loop. One by default for the
    same reason again: every row stored before §2.7 ran the unsplit grid, and a
    stored number has to keep describing the grid it was measured on. `None`
    hands the choice to the kernel, which is what the split rows measure."""
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
    # --- liteinfer: fused attention kernel (§3.3) ---
    BenchmarkConfig(
        name="liteinfer-sdpa",
        engine="liteinfer",
        max_num_seqs=32,
        attn_implementation="sdpa",
        baseline="liteinfer-continuous",
        description="Continuous batching, attention through PyTorch SDPA",
    ),
    # --- liteinfer: paged decode attention (§2.3) ---
    BenchmarkConfig(
        name="liteinfer-paged-attn",
        engine="liteinfer",
        max_num_seqs=32,
        attn_implementation="paged",
        baseline="liteinfer-sdpa",
        description="Continuous batching, decode attention reads the KV pool in-kernel",
    ),
    # --- liteinfer: captured decode forward (§3.2) ---
    BenchmarkConfig(
        name="liteinfer-graphs",
        engine="liteinfer",
        max_num_seqs=32,
        attn_implementation="paged",
        enable_cuda_graphs=True,
        baseline="liteinfer-paged-attn",
        description="Paged decode replayed from a CUDA graph instead of launched per kernel",
    ),
    # --- liteinfer: the batch width a flat step pays for (§1.4) ---
    BenchmarkConfig(
        name="liteinfer-graphs-b128",
        engine="liteinfer",
        max_num_seqs=128,
        max_model_len=1024,
        attn_implementation="paged",
        enable_cuda_graphs=True,
        description="Captured decode at 128 concurrent sequences",
    ),
    # --- liteinfer: split the decode key loop when the batch is narrow (§2.7) ---
    BenchmarkConfig(
        name="liteinfer-splitk",
        engine="liteinfer",
        max_num_seqs=32,
        attn_implementation="paged",
        enable_cuda_graphs=True,
        paged_decode_splits=None,
        baseline="liteinfer-graphs",
        description="Captured decode, narrow batches splitting their key loop across programs",
    ),
    # --- liteinfer: the same pair past the 4,096-token context the matrix stopped at ---
    BenchmarkConfig(
        name="liteinfer-graphs-16k",
        engine="liteinfer",
        max_num_seqs=8,
        max_model_len=16384,
        attn_implementation="paged",
        enable_cuda_graphs=True,
        description="Captured decode with a 16k context budget, one program per KV head",
    ),
    BenchmarkConfig(
        name="liteinfer-splitk-16k",
        engine="liteinfer",
        max_num_seqs=8,
        max_model_len=16384,
        attn_implementation="paged",
        enable_cuda_graphs=True,
        paged_decode_splits=None,
        baseline="liteinfer-graphs-16k",
        description="Split decode where attention is most of the step: long context, narrow batch",
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
    BenchmarkConfig(
        name="vllm-b128",
        engine="vllm",
        max_num_seqs=128,
        max_model_len=1024,
        description="vLLM, up to 128 concurrent sequences",
    ),
)

CONFIGS: dict[str, BenchmarkConfig] = {c.name: c for c in _ENTRIES}


def get(name: str) -> BenchmarkConfig:
    if name not in CONFIGS:
        raise KeyError(f"Unknown config {name!r}. Known: {', '.join(CONFIGS)}")
    return CONFIGS[name]
