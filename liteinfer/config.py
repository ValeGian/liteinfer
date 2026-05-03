"""Engine configuration.

`EngineConfig` is owned by `LLMEngine` and consumed by every component
below it (model loader, KV cache, scheduler, model runner). Treat
instances as immutable after construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EngineConfig:
    model: str
    """HuggingFace repo ID or local path to a model directory."""

    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"

    # KV cache / batching
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    block_size: int = 16
    gpu_memory_utilization: float = 0.9

    # Parallelism
    tensor_parallel_size: int = 1

    # Optimization toggles — opt-in, off by default while features mature.
    enable_torch_compile: bool = False
    enable_cuda_graph: bool = False
    enable_prefix_caching: bool = False

    # Reproducibility
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
