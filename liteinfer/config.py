"""Engine configuration.

`EngineConfig` is owned by `LLMEngine` and consumed by every component
below it (model loader, KV cache, scheduler, model runner). Treat
instances as immutable after construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

CacheMode = Literal["eager", "none"]
"""KV cache policy.

- ``eager``: keep per-sequence K/V tensors across decode steps.
  Standard transformer inference; recompute is avoided.
- ``none``: drop cache between steps; feed the full sequence on every
  forward pass. Useful as a didactic reference and as a parity baseline
  for cache implementations.
"""


@dataclass
class EngineConfig:
    model: str
    """Local path to a HuggingFace-format model directory.

    Must contain ``config.json`` and one or more ``*.safetensors``
    shards. Remote download is intentionally out of scope: the engine
    consumes already-materialized weights.
    """

    dtype: torch.dtype = torch.bfloat16
    device: str = "auto"
    """Execution device. ``"auto"`` resolves to ``"cuda"`` when CUDA is
    available, else ``"cpu"``. Pass an explicit device string to pin."""

    # Cache and batching
    cache_mode: CacheMode = "eager"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 32
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # Parallelism
    tensor_parallel_size: int = 1

    # Optimization toggles. Opt-in, off by default while features mature.
    # Each one gates a strict drop-in replacement for the eager path.
    enable_torch_compile: bool = False
    enable_cuda_graph: bool = False
    enable_prefix_caching: bool = False

    # Reproducibility
    seed: int = 42

    # Stats
    collect_stats: bool = True
    """When True, the engine emits a `StepMetrics` per `step()` and
    accumulates them in `LLMEngine.stats`. Disable to skip the
    bookkeeping in tight loops."""

    def __post_init__(self) -> None:
        assert 1 <= self.tensor_parallel_size <= 8, "tensor_parallel_size must be in [1, 8]"
        assert 0.0 < self.gpu_memory_utilization <= 1.0, "gpu_memory_utilization must be in (0, 1]"
        assert self.kvcache_block_size >= 1, "kvcache_block_size must be >= 1"
        assert self.max_num_seqs >= 1, "max_num_seqs must be >= 1"
        assert self.max_num_batched_tokens >= 1, "max_num_batched_tokens must be >= 1"
        assert self.max_model_len >= 1, "max_model_len must be >= 1"
        assert self.cache_mode in ("eager", "none"), f"cache_mode must be 'eager' or 'none', got {self.cache_mode!r}"

    def resolved_device(self) -> torch.device:
        """Return the concrete `torch.device` after resolving ``"auto"``."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
