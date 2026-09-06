# pyright: reportPrivateImportUsage=false
"""Engine configuration. Immutable after construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from liteinfer.models.attention import DEFAULT_IMPLEMENTATION, resolve


@dataclass
class EngineConfig:
    model: str

    dtype: torch.dtype = torch.bfloat16
    device: str = "auto"

    max_num_seqs: int = 32
    max_model_len: int = 4096

    seed: int = 42

    # Attention kernel. "sdpa" fuses the softmax and never materialises the
    # score matrix; "eager" writes it out, which caps the prompt length that
    # fits in memory. See `models/attention.py`.
    attn_implementation: str = DEFAULT_IMPLEMENTATION

    # How many requests may sit queued but not yet running. `max_num_seqs` caps
    # what runs; without this nothing caps what is accepted, and a caller that
    # submits faster than the engine drains grows the queue until the process
    # dies holding work it never ran.
    max_waiting_seqs: int = 1024

    collect_stats: bool = True

    # KV block pool.
    block_size: int = 16
    kv_cache_memory_fraction: float = 0.85  # share of free VRAM the pool may claim
    num_gpu_blocks: int | None = None  # None → sized from the fraction and the workload

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.max_waiting_seqs < 1:
            raise ValueError("max_waiting_seqs must be >= 1")
        if not 0 < self.kv_cache_memory_fraction <= 1:
            raise ValueError("kv_cache_memory_fraction must be in (0, 1]")
        resolve(self.attn_implementation)  # raises on an unknown kernel name

    def resolved_device(self) -> torch.device:
        """Return the concrete `torch.device` after resolving ``"auto"``."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
