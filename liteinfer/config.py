# pyright: reportPrivateImportUsage=false
"""Engine configuration. Immutable after construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from liteinfer.models.attention import resolve


@dataclass
class EngineConfig:
    model: str

    dtype: torch.dtype = torch.bfloat16
    device: str = "auto"

    max_num_seqs: int = 32
    max_model_len: int = 4096

    seed: int = 42

    # Attention kernel, or None for the fastest one this device can run —
    # "paged" on CUDA with Triton installed, "sdpa" otherwise. Naming one asks
    # for it specifically and fails rather than downgrading. See
    # `models/attention.py`.
    attn_implementation: str | None = None

    # How many requests may sit queued but not yet running. `max_num_seqs` caps
    # what runs; without this nothing caps what is accepted, and a caller that
    # submits faster than the engine drains grows the queue until the process
    # dies holding work it never ran.
    max_waiting_seqs: int = 1024

    # How many programs share one sequence's decode key loop under the paged
    # kernel, or None to choose from the batch width and the device. Splitting
    # buys parallelism a narrow batch cannot; pinning it is what lets a
    # benchmark row keep measuring one shape. See `models/paged_decode.py`.
    paged_decode_splits: int | None = None

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
        if self.paged_decode_splits is not None and self.paged_decode_splits < 1:
            raise ValueError("paged_decode_splits must be >= 1")
        if self.attn_implementation is not None:
            resolve(self.attn_implementation)  # raises on an unknown kernel name

    def resolved_device(self) -> torch.device:
        """Return the concrete `torch.device` after resolving ``"auto"``."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
