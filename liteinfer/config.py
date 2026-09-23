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

    # Replay the decode forward from a CUDA graph instead of launching it kernel
    # by kernel. None captures wherever the preconditions hold (CUDA, and the
    # paged attention kernel, whose bounds are a tensor a capture cannot freeze);
    # True asks for it and fails rather than downgrading. See
    # `engine/cuda_graphs.py`.
    enable_cuda_graphs: bool | None = None

    # How many programs share one sequence's decode key loop under the paged
    # kernel, or None to choose from the batch width, the context bound and the
    # device. Splitting buys the parallelism a narrow batch cannot supply;
    # pinning it is what lets a benchmark row keep measuring one grid. See
    # `models/paged_decode.py`.
    paged_decode_splits: int | None = None

    # Prefill a batch as one flat run of tokens instead of left-padding every
    # prompt to the longest. None packs wherever FlashAttention's varlen entry
    # can run (CUDA, half precision); True asks for it and fails rather than
    # padding. See `models/attention.varlen_attention`.
    enable_packed_prefill: bool | None = None

    collect_stats: bool = True

    # KV block pool.
    block_size: int = 16
    # Share of the device's *total* memory the engine may occupy — weights,
    # activations and KV pool together. Total rather than free so the pool is a
    # function of the config and the device, not of what happened to be resident
    # when `load_model` ran. vLLM's `gpu_memory_utilization` means the same thing.
    gpu_memory_utilization: float = 0.85
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
        if self.paged_decode_splits is not None and self.paged_decode_splits < 1:
            raise ValueError("paged_decode_splits must be >= 1")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.attn_implementation is not None:
            resolve(self.attn_implementation)  # raises on an unknown kernel name

    def resolved_device(self) -> torch.device:
        """Return the concrete `torch.device` after resolving ``"auto"``."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
