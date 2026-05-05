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
    cache_mode: CacheMode = "none"
    max_num_seqs: int = 32
    max_model_len: int = 4096

    # Reproducibility
    seed: int = 42

    # Stats
    collect_stats: bool = True
    """When True, the engine emits a `StepMetrics` per `step()` and
    accumulates them in `LLMEngine.stats`. Disable to skip the
    bookkeeping in tight loops."""

    def __post_init__(self) -> None:
        assert self.max_num_seqs >= 1, "max_num_seqs must be >= 1"
        assert self.max_model_len >= 1, "max_model_len must be >= 1"
        assert self.cache_mode in ("eager", "none"), f"cache_mode must be 'eager' or 'none', got {self.cache_mode!r}"

    def resolved_device(self) -> torch.device:
        """Return the concrete `torch.device` after resolving ``"auto"``."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
