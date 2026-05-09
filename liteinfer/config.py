# pyright: reportPrivateImportUsage=false
"""Engine configuration. Immutable after construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

CacheMode = Literal["eager", "native_eager", "none"]


@dataclass
class EngineConfig:
    model: str

    dtype: torch.dtype = torch.bfloat16
    device: str = "auto"

    cache_mode: CacheMode = "none"
    max_num_seqs: int = 32
    max_model_len: int = 4096

    seed: int = 42

    collect_stats: bool = True

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")
        if self.cache_mode not in ("eager", "native_eager", "none"):
            raise ValueError(
                f"cache_mode must be 'eager', 'native_eager', or 'none', got {self.cache_mode!r}"
            )

    def resolved_device(self) -> torch.device:
        """Return the concrete `torch.device` after resolving ``"auto"``."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
