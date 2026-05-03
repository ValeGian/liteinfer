"""ModelRunner — runs one forward pass for a scheduled batch.

Owns the model weights and is the integration point for tensor
parallelism, `torch.compile`, and CUDA graph capture.
"""

from __future__ import annotations

import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.sequence import SequenceGroup


class ModelRunner:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def load_model(self) -> None:
        """Materialize model weights on the target device(s)."""
        raise NotImplementedError

    @torch.inference_mode()
    def execute(self, scheduled: list[SequenceGroup]) -> torch.Tensor:
        """Run one forward pass; return per-sequence next-token logits."""
        raise NotImplementedError
