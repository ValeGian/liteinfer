"""Sampler — turns logits into token ids per request."""

from __future__ import annotations

import torch

from liteinfer.sampling.params import SamplingParams


class Sampler:
    """Apply per-request sampling parameters to a batch of logits."""

    def __call__(
        self,
        logits: torch.Tensor,            # shape: [batch, vocab]
        params: list[SamplingParams],    # one entry per row of `logits`
    ) -> torch.Tensor:                   # shape: [batch], dtype long
        raise NotImplementedError
