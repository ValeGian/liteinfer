# pyright: reportPrivateImportUsage=false
"""Sampler — logits → token ids, per-row Python loop."""

from __future__ import annotations

import torch

from liteinfer.sampling.params import SamplingParams


class Sampler:
    """Apply per-request sampling parameters to a batch of logits."""

    def __init__(self) -> None:
        # Generators keyed by id(SamplingParams) so RNG state advances across
        # decode steps for the same request.
        self._generators: dict[str, torch.Generator] = {}

    def __call__(
        self,
        logits: torch.Tensor,  # shape: [batch, vocab]
        params: list[SamplingParams],  # one entry per row of `logits`
    ) -> torch.Tensor:  # shape: [batch], dtype long
        if logits.ndim != 2:
            raise ValueError(f"expected 2D logits, got shape {tuple(logits.shape)}")
        if len(params) != logits.shape[0]:
            raise ValueError(f"params length {len(params)} != batch size {logits.shape[0]}")
        out = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
        for i, p in enumerate(params):
            out[i] = self._sample_one(logits[i], p)
        return out

    def _sample_one(self, row_logits: torch.Tensor, p: SamplingParams) -> torch.Tensor:
        if p.greedy or p.top_k == 1:
            return torch.argmax(row_logits, dim=-1)

        scaled = row_logits / p.temperature
        if p.top_k > 0:
            scaled = _apply_top_k(scaled, p.top_k)
        probs = torch.softmax(scaled, dim=-1)
        if 0.0 < p.top_p < 1.0:
            probs = _apply_top_p(probs, p.top_p)

        gen = self._get_generator(p, device=row_logits.device)
        sample = torch.multinomial(probs, num_samples=1, generator=gen)
        return sample.squeeze(0)

    def _get_generator(self, p: SamplingParams, device: torch.device) -> torch.Generator | None:
        if p.seed is None:
            return None
        key = p.id
        gen = self._generators.get(key)
        if gen is None or gen.device != device:
            gen = torch.Generator(device=device).manual_seed(p.seed)
            self._generators[key] = gen
        return gen


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k >= logits.numel():
        return logits
    # Index-based masking avoids keeping >k tokens when values tie at the boundary.
    _, top_indices = torch.topk(logits, k)
    mask = torch.full_like(logits, float("-inf"))
    mask.scatter_(0, top_indices, logits[top_indices])
    return mask


def _apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # Drop entries strictly past the threshold; keep the first crossing.
    mask = cum - sorted_probs > p
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, sorted_probs)
    total = out.sum()
    if total > 0:
        out = out / total
    return out
