# pyright: reportPrivateImportUsage=false
"""Unit tests for the sampler — CPU-only, no model loading."""

from __future__ import annotations

import torch

from liteinfer.sampling.params import SamplingParams
from liteinfer.sampling.sampler import Sampler, _apply_top_k, _apply_top_p


def test_greedy_picks_argmax() -> None:
    sampler = Sampler()
    logits = torch.tensor([[0.1, 5.0, -1.0, 2.0]])
    out = sampler(logits, [SamplingParams(temperature=0.0)])
    assert out.tolist() == [1]


def test_greedy_per_row() -> None:
    sampler = Sampler()
    logits = torch.tensor(
        [
            [0.0, 0.0, 5.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ]
    )
    out = sampler(logits, [SamplingParams(temperature=0.0)] * 2)
    assert out.tolist() == [2, 0]


def test_seeded_sampling_is_deterministic() -> None:
    sampler = Sampler()
    logits = torch.randn(1, 32)
    p1 = SamplingParams(temperature=1.0, seed=123)
    p2 = SamplingParams(temperature=1.0, seed=123)
    a = sampler(logits, [p1])
    b = sampler(logits, [p2])
    assert a.tolist() == b.tolist()


def test_top_k_masks_below_kth() -> None:
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
    masked = _apply_top_k(logits, k=2)
    # Only the top-2 (5.0 and 3.0) should remain finite.
    finite = (masked > float("-inf")).tolist()
    assert finite == [False, True, True, False]


def test_top_p_keeps_smallest_set_above_threshold() -> None:
    probs = torch.tensor([0.6, 0.3, 0.05, 0.05])
    out = _apply_top_p(probs, p=0.8)
    # 0.6 alone is below 0.8; need 0.6 + 0.3 = 0.9 → first two kept.
    assert (out[:2] > 0).all()
    assert (out[2:] == 0).all()
    assert torch.isclose(out.sum(), torch.tensor(1.0))


def test_batch_size_mismatch_raises() -> None:
    sampler = Sampler()
    logits = torch.randn(2, 8)
    try:
        sampler(logits, [SamplingParams()])
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched params length")
