# pyright: reportPrivateImportUsage=false
"""Unit tests for the sampler — CPU-only, no model loading."""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from liteinfer.sampling.params import SamplingParams
from liteinfer.sampling.sampler import Sampler, _apply_top_k, _apply_top_p

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_N = 400  # sample count for statistical tests — enough to catch exclusion violations


def _sample_n(sampler: Sampler, row_logits: torch.Tensor, p: SamplingParams, n: int = _N) -> Counter:
    """Run sampler n times on a single row and return token frequency counts."""
    counts: Counter = Counter()
    logits = row_logits.unsqueeze(0)
    for _ in range(n):
        token = sampler(logits, [p]).item()
        counts[int(token)] += 1
    return counts


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


def test_top_k_ties_at_boundary_keeps_exactly_k() -> None:
    # All three 4.0 values tie at the boundary; only exactly k=2 should survive.
    logits = torch.tensor([5.0, 4.0, 4.0, 4.0])
    masked = _apply_top_k(logits, k=2)
    finite_count = (masked > float("-inf")).sum().item()
    assert finite_count == 2


def test_generator_key_is_stable_across_gc() -> None:
    # Two distinct SamplingParams objects with same seed must produce independent
    # generators (not collide via id() reuse after GC).
    sampler = Sampler()
    logits = torch.randn(1, 64)
    p1 = SamplingParams(temperature=1.0, seed=42)
    result1 = sampler(logits, [p1])
    # Advance p1's generator state so its next draw differs from a fresh seed=42 generator.
    sampler(logits, [p1])
    p2 = SamplingParams(temperature=1.0, seed=42)
    result2 = sampler(logits, [p2])
    # p2 is a fresh generator at seed=42, so it should match result1, not the advanced p1.
    assert result1.tolist() == result2.tolist()


def test_batch_size_mismatch_raises() -> None:
    sampler = Sampler()
    logits = torch.randn(2, 8)
    try:
        sampler(logits, [SamplingParams()])
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched params length")


# ===========================================================================
# greedy — behavioral contracts
# ===========================================================================


def test_greedy_always_returns_argmax_across_diverse_logits() -> None:
    # Greedy must pick the highest-logit token for any input, not just simple cases.
    sampler = Sampler()
    torch.manual_seed(0)
    for _ in range(30):
        logits = torch.randn(1, 50)
        expected = int(logits.argmax(dim=-1).item())
        out = sampler(logits, [SamplingParams(temperature=0.0)])
        assert out.item() == expected, f"expected argmax={expected}, got {out.item()}"


def test_greedy_is_deterministic_regardless_of_seed() -> None:
    # Greedy uses argmax — RNG seed must have no effect on the result.
    sampler = Sampler()
    logits = torch.tensor([[0.1, 5.0, -1.0, 2.0]])
    results = {sampler(logits, [SamplingParams(temperature=0.0, seed=s)]).item() for s in range(10)}
    assert results == {1}, "greedy must always return argmax regardless of seed"


def test_top_k_1_is_equivalent_to_greedy_at_any_temperature() -> None:
    # With only one candidate token, sampling must always return argmax.
    sampler = Sampler()
    torch.manual_seed(0)
    for _ in range(20):
        logits = torch.randn(1, 32)
        expected = int(logits.argmax(dim=-1).item())
        out = sampler(logits, [SamplingParams(temperature=2.0, top_k=1)])
        assert out.item() == expected


# ===========================================================================
# top-k — behavioral contracts
# ===========================================================================


def test_top_k_tokens_ranked_below_k_are_never_sampled() -> None:
    # Tokens at rank > k must never appear, regardless of how many samples we draw.
    # Logits are spread enough that the top-2 boundary is unambiguous.
    logits = torch.tensor([10.0, 9.0, -20.0, -20.0, -20.0])
    sampler = Sampler()
    counts = _sample_n(sampler, logits, SamplingParams(temperature=1.0, top_k=2, seed=0))
    forbidden = set(counts.keys()) - {0, 1}
    assert not forbidden, f"tokens outside top-2 were sampled: {forbidden}"


def test_top_k_all_k_tokens_are_reachable() -> None:
    # When logits within the top-k are nearly equal, all k tokens must be reachable.
    logits = torch.tensor([1.0, 1.0, 1.0, -1_000.0, -1_000.0])
    sampler = Sampler()
    counts = _sample_n(sampler, logits, SamplingParams(temperature=1.0, top_k=3, seed=1), n=600)
    assert {0, 1, 2}.issubset(counts.keys()), f"not all top-3 tokens reached: {dict(counts)}"


def test_top_k_disabled_makes_full_vocab_reachable() -> None:
    # top_k=-1 (disabled) with uniform logits must eventually sample every token.
    vocab_size = 10
    logits = torch.zeros(vocab_size)
    sampler = Sampler()
    counts = _sample_n(sampler, logits, SamplingParams(temperature=1.0, top_k=-1, seed=2), n=1_000)
    assert len(counts) == vocab_size, f"expected all {vocab_size} tokens reachable, got {sorted(counts)}"


def test_top_k_equals_vocab_size_has_no_restricting_effect() -> None:
    # top_k >= vocab means no filtering — same tokens reachable as with top_k disabled.
    vocab_size = 6
    logits = torch.zeros(vocab_size)
    sampler = Sampler()
    counts = _sample_n(sampler, logits, SamplingParams(temperature=1.0, top_k=vocab_size, seed=3), n=600)
    assert len(counts) == vocab_size, f"top_k=vocab_size must not restrict sampling: {dict(counts)}"


# ===========================================================================
# top-p — behavioral contracts
# ===========================================================================


def test_top_p_tokens_outside_nucleus_are_never_sampled() -> None:
    # Token 0 carries ~100% of the probability mass. With top_p=0.95 the nucleus
    # contains only token 0; tokens 1-3 must never be chosen.
    logits = torch.tensor([100.0, -100.0, -100.0, -100.0])
    sampler = Sampler()
    counts = _sample_n(sampler, logits, SamplingParams(temperature=1.0, top_p=0.95, seed=4))
    assert set(counts.keys()) == {0}, f"expected only token 0, got {dict(counts)}"


def test_top_p_boundary_token_is_included_in_nucleus() -> None:
    # Probs ≈ [0.73, 0.27, ~0, ~0]. With top_p=0.80, token 0 alone (0.73) does not
    # meet the threshold so token 1 (the boundary-crosser) must be included.
    # Both tokens 0 and 1 must appear; tokens 2+ must never appear.
    logits = torch.tensor([1.0, 0.0, -100.0, -100.0])  # softmax ≈ [0.73, 0.27, ~0, ~0]
    sampler = Sampler()
    counts = _sample_n(sampler, logits, SamplingParams(temperature=1.0, top_p=0.80, seed=5), n=600)
    assert 1 in counts, "boundary token must be included in nucleus and reachable"
    forbidden = set(counts.keys()) - {0, 1}
    assert not forbidden, f"tokens outside nucleus were sampled: {forbidden}"


def test_top_p_1_does_not_restrict_any_token() -> None:
    # top_p=1.0 is the full vocabulary — every token must be reachable.
    vocab_size = 8
    logits = torch.zeros(vocab_size)
    sampler = Sampler()
    counts = _sample_n(sampler, logits, SamplingParams(temperature=1.0, top_p=1.0, seed=6), n=800)
    assert len(counts) == vocab_size, f"top_p=1.0 must not restrict sampling: {dict(counts)}"


def test_top_k_and_top_p_combined_restrict_to_intersection() -> None:
    # top_k=3 keeps tokens 0,1,2. With logits [5,4,3,-100,-100]:
    # softmax ≈ [0.665, 0.245, 0.090].
    # top_p=0.70: cum-before-token = [0.0, 0.665, 0.910]; mask (> 0.70) = [F, F, T].
    # So top_p cuts token 2; tokens 0 and 1 survive the intersection.
    logits = torch.tensor([5.0, 4.0, 3.0, -100.0, -100.0])
    sampler = Sampler()
    counts = _sample_n(sampler, logits, SamplingParams(temperature=1.0, top_k=3, top_p=0.70, seed=7), n=600)
    forbidden = set(counts.keys()) - {0, 1}
    assert not forbidden, f"only tokens 0-1 should be reachable, got extra: {forbidden}"
    assert 1 in counts, "token 1 must be reachable after top_k ∩ top_p"


# ===========================================================================
# temperature — behavioral contracts
# ===========================================================================


def test_high_temperature_produces_more_diverse_samples_than_low() -> None:
    # Higher temperature flattens the distribution → more unique tokens drawn.
    logits = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5])
    sampler = Sampler()
    counts_high = _sample_n(sampler, logits, SamplingParams(temperature=10.0, seed=10), n=500)
    counts_low = _sample_n(sampler, logits, SamplingParams(temperature=0.1, seed=10), n=500)
    assert len(counts_high) > len(counts_low), (
        f"high temp unique tokens ({len(counts_high)}) must exceed "
        f"low temp ({len(counts_low)})"
    )


def test_near_zero_temperature_concentrates_on_argmax_token() -> None:
    # With temperature ≈ 0, the distribution is nearly one-hot. Argmax token must
    # dominate heavily (≥ 95% of draws).
    logits = torch.tensor([5.0, 2.0, 1.0, 0.0])
    sampler = Sampler()
    counts = _sample_n(sampler, logits, SamplingParams(temperature=0.001, seed=11), n=300)
    argmax_frac = counts[0] / sum(counts.values())
    assert argmax_frac >= 0.95, f"argmax fraction {argmax_frac:.2f} too low for near-zero temperature"


def test_temperature_does_not_change_which_token_wins_for_greedy() -> None:
    # Any temperature combined with greedy flag must still yield argmax.
    sampler = Sampler()
    logits = torch.tensor([[0.5, 3.0, -1.0, 1.0]])
    for _ in range(10):
        out = sampler(logits, [SamplingParams(temperature=0.0)])
        assert out.item() == 1


# ===========================================================================
# seeded RNG — behavioral contracts
# ===========================================================================


def test_seeded_independent_params_objects_produce_same_first_draw() -> None:
    # Each fresh SamplingParams with the same seed starts from the same RNG state
    # and must produce the same first token for the same logits.
    sampler = Sampler()
    logits = torch.randn(1, 64)
    first_draws = [sampler(logits, [SamplingParams(temperature=1.0, seed=42)]).item() for _ in range(8)]
    assert len(set(first_draws)) == 1, f"fresh seed=42 draws must match: {first_draws}"


def test_seeded_generator_state_advances_so_consecutive_draws_differ() -> None:
    # The same SamplingParams object must advance its RNG across calls — otherwise
    # every decode step would repeat the same token.
    sampler = Sampler()
    logits = torch.randn(1, 128)
    p = SamplingParams(temperature=1.0, seed=7)
    draws = [sampler(logits, [p]).item() for _ in range(15)]
    assert len(set(draws)) > 1, "generator must advance: repeated identical draws across 15 calls is a bug"


def test_seeded_params_are_independent_even_when_sharing_same_seed() -> None:
    # Two params with the same seed have independent generators. Advancing one must
    # not affect the other's sequence.
    sampler = Sampler()
    logits = torch.randn(1, 64)
    p_a = SamplingParams(temperature=1.0, seed=55)
    p_b = SamplingParams(temperature=1.0, seed=55)
    # Burn 5 draws from p_a to advance its state.
    for _ in range(5):
        sampler(logits, [p_a])
    # p_b's first draw must still match a fresh seed=55 generator.
    p_ref = SamplingParams(temperature=1.0, seed=55)
    assert sampler(logits, [p_b]).item() == sampler(logits, [p_ref]).item(), (
        "p_b's first draw must equal fresh seed=55 draw, unaffected by p_a's state"
    )


def test_different_seeds_produce_different_sequences() -> None:
    # Different seeds must diverge — shared seed collision would invalidate per-request reproducibility.
    sampler = Sampler()
    logits = torch.randn(1, 64)
    draws_a = [sampler(logits, [SamplingParams(temperature=1.0, seed=1)]).item() for _ in range(10)]
    draws_b = [sampler(logits, [SamplingParams(temperature=1.0, seed=2)]).item() for _ in range(10)]
    assert draws_a != draws_b, "seed=1 and seed=2 must not produce identical 10-token sequences"


# ===========================================================================
# batch — behavioral contracts
# ===========================================================================


def test_batch_output_has_correct_shape_and_dtype() -> None:
    sampler = Sampler()
    batch_size, vocab = 5, 16
    out = sampler(torch.randn(batch_size, vocab), [SamplingParams(temperature=0.0)] * batch_size)
    assert out.shape == (batch_size,), f"expected shape ({batch_size},), got {out.shape}"
    assert out.dtype == torch.long, f"expected dtype long, got {out.dtype}"


def test_batch_greedy_row_always_returns_argmax() -> None:
    # Row 0 is greedy; row 1 is stochastic. Row 0 must always be argmax over 20 passes.
    sampler = Sampler()
    logits = torch.tensor([
        [0.0, 0.0, 10.0, 0.0],  # argmax = 2
        [1.0, 1.0, 1.0, 1.0],   # uniform → stochastic
    ])
    for _ in range(20):
        out = sampler(logits, [SamplingParams(temperature=0.0), SamplingParams(temperature=1.0, seed=0)])
        assert out[0].item() == 2, "greedy row must always be argmax"


def test_batch_each_row_uses_its_own_params() -> None:
    # Row 0 top_k=1 restricted to its argmax; row 1 top_k=1 restricted to its argmax.
    sampler = Sampler()
    logits = torch.tensor([
        [-10.0, -10.0, -10.0, 10.0],  # argmax = 3
        [10.0, -10.0, -10.0, -10.0],  # argmax = 0
    ])
    out = sampler(logits, [SamplingParams(temperature=1.0, top_k=1)] * 2)
    assert out[0].item() == 3 and out[1].item() == 0, f"per-row params not respected: {out.tolist()}"


def test_batch_output_tokens_are_within_vocab_bounds() -> None:
    sampler = Sampler()
    vocab = 20
    logits = torch.randn(8, vocab)
    out = sampler(logits, [SamplingParams(temperature=1.0, seed=i) for i in range(8)])
    assert (out >= 0).all() and (out < vocab).all(), f"tokens out of vocab range: {out.tolist()}"


# ===========================================================================
# numerical robustness
# ===========================================================================


def test_extreme_positive_logits_do_not_produce_nan_or_crash() -> None:
    # Very large logits — softmax uses max-shift so this must not overflow to NaN.
    logits = torch.tensor([1e20, -1e20, 1e20, -1e20])
    sampler = Sampler()
    out = sampler(logits.unsqueeze(0), [SamplingParams(temperature=1.0, seed=0)])
    assert out.item() in {0, 2}, f"expected argmax-class token (0 or 2), got {out.item()}"


def test_uniform_logits_makes_all_tokens_reachable() -> None:
    # Equal logits → equal probability → every token reachable with enough draws.
    vocab = 10
    logits = torch.zeros(vocab)
    sampler = Sampler()
    counts = _sample_n(sampler, logits, SamplingParams(temperature=1.0, seed=0), n=1_000)
    assert len(counts) == vocab, f"expected all {vocab} tokens, got {sorted(counts.keys())}"


def test_single_token_vocab_always_returns_token_0() -> None:
    sampler = Sampler()
    logits = torch.tensor([[42.0]])
    for trial in range(10):
        p = SamplingParams(temperature=1.0, seed=trial)
        assert sampler(logits, [p]).item() == 0, "single-token vocab must always return token 0"


# ===========================================================================
# input validation
# ===========================================================================


def test_1d_logits_raise_value_error() -> None:
    sampler = Sampler()
    import pytest
    with pytest.raises(ValueError, match="2D"):
        sampler(torch.randn(8), [SamplingParams()])


# ---------------------------------------------------------------------------
# Batched greedy: the fast path must not change which token wins (§7)
# ---------------------------------------------------------------------------


def test_all_greedy_batch_matches_row_by_row_argmax() -> None:
    """The whole-batch shortcut must agree with taking each row on its own."""
    torch.manual_seed(0)
    logits = torch.randn(8, 50)
    params = [SamplingParams(temperature=0.0) for _ in range(8)]

    expected = torch.stack([torch.argmax(logits[i], dim=-1) for i in range(8)])
    assert torch.equal(Sampler()(logits, params), expected)


def test_mixed_batch_gives_greedy_rows_their_argmax() -> None:
    """A stochastic neighbour must not disturb a greedy row's answer."""
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, seed=1),
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, seed=2),
    ]

    out = Sampler()(logits, params)
    assert [int(out[0]), int(out[2])] == [
        int(torch.argmax(logits[0])),
        int(torch.argmax(logits[2])),
    ]


def test_mixed_batch_leaves_stochastic_rows_stochastic() -> None:
    """The greedy shortcut must not swallow rows that asked to sample."""
    logits = torch.zeros(2, 200)
    logits[1, 7] = 0.01  # nearly uniform, so an argmax would be a giveaway
    params = [SamplingParams(temperature=0.0), SamplingParams(temperature=100.0, seed=3)]

    sampler = Sampler()  # one instance, so the seeded stream advances between draws
    draws = {int(sampler(logits, params)[1]) for _ in range(30)}
    assert len(draws) > 1


def test_top_k_1_row_is_routed_through_the_greedy_path(monkeypatch) -> None:
    """`top_k=1` is greedy by another name, and must not reach `_sample_one`."""
    sampler = Sampler()
    monkeypatch.setattr(
        sampler, "_sample_one", lambda *a: pytest.fail("top_k=1 should take the greedy path")
    )
    logits = torch.randn(3, 20)

    sampler(logits, [SamplingParams(temperature=0.7, top_k=1) for _ in range(3)])
