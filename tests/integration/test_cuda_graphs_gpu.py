"""A replayed decode forward must answer what the eager one answers.

The interesting part is what a capture *does not* freeze. `paged_decode` bounds
its loop by `context_lens`, a tensor, so one graph per batch width serves every
context length and the slot table is one fixed-width buffer whose unread columns
are never touched. If that were wrong, output would diverge as the context grew
past the length the graph was recorded at — which is what these tests generate
past.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_model_runner import ContinuousModelRunner
from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.sampling.params import SamplingParams

pytestmark = pytest.mark.gpu

_PROMPT_LENS = (6, 9, 4)
_DECODE_STEPS = 24


def _runner(
    model_dir: Path, *, capture: bool, max_num_seqs: int, splits: int | None = None
) -> ContinuousModelRunner:
    config = EngineConfig(
        model=str(model_dir),
        device="cuda",
        dtype=torch.float32,  # type: ignore[arg-type]
        max_num_seqs=max_num_seqs,
        max_model_len=64,
        enable_cuda_graphs=capture,
        paged_decode_splits=splits,
    )
    runner = ContinuousModelRunner(config)
    runner.load_model()
    return runner


def _sequences(prompt_lens: tuple[int, ...]) -> list[Sequence]:
    params = SamplingParams(temperature=0.0, max_tokens=_DECODE_STEPS + 2, ignore_eos=True)
    return [
        Sequence(
            request_id=f"seq-{i}",
            prompt="",
            prompt_token_ids=list(range(2, 2 + prompt_len)),
            sampling_params=params,
            status=SequenceStatus.RUNNING,
        )
        for i, prompt_len in enumerate(prompt_lens)
    ]


def _greedy_tokens(runner: ContinuousModelRunner, prompt_lens: tuple[int, ...]) -> list[list[int]]:
    """Decode greedily for a fixed number of steps and return each sequence's tokens."""
    seqs = _sequences(prompt_lens)
    logits = runner.prefill(seqs)
    for i, seq in enumerate(seqs):
        seq.output_token_ids.append(int(logits[i].argmax()))
    for _ in range(_DECODE_STEPS):
        logits = runner.decode(seqs)
        for i, seq in enumerate(seqs):
            seq.output_token_ids.append(int(logits[i].argmax()))
    return [list(seq.output_token_ids) for seq in seqs]


def test_a_replayed_decode_gives_the_same_tokens_as_an_eager_one(tiny_llama_dir: Path):
    """The whole claim: same answer, fewer launches.

    The contexts here start at 4-9 tokens and grow by 24, so most of these steps
    are replays of a graph recorded at a shorter context than they run at.
    """
    eager = _greedy_tokens(_runner(tiny_llama_dir, capture=False, max_num_seqs=4), _PROMPT_LENS)
    captured = _greedy_tokens(_runner(tiny_llama_dir, capture=True, max_num_seqs=4), _PROMPT_LENS)

    assert captured == eager


def test_a_replayed_decode_matches_eager_for_a_single_sequence(tiny_llama_dir: Path):
    """Batch 1 is the width the item exists for, and its own capture."""
    eager = _greedy_tokens(_runner(tiny_llama_dir, capture=False, max_num_seqs=1), (5,))
    captured = _greedy_tokens(_runner(tiny_llama_dir, capture=True, max_num_seqs=1), (5,))

    assert captured == eager


def test_one_graph_is_captured_per_batch_width(tiny_llama_dir: Path):
    """Repeating a width must replay, not re-record."""
    runner = _runner(tiny_llama_dir, capture=True, max_num_seqs=4)

    _greedy_tokens(runner, _PROMPT_LENS)

    assert runner.captured_decode_widths == [len(_PROMPT_LENS)]


def test_a_narrowing_batch_captures_each_width_it_visits(tiny_llama_dir: Path):
    """Sequences finish at different times, so the batch drains through widths."""
    runner = _runner(tiny_llama_dir, capture=True, max_num_seqs=4)
    seqs = _sequences(_PROMPT_LENS)
    runner.prefill(seqs)
    for seq in seqs:
        seq.output_token_ids.append(3)

    while len(seqs) > 1:
        runner.decode(seqs)
        for seq in seqs:
            seq.output_token_ids.append(3)
        runner.deregister_sequence(seqs.pop())  # retire one, as the scheduler would
    runner.decode(seqs)

    assert sorted(runner.captured_decode_widths) == [1, 2, 3]


def test_the_split_count_does_not_follow_the_context(tiny_llama_dir: Path):
    """What makes one capture per width enough.

    A graph bakes scalar kernel arguments in, so a count chosen from this step's
    context would be frozen at the context the graph was recorded at and wrong
    for every step after. The policy reads `max_model_len` instead, so generating
    must not move it.
    """
    runner = _runner(tiny_llama_dir, capture=True, max_num_seqs=4)
    before = runner.splits_for_width(1)
    _greedy_tokens(runner, (5,))

    assert runner.splits_for_width(1) == before


def test_a_narrow_batch_is_split_more_finely_than_a_wide_one(tiny_llama_dir: Path):
    """The count is per width because the device fills at different widths."""
    runner = _runner(tiny_llama_dir, capture=True, max_num_seqs=4)

    assert runner.splits_for_width(1) >= runner.splits_for_width(4)


def test_a_pinned_split_count_overrides_the_policy(tiny_llama_dir: Path):
    """A stored benchmark row has to keep measuring the grid it was measured on."""
    runner = _runner(tiny_llama_dir, capture=True, max_num_seqs=4, splits=1)

    assert runner.splits_for_width(1) == 1


def test_a_replayed_split_decode_gives_the_same_tokens_as_an_eager_one(tiny_llama_dir: Path):
    """The split count is a scalar, and a capture freezes scalars.

    Four splits over contexts of 4 to 30 tokens means most programs hold one key
    or none, and the graph was recorded at the shortest context of all — so this
    generates past it. Divergence here would mean the count, or the emptiness of
    a split, had been baked in where it should follow `context_lens`.
    """
    eager = _greedy_tokens(
        _runner(tiny_llama_dir, capture=False, max_num_seqs=4, splits=4), _PROMPT_LENS
    )
    captured = _greedy_tokens(
        _runner(tiny_llama_dir, capture=True, max_num_seqs=4, splits=4), _PROMPT_LENS
    )

    assert captured == eager


def test_a_replayed_split_decode_matches_an_unsplit_one(tiny_llama_dir: Path):
    """Splitting is a grid, not an answer: the captured forward must not notice."""
    unsplit = _greedy_tokens(
        _runner(tiny_llama_dir, capture=True, max_num_seqs=4, splits=1), _PROMPT_LENS
    )
    split = _greedy_tokens(
        _runner(tiny_llama_dir, capture=True, max_num_seqs=4, splits=4), _PROMPT_LENS
    )

    assert split == unsplit


def test_capture_is_off_when_a_gathering_kernel_is_pinned(tiny_llama_dir: Path):
    """`sdpa` needs a mask as wide as the batch's longest context, which a graph cannot hold."""
    config = EngineConfig(
        model=str(tiny_llama_dir),
        device="cuda",
        dtype=torch.float32,  # type: ignore[arg-type]
        max_num_seqs=2,
        max_model_len=64,
        attn_implementation="sdpa",
    )
    runner = ContinuousModelRunner(config)
    runner.load_model()

    assert runner.captured_decode_widths == []
