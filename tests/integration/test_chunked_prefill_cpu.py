"""A prompt prefilled in chunks must leave the model where one whole pass leaves it.

Compared as logits rather than greedy tokens on purpose: the tiny random model
mostly repeats its input token whatever attention returns, so its greedy output
cannot see a chunk that attends to the wrong keys. Its logits can, and in single
precision on CPU the two paths should agree to rounding.

CPU runs the padded path: a continuing chunk reads its prefix back through the
right-aligned table and is masked by an offset causal diagonal. The packed path
is checked the same way in `test_token_budget_gpu.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_model_runner import ContinuousModelRunner
from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.sampling.params import SamplingParams

# Prime to each other and to the block size, so chunk windows start part-way
# into blocks and straddle their boundaries.
_BLOCK_SIZE = 4
_PROMPT_LENS = (13, 6)
_CHUNKS = ((5, 3), (3, 3), (5, 0))  # per pass, per sequence; 0 sits the pass out
_DECODE_STEPS = 3
_TOLERANCE = {"rtol": 0, "atol": 1e-5}


def _runner(model_dir: Path) -> ContinuousModelRunner:
    runner = ContinuousModelRunner(
        EngineConfig(
            model=str(model_dir), device="cpu", dtype=torch.float32,  # type: ignore[arg-type]
            max_num_seqs=len(_PROMPT_LENS), max_model_len=64, block_size=_BLOCK_SIZE,
        )
    )
    runner.load_model()
    return runner


def _sequences() -> list[Sequence]:
    return [
        Sequence(
            request_id=f"seq-{i}",
            prompt="",
            prompt_token_ids=[2 + (7 * i + j) % 250 for j in range(prompt_len)],
            sampling_params=SamplingParams(temperature=0.0),
            status=SequenceStatus.RUNNING,
        )
        for i, prompt_len in enumerate(_PROMPT_LENS)
    ]


def _prefill_in_chunks(runner: ContinuousModelRunner, seqs: list[Sequence]) -> torch.Tensor:
    """Run `_CHUNKS`; return each sequence's logits from the pass that ended its prompt."""
    last_logits: dict[str, torch.Tensor] = {}
    for counts in _CHUNKS:
        batch = [(seq, count) for seq, count in zip(seqs, counts, strict=True) if count]
        logits = runner.prefill([seq for seq, _ in batch], [count for _, count in batch])
        for row, (seq, _) in enumerate(batch):
            last_logits[seq.request_id] = logits[row]
    return torch.stack([last_logits[seq.request_id] for seq in seqs])


def _decode_logits(runner: ContinuousModelRunner, seqs: list[Sequence]) -> list[torch.Tensor]:
    """Feed a fixed token for a few steps, so both runs decode the same inputs."""
    steps = []
    for step in range(_DECODE_STEPS):
        for seq in seqs:
            seq.output_token_ids.append(3 + step)
        steps.append(runner.decode(seqs))
    return steps


def test_the_chunk_plan_covers_every_prompt_exactly() -> None:
    """Guards the fixture: a plan that stopped short would compare a partial prompt."""
    assert [sum(counts) for counts in zip(*_CHUNKS, strict=True)] == list(_PROMPT_LENS)


def test_a_prompt_prefilled_in_chunks_predicts_what_one_pass_predicts(tiny_llama_dir: Path):
    whole = _runner(tiny_llama_dir).prefill(_sequences())
    chunked = _prefill_in_chunks(_runner(tiny_llama_dir), _sequences())

    torch.testing.assert_close(chunked, whole, **_TOLERANCE)


@pytest.mark.parametrize("step", range(_DECODE_STEPS))
def test_decoding_after_a_chunked_prefill_reads_the_same_history(tiny_llama_dir: Path, step: int):
    """Decode reads the pool the chunks wrote, so a misplaced token shows up here."""
    whole_runner, whole_seqs = _runner(tiny_llama_dir), _sequences()
    whole_runner.prefill(whole_seqs)
    chunked_runner, chunked_seqs = _runner(tiny_llama_dir), _sequences()
    _prefill_in_chunks(chunked_runner, chunked_seqs)

    torch.testing.assert_close(
        _decode_logits(chunked_runner, chunked_seqs)[step],
        _decode_logits(whole_runner, whole_seqs)[step],
        **_TOLERANCE,
    )
