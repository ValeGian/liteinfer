# pyright: reportPrivateImportUsage=false
"""A step holding prompts and sampled tokens together must compute what separate passes do.

Under the paged kernel the engine packs, so such a step is one forward:
sampled tokens, a chunk continuing a cached prompt and a fresh prompt side by
side, every row read out of the pool by `paged_prefill`. Each row is checked
against a single-precision `eager` reference of the same schedule, and allowed at most
twice the drift the same rows show when the step is run as separate passes — a
fixed tolerance cannot work here, see `test_token_budget_gpu.py`.

Compared as logits, not greedy tokens: the tiny model's greedy output barely
depends on attention.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_model_runner import ContinuousModelRunner
from tests.integration.sequences import running_sequence

pytestmark = pytest.mark.gpu

# Two sequences decoding at different contexts, one prompt continuing a chunk
# cached by an earlier pass, and one prompt arriving whole.
_DECODING_PROMPT_LENS = (23, 9)
_CHUNKED_PROMPT_LEN = 30
_FIRST_CHUNK = 12
_MIXED_CHUNK = 10
_FRESH_PROMPT_LEN = 17
_DRIFT_ALLOWANCE = 2.0
# Mixed and separate bf16 runs measured identical on an A40. The bound leaves
# room for a kernel that reorders its sums, and sits well under the 0.31 a
# one-token RoPE shift moves a decoding row.
_SAME_RUN_TOLERANCE = {"rtol": 0, "atol": 0.02}


def _runner(model_dir: Path, is_reference: bool, capture: bool = True) -> ContinuousModelRunner:
    """The bf16 engine under test, or the reference: fp32 through `eager`.

    The reference names its kernel because the default one packs in fp32 too,
    and would then share `paged_prefill` with the runs it is meant to judge.
    `eager` pads, writes every score out, and cannot be captured.
    """
    kernel = {"dtype": torch.float32, "attn_implementation": "eager"} if is_reference else {
        "dtype": torch.bfloat16, "enable_cuda_graphs": capture
    }
    runner = ContinuousModelRunner(
        EngineConfig(
            model=str(model_dir), device="cuda", max_num_seqs=4, max_model_len=64,
            **kernel,  # type: ignore[arg-type]
        )
    )
    runner.load_model()
    return runner


class _Schedule:
    """The same four sequences brought to the same mixed step, on one runner."""

    def __init__(self, model_dir: Path, is_reference: bool = False, capture: bool = True) -> None:
        self.runner = _runner(model_dir, is_reference, capture)
        self.decoding = [
            running_sequence(f"decoding-{i}", length, 40 * i)
            for i, length in enumerate(_DECODING_PROMPT_LENS)
        ]
        self.chunked = running_sequence("chunked", _CHUNKED_PROMPT_LEN, 100)
        self.fresh = running_sequence("fresh", _FRESH_PROMPT_LEN, 160)

        self.runner.execute(self.decoding)
        self.runner.execute([self.chunked], [_FIRST_CHUNK])
        for seq in self.decoding:
            seq.output_token_ids.append(5)

    def mixed_step(self) -> torch.Tensor:
        """Every row in one call: decode rows first, as the scheduler orders them."""
        return self.runner.execute(
            [*self.decoding, self.chunked, self.fresh],
            [1] * len(self.decoding) + [_MIXED_CHUNK, _FRESH_PROMPT_LEN],
        )

    def separate_steps(self) -> torch.Tensor:
        """The same rows one kind per call, in the same order."""
        return torch.cat([
            self.runner.execute(self.decoding),
            self.runner.execute([self.chunked], [_MIXED_CHUNK]),
            self.runner.execute([self.fresh]),
        ])

    def decode_after(self) -> torch.Tensor:
        """A uniform decode step over what the step wrote, which a captured graph replays."""
        after = [*self.decoding, self.fresh]
        for seq in after:
            seq.output_token_ids.append(6)
        return self.runner.execute(after)


_ROWS = {
    "decoding": slice(0, len(_DECODING_PROMPT_LENS)),
    "chunked": slice(len(_DECODING_PROMPT_LENS), len(_DECODING_PROMPT_LENS) + 1),
    "fresh": slice(len(_DECODING_PROMPT_LENS) + 1, None),
}


def _run(model_dir: Path, is_mixed: bool, is_reference: bool = False) -> dict[str, torch.Tensor]:
    schedule = _Schedule(model_dir, is_reference)
    assert schedule.runner._packs_prefill != is_reference, "the engine packs; the reference pads"
    step = schedule.mixed_step() if is_mixed else schedule.separate_steps()
    rows = {kind: step[rows].float() for kind, rows in _ROWS.items()}
    rows["after"] = schedule.decode_after().float()
    return rows


@pytest.fixture(scope="module")
def runs(tiny_llama_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "reference": _run(tiny_llama_dir, is_mixed=False, is_reference=True),
        "separate": _run(tiny_llama_dir, is_mixed=False),
        "mixed": _run(tiny_llama_dir, is_mixed=True),
    }


def _drift(runs: dict[str, dict[str, torch.Tensor]], run: str, kind: str) -> float:
    return (runs[run][kind] - runs["reference"][kind]).abs().max().item()


@pytest.mark.parametrize("kind", ["decoding", "chunked", "fresh"])
def test_each_row_of_a_mixed_step_computes_what_its_own_pass_does(runs, kind: str):
    assert _drift(runs, "mixed", kind) <= _DRIFT_ALLOWANCE * _drift(runs, "separate", kind)


@pytest.mark.parametrize("kind", ["decoding", "chunked", "fresh", "after"])
def test_a_mixed_step_matches_the_same_rows_run_separately(runs, kind: str):
    """Sharper than the reference bound, which a bug in code both runs share would loosen.

    Only what differs between the two runs is checked here — row order, one-query
    rows inside `paged_prefill`, their boundaries and write slots. Code both runs
    share, such as the position builder, is pinned by its own unit tests.
    """
    torch.testing.assert_close(runs["mixed"][kind], runs["separate"][kind], **_SAME_RUN_TOLERANCE)


def test_a_captured_decode_reads_what_a_mixed_step_wrote(runs):
    """Every row's K/V lands where the next step reads it, or this step's logits drift."""
    assert _drift(runs, "mixed", "after") <= _DRIFT_ALLOWANCE * _drift(runs, "separate", "after")


def test_a_mixed_step_runs_one_forward(tiny_llama_dir: Path):
    """Uncaptured, so a decode pass would count here rather than replay a graph unseen."""
    schedule = _Schedule(tiny_llama_dir, capture=False)
    model = schedule.runner.model
    assert model is not None
    num_forwards = 0
    original_forward = model.forward

    def _counting_forward(*args, **kwargs):
        nonlocal num_forwards
        num_forwards += 1
        return original_forward(*args, **kwargs)

    model.forward = _counting_forward  # type: ignore[method-assign]
    schedule.mixed_step()

    assert num_forwards == 1
