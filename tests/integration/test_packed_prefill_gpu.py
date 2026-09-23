"""A packed prefill must answer what a padded one answers.

Packing changes the layout a prompt arrives in, the kernel that attends to it,
and the addresses its K/V are written through. What it must not change is the
tokens that come out — so every test here runs the same prompts both ways.
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

# Lengths that differ, because equal lengths are the case padding costs nothing on.
_PROMPT_LENS = (11, 3, 27, 5)
_DECODE_STEPS = 8


def _runner(model_dir: Path, *, packed: bool) -> ContinuousModelRunner:
    config = EngineConfig(
        model=str(model_dir),
        device="cuda",
        dtype=torch.bfloat16,  # type: ignore[arg-type]
        max_num_seqs=len(_PROMPT_LENS),
        max_model_len=64,
        attn_implementation="paged",
        enable_cuda_graphs=False,
        enable_packed_prefill=packed,
    )
    runner = ContinuousModelRunner(config)
    runner.load_model()
    return runner


def _greedy_tokens(runner: ContinuousModelRunner) -> list[list[int]]:
    params = SamplingParams(temperature=0.0, max_tokens=_DECODE_STEPS + 2, ignore_eos=True)
    seqs = [
        Sequence(
            request_id=f"seq-{i}",
            prompt="",
            prompt_token_ids=list(range(2, 2 + prompt_len)),
            sampling_params=params,
            status=SequenceStatus.RUNNING,
        )
        for i, prompt_len in enumerate(_PROMPT_LENS)
    ]
    logits = runner.prefill(seqs)
    for i, seq in enumerate(seqs):
        seq.output_token_ids.append(int(logits[i].argmax()))
    for _ in range(_DECODE_STEPS):
        logits = runner.decode(seqs)
        for i, seq in enumerate(seqs):
            seq.output_token_ids.append(int(logits[i].argmax()))
    return [list(seq.output_token_ids) for seq in seqs]


def test_a_packed_prefill_generates_what_a_padded_one_generates(tiny_llama_dir: Path):
    """The whole claim: same tokens, less work."""
    padded = _greedy_tokens(_runner(tiny_llama_dir, packed=False))
    packed = _greedy_tokens(_runner(tiny_llama_dir, packed=True))

    assert packed == padded


def test_a_packed_prefill_fills_the_cache_the_decode_path_reads(tiny_llama_dir: Path):
    """Decode reads the pool the prefill wrote, so a wrong slot shows up as drift.

    The test above would catch that too, but only after the fact; this one says
    where to look — the first decoded token depends on the prompt's K/V alone.
    """
    padded = _greedy_tokens(_runner(tiny_llama_dir, packed=False))
    packed = _greedy_tokens(_runner(tiny_llama_dir, packed=True))

    assert [tokens[1] for tokens in packed] == [tokens[1] for tokens in padded]


def test_packing_is_on_by_default_where_it_can_run(tiny_llama_dir: Path):
    """Half precision on CUDA is where the flash kernel runs, so that is where packing happens."""
    runner = _runner(tiny_llama_dir, packed=None)  # type: ignore[arg-type]

    assert runner._packs_prefill


def test_asking_to_pack_where_flash_cannot_run_is_refused(tiny_llama_dir: Path):
    """Single precision has no flash kernel; a config that asks for one hears why."""
    config = EngineConfig(
        model=str(tiny_llama_dir),
        device="cuda",
        dtype=torch.float32,  # type: ignore[arg-type]
        max_model_len=64,
        attn_implementation="paged",
        enable_packed_prefill=True,
    )

    with pytest.raises(ValueError, match="half precision"):
        ContinuousModelRunner(config).load_model()


def test_a_dense_kernel_is_never_handed_a_packed_batch(tiny_llama_dir: Path):
    """`eager` and `sdpa` read a padded batch and a mask, so the engine keeps padding for them.

    Without this the packed payload would reach `eager_attention`, whose input
    type has the same field names — it would attend across prompt boundaries and
    return tokens rather than an error, which is how the e2e parity suite caught
    this in the first place.
    """
    config = EngineConfig(
        model=str(tiny_llama_dir),
        device="cuda",
        dtype=torch.bfloat16,  # type: ignore[arg-type]
        max_model_len=64,
        attn_implementation="eager",
    )
    runner = ContinuousModelRunner(config)
    runner.load_model()

    assert not runner._packs_prefill
