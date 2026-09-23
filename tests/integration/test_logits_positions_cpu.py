"""The LM head runs only where logits are read.

Its output is ``batch x positions x vocab``, which for a prefill is the largest
tensor the engine allocates — 3.91 GiB at batch 8 with 2,048-token prompts, of
which one row per sequence is ever read. These tests pin the contract that keeps
the rest from being computed: what `logits_positions` selects, and that selecting
does not change the answer.
"""

from __future__ import annotations

from pathlib import Path

import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.attention_mask import builders_for
from liteinfer.engine.continuous_model_runner import ContinuousModelRunner
from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.models import LAST_POSITION
from liteinfer.sampling.params import SamplingParams

_PROMPT_LENS = (7, 4, 9)


def _loaded_runner(model_dir: Path) -> ContinuousModelRunner:
    runner = ContinuousModelRunner(
        EngineConfig(
            model=str(model_dir),
            device="cpu",
            dtype=torch.float32,  # type: ignore[arg-type]
            max_num_seqs=len(_PROMPT_LENS),
            max_model_len=64,
        )
    )
    runner.load_model()
    return runner


def _prefill_logits(runner: ContinuousModelRunner, logits_positions: slice | None):
    """One prefill forward, built the way `prefill` builds it."""
    params = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    seqs = [
        Sequence(
            request_id=f"seq-{i}-{logits_positions}",
            prompt="",
            prompt_token_ids=list(range(2, 2 + prompt_len)),
            sampling_params=params,
            status=SequenceStatus.RUNNING,
        )
        for i, prompt_len in enumerate(_PROMPT_LENS)
    ]
    prompt_lens: list[int] = list(_PROMPT_LENS)
    request_ids = [s.request_id for s in seqs]
    cache = runner._cache
    assert cache is not None, "load_model() builds the cache"
    for request_id in request_ids:
        cache.register(request_id)
    chunks = runner._next_prompt_chunks(seqs, None)
    cache.advance(request_ids, prompt_lens)

    input_ids, position_ids = runner._build_prefill_inputs(chunks)
    build_prefill, _ = builders_for(type(runner.model).__name__)
    mask = build_prefill(prompt_lens, runner.config.dtype, runner.device)
    payload = cache.make_prefill_payload(request_ids, prompt_lens)

    with torch.inference_mode():
        out = runner.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=payload,
            attention_mask=mask,
            logits_positions=logits_positions,
        )
    for seq in seqs:
        runner.deregister_sequence(seq)
    return out.logits


def test_asking_for_the_last_position_returns_one_row_per_sequence(tiny_llama_dir: Path):
    """The row count is the whole saving: everything else is never projected."""
    logits = _prefill_logits(_loaded_runner(tiny_llama_dir), LAST_POSITION)

    assert logits.shape[1] == 1


def test_asking_for_nothing_in_particular_returns_every_position(tiny_llama_dir: Path):
    """`None` keeps the model a language model, which is what parity compares."""
    logits = _prefill_logits(_loaded_runner(tiny_llama_dir), None)

    assert logits.shape[1] == max(_PROMPT_LENS)


def test_the_sliced_logits_equal_the_row_they_were_sliced_from(tiny_llama_dir: Path):
    """Projecting fewer positions must not change the ones that are projected.

    Not bit-exact: the LM head is a GEMM over 1 position in one call and all of
    them in the other, and CPU BLAS blocks the reduction differently per shape,
    so fp32 sums round differently (observed up to 1.5e-5 on CI runners). A
    wrong row would differ at the scale of the logits themselves, far above atol.
    """
    runner = _loaded_runner(tiny_llama_dir)
    sliced = _prefill_logits(runner, LAST_POSITION)[:, -1, :]
    whole = _prefill_logits(runner, None)[:, -1, :]

    torch.testing.assert_close(sliced, whole, rtol=0, atol=1e-4)
