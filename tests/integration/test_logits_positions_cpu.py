# pyright: reportPrivateImportUsage=false
"""The LM head runs only where logits are read.

Its output is ``positions x vocab``, which for a prefill is the largest tensor
the engine allocates — 3.91 GiB for 8 prompts of 2,048 tokens, of which one row
per sequence is ever read. These tests pin the contract that keeps the rest from
being computed: what `logits_positions` selects, and that selecting does not
change the answer.
"""

from __future__ import annotations

from pathlib import Path

import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_model_runner import ContinuousModelRunner

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


def _last_tokens() -> torch.Tensor:
    """Where each prompt of the packed run ends."""
    return torch.tensor(_PROMPT_LENS).cumsum(0) - 1


def _prefill_logits(runner: ContinuousModelRunner, logits_positions: torch.Tensor | None):
    """One packed prefill forward over `_PROMPT_LENS`, built the way `execute` builds it."""
    cache, model = runner._cache, runner.model
    assert cache is not None and model is not None, "load_model() builds both"
    request_ids = [f"seq-{i}" for i in range(len(_PROMPT_LENS))]
    prompt_lens: list[int] = list(_PROMPT_LENS)
    for request_id in request_ids:
        cache.register(request_id)
    cache.advance(request_ids, prompt_lens)
    payload = cache.make_packed_payload(request_ids, prompt_lens)
    token_ids = [token for length in prompt_lens for token in range(2, 2 + length)]

    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([token_ids]),
            position_ids=payload.addresses.positions,
            past_key_values=payload,
            logits_positions=logits_positions,
        )
    for request_id in request_ids:
        cache.deregister(request_id)
    return out.logits


def test_asking_for_each_prompt_s_last_token_returns_one_row_per_sequence(tiny_llama_dir: Path):
    """The row count is the whole saving: everything else is never projected."""
    logits = _prefill_logits(_loaded_runner(tiny_llama_dir), _last_tokens())

    assert logits.shape[1] == len(_PROMPT_LENS)


def test_asking_for_nothing_in_particular_returns_every_position(tiny_llama_dir: Path):
    """`None` keeps the model a language model, which is what parity compares."""
    logits = _prefill_logits(_loaded_runner(tiny_llama_dir), None)

    assert logits.shape[1] == sum(_PROMPT_LENS)


def test_the_selected_logits_equal_the_rows_they_were_selected_from(tiny_llama_dir: Path):
    """Projecting fewer positions must not change the ones that are projected.

    Not bit-exact: the LM head is a GEMM over 3 positions in one call and all of
    them in the other, and CPU BLAS blocks the reduction differently per shape,
    so fp32 sums round differently (observed up to 1.5e-5 on CI runners). A
    wrong row would differ at the scale of the logits themselves, far above atol.
    """
    runner = _loaded_runner(tiny_llama_dir)
    selected = _prefill_logits(runner, _last_tokens())
    whole = _prefill_logits(runner, None)[:, _last_tokens()]

    torch.testing.assert_close(selected, whole, rtol=0, atol=1e-4)
