"""The activation budget is measured, not guessed.

`schedule()` hands out up to `token_budget` tokens a step, so the widest forward
this config allows is that many prompt tokens in one packed pass — or, for a
kernel that copies each context out of the pool, a decode step at full width.
Running them once at load is what turns the KV pool's share from a fraction
someone picked into what this device had left — see §2.6.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.model_runner import ModelRunner

pytestmark = pytest.mark.gpu


def _runner(model_dir: Path, **overrides) -> ModelRunner:
    defaults = {
        "device": "cuda",
        "dtype": torch.float32,
        "max_num_seqs": 4,
        "max_model_len": 64,
    }
    runner = ModelRunner(
        EngineConfig(model=str(model_dir), **{**defaults, **overrides})  # type: ignore[arg-type]
    )
    runner.load_model()
    return runner


def test_the_widest_step_is_measured_rather_than_guessed(tiny_llama_dir: Path) -> None:
    """A real forward, run before the pool exists, is what replaces the fraction."""
    runner = _runner(tiny_llama_dir)

    assert runner._forward_bytes > 0


def test_a_wider_config_measures_more_activations(tiny_llama_dir: Path) -> None:
    """The measurement has to track the shape it is measuring, or it is still a guess."""
    narrow = _runner(tiny_llama_dir, max_num_seqs=2)._forward_bytes
    wide = _runner(tiny_llama_dir, max_num_seqs=8)._forward_bytes

    assert wide > narrow


def test_the_engine_still_serves_after_being_profiled(tiny_llama_dir: Path) -> None:
    """The profile allocates a full-width forward, so it has to leave the engine usable."""
    runner = _runner(tiny_llama_dir)

    assert runner._cache is not None


def _full_width_decode_copy_bytes(runner: ModelRunner) -> int:
    """K and V of every row's whole context, which a dense decode step holds at once (fp32)."""
    hf_config = runner.hf_config
    assert hf_config is not None
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    per_tensor = (
        runner.config.max_num_seqs * runner.config.max_model_len * hf_config.num_key_value_heads * head_dim * 4
    )
    return 2 * per_tensor


# A token budget as small as the config allows, against long contexts: the
# widest packed pass is then a few tokens, and a full-width decode is not.
_SMALL_BUDGET_LONG_CONTEXT = {"max_num_seqs": 8, "max_model_len": 2048, "max_num_batched_tokens": 8}


def test_a_dense_kernel_reserves_for_the_decode_copy_its_token_budget_does_not_cap(
    tiny_llama_dir: Path,
) -> None:
    """`sdpa` copies every row's context out of the pool, however few tokens a step may hold."""
    runner = _runner(tiny_llama_dir, attn_implementation="sdpa", **_SMALL_BUDGET_LONG_CONTEXT)

    assert runner._forward_bytes >= _full_width_decode_copy_bytes(runner)


def test_the_paged_kernel_reserves_nothing_for_a_decode_copy_it_never_makes(
    tiny_llama_dir: Path,
) -> None:
    """It reads the pool in place, so its budget is the widest packed pass alone."""
    runner = _runner(tiny_llama_dir, attn_implementation="paged", **_SMALL_BUDGET_LONG_CONTEXT)

    assert runner._forward_bytes < _full_width_decode_copy_bytes(runner)
