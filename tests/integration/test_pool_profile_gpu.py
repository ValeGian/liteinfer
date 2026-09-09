"""The activation budget is measured, not guessed.

`schedule()` prefills every admitted sequence in one pass, so the widest forward
this config allows is `max_num_seqs x max_model_len` tokens. Running it once at
load is what turns the KV pool's share from a fraction someone picked into what
this device had left — see §2.6.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_model_runner import ContinuousModelRunner

pytestmark = pytest.mark.gpu


def _runner(model_dir: Path, **overrides) -> ContinuousModelRunner:
    defaults = {
        "device": "cuda",
        "dtype": torch.float32,
        "max_num_seqs": 4,
        "max_model_len": 64,
    }
    runner = ContinuousModelRunner(
        EngineConfig(model=str(model_dir), **{**defaults, **overrides})  # type: ignore[arg-type]
    )
    runner.load_model()
    return runner


def test_the_widest_prefill_is_measured_rather_than_guessed(tiny_llama_dir: Path) -> None:
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
