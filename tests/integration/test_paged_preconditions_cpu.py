"""How the engine chooses its attention kernel, and what it refuses.

`attn_implementation=None` means "the fastest kernel that runs here". These
tests pin the CPU half of that decision — the paged kernel is CUDA-only, so on
CPU the choice must fall back rather than fail, and an explicit request for it
must fail rather than be silently downgraded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_model_runner import ContinuousModelRunner


def _runner(model_dir: Path, **overrides) -> ContinuousModelRunner:
    config = EngineConfig(
        model=str(model_dir),
        device="cpu",
        dtype=torch.float32,  # type: ignore[arg-type]
        max_num_seqs=2,
        max_model_len=64,
        **overrides,
    )
    return ContinuousModelRunner(config)


def test_the_default_falls_back_to_a_kernel_this_device_can_run(tiny_llama_dir: Path) -> None:
    runner = _runner(tiny_llama_dir)

    runner.load_model()

    assert runner.attn_implementation == "sdpa"


def test_an_explicit_kernel_is_honoured_rather_than_re_chosen(tiny_llama_dir: Path) -> None:
    """Asking for `eager` must get `eager`, even though it is not what would be picked."""
    runner = _runner(tiny_llama_dir, attn_implementation="eager")

    runner.load_model()

    assert runner.attn_implementation == "eager"


def test_asking_for_paged_where_it_cannot_run_is_refused(tiny_llama_dir: Path) -> None:
    """A silent downgrade would make a benchmark row or a parity test measure the wrong kernel."""
    runner = _runner(tiny_llama_dir, attn_implementation="paged")

    with pytest.raises(ValueError, match="cannot run here"):
        runner.load_model()
