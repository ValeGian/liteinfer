"""When the decode forward may be captured, and when it must not be.

A capture freezes every shape and every pointer it records, so it is only legal
where the engine can promise those do not change between steps. These tests pin
that promise; whether a replay answers correctly is
`tests/integration/test_cuda_graphs_gpu.py`.
"""

from __future__ import annotations

import pytest
import torch

from liteinfer.cache.block_pool import BlockPool
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache
from liteinfer.engine.cuda_graphs import (
    _MAX_CAPTURES,
    DecodeGraphs,
    graphs_are_enabled,
    unsupported_reason,
)

_CPU = torch.device("cpu")
_CUDA = torch.device("cuda")  # a device object, not a claim that one is present


def test_a_cpu_device_cannot_capture():
    assert "CUDA device" in (unsupported_reason(_CPU, "paged") or "")


def test_a_gathering_kernel_cannot_capture():
    """The dense kernels take a mask as wide as the batch's longest context."""
    assert "attention mask" in (unsupported_reason(_CUDA, "sdpa") or "")


def test_the_paged_kernel_on_cuda_can_capture():
    """Its bounds are a tensor, so one capture serves every context length."""
    assert unsupported_reason(_CUDA, "paged") is None


def test_capture_is_on_by_default_where_it_can_run():
    assert graphs_are_enabled(None, _CUDA, "paged") is True


def test_capture_is_off_by_default_where_it_cannot_run():
    """Falling back is right for a default; it is the *request* that must not be."""
    assert graphs_are_enabled(None, _CPU, "paged") is False


def test_capture_can_be_turned_off_where_it_would_have_run():
    assert graphs_are_enabled(False, _CUDA, "paged") is False


def test_asking_for_capture_where_it_cannot_run_is_refused():
    """A silent downgrade would make a benchmark row measure the wrong engine."""
    with pytest.raises(ValueError, match="cannot be captured"):
        graphs_are_enabled(True, _CPU, "paged")


def _graphs_on_cpu(max_num_seqs: int) -> DecodeGraphs:
    """A `DecodeGraphs` whose buffers exist but whose model is never called."""
    pool = BlockPool(
        num_blocks=4, block_size=4, num_layers=1, num_kv_heads=1,
        head_dim=8, dtype=torch.float32, device=_CPU,
    )
    return DecodeGraphs(
        torch.nn.Identity(),
        ContinuousKVCache(pool),
        device=_CPU,
        max_num_seqs=max_num_seqs,
        max_model_len=16,
    )


def test_a_batch_wider_than_the_engine_allows_is_refused():
    """The buffers are sized to `max_num_seqs`; a wider batch would read past them."""
    graphs = _graphs_on_cpu(max_num_seqs=2)

    with pytest.raises(ValueError, match="exceeds max_num_seqs"):
        graphs.run(
            torch.zeros(3, 1, dtype=torch.long),
            torch.zeros(3, 1, dtype=torch.long),
            torch.zeros(3, 4, dtype=torch.long),
            torch.zeros(3, dtype=torch.int32),
        )


def test_widths_past_the_capture_cap_run_eager_instead():
    """Graphs are bounded so a very wide engine cannot accumulate them without limit."""
    graphs = _graphs_on_cpu(max_num_seqs=_MAX_CAPTURES + 8)
    graphs._graphs = dict.fromkeys(range(1, _MAX_CAPTURES + 1))  # type: ignore[arg-type]

    assert graphs.has_capacity_for(_MAX_CAPTURES + 1) is False


def test_an_already_captured_width_is_always_replayable():
    """The cap bounds new captures, not the ones already paid for."""
    graphs = _graphs_on_cpu(max_num_seqs=_MAX_CAPTURES + 8)
    graphs._graphs = dict.fromkeys(range(1, _MAX_CAPTURES + 1))  # type: ignore[arg-type]

    assert graphs.has_capacity_for(_MAX_CAPTURES) is True


def test_a_slot_table_wider_than_the_engine_allows_is_refused():
    """Slicing the last `max_model_len` columns of a wider table would keep the wrong ones."""
    graphs = _graphs_on_cpu(max_num_seqs=2)

    with pytest.raises(ValueError, match="max_model_len=16"):
        graphs.run(
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 20, dtype=torch.long),
            torch.zeros(1, dtype=torch.int32),
        )
