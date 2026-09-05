"""How the KV block pool decides its size."""

from __future__ import annotations

import logging
import math

import pytest
import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.continuous_model_runner import ContinuousModelRunner

LAYERS, KV_HEADS, HEAD_DIM = 2, 2, 8
BLOCK_SIZE = 16


def _runner(**kwargs) -> ContinuousModelRunner:
    defaults = dict(model="unused", device="cpu", dtype=torch.float32, block_size=BLOCK_SIZE)
    return ContinuousModelRunner(EngineConfig(**{**defaults, **kwargs}))


def _blocks(runner: ContinuousModelRunner) -> int:
    return runner._compute_num_blocks(LAYERS, KV_HEADS, HEAD_DIM)


def test_pool_is_capped_at_what_the_config_can_hold() -> None:
    # max_num_seqs x max_model_len tokens is the most KV that can ever exist.
    runner = _runner(max_num_seqs=4, max_model_len=64)
    assert _blocks(runner) == math.ceil(4 * 64 / BLOCK_SIZE)


def test_a_wider_batch_needs_proportionally_more_blocks() -> None:
    narrow = _blocks(_runner(max_num_seqs=4, max_model_len=64))
    wide = _blocks(_runner(max_num_seqs=8, max_model_len=64))
    assert wide == 2 * narrow


def test_memory_caps_the_pool_when_the_config_asks_for_more() -> None:
    runner = _runner(max_num_seqs=32, max_model_len=10**7)
    assert _blocks(runner) < math.ceil(32 * 10**7 / BLOCK_SIZE)


def test_a_config_memory_cannot_serve_is_warned_about(caplog) -> None:
    runner = _runner(max_num_seqs=32, max_model_len=10**7)
    with caplog.at_level(logging.WARNING):
        _blocks(runner)
    assert "may exhaust it" in caplog.text


def test_a_config_that_fits_warns_about_nothing(caplog) -> None:
    runner = _runner(max_num_seqs=4, max_model_len=64)
    with caplog.at_level(logging.WARNING):
        _blocks(runner)
    assert caplog.text == ""


def test_an_explicit_block_count_overrides_the_calculation() -> None:
    runner = _runner(max_num_seqs=4, max_model_len=64, num_gpu_blocks=7)
    assert _blocks(runner) == 7


def test_the_chosen_size_is_reported(caplog) -> None:
    runner = _runner(max_num_seqs=4, max_model_len=64)
    with caplog.at_level(logging.INFO):
        _blocks(runner)
    assert "KV pool: 16 blocks" in caplog.text


def test_the_memory_fraction_is_tunable() -> None:
    # Only binds when memory is the constraint, so ask for more than fits.
    generous = _blocks(_runner(max_num_seqs=32, max_model_len=10**7))
    frugal = _blocks(_runner(max_num_seqs=32, max_model_len=10**7, kv_cache_memory_fraction=0.1))
    assert frugal < generous


def test_an_impossible_memory_fraction_is_rejected() -> None:
    with pytest.raises(ValueError, match="kv_cache_memory_fraction"):
        EngineConfig(model="unused", kv_cache_memory_fraction=1.5)


def test_pool_reports_its_own_footprint() -> None:
    from liteinfer.cache.block_pool import BlockPool

    pool = BlockPool(
        num_blocks=4, block_size=BLOCK_SIZE, num_layers=LAYERS, num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM, dtype=torch.float32, device=torch.device("cpu"),
    )
    # 5 blocks (one is the null block) x 16 slots x 2 layers x 2 heads x 8 dims
    # x 4 bytes, counted for keys and values.
    assert pool.nbytes == 5 * BLOCK_SIZE * LAYERS * KV_HEADS * HEAD_DIM * 4 * 2
