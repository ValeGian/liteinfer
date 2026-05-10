"""Unit tests for BlockPool."""

from __future__ import annotations

import pytest
import torch

from liteinfer.cache.block_pool import BlockPool, BlockPoolExhaustedError

_B = 8      # num_blocks
_S = 4      # block_size
_L = 2      # num_layers
_H = 2      # num_kv_heads
_D = 8      # head_dim
_DT = torch.float32
_DEV = torch.device("cpu")


def _pool(**kwargs) -> BlockPool:
    defaults = dict(num_blocks=_B, block_size=_S, num_layers=_L, num_kv_heads=_H, head_dim=_D, dtype=_DT, device=_DEV)
    defaults.update(kwargs)
    return BlockPool(**defaults)


def test_block_pool_initial_free_count_equals_num_blocks() -> None:
    pool = _pool()
    assert pool.num_free_blocks == _B


def test_block_pool_allocate_returns_valid_index() -> None:
    pool = _pool()
    idx = pool.allocate()
    assert 0 <= idx < _B


def test_block_pool_allocate_decrements_free_count() -> None:
    pool = _pool()
    pool.allocate()
    assert pool.num_free_blocks == _B - 1


def test_block_pool_free_after_allocate_restores_count() -> None:
    pool = _pool()
    idx = pool.allocate()
    pool.free(idx)
    assert pool.num_free_blocks == _B


def test_block_pool_exhaustion_raises_block_pool_exhausted_error() -> None:
    pool = _pool(num_blocks=2)
    pool.allocate()
    pool.allocate()
    with pytest.raises(BlockPoolExhaustedError):
        pool.allocate()


def test_block_pool_write_tokens_are_readable_via_get_key_block() -> None:
    pool = _pool()
    idx = pool.allocate()
    k = torch.randn(_H, 3, _D)
    v = torch.randn_like(k)
    pool.write_tokens(layer_idx=0, block_idx=idx, slot_offset=0, k=k, v=v)

    torch.testing.assert_close(pool.get_key_block(0, idx)[:, :3, :], k)
    torch.testing.assert_close(pool.get_value_block(0, idx)[:, :3, :], v)


def test_block_pool_write_tokens_at_nonzero_slot_offset() -> None:
    pool = _pool()
    idx = pool.allocate()
    k = torch.randn(_H, 2, _D)
    v = torch.randn_like(k)
    pool.write_tokens(layer_idx=0, block_idx=idx, slot_offset=2, k=k, v=v)

    torch.testing.assert_close(pool.get_key_block(0, idx)[:, 2:4, :], k)
    torch.testing.assert_close(pool.get_value_block(0, idx)[:, 2:4, :], v)


def test_block_pool_layers_store_independently() -> None:
    pool = _pool()
    idx = pool.allocate()
    k0 = torch.ones(_H, 1, _D)
    k1 = torch.zeros(_H, 1, _D)
    pool.write_tokens(0, idx, 0, k0, torch.zeros_like(k0))
    pool.write_tokens(1, idx, 0, k1, torch.zeros_like(k1))

    torch.testing.assert_close(pool.get_key_block(0, idx)[:, :1, :], k0)
    torch.testing.assert_close(pool.get_key_block(1, idx)[:, :1, :], k1)


def test_block_pool_freed_block_can_be_reallocated() -> None:
    pool = _pool(num_blocks=1)
    idx = pool.allocate()
    pool.free(idx)
    new_idx = pool.allocate()
    assert new_idx == idx
