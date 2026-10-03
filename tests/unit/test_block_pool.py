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


def _key_block(pool: BlockPool, layer_idx: int, block_idx: int) -> torch.Tensor:
    """One block's keys, ``[num_kv_heads, block_size, head_dim]``, read by slot arithmetic."""
    keys, _ = pool.slots(layer_idx)
    return keys[block_idx * _S : (block_idx + 1) * _S].transpose(0, 1)


def test_slots_written_by_index_are_readable_as_their_block() -> None:
    pool = _pool()
    idx = pool.allocate()
    keys, _ = pool.slots(layer_idx=0)
    k = torch.randn(3, _H, _D)
    keys[idx * _S : idx * _S + 3] = k

    torch.testing.assert_close(_key_block(pool, 0, idx)[:, :3, :], k.permute(1, 0, 2))


def test_slots_are_addressed_at_block_index_times_block_size() -> None:
    pool = _pool()
    idx = pool.allocate()
    keys, _ = pool.slots(layer_idx=0)
    k = torch.randn(2, _H, _D)
    keys[idx * _S + 2 : idx * _S + 4] = k

    torch.testing.assert_close(_key_block(pool, 0, idx)[:, 2:4, :], k.permute(1, 0, 2))


def test_block_pool_layers_store_independently() -> None:
    pool = _pool()
    idx = pool.allocate()
    for layer, value in ((0, 1.0), (1, 0.0)):
        keys, _ = pool.slots(layer)
        keys[idx * _S] = torch.full((_H, _D), value)

    assert (_key_block(pool, 0, idx)[0, 0, 0], _key_block(pool, 1, idx)[0, 0, 0]) == (1.0, 0.0)


def test_every_block_is_allocatable() -> None:
    """No block is held back: every read stops at its sequence's context, so none absorbs padding."""
    pool = _pool(num_blocks=3)
    assert {pool.allocate() for _ in range(3)} == {0, 1, 2}


def test_block_pool_freed_block_can_be_reallocated() -> None:
    pool = _pool(num_blocks=1)
    idx = pool.allocate()
    pool.free(idx)
    new_idx = pool.allocate()
    assert new_idx == idx
