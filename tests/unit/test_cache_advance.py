# pyright: reportPrivateImportUsage=false
"""A step's tokens are given blocks all together or not at all."""

from __future__ import annotations

import pytest
import torch

from liteinfer.cache.block_pool import BlockPool, BlockPoolExhaustedError
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache

CPU = torch.device("cpu")
BLOCK_SIZE = 4


def _cache(num_blocks: int) -> ContinuousKVCache:
    pool = BlockPool(
        num_blocks=num_blocks, block_size=BLOCK_SIZE, num_layers=1, num_kv_heads=1, head_dim=2,
        dtype=torch.float32, device=CPU,
    )
    return ContinuousKVCache(pool)


def _holding(cache: ContinuousKVCache, counts: dict[str, int]) -> ContinuousKVCache:
    for request_id, count in counts.items():
        cache.register(request_id)
        cache.advance([request_id], [count])
    return cache


def test_a_step_the_pool_cannot_hold_is_refused() -> None:
    cache = _holding(_cache(num_blocks=3), {"running": 4})
    cache.register("new")

    with pytest.raises(BlockPoolExhaustedError):
        cache.advance(["running", "new"], [1, 8])  # 1 + 2 blocks against 2 free


def test_a_refused_step_leaves_every_sequence_where_it_was() -> None:
    cache = _holding(_cache(num_blocks=3), {"running": 4})
    cache.register("new")

    with pytest.raises(BlockPoolExhaustedError):
        cache.advance(["running", "new"], [1, 8])

    assert (cache.seq_total_len("running"), cache.seq_total_len("new")) == (4, 0)


def test_a_refused_step_allocates_no_blocks() -> None:
    cache = _holding(_cache(num_blocks=3), {"running": 4})
    cache.register("new")

    with pytest.raises(BlockPoolExhaustedError):
        cache.advance(["running", "new"], [1, 8])

    assert cache._pool.num_free_blocks == 2


def test_a_step_that_exactly_fills_the_pool_is_accepted() -> None:
    cache = _holding(_cache(num_blocks=3), {"running": 4})
    cache.register("new")

    cache.advance(["running", "new"], [1, 4])  # one block each, two free

    assert cache._pool.num_free_blocks == 0
