# pyright: reportPrivateImportUsage=false
"""A step's tokens are given blocks all together or not at all."""

from __future__ import annotations

import pytest
import torch

from liteinfer.cache.block_pool import BlockPool, BlockPoolExhaustedError
from liteinfer.cache.kv_cache import KVCache

CPU = torch.device("cpu")
BLOCK_SIZE = 4


def _cache(num_blocks: int) -> KVCache:
    pool = BlockPool(
        num_blocks=num_blocks, block_size=BLOCK_SIZE, num_layers=1, num_kv_heads=1, head_dim=2,
        dtype=torch.float32, device=CPU,
    )
    return KVCache(pool)


def _holding(cache: KVCache, totals: dict[str, int]) -> KVCache:
    for request_id, total in totals.items():
        cache.reserve([request_id], [total])
    return cache


def test_a_step_the_pool_cannot_hold_is_refused() -> None:
    cache = _holding(_cache(num_blocks=3), {"running": 4})

    with pytest.raises(BlockPoolExhaustedError):
        cache.reserve(["running", "new"], [5, 8])  # 1 + 2 blocks against 2 free


def test_a_refused_step_leaves_every_block_table_as_it_was() -> None:
    cache = _holding(_cache(num_blocks=3), {"running": 4})

    with pytest.raises(BlockPoolExhaustedError):
        cache.reserve(["running", "new"], [5, 8])

    # "new" was never given room, so it has no table at all.
    assert {rid: len(table) for rid, table in cache._block_tables.items()} == {"running": 1}


def test_a_refused_step_allocates_no_blocks() -> None:
    cache = _holding(_cache(num_blocks=3), {"running": 4})

    with pytest.raises(BlockPoolExhaustedError):
        cache.reserve(["running", "new"], [5, 8])

    assert cache._pool.num_free_blocks == 2


def test_a_step_that_exactly_fills_the_pool_is_accepted() -> None:
    cache = _holding(_cache(num_blocks=3), {"running": 4})

    cache.reserve(["running", "new"], [5, 4])  # one block each, two free

    assert cache._pool.num_free_blocks == 0
