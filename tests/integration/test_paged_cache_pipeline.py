"""Integration tests for the paged KV cache pipeline.

Tests verify the observable contract of `PagedKVCache` and `BlockPool`:
shapes, values, lifecycle (reset), and memory accounting. They do NOT
reach into internal block-table layout — the same tests must remain
valid across any implementation that satisfies the component contracts.

All tests run on CPU (no GPU required).
"""

from __future__ import annotations

import pytest
import torch

from liteinfer.cache.block_pool import BlockPool, BlockPoolExhaustedError
from liteinfer.cache.kv_cache import KVCache
from liteinfer.cache.paged_kv_cache import PagedKVCache
from liteinfer.config import EngineConfig

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 4
_NUM_BLOCKS = 16
_NUM_LAYERS = 2
_NUM_KV_HEADS = 2
_HEAD_DIM = 8
_DTYPE = torch.float32
_DEVICE = torch.device("cpu")


def _make_pool() -> BlockPool:
    return BlockPool(
        num_blocks=_NUM_BLOCKS,
        block_size=_BLOCK_SIZE,
        num_layers=_NUM_LAYERS,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        dtype=_DTYPE,
        device=_DEVICE,
    )


def _make_config() -> EngineConfig:
    return EngineConfig(model="dummy", cache_mode="paged")


def _make_cache(pool: BlockPool, prompt_lens: list[int]) -> PagedKVCache:
    return PagedKVCache(_make_config(), pool, prompt_lens)


# ---------------------------------------------------------------------------
# BlockPool — allocation and lifecycle
# ---------------------------------------------------------------------------


def test_block_pool_allocate_decrements_free_count() -> None:
    pool = _make_pool()
    initial = pool.num_free_blocks
    pool.allocate()
    assert pool.num_free_blocks == initial - 1


def test_block_pool_free_returns_block_to_pool() -> None:
    pool = _make_pool()
    initial = pool.num_free_blocks
    block_idx = pool.allocate()
    pool.free(block_idx)
    assert pool.num_free_blocks == initial


def test_block_pool_exhaustion_raises() -> None:
    pool = BlockPool(
        num_blocks=2,
        block_size=_BLOCK_SIZE,
        num_layers=_NUM_LAYERS,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    pool.allocate()
    pool.allocate()
    with pytest.raises(BlockPoolExhaustedError):
        pool.allocate()

def test_paged_kv_cache_is_kv_cache_subclass() -> None:
    assert issubclass(PagedKVCache, KVCache)


def test_paged_kv_cache_get_seq_length_initially_zero() -> None:
    pool = _make_pool()
    cache = _make_cache(pool, [3, 5])
    assert cache.get_seq_length() == 0


def test_paged_kv_cache_reset_zeroes_seq_length() -> None:
    pool = _make_pool()
    cache = _make_cache(pool, [3, 5])
    payload = cache.payload
    max_prompt_len = 5

    k = torch.randn(2, _NUM_KV_HEADS, max_prompt_len, _HEAD_DIM)
    v = torch.randn_like(k)
    for layer_idx in range(_NUM_LAYERS):
        payload.update(k, v, layer_idx=layer_idx)

    assert cache.get_seq_length() > 0
    cache.reset()
    assert cache.get_seq_length() == 0


def test_paged_kv_cache_reset_frees_blocks_to_pool() -> None:
    """reset() must return every allocated block back to the pool."""
    pool = _make_pool()
    initial_free = pool.num_free_blocks

    cache = _make_cache(pool, [5])
    payload = cache.payload
    k = torch.randn(1, _NUM_KV_HEADS, 5, _HEAD_DIM)
    v = torch.randn_like(k)
    for layer_idx in range(_NUM_LAYERS):
        payload.update(k, v, layer_idx=layer_idx)

    assert pool.num_free_blocks < initial_free
    cache.reset()
    assert pool.num_free_blocks == initial_free


# ---------------------------------------------------------------------------
# _PagedCachePayload — update() shapes and values
# ---------------------------------------------------------------------------


def test_paged_payload_prefill_returns_input_unchanged() -> None:
    """During prefill, update() must return tensors with the same shape and values
    as the input so the attention layer can compute prefill attention correctly."""
    pool = _make_pool()
    cache = _make_cache(pool, [3, 5])
    payload = cache.payload
    max_prompt_len = 5

    k = torch.randn(2, _NUM_KV_HEADS, max_prompt_len, _HEAD_DIM)
    v = torch.randn_like(k)
    k_out, v_out = payload.update(k, v, layer_idx=0)

    assert k_out.shape == k.shape
    assert v_out.shape == v.shape
    torch.testing.assert_close(k_out, k)
    torch.testing.assert_close(v_out, v)


def test_paged_payload_decode_returns_gathered_shape() -> None:
    """After prefill, one decode update() must return [B, H, max_prompt_len+1, D]."""
    prompt_lens = [3, 5]
    max_prompt_len = max(prompt_lens)
    pool = _make_pool()
    cache = _make_cache(pool, prompt_lens)
    payload = cache.payload

    k_p = torch.randn(2, _NUM_KV_HEADS, max_prompt_len, _HEAD_DIM)
    v_p = torch.randn_like(k_p)
    payload.update(k_p, v_p, layer_idx=0)

    k_d = torch.randn(2, _NUM_KV_HEADS, 1, _HEAD_DIM)
    v_d = torch.randn_like(k_d)
    k_out, v_out = payload.update(k_d, v_d, layer_idx=0)

    expected_len = max_prompt_len + 1
    assert k_out.shape == (2, _NUM_KV_HEADS, expected_len, _HEAD_DIM)
    assert v_out.shape == (2, _NUM_KV_HEADS, expected_len, _HEAD_DIM)


def test_paged_payload_seq_length_grows_per_step() -> None:
    """get_seq_length() must return max(prompt_lens) after prefill, then grow by 1
    each decode step."""
    prompt_lens = [3, 5]
    max_prompt_len = max(prompt_lens)
    pool = _make_pool()
    cache = _make_cache(pool, prompt_lens)
    payload = cache.payload

    assert cache.get_seq_length() == 0

    k_p = torch.randn(2, _NUM_KV_HEADS, max_prompt_len, _HEAD_DIM)
    v_p = torch.randn_like(k_p)
    for layer_idx in range(_NUM_LAYERS):
        payload.update(k_p, v_p, layer_idx=layer_idx)
    assert cache.get_seq_length() == max_prompt_len

    k_d = torch.randn(2, _NUM_KV_HEADS, 1, _HEAD_DIM)
    v_d = torch.randn_like(k_d)
    for layer_idx in range(_NUM_LAYERS):
        payload.update(k_d, v_d, layer_idx=layer_idx)
    assert cache.get_seq_length() == max_prompt_len + 1


def test_paged_payload_decode_preserves_prefill_real_token_values() -> None:
    """Gathered K/V must contain the correct prefill token values at real positions.

    Seq 0 (prompt_len=3, max_prompt_len=5): real tokens occupy positions [2:5] in
    the gathered output (left-padded by 2).  Seq 1 (prompt_len=5): all positions
    [0:5] are real.  The new decode token appears at position 5 for both sequences.
    """
    prompt_lens = [3, 5]
    max_prompt_len = max(prompt_lens)
    pool = _make_pool()
    cache = _make_cache(pool, prompt_lens)
    payload = cache.payload

    k_p = torch.randn(2, _NUM_KV_HEADS, max_prompt_len, _HEAD_DIM)
    v_p = torch.randn_like(k_p)
    payload.update(k_p, v_p, layer_idx=0)

    k_d = torch.randn(2, _NUM_KV_HEADS, 1, _HEAD_DIM)
    v_d = torch.randn_like(k_d)
    k_out, _v_out = payload.update(k_d, v_d, layer_idx=0)

    # Seq 0: real prefill tokens at gathered positions [2:5]
    real_start_0 = max_prompt_len - prompt_lens[0]  # = 2
    torch.testing.assert_close(
        k_out[0, :, real_start_0:max_prompt_len, :],
        k_p[0, :, real_start_0:, :],
    )

    # Seq 1: all 5 positions are real prefill tokens
    torch.testing.assert_close(k_out[1, :, :max_prompt_len, :], k_p[1, :, :, :])

    # Both seqs: decode token at position max_prompt_len
    torch.testing.assert_close(k_out[0, :, max_prompt_len:, :], k_d[0, :, :, :])
    torch.testing.assert_close(k_out[1, :, max_prompt_len:, :], k_d[1, :, :, :])


def test_paged_payload_multi_decode_steps_shape() -> None:
    """Shape after N decode steps must be [B, H, max_prompt_len+N, D]."""
    prompt_lens = [4]
    max_prompt_len = max(prompt_lens)
    pool = _make_pool()
    cache = _make_cache(pool, prompt_lens)
    payload = cache.payload

    k_p = torch.randn(1, _NUM_KV_HEADS, max_prompt_len, _HEAD_DIM)
    v_p = torch.randn_like(k_p)
    payload.update(k_p, v_p, layer_idx=0)

    for step in range(1, 5):
        k_d = torch.randn(1, _NUM_KV_HEADS, 1, _HEAD_DIM)
        v_d = torch.randn_like(k_d)
        k_out, _v_out = payload.update(k_d, v_d, layer_idx=0)
        assert k_out.shape == (1, _NUM_KV_HEADS, max_prompt_len + step, _HEAD_DIM)


def test_paged_payload_blocks_span_across_block_boundary() -> None:
    """A sequence whose token count exceeds block_size must work correctly —
    tokens must spill into new blocks transparently."""
    prompt_lens = [_BLOCK_SIZE + 2]  # 6 tokens — spans 2 blocks (4 + 2)
    max_prompt_len = prompt_lens[0]
    pool = _make_pool()
    cache = _make_cache(pool, prompt_lens)
    payload = cache.payload

    k_p = torch.randn(1, _NUM_KV_HEADS, max_prompt_len, _HEAD_DIM)
    v_p = torch.randn_like(k_p)
    payload.update(k_p, v_p, layer_idx=0)

    # Decode enough steps to cross another block boundary
    for step in range(1, _BLOCK_SIZE + 1):
        k_d = torch.randn(1, _NUM_KV_HEADS, 1, _HEAD_DIM)
        v_d = torch.randn_like(k_d)
        k_out, _ = payload.update(k_d, v_d, layer_idx=0)
        assert k_out.shape[2] == max_prompt_len + step
