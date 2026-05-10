"""Unit tests for _PagedCachePayload and PagedKVCache."""

from __future__ import annotations

import math

import torch

from liteinfer.cache.block_pool import BlockPool
from liteinfer.cache.kv_cache import KVCache
from liteinfer.cache.paged_kv_cache import PagedKVCache, _PagedCachePayload
from liteinfer.config import EngineConfig

_S = 4      # block_size
_NB = 32    # num_blocks (enough for all tests)
_L = 2      # num_layers
_H = 2      # num_kv_heads
_D = 8      # head_dim
_DT = torch.float32
_DEV = torch.device("cpu")


def _pool() -> BlockPool:
    return BlockPool(num_blocks=_NB, block_size=_S, num_layers=_L, num_kv_heads=_H, head_dim=_D, dtype=_DT, device=_DEV)


def _config() -> EngineConfig:
    return EngineConfig(model="dummy", cache_mode="paged")


def _cache(pool: BlockPool, prompt_lens: list[int]) -> PagedKVCache:
    return PagedKVCache(_config(), pool, prompt_lens)


# ---------------------------------------------------------------------------
# PagedKVCache — interface
# ---------------------------------------------------------------------------


def test_paged_kv_cache_is_kv_cache_subclass() -> None:
    assert issubclass(PagedKVCache, KVCache)


def test_paged_kv_cache_initial_seq_length_is_zero() -> None:
    cache = _cache(_pool(), [3, 5])
    assert cache.get_seq_length() == 0


def test_paged_kv_cache_payload_is_paged_cache_payload_instance() -> None:
    cache = _cache(_pool(), [3])
    assert isinstance(cache.payload, _PagedCachePayload)


# ---------------------------------------------------------------------------
# _PagedCachePayload — prefill
# ---------------------------------------------------------------------------


def test_paged_payload_prefill_returns_input_tensors_unchanged() -> None:
    """prefill update() must return the original K/V — attention uses them directly."""
    pool = _pool()
    cache = _cache(pool, [3, 5])
    payload = cache.payload
    max_plen = 5
    k = torch.randn(2, _H, max_plen, _D)
    v = torch.randn_like(k)
    k_out, v_out = payload.update(k, v, layer_idx=0)

    torch.testing.assert_close(k_out, k)
    torch.testing.assert_close(v_out, v)


def test_paged_payload_prefill_allocates_ceil_div_blocks_per_seq() -> None:
    """Exactly ceil(prompt_len / block_size) blocks allocated per sequence after prefill."""
    pool = _pool()
    cache = _cache(pool, [3, 5])
    payload = cache.payload
    k = torch.randn(2, _H, 5, _D)
    v = torch.randn_like(k)
    payload.update(k, v, layer_idx=0)

    # seq 0 prompt_len=3: ceil(3/4)=1 block; seq 1 prompt_len=5: ceil(5/4)=2 blocks
    assert len(payload._block_tables[0]) == math.ceil(3 / _S)
    assert len(payload._block_tables[1]) == math.ceil(5 / _S)


def test_paged_payload_prefill_sets_token_counts_to_prompt_lens() -> None:
    pool = _pool()
    cache = _cache(pool, [3, 5])
    payload = cache.payload
    k = torch.randn(2, _H, 5, _D)
    payload.update(k, torch.randn_like(k), layer_idx=0)

    assert payload._token_counts == [3, 5]


def test_paged_payload_prefill_stores_only_real_tokens_not_padding() -> None:
    """Verify real token values are actually stored and retrievable, and the number
    of stored tokens matches prompt_len (not max_prompt_len)."""
    pool = _pool()
    prompt_lens = [3, 5]
    max_plen = 5
    cache = _cache(pool, prompt_lens)
    payload = cache.payload

    k_p = torch.randn(2, _H, max_plen, _D)
    v_p = torch.randn_like(k_p)
    payload.update(k_p, v_p, layer_idx=0)

    # For seq 0 (prompt_len=3): 3 real tokens stored in block_tables[0]
    # Decode one step to trigger gather and verify values
    k_d = torch.zeros(2, _H, 1, _D)
    v_d = torch.zeros_like(k_d)
    k_out, _ = payload.update(k_d, v_d, layer_idx=0)

    real_start_0 = max_plen - prompt_lens[0]
    # Positions [real_start_0:max_plen] in the output must match the real prefill tokens
    torch.testing.assert_close(k_out[0, :, real_start_0:max_plen, :], k_p[0, :, real_start_0:, :])
    torch.testing.assert_close(k_out[1, :, :max_plen, :], k_p[1, :, :, :])


# ---------------------------------------------------------------------------
# _PagedCachePayload — decode
# ---------------------------------------------------------------------------


def test_paged_payload_decode_output_shape_one_step() -> None:
    pool = _pool()
    prompt_lens = [3, 5]
    max_plen = max(prompt_lens)
    cache = _cache(pool, prompt_lens)
    payload = cache.payload

    k_p = torch.randn(2, _H, max_plen, _D)
    payload.update(k_p, torch.randn_like(k_p), layer_idx=0)

    k_d = torch.randn(2, _H, 1, _D)
    k_out, v_out = payload.update(k_d, torch.randn_like(k_d), layer_idx=0)

    assert k_out.shape == (2, _H, max_plen + 1, _D)
    assert v_out.shape == (2, _H, max_plen + 1, _D)


def test_paged_payload_decode_token_appears_at_correct_position() -> None:
    pool = _pool()
    prompt_lens = [5]
    max_plen = 5
    cache = _cache(pool, prompt_lens)
    payload = cache.payload

    k_p = torch.randn(1, _H, max_plen, _D)
    payload.update(k_p, torch.randn_like(k_p), layer_idx=0)

    k_d = torch.randn(1, _H, 1, _D)
    k_out, _ = payload.update(k_d, torch.randn_like(k_d), layer_idx=0)

    torch.testing.assert_close(k_out[0, :, max_plen:, :], k_d[0, :, :, :])


def test_paged_payload_decode_crosses_block_boundary() -> None:
    """Sequences requiring more than block_size tokens must span multiple blocks."""
    pool = _pool()
    prompt_lens = [_S]  # exactly one full block for prefill
    cache = _cache(pool, prompt_lens)
    payload = cache.payload

    k_p = torch.randn(1, _H, _S, _D)
    payload.update(k_p, torch.randn_like(k_p), layer_idx=0)

    for step in range(1, _S + 2):
        k_d = torch.randn(1, _H, 1, _D)
        k_out, _ = payload.update(k_d, torch.randn_like(k_d), layer_idx=0)
        assert k_out.shape[2] == _S + step


def test_paged_payload_seq_length_after_prefill_equals_max_prompt_len() -> None:
    pool = _pool()
    cache = _cache(pool, [3, 5])
    payload = cache.payload
    k_p = torch.randn(2, _H, 5, _D)
    for layer_idx in range(_L):
        payload.update(k_p, torch.randn_like(k_p), layer_idx=layer_idx)
    assert cache.get_seq_length() == 5


def test_paged_payload_seq_length_grows_by_one_per_decode_step() -> None:
    pool = _pool()
    cache = _cache(pool, [5])
    payload = cache.payload
    k_p = torch.randn(1, _H, 5, _D)
    for layer_idx in range(_L):
        payload.update(k_p, torch.randn_like(k_p), layer_idx=layer_idx)

    for step in range(1, 5):
        k_d = torch.randn(1, _H, 1, _D)
        for layer_idx in range(_L):
            payload.update(k_d, torch.randn_like(k_d), layer_idx=layer_idx)
        assert cache.get_seq_length() == 5 + step


# ---------------------------------------------------------------------------
# PagedKVCache — reset
# ---------------------------------------------------------------------------


def test_paged_kv_cache_reset_zeroes_seq_length() -> None:
    pool = _pool()
    cache = _cache(pool, [3])
    payload = cache.payload
    k_p = torch.randn(1, _H, 3, _D)
    payload.update(k_p, torch.randn_like(k_p), layer_idx=0)
    assert cache.get_seq_length() > 0
    cache.reset()
    assert cache.get_seq_length() == 0


def test_paged_kv_cache_reset_returns_all_blocks_to_pool() -> None:
    pool = _pool()
    before_free = pool.num_free_blocks

    cache = _cache(pool, [5, 7])
    payload = cache.payload
    k_p = torch.randn(2, _H, 7, _D)
    for layer_idx in range(_L):
        payload.update(k_p, torch.randn_like(k_p), layer_idx=layer_idx)

    assert pool.num_free_blocks < before_free
    cache.reset()
    assert pool.num_free_blocks == before_free
