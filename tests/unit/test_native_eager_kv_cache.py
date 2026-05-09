"""Unit tests for NativeEagerKVCache and its inner _NativeCachePayload."""

from __future__ import annotations

import pytest
import torch

from liteinfer.cache.native_eager_kv_cache import NativeEagerKVCache, _NativeCachePayload
from liteinfer.config import EngineConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kv(B: int = 1, H: int = 2, S: int = 4, D: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(B, H, S, D), torch.randn(B, H, S, D)


def _make_config() -> EngineConfig:
    return EngineConfig(model="dummy")


# ---------------------------------------------------------------------------
# _NativeCachePayload — first update (no prior state)
# ---------------------------------------------------------------------------


def test_native_cache_payload_first_update_returns_input_tensors() -> None:
    cache = _NativeCachePayload()
    k, v = _make_kv(S=4)
    rk, rv = cache.update(k, v, layer_idx=0)
    torch.testing.assert_close(rk, k)
    torch.testing.assert_close(rv, v)


def test_native_cache_payload_first_update_reports_seq_length() -> None:
    cache = _NativeCachePayload()
    k, v = _make_kv(S=7)
    cache.update(k, v, layer_idx=0)
    assert cache.get_seq_length(layer_idx=0) == 7


# ---------------------------------------------------------------------------
# _NativeCachePayload — subsequent updates concatenate on dim=2
# ---------------------------------------------------------------------------


def test_native_cache_payload_subsequent_update_concatenates_on_seq_dim() -> None:
    cache = _NativeCachePayload()
    k1, v1 = _make_kv(S=4)
    k2, v2 = _make_kv(S=1)

    cache.update(k1, v1, layer_idx=0)
    rk, rv = cache.update(k2, v2, layer_idx=0)

    assert rk.shape == (1, 2, 5, 8)
    torch.testing.assert_close(rk[:, :, :4, :], k1)
    torch.testing.assert_close(rk[:, :, 4:, :], k2)
    torch.testing.assert_close(rv[:, :, :4, :], v1)
    torch.testing.assert_close(rv[:, :, 4:, :], v2)


def test_native_cache_payload_subsequent_update_reports_accumulated_seq_length() -> None:
    cache = _NativeCachePayload()
    k1, v1 = _make_kv(S=4)
    k2, v2 = _make_kv(S=1)
    cache.update(k1, v1, layer_idx=0)
    cache.update(k2, v2, layer_idx=0)
    assert cache.get_seq_length(layer_idx=0) == 5


# ---------------------------------------------------------------------------
# _NativeCachePayload — multiple layers are tracked independently
# ---------------------------------------------------------------------------


def test_native_cache_payload_layers_tracked_independently() -> None:
    cache = _NativeCachePayload()
    k0, v0 = _make_kv(S=3)
    k1, v1 = _make_kv(S=5)
    cache.update(k0, v0, layer_idx=0)
    cache.update(k1, v1, layer_idx=1)
    assert cache.get_seq_length(layer_idx=0) == 3
    assert cache.get_seq_length(layer_idx=1) == 5


def test_native_cache_payload_non_contiguous_layer_idx_handled() -> None:
    """Layers may be updated out of order; gaps are filled with empty slots."""
    cache = _NativeCachePayload()
    k, v = _make_kv(S=2)
    cache.update(k, v, layer_idx=3)
    assert cache.get_seq_length(layer_idx=3) == 2
    assert cache.get_seq_length(layer_idx=0) == 0


# ---------------------------------------------------------------------------
# _NativeCachePayload — get_seq_length on empty cache
# ---------------------------------------------------------------------------


def test_native_cache_payload_get_seq_length_empty_returns_zero() -> None:
    cache = _NativeCachePayload()
    assert cache.get_seq_length() == 0


def test_native_cache_payload_get_seq_length_out_of_bounds_returns_zero() -> None:
    cache = _NativeCachePayload()
    assert cache.get_seq_length(layer_idx=99) == 0


# ---------------------------------------------------------------------------
# _NativeCachePayload — clear resets state
# ---------------------------------------------------------------------------


def test_native_cache_payload_clear_resets_seq_length_to_zero() -> None:
    cache = _NativeCachePayload()
    k, v = _make_kv(S=4)
    cache.update(k, v, layer_idx=0)
    cache.clear()
    assert cache.get_seq_length(layer_idx=0) == 0


def test_native_cache_payload_clear_allows_fresh_update() -> None:
    cache = _NativeCachePayload()
    k1, v1 = _make_kv(S=4)
    cache.update(k1, v1, layer_idx=0)
    cache.clear()
    k2, v2 = _make_kv(S=2)
    rk, _ = cache.update(k2, v2, layer_idx=0)
    assert rk.shape[2] == 2


# ---------------------------------------------------------------------------
# NativeEagerKVCache — KVCache interface
# ---------------------------------------------------------------------------


def test_native_eager_kv_cache_payload_returns_inner_instance() -> None:
    cache = NativeEagerKVCache(_make_config())
    assert isinstance(cache.payload, _NativeCachePayload)


def test_native_eager_kv_cache_get_seq_length_delegates_to_inner() -> None:
    cache = NativeEagerKVCache(_make_config())
    k, v = _make_kv(S=6)
    cache.payload.update(k, v, layer_idx=0)
    assert cache.get_seq_length() == 6


def test_native_eager_kv_cache_reset_clears_inner_state() -> None:
    cache = NativeEagerKVCache(_make_config())
    k, v = _make_kv(S=4)
    cache.payload.update(k, v, layer_idx=0)
    cache.reset()
    assert cache.get_seq_length() == 0


def test_native_eager_kv_cache_payload_same_object_across_calls() -> None:
    """payload must return the same object so the model mutates the cache in-place."""
    cache = NativeEagerKVCache(_make_config())
    assert cache.payload is cache.payload


# ---------------------------------------------------------------------------
# Batch-size > 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 4])
def test_native_cache_payload_batched_update_preserves_batch_dim(batch_size: int) -> None:
    cache = _NativeCachePayload()
    k, v = _make_kv(B=batch_size, S=5)
    rk, rv = cache.update(k, v, layer_idx=0)
    assert rk.shape == (batch_size, 2, 5, 8)
    assert rv.shape == (batch_size, 2, 5, 8)
