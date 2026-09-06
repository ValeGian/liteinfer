"""What each decode payload hands the attention kernel.

The two decode payloads write the same token to the same slot and differ only
in what they return: the gathering one a copy of the history, the paged one the
pool plus the addresses. Both halves of that are checked here, on CPU — the
Triton kernel that consumes the addresses is tested in `test_paged_decode.py`.
"""

from __future__ import annotations

import torch

from liteinfer.cache.block_pool import BlockPool
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache
from liteinfer.models.attention import DenseKV, PagedKV

_BLOCK_SIZE = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 8
_LAYER = 0


def _cache() -> ContinuousKVCache:
    return ContinuousKVCache(
        BlockPool(
            num_blocks=8,
            block_size=_BLOCK_SIZE,
            num_layers=2,
            num_kv_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )


def _decode_token(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One K and V column per sequence, shaped as attention produces them."""
    generator = torch.Generator().manual_seed(0)
    shape = (batch, _NUM_KV_HEADS, 1, _HEAD_DIM)
    return (
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
    )


def _two_sequences_mid_decode() -> tuple[ContinuousKVCache, list[str]]:
    """A cache holding two prefilled sequences of different lengths, ready to decode."""
    cache = _cache()
    request_ids = ["short", "long"]
    prompt_lens = [2, 6]
    for request_id, prompt_len in zip(request_ids, prompt_lens, strict=True):
        cache.register(request_id, prompt_len)

    prompt_kv = torch.zeros(len(request_ids), _NUM_KV_HEADS, max(prompt_lens), _HEAD_DIM)
    cache.make_prefill_payload(request_ids, prompt_lens).update(prompt_kv, prompt_kv, _LAYER)
    cache.advance(request_ids)
    return cache, request_ids


def test_paged_payload_returns_the_pool_itself_rather_than_a_copy():
    """The point of the path: no bytes move when the kernel is handed its input."""
    cache, request_ids = _two_sequences_mid_decode()
    payload = cache.make_paged_decode_payload(
        cache.slot_table_for(request_ids), cache.context_lens_for(request_ids)
    )

    kv = payload.update(*_decode_token(len(request_ids)), _LAYER)

    assert kv.key_pool.data_ptr() == cache.layer_storage(_LAYER)[0].data_ptr()


def test_paged_payload_reports_where_each_sequence_history_ends():
    cache, request_ids = _two_sequences_mid_decode()

    context_lens = cache.context_lens_for(request_ids)

    torch.testing.assert_close(context_lens, torch.tensor([3, 7], dtype=torch.int32))


def test_both_decode_payloads_store_the_new_token_in_the_same_slot():
    """The write side is shared; only the read side differs."""
    paged_cache, request_ids = _two_sequences_mid_decode()
    gathering_cache, _ = _two_sequences_mid_decode()
    key_states, value_states = _decode_token(len(request_ids))

    paged_cache.make_paged_decode_payload(
        paged_cache.slot_table_for(request_ids), paged_cache.context_lens_for(request_ids)
    ).update(key_states, value_states, _LAYER)
    gathering_cache.make_decode_payload(gathering_cache.slot_table_for(request_ids)).update(
        key_states, value_states, _LAYER
    )

    torch.testing.assert_close(
        paged_cache.layer_storage(_LAYER)[0], gathering_cache.layer_storage(_LAYER)[0]
    )


def test_paged_payload_returns_a_paged_kv():
    """The returned type is what selects the kernel, so it is part of the contract."""
    cache, request_ids = _two_sequences_mid_decode()
    payload = cache.make_paged_decode_payload(
        cache.slot_table_for(request_ids), cache.context_lens_for(request_ids)
    )

    assert isinstance(payload.update(*_decode_token(len(request_ids)), _LAYER), PagedKV)


def test_gathering_payload_returns_a_dense_kv():
    cache, request_ids = _two_sequences_mid_decode()
    payload = cache.make_decode_payload(cache.slot_table_for(request_ids))

    assert isinstance(payload.update(*_decode_token(len(request_ids)), _LAYER), DenseKV)
