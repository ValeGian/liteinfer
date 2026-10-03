"""What each payload writes to the pool and hands the attention kernel.

The two decode payloads write the same token to the same slot and differ only
in what they return: the gathering one a copy of the history, the paged one the
pool plus the addresses. Both halves of that are checked here, on CPU — the
Triton kernel that consumes the addresses is tested in `test_paged_decode.py`.

The prefill payloads are checked for the same two things, including for a
prompt chunked across passes, whose later chunks must read back what the
earlier ones wrote.
"""

from __future__ import annotations

import torch

from liteinfer.cache.block_pool import BlockPool
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache
from liteinfer.models.attention import DenseKV, PagedKV, VarlenKV

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
    for request_id in request_ids:
        cache.register(request_id)
    cache.advance(request_ids, prompt_lens)

    prompt_kv = torch.zeros(len(request_ids), _NUM_KV_HEADS, max(prompt_lens), _HEAD_DIM)
    cache.make_prefill_payload(request_ids, prompt_lens).update(prompt_kv, prompt_kv, _LAYER)
    cache.advance(request_ids, [1, 1])
    return cache, request_ids


def _decode_addresses(cache: ContinuousKVCache, request_ids: list[str]):
    """The write mapping and read table `decode` builds, in the order payloads take them."""
    one_each = [1] * len(request_ids)
    return cache.slot_mapping_for(request_ids, one_each), cache.slot_table_for(request_ids)


def _paged_decode_payload(cache: ContinuousKVCache, request_ids: list[str], num_splits=None):
    return cache.make_paged_decode_payload(
        *_decode_addresses(cache, request_ids), cache.context_lens_for(request_ids), num_splits
    )


def test_paged_payload_returns_the_pool_itself_rather_than_a_copy():
    """The point of the path: no bytes move when the kernel is handed its input."""
    cache, request_ids = _two_sequences_mid_decode()
    payload = _paged_decode_payload(cache, request_ids)

    kv = payload.update(*_decode_token(len(request_ids)), _LAYER)

    assert kv.key_pool.data_ptr() == cache.layer_storage(_LAYER)[0].data_ptr()


def test_paged_payload_reports_where_each_sequence_history_ends():
    cache, request_ids = _two_sequences_mid_decode()

    context_lens = cache.context_lens_for(request_ids)

    torch.testing.assert_close(context_lens, torch.tensor([3, 7], dtype=torch.int32))


def test_a_packed_decode_write_fills_the_pool_the_padded_table_s_last_column_did():
    """One slot per sequence addresses what the right-aligned table's last column does.

    The last column of `slot_table` is each sequence's newest token only because
    the table is right-aligned. The flat mapping says the same thing without the
    alignment, and must say it byte for byte.
    """
    packed_cache, request_ids = _two_sequences_mid_decode()
    padded_cache, _ = _two_sequences_mid_decode()
    key_states, value_states = _decode_token(len(request_ids))

    _paged_decode_payload(packed_cache, request_ids).update(key_states, value_states, _LAYER)
    newest_column = padded_cache.slot_table_for(request_ids)[:, -1:]
    padded_cache.scatter(_LAYER, newest_column, key_states, value_states)

    assert torch.equal(
        packed_cache.layer_storage(_LAYER)[0], padded_cache.layer_storage(_LAYER)[0]
    )


def test_both_decode_payloads_store_the_new_token_in_the_same_slot():
    """The write side is shared; only the read side differs."""
    paged_cache, request_ids = _two_sequences_mid_decode()
    gathering_cache, _ = _two_sequences_mid_decode()
    key_states, value_states = _decode_token(len(request_ids))

    _paged_decode_payload(paged_cache, request_ids).update(key_states, value_states, _LAYER)
    gathering_cache.make_decode_payload(*_decode_addresses(gathering_cache, request_ids)).update(
        key_states, value_states, _LAYER
    )

    torch.testing.assert_close(
        paged_cache.layer_storage(_LAYER)[0], gathering_cache.layer_storage(_LAYER)[0]
    )


def test_paged_payload_returns_a_paged_kv():
    """The returned type is what selects the kernel, so it is part of the contract."""
    cache, request_ids = _two_sequences_mid_decode()
    payload = _paged_decode_payload(cache, request_ids)

    assert isinstance(payload.update(*_decode_token(len(request_ids)), _LAYER), PagedKV)


def test_gathering_payload_returns_a_dense_kv():
    cache, request_ids = _two_sequences_mid_decode()
    payload = cache.make_decode_payload(*_decode_addresses(cache, request_ids))

    assert isinstance(payload.update(*_decode_token(len(request_ids)), _LAYER), DenseKV)


def test_the_profile_payload_hands_back_what_it_was_given():
    """It measures a forward's memory, so it must not need a pool to write into.

    Prefill attention reads the K/V the pass just computed, which is what the real
    prefill payload returns after storing it — so returning them untouched gives a
    forward with the same shapes and the same activation peak.
    """
    from liteinfer.cache.continuous_kv_cache import ProfilePayload

    key_states, value_states = _decode_token(2)

    kv = ProfilePayload().update(key_states, value_states, _LAYER)

    assert kv.keys is key_states


def test_paged_payload_hands_the_kernel_a_pinned_split_count():
    """How many programs share the key loop travels with the addresses.

    The kernel chooses for itself when the engine has no opinion; pinning the
    count is what lets a benchmark row keep measuring one shape of the grid, so
    the value has to survive the trip from the config to the launch.
    """
    cache, request_ids = _two_sequences_mid_decode()
    payload = _paged_decode_payload(cache, request_ids, num_splits=4)

    kv = payload.update(*_decode_token(len(request_ids)), _LAYER)

    assert kv.num_splits == 4


def test_paged_payload_leaves_the_split_count_to_the_kernel_by_default():
    """`None` is "choose from the batch width and the device", which is the engine default."""
    cache, request_ids = _two_sequences_mid_decode()
    payload = _paged_decode_payload(cache, request_ids)

    kv = payload.update(*_decode_token(len(request_ids)), _LAYER)

    assert kv.num_splits is None


def _prompt_kv(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """K/V for prompts laid end to end, as a packed prefill pass computes them."""
    generator = torch.Generator().manual_seed(1)
    shape = (1, _NUM_KV_HEADS, sum(lengths), _HEAD_DIM)
    return torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)


def _registered(cache: ContinuousKVCache, request_ids: list[str], counts: list[int]) -> None:
    """Register sequences and account for a first pass of `counts` tokens, as `prefill` does."""
    for request_id in request_ids:
        cache.register(request_id)
    cache.advance(request_ids, counts)


def _left_padded(keys: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """Packed `[1, H, sum, D]` K/V rearranged as the padded pass computes it: zeros in front."""
    padded = torch.zeros(len(lengths), _NUM_KV_HEADS, max(lengths), _HEAD_DIM)
    start = 0
    for row, length in enumerate(lengths):
        padded[row, :, max(lengths) - length :] = keys[0, :, start : start + length]
        start += length
    return padded


def _packed_whole_prompts(
    cache: ContinuousKVCache, request_ids: list[str], lengths: list[int], keys, values
) -> VarlenKV:
    """What a packed pass over whole prompts hands attention: this pass's K/V and boundaries."""
    kv = cache.make_packed_prefill_payload(request_ids, lengths).update(keys, values, _LAYER)
    assert isinstance(kv, VarlenKV), "nothing was cached before the pass"
    return kv


def test_packed_prefill_payload_reports_where_each_prompt_starts():
    """`cu_seqlens` marks where each prompt starts; a padded pass needs padding and a mask for that."""
    cache = _cache()
    request_ids, lengths = ["a", "b"], [2, 5]
    _registered(cache, request_ids, lengths)

    kv = _packed_whole_prompts(cache, request_ids, lengths, *_prompt_kv(lengths))

    assert kv.cu_seqlens.tolist() == [0, 2, 7]


def test_a_whole_prompt_attends_to_the_keys_its_own_pass_computed():
    """Nothing was cached before the pass, so reading the pool back would be a wasted copy."""
    cache = _cache()
    _registered(cache, ["a"], [3])
    keys, values = _prompt_kv([3])

    kv = _packed_whole_prompts(cache, ["a"], [3], keys, values)

    assert kv.keys is keys


def test_packed_prefill_payload_writes_the_same_pool_as_the_padded_one():
    """Same prompts, same slots: the layout of the pass changes, the cache does not."""
    lengths = [2, 5]
    request_ids = ["a", "b"]
    packed_cache, padded_cache = _cache(), _cache()
    for cache in (packed_cache, padded_cache):
        _registered(cache, request_ids, lengths)

    keys, values = _prompt_kv(lengths)
    packed_cache.make_packed_prefill_payload(request_ids, lengths).update(keys, values, _LAYER)
    padded_cache.make_prefill_payload(request_ids, lengths).update(
        _left_padded(keys, lengths), _left_padded(values, lengths), _LAYER
    )

    torch.testing.assert_close(
        packed_cache.layer_storage(_LAYER)[0], padded_cache.layer_storage(_LAYER)[0]
    )


def test_packed_prefill_payload_returns_a_varlen_kv():
    """The returned type is what selects the kernel, so it is part of the contract."""
    cache = _cache()
    _registered(cache, ["a"], [3])
    payload = cache.make_packed_prefill_payload(["a"], [3])

    assert isinstance(payload.update(*_prompt_kv([3]), _LAYER), VarlenKV)


# --- a prompt chunked across passes ------------------------------------------

_CHUNKED_LENGTHS = [6, 3]
_FIRST_CHUNK = [4, 2]
_SECOND_CHUNK = [2, 1]


def _chunked_packed(cache: ContinuousKVCache) -> PagedKV:
    """Prefill `_CHUNKED_LENGTHS` in two packed passes; return what the second one hands attention."""
    request_ids = ["a", "b"]
    keys, values = _prompt_kv(_CHUNKED_LENGTHS)
    _registered(cache, request_ids, _FIRST_CHUNK)
    first = [slice(0, count) for count in _FIRST_CHUNK]
    cache.make_packed_prefill_payload(request_ids, _FIRST_CHUNK).update(
        _ragged_chunk(keys, first), _ragged_chunk(values, first), _LAYER
    )
    cache.advance(request_ids, _SECOND_CHUNK)
    second = [slice(count, None) for count in _FIRST_CHUNK]
    kv = cache.make_packed_prefill_payload(request_ids, _SECOND_CHUNK).update(
        _ragged_chunk(keys, second), _ragged_chunk(values, second), _LAYER
    )
    assert isinstance(kv, PagedKV), "a continuing chunk reads the pool"
    return kv


def _ragged_chunk(keys: torch.Tensor, chunks: list[slice]) -> torch.Tensor:
    """A different slice of each packed sequence of `_CHUNKED_LENGTHS`, packed again."""
    pieces, start = [], 0
    for length, chunk in zip(_CHUNKED_LENGTHS, chunks, strict=True):
        pieces.append(keys[:, :, start : start + length][:, :, chunk])
        start += length
    return torch.cat(pieces, dim=2)


def test_a_continuing_chunk_hands_the_kernel_the_pool_rather_than_a_copy():
    """The prefix stays where an earlier pass wrote it."""
    cache = _cache()

    kv = _chunked_packed(cache)

    assert kv.key_pool.data_ptr() == cache.layer_storage(_LAYER)[0].data_ptr()


def test_a_continuing_chunk_brings_only_its_own_queries():
    kv = _chunked_packed(_cache())

    assert kv.query_start_loc is not None and kv.query_start_loc.tolist() == [0, 2, 3]


def test_a_continuing_chunk_attends_to_everything_its_sequence_holds():
    """The prefix an earlier pass wrote is part of the context, bounded per sequence."""
    kv = _chunked_packed(_cache())

    assert kv.context_lens.tolist() == [6, 3]


def test_a_continuing_chunk_addresses_the_prefix_an_earlier_pass_wrote():
    """Read through its addresses, the pool holds the whole prompt, as if computed in one pass."""
    kv = _chunked_packed(_cache())

    rows = [row[-length:] for row, length in zip(kv.slot_table, _CHUNKED_LENGTHS, strict=True)]
    read_back = kv.key_pool[torch.cat(rows)].permute(1, 0, 2).unsqueeze(0)
    torch.testing.assert_close(read_back, _prompt_kv(_CHUNKED_LENGTHS)[0])


def test_a_padded_continuing_chunk_reads_back_the_whole_prompt_left_padded():
    """The dense kernels get the same context, in the layout their mask expects."""
    cache = _cache()
    request_ids = ["a", "b"]
    keys, values = _prompt_kv(_CHUNKED_LENGTHS)
    _registered(cache, request_ids, _FIRST_CHUNK)
    first = [slice(0, count) for count in _FIRST_CHUNK]
    cache.make_prefill_payload(request_ids, _FIRST_CHUNK).update(
        _left_padded(_ragged_chunk(keys, first), _FIRST_CHUNK),
        _left_padded(_ragged_chunk(values, first), _FIRST_CHUNK),
        _LAYER,
    )
    cache.advance(request_ids, _SECOND_CHUNK)
    second = [slice(count, None) for count in _FIRST_CHUNK]

    kv = cache.make_prefill_payload(request_ids, _SECOND_CHUNK).update(
        _left_padded(_ragged_chunk(keys, second), _SECOND_CHUNK),
        _left_padded(_ragged_chunk(values, second), _SECOND_CHUNK),
        _LAYER,
    )

    real = torch.cat([kv.keys[0], kv.keys[1, :, -3:]], dim=1).unsqueeze(0)
    torch.testing.assert_close(real, keys)
