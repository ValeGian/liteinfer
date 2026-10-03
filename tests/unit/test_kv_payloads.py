# pyright: reportPrivateImportUsage=false
"""What each payload writes to the pool and hands the attention kernel.

Every payload hands back a `PagedKV`: the pool itself plus the addresses of
every token each sequence holds. These tests check both halves on CPU — what
lands in the pool, and what the addresses say — while the Triton kernel that
consumes them is tested in `test_paged_decode.py` and `test_paged_prefill.py`.
A prompt chunked across passes must read back what the earlier ones wrote.
"""

from __future__ import annotations

import torch

from liteinfer.cache.block_pool import BlockPool
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache, ProfilePayload
from liteinfer.models.attention import PagedKV

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


def _prompt_kv(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """K/V for prompts laid end to end, as a packed pass computes them."""
    generator = torch.Generator().manual_seed(1)
    shape = (1, _NUM_KV_HEADS, sum(lengths), _HEAD_DIM)
    return torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)


def _registered(cache: ContinuousKVCache, request_ids: list[str], counts: list[int]) -> None:
    """Register sequences and account for a first pass of `counts` tokens, as `execute` does."""
    for request_id in request_ids:
        cache.register(request_id)
    cache.advance(request_ids, counts)


def _read_back_keys(kv: PagedKV, lengths: list[int]) -> torch.Tensor:
    """Every key a `PagedKV` addresses, packed `[1, H, sum, D]` as a pass computes them.

    The slot table is left-aligned, so each sequence's context is the first
    `length` columns of its row.
    """
    rows = [row[:length] for row, length in zip(kv.slot_table, lengths, strict=True)]
    return kv.key_pool[torch.cat(rows)].permute(1, 0, 2).unsqueeze(0)


# --- decode ------------------------------------------------------------------


def _two_sequences_mid_decode() -> tuple[ContinuousKVCache, list[str]]:
    """A cache holding two prefilled sequences of different lengths, ready to decode."""
    cache = _cache()
    request_ids, prompt_lens = ["short", "long"], [2, 6]
    _registered(cache, request_ids, prompt_lens)
    prompt_kv = torch.zeros(1, _NUM_KV_HEADS, sum(prompt_lens), _HEAD_DIM)
    cache.make_packed_payload(request_ids, prompt_lens).update(prompt_kv, prompt_kv, _LAYER)
    cache.advance(request_ids, [1, 1])
    return cache, request_ids


def _decode_payload(cache: ContinuousKVCache, request_ids: list[str], num_splits=None):
    """The payload `execute` builds for a decode step, from the same addresses."""
    return cache.make_decode_payload(
        cache.newest_slots_for(request_ids),
        cache.slot_table_for(request_ids),
        cache.context_lens_for(request_ids),
        num_splits,
    )


def test_decode_payload_returns_the_pool_itself_rather_than_a_copy():
    """No bytes move when the kernel is handed its input."""
    cache, request_ids = _two_sequences_mid_decode()
    payload = _decode_payload(cache, request_ids)

    kv = payload.update(*_decode_token(len(request_ids)), _LAYER)

    assert kv.key_pool.data_ptr() == cache.layer_storage(_LAYER)[0].data_ptr()


def test_decode_payload_reports_where_each_sequence_history_ends():
    cache, request_ids = _two_sequences_mid_decode()

    kv = _decode_payload(cache, request_ids).update(*_decode_token(len(request_ids)), _LAYER)

    torch.testing.assert_close(kv.context_lens, torch.tensor([3, 7], dtype=torch.int32))


def test_decode_payload_stores_each_token_at_its_sequence_s_newest_position():
    """Read through the table at the last position, the pool holds the token just written."""
    cache, request_ids = _two_sequences_mid_decode()
    key_states, value_states = _decode_token(len(request_ids))

    kv = _decode_payload(cache, request_ids).update(key_states, value_states, _LAYER)

    newest = kv.slot_table[torch.arange(2), kv.context_lens.long() - 1]
    torch.testing.assert_close(kv.key_pool[newest], key_states.squeeze(2))


def test_decode_payload_hands_the_kernel_a_pinned_split_count():
    """How many programs share the key loop travels with the addresses.

    The kernel chooses for itself when the engine has no opinion; pinning the
    count is what lets a benchmark row keep measuring one shape of the grid, so
    the value has to survive the trip from the config to the launch.
    """
    cache, request_ids = _two_sequences_mid_decode()
    payload = _decode_payload(cache, request_ids, num_splits=4)

    kv = payload.update(*_decode_token(len(request_ids)), _LAYER)

    assert kv.num_splits == 4


def test_decode_payload_leaves_the_split_count_to_the_kernel_by_default():
    """`None` is "choose from the batch width and the device", which is the engine default."""
    cache, request_ids = _two_sequences_mid_decode()

    kv = _decode_payload(cache, request_ids).update(*_decode_token(len(request_ids)), _LAYER)

    assert kv.num_splits is None


# --- the profile pass --------------------------------------------------------


def test_the_profile_payload_reads_back_exactly_what_its_pass_computed():
    """It measures a forward's memory, so it must not need a pool to write into.

    Every sequence is a whole prompt, so its context is the K/V the pass just
    computed; addressed through the payload's table, those are what attention reads.
    """
    lengths = [3, 5]
    keys, values = _prompt_kv(lengths)

    kv = ProfilePayload(lengths, torch.device("cpu")).update(keys, values, _LAYER)

    torch.testing.assert_close(_read_back_keys(kv, lengths), keys)


def test_the_profile_payload_bounds_each_prompt_as_a_packed_pass_does():
    kv = ProfilePayload([3, 5], torch.device("cpu")).update(*_prompt_kv([3, 5]), _LAYER)

    assert kv.query_start_loc is not None and kv.query_start_loc.tolist() == [0, 3, 8]


def test_the_profile_payload_positions_count_from_each_prompt_s_start():
    payload = ProfilePayload([3, 2], torch.device("cpu"))

    assert payload.positions.tolist() == [[0, 1, 2, 0, 1]]


# --- whole prompts, packed ---------------------------------------------------


def _packed_whole_prompts(
    cache: ContinuousKVCache, request_ids: list[str], lengths: list[int], keys, values
) -> PagedKV:
    """What a packed pass over whole prompts hands attention: the pool, addresses and boundaries."""
    return cache.make_packed_payload(request_ids, lengths).update(keys, values, _LAYER)


def test_packed_payload_reports_where_each_prompt_starts():
    """`query_start_loc` marks where each prompt starts; nothing else separates them."""
    cache = _cache()
    request_ids, lengths = ["a", "b"], [2, 5]
    _registered(cache, request_ids, lengths)

    kv = _packed_whole_prompts(cache, request_ids, lengths, *_prompt_kv(lengths))

    assert kv.query_start_loc is not None and kv.query_start_loc.tolist() == [0, 2, 7]


def test_a_whole_prompt_attends_to_exactly_the_tokens_its_own_pass_wrote():
    """Nothing was cached before the pass, so each prompt's context is the prompt itself."""
    cache = _cache()
    request_ids, lengths = ["a", "b"], [2, 5]
    _registered(cache, request_ids, lengths)

    kv = _packed_whole_prompts(cache, request_ids, lengths, *_prompt_kv(lengths))

    assert kv.context_lens.tolist() == lengths


def test_a_whole_prompt_addresses_the_keys_its_own_pass_wrote():
    """Read through its addresses, the pool holds exactly the K/V the pass computed."""
    cache = _cache()
    request_ids, lengths = ["a", "b"], [2, 5]
    _registered(cache, request_ids, lengths)
    keys, values = _prompt_kv(lengths)

    kv = _packed_whole_prompts(cache, request_ids, lengths, keys, values)

    torch.testing.assert_close(_read_back_keys(kv, lengths), keys)


def test_packed_payload_hands_the_dense_kernels_each_query_count_on_the_host():
    cache = _cache()
    request_ids, lengths = ["a", "b"], [2, 5]
    _registered(cache, request_ids, lengths)

    kv = _packed_whole_prompts(cache, request_ids, lengths, *_prompt_kv(lengths))

    assert kv.host_query_lens == (2, 5)


# --- a prompt chunked across passes ------------------------------------------

_CHUNKED_LENGTHS = [6, 3]
_FIRST_CHUNK = [4, 2]
_SECOND_CHUNK = [2, 1]


def _ragged_chunk(keys: torch.Tensor, chunks: list[slice]) -> torch.Tensor:
    """A different slice of each packed sequence of `_CHUNKED_LENGTHS`, packed again."""
    pieces, start = [], 0
    for length, chunk in zip(_CHUNKED_LENGTHS, chunks, strict=True):
        pieces.append(keys[:, :, start : start + length][:, :, chunk])
        start += length
    return torch.cat(pieces, dim=2)


def _chunked_packed(cache: ContinuousKVCache) -> PagedKV:
    """Prefill `_CHUNKED_LENGTHS` in two packed passes; return what the second one hands attention."""
    request_ids = ["a", "b"]
    keys, values = _prompt_kv(_CHUNKED_LENGTHS)
    _registered(cache, request_ids, _FIRST_CHUNK)
    first = [slice(0, count) for count in _FIRST_CHUNK]
    cache.make_packed_payload(request_ids, _FIRST_CHUNK).update(
        _ragged_chunk(keys, first), _ragged_chunk(values, first), _LAYER
    )
    cache.advance(request_ids, _SECOND_CHUNK)
    second = [slice(count, None) for count in _FIRST_CHUNK]
    return cache.make_packed_payload(request_ids, _SECOND_CHUNK).update(
        _ragged_chunk(keys, second), _ragged_chunk(values, second), _LAYER
    )


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


def test_a_continuing_chunk_hands_the_dense_kernels_its_whole_context_on_the_host():
    kv = _chunked_packed(_cache())

    assert kv.host_context_lens == (6, 3)


def test_a_continuing_chunk_addresses_the_prefix_an_earlier_pass_wrote():
    """Read through its addresses, the pool holds the whole prompt, as if computed in one pass."""
    kv = _chunked_packed(_cache())

    torch.testing.assert_close(_read_back_keys(kv, _CHUNKED_LENGTHS), _prompt_kv(_CHUNKED_LENGTHS)[0])
