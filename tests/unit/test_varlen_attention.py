"""Packed attention against the reference kernel, on the batch padding exists to serve.

`varlen_attention` goes through `torch.ops.aten._flash_attention_forward`, a
private torch op, and reads a packed layout no other kernel here reads. Both are
reasons to pin its answer rather than trust it: every test below states the same
property, that packing changes the layout and not the attention.
"""

from __future__ import annotations

import pytest
import torch

from liteinfer.engine.attention_mask import build_prefill_mask
from liteinfer.models.attention import (
    DenseKV,
    VarlenKV,
    eager_attention,
    varlen_attention,
    varlen_unsupported_reason,
)

pytestmark = pytest.mark.gpu

_HEAD_DIM = 64
_QUERY_HEADS = 8
_DTYPE = torch.bfloat16


def _packed(prompt_lens: list[int], kv_heads: int = 8) -> tuple[torch.Tensor, VarlenKV]:
    """One flat run of tokens, and the K/V a prefill pass would have computed for it."""
    torch.manual_seed(0)
    total = sum(prompt_lens)
    query = torch.randn(1, _QUERY_HEADS, total, _HEAD_DIM, dtype=_DTYPE, device="cuda")
    keys = torch.randn(1, kv_heads, total, _HEAD_DIM, dtype=_DTYPE, device="cuda")
    values = torch.randn_like(keys)

    cu_seqlens = _prefix_sums(prompt_lens)
    return query, VarlenKV(
        keys, values, cu_seqlens, max(prompt_lens), cu_seqlens, max(prompt_lens)
    )


def _prefix_sums(lengths: list[int]) -> torch.Tensor:
    boundaries = torch.tensor(lengths, device="cuda").cumsum(0)
    return torch.cat([torch.zeros(1, device="cuda"), boundaries]).to(torch.int32)


def _padded_reference(
    query: torch.Tensor, kv: VarlenKV, prompt_lens: list[int], kv_heads: int = 8
) -> torch.Tensor:
    """The same attention the padded path computes, returned in packed layout.

    Each sequence is attended on its own — which is what padding plus a mask adds
    up to — and the answers are concatenated back into one run.
    """
    num_kv_groups = _QUERY_HEADS // kv_heads
    outputs = []
    start = 0
    for prompt_len in prompt_lens:
        window = slice(start, start + prompt_len)
        mask = build_prefill_mask([prompt_len], _DTYPE, torch.device("cuda"))
        outputs.append(
            eager_attention(
                query[:, :, window, :],
                DenseKV(kv.keys[:, :, window, :], kv.values[:, :, window, :]),
                mask,
                _HEAD_DIM**-0.5,
                num_kv_groups,
            )
        )
        start += prompt_len
    return torch.cat(outputs, dim=2)


def test_packed_attention_matches_the_reference_kernel_on_mixed_lengths():
    """The case padding exists to serve: prompts whose lengths differ."""
    prompt_lens = [17, 300, 5, 129]
    query, kv = _packed(prompt_lens)

    packed = varlen_attention(query, kv, None, _HEAD_DIM**-0.5, 1)

    torch.testing.assert_close(
        packed, _padded_reference(query, kv, prompt_lens), rtol=0, atol=2**-6
    )


def test_packed_attention_never_reads_across_a_sequence_boundary():
    """Two sequences in one run must not see each other, whatever sits next to them."""
    prompt_lens = [4, 4]
    query, kv = _packed(prompt_lens)
    alone = varlen_attention(
        query[:, :, :4, :],
        VarlenKV(
            kv.keys[:, :, :4, :], kv.values[:, :, :4, :],
            kv.cu_seqlens_q[:2], 4, kv.cu_seqlens_k[:2], 4,
        ),
        None,
        _HEAD_DIM**-0.5,
        1,
    )

    together = varlen_attention(query, kv, None, _HEAD_DIM**-0.5, 1)

    torch.testing.assert_close(together[:, :, :4, :], alone)


def test_a_chunk_attends_to_its_cached_prefix_and_its_own_past():
    """A chunk continuing a prompt has fewer queries than keys, and they are the *last* ones.

    Flash aligns its causal diagonal to the bottom right in that case, which is
    what makes every query see the whole prefix — but the op's signature does
    not say so, so the alignment is pinned against `eager` under the padded
    path's own mask. The second sequence brings its whole prompt, so the batch
    also mixes a chunk with a sequence that has no prefix at all.
    """
    torch.manual_seed(0)
    query_lens, context_lens = [3, 5], [9, 5]
    query = torch.randn(1, _QUERY_HEADS, sum(query_lens), _HEAD_DIM, dtype=_DTYPE, device="cuda")
    keys = torch.randn(1, _QUERY_HEADS, sum(context_lens), _HEAD_DIM, dtype=_DTYPE, device="cuda")
    values = torch.randn_like(keys)
    kv = VarlenKV(
        keys, values,
        _prefix_sums(query_lens), max(query_lens), _prefix_sums(context_lens), max(context_lens),
    )

    packed = varlen_attention(query, kv, None, _HEAD_DIM**-0.5, 1)

    outputs, query_start, key_start = [], 0, 0
    for query_len, context_len in zip(query_lens, context_lens, strict=True):
        queries = slice(query_start, query_start + query_len)
        context = slice(key_start, key_start + context_len)
        mask = build_prefill_mask([query_len], _DTYPE, torch.device("cuda"), [context_len])
        outputs.append(
            eager_attention(
                query[:, :, queries, :],
                DenseKV(keys[:, :, context, :], values[:, :, context, :]),
                mask,
                _HEAD_DIM**-0.5,
                1,
            )
        )
        query_start += query_len
        key_start += context_len
    torch.testing.assert_close(packed, torch.cat(outputs, dim=2), rtol=0, atol=2**-6)


def test_packed_attention_serves_grouped_query_heads():
    """Flash reads one KV head per group, so the dense path's `_repeat_kv` copy is gone."""
    prompt_lens = [64, 16]
    query, kv = _packed(prompt_lens, kv_heads=2)

    packed = varlen_attention(query, kv, None, _HEAD_DIM**-0.5, _QUERY_HEADS // 2)

    torch.testing.assert_close(
        packed, _padded_reference(query, kv, prompt_lens, kv_heads=2), rtol=0, atol=2**-6
    )


def test_packed_attention_refuses_a_mask():
    """A mask means someone still thinks there is padding to hide."""
    query, kv = _packed([4, 4])
    mask = build_prefill_mask([4, 4], _DTYPE, torch.device("cuda"))

    with pytest.raises(ValueError, match="takes no mask"):
        varlen_attention(query, kv, mask, _HEAD_DIM**-0.5, 1)


def test_packing_is_refused_where_flash_cannot_run():
    """CPU has no flash kernel, so the engine pads there rather than pretending."""
    assert varlen_unsupported_reason("paged", torch.device("cpu"), torch.bfloat16) is not None


def test_packing_is_refused_in_single_precision():
    """Flash computes in half precision; float32 is the parity path's dtype."""
    assert varlen_unsupported_reason("paged", torch.device("cuda"), torch.float32) is not None


def test_packing_is_refused_for_a_kernel_that_cannot_read_boundaries():
    """`eager` and `sdpa` take a padded batch and a mask; neither can honour cu_seqlens."""
    assert varlen_unsupported_reason("eager", torch.device("cuda"), torch.bfloat16) is not None


def test_the_dense_kernels_refuse_a_packed_batch_rather_than_misread_it():
    """The failure this guard exists for is a wrong answer, not a crash.

    `VarlenKV` names its fields `keys` and `values` exactly as the dense payloads
    do, so `eager` would read one prompt's queries against the next prompt's keys
    — and with no mask, against its own future — and return a plausible tensor.
    """
    prompt_lens = [4, 4]
    query, kv = _packed(prompt_lens)

    with pytest.raises(ValueError, match="cannot read a packed batch"):
        eager_attention(query, kv, None, _HEAD_DIM**-0.5, 1)  # type: ignore[arg-type]
