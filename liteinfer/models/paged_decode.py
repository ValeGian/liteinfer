# pyright: reportPrivateImportUsage=false
# `tl.constexpr` annotates a compile-time value, not a runtime type, so pyright
# reads every kernel launch as passing an int where a constexpr was declared.
# pyright: reportArgumentType=false
"""Decode attention that reads the KV pool where it lies, in Triton.

The dense kernels in `attention.py` want K and V as one contiguous
``[batch, kv_heads, keys, head_dim]`` tensor, so a paged cache has to copy every
sequence's whole history out of the pool before it can call them — per layer,
per step. This kernel takes the slot table instead and reads each token from
the pool directly, so that copy never happens.

Two more properties fall out of writing the loop by hand rather than handing
the operation to a general kernel:

* **Grouped-query heads are never expanded.** One program owns one KV head and
  every query head that shares it, so K and V are read once per group instead
  of once per query head (`_repeat_kv` in the dense path materialises that copy).
* **Padding is never read.** ``context_lens`` says how many tokens each sequence
  really has, so the loop stops there. The dense path pads every sequence out
  to the longest in the batch and masks the difference away afterwards, having
  already paid to move it.

Decode only: one query per sequence. That is what makes the whole history
readable in a single pass with no causal mask — the newest token attends to
everything before it, which is every token in the cache.

**Slot table, not block table.** vLLM's kernels take the block table and do the
`block_idx * block_size + offset` arithmetic themselves; this one takes the slot
table the cache already builds for its writes, one integer per token. That is 16x
more index than a block table at `block_size=16` — and 8 bytes per token against
the 2 KiB of K and V that token costs at Llama-3.2's 8 KV heads, so 0.4% more
traffic to reuse the addressing the rest of the engine already has. The table is
built once per step and read by every layer, which is what makes that trade cheap.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Tokens folded into the running softmax per iteration. 64 was the best of
# {32, 64, 128} on an A40; the tile has to be at least 16 for `tl.dot`.
_BLOCK_KV = 64

# `tl.dot` needs at least 16 rows. Models have far fewer query heads per KV head
# than that (4 for Llama-3.2, 8 for Llama-3-70B), so the query tile is padded up
# and the extra rows are masked off at the store. They cost tensor-core flops on
# a kernel that is bound by KV bandwidth, which is the cheap thing to waste.
_MIN_QUERY_ROWS = 16


@triton.jit
def _paged_decode_kernel(
    query_ptr,
    key_pool_ptr,
    value_pool_ptr,
    slot_table_ptr,
    context_lens_ptr,
    out_ptr,
    query_stride_seq,
    query_stride_head,
    pool_stride_slot,
    pool_stride_head,
    slot_table_stride_seq,
    out_stride_seq,
    out_stride_head,
    scaling,
    max_context,
    NUM_GROUPS: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One program per (sequence, KV head). Folds the sequence's slots into a running softmax.

    The accumulator is kept in fp32 and rescaled whenever a block raises the
    running maximum — the online-softmax formulation FlashAttention uses, which
    is what lets the pass finish without ever holding the scores.
    """
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)

    context_len = tl.load(context_lens_ptr + seq)
    # The slot table is right-aligned (see `cache/block_pool.slot_table`), so a
    # sequence's real tokens are its *last* `context_len` columns.
    pad = max_context - context_len

    dims = tl.arange(0, HEAD_DIM)
    query_rows = tl.arange(0, QUERY_ROWS)
    is_real_head = query_rows < NUM_GROUPS
    head_offsets = kv_head * NUM_GROUPS + query_rows

    query = tl.load(
        query_ptr + seq * query_stride_seq + head_offsets[:, None] * query_stride_head + dims[None, :],
        mask=is_real_head[:, None],
        other=0.0,
    )

    running_max = tl.full([QUERY_ROWS], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([QUERY_ROWS], dtype=tl.float32)
    accumulator = tl.zeros([QUERY_ROWS, HEAD_DIM], dtype=tl.float32)

    for start in range(0, context_len, BLOCK_KV):
        offsets = start + tl.arange(0, BLOCK_KV)
        is_real_key = offsets < context_len
        slots = tl.load(
            slot_table_ptr + seq * slot_table_stride_seq + pad + offsets,
            mask=is_real_key,
            other=0,
        )
        pool_offsets = slots[:, None] * pool_stride_slot + kv_head * pool_stride_head + dims[None, :]
        keys = tl.load(key_pool_ptr + pool_offsets, mask=is_real_key[:, None], other=0.0)
        values = tl.load(value_pool_ptr + pool_offsets, mask=is_real_key[:, None], other=0.0)

        scores = tl.dot(query, tl.trans(keys), input_precision="ieee") * scaling
        scores = tl.where(is_real_key[None, :], scores, float("-inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        rescale = tl.exp(running_max - new_max)
        weights = tl.exp(scores - new_max[:, None])

        accumulator = accumulator * rescale[:, None] + tl.dot(
            weights.to(values.dtype), values, input_precision="ieee"
        )
        running_sum = running_sum * rescale + tl.sum(weights, axis=1)
        running_max = new_max

    tl.store(
        out_ptr + seq * out_stride_seq + head_offsets[:, None] * out_stride_head + dims[None, :],
        (accumulator / running_sum[:, None]).to(out_ptr.dtype.element_ty),
        mask=is_real_head[:, None],
    )


def supports_head_dim(head_dim: int) -> bool:
    """Whether this kernel can serve a model with this head dimension.

    Triton indexes with `tl.arange`, whose length must be a power of two, and the
    head dimension is one of those lengths. Llama's 64 and 128 qualify; a model
    with 96 does not, and has to run on a dense kernel instead.
    """
    return head_dim > 0 and head_dim & (head_dim - 1) == 0


def paged_decode(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    slot_table: torch.Tensor,
    context_lens: torch.Tensor,
    scaling: float,
    num_kv_groups: int,
    block_kv: int = _BLOCK_KV,
) -> torch.Tensor:
    """Attend one query per sequence over the KV each sequence's slots address.

    Args:
        query: ``[batch, num_heads, head_dim]`` — the decode token's queries.
        key_pool: this layer's flat key store, ``[num_slots, num_kv_heads, head_dim]``.
        value_pool: the matching value store.
        slot_table: ``[batch, max_context]`` physical slot per logical position,
            right-aligned, padded columns pointing at the null block.
        context_lens: ``[batch]`` real cached tokens per sequence.
        scaling: softmax scale, normally ``head_dim ** -0.5``.
        num_kv_groups: query heads per KV head.

    Returns:
        ``[batch, num_heads, head_dim]``, same dtype as ``query``.
    """
    batch, num_heads, head_dim = query.shape
    if not supports_head_dim(head_dim):
        raise ValueError(f"paged decode needs a power-of-two head_dim, got {head_dim}")
    if key_pool.stride() != value_pool.stride():
        raise ValueError("key and value pools must be laid out identically")

    out = torch.empty_like(query)
    grid = (batch, num_heads // num_kv_groups)
    _paged_decode_kernel[grid](
        query,
        key_pool,
        value_pool,
        slot_table,
        context_lens,
        out,
        query.stride(0),
        query.stride(1),
        key_pool.stride(0),
        key_pool.stride(1),
        slot_table.stride(0),
        out.stride(0),
        out.stride(1),
        scaling,
        slot_table.shape[1],
        NUM_GROUPS=num_kv_groups,
        QUERY_ROWS=max(_MIN_QUERY_ROWS, triton.next_power_of_2(num_kv_groups)),
        BLOCK_KV=block_kv,
        HEAD_DIM=head_dim,
    )
    return out
