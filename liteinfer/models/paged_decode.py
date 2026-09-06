# pyright: reportPrivateImportUsage=false
# `tl.constexpr` annotates a compile-time value, not a runtime type, so pyright
# reads every kernel launch as passing an int where a constexpr was declared.
# pyright: reportArgumentType=false
# A `@triton.jit` helper is typed as never returning, because calling one from
# Python raises — it is only callable from inside another Triton function, which
# is where the calls below are.
# pyright: reportGeneralTypeIssues=false
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

One pass or two
---------------
The natural grid is one program per (sequence, KV head), which is 256 programs at
batch 32 and **8** at batch 1 — and a GPU with 84 SMs runs the second of those on
a tenth of the device, however well each program is written. So a narrow batch
takes a second shape: cut each sequence's keys into `num_splits` contiguous
slices, give each slice its own program, and combine the partial softmaxes in a
second pass. That is flash-decoding's split-K, and it buys the parallelism the
batch cannot.

`_choose_num_splits` picks between the two from the batch width, the context and
the device, and picks the single pass whenever the grid is already full or the
context holds too little sequential work to be worth cutting — so the wide batch
the unsplit kernel was measured on keeps exactly the kernel it was measured with.
What makes the combine possible is that a partial softmax carries its own
log-sum-exp: the mass its slice contributed, which is what the second pass weighs
the slices by.

Both shapes read the pool the same way, so the online-softmax loop and the query
load are written once and shared (`_fold_keys_into_running_softmax`,
`_load_query_group`). The two kernels differ only in which keys they are given
and where they put the answer.

The partial buffers keep their last axis contiguous, as the query and the pool
do: a kernel here is passed the strides it needs to reach a row and indexes
within that row directly.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

# Tokens folded into the running softmax per iteration. 64 was the best of
# {16, 32, 64, 128} on an A40; the tile has to be at least 16 for `tl.dot`, and
# the answer must not depend on it — `block_kv` stays a parameter so the sweep
# that chose 64 is re-runnable, and the tests sweep the same values. A split
# slice can be shorter than the tile, in which case the tile is masked down to it.
_BLOCK_KV = 64

# `tl.dot` needs tiles of at least 16 and `tl.arange` needs a power-of-two
# length. Neither the query-head group (4 for Llama-3.2, 8 for Llama-3-70B) nor
# every model's head dimension (96 for Phi-3-mini, 80 for Phi-2) is such a
# number, so both axes are padded up to a legal tile. Padding costs tensor-core
# flops on a kernel bound by KV bandwidth, which is the cheap thing to waste;
# where an axis is already legal the padded and real extents coincide.
_MIN_TILE = 16

# The most programs one sequence's key loop is ever cut into. The combine pass
# holds `[num_splits, head_dim]` fp32 per (sequence, query head) in registers, so
# this is what bounds that tile — 32 splits x 128 dims is 16 KiB. It binds only
# where the batch is narrow and the KV heads few: at one KV head and batch 1 the
# device would otherwise ask for two programs per SM.
_MAX_SPLITS = 32

# Programs the split grid aims for, per SM. Covering the SMs once is not the
# target: a program spends most of its time waiting on the pool, so a second
# resident wave hides that latency rather than queueing behind it. Measured per
# layer at Llama-3.2-1B's decode shapes, batch 1 and 4,096 tokens of context:
# 84.6 us unsplit, 11.9 us at 21 splits (168 programs), 10.5 us at 32 (256). The
# curve is flat past two waves and the combine pass is not, so it stops there.
#
# It doubles as the gate on batch width, because `target // programs` falls to 1
# once the unsplit grid is already there: from batch 12 up nothing is split, and
# measured across every context that is the right answer — splitting there is
# 0.85x to 0.97x.
_TARGET_PROGRAMS_PER_SM = 2

# Key tiles the context must hold before splitting it is worth a second pass.
# The combine pass costs a fixed 0.4 us at batch 1 and about 1.2 us at batch 8,
# however little the splits saved, and a short context has nothing to save: at
# 128 tokens — two tiles — the unsplit kernel is already at its floor of 4.7 us
# and every split count measured is slower. Four tiles is where the sequential
# work starts to exceed the fixed cost (1.41x at batch 1, 256 tokens).
_MIN_TILES_TO_SPLIT = 4


@triton.jit
def _sequence_slots(slot_table_ptr, seq, slot_table_stride_seq, max_context, context_len):
    """Address of this sequence's slot for logical position 0.

    The slot table is right-aligned (see `cache/block_pool.slot_table`), so a
    sequence's real tokens are its *last* `context_len` columns.
    """
    return slot_table_ptr + seq * slot_table_stride_seq + (max_context - context_len)


@triton.jit
def _load_query_group(
    query_ptr,
    seq,
    kv_head,
    query_stride_seq,
    query_stride_head,
    dims,
    is_real_dim,
    NUM_GROUPS: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
):
    """Load the queries of every head that shares one KV head, as one tile.

    Returns the ``[QUERY_ROWS, HEAD_DIM_TILE]`` tile, the query-head index behind
    each of its rows, and which of those rows is a real head rather than tile
    padding.

    Zeroing the padded lanes is what makes the padding arithmetically free: a
    zero contributes nothing to the score dot however much rubbish sits opposite
    it in K, and the masked stores drop the padded outputs. Every other mask on
    these two axes is therefore about memory, not about answers.
    """
    query_rows = tl.arange(0, QUERY_ROWS)
    is_real_head = query_rows < NUM_GROUPS
    head_offsets = kv_head * NUM_GROUPS + query_rows
    query = tl.load(
        query_ptr + seq * query_stride_seq + head_offsets[:, None] * query_stride_head + dims[None, :],
        mask=is_real_head[:, None] & is_real_dim[None, :],
        other=0.0,
    )
    return query, head_offsets, is_real_head


@triton.jit
def _fold_keys_into_running_softmax(
    query,
    key_pool_ptr,
    value_pool_ptr,
    slot_ptr,
    pool_stride_slot,
    pool_stride_head,
    kv_head,
    first_key,
    last_key,
    dims,
    is_real_dim,
    scaling,
    QUERY_ROWS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM_TILE: tl.constexpr,
):
    """Attend `query` over the logical key positions ``[first_key, last_key)``.

    The accumulator is kept in fp32 and rescaled whenever a tile raises the
    running maximum — the online-softmax formulation FlashAttention uses, which
    is what lets the pass finish without ever holding the scores. It comes back
    unnormalised, alongside the maximum and the denominator it is relative to, so
    a caller holding only a slice of the key axis can hand its slice on to be
    weighed against the others.

    `slot_ptr` addresses this sequence's slot for logical position 0.
    """
    running_max = tl.full([QUERY_ROWS], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([QUERY_ROWS], dtype=tl.float32)
    accumulator = tl.zeros([QUERY_ROWS, HEAD_DIM_TILE], dtype=tl.float32)

    for start in range(first_key, last_key, BLOCK_KV):
        offsets = start + tl.arange(0, BLOCK_KV)
        is_real_key = offsets < last_key
        slots = tl.load(slot_ptr + offsets, mask=is_real_key, other=0)
        pool_offsets = slots[:, None] * pool_stride_slot + kv_head * pool_stride_head + dims[None, :]
        # `is_real_dim` here is a bounds guard, not a correctness one: the last KV
        # head of the last slot is the end of the allocation, so a padded lane
        # would read past it. The values it would load could not reach the output
        # either way, which is why no test can distinguish this mask being wrong.
        is_addressable = is_real_key[:, None] & is_real_dim[None, :]
        keys = tl.load(key_pool_ptr + pool_offsets, mask=is_addressable, other=0.0)
        values = tl.load(value_pool_ptr + pool_offsets, mask=is_addressable, other=0.0)

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

    return accumulator, running_sum, running_max


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
    HEAD_DIM_TILE: tl.constexpr,
):
    """One program per (sequence, KV head), folding a sequence's whole history in one pass.

    This is the shape the grid takes when it already covers the device. A narrow
    batch cannot fill it, and splits the key axis instead — see
    `_paged_decode_split_kernel`.
    """
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)

    context_len = tl.load(context_lens_ptr + seq)
    dims = tl.arange(0, HEAD_DIM_TILE)
    is_real_dim = dims < HEAD_DIM
    query, head_offsets, is_real_head = _load_query_group(
        query_ptr,
        seq,
        kv_head,
        query_stride_seq,
        query_stride_head,
        dims,
        is_real_dim,
        NUM_GROUPS,
        QUERY_ROWS,
    )

    accumulator, running_sum, _ = _fold_keys_into_running_softmax(
        query,
        key_pool_ptr,
        value_pool_ptr,
        _sequence_slots(slot_table_ptr, seq, slot_table_stride_seq, max_context, context_len),
        pool_stride_slot,
        pool_stride_head,
        kv_head,
        0,
        context_len,
        dims,
        is_real_dim,
        scaling,
        QUERY_ROWS,
        BLOCK_KV,
        HEAD_DIM_TILE,
    )

    tl.store(
        out_ptr + seq * out_stride_seq + head_offsets[:, None] * out_stride_head + dims[None, :],
        (accumulator / running_sum[:, None]).to(out_ptr.dtype.element_ty),
        mask=is_real_head[:, None] & is_real_dim[None, :],
    )


@triton.jit
def _paged_decode_split_kernel(
    query_ptr,
    key_pool_ptr,
    value_pool_ptr,
    slot_table_ptr,
    context_lens_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    query_stride_seq,
    query_stride_head,
    pool_stride_slot,
    pool_stride_head,
    slot_table_stride_seq,
    partial_out_stride_seq,
    partial_out_stride_head,
    partial_out_stride_split,
    partial_lse_stride_seq,
    partial_lse_stride_head,
    scaling,
    max_context,
    NUM_GROUPS: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_TILE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """One program per (sequence, KV head, split), each folding its own slice of the keys.

    The grid is `NUM_SPLITS` times wider than the unsplit pass, which is the
    whole point: at batch 1 that pass is one program per KV head and leaves most
    of the device idle. Each program writes its slice's softmax, normalised, plus
    the log-sum-exp saying how much mass the slice carried —
    `_combine_splits_kernel` needs both to weigh the slices against each other.
    """
    seq = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)

    context_len = tl.load(context_lens_ptr + seq)
    keys_per_split = tl.cdiv(context_len, NUM_SPLITS)
    first_key = split * keys_per_split
    last_key = tl.minimum(first_key + keys_per_split, context_len)

    dims = tl.arange(0, HEAD_DIM_TILE)
    is_real_dim = dims < HEAD_DIM
    query, head_offsets, is_real_head = _load_query_group(
        query_ptr,
        seq,
        kv_head,
        query_stride_seq,
        query_stride_head,
        dims,
        is_real_dim,
        NUM_GROUPS,
        QUERY_ROWS,
    )

    accumulator, running_sum, running_max = _fold_keys_into_running_softmax(
        query,
        key_pool_ptr,
        value_pool_ptr,
        _sequence_slots(slot_table_ptr, seq, slot_table_stride_seq, max_context, context_len),
        pool_stride_slot,
        pool_stride_head,
        kv_head,
        first_key,
        last_key,
        dims,
        is_real_dim,
        scaling,
        QUERY_ROWS,
        BLOCK_KV,
        HEAD_DIM_TILE,
    )

    # One split count serves the whole batch, so a sequence shorter than the
    # longest can leave its last splits with no keys at all. `-inf` is the mass
    # the combine pass reads as "contributed nothing"; the denominator has to
    # sidestep the 0/0 that would otherwise put a NaN opposite that zero.
    is_empty = running_sum == 0.0
    denominator = tl.where(is_empty, 1.0, running_sum)
    log_sum_exp = tl.where(is_empty, float("-inf"), running_max + tl.log(running_sum))

    tl.store(
        partial_out_ptr
        + seq * partial_out_stride_seq
        + head_offsets[:, None] * partial_out_stride_head
        + split * partial_out_stride_split
        + dims[None, :],
        accumulator / denominator[:, None],
        mask=is_real_head[:, None] & is_real_dim[None, :],
    )
    tl.store(
        partial_lse_ptr
        + seq * partial_lse_stride_seq
        + head_offsets * partial_lse_stride_head
        + split,
        log_sum_exp,
        mask=is_real_head,
    )


@triton.jit
def _combine_splits_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    partial_out_stride_seq,
    partial_out_stride_head,
    partial_out_stride_split,
    partial_lse_stride_seq,
    partial_lse_stride_head,
    out_stride_seq,
    out_stride_head,
    NUM_SPLITS: tl.constexpr,
    SPLIT_TILE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_TILE: tl.constexpr,
):
    """Weigh each split's softmax by the mass its keys carried. One program per (sequence, head).

    A slice whose scores were all small should barely move the answer, and its
    log-sum-exp is exactly that mass: with ``w_s = exp(lse_s - max_s lse_s)`` the
    answer is ``sum_s w_s o_s / sum_s w_s``. Subtracting the maximum is what keeps
    the exponentials in range, and it is also what sends an empty split's ``-inf``
    to a weight of zero.
    """
    seq = tl.program_id(0)
    head = tl.program_id(1)

    splits = tl.arange(0, SPLIT_TILE)
    is_real_split = splits < NUM_SPLITS
    dims = tl.arange(0, HEAD_DIM_TILE)
    is_real_dim = dims < HEAD_DIM

    log_sum_exp = tl.load(
        partial_lse_ptr + seq * partial_lse_stride_seq + head * partial_lse_stride_head + splits,
        mask=is_real_split,
        other=float("-inf"),
    )
    weights = tl.exp(log_sum_exp - tl.max(log_sum_exp, axis=0))

    partials = tl.load(
        partial_out_ptr
        + seq * partial_out_stride_seq
        + head * partial_out_stride_head
        + splits[:, None] * partial_out_stride_split
        + dims[None, :],
        mask=is_real_split[:, None] & is_real_dim[None, :],
        other=0.0,
    )
    combined = tl.sum(partials * weights[:, None], axis=0) / tl.sum(weights, axis=0)

    tl.store(
        out_ptr + seq * out_stride_seq + head * out_stride_head + dims,
        combined.to(out_ptr.dtype.element_ty),
        mask=is_real_dim,
    )


def _tile(extent: int) -> int:
    """The smallest tile Triton can index that covers `extent`."""
    # `next_power_of_2` is annotated as returning a Triton `constexpr`; it is an
    # int here, and the kernel wants it as one.
    return max(_MIN_TILE, int(triton.next_power_of_2(extent)))


@functools.cache
def _multiprocessor_count(device_index: int) -> int:
    """How many SMs this device has. Cached: the choice below is made per layer per step."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _choose_num_splits(
    batch: int,
    num_kv_heads: int,
    max_context: int,
    block_kv: int,
    device_index: int,
) -> int:
    """How many programs to cut each sequence's key loop into.

    The unsplit grid is one program per (sequence, KV head) — 256 at batch 32,
    and 8 at batch 1, which on an 84-SM GPU runs a single request's decode on a
    tenth of the device. Splitting the key axis buys the programs the batch
    cannot supply; the count wanted is whatever brings the grid up to
    `_TARGET_PROGRAMS_PER_SM` waves.

    Two measured bounds keep that from being taken too far, and between them they
    are why the wide batch and the short context both come back as one pass:

    * **The grid may already be full**, which `target // programs` says on its
      own: from batch 12 up it is 1 and nothing is split.
    * **The context may hold nothing worth cutting**, which `_MIN_TILES_TO_SPLIT`
      says. One tile per split is the finest cut worth making, so the tile count
      bounds the answer from above as well.

    Inside those bounds the win is large and grows with context, because context
    is what the unsplit program traverses one tile at a time. Measured per layer
    at batch 1: 1.41x at 256 tokens, 2.23x at 512, 3.47x at 1,024, 7.10x at 4,096.

    One shape pays for the rest: at batch 8 and 256 tokens the two splits this
    picks measure **0.94x**, the fixed cost of the combine pass over a context too
    short to make it back. Excluding it means either giving up the whole batch-8
    band (1.04x to 1.22x from 384 tokens up) or the batch-1 win at 256 tokens
    (1.41x), and both are worth more than 6% of 8 us.

    The count changes with the batch width, and each distinct value is a kernel
    Triton compiles once and caches. That is a handful of compiles over a run —
    bounded by `_MAX_SPLITS`, and paid where the batch is draining.
    """
    tiles = int(triton.cdiv(max_context, block_kv))
    if tiles < _MIN_TILES_TO_SPLIT:
        return 1

    programs = batch * num_kv_heads
    target_programs = _TARGET_PROGRAMS_PER_SM * _multiprocessor_count(device_index)
    return max(1, min(target_programs // programs, tiles, _MAX_SPLITS))


def paged_decode(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    slot_table: torch.Tensor,
    context_lens: torch.Tensor,
    scaling: float,
    num_kv_groups: int,
    block_kv: int = _BLOCK_KV,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Attend one query per sequence over the KV each sequence's slots address.

    Any head dimension and any query-head grouping are served: an axis Triton
    cannot index directly is padded to a tile and masked.

    Args:
        query: ``[batch, num_heads, head_dim]`` — the decode token's queries.
        key_pool: this layer's flat key store, ``[num_slots, num_kv_heads, head_dim]``.
        value_pool: the matching value store.
        slot_table: ``[batch, max_context]`` physical slot per logical position,
            right-aligned, padded columns pointing at the null block.
        context_lens: ``[batch]`` real cached tokens per sequence.
        scaling: softmax scale, normally ``head_dim ** -0.5``.
        num_kv_groups: query heads per KV head.
        block_kv: keys folded into the running softmax per iteration.
        num_splits: how many programs share each sequence's key loop. ``None``
            chooses from the batch width and the device; ``1`` is the single-pass
            kernel. Exposed so the choice can be pinned and swept.

    Returns:
        ``[batch, num_heads, head_dim]``, same dtype as ``query``.
    """
    batch, num_heads, head_dim = query.shape
    if key_pool.stride() != value_pool.stride():
        raise ValueError("key and value pools must be laid out identically")
    if num_splits is not None and num_splits < 1:
        raise ValueError(f"num_splits must be >= 1, got {num_splits}")

    num_kv_heads = num_heads // num_kv_groups
    max_context = slot_table.shape[1]
    if num_splits is None:
        num_splits = _choose_num_splits(
            batch, num_kv_heads, max_context, block_kv, query.device.index
        )

    out = torch.empty_like(query)
    tiles = {
        "NUM_GROUPS": num_kv_groups,
        "QUERY_ROWS": _tile(num_kv_groups),
        "BLOCK_KV": block_kv,
        "HEAD_DIM": head_dim,
        "HEAD_DIM_TILE": _tile(head_dim),
    }

    if num_splits == 1:
        _paged_decode_kernel[(batch, num_kv_heads)](
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
            max_context,
            **tiles,
        )
        return out

    # The partials stay fp32 whatever the query's dtype: the combine pass sums
    # them against each other, and rounding each slice first would fold that
    # error into the sum rather than only into the result.
    partial_out = torch.empty(
        batch, num_heads, num_splits, head_dim, dtype=torch.float32, device=query.device
    )
    partial_lse = torch.empty(
        batch, num_heads, num_splits, dtype=torch.float32, device=query.device
    )
    _paged_decode_split_kernel[(batch, num_kv_heads, num_splits)](
        query,
        key_pool,
        value_pool,
        slot_table,
        context_lens,
        partial_out,
        partial_lse,
        query.stride(0),
        query.stride(1),
        key_pool.stride(0),
        key_pool.stride(1),
        slot_table.stride(0),
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        scaling,
        max_context,
        NUM_SPLITS=num_splits,
        **tiles,
    )
    _combine_splits_kernel[(batch, num_heads)](
        partial_out,
        partial_lse,
        out,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        out.stride(0),
        out.stride(1),
        NUM_SPLITS=num_splits,
        SPLIT_TILE=int(triton.next_power_of_2(num_splits)),
        HEAD_DIM=head_dim,
        HEAD_DIM_TILE=_tile(head_dim),
    )
    return out
