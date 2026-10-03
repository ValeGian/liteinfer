# pyright: reportPrivateImportUsage=false
"""Attention kernels. One entry per `EngineConfig.attn_implementation`.

Every kernel reads the same thing: a `PagedKV` — the layer's pool, the slots
each sequence's context occupies, and how long each context is. The queries
are each sequence's newest tokens, one per batch row in a decode step and laid
end to end in a packed pass, where `query_start_loc` says where each sequence's
queries begin. `EngineConfig.attn_implementation=None` — the default — means "the
fastest one that runs here", which `select_implementation` resolves once per
engine from the device. Naming one asks for it specifically, and is refused
rather than downgraded when it cannot run.

`paged` reads the pool in place, in Triton (see `models/paged_decode.py`), and
takes its bounds as device tensors, which is what lets a decode step be
captured and split. `eager` and `sdpa` copy the contexts out of the pool: a
decode step all at once, each row bounded by its context length, and a packed
pass one sequence at a time, which costs a few launches per sequence per layer.
They are the correctness reference and the path for CPU and Triton-less
installs, not performance paths. `eager` writes the `[heads, queries, keys]` score matrix out,
which is what the arithmetic looks like and what the fused kernels are checked
against; `sdpa` hands the operation to PyTorch, which tiles it and never
materialises that matrix.

Causality is a rule rather than a mask: a sequence's queries are the last
positions of its context, so query ``j`` of ``q`` sees the keys up to position
``context_len - q + j``. A decode query is the last position, and sees
everything.
"""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import nn


class PagedKV(NamedTuple):
    """K and V left in the pool, plus the addresses a kernel needs to find them."""

    key_pool: torch.Tensor
    """This layer's flat key store, ``[num_slots, kv_heads, head_dim]``."""
    value_pool: torch.Tensor
    slot_table: torch.Tensor
    """``[batch, max_context]`` physical slot per logical position, left-aligned;
    columns past a row's context are never read."""
    context_lens: torch.Tensor
    """``[batch]`` tokens each sequence holds, this pass's included — where each
    row's context ends."""
    num_splits: int | None = None
    """How many programs share each sequence's key loop; None lets the kernel choose."""
    query_start_loc: torch.Tensor | None = None
    """``[batch + 1]`` int32 prefix sums of each sequence's queries, for a packed
    pass — whole prompts, chunks and sampled tokens laid end to end, any number
    per row. None is a decode step: one query per sequence, laid out one per
    batch row."""
    max_query_len: int = 1
    """Most queries any one sequence brings, which sizes the kernel's grid."""
    host_context_lens: tuple[int, ...] = ()
    """`context_lens` on the host, for a dense kernel slicing a packed pass one
    sequence at a time. Empty for a decode step, which needs no slicing."""
    host_query_lens: tuple[int, ...] = ()
    """Each sequence's query count on the host, for the same slicing."""


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query KV heads back to the number of query heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, slen, head_dim)
        .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    )


def _future_keys(num_queries: int, num_keys: int, device: torch.device) -> torch.Tensor:
    """``[queries, keys]``, True where a key lies after the query: the queries are the last positions."""
    query_positions = torch.arange(num_queries, device=device).unsqueeze(1) + (num_keys - num_queries)
    return torch.arange(num_keys, device=device).unsqueeze(0) > query_positions


def _gathered(pool: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Contexts out of a flat pool: ``[B, W]`` slots -> ``[B, kv_heads, W, head_dim]``."""
    return pool[slots].permute(0, 2, 1, 3)


# One sequence's or one batch's attention: (query, keys, values, visible keys or
# None, whether the rule is square-causal) -> output.
_Attend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, bool], torch.Tensor]


def _dense(query: torch.Tensor, kv: PagedKV, attend: _Attend) -> torch.Tensor:
    """Route a dense kernel: a decode step batched, a packed pass one sequence at a time."""
    if kv.query_start_loc is None:
        return _decode_batch(query, kv, attend)
    return _per_sequence(query, kv, attend)


def _decode_batch(query: torch.Tensor, kv: PagedKV, attend: _Attend) -> torch.Tensor:
    """One query per row, every row's context gathered at once and bounded by its length.

    The table is as wide as the batch's longest context; a row's columns past
    its own context address whatever the padding names, and the bound hides
    them — the rule the paged kernel applies by stopping its loop there.
    """
    width = kv.slot_table.shape[1]
    visible = torch.arange(width, device=query.device) < kv.context_lens.unsqueeze(1)
    keys, values = _gathered(kv.key_pool, kv.slot_table), _gathered(kv.value_pool, kv.slot_table)
    return attend(query, keys, values, visible[:, None, None, :], False)


def _per_sequence(query: torch.Tensor, kv: PagedKV, attend: _Attend) -> torch.Tensor:
    """A packed pass, one sequence at a time, over the context its slots address.

    Only attention loops: projections, MLP and LM head run once over every
    token around it. Batching the sequences would mean padding their queries
    to the longest, which is the layout the engine does not have.
    """
    if not (kv.host_context_lens and kv.host_query_lens):
        raise ValueError(
            "a dense kernel slices each sequence of a packed pass on the host, and was "
            "handed no host_context_lens / host_query_lens"
        )
    outputs = []
    rows = query.split(list(kv.host_query_lens), dim=2)
    for seq, (queries, context_len) in enumerate(zip(rows, kv.host_context_lens, strict=True)):
        slots = kv.slot_table[seq : seq + 1, :context_len]
        num_queries = queries.shape[2]
        # A single query sees everything; a whole prompt is the square causal
        # case; a chunk after a cached prefix needs its rule spelled out.
        is_whole_prompt = num_queries == context_len
        is_causal = num_queries > 1 and is_whole_prompt
        visible = None
        if num_queries > 1 and not is_whole_prompt:
            visible = ~_future_keys(num_queries, context_len, query.device)
        keys, values = _gathered(kv.key_pool, slots), _gathered(kv.value_pool, slots)
        outputs.append(attend(queries, keys, values, visible, is_causal))
    return torch.cat(outputs, dim=2)


def _eager(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    visible: torch.Tensor | None,
    is_causal: bool,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    """Attention written out in matmuls."""
    key = _repeat_kv(keys, num_kv_groups)
    value = _repeat_kv(values, num_kv_groups)
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if is_causal:
        visible = ~_future_keys(query.shape[2], key.shape[2], query.device)
    if visible is not None:
        scores = scores.masked_fill(~visible, float("-inf"))
    weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value)


def _sdpa(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    visible: torch.Tensor | None,
    is_causal: bool,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    """The same attention through `torch.nn.functional.scaled_dot_product_attention`.

    A whole prompt passes `is_causal` and no mask, which keeps FlashAttention
    eligible on CUDA; a mask rules it out in favour of the memory-efficient
    backend, which also never materialises the scores.
    """
    key = _repeat_kv(keys, num_kv_groups)
    value = _repeat_kv(values, num_kv_groups)
    return nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=visible, is_causal=is_causal, scale=scaling
    )


def eager_attention(query: torch.Tensor, kv: PagedKV, scaling: float, num_kv_groups: int) -> torch.Tensor:
    """Attention written out in matmuls. Readable, and the memory ceiling.

    Kept because it is what the arithmetic looks like — every step of scaled
    dot-product attention is a line here — and because it is the reference the
    fused kernels are checked against.
    """
    return _dense(query, kv, functools.partial(_eager, scaling=scaling, num_kv_groups=num_kv_groups))


def sdpa_attention(query: torch.Tensor, kv: PagedKV, scaling: float, num_kv_groups: int) -> torch.Tensor:
    """The same attention through PyTorch's fused kernel, which never materialises the scores."""
    return _dense(query, kv, functools.partial(_sdpa, scaling=scaling, num_kv_groups=num_kv_groups))


def paged_attention(query: torch.Tensor, kv: PagedKV, scaling: float, num_kv_groups: int) -> torch.Tensor:
    """Attend straight out of the KV pool: a decode step one query per row, a packed pass any number.

    Which kernel runs is decided by what the cache handed over, not by a flag:
    `query_start_loc` is what a packed pass carries and a decode step does not.
    """
    if kv.query_start_loc is None:
        return _paged_decode_step(query, kv, scaling, num_kv_groups)
    return _paged_chunks(query, kv, scaling, num_kv_groups)


def _paged_decode_step(
    query: torch.Tensor, kv: PagedKV, scaling: float, num_kv_groups: int
) -> torch.Tensor:
    """One query per sequence, ``[batch, heads, 1, dim]``."""
    if query.shape[2] != 1:
        raise ValueError(f"paged decode takes one query per sequence, got {query.shape[2]}")

    # Imported here rather than at module scope: the kernel needs Triton, which
    # torch's CPU-only builds do not ship, and the other kernels must keep
    # importing on those machines.
    from liteinfer.models.paged_decode import paged_decode

    attn_output = paged_decode(
        query.squeeze(2),
        kv.key_pool,
        kv.value_pool,
        kv.slot_table,
        kv.context_lens,
        scaling,
        num_kv_groups,
        num_splits=kv.num_splits,
    )
    return attn_output.unsqueeze(2)


def _paged_chunks(query: torch.Tensor, kv: PagedKV, scaling: float, num_kv_groups: int) -> torch.Tensor:
    """Packed queries, ``[1, heads, total_queries, dim]``, with their boundaries on `kv`."""
    assert kv.query_start_loc is not None, "the caller dispatched on it"
    from liteinfer.models.paged_decode import paged_prefill

    # The layers speak `[1, heads, tokens, dim]`; the kernel wants the token axis first.
    attn_output = paged_prefill(
        query.squeeze(0).transpose(0, 1),
        kv.key_pool,
        kv.value_pool,
        kv.slot_table,
        kv.context_lens,
        kv.query_start_loc,
        kv.max_query_len,
        scaling,
        num_kv_groups,
    )
    return attn_output.transpose(0, 1).unsqueeze(0)


IMPLEMENTATIONS = {
    "eager": eager_attention,
    "sdpa": sdpa_attention,
    "paged": paged_attention,
}
PAGED_IMPLEMENTATION = "paged"
"""Fastest everywhere, and the preferred choice wherever its preconditions hold."""

UNIVERSAL_IMPLEMENTATION = "sdpa"
"""Runs on any device and any model shape, so it is what the preference falls back to."""


def resolve(name: str):
    """Look up a kernel by name."""
    if name not in IMPLEMENTATIONS:
        raise ValueError(
            f"unknown attn_implementation {name!r}; known: {sorted(IMPLEMENTATIONS)}"
        )
    return IMPLEMENTATIONS[name]


def unsupported_reason(name: str, device: torch.device) -> str | None:
    """Why this kernel cannot run here, or `None` if it can.

    Only the paged kernel has preconditions, and both are about where the code
    can execute rather than what it is asked to compute: it is a Triton kernel,
    so it needs CUDA and the package installed. Any head dimension and any
    query-head grouping are served, by padding those axes to a tile Triton can
    index (see `models/paged_decode`).
    """
    if name != PAGED_IMPLEMENTATION:
        return None
    if device.type != "cuda":
        return f"it is a CUDA kernel and the device is {device}"
    if importlib.util.find_spec("triton") is None:
        return "Triton is not installed"
    return None


def select_implementation(requested: str | None, device: torch.device) -> str:
    """Resolve a kernel name, choosing one when the caller did not.

    An explicit request that cannot run is an error rather than a silent
    downgrade: a benchmark row or a parity test that asks for a kernel has to get
    that kernel, or hear why it could not.
    """
    if requested is None:
        blocked = unsupported_reason(PAGED_IMPLEMENTATION, device)
        return UNIVERSAL_IMPLEMENTATION if blocked else PAGED_IMPLEMENTATION

    blocked = unsupported_reason(requested, device)
    if blocked is not None:
        raise ValueError(
            f"attn_implementation={requested!r} cannot run here: {blocked}. "
            f"Leave it unset to choose automatically, or pass {UNIVERSAL_IMPLEMENTATION!r}."
        )
    return requested


def reads_pool_in_place(name: str) -> bool:
    """Whether this kernel reads the pool where it lies, bounded by device-side lengths.

    That is what a captured decode step needs — no host-side slicing inside the
    forward — and what the split count applies to. The dense kernels slice each
    sequence on the host instead.
    """
    return name == PAGED_IMPLEMENTATION
