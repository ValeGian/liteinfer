# pyright: reportPrivateImportUsage=false
"""Attention kernels. One entry per `EngineConfig.attn_implementation`.

The three entries differ in what they keep in memory and in where they read
K and V from. `EngineConfig.attn_implementation=None` — the default — means
"the fastest one that runs here", which `select_implementation` resolves once
per engine from the device. Naming one asks for it specifically, and is refused
rather than downgraded when it cannot run.

`eager` writes the full `[batch, heads, queries, keys]` score matrix out and
reads it back; `sdpa` hands the whole operation to PyTorch, which tiles it and
never materialises that matrix. The scores are the largest tensor in the
forward pass — at 32 sequences x 32 heads x 1024 queries x 1024 keys the fp32
softmax alone is 4 GiB — so which kernel runs decides what prompt length the
engine can serve, not just how fast it serves it.

Both of those want K and V as one contiguous tensor, which for a paged cache
means copying every sequence's history out of the pool before every decode
step. `paged` takes the slot table instead and reads the pool in place: every
decode step, and every packed prefill chunk that continues a cached prompt. A
prefill with nothing cached before it goes to `varlen` (packed) or `sdpa`
(padded), because its keys are the tensors the pass has just computed and
nothing is paged yet. See `models/paged_decode.py`.

Causality is not passed as a flag to the dense kernels: left-padded batches
need an explicit mask anyway (see `engine/attention_mask.py`), and that mask
already carries it. The paged kernel needs neither — a decode query attends to
its sequence's whole history, and `context_lens` says where that history ends.

The kernels agree everywhere the engine reads. `eager` and `sdpa` disagree on
fully masked query rows — the left padding — where sdpa returns zeros and eager
returns the average of every value vector, because the mask is `finfo.min`
rather than `-inf` and a softmax over equal scores is uniform. A row that
attends to nothing has no defined answer; the engine takes logits from the last
column, which is never padding, so neither answer reaches the output.
"""

from __future__ import annotations

import importlib.util
from typing import NamedTuple

import torch
from torch import nn


class DenseKV(NamedTuple):
    """K and V as contiguous tensors, ``[batch, kv_heads, keys, head_dim]``."""

    keys: torch.Tensor
    values: torch.Tensor


class PagedKV(NamedTuple):
    """K and V left in the pool, plus the addresses a kernel needs to find them."""

    key_pool: torch.Tensor
    """This layer's flat key store, ``[num_slots, kv_heads, head_dim]``."""
    value_pool: torch.Tensor
    slot_table: torch.Tensor
    """``[batch, max_context]`` physical slot per logical position, right-aligned."""
    context_lens: torch.Tensor
    """``[batch]`` real cached tokens per sequence — where each row's history ends."""
    num_splits: int | None = None
    """How many programs share each sequence's key loop; None lets the kernel choose."""
    query_start_loc: torch.Tensor | None = None
    """``[batch + 1]`` int32 prefix sums of each sequence's queries, for a pass that
    brings more than one per sequence — a chunk continuing a cached prompt. None
    is a decode step: one query per sequence, laid out one per batch row."""
    max_query_len: int = 1
    """Most queries any one sequence brings, which sizes the kernel's grid."""


class VarlenKV(NamedTuple):
    """K/V for a prefill batch packed end to end, with the sequence boundaries.

    The dense payloads hand back `[batch, heads, keys, dim]` and rely on padding
    plus a mask to keep one sequence's queries away from another's keys. A packed
    batch has no padding to mask: it is one flat run of tokens, and `cu_seqlens`
    is where each sequence starts.
    """

    keys: torch.Tensor
    """`[1, kv_heads, total_tokens, head_dim]` — the prompt K this pass computed."""
    values: torch.Tensor
    """The matching V."""
    cu_seqlens: torch.Tensor
    """`[num_sequences + 1]` int32 prefix sums: sequence `i` owns `[cu[i], cu[i + 1])`."""
    max_seqlen: int
    """Longest sequence in the batch, which the kernel needs as a host-side int."""


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


def _reject_packed(kv, kernel: str) -> None:
    """A packed batch carries its boundaries in `cu_seqlens`, which only `varlen` reads.

    `VarlenKV` holds fields named `keys` and `values` like the dense payloads do,
    so a kernel that does not check would attend straight across the boundary
    between two prompts — and, with no mask to stop it, across a prompt's own
    future. That is a wrong answer rather than a crash, which is why it is
    checked rather than assumed.
    """
    if isinstance(kv, VarlenKV):
        raise ValueError(
            f"{kernel} cannot read a packed batch: it has no way to honour cu_seqlens. "
            "Packed prefill requires the paged implementation."
        )


def eager_attention(
    query: torch.Tensor,
    kv: DenseKV,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    """Attention written out in matmuls. Readable, and the memory ceiling.

    Kept because it is what the arithmetic looks like — every step of scaled
    dot-product attention is a line here — and because it is the reference the
    fused kernels are checked against.
    """
    _reject_packed(kv, "eager")
    key = _repeat_kv(kv.keys, num_kv_groups)
    value = _repeat_kv(kv.values, num_kv_groups)

    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]
    weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value)


def sdpa_attention(
    query: torch.Tensor,
    kv: DenseKV,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    """The same attention through `torch.nn.functional.scaled_dot_product_attention`.

    PyTorch dispatches to a fused backend — FlashAttention or memory-efficient
    attention on CUDA — which computes the softmax in tiles that stay in SRAM.
    An additive float mask rules FlashAttention out (it takes only `is_causal`),
    so left-padded batches land on the memory-efficient backend; both avoid the
    score matrix, which is the property that matters here.
    """
    _reject_packed(kv, "sdpa")
    key = _repeat_kv(kv.keys, num_kv_groups)
    value = _repeat_kv(kv.values, num_kv_groups)

    mask = attention_mask[:, :, :, : key.shape[-2]] if attention_mask is not None else None
    return nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling)


def varlen_attention(
    query: torch.Tensor,
    kv: VarlenKV,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    """Attention over a packed prefill batch, through FlashAttention's varlen entry.

    Two things fall away against `sdpa_attention` on the same batch. There is no
    mask, because there is no padding to hide — `cu_seqlens` says where each
    sequence ends, so `is_causal` is the whole rule. And there is no `_repeat_kv`:
    flash reads one KV head per query group directly, where the dense path
    materialises a copy of K and V per query head first.

    The entry point is `torch.ops.aten._flash_attention_forward`, which is what
    PyTorch's own SDPA calls once it has decided flash applies. Going through it
    directly is what lets the call carry `cu_seqlens`; SDPA's public signature has
    nowhere to put them. It is a private op, so `varlen_unsupported_reason` states
    its preconditions and `tests/unit/test_varlen_attention.py` pins its answer
    against `eager`.
    """
    if attention_mask is not None:
        raise ValueError("packed attention takes no mask; cu_seqlens bounds each sequence")

    # `[1, heads, tokens, dim]` is the layout the layers speak; flash wants the
    # token axis first. The batch axis is 1 by construction — a packed batch is
    # one run of tokens, and `cu_seqlens` carries what the batch axis used to.
    packed_query = query.squeeze(0).transpose(0, 1)
    packed_key = kv.keys.squeeze(0).transpose(0, 1)
    packed_value = kv.values.squeeze(0).transpose(0, 1)

    attn_output, *_ = torch.ops.aten._flash_attention_forward(
        packed_query,
        packed_key,
        packed_value,
        kv.cu_seqlens,
        kv.cu_seqlens,
        kv.max_seqlen,
        kv.max_seqlen,
        dropout_p=0.0,
        is_causal=True,
        return_debug_mask=False,
        scale=scaling,
    )
    return attn_output.transpose(0, 1).unsqueeze(0)


def paged_attention(
    query: torch.Tensor,
    kv: DenseKV | VarlenKV | PagedKV,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    """Attend straight out of the KV pool; a whole-prompt prefill through `sdpa` or `varlen`.

    Which happens is decided by what the cache handed over, not by a flag. A
    prefill with nothing cached before it returns the K and V the pass just
    computed — packed if the batch was packed, padded if it was padded — and
    there is nothing paged about them yet. Everything else reads the pool: a
    decode step one query per sequence, and a packed chunk continuing a cached
    prompt many, through the same kernel.
    """
    if isinstance(kv, DenseKV):
        return sdpa_attention(query, kv, attention_mask, scaling, num_kv_groups)
    if isinstance(kv, VarlenKV):
        return varlen_attention(query, kv, attention_mask, scaling, num_kv_groups)
    if attention_mask is not None:
        raise ValueError("paged attention takes no mask; context_lens bounds each sequence")
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
"""Fastest decode, and the preferred choice wherever its preconditions hold."""

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


def varlen_unsupported_reason(
    implementation: str, device: torch.device, dtype: torch.dtype
) -> str | None:
    """Why a prefill batch cannot be packed here, or `None` if it can.

    Two of the three preconditions belong to the flash kernel underneath rather
    than to packing: it is CUDA-only, and it computes in half precision. The
    third is the engine's own: a packed batch may only be handed to a kernel
    that reads `cu_seqlens`. Where any of them fails the engine pads the batch
    and masks it, as it did everywhere before.
    """
    if not handles_packed_prefill(implementation):
        return f"the {implementation} kernel reads a padded batch and a mask"
    if device.type != "cuda":
        return f"FlashAttention is CUDA-only and the device is {device}"
    if dtype not in (torch.float16, torch.bfloat16):
        return f"FlashAttention computes in half precision and the dtype is {dtype}"
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


def handles_packed_prefill(name: str) -> bool:
    """Whether this kernel can attend to a prefill batch that carries `cu_seqlens`.

    Only the paged implementation dispatches on the payload type, so only it
    reaches `varlen_attention`. The dense kernels want a padded batch and a mask,
    which is what the engine keeps building wherever this is False.
    """
    return name == PAGED_IMPLEMENTATION


def reads_paged_kv(name: str) -> bool:
    """Whether this kernel reads the pool directly, so decode must page rather than gather."""
    return name == PAGED_IMPLEMENTATION
