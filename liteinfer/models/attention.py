# pyright: reportPrivateImportUsage=false
"""Attention kernels. One entry per `EngineConfig.attn_implementation`.

The three entries differ in what they keep in memory and in where they read
K and V from. `EngineConfig.attn_implementation=None` — the default — means
"the fastest one that runs here", which `select_implementation` resolves once
per engine from the device and the model's head dimension. Naming one asks for
it specifically, and is refused rather than downgraded when it cannot run.

`eager` writes the full `[batch, heads, queries, keys]` score matrix out and
reads it back; `sdpa` hands the whole operation to PyTorch, which tiles it and
never materialises that matrix. The scores are the largest tensor in the
forward pass — at 32 sequences x 32 heads x 1024 queries x 1024 keys the fp32
softmax alone is 4 GiB — so which kernel runs decides what prompt length the
engine can serve, not just how fast it serves it.

Both of those want K and V as one contiguous tensor, which for a paged cache
means copying every sequence's history out of the pool before every decode
step. `paged` takes the slot table instead and reads the pool in place; it is
decode-only, and falls back to `sdpa` for prefill, where the keys are the
tensors the pass has just computed and nothing is paged yet. See
`models/paged_decode.py`.

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
    key = _repeat_kv(kv.keys, num_kv_groups)
    value = _repeat_kv(kv.values, num_kv_groups)

    mask = attention_mask[:, :, :, : key.shape[-2]] if attention_mask is not None else None
    return nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling)


def paged_attention(
    query: torch.Tensor,
    kv: DenseKV | PagedKV,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    """Decode straight out of the KV pool; prefill through `sdpa`.

    Which of the two happens is decided by what the cache handed over, not by a
    flag: a prefill payload returns the K and V the pass just computed, and
    there is nothing paged about them yet.
    """
    if isinstance(kv, DenseKV):
        return sdpa_attention(query, kv, attention_mask, scaling, num_kv_groups)

    if attention_mask is not None:
        raise ValueError("paged decode takes no mask; context_lens bounds each sequence")
    if query.shape[2] != 1:
        raise ValueError(f"paged decode takes one query per sequence, got {query.shape[2]}")

    # Imported here rather than at module scope: the kernel needs Triton, which
    # torch's CPU-only builds do not ship, and the other two kernels must keep
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
    )
    return attn_output.unsqueeze(2)


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


def validate_name(name: str | None) -> None:
    """Reject an unknown kernel name at config time. `None` means "choose for me"."""
    if name is not None:
        resolve(name)


def unsupported_reason(name: str, device: torch.device, head_dim: int) -> str | None:
    """Why this kernel cannot serve this device and model, or `None` if it can.

    Only the paged kernel has preconditions: it is Triton, so it needs CUDA, the
    package installed, and a power-of-two head dimension (see
    `models/paged_decode.supports_head_dim`). The Triton check comes before that
    import for a reason — on a CPU-only install the module cannot be imported at
    all.
    """
    if name != PAGED_IMPLEMENTATION:
        return None
    if device.type != "cuda":
        return f"it is a CUDA kernel and the device is {device}"
    if importlib.util.find_spec("triton") is None:
        return "Triton is not installed"

    from liteinfer.models.paged_decode import supports_head_dim

    if not supports_head_dim(head_dim):
        return f"it tiles the head dimension in powers of two and this model's is {head_dim}"
    return None


def select_implementation(requested: str | None, device: torch.device, head_dim: int) -> str:
    """Resolve a kernel name, choosing one when the caller did not.

    An explicit request that cannot run is an error rather than a silent
    downgrade: a benchmark row or a parity test that asks for a kernel has to get
    that kernel, or hear why it could not.
    """
    if requested is None:
        blocked = unsupported_reason(PAGED_IMPLEMENTATION, device, head_dim)
        return UNIVERSAL_IMPLEMENTATION if blocked else PAGED_IMPLEMENTATION

    blocked = unsupported_reason(requested, device, head_dim)
    if blocked is not None:
        raise ValueError(
            f"attn_implementation={requested!r} cannot run here: {blocked}. "
            f"Leave it unset to choose automatically, or pass {UNIVERSAL_IMPLEMENTATION!r}."
        )
    return requested


def reads_paged_kv(name: str) -> bool:
    """Whether this kernel reads the pool directly, so decode must page rather than gather."""
    return name == PAGED_IMPLEMENTATION
