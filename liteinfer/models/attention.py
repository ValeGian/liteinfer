# pyright: reportPrivateImportUsage=false
"""Attention kernels. One entry per `EngineConfig.attn_implementation`.

Both kernels compute the same thing and differ only in what they keep in
memory. `eager` writes the full `[batch, heads, queries, keys]` score matrix
out and reads it back; `sdpa` hands the whole operation to PyTorch, which
tiles it and never materialises that matrix. The scores are the largest
tensor in the forward pass — at 32 sequences x 32 heads x 1024 queries x 1024
keys the fp32 softmax alone is 4 GiB — so which kernel runs decides what
prompt length the engine can serve, not just how fast it serves it.

Causality is not passed as a flag: left-padded batches need an explicit mask
anyway (see `engine/attention_mask.py`), and that mask already carries it.

The kernels agree everywhere the engine reads. They disagree on fully masked
query rows — the left padding — where sdpa returns zeros and eager returns the
average of every value vector, because the mask is `finfo.min` rather than
`-inf` and a softmax over equal scores is uniform. A row that attends to nothing
has no defined answer; the engine takes logits from the last column, which is
never padding, so neither answer reaches the output.
"""

from __future__ import annotations

import torch
from torch import nn


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
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_kv_groups: int,
) -> torch.Tensor:
    """Attention written out in matmuls. Readable, and the memory ceiling.

    Kept because it is what the arithmetic looks like — every step of scaled
    dot-product attention is a line here — and because it is the reference the
    fused kernels are checked against.
    """
    key = _repeat_kv(key, num_kv_groups)
    value = _repeat_kv(value, num_kv_groups)

    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]
    weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value)


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
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
    key = _repeat_kv(key, num_kv_groups)
    value = _repeat_kv(value, num_kv_groups)

    mask = attention_mask[:, :, :, : key.shape[-2]] if attention_mask is not None else None
    return nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling)


IMPLEMENTATIONS = {"eager": eager_attention, "sdpa": sdpa_attention}
DEFAULT_IMPLEMENTATION = "sdpa"


def resolve(name: str | None):
    """Look up a kernel by name. `None` means the caller stated no preference."""
    name = name or DEFAULT_IMPLEMENTATION
    if name not in IMPLEMENTATIONS:
        raise ValueError(
            f"unknown attn_implementation {name!r}; known: {sorted(IMPLEMENTATIONS)}"
        )
    return IMPLEMENTATIONS[name]
