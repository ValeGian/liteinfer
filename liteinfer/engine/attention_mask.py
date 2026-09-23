# pyright: reportPrivateImportUsage=false
"""Additive attention masks: 0 to attend, ``finfo(dtype).min`` to not.

Prompts of different lengths are left-padded, so every mask has to hide the
padded columns as well as enforce causality. Prefill and decode need different
shapes, so they have separate builders.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


def build_prefill_mask(
    prompt_lens: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
    context_lens: Sequence[int] | None = None,
) -> torch.Tensor:
    """Causal mask for one prefill pass over a left-padded batch.

    `prompt_lens` are the tokens each sequence brings to this pass — its queries.
    `context_lens` are the keys it attends to: the same by default, and more when
    a chunk continues a prompt whose start is already cached, in which case the
    chunk's queries are the *last* `prompt_len` of its `context_len` positions.

    Queries and keys are both right-aligned, so query column `j` and key column
    `k` of any row are separated by the same `max_context - max_query` columns
    their logical positions are: causality is one diagonal for the whole batch,
    offset by that amount. Each row then hides its own left padding — every
    padded query row entirely, and padded key columns from its real ones.

    Returns ``[B, 1, max(prompt_lens), max(context_lens)]`` additive mask.
    """
    context_lens = prompt_lens if context_lens is None else context_lens
    batch = len(prompt_lens)
    max_query, max_context = max(prompt_lens), max(context_lens)

    # One transfer for the lengths, then every rule is a broadcast comparison:
    # a slice assignment per sequence would be a kernel launch each.
    lengths = torch.tensor([list(prompt_lens), list(context_lens)], device=device)
    query_col = torch.arange(max_query, device=device).view(1, max_query, 1)
    key_col = torch.arange(max_context, device=device).view(1, 1, max_context)

    is_future = key_col > query_col + (max_context - max_query)
    is_pad_query = query_col < (max_query - lengths[0]).view(batch, 1, 1)
    is_pad_key = key_col < (max_context - lengths[1]).view(batch, 1, 1)

    # Filled one rule at a time, each broadcast into the mask, rather than OR-ing
    # them first: the union is a `[B, queries, keys]` boolean, 512 MiB per copy at
    # 32 x 4,096 — and this mask is built inside the profile that sizes the pool.
    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, max_query, max_context), dtype=dtype, device=device)
    for is_hidden in (is_future, is_pad_query, is_pad_key):
        mask.masked_fill_(is_hidden.unsqueeze(1), neg_inf)
    return mask


def build_continuous_decode_mask(
    seq_total_lens: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive attention mask for a continuous-batching decode step.

    Sequences in a continuous batch hold different numbers of cached tokens, and
    the cache returns them left-padded to ``max(seq_total_lens)``, so each row
    masks its own pad prefix. ``seq_total_lens`` counts prompt + output so far,
    including the token being decoded. Returns ``[B, 1, 1, max_total]``.

    That is a prefill chunk of one query over the sequence's whole history, so
    it is built by that rule.
    """
    if not seq_total_lens:
        raise ValueError("seq_total_lens must be non-empty")
    return build_prefill_mask([1] * len(seq_total_lens), dtype, device, seq_total_lens)


_BUILDERS = {"LlamaForCausalLM": (build_prefill_mask, build_continuous_decode_mask)}


def builders_for(model_class_name: str):
    """Return this architecture's ``(prefill, decode)`` mask builders."""
    if model_class_name not in _BUILDERS:
        raise NotImplementedError(f"no attention-mask builders for {model_class_name!r}")
    return _BUILDERS[model_class_name]
