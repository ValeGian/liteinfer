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
) -> torch.Tensor:
    """Causal mask for one prefill pass over a left-padded batch.

    Prompts shorter than the longest are padded on the left, so each row's
    first ``max_prompt_len - prompt_len`` columns are padding: padded query
    rows are masked out entirely, and real query rows must not attend to
    padded key columns.

    Returns ``[B, 1, max_prompt_len, max_prompt_len]`` additive mask.
    """
    neg_inf = torch.finfo(dtype).min
    max_prompt_len = max(prompt_lens)
    mask = (
        torch.triu(
            torch.full((max_prompt_len, max_prompt_len), neg_inf, dtype=dtype, device=device),
            diagonal=1,
        )
        .expand(len(prompt_lens), 1, -1, -1)
        .clone()
    )

    for i, prompt_len in enumerate(prompt_lens):
        pad = max_prompt_len - prompt_len
        if pad:
            mask[i, 0, :pad, :] = neg_inf
            mask[i, 0, pad:, :pad] = neg_inf
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
    """
    if not seq_total_lens:
        raise ValueError("seq_total_lens must be non-empty")
    max_total = max(seq_total_lens)
    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros((len(seq_total_lens), 1, 1, max_total), dtype=dtype, device=device)
    for i, total in enumerate(seq_total_lens):
        pad_len = max_total - total
        if pad_len > 0:
            mask[i, 0, 0, :pad_len] = neg_inf
    return mask


_BUILDERS = {"LlamaForCausalLM": (build_prefill_mask, build_continuous_decode_mask)}


def builders_for(model_class_name: str):
    """Return this architecture's ``(prefill, decode)`` mask builders."""
    if model_class_name not in _BUILDERS:
        raise NotImplementedError(f"no attention-mask builders for {model_class_name!r}")
    return _BUILDERS[model_class_name]
