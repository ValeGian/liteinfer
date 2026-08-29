# pyright: reportPrivateImportUsage=false
"""Additive attention mask builder for static batching with left-padded prefill.

Returns a mask of shape ``[B, 1, query_len, past_len + query_len]`` whose
entries are 0 for "attend" and ``finfo(dtype).min`` for "do not attend".
Combines two sources of masking:

* **Causal**: query token *q* may not attend to key token *k > q*.
* **Left padding**: when prompts in a batch have different lengths, the
  shorter ones are padded on the left during prefill. Padded positions
  are stored in the KV cache (their values are arbitrary). Real query
  rows must not attend to padded key columns; padded query rows are
  fully masked since they do not produce any sampled token.

The builder takes raw prompt lengths and reconstructs both effects in a
single tensor that the model can add to attention scores directly.
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


def build_prefill_for_model(
    model_class_name: str,
    *,
    prompt_lens: Sequence[int],
    dtype: torch.dtype,
    device: torch.device,
):
    """Per-model dispatch for the prefill mask.

    New architectures register themselves by extending this dispatch.
    """
    if model_class_name == "LlamaForCausalLM":
        return build_prefill_mask(prompt_lens=prompt_lens, dtype=dtype, device=device)
    raise NotImplementedError(
        f"no attention-mask builder registered for model class {model_class_name!r}"
    )


def build_continuous_decode_mask(
    seq_total_lens: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive attention mask for a continuous-batching decode step.

    Unlike ``build_additive_mask``, sequences in a continuous batch may have
    different total cached lengths (prompt + output tokens so far). The paged
    KV cache returns tensors LEFT-PADDED to ``max(seq_total_lens)``. This mask
    unmasks each sequence's real token positions (right side) and masks the
    left-pad zeros.

    Args:
        seq_total_lens: total token count per sequence (prompt + output so far
            PLUS the one new token being decoded). The length of this list
            equals the batch size.
        dtype: must match the attention score tensor.
        device: target device.

    Returns:
        Tensor shaped ``[B, 1, 1, max_total]``.
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


def build_continuous_decode_for_model(
    model_class_name: str,
    *,
    seq_total_lens: list[int],
    dtype: torch.dtype,
    device: torch.device,
):
    """Per-model dispatch for continuous-batching decode masks."""
    if model_class_name == "LlamaForCausalLM":
        return build_continuous_decode_mask(
            seq_total_lens=seq_total_lens,
            dtype=dtype,
            device=device,
        )
    raise NotImplementedError(
        f"no continuous-decode mask builder for model class {model_class_name!r}"
    )
