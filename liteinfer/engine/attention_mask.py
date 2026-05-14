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


def build_additive_mask(
    prompt_lens: Sequence[int],
    query_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the additive attention mask for a static batch.

    Args:
        prompt_lens: True (unpadded) prompt length per batch row.
        query_len: Number of query tokens in this forward pass.
            Equals max(prompt_lens) for prefill (with no past) and
            ``1`` for a single decode step.
        past_len: Number of key/value columns already cached. Equals
            ``0`` for prefill, ``max(prompt_lens) + decode_step`` for
            decode steps.
        dtype: Mask dtype; must match the attention score tensor's dtype.
        device: Mask device.

    Returns:
        Tensor shaped ``[B, 1, query_len, past_len + query_len]``.
    """
    if not prompt_lens:
        raise ValueError("prompt_lens must be non-empty")
    max_prompt_len = max(prompt_lens)
    is_prefill = past_len == 0
    if is_prefill and query_len < max_prompt_len:
        raise ValueError(f"prefill query_len={query_len} cannot be smaller than max prompt length {max_prompt_len}")

    batch_size = len(prompt_lens)
    key_len = past_len + query_len
    neg_inf = torch.finfo(dtype).min

    mask = torch.zeros((batch_size, 1, query_len, key_len), dtype=dtype, device=device)

    if query_len > 1:
        causal = torch.triu(
            torch.full((query_len, query_len), neg_inf, dtype=dtype, device=device),
            diagonal=1,
        )
        mask[:, :, :, past_len:] = causal[None, None]

    if is_prefill:
        # Left padding lives in the first (max_prompt_len - prompt_len_i) columns
        # of each row. Padded query rows mirror the same columns; padded key
        # columns are masked from real query rows.
        for i, pl in enumerate(prompt_lens):
            pad = max_prompt_len - pl
            if pad == 0:
                continue
            # Padded query rows: rows [0, pad) are pure padding.
            mask[i, 0, :pad, :] = neg_inf
            # Real query rows must not see padded key columns.
            mask[i, 0, pad:, :pad] = neg_inf
    else:
        # Decode step: cache layout matches prefill — padded prefix lives in
        # the first (max_prompt_len - prompt_len_i) columns of past_len.
        for i, pl in enumerate(prompt_lens):
            pad = max_prompt_len - pl
            if pad == 0:
                continue
            mask[i, 0, :, :pad] = neg_inf

    return mask


def build_for_model(
    model_class_name: str,
    *,
    prompt_lens: Sequence[int],
    query_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Per-model dispatch returning the right ``attention_mask`` payload.

    New architectures register themselves by extending this dispatch.
    """
    if model_class_name == "LlamaForCausalLM":
        return build_additive_mask(
            prompt_lens=prompt_lens,
            query_len=query_len,
            past_len=past_len,
            dtype=dtype,
            device=device,
        )
    raise NotImplementedError(f"no attention-mask builder registered for model class {model_class_name!r}")


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
    raise NotImplementedError(f"no continuous-decode mask builder for model class {model_class_name!r}")
