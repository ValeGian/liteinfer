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
    sliding_window: int | None = None,
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
        sliding_window: When set, additionally masks key columns more
            than ``sliding_window - 1`` positions before each query.
            Because all batch rows share the same left-padding scheme,
            the column distance between query and key equals the true
            position distance — the constraint is therefore batch- and
            pad-independent.

    Returns:
        Tensor shaped ``[B, 1, query_len, past_len + query_len]``.
    """
    if not prompt_lens:
        raise ValueError("prompt_lens must be non-empty")
    max_prompt_len = max(prompt_lens)
    is_prefill = past_len == 0
    if is_prefill and query_len < max_prompt_len:
        raise ValueError(f"prefill query_len={query_len} cannot be smaller than max prompt length {max_prompt_len}")
    if sliding_window is not None and sliding_window < 1:
        raise ValueError(f"sliding_window must be >= 1, got {sliding_window}")

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

    if sliding_window is not None and sliding_window < key_len:
        # Mask key columns farther than (sliding_window - 1) before each query.
        # Query at column index (past_len + q) sees keys at columns
        # [past_len + q - sliding_window + 1, past_len + q].
        for q in range(query_len):
            cutoff = past_len + q - sliding_window + 1
            if cutoff > 0:
                mask[:, :, q, :cutoff] = neg_inf

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
    hf_config,
    prompt_lens: Sequence[int],
    query_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Per-model dispatch returning the right ``attention_mask`` payload.

    ``LlamaForCausalLM`` consumes a single additive tensor; Gemma4 consumes a
    ``{"full_attention", "sliding_attention"}`` dict because its layers mix
    full and sliding attention. New architectures register themselves by
    extending this dispatch.
    """
    if model_class_name == "LlamaForCausalLM":
        return build_additive_mask(
            prompt_lens=prompt_lens,
            query_len=query_len,
            past_len=past_len,
            dtype=dtype,
            device=device,
        )
    if model_class_name == "Gemma4ForCausalLM":
        text_config = getattr(hf_config, "text_config", hf_config)
        sliding_window = int(text_config.sliding_window)
        return build_gemma4_mask_dict(
            prompt_lens=prompt_lens,
            query_len=query_len,
            past_len=past_len,
            dtype=dtype,
            device=device,
            sliding_window=sliding_window,
        )
    raise NotImplementedError(f"no attention-mask builder registered for model class {model_class_name!r}")


def build_continuous_decode_mask(
    seq_total_lens: list[int],
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: int | None = None,
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
        sliding_window: when set, also masks key columns farther than
            ``sliding_window - 1`` positions before the current query.

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
        if sliding_window is not None:
            # Mask keys more than (sliding_window - 1) positions before the query.
            # The query is at logical position (total - 1); the earliest
            # attendable key is at logical position (total - sliding_window).
            # In the left-padded layout the earliest attendable column is
            # max_total - sliding_window.
            cutoff = max_total - sliding_window
            if cutoff > pad_len:
                mask[i, 0, 0, pad_len:cutoff] = neg_inf
    return mask


def build_continuous_decode_for_model(
    model_class_name: str,
    *,
    hf_config,
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
    if model_class_name == "Gemma4ForCausalLM":
        text_config = getattr(hf_config, "text_config", hf_config)
        sliding_window = int(text_config.sliding_window)
        full = build_continuous_decode_mask(
            seq_total_lens=seq_total_lens,
            dtype=dtype,
            device=device,
        )
        sliding = build_continuous_decode_mask(
            seq_total_lens=seq_total_lens,
            dtype=dtype,
            device=device,
            sliding_window=sliding_window,
        )
        return {"full_attention": full, "sliding_attention": sliding}
    raise NotImplementedError(f"no continuous-decode mask builder for model class {model_class_name!r}")


def build_gemma4_mask_dict(
    prompt_lens: Sequence[int],
    query_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: int,
) -> dict[str, torch.Tensor]:
    """Two-layer-type mask dict consumed by ``Gemma4TextModel.forward``.

    Returns a ``{"full_attention": ..., "sliding_attention": ...}`` map
    whose values are pad- and causal-aware additive masks. The full
    layers receive a standard causal+pad mask; sliding layers receive
    the same plus a sliding-window cutoff. The dict is passed verbatim
    via ``attention_mask=`` so the model skips its internal
    ``create_causal_mask`` / ``create_sliding_window_causal_mask``
    helpers, which expect a 2D ``[B, S]`` boolean and would otherwise
    drop our padding information.
    """
    full = build_additive_mask(
        prompt_lens=prompt_lens,
        query_len=query_len,
        past_len=past_len,
        dtype=dtype,
        device=device,
    )
    sliding = build_additive_mask(
        prompt_lens=prompt_lens,
        query_len=query_len,
        past_len=past_len,
        dtype=dtype,
        device=device,
        sliding_window=sliding_window,
    )
    return {"full_attention": full, "sliding_attention": sliding}
