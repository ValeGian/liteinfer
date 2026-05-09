# pyright: reportPrivateImportUsage=false
"""Unit tests for the additive attention mask builder used by static batching."""

from __future__ import annotations

import pytest
import torch

from liteinfer.engine.attention_mask import build_additive_mask


def _min(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).min)


def test_single_seq_no_padding_matches_pure_causal() -> None:
    """B=1 with prompt_len == query_len: mask is the standard causal triangle."""
    mask = build_additive_mask(
        prompt_lens=[3],
        query_len=3,
        past_len=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, 3, 3)
    expected = torch.tensor(
        [
            [0.0, _min(torch.float32), _min(torch.float32)],
            [0.0, 0.0, _min(torch.float32)],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(mask[0, 0], expected)


def test_left_padded_prefill_masks_pad_columns_and_rows() -> None:
    """Prefill with left padding: rows < pad are entirely masked out as queries
    cannot attend to anything; padded key columns are masked from real query rows."""
    mask = build_additive_mask(
        prompt_lens=[2, 4],
        query_len=4,
        past_len=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (2, 1, 4, 4)

    # Sequence 0 has prompt_len=2 → 2 padded positions on the left (cols 0, 1).
    # Real query rows (rows 2, 3) must NOT attend to padded cols (0, 1).
    seq0 = mask[0, 0]
    assert seq0[2, 0].item() == _min(torch.float32)
    assert seq0[2, 1].item() == _min(torch.float32)
    assert seq0[2, 2].item() == 0.0
    assert seq0[3, 0].item() == _min(torch.float32)
    assert seq0[3, 1].item() == _min(torch.float32)
    assert seq0[3, 2].item() == 0.0
    assert seq0[3, 3].item() == 0.0
    # Real query rows respect causality among real positions:
    assert seq0[2, 3].item() == _min(torch.float32)

    # Sequence 1 has no padding, identical to a pure causal triangle.
    seq1 = mask[1, 0]
    expected_seq1 = torch.tensor(
        [
            [0.0, _min(torch.float32), _min(torch.float32), _min(torch.float32)],
            [0.0, 0.0, _min(torch.float32), _min(torch.float32)],
            [0.0, 0.0, 0.0, _min(torch.float32)],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(seq1, expected_seq1)


def test_decode_step_masks_padded_history() -> None:
    """Decode step: query_len=1, past_len=max_prompt_len. Each row must mask the
    padded prefix of its own KV history but not the real prompt+output positions."""
    # Two sequences: prompt_lens 2 and 4, batched at max 4. After prefill the
    # cache holds 4 columns per row. After 1 decode step, key length = 5 and
    # we are sampling token at position 4 (zero-indexed) for the longest seq,
    # at position 2 for the shorter one.
    mask = build_additive_mask(
        prompt_lens=[2, 4],
        query_len=1,
        past_len=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.shape == (2, 1, 1, 5)

    seq0 = mask[0, 0, 0]
    # Cols 0, 1 are padding from prefill; col 2 is the real prompt start
    # because prompt_len=2 occupies columns 2 and 3 after left padding.
    assert seq0[0].item() == _min(torch.float32)
    assert seq0[1].item() == _min(torch.float32)
    assert seq0[2].item() == 0.0
    assert seq0[3].item() == 0.0
    assert seq0[4].item() == 0.0  # the new decode token

    seq1 = mask[1, 0, 0]
    # No padding for the longest sequence.
    for c in range(5):
        assert seq1[c].item() == 0.0


def test_dtype_min_used_for_masked_positions() -> None:
    """bfloat16 mask uses bfloat16's finfo.min, not float32's."""
    mask = build_additive_mask(
        prompt_lens=[2],
        query_len=2,
        past_len=0,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    bf16_min = torch.finfo(torch.bfloat16).min
    # Upper-triangle entry in the causal mask must equal bf16_min.
    assert mask[0, 0, 0, 1].item() == bf16_min


def test_rejects_query_len_smaller_than_max_prompt_in_prefill() -> None:
    """Prefill semantics: query_len at least covers each prompt's tokens."""
    with pytest.raises(ValueError):
        build_additive_mask(
            prompt_lens=[3, 5],
            query_len=4,  # < max(prompt_lens) for prefill (past_len=0)
            past_len=0,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
