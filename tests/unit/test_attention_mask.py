# pyright: reportPrivateImportUsage=false
"""Unit tests for the attention-mask builders."""

from __future__ import annotations

import torch

from liteinfer.engine.attention_mask import (
    build_continuous_decode_mask,
    build_prefill_mask,
)


def _min(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).min)


def test_single_seq_no_padding_matches_pure_causal() -> None:
    """B=1 with no padding: the mask is the standard causal triangle."""
    mask = build_prefill_mask(
        prompt_lens=[3],
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
    mask = build_prefill_mask(
        prompt_lens=[2, 4],
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


def test_dtype_min_used_for_masked_positions() -> None:
    """bfloat16 mask uses bfloat16's finfo.min, not float32's."""
    mask = build_prefill_mask(
        prompt_lens=[2],
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    bf16_min = torch.finfo(torch.bfloat16).min
    # Upper-triangle entry in the causal mask must equal bf16_min.
    assert mask[0, 0, 0, 1].item() == bf16_min



def test_decode_mask_hides_exactly_the_pad_prefix_of_every_row() -> None:
    """Vectorised or not, each row must mask its own prefix and nothing else."""
    lens = [5, 3, 1]
    mask = build_continuous_decode_mask(lens, torch.float32, torch.device("cpu"))

    masked_per_row = (mask[:, 0, 0, :] == torch.finfo(torch.float32).min).sum(dim=1)
    assert masked_per_row.tolist() == [max(lens) - n for n in lens]


# ---------------------------------------------------------------------------
# A chunk continuing a cached prompt
# ---------------------------------------------------------------------------


def _chunk_mask() -> torch.Tensor:
    """Two queries at positions 3 and 4 of a 5-token context, beside a 2-token prompt."""
    return build_prefill_mask(
        prompt_lens=[2, 2],
        dtype=torch.float32,
        device=torch.device("cpu"),
        context_lens=[5, 2],
    )


def test_a_chunk_mask_has_a_column_per_key_the_sequence_holds() -> None:
    assert _chunk_mask().shape == (2, 1, 2, 5)


def test_a_chunk_s_first_query_sees_the_whole_cached_prefix_and_itself() -> None:
    assert _chunk_mask()[0, 0, 0].tolist() == [0.0, 0.0, 0.0, 0.0, _min(torch.float32)]


def test_a_chunk_s_last_query_sees_every_key() -> None:
    assert _chunk_mask()[0, 0, 1].tolist() == [0.0] * 5


def test_a_prompt_beside_a_chunk_masks_the_columns_it_does_not_hold() -> None:
    """Right-aligned: its two keys are the last two columns, and it stays causal among them."""
    neg = _min(torch.float32)
    assert _chunk_mask()[1, 0].tolist() == [[neg, neg, neg, 0.0, neg], [neg, neg, neg, 0.0, 0.0]]
