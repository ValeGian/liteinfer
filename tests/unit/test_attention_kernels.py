# pyright: reportPrivateImportUsage=false
"""The two kernels must compute the same attention, and sdpa must be the cheaper one."""

from __future__ import annotations

import pytest
import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.attention_mask import build_continuous_decode_mask, build_prefill_mask
from liteinfer.models.attention import IMPLEMENTATIONS, eager_attention, sdpa_attention

BATCH, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 2, 8, 2, 16
KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
DTYPE = torch.float32


def _qkv(num_queries: int, num_keys: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)

    def randn(heads: int, length: int) -> torch.Tensor:
        return torch.randn(BATCH, heads, length, HEAD_DIM, generator=generator, dtype=DTYPE)

    return randn(NUM_HEADS, num_queries), randn(NUM_KV_HEADS, num_keys), randn(NUM_KV_HEADS, num_keys)


def test_kernels_agree_on_a_prefill_pass():
    query, key, value = _qkv(num_queries=6, num_keys=6)
    mask = build_prefill_mask([6, 4], DTYPE, torch.device("cpu"))
    args = (query, key, value, mask, HEAD_DIM**-0.5, KV_GROUPS)

    torch.testing.assert_close(sdpa_attention(*args), eager_attention(*args))


def test_kernels_agree_on_a_decode_pass():
    query, key, value = _qkv(num_queries=1, num_keys=6)
    mask = build_continuous_decode_mask([6, 4], DTYPE, torch.device("cpu"))
    args = (query, key, value, mask, HEAD_DIM**-0.5, KV_GROUPS)

    torch.testing.assert_close(sdpa_attention(*args), eager_attention(*args))


def test_kernels_agree_without_a_mask():
    query, key, value = _qkv(num_queries=6, num_keys=6)
    args = (query, key, value, None, HEAD_DIM**-0.5, KV_GROUPS)

    torch.testing.assert_close(sdpa_attention(*args), eager_attention(*args))


def test_padded_query_rows_are_finite():
    """A fully masked row has no defined value; it must still not be a NaN.

    The two kernels answer it differently — sdpa returns zeros, eager returns
    the average of every value vector, because the mask is `finfo.min` rather
    than `-inf` and a softmax over equal scores is uniform. Neither is more
    correct, and the engine reads only the last column, which is never padding.
    A NaN would be a real problem: it would poison the residual stream.
    """
    query, key, value = _qkv(num_queries=6, num_keys=6)
    mask = build_prefill_mask([6, 2], DTYPE, torch.device("cpu"))

    padded_rows = sdpa_attention(query, key, value, mask, HEAD_DIM**-0.5, KV_GROUPS)[1, :, :4]

    assert padded_rows.isfinite().all()


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_every_kernel_name_is_an_accepted_config_value(name):
    assert EngineConfig(model="stub", attn_implementation=name).attn_implementation == name


def test_unknown_kernel_is_rejected_at_config_time():
    with pytest.raises(ValueError, match="unknown attn_implementation"):
        EngineConfig(model="stub", attn_implementation="flash")


@pytest.mark.gpu
def test_kernels_agree_in_bfloat16_to_within_rounding():
    """bf16 is the engine's working precision, and the kernels sum in different orders.

    Compared on the real query rows only: a padded row attends to nothing, and
    the two kernels disagree there by construction (see the test below).
    """
    query, key, value = (t.to(torch.bfloat16).cuda() for t in _qkv(num_queries=6, num_keys=6))
    mask = build_prefill_mask([6, 4], torch.bfloat16, torch.device("cuda"))
    args = (query, key, value, mask, HEAD_DIM**-0.5, KV_GROUPS)
    real_rows = (slice(None), slice(None), slice(2, None))

    torch.testing.assert_close(
        sdpa_attention(*args)[real_rows], eager_attention(*args)[real_rows], rtol=0, atol=2**-6
    )


@pytest.mark.gpu
def test_sdpa_does_not_materialise_the_score_matrix():
    """The point of the kernel: peak memory stops scaling with queries x keys.

    At these shapes the eager score matrix is 512 MiB in bf16 and 1 GiB again
    once softmax upcasts it, so a kernel that keeps it in SRAM shows up as a
    peak-memory difference far larger than the inputs themselves.
    """
    device = torch.device("cuda")
    seq_len = 2048
    query = torch.randn(4, 32, seq_len, 64, dtype=torch.bfloat16, device=device)
    key = value = torch.randn(4, 8, seq_len, 64, dtype=torch.bfloat16, device=device)
    mask = build_continuous_decode_mask([seq_len] * 4, torch.bfloat16, device).expand(-1, -1, seq_len, -1)

    def peak_bytes(kernel) -> int:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.max_memory_allocated(device)
        kernel(query, key, value, mask, 64**-0.5, 4)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) - before

    assert peak_bytes(sdpa_attention) < peak_bytes(eager_attention) / 2
