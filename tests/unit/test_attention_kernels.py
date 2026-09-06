# pyright: reportPrivateImportUsage=false
"""The dense kernels must compute the same attention, and sdpa must be the cheaper one."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from liteinfer.config import EngineConfig
from liteinfer.engine.attention_mask import build_continuous_decode_mask, build_prefill_mask
from liteinfer.models.attention import (
    IMPLEMENTATIONS,
    DenseKV,
    PagedKV,
    eager_attention,
    paged_attention,
    sdpa_attention,
    select_implementation,
    unsupported_reason,
)

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
    args = (query, DenseKV(key, value), mask, HEAD_DIM**-0.5, KV_GROUPS)

    torch.testing.assert_close(sdpa_attention(*args), eager_attention(*args))


def test_kernels_agree_on_a_decode_pass():
    query, key, value = _qkv(num_queries=1, num_keys=6)
    mask = build_continuous_decode_mask([6, 4], DTYPE, torch.device("cpu"))
    args = (query, DenseKV(key, value), mask, HEAD_DIM**-0.5, KV_GROUPS)

    torch.testing.assert_close(sdpa_attention(*args), eager_attention(*args))


def test_kernels_agree_without_a_mask():
    query, key, value = _qkv(num_queries=6, num_keys=6)
    args = (query, DenseKV(key, value), None, HEAD_DIM**-0.5, KV_GROUPS)

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

    padded_rows = sdpa_attention(query, DenseKV(key, value), mask, HEAD_DIM**-0.5, KV_GROUPS)[1, :, :4]

    assert padded_rows.isfinite().all()


# ---------------------------------------------------------------------------
# The paged entry: dense for prefill, and a decode contract it enforces
# ---------------------------------------------------------------------------


def _paged_kv() -> PagedKV:
    """Addresses that no kernel will get as far as reading — the guards fire first."""
    pool = torch.zeros(4, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
    return PagedKV(pool, pool, torch.zeros(BATCH, 2, dtype=torch.long), torch.ones(BATCH))


def test_paged_hands_a_prefill_pass_to_sdpa():
    """Prefill K/V are the tensors the pass just computed; there is nothing paged to read."""
    query, key, value = _qkv(num_queries=6, num_keys=6)
    mask = build_prefill_mask([6, 4], DTYPE, torch.device("cpu"))
    args = (query, DenseKV(key, value), mask, HEAD_DIM**-0.5, KV_GROUPS)

    torch.testing.assert_close(paged_attention(*args), sdpa_attention(*args))


def test_paged_decode_rejects_a_mask():
    """`context_lens` already bounds each sequence, so a mask would be a second answer."""
    query, _, _ = _qkv(num_queries=1, num_keys=6)
    mask = build_continuous_decode_mask([6, 4], DTYPE, torch.device("cpu"))

    with pytest.raises(ValueError, match="takes no mask"):
        paged_attention(query, _paged_kv(), mask, HEAD_DIM**-0.5, KV_GROUPS)


def test_paged_decode_rejects_more_than_one_query_per_sequence():
    """The kernel reads the whole history uncausally, which only holds for one query."""
    query, _, _ = _qkv(num_queries=2, num_keys=6)

    with pytest.raises(ValueError, match="one query per sequence"):
        paged_attention(query, _paged_kv(), None, HEAD_DIM**-0.5, KV_GROUPS)


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_every_kernel_name_is_an_accepted_config_value(name):
    assert EngineConfig(model="stub", attn_implementation=name).attn_implementation == name


def test_no_kernel_named_is_an_accepted_config_value():
    """`None` is "choose for me", resolved at load once the device is known."""
    assert EngineConfig(model="stub").attn_implementation is None


def test_unknown_kernel_is_rejected_at_config_time():
    with pytest.raises(ValueError, match="unknown attn_implementation"):
        EngineConfig(model="stub", attn_implementation="flash")


# ---------------------------------------------------------------------------
# Choosing a kernel: the fastest that runs here, or the one that was asked for
# ---------------------------------------------------------------------------

# Three of these tests assert what happens when the paged kernel *can* run, which
# is only true where Triton is installed. That is the precondition itself, so it
# is the condition to skip on rather than CUDA.
_needs_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None,
    reason="the paged kernel's preconditions include a Triton install",
)


@_needs_triton
def test_the_choice_is_paged_where_its_preconditions_hold():
    assert select_implementation(None, torch.device("cuda")) == "paged"


def test_the_choice_falls_back_off_cuda():
    assert select_implementation(None, torch.device("cpu")) == "sdpa"


@_needs_triton
@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_a_named_kernel_that_can_run_is_returned_unchanged(name):
    assert select_implementation(name, torch.device("cuda")) == name


def test_a_named_kernel_that_cannot_run_is_refused_rather_than_downgraded():
    """A silent downgrade would make a benchmark row measure a kernel it did not name."""
    with pytest.raises(ValueError, match="cannot run here"):
        select_implementation("paged", torch.device("cpu"))


def test_the_reason_a_kernel_cannot_run_names_the_precondition_that_failed():
    reason = unsupported_reason("paged", torch.device("cpu"))

    assert reason is not None and "CUDA" in reason


def test_the_universal_kernel_has_no_preconditions():
    assert unsupported_reason("sdpa", torch.device("cpu")) is None


@pytest.mark.gpu
def test_kernels_agree_in_bfloat16_to_within_rounding():
    """bf16 is the engine's working precision, and the kernels sum in different orders.

    Compared on the real query rows only: a padded row attends to nothing, and
    the two kernels disagree there by construction (see the test below).
    """
    query, key, value = (t.to(torch.bfloat16).cuda() for t in _qkv(num_queries=6, num_keys=6))
    mask = build_prefill_mask([6, 4], torch.bfloat16, torch.device("cuda"))
    args = (query, DenseKV(key, value), mask, HEAD_DIM**-0.5, KV_GROUPS)
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
        kernel(query, DenseKV(key, value), mask, 64**-0.5, 4)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) - before

    assert peak_bytes(sdpa_attention) < peak_bytes(eager_attention) / 2
