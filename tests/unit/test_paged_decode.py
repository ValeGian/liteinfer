# pyright: reportPrivateImportUsage=false
"""The fused paged decode kernel must answer what the dense kernels answer.

Everything here runs on GPU: the kernel is Triton, and Triton is CUDA-only.
The reference is `eager_attention` over the same K/V gathered out of the pool
by hand — the path the paged kernel exists to replace.
"""

from __future__ import annotations

import pytest
import torch

from liteinfer.engine.attention_mask import build_continuous_decode_mask
from liteinfer.models.attention import DenseKV, eager_attention
from liteinfer.models.paged_decode import paged_decode

pytestmark = pytest.mark.gpu

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 8, 2, 16
KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
NUM_SLOTS = 512
SCALING = HEAD_DIM**-0.5


class _PagedBatch:
    """One decode step's worth of pool, addresses and queries."""

    def __init__(
        self,
        context_lens: list[int],
        dtype: torch.dtype,
        num_heads: int = NUM_HEADS,
        num_kv_heads: int = NUM_KV_HEADS,
        head_dim: int = HEAD_DIM,
        seed: int = 0,
    ) -> None:
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(seed)

        def randn(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=generator, dtype=dtype, device=device)

        self.dtype = dtype
        self.device = device
        self.context_lens = context_lens
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.key_pool = randn(NUM_SLOTS, num_kv_heads, head_dim)
        self.value_pool = randn(NUM_SLOTS, num_kv_heads, head_dim)
        self.query = randn(len(context_lens), num_heads, head_dim)

        # Right-aligned, exactly as `cache.block_pool.slot_table` builds it:
        # a sequence's real tokens are its last `context_len` columns, and the
        # padding in front of them points at the null block.
        max_context = max(context_lens)
        self.slot_table = torch.zeros(len(context_lens), max_context, dtype=torch.long, device=device)
        for row, context_len in enumerate(context_lens):
            self.slot_table[row, max_context - context_len :] = torch.randint(
                1, NUM_SLOTS, (context_len,), generator=generator, device=device
            )

    def paged(self, **kwargs) -> torch.Tensor:
        context_lens = torch.tensor(self.context_lens, dtype=torch.int32, device=self.device)
        return paged_decode(
            self.query,
            self.key_pool,
            self.value_pool,
            self.slot_table,
            context_lens,
            self.scaling,
            self.num_kv_groups,
            **kwargs,
        )

    def dense(self) -> torch.Tensor:
        """The same attention through the gather the paged kernel avoids."""
        keys = self.key_pool[self.slot_table].permute(0, 2, 1, 3)
        values = self.value_pool[self.slot_table].permute(0, 2, 1, 3)
        mask = build_continuous_decode_mask(self.context_lens, self.dtype, self.device)
        out = eager_attention(
            self.query.unsqueeze(2), DenseKV(keys, values), mask, self.scaling, self.num_kv_groups
        )
        return out.squeeze(2)


def test_paged_decode_matches_the_dense_kernel_in_float32():
    batch = _PagedBatch([6, 4, 1], torch.float32)

    torch.testing.assert_close(batch.paged(), batch.dense(), rtol=1e-5, atol=1e-5)


def test_paged_decode_matches_the_dense_kernel_in_bfloat16_to_within_rounding():
    """bf16 is the engine's working precision; the two sum over the keys in different orders."""
    batch = _PagedBatch([6, 4, 1], torch.bfloat16)

    torch.testing.assert_close(batch.paged(), batch.dense(), rtol=0, atol=2**-6)


@pytest.mark.parametrize("block_kv", [16, 32, 64, 128])
def test_paged_decode_matches_the_dense_kernel_across_several_key_tiles(block_kv):
    """The answer must not depend on how many keys are folded in at a time.

    Splitting the key axis is where the kernel does its only stateful arithmetic:
    each tile can raise the running maximum, and the accumulator and the
    denominator both have to be rescaled when it does. A tile size that divides
    the context evenly hides a boundary bug, so the contexts here straddle every
    one of these sizes — and the sweep is what picked 64 as the default, so it is
    also what keeps that choice re-runnable.
    """
    batch = _PagedBatch([200, 65, 64, 63, 17, 1], torch.float32)

    torch.testing.assert_close(
        batch.paged(block_kv=block_kv), batch.dense(), rtol=1e-5, atol=1e-5
    )


def test_paged_decode_matches_the_dense_kernel_when_the_group_is_not_a_power_of_two():
    """The query tile is padded to a power of two; the rows past the group must not count."""
    batch = _PagedBatch([70, 5], torch.float32, num_heads=10, num_kv_heads=2)

    torch.testing.assert_close(batch.paged(), batch.dense(), rtol=1e-5, atol=1e-5)


def test_paged_decode_matches_the_dense_kernel_without_grouped_query_heads():
    """One query head per KV head is the degenerate group, and must not be a special case."""
    batch = _PagedBatch([9, 3], torch.float32, num_kv_heads=NUM_HEADS)

    torch.testing.assert_close(batch.paged(), batch.dense(), rtol=1e-5, atol=1e-5)


def test_paged_decode_never_reads_a_slot_outside_the_context():
    """What the padding columns point at must not reach the answer.

    The dense path gathers those columns and masks them away afterwards. The
    paged kernel stops at `context_lens`, so filling them with real slots must
    change nothing — which is what makes the decode mask unnecessary.
    """
    batch = _PagedBatch([200, 65, 3], torch.float32)
    with_null_padding = batch.paged()
    is_padding = batch.slot_table == 0
    batch.slot_table[is_padding] = torch.randint(
        1, NUM_SLOTS, (int(is_padding.sum()),), device=batch.device
    )

    torch.testing.assert_close(batch.paged(), with_null_padding, rtol=0, atol=0)


@pytest.mark.parametrize("head_dim", [96, 80, 48, 24])
def test_paged_decode_matches_the_dense_kernel_on_a_head_dim_triton_cannot_index(head_dim):
    """`tl.arange` needs a power-of-two length, so these head dimensions are padded.

    96 is Phi-3-mini's and 80 is Phi-2's, so this is the case a second
    architecture brings rather than a hypothetical. What the padding must not do
    is leak into the answer, which is what these shapes check.
    """
    batch = _PagedBatch([70, 5], torch.float32, head_dim=head_dim)

    torch.testing.assert_close(batch.paged(), batch.dense(), rtol=1e-5, atol=1e-5)


def test_paged_decode_rejects_pools_that_are_not_laid_out_alike():
    """One set of strides addresses both pools, so differing layouts would read garbage."""
    batch = _PagedBatch([4], torch.float32)
    context_lens = torch.tensor([4], dtype=torch.int32, device=batch.device)

    with pytest.raises(ValueError, match="laid out identically"):
        paged_decode(
            batch.query,
            batch.key_pool,
            batch.value_pool.transpose(1, 2),
            batch.slot_table,
            context_lens,
            SCALING,
            KV_GROUPS,
        )
