# pyright: reportPrivateImportUsage=false
"""The paged kernel with a query dimension must answer what the dense kernels answer.

A chunk's queries are the last positions of its sequence's context: each reads
the whole cached prefix and its own chunk only up to itself. The reference is
`eager_attention` over the same K/V gathered out of the pool by hand, under the
padded path's own chunk mask — the path this kernel exists to replace.
"""

from __future__ import annotations

import pytest
import torch

from liteinfer.engine.attention_mask import build_prefill_mask
from liteinfer.models.attention import DenseKV, eager_attention
from liteinfer.models.paged_decode import paged_decode, paged_prefill

pytestmark = pytest.mark.gpu

NUM_SLOTS = 2048


class _ChunkBatch:
    """One prefill pass's pool, addresses and queries: `query_lens[i]` queries over `context_lens[i]` keys."""

    def __init__(
        self,
        query_lens: list[int],
        context_lens: list[int],
        dtype: torch.dtype,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        head_dim: int = 16,
        seed: int = 0,
    ) -> None:
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(seed)

        def randn(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=generator, dtype=dtype, device=device)

        self.device, self.dtype = device, dtype
        self.query_lens, self.context_lens = query_lens, context_lens
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.key_pool = randn(NUM_SLOTS, num_kv_heads, head_dim)
        self.value_pool = randn(NUM_SLOTS, num_kv_heads, head_dim)
        self.query = randn(sum(query_lens), num_heads, head_dim)

        # Right-aligned, exactly as `cache.block_pool.slot_table` builds it.
        max_context = max(context_lens)
        self.slot_table = torch.zeros(len(context_lens), max_context, dtype=torch.long, device=device)
        for row, context_len in enumerate(context_lens):
            self.slot_table[row, max_context - context_len :] = torch.randint(
                1, NUM_SLOTS, (context_len,), generator=generator, device=device
            )

    def paged(self, **kwargs) -> torch.Tensor:
        starts = torch.tensor([0, *self.query_lens], device=self.device).cumsum(0)
        return paged_prefill(
            self.query,
            self.key_pool,
            self.value_pool,
            self.slot_table,
            torch.tensor(self.context_lens, dtype=torch.int32, device=self.device),
            starts.to(torch.int32),
            max(self.query_lens),
            self.scaling,
            self.num_kv_groups,
            **kwargs,
        )

    def dense(self) -> torch.Tensor:
        """Each sequence attended on its own, over its gathered context, and packed again."""
        outputs, first = [], 0
        max_context = self.slot_table.shape[1]
        for row, (query_len, context_len) in enumerate(zip(self.query_lens, self.context_lens, strict=True)):
            slots = self.slot_table[row, max_context - context_len :]
            keys = self.key_pool[slots].permute(1, 0, 2).unsqueeze(0)
            values = self.value_pool[slots].permute(1, 0, 2).unsqueeze(0)
            query = self.query[first : first + query_len].permute(1, 0, 2).unsqueeze(0)
            mask = build_prefill_mask([query_len], self.dtype, self.device, [context_len])
            out = eager_attention(query, DenseKV(keys, values), mask, self.scaling, self.num_kv_groups)
            outputs.append(out.squeeze(0).permute(1, 0, 2))
            first += query_len
        return torch.cat(outputs)


def _assert_matches_in_float32(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_a_chunk_reads_its_prefix_and_its_own_past():
    batch = _ChunkBatch([5], [23], torch.float32)

    _assert_matches_in_float32(batch.paged(), batch.dense())


def test_a_whole_prompt_is_the_chunk_whose_context_is_itself():
    batch = _ChunkBatch([37], [37], torch.float32)

    _assert_matches_in_float32(batch.paged(), batch.dense())


def test_a_batch_mixing_a_decode_row_with_a_chunk_matches_the_dense_kernel():
    """The roadmap's parity case: one query beside many, in a single launch."""
    batch = _ChunkBatch([1, 40, 1, 7], [90, 60, 3, 7], torch.float32)

    _assert_matches_in_float32(batch.paged(), batch.dense())


def test_a_chunk_matches_the_dense_kernel_in_bfloat16_to_within_rounding():
    batch = _ChunkBatch([33, 1, 12], [130, 64, 12], torch.bfloat16)

    torch.testing.assert_close(batch.paged(), batch.dense(), rtol=0, atol=2**-6)


@pytest.mark.parametrize("block_q", [1, 4, 16, 32])
def test_the_answer_does_not_depend_on_how_many_queries_a_program_takes(block_q):
    """Chunks that straddle every one of these sizes, so a block boundary bug shows."""
    batch = _ChunkBatch([33, 16, 15, 1], [100, 16, 47, 9], torch.float32)

    _assert_matches_in_float32(batch.paged(block_q=block_q), batch.dense())


@pytest.mark.parametrize("block_kv", [16, 64, 128])
def test_the_answer_does_not_depend_on_the_key_tile(block_kv):
    """Causal masking meets the online softmax at tile boundaries; contexts straddle them."""
    batch = _ChunkBatch([70, 3], [200, 65], torch.float32)

    _assert_matches_in_float32(batch.paged(block_kv=block_kv), batch.dense())


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "head_dim"),
    [(8, 8, 16), (8, 1, 16), (10, 2, 16), (8, 2, 80)],
    ids=["no-grouping", "one-kv-head", "group-not-a-power-of-two", "head-dim-not-a-power-of-two"],
)
def test_grouping_and_head_dims_the_tiles_pad_are_served(num_heads, num_kv_heads, head_dim):
    batch = _ChunkBatch(
        [19, 1], [50, 30], torch.float32,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )

    _assert_matches_in_float32(batch.paged(), batch.dense())


def test_one_query_per_sequence_answers_what_the_decode_build_answers():
    """Decode is the `q = 1` case of the same kernel; the chunk build must agree with it."""
    batch = _ChunkBatch([1, 1, 1], [90, 17, 1], torch.float32)
    decode = paged_decode(
        batch.query,
        batch.key_pool,
        batch.value_pool,
        batch.slot_table,
        torch.tensor(batch.context_lens, dtype=torch.int32, device=batch.device),
        batch.scaling,
        batch.num_kv_groups,
        num_splits=1,
    )

    _assert_matches_in_float32(batch.paged(), decode)
