# pyright: reportPrivateImportUsage=false
"""The dense kernels must compute the same attention, and sdpa must be the cheaper one.

Every kernel reads a `PagedKV`. The dense ones copy each sequence's context out of
the pool and attend one sequence at a time, with the causal rule that a
sequence's queries are the last positions of its context.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from liteinfer.config import EngineConfig
from liteinfer.models.attention import (
    IMPLEMENTATIONS,
    PagedKV,
    eager_attention,
    paged_attention,
    sdpa_attention,
    select_implementation,
    unsupported_reason,
)

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 8, 2, 16
KV_GROUPS = NUM_HEADS // NUM_KV_HEADS
SCALING = HEAD_DIM**-0.5


class _Batch:
    """Sequences laid end to end in a toy pool, and the queries each one brings.

    Sequence ``i``'s context occupies the pool's next ``context_lens[i]`` slots and
    its ``query_lens[i]`` queries are its newest positions. No `query_lens` is a
    decode step: one query per sequence, one per batch row. A shorter row's
    columns past its context address real slots, as the engine's padding does —
    the next sequence's, or spare ones at the end of the pool.
    """

    def __init__(self, context_lens, query_lens=None, dtype=torch.float32, device="cpu", seed=0):
        generator = torch.Generator().manual_seed(seed)

        def randn(*shape):
            return torch.randn(*shape, generator=generator, dtype=torch.float32).to(dtype).to(device)

        self.context_lens, self.query_lens = list(context_lens), query_lens
        num_slots = sum(context_lens) + max(context_lens)
        self.keys = randn(num_slots, NUM_KV_HEADS, HEAD_DIM)
        self.values = randn(num_slots, NUM_KV_HEADS, HEAD_DIM)
        starts = torch.tensor([0, *context_lens[:-1]]).cumsum(0)
        self.slot_table = (starts.unsqueeze(1) + torch.arange(max(context_lens))).to(device)
        if query_lens is None:
            self.query = randn(len(context_lens), NUM_HEADS, 1, HEAD_DIM)
        else:
            self.query = randn(1, NUM_HEADS, sum(query_lens), HEAD_DIM)

    def kv(self) -> PagedKV:
        query_start_loc = None
        if self.query_lens is not None:
            query_start_loc = torch.tensor([0, *self.query_lens]).cumsum(0).to(torch.int32)
            query_start_loc = query_start_loc.to(self.query.device)
        return PagedKV(
            self.keys,
            self.values,
            self.slot_table,
            torch.tensor(self.context_lens, dtype=torch.int32, device=self.query.device),
            query_start_loc=query_start_loc,
            max_query_len=max(self.query_lens or [1]),
            host_context_lens=tuple(self.context_lens),
            host_query_lens=tuple(self.query_lens or ()),
        )

    def run(self, kernel) -> torch.Tensor:
        return kernel(self.query, self.kv(), SCALING, KV_GROUPS)


def test_kernels_agree_on_whole_prompts():
    batch = _Batch([6, 3], query_lens=[6, 3])

    torch.testing.assert_close(batch.run(sdpa_attention), batch.run(eager_attention))


def test_kernels_agree_on_chunks_continuing_a_cached_prefix():
    """A chunk shorter than its context is the case that needs an explicit causal rule."""
    batch = _Batch([9, 5], query_lens=[4, 1])

    torch.testing.assert_close(batch.run(sdpa_attention), batch.run(eager_attention))


def test_kernels_agree_on_a_decode_step():
    batch = _Batch([6, 4])

    torch.testing.assert_close(batch.run(sdpa_attention), batch.run(eager_attention))


def test_a_decode_row_ignores_what_its_columns_past_the_context_address():
    """The batch is gathered as wide as its longest context; the bound must hide the rest."""
    batch = _Batch([6, 4])
    short_row = batch.run(eager_attention)[1]

    batch.keys[10:] += 10.0
    batch.values[10:] += 10.0

    torch.testing.assert_close(batch.run(eager_attention)[1], short_row)


def test_a_sequence_never_attends_to_another_s_keys():
    """With nothing but slot ranges separating them, a leak would change the first row's answer."""
    batch = _Batch([5, 4], query_lens=[5, 4])
    alone = batch.run(eager_attention)[:, :, :5]

    batch.keys[5:] += 10.0
    together = batch.run(eager_attention)[:, :, :5]

    torch.testing.assert_close(together, alone)


def test_a_chunk_answers_what_the_same_rows_of_the_whole_prompt_answer():
    """Its queries are the last positions of the context: the causal offset has to say so."""
    whole = _Batch([7], query_lens=[7]).run(eager_attention)[:, :, -3:]
    chunk = _Batch([7], query_lens=[3])
    chunk.query = _Batch([7], query_lens=[7]).query[:, :, -3:]

    torch.testing.assert_close(chunk.run(eager_attention), whole)


def test_a_dense_kernel_without_host_lengths_for_a_packed_pass_is_refused():
    """It slices each sequence on the host, and would otherwise have to sync to learn where."""
    batch = _Batch([6, 4], query_lens=[2, 4])

    with pytest.raises(ValueError, match="host_context_lens"):
        eager_attention(batch.query, batch.kv()._replace(host_context_lens=()), SCALING, KV_GROUPS)


def test_paged_decode_rejects_more_than_one_query_per_sequence():
    """The kernel reads the whole history uncausally, which only holds for one query."""
    batch = _Batch([6, 4])
    query = torch.cat([batch.query, batch.query], dim=2)

    with pytest.raises(ValueError, match="one query per sequence"):
        paged_attention(query, batch.kv(), SCALING, KV_GROUPS)


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
    """bf16 is the engine's working precision, and the kernels sum in different orders."""
    batch = _Batch([6, 9], query_lens=[6, 4], dtype=torch.bfloat16, device="cuda")

    torch.testing.assert_close(batch.run(sdpa_attention), batch.run(eager_attention), rtol=0, atol=2**-6)


@pytest.mark.gpu
def test_sdpa_does_not_materialise_the_score_matrix():
    """The point of the kernel: peak memory stops scaling with queries x keys.

    At this length the eager score matrix is 256 MiB in bf16 and 512 MiB again
    once softmax upcasts it, so a kernel that keeps it in SRAM shows up as a
    peak-memory difference far larger than the inputs themselves.
    """
    device = torch.device("cuda")
    seq_len = 2048
    query = torch.randn(1, 32, seq_len, 64, dtype=torch.bfloat16, device=device)
    pool = torch.randn(seq_len, 8, 64, dtype=torch.bfloat16, device=device)
    kv = PagedKV(
        pool, pool,
        torch.arange(seq_len, device=device).unsqueeze(0),
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        max_query_len=seq_len,
        host_context_lens=(seq_len,),
        host_query_lens=(seq_len,),
    )

    def peak_bytes(kernel) -> int:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.max_memory_allocated(device)
        kernel(query, kv, 64**-0.5, 4)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) - before

    assert peak_bytes(sdpa_attention) < peak_bytes(eager_attention) / 2
