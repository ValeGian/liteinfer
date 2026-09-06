# pyright: reportPrivateImportUsage=false
"""Captured decode forward passes. One graph per batch width.

A decode step issues roughly 700 kernels to do about 7 ms of GPU work inside a
13 ms step, and the gaps between those launches are the rest of it. They do not
shrink with batch width — measured at a flat 18.7 us of wall per launch at batch
1 and batch 32 alike — so 32x the tokens buys 1.29x the GPU time and the step
does not move. A captured graph submits the whole sequence in one go, which is
what removes them: replaying the same forward measures 5.84 ms against 11.18 ms
eager at batch 1, and 7.28 against 11.67 at batch 32.

What a capture freezes, and what it does not
--------------------------------------------
A graph bakes in every kernel argument that is a scalar and every pointer it was
recorded with. So shapes, batch width and buffer addresses are fixed: the host
refills **static input buffers** before each replay, and there is **one graph per
batch width**.

What is *not* frozen is anything the kernels read out of device memory — and that
is where the context length lives. `paged_decode` bounds its loop by
`context_lens`, a tensor, so one capture serves every context length and the slot
table can be a single fixed ``[max_num_seqs, max_model_len]`` buffer. Its unused
columns are never even read: the kernel starts at ``max_context - context_len``,
so a wide table costs nothing and needs no zeroing between steps.

That is why this is worth doing now and measured 1.06x when it was first tried.
Until §2.3 the decode read *gathered* each sequence's KV history into a
contiguous tensor, so the table's width was the size of a copy: a fixed-width
table meant padding real bytes every layer of every step, and the item had to
bucket the KV length and pay for the padding anyway. The paged kernel reads the
pool where it lies and skips what `context_lens` excludes, so there are no KV
buckets here, and no padded rows to discard either — capturing per exact batch
width means every row of every capture is a real sequence.

Which is also the precondition. The dense kernels take an additive mask whose
width follows the batch's longest context, and a capture cannot hold a shape that
changes every step; only the paged kernel takes its bounds as a tensor.
"""

from __future__ import annotations

import logging

import torch

from liteinfer.cache.continuous_kv_cache import ContinuousKVCache
from liteinfer.models.attention import reads_paged_kv

_LOGGER = logging.getLogger(__name__)

# Forwards run on a side stream before a capture. The first call through a shape
# allocates cuBLAS workspaces and picks kernels, and neither may happen inside
# the capture — three is what PyTorch's own graph guidance uses.
_WARMUP_FORWARDS = 3

# Distinct batch widths that may be captured before the rest run eager.
#
# A decode batch is at most `max_num_seqs` sequences, so at the default 32 this
# never binds and every width a run visits gets its own graph — which is why
# there are no padded rows here. vLLM pads each batch up to a ladder of sizes
# instead, because its batch width is a *token* count that can be any value in
# the thousands; capturing one graph per value is only possible because
# liteinfer's decode batch is bounded by a small config. An engine configured
# far wider than the default would want vLLM's ladder rather than this cap, and
# until it has one, this is the bound that keeps the graphs from growing without
# limit. See the §3.2 milestone.
_MAX_CAPTURES = 64


def unsupported_reason(device: torch.device, attn_implementation: str) -> str | None:
    """Why the decode forward cannot be captured here, or `None` if it can."""
    if device.type != "cuda":
        return f"CUDA graphs need a CUDA device and this is {device}"
    if not reads_paged_kv(attn_implementation):
        return (
            f"attn_implementation={attn_implementation!r} builds an attention mask whose "
            "width follows the batch's longest context, and a capture freezes every shape; "
            "only the paged kernel takes its context lengths as a tensor"
        )
    return None


def graphs_are_enabled(
    requested: bool | None, device: torch.device, attn_implementation: str
) -> bool:
    """Resolve `EngineConfig.enable_cuda_graphs` against what this engine can do.

    `None` means "capture where the preconditions hold", which is the default.
    `True` asks for capture specifically and is refused rather than downgraded,
    for the same reason naming an attention kernel is: a benchmark row or a
    parity test that asks for graphs has to get them or hear why not.
    """
    if requested is False:
        return False
    reason = unsupported_reason(device, attn_implementation)
    if reason is None:
        return True
    if requested:
        raise ValueError(f"enable_cuda_graphs=True, but decode cannot be captured: {reason}")
    _LOGGER.info("CUDA graphs off: %s", reason)
    return False


class DecodeGraphs:
    """The decode forward, captured once per batch width and replayed thereafter.

    Captures happen lazily, on the first step at a width, because a run only ever
    visits the widths its own scheduling produces — a full batch for most of a
    throughput run, one sequence for a latency run, and the widths in between
    only while the batch drains. Every graph after the first shares the first
    one's memory pool, so N graphs cost about what one does.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cache: ContinuousKVCache,
        *,
        device: torch.device,
        max_num_seqs: int,
        max_model_len: int,
    ) -> None:
        self._model = model
        self._cache = cache
        self._max_num_seqs = max_num_seqs
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._memory_pool = None

        self._input_ids = torch.zeros((max_num_seqs, 1), dtype=torch.long, device=device)
        self._position_ids = torch.zeros((max_num_seqs, 1), dtype=torch.long, device=device)
        # One table for every context length there will ever be. Unread columns
        # hold whatever they last held; see the module docstring.
        self._slots = torch.zeros((max_num_seqs, max_model_len), dtype=torch.long, device=device)
        self._context_lens = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        self._logits: torch.Tensor | None = None

    @property
    def captured_widths(self) -> list[int]:
        """Batch widths that have been captured, in the order they were first seen."""
        return list(self._graphs)

    def run(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        slots: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Replay this step's decode forward, capturing first if the width is new.

        Returns ``[batch, vocab_size]`` logits for the decoded token.
        """
        batch_size = input_ids.shape[0]
        if batch_size > self._max_num_seqs:
            raise ValueError(
                f"decode batch of {batch_size} exceeds max_num_seqs={self._max_num_seqs}"
            )
        self._fill(batch_size, input_ids, position_ids, slots, context_lens)

        graph = self._graphs.get(batch_size)
        if graph is None:
            graph = self._graphs[batch_size] = self._capture(batch_size)
        graph.replay()

        assert self._logits is not None, "_capture allocates the logits buffer"
        # Copied rather than returned as a view: the next replay overwrites this
        # buffer, while the eager path returns a tensor that lives as long as its
        # caller holds it. One function, one lifetime. vLLM hands back the
        # graph-owned tensor instead; the copy is ~11 us against a step this is
        # trying to cut by 4 ms, which is worth not having an aliasing contract.
        return self._logits[:batch_size].clone()

    def has_capacity_for(self, batch_size: int) -> bool:
        """Whether this width can be captured, or has to run eager."""
        return batch_size in self._graphs or len(self._graphs) < _MAX_CAPTURES

    def _fill(
        self,
        batch_size: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        slots: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> None:
        """Copy this step's inputs into the buffers the capture was recorded against."""
        width = slots.shape[1]
        if width > self._slots.shape[1]:
            # A negative slice would silently keep only the last `max_model_len`
            # columns, which is the newest tokens of the wrong sequences.
            raise ValueError(
                f"slot table is {width} columns wide but the engine was built for "
                f"max_model_len={self._slots.shape[1]}"
            )
        self._input_ids[:batch_size].copy_(input_ids)
        self._position_ids[:batch_size].copy_(position_ids)
        self._context_lens[:batch_size].copy_(context_lens)
        # The slot table is right-aligned, so it lands in the table's last columns
        # and every row keeps its own alignment — right-alignment composes, which
        # is what lets one fixed-width buffer serve every context. The columns in
        # front are the padding the kernel starts after, so they are left as they
        # are rather than cleared.
        self._slots[:batch_size, -width:].copy_(slots)

    def _forward(self, batch_size: int) -> torch.Tensor:
        """The decode forward over the static buffers, at one batch width."""
        payload = self._cache.make_paged_decode_payload(
            self._slots[:batch_size], self._context_lens[:batch_size]
        )
        out = self._model(
            input_ids=self._input_ids[:batch_size],
            position_ids=self._position_ids[:batch_size],
            past_key_values=payload,
            attention_mask=None,
        )
        return out.logits[:, -1, :]

    def _capture(self, batch_size: int) -> torch.cuda.CUDAGraph:
        """Record one batch width's forward. The buffers must already hold a real step."""
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(_WARMUP_FORWARDS):
                warmup_logits = self._forward(batch_size)
        torch.cuda.current_stream().wait_stream(side_stream)

        if self._logits is None:
            # Sized from the forward rather than from the config: the vocabulary
            # and the output dtype are the model's to state, not ours to assume.
            self._logits = torch.empty(
                (self._max_num_seqs, warmup_logits.shape[-1]),
                dtype=warmup_logits.dtype,
                device=warmup_logits.device,
            )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._memory_pool):
            self._logits[:batch_size].copy_(self._forward(batch_size))
        if self._memory_pool is None:
            self._memory_pool = graph.pool()

        _LOGGER.info(
            "captured decode graph for batch width %d (%d captured so far)",
            batch_size, len(self._graphs) + 1,
        )
        return graph
