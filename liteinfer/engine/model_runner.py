# pyright: reportPrivateImportUsage=false
"""ModelRunner — forward-pass execution for continuous batching.

``execute(seqs, num_tokens)`` runs one step: the next ``num_tokens`` of each
sequence, which is a chunk of its prompt (the whole prompt unless the
scheduler's token budget split it) or the one token it last sampled. A step
holding both is one forward.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import torch

from liteinfer.cache.block_pool import BlockPool
from liteinfer.cache.block_tables import BlockTables
from liteinfer.cache.kv_cache import (
    DecodeProfilePayload,
    KVCache,
    ProfilePayload,
)
from liteinfer.config import EngineConfig
from liteinfer.engine.cuda_graphs import DecodeGraphs, graphs_are_enabled
from liteinfer.engine.sequence import Sequence
from liteinfer.hub import resolve_model_path
from liteinfer.models import LAST_POSITION
from liteinfer.models.attention import reads_pool_in_place
from liteinfer.models.loader import load_hf_model
from liteinfer.models.paged_kernels import choose_num_splits
from liteinfer.tokenizer import Tokenizer

if TYPE_CHECKING:
    from transformers import PretrainedConfig

_LOGGER = logging.getLogger(__name__)
_GIB = 1 << 30

# Stand-in device size for CPU runs, which exist to test the sizing logic rather
# than to serve anything. A constant keeps those tests independent of the host.
_CPU_NOMINAL_TOTAL_BYTES = 1 << 30

# Tokens in the throwaway forward that precedes the profile, so one-time
# workspace allocations land outside the measurement rather than inside it.
_PROFILE_WARMUP_TOKENS = 16


class _Chunk(NamedTuple):
    """The tokens one sequence computes in a pass, and where they start."""

    token_ids: list[int]
    start: int

    @property
    def end(self) -> int:
        """One past the chunk's last position: what its sequence holds once the pass is written."""
        return self.start + len(self.token_ids)


def _tokens_at(seq: Sequence, start: int, count: int) -> list[int]:
    """`count` of the sequence's tokens from `start`, read from the prompt or the output.

    A chunk never crosses from one into the other: a sequence still computing
    its prompt has sampled nothing, and one past its prompt owes exactly the
    token it last sampled. So this slices one list rather than concatenating
    prompt and output, which would copy the whole sequence for every row of
    every step.
    """
    prompt_len = len(seq.prompt_token_ids)
    if start >= prompt_len:
        return seq.output_token_ids[start - prompt_len : start - prompt_len + count]
    if start + count > prompt_len:
        raise ValueError(
            f"{seq.request_id}: a chunk from position {start} of {count} tokens crosses "
            f"the end of its {prompt_len}-token prompt"
        )
    return seq.prompt_token_ids[start : start + count]


def _head_dim(hf_config: PretrainedConfig) -> int:
    """Head dimension, which most configs state and the rest imply."""
    return getattr(
        hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
    )


class ModelRunner:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = config.resolved_device()
        self.model: torch.nn.Module | None = None
        self.hf_config = None
        self.tokenizer: Tokenizer | None = None
        self._block_tables: BlockTables | None = None
        self._cache: KVCache | None = None
        self._graphs: DecodeGraphs | None = None
        self._forward_bytes = 0

    def load_model(self) -> None:
        model_path = resolve_model_path(self.config.model)
        self.model, self.hf_config = load_hf_model(self.config, model_path)
        self.tokenizer = Tokenizer(model_path)
        # Measured before the pool exists, because the pool gets whatever the
        # forward turns out not to need.
        self._forward_bytes = self._profile_forward_bytes()
        pool = self._create_block_pool()
        self._block_tables = BlockTables(pool.num_blocks, pool.block_size)
        self._cache = KVCache(pool, self._block_tables)
        self._graphs = self._create_decode_graphs()

    @property
    def block_tables(self) -> BlockTables:
        """Which blocks each sequence holds and which are free; the scheduler admits by it.

        Built by `load_model`, because how many blocks there are is only known
        once the activation profile has said what the forward needs.
        """
        assert self._block_tables is not None, "load_model() builds the block tables"
        return self._block_tables

    @property
    def attn_implementation(self) -> str:
        """The kernel this engine resolved to, which may not be the one requested.

        `EngineConfig.attn_implementation` can be `None` for "choose for me";
        `load_hf_model` makes the choice and records it, because that is where
        the device and the model's head dimension are both known. Reading it
        back from there keeps one answer rather than two.
        """
        resolved = getattr(self.hf_config, "_attn_implementation", None)
        assert isinstance(resolved, str), "load_model() records the resolved kernel"
        return resolved

    @property
    def captured_decode_widths(self) -> list[int]:
        """Batch widths whose decode forward is replayed from a graph, first seen first.

        Empty when capture is off or nothing has been captured yet, which is the
        same answer from the caller's side: this step ran kernel by kernel.
        """
        return [] if self._graphs is None else self._graphs.captured_widths

    @torch.inference_mode()
    def execute(self, seqs: list[Sequence], num_tokens: list[int] | None = None) -> torch.Tensor:
        """One step's forward over the next `num_tokens` tokens of each sequence.

        Each sequence continues from its `num_computed_tokens`, with the next
        chunk of its prompt or with the token it last sampled. `None` computes
        everything the cache does not hold yet — a whole prompt for a new
        sequence, the sampled token for one past its prompt. Once the forward
        returns, each sequence's `num_computed_tokens` counts what the pass wrote:
        the runner is what knows the pass ran, and a pass that raises leaves the
        counts where they were. The blocks the pass writes are reserved here, all
        or nothing; under the engine the scheduler granted only what is free, so a
        refusal here is a scheduling bug rather than a full pool.

        Returns logits ``[len(seqs), vocab_size]`` at each sequence's last token,
        in `seqs` order. Only a sequence whose prompt is now complete has a next
        token worth sampling, and the caller is the one that knows which rows
        those are.

        How the step runs follows from the tokens it holds, not from a phase.
        One token per sequence is a decode batch, laid out one per row, which the
        paged kernel serves with split-K and a captured graph replays. Anything
        else is one packed forward: prompts, chunks and sampled tokens side by
        side, every row's K/V read where it lies.
        """
        assert self._cache is not None and self.model is not None

        request_ids = [s.request_id for s in seqs]
        chunks = self._next_chunks(seqs, num_tokens)
        self.block_tables.reserve(request_ids, [chunk.end for chunk in chunks])

        if all(len(chunk.token_ids) == 1 for chunk in chunks):
            logits = self._decode(request_ids, chunks)
        else:
            logits = self._packed_forward(request_ids, chunks)
        for seq, chunk in zip(seqs, chunks, strict=True):
            seq.num_computed_tokens = chunk.end
        return logits

    def _next_chunks(self, seqs: list[Sequence], num_tokens: list[int] | None) -> list[_Chunk]:
        """The tokens each sequence computes next, starting after what is cached.

        A chunk running past what its sequence holds is refused rather than
        truncated: the sequence would count tokens the pass never wrote.
        """
        chunks = []
        for i, seq in enumerate(seqs):
            start = seq.num_computed_tokens
            count = len(seq) - start if num_tokens is None else num_tokens[i]
            if count < 1 or start + count > len(seq):
                raise ValueError(
                    f"{seq.request_id}: cannot compute {count} tokens from position {start} "
                    f"of a {len(seq)}-token sequence"
                )
            chunks.append(_Chunk(_tokens_at(seq, start, count), start))
        return chunks

    def _packed_forward(self, request_ids: list[str], chunks: list[_Chunk]) -> torch.Tensor:
        """Every chunk as one flat run of tokens: each position computed is one kept.

        A sampled token is a chunk of one, so a step that admits prompts while
        others decode is still one pass, every row read out of the pool.
        """
        assert self._cache is not None and self.model is not None

        counts = [len(chunk.token_ids) for chunk in chunks]
        token_ids = [token for chunk in chunks for token in chunk.token_ids]
        payload = self._cache.make_packed_payload(request_ids, [chunk.start for chunk in chunks], counts)
        # The leading axis is 1 because the batch is the token run itself; where
        # each sequence begins is in the payload's addresses, which also give
        # every token's position for RoPE and each sequence's last token, the
        # one row it samples from.
        out = self.model(
            input_ids=torch.tensor([token_ids], dtype=torch.long, device=self.device),
            position_ids=payload.addresses.positions,
            past_key_values=payload,
            logits_positions=payload.addresses.query_start_loc[1:] - 1,
        )
        return out.logits[0]

    def _decode(self, request_ids: list[str], chunks: list[_Chunk]) -> torch.Tensor:
        """One token per sequence, laid out one per batch row, which is what a capture holds.

        Every address is built here rather than on the first layer: the forward
        pass must contain no host-side work. The write is one slot per sequence;
        the read is each sequence's whole history, bounded by its context length.
        """
        assert self._cache is not None and self.model is not None

        input_ids, position_ids = self._build_decode_inputs(chunks)
        totals = [chunk.end for chunk in chunks]
        write_slots = self._cache.newest_slots_for(request_ids, [chunk.start for chunk in chunks])
        slots = self._cache.slot_table_for(request_ids, totals)
        context_lens = torch.tensor(totals, dtype=torch.int32, device=self.device)

        if self._graphs is not None and self._graphs.has_capacity_for(len(request_ids)):
            return self._graphs.run(input_ids, position_ids, write_slots, slots, context_lens)

        # The split count is the paged kernel's alone, and choosing one asks the
        # device how many SMs it has.
        is_paged = reads_pool_in_place(self.attn_implementation)
        num_splits = self.splits_for_width(len(request_ids)) if is_paged else None
        payload = self._cache.make_decode_payload(write_slots, slots, context_lens, num_splits)
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=payload,
            logits_positions=LAST_POSITION,
        )
        return out.logits[:, -1, :]

    # ------------------------------------------------------------------
    # Input builders
    # ------------------------------------------------------------------

    def splits_for_width(self, batch_size: int) -> int:
        """How many programs share one sequence's decode key loop at this batch width.

        Chosen from ``max_model_len`` rather than from this step's context, so the
        count is a function of the batch width alone. That is what a capture needs:
        a CUDA graph bakes scalar kernel arguments in, while one capture serves
        every context that fits its slot buffer — and it keeps the captured and
        eager paths choosing the same grid for the same batch.

        The cost is that a short context runs more splits than it would pick for
        itself. Measured per layer at batch 1 against the count the context would
        choose: 0.83x at 128 tokens, 1.32x at 256, 3.77x at 1,024, 4.47x at 4,096.
        Only the first of those is a loss, and it is 0.9 us on a 6.1 ms step.
        """
        if self.config.paged_decode_splits is not None:
            return self.config.paged_decode_splits
        assert self.hf_config is not None
        return choose_num_splits(
            batch_size,
            self.hf_config.num_key_value_heads,
            self.config.max_model_len,
            self.device.index or 0,
        )

    def _build_decode_inputs(self, chunks: list[_Chunk]) -> tuple[torch.Tensor, torch.Tensor]:
        """Each sequence's one token and its position, as ``[B, 1]`` each, in one transfer."""
        tokens_and_positions = torch.tensor(
            [[chunk.token_ids[0] for chunk in chunks], [chunk.start for chunk in chunks]],
            dtype=torch.long,
            device=self.device,
        )
        return tokens_and_positions[0].unsqueeze(1), tokens_and_positions[1].unsqueeze(1)

    def _create_decode_graphs(self) -> DecodeGraphs | None:
        """Build the capture cache, or `None` where decode cannot be captured.

        Built here rather than lazily on the first decode step because the preconditions are
        known once the model is loaded, and because a `None` is the whole signal
        the decode path needs — no flag to re-read per step.
        """
        assert self.model is not None and self._cache is not None
        if not graphs_are_enabled(
            self.config.enable_cuda_graphs, self.device, self.attn_implementation
        ):
            return None
        return DecodeGraphs(
            self.model,
            self._cache,
            device=self.device,
            max_num_seqs=self.config.max_num_seqs,
            max_model_len=self.config.max_model_len,
            splits_for_width=self.splits_for_width,
        )

    # ------------------------------------------------------------------
    # Block pool
    # ------------------------------------------------------------------

    def _create_block_pool(self) -> BlockPool:
        assert self.hf_config is not None
        num_layers: int = self.hf_config.num_hidden_layers
        num_kv_heads: int = self.hf_config.num_key_value_heads
        head_dim = _head_dim(self.hf_config)
        num_blocks = self._compute_num_blocks(num_layers, num_kv_heads, head_dim)
        return BlockPool(
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=self.config.dtype,
            device=self.device,
        )

    def _compute_num_blocks(self, num_layers: int, num_kv_heads: int, head_dim: int) -> int:
        """Size the pool to the smaller of what memory allows and what the engine can reach.

        `max_num_seqs` sequences of `max_model_len` tokens is the most KV that can
        ever exist, so blocks beyond that are memory the engine is structurally
        unable to use — and that surplus is what the forward pass needs for
        activations.
        """
        dtype_bytes = torch.finfo(self.config.dtype).bits // 8
        bytes_per_block = (
            self.config.block_size * num_kv_heads * head_dim * dtype_bytes * 2 * num_layers
        )
        if self.config.num_gpu_blocks is not None:
            self._log_pool(self.config.num_gpu_blocks, bytes_per_block, "set by num_gpu_blocks")
            return self.config.num_gpu_blocks

        affordable = self._affordable_blocks(bytes_per_block)
        reachable = math.ceil(
            self.config.max_num_seqs * self.config.max_model_len / self.config.block_size
        )

        if affordable < reachable:
            # Naming the concurrency it can actually serve is the part a caller can
            # act on: `max_num_seqs` is otherwise a promise the pool cannot keep.
            servable = affordable * self.config.block_size // self.config.max_model_len
            _LOGGER.warning(
                "KV pool holds %d blocks but this config could need %d: at "
                "max_model_len=%d it can serve %d concurrent sequences, not "
                "max_num_seqs=%d. Lower max_num_seqs or max_model_len, or raise "
                "gpu_memory_utilization.",
                affordable, reachable, self.config.max_model_len, servable,
                self.config.max_num_seqs,
            )
            num_blocks = max(1, affordable)
            reason = "limited by the memory budget"
        else:
            num_blocks = max(1, reachable)
            reason = f"sized for {self.config.max_num_seqs} x {self.config.max_model_len} tokens"
        self._log_pool(num_blocks, bytes_per_block, reason)
        return num_blocks

    def _affordable_blocks(self, bytes_per_block: int) -> int:
        """Blocks the memory budget allows, given the weights already resident.

        `mem_get_info`'s *free* figure was the obvious budget and the wrong one:
        the same config on the same GPU sized differently depending on what else
        happened to be resident a second earlier, which made a benchmark's pool a
        property of the machine's history rather than of the run. Total memory is
        a device constant and the weights are deterministic, so this answer is
        reproducible across loads.

        What it still does not measure is the forward pass. `memory_allocated`
        counts tensors torch allocated, so the CUDA context and cuBLAS workspaces
        fall outside it, and the activations have not been allocated yet at all —
        the fraction is what covers both — and `_profile_forward_bytes` measures
        the activations, which are subtracted here.
        """
        if self.device.type == "cuda":
            total = torch.cuda.get_device_properties(self.device).total_memory
            resident = torch.cuda.memory_allocated(self.device)
        else:
            total, resident = _CPU_NOMINAL_TOTAL_BYTES, 0
        budget = int(total * self.config.gpu_memory_utilization) - resident
        return max(0, budget - self._forward_bytes) // bytes_per_block

    def _profile_forward_bytes(self) -> int:
        """Peak allocation the widest step needs, beyond the weights.

        `schedule()` hands out up to `token_budget` tokens per step, so the
        widest packed pass is not hypothetical — it is what a cold engine does
        when that many prompt tokens are already waiting. A kernel that copies
        each row's context out of the pool has a second worst case the budget
        does not cap: a decode step at full width (see `DecodeProfilePayload`).
        Running both here turns the activation budget from a fraction someone
        guessed into a number this device measured, through the kernel the engine
        will serve with, and the larger of the two is what the pool leaves free.

        The payloads write to no cache, so this needs no pool. That is what lets
        the measurement happen before the pool is sized.

        A forward that will not fit is not fatal. It means the configuration
        cannot serve its own worst case, which is worth saying rather than
        crashing on — that forward is left out of the budget, and the WARNING in
        `_compute_num_blocks` still reports what the pool can serve.
        """
        if self.device.type != "cuda":
            return 0
        try:
            # The first forward in a process allocates cuBLAS workspaces that every
            # later one reuses, and they land in the peak. Unwarmed, a narrow config
            # measured 8.49 MiB where a config sixteen times wider measured 5.77 —
            # so the first engine in a process would have been handed the smallest
            # pool. One throwaway pass absorbs that. It is left to raise: an engine
            # that cannot run 16 tokens has no worst case worth measuring.
            self._profile_packed([min(self.config.max_model_len, _PROFILE_WARMUP_TOKENS)])
            measured = self._measure_widest_packed_step()
            if not reads_pool_in_place(self.attn_implementation):
                measured = max(measured, self._measure_widest_decode())
        finally:
            torch.cuda.empty_cache()
        return measured

    def _measure_widest_packed_step(self) -> int:
        """Peak allocation of `_widest_packed_step`, or 0 if it does not fit."""
        lengths = self._widest_packed_step()
        return self._peak_bytes(
            lambda: self._profile_packed(lengths),
            f"packed step ({sum(lengths)} tokens as {len(lengths)} sequences)",
            "Lower max_num_batched_tokens or max_model_len to serve that case.",
        )

    def _measure_widest_decode(self) -> int:
        """Peak allocation of a decode step at full width and full context, or 0 if it does not fit."""
        assert self.hf_config is not None
        batch, width = self.config.max_num_seqs, self.config.max_model_len
        payload = DecodeProfilePayload(
            batch,
            width,
            self.hf_config.num_key_value_heads,
            _head_dim(self.hf_config),
            self.config.dtype,
            self.device,
        )
        return self._peak_bytes(
            lambda: self._profile_decode(payload, batch, width),
            f"decode step ({batch} sequences x {width} tokens of context)",
            "Lower max_num_seqs or max_model_len, or serve with the paged kernel, which reads "
            "the pool in place.",
        )

    def _peak_bytes(self, forward: Callable[[], None], step: str, remedy: str) -> int:
        """What `forward` allocates at its peak beyond what was resident, or 0 if it does not fit."""
        torch.cuda.reset_peak_memory_stats(self.device)
        before = torch.cuda.memory_allocated(self.device)
        try:
            forward()
        except torch.OutOfMemoryError:
            _LOGGER.warning(
                "the widest %s this config allows does not fit, so the KV pool is sized "
                "without measuring it. %s", step, remedy,
            )
            return 0
        measured = max(0, torch.cuda.max_memory_allocated(self.device) - before)
        _LOGGER.info("widest %s needs %.2f GiB of activations", step, measured / _GIB)
        return measured

    def _widest_packed_step(self) -> list[int]:
        """The packed step needing the most activations: a full token budget in the fewest, longest prompts.

        The linear layers cost the same however a step's tokens are split, but
        attention costs more the longer each sequence is — the dense kernels copy
        each context out, and `eager` writes its square score matrix — so the
        worst case is whole `max_model_len` prompts, the remainder in one more.
        """
        longest = self.config.max_model_len
        budget = min(self.config.token_budget, self.config.max_num_seqs * longest)
        full, rest = divmod(budget, longest)
        return [longest] * full + ([rest] if rest else [])

    def _profile_packed(self, lengths: list[int]) -> None:
        """One packed forward over whole prompts of dummy tokens, writing to no cache."""
        assert self.model is not None
        payload = ProfilePayload(lengths, self.device)
        with torch.inference_mode():
            self.model(
                input_ids=torch.zeros((1, sum(lengths)), dtype=torch.long, device=self.device),
                position_ids=payload.positions,
                past_key_values=payload,
                logits_positions=payload.query_start_loc[1:] - 1,
            )

    def _profile_decode(self, payload: DecodeProfilePayload, batch: int, width: int) -> None:
        """One decode forward of dummy tokens, each row the last position of a `width`-token context."""
        assert self.model is not None
        with torch.inference_mode():
            self.model(
                input_ids=torch.zeros((batch, 1), dtype=torch.long, device=self.device),
                position_ids=torch.full((batch, 1), width - 1, dtype=torch.long, device=self.device),
                past_key_values=payload,
                logits_positions=LAST_POSITION,
            )

    def _log_pool(self, num_blocks: int, bytes_per_block: int, reason: str) -> None:
        _LOGGER.info(
            "KV pool: %d blocks x %d tokens = %.2f GiB (%s)",
            num_blocks, self.config.block_size, num_blocks * bytes_per_block / _GIB, reason,
        )
