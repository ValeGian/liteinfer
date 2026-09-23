"""ContinuousModelRunner — forward-pass execution for continuous batching.

* ``prefill(seqs, num_tokens)`` — the next chunk of each prompt, which is the
  whole prompt unless the scheduler's token budget split it.
* ``decode(seqs)`` — single-token pass for sequences already past prefill.

A step where new and running sequences coexist issues both, rather than one
mixed pass, which would need a flash-attention-style kernel. See roadmap §1.3.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, NamedTuple

import torch

from liteinfer.cache.block_pool import BlockPool
from liteinfer.cache.continuous_kv_cache import ContinuousKVCache, KVPayload, ProfilePayload
from liteinfer.config import EngineConfig
from liteinfer.engine.attention_mask import builders_for
from liteinfer.engine.cuda_graphs import DecodeGraphs, graphs_are_enabled
from liteinfer.engine.sequence import Sequence
from liteinfer.hub import resolve_model_path
from liteinfer.models import LAST_POSITION
from liteinfer.models.attention import reads_paged_kv, varlen_unsupported_reason
from liteinfer.models.loader import load_hf_model
from liteinfer.models.paged_decode import choose_num_splits
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


def _packing_is_enabled(
    requested: bool | None, implementation: str, device: torch.device, dtype: torch.dtype
) -> bool:
    """Whether prefill packs its batch, from the config and what the device can run.

    `None` packs wherever the preconditions hold, which is the same rule
    `enable_cuda_graphs` follows. An explicit `True` that cannot run raises
    rather than quietly padding: a benchmark row that asks for the packed path
    has to get it, or hear why it could not.
    """
    if requested is False:
        return False
    reason = varlen_unsupported_reason(implementation, device, dtype)
    if reason is None:
        return True
    if requested is True:
        raise ValueError(f"enable_packed_prefill was asked for but {reason}")
    return False


def _last_token_indices(chunk_lens: list[int], device: torch.device) -> torch.Tensor:
    """Where each chunk's last token sits in the packed run.

    Sampling reads one row per sequence, and in a packed batch those rows are at
    the end of each chunk rather than in a shared last column.
    """
    ends = torch.tensor(chunk_lens, device=device).cumsum(0)
    return ends - 1


class _PromptChunk(NamedTuple):
    """The prompt tokens one sequence computes in a prefill pass, and where they start."""

    token_ids: list[int]
    start: int

    @property
    def end(self) -> int:
        """One past the chunk's last position, which is what the sequence holds afterwards."""
        return self.start + len(self.token_ids)


def _head_dim(hf_config: PretrainedConfig) -> int:
    """Head dimension, which most configs state and the rest imply."""
    return getattr(
        hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
    )


class ContinuousModelRunner:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = config.resolved_device()
        self.model: torch.nn.Module | None = None
        self.hf_config = None
        self.tokenizer: Tokenizer | None = None
        self._cache: ContinuousKVCache | None = None
        self._graphs: DecodeGraphs | None = None
        self._forward_bytes = 0
        self._packed_prefill_enabled = False

    def load_model(self) -> None:
        model_path = resolve_model_path(self.config.model)
        self.model, self.hf_config = load_hf_model(self.config, model_path)
        self.tokenizer = Tokenizer(model_path)
        # After the model, because the kernel is what decides whether a packed
        # batch can be read at all, and `load_hf_model` is where it is resolved.
        self._packed_prefill_enabled = _packing_is_enabled(
            self.config.enable_packed_prefill,
            self.attn_implementation,
            self.device,
            self.config.dtype,
        )
        # Measured before the pool exists, because the pool gets whatever the
        # forward turns out not to need.
        self._forward_bytes = self._profile_forward_bytes()
        self._cache = ContinuousKVCache(self._create_block_pool())
        self._graphs = self._create_decode_graphs()

    @property
    def _packs_prefill(self) -> bool:
        """Whether prefill goes in packed, resolved once at load rather than per step."""
        return self._packed_prefill_enabled

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

    def deregister_sequence(self, seq: Sequence) -> None:
        """Free paged KV blocks allocated for a finished sequence."""
        assert self._cache is not None
        self._cache.deregister(seq.request_id)

    @torch.inference_mode()
    def prefill(self, seqs: list[Sequence], num_tokens: list[int] | None = None) -> torch.Tensor:
        """Prefill pass over the next `num_tokens` prompt tokens of each sequence.

        Each sequence continues where the cache left off, so a prompt chunked
        across steps is this call made once per chunk; the first one registers
        the sequence. `None` prefills the rest of every prompt, which for a
        sequence the cache has not seen is all of it.

        Returns logits ``[B, vocab_size]`` at each chunk's last token. Only a
        chunk that ends its prompt has a next token worth sampling, and the
        caller is the one that knows which rows those are.
        """
        assert self._cache is not None and self.model is not None

        request_ids = [s.request_id for s in seqs]
        for request_id in request_ids:
            if not self._cache.is_registered(request_id):
                self._cache.register(request_id)
        chunks = self._next_prompt_chunks(seqs, num_tokens)
        counts = [len(chunk.token_ids) for chunk in chunks]
        self._cache.advance(request_ids, counts)

        if self._packs_prefill:
            return self._packed_prefill(request_ids, chunks)

        input_ids, position_ids = self._build_prefill_inputs(chunks)
        build_prefill, _ = builders_for(type(self.model).__name__)
        attention_mask = build_prefill(
            counts, self.config.dtype, self.device, [chunk.end for chunk in chunks]
        )
        payload = self._cache.make_prefill_payload(request_ids, counts)
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=payload,
            attention_mask=attention_mask,
            logits_positions=LAST_POSITION,
        )
        return out.logits[:, -1, :]  # left-padded, so the last column is the last real token

    def _next_prompt_chunks(
        self, seqs: list[Sequence], num_tokens: list[int] | None
    ) -> list[_PromptChunk]:
        """The prompt tokens each sequence computes next, starting after what is cached.

        A chunk that runs past its prompt is refused rather than truncated: the
        cache would account for tokens the pass never wrote.
        """
        assert self._cache is not None
        chunks = []
        for i, seq in enumerate(seqs):
            start = self._cache.seq_total_len(seq.request_id)
            prompt_len = len(seq.prompt_token_ids)
            count = prompt_len - start if num_tokens is None else num_tokens[i]
            if count < 1 or start + count > prompt_len:
                raise ValueError(
                    f"{seq.request_id}: cannot prefill {count} tokens from position {start} "
                    f"of a {prompt_len}-token prompt"
                )
            chunks.append(_PromptChunk(seq.prompt_token_ids[start : start + count], start))
        return chunks

    def _packed_prefill(self, request_ids: list[str], chunks: list[_PromptChunk]) -> torch.Tensor:
        """Prefill the same chunks as one flat run of tokens, with no padding.

        A padded batch computes `len(seqs) * max(prompt_lens)` positions to keep
        `sum(prompt_lens)` of them. On prompts whose lengths vary — which is what
        real traffic is — that ratio reaches 13.4x at 32 sequences, and it is
        entirely wasted work plus a mask to hide it afterwards.
        """
        assert self._cache is not None and self.model is not None

        counts = [len(chunk.token_ids) for chunk in chunks]
        input_ids, position_ids = self._build_packed_prefill_inputs(chunks)
        payload = self._cache.make_packed_prefill_payload(request_ids, counts)
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=payload,
            attention_mask=None,
            logits_positions=_last_token_indices(counts, self.device),
        )
        return out.logits[0]

    def _build_packed_prefill_inputs(
        self, chunks: list[_PromptChunk]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every chunk end to end as `[1, total_tokens]`, positions restarting per chunk.

        The leading axis is 1 because the batch is the token run itself; where
        each sequence begins lives in `cu_seqlens_q` on the payload, and in the
        positions, which count from where each chunk starts in its prompt so RoPE
        sees every token where it sits.
        """
        token_ids = [token for chunk in chunks for token in chunk.token_ids]
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        position_ids = torch.cat(
            [torch.arange(chunk.start, chunk.end, device=self.device) for chunk in chunks]
        ).unsqueeze(0)
        return input_ids, position_ids

    @torch.inference_mode()
    def decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """Decode pass for sequences already past their prefill step.

        Feeds one token per sequence (the last sampled token) and returns
        logits ``[B, vocab_size]`` for sampling the next token.
        """
        assert self._cache is not None and self.model is not None

        request_ids = [s.request_id for s in seqs]
        input_ids, position_ids = self._build_decode_inputs(seqs)

        # Account for this step's token before addressing it, then build every
        # address here rather than on the first layer: the forward pass must
        # contain no host-side work. The write is one slot per sequence — a
        # packed run where every count is 1 — and only the read keeps the
        # right-aligned table, because `paged_decode` walks one row per sequence.
        one_each = [1] * len(seqs)
        self._cache.advance(request_ids, one_each)
        write_slots = self._cache.slot_mapping_for(request_ids, one_each)
        slots = self._cache.slot_table_for(request_ids)

        if self._graphs is not None and self._graphs.has_capacity_for(len(seqs)):
            return self._graphs.run(
                input_ids,
                position_ids,
                write_slots,
                slots,
                self._cache.context_lens_for(request_ids),
            )

        payload, attention_mask = self._build_decode_kv(request_ids, write_slots, slots)
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=payload,
            attention_mask=attention_mask,
            logits_positions=LAST_POSITION,
        )
        return out.logits[:, -1, :]

    # ------------------------------------------------------------------
    # Input builders
    # ------------------------------------------------------------------

    def _build_decode_kv(
        self, request_ids: list[str], write_slots: torch.Tensor, slots: torch.Tensor
    ) -> tuple[KVPayload, torch.Tensor | None]:
        """Pair this step's KV payload with the mask its attention kernel needs.

        The paged kernel stops each sequence at its own context length, so there
        is no padding to hide and no mask to build. The dense kernels read a
        gather padded out to the longest sequence in the batch, so there is.
        """
        assert self._cache is not None
        if reads_paged_kv(self.attn_implementation):
            context_lens = self._cache.context_lens_for(request_ids)
            payload = self._cache.make_paged_decode_payload(
                write_slots, slots, context_lens, self.splits_for_width(len(request_ids))
            )
            return payload, None

        # The cache's token counts already include this step's token, and they are
        # what addressed the slots above — so the mask is built from the same source.
        seq_total_lens = [self._cache.seq_total_len(rid) for rid in request_ids]
        _, build_decode = builders_for(type(self.model).__name__)
        return (
            self._cache.make_decode_payload(write_slots, slots),
            build_decode(seq_total_lens, self.config.dtype, self.device),
        )

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

    def _build_prefill_inputs(self, chunks: list[_PromptChunk]) -> tuple[torch.Tensor, torch.Tensor]:
        """Every chunk left-padded to the longest, positions counting from where it starts."""
        max_len = max(len(chunk.token_ids) for chunk in chunks)
        shape = (len(chunks), max_len)
        input_ids = torch.zeros(shape, dtype=torch.long, device=self.device)
        position_ids = torch.zeros(shape, dtype=torch.long, device=self.device)
        for i, chunk in enumerate(chunks):
            offset = max_len - len(chunk.token_ids)
            input_ids[i, offset:] = torch.tensor(chunk.token_ids, dtype=torch.long, device=self.device)
            position_ids[i, offset:] = torch.arange(chunk.start, chunk.end, device=self.device)
        return input_ids, position_ids

    def _build_decode_inputs(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        last_tokens = [s.output_token_ids[-1] for s in seqs]
        positions = [len(s.prompt_token_ids) + len(s.output_token_ids) - 1 for s in seqs]
        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        position_ids = torch.tensor(positions, dtype=torch.long, device=self.device).unsqueeze(1)
        return input_ids, position_ids

    def _create_decode_graphs(self) -> DecodeGraphs | None:
        """Build the capture cache, or `None` where decode cannot be captured.

        Built here rather than lazily in `decode` because the preconditions are
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
        the fraction is what covers both. Replacing it with a profiled figure is
        the rest of §2.6, and needs the worst-case forward to be bounded first.
        """
        if self.device.type == "cuda":
            total = torch.cuda.get_device_properties(self.device).total_memory
            resident = torch.cuda.memory_allocated(self.device)
        else:
            total, resident = _CPU_NOMINAL_TOTAL_BYTES, 0
        budget = int(total * self.config.gpu_memory_utilization) - resident
        return max(0, budget - self._forward_bytes) // bytes_per_block

    def _profile_forward_bytes(self) -> int:
        """Peak allocation the widest prefill needs, beyond the weights.

        `schedule()` admits up to `max_num_seqs` sequences at once and prefills
        them in a single pass, so `max_num_seqs x max_model_len` tokens in one
        forward is not hypothetical — it is what a cold engine does when that many
        requests are already waiting. Running it here turns the activation budget
        from a fraction someone guessed into a number this device measured, which
        is what the fraction was standing in for. On Llama-3.2-1B at 32 x 4,096 it
        is **9.04 GiB**, and it was 32.82 GiB before §3.7 stopped the LM head
        running over every position — which is why that had to land first.

        The payload does not write to a cache, so this needs no pool: prefill
        attention reads the K/V the pass just computed, and `ProfilePayload` hands
        exactly that back.

        A forward that will not fit is not fatal. It means the configuration
        cannot serve its own worst case, which is worth saying rather than
        crashing on — sizing falls back to the fraction alone, and the WARNING in
        `_compute_num_blocks` still reports what the pool can serve.
        """
        if self.device.type != "cuda":
            return 0
        assert self.model is not None

        batch, length = self.config.max_num_seqs, self.config.max_model_len
        try:
            # The first forward in a process allocates cuBLAS workspaces that every
            # later one reuses, and they land in the peak. Unwarmed, a narrow config
            # measured 8.49 MiB where a config sixteen times wider measured 5.77 —
            # so the first engine in a process would have been handed the smallest
            # pool. One throwaway pass absorbs that.
            self._prefill_forward(1, min(length, _PROFILE_WARMUP_TOKENS))
            torch.cuda.reset_peak_memory_stats(self.device)
            before = torch.cuda.memory_allocated(self.device)
            self._prefill_forward(batch, length)
            measured = torch.cuda.max_memory_allocated(self.device) - before
        except torch.OutOfMemoryError:
            _LOGGER.warning(
                "the widest prefill this config allows — %d sequences x %d tokens — does "
                "not fit, so the KV pool is sized without a measured activation budget. "
                "Lower max_num_seqs or max_model_len to serve that case.",
                batch, length,
            )
            measured = 0
        finally:
            torch.cuda.empty_cache()

        _LOGGER.info(
            "widest prefill (%d x %d) needs %.2f GiB of activations",
            batch, length, measured / _GIB,
        )
        return max(0, measured)

    def _prefill_forward(self, batch: int, length: int) -> None:
        """One prefill-shaped forward over dummy tokens, writing to no cache."""
        assert self.model is not None
        build_prefill, _ = builders_for(type(self.model).__name__)
        with torch.inference_mode():
            self.model(
                input_ids=torch.zeros((batch, length), dtype=torch.long, device=self.device),
                position_ids=torch.arange(length, device=self.device).expand(batch, length),
                past_key_values=ProfilePayload(),
                attention_mask=build_prefill([length] * batch, self.config.dtype, self.device),
                logits_positions=LAST_POSITION,
            )

    def _log_pool(self, num_blocks: int, bytes_per_block: int, reason: str) -> None:
        _LOGGER.info(
            "KV pool: %d blocks x %d tokens = %.2f GiB (%s)",
            num_blocks, self.config.block_size, num_blocks * bytes_per_block / _GIB, reason,
        )
