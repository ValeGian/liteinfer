"""Scheduler — iteration-level scheduling for continuous batching.

Every step hands out a token budget. Each sequence carries `num_computed_tokens`,
and what it is owed is the rest of its length: the remainder of the prompt for
a sequence still prefilling, one token for one that is decoding. The scheduler
grants each sequence what it is owed clamped by what is left of the budget, and
admits waiting sequences into free slots while any is left. There are no phases
here — a decode and a prefill are the same request with different counts.

Three caps, and all bind: `max_num_seqs` bounds how many sequences run at once,
`EngineConfig.token_budget` how many tokens one step computes across them, and
the KV blocks free in the pool what those tokens can be written to. A prompt
larger than what is left of the budget is not deferred, it is *chunked*: it gets
what is left, and the rest on later steps.

Running sequences are served before waiting ones, in the order they were
admitted. A chunked prompt takes everything left of the budget, so nothing is
admitted behind it until it completes — which means at most one sequence is ever
part-way through its prompt, and it is the last one running. With the budget at
least `max_num_seqs`, the decodes ahead of it can never take all of it.

Blocks
------
The free blocks are counted down as the step hands them out, running sequences
first. A running sequence gets what fits: a prompt part-way through takes a
smaller chunk, and one that fits nothing — a decode whose last block is full, a
chunk with no block to start — sits the step out while the others go on, which
is what frees blocks. A waiting prompt is admitted only when its whole grant
fits, and in arrival order: one that does not fit waits, and nothing is admitted
past it, so it cannot be starved by shorter prompts behind it. A prompt no empty
pool could hold is refused, since waiting cannot help it.

When every running sequence sits out, nothing will free a block, so the most
recently admitted one is refused and the rest go on once its blocks return.
vLLM preempts that sequence and recomputes it later instead; that is §1.8's
second half.

This is the loop at the top of vLLM's scheduler and nothing else: no
preemption yet, no speculative tokens, no encoder budget.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

from liteinfer.cache.block_tables import BlockTables
from liteinfer.config import EngineConfig
from liteinfer.engine.sequence import Sequence, SequenceStatus


class Refusal(NamedTuple):
    """A sequence the pool cannot serve, and why — what its caller is told."""

    seq: Sequence
    reason: str


@dataclass
class SchedulerOutput:
    """Decisions for one continuous-batching step."""

    seqs: list[Sequence] = field(default_factory=list)
    """Every sequence that computes tokens this step, in scheduling order."""

    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    """Tokens each of `seqs` computes this step, by request id."""

    refused: list[Refusal] = field(default_factory=list)
    """Sequences the engine fails this step for want of KV blocks; none of them is in `seqs`."""

    @property
    def is_empty(self) -> bool:
        return not self.seqs


@dataclass
class _Left:
    """What is still to hand out this step."""

    tokens: int
    blocks: int


class Scheduler:
    """Iteration-level scheduler: spends a token budget and the free KV blocks every step.

    * Admits waiting sequences whenever ``len(running) < max_num_seqs``, some of
      the step's token budget is left, and the blocks free hold their grant.
    * ``remove_finished`` evicts individual finished sequences immediately
      rather than waiting for the whole batch to complete, and frees their blocks.
    """

    def __init__(self, config: EngineConfig, blocks: BlockTables) -> None:
        self.config = config
        self._blocks = blocks
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, sequence: Sequence) -> None:
        sequence.status = SequenceStatus.WAITING
        self.waiting.append(sequence)

    def schedule(self) -> SchedulerOutput:
        """Grant running sequences what they are owed and what fits, then admit waiting ones."""
        out = SchedulerOutput()
        left = _Left(tokens=self.config.token_budget, blocks=self._blocks.num_free_blocks)
        is_any_sitting_out = self._serve_running(out, left)
        self._admit_waiting(out, left)
        if out.is_empty and is_any_sitting_out:
            out.refused.append(self._refuse_newest_running())
        return out

    def _serve_running(self, out: SchedulerOutput, left: _Left) -> bool:
        """Grant each running sequence what fits; return whether any fit nothing and sat out.

        The block arithmetic and the grant are written out over locals rather
        than through `BlockTables.blocks_short_of` and `_grant`: this is the one
        loop that visits every running sequence every step, and through those
        calls and attribute writes `schedule()` costs twice as much at 128.
        """
        tokens, free, block_size = left.tokens, left.blocks, self._blocks.block_size
        capacity_of = self._blocks.capacity
        is_any_sitting_out = False
        for seq in self.running:
            if tokens == 0:
                break
            if seq.is_finished:
                continue
            computed = seq.num_computed_tokens
            owed = len(seq) - computed
            # Every unfinished sequence owes at least its sampled token; zero means
            # the engine did not record what the last forward computed.
            assert owed > 0, f"{seq.request_id} was scheduled with nothing to compute"
            capacity = capacity_of(seq.request_id)
            num_tokens = min(owed, tokens, capacity + free * block_size - computed)
            if num_tokens == 0:
                is_any_sitting_out = True
                continue
            overflow = computed + num_tokens - capacity
            if overflow > 0:
                free -= -(-overflow // block_size)
            tokens -= num_tokens
            out.seqs.append(seq)
            out.num_scheduled_tokens[seq.request_id] = num_tokens
        left.tokens, left.blocks = tokens, free
        return is_any_sitting_out

    def _admit_waiting(self, out: SchedulerOutput, left: _Left) -> None:
        """Admit waiting sequences in arrival order while slots, budget and blocks allow."""
        while left.tokens > 0 and self.waiting and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            if self._blocks.blocks_for(len(seq.prompt_token_ids)) > self._blocks.num_blocks:
                out.refused.append(self._refuse_unservable(self.waiting.popleft()))
                continue
            num_tokens = min(seq.num_uncomputed_tokens, left.tokens)
            blocks = self._blocks.blocks_short_of(seq.request_id, seq.num_computed_tokens + num_tokens)
            if blocks > left.blocks:
                return
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            self._grant(out, seq, num_tokens, blocks, left)

    def _grant(
        self, out: SchedulerOutput, seq: Sequence, num_tokens: int, blocks: int, left: _Left
    ) -> None:
        """Schedule `num_tokens` of `seq` and take them, and the `blocks` they need, from `left`."""
        # Every unfinished sequence owes at least its sampled token; zero means the
        # engine did not record what the last forward computed.
        assert num_tokens > 0, f"{seq.request_id} was scheduled with nothing to compute"
        left.blocks -= blocks
        left.tokens -= num_tokens
        out.seqs.append(seq)
        out.num_scheduled_tokens[seq.request_id] = num_tokens

    def _refuse_unservable(self, seq: Sequence) -> Refusal:
        needed = self._blocks.blocks_for(len(seq.prompt_token_ids))
        return Refusal(
            seq,
            f"{seq.request_id}: its {len(seq.prompt_token_ids)}-token prompt needs {needed} KV "
            f"blocks and the whole pool has {self._blocks.num_blocks}. Increase num_gpu_blocks "
            "or gpu_memory_utilization, or shorten the prompt.",
        )

    def _refuse_newest_running(self) -> Refusal:
        """The most recently admitted unfinished sequence, refused so the rest can grow."""
        seq = next(seq for seq in reversed(self.running) if not seq.is_finished)
        return Refusal(
            seq,
            f"{seq.request_id}: the KV pool is full ({self._blocks.num_blocks} blocks of "
            f"{self._blocks.block_size} tokens) and no running sequence can grow, so the most "
            "recently admitted one is refused. Increase num_gpu_blocks or reduce max_num_seqs "
            "/ max_model_len.",
        )

    def remove_finished(self) -> list[Sequence]:
        """Evict individually-finished sequences and free their blocks. Does not wait for the whole batch."""
        finished = [s for s in self.running if s.is_finished]
        self.running = [s for s in self.running if not s.is_finished]
        for seq in finished:
            self._blocks.release(seq.request_id)
        return finished

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)
