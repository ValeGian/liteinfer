"""ContinuousScheduler — iteration-level scheduling for continuous batching.

Every step hands out a token budget. Each sequence carries `num_computed_tokens`,
and what it is owed is the rest of its length: the remainder of the prompt for
a sequence still prefilling, one token for one that is decoding. The scheduler
grants each sequence what it is owed clamped by what is left of the budget, and
admits waiting sequences into free slots while any is left. There are no phases
here — a decode and a prefill are the same request with different counts.

Two caps, and both bind: `max_num_seqs` bounds how many sequences run at once,
`EngineConfig.token_budget` how many tokens one step computes across them. A
prompt larger than what is left of the budget is not deferred, it is *chunked*:
it gets what is left, and the rest on later steps.

Running sequences are served before waiting ones, in the order they were
admitted. A chunked prompt takes everything left of the budget, so nothing is
admitted behind it until it completes — which means at most one sequence is ever
part-way through its prompt, and it is the last one running. With the budget at
least `max_num_seqs`, the decodes ahead of it can never take all of it.

This is the loop at the top of vLLM's scheduler and nothing else: no
preemption, no speculative tokens, no encoder budget.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from liteinfer.config import EngineConfig
from liteinfer.engine.sequence import Sequence, SequenceStatus


@dataclass
class ContinuousSchedulerOutput:
    """Decisions for one continuous-batching step."""

    seqs: list[Sequence] = field(default_factory=list)
    """Every sequence that computes tokens this step, in scheduling order."""

    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    """Tokens each of `seqs` computes this step, by request id."""

    @property
    def is_empty(self) -> bool:
        return not self.seqs


class ContinuousScheduler:
    """Iteration-level scheduler: spends a token budget every step.

    * Admits waiting sequences whenever ``len(running) < max_num_seqs`` and some
      of the step's token budget is left.
    * ``remove_finished`` evicts individual finished sequences immediately
      rather than waiting for the whole batch to complete.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, sequence: Sequence) -> None:
        sequence.status = SequenceStatus.WAITING
        self.waiting.append(sequence)

    def schedule(self) -> ContinuousSchedulerOutput:
        """Grant running sequences what they are owed, then admit waiting ones with the rest."""
        out = ContinuousSchedulerOutput()
        budget = self.config.token_budget

        for seq in self.running:
            if budget == 0:
                break
            if not seq.is_finished:
                budget -= self._grant(out, seq, budget)

        while budget > 0 and self.waiting and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            budget -= self._grant(out, seq, budget)

        return out

    @staticmethod
    def _grant(out: ContinuousSchedulerOutput, seq: Sequence, budget: int) -> int:
        """Schedule what `seq` is owed, clamped to `budget`, and return what it took."""
        num_tokens = min(seq.num_uncomputed_tokens, budget)
        # Every unfinished sequence owes at least its sampled token; zero means the
        # engine did not record what the last forward computed.
        assert num_tokens > 0, f"{seq.request_id} was scheduled with nothing to compute"
        out.seqs.append(seq)
        out.num_scheduled_tokens[seq.request_id] = num_tokens
        return num_tokens

    def remove_finished(self) -> list[Sequence]:
        """Evict individually-finished sequences. Does not wait for the whole batch."""
        finished = [s for s in self.running if s.is_finished]
        self.running = [s for s in self.running if not s.is_finished]
        return finished

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)
