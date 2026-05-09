"""Scheduler — v0 policy: static batching."""

from __future__ import annotations

from dataclasses import dataclass, field

from liteinfer.config import EngineConfig
from liteinfer.engine.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    """Decisions for one scheduling step."""

    scheduled: list[Sequence] = field(default_factory=list)
    """Sequences that will run a forward pass this step."""

    is_new_batch: bool = False
    """True on the first step of a new static batch (engine triggers prefill)."""

    preempted: list[Sequence] = field(default_factory=list)
    """Reserved for future variants that evict running sequences. Always
    empty under the static-batching policy."""


class Scheduler:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add(self, sequence: Sequence) -> None:
        sequence.status = SequenceStatus.WAITING
        self.waiting.append(sequence)

    def schedule(self) -> SchedulerOutput:
        is_new_batch = False
        if not self.running and self.waiting:
            n = min(len(self.waiting), self.config.max_num_seqs)
            batch, self.waiting = self.waiting[:n], self.waiting[n:]
            for seq in batch:
                seq.status = SequenceStatus.RUNNING
            self.running = batch
            is_new_batch = True
        return SchedulerOutput(scheduled=list(self.running), is_new_batch=is_new_batch)

    def remove_finished(self) -> list[Sequence]:
        """Drain the running batch only when *every* member has finished.

        Strict static-batching policy: a batch enters together (one prefill),
        decodes together, and leaves together. Members that finish early stay
        in ``running`` so the runner keeps a stable shape until the slowest
        sequence is done.
        """
        if not self.running or any(not s.is_finished for s in self.running):
            return []
        finished, self.running = self.running, []
        return finished

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)
