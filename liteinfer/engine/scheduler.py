"""Scheduler — picks which sequences run on the next forward pass.

v0 policy is **static batching**: drain up to ``max_num_seqs`` waiting
requests into one batch, run them lockstep until every sequence
finishes, then move on to the next batch. Continuous batching, paged
KV reuse, and prefix-aware reordering are deliberately deferred — each
gets its own subclass behind a flag once the eager baseline is solid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from liteinfer.config import EngineConfig
from liteinfer.engine.sequence import SequenceGroup, SequenceStatus


@dataclass
class SchedulerOutput:
    """Decisions for one scheduling step."""

    scheduled: list[SequenceGroup] = field(default_factory=list)
    """Sequence groups that will run a forward pass this step."""

    is_new_batch: bool = False
    """True on the first step of a new static batch (engine triggers prefill)."""

    preempted: list[SequenceGroup] = field(default_factory=list)
    """Reserved for future variants that evict running sequences. Always
    empty under the static-batching policy."""


class Scheduler:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.waiting: list[SequenceGroup] = []
        self.running: list[SequenceGroup] = []

    def add(self, group: SequenceGroup) -> None:
        """Enqueue a new sequence group. Marked ``WAITING`` until scheduled."""
        for seq in group.sequences:
            seq.status = SequenceStatus.WAITING
        self.waiting.append(group)

    def schedule(self) -> SchedulerOutput:
        """Pick the next batch under the static-batching policy."""
        is_new_batch = False
        if not self.running and self.waiting:
            n = min(len(self.waiting), self.config.max_num_seqs)
            batch, self.waiting = self.waiting[:n], self.waiting[n:]
            for group in batch:
                for seq in group.sequences:
                    seq.status = SequenceStatus.RUNNING
            self.running = batch
            is_new_batch = True
        return SchedulerOutput(scheduled=list(self.running), is_new_batch=is_new_batch)

    def remove_finished(self) -> list[SequenceGroup]:
        """Drop finished groups from the running batch and return them.

        Called once per step by the engine. Under the static policy the
        next batch only starts after this list empties the running set.
        """
        finished, self.running = (
            [g for g in self.running if g.is_finished],
            [g for g in self.running if not g.is_finished],
        )
        return finished

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)
