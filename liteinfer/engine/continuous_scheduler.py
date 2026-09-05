"""ContinuousScheduler — iteration-level scheduling for continuous batching.

Unlike the static ``Scheduler``, which holds a batch until every member
finishes, the continuous scheduler admits new sequences into free slots on
every scheduling call and evicts finished sequences individually.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from liteinfer.config import EngineConfig
from liteinfer.engine.sequence import Sequence, SequenceStatus


@dataclass
class ContinuousSchedulerOutput:
    """Decisions for one continuous-batching step."""

    prefill_seqs: list[Sequence] = field(default_factory=list)
    """Newly admitted sequences that need a prefill pass this step."""

    decode_seqs: list[Sequence] = field(default_factory=list)
    """Sequences already past prefill that need a decode pass this step."""

    @property
    def all_seqs(self) -> list[Sequence]:
        return self.prefill_seqs + self.decode_seqs


class ContinuousScheduler:
    """Iteration-level scheduler: fills empty slots every step.

    Key differences from ``Scheduler``:
    * Admits new sequences whenever ``len(running) < max_num_seqs``.
    * ``remove_finished`` evicts individual finished sequences immediately
      rather than waiting for the whole batch to complete.
    * ``schedule`` returns separate ``prefill_seqs`` and ``decode_seqs``
      lists so the engine can issue two targeted forward passes.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add(self, sequence: Sequence) -> None:
        sequence.status = SequenceStatus.WAITING
        self.waiting.append(sequence)

    def schedule(self) -> ContinuousSchedulerOutput:
        """Fill empty slots from waiting, then split running into prefill/decode."""
        available = self.config.max_num_seqs - len(self.running)
        if available > 0 and self.waiting:
            n = min(available, len(self.waiting))
            admitted, self.waiting = self.waiting[:n], self.waiting[n:]
            for seq in admitted:
                seq.status = SequenceStatus.RUNNING
            self.running.extend(admitted)

        prefill_seqs = [s for s in self.running if not s.is_finished and not s.output_token_ids]
        decode_seqs = [s for s in self.running if not s.is_finished and s.output_token_ids]
        return ContinuousSchedulerOutput(prefill_seqs=prefill_seqs, decode_seqs=decode_seqs)

    def remove_finished(self) -> list[Sequence]:
        """Evict individually-finished sequences. Does not wait for the whole batch."""
        finished = [s for s in self.running if s.is_finished]
        self.running = [s for s in self.running if not s.is_finished]
        return finished

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)
