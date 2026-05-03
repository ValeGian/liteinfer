"""Scheduler — chooses which sequences run on the next forward pass.

Continuous-batching aware. Will eventually be prefix-cache aware: with
prefix caching enabled, scheduling decisions depend on which prefixes
are already resident in the KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from liteinfer.config import EngineConfig
from liteinfer.engine.sequence import SequenceGroup


@dataclass
class SchedulerOutput:
    """Decisions for one scheduling step."""

    scheduled: list[SequenceGroup] = field(default_factory=list)
    """Sequence groups that will run a forward pass this step."""

    preempted: list[SequenceGroup] = field(default_factory=list)
    """Sequences evicted from the running set (e.g., to free KV blocks)."""


class Scheduler:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.waiting: list[SequenceGroup] = []
        self.running: list[SequenceGroup] = []

    def add(self, group: SequenceGroup) -> None:
        """Enqueue a new sequence group for scheduling."""
        raise NotImplementedError

    def schedule(self) -> SchedulerOutput:
        """Pick the next batch under token / sequence / KV-block budgets."""
        raise NotImplementedError
