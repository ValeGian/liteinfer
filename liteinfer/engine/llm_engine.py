"""LLMEngine — orchestrates scheduling, execution, and KV cache lifecycle."""

from __future__ import annotations

from liteinfer.config import EngineConfig
from liteinfer.engine.model_runner import ModelRunner
from liteinfer.engine.scheduler import Scheduler
from liteinfer.engine.sequence import SequenceGroup
from liteinfer.sampling.params import SamplingParams


class LLMEngine:
    """The core inference engine.

    Owns the lifecycle of every request from arrival to completion. Each
    `step()` call runs at most one forward pass: ask the scheduler what
    to run, ask the model runner to run it, route outputs back to the
    originating sequences.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = ModelRunner(config)

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> None:
        """Tokenize, wrap as a `SequenceGroup`, and enqueue with the scheduler."""
        raise NotImplementedError

    def step(self) -> list[SequenceGroup]:
        """Run one schedule + forward iteration. Return finished sequence groups."""
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        raise NotImplementedError
