"""Sequence — the in-flight representation of a generation request."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from liteinfer.sampling.params import SamplingParams


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "finished_stopped"   # hit a stop string / EOS
    FINISHED_LENGTH = "finished_length"     # reached max_tokens
    FINISHED_ABORTED = "finished_aborted"   # cancelled by the user


@dataclass
class Sequence:
    """A single in-flight token stream."""

    seq_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    @property
    def is_finished(self) -> bool:
        return self.status.name.startswith("FINISHED_")

    def __len__(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


@dataclass
class SequenceGroup:
    """A set of sequences sharing a request id and sampling params.

    A group has multiple sequences when the user asks for `n > 1`
    candidates from the same prompt.
    """

    request_id: str
    sequences: list[Sequence]
    sampling_params: SamplingParams

    @property
    def is_finished(self) -> bool:
        return all(s.is_finished for s in self.sequences)
