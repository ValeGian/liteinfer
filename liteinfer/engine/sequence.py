"""Sequence — the in-flight representation of a generation request."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from liteinfer.sampling.params import SamplingParams


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "finished_stopped"  # hit a stop string / EOS
    FINISHED_LENGTH = "finished_length"  # reached max_tokens
    FINISHED_ABORTED = "finished_aborted"  # cancelled by the user


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

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids


@dataclass
class SequenceGroup:
    """A set of sequences sharing a request id and sampling params.

    A group has multiple sequences when the user asks for `n > 1`
    candidates from the same prompt. v0 supports `n == 1` only.
    """

    request_id: str
    sequences: list[Sequence]
    sampling_params: SamplingParams
    prompt: str = ""

    @property
    def is_finished(self) -> bool:
        return all(s.is_finished for s in self.sequences)

    @property
    def primary(self) -> Sequence:
        """The single in-flight sequence under the v0 ``n == 1`` constraint."""
        if len(self.sequences) != 1:
            raise RuntimeError("v0 only supports n=1 per request")
        return self.sequences[0]
