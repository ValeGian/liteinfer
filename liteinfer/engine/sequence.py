"""Sequence — the in-flight representation of a generation request."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from liteinfer.sampling.params import SamplingParams
from liteinfer.tokenizer import IncrementalDetokenizer


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "finished_stopped"  # hit a stop string / EOS
    FINISHED_LENGTH = "finished_length"    # reached max_tokens
    FINISHED_ABORTED = "finished_aborted"  # cancelled by the user


@dataclass
class Sequence:
    """In-flight token stream for a single generation request."""

    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    detokenizer: IncrementalDetokenizer = field(default_factory=IncrementalDetokenizer)

    @property
    def output_text(self) -> str:
        """Text decoded so far. Advanced once per step, not rebuilt from scratch."""
        return self.detokenizer.text

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
