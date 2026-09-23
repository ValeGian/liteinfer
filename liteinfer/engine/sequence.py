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


_FINISHED_STATUSES = frozenset(
    {
        SequenceStatus.FINISHED_STOPPED,
        SequenceStatus.FINISHED_LENGTH,
        SequenceStatus.FINISHED_ABORTED,
    }
)
"""Membership rather than a name prefix: this is read seven times per sequence per
step — 229,376 times in one wide-batch run — and building `.name` to match a
string prefix made that measurable. Naming the states also stops the check
depending on how they happen to be spelled."""


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
    num_computed_tokens: int = 0
    """Tokens whose K/V are in the cache. What is left of `len(self)` is what the
    next step has to compute: the rest of the prompt, or the one sampled token."""

    @property
    def output_text(self) -> str:
        """Text decoded so far. Advanced once per step, not rebuilt from scratch."""
        return self.detokenizer.text

    @property
    def is_finished(self) -> bool:
        return self.status in _FINISHED_STATUSES

    def __len__(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        """Tokens a forward still has to run before this sequence can sample again."""
        return len(self) - self.num_computed_tokens

    @property
    def is_prompt_computed(self) -> bool:
        """Whether the whole prompt is cached, so each step computes one sampled token."""
        return self.num_computed_tokens >= len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids
