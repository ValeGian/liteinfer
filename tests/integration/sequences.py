"""Requests for integration tests, built the same way wherever a test needs one."""

from __future__ import annotations

from liteinfer.engine.sequence import Sequence, SequenceStatus
from liteinfer.sampling.params import SamplingParams


def prompt_text(length: int, offset: int) -> str:
    """A prompt of `length` tiny-model tokens, varied by `offset` so prompts differ."""
    return " ".join(f"tok{2 + (offset + i) % 200}" for i in range(length))


def running_sequence(request_id: str, prompt_len: int, offset: int) -> Sequence:
    """A greedy sequence already admitted, for driving the runner without the engine."""
    return Sequence(
        request_id=request_id,
        prompt="",
        prompt_token_ids=[2 + (offset + i) % 250 for i in range(prompt_len)],
        sampling_params=SamplingParams(temperature=0.0),
        status=SequenceStatus.RUNNING,
    )
