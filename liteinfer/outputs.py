"""User-facing result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RequestOutput:
    """A completed generation."""

    request_id: str
    prompt: str
    text: str
    token_ids: list[int]
    finish_reason: str  # "stop" | "length" | "abort"


@dataclass
class StreamEvent:
    """One streaming event from ``AsyncLLM.stream()``.

    Each event carries the *cumulative* output so far: ``text`` for the decoded
    string, ``output_token_ids`` for the token list. The final event has
    ``is_finished=True`` and a non-``None`` ``finish_reason``.
    """

    request_id: str
    prompt: str
    output_token_ids: list[int]
    text: str
    is_finished: bool
    finish_reason: str | None  # "stop" | "length" | "abort" | None while running
