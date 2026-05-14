"""Shared types for the async inference API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamEvent:
    """One streaming generation event from ``AsyncLLM.stream()``.

    Each event carries the *cumulative* output so far. Consumers can read
    ``text`` for the full decoded string or ``output_token_ids`` for the token
    list. The final event has ``is_finished=True`` and a non-``None``
    ``finish_reason``.
    """

    request_id: str
    prompt: str
    output_token_ids: list[int]
    text: str
    is_finished: bool
    finish_reason: str | None  # "stop" | "length" | "abort" | None if still running
