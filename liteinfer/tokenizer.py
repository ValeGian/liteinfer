"""Tokenizer — thin wrapper around HuggingFace `AutoTokenizer`, plus incremental decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from transformers import AutoTokenizer

_REPLACEMENT_CHAR = "\ufffd"


class SupportsDecode(Protocol):
    """All `IncrementalDetokenizer` needs of a tokenizer."""

    def decode(self, token_ids: list[int]) -> str: ...


class Tokenizer:
    def __init__(self, model_dir: str | Path) -> None:
        self._hf: Any = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        # Multimodal models can return a list; normalize to tuple of ints.
        eos = self._hf.eos_token_id
        if isinstance(eos, list):
            self.eos_token_ids: tuple[int, ...] = tuple(eos)
        elif eos is None:
            self.eos_token_ids = ()
        else:
            self.eos_token_ids = (int(eos),)

    @property
    def vocab_size(self) -> int:
        return int(self._hf.vocab_size)

    def encode(self, text: str) -> list[int]:
        """Encode without adding special tokens; the engine controls templating."""
        return self._hf.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self._hf.decode(token_ids, skip_special_tokens=True)


@dataclass
class IncrementalDetokenizer:
    """The text of a growing token sequence, without re-decoding the whole prefix.

    Decoding only the newest token would be wrong twice over: a UTF-8 character
    can span two tokens, and a byte-level BPE piece renders differently
    depending on what precedes it. So each step decodes a short *window* ending
    at the new token, decodes the same window one token shorter, and keeps the
    difference — which is exactly the text the new token added.

    When the window ends mid-character the decoder emits U+FFFD; that is not
    text yet, so the window is left to grow and the next token completes it.
    """

    text: str = ""
    prefix_offset: int = 0
    read_offset: int = 0

    def update(self, tokenizer: SupportsDecode, token_ids: list[int]) -> str:
        """Fold every token after `read_offset` into `text`, and return it."""
        prefix = tokenizer.decode(token_ids[self.prefix_offset : self.read_offset])
        whole = tokenizer.decode(token_ids[self.prefix_offset :])
        if len(whole) <= len(prefix) or whole.endswith(_REPLACEMENT_CHAR):
            return self.text  # mid-character: wait for the token that completes it

        self.text += whole[len(prefix) :]
        self.prefix_offset = self.read_offset
        self.read_offset = len(token_ids)
        return self.text
