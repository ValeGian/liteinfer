"""Tokenizer — thin wrapper around HuggingFace `AutoTokenizer`.

Centralizing the wrapper keeps engine code free of HF-specific imports
and gives one place to swap implementations later (custom BPE, faster
tokenizers, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


class Tokenizer:
    """Encode/decode text. Backed by HF `PreTrainedTokenizerFast` when available."""

    def __init__(self, model_dir: str | Path) -> None:
        self._hf: Any = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        # `eos_token_id` may be a list on multimodal models; reduce to a single id
        # for the v0 stop-criterion logic. Stored as a tuple so callers can
        # iterate without copying.
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
