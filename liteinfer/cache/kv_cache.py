"""KV cache abstract base. Threaded through `model.forward(past_key_values=…)`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from liteinfer.config import EngineConfig


class KVCache(ABC):
    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    @abstractmethod
    def reset(self) -> None:
        """Clear cached state. Called between static batches."""

    @abstractmethod
    def get_seq_length(self) -> int:
        """Number of tokens currently cached."""

    @property
    @abstractmethod
    def payload(self) -> Any:
        """Object passed to `model.forward(past_key_values=…)`. Opaque to engine."""
