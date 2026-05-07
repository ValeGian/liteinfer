# pyright: reportPrivateImportUsage=false
"""ModelRunner — one forward pass per call. v0 caps batch size at 1."""

from __future__ import annotations

import torch

from liteinfer.cache.eager_kv_cache import EagerKVCache
from liteinfer.cache.kv_cache import KVCache
from liteinfer.config import EngineConfig
from liteinfer.engine.sequence import SequenceGroup
from liteinfer.hub import resolve_model_path
from liteinfer.models.loader import load_hf_model
from liteinfer.tokenizer import Tokenizer


class ModelRunner:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = config.resolved_device()
        self.model: torch.nn.Module | None = None
        self.hf_config = None
        self.tokenizer: Tokenizer | None = None

        self._cache: KVCache | None = None
        self._batch: list[SequenceGroup] = []

    def load_model(self) -> None:
        model_path = resolve_model_path(self.config.model)
        self.model, self.hf_config = load_hf_model(self.config, model_path)
        self.tokenizer = Tokenizer(model_path)

    def start_batch(self, scheduled: list[SequenceGroup]) -> None:
        if len(scheduled) != 1:
            raise NotImplementedError(
                "v0 supports batch size 1; static batching of multiple "
                "sequences is the first scheduled optimization"
            )
        self._batch = scheduled
        if self.config.cache_mode == "eager":
            self._cache = EagerKVCache(self.config, self.hf_config)
        else:
            self._cache = None

    def end_batch(self) -> None:
        if self._cache is not None:
            self._cache.reset()
        self._cache = None
        self._batch = []

    @torch.inference_mode()
    def execute(
        self, scheduled: list[SequenceGroup], is_new_batch: bool
    ) -> tuple[torch.Tensor, int]:
        """Run one forward pass. Returns (logits [batch, vocab], input_tokens)."""
        if scheduled != self._batch:
            raise RuntimeError("scheduled batch differs from registered batch")

        seq = scheduled[0].primary
        if self.config.cache_mode == "eager":
            return self._execute_eager(seq, is_new_batch)
        return self._execute_no_cache(seq)

    def _execute_eager(self, seq, is_new_batch: bool) -> tuple[torch.Tensor, int]:
        cache_payload = self._cache.payload if self._cache is not None else None

        if is_new_batch:
            input_ids = torch.tensor([seq.prompt_token_ids], dtype=torch.long, device=self.device)
            position_ids = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)
        else:
            last_token = seq.output_token_ids[-1]
            input_ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
            position_ids = torch.tensor([[len(seq) - 1]], dtype=torch.long, device=self.device)

        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache_payload,
        )
        logits = out.logits[:, -1, :]
        return logits, int(input_ids.shape[1])

    def _execute_no_cache(self, seq) -> tuple[torch.Tensor, int]:
        all_tokens = seq.all_token_ids()
        input_ids = torch.tensor([all_tokens], dtype=torch.long, device=self.device)
        position_ids = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)

        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=None,
        )
        logits = out.logits[:, -1, :]
        return logits, int(input_ids.shape[1])
