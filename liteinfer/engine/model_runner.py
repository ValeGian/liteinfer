# pyright: reportPrivateImportUsage=false
"""ModelRunner — runs one forward pass for a scheduled batch.

Owns the model weights, tokenizer, and (for the eager path) the active
KV cache. Acts as the seam between the engine and the modeling code:
the engine deals in `Sequence` objects; the runner translates them into
tensors and back.

v0 limitation: batch size = 1. Variable-length static batching plus
finish-mask bookkeeping is the first optimization to add; doing it
correctly with the eager cache requires careful padding/positions and
is intentionally deferred so the first end-to-end path stays small and
obviously correct.
"""

from __future__ import annotations

import torch

from liteinfer.cache.eager_kv_cache import EagerKVCache
from liteinfer.cache.kv_cache import KVCache
from liteinfer.config import EngineConfig
from liteinfer.engine.sequence import SequenceGroup
from liteinfer.models.loader import load_hf_model
from liteinfer.tokenizer import Tokenizer


class ModelRunner:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = config.resolved_device()
        self.model: torch.nn.Module | None = None
        self.hf_config = None
        self.tokenizer: Tokenizer | None = None

        # Per-static-batch state. Reset on `start_batch` / `end_batch`.
        self._cache: KVCache | None = None
        self._batch: list[SequenceGroup] = []

    def load_model(self) -> None:
        """Materialize model weights and tokenizer on the target device."""
        self.model, self.hf_config = load_hf_model(self.config)
        self.tokenizer = Tokenizer(self.config.model)

    def start_batch(self, scheduled: list[SequenceGroup]) -> None:
        """Register a static batch and (re)initialize cache state."""
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
        """Run one forward pass.

        Returns
        -------
        logits : tensor of shape ``[batch, vocab_size]`` — the next-token
            logits for each scheduled sequence.
        input_tokens : how many tokens were fed into the forward pass
            in total. Used by the metrics layer.
        """
        if self.model is None:
            raise RuntimeError("model not loaded; call load_model() first")
        if scheduled != self._batch:
            raise RuntimeError("scheduled batch differs from registered batch")

        seq = scheduled[0].primary
        if self.config.cache_mode == "eager":
            return self._execute_eager(seq, is_new_batch)
        return self._execute_no_cache(seq)

    def _execute_eager(self, seq, is_new_batch: bool) -> tuple[torch.Tensor, int]:
        """Eager-cache path: prefill on the first step, single-token decodes after."""
        assert self.model is not None
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
            use_cache=True,
        )
        logits = out.logits[:, -1, :]  # [1, vocab]
        return logits, int(input_ids.shape[1])

    def _execute_no_cache(self, seq) -> tuple[torch.Tensor, int]:
        """No-cache path: feed the full sequence each step, no past KV reused."""
        assert self.model is not None
        all_tokens = seq.all_token_ids()
        input_ids = torch.tensor([all_tokens], dtype=torch.long, device=self.device)
        position_ids = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)

        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        logits = out.logits[:, -1, :]
        return logits, int(input_ids.shape[1])
