# pyright: reportPrivateImportUsage=false
"""ModelRunner — one forward pass per call.

Static batching: a batch is registered once via ``start_batch`` and runs
to completion. Variable-length prompts are left-padded for prefill so
all sequences end at the same column; an additive attention mask hides
padded positions on every step. Sequences that finish early stay in the
batch (their sampled tokens are ignored by the engine) until every
member completes.
"""

from __future__ import annotations

import torch

from liteinfer.cache.eager_kv_cache import EagerKVCache
from liteinfer.cache.kv_cache import KVCache
from liteinfer.config import EngineConfig
from liteinfer.engine.attention_mask import build_for_model
from liteinfer.engine.sequence import Sequence
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
        self._batch: list[Sequence] = []
        self._prompt_lens: list[int] = []
        self._max_prompt_len: int = 0

    def load_model(self) -> None:
        model_path = resolve_model_path(self.config.model)
        self.model, self.hf_config = load_hf_model(self.config, model_path)
        self.tokenizer = Tokenizer(model_path)

    def start_batch(self, scheduled: list[Sequence]) -> None:
        if not scheduled:
            raise ValueError("start_batch requires at least one sequence")
        self._batch = scheduled
        self._prompt_lens = [len(seq.prompt_token_ids) for seq in scheduled]
        self._max_prompt_len = max(self._prompt_lens)
        if self.config.cache_mode == "eager":
            self._cache = EagerKVCache(self.config, self.hf_config)
        else:
            self._cache = None

    def end_batch(self) -> None:
        if self._cache is not None:
            self._cache.reset()
        self._cache = None
        self._batch = []
        self._prompt_lens = []
        self._max_prompt_len = 0

    @torch.inference_mode()
    def execute(
        self, scheduled: list[Sequence], is_new_batch: bool
    ) -> tuple[torch.Tensor, int]:
        """Run one forward pass. Returns ``(logits[B, vocab], input_tokens)``."""
        if scheduled != self._batch:
            raise RuntimeError("scheduled batch differs from registered batch")

        if self.config.cache_mode == "eager":
            return self._execute_eager(is_new_batch)
        return self._execute_no_cache()

    # -----------------------------------------------------------------------
    # Eager KV-cache path
    # -----------------------------------------------------------------------

    def _execute_eager(self, is_new_batch: bool) -> tuple[torch.Tensor, int]:
        cache_payload = self._cache.payload if self._cache is not None else None
        if is_new_batch:
            input_ids, position_ids = self._build_prefill_inputs()
            past_len = 0
        else:
            input_ids, position_ids = self._build_decode_inputs()
            past_len = self._max_prompt_len + (self._current_decode_step() - 1)

        attention_mask = build_for_model(
            type(self.model).__name__,
            hf_config=self.hf_config,
            prompt_lens=self._prompt_lens,
            query_len=int(input_ids.shape[1]),
            past_len=past_len,
            dtype=self.config.dtype,
            device=self.device,
        )

        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache_payload,
            attention_mask=attention_mask,
        )
        logits = self._select_last_real_logits(out.logits, is_new_batch)
        return logits, int(input_ids.numel())

    # -----------------------------------------------------------------------
    # Cache-less path: full re-feed every step.
    # -----------------------------------------------------------------------

    def _execute_no_cache(self) -> tuple[torch.Tensor, int]:
        # Build a left-padded batch of all tokens generated so far per seq.
        token_lists = [seq.all_token_ids() for seq in self._batch]
        seq_lens = [len(t) for t in token_lists]
        max_len = max(seq_lens)

        pad_id = 0
        input_ids = torch.full((len(self._batch), max_len), pad_id, dtype=torch.long, device=self.device)
        position_ids = torch.zeros((len(self._batch), max_len), dtype=torch.long, device=self.device)
        for i, tokens in enumerate(token_lists):
            offset = max_len - len(tokens)
            input_ids[i, offset:] = torch.tensor(tokens, dtype=torch.long, device=self.device)
            position_ids[i, offset:] = torch.arange(len(tokens), device=self.device)

        attention_mask = build_for_model(
            type(self.model).__name__,
            hf_config=self.hf_config,
            prompt_lens=seq_lens,
            query_len=max_len,
            past_len=0,
            dtype=self.config.dtype,
            device=self.device,
        )

        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=None,
            attention_mask=attention_mask,
        )
        logits = out.logits[:, -1, :]
        return logits, int(input_ids.numel())

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _build_prefill_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad all prompts to ``max_prompt_len``. Position ids restart
        at 0 for each sequence's real tokens; padded slots get position 0
        (their attention is fully masked, so the value is irrelevant)."""
        batch_size = len(self._batch)
        pad_id = 0
        input_ids = torch.full((batch_size, self._max_prompt_len), pad_id, dtype=torch.long, device=self.device)
        position_ids = torch.zeros((batch_size, self._max_prompt_len), dtype=torch.long, device=self.device)
        for i, seq in enumerate(self._batch):
            offset = self._max_prompt_len - self._prompt_lens[i]
            input_ids[i, offset:] = torch.tensor(seq.prompt_token_ids, dtype=torch.long, device=self.device)
            position_ids[i, offset:] = torch.arange(self._prompt_lens[i], device=self.device)
        return input_ids, position_ids

    def _build_decode_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """One token per row. For each sequence we feed the most-recently
        sampled token at its own real position. Sequences that have already
        finished still need a column — we re-feed their last token; their
        sampled output is ignored by the engine via ``Sequence.is_finished``."""
        last_tokens: list[int] = []
        positions: list[int] = []
        for seq, prompt_len in zip(self._batch, self._prompt_lens, strict=True):
            if seq.output_token_ids:
                last_tokens.append(seq.output_token_ids[-1])
            else:
                last_tokens.append(seq.prompt_token_ids[-1])
            positions.append(prompt_len + max(0, len(seq.output_token_ids) - 1))

        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        position_ids = torch.tensor(positions, dtype=torch.long, device=self.device).unsqueeze(1)
        return input_ids, position_ids

    def _select_last_real_logits(self, logits: torch.Tensor, is_new_batch: bool) -> torch.Tensor:
        """Return ``[B, vocab]`` logits at each sequence's last real token.

        On prefill, the "last real token" sits at column ``max_prompt_len - 1``
        for the longest sequence and earlier for left-padded shorter ones.
        On decode, ``query_len == 1`` so there is only one column to pick.
        """
        if logits.shape[1] == 1:
            return logits[:, -1, :]

        batch_size = logits.shape[0]
        if is_new_batch:
            last_indices = torch.tensor(
                [self._max_prompt_len - 1] * batch_size,
                dtype=torch.long,
                device=logits.device,
            )
        else:
            last_indices = torch.full((batch_size,), logits.shape[1] - 1, dtype=torch.long, device=logits.device)
        return logits[torch.arange(batch_size, device=logits.device), last_indices]

    def _current_decode_step(self) -> int:
        """Number of decode steps completed so far for the longest member.

        After prefill, every sequence has at most one new sampled token.
        Subsequent steps grow ``output_token_ids`` for non-finished members.
        """
        return max(seq.num_output_tokens for seq in self._batch)
