"""User-facing entry point: the `LLM` class."""

from __future__ import annotations

from collections.abc import Sequence as _Sequence
from dataclasses import dataclass
from itertools import count

from liteinfer.config import EngineConfig
from liteinfer.engine.llm_engine import LLMEngine
from liteinfer.engine.metrics import EngineStats
from liteinfer.engine.sequence import SequenceStatus
from liteinfer.sampling.params import SamplingParams


@dataclass
class RequestOutput:
    request_id: str
    prompt: str
    text: str
    token_ids: list[int]
    finish_reason: str  # "stop" | "length" | "abort"


_FINISH_REASONS: dict[SequenceStatus, str] = {
    SequenceStatus.FINISHED_STOPPED: "stop",
    SequenceStatus.FINISHED_LENGTH: "length",
    SequenceStatus.FINISHED_ABORTED: "abort",
}


class LLM:
    """High-level offline inference API. Facade over `LLMEngine`."""

    def __init__(self, model: str, **engine_kwargs) -> None:
        self.config = EngineConfig(model=model, **engine_kwargs)
        self.engine = LLMEngine(self.config)
        self._req_id_gen = count(0)

        self.engine.load_model()

    @property
    def stats(self) -> EngineStats:
        return self.engine.stats

    def generate(
        self,
        prompts: str | _Sequence[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Generate completions. Drains all requests before returning."""
        if isinstance(prompts, str):
            prompts = [prompts]
        params = sampling_params or SamplingParams()

        # v0 runs one prompt per static batch — submit and drain each
        # request before moving on. Collapses into one submit-then-drain
        # once batched execution lands.
        results: list[RequestOutput] = []
        for prompt in prompts:
            req_id = f"req-{next(self._req_id_gen)}"
            self.engine.add_request(req_id, prompt, params)
            while self.engine.has_unfinished_requests():
                for group in self.engine.step():
                    seq = group.primary
                    tokenizer = self.engine.model_runner.tokenizer
                    if tokenizer is None:
                        raise RuntimeError("model not loaded; call load_model() first")
                    results.append(
                        RequestOutput(
                            request_id=group.request_id,
                            prompt=group.prompt,
                            text=tokenizer.decode(seq.output_token_ids),
                            token_ids=list(seq.output_token_ids),
                            finish_reason=_FINISH_REASONS.get(seq.status, "stop"),
                        )
                    )
        return results
