"""User-facing entry point: the `LLM` class."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

from liteinfer.config import EngineConfig
from liteinfer.engine.llm_engine import LLMEngine
from liteinfer.engine.metrics import EngineStats
from liteinfer.engine.sequence import SequenceStatus
from liteinfer.sampling.params import SamplingParams
from liteinfer.tokenizer import Tokenizer


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

    @property
    def tokenizer(self) -> Tokenizer:
        return self.engine.model_runner.tokenizer

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Generate completions. Submits all requests, then drains the engine.

        Output order matches input ``prompts`` order regardless of which
        sequences finish first inside a static batch.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        params = sampling_params or SamplingParams()

        request_ids: list[str] = []
        for prompt in prompts:
            req_id = f"req-{next(self._req_id_gen)}"
            self.engine.add_request(req_id, prompt, params)
            request_ids.append(req_id)

        finished_by_id: dict[str, RequestOutput] = {}
        while self.engine.has_unfinished_requests():
            for seq in self.engine.step():
                finished_by_id[seq.request_id] = RequestOutput(
                    request_id=seq.request_id,
                    prompt=seq.prompt,
                    text=self.tokenizer.decode(seq.output_token_ids),
                    token_ids=list(seq.output_token_ids),
                    finish_reason=_FINISH_REASONS.get(seq.status, "stop"),
                )

        return [finished_by_id[req_id] for req_id in request_ids]
