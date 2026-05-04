"""User-facing entry point: the `LLM` class."""

from __future__ import annotations

from collections.abc import Sequence as _Sequence
from dataclasses import dataclass
from itertools import count

from liteinfer.config import EngineConfig
from liteinfer.engine.llm_engine import LLMEngine
from liteinfer.engine.metrics import EngineStats
from liteinfer.sampling.params import SamplingParams


@dataclass
class RequestOutput:
    """Result for a single generation request."""

    request_id: str
    prompt: str
    text: str
    token_ids: list[int]
    finish_reason: str  # "stop" | "length" | "abort"


_FINISH_REASONS = {
    "FINISHED_STOPPED": "stop",
    "FINISHED_LENGTH": "length",
    "FINISHED_ABORTED": "abort",
}


class LLM:
    """High-level offline inference API.

    A thin facade over `LLMEngine`. The intent is for this public surface
    to stay stable while engine internals evolve.
    """

    def __init__(self, model: str, **engine_kwargs) -> None:
        self.config = EngineConfig(model=model, **engine_kwargs)
        self.engine = LLMEngine(self.config)
        self._req_id_gen = count(0)

    @property
    def stats(self) -> EngineStats:
        return self.engine.stats

    def generate(
        self,
        prompts: str | _Sequence[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Generate completions for one or more prompts.

        Drains every request to completion before returning. Per-step
        metrics accumulate in ``self.stats`` and can be inspected
        afterwards or subscribed to via ``self.stats.on_step``.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        params = sampling_params or SamplingParams()

        # v0 runs one prompt per static batch (see ModelRunner). Submit
        # and drain each request before moving on so multi-prompt input
        # still works under the B=1 limitation. Once batched execution
        # lands, this loop collapses into a single submit-then-drain.
        results: list[RequestOutput] = []
        for prompt in prompts:
            req_id = f"req-{next(self._req_id_gen)}"
            self.engine.add_request(req_id, prompt, params)
            while self.engine.has_unfinished_requests():
                for group in self.engine.step():
                    seq = group.primary
                    tokenizer = self.engine.model_runner.tokenizer
                    assert tokenizer is not None
                    results.append(
                        RequestOutput(
                            request_id=group.request_id,
                            prompt=group.prompt,
                            text=tokenizer.decode(seq.output_token_ids),
                            token_ids=list(seq.output_token_ids),
                            finish_reason=_FINISH_REASONS.get(seq.status.name, "stop"),
                        )
                    )
        return results
