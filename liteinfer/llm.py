"""User-facing entry point: the `LLM` class."""

from __future__ import annotations

import asyncio

from liteinfer.async_llm import AsyncLLM
from liteinfer.engine.metrics import EngineStats
from liteinfer.outputs import RequestOutput
from liteinfer.sampling.params import SamplingParams
from liteinfer.tokenizer import Tokenizer


class LLM:
    """Synchronous offline inference, backed by the continuous-batching engine.

    Owns a private event loop so the async engine can be driven from ordinary
    code. Inside an existing event loop, use `AsyncLLM` directly.

    Usage::

        llm = LLM("meta-llama/Llama-3.2-1B-Instruct")
        outputs = llm.generate(prompts, SamplingParams(max_tokens=64))
        llm.close()
    """

    def __init__(self, model: str, **engine_kwargs) -> None:
        if _event_loop_is_running():
            raise RuntimeError(
                "LLM owns an event loop and cannot be built inside a running one. "
                "Use AsyncLLM here instead."
            )
        self._loop = asyncio.new_event_loop()
        self._llm = AsyncLLM(model=model, **engine_kwargs)
        self._loop.run_until_complete(self._llm.start())

    @property
    def config(self):
        return self._llm.config

    @property
    def stats(self) -> EngineStats:
        return self._llm.engine.stats

    @property
    def tokenizer(self) -> Tokenizer:
        return self._llm.engine.tokenizer

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Generate completions, returned in the order the prompts were given."""
        return self._loop.run_until_complete(self._llm.generate(prompts, sampling_params))

    def close(self) -> None:
        """Stop the engine and release its event loop."""
        if self._loop.is_closed():
            return
        self._loop.run_until_complete(self._llm.stop())
        self._loop.close()

    def __enter__(self) -> LLM:
        return self

    def __exit__(self, *_) -> None:
        self.close()


def _event_loop_is_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
