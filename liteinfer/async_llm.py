"""AsyncLLM — user-facing async inference API with continuous batching."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from itertools import count

from liteinfer.config import EngineConfig
from liteinfer.engine.async_llm_engine import AsyncLLMEngine
from liteinfer.outputs import RequestOutput, StreamEvent
from liteinfer.sampling.params import SamplingParams


class AsyncLLM:
    """Continuous-batching inference engine with an asyncio interface.

    Usage::

        async with AsyncLLM("meta-llama/Llama-3.2-1B-Instruct") as llm:
            # Batch: await all completions
            results = await llm.generate(prompts, SamplingParams(max_tokens=64))

            # Streaming: per-token events
            async for event in llm.stream(prompt, SamplingParams(max_tokens=64)):
                print(event.text, end="\\r")

    The engine loop runs as a background asyncio Task between ``start()`` and
    ``stop()`` (called automatically by the context manager). New requests are
    admitted into empty batch slots on every engine step; finished sequences
    are evicted immediately without waiting for the whole batch to complete.
    """

    def __init__(self, model: str, **engine_kwargs) -> None:
        self.config = EngineConfig(model=model, **engine_kwargs)
        self.engine = AsyncLLMEngine(self.config)
        self._req_id_gen = count(0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load model weights and start the background generate loop."""
        await self.engine.start()

    async def stop(self) -> None:
        """Drain pending requests and stop the background loop."""
        await self.engine.stop()

    async def __aenter__(self) -> AsyncLLM:
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Generation API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Submit all prompts and await their completions.

        Returns outputs in the same order as the input ``prompts`` list,
        regardless of which sequences finish first.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        params = sampling_params or SamplingParams()

        async def _collect(prompt: str, req_id: str) -> RequestOutput:
            final: StreamEvent | None = None
            async for event in self.engine.generate_stream(req_id, prompt, params):
                final = event
            assert final is not None
            return RequestOutput(
                request_id=req_id,
                prompt=prompt,
                text=final.text,
                token_ids=final.output_token_ids,
                finish_reason=final.finish_reason or "stop",
            )

        req_ids = [f"req-{next(self._req_id_gen)}" for _ in prompts]
        results = await asyncio.gather(
            *(_collect(p, rid) for p, rid in zip(prompts, req_ids, strict=False))
        )
        return list(results)

    async def stream(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Submit a single prompt and stream ``StreamEvent`` objects token-by-token."""
        params = sampling_params or SamplingParams()
        req_id = f"req-{next(self._req_id_gen)}"
        async for event in self.engine.generate_stream(req_id, prompt, params):
            yield event
