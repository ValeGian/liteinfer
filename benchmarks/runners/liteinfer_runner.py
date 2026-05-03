"""Run benchmarks against `liteinfer`."""

from __future__ import annotations

from benchmarks.runners.base import GenerationResult, SamplingSpec


class LiteInferRunner:
    name = "liteinfer"

    def setup(self, model: str, **kwargs) -> None:
        raise NotImplementedError

    def generate(
        self, prompts: list[str], sampling: SamplingSpec
    ) -> list[GenerationResult]:
        raise NotImplementedError

    def teardown(self) -> None:
        pass
