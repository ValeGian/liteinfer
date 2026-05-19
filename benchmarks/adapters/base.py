from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class BenchmarkSample:
    prompt: str
    input_token_count: int
    forced_output_token_count: int


@dataclass
class RequestMeasurement:
    sample_index: int
    input_token_count: int
    output_token_count: int
    ttft_s: float
    token_timestamps_s: list[float] | None  # absolute wall times of each output token
    e2e_s: float


class EngineAdapter(Protocol):
    name: str

    def __enter__(self) -> EngineAdapter: ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    def run(
        self,
        samples: list[BenchmarkSample],
        model: str,
        benchmark_type: Literal["throughput", "latency"],
    ) -> tuple[list[RequestMeasurement], float]:
        """
        Run the benchmark and return (measurements, wall_time_s).

        wall_time_s is the elapsed time from the moment the first real request
        is submitted to the engine until the last response is fully received.
        It must not include model loading, server startup, dataset I/O, or
        warmup time.

        Before starting the clock, the adapter must run some warmup
        requests using a short fixed prompt ("Hello", max_tokens=16).
        Warmup requests are not included in measurements or wall_time_s.

        Raise BenchmarkError (defined in benchmarks/harness.py) on any
        non-recoverable failure. Never return partial measurements.
        """
        ...
