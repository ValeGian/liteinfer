"""TRT-LLM adapter: spawns runner.py in the TRT-LLM venv as a subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from benchmarks.adapters.base import BenchmarkSample, RequestMeasurement

_TRTLLM_PYTHON = Path(
    os.environ.get("BENCH_TRTLLM_PYTHON", "benchmarks/envs/trtllm/bin/python")
)
_RUNNER = Path(__file__).parent / "runner.py"


class TRTLLMAdapter:
    name = "trtllm"

    def __enter__(self) -> TRTLLMAdapter:
        return self

    def __exit__(self, *_) -> None:
        pass

    def run(
        self,
        samples: list[BenchmarkSample],
        model: str,
        benchmark_type: Literal["throughput", "latency"],
    ) -> tuple[list[RequestMeasurement], float]:
        from benchmarks.harness import BenchmarkError

        samples_data = [
            {
                "prompt": s.prompt,
                "input_token_count": s.input_token_count,
                "forced_output_token_count": s.forced_output_token_count,
            }
            for s in samples
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(samples_data, tmp)
            tmp_path = tmp.name

        try:
            proc = subprocess.run(
                [
                    str(_TRTLLM_PYTHON),
                    str(_RUNNER),
                    "--dataset",
                    tmp_path,
                    "--model",
                    model,
                    "--benchmark-type",
                    benchmark_type,
                ],
                capture_output=True,
                text=True,
            )
        finally:
            os.unlink(tmp_path)

        if proc.returncode != 0:
            raise BenchmarkError(
                f"TRT-LLM runner exited with code {proc.returncode}.\n"
                f"stderr:\n{proc.stderr}"
            )

        return _parse_runner_output(proc.stdout)


def _parse_runner_output(stdout: str) -> tuple[list[RequestMeasurement], float]:
    from benchmarks.harness import BenchmarkError

    measurements: list[RequestMeasurement] = []
    wall_time_s: float | None = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "__wall_time_s" in record:
            wall_time_s = record["__wall_time_s"]
        else:
            measurements.append(
                RequestMeasurement(
                    sample_index=record["sample_index"],
                    input_token_count=record["input_token_count"],
                    output_token_count=record["output_token_count"],
                    ttft_s=record["ttft_s"],
                    token_timestamps_s=record.get("token_timestamps_s"),
                    e2e_s=record["e2e_s"],
                )
            )

    if wall_time_s is None:
        raise BenchmarkError("TRT-LLM runner did not emit wall_time_s")
    if not measurements:
        raise BenchmarkError("TRT-LLM runner emitted no measurements")

    return measurements, wall_time_s
