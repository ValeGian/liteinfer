"""TGI adapter: manages TGI Docker container lifecycle and HTTP client.

__enter__ starts the TGI server using --network host.
__exit__ stops and removes the container.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Literal

from benchmarks.adapters.base import BenchmarkSample, RequestMeasurement

_TGI_IMAGE = "ghcr.io/huggingface/text-generation-inference"
_TGI_PORT = 8080
_HEALTH_POLL_INTERVAL_S = 5
_HEALTH_TIMEOUT_S = 120
_WARMUP_PROMPT = "Hello"
_WARMUP_MAX_TOKENS = 16
_WARMUP_COUNT = 4


class TGIAdapter:
    name = "tgi"

    def __init__(self) -> None:
        self._container_id: str | None = None

    def __enter__(self) -> TGIAdapter:
        return self

    def __exit__(self, *_) -> None:
        if self._container_id:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True,
            )
            self._container_id = None

    def _start_server(self, model: str) -> None:
        from benchmarks.harness import BenchmarkError

        proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--gpus",
                "all",
                "--network",
                "host",
                "--detach",
                _TGI_IMAGE,
                "--model-id",
                model,
                "--port",
                str(_TGI_PORT),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            raise BenchmarkError(f"Failed to start TGI container.\nstderr:\n{stderr}")
        self._container_id = stdout.strip()

    def _wait_for_healthy(self) -> None:
        import urllib.request

        from benchmarks.harness import BenchmarkError

        health_url = f"http://localhost:{_TGI_PORT}/health"
        deadline = time.monotonic() + _HEALTH_TIMEOUT_S

        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            time.sleep(_HEALTH_POLL_INTERVAL_S)

        # Capture Docker stderr for diagnostics
        stderr_output = ""
        if self._container_id:
            result = subprocess.run(
                ["docker", "logs", self._container_id],
                capture_output=True,
                text=True,
            )
            stderr_output = result.stderr or result.stdout

        raise BenchmarkError(
            f"TGI server did not become healthy within {_HEALTH_TIMEOUT_S}s.\n"
            f"Docker logs:\n{stderr_output}"
        )

    def run(
        self,
        samples: list[BenchmarkSample],
        model: str,
        benchmark_type: Literal["throughput", "latency"],
    ) -> tuple[list[RequestMeasurement], float]:
        self._start_server(model)
        self._wait_for_healthy()
        return asyncio.run(self._run_async(samples, benchmark_type))

    async def _run_async(
        self,
        samples: list[BenchmarkSample],
        benchmark_type: Literal["throughput", "latency"],
    ) -> tuple[list[RequestMeasurement], float]:
        import httpx

        base_url = f"http://localhost:{_TGI_PORT}"

        async with httpx.AsyncClient(base_url=base_url, timeout=300.0) as client:
            # Warmup
            warmup_tasks = [
                asyncio.create_task(
                    self._send_request(
                        client,
                        prompt=_WARMUP_PROMPT,
                        max_tokens=_WARMUP_MAX_TOKENS,
                        sample_index=-1,
                        forced_output_token_count=_WARMUP_MAX_TOKENS,
                        input_token_count=1,
                    )
                )
                for _ in range(_WARMUP_COUNT)
            ]
            await asyncio.gather(*warmup_tasks)

            wall_start = time.perf_counter()

            if benchmark_type == "throughput":
                tasks = [
                    asyncio.create_task(
                        self._send_request(
                            client,
                            prompt=s.prompt,
                            max_tokens=s.forced_output_token_count,
                            sample_index=i,
                            forced_output_token_count=s.forced_output_token_count,
                            input_token_count=s.input_token_count,
                        )
                    )
                    for i, s in enumerate(samples)
                ]
                measurements = list(await asyncio.gather(*tasks))
            else:
                measurements = []
                for i, s in enumerate(samples):
                    m = await self._send_request(
                        client,
                        prompt=s.prompt,
                        max_tokens=s.forced_output_token_count,
                        sample_index=i,
                        forced_output_token_count=s.forced_output_token_count,
                        input_token_count=s.input_token_count,
                    )
                    measurements.append(m)

            wall_time_s = time.perf_counter() - wall_start

        return measurements, wall_time_s

    async def _send_request(
        self,
        client,
        prompt: str,
        max_tokens: int,
        sample_index: int,
        forced_output_token_count: int,
        input_token_count: int,
    ) -> RequestMeasurement:
        import json as _json

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "ignore_eos_token": True,
            },
        }

        submit_time = time.perf_counter()
        first_token_time: float | None = None
        token_times: list[float] = []
        output_text = ""

        async with client.stream("POST", "/generate_stream", json=payload) as response:
            async for chunk in response.aiter_lines():
                if not chunk.startswith("data:"):
                    continue
                data_str = chunk[len("data:"):].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    data = _json.loads(data_str)
                except _json.JSONDecodeError:
                    continue

                now = time.perf_counter()
                token_times.append(now)
                if first_token_time is None:
                    first_token_time = now
                if "token" in data:
                    output_text += data.get("token", {}).get("text", "")

        end_time = time.perf_counter()
        output_token_count = len(token_times)

        if output_token_count != forced_output_token_count and sample_index >= 0:
            import warnings
            warnings.warn(
                f"TGI sample {sample_index}: expected {forced_output_token_count} tokens, "
                f"got {output_token_count}. Consider --strict-osl.",
                stacklevel=2,
            )

        ttft_s = (first_token_time - submit_time) if first_token_time else (end_time - submit_time)

        return RequestMeasurement(
            sample_index=sample_index,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            ttft_s=ttft_s,
            token_timestamps_s=token_times if token_times else None,
            e2e_s=end_time - submit_time,
        )
