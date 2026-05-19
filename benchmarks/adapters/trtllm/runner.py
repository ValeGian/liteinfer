"""TRT-LLM runner: subprocess entry point executed in the TRT-LLM venv.

Uses tensorrt_llm.LLM Python API directly (PyTorch backend).
No trtllm-build compilation step required.

IPC format (stdout): identical to vLLM runner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

_WARMUP_PROMPT = "Hello"
_WARMUP_MAX_TOKENS = 16
_WARMUP_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark-type", required=True, choices=["throughput", "latency"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.dataset) as f:
        samples = json.load(f)

    if args.benchmark_type == "throughput":
        asyncio.run(run_throughput(samples, args.model))
    else:
        run_latency(samples, args.model)


async def run_throughput(samples: list[dict], model: str) -> None:
    from tensorrt_llm import LLM, SamplingParams

    llm = LLM(model=model, backend="pytorch")

    # Warmup
    warmup_params = SamplingParams(max_new_tokens=_WARMUP_MAX_TOKENS, end_id=-1)
    warmup_inputs = [_WARMUP_PROMPT] * _WARMUP_COUNT
    async for _ in llm.generate_async(warmup_inputs, sampling_params=warmup_params):
        pass

    # Real benchmark
    wall_start = time.perf_counter()
    tasks = [
        asyncio.create_task(_run_sample_async(llm, sample, idx))
        for idx, sample in enumerate(samples)
    ]
    measurements = await asyncio.gather(*tasks)
    wall_time_s = time.perf_counter() - wall_start

    for m in measurements:
        print(json.dumps(m), flush=True)
    print(json.dumps({"__wall_time_s": wall_time_s}), flush=True)


async def _run_sample_async(llm, sample: dict, idx: int) -> dict:
    from tensorrt_llm import SamplingParams

    params = SamplingParams(
        max_new_tokens=sample["forced_output_token_count"],
        end_id=-1,
    )
    submit_time = time.perf_counter()
    first_token_time: float | None = None
    output_token_count = 0
    token_times: list[float] = []

    async for output in llm.generate_async([sample["prompt"]], sampling_params=params):
        now = time.perf_counter()
        token_times.append(now)
        if first_token_time is None:
            first_token_time = now
        if hasattr(output, "outputs") and output.outputs:
            output_token_count = len(output.outputs[0].token_ids)

    end_time = time.perf_counter()
    ttft_s = (first_token_time - submit_time) if first_token_time else (end_time - submit_time)

    return {
        "sample_index": idx,
        "input_token_count": sample["input_token_count"],
        "output_token_count": output_token_count,
        "ttft_s": ttft_s,
        "token_timestamps_s": token_times if len(token_times) > 1 else None,
        "e2e_s": end_time - submit_time,
    }


def run_latency(samples: list[dict], model: str) -> None:
    from tensorrt_llm import LLM, SamplingParams

    llm = LLM(model=model, backend="pytorch")

    # Warmup
    warmup_params = SamplingParams(max_new_tokens=_WARMUP_MAX_TOKENS, end_id=-1)
    llm.generate([_WARMUP_PROMPT] * _WARMUP_COUNT, sampling_params=warmup_params)

    measurements: list[dict] = []
    wall_start = time.perf_counter()

    for idx, sample in enumerate(samples):
        params = SamplingParams(
            max_new_tokens=sample["forced_output_token_count"],
            end_id=-1,
        )
        start = time.perf_counter()
        outputs = llm.generate([sample["prompt"]], sampling_params=params)
        end = time.perf_counter()

        output_token_count = 0
        if outputs and hasattr(outputs[0], "outputs") and outputs[0].outputs:
            output_token_count = len(outputs[0].outputs[0].token_ids)

        measurements.append(
            {
                "sample_index": idx,
                "input_token_count": sample["input_token_count"],
                "output_token_count": output_token_count,
                "ttft_s": end - start,
                "token_timestamps_s": None,
                "e2e_s": end - start,
            }
        )

    wall_time_s = time.perf_counter() - wall_start

    for m in measurements:
        print(json.dumps(m), flush=True)
    print(json.dumps({"__wall_time_s": wall_time_s}), flush=True)


if __name__ == "__main__":
    main()
