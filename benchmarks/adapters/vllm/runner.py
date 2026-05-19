"""vLLM runner: subprocess entry point executed in the vLLM venv.

Reads dataset JSON from --dataset, runs vLLM, writes per-request
measurements as JSONL to stdout, then exits.

IPC format (stdout):
  One JSON object per line for each measurement.
  Final line: {"__wall_time_s": <float>}
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
    from vllm import AsyncLLM, SamplingParams

    llm = AsyncLLM(model=model)

    # Warmup
    warmup_params = SamplingParams(
        max_tokens=_WARMUP_MAX_TOKENS,
        min_tokens=_WARMUP_MAX_TOKENS,
        ignore_eos=True,
    )
    warmup_tasks = [
        asyncio.create_task(_collect_stream(llm, _WARMUP_PROMPT, warmup_params, f"warmup-{i}"))
        for i in range(_WARMUP_COUNT)
    ]
    await asyncio.gather(*warmup_tasks)

    # Real benchmark
    wall_start = time.perf_counter()
    tasks = [
        asyncio.create_task(_run_sample(llm, sample, idx))
        for idx, sample in enumerate(samples)
    ]
    measurements = await asyncio.gather(*tasks)
    wall_time_s = time.perf_counter() - wall_start

    for m in measurements:
        print(json.dumps(m), flush=True)
    print(json.dumps({"__wall_time_s": wall_time_s}), flush=True)


async def _collect_stream(llm, prompt: str, params, request_id: str) -> None:
    async for _ in llm.generate(prompt, params, request_id=request_id):
        pass


async def _run_sample(llm, sample: dict, idx: int) -> dict:
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=sample["forced_output_token_count"],
        min_tokens=sample["forced_output_token_count"],
        ignore_eos=True,
    )
    request_id = f"req-{idx}"
    submit_time = time.perf_counter()
    first_token_time: float | None = None
    output_token_count = 0
    last_time = submit_time

    async for output in llm.generate(sample["prompt"], params, request_id=request_id):
        now = time.perf_counter()
        if first_token_time is None and output.outputs:
            # Check metrics for first_token_ts; fall back to perf_counter
            metrics = output.metrics
            if metrics and hasattr(metrics, "first_token_ts") and metrics.first_token_ts:
                first_token_time = metrics.first_token_ts - (
                    metrics.queued_ts if hasattr(metrics, "queued_ts") else submit_time
                ) + submit_time
            else:
                first_token_time = now
        if output.outputs:
            output_token_count = len(output.outputs[0].token_ids)
        last_time = now

    ttft_s = (first_token_time - submit_time) if first_token_time else (last_time - submit_time)
    e2e_s = last_time - submit_time

    return {
        "sample_index": idx,
        "input_token_count": sample["input_token_count"],
        "output_token_count": output_token_count,
        "ttft_s": ttft_s,
        "token_timestamps_s": None,
        "e2e_s": e2e_s,
    }


def run_latency(samples: list[dict], model: str) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_num_seqs=1)

    # Warmup
    warmup_params = SamplingParams(
        max_tokens=_WARMUP_MAX_TOKENS,
        min_tokens=_WARMUP_MAX_TOKENS,
        ignore_eos=True,
    )
    llm.generate([_WARMUP_PROMPT] * _WARMUP_COUNT, warmup_params)

    measurements: list[dict] = []
    wall_start = time.perf_counter()

    for idx, sample in enumerate(samples):
        params = SamplingParams(
            max_tokens=sample["forced_output_token_count"],
            min_tokens=sample["forced_output_token_count"],
            ignore_eos=True,
        )
        start = time.perf_counter()
        outputs = llm.generate([sample["prompt"]], params)
        end = time.perf_counter()

        output_token_count = len(outputs[0].outputs[0].token_ids) if outputs else 0

        measurements.append(
            {
                "sample_index": idx,
                "input_token_count": sample["input_token_count"],
                "output_token_count": output_token_count,
                "ttft_s": end - start,  # latency mode: TTFT ≈ e2e for batch_size=1
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
