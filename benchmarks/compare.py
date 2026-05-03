"""CLI: run one workload across one or more engines and emit JSON results.

Example:
    python -m benchmarks.compare \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --engines liteinfer vllm \\
        --workload throughput \\
        --output benchmarks/results/throughput.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.metrics import summarize
from benchmarks.runners import RUNNERS
from benchmarks.workloads import WORKLOADS


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM inference engines.")
    parser.add_argument("--model", required=True, help="HF repo id or local path.")
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["liteinfer", "vllm"],
        choices=sorted(RUNNERS),
    )
    parser.add_argument("--workload", default="throughput", choices=sorted(WORKLOADS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    workload = WORKLOADS[args.workload]()
    all_metrics = []

    for engine_name in args.engines:
        runner = RUNNERS[engine_name]()
        runner.setup(args.model)
        try:
            t0 = time.perf_counter()
            results = runner.generate(workload.prompts, workload.sampling)
            wall = time.perf_counter() - t0
        finally:
            runner.teardown()

        metrics = summarize(engine_name, results, wall_time_s=wall)
        all_metrics.append(metrics.as_dict())
        print(
            f"[{engine_name}] {metrics.requests_per_second:.2f} req/s, "
            f"{metrics.output_tokens_per_second:.1f} tok/s, "
            f"TTFT p50={metrics.ttft_p50_s * 1000:.1f} ms"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "workload": workload.name,
                    "model": args.model,
                    "engines": all_metrics,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
