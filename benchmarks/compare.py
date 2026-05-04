"""CLI: run one workload across one or more engines and emit results.

Examples:
    python -m benchmarks.compare \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --engines liteinfer vllm \\
        --workload throughput \\
        --output benchmarks/results/throughput.json \\
        --append-history benchmarks/results/history.jsonl \\
        --tag baseline
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from benchmarks.metrics import BenchmarkMetrics, summarize
from benchmarks.runners import RUNNERS
from benchmarks.workloads import WORKLOADS

_MS = 1000.0

_COL_DEFS: list[tuple[str, int, str, str | None]] = [
    # (header, width, attribute, format)
    ("Engine", 16, "engine", None),
    ("req/s", 8, "requests_per_second", ".2f"),
    ("tok/s", 8, "output_tokens_per_second", ".1f"),
    ("TTFT p50", 11, "ttft_p50_s", "ms"),
    ("TTFT p99", 11, "ttft_p99_s", "ms"),
    ("E2E p50", 11, "e2e_latency_p50_s", "ms"),
    ("E2E p99", 11, "e2e_latency_p99_s", "ms"),
    ("Tokens", 7, "output_tokens", "d"),
]


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
    parser.add_argument("--output", type=Path, default=None, help="Write JSON result to this path.")
    parser.add_argument(
        "--append-history",
        type=Path,
        default=None,
        metavar="PATH",
        help="Append this run to a JSONL history file (creates the file if absent).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Short label for this run, e.g. 'baseline' or 'prefix-cache'.",
    )
    args = parser.parse_args()

    workload = WORKLOADS[args.workload]()
    timestamp = datetime.now().isoformat(timespec="seconds")
    all_metrics: list[BenchmarkMetrics] = []

    for engine_name in args.engines:
        runner = RUNNERS[engine_name]()
        runner.setup(args.model)
        try:
            t0 = time.perf_counter()
            results = runner.generate(workload.prompts, workload.sampling)
            wall = time.perf_counter() - t0
        finally:
            runner.teardown()

        peak_memory = getattr(runner, "peak_memory_bytes", None)
        metrics = summarize(engine_name, results, wall_time_s=wall, peak_memory_bytes=peak_memory)
        all_metrics.append(metrics)

    _print_table(all_metrics, workload.name, args.model, timestamp, args.tag)

    run_record = {
        "timestamp": timestamp,
        "tag": args.tag,
        "workload": workload.name,
        "model": args.model,
        "results": [m.as_dict() for m in all_metrics],
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(run_record, indent=2))

    if args.append_history is not None:
        args.append_history.parent.mkdir(parents=True, exist_ok=True)
        with args.append_history.open("a") as fh:
            fh.write(json.dumps(run_record) + "\n")
        print(f"\nAppended to history: {args.append_history}")


def _print_table(
    all_metrics: list[BenchmarkMetrics],
    workload_name: str,
    model: str,
    timestamp: str,
    tag: str | None,
) -> None:
    tag_part = f"  [{tag}]" if tag else ""
    print(f"\nWorkload: {workload_name}  |  Model: {model}  |  {timestamp}{tag_part}\n")

    header = "  ".join(
        f"{label:<{w}}" if i == 0 else f"{label:>{w}}"
        for i, (label, w, _, _) in enumerate(_COL_DEFS)
    )
    separator = "  ".join("-" * w for _, w, _, _ in _COL_DEFS)
    print(header)
    print(separator)

    for m in all_metrics:
        print(_format_row(m))

    if len(all_metrics) > 1:
        _print_speedup(all_metrics)


def _format_row(m: BenchmarkMetrics) -> str:
    cells = []
    for i, (_, w, attr, fmt) in enumerate(_COL_DEFS):
        val = getattr(m, attr)
        if i == 0:
            cells.append(f"{val:<{w}}")
        elif fmt == "ms":
            cells.append(f"{val * _MS:.1f} ms".rjust(w))
        elif fmt == "d":
            cells.append(f"{int(val):>{w}}")
        else:
            cells.append(f"{val:{fmt}}".rjust(w))
    return "  ".join(cells)


def _print_speedup(all_metrics: list[BenchmarkMetrics]) -> None:
    baseline = all_metrics[0]
    print(f"\nSpeedup vs {baseline.engine}:")
    for m in all_metrics[1:]:
        thr_ratio = (
            m.requests_per_second / baseline.requests_per_second
            if baseline.requests_per_second > 0
            else 0.0
        )
        # For latency, lower is better: speedup = baseline / current
        ttft_ratio = baseline.ttft_p50_s / m.ttft_p50_s if m.ttft_p50_s > 0 else 0.0
        direction = "faster" if thr_ratio > 1.0 else "slower"
        print(
            f"  {m.engine}:  {thr_ratio:.2f}x throughput  |  {ttft_ratio:.2f}x TTFT p50  ({direction})"
        )


if __name__ == "__main__":
    main()
