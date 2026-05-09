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

# Throughput: all requests submitted at once, engine queues at its max_num_seqs. Shows req/s and E2E under load.
_THROUGHPUT_COLS: list[tuple[str, int, str]] = [
    ("Engine", 20, "engine"),
    ("B", 4, "batch_size"),
    ("req/s", 8, "requests_per_second"),
    ("tok/s", 8, "output_tokens_per_second"),
    ("E2E p50", 13, "e2e_latency_p50_s"),
    ("E2E p99", 13, "e2e_latency_p99_s"),
]

# Latency: one request at a time, no queue. Shows TTFT and per-request E2E.
_LATENCY_COLS: list[tuple[str, int, str]] = [
    ("Engine", 20, "engine"),
    ("B", 4, "batch_size"),
    ("TTFT p50", 13, "ttft_p50_s"),
    ("TTFT p99", 13, "ttft_p99_s"),
    ("E2E p50", 13, "e2e_latency_p50_s"),
    ("tok/s", 8, "output_tokens_per_second"),
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
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of single-prompt warmup calls before timing.",
    )
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
            for _ in range(args.warmup):
                runner.generate([workload.prompts[0]], workload.sampling)

            t0 = time.perf_counter()
            results = _run_workload(runner, workload)
            wall = time.perf_counter() - t0
        finally:
            runner.teardown()

        peak_memory = getattr(runner, "peak_memory_bytes", None)
        batch_size = int(getattr(runner, "batch_size", 1))
        metrics = summarize(
            engine_name,
            results,
            wall_time_s=wall,
            peak_memory_bytes=peak_memory,
            batch_size=batch_size,
        )
        all_metrics.append(metrics)

    _print_table(all_metrics, workload, timestamp, args.tag)

    run_record = {
        "timestamp": timestamp,
        "tag": args.tag,
        "workload": workload.name,
        "model": args.model,
        "sequential": workload.sequential,
        "num_prompts": len(workload.prompts),
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


def _run_workload(runner, workload) -> list:
    """Execute workload respecting sequential vs batch submission semantics."""
    from benchmarks.runners.base import GenerationResult

    if workload.sequential:
        results: list[GenerationResult] = []
        for prompt in workload.prompts:
            results.extend(runner.generate([prompt], workload.sampling))
        return results
    else:
        return runner.generate(workload.prompts, workload.sampling)


def _print_table(
    all_metrics: list[BenchmarkMetrics],
    workload,
    timestamp: str,
    tag: str | None,
) -> None:
    cols = _LATENCY_COLS if workload.sequential else _THROUGHPUT_COLS
    submission = "sequential (no queue)" if workload.sequential else "all submitted at once"

    tag_part = f"  [{tag}]" if tag else ""
    print(
        f"\nWorkload: {workload.name}  |  {len(workload.prompts)} prompts  |  {submission}"
        f"\n{timestamp}{tag_part}\n"
    )

    header = "  ".join(
        f"{label:<{w}}" if i == 0 else f"{label:>{w}}"
        for i, (label, w, _) in enumerate(cols)
    )
    separator = "  ".join("-" * w for _, w, _ in cols)
    print(header)
    print(separator)

    for m in all_metrics:
        print(_format_row(m, cols))

    if len(all_metrics) > 1:
        _print_speedup(all_metrics)


def _format_row(m: BenchmarkMetrics, cols: list[tuple[str, int, str]]) -> str:
    cells = []
    for i, (_, w, attr) in enumerate(cols):
        val = getattr(m, attr)
        if i == 0:
            cells.append(f"{val:<{w}}")
        elif isinstance(val, int):
            cells.append(f"{val}".rjust(w))
        elif attr.endswith("_s"):
            cells.append(f"{val * _MS:.1f} ms".rjust(w))
        else:
            cells.append(f"{val:.2f}".rjust(w))
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
        e2e_ratio = baseline.e2e_latency_p50_s / m.e2e_latency_p50_s if m.e2e_latency_p50_s > 0 else 0.0
        direction = "faster" if thr_ratio > 1.0 else "slower"
        print(
            f"  {m.engine}:  {thr_ratio:.2f}x throughput  |  {e2e_ratio:.2f}x E2E p50  ({direction})"
        )


if __name__ == "__main__":
    main()
