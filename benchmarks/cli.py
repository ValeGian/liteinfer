"""Benchmark CLI.

bench dataset --model M --isl 128 --osl 256 -n 200
bench run --all --dataset D --mode both
bench report --out docs/index.html
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

from benchmarks import configs, dataset, harness, report
from benchmarks.harness import Mode

DEFAULT_RESULTS_DIR = "benchmarks/results"
DEFAULT_DATASET_DIR = "benchmarks/datasets"


def _cmd_dataset(args: argparse.Namespace) -> int:
    path = dataset.generate(
        model=args.model,
        target_isl=args.isl,
        target_osl=args.osl,
        num_samples=args.num_samples,
        output_dir=args.out,
    )
    print(f"Wrote {path}")
    return 0


def _run_one(args: argparse.Namespace, name: str, mode: Mode) -> int:
    data = dataset.load(args.dataset).head(args.num_samples)
    result = harness.run(configs.get(name), data, args.dataset, mode, args.results_dir)
    metrics = "  ".join(f"{k}={v:,.1f}" for k, v in result.summary.items())
    print(f"{name} [{mode}] {metrics}")
    return 0


@dataclass(frozen=True)
class Slot:
    """One GPU plus a private block of CPU cores."""

    gpu: int
    cores: str
    threads: int


def _slots(gpus: list[int]) -> list[Slot]:
    """Give each GPU a disjoint block of cores.

    Pinning is what makes parallel runs comparable to sequential ones: decode is
    bound by how fast Python can launch kernels, so workers sharing cores would
    measure each other's contention instead of the engine.
    """
    per = max(1, (os.cpu_count() or len(gpus)) // len(gpus))
    return [
        Slot(gpu=gpu, cores=f"{i * per}-{i * per + per - 1}", threads=per)
        for i, gpu in enumerate(gpus)
    ]


def _run_isolated(args: argparse.Namespace, name: str, mode: Mode, slot: Slot | None) -> int:
    """Each run gets a fresh process so GPU state never leaks between configs."""
    command = [
        sys.executable,
        "-m",
        "benchmarks.cli",
        "run",
        "--config",
        name,
        "--mode",
        mode,
        "--dataset",
        str(args.dataset),
        "--results-dir",
        str(args.results_dir),
        "--no-isolate",
    ]
    if args.num_samples is not None:
        command += ["--num-samples", str(args.num_samples)]

    env = dict(os.environ)
    if slot is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
        env["OMP_NUM_THREADS"] = str(slot.threads)
        if shutil.which("taskset"):
            command = ["taskset", "-c", slot.cores, *command]

    done = subprocess.run(command, env=env, capture_output=True, text=True)
    label = f"{name} [{mode}]" + (f" gpu{slot.gpu}" if slot else "")
    if done.returncode == 0:
        summary = [ln for ln in done.stdout.splitlines() if ln.startswith(name)]
        print(summary[-1] if summary else f"{label} ok", flush=True)
    else:
        print(f"{label} FAILED\n{done.stderr[-1500:]}", file=sys.stderr, flush=True)
    return done.returncode


def _longest_first(job: tuple[str, Mode]) -> tuple:
    """Start the long jobs first so the tail of the schedule is short.

    Throughput at a narrow batch width is the long pole; vLLM is fast everywhere.
    """
    config = configs.get(job[0])
    return (job[1] == "latency", config.max_num_seqs, config.engine == "vllm")


def _cmd_run(args: argparse.Namespace) -> int:
    names = list(configs.CONFIGS) if args.all else args.config
    if not names:
        print("Nothing to run: pass --config NAME ... or --all", file=sys.stderr)
        return 2
    modes: list[Mode] = ["throughput", "latency"] if args.mode == "both" else [args.mode]
    jobs: list[tuple[str, Mode]] = [(n, m) for n in names for m in modes]

    if args.no_isolate:
        return max((_run_one(args, n, m) for n, m in jobs), default=0)

    slots = _slots(args.gpus) if args.gpus else [None]
    if len(slots) > 1 and "latency" in modes:
        # Latency is dominated by per-call overhead, which is CPU-scheduling
        # sensitive: running workers side by side inflated vLLM's TTFT by 24%
        # while leaving its ITL untouched. Throughput is GPU-bound and safe.
        print(
            "WARNING: latency mode is sensitive to CPU contention; run it without "
            "--gpus for publishable TTFT numbers.",
            file=sys.stderr,
        )
    jobs.sort(key=_longest_first)
    print(f"{len(jobs)} runs over {len(slots)} worker(s)", flush=True)

    free: Queue = Queue()
    for slot in slots:
        free.put(slot)

    def run_job(job: tuple[str, Mode]) -> int:
        slot = free.get()
        try:
            return _run_isolated(args, job[0], job[1], slot)
        finally:
            free.put(slot)

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        codes = list(pool.map(run_job, jobs))

    failed = [f"{n}/{m}" for (n, m), code in zip(jobs, codes, strict=True) if code != 0]
    if failed:
        print(f"\nFailed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


def _cmd_report(args: argparse.Namespace) -> int:
    results = report.load_results(args.results_dir)
    print(report.as_text(results))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.as_html(results), encoding="utf-8")
        print(f"\nWrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="liteinfer benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("dataset", help="Generate a canonical dataset")
    make.add_argument("--model", required=True)
    make.add_argument("--isl", type=int, required=True)
    make.add_argument("--osl", type=int, required=True)
    make.add_argument("-n", "--num-samples", type=int, default=200)
    make.add_argument("--out", default=DEFAULT_DATASET_DIR)
    make.set_defaults(func=_cmd_dataset)

    run = sub.add_parser("run", help="Run configs against a dataset")
    run.add_argument("--config", nargs="+", default=[], choices=list(configs.CONFIGS))
    run.add_argument("--all", action="store_true", help="Run every known config")
    run.add_argument("--dataset", required=True)
    run.add_argument("--mode", choices=["throughput", "latency", "both"], default="both")
    run.add_argument("-n", "--num-samples", type=int, default=None)
    run.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    run.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="GPU ids to spread runs over, one run per GPU at a time (default: one worker)",
    )
    run.add_argument(
        "--no-isolate", action="store_true", help="Run in this process instead of a subprocess"
    )
    run.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="Print and optionally write the comparison report")
    rep.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    rep.add_argument("--out", default=None, help="Write a standalone HTML page here")
    rep.set_defaults(func=_cmd_report)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
