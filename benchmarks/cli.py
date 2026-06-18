"""CLI entry point for the benchmark system.

Usage:
    bench dataset generate  -- generate a canonical JSONL dataset
    bench run               -- run a benchmark for one engine
    bench run-suite         -- run all engines on the same dataset
    bench results list      -- list saved result files
    bench results show      -- print a result file as formatted text
    bench dashboard compare -- build comparison HTML from cherry-picked results
    bench dashboard promote -- pin result(s) to the main dashboard
    bench dashboard demote  -- unpin result(s) from the main dashboard
    bench dashboard build   -- rebuild the main dashboard HTML
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parsers
# ---------------------------------------------------------------------------


def _add_dataset_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HuggingFace model ID (for tokenizer)")
    parser.add_argument("--isl", required=True, type=int, help="Target input sequence length")
    parser.add_argument("--osl", required=True, type=int, help="Forced output sequence length")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples")
    parser.add_argument(
        "--output", default="benchmarks/datasets/", help="Output directory for JSONL file"
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine", required=True, help="Engine: liteinfer, vllm, trtllm"
    )
    parser.add_argument(
        "--type", required=True, dest="benchmark_type", choices=["throughput", "latency"]
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--dataset", required=True, help="Path to canonical JSONL dataset")
    parser.add_argument("--num-samples", type=int, default=None, help="Use first N samples")
    parser.add_argument("--tag", default="", help="Short label for this run")
    parser.add_argument(
        "--results-dir", default="benchmarks/results/", help="Where to save result JSON"
    )
    parser.add_argument(
        "--strict-osl",
        action="store_true",
        default=False,
        help="Fail if >5%% of samples have wrong output length",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=None,
        help="Max tokens per forward pass (engine-specific; maps to max_num_batched_tokens / max_num_tokens)",
    )


def _add_run_suite_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engines", nargs="+", required=True, help="Engines to run (space-separated)"
    )
    parser.add_argument(
        "--type", required=True, dest="benchmark_type", choices=["throughput", "latency"]
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--dataset", required=True, help="Path to canonical JSONL dataset")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--results-dir", default="benchmarks/results/")
    parser.add_argument("--strict-osl", action="store_true", default=False)
    parser.add_argument("--max-num-tokens", type=int, default=None)
    parser.add_argument(
        "--promote", action="store_true", default=False, help="Auto-promote all results"
    )


def _add_results_list_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", default=None, help="Filter by engine")
    parser.add_argument("--type", dest="benchmark_type", default=None, help="Filter by type")
    parser.add_argument("--model", default=None, help="Filter by model")
    parser.add_argument(
        "--pinned", action="store_true", default=False, help="Show only pinned results"
    )
    parser.add_argument("--results-dir", default="benchmarks/results/")


def _add_results_show_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_id", help="Run ID to display")
    parser.add_argument("--results-dir", default="benchmarks/results/")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_dataset_generate(args: argparse.Namespace) -> None:
    from benchmarks.dataset import generate_dataset

    output_file = generate_dataset(
        model_id=args.model,
        target_isl=args.isl,
        target_osl=args.osl,
        num_samples=args.num_samples,
        output_path=args.output,
    )
    print(f"Dataset written: {output_file}")


def _cmd_run(args: argparse.Namespace) -> None:
    from benchmarks.harness import run_benchmark

    result = run_benchmark(
        engine_name=args.engine,
        benchmark_type=args.benchmark_type,
        model=args.model,
        dataset_path=args.dataset,
        num_samples=args.num_samples,
        tag=args.tag,
        results_dir=args.results_dir,
        strict_osl=args.strict_osl,
        max_num_tokens=args.max_num_tokens,
    )
    _print_result_summary(result)
    print(f"\nResult saved: {args.results_dir}/{result['run_id']}.json")


def _cmd_run_suite(args: argparse.Namespace) -> None:
    from benchmarks.harness import run_benchmark

    run_ids = []
    for engine in args.engines:
        print(f"\n--- Running {engine} ---")
        result = run_benchmark(
            engine_name=engine,
            benchmark_type=args.benchmark_type,
            model=args.model,
            dataset_path=args.dataset,
            num_samples=args.num_samples,
            tag=args.tag,
            results_dir=args.results_dir,
            strict_osl=args.strict_osl,
            max_num_tokens=args.max_num_tokens,
        )
        _print_result_summary(result)
        run_ids.append(result["run_id"])

    if args.promote:
        for run_id in run_ids:
            _promote_run_id(run_id, args.results_dir)
        print(f"\nPromoted {len(run_ids)} results to main dashboard.")


def _cmd_results_list(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print("No results directory found.")
        return

    main_json = results_dir / "main.json"
    pinned_ids: set[str] = set()
    if main_json.exists():
        index = json.loads(main_json.read_text())
        pinned_ids = set(index.get("pinned", []))

    rows = []
    for result_file in sorted(results_dir.glob("*.json")):
        if result_file.name == "main.json":
            continue
        try:
            r = json.loads(result_file.read_text())
        except Exception:
            continue

        if args.engine and r.get("engine") != args.engine:
            continue
        if args.benchmark_type and r.get("benchmark_type") != args.benchmark_type:
            continue
        if args.model and r.get("model") != args.model:
            continue

        run_id = r.get("run_id", result_file.stem)
        is_pinned = "✓" if run_id in pinned_ids else ""
        if args.pinned and not is_pinned:
            continue

        rows.append(
            (
                run_id,
                r.get("engine", ""),
                r.get("benchmark_type", ""),
                r.get("model", "").split("/")[-1],
                str(r["dataset"].get("target_isl", "")),
                str(r["dataset"].get("target_osl", "")),
                r.get("tag", ""),
                is_pinned,
            )
        )

    rows.sort(key=lambda x: x[0], reverse=True)

    header = f"{'run_id':<50} {'engine':<12} {'type':<12} {'model':<30} {'ISL':>5} {'OSL':>5} {'tag':<20} pinned"
    print(header)
    print("-" * len(header))
    for row in rows:
        run_id, engine, btype, model, isl, osl, tag, pinned = row
        print(f"{run_id:<50} {engine:<12} {btype:<12} {model:<30} {isl:>5} {osl:>5} {tag:<20} {pinned}")


def _cmd_results_show(args: argparse.Namespace) -> None:
    result_file = Path(args.results_dir) / f"{args.run_id}.json"
    if not result_file.exists():
        print(f"Result not found: {result_file}", file=sys.stderr)
        sys.exit(1)
    result = json.loads(result_file.read_text())
    _print_result_summary(result)


def _cmd_dashboard_compare(args: argparse.Namespace) -> None:
    from benchmarks.dashboard.builder import build_comparison_dashboard

    build_comparison_dashboard(
        result_ids=args.run_ids,
        results_dir=args.results_dir,
        output_path=args.output,
    )
    print(f"Dashboard written: {args.output}")


def _cmd_dashboard_promote(args: argparse.Namespace) -> None:
    for run_id in args.run_ids:
        _promote_run_id(run_id, args.results_dir)
        print(f"Promoted: {run_id}")


def _cmd_dashboard_demote(args: argparse.Namespace) -> None:
    main_json = Path(args.results_dir) / "main.json"
    if not main_json.exists():
        print("main.json not found.", file=sys.stderr)
        sys.exit(1)
    index = json.loads(main_json.read_text())
    for run_id in args.run_ids:
        if run_id in index["pinned"]:
            index["pinned"].remove(run_id)
            print(f"Demoted: {run_id}")
        else:
            print(f"Not pinned: {run_id}")
    main_json.write_text(json.dumps(index, indent=2))


def _cmd_dashboard_build(args: argparse.Namespace) -> None:
    from benchmarks.dashboard.builder import build_main_dashboard

    build_main_dashboard(results_dir=args.results_dir, output_path=args.output)
    print(f"Dashboard written: {args.output}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _promote_run_id(run_id: str, results_dir: str) -> None:
    main_json = Path(results_dir) / "main.json"
    index = json.loads(main_json.read_text()) if main_json.exists() else {"pinned": []}
    if run_id not in index["pinned"]:
        index["pinned"].append(run_id)
    main_json.write_text(json.dumps(index, indent=2))


def _print_result_summary(result: dict) -> None:
    engine = result.get("engine", "?")
    btype = result.get("benchmark_type", "?")
    dataset = result.get("dataset", {})
    isl = dataset.get("target_isl", "?")
    osl = dataset.get("target_osl", "?")
    num_samples = dataset.get("num_samples", "?")
    tag = result.get("tag", "")
    timestamp = result.get("timestamp", "?")
    summary = result.get("summary", {})

    print(
        f"\nEngine: {engine}  |  {btype}  |  ISL={isl}  OSL={osl}  |  {num_samples} samples"
    )
    if tag:
        print(f"Tag: {tag}  |  {timestamp}")
    else:
        print(f"Timestamp: {timestamp}")

    if btype == "throughput":
        tps = summary.get("tokens_per_second")
        rps = summary.get("requests_per_second")
        if tps is not None:
            print("\nThroughput")
            print(f"  tok/s      {tps:>10.1f}")
        if rps is not None:
            print(f"  req/s      {rps:>10.1f}")
        print(
            "\nLatency (note: TTFT in throughput mode includes scheduling queue time and"
        )
        print("         is not comparable to TTFT from a latency-mode run)")

    else:
        print("\nLatency (batch_size=1)")

    for label, key in [
        ("TTFT  p50", "ttft_p50_ms"),
        ("TTFT  p95", "ttft_p95_ms"),
        ("TTFT  p99", "ttft_p99_ms"),
        ("ITL   p50", "itl_p50_ms"),
        ("ITL   p95", "itl_p95_ms"),
        ("ITL   p99", "itl_p99_ms"),
        ("E2E   p50", "e2e_p50_ms"),
    ]:
        v = summary.get(key)
        if v is not None:
            print(f"  {label}  {v:>10.0f} ms")


# ---------------------------------------------------------------------------
# Parser construction (also exported for test use)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="liteinfer benchmark CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # dataset
    dataset_parser = subparsers.add_parser("dataset", help="Dataset commands")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command")
    gen_parser = dataset_sub.add_parser("generate", help="Generate a canonical dataset")
    _add_dataset_generate_args(gen_parser)
    gen_parser.set_defaults(func=_cmd_dataset_generate)

    # run
    run_parser = subparsers.add_parser("run", help="Run a benchmark for one engine")
    _add_run_args(run_parser)
    run_parser.set_defaults(func=_cmd_run)

    # run-suite
    suite_parser = subparsers.add_parser("run-suite", help="Run all engines sequentially")
    _add_run_suite_args(suite_parser)
    suite_parser.set_defaults(func=_cmd_run_suite)

    # results
    results_parser = subparsers.add_parser("results", help="Result management commands")
    results_sub = results_parser.add_subparsers(dest="results_command")

    list_parser = results_sub.add_parser("list", help="List saved result files")
    _add_results_list_args(list_parser)
    list_parser.set_defaults(func=_cmd_results_list)

    show_parser = results_sub.add_parser("show", help="Print a result file")
    _add_results_show_args(show_parser)
    show_parser.set_defaults(func=_cmd_results_show)

    # dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Dashboard commands")
    dash_sub = dash_parser.add_subparsers(dest="dashboard_command")

    compare_parser = dash_sub.add_parser("compare", help="Build comparison dashboard")
    compare_parser.add_argument("run_ids", nargs="+", help="Run IDs to compare")
    compare_parser.add_argument("--output", required=True, help="Output HTML path")
    compare_parser.add_argument("--results-dir", default="benchmarks/results/")
    compare_parser.set_defaults(func=_cmd_dashboard_compare)

    promote_parser = dash_sub.add_parser("promote", help="Pin result(s) to main dashboard")
    promote_parser.add_argument("run_ids", nargs="+", help="Run IDs to promote")
    promote_parser.add_argument("--results-dir", default="benchmarks/results/")
    promote_parser.set_defaults(func=_cmd_dashboard_promote)

    demote_parser = dash_sub.add_parser("demote", help="Unpin result(s) from main dashboard")
    demote_parser.add_argument("run_ids", nargs="+", help="Run IDs to demote")
    demote_parser.add_argument("--results-dir", default="benchmarks/results/")
    demote_parser.set_defaults(func=_cmd_dashboard_demote)

    build_parser = dash_sub.add_parser("build", help="Rebuild main dashboard HTML")
    build_parser.add_argument("--output", default="docs/index.html")
    build_parser.add_argument("--results-dir", default="benchmarks/results/")
    build_parser.set_defaults(func=_cmd_dashboard_build)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not hasattr(args, "func"):
        _build_parser().print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
