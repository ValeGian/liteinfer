"""Benchmark harness: orchestrates dataset → adapter → stats → result file."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from benchmarks.adapters import ADAPTER_REGISTRY
from benchmarks.dataset import dataset_sha256, load_dataset
from benchmarks.stats import compute_summary


class BenchmarkError(RuntimeError):
    """Raised for any non-recoverable benchmark failure."""


def run_benchmark(
    engine_name: str,
    benchmark_type: Literal["throughput", "latency"],
    model: str,
    dataset_path: str | Path,
    num_samples: int | None,
    tag: str,
    results_dir: str | Path,
    strict_osl: bool = False,
    max_num_tokens: int | None = None,
) -> dict[str, Any]:
    """Run a benchmark and save the result JSON.

    Sequence:
    1. Load dataset (first num_samples entries)
    2. Compute dataset_sha256
    3. Instantiate adapter from ADAPTER_REGISTRY
    4. Call adapter via context manager
    5. If strict_osl: check output length mismatch rate (>5% → BenchmarkError)
    6. Compute summary stats
    7. Assemble and write result JSON
    8. Return the result dict
    """
    dataset_path = Path(dataset_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(dataset_path, num_samples=num_samples)
    if not samples:
        raise BenchmarkError(f"Dataset at {dataset_path} is empty")

    sha256 = dataset_sha256(samples)

    if engine_name not in ADAPTER_REGISTRY:
        raise BenchmarkError(
            f"Unknown engine '{engine_name}'. "
            f"Available engines: {list(ADAPTER_REGISTRY.keys())}"
        )

    adapter_cls = ADAPTER_REGISTRY[engine_name]

    with adapter_cls(max_num_tokens=max_num_tokens) as adapter:
        measurements, wall_time_s = adapter.run(samples, model, benchmark_type)

    if not measurements:
        raise BenchmarkError("Adapter returned no measurements")

    if strict_osl:
        mismatches = sum(
            1
            for m, s in zip(measurements, samples, strict=False)
            if m.output_token_count != s.forced_output_token_count
        )
        total = len(measurements)
        if mismatches / total > 0.05:
            raise BenchmarkError(
                f"--strict-osl: {mismatches}/{total} samples have "
                f"output_token_count ≠ forced_output_token_count"
            )

    summary = compute_summary(measurements, benchmark_type, wall_time_s)

    timestamp = datetime.now(timezone.utc)
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    ts_iso = timestamp.strftime("%Y-%m-%dT%H:%M:%S")

    target_isl = samples[0].input_token_count
    target_osl = samples[0].forced_output_token_count
    run_id = f"{ts_str}_{engine_name}_{benchmark_type}_isl{target_isl}_osl{target_osl}"

    summary_dict: dict[str, Any] = {}
    if summary.latency:
        lat = summary.latency
        summary_dict.update(
            {
                "ttft_p50_ms": lat.ttft_p50_ms,
                "ttft_p95_ms": lat.ttft_p95_ms,
                "ttft_p99_ms": lat.ttft_p99_ms,
                "itl_p50_ms": lat.itl_p50_ms,
                "itl_p95_ms": lat.itl_p95_ms,
                "itl_p99_ms": lat.itl_p99_ms,
                "e2e_p50_ms": lat.e2e_p50_ms,
            }
        )
    if summary.throughput:
        thr = summary.throughput
        summary_dict.update(
            {
                "tokens_per_second": thr.tokens_per_second,
                "requests_per_second": thr.requests_per_second,
            }
        )

    raw = [
        {
            "sample_index": m.sample_index,
            "input_token_count": m.input_token_count,
            "output_token_count": m.output_token_count,
            "ttft_s": m.ttft_s,
            "token_timestamps_s": m.token_timestamps_s,
            "e2e_s": m.e2e_s,
        }
        for m in measurements
    ]

    result: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": ts_iso,
        "tag": tag,
        "engine": engine_name,
        "benchmark_type": benchmark_type,
        "model": model,
        "dataset": {
            "path": str(dataset_path),
            "num_samples": len(samples),
            "target_isl": target_isl,
            "target_osl": target_osl,
            "prompt_batch_sha256": sha256,
        },
        "summary": summary_dict,
        "raw": raw,
    }

    output_file = results_dir / f"{run_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result
