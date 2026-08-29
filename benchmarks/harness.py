"""Runs one config in one mode and writes a result file."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import cycle, islice
from pathlib import Path
from typing import Literal

from benchmarks import adapters, stats
from benchmarks.configs import BenchmarkConfig
from benchmarks.dataset import Dataset

Mode = Literal["throughput", "latency"]

WARMUP_TOKENS = 32
WARMUP_ROUNDS = 2


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class Result:
    config: BenchmarkConfig
    mode: Mode
    dataset: Dataset
    dataset_path: Path
    summary: dict
    raw: dict
    timestamp: str

    @property
    def filename(self) -> str:
        return (
            f"{self.config.name}__{self.mode}"
            f"__isl{self.dataset.target_isl}_osl{self.dataset.target_osl}.json"
        )

    def as_dict(self) -> dict:
        return {
            "config": self.config.name,
            "engine": self.config.engine,
            "description": self.config.description,
            "baseline": self.config.baseline,
            "mode": self.mode,
            "model": self.dataset.model,
            "max_num_seqs": self.config.max_num_seqs,
            "timestamp": self.timestamp,
            "dataset": {
                "path": str(self.dataset_path),
                "target_isl": self.dataset.target_isl,
                "target_osl": self.dataset.target_osl,
                "num_samples": len(self.dataset.samples),
                "sha256": self.dataset.sha256,
            },
            "summary": self.summary,
            "raw": self.raw,
        }


def _check_lengths(counts: list[int], expected: int) -> None:
    wrong = [c for c in counts if c != expected]
    if wrong:
        raise BenchmarkError(
            f"{len(wrong)}/{len(counts)} requests returned a wrong output length "
            f"(expected {expected}, saw e.g. {wrong[0]}). Forced-length control is "
            f"broken, so the run is not comparable."
        )


def _warmup(adapter: adapters.Adapter, config: BenchmarkConfig, data: Dataset, mode: Mode) -> None:
    """Warm the exact path about to be measured, before the clock starts.

    Real prompts, so prefill is warmed at the benchmark's ISL rather than at
    length 1, and the real batch width, so kernel autotuning and CUDA graph
    capture happen here instead of inside the timed region.
    """
    width = 1 if mode == "latency" else config.max_num_seqs
    prompts = list(islice(cycle(s.prompt for s in data.samples), width))
    for _ in range(WARMUP_ROUNDS):
        if mode == "latency":
            adapter.generate(prompts, 1)  # latency mode times a 1-token pass too
        adapter.generate(prompts, WARMUP_TOKENS)


def _run_throughput(adapter: adapters.Adapter, data: Dataset) -> tuple[dict, dict]:
    prompts = [s.prompt for s in data.samples]
    start = time.perf_counter()
    counts = adapter.generate(prompts, data.target_osl)
    wall_time_s = time.perf_counter() - start

    _check_lengths(counts, data.target_osl)
    summary = stats.throughput_summary(sum(counts), len(counts), wall_time_s)
    return summary.as_dict(), {"wall_time_s": wall_time_s, "output_tokens": sum(counts)}


def _run_latency(adapter: adapters.Adapter, data: Dataset) -> tuple[dict, dict]:
    # Two passes per sample: one capped at a single token to time the prefill
    # exactly, one full-length to time the whole request.
    ttft_s: list[float] = []
    e2e_s: list[float] = []
    for sample in data.samples:
        start = time.perf_counter()
        adapter.generate([sample.prompt], 1)
        ttft_s.append(time.perf_counter() - start)

        start = time.perf_counter()
        counts = adapter.generate([sample.prompt], data.target_osl)
        e2e_s.append(time.perf_counter() - start)
        _check_lengths(counts, data.target_osl)

    summary = stats.latency_summary(ttft_s, e2e_s, data.target_osl)
    return summary.as_dict(), {"ttft_s": ttft_s, "e2e_s": e2e_s}


def run(
    config: BenchmarkConfig,
    data: Dataset,
    dataset_path: str | Path,
    mode: Mode,
    results_dir: str | Path,
) -> Result:
    if not data.samples:
        raise BenchmarkError(f"Dataset {dataset_path} has no samples")

    with adapters.build(config, data.model) as adapter:
        _warmup(adapter, config, data, mode)
        runner = _run_throughput if mode == "throughput" else _run_latency
        summary, raw = runner(adapter, data)

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = Result(config, mode, data, Path(dataset_path), summary, raw, timestamp)
    output = Path(results_dir) / result.filename
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return result
