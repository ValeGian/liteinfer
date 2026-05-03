# Benchmarks

Compare `liteinfer` against `vllm` (and any other engine or variant you wire up)
on identical workloads, with the same measurement code on both sides.

## Design

Every engine implements the `EngineRunner` interface in
`benchmarks/runners/base.py`:

```python
class EngineRunner(Protocol):
    name: str
    def setup(self, model: str, **kwargs) -> None: ...
    def generate(self, prompts: list[str], sampling: SamplingSpec) -> list[GenerationResult]: ...
    def teardown(self) -> None: ...
```

Workloads (`benchmarks/workloads.py`) are pure data — a list of prompts
plus sampling parameters. Measurement (`benchmarks/metrics.py`) is
engine-agnostic.

## Running a single comparison

```bash
python -m benchmarks.compare \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --engines liteinfer vllm \
    --workload throughput
```

The terminal prints a side-by-side table of all metrics plus a speedup
summary vs the first engine listed.

## Tracking optimization impact over time

Tag every run with `--tag` and accumulate results in a JSONL history file
with `--append-history`:

```bash
# Before the optimization
python -m benchmarks.compare \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --engines liteinfer vllm \
    --workload throughput \
    --tag baseline \
    --append-history benchmarks/results/history.jsonl

# After implementing the optimization
python -m benchmarks.compare \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --engines liteinfer vllm \
    --workload throughput \
    --tag prefix-cache \
    --append-history benchmarks/results/history.jsonl
```

Generate the HTML dashboard from the accumulated history:

```bash
python -m benchmarks.dashboard \
    --history benchmarks/results/history.jsonl \
    --output benchmarks/results/dashboard.html
```

Open `dashboard.html` in a browser. Rows are ordered chronologically;
cells that improved ≥5 % vs the previous run are **green**, regressions
are **red**. Each workload gets its own section; all engines appear as
column groups so you can read across a run horizontally and down a column
to see the trend.

## Comparing two liteinfer variants

Register a second runner alongside the default one:

1. Copy `benchmarks/runners/liteinfer_runner.py` to e.g. `liteinfer_prefix_runner.py`.
2. Adjust `setup()` to enable the feature (e.g. `enable_prefix_caching=True`).
3. Register it in `benchmarks/runners/__init__.py`:
   ```python
   RUNNERS["liteinfer-prefix"] = LiteInferPrefixRunner
   ```
4. Pass both names to `--engines`:
   ```bash
   python -m benchmarks.compare \
       --engines liteinfer liteinfer-prefix vllm ...
   ```

## Saving raw results to JSON

`--output PATH` writes the full run record (timestamp, tag, workload,
model, per-engine metrics) as JSON to the given path.

## Adding a new engine

1. Drop a file in `benchmarks/runners/` implementing the interface.
2. Register it in `benchmarks/runners/__init__.py`.

## Workloads

| Workload       | Stresses                                       |
| -------------- | ---------------------------------------------- |
| `throughput`   | Continuous batching, request scheduling.       |
| `latency`      | TTFT, end-to-end latency on a single request.  |
| `prefix_share` | Prefix caching (shared system prompts).        |

Add new workloads by appending to `benchmarks/workloads.py`.

## Metrics

| Metric             | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `req/s`            | Requests completed per wall-second.                                  |
| `tok/s`            | Output tokens emitted per wall-second.                               |
| `TTFT p50/p99`     | Wall time from request start to first emitted token.                 |
| `E2E p50/p99`      | Wall time for the full generation of one request.                    |
| `peak_memory_bytes`| Peak GPU memory allocated during `generate()` (when measurable).    |

Runners that track peak CUDA memory expose a `peak_memory_bytes`
attribute after `generate()` returns; `compare.py` reads it with
`getattr` so runners without it are unaffected.
