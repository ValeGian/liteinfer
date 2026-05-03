# Benchmarks

Compare `liteinfer` against `vllm` (and any other engine you wire up)
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

## Adding a new engine or variant

1. Drop a file in `benchmarks/runners/` implementing the interface.
2. Register it in `benchmarks/runners/__init__.py`.

That's it. The same recipe applies to comparing two **variants of
liteinfer** (e.g., with vs. without prefix caching) — register them as
separate runners.

## Running

```bash
python -m benchmarks.compare \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --engines liteinfer vllm \
    --workload throughput \
    --output benchmarks/results/throughput.json
```

Results are JSON. Re-run with the same `--output` path to overwrite, or
diff manually against an earlier file.

## Workloads

| Workload       | Stresses                                       |
| -------------- | ---------------------------------------------- |
| `throughput`   | Continuous batching, request scheduling.       |
| `latency`      | TTFT, inter-token latency on a single request. |
| `prefix_share` | Prefix caching (shared system prompts).        |

Add new workloads by appending to `benchmarks/workloads.py`.

## Metrics

- **Throughput**: requests / sec, output tokens / sec.
- **TTFT** (time to first token): wall time from `generate()` to first
  emitted token; reported as p50 and p99.
- **ITL** (inter-token latency): per-token wall time after TTFT.
- **Peak memory**: `torch.cuda.max_memory_allocated`.
