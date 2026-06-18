# Benchmark Harness

CLI-driven benchmark harness for comparing LLM inference engines on
identical ISL/OSL-controlled datasets.

## Quick start

```bash
# Generate a canonical dataset (once per model/ISL/OSL combination)
bench dataset generate \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --isl 128 --osl 256 --num-samples 200

# Run a benchmark
bench run \
  --engine liteinfer --type throughput \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset benchmarks/datasets/isl128_osl256_n200_meta_llama_llama_3_2_1b_instruct.jsonl \
  --tag my-feature

# Promote to main dashboard and publish
bench dashboard promote <run_id>
bench dashboard build --output docs/index.html
```

## Engines

| Key | Description |
|---|---|
| `liteinfer` | In-process (this repo), paged KV cache, async/sync |
| `vllm` | vLLM 0.21.0, subprocess in isolated venv |
| `trtllm` | TensorRT-LLM 1.2.1, subprocess, PyTorch backend |

## Adding a new engine

1. Create `benchmarks/adapters/<name>/adapter.py` implementing `EngineAdapter`.
2. Add one line to `benchmarks/adapters/__init__.py::ADAPTER_REGISTRY`.

No changes to `cli.py`, `harness.py`, or `stats.py`.

## Environment setup

See `benchmarks/envs/README.md` for one-time setup instructions per engine,
including hardware prerequisites, HuggingFace authentication, and model downloads.

## Architecture

| Module | Responsibility |
|---|---|
| `dataset.py` | Generate / load canonical JSONL datasets (corpus: ShareGPT_V3) |
| `harness.py` | Orchestrate: dataset → adapter → stats → result JSON |
| `adapters/` | Per-engine translation of canonical samples to engine API |
| `stats.py` | Compute `BenchmarkSummary` from raw `RequestMeasurement` objects |
| `dashboard/builder.py` | Read result JSONs → emit self-contained HTML |
| `cli.py` | argparse entry point, all subcommands |
