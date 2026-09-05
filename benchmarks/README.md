# Benchmarks

Measures every liteinfer configuration against the one it improves on, and
against vLLM, on byte-identical prompts.

## Quick start

```bash
pip install -e ".[dev,bench]"

# 1. Build a dataset (once per model + ISL/OSL)
bench dataset --model meta-llama/Llama-3.2-1B-Instruct --isl 128 --osl 256 -n 200

DS=benchmarks/datasets/isl128_osl256_n200_meta_llama_llama_3_2_1b_instruct.json

# 2. Run. Throughput can spread over GPUs; latency must not (see Controls).
bench run --all --dataset "$DS" --mode throughput -n 200 --gpus 0 1 2 3 4 5 6 7
bench run --all --dataset "$DS" --mode latency    -n 50

# 3. Report
bench report --out docs/index.html
```

Run one config while iterating:

```bash
bench run --config liteinfer-paged --dataset "$DS" --mode latency -n 20
```

## The two modes measure different things

| | `throughput` | `latency` |
|---|---|---|
| Offered load | every prompt at once | one request at a time |
| Reports | output tok/s, req/s | TTFT, ITL, E2E percentiles |

They report disjoint metrics on purpose. Under saturation a per-request latency
mostly records where the request sat in the queue, so it says little about the
engine; at one request in flight there is no throughput to speak of. Mixing the
two is how benchmarks end up comparing queue depth and calling it speed.

ITL is derived, not instrumented: with one request in flight and a forced output
length, `(e2e - ttft) / (osl - 1)` is the mean decode-step cost. TTFT comes from a
separate pass capped at a single token. Both engines are therefore measured by
the same clock, with no per-token hooks that each would implement differently.

## Shipping an improvement

This harness exists so that a faster path can *replace* a slower one instead of
sitting next to it — when it genuinely covers the same ground. The loop (measure
the baseline, add a config naming what it improves on, read the mode that matches
the change, then either delete what it beat and simplify, or keep both because
the win is scoped to a precondition) is written up in
[`docs/roadmap.md`](../docs/roadmap.md) under "Shipping an improvement".

## Configs

`benchmarks/configs.py` is the matrix. Each entry names the config it improves
on via `baseline`, and the report turns that into a 1:1 delta — which is how a
change is judged. Adding a config is one entry; no other file changes.

## Controls

- **Identical prompts.** One dataset file per (model, ISL, OSL). Datasets are not
  committed — they hold raw scraped text, secrets and all — but regenerate byte
  for byte from the pinned corpus revision and seed. Every result records the
  SHA-256 of its prompt set, and the report flags any group whose members disagree.
- **Forced output length.** `min_tokens = max_tokens = OSL`, `ignore_eos=True`, so
  output-length variance can never be mistaken for an engine difference. A run
  whose lengths come back wrong fails rather than reporting.
- **Warmup.** Before the clock starts, each run exercises the path it is about to
  measure — real prompts at the benchmark's ISL, at the real batch width — so
  kernel autotuning and CUDA graph capture land outside the timed region.
- **Isolation.** Each (config, mode) runs in a fresh process, so no GPU state or
  allocator fragmentation carries between configs. `--gpus` spreads runs over
  several GPUs, each worker pinned to its own block of CPU cores.
- **Latency runs sequentially.** TTFT is largely fixed per-call overhead, which is
  CPU-scheduling sensitive: measured inside an 8-worker sweep, vLLM's TTFT read 24%
  high while its ITL was untouched. Throughput is GPU-bound and parallelises safely.
  `bench run` warns if you combine `--gpus` with latency mode.
- **Greedy decoding**, fixed seed.

## Layout

| Module | Responsibility |
|---|---|
| `configs.py` | The benchmark matrix and its comparison lineage |
| `dataset.py` | Build / load a canonical dataset; prompt-set digest |
| `adapters.py` | Per-engine translation of one primitive: prompts → output lengths |
| `harness.py` | Warm up, time, verify, write the result file |
| `stats.py` | Percentiles and rates |
| `report.py` | Result files → text table and a standalone HTML page |
| `cli.py` | `bench dataset` / `bench run` / `bench report` |
