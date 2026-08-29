# liteinfer

[![PyPI](https://img.shields.io/pypi/v/liteinfer)](https://pypi.org/project/liteinfer/)

A lightweight, hackable LLM inference engine built from scratch — designed
to make state-of-the-art inference techniques (paged KV cache, prefix
caching, tensor parallelism, `torch.compile`, CUDA graphs, …) easy to read,
test, and benchmark.

## Goals

1. **Fast offline inference** — throughput in the same league as vLLM on a single node.
2. **Readable codebase** — clean, minimal, well-structured. The core engine should fit in your head.
3. **Optimization suite** — a clear place for each technique (prefix caching, TP, `torch.compile`, CUDA graphs, …) with isolated, testable implementations.
4. **HuggingFace compatibility** — load any compatible HF model from the Hub or a local safetensors directory.

## Status

v0 — minimal end-to-end greedy/sampled inference on local safetensors.
Static batching (B > 1), paged KV cache. Continuous batching via `AsyncLLM` (async context manager + streaming API). Paged KV cache. See
[`docs/milestones.md`](docs/milestones.md) for what is in, and
[`docs/roadmap.md`](docs/roadmap.md) for what is queued.

## Installation

```bash
pip install liteinfer
```

For development or benchmark comparisons:

```bash
git clone https://github.com/ValeGian/liteinfer.git
cd liteinfer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
from liteinfer import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
params = SamplingParams(temperature=0.8, max_tokens=128)

outputs = llm.generate(["Explain paged attention in one paragraph."], params)
print(outputs[0].text)
```

```python
# Async continuous batching with streaming
import asyncio
from liteinfer import AsyncLLM, SamplingParams

async def main():
    async with AsyncLLM("meta-llama/Llama-3.2-1B-Instruct") as llm:
        # Batch — all requests processed concurrently
        results = await llm.generate(prompts, SamplingParams(max_tokens=128))

        # Streaming — per-token events
        async for event in llm.stream("Explain paged attention.", SamplingParams(max_tokens=128)):
            print(event.text, end="\r")

asyncio.run(main())
```

## Repository layout

```
liteinfer/
├── liteinfer/             # Library source
│   ├── llm.py             # User-facing LLM class
│   ├── config.py          # EngineConfig
│   ├── engine/            # Orchestration: scheduler, sequence, model runner
│   ├── models/            # Model loaders + per-architecture implementations
│   ├── layers/            # Reusable building blocks (attention, RMSNorm, …)
│   ├── cache/             # KV cache (paged, prefix-cached, …)
│   └── sampling/          # SamplingParams + Sampler
├── tests/                 # Unit / integration / e2e tests
├── benchmarks/            # Benchmark matrix, harness, and report
└── pyproject.toml
```

Each module's `__init__.py` documents the contract it owns.

## Architecture (brief)

User code calls **`LLM`**, a thin facade over **`LLMEngine`**, which owns:

- **`Scheduler`** — picks which sequences run on the next forward pass
  (static batching (see `Scheduler`) or continuous batching (see `ContinuousScheduler`)).
- **`ModelRunner`** — runs the actual forward pass for the selected batch
  on the GPU. Tensor parallelism, `torch.compile`, and CUDA graph capture
  plug in here.
- **`KVCache`** — paged blocks shared across sequences. Prefix caching
  is a `KVCache` variant.

Sampling is a separate stage so strategies (greedy, top-p, …) can be
swapped without touching the engine.

## Testing

`liteinfer` is **test-first**: every feature ships with the tests that
pin its contract.

```bash
pytest                              # full suite — runs sequentially, GPU-safe
pytest -m "not gpu and not slow"    # fast suite (no model downloads, no GPU)
pytest -m gpu                       # GPU tests only — sequential, never use -n auto
pytest -n auto                      # parallel mode — CPU-only tests only
pytest tests/unit/                  # one directory
```

> **GPU tests**: always run sequentially. The e2e tests (`tests/e2e/`) load
> real models onto GPU. Running them in parallel (`-n auto`) will cause OOM or
> cross-process interference.

Test layout:

- `tests/unit/` — single-component tests. CPU-only, fast, no model loading.
- `tests/integration/` — multiple components wired together (still no HF download).
- `tests/e2e/` — load a small real model and verify generation against `transformers`.

See `tests/README.md` for conventions.

## Performance

A40 · Llama-3.2-1B-Instruct · ISL 128 · OSL 256 · vs vLLM 0.28.0 at matched batch width.

| | liteinfer | vLLM | |
|---|---:|---:|---|
| Decode step (ITL p50) | 13.7 ms | 5.2 ms | 2.6× behind |
| Throughput, B=4 | 282 tok/s | 724 tok/s | 2.6× behind |
| Throughput, B=32 | 1,268 tok/s | 4,467 tok/s | 3.5× behind |
| Batching gain, B=1 → B=4 | 3.84× | 3.84× | on par |
| Time to first token (p50) | 15.7 ms | 22.7 ms | 1.4× ahead\* |

\* At this prompt length TTFT is mostly fixed per-call API overhead rather than
prefill compute, so read it as offline round-trip latency, not prefill speed.

Batching is competitive at both tiers; what remains is a roughly flat ~3× per-step
decode constant at every batch width — no CUDA graphs, no fused attention.
Throughput figures are certified against `vllm bench throughput` to within 1%. Full tables, methodology, and per-milestone deltas:
[`docs/benchmarks.md`](docs/benchmarks.md) · [live dashboard](https://valegian.github.io/liteinfer/).

## Benchmarking

Every liteinfer configuration is measured against the one it improves on, and
against vLLM, on byte-identical prompts.

```bash
pip install -e ".[dev,bench]"

# 1. Build a dataset (once per model + ISL/OSL)
bench dataset --model meta-llama/Llama-3.2-1B-Instruct --isl 128 --osl 256 -n 200

DS=benchmarks/datasets/isl128_osl256_n200_meta_llama_llama_3_2_1b_instruct.json

# 2. Run every config, in both modes
bench run --all --dataset "$DS" --mode throughput -n 200
bench run --all --dataset "$DS" --mode latency    -n 50

# 3. Report
bench report --out docs/index.html
```

Engines: `liteinfer`, `vllm`. See [`benchmarks/README.md`](benchmarks/README.md)
for the config matrix and controls, and [`docs/benchmarks.md`](docs/benchmarks.md)
for methodology and results.

## Roadmap

High-level direction:

- [x] Single-prompt greedy/sampled generation from local safetensors
- [x] Static batching (B > 1)
- [x] Paged KV cache
- [x] Continuous batching (async, streaming, paged KV)
- [ ] prefix caching
- [ ] `torch.compile` and CUDA graphs for decode
- [ ] Tensor parallelism (single node)
- [ ] Speculative decoding

Detailed, fine-grained backlog with scope and parity-test notes lives
in [`docs/roadmap.md`](docs/roadmap.md). Achieved milestones are
tracked separately in [`docs/milestones.md`](docs/milestones.md).

## License

Apache-2.0.
