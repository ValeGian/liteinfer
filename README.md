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
# Optional: install vLLM for benchmark comparisons
pip install -e ".[dev,bench]"
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
├── benchmarks/            # vLLM comparison harness
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
> cross-process interference. `tests/e2e/test_vllm_runner.py` additionally
> requires the `bench` extras (`pip install -e ".[dev,bench]"`).

Test layout:

- `tests/unit/` — single-component tests. CPU-only, fast, no model loading.
- `tests/integration/` — multiple components wired together (still no HF download).
- `tests/e2e/` — load a small real model and verify generation against `transformers`.

See `tests/README.md` for conventions.

## Performance

Llama-3.2-1B-Instruct · greedy · NVIDIA A40 — see [`docs/benchmarks.md`](docs/benchmarks.md) for full setup and methodology, or open the **[interactive dashboard](https://htmlpreview.github.io/?https://github.com/ValeGian/liteinfer/blob/master/docs/dashboard.html)** for a visual comparison.

Engines: `liteinfer` (B=1, no KV cache) · `liteinfer-b4` (eager, B=4) · `liteinfer-native-kvcache-b4` (native eager, B=4) · `liteinfer-paged-b4` (paged static, B=4) · `liteinfer-continuous` (paged continuous, B=4) · `vllm` (B=1) · `vllm-b4` (B=4) · `vllm-continuous` (continuous B=4).

**Throughput** — 32 requests submitted at once. E2E includes queue wait time.

| Engine | B | req/s | tok/s | E2E p50 | E2E p99 |
|---|---:|---:|---:|---:|---:|
| liteinfer | 1 | 1.41 | 58 | 14832 ms | 22763 ms |
| liteinfer-b4 | 4 | 3.71 | 152 | 4860 ms | 8624 ms |
| liteinfer-native-kvcache-b4 | 4 | 3.52 | 144 | 5274 ms | 9094 ms |
| liteinfer-paged-b4 | 4 | 2.31 | 95 | 8230 ms | 13872 ms |
| liteinfer-continuous | 4 | 3.14 | 125 | 5756 ms | 10206 ms |
| vllm | 1 | 2.88 | 179 | 5822 ms | 11098 ms |
| vllm-b4 | 4 | 10.60 | 645 | 1683 ms | 2974 ms |
| vllm-continuous | 4 | 10.72 | 653 | 1636 ms | 2947 ms |

**Latency** — sequential, no queue; each request sent only after previous finishes.

| Engine | B | TTFT p50 | TTFT p99 | E2E p50 | tok/s |
|---|---:|---:|---:|---:|---:|
| liteinfer | 1 | 16.1 ms | 17.0 ms | 1948 ms | 64 |
| liteinfer-kvcache | 1 | 18.8 ms | 29.6 ms | 1936 ms | 54 |
| liteinfer-native-kvcache | 1 | 18.4 ms | 26.6 ms | 1974 ms | 56 |
| liteinfer-paged-kvcache | 1 | 18.1 ms | 27.6 ms | 2104 ms | 51 |
| vllm | 1 | 25.9 ms | 29.4 ms | 692 ms | 183 |

## Benchmarking against vLLM

Every engine implements the same `EngineRunner` interface, so comparing
`liteinfer` against vLLM (or future variants of `liteinfer` itself) is a
single command:

```bash
python -m benchmarks.compare \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --engines liteinfer vllm \
    --workload throughput \
    --output benchmarks/results/throughput.json
```

Metrics: requests/sec, output tokens/sec, TTFT (p50/p99), inter-token
latency, peak GPU memory. See `benchmarks/README.md` for adding
workloads or new engines.

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
