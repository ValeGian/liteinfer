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
Single-prompt at a time, no continuous batching, no paged cache. See
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
  (continuous batching; later, prefix-cache aware).
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

Engines: `liteinfer` (no KV cache, RECOMPUTE) · `liteinfer-kvcache` (eager KV cache) · `liteinfer-b4` (eager KV cache + static batching B=4) · `vllm` (B=1) · `vllm-b4` (B=4).

**Throughput** — 32 requests submitted at once. E2E includes queue wait time.

| Engine | B | req/s | tok/s | E2E p50 | E2E p99 |
|---|---:|---:|---:|---:|---:|
| liteinfer | 1 | 1.81 | 75 | 11193 ms | 17662 ms |
| liteinfer-kvcache | 1 | 1.87 | 75 | 9865 ms | 17100 ms |
| liteinfer-b4 | 4 | 4.42 | 182 | 4096 ms | 7234 ms |
| vllm | 1 | 2.89 | 180 | 5785 ms | 11023 ms |
| vllm-b4 | 4 | 10.67 | 650 | 1656 ms | 2963 ms |

**Latency** — sequential, no queue; each request sent only after previous finishes. B>1 has no effect because at most one prompt is in flight.

| Engine | B | TTFT p50 | TTFT p99 | E2E p50 | tok/s |
|---|---:|---:|---:|---:|---:|
| liteinfer | 1 | 13 ms | 14 ms | 1654 ms | 75 |
| liteinfer-kvcache | 1 | 15 ms | 16 ms | 1490 ms | 75 |
| liteinfer-b4 | 4 | 15 ms | 15 ms | 1501 ms | 75 |
| vllm | 1 | 26 ms | 27 ms | 693 ms | 183 |
| vllm-b4 | 4 | 26 ms | 29 ms | 693 ms | 183 |

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
- [ ] Static batching (B > 1)
- [ ] Paged KV cache → continuous batching → prefix caching
- [ ] `torch.compile` and CUDA graphs for decode
- [ ] Tensor parallelism (single node)
- [ ] Speculative decoding

Detailed, fine-grained backlog with scope and parity-test notes lives
in [`docs/roadmap.md`](docs/roadmap.md). Achieved milestones are
tracked separately in [`docs/milestones.md`](docs/milestones.md).

## License

Apache-2.0.
