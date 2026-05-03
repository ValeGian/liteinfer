# liteinfer

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

Pre-alpha. The repository is a skeleton — most components currently raise
`NotImplementedError`. The structure, contracts, and test/benchmark
harness are in place so that each feature can be implemented and verified
in isolation.

## Installation

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
pytest                              # everything that runs in this env
pytest -m "not gpu and not slow"    # the fast suite
pytest -m gpu                       # GPU-only tests
pytest tests/unit/                  # one directory
```

Test layout:

- `tests/unit/` — single-component tests. CPU-only, fast, no model loading.
- `tests/integration/` — multiple components wired together (still no HF download).
- `tests/e2e/` — load a small real model and verify generation against `transformers`.

See `tests/README.md` for conventions.

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

- [ ] Single-GPU greedy generation on Llama-family models
- [ ] Continuous batching
- [ ] Paged KV cache
- [ ] Prefix caching
- [ ] `torch.compile` integration
- [ ] CUDA graph capture for decode
- [ ] Tensor parallelism (single node)
- [ ] Speculative decoding

## License

Apache-2.0.
