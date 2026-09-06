# liteinfer

[![PyPI](https://img.shields.io/pypi/v/liteinfer)](https://pypi.org/project/liteinfer/)

A lightweight, hackable LLM inference engine built from scratch — designed
to make state-of-the-art inference techniques (paged KV cache, prefix
caching, tensor parallelism, `torch.compile`, CUDA graphs, …) easy to read,
test, and benchmark.

**New to inference serving?** [**Inside liteinfer**](https://claude.ai/code/artifact/52a9e43f-c529-4a70-af07-d55dbffb1cbf)
is an illustrated walkthrough of how the engine works and why each design was
chosen — prefill and decode, KV caching, paged memory, and continuous batching —
following the measured progression that produced this codebase, including the
steps that lost. It assumes PyTorch, not serving experience.

## Goals

1. **Fast offline inference** — throughput in the same league as vLLM on a single node.
2. **Readable codebase** — clean, minimal, well-structured. The core engine should fit in your head.
3. **Optimization suite** — a clear place for each technique (prefix caching, TP, `torch.compile`, CUDA graphs, …) with isolated, testable implementations.
4. **HuggingFace compatibility** — load any compatible HF model from the Hub or a local safetensors directory.

## Status

v0 — greedy/sampled inference on local safetensors via continuous batching over
a paged KV cache. `AsyncLLM` is the native interface (async context manager plus
streaming); `LLM` is a synchronous facade for offline batch use. See
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

The attention kernel is chosen for the device and the model unless you name
one — on CUDA that is `paged`, a Triton kernel that reads the KV pool in place
instead of gathering it, so the decode step stops growing with context. Pass
`attn_implementation="sdpa"` or `"eager"` to override; naming a kernel that
cannot run here is an error rather than a silent downgrade.

## Repository layout

```
liteinfer/
├── liteinfer/             # Library source
│   ├── llm.py             # Synchronous LLM facade
│   ├── async_llm.py       # AsyncLLM: the engine's native interface
│   ├── config.py          # EngineConfig
│   ├── engine/            # Orchestration: scheduler, sequence, model runner, metrics
│   ├── models/            # Loader + per-architecture implementations (layers included)
│   ├── cache/             # Paged KV cache: block pool + slot addressing
│   └── sampling/          # SamplingParams + Sampler
├── tests/                 # Unit / integration / e2e tests
├── benchmarks/            # Benchmark matrix, harness, and report
└── pyproject.toml
```

Each module's `__init__.py` documents the contract it owns.

## Architecture (brief)

One engine: continuous batching over a paged KV cache.

- **`AsyncLLM`** — asyncio API: `await llm.generate(...)`, or `async for` to
  stream tokens. **`LLM`** is a synchronous facade over it for offline batch use.
- **`ContinuousScheduler`** — fills empty batch slots from the waiting queue on
  every step and evicts finished sequences individually.
- **`ContinuousModelRunner`** — runs one prefill or decode forward pass.
  `torch.compile`, CUDA graph capture and tensor parallelism plug in here.
- **`ContinuousKVCache`** — per-sequence blocks drawn from a shared `BlockPool`;
  `slot_table` maps logical token positions to physical slots, so a whole batch
  is read or written with a single indexing op.
- **`models/attention.py`** — the attention kernel, one per
  `attn_implementation`. `sdpa` (default) never materialises the score matrix;
  `eager` writes it out in plain matmuls, which reads better and is the parity
  reference, but caps the prompt length that fits in memory. `paged` is the fast
  decode path: a Triton kernel that reads the KV pool through the slot table
  instead of gathering it, so the decode step stops growing with context. The
  engine picks between them from the device and the model's head dimension.

Sampling is a separate stage so strategies (greedy, top-p, …) can be swapped
without touching the engine. `stats` records a `StepMetrics` per forward pass.

Each of these components is explained, with diagrams and interactive figures, in
[Inside liteinfer](https://claude.ai/code/artifact/52a9e43f-c529-4a70-af07-d55dbffb1cbf).

Earlier designs — no cache, `DynamicCache`, plain-tensor cache, static
batching — were measured against each other and then removed; the numbers live
in [`docs/benchmarks.md`](docs/benchmarks.md) and the reasoning in
[`docs/milestones.md`](docs/milestones.md).

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
| Throughput, B=32 | 1,930 tok/s | 4,467 tok/s | 2.3× behind |
| Throughput, ISL 1024 / OSL 1024 | 1,784 tok/s | 3,241 tok/s | 1.8× behind |
| Decode step (ITL p50) | 13.4 ms | 5.2 ms | 2.6× behind |
| Time to first token (p50) | 14.5 ms | 23.2 ms | 1.6× ahead\* |
| Long prompts (ISL 2048) | runs | runs | eager attention cannot |

\* At this prompt length TTFT is mostly fixed per-call API overhead rather than
prefill compute, so read it as offline round-trip latency, not prefill speed.

Rows 1-4 are ISL 128 / OSL 256 unless stated, on the kernel the engine picks
for an A40 — `paged`, the Triton decode kernel. Where its preconditions do not
hold the choice falls back to `sdpa`, which runs anywhere and measures
1,745 tok/s on row 1.

The second row is the one that moved most this cycle: reading the KV pool in
place instead of gathering it makes the decode step **flat in context length**
(13.3 ms at 188 tokens, 13.1 ms at 1,028), which is worth 1.11× at the shape
above and 2.59× at ISL 1024 / OSL 1024. What remains against vLLM is a per-step
constant: no CUDA graphs, no varlen packing, and a decode grid that leaves an
84-SM GPU idle at B=1. All three are measured and queued, not guessed —
[`docs/roadmap.md`](docs/roadmap.md) carries the numbers, including two
optimisations that were built, measured and reverted.
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

Every performance change is measured against the config it improves on. A
general win replaces that config; a win scoped to a precondition (MoE,
quantization, long context) joins it instead — see "Shipping an improvement" in
[`docs/roadmap.md`](docs/roadmap.md).

## Roadmap

High-level direction:

- [x] Greedy/sampled generation from local safetensors
- [x] Paged KV cache
- [x] Continuous batching (async, streaming) — superseded static batching, now removed
- [ ] prefix caching
- [ ] `torch.compile` and CUDA graphs for decode
- [ ] Tensor parallelism (single node)
- [ ] Speculative decoding

Detailed, fine-grained backlog with scope and parity-test notes lives
in [`docs/roadmap.md`](docs/roadmap.md). Achieved milestones are
tracked separately in [`docs/milestones.md`](docs/milestones.md).

## License

Apache-2.0.
