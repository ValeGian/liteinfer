# Benchmark Results

> Last updated: 2026-05-19 · [live dashboard](https://valegian.github.io/liteinfer/)

---

## Setup

| | |
|---|---|
| **Model** | `meta-llama/Llama-3.2-1B-Instruct` (bfloat16) |
| **GPU** | NVIDIA A40 (46 GB VRAM) |
| **vLLM** | 0.21.0 |
| **Engines** | `liteinfer`, `vllm`, `trtllm` |
| **Decoding** | Greedy (temperature=0) |

ISL/OSL-controlled inputs via canonical JSONL dataset; all engines receive
identical prompts (SHA-256 verified per run). 4 warmup requests per engine
before wall clock starts.

---

## Methodology

- **Canonical JSONL dataset:** one fixed file per `(model, ISL, OSL)` tuple, generated
  once and reused across all engines. Corpus: ShareGPT_V3 human turns
  (`anon8231489123/ShareGPT_Vicuna_unfiltered`, revision pinned in `dataset.py`),
  the same source used by vLLM's own benchmarks — results are directly comparable.
  See `benchmarks/dataset.py`.
- **Forced output length:** `min_tokens=max_tokens=OSL`, `ignore_eos=True`. This
  eliminates output-length variance as a confound.
- **Throughput mode:** all requests submitted concurrently; `tok/s` and `req/s` are
  the primary metrics.
- **Latency mode:** `batch_size=1`, sequential submission; TTFT/ITL/E2E percentiles
  are primary.
- **TTFT in throughput mode** includes scheduling contention and is not comparable to
  TTFT from a latency-mode run. Compare TTFT only within runs of the same benchmark
  type.

---

## Results

See the **[live dashboard](https://valegian.github.io/liteinfer/)** for current
promoted results across all engines and ISL/OSL configurations.

---

## What these numbers mean

| Gap | Root cause | Roadmap item |
|---|---|---|
| 2.86× slower throughput at B=4 (`liteinfer-b4` vs `vllm-b4`) | Eager attention, no CUDA graphs, no FlashAttention | [§3.1](roadmap.md#31-torchcompile-of-the-forward-path), [§3.2](roadmap.md#32-cuda-graph-capture-for-decode), [§3.3](roadmap.md#33-flash--sdpa-attention) |
| `liteinfer-paged-b4` 38% slower than `liteinfer-b4` at B=4 | Block-table scatter/gather on each decode step; no fused kernel | [§2.3](roadmap.md#23-paged-kv-cache-performance-fused-kernel) |
| `liteinfer-continuous` ≈ `liteinfer-paged-b4` at matched B | Two-pass step overhead when new seqs join decode batch | [§1.3](roadmap.md#13-chunked-prefill--single-pass-mixed-batching) |
| ~2.8× slower decode tok/s at B=1 vs vLLM | No CUDA graphs, no FlashAttention | [§3.2](roadmap.md#32-cuda-graph-capture-for-decode), [§3.3](roadmap.md#33-flash--sdpa-attention) |
| No prefix-cache benefit yet | Prefix caching not implemented | [§2.2](roadmap.md#22-prefix-sharing) |

See [`docs/roadmap.md`](roadmap.md) for the full backlog.

---

<details>
<summary>Historical results — generated with v0 benchmark system (no ISL/OSL control). See the live dashboard for current results.</summary>

## Throughput workload (v0)

32 short independent prompts · all submitted at once · greedy · max 64 tokens · 1 warmup

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

## Latency workload (v0)

20 calls · sequential · greedy · max 128 tokens · 1 warmup

| Engine | B | TTFT p50 | TTFT p99 | E2E p50 | tok/s |
|---|---:|---:|---:|---:|---:|
| liteinfer | 1 | 16.1 ms | 17.0 ms | 1948 ms | 64 |
| liteinfer-kvcache | 1 | 18.8 ms | 29.6 ms | 1936 ms | 54 |
| liteinfer-native-kvcache | 1 | 18.4 ms | 26.6 ms | 1974 ms | 56 |
| liteinfer-paged-kvcache | 1 | 18.1 ms | 27.6 ms | 2104 ms | 51 |
| vllm | 1 | 25.9 ms | 29.4 ms | 692 ms | 183 |

</details>
