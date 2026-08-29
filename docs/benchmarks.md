# Benchmarks

Every liteinfer configuration is measured against the one it improves on, and
against vLLM, on byte-identical prompts.

Live table: **[dashboard](https://valegian.github.io/liteinfer/)** ·
How to run: [`benchmarks/README.md`](../benchmarks/README.md)

---

## Setup

| | |
|---|---|
| **Model** | `Llama-3.2-1B-Instruct` (bfloat16) |
| **GPU** | NVIDIA A40 (46 GB) |
| **vLLM** | 0.28.0 |
| **Shape** | ISL 128 · OSL 256 |
| **Samples** | 200 (throughput) · 50 (latency) |
| **Decoding** | greedy, forced output length |

---

## Methodology

**Identical inputs.** One canonical dataset per (model, ISL, OSL), reused by every
engine. The file is not committed — it holds raw scraped text, secrets included —
but regenerates byte for byte from the pinned corpus revision and seed. Prompts
are ISL-controlled windows over
ShareGPT_V3 human turns — the corpus vLLM's own benchmarks use — at a pinned
dataset revision. Tokenizer round-trips move the realised length by at most a
few tokens (max deviation 3, mean 0.34 at ISL=128); each sample records the
length it actually has. Every result stores the SHA-256 of its prompt set, and
the report flags any group whose members disagree.

**Forced output length.** `min_tokens = max_tokens = OSL` with `ignore_eos`, so
output-length variance can never be mistaken for an engine difference. Lengths
are verified after every run; a run that violates this fails instead of
reporting.

**Warmup.** Before the clock starts, each run exercises the path it is about to
measure — real prompts at the benchmark's ISL, at the real batch width, twice —
so kernel autotuning and CUDA graph capture land outside the timed region.

**Isolation.** Each (config, mode) pair runs in a fresh process, so no GPU state
or allocator fragmentation carries between configs. Runs are sequential on a
single GPU; nothing else shares it.

**vLLM configuration.** Left at its strongest: its own scheduler, CUDA graphs
enabled, `gpu_memory_utilization=0.90`, bfloat16. The comparison is against
vLLM at its best, not a hobbled version of it.

### The two modes report disjoint metrics

| | `throughput` | `latency` |
|---|---|---|
| Offered load | every prompt at once | one request in flight |
| Reports | output tok/s, req/s | TTFT, ITL, E2E percentiles |

This split is deliberate. A per-request latency measured under saturation mostly
records where the request sat in the queue, so it characterises offered load
rather than the engine; at one request in flight there is no throughput to speak
of. Reporting both from one run is how benchmarks end up comparing queue depth
and calling it speed.

ITL is derived, not instrumented: with one request in flight and a forced output
length, `(e2e - ttft) / (osl - 1)` is the mean decode-step cost. TTFT comes from
a separate pass capped at one token. Both engines are therefore measured by the
same clock, with no per-token callbacks that each would implement differently.

---

## Configurations

`benchmarks/configs.py` is the matrix; each entry names the config it improves
on, and the report renders that as a 1:1 delta.

| Config | What it adds | Baseline |
|---|---|---|
| `liteinfer-nocache` | No KV cache; every step re-feeds the sequence | — |
| `liteinfer-eager` | KV cache via `DynamicCache` | `liteinfer-nocache` |
| `liteinfer-native-eager` | KV cache as plain tensors | `liteinfer-eager` |
| `liteinfer-paged` | Paged KV cache, block pool | `liteinfer-native-eager` |
| `liteinfer-eager-b4` | Static batching, B=4 | `liteinfer-eager` |
| `liteinfer-paged-b4` | Static batching, B=4, paged | `liteinfer-paged` |
| `liteinfer-continuous` | Continuous batching, up to 32 | `liteinfer-paged-b4` |
| `vllm`, `vllm-b4`, `vllm-continuous` | Reference, matched batch widths | — |

---

## Results

A40 · Llama-3.2-1B-Instruct · ISL 128 · OSL 256 · greedy · vLLM 0.28.0.
`vs base` compares each config to the one it improves on. `vs vLLM` compares to
vLLM at the **same batch width** — the only fair cross-engine comparison.

### Throughput — 200 prompts, all offered at once

| config | B | tok/s | req/s | wall (s) | vs base | vs vLLM |
|---|---:|---:|---:|---:|---:|---:|
| `liteinfer-nocache` | 1 | 60.9 | 0.2 | 840.5 | — | 0.32× |
| `liteinfer-eager` | 1 | 73.2 | 0.3 | 699.9 | **1.20×** | 0.39× |
| `liteinfer-native-eager` | 1 | 73.3 | 0.3 | 698.1 | 1.00× | 0.39× |
| `liteinfer-paged` | 1 | 52.7 | 0.2 | 971.0 | **0.72×** | 0.28× |
| `liteinfer-eager-b4` | 4 | 271.1 | 1.1 | 188.8 | **3.71×** | 0.37× |
| `liteinfer-native-eager-b4` | 4 | 270.9 | 1.1 | 189.0 | **3.69×** | 0.37× |
| `liteinfer-paged-b4` | 4 | 116.7 | 0.5 | 438.6 | 2.21× | 0.16× |
| `liteinfer-continuous` | 32 | 180.5 | 0.7 | 283.6 | 1.55× | 0.04× |
| `vllm` | 1 | 188.4 | 0.7 | 271.8 | — | — |
| `vllm-b4` | 4 | 724.0 | 2.8 | 70.7 | — | — |
| `vllm-continuous` | 32 | 4,465.9 | 17.4 | 11.5 | — | — |

### Latency — one request in flight

Run sequentially on an idle GPU. See *Certification* below for why this mode must
not be parallelised.

| config | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | E2E p50 | vs base |
|---|---:|---:|---:|---:|---:|---:|
| `liteinfer-nocache` | 14.7 ms | 15.9 ms | 16.5 ms | 16.5 ms | 4,210.3 ms | — |
| `liteinfer-eager` | 15.7 ms | 17.1 ms | 13.8 ms | 13.9 ms | 3,535.1 ms | **1.19×** |
| `liteinfer-native-eager` | 15.1 ms | 16.7 ms | 14.1 ms | 14.1 ms | 3,598.0 ms | 0.98× |
| `liteinfer-paged` | 18.5 ms | 20.2 ms | 18.8 ms | 18.9 ms | 4,816.7 ms | **0.75×** |
| `liteinfer-eager-b4` | 16.0 ms | 17.8 ms | 14.0 ms | 14.2 ms | 3,576.5 ms | 0.99× |
| `liteinfer-native-eager-b4` | 15.2 ms | 17.1 ms | 14.2 ms | 14.3 ms | 3,645.9 ms | 0.99× |
| `liteinfer-paged-b4` | 18.7 ms | 20.6 ms | 19.0 ms | 19.1 ms | 4,870.1 ms | 0.99× |
| `liteinfer-continuous` | 18.8 ms | 20.5 ms | 18.9 ms | 19.0 ms | 4,843.4 ms | 1.01× |
| `vllm` | 21.3 ms | 25.7 ms | 5.2 ms | 5.3 ms | 1,359.6 ms | — |
| `vllm-b4` | 22.1 ms | 31.3 ms | 5.2 ms | 5.3 ms | 1,358.9 ms | — |
| `vllm-continuous` | 21.6 ms | 33.1 ms | 5.2 ms | 5.3 ms | 1,358.8 ms | — |

---

## What the numbers say

**The paged KV cache is a net regression.** It reaches 0.72× of native-eager on
throughput (73.3 → 52.7 tok/s) and 0.75× on ITL (14.1 → 18.8 ms) — two
independent measurements agreeing within 4%, so the cost is a fixed per-decode-step
tax, not a scheduling artifact. `PagedKVCache._gather_all`
(`liteinfer/cache/paged_kv_cache.py:193`) runs once per layer per step: a Python
loop over every sequence that slices each block and `torch.cat`s the whole K/V
history into a fresh padded tensor. At B=32 across 16 layers that is roughly 8k
Python-level tensor ops and ~200 MB copied **per step** — a batch-32 decode step
costs 130 ms against 19 ms at batch 1, where a bandwidth-bound decode should cost
about the same at both widths. And the blunt version: **paged decode (18.8 ms) is
slower per step than recomputing the whole sequence with no cache at all
(16.5 ms)**. §2.3 is not an optimization, it is what makes the paged path viable.

**Continuous batching inherits that tax, and under-scales.** Going from B=4 to
B=32 — an 8× wider batch — buys liteinfer 1.55× where vLLM gets 6.17×. This is
*not* a scheduling failure: instrumenting `ContinuousScheduler.schedule` shows
every step running at full width 32. The batch fills; each batched step is simply
enormous. Because continuous batching is built on paged, its 180.5 tok/s lands
*below* static batching on the eager cache (271.1 tok/s) — the async engine is
currently slower than the synchronous one. Causes: the per-step paged tax above,
and the two-pass step whenever prefill and decode coexist (§1.3).

**Static batching already works.** liteinfer converts B=1 → B=4 at 3.70×; vLLM
manages 3.84× on the same transition. Scheduling and batching are not the problem.

**The decode-step gap to vLLM is a flat ~2.6× constant** (ITL 13.8 vs 5.2 ms;
throughput 0.39× at B=1, 0.37× at B=4). It does not widen with batch size — the
blow-up at B=32 is entirely the paged/continuous tax layered on top. In absolute
terms vLLM sustains 68% of the A40's 696 GB/s memory bandwidth on a 2.48 GB model
(5.2 ms vs a 3.56 ms hardware floor); liteinfer sustains 26%. That headroom is
§3.1–3.3: CUDA graphs and fused attention.

**The native-eager rewrite is performance-neutral.** eager vs native-eager:
73.2 vs 73.3 tok/s at B=1, 271.1 vs 270.9 at B=4, 13.8 vs 14.1 ms ITL. All inside
run-to-run variance. Dropping `DynamicCache` removed a transformers coupling and
gave the paged cache something to build on; it did not make anything faster, and
the docs should not imply otherwise.

**The KV cache itself buys only ~1.2× at this shape** — 1.20× on throughput
(60.9 → 73.2 tok/s) and 1.19× on ITL (16.5 → 13.8 ms). At ISL=128/OSL=256 on a 1B
model decode is kernel-launch bound, so re-feeding the whole sequence costs little
more than feeding one token while the cache adds per-step `torch.cat` growth.
Expect this ratio to widen sharply with sequence length; measuring that is §8.3.

**TTFT: liteinfer is ahead, but read it narrowly.** 15.7 ms vs vLLM's 21.3 ms,
1.36×. At ISL=128 both engines' TTFT is dominated by fixed per-call API overhead
rather than prefill compute — measured on a 2-token prompt, vLLM spends 11.1 ms
before any real work, and prefilling 128 tokens costs it only 3.4 ms more. So this
measures offline-API round-trip latency, where vLLM pays IPC to a separate engine
process, and says little about prefill throughput at realistic prompt lengths.
A longer-ISL sweep (§8.3) is what would settle it.

### Internal consistency

The two modes are independent measurement paths — saturated wall-clock versus
per-request percentiles at B=1 — and they agree wherever they overlap: the KV
cache gain (1.20× / 1.19×) and the paged penalty (0.72× / 0.75×). Single-request
latency is also flat across `max_num_seqs` for both engines (liteinfer ~14 ms at
B=1 and B=4; vLLM 5.2 ms at B=1, 4 and 32), confirming latency mode really does
keep one request in flight. The derived ITL reconstructs E2E to 0.3%
(28.1 + 255 × 5.2 = 1,354 ms vs 1,350 ms measured).

## Certification

**Against vLLM's own tooling.** At the identical shape (ISL 128, OSL 256, 200
prompts, `max_num_seqs=32`), `vllm bench throughput` reports 4,421.68 output
tok/s and 17.27 req/s; this harness reports 4,465.9 and 17.4 — **1.0% apart**.
Both emit exactly 51,200 output tokens. vLLM's own script also uses
`ignore_eos=True`, and defaults to *zero* warmup where this harness does two
rounds.

**Variance.** Re-running configs alone versus inside the parallel sweep moved
throughput by 2.0% in one direction and ITL by 3.8% in the other — noise, not a
systematic parallel bias, so CPU-core pinning is doing its job. Treat ±4% as the
resolution floor; every effect reported above is ≥1.19×.

**One measurement was thrown out.** In the first parallel sweep vLLM's TTFT read
28.1 ms. Because vLLM's offline TTFT is mostly fixed IPC overhead to its engine
process, it is CPU-scheduling sensitive in a way its ITL is not: re-run
sequentially it dropped to 21.3 ms (−24%) while every ITL figure and every
liteinfer number stayed put. The latency tables above are all from the sequential
re-run, and `bench run` now warns when `--gpus` is combined with latency mode.

---

## Known gaps

Ordered by cost, largest first.

| Gap | Measured | Root cause | Roadmap |
|---|---|---|---|
| Paged cache costs more than it saves | 0.72× vs native-eager; slower per step than no cache at all | Block-table gather rebuilds K/V before every forward pass | [§2.3](roadmap.md#23-paged-kv-cache-performance-fused-kernel) |
| Continuous batching barely scales with width | 1.55× for 8× batch, vs vLLM's 6.17× | Paged tax per step, plus two-pass step when prefill and decode coexist | [§1.3](roadmap.md#13-chunked-prefill--single-pass-mixed-batching), [§2.3](roadmap.md#23-paged-kv-cache-performance-fused-kernel) |
| Decode step ~2.6× slower than vLLM at every width | ITL 13.8 ms vs 5.2 ms | No CUDA graphs, no fused attention | [§3.1](roadmap.md#31-torchcompile-of-the-forward-path), [§3.2](roadmap.md#32-cuda-graph-capture-for-decode), [§3.3](roadmap.md#33-flash--sdpa-attention) |
| KV-cache benefit unquantified across shapes | 1.20× at ISL 128 / OSL 256 only | Single measured shape; the crossover is sequence-length dependent | [§8.3](roadmap.md#83-sequence-length-sweep-workload) |
| No prefix-cache benefit | not measured | Prefix caching not implemented | [§2.2](roadmap.md#22-prefix-sharing) |

See [`docs/roadmap.md`](roadmap.md) for the full backlog.

---

<details>
<summary>Historical results — v0 benchmark system, no ISL/OSL control, not comparable to the numbers above</summary>

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
|---|---|---:|---:|---:|---:|
| liteinfer | 1 | 16.1 ms | 17.0 ms | 1948 ms | 64 |
| liteinfer-kvcache | 1 | 18.8 ms | 29.6 ms | 1936 ms | 54 |
| liteinfer-native-kvcache | 1 | 18.4 ms | 26.6 ms | 1974 ms | 56 |
| liteinfer-paged-kvcache | 1 | 18.1 ms | 27.6 ms | 2104 ms | 51 |
| vllm | 1 | 25.9 ms | 29.4 ms | 692 ms | 183 |

</details>
