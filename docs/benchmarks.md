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

Everything above `liteinfer-continuous` was measured and then **removed from the
codebase** — the engine now has a single execution path. Their entries stay in
the matrix, flagged `historical`, so the progression still renders; `bench run`
refuses to run them.

| Config | What it added | Baseline | |
|---|---|---|---|
| `liteinfer-nocache` | No KV cache; every step re-feeds the sequence | — | removed |
| `liteinfer-eager` | KV cache via `DynamicCache` | `liteinfer-nocache` | removed |
| `liteinfer-native-eager` | KV cache as plain tensors | `liteinfer-eager` | removed |
| `liteinfer-paged` | Paged KV cache, block pool | `liteinfer-native-eager` | removed |
| `liteinfer-eager-b4` | Static batching, B=4 | `liteinfer-eager` | removed |
| `liteinfer-native-eager-b4` | Static batching, B=4, plain tensors | `liteinfer-native-eager` | removed |
| `liteinfer-paged-b4` | Static batching, B=4, paged | `liteinfer-paged` | removed |
| `liteinfer-continuous` | Continuous batching, up to 32 | `liteinfer-paged-b4` | eager kernel |
| `liteinfer-sdpa` | Attention through PyTorch SDPA | `liteinfer-continuous` | **current** |
| `vllm`, `vllm-b4`, `vllm-continuous` | Reference, matched batch widths | — | |

`liteinfer-continuous` stays runnable rather than becoming `historical`. Note what
that is not: the replacement rule, read literally, says delete the eager kernel —
`sdpa` covers its whole domain and wins there. It is kept as a deliberate
exception, because it is the only attention written out in the repo and the
independent reference the fused path is checked against, and because that costs
22 lines. See `milestones.md`.

---

## Results

A40 · Llama-3.2-1B-Instruct · ISL 128 · OSL 256 · greedy · vLLM 0.28.0.
`vs base` compares each config to the one it improves on. `vs vLLM` compares to
vLLM at the **same batch width** — the only fair cross-engine comparison.

### Throughput — 200 prompts, all offered at once

| config | B | tok/s | req/s | wall (s) | vs base | vs vLLM |
|---|---:|---:|---:|---:|---:|---:|
| `liteinfer-nocache` | 1 | 60.5 | 0.2 | 846.7 | — | 0.32× |
| `liteinfer-eager` | 1 | 73.4 | 0.3 | 697.9 | **1.21×** | 0.39× |
| `liteinfer-native-eager` | 1 | 72.7 | 0.3 | 704.6 | 0.99× | 0.39× |
| `liteinfer-paged` | 1 | 66.8 | 0.3 | 766.7 | 0.92× | 0.35× |
| `liteinfer-eager-b4` | 4 | 281.6 | 1.1 | 181.8 | **3.84×** | 0.39× |
| `liteinfer-native-eager-b4` | 4 | 277.6 | 1.1 | 184.4 | 3.82× | 0.38× |
| `liteinfer-paged-b4` | 4 | 252.9 | 1.0 | 202.4 | 3.79× | 0.35× |
| `liteinfer-continuous` | 32 | 1,268.3 | 5.0 | 40.4 | **5.01×** | 0.28× |
| `vllm` | 1 | 188.4 | 0.7 | 271.8 | — | — |
| `vllm-b4` | 4 | 724.0 | 2.8 | 70.7 | — | — |
| `vllm-continuous` | 32 | 4,466.6 | 17.4 | 11.5 | — | — |

### Latency — one request in flight

Run sequentially on an idle GPU. See *Certification* below for why this mode must
not be parallelised.

| config | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | E2E p50 | vs base |
|---|---:|---:|---:|---:|---:|---:|
| `liteinfer-nocache` | 14.8 ms | 16.1 ms | 16.5 ms | 16.5 ms | 4,215.6 ms | — |
| `liteinfer-eager` | 15.7 ms | 16.3 ms | 13.7 ms | 13.9 ms | 3,521.2 ms | **1.20×** |
| `liteinfer-native-eager` | 14.8 ms | 16.1 ms | 13.7 ms | 13.9 ms | 3,518.5 ms | 1.00× |
| `liteinfer-paged` | 19.3 ms | 21.1 ms | 15.3 ms | 15.3 ms | 3,910.7 ms | 0.90× |
| `liteinfer-eager-b4` | 15.8 ms | 17.7 ms | 14.0 ms | 14.1 ms | 3,596.6 ms | 0.98× |
| `liteinfer-native-eager-b4` | 14.9 ms | 16.7 ms | 13.9 ms | 13.9 ms | 3,561.1 ms | 0.99× |
| `liteinfer-paged-b4` | 19.0 ms | 20.9 ms | 15.0 ms | 15.0 ms | 3,832.2 ms | 1.02× |
| `liteinfer-continuous` | 19.0 ms | 21.0 ms | 14.9 ms | 14.9 ms | 3,807.6 ms | 1.01× |
| `vllm` | 22.7 ms | 27.4 ms | 5.2 ms | 5.2 ms | 1,354.8 ms | — |
| `vllm-b4` | 22.6 ms | 27.0 ms | 5.2 ms | 5.2 ms | 1,353.9 ms | — |
| `vllm-continuous` | 23.2 ms | 32.2 ms | 5.2 ms | 5.2 ms | 1,356.2 ms | — |

---

## What the numbers say

**The gap to vLLM is one flat constant, not a compounding one.** liteinfer runs
at 0.35× of vLLM at B=1, 0.35× at B=4 and 0.28× at B=32. A roughly uniform ~3×
across every batch width is the signature of a per-step cost — no CUDA graphs, no
fused attention — and that is §3.1–3.3. In absolute terms vLLM sustains 68% of
the A40's 696 GB/s on a 2.48 GB model (5.2 ms per decode step against a 3.56 ms
hardware floor); liteinfer sustains about 25%. A deficit that *widened* with
concurrency would point at something scaling with batch × sequence length; that
is no longer what the numbers show.

**Batching works at both tiers.** Static batching converts B=1 → B=4 at 3.84×
against vLLM's 3.84× on the same transition. Continuous batching converts
B=4 → B=32 at 5.01× against vLLM's 6.17×. Whatever the two-pass prefill+decode
step (§1.3) costs, it is a modest residual rather than a dominant term:
continuous ITL (14.9 ms) sits level with static paged (15.0 ms).

**Paging costs ~8-10%, and that cost is now memory traffic.** Paged reaches 0.92×
of native-eager on throughput at B=1, 0.90× at B=4, and 0.90× on ITL. The gather
still copies the whole K/V history every decode step; removing the copy is what
§2.3's fused kernel is for. Paged is faster than running with no cache (66.8 vs
60.5 tok/s), and continuous batching is far faster than static (1,268 vs 282
tok/s) — both of which were *false* before the gather was vectorised, and both of
which were implementation artifacts rather than properties of the designs.

**The native-eager rewrite is performance-neutral.** eager vs native-eager: 73.4
vs 72.7 tok/s at B=1, 281.6 vs 277.6 at B=4, 13.7 vs 13.7 ms ITL. All inside
run-to-run variance. Dropping `DynamicCache` removed a transformers coupling and
gave the paged cache something to build on; it did not make anything faster.

**The KV cache itself buys ~1.2× at this shape** — 1.21× on throughput
(60.5 → 73.4 tok/s) and 1.20× on ITL (16.5 → 13.7 ms). At ISL=128/OSL=256 on a 1B
model decode is kernel-launch bound, so re-feeding the whole sequence costs little
more than feeding one token while the cache adds per-step `torch.cat` growth.
Expect this to widen sharply with sequence length; measuring that is §8.3.

**TTFT: liteinfer is ahead, but read it narrowly.** 15.7 ms against vLLM's
22.7 ms. At ISL=128 both engines' TTFT is dominated by fixed per-call API overhead
rather than prefill compute — measured on a 2-token prompt, vLLM spends 11.1 ms
before any real work and prefilling 128 tokens costs it only 3.4 ms more. This
measures offline round-trip latency, where vLLM pays IPC to a separate engine
process, and says little about prefill throughput at realistic prompt lengths.
Paged pays ~4 ms more TTFT than eager because it builds its slot table on the
first decode step.

### Internal consistency

The two modes are independent measurement paths — saturated wall-clock versus
per-request percentiles at B=1 — and they agree wherever they overlap: the KV
cache gain (1.21× / 1.20×) and the paged penalty (0.92× / 0.90×). Single-request
latency is also flat across `max_num_seqs` for both engines (liteinfer 13.7 ms at
B=1 and 13.9 ms at B=4; vLLM 5.2 ms at B=1, 4 and 32), confirming latency mode really does
keep one request in flight. The derived ITL reconstructs E2E to 0.3%
(28.1 + 255 × 5.2 = 1,354 ms vs 1,350 ms measured).

## Shape sensitivity

Everything above is one shape: ISL 128 / OSL 256. `bench sweep` measures across a
grid, and the first sweep showed that treating a single shape as general would
have been a mistake twice over.

### The KV cache's advantage grows with generated length

Measured on the pre-#17 tree, the last commit where `liteinfer-nocache` still
exists (n=32, ISL 128, one A40):

| OSL | no cache | KV cache | cache is worth |
|---:|---:|---:|---:|
| 256 | 60.7 | 69.5 | 1.15× |
| 512 | 44.6 | 71.9 | 1.61× |
| 1024 | 24.1 | 69.6 | **2.88×** |

Cached throughput barely moves; the recompute path collapses, because its work per
step grows with the sequence it re-reads. **The 1.21× headline is the weakest point
on this curve, not a general figure** — at OSL 1024 the cache is worth nearly 3×,
and it keeps climbing with length.

Re-measuring it required checking out history, which is the standing cost of the
"a clear win replaces what it beat" rule: once the loser's code is deleted, its
claims can only be re-validated against an old tree.

### Long prompts are liteinfer's weak shape

| shape | liteinfer | vLLM |
|---|---:|---:|
| 128 / 256 | 1,369.1 | 4,458.7 |
| 128 / 1024 | 942.3 | 4,211.9 |
| 1024 / 256 | 646.0 | 2,871.2 |
| 1024 / 1024 | 574.1 | 3,240.6 |

An 8x longer prompt costs liteinfer 2.1x throughput while vLLM gives up 1.6x, so
the gap widens from 3.3x to 4.4x. The bottom two rows are also the ones the sweep
was worth running for: on the first pass liteinfer did not produce them at all. It
died with an out-of-memory in `softmax`, and only vLLM finished the shape. Both
halves of that failure are now fixed — the pool below, the allocation itself in
the section after it.

Eager attention materialises the score matrix and the softmax upcasts it to fp32:
at 32 sequences × 32 heads × 1024², that is 4.00 GiB in one allocation. vLLM never
materialises it, which is what FlashAttention is for — filed as §3.3, and a
capability gate rather than an optimisation.

The KV pool compounded it. It claimed 32.74 GiB when the configuration could only
ever use 4.00 GiB — `max_num_seqs` × `max_model_len` × 32 KB — so the engine ran
out of memory while holding ~29 GiB of KV space it was structurally unable to
reach. Sizing the pool to that ceiling freed the surplus, which is why the shape
runs at all.

The materialisation itself is untouched by that fix and fails again at longer
prompts or wider batches. Removing it is §3.3, below.

### The fused kernel buys prompt length, not throughput

Attention now goes through `torch.nn.functional.scaled_dot_product_attention`,
which tiles the softmax and never assembles the score matrix. Measured directly
on one attention call — 4 sequences × 32 heads × 2048 tokens, bf16, the mask the
engine actually builds:

| kernel | peak allocation for the call |
|---|---:|
| `eager` | 5,192 MiB |
| `sdpa` | 96 MiB |

End to end it changes almost nothing until it changes everything. Both kernels
re-measured on the same tree, same datasets:

| shape | `eager` | `sdpa` | |
|---|---:|---:|---:|
| 128 / 256 | 1,344.4 | 1,445.5 | 1.08× |
| 128 / 1024 | 937.9 | 959.7 | 1.02× |
| 1024 / 256 | 645.2 | 711.1 | 1.10× |
| 1024 / 1024 | 572.9 | 592.8 | 1.03× |
| **2048 / 128** | **OOM** | **385.3** | *runs at all* |

Four of those five rows are inside run-to-run variance: **this is not a
speedup**, and the ~1.1× at 1024/256 is the top of the noise band rather than an
effect. The last row is the whole feature. At ISL 2048 eager asks for
**16.02 GiB in one allocation** and dies — 32 sequences × 32 heads × 2048², at
4 bytes once the softmax upcasts — while `sdpa` completes the same workload.
That number is the score matrix and nothing else, which is why removing it moves
the ceiling and not the clock.

It is worth being precise about which fix unlocked which shape. The ISL 1024
rows ran before this change: **PR #20's pool sizing** freed the memory they
needed, and both kernels serve them. ISL 2048 is the first shape only the fused
kernel can reach.

Which backend serves it is not the obvious one. Left-padded batches need an
explicit additive mask, and FlashAttention accepts only `is_causal`, so PyTorch
falls to its **memory-efficient** backend, which does take a mask and tiles the
same way. Probed on these tensors: flash unavailable, mem-efficient available,
and forcing the math fallback instead costs +4,872 MiB against its +32 MiB.

---

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
| ~3x slower than vLLM per decode step, at every batch width | ITL 13.7 ms vs 5.2 ms; ~25% of memory bandwidth vs vLLM's 68% | No CUDA graphs, no fused attention | [§3.1](roadmap.md#31-torchcompile-of-the-forward-path), [§3.2](roadmap.md#32-cuda-graph-capture-for-decode), [§3.3](roadmap.md#33-flash--sdpa-attention) |
| Paging costs 8-10% against eager | 0.92x throughput at B=1, 0.90x at B=4 and on ITL | The gather still copies the whole K/V history each step | [§2.3](roadmap.md#23-paged-kv-cache-performance-fused-kernel) |
| Continuous batching scales slightly below vLLM | 5.01x for 8x width vs vLLM's 6.17x | Two-pass step when prefill and decode coexist | [§1.3](roadmap.md#13-chunked-prefill--single-pass-mixed-batching) |
| KV-cache benefit unquantified across shapes | 1.21x at ISL 128 / OSL 256 only | Single measured shape; the crossover is sequence-length dependent | [§8.3](roadmap.md#83-sequence-length-sweep-workload) |
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
