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
| **Samples** | 200 (throughput) · 200 (latency; the historical rows used 50) |
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
| `liteinfer-sdpa` | Attention through PyTorch SDPA | `liteinfer-continuous` | fallback |
| `liteinfer-paged-attn` | Decode attention reads the KV pool in-kernel | `liteinfer-sdpa` | **ships** |
| `vllm`, `vllm-b4`, `vllm-continuous` | Reference, matched batch widths | — | |

`liteinfer-paged-attn` is the engine as it ships on CUDA: `EngineConfig()`
resolves its attention kernel from the device and the model, and picks the paged
one wherever its preconditions hold. `liteinfer-sdpa` is what that choice falls
back to — off CUDA, without Triton, or on a model whose head dimension the
Triton kernel cannot tile — so both rows describe shipping configurations rather
than one being an experiment. Every row above them is a design that was measured
and then removed, kept so the progression still renders.

Each row pins its kernel by name, so the matrix measures the kernel the row
claims and not whatever this machine would have chosen.
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

**Two kinds of `vs base`, and the table cannot mark which is which.** The three
runnable liteinfer configs are re-measured together, so the deltas between them
isolate what their code differs by. A delta against a *removed* config cannot be:
its number is frozen at the engine of the day it was deleted, so
`liteinfer-continuous`'s 6.48× over `liteinfer-paged-b4` is "the engine now
against the engine then", not "continuous batching against static batching". That
is the standing cost of deleting a path the benchmark proved worse — the roadmap
takes that trade deliberately, and this is where the bill arrives.

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
| `liteinfer-continuous` | 32 | 1,638.8 | 6.4 | 31.2 | 6.48×† | 0.37× |
| `liteinfer-sdpa` | 32 | 1,744.8 | 6.8 | 29.3 | 1.06× | 0.39× |
| `liteinfer-paged-attn` | 32 | **1,930.0** | 7.5 | 26.5 | **1.11×** | 0.43× |
| `vllm` | 1 | 188.4 | 0.7 | 271.8 | — | — |
| `vllm-b4` | 4 | 724.0 | 2.8 | 70.7 | — | — |
| `vllm-continuous` | 32 | 4,466.6 | 17.4 | 11.5 | — | — |

† against a removed config, so it measures two engines two milestones apart. The
two rows above it are same-session and measure only their own code.

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
| `liteinfer-sdpa` | **14.0 ms** | 15.9 ms | 13.9 ms | 14.0 ms | 3,546.6 ms | **1.07×** |
| `liteinfer-paged-attn` | 14.5 ms | 16.2 ms | **13.4 ms** | 13.5 ms | **3,431.3 ms** | 1.03× |
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

**Paging cost ~8-10%, and paying it back is what §2.3 did.** Paged reached 0.92×
of native-eager on throughput at B=1, 0.90× at B=4, and 0.90× on ITL, because the
gather copied the whole K/V history every decode step. `liteinfer-paged-attn`
removes that copy — the block-addressed cache is now the *faster* design rather
than a slower one bought for its memory behaviour, by 1.11× at this shape and up
to 2.59× at ISL 1024 / OSL 1024. Paged was already faster than running with no cache (66.8 vs
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

**This is the shape §2.3 helped most.** A long prompt is a long KV history from
the first decode step, which is exactly what the gather was charging for: with
paged decode the 1024 / 1024 row goes 688.9 → 1,783.6 tok/s and the gap to vLLM
at that shape closes from 4.7× to 1.8×. See *Paged decode stops the step growing
with context*.

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

### Paged decode stops the step growing with context

`liteinfer-paged-attn` replaces the decode gather with a Triton kernel that reads
the KV pool through the slot table. All three runnable configs re-measured
together, which is what these stored rows are:

| shape | `continuous` | `sdpa` | `paged` | paged vs sdpa |
|---|---:|---:|---:|---:|
| 128 / 256 | 1,638.8 | 1,744.8 | 1,930.0 | 1.11× |
| 128 / 1024 | 1,181.1 | 1,205.9 | 1,921.9 | 1.59× |
| 1024 / 256 | 702.9 | 781.0 | 1,542.0 | 1.97× |
| 1024 / 1024 | 664.9 | 688.9 | 1,783.6 | **2.59×** |
| 2048 / 128 | OOM | 398.3 | 820.1 | 2.06× |

Read the first row against the fourth: at the headline shape the change is 1.11×,
only just past the 1.1× floor this repo treats as an effect at all, and at ISL
1024 / OSL 1024 it is 2.59×. Same code in every row. What differs is how much KV
there is to copy — and the shape this repo has always led with is the one with
the least of it.

The `continuous` column is why all three configs were re-run and not just the two
being compared. Its stored rows predated PRs #25-#31, so refreshing `sdpa` alone
would have moved the sdpa-vs-continuous delta at 1024 / 1024 from 1.03× to 1.20×
and credited the SDPA kernel with five other PRs' work — the same error one row
further down. A `vs base` delta is only a claim about code if both engines are the
same age.

The mechanism is visible one level down. Timing `ContinuousModelRunner.decode`
directly at B=32, so the number is the forward pass and nothing else:

| context | `sdpa` | `paged` | |
|---:|---:|---:|---:|
| 188 | 17.14 ms | 13.28 ms | 1.29× |
| 1,028 | 21.70 ms | **13.05 ms** | 1.66× |

The paged step is **flat**: 13.28 ms at 188 tokens of context and 13.05 ms at
1,028. The gathering step is not, because the bytes it moves are proportional to
the history it copies. That is the whole result — decode stops paying for context
it is not reading — and it is why the win grows with every shape that generates
more or prompts longer.

Per attention call, bf16, 8 KV heads and 32 query heads, the operation the kernel
replaces:

| B | context | gather + sdpa | paged | |
|---:|---:|---:|---:|---:|
| 1 | 190 | 0.107 ms | 0.047 ms | 2.27× |
| 32 | 190 | 0.294 ms | 0.038 ms | 7.73× |
| 32 | 1,024 | 1.478 ms | 0.131 ms | 11.31× |
| 32 | 2,048 | 2.916 ms | 0.253 ms | 11.54× |

Three things the kernel does not do explain the ratio: it never copies the
history, it never expands 8 KV heads to 32 (`_repeat_kv` was 0.74 GiB of writes
per step), and it never touches the padding, because it takes each sequence's
context length rather than a mask over the batch's longest.

**At B=1 it is worth nothing, and that is the same story.** Latency mode runs one
request at a time: ITL p50 13.9 → 13.4 ms, E2E 3,546.6 → 3,431.3 ms, TTFT 14.0 →
14.5 ms. All three are inside the ±4% floor, so the honest reading is *no effect*.
Two reasons, and they compound: a single request has ~190 tokens of history to
gather, which is the cheapest case in the table above, and the kernel's grid is one
program per (sequence, KV head) — 8 programs on an 84-SM GPU. The B=1 row of the
per-call table is 2.27× where B=32 is 7.73×, and closing that is §2.7.

**The isolated number predicted this one.** 16 layers × (0.294 − 0.038) = 4.09 ms
of predicted saving against 3.86 ms measured — 94% realised, where the three
items before it came in at 30-60% of their component benchmarks. The difference is
what is being removed. Those items removed *overhead* that partly overlapped with
real work, so the isolated figure double-counted; this one removes a memory copy
that was serialised with everything else, and a copy that does not happen is
worth exactly what it cost.

### What five changes bought, in one place

`liteinfer-continuous` is the engine at the start of that sequence and
`liteinfer-sdpa` is where it ended. **The "before" column is a stored
measurement from before those five PRs, and the live matrix no longer reproduces
it**: `liteinfer-continuous` has since been re-measured on today's engine, which
is what makes the current `vs base` deltas mean what they say. Kept here because
it is the only place the sequence is accounted for.

| | before (as measured then) | after | |
|---|---:|---:|---:|
| Throughput, ISL 128 / OSL 256 | 1,268.3 tok/s | **1,732.8** | 1.37× |
| Throughput, ISL 128 / OSL 1024 | 942.3 | **1,230.4** | 1.31× |
| Decode step, ITL p50 | 14.9 ms | **13.9 ms** | 1.07× |
| Time to first token, p50 | 19.0 ms | **14.0 ms** | 1.36× |
| Longest prompt that runs | ISL 1024 | **ISL 2048** | — |

The ITL row is the one worth reading twice. Throughput moved 1.37x while the
decode step moved 1.07x, and both are correct: three of the five changes —
incremental detokenisation, vectorised sampling, one transfer per step — cost
work *per sequence in the batch*, so removing them shows up in aggregate
throughput and barely in a single request's step time. Latency mode runs one
request at a time and cannot see them. A benchmark that reported only one of
these two numbers would have told half the story either way.

**And the 1.37x was never the SDPA kernel's.** Re-measured together on today's
engine, `liteinfer-sdpa` is **1.06×** over `liteinfer-continuous` at this shape —
the two configs differ only by the attention kernel, and that is what a kernel
swap is worth here. The rest of the 1.37x was the other four PRs, which the
stale baseline had been silently crediting to the kernel for five milestones.
Re-running a baseline is not bookkeeping; it is how a delta stops being a
different claim than it appears to be.

---

## Where the engine's time goes

Every performance item on the roadmap optimises the forward pass. §6.4 added the
attribution that checks whether the forward pass is where the time is:
`EngineStats.time` charges each part of a step to a stage, and the loop accounts
for 99% of its own wall time. Measured on 32 sequences, A40, Llama-3.2-1B:

| stage | OSL 256 | OSL 1024 |
|---|---:|---:|
| forward pass | 84.9% | 76.6% |
| **deliver** — one `StreamEvent` per sequence, which detokenises | **6.9%** | **16.9%** |
| sample | 7.0% | 5.5% |
| schedule | 0.3% | 0.2% |
| unattributed — asyncio, queue puts | 1.0% | 0.8% |

Two things worth taking from it. **Scheduling and the async plumbing are free** —
together under 1.5%, and no candidate for optimisation despite being the parts
that look most like overhead. And **detokenisation was not**: `_build_event`
re-decoded a sequence's entire output text on every step, so four times the
output length cost twelve times the work — the second-largest cost in the engine
at OSL 1024, ahead of sampling and behind only the model itself.

§5.2 fixed that by decoding a short window per token instead of the whole
prefix, and the same attribution measures the result:

| stage | OSL 256 | OSL 1024 |
|---|---:|---:|
| deliver, before | 6.9% | 16.9% |
| **deliver, after** | **1.0%** | **1.0%** |

Flat with output length, which is the property that was missing. End to end, in
the harness:

| shape | before | after | |
|---|---:|---:|---:|
| ISL 128 / OSL 256 | 1,445.5 | 1,509.7 | 1.04× |
| ISL 128 / OSL 1024 | 959.7 | 1,106.4 | **1.15×** |

Read those two rows together. Removing a cost worth ~7% of the loop shows up as
1.04×, which is *inside* the ±4% noise band and not a result on its own;
removing one worth ~17% shows up as 1.15×, which is. The fix is the same code in
both rows — what changes is how much there was to remove, and that is a property
of the workload. A shorter benchmark would have found nothing here.

Sampling was the next item that showed up here, at 10.7% of loop time: a Python
loop taking one `argmax` per row, and then one `.item()` per row to read the
tokens back. Taking the greedy rows in a single kernel and the batch in a single
`.tolist()` drops it to **3.1%**, worth 1.07x end to end.

Counting those launches found the next item, which was not on the roadmap at
all. Two per-step tensors — the cache's block tables and the decode mask — were
built by looping over sequences in Python, one `torch.tensor(..., device=cuda)`
per sequence. Those are pageable copies, and pageable copies block: **35 of them
per decode step**, a count that tracked the batch size, which is the signature
of a per-sequence loop. Padding on the host and moving the batch in one transfer
leaves 5.

Inside the forward pass, one decode step at B=32 now profiles as 13.96 ms wall
against a 3.56 ms weight-read floor: 11.02 ms of GPU work spread over **822
kernel launches**, and 2.93 ms — a fifth of the step — with the GPU idle waiting
for Python to issue the next one. That idle fifth is what §3.2's CUDA graphs
recover. It was a quarter before the transfers were fixed, and unchanged by the
sampling fix, because sampling sits outside the forward pass.

### A kernel benchmark is not an engine benchmark

Worth recording because it cost a day and nearly shipped a regression. Attention
expands 8 KV heads to 32 with a real copy — 0.74 GiB of writes per decode step,
~15% of GPU time. `scaled_dot_product_attention(..., enable_gqa=True)` asks the
kernel to broadcast instead, and only cuDNN will do that under an additive mask.
Benchmarked on one attention call it looked decisive:

| attention call, B=32 | `_repeat_kv` + memory-efficient | cuDNN + `enable_gqa` |
|---|---:|---:|
| fixed shape | 0.223 ms | **0.047 ms** |
| shape grows by 1 each call | 0.293 ms | **36.177 ms** |
| padded to 64-token buckets | 0.345 ms | 0.352 ms |

The first row is the one a microbenchmark reports. The second is what decode
actually does: the KV length grows by one every step, and cuDNN builds an
execution plan per shape. In the engine it measured **0.64x** at ISL 128 /
OSL 256 — reverted.

This is the fourth change in a row where the component measurement and the
engine measurement disagreed, and the first where they disagreed about the
*sign*. The other three read 30-60% high. The rule the benchmark exists to
enforce: **measure the component to find the target, measure the engine to size
it** — and never ship on the first number alone.

§2.3 is the counter-example that sharpens the rule rather than breaking it. Its
component number predicted 4.09 ms of saving and the engine gave 3.86 ms, 94%.
What it removes is a *copy*, which was serialised with the rest of the pass and
therefore cost its full measured time; the four items that read high all removed
overhead that partly overlapped with real work. So the discount is not a property
of component benchmarks in general — it is a property of measuring something that
was never entirely on the critical path.

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
resolution floor. Two rows above sit inside it and are reported as no effect: the
paged kernel's latency at B=1, and the SDPA kernel's 1.06× over the eager one at
ISL 128 / OSL 256 once both are measured on the same engine.

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
| Decode is single-request-slow at narrow batches | paged attention is 7.73x per layer at B=32 but 2.27x at B=1 | One program per (sequence, KV head) leaves an 84-SM GPU idle at B=1 | [§2.7](roadmap.md#27-split-the-key-loop-when-the-batch-is-narrow) |
| Prefill still gathers, pads and expands | not isolated; `_repeat_kv` and the prefill mask are unchanged | §2.3 addressed decode only | [§3.5](roadmap.md#35-broadcast-the-grouped-query-heads-instead-of-expanding-them), [§3.6](roadmap.md#36-pack-the-batch-instead-of-padding-it) |
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
