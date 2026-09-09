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
resolves its attention kernel from the device and picks the paged one wherever it
can run. `liteinfer-sdpa` is what that choice falls back to — off CUDA, or
without a Triton install — so both rows describe shipping configurations rather
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
| `liteinfer-continuous` | 32 | 1,626.8 | 6.4 | 31.5 | 6.43×† | 0.36× |
| `liteinfer-sdpa` | 32 | 1,748.1 | 6.8 | 29.3 | 1.07× | 0.39× |
| `liteinfer-paged-attn` | 32 | 1,931.5 | 7.5 | 26.5 | **1.10×** | 0.43× |
| `liteinfer-graphs` | 32 | **3,152.4** | 12.3 | 16.2 | **1.63×** | 0.71× |
| `liteinfer-graphs-b128` | 128 | **6,945.3** | 27.1 | 7.4 | — | 0.63× |
| `vllm` | 1 | 188.4 | 0.7 | 271.8 | — | — |
| `vllm-b4` | 4 | 724.0 | 2.8 | 70.7 | — | — |
| `vllm-continuous` | 32 | 4,466.6 | 17.4 | 11.5 | — | — |
| `vllm-b128` | 128 | 11,097.5 | 43.3 | 4.6 | — | — |

† against a removed config, so it measures two engines two milestones apart. The
two rows above it are same-session and measure only their own code.

### Latency — one request in flight

Run sequentially on an idle GPU. See *Certification* below for why this mode must
not be parallelised.

**`bench report` prints a prompt-mismatch warning on this group, and it is
expected.** `benchmarks/datasets/` is gitignored, so a regenerated dataset gets a
new sha256 while older results keep the old one. The seven `historical` rows here
were measured on a dataset that no longer exists and cannot be re-run — `bench
run` refuses them by design — so the group will always straddle two shas. What
matters is that every *runnable* row, liteinfer and vLLM alike, is on the current
one; that was not true until the vLLM rows were re-measured, which had quietly
made the headline decode-step comparison cross-dataset. §8.4 is about making the
report say which ratios a mismatch actually affects instead of flagging the whole
group.

| config | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | E2E p50 | vs base |
|---|---:|---:|---:|---:|---:|---:|
| `liteinfer-nocache` | 14.8 ms | 16.1 ms | 16.5 ms | 16.5 ms | 4,215.6 ms | — |
| `liteinfer-eager` | 15.7 ms | 16.3 ms | 13.7 ms | 13.9 ms | 3,521.2 ms | **1.20×** |
| `liteinfer-native-eager` | 14.8 ms | 16.1 ms | 13.7 ms | 13.9 ms | 3,518.5 ms | 1.00× |
| `liteinfer-paged` | 19.3 ms | 21.1 ms | 15.3 ms | 15.3 ms | 3,910.7 ms | 0.90× |
| `liteinfer-eager-b4` | 15.8 ms | 17.7 ms | 14.0 ms | 14.1 ms | 3,596.6 ms | 0.98× |
| `liteinfer-native-eager-b4` | 14.9 ms | 16.7 ms | 13.9 ms | 13.9 ms | 3,561.1 ms | 0.99× |
| `liteinfer-paged-b4` | 19.0 ms | 20.9 ms | 15.0 ms | 15.0 ms | 3,832.2 ms | 1.02× |
| `liteinfer-continuous` | 16.1 ms | 17.1 ms | 15.4 ms | 15.5 ms | 3,947.1 ms | 0.97× |
| `liteinfer-sdpa` | 13.9 ms | 15.5 ms | 13.9 ms | 14.0 ms | 3,561.0 ms | **1.11×** |
| `liteinfer-paged-attn` | **13.9 ms** | 15.3 ms | 13.3 ms | 13.4 ms | 3,407.6 ms | 1.05× |
| `liteinfer-graphs` | 14.0 ms | 15.8 ms | **6.6 ms** | **6.6 ms** | **1,698.5 ms** | **2.01×** |
| `vllm` | 27.5 ms | 32.1 ms | 5.2 ms | 5.5 ms | 1,354.9 ms | — |
| `vllm-b4` | 28.7 ms | 31.2 ms | 5.2 ms | 5.2 ms | 1,352.8 ms | — |
| `vllm-continuous` | 28.6 ms | 31.4 ms | 5.2 ms | 5.2 ms | 1,352.7 ms | — |

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
continuous ITL (15.4 ms) sits level with static paged (15.0 ms).

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

**TTFT: liteinfer is ahead, but read it narrowly.** 14.0 ms against vLLM's
28.6 ms — and vLLM's figure moved 22.7 → 28.6 ms purely by being re-measured on
the current prompt set, which is how the cross-dataset split described above was
found. ITL did not move at all (5.2 ms either way), because these runs pin the
output length, so only TTFT sees the prompts change. At ISL=128 both engines' TTFT is dominated by fixed per-call API overhead
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
paged decode the 1024 / 1024 row goes 696.3 → 1,817.4 tok/s and the gap to vLLM
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
| 128 / 256 | 1,626.8 | 1,748.1 | 1,931.5 | 1.10× |
| 128 / 1024 | 1,197.3 | 1,232.2 | 1,930.6 | 1.57× |
| 1024 / 256 | 713.4 | 793.1 | 1,569.6 | 1.98× |
| 1024 / 1024 | 670.1 | 696.3 | 1,817.4 | **2.61×** |
| 2048 / 128 | OOM | 418.8 | 892.6 | 2.13× |

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
request at a time: ITL p50 13.9 → 13.3 ms, E2E 3,561.0 → 3,407.6 ms, TTFT 13.9 →
13.9 ms. All three are inside the ±4% floor, so the honest reading is *no effect*.
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
went on to recover — see *The step was 40-50% launch overhead* below, where the
same measurement at 725 launches priced it and the capture collected it. It was a
quarter before the transfers were fixed, and unchanged by the sampling fix,
because sampling sits outside the forward pass.

### One ULP decides a token, and that is why parity is claimed in fp32

The 8B parity test failed when §2.3 landed, and what it caught is worth keeping.
Its fixture compared liteinfer against `transformers` in bf16 without pinning the
attention kernel, so it silently started measuring the *new* kernel against an
eager reference. One prompt of three diverged at token 11. Both kernels rank the
same three candidates there:

| kernel | top-3 logits | top-2 gap | picks |
|---|---|---:|---|
| `sdpa` | 20.25, 20.125, 19.875 | 0.125 | 3169 |
| `paged` | 20.125, 20.125, 19.875 | **0.000** | 279 |

bf16 carries 8 mantissa bits, so between 16 and 32 the representable values are
0.125 apart: `sdpa`'s "gap" is **one ULP**, the smallest disagreement the format
can express, and `paged` rounds both candidates onto the same value. The
tie-break then takes the lower token id, and twenty tokens of text change.

Nothing here is wrong. The kernels are token-identical in fp32, agree to 1e-5
there and to 2⁻⁶ in bf16 on real rows, and rank these candidates identically —
what differs is the order in which the key axis is summed, since `paged` folds
64 keys at a time through an online softmax where the dense kernels take the
whole axis in one GEMM. What the failure actually shows is that **greedy
decoding is a discontinuous readout of a continuous quantity**: at a tie it
amplifies the last bit of the last accumulation into a completely different
continuation.

Two rules come out of it, and both were already written down — this is the run
that made them concrete. A parity claim between kernels is made in **fp32**,
where there is no tie to break. And a test that holds the *model* constant must
pin the *kernel* on both sides, or the engine's own choice of kernel will
eventually turn it into a precision test nobody meant to write.

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

### A 7x faster kernel that made the engine slower (§2.7)

§2.3 left the paged kernel running one program per (sequence, KV head) — 8
programs at B=1, on 84 SMs. §2.7 cut each sequence's keys into slices, gave each
its own program, and combined the partial softmaxes weighted by their
log-sum-exp. It was built, measured, and reverted. Both halves of that are worth
keeping.

**On the GPU it did exactly what it was supposed to.** Per layer at Llama-3.2-1B's
decode shapes (32 query heads, 8 KV heads, head_dim 64, bf16), unsplit against a
`num_splits` chooser fitted to a sweep over batch width and context. Timed through
a captured CUDA graph, five repeats of 500 replays, best repeat kept:

| context | B=1 | B=2 | B=4 | B=8 | B=12 | B=32 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| 256 | **1.42x** | 1.35x | 1.14x | 0.94x | 1.00x | 1.00x |
| 384 | **1.81x** | 1.58x | 1.24x | 1.04x | 1.00x | 1.00x |
| 512 | **2.24x** | 1.92x | 1.52x | 1.13x | 1.00x | 1.00x |
| 1024 | **3.45x** | 2.77x | 1.95x | 1.12x | 1.00x | 1.00x |
| 2048 | **5.18x** | 3.66x | 1.87x | 1.05x | 1.00x | 1.00x |
| 4096 | **7.15x** | 3.44x | 1.76x | 1.06x | 1.00x | 1.00x |

The 1.00x entries are the unsplit pass, chosen: from B=12 up the grid already
holds two programs per SM, and below four key tiles the context holds less
sequential work than the combine pass costs to add. And the saving is not an
artefact of the microbenchmark — profiled inside the engine, GPU kernel time per
decode step falls **7.35 → 6.31 ms** at B=1 and 3,645 tokens, which is the 1.04 ms
the per-layer numbers predict.

**In the engine it lost, at every shape measured.** The same profile counts
**695 → 711 kernel launches** per step — one extra per layer:

| latency, B=1 | ISL 128 / OSL 256 | ISL 3584 / OSL 128 |
|---|---:|---:|
| ITL p50, unsplit | 13.4 ms | 13.8 ms |
| ITL p50, split-K | **14.2 ms** | **14.3 ms** |
| | 0.94x | 0.97x |
| e2e p50 | 3,431.6 → 3,629.8 ms | 1,921.6 → 1,986.7 ms |
| TTFT p50 | 14.5 → 14.5 ms | 164.1 → 162.6 ms |

‡ This shape is the least trustworthy row in the file, and it is worth saying
why. Its ITL spreads about 5% run to run — the unsplit config read 16.1, 16.2,
16.3, 16.9, 17.0 and 17.1 ms across six runs on two GPUs, and the captured one
7.9 to 8.4 — which is wider than the ±4% floor claimed above. An earlier reading
of **13.8 ms** was published here and is now discredited: neither the current
engine nor `master` reproduces it, and a two-pass A/B on `master` put the same
configuration at **19.8 and 19.9 ms**. So the honest range for this delta is
roughly 2.0-2.4×, and the number in the table is the stored pair rather than a
best case.

That A/B also said something about §3.7 that the throughput rows understated:
`master`'s TTFT at this shape is **240 ms against the branch's ~151**, far more
than the 1.10× §3.7 measured at ISL 2048. Removing a 0.92 GiB logits allocation
evidently saves allocator time as well as arithmetic, so §3.7 is worth more at
long prompts than its throughput delta suggested.

ISL 3584 is the most favourable shape `max_model_len` 4096 allows — the context
where the kernel is 7x faster — and it still loses 3%. `decode()` timed directly
agrees, 12.83/13.51 ms against 13.99/14.24 ms, and keeps that ordering when the
split run goes first, so it is not run-order bias. TTFT does not move because
prefill delegates to `sdpa`.

**Why a 1.04 ms GPU saving loses to 16 extra launches.** Because at B=1 the GPU is
not the constraint. Of a 12.8 ms decode step, **7.35 ms is GPU kernel time and the
rest is host** — 43% idle across ~700 launches — and `paged_decode` costs **74 us of
host time per call against 11.8 us of kernel**, most of it Triton's own dispatch.
Removing GPU work from the idle side of that ledger buys nothing; adding a launch
to the busy side costs immediately.

That is a precondition, not a verdict, and it has a name: **§3.2**. A captured
graph replays the launches with no host work between them, which is the exact
condition under which the GPU column above becomes step time. So the order is
§3.2 first. It also means §3.2's own 1.06x is measured on the wrong batch width:
that number is B=32, where a step has real GPU work in it, and nobody has
measured graphs at B=1 where the step is 43% idle.

**Two measurement traps, both of which nearly shipped the wrong answer.**

*Timing a Triton kernel from Python measures Python.* The first attempt at the
table above launched the kernel in a loop and reported ~70 us per call at *every*
batch width and context, with split-K uniformly 0.43x-0.83x — the arithmetic
inverted, because the CPU cannot enqueue faster than that and the GPU idles
waiting. Under a captured graph the same shapes are 4.7-490 us and ordered as
shown. The irony is that the bogus measurement was reporting the real problem:
74 us of host time per call is why this item is blocked.

*At single-digit microseconds a single mean is noise.* The second attempt used one
mean per shape and read a 1.42x win as **0.88x**, and a policy fitted to that would
have disabled splitting exactly where it helps most. Five repeats and a minimum
fixed it. Anything measured near the kernel's ~4.7 us floor needs repeats before
it is allowed to decide anything.

**And the cheapest check was the one not done first.** Sixteen layers times a
per-layer time, over a step time already stored in `benchmarks/results/`, sizes
this item in one line of arithmetic — and the profile that says 43% of the step is
idle takes three minutes. Both were run after the kernel was written. The rule
from the section above still holds, with a corollary: **measure the component to
find the target, measure the engine to size it — and compute the ceiling before
writing the kernel.**

### The step was 40-50% launch overhead, at every batch width (§3.2)

Asked where the next general win is, the engine's own instrumentation answers
first. `TimeBreakdown` has recorded the loop's stages since §6.4 and nothing had
read it. At B=32, ISL 128 / OSL 256, 64 prompts:

| stage | share of loop |
|---|---:|
| forward | **93.0%** |
| sample | 3.6% |
| unattributed | 1.8% |
| deliver | 1.3% |
| schedule | 0.4% |

So the five PRs that went after sampling, detokenisation and per-step transfers
have finished that job: everything outside the forward is 7% of the loop. Whatever
comes next has to come out of the forward.

**Inside the forward, the GPU is idle about half the time — and that does not
change with batch width.** Profiling a decode forward at three widths, same
context:

| | B=1 | B=8 | B=32 |
|---|---:|---:|---:|
| forward wall / step | 12.80 ms | 13.60 ms | 13.55 ms |
| GPU busy / step | 6.19 ms | 6.88 ms | 7.97 ms |
| GPU idle / step | **6.61 ms (52%)** | **6.72 ms (49%)** | **5.58 ms (41%)** |
| kernel launches / step | 693 | 725 | 725 |
| wall per launch | 18.5 µs | 18.8 µs | 18.7 µs |

Read the last row first. The step costs about 18.7 µs per launch whatever the
kernels are doing, and 32x the tokens buys only 1.29x the GPU time — which is
also why throughput scales with batch while the step does not move. This is a
launch-bound loop.

**Capturing the forward and replaying it prices the fix exactly.** Same forward,
same inputs, interleaved and repeated in one process:

| | B=1 | B=8 | B=32 |
|---|---:|---:|---:|
| `decode()` step, eager | 12.39 ms | 12.96 ms | 13.20 ms |
| its forward, eager | 11.18 ms | 11.67 ms | 11.67 ms |
| its forward, replayed | 5.84 ms | 6.41 ms | 7.28 ms |
| **step if launches were free** | **1.76x** | **1.68x** | **1.50x** |

Two independent methods agreeing — the profiler's idle share and the replay
delta — is the standard this file asks for, and they do. Replay also comes in
just under profiled GPU-busy time, as it should, since profiling inflates kernel
durations slightly.

And replay is not just fast, it is correct against changing inputs: refilling
static buffers from live engine state over six steps, with per-sequence contexts
of 207-215 and a 512-slot KV bucket 2.4x wider than the real context, matches
eager **bit for bit** — max logit difference 0, same argmax every step. The
padding is invisible because `context_lens` bounds the paged kernel's loop, which
also retires the fp32-only parity caveat §3.2 recorded for the dense path.

**Where that leaves the engine against the roofline.** Llama-3.2-1B in bf16 is
~2.47 GB of weights to read per decode step; an A40 at 696 GB/s puts the floor at
**3.55 ms**:

| | step | vs roofline |
|---|---:|---:|
| roofline | 3.55 ms | 1.00x |
| vLLM ITL | 5.20 ms | 1.47x |
| liteinfer, GPU busy only (B=32) | 7.97 ms | 2.25x |
| liteinfer, replayed forward (B=32) | 7.28 ms | 2.05x |
| liteinfer, actual step (B=32) | 13.20 ms | 3.72x |

liteinfer's *kernels* are already within ~1.4x of vLLM's entire step. The 2.58x
ITL gap is host dispatch, and it is the one line in this table that a single
change removes. What survives it is the 2.05x — 725 launches is ~45 kernels per
layer where a Llama layer needs about ten — and that is §3.1, which should be
sized after graphs land rather than against a step where a launch costs 18.7 µs.

**Two measurement traps caught on the way, both the same trap as §2.7's.** Timed
sequentially, the second of two configurations came out ~25% slower whichever one
it was: clock drift over a run, wide enough here to invert the comparison, so
everything above is interleaved with per-configuration minima. And the bare
forward first measured *slower than the whole step containing it* — because
`decode()` is decorated `@torch.inference_mode()` and a hand-written call is not,
so it was paying autograd bookkeeping the engine never pays. That one read the
launch overhead 15 points high before it was found.

#### What it actually paid

Built as `liteinfer-graphs` and measured against `liteinfer-paged-attn`, whose
stored rows are the same engine with capture pinned off:

| throughput | paged-attn | graphs | |
|---|---:|---:|---:|
| 128 / 256 | 1,931.5 | **3,152.4** | **1.63×** |
| 128 / 1024 | 1,930.6 | **2,972.1** | **1.54×** |
| 1024 / 1024 | 1,817.4 | **2,389.2** | 1.31× |
| 1024 / 256 | 1,569.6 | **2,071.6** | 1.32× |
| 2048 / 128 | 892.6 | **979.5** | 1.10× |

| latency, B=1 | paged-attn | graphs | |
|---|---:|---:|---:|
| ITL p50, ISL 128 / OSL 256 | 13.3 ms | **6.6 ms** | **2.01×** |
| e2e p50 | 3,407.6 ms | **1,698.5 ms** | 2.01× |
| ITL p50, ISL 3584 / OSL 128 | 16.9 ms | **7.9 ms** | 2.14×‡ |
| TTFT p50 | 13.9 ms | 14.0 ms | — |

The gap to vLLM closes from **0.43× to 0.71×** on throughput and from **2.56× to
1.27×** on the decode step. It is the largest single change the project has
measured, and it is general: every batch width, every context, both modes.

**What it cost.** TTFT does not move, because prefill is not captured — its
shapes follow the prompt, and there is one prefill per request against hundreds
of decodes. ISL 2048 / OSL 128 is the weakest row at 1.11×, barely past the
variance floor, for the same reason: it is the shape with the most prefill and
the fewest decode steps to spread a fixed saving over. The first step at each
batch width pays three warm-up forwards and a capture. And a very wide engine
would accumulate graphs, so `_MAX_CAPTURES` bounds them at 64 — untouched at the
default `max_num_seqs` of 32, where a draining batch captures all 32 widths and
**reserved memory went down** 2.1 GiB, the eager path's transient allocations
having been larger than the graph pool that replaced them.

**The component estimate read *low* this time, which is new.** The ceiling above
predicted 1.76× at B=1 and the engine gave 2.03×. Every previous disagreement in
this file went the other way, by 30-60%. The reason is the shape: the ceiling was
measured at 765 tokens of context and the benchmark's is 128-384, so there is
less GPU work under the same fixed overhead and a larger share of the step to
remove. An overhead-removal estimate scales with how little work the step
contains — which is the mirror image of §2.3's rule about removing a copy.

**Where the engine now sits against the roofline.** Reading ~2.47 GB of bf16
weights per step on an A40 at 696 GB/s is a 3.55 ms floor:

| | step | vs roofline |
|---|---:|---:|
| roofline | 3.55 ms | 1.00× |
| vLLM ITL | 5.20 ms | 1.47× |
| liteinfer ITL, after §3.2 | 6.60 ms | 1.86× |
| liteinfer ITL, before | 13.30 ms | 3.75× |

What is left is arithmetic, not overhead: ~725 launches is about 45 kernels per
layer where a Llama layer needs ten, and the elementwise work among them costs
far more in launches than in GPU time. That is §3.1, and it can now be sized
against a step where a launch is nearly free rather than one where it cost
18.7 µs.

**And it unblocks §2.7 with a caveat worth writing down.** Split-K decode was
reverted because it added one launch per layer to a step that paid per launch;
inside a capture that cost is gone, and its GPU win was up to 7.15× per layer.
But `num_splits` is a *scalar* kernel argument, and a graph bakes those in while
the split count this engine chooses varies with context — so §2.7 after §3.2 has
to pin one count per capture rather than choose per step.

### What §3.2 left behind, and it is not fusion (§3.1)

§3.1 was next in line — with a launch nearly free, the step is arithmetic — so it
was profiled before it was built. Inside a captured forward at B=1, 455 tokens of
context:

| op | ms / step | share | calls | per layer |
|---|---:|---:|---:|---:|
| `mm` | 4.730 | **79.8%** | 113 | 7.06 |
| `mul` | 0.316 | 5.3% | 149 | 9.31 |
| `copy_` | 0.158 | 2.7% | 75 | 4.69 |
| `add` | 0.150 | 2.5% | 98 | 6.12 |
| `_index_put_impl_` | 0.126 | 2.1% | 32 | 2.00 |
| `cat` | 0.109 | 1.8% | 33 | 2.06 |
| `mean` | 0.107 | 1.8% | 33 | 2.06 |
| `neg` | 0.072 | 1.2% | 32 | 2.00 |
| `rsqrt` | 0.049 | 0.8% | 33 | 2.06 |
| `pow` | 0.047 | 0.8% | 33 | 2.06 |
| `silu` | 0.038 | 0.6% | 16 | 1.00 |

The matmuls are 79.8% of it over exactly **113 launches — 7 per layer plus the
head**, which is already the fewest a Llama layer can be written with. Everything
fusible is the other 1.20 ms of 5.93, and fusing RMSNorm, RoPE and SiLU·mul while
merging the QKV projection comes to about 0.72 ms: a 6.6 ms step becoming 5.9,
**1.12x**, for three custom kernels with parity tests and a change to weight
loading. Making every non-`mm` kernel free would be 1.22x. That is the ceiling,
and by this file's own standard — an effect under ~1.1x is not an effect — 1.12x
is barely one.

Two details worth keeping from the attempt that did not happen. **Only one of the
two obvious GEMM merges pays:** measured on the projections alone through a
captured graph, merging q/k/v is 1.16x at B=1 and **1.38x at B=32** (411 → 477
GB/s, three skinny GEMMs each paying their own tail), while merging gate/up is
0.97-1.02x because at 2048×8192 each already saturates. "Fuse the projections"
would have done both. And **the elementwise cost is kernel count, not bytes**: 475
launches at ~1.8 µs each, touching 2,048 elements apiece at B=1, which is far too
little to be bandwidth-bound. That is a per-kernel floor, which is why the saving
hardly moves between B=1 and B=32.

### The flat step is throughput nobody is collecting (§1.4)

The same measurement that priced §3.2 said something else in passing: the step is
nearly flat in batch width. Flat means the weights are read once per step no
matter how many sequences share it, so every extra sequence is nearly free.
Measured on the captured step at 256 tokens of context:

| `max_num_seqs` | step | decode tok/s | |
|---:|---:|---:|---:|
| 32 | 7.63 ms | 4,196 | — |
| 64 | 8.93 ms | 7,165 | 1.71× |
| 128 | 11.25 ms | 11,381 | **2.71×** |
| 256 | 17.30 ms | 14,797 | 3.53× |

Eight times the batch for 2.3× the step. Through the harness at ISL 128 / OSL 256,
`max_num_seqs=128` measures **6,945.3 tok/s against 3,152.4 — 2.20×** — with wall
time 16.2 → 7.4 s.

#### What it paid, and the deficit it exposed

Measured through the harness at ISL 128 / OSL 256, against vLLM at the **same**
width — which is the only comparison worth making, and the reason the wide row
carries no `baseline`:

| | B=32 | B=128 | |
|---|---:|---:|---:|
| liteinfer | 3,152.4 | **6,945.3** | 2.20× |
| vLLM | 4,466.6 | **11,097.5** | 2.48× |
| gap | 0.71× | **0.63×** | |

So the win is real — 2.20×, wall 16.2 → 7.4 s — and it does **not** close the gap;
it widens it, because vLLM gains more from the same width. That is the finding
worth having, and it is invisible if you only compare a wide engine to your own
narrower self.

Where it goes is measurable, and it is not the forward. The loop's stages by
width:

| stage | B=32 | B=128 |
|---|---:|---:|
| forward | 88.7% | **81.5%** |
| sample | 6.0% | **11.3%** |
| deliver | 2.0% | 4.2% |
| schedule + unattributed | 3.4% | 3.0% |

The forward is nearly flat in batch width — that is what §1.4 spends — but the
per-sequence work around it is not, and at 128 sequences it is **18.5% of the
loop** against 11.3% at 32. Removing it entirely would put throughput near 8,550
tok/s and the gap at ~0.77×. Filed as §1.5.

It is also not merely a bigger default: `max_num_seqs=128` is reachable by any
caller today, and what was missing is the engine being *safe* at it. The pool is sized to `max_num_seqs` ×
`max_model_len` or to a fraction of free memory, whichever is smaller — 17.2 GiB
at 128 × 4096, which fits an A40 and does not fit a 24 GiB card, where the pool
goes quietly under what the config promises. And `DecodeGraphs` records one graph
per exact batch width with a cap of 64, which never binds at 32 and binds
immediately at 128. Those are §2.6 and a capture ladder, filed as §1.4's
prerequisites.

The number above is deliberately **not** in the results table. Comparing a
128-wide row to a 32-wide one measures the batch width, which is the one
comparison this file refuses to make against vLLM and should not make against
itself either. It belongs in the table when there is a `vllm-b128` beside it.

### The pool is sized from the device now, not from the machine's history (§2.6)

The block pool took a fraction of *free* VRAM, which made its size depend on
what else happened to be resident. The sharper version of that bug is that it
depends on what was resident a moment **earlier**: torch's caching allocator
keeps freed blocks, so CUDA's free figure stays low after an allocation is
released.

| blocks a memory-bound pool would take | from free VRAM | from total VRAM |
|---|---:|---:|
| weights only | 72,516 | 72,208 |
| after 8 GiB allocated **and freed** | 58,589 | 72,208 |
| | **−19%** | unchanged |

`gpu_memory_utilization` now applies to total memory less the weights already
resident — vLLM's meaning for the same name — so the size is a function of the
config and the device. The benchmark config is unaffected, because
`max_num_seqs × max_model_len` binds it there: 8,192 blocks / 4.00 GiB before and
after, so no stored result moves.

It is a behaviour change elsewhere, though, and worth stating plainly: loading
weights leaves enough allocator residue that CUDA's free figure reads well below
`total − weights`. At Llama-3-8B's defaults the old formula produced **5,752
blocks (11.24 GiB) where the config asks for 8,192 (16.00 GiB)** — the engine was
quietly taking *less* than it was told whenever loading was untidy. It now takes
what it was told, which is the point, and which means a process holding something
else large has to say so. The 8B parity test co-loads a second copy of the same
model as its reference and had been fitting on that accident; it now asks for the
one short sequence it decodes.

**The second half replaces the fraction with a measurement, and §3.7 is what made
it possible.** The pool now subtracts a profiled figure: one prefill at
`max_num_seqs × max_model_len`, which is not hypothetical because `schedule()`
admits and prefills that many at once. On Llama-3.2-1B at 32 × 4,096 that is
**9.04 GiB** — and **32.82 GiB** with the LM head left unsliced, which a 44 GiB
card has nowhere to put beside weights and a pool. The roadmap had this step
blocked on "bounding the worst case" without knowing what was making it large.

The profile needs no pool, which is the trick that lets it run before the thing it
is sizing exists: `ProfilePayload` returns the K/V the pass just computed, exactly
as the prefill payload does after storing it, so the forward has the same shapes
and the same peak while writing nowhere.

**A test caught the first forward measuring itself.** Unwarmed, a 2-sequence
config profiled **8.49 MiB where a 32-sequence one profiled 5.77** — the first
forward in a process allocates cuBLAS workspaces that every later one reuses, and
they landed inside the measurement, so the first engine loaded in any process
would have been handed the smallest pool. One throwaway 16-token pass fixes it,
after which the figures scale exactly 2× per doubling of the batch:

| `max_num_seqs` | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| profiled (MiB) | 0.361 | 0.721 | 1.442 | 2.883 | 5.766 |

### Prefill's largest allocation is logits nobody reads (§3.7)

`LlamaForCausalLM.forward` applies the LM head to every position; `prefill`
returns `logits[:, -1, :]`. Measured at batch 8 with 2,048-token prompts:

| | |
|---|---:|
| prefill peak allocation over baseline | 4.05 GiB |
| the logits tensor inside it | **3.91 GiB — 96% of the peak** |
| rows of it that are read | 8 of 16,384 — **0.05%** |
| LM head flops | 8.61 TFLOP, 0.05% useful |

The prefill's peak memory *is* this tensor, and it scales with
`batch × prompt_len`: 15.6 GiB at 32 × 2,048, 31 GiB at 32 × 4,096, and **62.6 GiB
at 128 × 2,048**, which does not fit an A40 at all.

Three things follow. It is where the surplus §2.1 freed from the pool was
actually going, and a large part of why *Long prompts are liteinfer's weak shape*
above. It blocks §2.6's profile run, since the thing to be profiled is dominated
by a tensor that should not exist. And it blocks §1.4 outright: raising
concurrency to 128 would not be slow at long prompts, it would fail to run.

#### What it paid, and what it did not

`forward` now takes a `logits_positions` slice — `None` still computes every
position, which is what the `transformers` parity tests compare — and every caller
in the engine passes `LAST_POSITION`. The logits come back bit-identical and the
prefill peak goes **3.98 → 1.01 GiB**.

Both columns below are `liteinfer-graphs` — the same config either side of the
change, which is the only way an ungated improvement can be measured. Neither
column survives in `benchmarks/results/`: the *before* was overwritten by the
*after*, and the *after* was itself superseded when every config was re-measured
in one session (see below). The results table above is the current engine; this
is the delta.

| | before | after | |
|---|---:|---:|---:|
| TTFT p50, ISL 3584 | 162.5 ms | **147.2 ms** | **1.10×** |
| throughput, 2048 / 128 | 908.2 | **979.0** | 1.08× |
| throughput, 1024 / 256 | 1,990.8 | 2,069.3 | 1.04× |
| throughput, 1024 / 1024 | 2,356.2 | 2,376.1 | 1.01× |
| throughput, 128 / 256 | 3,111.7 | 3,115.6 | 1.00× |
| ITL p50 | 6.6 ms | 6.6 ms | — |

So it is a capability fix with a small bonus, not a throughput win: the head is
about a fifth of prefill's arithmetic, and prefill is only part of a run. The
prediction written here before it was built — "a throughput win at prefill-heavy
shapes" — was optimistic by roughly half, which is worth leaving on the record
next to the memory figure that was not.

#### And it corrupted its own neighbours, which is a property of the harness

Because it is ungated, §3.7 landed in *every* config at once — while the stored
rows it would be compared against had been measured before it. `vs base` ratios
two stored rows with no notion of when either was taken, so at ISL 2048 / OSL 128
the report printed **1.19×** for `liteinfer-graphs` over `liteinfer-paged-attn`
where CUDA graphs alone are worth 1.10×; the other 8% was §3.7, present in the
newer row only. Re-measuring every runnable config in one session fixed it —
`liteinfer-paged-attn` at that shape went 820.1 → 892.6, which is §3.7 arriving
in the baseline, and the printed delta fell to the 1.10× it should always have
been.

Worth stating as a rule: **an ungated change invalidates every stored delta whose
baseline predates it.** §8.4 now enforces it — each result records the git
revision it measured, and a ratio whose two rows disagree on revision, prompt
digest or (for older rows) the clock is printed with a `~`. A staleness window
alone would not have caught this one: these baselines were hours old, not days,
and what differed was the code.

### Would capturing prefill help? Only where prefill is small (§3.2 follow-up)

vLLM graphs prefill, so the question is fair. It does it through **piecewise**
capture: the inductor graph is split at the attention ops, attention runs outside
the graph, and every other piece is captured — which works for any shape,
including mixed prefill/decode batches. Full-forward graphs are reserved for
uniform decode batches, which is exactly what `DecodeGraphs` is, and what vLLM
calls `FULL_DECODE_ONLY`.

Whether liteinfer wants the piecewise version depends on how much of a prefill is
waiting rather than working:

| prompt length | prefill wall | GPU busy | GPU idle |
|---:|---:|---:|---:|
| 128 | 12.92 ms | 8.08 ms | **4.84 ms (37%)** |
| 512 | 20.11 ms | 18.83 ms | 1.28 ms (6%) |
| 2048 | 80.60 ms | 79.83 ms | 0.77 ms (1%) |

A liteinfer prefill is the whole prompt in one pass, so it is large and
GPU-bound as soon as the prompt is non-trivial — the idle share collapses by 512
tokens. That is *why* the value looks low here, and it is a property of this
engine rather than of the technique: vLLM's prefills are chunked to
`max_num_batched_tokens`, so most of its forwards are small mixed batches where
the launch overhead dominates and piecewise capture pays. Two consequences worth
recording:

- Capturing prefill on its own would buy about 1.6x on TTFT at ISL 128 and
  nothing past 512, at the cost of one capture per (prompt-length bucket × batch
  width) — and unlike decode's slot-table padding, padding a prompt wastes real
  compute.
- **§1.3 changes the arithmetic and partly undoes §3.2.** Chunked prefill merges
  prefill and decode into one pass, and a mixed pass is not a uniform decode
  batch, so `DecodeGraphs` would not cover it. At OSL 256 with 32 slots that is
  roughly one step in eight, so the loss is modest — but §1.3 should land with a
  plan for capturing mixed batches, which is the point at which piecewise stops
  being low-value here.

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
| 1.3x slower than vLLM per decode step | ITL 6.6 ms vs 5.2 ms; 1.86x the memory roofline vs vLLM's 1.47x | Unfused elementwise work — ~45 kernels per layer where ten would do | [§3.1](roadmap.md#31-fuse-the-forwards-elementwise-work) |
| Decode is single-request-slow at narrow batches | paged attention is 7.73x per layer at B=32 but 2.27x at B=1 | One program per (sequence, KV head) leaves an 84-SM GPU idle at B=1 | [§2.7](roadmap.md#27-split-the-key-loop-when-the-batch-is-narrow) |
| Prefill still gathers, pads and expands | not isolated; `_repeat_kv` and the prefill mask are unchanged | §2.3 addressed decode only | [§3.5](roadmap.md#35-broadcast-the-grouped-query-heads-instead-of-expanding-them), [§3.6](roadmap.md#36-pack-the-batch-instead-of-padding-it) |
| Continuous batching scales slightly below vLLM | 5.01x for 8x width vs vLLM's 6.17x | Two-pass step when prefill and decode coexist | [§1.3](roadmap.md#13-chunked-prefill--single-pass-mixed-batching) |
| KV-cache benefit unquantified across shapes | 1.21x at ISL 128 / OSL 256 only | Single measured shape, and not re-measurable: the no-cache and DynamicCache configs are `historical`, so the number is frozen at the engine of the day they were deleted | — |
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
