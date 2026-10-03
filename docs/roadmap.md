# Roadmap

Fine-grained backlog of possible future work. Items are grouped by
area, not priority.

## Item template

Each item starts with a status badge and an optional PR list, followed
by the substantive bullets:

```
### N.N <Title>
- **Status.** `planned` | `in-progress` | `landed`
- **PRs.** _none yet_ | #12, #14
- **Why.** <user-facing benefit>
- **Scope.** <files / surfaces touched>
- **Parity test.** <how correctness is pinned>
```

## Shipping an improvement

Every change that claims to make liteinfer faster follows the same loop: the
claim is backed by a number measured the same way before and after, and a
general improvement that wins outright replaces the path it beats rather than
sitting beside it. A specialised improvement — one that only applies under some
precondition — joins instead. Step 6 is where that is decided.

**1. Measure the baseline.** The config being improved must already have stored
results in `benchmarks/results/`. If it does not, run it first — a claim needs a
before.

**2. Add the config, not a branch.** New work is a `BenchmarkConfig` entry in
`benchmarks/configs.py` whose `baseline` names the config it improves on. That
is what makes `bench report` print the 1:1 delta.

**3. Measure the right thing.** The two modes answer different questions, and
using the wrong one hides the effect:

| The change affects | Run | Read |
|---|---|---|
| how much work fits at once (batching, scheduling, memory) | `throughput` | tok/s, req/s |
| how long one step costs (kernels, cache layout, attention) | `latency` | ITL, TTFT |

Same dataset, same ISL/OSL, same sample count as the baseline. Latency runs
sequentially — `--gpus` distorts TTFT. Against vLLM, compare only at **matched
batch width**; a B=1 engine against a B=32 one measures the batch width.
Run-to-run variance is about ±4%, so an effect under ~1.1x is not an effect.

**4. Say what it cost.** Report the metric that got worse as plainly as the one
that got better. Continuous batching won throughput 4.5x and lost ~9% on
single-request ITL; both belong in the write-up.

**5. Update the docs in the same PR.** Results and analysis in
`docs/benchmarks.md`, headline numbers in `README.md`, a milestone entry with
its PR link, and the roadmap item flipped to `landed`.

Check what the headline numbers still refer to while you are there. Every entry
in a sequence of five re-ran `throughput` and none re-ran `latency`, and the
README ended up quoting a config that had been deleted two milestones earlier —
a number does not stop being published when the code behind it is removed.

**6. Does it replace, or does it join?** Two questions, in this order.

*Does it cover the whole domain of the thing it beats?* Continuous batching
serves every workload static batching served — any model, any batch size — so
static batching had no remaining reason to exist. An MoE kernel, a quantized
path, a long-context attention variant: each may win by a wide margin inside its
domain and still replace nothing, because outside that domain it does not apply.
If there is any workload the old path serves and the new one cannot, they both
stay. Two paths chosen by a precondition are not debt; they are the feature.

*If it does cover the domain, is it a clear win there?* Better on the mode the
feature targets, by more than run-to-run variance, with any regression elsewhere
small enough to state and accept.

**Both yes → delete what it beat**, in order:

   - confirm the superseded config's results are stored — that measurement is
     the only record once the code is gone;
   - flag its `BenchmarkConfig` entry `historical` so the report keeps rendering
     the progression while `bench run` refuses it;
   - delete the code, and everything that existed only to serve it;
   - **simplify what is left.** Removing one of two paths usually makes an
     abstraction pointless: a mode flag with one value, a dispatcher with one
     entry, a base class with one subclass. Collapse them in the same PR, or
     the codebase keeps the shape of a choice it no longer offers.

**Conditional win → keep both**, and say so explicitly: the config stays
runnable and its `description` names the precondition, so the report shows what
the row applies to. Measure it *inside its domain*. Comparing an MoE kernel against a dense baseline measures
the model, not the kernel — so a specialised path usually needs its own dataset
or model in the matrix, not just its own config.

Keeping a slower *general* path "just in case" is how the codebase stops being
readable, and the benchmark exists so that deleting it is safe. Keeping a
*specialised* path is not the same thing: it is the only thing serving its case.

## Marking an item done

When an item lands:

1. Flip `Status` to `landed` and append the merging PR(s) to `PRs`.
2. Move the item out of this file into `milestones.md`, under the
   current month's heading. Keep the `PRs` line so the milestone log
   has the same audit trail — **always link the PR, even before it is
   merged**. A milestone written in the same PR that delivers it knows
   its own number; `_none yet_` there is a link nobody goes back to add.
3. If a follow-up is created (e.g. v1 lands but v2 is the polished
   version), leave a stub here with status `planned` and a backlink
   to the original PR.

Number new items by area: section 2 is KV cache, 5 is engine ergonomics, 6 is
observability. Check `milestones.md` before picking a number — landed items keep
theirs.

Status badges keep half-done work visible: an item can sit at
`in-progress` with one PR linked while the remaining scope stays
listed.

---

## 1. Batching and scheduling

### 1.6 What is left of the loop outside the forward
- **Status.** `planned` — follow-up to §1.5, and small.
- **PRs.** _none yet_
- **Why.** §1.5 took the non-forward share of the loop from 29.9% to **12.6%** at
  128 concurrent sequences. What remains is almost all `sample`, at 7.9%, and
  about half of that is the incremental detokeniser: two `decode` calls per
  sequence per step, of which the Rust work is now 0.055 s against the frame's
  0.167 s.
- **Scope.** The two calls decode overlapping windows — `[prefix_offset,
  read_offset)` and `[prefix_offset, end)` — and the first re-decodes a window
  the previous step already covered, so there may be one call's worth of text to
  carry forward instead of recomputing. That is a change to the one piece of the
  engine whose output is user-visible text, so it needs the byte-equality checks
  §1.5 used: random windows, then real generations across emoji, CJK, Arabic and
  accents, compared both incrementally and against a whole-sequence decode.
- **Size it first.** 4% of the loop is the whole prize, so this is worth doing
  only if it is genuinely small. Measure against `TimeBreakdown`'s stage timings
  rather than in isolation.

### 1.8 Admit by what the pool can hold
- **Status.** `planned` — raised in review of §1.3 ([#45](https://github.com/ValeGian/liteinfer/pull/45)).
- **PRs.** _none yet_
- **Why.** The scheduler admits by slots and by the token budget, never by free
  KV blocks, so a step can be scheduled that the pool cannot hold. Since §1.3,
  `advance` refuses such a step whole and the engine fails its least invested
  tier — every sequence the step would have started, even where one fewer would
  have fitted, and without retrying them once blocks free up. Before §1.3 a
  refused prefill pass failed every prompt in it, so this is not a regression,
  but it is a request failed for want of waiting.
- **Scope.** `ContinuousScheduler.schedule()` asks the cache how many blocks a
  grant needs (`_blocks_short_of` already computes it) and stops admitting when
  the free pool would run out, leaving the rest waiting. Running sequences that
  outgrow the pool still need an answer — vLLM preempts and recomputes — and
  that is the larger half; without it, a decoding sequence the pool cannot grow
  is still failed.
- **Parity test.** A pool too small for every waiting prompt at once completes
  all of them, in more steps.

---

## 2. KV cache implementations

### 2.2 Prefix sharing
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Multi-turn chat and few-shot batches share long prefixes;
  recomputing them dominates wall time.
- **Scope.** Subclass of paged cache that hashes prompt prefixes and
  reuses blocks. Scheduler becomes prefix-aware: pick batch members
  whose prefixes are already resident.
- **Pre-req.** §2.1.
- **Parity test.** Identical greedy output to non-shared paged cache
  on a workload designed to share prefixes.

### 2.5 Quantized cache (KV-cache fp8 / int8)
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Memory bound on long contexts.
- **Scope.** Storage-level quant of K and V; dequant on read. Behind
  `cache_quant: str | None` flag.
- **Risk.** Quality regression on long contexts; needs a tolerance
  parity test against fp16/bf16.

### 2.8 Re-fit the split chooser past 4,096 tokens
- **Status.** `planned` — follow-up to §2.7, and small.
- **PRs.** _none yet_
- **Why.** `_TARGET_PROGRAMS_PER_SM` and `_MAX_SPLITS` were fitted to a sweep that
  stopped at 4,096 tokens, because `max_model_len` did. §2.7 then measured the
  kernel at 16k and 32k, where the chooser returns the same 21 splits it returns
  at 4k — the cap is not binding and the target is not re-derived, so whether 21
  is still the right answer over 256 key tiles is unmeasured. Per layer at batch 1
  the unsplit pass is 368.6 us at 16k against 64.6 split, so a further 10% there is
  worth about as much as the whole win at ISL 3584.
- **Scope.** Re-run the `num_splits` sweep at 8k, 16k and 32k, and at the KV-head
  counts a larger model brings (Llama-3-8B is 8 heads, 70B is 8 over more layers).
  The chooser's shape is fine; it is the two constants that were fitted on a
  narrower domain than the kernel now serves.
- **Measure it** per layer through a captured graph, five repeats and a minimum —
  the two traps in `docs/benchmarks.md` under §2.7 both bite at this scale.

### 2.11 Choose the prefill tiles from the chunk length
- **Status.** `planned` — follow-up to §2.10, and small.
- **PRs.** _none yet_
- **Why.** `paged_prefill` attends every chunk in tiles of 16 queries x 32 keys,
  which the §2.10 sweep found fastest for every chunk of 512 tokens or more. A
  short chunk over a long prefix is bound by reading the pool rather than by the
  dot products, and would rather have 8 x 128: 92 against 145 us per layer for
  64 queries over 4,096 keys, 300 against 492 over 15,000.
- **Scope.** Pick the pair from `max_query_len` in `paged_prefill`, the way
  `choose_num_splits` picks decode's split count from the batch width. Each
  pair is one more compiled kernel, so the chooser should return few of them.
- **Parity test.** Already there: the kernel tests sweep both tiles against
  `eager`, so a chooser can only change speed.

### 2.12 Split the one-query rows of a mixed pass
- **Status.** `planned` — follow-up to §1.3 ([#45](https://github.com/ValeGian/liteinfer/pull/45)).
- **PRs.** _none yet_
- **Why.** A mixed pass reads every row through `paged_prefill`, which has no
  split-K, so its decode rows lose the parallelism `paged_decode` buys a narrow
  batch. At a 32-wide batch and contexts up to 2,000 that costs nothing
  (0.88x-1.01x of a separate decode launch per layer); at 16k context it does: **1.08x-1.51x** per
  layer for 1-7 decode rows beside a chunk, about 3.4 ms over 16 layers on the
  worst shape. Still well under the graphed decode pass §1.3 removed, so not a
  regression against two passes — but money left on the table where long
  context meets a narrow batch.
- **Scope.** Either split the key loop for one-query rows inside `paged_prefill`,
  or give those rows a second launch per layer. The second is simpler and cost
  48-100 us of host time per layer when measured, which is why §1.3 did not take
  it at short context; choose by the split count `choose_num_splits` would pick
  for the decode rows, so it only applies where splitting pays.
- **Measure it** per layer through a captured graph, as §1.3 did
  (`docs/benchmarks.md`, §1.3), then in the engine on a long-context packed row.

---

## 3. Performance optimizations

### 3.1 Fuse the forward's elementwise work
- **Status.** `planned` — **ceiling measured at ~1.12x and not built.** It was
  next in line after §3.2 and the profile disagreed; §1.4 is worth 2.19x for less
  work. Re-open it when the batch width has been spent.
- **Blocked on.** nothing. What stops it is its own size.
- **What the profile says, inside a captured forward.** `mm` is **79.8%** of the
  GPU time at B=1 and 78.5% at B=32, over exactly 113 launches — 7 per layer plus
  the head, which is already the minimum a Llama layer needs. Everything fusible
  is the other fifth:

  | op | ms / step (B=1) | share | calls | per layer |
  |---|---:|---:|---:|---:|
  | `mm` | 4.730 | 79.8% | 113 | 7.06 |
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

  Non-`mm` work totals **1.20 ms of 5.93**. Fusing RMSNorm (`pow`+`mean`+`rsqrt`+
  two `mul` into one), RoPE (`neg`+`cat`+two `mul`+`add` into one) and SiLU·mul,
  and merging the QKV projection, adds up to about **0.72 ms** — a 6.6 ms step
  becoming 5.9, so **1.12x**, for three custom kernels with parity tests and a
  change to how weights are loaded. Making *every* non-`mm` kernel free would be
  1.22x, which is the hard ceiling.
- **The one GEMM merge that pays, and the one that does not.** Measured on the
  projections alone through a captured graph: merging q/k/v into one GEMM is
  **1.16x at B=1 and 1.38x at B=32** (411 → 477 GB/s — three skinny GEMMs each
  pay their own tail), while merging gate/up is **0.97-1.02x**, because at
  2048x8192 each one already saturates. So the QKV merge is worth having and the
  MLP merge is not, which is not what "fuse the projections" would have assumed.
- **Why the elementwise work costs what it does.** 475 elementwise launches per
  step at ~1.8 us each, and at B=1 each one touches 2,048 elements — far too
  little to be bandwidth-bound. That is a per-kernel floor, so fusion helps by
  removing kernel *instances*, not by moving fewer bytes. It is also why the
  saving barely changes between B=1 and B=32.
- **`torch.compile` is still the wrong tool for it.** Measured at **0.06x** before
  §2.3 because inductor functionalises the in-place pool write into a copy of the
  whole pool; excluding the cache mutation recovered 1.02x and then recompiled
  once per `layer_idx`. If this is built, build it as three Triton kernels against
  a dense reference — the shape `models/paged_decode.py` already established.
- **PRs.** _none yet_
- **What it measured.** `torch.compile` on the decode forward is 206.66 ms
  against eager's 12.87 ms at a fixed shape — and it is *not* recompiling.
  Inductor functionalises the paged cache's in-place write to a multi-GB pool
  into a copy of the whole pool: four generated kernels at 43-52 ms each,
  182 ms of the 206. Excluding the cache mutation from the compiled region
  recovers it to 12.61 ms, which is **1.02x** — neutral, not a win, and it then
  recompiles once per `layer_idx` because that is a static int attribute.
  So the fusion this item wants is worth nothing until the cache stops being
  compiled with it.
- **What §2.3 changed, and what it did not.** The gather inductor was compiling
  is gone: paged decode reads the pool where it lies. The *write* is not — every
  layer still scatters one K/V column into the pool in place, which is the
  operation inductor functionalises into a whole-pool copy. Re-measure before
  sizing this: the 0.06x figure describes a forward pass that no longer exists,
  but the mechanism that produced it is still in the one that does.
- **Why.** Elementwise work is the second-largest slice of decode GPU time and
  the most fusible: profiled at B=32, `elementwise_kernel` accounts for
  **17.9% + 2.6% across ~143 launches per step** — RoPE, residual adds, norms
  and the mask fills — against 30.1% for the projection GEMMs that actually do
  the model's arithmetic. §3.2 has since collected the launch half of that, which
  is what makes the remaining half the thing worth measuring: re-profile inside a
  captured forward before sizing this.
- **Scope.** Wrap the `ContinuousModelRunner` forward passes in `torch.compile`,
  gated by `EngineConfig.enable_torch_compile`. First call pays
  compile cost; subsequent calls run the compiled graph.
- **Risk.** Dynamic shapes (variable sequence length) defeat compile
  cache. Use `torch._dynamo` shape specialization or pad to bucket
  sizes.
- **Parity test.** Compiled vs eager: identical greedy outputs.

### 3.4 Tensor parallelism (single-node)
- **Status.** `planned`
- **PRs.** _none yet_
- **Scope.** Per-rank `ContinuousModelRunner` plus a process group. Layer
  weights sharded along output dim (column-parallel) or input dim
  (row-parallel) per HF `_tp_plan`. Already declared in vendored
  models.
- **Surface change.** Loader streams shards onto the right rank;
  attention layers all-reduce.

### 3.5 Broadcast the grouped-query heads instead of expanding them
- **Status.** `planned` — **superseded by §2.3 on decode and by §3.6 on prefill.**
- **What is left of it, which is now only the dense kernels.** This item existed
  to stop `_repeat_kv` materialising a 4x copy of K and V, 0.74 GiB of writes per
  decode step. §2.3's kernel reads each KV head once and broadcasts over its
  query group in-register, so the paged decode path never makes that copy; §3.6's
  packed prefill goes through FlashAttention's varlen entry, which reads one KV
  head per query group natively, so the paged prefill path does not either.
  `_repeat_kv` now runs only in `eager` and `sdpa` — the reference kernel and the
  path CPU and Triton-less installs use. Neither is a performance path, so what
  is left of this item is worth close to nothing; close it rather than build it
  unless a dense path becomes load-bearing again.
- **What it measured on its own.** Only cuDNN broadcasts KV heads under an
  additive mask, and cuDNN builds a plan per shape while decode changes shape
  every step: 0.223 ms fixed-shape, **36.177 ms** when the shape grows by one
  each call. In the engine, **0.64x** — reverted. Bucketing the KV length to 64
  cuts 128 distinct shapes to 3 and reaches parity, not a win.

### 3.8 Capture mixed batches by splitting the graph at attention
- **Status.** `planned` — **low value today and not for a general reason**, which
  is the part worth keeping. Re-priced after §1.3: still low.
- **PRs.** _none yet_
- **Why the question comes up.** §3.2 captures the decode forward whole, which
  works because a decode batch is uniform: one query per sequence, and the only
  thing that varies is read from a tensor. That is precisely vLLM's
  `FULL_DECODE_ONLY` mode. vLLM's default is `FULL_AND_PIECEWISE`, where anything
  that is *not* a uniform decode batch — prefill, and mixed prefill/decode — is
  covered by **piecewise** capture instead: the inductor graph is split at the
  attention ops (`splitting_ops = self._attention_ops`), attention runs outside
  the graph, and every other piece is captured. Splitting there is what makes the
  captured pieces shape-agnostic, because attention is the only part whose shape
  follows per-sequence lengths.
- **Why it buys little here, and it is the engine's fault not the technique's.** A
  liteinfer prefill is the whole prompt in one pass, so it is GPU-bound as soon as
  the prompt is non-trivial:

  | prompt length | prefill wall | GPU busy | GPU idle |
  |---:|---:|---:|---:|
  | 128 | 12.92 ms | 8.08 ms | **4.84 ms (37%)** |
  | 512 | 20.11 ms | 18.83 ms | 1.28 ms (6%) |
  | 2048 | 80.60 ms | 79.83 ms | 0.77 ms (1%) |

  There is nothing to recover past 512 tokens. vLLM's prefills are chunked to
  `max_num_batched_tokens`, so most of *its* forwards are small mixed batches
  where launch overhead dominates and piecewise pays. Capturing prefill on its own
  here would buy about 1.6x on TTFT at ISL 128 and nothing beyond, for one capture
  per (prompt-length bucket x batch width) — and unlike decode's slot-table
  padding, padding a prompt wastes real compute.
- **§1.3 did not undo §3.2.** A mixed step was an eager prefill pass plus a
  graphed decode pass; it is now one eager pass, and every step that was a
  uniform decode batch still is one and still replays its graph. What a mixed
  pass costs uncaptured is its own launches, which the eager prefill pass paid
  before. In the throughput harness mixed steps are also rare — **1.0% to 1.3%**
  of steps under a 2,048-token budget, none without one, because every
  request arrives at once and asks for the same output length. That changes
  under open-loop arrivals (§8.8), where most steps admit: re-price this there.
- **It is also half of §3.1.** vLLM's piecewise mode requires inductor
  compilation, so its pieces are fused *and* captured; the two wins arrive
  together. liteinfer has no compilation step, which is why §3.2 could only take
  the capture half, and why §3.1's 1.12x has to be earned separately by hand.
- **Scope.** A compilation step this engine does not have yet, so the honest
  prerequisite is either `torch.compile` on the pieces between attention calls —
  which runs into the same in-place pool write that measured 0.06x in §3.1 — or
  hand-partitioned capture of the non-attention spans. Neither is small, and
  neither is worth building before §8.8 says how many steps it would cover.

### 3.9 Retire the padded path
- **Status.** `planned` — the closing stage; nothing to build, everything to delete.
- **Stage 7 of the packed-batch move**, which runs ~~§8.7~~ → ~~§3.6~~ → ~~§2.9~~ → ~~§1.7~~ → ~~§2.10~~ → ~~§1.3~~ (all six landed) → §3.9.
- **PRs.** _none yet_
- **Why.** Padding is currently undone by five mechanisms that exist only to
  cancel each other: left-padded inputs, a right-aligned slot table, a null block
  absorbing pad positions, two mask builders, and a two-pass route inside the
  runner's one entry point. Since §1.3 every one of them has a packed equivalent,
  and the engine keeps both.
  Keeping a slower general path "just in case" is how the codebase stops being
  readable.
- **Scope.** Delete `build_prefill_mask` and the padded input builders, drop the
  slot table's right-alignment and the null block if decode capture no longer
  needs it, collapse `_padded_forward`'s two passes inside `execute` into the
  packed route §1.3 left, and then simplify what the choice left behind — a
  dispatcher with one entry is the shape to look for.
- **`varlen_attention` goes too, if an engine run agrees.** A packed pass with
  nothing cached still goes to FlashAttention's varlen entry; every other packed
  pass already reads through `paged_prefill`. Per layer, `paged_prefill` over
  whole prompts is 0.30x of flash at 32 x 18 tokens and within 4% up to 4,096
  (§2.10), so routing every packed pass through it would delete
  `varlen_attention`, `VarlenKV` and `_PackedPrefillPayload` — after a throughput
  run on the mixed dataset, the prefill-heaviest shape at OSL 16, shows no loss.
  §1.3 kept varlen so its own number measured one change.
- **The activation profile has to follow.** `_profile_forward_bytes` sizes the
  pool from a *padded* `sdpa` forward over `max_num_seqs x max_model_len` with a
  full `[B, 1, L, L]` mask, even on an engine that packs. That over-reserves
  today, which is safe; with the padded path gone it is the wrong basis, and the
  worst case to profile becomes one packed pass of `token_budget` tokens.
- **What stays.** `eager` and `sdpa` keep a packed per-sequence loop. They are the
  correctness reference and the CPU path, not performance paths, and should not
  pretend otherwise. `eager` in particular is the oracle the fused kernels are
  checked against, and it is what caught §3.6 attending across prompt boundaries
  when no benchmark did.
- **Price the loop before landing it, because it is not free.** A per-sequence
  loop wastes launches where padding wastes arithmetic, and at short prompts the
  launches cost more. Measured on 32 mixed prompts, padded batch against one
  forward per sequence: **273.2 ms vs 448.5** and 276.5 vs 426.9 where the longest
  prompt was ~320 tokens, but 571.2 vs **456.0** where it was 657. So retiring the
  padded layout makes the dense paths slower on exactly the workloads they serve
  — short prompts, many of them — unless the loop is replaced by something
  better. Say what it costs rather than discovering it afterwards.
- **The rule this follows** is "Shipping an improvement", step 6: confirm the
  superseded configs' results are stored, flag them `historical`, delete the code,
  then collapse the abstraction.

---

## 4. Modeling and parity

### 4.2 Detangle remaining transformers helpers from modeling files
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Llama still imports `ROPE_INIT_FUNCTIONS` and `ACT2FN` from
  `transformers` and subclasses `PreTrainedModel`.
- **Scope.** Bring in-tree incrementally: Llama-3.x RoPE init
  (~15 lines), `ACT2FN["silu"]`. `DynamicCache` tracked separately in §2.4.
- **Parity test.** `tests/e2e/test_llama_parity.py` stays bit-exact
  vs `transformers.AutoModelForCausalLM.generate`.

### 4.3 Add architecture: Qwen-MoE / Mixtral / DBRX
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Exercise the dispatch table and stretch the engine to a
  classical top-k MoE.
- **Scope.** New entry in `_DISPATCH`, vendored modeling file, parity
  test.

---

## 5. Engine ergonomics

### 5.6 Request cancellation
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** `FINISHED_ABORTED` only ever comes from the engine aborting a failed
  forward pass (§5.4). A caller has no way to abort its own request: a client
  that disconnects mid-stream, or an `async for` that breaks out early, leaves
  the sequence running to `max_tokens`, holding a slot and its KV blocks. Under
  a bounded batch that is capacity another request could have used.
- **Scope.** `AsyncLLM.abort(request_id)` marking the sequence
  `FINISHED_ABORTED` so the scheduler evicts it and frees its blocks on the next
  step; `stream()` aborts on generator close, so breaking out of the loop does
  the right thing without the caller knowing the API exists.
- **Parity test.** Breaking out of `stream()` releases the sequence's blocks
  back to the pool within one step.

---

## 6. Observability

### 6.1 Per-request stats
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** `EngineStats` is engine-wide. Per-request TTFT, decode
  latency distribution, output length, finish reason distribution
  belong on `RequestOutput`.
- **Scope.** Track per-request first-token wall and total wall in the
  engine; surface on `RequestOutput`.

### 6.2 Live dashboard runner
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** `EngineStats` records a `StepMetrics` per step (§6.3) and a
  `TimeBreakdown` per loop (§6.4), and nothing consumes either.
- **Scope.** `python -m liteinfer.dashboard` printing rolling
  prefill/decode tok/s, batch width and KV usage from the stats stream.

---

## 7. Hygiene / housekeeping

- Trim `EngineStats`: the two overall throughput properties and the `on_step`
  listener have no callers outside their own tests. §1.3 removed the per-phase
  totals and averages, whose meaning a mixed step broke; `on_step` is what §6.2
  plans to read, so decide it with §6.2.
- Fix the five `reportOptionalMemberAccess` errors pyright reports on package
  code: `hf_config` and `tokenizer` are declared optional because they are
  assigned in `load_model` rather than `__init__`, so every read of them is an
  error. Either construct the runner already loaded, or keep a loaded-state
  object the type system can see. The remaining pyright output is
  `reportPrivateImportUsage` against torch's re-exports, which is noise.

---

## 8. Benchmark harness

### 8.2 Plain HuggingFace `transformers` benchmark runner
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** vLLM is a strong production baseline, but a plain
  `transformers.AutoModelForCausalLM.generate` runner gives a
  simpler, dependency-free lower bound and makes liteinfer's
  overhead vs raw HF visible without the vLLM install requirement.
- **Scope.** New adapter class in `benchmarks/adapters.py`, plus one
  `BenchmarkConfig` entry in `benchmarks/configs.py`.
- **Parity test.** HF runner greedy outputs match liteinfer eager
  outputs on the same prompts (already validated by existing e2e
  parity tests; benchmark runner just reuses that path).

### 8.5 Notice when a run was disturbed
- **Status.** `planned` — **the data is already stored and nothing reads it.**
- **PRs.** _none yet_
- **Why.** Every latency result keeps its 200 per-request `e2e_s` and `ttft_s`
  values, and the report only ever takes percentiles of them. Two rows in the §1.5
  re-measurement had an intra-run spread of **1.27x and 1.48x** where every other
  row in the file is at most 1.09x, and drifted in opposite directions across the
  boundary between them — an external disturbance of about ten minutes on a host
  that has 27 users. Unnoticed, it made the report claim paged decode was 0.91x of
  `sdpa`, reversing §2.3.
- **Scope.** `report.py` computes max/min and a first-quarter-to-last-quarter
  drift per result and marks the row when either leaves a band the rest of the
  file sets. Throughput results record no per-request series, so this covers
  latency only — recording one is the alternative and a bigger change.
- **Why it is worth having rather than remembering.** §8.4 catches a delta between
  two runs that should not be compared. This catches a single run that should not
  be believed, which is the failure the §1.5 numbers actually hit, and the
  criterion has to exist before a number is seen to be worth anything.
- **It also bounds the variance claim.** `docs/benchmarks.md` says run-to-run
  variance is about ±4%. Within a run that holds; between sessions the same
  configuration has read 13.4 to 19.9 ms at ISL 3584. Whatever this measures
  should replace that number with a measured one.

### 8.6 A long-context vLLM reference row
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** §2.7's win lands at 15,360 tokens, and the vLLM rows stop at ISL 2048 —
  so `liteinfer-splitk-16k` is the first liteinfer row with no reference point.
  Comparing it to a 128-token vLLM row would measure the prompt, not the engine,
  and the report is right to print nothing there. Whether vLLM's long-context
  decode is faster than 7.5 ms is simply unknown.
- **Scope.** A `vllm-16k` entry matched to `liteinfer-graphs-16k` on batch width
  and `max_model_len`, run on the ISL 15360 dataset that now exists.
- **Watch the pool.** vLLM sizes its own KV cache from `gpu_memory_utilization`;
  at a 16k budget the two engines must be given the same ceiling or the row
  measures the cache, not the decode.

### 8.8 Open-loop throughput: requests that arrive over time
- **Status.** `planned` — what §1.3's claim is waiting on ([#45](https://github.com/ValeGian/liteinfer/pull/45)).
- **PRs.** _none yet_
- **Why.** Throughput mode submits every request at once, and every request
  asks for the same output length, so the engine admits in waves: a wave starts
  together and finishes together, and the next one is admitted into an empty
  batch. No step ever holds prompts beside decodes unless a token budget splits a
  wave, and even then 1.0%-1.3% of steps do at a 2,048-token budget. A server sees the opposite — requests
  arrive while others decode, so most steps admit — and that is the workload
  §1.3 (one forward per mixed step), §3.8 (capturing mixed passes) and chunking
  itself exist to serve. None of them can show a number here.
- **Scope.** A `--request-rate` on throughput mode: Poisson arrivals at a fixed
  rate and seed, so a run is reproducible, recording per-request TTFT and e2e as
  latency mode does. That second half needs §6.1's per-request timings, or the
  harness timing its own stream events. A rate is part of the shape, like
  `max_isl`, so it travels in the result's key.
- **First use.** Re-measure `liteinfer-packed-budget` against
  `liteinfer-onepass-budget` there. The before is `historical` by then, so the
  comparison has to be run from §1.3's parent revision.
