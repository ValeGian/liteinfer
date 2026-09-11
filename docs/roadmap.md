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

### 1.3 Chunked prefill / single-pass mixed batching
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** The current continuous-batching step issues two separate
  forward passes when newly admitted sequences (prefill) and running
  sequences (decode) coexist: one prefill pass and one decode pass.
  Chunked prefill merges both into a single forward pass by interleaving
  prefill tokens and decode tokens in the same batch tensor. This halves
  kernel launches in the common case and reduces TTFT for waiting
  sequences.
  It also bounds peak activation memory: a chunk size caps how many prefill
  tokens enter one pass, so prompt length stops setting the size of the
  largest allocation. That is the second half of the ISL 1024 failure
  (`docs/benchmarks.md`, "Long prompts are liteinfer's weak shape") — §3.3
  removes the score matrix, §1.3 caps what feeds it.
- **Scope.** Requires a flash-attention-style kernel that accepts
  per-sequence key-length metadata (block tables + variable query
  lengths). `ContinuousModelRunner` grows a `mixed_step(prefill_seqs,
  decode_seqs)` path; `AsyncLLMEngine._step` uses it. The two-pass path stays as
  a fallback for the dense kernels.
- **Pre-req.** §3.6, which is where the per-sequence length metadata comes from.
  §2.3's kernel is the other half: it already takes per-sequence key lengths and
  a slot table, so mixing prefill and decode in one pass is a question of giving
  it more than one query per sequence.
- **It costs some of §3.2, and §3.8 is the repair.** A mixed pass is not a uniform
  decode batch, so the captured decode graph does not cover it — roughly one step
  in eight at OSL 256 with 32 slots. Land this with a plan for capturing mixed
  batches, which is what §3.8 exists to be.

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
- **Status.** `planned` — **superseded by §2.3 on the decode path.**
- **What is left of it.** This item existed to stop `_repeat_kv` materialising a
  4x copy of K and V, 0.74 GiB of writes per decode step. §2.3's kernel reads
  each KV head once and broadcasts over its query group in-register, so on the
  paged path that copy is gone. `_repeat_kv` still runs in **prefill**, and on
  the dense decode path that CPU and Triton-less installs still use — so
  the item is not closed, it is re-scoped to prefill, where the copy is made once
  per prompt rather than once per token and is worth much less.
- **What it measured on its own.** Only cuDNN broadcasts KV heads under an
  additive mask, and cuDNN builds a plan per shape while decode changes shape
  every step: 0.223 ms fixed-shape, **36.177 ms** when the shape grows by one
  each call. In the engine, **0.64x** — reverted. Bucketing the KV length to 64
  cuts 128 distinct shapes to 3 and reaches parity, not a win.

### 3.6 Pack the batch instead of padding it
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Prompts of different lengths are left-padded, so every attention call
  needs an explicit additive mask to hide each row's pad prefix. That mask is
  what keeps liteinfer off FlashAttention: PyTorch reports "Flash Attention does
  not support non-null attn_mask", so SDPA falls to the memory-efficient
  backend. Both tile the softmax, so §3.3's memory win is unaffected — but flash
  is the faster of the two, and the padding also costs real compute on positions
  that are thrown away. Verified: the same tensors with no mask and
  `is_causal=True` make flash available.
- **Scope.** Pack the batch as one flat token run plus cumulative sequence-length
  offsets (`cu_seqlens`), the varlen entry point vLLM uses. Padding stops
  existing, so `engine/attention_mask.py` has nothing to mask and `is_causal`
  replaces it. Touches the runner's input builders, the cache's `slot_table`
  right-alignment, and the null block, which exists to absorb padded positions.
- **Unblocks.** §1.3 needs the same per-sequence length metadata to mix prefill
  and decode in one pass, so this is its prerequisite as much as its own change.
  It also buys flash over the memory-efficient backend for the padded case, and
  stops computing on positions that are discarded. Note what it does *not* gate:
  §3.5 turned out to be reachable without it, on a different backend.
- **Parity test.** Greedy output unchanged on a variable-length batch, which is
  the case padding exists to serve.

### 3.8 Capture mixed batches by splitting the graph at attention
- **Status.** `planned` — **low value today and not for a general reason**, which
  is the part worth keeping. Re-price it when §1.3 lands.
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
- **§1.3 is what changes the arithmetic, and it partly undoes §3.2.** Chunked
  prefill merges prefill and decode into one pass, and a mixed pass is not a
  uniform decode batch, so `DecodeGraphs` would not cover it. At OSL 256 with 32
  slots that is roughly one step in eight, so the loss is modest — but §1.3 should
  not land without a plan for capturing mixed batches, and this is that plan.
- **It is also half of §3.1.** vLLM's piecewise mode requires inductor
  compilation, so its pieces are fused *and* captured; the two wins arrive
  together. liteinfer has no compilation step, which is why §3.2 could only take
  the capture half, and why §3.1's 1.12x has to be earned separately by hand.
- **Scope.** A compilation step this engine does not have yet, so the honest
  prerequisite is either `torch.compile` on the pieces between attention calls —
  which runs into the same in-place pool write that measured 0.06x in §3.1 — or
  hand-partitioned capture of the non-attention spans. Neither is small; both are
  worth less than §1.4 until §1.3 exists.

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
- **Why.** `EngineStats` records a `StepMetrics` per forward pass (§6.3) and a
  `TimeBreakdown` per loop (§6.4), and nothing consumes either.
- **Scope.** `python -m liteinfer.dashboard` printing rolling
  prefill/decode tok/s, batch width and KV usage from the stats stream.

---

## 7. Hygiene / housekeeping

- Trim `EngineStats`: six derived throughput properties, the `on_step`
  listener and four running totals have no callers outside their own tests.
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
