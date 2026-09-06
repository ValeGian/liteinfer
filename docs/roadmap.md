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
  decode_seqs)` path; `AsyncLLMEngine._step` uses it once §3.3 (SDPA /
  Flash) lands. The two-pass path stays as a fallback for eager attention.
- **Pre-req.** §3.3 (Flash / SDPA attention backend).

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

### 2.3 Paged attention that reads the block table in-kernel
- **Status.** `planned` — **the keystone. Everything else in §3 is waiting on it.**
- **PRs.** _none yet_ — prototype and measurements in `benchmarks/paged_attention_probe.py`
- **Why.** Decode gathers every sequence's whole KV history into a contiguous
  tensor, per layer, per step, and then runs a general attention kernel over it.
  A kernel that walks the block table itself never gathers, never expands the
  grouped-query heads, and is shaped for `q_len == 1`. A ~60-line Triton
  prototype, B=32, 8 KV heads, 32 query heads, 190 tokens of context:

  | per layer | |
  |---|---:|
  | gather + memory-efficient attention *(today)* | 0.2961 ms |
  | **Triton paged decode** | **0.0497 ms** |
  | | **5.96x** |

  Output matches to 3.9e-03 in bf16, about two ULP. Per decode step that is
  4.738 ms of measured-in-isolation work falling to 0.795 ms. Isolated figures
  have read 30-60% high all through this roadmap, so **expect roughly 1.25x in
  the engine, not 6x** — and measure it there before claiming anything.
- **An earlier estimate here was wrong.** This item was sized at a 1.038x
  ceiling on the theory that a fused kernel could at best match the existing
  attention and save only the gather. It does not match it; it beats it, because
  the existing path expands 8 KV heads to 32 and hands a general kernel a padded
  key axis, while the paged kernel reads each KV head once and broadcasts over
  its query group in-register. That also makes §3.5 unnecessary rather than
  blocked: there is no `_repeat_kv` left to remove.
- **What it unblocks, and why they are all stuck without it.** Three separate
  optimisations were built and measured this cycle, and the paged cache is why
  each failed:

  | item | measured | what the cache did to it |
  |---|---|---|
  | §3.2 CUDA graphs | 1.06x | a graph fixes the shape, so the *gather* has to be padded — real bytes per layer per step |
  | §3.5 cuDNN broadcast KV | 0.64x | the KV length changes every step, so cuDNN re-plans; 36 ms per call |
  | §3.1 `torch.compile` | 0.06x | inductor functionalises the in-place pool write into a **copy of the whole pool**, 182 ms of a 206 ms forward |

  Remove the gather and the in-graph pool mutation and all three change
  character: there is nothing left for bucketing to inflate, and nothing for
  inductor to functionalise.
- **Scope, and the design question to settle first.** The kernel is the easy
  part. The open question is how a query reaches the block table without
  putting attention inside the cache or the cache inside the model:
  `_DecodePayload.update` currently scatters *and* gathers, returning `(k, v)`
  to `LlamaAttention`, which is exactly the contract a paged kernel does not
  want. Options are a third `attn_implementation` that takes the pool and slots
  instead of `(k, v)`, or a payload that answers a query directly. Pick one
  deliberately — `models/llama.py` is the file this codebase most expects to be
  readable.
- **Parity test.** Greedy output identical to `sdpa` in fp32, bf16 agreement to
  ULP on real rows, and a decode benchmark rather than a kernel benchmark.

### 2.5 Quantized cache (KV-cache fp8 / int8)
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Memory bound on long contexts.
- **Scope.** Storage-level quant of K and V; dequant on read. Behind
  `cache_quant: str | None` flag.
- **Risk.** Quality regression on long contexts; needs a tolerance
  parity test against fp16/bf16.

### 2.6 KV pool sizing from a measured activation budget
- **Status.** `planned`
- **PRs.** follow-up to [#20](https://github.com/ValeGian/liteinfer/pull/20)
- **Why.** #20 sized the pool as `min(affordable, reachable)`, which stops it
  hoarding VRAM it can never address. `affordable` is still a guess: a fraction
  of whatever happens to be free when `load_model` runs. Two consequences. It is
  not reproducible — the same config on the same GPU sizes differently depending
  on what else was resident a second earlier, which makes a benchmark's pool a
  property of the machine's history. And the fraction is a stand-in for the real
  question, which is how much memory the forward pass needs at full width.
- **Scope.** Two steps, in order. First move the fraction to *total* VRAM, so
  the size is a function of the config and the device only. Then replace it:
  run one worst-case forward pass at `max_num_seqs` × `max_model_len` during
  load, record peak allocation, and give the pool what is left — vLLM's
  approach. The WARNING #20 added stays useful either way; it names which of
  the two constraints bound the pool.
- **Parity test.** Pool size is identical across two loads separated by an
  unrelated allocation; profiled size leaves a forward pass at full width
  headroom to complete.

---

## 3. Performance optimizations

### 3.1 `torch.compile` of the forward path
- **Status.** `planned`
- **Blocked on.** §2.3. Measured at **0.06x** as the code stands.
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
- **Why.** Elementwise work is the second-largest slice of decode GPU time and
  the most fusible: profiled at B=32, `elementwise_kernel` accounts for
  **17.9% + 2.6% across ~143 launches per step** — RoPE, residual adds, norms
  and the mask fills — against 30.1% for the projection GEMMs that actually do
  the model's arithmetic. Fusion removes both the time and the launches, so this
  overlaps §3.2; do §3.2 first and re-measure before sizing this one.
- **Scope.** Wrap the `ContinuousModelRunner` forward passes in `torch.compile`,
  gated by `EngineConfig.enable_torch_compile`. First call pays
  compile cost; subsequent calls run the compiled graph.
- **Risk.** Dynamic shapes (variable sequence length) defeat compile
  cache. Use `torch._dynamo` shape specialization or pad to bucket
  sizes.
- **Parity test.** Compiled vs eager: identical greedy outputs.

### 3.2 CUDA graph capture for decode
- **Status.** `planned`
- **Blocked on.** §2.3. Built, measured at 1.06x, and reverted — see below.
- **Why it looked compelling.** A decode step issues **822 kernels** to do 10.34 ms
  of GPU work in a 13.96 ms step; the gaps between launches are idle time, and a
  graph replays the whole sequence as one submission.
- **What it actually measured.** The saving is real but small, and the padding a
  graph forces costs more than it saves at most bucket sizes. A graph fixes
  every shape, so the KV length has to be rounded up to a bucket:

  | KV bucket | throughput | vs no graphs |
  |---:|---:|---:|
  | none | 1,809 tok/s | — |
  | 256 | 1,759 | 0.97x |
  | **64** | **1,912** | **1.06x** |
  | 32 | 1,840 | 1.02x |
  | 16 | 1,679 | 0.93x |

  Large buckets waste attention and gather work on padding; small ones spend the
  gain on captures. The best point is 1.06x — and the same no-graph
  configuration measured 1,809 and 1,932 tok/s on two consecutive runs, so the
  effect is smaller than the noise. Replaying the forward in isolation saves
  0.858 ms of 12.85 ms, which is the whole prize.
- **Why it is blocked, and on what.** The cost is the *gather*. liteinfer copies
  the whole KV history into a contiguous tensor every decode step, so padding
  the KV length pads a copy — real bytes, per layer, per step. §2.3's fused
  paged-attention kernel reads the block table inside the kernel and never
  gathers, so there is nothing for the padding to inflate. That is why vLLM
  captures graphs profitably and liteinfer cannot yet: **§2.3 first, then this.**
- **Scope, when it comes back.** Static buffers written in place, one capture per
  (batch width, KV bucket), padded rows attending to nothing and discarded.
  Roughly 150 lines; the implementation is in the history of the PR that filed
  this measurement.
- **Prerequisite already landed.** The slot table and block allocation used to be
  computed on the first layer, inside the forward pass. They now happen in
  `decode()` before it, so the forward contains no host-side work — which any
  capture requires, and which §2.3 wants anyway.
- **Parity test.** Graphed output is token-identical to eager in fp32. In bf16 it
  diverges around token 11, because attention then sums over a padded key axis
  in a different order — the same class of difference as §3.3, and the reason
  the parity claim has to be made in fp32.

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
- **Status.** `planned` — **likely superseded by §2.3.**
- **Why it may not be needed.** This item exists to stop `_repeat_kv`
  materialising a 4x copy of K and V, 0.74 GiB of writes per decode step. §2.3's
  paged kernel reads each KV head once and broadcasts over its query group
  in-register, so there is no expansion left to remove. Keep the item open only
  until §2.3 lands and the profile confirms the copy is gone.
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

### 5.5 Backpressure on admission
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** `max_num_seqs` caps how many sequences run, not how many are
  accepted. `AsyncLLMEngine._pending` and `ContinuousScheduler.waiting` are both
  unbounded, so a caller that submits faster than the engine drains grows two
  Python lists without limit until the process dies — and it dies holding
  admitted work that never ran. Every production serving stack rejects or blocks
  instead; that liteinfer does not is a gap, not a simplification.
- **Scope.** `EngineConfig.max_waiting_seqs`, enforced in `add_request`. Decide
  the failure mode explicitly: `await` on a bounded `asyncio.Queue` gives the
  caller backpressure, raising gives it a fast 429-shaped answer. Prefer
  raising — it is visible, and a hung `add_request` is the bug this item
  exists to avoid.
- **Parity test.** Submitting `max_waiting_seqs + 1` requests raises on the
  last one and leaves the first ones running.

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
