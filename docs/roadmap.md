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

### 2.3 Paged KV cache performance (fused kernel)
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Paged gathers non-contiguous KV blocks into a contiguous buffer before
  attention. That gather is now measured rather than inferred: profiled at B=32
  it is `vectorized_gather_kernel` at **0.652 ms, 5.9% of GPU time** per decode
  step. Worth having, and smaller than this item used to claim — the ~8-10%
  figure it carried was measured at B=1 before the flat-slot rewrite, and is no
  longer the number to plan against. A fused paged-attention kernel reads the
  block table inside the CUDA kernel and removes the copy entirely, the same
  approach as vLLM's PagedAttention.
- **Scope.** Custom CUDA/Triton kernel for block-table attention;
  `ContinuousModelRunner` switches to it for the decode pass.
- **Pre-req.** §3.3 (SDPA / FlashAttention switch) provides the entry point to
  swap in a custom attention backend.
- **Parity test.** Paged greedy output identical to eager; benchmark shows
  paged >= eager tok/s at B=1.

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
- **PRs.** _none yet_
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
- **PRs.** _none yet_
- **Why.** The largest measured cost in the decode path is not a kernel — it is
  the gaps between kernels. Profiled on the current tree, one decode step at
  B=32 (Llama-3.2-1B, ISL 128, A40):

  | | |
  |---|---:|
  | step wall time | 13.96 ms |
  | weight-read floor | 3.56 ms |
  | GPU busy | 11.02 ms (79%) |
  | **GPU idle, waiting on Python** | **2.93 ms (21%)** |
  | CUDA kernels launched per step | **822** |

  Graph replay collapses those 883 launches into one, so the idle quarter is
  what it directly recovers — about 1.3x before any kernel gets faster.
  These numbers have been re-measured twice, because two other changes might
  have eaten into them. §7's sampler vectorisation did not: sampling sits
  outside the forward pass, and the step was unchanged at 14.75 ms, 25% idle,
  883 launches. Building the per-step tensors in one transfer did: 35 pageable
  host-to-device copies per step became 5, which is where the drop to 13.96 ms,
  21% and 822 launches came from. What is left is launch overhead proper. It is
  also the only item that attacks the deficit the benchmark has shown all along:
  liteinfer sits at 0.28-0.35x of vLLM at *every* batch width, and a gap that
  stays flat as concurrency grows is a fixed per-step cost, not an algorithmic
  one.
- **Scope.** Capture the single-token-decode step and replay it, behind
  `enable_cuda_graph`. Continuous batching varies the decode width per
  step, so capture needs a set of graphs at padded batch sizes rather
  than one — the batch is padded up to the nearest captured width.
  Capture also removes the per-step `torch.zeros` allocation in
  `build_continuous_decode_mask` and its Python-side slice fills, which
  cost a GPU allocation plus several kernel launches every pass.
- **Pre-req.** §3.1 helps but is not strictly required.

### 3.4 Tensor parallelism (single-node)
- **Status.** `planned`
- **PRs.** _none yet_
- **Scope.** Per-rank `ContinuousModelRunner` plus a process group. Layer
  weights sharded along output dim (column-parallel) or input dim
  (row-parallel) per HF `_tp_plan`. Already declared in vendored
  models.
- **Surface change.** Loader streams shards onto the right rank;
  attention layers all-reduce.

### 3.5 Broadcast the grouped-query heads on the cuDNN backend
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Two costs, one fix. `_repeat_kv` materialises a 4x copy of K and V so
  8 KV heads line up with 32 query heads, and the expanded tensors then make the
  attention kernel read four times the KV it needs. Asking the kernel to
  broadcast instead removes both — but *only on the right backend*, and picking
  it is the whole content of this item.

  Measured on one attention call, bf16, with the mask the engine builds:

  | | peak | decode B=32 ctx 2048 | ctx 256 | prefill B=8 len 1024 |
  |---|---:|---:|---:|---:|
  | expanded heads *(ships today)* | 96 MiB | 2.329 ms | 0.312 ms | 1.632 ms |
  | `enable_gqa`, default dispatch | **4,904 MiB** | — | — | — |
  | `enable_gqa` on **cuDNN** | **32 MiB** | **0.229 ms** | **0.047 ms** | 1.706 ms |

  **6.7-10x on the decode attention kernel**, a third of the memory, and roughly
  neutral on prefill — which is the right trade, since a request pays prefill
  once and decode hundreds of times.
- **Why it is not automatic.** PyTorch will not choose this on its own. With
  broadcast heads and a mask, the memory-efficient backend refuses ("both fused
  kernels require query, key and value to have the same num_heads") and flash is
  already out on the mask, so the default dispatch falls to **math**, which
  materialises the score matrix and undoes §3.3 — the 4,904 MiB row. cuDNN
  attention serves it, but is runtime-disabled by default in this build, so it
  has to be selected explicitly with
  `torch.nn.attention.sdpa_kernel(SDPBackend.CUDNN_ATTENTION)`.
- **Scope.** Select the backend explicitly in `sdpa_attention` and drop
  `_repeat_kv` on that path; `eager_attention` keeps the expansion, since writing
  it out is what makes it the readable reference. **A fallback is required**: on
  a machine where cuDNN attention is unavailable, forcing it raises "No available
  kernel", and the broadcast path cannot fall back to memory-efficient — it has
  to fall back to expanding the heads. That fallback is the reason this is not a
  one-line change.
- **Predicted effect, written before the run.** Amdahl caps this well below the
  kernel number. Attention is 1.57 ms of a 14.68 ms decode step (14.2% of the
  11.08 ms the GPU is busy; the other 3.60 ms it is idle). A 10x attention
  kernel therefore buys **1.11x** on the step, or about **1.13-1.16x** once the
  `_repeat_kv` expansion goes with it — against a 1.1x threshold and ±4%
  run-to-run variance. The end-to-end benchmark figure will move less again,
  since prefill is neutral and some of the wall is outside the forward pass
  (§6.4). **Do not report the 6.7-10x as a liteinfer speedup**; it is a kernel
  measurement, and quoting it as an engine result is the mistake this item has
  already made three times in other forms.
- **Parity test.** The kernels differ by ~1e-3 in bf16 here, more than the
  current path, so pin equivalence in fp32 as §3.3 did — and assert peak memory
  no worse than the expanded-head path, since numerical agreement is exactly the
  check that misses a math-backend fallback.
- **Note.** An earlier version of this item claimed `enable_gqa` was verified
  viable, then that it was blocked on §3.6. Both were wrong: the first tested
  values and not memory, the second assumed the mask was the obstacle when the
  obstacle was the backend.

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
