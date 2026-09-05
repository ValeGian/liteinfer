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

Every change that claims to make liteinfer faster follows the same loop. The
point is that the claim is always backed by a number measured the same way
before and after, and that a feature which wins outright replaces the one it
beats rather than sitting beside it.

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

**6. If it is a clear win, delete what it beat.** A clear win is: better on the
mode the feature targets, by more than variance, with any regression elsewhere
small enough to state and accept. Then, in order:

   - confirm the superseded config's results are stored — that measurement is
     the only record once the code is gone;
   - flag its `BenchmarkConfig` entry `historical` so the report keeps rendering
     the progression while `bench run` refuses it;
   - delete the code, and everything that existed only to serve it;
   - **simplify what is left.** Removing one of two paths usually makes an
     abstraction pointless: a mode flag with one value, a dispatcher with one
     entry, a base class with one subclass. Collapse them in the same PR, or
     the codebase keeps the shape of a choice it no longer offers.

Keeping a slower path "just in case" is how the codebase stops being readable.
The benchmark exists so that deleting it is safe.

## Marking an item done

When an item lands:

1. Flip `Status` to `landed` and append the merging PR(s) to `PRs`.
2. Move the item out of this file into `milestones.md`, under the
   current month's heading. Keep the `PRs` line so the milestone log
   has the same audit trail.
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
  attention. Since the gather was vectorised that costs ~8-10% against
  native-eager (0.92x throughput at B=1, 0.90x at B=4 and on ITL), and what is
  left is memory traffic rather than overhead: the gather still copies the whole
  K/V history every decode step. A fused paged-attention kernel reads the block
  table inside the CUDA kernel and removes the copy entirely, the same approach
  as vLLM's PagedAttention.
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

---

## 3. Performance optimizations

### 3.1 `torch.compile` of the forward path
- **Status.** `planned`
- **PRs.** _none yet_
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
- **Scope.** Capture the single-token-decode step and replay it, behind
  `enable_cuda_graph`. Continuous batching varies the decode width per
  step, so capture needs a set of graphs at padded batch sizes rather
  than one — the batch is padded up to the nearest captured width.
  Capture also removes the per-step `torch.zeros` allocation in
  `build_continuous_decode_mask` and its Python-side slice fills, which
  cost a GPU allocation plus several kernel launches every pass.
- **Pre-req.** §3.1 helps but is not strictly required.

### 3.3 Flash / SDPA attention
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Vendored modeling already supports `_attn_implementation`
  switching; loader pins `eager` for v0.
- **Scope.** Allow `EngineConfig.attn_implementation` and pass through
  to `hf_config._attn_implementation`. Default to `sdpa` once parity
  proven.
- **Parity test.** SDPA vs eager: outputs match within `torch.testing
  .assert_close` tolerances.

### 3.4 Tensor parallelism (single-node)
- **Status.** `planned`
- **PRs.** _none yet_
- **Scope.** Per-rank `ContinuousModelRunner` plus a process group. Layer
  weights sharded along output dim (column-parallel) or input dim
  (row-parallel) per HF `_tp_plan`. Already declared in vendored
  models.
- **Surface change.** Loader streams shards onto the right rank;
  attention layers all-reduce.

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

### 5.2 Incremental detokenization
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Stop-string detection currently re-decodes the entire
  output every step (`engine/llm_engine.py::_maybe_finish`). Fine for
  v0; pathological on long outputs.
- **Scope.** Cache the last decoded suffix and only decode the new
  token's contribution.

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
- **Why.** `EngineStats` now records a `StepMetrics` per forward pass
  (§6.3), and nothing consumes it.
- **Scope.** `python -m liteinfer.dashboard` printing rolling
  prefill/decode tok/s, batch width and KV usage from the stats stream.

---

## 7. Hygiene / housekeeping

- Drop loader's `_attn_implementation = "eager"` override once §3.3
  lands.
- Vectorize `Sampler.__call__` per-row loop when params are
  homogeneous across batch.
- Trim `EngineStats`: six derived throughput properties, the `on_step`
  listener and four running totals have no callers outside their own tests.

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

### 8.3 Sequence-length sweep workload
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** At short sequence lengths (small prompts, low `max_tokens`)
  the RECOMPUTE engine outperforms KV-cache engines because single-token
  decode steps are memory-bandwidth-bound and GPU-underutilised, while
  `torch.cat` KV growth adds per-step copy overhead. The crossover point
  — where KV cache starts winning — is model- and hardware-dependent and
  currently invisible in the benchmark suite. A sequence-length sweep
  makes this crossover explicit and guards against regressions.
- **Scope.** New `bench run-sweep` subcommand iterating over `(ISL, OSL)` pairs,
  calling `harness.run()` for each. `report.py` renders a tok/s-vs-OSL section.
- **Pre-req.** §8.1 (ISL/OSL-controlled workloads).
- **Parity test.** Greedy outputs identical across engines at each
  (ISL, OSL) point.

