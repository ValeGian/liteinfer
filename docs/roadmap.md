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

## Marking an item done

When an item lands:

1. Flip `Status` to `landed` and append the merging PR(s) to `PRs`.
2. Move the item out of this file into `milestones.md`, under the
   current month's heading. Keep the `PRs` line so the milestone log
   has the same audit trail.
3. If a follow-up is created (e.g. v1 lands but v2 is the polished
   version), leave a stub here with status `planned` and a backlink
   to the original PR.

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
- **Why.** The initial paged impl (§2.1, landed) gathers non-contiguous KV blocks
  into a contiguous buffer before attention. Measured 29-08-2026 at ISL 128 /
  OSL 256 that costs far more than the ~13% first recorded at OSL 64: paged runs
  at **0.72x of native-eager** on both throughput (73.3 -> 52.7 tok/s) and ITL
  (13.8 -> 19.0 ms), and is **slower per decode step than running with no KV cache
  at all** (19.0 vs 16.5 ms). `PagedKVCache._gather_all` rebuilds the whole K/V
  history into a fresh padded tensor once per layer per step — roughly 8k Python
  tensor ops and ~200 MB copied per step at B=32, so a batch-32 step costs 130 ms
  against 19 ms at batch 1, where a bandwidth-bound decode should cost about the
  same at both widths. Continuous batching (§1.2) is built on paged and inherits
  it: batch occupancy is full (measured: every step at width 32) yet an 8x wider
  batch yields only 1.55x throughput, where vLLM gets 6.17x. A fused
  paged-attention kernel reads the block table inside the CUDA kernel and removes
  the gather entirely, the same approach as vLLM's PagedAttention.
- **Scope.** Custom CUDA/Triton kernel for block-table attention;
  `ModelRunner` switches to it when `cache_mode="paged"`.
- **Pre-req.** §3.3 (SDPA / FlashAttention switch) provides the
  entry point to swap in a custom attention backend.
- **Parity test.** Paged greedy output identical to eager; benchmark
  shows paged ≥ eager tok/s at B=1.

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

### 2.7 Async engine step metrics
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** `AsyncLLMEngine` never calls `stats.record()`, so the continuous
  path emits no `StepMetrics` at all — no phase, batch width, token count or
  per-step wall time. `LLMEngine` records all of it. Diagnosing the continuous
  batching shortfall behind §2.3 required monkey-patching
  `ContinuousScheduler.schedule` from a throwaway script to recover batch
  occupancy, which is not a workflow anyone should need to repeat.
- **Scope.** Record a `StepMetrics` per step in `AsyncLLMEngine._step`, with a
  phase for the mixed prefill+decode case the two-pass step creates. Reuse
  `EngineStats`; no new types.
- **Parity test.** A continuous run reports the same total token count through
  `EngineStats` as the returned outputs contain, and batch width never exceeds
  `max_num_seqs`.

### 3.1 `torch.compile` of the forward path
- **Status.** `planned`
- **PRs.** _none yet_
- **Scope.** Wrap `ModelRunner._execute_eager` in `torch.compile`,
  gated by `EngineConfig.enable_torch_compile`. First call pays
  compile cost; subsequent calls run the compiled graph.
- **Risk.** Dynamic shapes (variable sequence length) defeat compile
  cache. Use `torch._dynamo` shape specialization or pad to bucket
  sizes.
- **Parity test.** Compiled vs eager: identical greedy outputs.

### 3.2 CUDA graph capture for decode
- **Status.** `planned`
- **PRs.** _none yet_
- **Scope.** Capture the single-token-decode step (fixed shape after
  prefill) and replay. Behind `enable_cuda_graph`. Requires §1.1
  (fixed batch size during capture). Graph capture also eliminates the
  per-step `torch.zeros` allocation in `build_additive_mask` and the
  associated Python-side slice fills, which currently incur a GPU
  allocation + multiple kernel launches every forward pass.
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
- **Scope.** Per-rank `ModelRunner` plus a process group. Layer
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
- **Why.** Listener API exists; no canonical consumer.
- **Scope.** `python -m liteinfer.dashboard` printing rolling
  prefill/decode tok/s, batch size, KV usage. Builds on §1.1 to be
  meaningful.

---

## 7. Hygiene / housekeeping

- Drop loader's `_attn_implementation = "eager"` override once §3.3
  lands.
- Vectorize `Sampler.__call__` per-row loop when params are
  homogeneous across batch.
- Tighten `KVCache` ABC — richer `payload` contract landed with §2.1; may still obviate eager wrapper.
- Add `tests/integration/` B=1 vs B=N parity tests once §1.1 lands.

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
  makes this crossover explicit, validates that §2.6 (pre-allocated
  buffer) actually closes the gap, and guards against regressions.
- **Scope.** New `bench run-sweep` subcommand iterating over `(ISL, OSL)` pairs,
  calling `harness.run()` for each. `report.py` renders a tok/s-vs-OSL section.
- **Pre-req.** §8.1 (ISL/OSL-controlled workloads).
- **Parity test.** Greedy outputs identical across engines at each
  (ISL, OSL) point.

