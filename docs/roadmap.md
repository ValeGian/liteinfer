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

### 1.1 Static batching with B > 1
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** v0 caps the runner at batch size 1; throughput on multi-prompt
  workloads is a fraction of single-GPU peak.
- **Scope.** `engine/model_runner.py` (left-padding, attention masks,
  per-seq position ids), `engine/scheduler.py` (`start_batch` >1),
  finish-mask bookkeeping in `engine/llm_engine.py`. Early-finished
  sequences either masked out or dropped from cache mid-batch.
- **Parity test.** B=1 vs B=8: greedy outputs identical.

### 1.2 Continuous batching
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Static batching wastes compute when prompts finish at
  different steps and new requests arrive mid-batch.
- **Scope.** New `engine/scheduler_continuous.py` (or behind a config
  flag). Scheduler must merge new arrivals into the running set
  step-by-step, evict on KV-block pressure once paging lands.
- **Pre-req.** §2.1 paged KV cache.
- **Parity test.** Same multi-prompt workload: continuous mode produces
  the same outputs as static, with strictly higher throughput.

### 1.3 Multi-sequence per request (`n > 1`)
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Sampling `n` candidates from one prompt is a common UX
  pattern; v0 rejects it.
- **Scope.** `SequenceGroup` already supports a list of sequences. The
  runner must duplicate the prompt KV across `n` rows once and then
  diverge per row. Sampler must consume one params instance per row.
- **Parity test.** `n=1` reproduces v0; `n=4` with seeded sampling
  produces 4 distinct, deterministic outputs.

---

## 2. KV cache implementations

### 2.1 Paged KV cache
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Eager cache wastes memory on padding and forbids prefix
  reuse. Paged blocks are the foundation for §2.2 and §1.2.
- **Scope.** New `liteinfer/cache/paged_kv_cache.py`: block-allocated
  tensors with free list. Vendored attention reads via block table.
  `KVCache` ABC needs richer `payload` protocol.
- **Parity test.** Greedy identical to eager on Llama-3.2-1B for
  prompt lengths 1, 17, 512.

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

### 2.3 Quantized cache (KV-cache fp8 / int8)
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Memory bound on long contexts.
- **Scope.** Storage-level quant of K and V; dequant on read. Behind
  `cache_quant: str | None` flag.
- **Risk.** Quality regression on long contexts; needs a tolerance
  parity test against fp16/bf16.

### 2.4 Native `EagerKVCache` (drop `DynamicCache` wrapper)
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Removes last inference-path dependency on
  `transformers.cache_utils`.
- **Scope.** Rewrite `cache/eager_kv_cache.py` as per-layer
  `(K, V)` tensor list; update vendored attention layers to consume
  it directly.
- **Parity test.** Greedy outputs identical to wrapper-based eager
  cache on Llama-3.2-1B.

---

## 3. Performance optimizations

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
  (fixed batch size during capture).
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
- **Why.** `gemma4.py` still imports `transformers.{masking_utils,
  modeling_layers, modeling_utils, modeling_rope_utils, integrations}`
  and subclasses `PreTrainedModel`. Llama still depends on
  `ROPE_INIT_FUNCTIONS` and `ACT2FN`.
- **Scope.** Bring in-tree incrementally: Llama-3.x RoPE init
  (~15 lines), `ACT2FN["silu"]`, Gemma4 causal/sliding-window mask.
  `DynamicCache` tracked separately in §2.4.
- **Parity test.** `tests/e2e/test_llama_parity.py` stays bit-exact
  vs `transformers.AutoModelForCausalLM.generate`.

### 4.3 Add architecture: Qwen-MoE / Mixtral / DBRX
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Exercise the dispatch table and stretch the engine to a
  classical top-k MoE, complementing Gemma4.
- **Scope.** New entry in `_DISPATCH`, vendored modeling file, parity
  test.

---

## 5. Engine ergonomics

### 5.1 Streaming output
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** UX for long generations.
- **Scope.** New iterator API on `LLM`: `for token in llm.stream(...)`.
  Probably yields `TokenEvent{request_id, token_id, text_delta,
   step_metrics}`.

### 5.2 Incremental detokenization
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Stop-string detection currently re-decodes the entire
  output every step (`engine/llm_engine.py::_maybe_finish`). Fine for
  v0; pathological on long outputs.
- **Scope.** Cache the last decoded suffix and only decode the new
  token's contribution.

### 5.3 Async / server interface
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Multiple concurrent clients without one blocking another.
- **Scope.** `LLM.add_request_async` + `await response`. Probably a
  thin asyncio wrapper around the existing engine loop running in a
  background thread.

### 5.4 Chat templating
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Tokenizer wrapper exposes `apply_chat_template` but the
  facade does not. Users must call into `tokenizer` themselves.
- **Scope.** `LLM.chat(messages: list[dict], ...)` shortcut.

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

### 6.3 Trace export
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Profiling deep dives benefit from a Chrome-trace JSON.
- **Scope.** `EngineStats.to_chrome_trace()` emitting one event per
  step.

---

## 7. Hygiene / housekeeping

- Strip HF training-flow decorators from `gemma4.py`
  (`@auto_docstring`, `@can_return_tuple`, `@capture_outputs`,
  `@merge_with_config_defaults`, `@use_kernelized_func`,
  `@use_experts_implementation`, `@dynamic_rope_update`). No-ops at
  inference; drag in `transformers.utils` imports.
- Replace `Gemma4PreTrainedModel(PreTrainedModel)` with `nn.Module`
  once weight-tying / `_attn_implementation` plumbing is in-tree.
- Drop loader's `_attn_implementation = "eager"` override once §3.3
  lands.
- Vectorize `Sampler.__call__` per-row loop when params are
  homogeneous across batch.
- Tighten `KVCache` ABC after §2.1 — richer `payload` may obviate
  eager wrapper.
- Add `tests/integration/` B=1 vs B=N parity tests once §1.1 lands.

---

## 8. Benchmark harness

### 8.1 ISL / OSL-controlled workloads
- **Status.** `planned`
- **PRs.** _none yet_
- **Why.** Current workloads let the model decide output length (EOS
  or `max_tokens`), so two engines may generate different numbers of
  tokens for the same prompt. This makes tok/s and E2E comparisons
  unreliable across engines. Fixed ISL (Input Sequence Length) and
  OSL (Output Sequence Length) give reproducible, apples-to-apples
  numbers.
- **Scope.** Add `isl` / `osl` parameters to `Workload` and to the
  `--workload` CLI flag. Prompt construction: tokenize a template to
  exactly ISL tokens (pad or truncate). Output forcing: pass
  `min_tokens=OSL, max_tokens=OSL` to each engine (requires
  liteinfer `SamplingParams` to expose `min_tokens`). Add standard
  workload presets such as `(ISL=128, OSL=128)` and
  `(ISL=512, OSL=512)` to `benchmarks/workloads.py`.
- **Parity test.** Assert `len(output_token_ids) == OSL` for every
  result in both engines.
