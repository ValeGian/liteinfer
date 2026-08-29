# Milestones

Achieved milestones, newest first. When a roadmap item lands: flip its `Status` to `landed` in `roadmap.md`, then add an entry here.

---

## 29-08-2026 — §8.1 ISL/OSL benchmark harness refactor

- **PRs.** _pending merge_
- **What.** Replaced the old benchmark system (append-only history file, one runner
  class per engine/config) with a data-driven matrix. `benchmarks/configs.py` holds
  every configuration and names the config each one improves on, so the report emits
  a 1:1 delta per change rather than an undifferentiated table. Datasets are canonical
  and committed; every result carries the SHA-256 of its prompt set.
- **Methodology.** `throughput` and `latency` report disjoint metric sets — rates
  under saturation, percentiles at one request in flight — because a per-request
  latency measured under saturation mostly records queue position. ITL is derived
  from `(e2e - ttft) / (osl - 1)`, and TTFT from a separate single-token pass, so both
  engines are timed identically with no per-token hooks. Forced output length
  (`min_tokens = max_tokens = OSL`, `ignore_eos`) is verified per run and fails the
  run when violated. Warmup exercises the measured path at the real ISL and batch
  width; each (config, mode) runs in a fresh process.
- **Engines.** liteinfer and vLLM 0.28.0, both in-process. TRT-LLM dropped.
- **Fixed.** `AsyncLLMEngine` carried its own copy of the stop rule that ignored
  `ignore_eos` and `min_tokens`, silently truncating every continuous-batching run.
  Both engines now share `liteinfer/engine/stopping.py`.
- **First findings.** The paged KV cache is a net regression at this shape (0.72x of
  native-eager on both throughput and ITL; slower per decode step than running with
  no cache at all), and continuous batching inherits it — 1.55x for an 8x wider batch
  where vLLM gets 6.17x. Static batching is on par with vLLM (3.70x vs 3.84x for
  B=1 -> B=4), and liteinfer leads on TTFT (15.7 ms vs 21.3 ms, though at this prompt
  length TTFT is mostly per-call API overhead). The native-eager rewrite is
  performance-neutral across every measurement.
- **Certification.** Throughput matches `vllm bench throughput` to 1.0% at the same
  shape. Continuous-batching occupancy was verified to reach full width 32 on every
  step, so the scaling shortfall is per-step cost, not scheduling. An initial vLLM
  TTFT reading was discarded: it is IPC-bound and was inflated 24% by the parallel
  sweep, so latency now runs sequentially and the CLI warns otherwise.

---

## 12-05-2026 — §1.2 Continuous batching (`AsyncLLM`) + §5.3 Async / server interface + §5.1 Streaming output

- **PRs.** [#13](https://github.com/ValeGian/liteinfer/pull/13)
- **What.** `AsyncLLM` exposes an asyncio interface (`async with AsyncLLM(...) as llm`, `await llm.generate(...)`, `async for event in llm.stream(...)`). `AsyncLLMEngine` runs a background asyncio Task; `ContinuousScheduler` fills empty slots from `waiting` on every step and evicts finished sequences individually. `ContinuousModelRunner` issues two forward passes per step when prefill and decode sequences coexist (single-pass chunked prefill is §1.3). Always uses `cache_mode="paged"`. Benchmark runners added for `liteinfer-continuous` and `vllm-continuous`.

---

## 10-05-2026 — §2.1 Paged KV cache (block-pool allocator)

- **PRs.** [#12](https://github.com/ValeGian/liteinfer/pull/12)
- **What.** `PagedKVCache` stores tokens in fixed-size blocks drawn from a pre-allocated pool. Block table maps logical sequence positions to physical block slots. `cache_mode="paged"` enabled in `EngineConfig`. Foundation for prefix sharing (§2.2) and continuous batching (§1.2). Initial overhead vs eager: ~13% throughput, ~19% E2E latency at B=1 (Llama-3.2-1B).

---

## 10-05-2026 — §2.4 Native eager KV cache (plain tensors, no `DynamicCache`)

- **PRs.** [#11](https://github.com/ValeGian/liteinfer/pull/11)
- **What.** `NativeEagerKVCache` stores per-layer `(K, V)` as plain tensors; duck-types `DynamicCache.update()` so attention layers consume it transparently. New `cache_mode="native_eager"`. Perf matches `EagerKVCache` (1.12× vs RECOMPUTE, Llama-3.2-1B B=1).

---

## 09-05-2026 — §1.1 Static batching with B > 1

- **PRs.** [#10](https://github.com/ValeGian/liteinfer/pull/10)
- **What.** Raises batch size from B=1 to `max_num_seqs`. Variable-length prompts left-padded; additive attention mask (`attention_mask.py`) threaded through `forward`. Strict-static: batch enters and exits together, early finishers wait. Llama (single tensor) and Gemma4 (full/sliding dict) both supported.

---

## 07-05-2026 — §4.1 e2e parity test vs `transformers`

- **PRs.** [#3](https://github.com/ValeGian/liteinfer/pull/3)
- **What.** `tests/e2e/test_llama_gpu.py` validates bit-equivalent greedy output vs `AutoModelForCausalLM.generate` on Llama-3.2-1B-Instruct (3 prompts, 20 tokens). Requires CUDA.

---

## 07-05-2026 — v0: minimal end-to-end inference

- **PRs.** [#3](https://github.com/ValeGian/liteinfer/pull/3)
- **What.** First working inference path: local safetensors loading, Llama + Gemma4 dispatch, KV cache (eager / none), greedy sampler (temperature / top-k / top-p), static batch scheduler (B=1), per-step metrics, `device="auto"`. 47 unit tests, lint + pyright clean.
