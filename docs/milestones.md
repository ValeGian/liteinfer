# Milestones

Achieved milestones, newest first. When a roadmap item lands: flip its `Status` to `landed` in `roadmap.md`, then add an entry here.

---

## 29-08-2026 — Vectorised block-table gather (precursor to §2.3)

- **PRs.** [#16](https://github.com/ValeGian/liteinfer/pull/16)
- **What.** `BlockPool` stores a flat run of token slots, so a token's address is one integer (`block_idx * block_size + offset`), and block 0 is a null block that absorbs the reads and writes padded batch positions generate. A shared `slot_table()` maps logical positions to physical slots once per decode step, and `PagedKVCache` and `ContinuousKVCache` both read and write a whole batch with a single indexing op. Each previously carried its own Python loop over sequences and blocks, run per layer per step — roughly 8k tensor ops and ~200 MB copied per step at B=32 — which left paged at 0.72x of native-eager and slower per decode step than running with no cache at all, and continuous batching slower than synchronous static batching. Neither followed from the design. Measured on A40 / Llama-3.2-1B-Instruct at ISL 128 / OSL 256: paged 52.7 → 66.8 tok/s (0.92x of native-eager), paged B=4 116.7 → 252.9 (0.90x of eager-b4), continuous 180.5 → 1,268.3 (7.0x). The deficit against vLLM stops widening with concurrency — 0.28-0.35x at every batch width, which is the flat per-step constant §3.1-3.3 addresses — and what paging still costs is the copy itself, which is §2.3. Greedy output stays identical to `native_eager` and `eager` at B=1 and B=4 and for the continuous engine, including variable-length prompts. The duplication is why this needed fixing twice: vectorising only `PagedKVCache` made continuous batching 18% slower, because the layout change left `ContinuousKVCache` reading transposed non-contiguous views.

---

## 29-08-2026 — §8.1 ISL / OSL-controlled workloads

- **PRs.** [#15](https://github.com/ValeGian/liteinfer/pull/15)
- **What.** Replaced the per-engine runner classes with a data-driven matrix: `benchmarks/configs.py` holds every configuration and names the one it improves on, so `bench report` scores each change against its baseline, cumulatively against its lineage root, and against vLLM at the same batch width. Both engines expose one primitive — prompts in, output lengths out — so they are timed by the same clock. `throughput` and `latency` report disjoint metric sets, because a per-request latency measured under saturation records queue position rather than the engine; ITL is derived from `(e2e - ttft) / (osl - 1)` and TTFT from a separate single-token pass, avoiding per-token hooks each engine would implement differently. Forced output length (`min_tokens = max_tokens = OSL`, `ignore_eos`) is verified every run and fails the run when violated, warmup exercises the measured path at the real ISL and batch width, and each (config, mode) runs in a fresh process. TensorRT-LLM dropped; vLLM 0.28.0 runs in-process, deleting the subprocess/venv/IPC layer, the promote/demote pinning and `main.json`. Fixed `AsyncLLMEngine`, which carried its own copy of the stop rule and ignored `ignore_eos` and `min_tokens`, silently truncating every continuous-batching run; both engines now share `engine/stopping.py`. Throughput is certified against `vllm bench throughput` to within 1% at the same shape, and latency runs sequentially because vLLM's TTFT is IPC-bound and read 24% high inside a parallel sweep.

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
