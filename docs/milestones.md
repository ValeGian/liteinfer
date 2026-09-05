# Milestones

Achieved milestones, newest first. When a roadmap item lands: flip its `Status` to `landed` in `roadmap.md`, then add an entry here.

---

## 05-09-2026 — §3.3 Fused SDPA attention

- **PRs.** [#22](https://github.com/ValeGian/liteinfer/pull/22)
- **What.** Attention is now a named kernel picked by `EngineConfig.attn_implementation`, and the default `sdpa` hands the operation to PyTorch, which tiles the softmax and never assembles the `[batch, heads, queries, keys]` score matrix. That matrix was the largest allocation in the forward pass and grew with the square of sequence length. On one attention call at 4 seqs × 32 heads × 2048 tokens, peak allocation drops 5,192 MiB → 96 MiB.
- **What it is not.** A speedup. Re-measured against the eager kernel on the same tree: 1.08× at 128/256, 1.02× at 128/1024, 1.10× at 1024/256, 1.03× at 1024/1024 — four rows inside run-to-run variance. The feature is the fifth row. At ISL 2048 eager asks for 16.02 GiB in one allocation and dies; `sdpa` completes at 385.3 tok/s. That is the score matrix exactly (32 × 32 × 2048², fp32 after the softmax upcast), so what the kernel moves is the ceiling, not the clock. Credit where it belongs: the ISL 1024 shapes were unlocked by #20's pool sizing, not by this — ISL 2048 is the first shape only the fused kernel reaches.
- **Eager stays — as a stated exception, not because the rule says so.** Read literally, the replacement rule says delete it: `sdpa` serves every workload `eager` serves, plus one it cannot, and wins on memory with no regression elsewhere. Both questions answer yes. It stays anyway, for two reasons the rule does not weigh. It is the only attention *in* the repo — delete it and liteinfer no longer contains the algorithm, it contains a call to something that does it, in an engine whose stated purpose is being read end to end. And it is the independent reference the fused path is checked against, which catches our own mistakes — a wrong scale, a mis-shaped mask, a bad slot table — rather than PyTorch's. The cost is 22 lines, one e2e fixture and one benchmark row that keeps being run, so `liteinfer-continuous` stays runnable rather than `historical`. The exception holds only while it is exercised: a reference nobody runs is dead code with a story attached.
- **Which backend.** Not FlashAttention. Left-padded batches need an explicit additive mask and flash accepts only `is_causal` — PyTorch reports "Flash Attention does not support non-null attn_mask" — so SDPA takes the memory-efficient backend, which tiles the same way. Getting flash means dropping the padding, filed as §3.6.
- **Parity.** The transformers reference test now pins liteinfer to `eager`, matching the kernel `hf_model` runs, because that test isolates the *model* — RoPE, norms, projections, the cache — and a different kernel on one side turns it into a rounding test. So held, it is exact: transformers-eager and liteinfer-eager agree token-for-token on every parity prompt. Kernel equivalence is pinned separately, in fp32, where the two are token-identical on a single prompt and on a left-padded batch. In bf16 they agree to ~2 ULP on the rows the engine reads, which greedy decoding turns into a different token around position 11 — precision, not a difference in what is computed; liteinfer-eager still matches transformers exactly at that precision. They do differ on fully masked rows, where `sdpa` returns zeros and eager the uniform average of every value vector; a row attending to nothing has no defined answer, and the engine takes logits from the last column, which is never padding.
- **Also.** The loader's `_attn_implementation = "eager"` pin had been dead since the modeling code stopped going through transformers, and is gone. Two follow-ups the work exposed are filed: §3.5 (`enable_gqa` removes the 4× K/V copy the profile put at ~10% of decode time) and §3.6 (varlen packing, which unblocks flash and §1.3).

---

## 05-09-2026 — KV pool sized to demand

- **PRs.** [#20](https://github.com/ValeGian/liteinfer/pull/20)
- **What.** The block pool took 85% of *free* VRAM and never asked how much the engine could actually use. `max_num_seqs` × `max_model_len` is a hard ceiling on how much KV can ever exist — 4.00 GiB for the default config — but the pool allocated 32.74 GiB, leaving 5.77 GiB for activations, and then died trying to allocate a 4.02 GiB attention score matrix while holding ~29 GiB it could never reach. The pool is now `min(affordable, reachable)`, which frees that surplus and lets ISL 1024 run at all (646.0 and 574.1 tok/s, previously OOM). Sizing is also no longer silent or fixed: it logs the size it chose and why, warns when memory rather than the workload is the binding constraint — which means the configured concurrency may exhaust the pool under load — and the hardcoded 0.85 becomes `EngineConfig.kv_cache_memory_fraction`, applied identically on CPU and CUDA so the knob is testable without a GPU. `BlockPool.nbytes` reports the footprint. Pool sizing previously had no tests at all, which is how an 8x oversize went unnoticed; it now has ten.

---

## 05-09-2026 — §8.3 Sequence-length sweep

- **PRs.** [#19](https://github.com/ValeGian/liteinfer/pull/19)
- **What.** `bench sweep` runs the configs across a grid of (ISL, OSL) shapes, generating any missing datasets, and `bench report` grows an *across shapes* table once more than one shape has results — because a ratio measured at one shape is a claim about that shape only. The sweep earned itself twice over on its first run. It showed that the KV cache's advantage is strongly shape-dependent: measured on the pre-#17 tree, where `liteinfer-nocache` still exists, the cache is worth 1.15x at OSL 256, 1.61x at OSL 512 and 2.88x at OSL 1024 — cached throughput stays flat (69.5, 71.9, 69.6 tok/s) while the recompute path collapses (60.7, 44.6, 24.1). The 1.21x recorded at ISL 128 / OSL 256 was the weakest point on that curve, not a general figure. It also found that liteinfer could not run ISL 1024 at all: eager attention materialises a `[32, 32, 1024, 1024]` score matrix, which the softmax upcasts to fp32 — 4 GiB in a single allocation — and the run died where vLLM completed. Removing that materialisation is §3.3; the KV pool's share of the blame is fixed separately.

---

## 29-08-2026 — §5.4 Per-request failure isolation

- **PRs.** [#17](https://github.com/ValeGian/liteinfer/pull/17)
- **What.** Errors are now scoped to the request that caused them. Per-request queues carry `StreamEvent | Exception | None` and `generate_stream` re-raises, so a rejected prompt reaches its own caller; a forward pass that raises aborts only that pass's sequences; and a loop that dies for any other reason fails every outstanding waiter before exiting rather than leaving them parked. `generate_stream` also refuses to run before `start()` instead of hanging. Previously any exception in admission or execution killed the background loop silently: the caller waited forever on a queue nobody would post to, every subsequent request hung the same way, and the original error only surfaced later from `stop()`. An over-long prompt was enough to trigger it.

---

## 29-08-2026 — One engine: continuous batching only (+ §6.3 async step metrics)

- **PRs.** [#17](https://github.com/ValeGian/liteinfer/pull/17)
- **What.** Removed every superseded execution path now that the benchmark has measured each one against its predecessor. Gone: the synchronous `LLMEngine`, `Scheduler` and `ModelRunner` (static batching), the `KVCache` ABC with its no-cache, `DynamicCache` and plain-tensor implementations, the batch-level `PagedKVCache`, `cache_mode`, and `AsyncEngineConfig` — which had become identical to `EngineConfig`. What remains is one engine: `AsyncLLM` over continuous batching on a paged block cache, with `LLM` kept as a synchronous facade that owns a private event loop, so offline batch use is unchanged. The attention-mask builder loses its decode branch and `past_len` parameter, since continuous prefill always passes zero and continuous decode has its own builder. `StepMetrics` is now recorded per forward pass rather than per step, which is what makes the two-pass prefill+decode cost (§1.3) visible and closes §6.3; `collect_stats` finally means something, having never been read before. The benchmark collapses to one liteinfer adapter — the event-loop handling it used to carry now lives in `LLM` — and the removed configurations stay in `benchmarks/configs.py` flagged `historical` so the report keeps rendering the full progression, marked as no longer runnable. Package source drops from ~2,900 to ~2,000 lines.
- **Why these and not others.** Each removed path was general, not scoped to a case the survivor cannot serve: continuous batching serves every workload static batching did, so nothing was left without a home. On the numbers, `nocache` was strictly dominated (60.5 vs 73.4 tok/s, 16.5 vs 13.7 ms ITL); `eager` and `native-eager` measured identical, so one was redundant; static batching was surpassed 4.5x by continuous batching (281.6 → 1,268.3 tok/s). The one cost is ~9% single-request ITL, since eager's 13.7 ms beat continuous's 14.9 ms at B=1 — judged not worth a second engine and ~900 lines in a codebase meant to be read.
- **Verified.** Surviving flow re-measured at 1,368 tok/s, unchanged within run-to-run variance, so per-pass metrics cost nothing measurable. 177 non-e2e tests and 11 GPU e2e tests pass; the sync-pipeline suite was ported to the facade rather than deleted.

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
