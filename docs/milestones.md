# Milestones

Achieved milestones, newest first. When a roadmap item lands: flip its `Status` to `landed` in `roadmap.md`, then add an entry here.

---

## 06-09-2026 — The measured record, refreshed

- **PRs.** [#30](https://github.com/ValeGian/liteinfer/pull/30)
- **What.** `liteinfer-sdpa` had no latency results at all: every row in the latency table was dated before the five changes that followed, and the README quoted `liteinfer-eager` — a config **deleted in #17** — as the engine's decode step, alongside a throughput figure from before #22. The whole table described an engine that no longer exists. Latency is now measured on the shipping engine and both documents carry it.
- **Measured.** ITL p50 14.9 → **13.9 ms**, TTFT p50 19.0 → **14.0 ms**, e2e p50 3,807.6 → **3,546.6 ms**. Against vLLM the gap is **2.6x on throughput and 2.7x on the decode step**, not the 3.5x the README claimed.
- **The two numbers disagree, and both are right.** Throughput moved 1.37x over this sequence while the decode step moved 1.07x. Three of the five changes — incremental detokenisation, vectorised sampling, one transfer per step — cost work *per sequence in the batch*, so removing them shows up in aggregate throughput and barely in a single request's step time. Latency mode runs one request at a time and structurally cannot see them. Reporting either number alone would have told half the story.
- **Why it went stale.** Every PR in the sequence re-ran `throughput`, because that is the mode the roadmap's table says a scheduling or memory change belongs in. None re-ran `latency`, and nothing checked whether the published numbers still referred to code that existed.

---

## 05-09-2026 — No host-side work inside the forward pass

- **PRs.** [#29](https://github.com/ValeGian/liteinfer/pull/29)
- **What.** The decode payload used to allocate blocks and build the slot table on layer 0 — Python running in the middle of a forward pass, with the GPU waiting on it. `decode()` now calls `cache.advance()` and `cache.slot_table_for()` before the model, and the payload is handed a slot tensor it only reads. Measured at 1.00x: the work moved rather than went away. It ships because the shape is right — a forward pass that decides nothing on the host is what CUDA graph capture requires, and what a fused paged-attention kernel (§2.3) wants too.
- **The rest of the attempt did not ship.** This was the prerequisite for §3.2, which was then built, measured at **1.06x** at its best bucket size, and reverted. A graph fixes every shape, so the KV length must be rounded up to a bucket: 256-token buckets waste attention and gather work and measure 0.97x, 16-token buckets spend the gain on captures and measure 0.93x, and the 1.06x peak at 64 is smaller than the run-to-run variance the same configuration showed on consecutive runs (1,809 and 1,932 tok/s).
- **Why, and what unblocks it.** The cost is the gather: liteinfer copies the whole KV history into a contiguous tensor every step, so padding the KV length pads a copy — real bytes, per layer, per step. §2.3's fused kernel reads the block table directly and never gathers, leaving the padding nothing to inflate. That is the difference between liteinfer and an engine that captures graphs profitably, and it makes §3.2 depend on §2.3.
- **Two large items in a row that measurement killed** — §3.5 at 0.64x, §3.2 at 1.06x. Both had a plausible mechanism and a promising component benchmark; neither survived the engine. The roadmap now carries the numbers and the dependency for each, so the next attempt starts from what was learned rather than from the same estimate.

---

## 05-09-2026 — One transfer per step, not one per sequence

- **PRs.** [#27](https://github.com/ValeGian/liteinfer/pull/27)
- **What.** Two per-step tensors were built by looping over sequences in Python, each iteration making its own `torch.tensor(..., device=cuda)` — a separate **pageable** host-to-device copy, which blocks. `slot_table` did it once per sequence to assemble the block tables, and `build_continuous_decode_mask` did a slice assignment per sequence to write its pad prefix. Both now pad on the host and move the batch in a single transfer. `slot_table` also read `int(count.max())` off the device to learn a width the Python counts already knew, which synced the whole queue every step; it uses `max(counts)`.
- **Measured.** Host-to-device copies per decode step fall from **35 to 5**, kernels from 883 to 822, and the step from 14.75 to **13.96 ms** — GPU idle 3.67 → **2.93 ms**. In the harness, 1,614.7 → **1,732.8** tok/s at ISL 128 / OSL 256 and 1,173.0 → **1,230.4** at OSL 1024, so 1.07x and 1.05x.
- **How it was found.** Not from the roadmap — from counting the 883 kernel launches after §7, looking for what §3.2 would have to capture. The 35 pageable memcpys stood out because their *count* tracked the batch size, which is the signature of a Python loop over sequences.
- **The isolated numbers over-predicted again.** Benchmarked on their own the two functions saved 1.021 + 0.290 = 1.31 ms; in the engine the step moved 0.79 ms, about 60% of that. The isolated figures include synchronisation that partly overlapped with real work. This is the third time in a row that a microbenchmark has read high — §3.5's 10x kernel win predicted 1.15x end to end, §7's predicted 1.12x and gave 1.07x — and the pattern is worth keeping: **measure the component to find the target, measure the engine to size it.**
- **§3.2 re-sized.** Its case rested on 883 launches and a 25% idle step. Both were partly this: it is now 822 launches and 21%, which is what CUDA graphs actually stand to recover. The roadmap entry carries the corrected numbers rather than the ones that made the item look biggest.

---

## 05-09-2026 — Vectorised sampling

- **PRs.** [#26](https://github.com/ValeGian/liteinfer/pull/26)
- **What.** `Sampler.__call__` ran a Python loop over rows, taking one `argmax` per sequence, and `_apply_sampled` then pulled each token back with its own `.item()` — 32 separate device-to-host copies per step. Greedy rows do not depend on their parameters, so they are now taken together in one kernel, and the batch comes back in one `.tolist()`. Stochastic rows still need their own temperature, top-k, top-p and RNG stream, so they stay a loop.
- **Measured in isolation.** One batched `argmax` against the per-row loop: 1.271 → 0.040 ms, 31.5x. One `.tolist()` against 32 `.item()` calls: 0.372 → 0.014 ms, 27x.
- **Measured in the engine.** The `sample` stage (§6.4) falls from 10.7% of loop time to **3.1%** at OSL 256, and from 8.6% to **2.4%** at OSL 1024. In the harness, 1,509.7 → **1,614.7** tok/s at ISL 128 / OSL 256 and 1,106.4 → **1,173.0** at OSL 1024 — 1.07x and 1.06x.
- **The prediction was 1.12x and the answer was 1.07x.** The error was the denominator: 1.588 ms of avoidable work against a *14.68 ms decode step*, when sampling is not inside that step. Against loop time the sum is 7.6%, which predicts 1.08x — and that is what happened. A saving is only as large as the total it is measured against.
- **A guess that did not survive.** Ranking this item, I argued it would partly displace §3.2, on the theory that 32 per-row `.item()` syncs were stalling the pipeline and inflating the "25% GPU idle" figure. Re-profiled afterwards: the forward-only step is **unchanged** at 14.75 ms, 25% idle, 883 launches. Sampling happens outside the forward pass, so it was never part of that idle. §3.2 is worth its full size.

---

## 05-09-2026 — §5.2 Incremental detokenization

- **PRs.** [#25](https://github.com/ValeGian/liteinfer/pull/25)
- **What.** The engine decoded each sequence's *entire* output text on every step, so the work per token grew with the tokens already generated. It now decodes a short window ending at the new token, decodes the same window one token shorter, and keeps the difference. The text is advanced once per step and read by both the stop rule and the stream event, where before each rebuilt it independently.
- **Why not simply append the new token's text.** Two reasons, and both are in the tests. A UTF-8 character can span two tokens, so decoding one token alone yields U+FFFD; and a byte-level BPE piece renders differently depending on what precedes it. The window-and-difference approach handles both: when the window ends mid-character the decoder emits U+FFFD, which is not text yet, so the window is left to grow until the next token completes it.
- **Measured.** The `deliver` stage (§6.4) goes from 6.9% of loop time at OSL 256 and 16.9% at OSL 1024 to **1.0% at both** — flat with output length, which is the property that was missing. In the harness: 1,445.5 → 1,509.7 tok/s at ISL 128 / OSL 256, and 959.7 → **1,106.4** at OSL 1024.
- **Read those two rows together.** Removing a cost worth ~7% of the loop reads as 1.04x, which is inside the ±4% noise band and not a result on its own; removing one worth ~17% reads as 1.15x, which is. Same code in both rows — what differs is how much there was to remove. Measuring this at one short shape would have found nothing.
- **Cost.** None measured. `Sequence` carries an `IncrementalDetokenizer`; the stop rule reads `seq.output_text` rather than decoding, which also removes the second full decode it was doing whenever stop strings were set.

---

## 05-09-2026 — §6.4 The engine accounts for its own time

- **PRs.** [#24](https://github.com/ValeGian/liteinfer/pull/24)
- **What.** `EngineStats.time` is a `TimeBreakdown` charging every part of a step to a stage — `forward`, `sample`, `deliver`, `schedule` — with `unattributed` for what is left. The loop now accounts for 99% of its own wall time, so "where does it go" is a property you can read instead of a script someone has to write.
- **Why it was needed.** Every performance item on the roadmap optimises the forward pass. Nothing had checked whether the forward pass is where the time is.
- **What it found.** It is — but less so the longer the output runs, and the second-biggest item is not a kernel. Measured on 32 sequences, A40, Llama-3.2-1B:

  | stage | OSL 256 | OSL 1024 |
  |---|---:|---:|
  | forward | 84.9% | 76.6% |
  | **deliver** (detokenise) | **6.9%** | **16.9%** |
  | sample | 7.0% | 5.5% |
  | schedule | 0.3% | 0.2% |
  | unattributed (asyncio, queues) | 1.0% | 0.8% |

  `_build_event` re-decodes a sequence's whole output text every step, so four times the output length costs twelve times the detokenisation. That promotes §5.2 from a "fine for v0" nicety to the second-largest cost in the engine at OSL 1024, and it is pure Python with no kernel work involved. Scheduling and the asyncio round trip, which might plausibly have been suspects, are together under 1.5% and need no attention at all.
- **A measurement bug fixed on the way.** `StepMetrics.wall_time_s` was documented as one forward pass but timed the forward pass *plus* sampling and the stop check. Sampling now has its own stage, so the two are no longer conflated — which is what made the 5.5-7% sampling cost visible as a separate line.

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
