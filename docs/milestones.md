# Milestones

Achieved milestones, newest first. When a roadmap item lands: flip its `Status` to `landed` in `roadmap.md`, then add an entry here.

---

## 06-09-2026 — §3.2 The decode forward is replayed, not launched

- **PRs.** [#35](https://github.com/ValeGian/liteinfer/pull/35)
- **What.** A decode step issued about 700 kernels to do 7 ms of GPU work inside a 13 ms step. `EngineConfig.enable_cuda_graphs` now captures that forward once per batch width and replays it, which submits the whole sequence in one go instead of paying host dispatch per kernel. Decode step **13.4 → 6.6 ms**, throughput **1,930 → 3,112 tok/s**.
- **Measured, against the same engine with capture pinned off.**

  | throughput | before | after | |
  |---|---:|---:|---:|
  | 128 / 256 | 1,930.0 | **3,111.7** | **1.61x** |
  | 128 / 1024 | 1,921.9 | 2,959.1 | 1.54x |
  | 1024 / 1024 | 1,783.6 | 2,356.2 | 1.32x |
  | 1024 / 256 | 1,542.0 | 1,990.8 | 1.29x |
  | 2048 / 128 | 820.1 | 908.2 | 1.11x |

  Latency at B=1: ITL p50 13.4 → **6.6 ms** (2.03x), e2e 3,431.3 → **1,698.3 ms**, and 13.8 → 7.9 ms at 3,584 tokens of context. The gap to vLLM closes from 0.43x to **0.70x** on throughput and from 2.58x to **1.27x** on the step. Largest single change the project has measured, and general: every batch width, every context, both modes.
- **The instrumentation named it before any code was written.** `TimeBreakdown` had recorded the loop's stages since §6.4 and nothing had ever read it: the forward is **93%** of the loop, so the five PRs that went after sampling, detokenisation and per-step transfers had finished that job. Inside the forward, profiling showed the GPU **41-52% idle at every batch width** across 693-725 launches at a flat **18.7 us of wall per launch** — 32x the tokens buying 1.29x the GPU time, which is why throughput scaled with batch while the step never moved.
- **Then the ceiling was priced before the build.** Capturing the existing forward and timing replay gave 1.76x / 1.68x / 1.50x at B=1 / 8 / 32, and replaying against static buffers refilled from live engine state was bit-identical to eager over six steps. Both numbers were on record before a line of `cuda_graphs.py` existed, which is the rule §2.7 paid to learn.
- **Why it measured 1.06x the first time it was tried (#29) and 1.61x now.** That attempt predates §2.3. A graph fixes every shape, so the KV length has to be bucketed — and back then padding the KV length padded a *gather*, real bytes per layer per step. §2.3 deleted the gather, and the padding became a few masked loads inside one kernel.
- **No KV buckets at all, in the end — the roadmap's own scope was heavier than needed.** A graph freezes scalars and pointers, not what a kernel reads out of device memory, and `paged_decode` bounds its loop by `context_lens`, which is a tensor. So one capture serves every context length and the slot table is a single fixed `[max_num_seqs, max_model_len]` buffer whose unread columns are never touched — no bucketing, no zeroing, and right-alignment composing is what makes a wide table hold a short sequence correctly.
- **One graph per exact batch width, so there are no padded rows either.** vLLM pads each batch up to a ladder of captured sizes because its batch width is a token count that can be anything in the thousands; liteinfer's decode batch is bounded by `max_num_seqs`, so capturing per width is affordable and wastes nothing. Lazily, too: a run visits the widths its own scheduling produces, and a batch fills before it drains, so the largest width is captured first — which is the ordering vLLM arranges deliberately at startup. `_MAX_CAPTURES` bounds the total at 64 for an engine configured far wider than the default.
- **What it cost.** TTFT does not move: prefill is not captured, its shapes follow the prompt, and there is one prefill per request against hundreds of decodes. ISL 2048 / OSL 128 is the weakest row at 1.11x — the shape with the most prefill and the fewest decode steps to spread a fixed saving over. The first step at each width pays three warm-up forwards and a capture. Memory went the right way: a draining batch captured all 32 widths and **reserved memory fell 2.1 GiB**, the eager path's transients having been larger than the graph pool that replaced them.
- **The component estimate read low, which is new here.** 1.76x predicted at B=1 against 2.03x measured; every previous disagreement in `docs/benchmarks.md` went the other way by 30-60%. The ceiling was measured at 765 tokens of context and the benchmark's is 128-384, so there was less GPU work under the same fixed overhead. An overhead-removal estimate scales with how little work the step contains — the mirror image of §2.3's rule about removing a copy.
- **What it leaves.** ITL is now **1.86x** the memory roofline (3.55 ms) against vLLM's 1.47x, so the remainder is arithmetic rather than waiting: ~45 kernels per layer where a Llama layer needs ten. That is §3.1, now sizeable against a step where a launch is nearly free. It also unblocks §2.7, whose split-K was reverted only for adding a launch per layer — with one caveat now recorded: `num_splits` is a scalar a graph bakes in, so it has to be pinned per capture rather than chosen per step.

---

## 06-09-2026 — §2.3 Decode reads the KV pool where it lies

- **PRs.** [#33](https://github.com/ValeGian/liteinfer/pull/33)
- **What.** Decode copied every sequence's whole KV history out of the block pool into a contiguous `[B, heads, max_total, head_dim]` tensor — per layer, per step — and then ran a general attention kernel over it. `attn_implementation="paged"` replaces that with a Triton kernel that takes the slot table and reads the pool in place. The copy does not happen, the grouped-query heads are never expanded, and the padding is never touched, because the kernel takes each sequence's context length instead of a mask over the batch's longest.
- **Measured, both kernels in one session, same datasets.**

  | shape | `sdpa` | `paged` | |
  |---|---:|---:|---:|
  | 128 / 256 | 1,744.8 | 1,930.0 | 1.11x |
  | 128 / 1024 | 1,205.9 | 1,921.9 | 1.59x |
  | 1024 / 256 | 781.0 | 1,542.0 | 1.97x |
  | 1024 / 1024 | 688.9 | 1,783.6 | **2.59x** |
  | 2048 / 128 | 398.3 | 820.1 | 2.06x |

  The headline shape is the weakest row on this curve, not a general figure — the same mistake the KV cache's 1.21x was. What differs across rows is how much history there is to copy, and ISL 128 / OSL 256 is the shape with the least of it.
- **The decode step stops growing with context.** Timing `ContinuousModelRunner.decode` directly at B=32: 17.14 → **13.28 ms** at 188 tokens of context, and 21.70 → **13.05 ms** at 1,028. The paged step is flat; the gathering step is not, because its bytes are proportional to the history. That is the whole mechanism, and it is why the win tracks context length rather than batch width.
- **The design question the roadmap asked, and the answer.** `_DecodePayload.update` scattered *and* gathered, returning `(k, v)` — the one contract a paged kernel cannot use. Rather than put attention inside the cache or the cache inside the model, `update` now returns a **`DenseKV` or a `PagedKV`**: tensors, or the pool plus the addresses. `LlamaAttention` gained one line and lost none; the kernel it calls is picked by config as before, and which of the two shapes it gets is decided by the payload the runner built. Decode under `paged` builds no attention mask at all, because there is no padding left to hide.
- **The component number predicted this one, and the four before it did not.** 16 layers x (0.294 - 0.038) ms of measured attention saving is 4.09 ms; the step moved 3.86 ms, 94% of it. Three of those four read 30-60% high and the fourth disagreed about the sign. The difference is what is being removed: those removed *overhead* that partly overlapped with real work, so the isolated figure double-counted it; this removes a memory copy that was serialised with everything else, and a copy that does not happen is worth exactly what it cost. The rule survives, better stated — the discount belongs to things that were never entirely on the critical path.
- **It joins rather than replaces, by the roadmap's own test — and then the engine chooses.** The kernel needs CUDA and a Triton install, and `sdpa` serves everything else, so both paths stay. That would normally leave the fastest path behind a flag nobody sets, so `EngineConfig.attn_implementation` now defaults to **`None`, meaning "the fastest kernel that runs here"**: `select_implementation` resolves it once in `load_hf_model`, which is where the model is told which kernel to use, and records it on `hf_config._attn_implementation` for both the layers and the runner to read back. On CUDA with Triton installed that is `paged`; everywhere else it is `sdpa`. Naming a kernel that cannot run **raises rather than downgrading**, because a benchmark row or a parity test that asks for a kernel has to get it or hear why not — and each benchmark row pins its kernel, so the matrix is unaffected by the choice.
- **What it unblocks.** §3.2 (CUDA graphs) measured 1.06x because a graph fixes every shape and the padding therefore padded a *gather* — real bytes per layer per step. There is no gather now; what a capture pads is how many slots a kernel skips. §3.5 is superseded on the decode path: `_repeat_kv`'s 0.74 GiB of writes per step are gone, and what is left of the item is prefill. §3.1 is only half unblocked, and the honest half is the small one — inductor's whole-pool copy came from the in-place pool *write*, which every layer still does.
- **The 8B parity test caught it, and what it caught was one ULP.** That test compared liteinfer against `transformers` in bf16 without pinning the attention kernel, so when the engine started choosing `paged` it silently began measuring the new kernel against an eager reference — and one prompt of three diverged at token 11. Both kernels rank the same three candidates there; `sdpa` separates the top two by 0.125, which between 16 and 32 is **one bf16 ULP**, and `paged` rounds both onto 20.125. The argmax tie-break then takes the lower token id and twenty tokens of text change. The kernels are token-identical in fp32 and agree to 2⁻⁶ in bf16, so nothing is wrong: greedy decoding is a discontinuous readout of a continuous quantity, and at a tie it amplifies the last bit of the last accumulation. The fixture now pins `eager` on both sides, which is what the 1B file already documents and does; `docs/benchmarks.md` carries the logits.
- **Any head dimension, after a follow-up in the same PR.** The kernel first shipped requiring a power-of-two `head_dim`, because `tl.arange` needs a power-of-two length and the head axis is one of those lengths. That was the implementation's limit rather than the algorithm's, and it was already inconsistent: the *query* axis was padded to a legal tile and masked, since `tl.dot` wants 16 rows and Llama has 4 query heads per KV head. Applying the same treatment to the head axis retires the precondition — 96 (Phi-3-mini) and 80 (Phi-2) are the shapes a second architecture would have brought — and it removes `head_dim` from the kernel-selection inputs entirely, leaving two preconditions that are both about where code can execute rather than what it computes. Measured: the extra masking costs about 5 microseconds per launch, flat across batch widths and therefore launch overhead rather than kernel work; per 16-layer decode step that is 0.08 ms, and the engine cannot see it (13.28 ms against 13.53 ms, either side of the published figure and inside the ±4% floor). A `constexpr` guard to compile the mask away where it is unnecessary recovered nothing measurable and was dropped rather than kept as unpaid-for complexity.
- **At B=1 it buys nothing.** Latency mode, 200 requests one at a time: ITL p50 13.9 → **13.4 ms**, E2E 3,546.6 → **3,431.3 ms**, TTFT 14.0 → 14.5 ms. Every one of those is inside the ±4% floor, so the result is *no effect*, and it is the same result as the 1.12x row read differently. A single request has ~190 tokens of history to gather, which is the least this change can save, and the kernel's grid is one program per (sequence, KV head) — 8 programs on an 84-SM GPU. Per call it is 2.27x at B=1 against 7.73x at B=32. Nothing regressed; the single-request case is simply where this design is weakest, and flash-decoding's split-K is filed as §2.7.
- **The stored baselines were refreshed, and they had to be.** `liteinfer-sdpa` and `liteinfer-continuous` had throughput rows at ISL ≥ 1024 measured before PRs #25-#31, so a delta against them would have credited this change with five other PRs' work — `bench report` printed 3.09x at 1024/1024 where the same-session answer is 2.59x. Both configs were re-run across every stored shape in one session alongside the new one, which is the only way `vs base` compares engines of the same age. Refreshing `sdpa` alone would have been worse than leaving it be: `liteinfer-continuous` gained ~9% at that shape too, so the sdpa-vs-continuous delta would have moved from 1.03x to 1.20x and credited the SDPA kernel with work done by the five PRs after it. Refreshing both surfaced a claim that had been wrong for five milestones: **`liteinfer-sdpa` is 1.06x over `liteinfer-continuous` at ISL 128 / OSL 256**, not the 1.37x the report printed. The two configs differ only by the attention kernel; the rest of that 1.37x was the four PRs measured between them, and the stale baseline had been crediting it to SDPA. `docs/benchmarks.md` now carries both the corrected delta and the historical sequence it was confused with.
- **`docs/index.html` was 16 PRs stale**, because it is a generated file that is committed so GitHub Pages can serve it, and regenerating it was a manual step no checklist enforced. `tests/unit/test_dashboard_is_current.py` now re-renders it from the committed results and fails if the two differ, naming the command that fixes it. The check needs no GPU: rendering the report is a pure transform of data already in the tree.

---

## 06-09-2026 — §5.5 Backpressure on admission

- **PRs.** [#31](https://github.com/ValeGian/liteinfer/pull/31)
- **What.** `max_num_seqs` caps what runs; nothing capped what was accepted. `_pending` and `scheduler.waiting` were both unbounded, so a caller submitting faster than the engine drains grew two Python lists until the process died — holding admitted work it never ran. `EngineConfig.max_waiting_seqs` (default 1024) now bounds them, and `generate_stream` raises `EngineOverloaded` past it.
- **Why raising rather than blocking.** A bounded queue that makes submission wait turns an overload into a hang, which is harder to see than a refusal and impossible to tell from a slow engine. `EngineOverloaded` is its own type so a caller can distinguish "come back later" from "this request is malformed" — the difference between retrying and giving up.
- **The batch API paces itself.** `generate` owns the whole prompt list and submits it with `asyncio.gather`, so a naive cap would have made `llm.generate(2000_prompts)` fail where it used to work. It now keeps at most `max_waiting_seqs` requests outstanding, so a list of any length still runs and the caller that can see the whole list never trips the cap. Measured at 1,744.8 tok/s against a 1,732.8 baseline — no cost.
- **What the test had to work around.** The obvious test — submit past the cap, expect a raise — passed for the wrong reason at first: awaiting a stream's first event waits for a *token*, by which point the request has already left the queue for the running set. It now submits more requests than can run or wait, with outputs long enough that none finishes while the rest arrive, and asserts one is refused.

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
