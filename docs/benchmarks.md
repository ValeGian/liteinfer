# Benchmark Results

Comparison of liteinfer variants against vLLM as a reference engine, with and without static batching (B=4).

> Last updated: 2026-05-12 · tag: `v0.1.5` · [interactive dashboard](https://htmlpreview.github.io/?https://github.com/ValeGian/liteinfer/blob/master/docs/dashboard.html)

---

## Setup

| | |
|---|---|
| **Model** | `meta-llama/Llama-3.2-1B-Instruct` (bfloat16) |
| **GPU** | NVIDIA A40 (46 GB VRAM) |
| **vLLM** | 0.20.0 |
| **PyTorch** | 2.11.0+cu130 |
| **Decoding** | Greedy (temperature=0) |

**Engines under test:**

| Key | B | Description |
|---|---:|---|
| `liteinfer` | 1 | liteinfer, no KV cache (`cache_mode="none"`); every step re-feeds the full and growing sequence (RECOMPUTE). Baseline. |
| `liteinfer-kvcache` | 1 | liteinfer, eager KV cache (`cache_mode="eager"`); prefill populates a `DynamicCache`, decode steps pass only the new token |
| `liteinfer-native-kvcache` | 1 | liteinfer, native eager KV cache (`cache_mode="native_eager"`); plain per-layer tensor store, no `DynamicCache` wrapper |
| `liteinfer-paged-kvcache` | 1 | liteinfer, paged KV cache (`cache_mode="paged"`); tokens stored in fixed-size blocks drawn from a pre-allocated pool |
| `liteinfer-b4` | 4 | liteinfer, eager KV cache, `max_num_seqs=4`; static batching — left-padded prefill with pad-aware additive attention mask |
| `liteinfer-native-kvcache-b4` | 4 | liteinfer, native eager KV cache, `max_num_seqs=4`; same static-batching policy as `liteinfer-b4`, no `DynamicCache` wrapper |
| `liteinfer-paged-b4` | 4 | liteinfer, paged KV cache, `max_num_seqs=4`; static batching with block-allocated memory |
| `liteinfer-continuous` | 4 | liteinfer, async continuous batching (`AsyncLLM`, `cache_mode="paged"`, `max_num_seqs=4`); sequences admitted every step, evicted individually on completion |
| `vllm` | 1 | vLLM 0.20.0, `max_num_seqs=1`, FlashAttention 2, CUDA graphs, paged KV cache |
| `vllm-b4` | 4 | vLLM 0.20.0, `max_num_seqs=4`; same kernels as `vllm`, four sequences in flight |
| `vllm-continuous` | 4 | vLLM 0.20.0, `max_num_seqs=4`, full continuous batching with FlashAttention 2, CUDA graphs, paged KV cache, prefix caching |

**TTFT measurement:** wall-clock time from request submission to when the first token is ready (measured via `time.perf_counter()` at step-listener fire time — includes Python/scheduling overhead, CUDA sync, forward pass, sampling, and token application).

---

## Throughput workload

32 short independent prompts · **all submitted at once** · greedy · max 64 tokens · 1 warmup

All 32 requests are queued at t₀; the engine processes them up to `max_num_seqs` at a time. E2E is measured from t₀ per request — it includes queue wait time, so later requests in a queue have higher E2E than earlier ones.

| Engine | B | req/s | tok/s | E2E p50 | E2E p99 |
|---|---:|---:|---:|---:|---:|
| liteinfer | 1 | 1.41 | 58 | 14832 ms | 22763 ms |
| liteinfer-b4 | 4 | 3.71 | 152 | 4860 ms | 8624 ms |
| liteinfer-native-kvcache-b4 | 4 | 3.52 | 144 | 5274 ms | 9094 ms |
| liteinfer-paged-b4 | 4 | 2.31 | 95 | 8230 ms | 13872 ms |
| liteinfer-continuous | 4 | 3.14 | 125 | 5756 ms | 10206 ms |
| vllm | 1 | 2.88 | 179 | 5822 ms | 11098 ms |
| vllm-b4 | 4 | 10.60 | 645 | 1683 ms | 2974 ms |
| vllm-continuous | 4 | 10.72 | 653 | 1636 ms | 2947 ms |

**Static B=4 comparison:** `liteinfer-b4` (eager, 3.71 req/s) is the fastest liteinfer variant at B=4. `liteinfer-native-kvcache-b4` is within 5% (3.52 req/s). `liteinfer-paged-b4` is ~38% slower (2.31 req/s) — block-table scatter/gather overhead on every decode step with no fused kernel.

**Continuous batching (liteinfer-continuous):** 3.14 req/s — between `liteinfer-paged-b4` (2.31) and `liteinfer-b4` (3.71). Does not yet outperform static at matched B=4: the two-pass step (separate prefill + decode forward calls when new sequences join) adds overhead that cancels the slot-filling benefit. Roadmap §1.3 (chunked prefill) will merge them.

**vLLM B=1 vs liteinfer B=4:** `vllm` at B=1 reaches 2.88 req/s — lower than all liteinfer B=4 variants. vLLM's higher tok/s (179 vs 152) reflects FlashAttention 2 and CUDA graphs on individual sequences; liteinfer's advantage at B=4 comes from static batching amortising Python/launch overhead across sequences.

**vllm-continuous vs vllm-b4:** +1% gain (10.72 vs 10.60 req/s) — vLLM's chunked prefill already mixes prefill and decode in one pass, so the scheduling policy difference is minimal at B=4.

**liteinfer vs vLLM gap (B=4):** `liteinfer-b4` is **2.86×** slower than `vllm-b4` (3.71 vs 10.60 req/s). Gap: eager attention, no CUDA graphs, no FlashAttention (§3.1–§3.3).

---

## Latency workload

20 calls of the same prompt · **sequential, no queue** · greedy · max 128 tokens · 1 warmup

Each request is submitted only after the previous has finished — no queue contamination.

| Engine | B | TTFT p50 | TTFT p99 | E2E p50 | tok/s |
|---|---:|---:|---:|---:|---:|
| liteinfer | 1 | 16.1 ms | 17.0 ms | 1948 ms | 64 |
| liteinfer-kvcache | 1 | 18.8 ms | 29.6 ms | 1936 ms | 54 |
| liteinfer-native-kvcache | 1 | 18.4 ms | 26.6 ms | 1974 ms | 56 |
| liteinfer-paged-kvcache | 1 | 18.1 ms | 27.6 ms | 2104 ms | 51 |
| vllm | 1 | 25.9 ms | 29.4 ms | 692 ms | 183 |

**KV cache vs no-cache:** KV cache gives marginal E2E improvement at B=1 (~1%: 1936 ms vs 1948 ms). Modest because liteinfer uses eager PyTorch attention without CUDA graphs — each decode step still pays full Python/GPU launch overhead.

**Native KV cache:** matches eager cache on E2E (1974 ms vs 1936 ms). No `DynamicCache` wrapper.

**Paged KV cache latency:** slightly higher E2E (2104 ms vs 1936 ms) — block-table scatter/gather on each decode step.

**TTFT:** liteinfer variants show lower TTFT than vLLM for this short prompt. vLLM's multi-process architecture (IPC to EngineCore subprocess) adds fixed overhead regardless of prompt length.

**E2E gap vs vLLM (~2.8×):** decode-bound. vLLM uses CUDA graphs and FlashAttention 2; liteinfer uses standard eager attention (§3.1–§3.3).

---

## What these numbers mean

| Gap | Root cause | Roadmap item |
|---|---|---|
| 2.86× slower throughput at B=4 (`liteinfer-b4` vs `vllm-b4`) | Eager attention, no CUDA graphs, no FlashAttention | §3.1 `torch.compile`, §3.2 CUDA graphs, §3.3 FlashAttention |
| `liteinfer-paged-b4` 38% slower than `liteinfer-b4` at B=4 | Block-table scatter/gather on each decode step; no fused kernel | §2.3 fused paged-attention kernel, §3.3 FlashAttention |
| `liteinfer-continuous` ≈ `liteinfer-paged-b4` at matched B | Two-pass step overhead when new seqs join decode batch | §1.3 chunked prefill (single-pass mixed batching) |
| ~2.8× slower decode tok/s at B=1 vs vLLM | No CUDA graphs, no FlashAttention | §3.2 CUDA graphs, §3.3 FlashAttention |
| No prefix-cache benefit yet | Prefix caching not implemented | §2.2 prefix sharing |

See [`docs/roadmap.md`](roadmap.md) for the full backlog.
