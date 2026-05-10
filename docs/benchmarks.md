# Benchmark Results

Comparison of liteinfer variants against vLLM as a reference engine, with and without static batching (B=4).

> Last updated: 2026-05-10 · tag: `paged-kv-cache` · [interactive dashboard](https://htmlpreview.github.io/?https://github.com/ValeGian/liteinfer/blob/master/docs/dashboard.html)

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
| `liteinfer` | 1 | liteinfer, no KV cache (`cache_mode="none"`); every step re-feeds the full and growing sequence (RECOMPUTE) |
| `liteinfer-kvcache` | 1 | liteinfer, eager KV cache (`cache_mode="eager"`); prefill populates a `DynamicCache`, decode steps pass only the new token |
| `liteinfer-native-kvcache` | 1 | liteinfer, native eager KV cache (`cache_mode="native_eager"`); plain per-layer tensor store, no `DynamicCache` wrapper (roadmap §2.4) |
| `liteinfer-paged-kvcache` | 1 | liteinfer, paged KV cache (`cache_mode="paged"`); tokens stored in fixed-size blocks drawn from a pre-allocated pool (roadmap §2.1) |
| `liteinfer-b4` | 4 | liteinfer, eager KV cache + `max_num_seqs=4`; static batching from roadmap §1.1 — left-padded prefill with pad-aware additive attention mask |
| `vllm` | 1 | vLLM 0.20.0, `max_num_seqs=1`, FlashAttention 2, CUDA graphs, paged KV cache |
| `vllm-b4` | 4 | vLLM 0.20.0, `max_num_seqs=4`; same kernels as `vllm`, four sequences in flight |

**TTFT measurement:** wall-clock time from request submission to when the first token is ready (measured via `time.perf_counter()` at step-listener fire time — includes Python/scheduling overhead, CUDA sync, forward pass, sampling, and token application).

---

## Throughput workload

32 short independent prompts · **all submitted at once** · greedy · max 64 tokens · 1 warmup

All 32 requests are queued at t₀; the engine processes them up to `max_num_seqs` at a time. E2E is measured from t₀ per request — it includes queue wait time, so later requests in a queue have higher E2E than earlier ones.

| Engine | B | req/s | tok/s | E2E p50 | E2E p99 |
|---|---:|---:|---:|---:|---:|
| liteinfer | 1 | 1.75 | 73 | 11607 ms | 18249 ms |
| liteinfer-kvcache | 1 | 1.81 | 72 | 10190 ms | 17648 ms |
| liteinfer-native-kvcache | 1 | 1.81 | 72 | 10233 ms | 17725 ms |
| liteinfer-paged-kvcache | 1 | 1.57 | 63 | 11772 ms | 20397 ms |
| liteinfer-b4 | 4 | 4.31 | 177 | 4220 ms | 7421 ms |
| vllm | 1 | 2.90 | 181 | 5796 ms | 10999 ms |
| vllm-b4 | 4 | 10.77 | 656 | 1625 ms | 2932 ms |

**Paged KV cache overhead:** `liteinfer-paged-kvcache` is ~13% slower than `liteinfer-kvcache` (1.57 vs 1.81 req/s). Block-table indirection on every decode step adds overhead not present in the direct-tensor eager path. This is the initial §2.1 implementation — the foundation for prefix sharing (§2.2) and continuous batching (§1.2).

**Native KV cache parity:** `liteinfer-native-kvcache` matches `liteinfer-kvcache` exactly (1.81 req/s, 72 tok/s) with no `DynamicCache` dependency. §2.4 landed with zero performance regression.

**Static batching impact (liteinfer):** B=4 gives **2.7×** throughput over B=1 eager-cache (4.31 vs 1.81 req/s) and reaches per-token throughput parity with `vllm` at B=1 (177 vs 181 tok/s). Median E2E drops from 10190 ms to 4220 ms because four prompts share each prefill+decode step instead of waiting in queue.

**vLLM headroom remaining:** `vllm-b4` is **2.5×** the throughput of `liteinfer-b4` (10.77 vs 4.31 req/s) at the same batch size — gap is now CUDA graphs, FlashAttention 2, and continuous batching, not paging.

---

## Latency workload

20 calls of the same prompt · **sequential, no queue** · greedy · max 128 tokens · 1 warmup

Each request is submitted only after the previous has finished — no queue contamination.

| Engine | B | TTFT p50 | TTFT p99 | E2E p50 | tok/s |
|---|---:|---:|---:|---:|---:|
| liteinfer | 1 | 13.7 ms | 15.4 ms | 1710 ms | 72 |
| liteinfer-kvcache | 1 | 14.6 ms | 16.8 ms | 1541 ms | 73 |
| liteinfer-native-kvcache | 1 | 13.9 ms | 15.6 ms | 1540 ms | 73 |
| liteinfer-paged-kvcache | 1 | 14.7 ms | 16.7 ms | 1829 ms | 61 |
| vllm | 1 | 25.9 ms | 28.8 ms | 693 ms | 182 |

**Paged KV cache latency:** `liteinfer-paged-kvcache` E2E is ~19% higher than `liteinfer-kvcache` (1829 ms vs 1541 ms) and tok/s drops from 73 to 61. Block-table scatter/gather on each decode step adds per-step overhead beyond what the eager path pays.

**TTFT:** liteinfer variants show lower TTFT than vLLM for this short prompt (7 tokens). In RECOMPUTE mode the "prefill" is a 7-token forward pass with no KV-cache setup. In KV-cache mode, prefill populates the cache but there is still no subprocess IPC. vLLM's higher TTFT reflects its multi-process architecture (IPC to EngineCore subprocess, chunked-prefill scheduler) regardless of prompt length.

**Native KV cache TTFT:** `liteinfer-native-kvcache` shows slightly lower TTFT p50 than `liteinfer-kvcache` (13.9 ms vs 14.6 ms) — no `DynamicCache` initialization overhead on prefill. E2E and tok/s are identical.

**KV cache vs no cache:** KV cache reduces E2E by ~10% (1541 ms vs 1710 ms) and tightens TTFT p99 variance. The improvement is modest because liteinfer uses eager PyTorch attention without CUDA graphs — each decode step still pays full Python/GPU launch overhead.

**E2E gap vs vLLM (~2.2×):** comes from decode. vLLM uses CUDA graphs (near-zero Python overhead per decode step) and FlashAttention 2, while liteinfer uses standard eager attention without any fusion or graph capture.

---

## What these numbers mean

| Gap | Root cause | Roadmap item |
|---|---|---|
| 2.5× slower decode tok/s at B=4 (`liteinfer-b4` vs `vllm-b4`) | Eager attention, no CUDA graphs, no FlashAttention | §3.1 `torch.compile`, §3.2 CUDA graphs, §3.3 FlashAttention |
| Modest KV cache gain (~10%) at B=1 | KV cache without CUDA graphs still pays per-step Python overhead | §3.2 CUDA graphs for decode |
| Paged KV ~13–19% slower than eager at B=1 | Block-table scatter/gather overhead; no fused paged-attention kernel | §3.3 FlashAttention (paged attn kernel), future §2.1 optimisation pass |
| No prefix-cache benefit yet | Prefix caching not implemented | §2.2 prefix sharing |
| Continuous batching not yet supported | Static batches drain to completion before refilling | §1.2 continuous batching |

See [`docs/roadmap.md`](roadmap.md) for the full backlog.
