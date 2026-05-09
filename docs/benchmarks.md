# Benchmark Results

Comparison of liteinfer variants against vLLM as a reference engine, with and without static batching (B=4).

> Last updated: 2026-05-09 · tag: `static-batching-b4` · [interactive dashboard](https://htmlpreview.github.io/?https://github.com/ValeGian/liteinfer/blob/master/docs/dashboard.html)

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
| liteinfer | 1 | 1.81 | 75 | 11193 ms | 17662 ms |
| liteinfer-kvcache | 1 | 1.87 | 75 | 9865 ms | 17100 ms |
| liteinfer-b4 | 4 | 4.42 | 182 | 4096 ms | 7234 ms |
| vllm | 1 | 2.89 | 180 | 5785 ms | 11023 ms |
| vllm-b4 | 4 | 10.67 | 650 | 1656 ms | 2963 ms |

**Static batching impact (liteinfer):** B=4 gives **2.4×** throughput over B=1 eager-cache (4.42 vs 1.87 req/s) and reaches per-token throughput parity with `vllm` at B=1 (182 vs 180 tok/s). Median E2E drops from 9865 ms to 4096 ms because four prompts share each prefill+decode step instead of waiting in queue.

**vLLM headroom remaining:** `vllm-b4` is **2.4×** the throughput of `liteinfer-b4` (10.67 vs 4.42 req/s) at the same batch size — gap is now CUDA graphs, FlashAttention 2, and paged KV cache, not batching.

---

## Latency workload

20 calls of the same prompt · **sequential, no queue** · greedy · max 128 tokens · 1 warmup

Each request is submitted only after the previous has finished — no queue contamination. B>1 has no effect because at most one prompt is in flight; the B=4 rows are reported only for completeness.

| Engine | B | TTFT p50 | TTFT p99 | E2E p50 | tok/s |
|---|---:|---:|---:|---:|---:|
| liteinfer | 1 | 13 ms | 14 ms | 1654 ms | 75 |
| liteinfer-kvcache | 1 | 15 ms | 16 ms | 1490 ms | 75 |
| liteinfer-b4 | 4 | 15 ms | 15 ms | 1501 ms | 75 |
| vllm | 1 | 26 ms | 27 ms | 693 ms | 183 |
| vllm-b4 | 4 | 26 ms | 29 ms | 693 ms | 183 |

**TTFT:** liteinfer variants show lower TTFT than vLLM for this short prompt (7 tokens). In RECOMPUTE mode the "prefill" is a 7-token forward pass with no KV-cache setup. In KV-cache mode, prefill populates the cache but there is still no subprocess IPC. vLLM's higher TTFT reflects its multi-process architecture (IPC to EngineCore subprocess, chunked-prefill scheduler) regardless of prompt length.

**KV cache vs no cache:** KV cache reduces E2E by ~10% (1490 ms vs 1654 ms) and tightens TTFT p99 variance. The improvement is modest because liteinfer uses eager PyTorch attention without CUDA graphs — each decode step still pays full Python/GPU launch overhead.

**E2E gap vs vLLM (~2.2×):** comes from decode. vLLM uses CUDA graphs (near-zero Python overhead per decode step) and FlashAttention 2, while liteinfer uses standard eager attention without any fusion or graph capture.

---

## What these numbers mean

| Gap | Root cause | Roadmap item |
|---|---|---|
| 2.4× slower decode tok/s at B=4 (`liteinfer-b4` vs `vllm-b4`) | Eager attention, no CUDA graphs, no FlashAttention | §3.1 `torch.compile`, §3.2 CUDA graphs, §3.3 FlashAttention |
| Modest KV cache gain (~10%) at B=1 | KV cache without CUDA graphs still pays per-step Python overhead | §3.2 CUDA graphs for decode |
| No prefix-cache benefit on `prefix_share` workload yet | Prefix caching not implemented | §2.1 paged KV, §2.2 prefix sharing |
| Continuous batching not yet supported | Static batches drain to completion before refilling | §1.2 continuous batching |

See [`docs/roadmap.md`](roadmap.md) for the full backlog.
