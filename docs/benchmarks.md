# Benchmark Results

Comparison of liteinfer v0 variants against vLLM as a reference engine, all at **batch size 1**.

> Last updated: 2026-05-07 · tag: `v0_baseline` · [interactive dashboard](https://htmlpreview.github.io/?https://github.com/ValeGian/liteinfer/blob/master/docs/dashboard.html)

---

## Setup

| | |
|---|---|
| **Model** | `meta-llama/Llama-3.2-1B-Instruct` (bfloat16) |
| **GPU** | NVIDIA A40 (46 GB VRAM) |
| **liteinfer** | v0.0.6 |
| **vLLM** | 0.20.0 |
| **PyTorch** | 2.11.0+cu130 |
| **Batch size** | **1** (all engines) |
| **Decoding** | Greedy (temperature=0) |

**Engines under test:**

| Key | Description |
|---|---|
| `vllm` | vLLM 0.20.0, `max_num_seqs=1`, FlashAttention 2, CUDA graphs, KV cache |
| `liteinfer` | liteinfer v0, no KV cache (`cache_mode="none"`); every step re-feeds the full and growing sequence (RECOMPUTE) |
| `liteinfer-kvcache` | liteinfer v0, eager KV cache (`cache_mode="eager"`); prefill populates a `DynamicCache`, decode steps pass only the new token |

**Batch size 1 enforcement:**
- vLLM: `max_num_seqs=1` at construction.
- liteinfer: hard-capped at B=1 in the model runner; `max_num_seqs=1` also set so the scheduler picks one request at a time from the queue.

**TTFT measurement:** wall-clock time from request submission to when the first token is ready (measured via `time.perf_counter()` at step-listener fire time — includes Python/scheduling overhead, CUDA sync, forward pass, sampling, and token application).

---

## Throughput workload

32 short independent prompts · **all submitted at once** · greedy · max 64 tokens · 1 warmup

All 32 requests are queued at t₀; the engine processes them one at a time (B=1). E2E is measured from t₀ per request — it includes queue wait time, so the median request (16th) has waited for ~15 prior requests.

| Engine (B=1) | req/s | tok/s | E2E p50 | E2E p99 |
|---|---:|---:|---:|---:|
| liteinfer | 1.69 | 70 | 11956 ms | 18914 ms |
| liteinfer-kvcache | 1.77 | 71 | 10480 ms | 18114 ms |
| vllm | 2.89 | 180 | 5786 ms | 11042 ms |

KV cache gives a modest ~12% E2E improvement for short sequences (max 64 output tokens). The RECOMPUTE cost is relatively low here; the gap widens at longer OSL. vLLM is **1.7× faster** throughput due to CUDA graphs and FlashAttention 2.

---

## Latency workload

20 calls of the same prompt · **sequential, no queue** · greedy · max 128 tokens · 1 warmup

Each request is submitted only after the previous has finished — no queue contamination. Measures pure per-request engine latency.

| Engine (B=1) | TTFT p50 | TTFT p99 | E2E p50 | tok/s |
|---|---:|---:|---:|---:|
| liteinfer | 14 ms | 16 ms | 1726 ms | 72 |
| liteinfer-kvcache | 15 ms | 17 ms | 1541 ms | 72 |
| vllm | 26 ms | 31 ms | 694 ms | 182 |

**TTFT:** liteinfer variants show lower TTFT than vLLM for this short prompt (7 tokens). In RECOMPUTE mode the "prefill" is a 7-token forward pass with no KV-cache setup. In KV-cache mode, prefill populates the cache but there is still no subprocess IPC. vLLM's higher TTFT reflects its multi-process architecture (IPC to EngineCore subprocess, chunked-prefill scheduler) regardless of prompt length. The liteinfer TTFT advantage will narrow for longer prompts.

**KV cache vs no cache:** KV cache reduces E2E by **11%** (1541 ms vs 1726 ms) and tightens TTFT p99 variance. The improvement is modest because liteinfer uses eager PyTorch attention without CUDA graphs — each decode step still pays full Python/GPU launch overhead.

**E2E gap vs vLLM (2.5×):** comes from decode. vLLM uses CUDA graphs (near-zero Python overhead per decode step) and FlashAttention 2, while liteinfer uses standard eager attention without any fusion or graph capture.

---

## What these numbers mean for v0

| Gap | Root cause | Roadmap item |
|---|---|---|
| 2.5× slower decode tok/s | Eager attention, no CUDA graphs, no KV cache in default mode | §2 KV cache, §3.1 `torch.compile`, §3.2 CUDA graphs, §3.3 FlashAttention |
| 1.7× lower throughput req/s | All of the above + no continuous batching | §1.1 static batching, §1.2 continuous batching |
| Modest KV cache gain (~12%) | KV cache without CUDA graphs still pays per-step Python overhead | §3.2 CUDA graphs for decode |

See [`docs/roadmap.md`](roadmap.md) for the full backlog.
