# Milestones

Achieved milestones, newest first. When a roadmap item lands: flip its `Status` to `landed` in `roadmap.md`, then add an entry here.

---

## 2026-05 — §1.1 Static batching with B > 1

- **PRs.** [#10](https://github.com/ValeGian/liteinfer/pull/10)
- **What.** Static batching now respects `EngineConfig.max_num_seqs` end-to-end. `LLM.generate` submits all prompts up front and drains the engine, letting the scheduler form batches of up to `max_num_seqs` sequences. `ModelRunner` left-pads variable-length prompts for prefill and threads an additive attention mask (new `liteinfer/engine/attention_mask.py`) through `LlamaModel.forward(attention_mask=...)`. Strict-static policy: a batch enters together (one prefill), runs to completion, exits together — early finishers stay in the running set so tensor shapes are stable.

---

## 2026-05 — §4.1 e2e parity test vs `transformers`

- **PRs.** [#3](https://github.com/ValeGian/liteinfer/pull/3)
- **What.** `tests/e2e/test_llama_gpu.py` validates bit-equivalent greedy output vs `AutoModelForCausalLM.generate` on Llama-3.2-1B-Instruct (3 prompts, 20 tokens). Requires CUDA.

---

## 2026-05 — v0: minimal end-to-end inference

- **PRs.** [#3](https://github.com/ValeGian/liteinfer/pull/3)
- **What.** First working inference path: local safetensors loading, Llama + Gemma4 dispatch, KV cache (eager / none), greedy sampler (temperature / top-k / top-p), static batch scheduler (B=1), per-step metrics, `device="auto"`. 47 unit tests, lint + pyright clean.
