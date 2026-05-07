# Milestones

Achieved milestones, newest section at the bottom. Mirrors the
roadmap's per-item format so the audit trail is preserved on the way
in.

## Entry template

```
### <year-month label> — <short milestone name>
- **PRs.** #12, #14
- **Why.** <user-facing benefit, copied or distilled from the roadmap>
- **Notes.** <anything unusual: scope changes, follow-ups, parity gaps>
```

## Workflow

When a roadmap item lands:

1. In `roadmap.md`: flip the item's `Status` to `landed`.
2. Move the item's body here, under a dated heading. Carry over the
   `PRs` list.
3. If the milestone bundles several roadmap items, link each one's
   PR(s) and reference the roadmap section ID (`§1.1`, `§2.1`, …) for
   traceability.

---

## 2026-05 — v0: minimal end-to-end inference

- **PRs.** #3
- **Roadmap items closed.** _N/A — pre-roadmap baseline_

What landed:

- [x] **Local safetensors loading.** `liteinfer/models/loader.py`
      reads `config.json` + `*.safetensors` from a local directory.
      Remote download is intentionally out of scope.
- [x] **Architecture dispatch.** `LlamaForCausalLM` and
      `Gemma4ForCausalLM` selected via `config.json["architectures"]`.
      Modeling vendored from `transformers` 5.7.0 into
      `liteinfer/models/{llama,gemma4}.py` (Llama rewritten clean
      ~340 lines; Gemma4 trimmed to text-generation path only).
- [x] **HF tokenizer wrapper.** `liteinfer.tokenizer.Tokenizer` over
      `AutoTokenizer` keeps engine code provider-agnostic.
- [x] **Two cache modes.** `cache_mode="eager"` wraps
      `transformers.DynamicCache` behind the `KVCache` ABC;
      `cache_mode="none"` recomputes the full sequence each step.
      Greedy parity verified between modes on Llama-3.2-1B.
- [x] **Sampler.** Greedy / temperature / top-k / top-p with per-seq
      seeded generators.
- [x] **Per-step metrics.** `StepMetrics` snapshot per
      `LLMEngine.step()` (`phase`, `input_tokens`, `new_tokens`,
      `wall_time_s`, throughput properties). `EngineStats` accumulates
      and exposes prefill/decode-tagged averages. `on_step` listener
      hook for live dashboards.
- [x] **Static scheduler skeleton.** `Scheduler` drains waiting → one
      static batch, lockstep until drain, then next batch. Currently
      hard-capped to batch size 1 by the runner (see roadmap §1.1).
- [x] **Device auto-resolution.** `device="auto"` picks `cuda` if
      available else `cpu`.
- [x] **End-to-end demo.** `LLM(model=...).generate(...)` returns
      coherent greedy output on Llama-3.2-1B locally.
- [x] **Tests.** 47 unit tests across config, sampling params, sampler,
      scheduler, metrics, smoke. Lint + pyright clean on `liteinfer/`.

---

## 2026-05 — §4.1 real e2e parity test against `transformers`

- **PRs.** #3
- **Why.** Validates liteinfer's vendored layers and engine produce
  bit-equivalent outputs to `AutoModelForCausalLM.generate(..., do_sample=False)`.
- **Notes.** Landed as `tests/e2e/test_llama_gpu.py` (originally planned
  as `test_llama_parity.py`). GPU + e2e marked; requires CUDA and
  `meta-llama/Llama-3.2-1B-Instruct`. Parity pinned on three prompts,
  20 greedy tokens each.
