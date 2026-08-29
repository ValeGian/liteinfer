# Testing strategy

`liteinfer` is **test-first**: every new feature ships with the tests
that pin its contract, and most features should be exercisable without
loading a real model.

## Layout

- `tests/unit/` — single-component tests. CPU-only, fast (< 1s each), no
  model loading. Layers, sampling logic, scheduler decisions, and KV-cache
  bookkeeping live here.
- `tests/integration/` — multi-component tests. Wire two or three
  components together (e.g. scheduler + KV cache + a toy model on CPU).
  Still fast; no HF download.
- `tests/e2e/` — load a small real model (e.g. `Llama-3.2-1B`) and
  verify behavior against `transformers`. Requires GPU; mark with
  `@pytest.mark.gpu` and `@pytest.mark.e2e`.

## Markers

| Marker      | Meaning                                                         |
| ----------- | --------------------------------------------------------------- |
| `gpu`       | Requires CUDA. Auto-skipped on CPU-only machines.               |
| `slow`      | Takes more than a few seconds.                                  |
| `e2e`       | Loads a real model.                                             |
| `benchmark` | Performance measurement, not correctness — skipped by default.  |

## Running

```bash
pytest                              # everything that runs in this env
pytest -m "not gpu and not slow"    # the fast suite (CI default)
pytest -m gpu                       # GPU-only
pytest tests/unit/test_smoke.py     # one file
```

## Conventions

- One assertion family per test. Name tests
  `test_<unit>_<condition>_<expected>`.
- Use small, deterministic toy tensors for layer tests — favour shapes
  like `(2, 4, 8)` over realistic shapes.
- For numerical checks against a reference (HF, naive impl), use
  `torch.testing.assert_close`, not `torch.equal`.
- New optimizations must come with: (1) a unit test for the new code
  path, (2) a parity test against the existing path, and (3) a `BenchmarkConfig`
  in `benchmarks/configs.py` whose `baseline` names the config it improves on,
  so `bench report` quantifies the change 1:1.
