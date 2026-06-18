# Engine Environment Setup

One-time setup per engine. Run from the repository root.

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| VRAM (Llama-3.2-1B benchmarks) | 8 GB | FP16 weights (~2.5 GB) + KV cache |
| VRAM (Llama-3-8B benchmarks) | 24 GB | FP16 weights (~16 GB) + KV cache |
| CUDA | ≥ 12.1 | Required by TensorRT-LLM 1.2.1 |
| RAM | ≥ 32 GB | vLLM and TRT-LLM load large buffers during init |

## HuggingFace Authentication

Required before running any setup scripts (Llama models are gated):

```bash
huggingface-cli login
```

Then accept the Meta Llama license at:
- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

## vLLM (pinned to 0.21.0)

```bash
bash benchmarks/envs/setup_vllm.sh
```

Creates `benchmarks/envs/vllm/` venv. Override Python path via `BENCH_VLLM_PYTHON`.

## TensorRT-LLM (pinned to 1.2.1)

```bash


```

Creates `benchmarks/envs/trtllm/` venv. Uses the PyTorch backend — no `trtllm-build`
compilation required. Override Python path via `BENCH_TRTLLM_PYTHON`.

## Model Downloads

```bash
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
```

## Dataset Generation

```bash
bench dataset generate --model meta-llama/Llama-3.2-1B-Instruct --isl 128 --osl 256 --num-samples 200
bench dataset generate --model meta-llama/Llama-3.2-1B-Instruct --isl 512 --osl 512 --num-samples 200
bench dataset generate --model meta-llama/Meta-Llama-3-8B-Instruct --isl 128 --osl 256 --num-samples 200
bench dataset generate --model meta-llama/Meta-Llama-3-8B-Instruct --isl 512 --osl 512 --num-samples 200
```
