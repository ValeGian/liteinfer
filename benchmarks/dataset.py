"""
Canonical benchmark dataset generation and loading.

Corpus source: ShareGPT_V3 (anon8231489123/ShareGPT_Vicuna_unfiltered on HuggingFace).
All human turns from ~100K real conversations are concatenated into a single text corpus;
a sliding token window then extracts ISL-controlled samples. This matches the dataset
used by vLLM's own benchmarks, making results directly comparable.

dataset_revision in generate_dataset should be pinned to a git commit hash for
strict cross-machine reproducibility. Using "main" (the default) is convenient but
means the corpus may change if the dataset is updated upstream.

ISL variance: tokenizer encode->decode round-trip introduces +-0-5 token variance
from the target ISL. The actual input_token_count is recorded per sample, so
result JSONs correctly reflect what each engine received. ±5 token variance is
acceptable for throughput/latency benchmarking.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from benchmarks.adapters.base import BenchmarkSample

# ---------------------------------------------------------------------------
# Corpus — ShareGPT (~100K real user conversations, same source as vLLM benchmarks)
# ---------------------------------------------------------------------------

_SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
_SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"

# Pin to a specific git commit hash for strict cross-machine reproducibility.
# To update: find the latest commit at
# https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/commits/main
# then update this constant and regenerate all datasets.
_SHAREGPT_DEFAULT_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"

# Per-revision in-process cache: avoids re-downloading within one process.
_corpus_by_revision: dict[str, str] = {}


def _build_sharegpt_corpus(revision: str) -> str:
    """Download ShareGPT and concatenate all human turns into a single text corpus."""
    json_path = hf_hub_download(
        repo_id=_SHAREGPT_REPO_ID,
        filename=_SHAREGPT_FILENAME,
        repo_type="dataset",
        revision=revision,
    )

    with open(json_path, encoding="utf-8") as f:
        conversations: list[dict] = json.load(f)

    texts: list[str] = []
    for conv in conversations:
        for turn in conv.get("conversations", []):
            if turn.get("from") == "human":
                text = turn.get("value", "").strip()
                if text:
                    texts.append(text)

    return "\n\n".join(texts)


def _get_corpus(revision: str) -> str:
    if revision not in _corpus_by_revision:
        _corpus_by_revision[revision] = _build_sharegpt_corpus(revision)
    return _corpus_by_revision[revision]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dataset_filename(model_id: str, isl: int, osl: int, num_samples: int) -> str:
    """Derive canonical output filename from benchmark parameters.

    Slugifies model_id: lowercase, replace '/' and '-' with '_', compress
    repeated '_'.

    Example:
        "meta-llama/Llama-3.2-1B-Instruct" → "meta_llama_llama_3_2_1b_instruct"
        → "isl128_osl256_n100_meta_llama_llama_3_2_1b_instruct.jsonl"
    """
    slug = model_id.lower()
    slug = re.sub(r"[/\-\.]", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("_")
    return f"isl{isl}_osl{osl}_n{num_samples}_{slug}.jsonl"


def generate_dataset(
        model_id: str,
        target_isl: int,
        target_osl: int,
        num_samples: int,
        output_path: str | Path,
        seed: int = 42,
        dataset_revision: str = _SHAREGPT_DEFAULT_REVISION,
) -> Path:
    """Generate a canonical benchmark dataset JSONL file.

    Downloads ShareGPT at the given revision, tokenizes the concatenated corpus of
    human turns, and slides a window of target_isl tokens to extract samples. Each
    window is decoded back to text to form the prompt.

    Pin dataset_revision to a git commit hash for cross-machine reproducibility.
    Actual input_token_count may differ from target_isl by ±5 tokens due to
    tokenizer round-trip; this variance is acceptable and documented (§9.4).
    """
    output_path = Path(output_path)
    if output_path.is_dir():
        filename = build_dataset_filename(model_id, target_isl, target_osl, num_samples)
        output_path = output_path / filename

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    corpus = _get_corpus(dataset_revision)
    token_ids = tokenizer.encode(corpus, add_special_tokens=False)

    max_start = len(token_ids) - target_isl
    if max_start < num_samples:
        raise ValueError(
            f"Corpus too small: need {num_samples} windows of {target_isl} tokens "
            f"but corpus has only {len(token_ids)} tokens ({max_start} valid starts)"
        )

    rng = random.Random(seed)
    selected_starts = rng.sample(range(max_start), num_samples)

    samples: list[dict] = []
    for start in selected_starts:
        window = token_ids[start: start + target_isl]
        prompt = tokenizer.decode(window, skip_special_tokens=True)
        actual_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        samples.append(
            {
                "prompt": prompt,
                "input_token_count": actual_count,
                "forced_output_token_count": target_osl,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return output_path


def load_dataset(
        path: str | Path,
        num_samples: int | None = None,
) -> list[BenchmarkSample]:
    """Load a canonical benchmark dataset from a JSONL file."""
    path = Path(path)
    samples: list[BenchmarkSample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(
                BenchmarkSample(
                    prompt=record["prompt"],
                    input_token_count=record["input_token_count"],
                    forced_output_token_count=record["forced_output_token_count"],
                )
            )
            if num_samples is not None and len(samples) >= num_samples:
                break
    return samples


def dataset_sha256(samples: list[BenchmarkSample]) -> str:
    """SHA-256 of the sorted prompt list (order-independent).

    Used for cross-engine identity checks: if two result files carry the same
    sha256, they were run against the same set of prompts.
    """
    sorted_prompts = sorted(s.prompt for s in samples)
    payload = json.dumps(sorted_prompts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
