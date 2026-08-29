"""Canonical benchmark dataset: one fixed file per (model, ISL, OSL).

Corpus is ShareGPT_V3 (the source vLLM's own benchmarks use). All human turns
are concatenated, then a sliding token window extracts ISL-controlled prompts.
Every engine is then fed byte-identical prompts, which `sha256` proves.

Tokenizer encode->decode round-trips shift the realised prompt length by a few
tokens, so each sample records the length it actually has; `target_isl` in the
metadata is the length that was asked for.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
# Pinned for cross-machine reproducibility. To update, pick a newer commit from
# https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/commits/main
_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"

_corpus_cache: dict[str, list[str]] = {}


@dataclass(frozen=True)
class Sample:
    prompt: str
    input_tokens: int


@dataclass(frozen=True)
class Dataset:
    model: str
    target_isl: int
    target_osl: int
    samples: list[Sample]

    @property
    def sha256(self) -> str:
        """Order-independent digest of the prompt set."""
        payload = json.dumps(sorted(s.prompt for s in self.samples), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def head(self, n: int | None) -> Dataset:
        if n is None or n >= len(self.samples):
            return self
        return Dataset(self.model, self.target_isl, self.target_osl, self.samples[:n])


def _human_turns(revision: str) -> list[str]:
    """Every human turn in the corpus, cached per revision."""
    if revision not in _corpus_cache:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=_REPO_ID, filename=_FILENAME, repo_type="dataset", revision=revision
        )
        conversations = json.loads(Path(path).read_text(encoding="utf-8"))
        _corpus_cache[revision] = [
            turn["value"].strip()
            for conv in conversations
            for turn in conv.get("conversations", [])
            if turn.get("from") == "human" and turn.get("value", "").strip()
        ]
    return _corpus_cache[revision]


def filename_for(model: str, isl: int, osl: int, num_samples: int) -> str:
    slug = re.sub(r"_+", "_", re.sub(r"[/\-.]", "_", model.lower())).strip("_")
    return f"isl{isl}_osl{osl}_n{num_samples}_{slug}.json"


def generate(
    model: str,
    target_isl: int,
    target_osl: int,
    num_samples: int,
    output_dir: str | Path,
    seed: int = 42,
    revision: str = _REVISION,
) -> Path:
    """Build a dataset file and return its path.

    Only as much of the corpus as the run needs is tokenised: turns are shuffled
    with ``seed``, concatenated until they cover ``num_samples * target_isl``
    tokens, then cut into consecutive non-overlapping windows. Tokenising the
    whole corpus to keep a thousandth of it would cost minutes and gigabytes.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    turns = list(_human_turns(revision))
    random.Random(seed).shuffle(turns)

    needed = num_samples * target_isl
    token_ids: list[int] = []
    for turn in turns:
        token_ids += tokenizer.encode(turn, add_special_tokens=False)
        if len(token_ids) >= needed:
            break
    if len(token_ids) < needed:
        raise ValueError(
            f"Corpus holds {len(token_ids)} tokens, need {needed} "
            f"({num_samples} samples of {target_isl})"
        )

    samples = []
    for index in range(num_samples):
        window = token_ids[index * target_isl : (index + 1) * target_isl]
        prompt = tokenizer.decode(window, skip_special_tokens=True)
        samples.append(Sample(prompt, len(tokenizer.encode(prompt, add_special_tokens=False))))

    path = Path(output_dir) / filename_for(model, target_isl, target_osl, num_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": model,
                "target_isl": target_isl,
                "target_osl": target_osl,
                "samples": [{"prompt": s.prompt, "input_tokens": s.input_tokens} for s in samples],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load(path: str | Path) -> Dataset:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return Dataset(
        model=record["model"],
        target_isl=record["target_isl"],
        target_osl=record["target_osl"],
        samples=[Sample(s["prompt"], s["input_tokens"]) for s in record["samples"]],
    )
