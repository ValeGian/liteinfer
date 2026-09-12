"""Canonical benchmark dataset: one fixed file per (model, shape).

Corpus is ShareGPT_V3 (the source vLLM's own benchmarks use). Every engine is
fed byte-identical prompts, which `sha256` proves.

There are two shapes, and the difference is what the prompt lengths do.

**Fixed ISL** concatenates all human turns and cuts consecutive windows of
`target_isl` tokens, so every prompt in a run is the same length. That isolates
one shape, which is what a kernel measurement wants. Tokenizer encode->decode
round-trips shift the realised length by a few tokens, so each sample records
the length it actually has; `target_isl` is the length that was asked for.

**Mixed** (`target_isl=None`) keeps whole turns instead, so the run carries the
corpus's own length spread — median 17 tokens, mean 74, p99 1,286. That
distinction is not cosmetic: a batch is left-padded to its longest prompt, so
at a fixed ISL padding wastes *exactly nothing*, while on the real spread a
batch of 32 computes 11.64x the positions it keeps. Every liteinfer row measured
before this existed was measured on the one workload where that cost is
invisible (§3.6, §8.7).
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

# Longest prompt a mixed run admits unless asked otherwise. The corpus reaches
# 10,046 tokens and the configs declare `max_model_len` 4,096, so an uncapped
# spread would not fit the engine it is meant to measure. 2,048 keeps the
# distribution's shape — it is past the corpus's p99 of 1,286 — while leaving
# room for the output tokens the run then generates.
DEFAULT_MAX_MIXED_ISL = 2048


@dataclass(frozen=True)
class Sample:
    prompt: str
    input_tokens: int


@dataclass(frozen=True)
class Dataset:
    model: str
    target_isl: int | None
    """Tokens every prompt holds, or `None` for the corpus's own spread."""
    target_osl: int
    samples: list[Sample]
    max_isl: int | None = None
    """Longest prompt a mixed run admits. `None` on a fixed-ISL run, where
    `target_isl` already says it. It belongs in the shape's name because two
    mixed datasets with different caps are different workloads, not one."""

    @property
    def sha256(self) -> str:
        """Order-independent digest of the prompt set."""
        payload = json.dumps(sorted(s.prompt for s in self.samples), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def head(self, n: int | None) -> Dataset:
        if n is None or n >= len(self.samples):
            return self
        return Dataset(
            self.model, self.target_isl, self.target_osl, self.samples[:n], self.max_isl
        )


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


def shape_slug(target_isl: int | None, target_osl: int, max_isl: int | None = None) -> str:
    """The shape's name, as it appears in a dataset file and in a result file.

    Both spell it the same way on purpose: a result is findable from the dataset
    that produced it without either side knowing how the other builds names.
    """
    isl = f"mixed{max_isl}" if target_isl is None else str(target_isl)
    return f"isl{isl}_osl{target_osl}"


def filename_for(
    model: str, isl: int | None, osl: int, num_samples: int, max_isl: int | None = None
) -> str:
    slug = re.sub(r"_+", "_", re.sub(r"[/\-.]", "_", model.lower())).strip("_")
    return f"{shape_slug(isl, osl, max_isl)}_n{num_samples}_{slug}.json"


def _windowed_samples(tokenizer, turns: list[str], target_isl: int, num_samples: int) -> list[Sample]:
    """Prompts of exactly ``target_isl`` tokens, cut from the concatenated corpus.

    Only as much of the corpus as the run needs is tokenised: turns are
    concatenated until they cover ``num_samples * target_isl`` tokens, then cut
    into consecutive non-overlapping windows. Tokenising the whole corpus to keep
    a thousandth of it would cost minutes and gigabytes.
    """
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
    return samples


def _whole_turn_samples(tokenizer, turns: list[str], max_isl: int, num_samples: int) -> list[Sample]:
    """Whole turns, so the run carries the length spread the corpus has.

    A turn is taken as it stands, which also makes ``input_tokens`` exact — there
    is no decode step to shift it. Turns longer than ``max_isl`` are skipped
    rather than truncated: truncating would pile mass on the cap and report a
    spread the corpus does not have, and the cap exists so a run fits the
    ``max_model_len`` the configs declare.
    """
    samples: list[Sample] = []
    for turn in turns:
        if len(samples) == num_samples:
            return samples
        length = len(tokenizer.encode(turn, add_special_tokens=False))
        if 0 < length <= max_isl:
            samples.append(Sample(turn, length))
    raise ValueError(
        f"Corpus holds {len(samples)} turns of at most {max_isl} tokens, need {num_samples}"
    )


def generate(
    model: str,
    target_isl: int | None,
    target_osl: int,
    num_samples: int,
    output_dir: str | Path,
    seed: int = 42,
    revision: str = _REVISION,
    max_isl: int = DEFAULT_MAX_MIXED_ISL,
) -> Path:
    """Build a dataset file and return its path.

    ``target_isl=None`` asks for the corpus's own length spread, capped at
    ``max_isl``; an integer asks for prompts of exactly that length. Turns are
    shuffled with ``seed`` either way, so the same seed gives the same prompts.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    turns = list(_human_turns(revision))
    random.Random(seed).shuffle(turns)

    is_mixed = target_isl is None
    samples = (
        _whole_turn_samples(tokenizer, turns, max_isl, num_samples)
        if is_mixed
        else _windowed_samples(tokenizer, turns, target_isl, num_samples)
    )
    cap = max_isl if is_mixed else None

    path = Path(output_dir) / filename_for(model, target_isl, target_osl, num_samples, cap)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": model,
                "target_isl": target_isl,
                "target_osl": target_osl,
                "max_isl": cap,
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
        # Absent from every file written before mixed shapes existed, and `None`
        # is what those files mean: a fixed ISL has no cap to state.
        max_isl=record.get("max_isl"),
    )
