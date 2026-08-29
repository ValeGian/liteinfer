"""Dataset generation, loading, and prompt-set identity."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from benchmarks import dataset

_TURNS = [f"instruction number {i} with some trailing words" for i in range(400)]


@pytest.fixture
def tokenizer() -> MagicMock:
    tok = MagicMock()
    # A true round-trip: decode(encode(text)) == text, so windows stay distinguishable.
    tok.encode.side_effect = lambda text, add_special_tokens=True: [ord(c) for c in text]
    tok.decode.side_effect = lambda ids, skip_special_tokens=True: "".join(map(chr, ids))
    return tok


def _generate(tmp_path, tokenizer, *, isl=8, osl=16, n=5, seed=42):
    with (
        patch("transformers.AutoTokenizer") as auto,
        patch.object(dataset, "_human_turns", return_value=_TURNS),
    ):
        auto.from_pretrained.return_value = tokenizer
        return dataset.generate("test/model", isl, osl, n, tmp_path, seed=seed)


def test_generated_file_round_trips(tmp_path, tokenizer) -> None:
    loaded = dataset.load(_generate(tmp_path, tokenizer, n=5))
    assert len(loaded.samples) == 5


def test_targets_survive_the_round_trip(tmp_path, tokenizer) -> None:
    loaded = dataset.load(_generate(tmp_path, tokenizer, isl=8, osl=16))
    assert (loaded.target_isl, loaded.target_osl) == (8, 16)


def test_same_seed_produces_the_same_prompt_set(tmp_path, tokenizer) -> None:
    first = dataset.load(_generate(tmp_path / "a", tokenizer, seed=7))
    second = dataset.load(_generate(tmp_path / "b", tokenizer, seed=7))
    assert first.sha256 == second.sha256


def test_different_seeds_produce_different_prompt_sets(tmp_path, tokenizer) -> None:
    first = dataset.load(_generate(tmp_path / "a", tokenizer, seed=1))
    second = dataset.load(_generate(tmp_path / "b", tokenizer, seed=2))
    assert first.sha256 != second.sha256


def test_sha256_ignores_sample_order(tmp_path, tokenizer) -> None:
    loaded = dataset.load(_generate(tmp_path, tokenizer))
    reversed_order = dataset.Dataset(
        loaded.model, loaded.target_isl, loaded.target_osl, list(reversed(loaded.samples))
    )
    assert loaded.sha256 == reversed_order.sha256


def test_head_trims_to_the_requested_size(tmp_path, tokenizer) -> None:
    loaded = dataset.load(_generate(tmp_path, tokenizer, n=5))
    assert len(loaded.head(2).samples) == 2


def test_head_of_none_keeps_every_sample(tmp_path, tokenizer) -> None:
    loaded = dataset.load(_generate(tmp_path, tokenizer, n=5))
    assert len(loaded.head(None).samples) == 5


def test_generation_rejects_a_corpus_that_is_too_small(tmp_path, tokenizer) -> None:
    with pytest.raises(ValueError, match="Corpus holds"):
        _generate(tmp_path, tokenizer, isl=8, n=10**6)


def test_filename_encodes_the_benchmark_shape() -> None:
    name = dataset.filename_for("meta-llama/Llama-3.2-1B-Instruct", 128, 256, 200)
    assert name == "isl128_osl256_n200_meta_llama_llama_3_2_1b_instruct.json"
