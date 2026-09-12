"""Dataset generation, loading, and prompt-set identity."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from benchmarks import dataset

_TURNS = [f"instruction number {i} with some trailing words" for i in range(400)]
# The corpus's own spread is the point of a mixed dataset, so the turns it is
# built from have to have one. Lengths here are the character counts the fake
# tokenizer below returns, which run from 4 to 400.
_VARIED_TURNS = ["x" * (4 * (i + 1)) for i in range(100)]


@pytest.fixture
def tokenizer() -> MagicMock:
    tok = MagicMock()
    # A true round-trip: decode(encode(text)) == text, so windows stay distinguishable.
    tok.encode.side_effect = lambda text, add_special_tokens=True: [ord(c) for c in text]
    tok.decode.side_effect = lambda ids, skip_special_tokens=True: "".join(map(chr, ids))
    return tok


def _generate(
    tmp_path, tokenizer, *, isl: int | None = 8, osl=16, n=5, seed=42, turns=None, max_isl=2048
):
    with (
        patch("transformers.AutoTokenizer") as auto,
        patch.object(dataset, "_human_turns", return_value=turns or _TURNS),
    ):
        auto.from_pretrained.return_value = tokenizer
        return dataset.generate(
            "test/model", isl, osl, n, tmp_path, seed=seed, max_isl=max_isl
        )


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


# ---------------------------------------------------------------------------
# Mixed shapes: prompts whose lengths are the corpus's, not the benchmark's
# ---------------------------------------------------------------------------


def test_a_fixed_isl_gives_every_prompt_the_same_length(tmp_path, tokenizer) -> None:
    """The property that makes padding free, and the reason mixed had to exist."""
    loaded = dataset.load(_generate(tmp_path, tokenizer, isl=8, n=5))

    assert {s.input_tokens for s in loaded.samples} == {8}


def test_a_mixed_dataset_keeps_the_corpus_spread(tmp_path, tokenizer) -> None:
    """Whole turns, so the lengths are whatever the corpus has."""
    loaded = dataset.load(
        _generate(tmp_path, tokenizer, isl=None, n=20, turns=_VARIED_TURNS)
    )

    assert len({s.input_tokens for s in loaded.samples}) > 1


def test_a_mixed_dataset_holds_no_prompt_past_its_cap(tmp_path, tokenizer) -> None:
    """The cap is what lets a run fit the `max_model_len` the configs declare."""
    loaded = dataset.load(
        _generate(tmp_path, tokenizer, isl=None, n=10, turns=_VARIED_TURNS, max_isl=100)
    )

    assert max(s.input_tokens for s in loaded.samples) <= 100


def test_a_mixed_prompt_reports_the_length_it_really_has(tmp_path, tokenizer) -> None:
    """A whole turn needs no decode step, so the recorded length is exact."""
    loaded = dataset.load(
        _generate(tmp_path, tokenizer, isl=None, n=5, turns=_VARIED_TURNS)
    )

    assert all(len(s.prompt) == s.input_tokens for s in loaded.samples)


def test_a_mixed_dataset_records_its_cap(tmp_path, tokenizer) -> None:
    """Two caps are two workloads, so the cap travels with the file."""
    loaded = dataset.load(
        _generate(tmp_path, tokenizer, isl=None, n=5, turns=_VARIED_TURNS, max_isl=128)
    )

    assert (loaded.target_isl, loaded.max_isl) == (None, 128)


def test_a_fixed_dataset_records_no_cap(tmp_path, tokenizer) -> None:
    """`target_isl` already says what the prompts are; a second number would drift from it."""
    loaded = dataset.load(_generate(tmp_path, tokenizer, isl=8))

    assert loaded.max_isl is None


def test_mixed_generation_rejects_a_corpus_with_too_few_short_turns(tmp_path, tokenizer) -> None:
    """Silently returning fewer prompts would shrink the run instead of failing it."""
    with pytest.raises(ValueError, match="turns of at most"):
        _generate(tmp_path, tokenizer, isl=None, n=50, turns=_VARIED_TURNS, max_isl=8)


def test_a_mixed_filename_names_the_cap_rather_than_a_length() -> None:
    name = dataset.filename_for("meta-llama/Llama-3.2-1B-Instruct", None, 128, 200, 2048)
    assert name == "islmixed2048_osl128_n200_meta_llama_llama_3_2_1b_instruct.json"


def test_head_keeps_the_cap(tmp_path, tokenizer) -> None:
    """Trimming a dataset must not turn it into a different workload."""
    loaded = dataset.load(
        _generate(tmp_path, tokenizer, isl=None, n=10, turns=_VARIED_TURNS, max_isl=256)
    )

    assert loaded.head(3).max_isl == 256
