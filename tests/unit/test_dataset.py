"""Tests for benchmarks.dataset module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from benchmarks.adapters.base import BenchmarkSample
from benchmarks.dataset import (
    build_dataset_filename,
    dataset_sha256,
    generate_dataset,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tokenizer(token_ids_per_char: int = 1) -> MagicMock:
    """Returns a mock tokenizer that encodes text as one token per character."""
    tokenizer = MagicMock()

    def _encode(text, add_special_tokens=True):
        return list(range(len(text)))

    def _decode(token_ids, skip_special_tokens=True):
        # Return a string with len == len(token_ids)
        return "a" * len(token_ids)

    tokenizer.encode.side_effect = _encode
    tokenizer.decode.side_effect = _decode
    return tokenizer


# ---------------------------------------------------------------------------
# build_dataset_filename
# ---------------------------------------------------------------------------


def test_build_dataset_filename_basic():
    result = build_dataset_filename("meta-llama/Llama-3.2-1B-Instruct", 128, 256, 100)
    assert result == "isl128_osl256_n100_meta_llama_llama_3_2_1b_instruct.jsonl"


def test_build_dataset_filename_slash_in_model_id():
    result = build_dataset_filename("org/Model-Name", 64, 128, 50)
    assert "/" not in result
    assert result == "isl64_osl128_n50_org_model_name.jsonl"


def test_build_dataset_filename_multiple_dashes():
    result = build_dataset_filename("meta-llama/Meta-Llama-3-8B-Instruct", 128, 256, 200)
    assert "/" not in result
    assert "-" not in result
    assert result == "isl128_osl256_n200_meta_llama_meta_llama_3_8b_instruct.jsonl"


def test_build_dataset_filename_no_double_underscores():
    result = build_dataset_filename("a//b--c", 1, 2, 3)
    assert "__" not in result


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------


def test_generate_dataset_writes_valid_jsonl(tmp_path):
    tokenizer = _make_mock_tokenizer()
    # Large enough that the sliding window can extract 5 samples of 20 tokens each.
    mock_corpus = "some realistic instruction text here " * 300
    with patch("benchmarks.dataset.AutoTokenizer") as mock_auto, \
            patch("benchmarks.dataset._get_corpus", return_value=mock_corpus):
        mock_auto.from_pretrained.return_value = tokenizer
        output_file = generate_dataset(
            model_id="test/model",
            target_isl=20,
            target_osl=10,
            num_samples=5,
            output_path=tmp_path,
            seed=42,
        )

    assert output_file.exists()
    lines = output_file.read_text().strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        record = json.loads(line)
        assert "prompt" in record
        assert "input_token_count" in record
        assert "forced_output_token_count" in record
        assert record["forced_output_token_count"] == 10
        # actual count should be within ±5 of target ISL
        assert abs(record["input_token_count"] - 20) <= 5


def test_generate_dataset_filename_derived_from_params(tmp_path):
    tokenizer = _make_mock_tokenizer()
    mock_corpus = "some realistic instruction text here " * 300
    with patch("benchmarks.dataset.AutoTokenizer") as mock_auto, \
            patch("benchmarks.dataset._get_corpus", return_value=mock_corpus):
        mock_auto.from_pretrained.return_value = tokenizer
        output_file = generate_dataset(
            model_id="org/model-id",
            target_isl=10,
            target_osl=5,
            num_samples=3,
            output_path=tmp_path,
            seed=42,
        )
    assert output_file.name == "isl10_osl5_n3_org_model_id.jsonl"


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_returns_all_samples(tmp_path):
    data_file = tmp_path / "test.jsonl"
    records = [
        {"prompt": f"prompt_{i}", "input_token_count": 10 + i, "forced_output_token_count": 5}
        for i in range(20)
    ]
    data_file.write_text("\n".join(json.dumps(r) for r in records))

    samples = load_dataset(data_file)
    assert len(samples) == 20


def test_load_dataset_returns_correct_count(tmp_path):
    data_file = tmp_path / "test.jsonl"
    records = [
        {"prompt": f"p{i}", "input_token_count": 10, "forced_output_token_count": 5}
        for i in range(50)
    ]
    data_file.write_text("\n".join(json.dumps(r) for r in records))

    samples = load_dataset(data_file, num_samples=10)
    assert len(samples) == 10
    assert samples[0].prompt == "p0"
    assert samples[9].prompt == "p9"


def test_load_dataset_returns_benchmark_sample_objects(tmp_path):
    data_file = tmp_path / "test.jsonl"
    data_file.write_text(
        json.dumps({"prompt": "hello", "input_token_count": 3, "forced_output_token_count": 7})
    )
    samples = load_dataset(data_file)
    assert len(samples) == 1
    assert isinstance(samples[0], BenchmarkSample)
    assert samples[0].prompt == "hello"
    assert samples[0].input_token_count == 3
    assert samples[0].forced_output_token_count == 7


# ---------------------------------------------------------------------------
# dataset_sha256
# ---------------------------------------------------------------------------


def test_dataset_sha256_is_order_independent():
    samples_a = [
        BenchmarkSample(prompt="hello", input_token_count=2, forced_output_token_count=4),
        BenchmarkSample(prompt="world", input_token_count=3, forced_output_token_count=4),
    ]
    samples_b = [
        BenchmarkSample(prompt="world", input_token_count=3, forced_output_token_count=4),
        BenchmarkSample(prompt="hello", input_token_count=2, forced_output_token_count=4),
    ]
    assert dataset_sha256(samples_a) == dataset_sha256(samples_b)


def test_dataset_sha256_detects_change():
    samples_original = [
        BenchmarkSample(prompt="hello world", input_token_count=2, forced_output_token_count=4),
    ]
    samples_modified = [
        BenchmarkSample(prompt="hello earth", input_token_count=2, forced_output_token_count=4),
    ]
    assert dataset_sha256(samples_original) != dataset_sha256(samples_modified)


def test_dataset_sha256_is_deterministic():
    samples = [
        BenchmarkSample(prompt=f"sample {i}", input_token_count=i, forced_output_token_count=5)
        for i in range(10)
    ]
    assert dataset_sha256(samples) == dataset_sha256(samples)
