"""Tests for benchmarks.cli module — argparse smoke tests."""

from __future__ import annotations

import pytest

from benchmarks.cli import parse_args


def test_cli_dataset_generate_requires_model():
    with pytest.raises(SystemExit):
        parse_args(["dataset", "generate", "--isl", "128", "--osl", "256"])


def test_cli_dataset_generate_requires_isl():
    with pytest.raises(SystemExit):
        parse_args(["dataset", "generate", "--model", "m", "--osl", "256"])


def test_cli_dataset_generate_requires_osl():
    with pytest.raises(SystemExit):
        parse_args(["dataset", "generate", "--model", "m", "--isl", "128"])


def test_cli_run_requires_engine():
    with pytest.raises(SystemExit):
        parse_args(
            ["run", "--type", "throughput", "--model", "m", "--dataset", "d"]
        )


def test_cli_run_requires_type():
    with pytest.raises(SystemExit):
        parse_args(
            ["run", "--engine", "liteinfer", "--model", "m", "--dataset", "d"]
        )


def test_cli_run_strict_osl_defaults_to_false():
    args = parse_args(
        [
            "run",
            "--engine",
            "liteinfer",
            "--type",
            "throughput",
            "--model",
            "m",
            "--dataset",
            "d",
        ]
    )
    assert args.strict_osl is False


def test_cli_dataset_generate_default_num_samples():
    args = parse_args(
        ["dataset", "generate", "--model", "m", "--isl", "128", "--osl", "256"]
    )
    assert args.num_samples == 200


def test_cli_run_suite_requires_engines():
    with pytest.raises(SystemExit):
        parse_args(
            ["run-suite", "--type", "throughput", "--model", "m", "--dataset", "d"]
        )


def test_cli_dashboard_promote_accepts_run_ids():
    args = parse_args(["dashboard", "promote", "run_a", "run_b"])
    assert args.run_ids == ["run_a", "run_b"]


def test_cli_dashboard_build_default_output():
    args = parse_args(["dashboard", "build"])
    assert args.output == "docs/index.html"
