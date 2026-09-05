"""Smoke tests — verify the package imports and basic invariants hold."""

from __future__ import annotations

import pytest

import liteinfer
from liteinfer import EngineConfig, SamplingParams


def test_package_exposes_public_api() -> None:
    for name in ("LLM", "SamplingParams", "EngineConfig", "RequestOutput", "StepMetrics"):
        assert hasattr(liteinfer, name), f"missing public export: {name}"


def test_sampling_params_rejects_negative_temperature() -> None:
    with pytest.raises(ValueError):
        SamplingParams(temperature=-0.1)


def test_sampling_params_rejects_zero_max_tokens() -> None:
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=0)


def test_sampling_params_rejects_negative_min_tokens() -> None:
    with pytest.raises(ValueError):
        SamplingParams(min_tokens=-1)


def test_sampling_params_rejects_min_tokens_greater_than_max_tokens() -> None:
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=5, min_tokens=10)


def test_sampling_params_ignore_eos_defaults_false() -> None:
    assert SamplingParams().ignore_eos is False


def test_sampling_params_min_tokens_defaults_zero() -> None:
    assert SamplingParams().min_tokens == 0


def test_sampling_params_greedy_when_temperature_zero() -> None:
    assert SamplingParams(temperature=0.0).greedy
    assert not SamplingParams(temperature=0.7).greedy


def test_engine_config_rejects_zero_max_num_seqs() -> None:
    with pytest.raises(ValueError):
        EngineConfig(model="dummy", max_num_seqs=0)


def test_engine_config_rejects_zero_max_model_len() -> None:
    with pytest.raises(ValueError):
        EngineConfig(model="dummy", max_model_len=0)


def test_engine_config_accepts_defaults() -> None:
    cfg = EngineConfig(model="dummy")
    assert cfg.device == "auto"
