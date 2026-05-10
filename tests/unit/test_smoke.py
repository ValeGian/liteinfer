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


def test_sampling_params_greedy_when_temperature_zero() -> None:
    assert SamplingParams(temperature=0.0).greedy
    assert not SamplingParams(temperature=0.7).greedy


def test_engine_config_rejects_zero_max_num_seqs() -> None:
    with pytest.raises(ValueError):
        EngineConfig(model="dummy", max_num_seqs=0)


def test_engine_config_rejects_zero_max_model_len() -> None:
    with pytest.raises(ValueError):
        EngineConfig(model="dummy", max_model_len=0)


def test_engine_config_rejects_invalid_cache_mode() -> None:
    with pytest.raises(ValueError):
        EngineConfig(model="dummy", cache_mode="turbo")  # type: ignore[arg-type]


def test_engine_config_accepts_paged_cache_mode() -> None:
    cfg = EngineConfig(model="dummy", cache_mode="paged")
    assert cfg.cache_mode == "paged"


def test_engine_config_accepts_defaults() -> None:
    cfg = EngineConfig(model="dummy")
    assert cfg.cache_mode == "none"
    assert cfg.device == "auto"
