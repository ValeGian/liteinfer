"""Smoke tests — verify the package imports and basic invariants hold."""

from __future__ import annotations

import pytest

import liteinfer
from liteinfer import EngineConfig, SamplingParams


def test_package_exposes_public_api() -> None:
    for name in ("LLM", "SamplingParams", "EngineConfig", "RequestOutput"):
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


def test_engine_config_rejects_invalid_tp_size() -> None:
    with pytest.raises(ValueError):
        EngineConfig(model="dummy", tensor_parallel_size=0)


def test_engine_config_rejects_invalid_memory_util() -> None:
    with pytest.raises(ValueError):
        EngineConfig(model="dummy", gpu_memory_utilization=1.5)


def test_engine_config_accepts_defaults() -> None:
    cfg = EngineConfig(model="dummy")
    assert cfg.tensor_parallel_size == 1
    assert cfg.block_size > 0
