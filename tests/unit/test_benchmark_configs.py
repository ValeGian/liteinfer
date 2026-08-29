"""The config matrix must form a valid comparison lineage."""

from __future__ import annotations

import pytest

from benchmarks.configs import CONFIGS, get


def test_every_baseline_names_a_known_config() -> None:
    unknown = {c.baseline for c in CONFIGS.values() if c.baseline} - set(CONFIGS)
    assert unknown == set()


def test_baseline_chains_terminate() -> None:
    for config in CONFIGS.values():
        seen, cursor = set(), config
        while cursor.baseline is not None:
            assert cursor.name not in seen, f"cycle through {cursor.name}"
            seen.add(cursor.name)
            cursor = CONFIGS[cursor.baseline]


def test_a_config_never_baselines_against_another_engine() -> None:
    crossed = [
        c.name for c in CONFIGS.values() if c.baseline and CONFIGS[c.baseline].engine != c.engine
    ]
    assert crossed == []


def test_continuous_configs_use_the_paged_cache() -> None:
    assert all(c.cache_mode == "paged" for c in CONFIGS.values() if c.continuous)


def test_get_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError):
        get("liteinfer-does-not-exist")
