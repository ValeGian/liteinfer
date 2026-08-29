"""Shared fixtures for CPU integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration import tiny_llama


@pytest.fixture(scope="session")
def tiny_llama_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model_dir = tmp_path_factory.mktemp("tiny_llama")
    tiny_llama.build(model_dir)
    return model_dir
