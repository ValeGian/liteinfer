"""Shared pytest fixtures and collection hooks.

GPU-only tests (`@pytest.mark.gpu`) are auto-skipped when CUDA is not
available, so the same suite runs in CPU-only and GPU environments.
"""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
