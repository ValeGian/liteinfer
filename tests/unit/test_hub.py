"""Tests for liteinfer.hub.resolve_model_path."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from liteinfer.hub import resolve_model_path


def test_resolve_model_path_returns_local_dir(tmp_path: Path) -> None:
    resolved = resolve_model_path(str(tmp_path))
    assert resolved == tmp_path


def test_resolve_model_path_local_dir_is_path_object(tmp_path: Path) -> None:
    resolved = resolve_model_path(str(tmp_path))
    assert isinstance(resolved, Path)


def test_resolve_model_path_calls_snapshot_download_for_hub_id() -> None:
    fake_cache = "/fake/cache/model"
    with patch("liteinfer.hub.snapshot_download", return_value=fake_cache) as mock_dl:
        resolved = resolve_model_path("org/model-name")
    mock_dl.assert_called_once_with("org/model-name")
    assert resolved == Path(fake_cache)


def test_resolve_model_path_snapshot_download_not_called_for_local_dir(
    tmp_path: Path,
) -> None:
    with patch("liteinfer.hub.snapshot_download") as mock_dl:
        resolve_model_path(str(tmp_path))
    mock_dl.assert_not_called()


def test_resolve_model_path_returns_path_from_snapshot_download() -> None:
    fake_cache = "/hf/hub/models--org--model/snapshots/abc123"
    with patch("liteinfer.hub.snapshot_download", return_value=fake_cache):
        resolved = resolve_model_path("org/model")
    assert isinstance(resolved, Path)
    assert str(resolved) == fake_cache
