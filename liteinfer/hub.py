"""Resolve a model identifier to a local directory path.

Accepts either a local directory path or a HuggingFace Hub repo ID
(e.g. ``"meta-llama/Llama-3.2-1B"``). Hub models are downloaded via
``snapshot_download`` and cached in the default HF cache directory.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def resolve_model_path(model: str) -> Path:
    """Return a local directory path for *model*.

    If *model* is an existing local directory, return it directly.
    Otherwise treat it as a HuggingFace Hub repo ID and download it
    (or return the already-cached snapshot).

    Raises:
        huggingface_hub.utils.RepositoryNotFoundError: if the repo ID
            does not exist on the Hub.
    """
    local = Path(model)
    if local.is_dir():
        return local

    cached = snapshot_download(model)
    return Path(cached)
