# pyright: reportPrivateImportUsage=false
"""Load HuggingFace-format models from a local directory.

Reads ``config.json``, dispatches on the architecture name to the
matching liteinfer model class, and streams weights from the directory's
``*.safetensors`` shards onto the target device.

Remote download is intentionally out of scope: the engine consumes
already-materialized weights. Run ``huggingface-cli download …`` (or
equivalent) ahead of time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig

from liteinfer.config import EngineConfig

if TYPE_CHECKING:
    from transformers import PretrainedConfig

# Architecture name (from `config.json["architectures"][0]`) → liteinfer class.
# Kept lazy to avoid importing every modeling module when only one is used.
_DISPATCH: dict[str, str] = {
    "LlamaForCausalLM": "liteinfer.models.llama:LlamaForCausalLM",
    "Gemma4ForCausalLM": "liteinfer.models.gemma4:Gemma4ForCausalLM",
}


def _resolve(qualified: str) -> type[nn.Module]:
    module_path, _, attr = qualified.partition(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _read_architecture(model_dir: Path) -> str:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json under {model_dir}")
    raw = json.loads(config_path.read_text())
    architectures = raw.get("architectures") or []
    if not architectures:
        raise ValueError(f"config.json under {model_dir} has no 'architectures' field")
    return architectures[0]


def _iter_safetensor_shards(model_dir: Path) -> list[Path]:
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors files under {model_dir}")
    return shards


def _load_weights(model: nn.Module, model_dir: Path, device: torch.device) -> None:
    """Stream weights from safetensors shards into ``model`` on ``device``.

    Missing keys raise; unexpected keys are tolerated and reported via the
    ``model._unexpected_keys`` attribute (set on the instance) so callers
    can audit them.
    """
    state_keys = set(model.state_dict().keys())
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    seen: set[str] = set()
    unexpected: list[str] = []

    for shard in _iter_safetensor_shards(model_dir):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 — safe_open keys() is API, not dict
                tensor = f.get_tensor(key)
                if key not in state_keys:
                    unexpected.append(key)
                    continue
                target = params.get(key)
                if target is None:
                    target = buffers.get(key)
                if target is None:
                    unexpected.append(key)
                    continue
                with torch.no_grad():
                    target.copy_(tensor.to(device=device, dtype=target.dtype))
                seen.add(key)

    missing = state_keys - seen
    # Tied weights (e.g. lm_head.weight ↔ model.embed_tokens.weight)
    # appear in `state_dict()` under both names but in `named_parameters()`
    # only once: they share storage. If the source name was loaded, the
    # tied name is already populated — just clear it from `missing`.
    tied_map: dict[str, str] = getattr(model, "_tied_weights_keys", {}) or {}
    for tied_key, source_key in tied_map.items():
        if tied_key in missing and source_key in seen:
            missing.discard(tied_key)
    if missing:
        raise RuntimeError(f"missing {len(missing)} weights, e.g. {sorted(missing)[:5]}")

    object.__setattr__(model, "_unexpected_keys", unexpected)


def load_hf_model(config: EngineConfig) -> tuple[nn.Module, PretrainedConfig]:
    """Load a model from a local HuggingFace-format directory.

    Returns ``(model, hf_config)``. The HF config is returned alongside
    so callers (model runner, scheduler) can read fields like
    ``num_hidden_layers`` and ``eos_token_id`` without reopening the
    directory.
    """
    model_dir = Path(config.model)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"{config.model!r} is not a directory; this loader only handles local paths"
        )

    architecture = _read_architecture(model_dir)
    if architecture not in _DISPATCH:
        raise ValueError(
            f"unsupported architecture {architecture!r}; known: {sorted(_DISPATCH)}"
        )

    hf_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    # Force eager attention path; SDPA/flash require correct masking and
    # are an optimization to add later behind a flag.
    hf_config._attn_implementation = "eager"

    model_cls = _resolve(_DISPATCH[architecture])
    text_config = getattr(hf_config, "text_config", hf_config)
    device = config.resolved_device()

    # Initialize on the target device so non-persistent buffers (e.g.
    # RoPE `inv_freq`) are computed correctly. Meta+`to_empty` would
    # leave those buffers as garbage, since they're not stored in
    # safetensors and never get re-derived.
    with torch.device(device):
        model = model_cls(text_config)
    model = model.to(dtype=config.dtype)
    model.eval()

    _load_weights(model, model_dir, device)
    return model, hf_config
