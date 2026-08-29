"""Model implementations and weight loading.

Per-architecture model code (e.g., `llama.py`, `qwen.py`) lives here.
Each model exposes a constructor that accepts an `EngineConfig` and a
`forward()` matching the shape the engine's runner expects.

The dispatch table from HF architecture name to local class is owned by
`load_hf_model`.
"""

from liteinfer.models.loader import load_hf_model

__all__ = ["load_hf_model"]
