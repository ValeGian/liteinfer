"""Model implementations and weight loading.

Per-architecture model code lives here — `llama.py` today. Each model is built
from its `transformers` config and exposes a `forward()` matching the shape the
engine's runner expects.

The dispatch table from HF architecture name to local class is owned by
`load_hf_model`.
"""

LAST_POSITION = slice(-1, None)
"""The positions an inference pass reads logits at, for any architecture here.

Part of that shared `forward()` contract: every model takes `logits_positions`
and projects only those positions to vocabulary logits. A decode batch lays one
token per row, so this one slice serves the whole batch. A packed batch has no
such column — each sequence ends where the next begins — and passes a tensor of
indices instead.
"""
