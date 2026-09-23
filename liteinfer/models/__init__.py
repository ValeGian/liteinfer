"""Model implementations and weight loading.

Per-architecture model code (e.g., `llama.py`, `qwen.py`) lives here.
Each model exposes a constructor that accepts an `EngineConfig` and a
`forward()` matching the shape the engine's runner expects.

The dispatch table from HF architecture name to local class is owned by
`load_hf_model`.
"""

LAST_POSITION = slice(-1, None)
"""The positions an inference pass reads logits at, for any architecture here.

Part of that shared `forward()` contract: every model takes `logits_positions`
and projects only those positions to vocabulary logits. Left-padded prompts put
every sequence's last real token in the last column whatever its length, so this
one slice serves the whole batch. A packed batch has no such column — each
sequence ends where the next begins — and passes a tensor of indices instead.

It matters because the head's output is `batch x positions x vocab`. At batch 8
and a 2,048-token prompt that is 3.91 GiB, **96% of the whole prefill's peak
allocation**, and an inference pass reads one row of it per sequence.
"""
