"""Reusable building blocks (attention, RMSNorm, RoPE, linear, …).

Each layer is a `torch.nn.Module` whose forward signature is the
contract the model code depends on. Keep them small, dependency-free,
and unit-testable on CPU with toy tensor shapes.

Convention: one layer family per file (`attention.py`, `rmsnorm.py`,
`rotary.py`, `linear.py`, …). Implementations are added as the engine
grows; this `__init__.py` stays empty until there is something stable
to re-export.
"""
