"""Sampling: parameters and the sampler itself.

Sampling is a separate stage from model execution so that strategies
(greedy, top-p, beam, …) can be swapped without touching the engine.
"""

from liteinfer.sampling.params import SamplingParams
from liteinfer.sampling.sampler import Sampler

__all__ = ["SamplingParams", "Sampler"]
