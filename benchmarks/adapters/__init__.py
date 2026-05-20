from benchmarks.adapters.base import EngineAdapter
from benchmarks.adapters.liteinfer import LiteInferAdapter
from benchmarks.adapters.trtllm.adapter import TRTLLMAdapter
from benchmarks.adapters.vllm.adapter import VLLMAdapter

ADAPTER_REGISTRY: dict[str, type[EngineAdapter]] = {
    "liteinfer": LiteInferAdapter,
    "vllm": VLLMAdapter,
    "trtllm": TRTLLMAdapter,
}
