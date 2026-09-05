"""Builds a tiny CPU Llama for integration tests. Small enough to run anywhere."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.llama.configuration_llama import LlamaConfig

VOCAB_SIZE = 256
EOS_ID = 1


def build(model_dir: Path) -> None:
    cfg = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_scaling={"rope_type": "linear", "factor": 1.0},
        architectures=["LlamaForCausalLM"],
        tie_word_embeddings=True,
    )
    cfg.save_pretrained(str(model_dir))

    from liteinfer.models.llama import LlamaForCausalLM

    torch.manual_seed(0)
    with torch.device("cpu"):
        model = LlamaForCausalLM(cfg)
    model = model.to(dtype=torch.float32)
    state = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, str(model_dir / "model.safetensors"))

    vocab: dict[str, int] = {"<unk>": 0, "<eos>": EOS_ID}
    for i in range(2, VOCAB_SIZE):
        vocab[f"tok{i}"] = i
    hf_tok = HFTokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    hf_tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=hf_tok, eos_token="<eos>", unk_token="<unk>"
    ).save_pretrained(str(model_dir))
