"""A tiny, randomly initialised chat model for checking the pipeline offline.

It exists so every stage (chat template -> masking -> training steps -> save -> reload ->
generate -> parse -> score) can run on a CPU with no model download. Its weights are
random and its tokenizer is trained on this dataset's text, so what it generates says
nothing about the check rules. Use a real base model for anything but plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import prompt

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
#: ChatML, the format the Qwen instruct models use.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def make_tiny_model(out_dir: str | Path, rows: list[dict], system: str) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = [system] + [prompt.user_message(r["input"]) for r in rows]
    corpus += [prompt.target_text(r["target"]) for r in rows if r.get("target")]

    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=2000, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"])
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(out)

    config = Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
        tie_word_embeddings=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model = Qwen2ForCausalLM(config)
    model.generation_config = GenerationConfig(eos_token_id=tokenizer.eos_token_id,
                                               pad_token_id=tokenizer.pad_token_id)
    model.save_pretrained(out)
    (out / "TINY_RANDOM_MODEL.json").write_text(json.dumps(
        {"warning": "random weights, dataset-trained tokenizer: for pipeline checks only"}) + "\n")
    return out
