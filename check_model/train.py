"""LoRA SFT with the base model's own chat template.

* Each row is rendered with `tokenizer.apply_chat_template`: the prompt (system + user,
  with the generation prompt) and the full conversation (plus the assistant target). The
  prompt must be a token prefix of the full conversation, or the run stops -- that is what
  makes "loss on the answer only" exact rather than approximate.
* Labels are -100 over the prompt, so loss is computed only on the target response (and
  the template's end-of-turn token after it, which teaches the model to stop).
* Nothing is truncated. A row longer than `max_seq_length` stops the run with its id.
* The run directory holds everything needed to reload and serve: the adapter, the
  tokenizer, the system prompt and catalog the model was trained with, the training log,
  and a manifest recording the base model, parameters and every example id used.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

from . import prompt
from .catalog import load_catalog
from .evaluate import input_fingerprint

IGNORE_INDEX = -100


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def encode(tokenizer, system: str, row: dict, max_seq_length: int, context_limit: int | None = None) -> dict:
    """Token ids and loss labels for one row. Raises ValueError naming the row on anything
    that would make the labels wrong or the answer truncated."""
    prompt_messages = prompt.messages(system, row["input"])
    full_messages = prompt.messages(system, row["input"], row["target"])
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError(f"{row['id']}: the chat template renders the prompt differently with and without "
                         "the answer; loss masking would be wrong for this base model")
    # The template already carries any BOS/special tokens it needs.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{row['id']}: prompt tokens are not a prefix of the full conversation's tokens")
    answer = prompt.target_text(row["target"])
    completion = tokenizer.decode(full_ids[len(prompt_ids):])
    if answer not in completion:
        raise ValueError(f"{row['id']}: the target is not recoverable from the completion tokens")
    if len(full_ids) > max_seq_length:
        raise ValueError(f"{row['id']}: {len(full_ids)} tokens (prompt {len(prompt_ids)}) exceeds "
                         f"max_seq_length {max_seq_length}; raise it rather than truncate the answer")
    if context_limit is not None and len(full_ids) > context_limit:
        raise ValueError(f"{row['id']}: {len(full_ids)} tokens exceeds the base model's context of "
                         f"{context_limit}; shorten the input or use a model with a longer context")
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels, "id": row["id"]}


def encode_all(tokenizer, system: str, rows: list[dict], max_seq_length: int,
               context_limit: int | None = None) -> tuple[list[dict], dict]:
    encoded, errors = [], []
    for row in rows:
        try:
            encoded.append(encode(tokenizer, system, row, max_seq_length, context_limit))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("cannot encode training rows:\n  " + "\n  ".join(errors))
    lengths = [len(e["input_ids"]) for e in encoded]
    stats = {"rows": len(lengths), "max_tokens": max(lengths, default=0),
             "mean_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0,
             "max_seq_length": max_seq_length, "model_context": context_limit}
    return encoded, stats


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [IGNORE_INDEX] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


PRECISIONS = ("auto", "bf16", "fp16", "fp32")


def resolve_precision(requested: str) -> str:
    if requested not in PRECISIONS:
        raise ValueError(f"train.precision must be one of {PRECISIONS}, got {requested!r}")
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


def load_dtype(precision: str, lora: bool) -> str:
    """The dtype the base weights are loaded in. bf16 loads bf16. fp16 loads fp16 only under
    LoRA, where the frozen base can sit in half precision while PEFT keeps the trainable adapter
    in fp32; a full fine-tune under fp16 AMP needs fp32 master weights."""
    if precision == "bf16":
        return "bfloat16"
    if precision == "fp16" and lora:
        return "float16"
    return "float32"


#: Where model configs keep their context window, in order of preference. Architectures differ.
CONTEXT_FIELDS = ("max_position_embeddings", "n_positions", "max_seq_len", "max_sequence_length",
                  "seq_length", "n_ctx")
#: Tokenizers report "no limit" as a huge sentinel (1e30); anything above this is not a limit.
_NO_LIMIT = 10_000_000


def context_limit_of(config, tokenizer=None) -> int | None:
    """A model's context window from its config -- or from a nested text config, or, failing
    both, the tokenizer's model_max_length -- or None when nothing states one."""
    for source in (config, getattr(config, "text_config", None)):
        for field in CONTEXT_FIELDS:
            value = getattr(source, field, None) if source is not None else None
            if isinstance(value, int) and 0 < value < _NO_LIMIT:
                return value
    value = getattr(tokenizer, "model_max_length", None)
    return value if isinstance(value, int) and 0 < value < _NO_LIMIT else None


def model_context_limit(model_cfg: dict, tokenizer=None) -> int | None:
    """The base model's context window, read from its config before any row is encoded."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_cfg["base_model"], revision=model_cfg["revision"],
                                        trust_remote_code=model_cfg["trust_remote_code"])
    return context_limit_of(config, tokenizer)


def load_tokenizer(name_or_path: str, revision: str | None, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision, trust_remote_code=trust_remote_code)
    if tokenizer.chat_template is None:
        raise ValueError(f"{name_or_path} has no chat template; use an instruct/chat base model")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def train(config: dict, train_rows: list[dict], val_rows: list[dict], *, run_dir: str | Path,
          source_files: list[dict], max_steps: int | None = None, mode: str = "train",
          split_sha256: dict | None = None, run_dir_created: bool = False) -> dict:
    """Trains, saves and returns the manifest. `train_rows` must all have a target;
    `source_files` is the prepare report's `sources` (paths and sha256 of the data)."""
    import peft
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    model_cfg, lora_cfg, train_cfg = config["model"], config["lora"], config["train"]
    run_dir = Path(run_dir)
    # Two runs never write into one directory: created here exclusively, or already created
    # exclusively by the caller for this run (smoke --tiny puts its base model inside it).
    run_dir.mkdir(parents=True, exist_ok=run_dir_created)
    transformers.set_seed(config["seed"])

    catalog = load_catalog(config["data"]["catalog"])
    system = prompt.system_prompt(catalog)
    tokenizer = load_tokenizer(model_cfg["base_model"], model_cfg["revision"], model_cfg["trust_remote_code"])
    context = model_context_limit(model_cfg, tokenizer)
    train_set, train_stats = encode_all(tokenizer, system, train_rows, train_cfg["max_seq_length"], context)
    val_trainable = [r for r in val_rows if r.get("target") is not None]
    val_set, val_stats = encode_all(tokenizer, system, val_trainable, train_cfg["max_seq_length"], context)

    precision = resolve_precision(train_cfg["precision"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"], revision=model_cfg["revision"],
        trust_remote_code=model_cfg["trust_remote_code"],
        dtype=getattr(torch, load_dtype(precision, lora_cfg["enabled"])),
    )
    if train_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if lora_cfg["enabled"]:
        model = peft.get_peft_model(model, peft.LoraConfig(
            task_type="CAUSAL_LM", r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"], target_modules=lora_cfg["target_modules"],
        ))

    steps = max_steps if max_steps is not None else train_cfg["max_steps"]
    args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        max_steps=steps,
        warmup_steps=train_cfg["warmup_ratio"],  # a float < 1 is a ratio of total steps
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=config["seed"],
        data_seed=config["seed"],
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_set,
                      data_collator=Collator(tokenizer.pad_token_id))
    started = time.time()
    result = trainer.train()
    metrics = dict(result.metrics)
    if val_set:
        metrics.update(trainer.evaluate(eval_dataset=val_set))

    scratch = run_dir / "trainer"
    if scratch.is_dir() and not any(scratch.iterdir()):
        scratch.rmdir()  # save_strategy "no": the Trainer's scratch dir stays empty
    model_dir = run_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (run_dir / "catalog.json").write_text(json.dumps(catalog.to_json(), ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    with (run_dir / "train_log.jsonl").open("w", encoding="utf-8") as f:
        for entry in trainer.state.log_history:
            f.write(json.dumps(entry) + "\n")

    base_config = model.get_base_model().config if lora_cfg["enabled"] else model.config
    manifest = {
        "mode": mode,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_seconds": round(time.time() - started, 1),
        "base_model": {
            "name_or_path": model_cfg["base_model"],
            "revision": model_cfg["revision"],
            "resolved_commit": getattr(base_config, "_commit_hash", None),
            "trust_remote_code": model_cfg["trust_remote_code"],
        },
        "adapter": "lora" if lora_cfg["enabled"] else None,
        "config": config,
        "precision": precision,
        "max_steps_used": steps,
        "global_steps": trainer.state.global_step,
        "prompt": {
            "format_version": prompt.PROMPT_FORMAT_VERSION,
            "system_prompt": system,
            "system_prompt_sha256": prompt.prompt_sha256(system),
            "language": config["data"]["language"],
        },
        "catalog_source": catalog.source,
        "examples": {
            "train_ids": [r["id"] for r in train_rows],
            "train_fingerprints": sorted(input_fingerprint(r["input"]) for r in train_rows),
            # Both languages, so evaluation can still recognise a near-duplicate of a training row
            # after that row has been removed or rewritten.
            "train_texts": [r.get("similarity_text", "") for r in train_rows],
            "val_ids": [r["id"] for r in val_rows],
            "source_files": source_files,
            # The prepared split files this run was trained from (splits_provenance.json).
            "split_sha256": split_sha256 or {},
            # Explicit links from each training row, so a row linked to one stays excluded from
            # scoring even after the training row is removed from the data.
            "train_links": {r["id"]: r.get("links", []) for r in train_rows},
            # Every id grouped with a training row at prepare time, transitively and including
            # records that were never trainable, so a later-annotated member stays excluded.
            "train_scenario_ids": sorted({i for r in train_rows for i in r.get("group_members", [])}),
        },
        "token_lengths": {"train": train_stats, "val": val_stats},
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "repo_commit": _git("rev-parse", "HEAD"),
            # Uncommitted code changes mean repo_commit alone does not reproduce this run.
            "repo_dirty": bool(_git("status", "--porcelain", "--", "check_model", "configs", "catalog")),
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    return manifest
