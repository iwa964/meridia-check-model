"""The run configuration: YAML over these defaults. An unknown key is an error, so a typo
cannot silently fall back to a default."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "run_name": "check-sft",
    "seed": 42,
    "data": {
        "sources": ["training_files/dice_rolling_train.json"],
        "catalog": "catalog/meridia_catalog.json",
        "language": "en",  # en | zh: which text of the bilingual source the prompt shows
        "prepared_dir": "build/data",
        "val_fraction": 0.2,
        "split_seed": 0,  # separate from `seed`, so changing the training seed keeps the split
        "near_duplicate_threshold": 0.28,  # null turns similarity grouping off; see configs/sft_example.yaml
        "extra_groups": [],  # [[id, id, ...], ...] scenario variations to keep together
    },
    "model": {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": None,  # pin a hub commit for a reproducible run
        "trust_remote_code": False,
    },
    "lora": {
        "enabled": True,
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": "all-linear",
    },
    "train": {
        "output_dir": "runs",
        "learning_rate": 2.0e-4,
        "num_train_epochs": 5,
        "max_steps": -1,  # > 0 overrides epochs
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 1024,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "precision": "auto",  # auto | bf16 | fp16 | fp32
        "gradient_checkpointing": False,
    },
    "inference": {
        "max_new_tokens": 64,
        "batch_size": 8,
    },
    "smoke": {
        # Explicit ids, or [] for the first `num_examples` trainable rows.
        "example_ids": [],
        "num_examples": 8,
        "max_steps": 10,
    },
}


def _merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key not in base:
            raise ValueError(f"unknown config key {path + key!r}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"config key {path + key!r} must be a mapping")
            out[key] = _merge(base[key], value, path + key + ".")
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULTS)
    return _merge(DEFAULTS, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
