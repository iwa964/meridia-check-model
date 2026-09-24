"""The run configuration: YAML over these defaults. An unknown key is an error, so a typo
cannot silently fall back to a default."""

from __future__ import annotations

import copy
import math
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
        "near_duplicate_threshold": 0.38,  # null turns similarity grouping off; see configs/sft_example.yaml
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


#: Keys whose value may also be null, and the one string key that may also be a list.
NULLABLE = {"data.near_duplicate_threshold", "model.revision"}
STR_OR_LIST = {"lora.target_modules"}


def _check_type(path: str, default, value) -> None:
    """A value must have its default's type: a quoted "false" is a truthy string, not False, and
    PyYAML reads 2e-4 (no dot) as a string -- both would otherwise pass through unnoticed."""
    if value is None and (default is None or path in NULLABLE):
        return
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif isinstance(default, list):
        ok = isinstance(value, list)
    elif isinstance(default, str):
        ok = isinstance(value, str) or (path in STR_OR_LIST and isinstance(value, list))
    else:  # a null default takes a string (model.revision)
        ok = isinstance(value, str)
    if not ok:
        expected = "string" if default is None else type(default).__name__
        raise ValueError(f"config key {path!r} must be a {expected}, got {value!r}")


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
            _check_type(path + key, base[key], value)
            out[key] = value
    return out


#: Values whose type is not enough: a negative or NaN val_fraction would put every group in
#: training and leave no validation cohort, with nothing said. (key, low, high, high inclusive)
RANGES = (("data.val_fraction", 0.0, 1.0, True),
          ("data.near_duplicate_threshold", 0.0, 1.0, True),
          ("train.warmup_ratio", 0.0, 1.0, False),
          ("lora.dropout", 0.0, 1.0, False))


#: Values that must be above zero. Zero is accepted by the libraries and trains nothing (a zero
#: learning rate or epoch count, a LoRA rank or alpha of zero) or fails far from the config.
POSITIVE = ("train.learning_rate", "train.num_train_epochs", "train.per_device_train_batch_size",
            "train.gradient_accumulation_steps", "train.max_seq_length", "train.logging_steps",
            "lora.r", "lora.alpha", "inference.max_new_tokens", "inference.batch_size",
            "smoke.num_examples", "smoke.max_steps")


def _check_ranges(config: dict) -> None:
    for key in POSITIVE:
        section, name = key.split(".")
        value = config[section][name]
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"config key {key!r} must be above 0, got {value!r}")
    for key, low, high, inclusive in RANGES:
        section, name = key.split(".")
        value = config[section][name]
        if value is None:
            continue
        inside = math.isfinite(value) and low <= value and (value <= high if inclusive else value < high)
        if not inside:
            bound = "]" if inclusive else ")"
            raise ValueError(f"config key {key!r} must be in [{low}, {high}{bound}, got {value!r}")


def load_config(path: str | Path | None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULTS)
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if document is None:  # an empty file: every default
        document = {}
    if not isinstance(document, dict):
        # [] or false would otherwise read as "no overrides" and train the defaults.
        raise ValueError(f"{path}: the config must be a mapping of sections, got {type(document).__name__}")
    config = _merge(DEFAULTS, document)
    _check_ranges(config)
    return config
