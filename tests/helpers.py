import copy


def record(data, rid):
    for section, value in data.items():
        if isinstance(value, list):
            for r in value:
                if r.get("id") == rid:
                    return r
    raise KeyError(rid)


def clone(data, rid, new_id):
    r = copy.deepcopy(record(data, rid))
    r["id"] = new_id
    return r


def valid_manifest(system_prompt: str = "p") -> dict:
    """The smallest run manifest read_manifest accepts: every field serving and evaluation read."""
    from check_model.config import load_config
    from check_model.prompt import PROMPT_FORMAT_VERSION, prompt_sha256

    return {
        "precision": "fp32", "adapter": "lora", "catalog_sha256": "0" * 64,
        "model_files": {"adapter_config.json": "0" * 64}, "global_steps": 1, "metrics": {},
        "config": load_config(None),
        "prompt": {"format_version": PROMPT_FORMAT_VERSION, "system_prompt": system_prompt,
                   "system_prompt_sha256": prompt_sha256(system_prompt), "language": "en"},
        "base_model": {"name_or_path": "base", "revision": None, "resolved_commit": None,
                       "local_sha256": None, "in_run": None, "trust_remote_code": False},
        "examples": {"train_ids": [], "val_ids": [], "train_fingerprints": [], "train_texts": [],
                     "train_scenario_ids": [], "split_sha256": {}, "train_links": {}, "source_files": []},
    }
