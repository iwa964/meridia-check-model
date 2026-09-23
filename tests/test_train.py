"""Training-stack tests on the tiny random model: they need torch/transformers/peft
(requirements-train.txt) and are skipped without them. They check the plumbing -- masking,
length checks, save and reload -- never what the model answers."""

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

from check_model import prompt  # noqa: E402
from check_model.adapter import load_rows  # noqa: E402
from check_model.config import load_config  # noqa: E402
from check_model.train import IGNORE_INDEX, encode  # noqa: E402

SUBSET = Path(__file__).resolve().parent / "fixtures" / "dice_subset.json"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from check_model.catalog import load_catalog
    from check_model.tiny import make_tiny_model

    catalog = load_catalog(Path(__file__).resolve().parent.parent / "catalog" / "meridia_catalog.json")
    rows, _ = load_rows([str(SUBSET)], catalog, "en")
    rows = [r.to_json() for r in rows if r.target is not None]
    system = prompt.system_prompt(catalog)
    model_dir = make_tiny_model(tmp_path_factory.mktemp("tiny"), rows, system)
    from transformers import AutoTokenizer

    return {"dir": model_dir, "rows": rows, "system": system,
            "tokenizer": AutoTokenizer.from_pretrained(model_dir)}


def test_loss_is_on_the_answer_only(tiny):
    row = tiny["rows"][0]
    enc = encode(tiny["tokenizer"], tiny["system"], row, 4096)
    first = next(i for i, label in enumerate(enc["labels"]) if label != IGNORE_INDEX)
    assert all(label == IGNORE_INDEX for label in enc["labels"][:first])
    answer = tiny["tokenizer"].decode(enc["labels"][first:])
    assert answer.startswith(prompt.target_text(row["target"]))
    assert "<|im_end|>" in answer  # the model is taught to stop


def test_too_long_is_refused_not_truncated(tiny):
    with pytest.raises(ValueError, match=tiny["rows"][0]["id"] + ".*exceeds max_seq_length"):
        encode(tiny["tokenizer"], tiny["system"], tiny["rows"][0], 32)


def test_train_save_reload_predict(tiny, tmp_path):
    from check_model.infer import CheckModel
    from check_model.train import train

    config = copy.deepcopy(load_config(None))
    config["model"]["base_model"] = str(tiny["dir"])
    config["train"].update(per_device_train_batch_size=2, gradient_accumulation_steps=1)
    run = tmp_path / "run"
    manifest = train(config, tiny["rows"][:3], tiny["rows"][3:], run_dir=run, source_files=[], max_steps=2)
    for name in ("model/adapter_config.json", "model/tokenizer_config.json", "catalog.json",
                 "train_log.jsonl", "run_manifest.json"):
        assert (run / name).exists(), name
    saved = json.loads((run / "run_manifest.json").read_text())
    assert saved["examples"]["train_ids"] == [r["id"] for r in tiny["rows"][:3]]
    assert saved["global_steps"] == 2 == manifest["global_steps"]
    assert saved["prompt"]["system_prompt"] == tiny["system"]

    model = CheckModel(run, max_new_tokens=8)
    result = model.predict(tiny["rows"][0]["input"])
    # Random weights: the reply is whatever it is; what matters is that it came back scored.
    assert set(result) == {"valid", "decision", "game_request", "note", "errors", "raw_output"}
    assert isinstance(result["raw_output"], str)
    assert result["valid"] == (result["decision"] is not None)


def test_query_past_the_context_window_is_refused(tiny, tmp_path):
    from check_model.infer import CheckModel
    from check_model.train import train

    config = copy.deepcopy(load_config(None))
    config["model"]["base_model"] = str(tiny["dir"])
    run = tmp_path / "run"
    train(config, tiny["rows"][:2], [], run_dir=run, source_files=[], max_steps=1)
    model = CheckModel(run, max_new_tokens=8)
    long_query = dict(tiny["rows"][0]["input"], scene="stone wall " * 5000)
    result = model.predict(long_query)
    assert result["valid"] is False and "query too long" in result["errors"][0]
    with pytest.raises(FileExistsError):  # a run directory is never shared
        train(config, tiny["rows"][:2], [], run_dir=run, source_files=[], max_steps=1)


def test_fp16_lora_keeps_the_adapter_in_fp32(tiny):
    import peft
    import torch
    from transformers import AutoModelForCausalLM

    from check_model.train import load_dtype

    model = AutoModelForCausalLM.from_pretrained(tiny["dir"], dtype=getattr(torch, load_dtype("fp16", True)))
    model = peft.get_peft_model(model, peft.LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8,
                                                       target_modules="all-linear"))
    assert {p.dtype for p in model.parameters() if p.requires_grad} == {torch.float32}
    assert {p.dtype for p in model.parameters() if not p.requires_grad} == {torch.float16}


@pytest.mark.parametrize("precision, device, dtype", [
    ("bf16", "cuda:1", "bfloat16"), ("fp16", "cuda:0", "float16"), ("bf16", "cuda", "bfloat16"),
    ("bf16", "cpu", "float32"), ("fp32", "cuda:1", "float32"),
])
def test_serving_dtype_on_any_cuda_device(precision, device, dtype):
    import torch

    from check_model.infer import serving_dtype

    assert serving_dtype(precision, device) is getattr(torch, dtype)


def test_each_tiny_smoke_run_keeps_its_own_base(tmp_path, monkeypatch):
    import yaml

    from check_model.__main__ import main

    root = Path(__file__).resolve().parent.parent
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(yaml.safe_dump({
        "data": {"sources": [str(SUBSET)], "catalog": str(root / "catalog" / "meridia_catalog.json"),
                 "prepared_dir": str(tmp_path / "build" / "data")},
        "train": {"output_dir": str(tmp_path / "runs"), "per_device_train_batch_size": 2},
        "smoke": {"max_steps": 1}}), encoding="utf-8")
    main(["smoke", "--config", str(cfg), "--tiny"])
    main(["smoke", "--config", str(cfg), "--tiny"])
    bases = [json.loads(m.read_text())["base_model"]["name_or_path"]
             for m in sorted((tmp_path / "runs").glob("*/run_manifest.json"))]
    assert len(bases) == 2 and bases[0] != bases[1]
    assert all(Path(b, "config.json").exists() for b in bases)
