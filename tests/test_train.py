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
    from check_model.evaluate import input_fingerprint

    assert saved["examples"]["train_fingerprints"] == sorted(input_fingerprint(r["input"]) for r in tiny["rows"][:3])
    assert saved["examples"]["train_texts"] == [r["similarity_text"] for r in tiny["rows"][:3]]
    assert all(saved["examples"]["train_texts"])
    assert saved["global_steps"] == 2 == manifest["global_steps"]
    assert saved["prompt"]["system_prompt"] == tiny["system"]

    model = CheckModel(run, max_new_tokens=8)
    result = model.predict(tiny["rows"][0]["input"])
    # Random weights: the reply is whatever it is; what matters is that it came back scored.
    assert set(result) == {"valid", "decision", "game_request", "note", "errors", "raw_output"}
    assert isinstance(result["raw_output"], str)
    assert result["valid"] == (result["decision"] is not None)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            model.predict_many([tiny["rows"][0]["input"]], batch_size=bad)


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
def test_serving_dtype_on_any_cuda_device(precision, device, dtype, monkeypatch):
    import torch

    import check_model.infer as infer

    monkeypatch.setattr(infer, "_bf16_supported", lambda device: True)
    assert infer.serving_dtype(precision, device) is getattr(torch, dtype)


def test_bf16_run_on_a_gpu_without_bf16_serves_in_fp32(monkeypatch):
    import torch

    import check_model.infer as infer

    monkeypatch.setattr(infer, "_bf16_supported", lambda device: False)
    assert infer.serving_dtype("bf16", "cuda:0") is torch.float32
    assert infer.serving_dtype("fp16", "cuda:0") is torch.float16


def test_rows_past_the_base_models_context_are_refused(tiny, tmp_path):
    import shutil

    from check_model.train import train

    base = tmp_path / "short-context"
    shutil.copytree(tiny["dir"], base)
    cfg = json.loads((base / "config.json").read_text())
    cfg["max_position_embeddings"] = 64
    (base / "config.json").write_text(json.dumps(cfg))
    config = copy.deepcopy(load_config(None))
    config["model"]["base_model"] = str(base)
    config["train"]["max_seq_length"] = 100000  # the configured limit alone would accept every row
    with pytest.raises(ValueError, match="exceeds the base model's context of 64"):
        train(config, tiny["rows"][:1], [], run_dir=tmp_path / "run", source_files=[], max_steps=1)


def test_each_tiny_smoke_run_keeps_its_own_base(tmp_path, monkeypatch):
    import yaml

    from check_model.__main__ import main

    root = Path(__file__).resolve().parent.parent
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(yaml.safe_dump({
        "data": {"sources": [str(SUBSET)], "catalog": str(root / "catalog" / "meridia_catalog.json"),
                 "prepared_dir": str(tmp_path / "build" / "data")},
        "train": {"output_dir": str(tmp_path / "runs"), "per_device_train_batch_size": 2},
        "inference": {"batch_size": 1},
        "smoke": {"max_steps": 1}}), encoding="utf-8")
    import check_model.evaluate as evaluate

    batch_sizes = []
    original = evaluate.evaluate_rows

    def spy(*args, **kwargs):
        batch_sizes.append(kwargs.get("batch_size"))
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluate, "evaluate_rows", spy)
    main(["smoke", "--config", str(cfg), "--tiny"])
    main(["smoke", "--config", str(cfg), "--tiny"])
    assert batch_sizes == [1, 1]  # the configured inference batch size, not evaluate_rows' default
    bases = [json.loads(m.read_text())["base_model"]["name_or_path"]
             for m in sorted((tmp_path / "runs").glob("*/run_manifest.json"))]
    assert len(bases) == 2 and bases[0] != bases[1]
    assert all(Path(b, "config.json").exists() for b in bases)

    # A smoke run is prepared like any other, so its split hashes are recorded and scoring it
    # on unchanged data needs no override; a run recorded without them is refused, by name.
    run = sorted((tmp_path / "runs").iterdir())[0]
    main(["evaluate", "--run", str(run), "--split", "val"])
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["examples"]["split_sha256"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="val split: trained on no recorded hash, now [0-9a-f]{12}"):
        main(["evaluate", "--run", str(run), "--split", "val"])

    # Each base lives inside its own run, so the run still loads once the regenerable
    # prepared data around it is cleaned away.
    runs = sorted((tmp_path / "runs").iterdir())
    assert [Path(b).parent for b in bases] == runs
    import shutil

    shutil.rmtree(tmp_path / "build")
    query = tmp_path / "query.json"
    query.write_text(json.dumps({"scene": "A cliff.", "player_action": "I climb it."}), encoding="utf-8")
    main(["predict", "--run", str(runs[1]), "--input", str(query)])


def test_evaluate_uses_the_runs_own_data_and_refuses_another(tiny, tmp_path, monkeypatch):
    import yaml

    from check_model.__main__ import main

    root = Path(__file__).resolve().parent.parent

    def config(name, **data):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump({
            "data": {"sources": [str(SUBSET)], "catalog": str(root / "catalog" / "meridia_catalog.json"),
                     "prepared_dir": str(tmp_path / name), "val_fraction": 0.5, **data},
            "model": {"base_model": str(tiny["dir"])},
            "train": {"output_dir": str(tmp_path / "runs"), "max_steps": 1, "per_device_train_batch_size": 2}}),
            encoding="utf-8")
        return str(path)

    main(["train", "--config", config("en")])
    (run,) = (tmp_path / "runs").iterdir()
    monkeypatch.chdir(tmp_path)  # no configs/sft_example.yaml here: only the run's config can be used
    main(["evaluate", "--run", str(run), "--split", "val"])
    assert (run / "eval" / "val" / "metrics.json").exists()

    main(["prepare", "--config", config("zh", language="zh")])
    import check_model.infer as infer

    def no_model(*args, **kwargs):  # a refusal must come before any base-model download
        raise AssertionError("the model was loaded before the data checks")

    monkeypatch.setattr(infer, "CheckModel", no_model)
    with pytest.raises(SystemExit, match="split with \\{'language': 'zh'"):
        main(["evaluate", "--run", str(run), "--split", "val", "--config", config("zh", language="zh")])


def test_serving_honours_trust_remote_code_for_the_tokenizer(tiny, tmp_path, monkeypatch):
    import transformers

    from check_model.infer import CheckModel
    from check_model.train import train

    config = copy.deepcopy(load_config(None))
    config["model"]["base_model"] = str(tiny["dir"])
    run = tmp_path / "run"
    train(config, tiny["rows"][:2], [], run_dir=run, source_files=[], max_steps=1)
    manifest = json.loads((run / "run_manifest.json").read_text())
    manifest["base_model"]["trust_remote_code"] = True
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    seen = []
    original = transformers.AutoTokenizer.from_pretrained

    def spy(*args, **kwargs):
        seen.append(kwargs.get("trust_remote_code"))
        return original(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", spy)
    CheckModel(run, max_new_tokens=4)
    assert seen == [True]


def test_evaluate_refuses_a_changed_data_revision_unless_asked(tiny, tmp_path, monkeypatch):
    import yaml

    from check_model.__main__ import main

    root = Path(__file__).resolve().parent.parent
    source = tmp_path / "source.json"
    source.write_text(SUBSET.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "data": {"sources": [str(source)], "catalog": str(root / "catalog" / "meridia_catalog.json"),
                 "prepared_dir": str(tmp_path / "prepared"), "val_fraction": 0.5},
        "model": {"base_model": str(tiny["dir"])},
        "train": {"output_dir": str(tmp_path / "runs"), "max_steps": 1, "per_device_train_batch_size": 2}}),
        encoding="utf-8")
    main(["train", "--config", str(cfg)])
    (run,) = (tmp_path / "runs").iterdir()

    data = json.loads(source.read_text(encoding="utf-8"))  # edited in place: same path, new content
    data["examples"][0]["scene"]["en"] += " The wall has been freshly whitewashed."
    source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    main(["prepare", "--config", str(cfg)])
    with pytest.raises(SystemExit, match="not the revision the run was trained on"):
        main(["evaluate", "--run", str(run), "--split", "val"])
    main(["evaluate", "--run", str(run), "--split", "val", "--allow-data-change"])
    metrics = json.loads((run / "eval" / "val" / "metrics.json").read_text())
    assert metrics["data_revision"]["matches_training"] is False
    assert metrics["data_revision"]["changed_sources"][0]["path"] == str(source)


def test_a_new_sibling_of_a_training_row_is_not_scored(tiny, tmp_path):
    """A near-duplicate added after training, with an id that sorts first, takes over the group
    key and can move the group into validation; the training row is excluded by id, and its new
    sibling must be excluded too."""
    import yaml

    from check_model.__main__ import main

    root = Path(__file__).resolve().parent.parent
    source = tmp_path / "source.json"
    source.write_text(SUBSET.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "data": {"sources": [str(source)], "catalog": str(root / "catalog" / "meridia_catalog.json"),
                 "prepared_dir": str(tmp_path / "prepared"), "val_fraction": 0.5},
        "model": {"base_model": str(tiny["dir"])},
        "train": {"output_dir": str(tmp_path / "runs"), "max_steps": 1, "per_device_train_batch_size": 2}}),
        encoding="utf-8")
    main(["train", "--config", str(cfg)])
    (run,) = (tmp_path / "runs").iterdir()
    trained = json.loads((run / "run_manifest.json").read_text())["examples"]["train_ids"]
    original = json.loads(source.read_text(encoding="utf-8"))

    landed = False
    for n, tid in enumerate(trained):
        data = json.loads(json.dumps(original))
        twin = next(json.loads(json.dumps(r)) for r in data["examples"] if r["id"] == tid)
        twin["id"] = f"dice_train_000000_{n}"  # sorts before every real id
        for lang in ("en", "zh"):
            twin["scene"][lang] += " (later)"
        data["examples"].append(twin)
        source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        main(["prepare", "--config", str(cfg)])
        val_ids = [json.loads(line)["id"] for line in (tmp_path / "prepared" / "val.jsonl").read_text().splitlines()]
        if twin["id"] in val_ids:
            landed = True
            break
    assert landed, "no sibling moved into validation; the scenario under test did not arise"
    main(["evaluate", "--run", str(run), "--split", "val", "--allow-data-change"])
    metrics = json.loads((run / "eval" / "val" / "metrics.json").read_text())
    scored = [json.loads(line)["id"] for line in (run / "eval" / "val" / "predictions.jsonl").read_text().splitlines()]
    assert twin["id"] in metrics["excluded_related_rows"] and twin["id"] not in scored


def test_evaluate_refuses_split_contents_that_changed_with_the_same_sources(tiny, tmp_path, monkeypatch):
    import yaml

    import check_model.adapter as adapter
    from check_model.__main__ import main

    root = Path(__file__).resolve().parent.parent
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "data": {"sources": [str(SUBSET)], "catalog": str(root / "catalog" / "meridia_catalog.json"),
                 "prepared_dir": str(tmp_path / "prepared"), "val_fraction": 0.5},
        "model": {"base_model": str(tiny["dir"])},
        "train": {"output_dir": str(tmp_path / "runs"), "max_steps": 1, "per_device_train_batch_size": 2}}),
        encoding="utf-8")
    main(["train", "--config", str(cfg)])
    (run,) = (tmp_path / "runs").iterdir()
    assert set(json.loads((run / "run_manifest.json").read_text())["examples"]["split_sha256"]) == {"train", "val", "test"}

    original = adapter.Row.to_json  # a changed adapter: same source bytes, different prepared rows
    monkeypatch.setattr(adapter.Row, "to_json", lambda self: dict(original(self), adapter_version=2))
    main(["prepare", "--config", str(cfg)])
    with pytest.raises(SystemExit, match="val split: trained on"):
        main(["evaluate", "--run", str(run), "--split", "val"])
    main(["evaluate", "--run", str(run), "--split", "val", "--allow-data-change"])
    revision = json.loads((run / "eval" / "val" / "metrics.json").read_text())["data_revision"]
    assert revision["matches_training"] is False and revision["changed_sources"] == []
    assert {c["split"] for c in revision["changed_splits"]} >= {"val"}


    # A split file edited after prepare, its provenance untouched: the file itself is hashed.
    monkeypatch.setattr(adapter.Row, "to_json", original)
    main(["prepare", "--config", str(cfg)])
    main(["evaluate", "--run", str(run), "--split", "val"])  # back to the trained contents
    val = tmp_path / "prepared" / "val.jsonl"
    provenance = (tmp_path / "prepared" / "splits_provenance.json").read_text()
    val.write_text(val.read_text().splitlines()[0] + "\n", encoding="utf-8")
    assert (tmp_path / "prepared" / "splits_provenance.json").read_text() == provenance
    with pytest.raises(SystemExit, match="val split: trained on [0-9a-f]{12}, now [0-9a-f]{12}"):
        main(["evaluate", "--run", str(run), "--split", "val"])

    # A split file cut off mid-line: the hash check refuses before anything is parsed, and with
    # --allow-data-change the damaged line is named instead of ending in a traceback.
    val.write_text(val.read_text() + '{"id": "dice_tr', encoding="utf-8")
    with pytest.raises(SystemExit, match="val split: trained on"):
        main(["evaluate", "--run", str(run), "--split", "val"])
    with pytest.raises(SystemExit, match=r"val\.jsonl:2: not a prepared row"):
        main(["evaluate", "--run", str(run), "--split", "val", "--allow-data-change"])


def test_the_manifest_records_every_training_scenario_member(tiny, tmp_path):
    from check_model.train import train

    config = copy.deepcopy(load_config(None))
    config["model"]["base_model"] = str(tiny["dir"])
    rows = [dict(r, group_members=[r["id"], "dice_train_000029"]) for r in tiny["rows"][:2]]
    train(config, rows, [], run_dir=tmp_path / "run", source_files=[], max_steps=1)
    saved = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert saved["examples"]["train_scenario_ids"] == sorted({r["id"] for r in rows} | {"dice_train_000029"})
