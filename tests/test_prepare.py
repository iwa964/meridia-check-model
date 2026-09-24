import json
from pathlib import Path

import pytest
import yaml
from helpers import record

from check_model.__main__ import main
from check_model.config import load_config
from check_model.infer import base_revision, check_format
from check_model.prepare import build
from check_model.train import load_dtype, resolve_precision

ROOT = Path(__file__).resolve().parent.parent
SUBSET = ROOT / "tests" / "fixtures" / "dice_subset.json"


def config_file(tmp_path, source, **data):
    cfg = {"data": {"sources": [str(source)], "catalog": str(ROOT / "catalog" / "meridia_catalog.json"),
                    "prepared_dir": str(tmp_path / "prepared"), **data}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_errors_leave_the_last_clean_splits_alone(tmp_path, subset, write_source, capsys):
    cfg = config_file(tmp_path, SUBSET)
    main(["prepare", "--config", cfg])
    train = tmp_path / "prepared" / "train.jsonl"
    clean = train.read_text(encoding="utf-8")

    record(subset, "dice_train_000001")["annotation"]["checks"][0]["name"] = "maintenance"
    broken = config_file(tmp_path, write_source(subset))
    with pytest.raises(SystemExit):
        main(["prepare", "--config", broken])
    assert train.read_text(encoding="utf-8") == clean
    report = json.loads((tmp_path / "prepared" / "report.json").read_text(encoding="utf-8"))
    assert report["errors"] and report["errors"][0]["id"] == "dice_train_000001"


def test_rows_that_changed_split_are_reported(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "val.jsonl").write_text(json.dumps({"id": "dice_train_000001"}) + "\n", encoding="utf-8")
    config = load_config(config_file(tmp_path, SUBSET, val_fraction=0.0))
    _, report = build(config)
    assert "dice_train_000001 moved from val to train since the last prepare" in report.warnings


def test_serving_uses_the_recorded_base_commit():
    assert base_revision({"revision": None, "resolved_commit": "abc123"}) == "abc123"
    assert base_revision({"revision": "v1", "resolved_commit": None}) == "v1"


def test_a_run_in_another_prompt_format_is_refused():
    with pytest.raises(ValueError, match="prompt format"):
        check_format({"prompt": {"format_version": "0"}})


@pytest.mark.parametrize("precision, lora, dtype", [
    ("bf16", True, "bfloat16"), ("fp16", True, "float16"), ("fp16", False, "float32"), ("fp32", True, "float32"),
])
def test_load_dtype(precision, lora, dtype):
    assert load_dtype(precision, lora) == dtype


def test_an_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="fp116"):
        resolve_precision("fp116")


def test_a_failed_prepare_does_not_vouch_for_the_kept_splits(tmp_path, subset, write_source):
    from check_model.__main__ import _data_mismatch

    main(["prepare", "--config", config_file(tmp_path, SUBSET)])  # source A: clean, splits written
    record(subset, "dice_train_000001")["annotation"]["checks"][0]["name"] = "maintenance"
    source_b = write_source(subset)
    with pytest.raises(SystemExit):  # source B: fails, splits from A are kept
        main(["prepare", "--config", config_file(tmp_path, source_b)])
    prepared = {"data": {"prepared_dir": str(tmp_path / "prepared")}}
    data_a = load_config(config_file(tmp_path, SUBSET))["data"]
    trained_on_b = {"config": {"data": dict(data_a, sources=[source_b])}, "prompt": {"language": "en"}}
    assert "trained on" in _data_mismatch(prepared, trained_on_b, [])
    trained_on_a = {"config": {"data": data_a}, "prompt": {"language": "en"}}
    assert _data_mismatch(prepared, trained_on_a, []) is None


def test_a_different_split_of_the_same_sources_is_refused(tmp_path):
    from check_model.__main__ import _data_mismatch

    main(["prepare", "--config", config_file(tmp_path, SUBSET, split_seed=1)])
    prepared = {"data": {"prepared_dir": str(tmp_path / "prepared")}}
    trained = {"config": {"data": load_config(config_file(tmp_path, SUBSET, split_seed=0))["data"]},
               "prompt": {"language": "en"}}
    assert "the validation rows differ" in _data_mismatch(prepared, trained, [])


@pytest.mark.parametrize("section, key, value", [
    ("model", "trust_remote_code", "false"),   # a truthy string, not False
    ("train", "learning_rate", "2e-4"),        # PyYAML reads 2e-4 without a dot as a string
    ("train", "max_steps", 1.5),
    ("data", "sources", "training_files/dice_rolling_train.json"),
])
def test_config_values_must_have_their_defaults_type(tmp_path, section, key, value):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({section: {key: value}}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"config key '{section}.{key}' must be"):
        load_config(path)


def test_documented_null_and_list_values_are_accepted(tmp_path):
    path = tmp_path / "ok.yaml"
    path.write_text(yaml.safe_dump({"data": {"near_duplicate_threshold": None},
                                    "model": {"revision": "abc123"},
                                    "lora": {"target_modules": ["q_proj", "v_proj"]}}), encoding="utf-8")
    config = load_config(path)
    assert config["data"]["near_duplicate_threshold"] is None
    assert config["lora"]["target_modules"] == ["q_proj", "v_proj"]
    assert load_config(ROOT / "configs" / "sft_example.yaml") and load_config(ROOT / "configs" / "smoke.yaml")
