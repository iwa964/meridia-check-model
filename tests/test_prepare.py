import json
import re
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


@pytest.mark.parametrize("section, key, value", [
    ("data", "val_fraction", -0.1), ("data", "val_fraction", float("nan")), ("data", "val_fraction", 1.5),
    ("data", "near_duplicate_threshold", 2.0), ("train", "warmup_ratio", 1.0),
])
def test_values_out_of_range_are_refused(tmp_path, section, key, value):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({section: {key: value}}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"config key '{section}.{key}' must be (in|finite)"):
        load_config(path)


@pytest.mark.parametrize("config, tokenizer, expected", [
    ({"max_position_embeddings": 4096}, None, 4096),
    ({"n_positions": 1024}, None, 1024),                   # GPT-2 style
    ({"max_seq_len": 2048}, None, 2048),
    ({"text_config": {"max_position_embeddings": 8192}}, None, 8192),  # nested text model
    ({}, {"model_max_length": 512}, 512),                 # only the tokenizer says
    ({}, {"model_max_length": int(1e30)}, None),          # the tokenizer's "no limit" sentinel
    ({}, None, None),
])
def test_context_limit_is_read_from_the_usual_fields(config, tokenizer, expected):
    from types import SimpleNamespace

    from check_model.train import context_limit_of

    def ns(d):
        return SimpleNamespace(**{k: ns(v) if isinstance(v, dict) else v for k, v in d.items()})

    assert context_limit_of(ns(config), ns(tokenizer) if tokenizer else None) == expected


@pytest.mark.parametrize("document", ["[]", "false", "0", "- a\n- b"])
def test_a_config_that_is_not_a_mapping_is_refused(tmp_path, document):
    path = tmp_path / "bad.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(path)


def test_an_empty_config_file_means_every_default(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == load_config(None)


@pytest.mark.parametrize("payload, message", [
    ('"a scene"', "got str"), ("null", "got NoneType"), ("3", "got int"), ("{not json", "not valid JSON")])
def test_predict_refuses_input_that_is_not_a_query_or_a_list(tmp_path, payload, message):
    from check_model.__main__ import main

    source = tmp_path / "queries.json"
    source.write_text(payload, encoding="utf-8")
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")
    # Refused before any model is loaded: the run directory does not even exist.
    with pytest.raises(SystemExit, match=message):
        main(["predict", "--run", str(tmp_path / "no-run"), "--config", str(config), "--input", str(source)])


def test_a_corrupt_previous_split_file_does_not_block_rebuilding_it(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "val.jsonl").write_text(json.dumps({"id": "dice_train_000001"}) + "\n" + '{"id": "dice_tr',
                                        encoding="utf-8")  # a write cut off mid-line
    config = load_config(config_file(tmp_path, SUBSET, val_fraction=0.0))
    _, report = build(config)
    assert "dice_train_000001 moved from val to train since the last prepare" in report.warnings
    assert any("1 line(s) of the previous prepare are unreadable" in w for w in report.warnings)


@pytest.mark.parametrize("section, key, value", [
    ("train", "learning_rate", 0.0), ("train", "learning_rate", -1.0e-4), ("train", "num_train_epochs", 0),
    ("lora", "r", 0), ("lora", "alpha", 0), ("lora", "dropout", 1.0), ("inference", "max_new_tokens", 0)])
def test_values_that_would_train_or_generate_nothing_are_refused(tmp_path, section, key, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({section: {key: value}}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"'{section}.{key}' must be"):
        load_config(path)


@pytest.mark.parametrize("section, key, value", [
    ("train", "weight_decay", ".inf"), ("train", "weight_decay", "-0.1"), ("train", "learning_rate", ".nan"),
    ("data", "val_fraction", ".nan")])
def test_non_finite_and_negative_float_settings_are_refused(tmp_path, section, key, value):
    path = tmp_path / "config.yaml"
    path.write_text(f"{section}:\n  {key}: {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"'{section}.{key}' must be"):
        load_config(path)


@pytest.mark.parametrize("section, key, value", [
    ("data", "sources", [None]), ("data", "sources", [""]), ("data", "extra_groups", [None]),
    ("data", "extra_groups", [["dice_train_000001", 2]]), ("smoke", "example_ids", [3]),
    ("lora", "target_modules", [None])])
def test_list_settings_with_malformed_elements_are_refused(tmp_path, section, key, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({section: {key: value}}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"'{section}.{key}' must be a list of"):
        load_config(path)


def test_a_config_key_repeated_in_one_mapping_is_refused(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  learning_rate: 1.0e-4\ntrain:\n  precision: fp32\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate config key 'train' at line 3"):
        load_config(path)


@pytest.mark.parametrize("section, key", [("data", "sources"), ("lora", "target_modules")])
def test_a_list_that_must_name_something_cannot_be_empty(tmp_path, section, key):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({section: {key: []}}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"'{section}.{key}' must list at least one entry"):
        load_config(path)


def test_predict_refuses_a_query_with_a_repeated_key(tmp_path):
    source = tmp_path / "queries.json"
    source.write_text('{"scene": "A cliff.", "scene": "A river.", "player_action": "I climb."}', encoding="utf-8")
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"duplicate key\(s\) \['scene'\]"):
        main(["predict", "--run", str(tmp_path / "no-run"), "--config", str(config), "--input", str(source)])


@pytest.mark.parametrize("section, key", [("lora", "target_modules"), ("model", "base_model"), ("data", "prepared_dir")])
def test_a_blank_string_setting_is_refused(tmp_path, section, key):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({section: {key: "  "}}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"'{section}.{key}' must not be blank"):
        load_config(path)


#: A prepared row with every field evaluation reads (the shape Row.to_json writes).
VALID_ROW = {"id": "r1", "source": "s.json", "split": "val", "group": "r1",
             "input": {"scene": "s", "player_action": "a"},
             "reference": {"roll_required": False, "options": []}}


@pytest.mark.parametrize("line", ["null", '"a string"', "[1, 2]"])
def test_a_prepared_line_that_is_not_an_object_is_named(tmp_path, line):
    from check_model.prepare import read_split

    (tmp_path / "val.jsonl").write_text(json.dumps(VALID_ROW) + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"val\.jsonl:2: not a prepared row \(a JSON object is required"):
        read_split(tmp_path, "val")


@pytest.mark.parametrize("row, problem", [
    ({}, "missing ['id', 'source', 'split', 'group', 'input', 'reference']"),
    ({**VALID_ROW, "input": "a scene"}, "wrong type for ['input']"),
    ({k: v for k, v in VALID_ROW.items() if k != "group"}, "missing ['group']")])
def test_a_prepared_row_missing_what_evaluation_reads_is_named(tmp_path, row, problem):
    from check_model.prepare import read_split

    (tmp_path / "val.jsonl").write_text(json.dumps(VALID_ROW) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"val\.jsonl:2: not a prepared row \(" + re.escape(problem)):
        read_split(tmp_path, "val")


@pytest.mark.parametrize("reference, problem", [
    ({}, "reference.roll_required must be true or false"),
    ({"roll_required": True, "options": [{"kind": "skill"}]}, "reference.options must be a list of option lists"),
    ({"roll_required": True, "options": [[{"kind": "skill", "name": "Climbing"}]]},
     "each reference check needs string kind, name and difficulty")])
def test_a_prepared_reference_of_the_wrong_shape_is_named(tmp_path, reference, problem):
    from check_model.prepare import read_split

    (tmp_path / "val.jsonl").write_text(json.dumps({**VALID_ROW, "reference": reference}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"val\.jsonl:1: not a prepared row \(" + re.escape(problem)):
        read_split(tmp_path, "val")


@pytest.mark.parametrize("payload", ["[]", '[{"scene": ""}, {"player_action": "a"}]'])
def test_predict_loads_no_model_when_no_query_can_run(tmp_path, monkeypatch, capsys, payload):
    import check_model.infer as infer
    from check_model.prompt import PROMPT_FORMAT_VERSION

    def no_model(*args, **kwargs):
        raise AssertionError("the model was loaded for a batch with nothing to generate")

    monkeypatch.setattr(infer, "CheckModel", no_model)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"prompt": {"format_version": PROMPT_FORMAT_VERSION}}),
                                           encoding="utf-8")
    source = tmp_path / "queries.json"
    source.write_text(payload, encoding="utf-8")
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")
    main(["predict", "--run", str(run), "--config", str(config), "--input", str(source)])
    results = json.loads(capsys.readouterr().out)
    assert all(not r["valid"] and r["errors"][0].startswith("bad query: ") for r in results)


@pytest.mark.parametrize("reference, problem", [
    ({"roll_required": True, "options": []}, "a roll-required reference needs options of 1 to 1 check(s)"),
    ({"roll_required": True, "options": [[]]}, "a roll-required reference needs options of 1 to 1 check(s)"),
    ({"roll_required": False, "options": [[{"kind": "skill", "name": "Climbing", "difficulty": "hard"}]]},
     "a no-roll reference cannot hold checks")])
def test_a_contradictory_prepared_reference_is_named(tmp_path, reference, problem):
    from check_model.prepare import read_split

    (tmp_path / "val.jsonl").write_text(json.dumps({**VALID_ROW, "reference": reference}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"val\.jsonl:1: not a prepared row \(" + re.escape(problem)):
        read_split(tmp_path, "val")


def test_a_prepared_input_that_is_not_a_query_is_named(tmp_path):
    from check_model.prepare import read_split

    (tmp_path / "val.jsonl").write_text(json.dumps({**VALID_ROW, "input": {}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"val\.jsonl:1: not a prepared row \(input: scene must be a non-empty string"):
        read_split(tmp_path, "val")


def test_every_row_prepare_writes_passes_read_split(tmp_path, subset, write_source):
    from helpers import clone

    from check_model.prepare import read_split

    # Plus a plain no-roll label (000040's without its optional-roll part): its reference is the
    # {roll_required: false, options: [[]]} form, which neither the fixture nor the dataset has yet.
    no_roll = clone(subset, "dice_train_000040", "dice_train_000940")
    no_roll["scene"]["en"] = "A variation. " + no_roll["scene"]["en"]
    for key in ("roll_optional", "optional_roll"):
        no_roll["annotation"].pop(key)
    subset["examples"].append(no_roll)
    main(["prepare", "--config", config_file(tmp_path, write_source(subset), val_fraction=0.5)])
    rows = [r for s in ("train", "val", "test") for r in read_split(tmp_path / "prepared", s)]
    assert {r["id"]: r["reference"] for r in rows}["dice_train_000940"] == {"roll_required": False, "options": [[]]}
