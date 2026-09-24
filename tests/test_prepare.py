import json
import sys
import re
from pathlib import Path

import pytest
import yaml
from helpers import record, valid_manifest

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


@pytest.mark.parametrize("value", [-1, 2**32])
def test_a_seed_numpy_cannot_take_is_refused(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(f"seed: {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"config key 'seed' must be in \[0, 4294967295\]"):
        load_config(path)
    path.write_text(f"seed: {2**32 - 1}\n", encoding="utf-8")
    assert load_config(path)["seed"] == 2**32 - 1


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


def test_predict_refuses_input_nested_past_the_recursion_limit(tmp_path):
    source = tmp_path / "queries.json"
    source.write_text('{"scene": "A cliff.", "player_action": "I climb.", "runtime_state": '
                      + '{"a":' * 5000 + "1" + "}" * 5000 + "}", encoding="utf-8")
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON .*nested more than 200 levels deep"):
        main(["predict", "--run", str(tmp_path / "no-run"), "--config", str(config), "--input", str(source)])


@pytest.mark.parametrize("section, key", [("lora", "target_modules"), ("model", "base_model"), ("data", "prepared_dir")])
def test_a_blank_string_setting_is_refused(tmp_path, section, key):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({section: {key: "  "}}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"'{section}.{key}' must not be blank"):
        load_config(path)


#: A prepared row with every field Row.to_json writes.
VALID_ROW = {"id": "r1", "source": "s.json", "section": "examples", "scope": "general", "split": "val",
             "group": "r1", "lang": "en", "input": {"scene": "s", "player_action": "a"},
             "reference": {"roll_required": False, "options": [[]]}, "target": None,
             "similarity_text": "s a", "links": [], "group_members": ["r1"]}


@pytest.mark.parametrize("line", ["null", '"a string"', "[1, 2]"])
def test_a_prepared_line_that_is_not_an_object_is_named(tmp_path, line):
    from check_model.prepare import read_split

    (tmp_path / "val.jsonl").write_text(json.dumps(VALID_ROW) + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"val\.jsonl:2: not a prepared row \(a JSON object is required"):
        read_split(tmp_path, "val")


@pytest.mark.parametrize("row, problem", [
    ({}, "missing ['id', 'source', 'section', 'scope', 'split', 'group', 'lang', 'input', 'reference', "
         "'similarity_text', 'links', 'group_members', 'target']"),
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

    def no_model(*args, **kwargs):
        raise AssertionError("the model was loaded for a batch with nothing to generate")

    monkeypatch.setattr(infer, "CheckModel", no_model)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps(valid_manifest()), encoding="utf-8")
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


@pytest.mark.parametrize("line, problem", [
    (b'{"id": "r1", "id": "r2"}', "duplicate key(s) ['id']"),
    (b'{"id": "r\xff"}', "not UTF-8")])
def test_a_split_line_is_decoded_strictly(tmp_path, line, problem):
    from check_model.prepare import read_split

    (tmp_path / "val.jsonl").write_bytes(line + b"\n")
    with pytest.raises(ValueError, match=r"val\.jsonl:1: not a prepared row \(" + re.escape(problem)):
        read_split(tmp_path, "val")


@pytest.mark.parametrize("change, problem", [
    ({"links": None}, "missing ['links']"),
    ({"group_members": None}, "missing ['group_members']"),
    ({"similarity_text": None}, "missing ['similarity_text']"),
    ({"group_members": [""]}, "group_members must hold ids"),
    ({"group": "", "group_members": []}, "group is blank"),
    ({"group_members": []}, "group_members must include the row's id and its group"),
    ({"group": "r0", "group_members": ["r1"]}, "group_members must include the row's id and its group"),
    ({"split": "train"}, "split 'train' in the val file")])
def test_a_prepared_row_without_its_relation_metadata_is_named(tmp_path, change, problem):
    from check_model.prepare import read_split

    row = dict(VALID_ROW)
    for key, value in change.items():
        if value is None:
            row.pop(key)
        else:
            row[key] = value
    (tmp_path / "val.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"val\.jsonl:1: not a prepared row \(" + re.escape(problem)):
        read_split(tmp_path, "val")


def test_repeated_ids_are_refused_within_and_across_splits(tmp_path):
    from check_model.prepare import duplicate_ids, read_split

    (tmp_path / "val.jsonl").write_text((json.dumps(VALID_ROW) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match=r"val\.jsonl:2: not a prepared row \(duplicate id 'r1' \(also at line 1\)"):
        read_split(tmp_path, "val")
    assert duplicate_ids({"train": [{"id": "r1"}], "val": [{"id": "r1"}, {"id": "r2"}], "test": []}) == ["r1"]


def test_reference_labels_are_checked_against_the_run_catalog(tmp_path, catalog):
    from check_model.prepare import read_split

    fireball = {"kind": "spell", "name": "Fireball", "difficulty": "impossible"}
    row = {**VALID_ROW, "reference": {"roll_required": True, "options": [[fireball]]}}
    (tmp_path / "val.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert read_split(tmp_path, "val")  # without a catalog only the shape is checked
    with pytest.raises(ValueError, match=r"val\.jsonl:1: not a prepared row \(reference label .*kind must be one of"):
        read_split(tmp_path, "val", catalog)


def test_an_edited_system_prompt_is_refused():
    from check_model.prompt import PROMPT_FORMAT_VERSION, prompt_sha256

    prompt_record = {"format_version": PROMPT_FORMAT_VERSION, "system_prompt": "p", "system_prompt_sha256": prompt_sha256("p")}
    check_format({"prompt": prompt_record})
    with pytest.raises(ValueError, match="does not match its recorded SHA-256"):
        check_format({"prompt": {**prompt_record, "system_prompt": "p, edited"}})


@pytest.mark.parametrize("content, problem", [
    ('{"sources": [', "is unreadable"), ('{"sources": [{"path": "a"}], "split_config": {}}', "is not a provenance record"),
    ("[]", "is not a provenance record")])
def test_a_damaged_provenance_file_is_a_refusal(tmp_path, content, problem):
    from check_model.__main__ import _data_mismatch

    (tmp_path / "splits_provenance.json").write_text(content, encoding="utf-8")
    config = {"data": {"prepared_dir": str(tmp_path)}}
    assert problem in _data_mismatch(config, {"config": {"data": {"sources": []}}, "prompt": {"language": "en"}})


@pytest.mark.parametrize("split, scope", [("val", "game_specific"), ("test", "general")])
def test_each_split_holds_only_its_scope(tmp_path, split, scope):
    from check_model.prepare import read_split

    row = {**VALID_ROW, "split": split, "scope": scope}
    (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"scope {scope!r} in the {split} file")):
        read_split(tmp_path, split)


def test_a_reference_check_with_an_extra_field_is_named(tmp_path):
    from check_model.prepare import read_split

    check = {"kind": "skill", "name": "Climbing", "difficulty": "hard", "comment": "legacy"}
    row = {**VALID_ROW, "reference": {"roll_required": True, "options": [[check]]}}
    (tmp_path / "val.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="and no other field"):
        read_split(tmp_path, "val")


def test_predict_needs_no_training_stack_when_nothing_can_run(tmp_path, monkeypatch, capsys):
    import check_model.__main__ as cli

    monkeypatch.setattr(cli, "_require_training_stack", lambda: sys.exit("peft is not installed"))
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps(valid_manifest()), encoding="utf-8")
    source = tmp_path / "queries.json"
    source.write_text('[{"scene": ""}]', encoding="utf-8")
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")
    main(["predict", "--run", str(run), "--config", str(config), "--input", str(source)])
    assert json.loads(capsys.readouterr().out)[0]["errors"][0].startswith("bad query: ")


def test_predict_names_an_input_that_is_not_utf8(tmp_path):
    source = tmp_path / "queries.json"
    source.write_bytes(b'{"scene": "\xff", "player_action": "a"}')
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        main(["predict", "--run", str(tmp_path / "no-run"), "--config", str(config), "--input", str(source)])


def test_a_blank_similarity_text_is_named(tmp_path):
    from check_model.prepare import read_split

    (tmp_path / "val.jsonl").write_text(json.dumps({**VALID_ROW, "similarity_text": "  "}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"not a prepared row \(similarity_text is blank"):
        read_split(tmp_path, "val")



@pytest.mark.parametrize("change, problem", [
    (lambda c: c["difficulties"].append("legendary"), "legendary"),
    (lambda c: c["kinds"].append("item"), "catalog kinds"),
])
def test_prepare_refuses_a_catalog_the_prompt_cannot_describe(tmp_path, change, problem):
    catalog = json.loads((ROOT / "catalog" / "meridia_catalog.json").read_text(encoding="utf-8"))
    change(catalog)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    cfg = config_file(tmp_path, SUBSET, catalog=str(path))
    with pytest.raises(ValueError, match=problem):
        build(load_config(cfg))
    with pytest.raises(SystemExit, match=f"data.catalog {re.escape(str(path))}: .*{problem}"):
        main(["prepare", "--config", cfg])
    assert not (tmp_path / "prepared" / "train.jsonl").exists()


def _repeat_first_skill_as_special(text: str) -> str:
    catalog = json.loads(text)
    catalog["skills"].append(dict(catalog["skills"][0], initial=""))
    return json.dumps(catalog)


@pytest.mark.parametrize("edit, problem", [
    (lambda text: text.replace('"skills": [', '"skills": [], "skills": [', 1), r"duplicate key\(s\) \['skills'\]"),
    (_repeat_first_skill_as_special, r"not a catalog: skills lists \['Investigation'\] more than once"),
    (lambda text: text.replace('"attributes": [', '"attributes": ["STR", ', 1),
     r"not a catalog: attributes lists \['STR'\] more than once"),
    (lambda text: text.replace('"attributes"', '"attribute"', 1), "not a catalog: expected an object with exactly the keys"),
    (lambda text: text.replace('"initial"', '"initial_value"', 1), "not a catalog: skills must be"),
    (lambda text: text.replace('"blob_sha"', '"sha"', 1), "not a catalog: source must record"),
])
def test_prepare_refuses_an_ambiguous_or_malformed_catalog(tmp_path, edit, problem):
    from check_model.catalog import load_catalog

    text = (ROOT / "catalog" / "meridia_catalog.json").read_text(encoding="utf-8")
    edited = edit(text)
    assert edited != text
    path = tmp_path / "catalog.json"
    path.write_text(edited, encoding="utf-8")
    with pytest.raises(ValueError, match=problem):
        load_catalog(path)
    with pytest.raises(SystemExit, match=f"data.catalog {re.escape(str(path))}: .*{problem}"):
        main(["prepare", "--config", config_file(tmp_path, SUBSET, catalog=str(path))])
    assert not (tmp_path / "prepared" / "train.jsonl").exists()


@pytest.mark.parametrize("value", [0, -2])
def test_max_steps_is_minus_one_or_positive(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(f"train:\n  max_steps: {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"'train.max_steps' must be -1 \(train num_train_epochs\) or above 0"):
        load_config(path)
    for fine in (-1, 5):
        path.write_text(f"train:\n  max_steps: {fine}\n", encoding="utf-8")
        assert load_config(path)["train"]["max_steps"] == fine


def test_a_run_manifest_with_a_repeated_key_is_refused(tmp_path):
    from check_model.infer import read_manifest

    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text('{"base_model": {"name_or_path": "a"}, "base_model": {"name_or_path": "b"}}',
                                           encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate key\(s\) \['base_model'\]"):
        read_manifest(run)
    with pytest.raises(SystemExit, match=r"--run .*duplicate key\(s\) \['base_model'\]"):
        main(["evaluate", "--run", str(run), "--split", "val"])
    query = tmp_path / "query.json"
    query.write_text('{"scene": "A cliff.", "player_action": "I climb."}', encoding="utf-8")
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"--run .*duplicate key\(s\) \['base_model'\]"):
        main(["predict", "--run", str(run), "--config", str(config), "--input", str(query)])


def test_prepare_with_a_source_holding_no_records_keeps_the_last_splits(tmp_path, subset, write_source):
    from check_model.adapter import SECTION_KIND

    source = write_source(subset)
    cfg = config_file(tmp_path, source)
    main(["prepare", "--config", cfg])
    before = {p.name: p.read_bytes() for p in (tmp_path / "prepared").glob("*.jsonl")}
    assert before["train.jsonl"]
    for section in SECTION_KIND:
        subset.pop(section, None)
    write_source(subset)
    with pytest.raises(SystemExit):
        main(["prepare", "--config", cfg])
    assert {p.name: p.read_bytes() for p in (tmp_path / "prepared").glob("*.jsonl")} == before


def test_duplicate_inputs_are_found_within_and_across_splits():
    from check_model.prepare import duplicate_inputs

    a = {"id": "a", "input": {"scene": "A cliff.", "player_action": "I climb."}}
    copy_ = {"id": "b", "input": {"player_action": "I  CLIMB.", "scene": "A cliff."}}  # order, space, case
    other = {"id": "c", "input": {"scene": "A river.", "player_action": "I swim."}}
    assert duplicate_inputs({"train": [a], "val": [copy_, other]}) == [["a", "b"]]
    assert duplicate_inputs({"val": [a, copy_]}) == [["a", "b"]]
    assert duplicate_inputs({"train": [a], "val": [other]}) == []


@pytest.mark.parametrize("damage, problem", [
    (lambda m: m.clear(), "precision is missing"),
    (lambda m: m.update(precision="garbage"), "precision has the wrong type or value"),
    (lambda m: m.update(precision="auto"), "precision has the wrong type or value"),
    (lambda m: m["base_model"].update(trust_remote_code="yes"), "base_model.trust_remote_code has the wrong type"),
    (lambda m: m["examples"].update(train_links={"a": "b"}), "examples.train_links has the wrong type"),
    (lambda m: m["config"]["data"].pop("split_seed"), r"config: the recorded config lacks \['data.split_seed'\]"),
    (lambda m: m["config"]["train"].update(learning_rate=0), "config: config key 'train.learning_rate' must be above 0"),
])
def test_a_damaged_run_manifest_is_a_named_refusal(tmp_path, damage, problem):
    from check_model.infer import read_manifest

    run = tmp_path / "run"
    run.mkdir()
    manifest = valid_manifest()
    assert read_manifest_ok(run, manifest)
    damage(manifest)
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not a run manifest .*" + problem):
        read_manifest(run)
    query = tmp_path / "query.json"
    query.write_text('{"scene": "A cliff.", "player_action": "I climb."}', encoding="utf-8")
    for argv in (["predict", "--run", str(run), "--input", str(query)],
                 ["evaluate", "--run", str(run), "--split", "val"]):
        with pytest.raises(SystemExit, match=f"--run {re.escape(str(run))}: .*not a run manifest .*{problem}"):
            main(argv)


def read_manifest_ok(run, manifest) -> bool:
    from check_model.infer import read_manifest

    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return read_manifest(run) == manifest


@pytest.mark.parametrize("text, where", [("train:\n  1: 2\n", "in train"), ("true: 1\n", "in the top level")])
def test_a_config_key_that_is_not_a_string_is_named(tmp_path, text, where):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=f"config keys must be strings, got .* {where}"):
        load_config(path)


def test_prepare_refuses_a_source_holding_an_unpaired_surrogate(tmp_path, subset):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(subset), encoding="utf-8")
    cfg = config_file(tmp_path, source)
    main(["prepare", "--config", cfg])
    before = {p.name: p.read_bytes() for p in (tmp_path / "prepared").glob("*.jsonl")}
    record(subset, "dice_train_000001")["scene"]["en"] = "\ud800"  # json.dumps writes the escape
    source.write_text(json.dumps(subset), encoding="utf-8")
    assert "\\ud800" in source.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["prepare", "--config", cfg])
    report = json.loads((tmp_path / "prepared" / "report.json").read_text(encoding="utf-8"))
    assert any("unpaired surrogate" in e["message"] for e in report["errors"])
    assert {p.name: p.read_bytes() for p in (tmp_path / "prepared").glob("*.jsonl")} == before


def test_an_integer_too_large_for_a_float_is_a_config_error_not_a_crash(tmp_path):
    path = tmp_path / "config.yaml"
    huge = 10 ** 310
    path.write_text(f"seed: {huge}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"config key 'seed' must be in \[0, 4294967295\]"):
        load_config(path)
    path.write_text(f"train:\n  learning_rate: {huge}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'train.learning_rate' must be a finite number, got an integer of 311 digits"):
        load_config(path)
    path.write_text(f"train:\n  per_device_train_batch_size: {huge}\n", encoding="utf-8")
    assert load_config(path)["train"]["per_device_train_batch_size"] == huge  # an int is exact


@pytest.mark.parametrize("name", ["report.json", "train.jsonl", "splits_provenance.json"])
def test_prepare_never_writes_over_a_source_or_the_catalog(tmp_path, subset, name):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    source = prepared / name
    source.write_text(json.dumps(subset), encoding="utf-8")
    before = source.read_bytes()
    cfg = config_file(tmp_path, source)
    with pytest.raises(SystemExit, match=rf"refusing to prepare: .*would overwrite the input\(s\) \[.*{re.escape(name)}"):
        main(["prepare", "--config", cfg])
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="would overwrite"):
        build(load_config(cfg))
    # The catalog too.
    catalog = prepared / "report.json"
    catalog.write_bytes((ROOT / "catalog" / "meridia_catalog.json").read_bytes())
    with pytest.raises(SystemExit, match="would overwrite"):
        main(["prepare", "--config", config_file(tmp_path, SUBSET, catalog=str(catalog))])


@pytest.mark.parametrize("groups", [[[]], [["dice_train_000001"]]])
def test_an_extra_group_that_cannot_join_two_records_is_refused(tmp_path, groups):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"data": {"extra_groups": groups}}), encoding="utf-8")
    with pytest.raises(ValueError, match="'data.extra_groups' must be a list of lists of two or more id strings"):
        load_config(path)


def test_a_hub_base_is_not_served_from_a_local_path_of_the_same_name(tmp_path, monkeypatch):
    from check_model.infer import check_local_base

    base = {"name_or_path": "org/model", "local_sha256": None}
    monkeypatch.chdir(tmp_path)
    check_local_base(base, "org/model")  # nothing of that name here: the hub is used
    (tmp_path / "org" / "model").mkdir(parents=True)
    with pytest.raises(ValueError, match="was trained from the hub, but a local path"):
        check_local_base(base, "org/model")
