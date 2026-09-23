import hashlib
from pathlib import Path

import pytest
from helpers import clone, record

from check_model.adapter import load_rows

SUBSET = Path(__file__).resolve().parent / "fixtures" / "dice_subset.json"


def load(catalog, *paths, lang="en"):
    return load_rows([str(p) for p in paths], catalog, lang)


def messages(report):
    return [(e["id"], e["message"]) for e in report.errors]


def test_every_record_lands_in_exactly_one_bucket(catalog, subset):
    rows, report = load(catalog, SUBSET)
    assert report.errors == []
    assert sorted(report.trainable) == [
        "dice_train_000001", "dice_train_000002", "dice_train_000021", "dice_train_000028", "dice_train_000033"]
    assert [e["id"] for e in report.eval_only] == ["dice_train_000038"]
    assert [e["id"] for e in report.pending] == ["dice_train_000047"]
    assert [e["id"] for e in report.skipped] == ["dice_train_000015"]
    reasons = {e["id"]: e["reason"] for e in report.unsupported}
    assert reasons["dice_train_000040"].startswith("optional roll")
    assert reasons["dice_train_000044"].startswith("pair check")
    assert reasons["dice_train_000019"].startswith("conditional")
    assert reasons["dice_train_000029"].startswith("parameterized")
    in_source = sum(len(v) for v in subset.values() if isinstance(v, list))
    bucketed = (len(report.trainable) + len(report.eval_only) + len(report.pending)
                + len(report.skipped) + len(report.unsupported))
    assert bucketed == in_source == len(report.all_ids)


def test_targets_and_inputs_come_from_the_annotation(catalog):
    rows, _ = load(catalog, SUBSET)
    by_id = {r.id: r for r in rows}
    legacy = by_id["dice_train_000001"]
    assert legacy.target == {"roll_required": True, "checks": [
        {"kind": "skill", "name": "Climbing", "difficulty": "extreme"}]}
    assert set(legacy.input) == {"scene", "player_action"}
    observed = by_id["dice_train_000021"]
    assert observed.target["checks"] == [{"kind": "attribute", "name": "STR", "difficulty": "success"}]
    assert set(observed.input) == {"scene", "observed_event", "runtime_state"}
    # The corrected label, not the one in annotation_history.
    assert by_id["dice_train_000033"].target["checks"][0]["difficulty"] == "success"
    alternatives = by_id["dice_train_000038"]
    assert alternatives.target is None
    assert [o[0]["name"] for o in alternatives.reference["options"]] == ["Economics", "Mathematics"]


def test_language_selects_the_text(catalog):
    rows, _ = load(catalog, SUBSET, lang="zh")
    row = next(r for r in rows if r.id == "dice_train_000001")
    assert row.input["player_action"] == "我抓住石缝，徒手爬上墙顶，再翻进院子。"


def test_source_is_read_not_written(catalog):
    before = hashlib.sha256(SUBSET.read_bytes()).hexdigest()
    _, report = load(catalog, SUBSET)
    assert hashlib.sha256(SUBSET.read_bytes()).hexdigest() == before == report.sources[0]["sha256"]


def test_no_roll_label_is_trainable(catalog, subset, write_source):
    # The policy's no-roll form (annotation_policy.no_roll); the dataset has none yet, so
    # 000040's optional roll is reduced to it here.
    annotation = record(subset, "dice_train_000040")["annotation"]
    del annotation["roll_optional"], annotation["optional_roll"]
    rows, report = load(catalog, write_source(subset))
    row = next(r for r in rows if r.id == "dice_train_000040")
    assert row.target == {"roll_required": False, "checks": []}
    assert report.errors == []


@pytest.mark.parametrize("mutate, rid, expected", [
    (lambda r: r["annotation"]["checks"][0].update(name="maintenance"), "dice_train_000001", "unknown skill"),
    (lambda r: r["annotation"]["checks"][0].update(difficulty="normal"), "dice_train_000001", "difficulty must be"),
    (lambda r: r["annotation"]["checks"][0].pop("roll_system"), "dice_train_000001", "no roll_system"),
    (lambda r: r["annotation"].update(raw_response=None), "dice_train_000001", "no raw_response"),
    (lambda r: r["annotation"].update(roll_required="yes"), "dice_train_000001", "roll_required must be"),
    (lambda r: r["scene"].pop("en"), "dice_train_000001", "missing scene.en"),
    (lambda r: r.update(annotation=None), "dice_train_000001", "has no annotation"),
    (lambda r: r.update(scenario_scope="meridia"), "dice_train_000001", "scenario_scope must be"),
    (lambda r: r["annotation"]["checks"].append(dict(r["annotation"]["checks"][0])) or
     r["annotation"]["checks"].append(dict(r["annotation"]["checks"][0])), "dice_train_000001", "3 checks"),
])
def test_invalid_records_are_named(catalog, subset, write_source, mutate, rid, expected):
    mutate(record(subset, rid))
    _, report = load(catalog, write_source(subset))
    assert any(i == rid and expected in m for i, m in messages(report)), messages(report)


@pytest.mark.parametrize("mutate, expected", [
    (lambda r: r["annotation"]["checks"][0].update(name="Void Sense"), "special skill check"),
    (lambda r: r["annotation"]["checks"][0].update(roll_system="bidirectional"), "roll_system"),
])
def test_unsupported_forms_are_reported_not_errors(catalog, subset, write_source, mutate, expected):
    mutate(record(subset, "dice_train_000001"))
    _, report = load(catalog, write_source(subset))
    assert report.errors == []
    assert any(e["id"] == "dice_train_000001" and expected in e["reason"] for e in report.unsupported)


def test_duplicate_id_is_an_error(catalog, subset, write_source):
    subset["examples"].append(clone(subset, "dice_train_000002", "dice_train_000001"))
    _, report = load(catalog, write_source(subset))
    assert any(i == "dice_train_000001" and "duplicate id" in m for i, m in messages(report))


def test_duplicate_id_across_sources_is_an_error(catalog, subset, write_source):
    _, report = load(catalog, SUBSET, write_source(subset))
    assert len([m for _, m in messages(report) if "duplicate id" in m]) == 12


def test_duplicate_input_is_an_error(catalog, subset, write_source):
    subset["examples"].append(clone(subset, "dice_train_000001", "dice_train_copy"))
    _, report = load(catalog, write_source(subset))
    assert ("dice_train_copy", "same input as dice_train_000001") in messages(report)


def test_unknown_section_and_schema_are_errors(catalog, subset, write_source):
    subset["surprise_examples"] = []
    _, report = load(catalog, write_source(subset))
    assert any("unknown section 'surprise_examples'" in m for _, m in messages(report))
    subset.pop("surprise_examples")
    subset["schema_version"] = "2.0"
    _, report = load(catalog, write_source(subset))
    assert any("schema_version '2.0'" in m for _, m in messages(report))


def test_catalog_version_mismatch_is_a_warning(catalog, subset, write_source):
    subset["skill_catalog"]["blob_sha"] = "0" * 40
    _, report = load(catalog, write_source(subset))
    assert report.errors == [] and any("0" * 40 in w for w in report.warnings)
