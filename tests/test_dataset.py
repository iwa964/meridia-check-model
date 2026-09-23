"""The real dataset, when this checkout has it: it must validate, and every record must be
accounted for. Skipped until training_files/dice_rolling_train.json exists here."""

import json
from pathlib import Path

import pytest

from check_model.config import load_config
from check_model.prepare import build

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "training_files" / "dice_rolling_train.json"

pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="training_files/dice_rolling_train.json not present")


def test_real_dataset_validates_and_accounts_for_every_record(monkeypatch):
    monkeypatch.chdir(ROOT)
    rows, report = build(load_config(ROOT / "configs" / "sft_example.yaml"))
    assert report.errors == [], report.errors
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    in_source = {r["id"] for key, v in data.items() if key.endswith("examples") for r in v}
    counted = (set(report.trainable) | {e["id"] for e in report.eval_only} | {e["id"] for e in report.pending}
               | {e["id"] for e in report.skipped} | {e["id"] for e in report.unsupported})
    assert counted == in_source
    assert {r.split for r in rows} <= {"train", "val", "test"}
