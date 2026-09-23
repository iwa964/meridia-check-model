import json
from pathlib import Path

import pytest

from check_model.catalog import load_catalog

ROOT = Path(__file__).resolve().parent.parent
SUBSET = ROOT / "tests" / "fixtures" / "dice_subset.json"


@pytest.fixture
def catalog():
    return load_catalog(ROOT / "catalog" / "meridia_catalog.json")


@pytest.fixture
def subset():
    """A fresh copy of the fixture dataset, to mutate per test."""
    return json.loads(SUBSET.read_text(encoding="utf-8"))


@pytest.fixture
def write_source(tmp_path):
    def write(data, name="source.json"):
        path = tmp_path / name
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return str(path)
    return write
