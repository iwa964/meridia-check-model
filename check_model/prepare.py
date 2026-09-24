"""`prepare`: validate the sources and write the split files plus a report.

Outputs (under `data.prepared_dir`): `train.jsonl`, `val.jsonl`, `test.jsonl`, one row per
example, each keeping its source id; and `report.json`, which lists every record that did
not become training data and why. Regenerate after every change to the sources -- the
outputs are build products and are not committed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .adapter import Report, Row, load_rows
from .catalog import load_catalog
from .splits import assign_splits


PROVENANCE = "splits_provenance.json"


def build(config: dict) -> tuple[list[Row], Report]:
    data = config["data"]
    catalog = load_catalog(data["catalog"])
    rows, report = load_rows(data["sources"], catalog, data["language"])
    assign_splits(
        rows, report,
        val_fraction=data["val_fraction"], split_seed=data["split_seed"],
        near_duplicate_threshold=data["near_duplicate_threshold"], extra_groups=data["extra_groups"],
    )
    _note_moves(rows, report, data["prepared_dir"])
    return rows, report


def _note_moves(rows: list[Row], report: Report, prepared_dir: str | Path) -> None:
    """Warn about rows whose split differs from the last prepare. A split is not frozen: a new
    row that joins an existing group (a link, or similarity -- whose IDF moves with the data)
    can change the group's key and so its split. Scoring stays sound, since `evaluate`
    excludes every row the run trained on; this makes the move visible for comparisons."""
    before = {}
    for split in ("train", "val", "test"):
        path = Path(prepared_dir) / f"{split}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    before[json.loads(line)["id"]] = split
    for row in rows:
        if row.id in before and before[row.id] != row.split:
            report.warnings.append(f"{row.id} moved from {before[row.id]} to {row.split} since the last prepare")


def write(rows: list[Row], report: Report, out_dir: str | Path, *, splits: bool = True) -> None:
    """The report always; the split files only when `splits` -- a prepare with errors must not
    replace the last clean split files with partial ones."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test") if splits else ():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                if row.split == split:
                    f.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
    if splits:
        # What the split files were built from. Written only beside them: report.json is also
        # rewritten by a failed prepare, so it cannot say where the kept split files came from.
        (out / PROVENANCE).write_text(json.dumps({"sources": report.sources}, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_split(prepared_dir: str | Path, split: str) -> list[dict]:
    path = Path(prepared_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `python -m check_model prepare` first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(report: Report) -> str:
    counts = report.to_json()["counts"]
    lines = ["counts: " + ", ".join(f"{k}={v}" for k, v in counts.items())]
    for w in report.warnings:
        lines.append(f"WARNING {w}")
    for e in report.errors:
        lines.append(f"ERROR {e['id'] or '-'} ({e['source']}): {e['message']}")
    for bucket in ("unsupported", "eval_only", "pending", "skipped"):
        for item in getattr(report, bucket):
            lines.append(f"{bucket.upper()} {item['id']}: {item['reason']}")
    for pair in report.near_duplicates:
        lines.append(f"GROUPED {pair['a']} + {pair['b']} (similarity {pair['similarity']}) -- check they are one scenario")
    if not report.splits.get("test"):
        lines.append("NOTE no game_specific rows: there is no independent test set yet")
    return "\n".join(lines)
