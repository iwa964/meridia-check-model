"""`prepare`: validate the sources and write the split files plus a report.

Outputs (under `data.prepared_dir`): `train.jsonl`, `val.jsonl`, `test.jsonl`, one row per
example, each keeping its source id; and `report.json`, which lists every record that did
not become training data and why. Regenerate after every change to the sources -- the
outputs are build products and are not committed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import prompt, strictjson
from .adapter import Report, Row, input_key, load_rows
from .catalog import Catalog, load_catalog
from .splits import assign_splits


PROVENANCE = "splits_provenance.json"
#: The data settings that decide which rows land in which split: re-preparing the same sources
#: with any of these changed yields a different validation cohort.
SPLIT_KEYS = ("language", "val_fraction", "split_seed", "near_duplicate_threshold", "extra_groups")


def checked_catalog(path: str | Path) -> Catalog:
    """The catalog at `path`, refused (ValueError) unless the prompt can describe it: a kind this
    code does not implement or a difficulty it has no rule for would otherwise pass `prepare`
    and only fail in training, after the run directory exists."""
    catalog = load_catalog(path)
    prompt.system_prompt(catalog)
    return catalog


def build(config: dict, catalog: Catalog | None = None) -> tuple[list[Row], Report]:
    """`catalog` is the snapshot to label against, loaded from `data.catalog` when not given; a
    caller that trains afterwards passes the same object on, not the path again."""
    data = config["data"]
    if catalog is None:
        catalog = checked_catalog(data["catalog"])
    else:
        prompt.system_prompt(catalog)
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
        if not path.exists():
            continue
        unreadable = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                before[json.loads(line)["id"]] = split
            except (ValueError, TypeError, KeyError):  # an interrupted write, say: it is rebuilt below
                unreadable += 1
        if unreadable:
            report.warnings.append(f"{path}: {unreadable} line(s) of the previous prepare are unreadable; "
                                   "the moved-row check skips them")
    for row in rows:
        if row.id in before and before[row.id] != row.split:
            report.warnings.append(f"{row.id} moved from {before[row.id]} to {row.split} since the last prepare")


def write(rows: list[Row], report: Report, out_dir: str | Path, *, splits: bool = True,
          split_config: dict | None = None) -> dict[str, str]:
    """The report always; the split files only when `splits` -- a prepare with errors must not
    replace the last clean split files with partial ones. Returns the SHA-256 of each split's
    bytes as written, so a caller training from `rows` records exactly those."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in ("train", "val", "test") if splits else ():
        data = "".join(json.dumps(row.to_json(), ensure_ascii=False) + "\n" for row in rows
                       if row.split == split).encode("utf-8")
        (out / f"{split}.jsonl").write_bytes(data)  # bytes: no newline translation between file and hash
        hashes[split] = hashlib.sha256(data).hexdigest()
    if splits:
        # What the split files were built from, and what they hold. Written only beside them:
        # report.json is also rewritten by a failed prepare, so it cannot say where the kept split
        # files came from. The content hashes catch a change the sources and settings do not --
        # new adapter or splitting code yields different rows from the same bytes.
        provenance = {"sources": report.sources, "split_config": split_config, "split_sha256": hashes}
        (out / PROVENANCE).write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return hashes


def split_hashes(prepared_dir: str | Path) -> dict[str, str | None]:
    """SHA-256 of each split file as it is on disk now (None if missing). Hashed from the files,
    never taken from splits_provenance.json: a split file edited or replaced after `prepare`
    would still match its stale recorded hash."""
    out = {}
    for split in ("train", "val", "test"):
        path = Path(prepared_dir) / f"{split}.jsonl"
        out[split] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return out


#: Every field Row.to_json writes, and its type (target, also written, is an object or null).
#: Evaluation depends on all of them: identity, what is asked and scored, and what decides
#: exclusion (group, links, group_members, similarity_text) -- a row missing one would be scored
#: with defaults nobody chose. The previous rounds checked a subset, one field at a time.
ROW_FIELDS = {"id": str, "source": str, "section": str, "scope": str, "split": str, "group": str,
              "lang": str, "input": dict, "reference": dict, "similarity_text": str,
              "links": list, "group_members": list}


def _reference_problem(reference: dict, catalog=None) -> str | None:
    """Why a reference is not {roll_required: bool, options: [[{kind, name, difficulty}, ...], ...]}."""
    if not isinstance(reference.get("roll_required"), bool):
        return "reference.roll_required must be true or false"
    options = reference.get("options")
    if not isinstance(options, list) or not all(isinstance(o, list) for o in options):
        return "reference.options must be a list of option lists"
    for option in options:
        for check in option:
            if not (isinstance(check, dict) and set(check) == {"kind", "name", "difficulty"}
                    and all(isinstance(check[k], str) for k in check)):
                # score() compares whole checks: an extra field could never be matched exactly.
                return "each reference check needs string kind, name and difficulty, and no other field"
    # What the adapter writes: a roll-required reference has options of 1..MAX_CHECKS checks, a
    # no-roll one only empty options. Anything else would be scored wrong whatever the model says.
    if reference["roll_required"]:
        if not options or not all(1 <= len(o) <= prompt.MAX_CHECKS for o in options):
            return f"a roll-required reference needs options of 1 to {prompt.MAX_CHECKS} check(s)"
    elif any(options):
        return "a no-roll reference cannot hold checks"
    if catalog is not None:
        # A label the run's parser can never accept would be scored wrong whatever the model says.
        for check in (c for option in options for c in option):
            error = catalog.entry_error(check)
            if error:
                return f"reference label {check}: {error}"
    return None


def _is_id(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _row_problem(row: dict, split: str, catalog=None) -> str | None:
    missing = [k for k in [*ROW_FIELDS, "target"] if k not in row]
    if missing:
        return f"missing {missing}"
    wrong = [k for k, kind in ROW_FIELDS.items() if not isinstance(row[k], kind)]
    if not (row["target"] is None or isinstance(row["target"], dict)):
        wrong.append("target")
    if wrong:
        return f"wrong type for {wrong}"
    if not _is_id(row["id"]):
        return "id is blank"
    if row["split"] != split:
        return f"split {row['split']!r} in the {split} file"
    scope = "game_specific" if split == "test" else "general"  # what assign_splits puts there
    if row["scope"] != scope:
        return f"scope {row['scope']!r} in the {split} file, which holds {scope!r} rows"
    for key in ("links", "group_members"):
        if not all(_is_id(i) for i in row[key]):
            return f"{key} must hold ids"
    # assign_splits names a group by one of its ids and lists every member, the row included.
    # Evaluation links rows to trained scenarios through these; empty, a renamed variation of a
    # trained row would be scored as held out.
    if not _is_id(row["group"]):
        return "group is blank"
    if row["id"] not in row["group_members"] or row["group"] not in row["group_members"]:
        return "group_members must include the row's id and its group"
    if not row["similarity_text"].strip():
        # The adapter always writes the scene and action here; blank, the near-duplicate check
        # against training texts would see nothing and let a renamed variant through.
        return "similarity_text is blank"
    # The same check predict applies: a damaged input would otherwise be scored as a wrong answer.
    query_problems = prompt.query_errors(row["input"])
    if query_problems:
        return "input: " + "; ".join(query_problems)
    return _reference_problem(row["reference"], catalog)


def read_split(prepared_dir: str | Path, split: str, catalog=None, hashes: dict | None = None) -> list[dict]:
    """The rows of one prepared split file, each checked against the full row schema: strict UTF-8
    and strict JSON, unique ids, and -- given the run's catalog -- reference labels it accepts."""
    path = Path(prepared_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `python -m check_model prepare` first")
    raw = path.read_bytes()
    if hashes is not None:  # the hash of exactly the bytes parsed here
        hashes[split] = hashlib.sha256(raw).hexdigest()
    rows, seen = [], {}
    for number, data in enumerate(raw.split(b"\n"), 1):
        def refuse(problem: str):
            return ValueError(f"{path}:{number}: not a prepared row ({problem}); re-run prepare")

        try:
            line = data.decode("utf-8")  # strictly: a replaced byte would change the text scored
        except UnicodeDecodeError:
            raise refuse("not UTF-8") from None
        if not line.strip():
            continue
        try:
            row = strictjson.loads(line)
        except ValueError as exc:
            raise refuse(str(exc)) from None
        if not isinstance(row, dict):
            raise refuse(f"a JSON object is required, got {type(row).__name__}")
        problem = _row_problem(row, split, catalog)
        if problem:
            raise refuse(problem)
        if row["id"] in seen:
            raise refuse(f"duplicate id {row['id']!r} (also at line {seen[row['id']]})")
        seen[row["id"]] = number
        rows.append(row)
    return rows


def duplicate_ids(splits: dict[str, list[dict]]) -> list[str]:
    """Ids that appear in more than one split file: each would be scored or excluded twice."""
    count: dict[str, int] = {}
    for rows in splits.values():
        for row in rows:
            count[row["id"]] = count.get(row["id"], 0) + 1
    return sorted(i for i, n in count.items() if n > 1)


def duplicate_inputs(splits: dict[str, list[dict]]) -> list[list[str]]:
    """Ids whose rows have the same input by prepare's duplicate check, within or across split
    files. prepare refuses these; in changed split files a copy under a new id would be scored
    twice and weigh twice in every metric."""
    by_key: dict[str, list[str]] = {}
    for rows in splits.values():
        for row in rows:
            by_key.setdefault(input_key(row["input"]), []).append(row["id"])
    return sorted(ids for ids in by_key.values() if len(ids) > 1)


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
