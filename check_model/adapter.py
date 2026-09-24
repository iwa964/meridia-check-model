"""Source annotations -> SFT rows, without touching the source.

Every record in a source file ends up in exactly one bucket, and the report says which:

* `trainable` -- a creator-labelled single check (or a confirmed no-roll) the model can be
  trained on and scored against.
* `eval_only` -- the creator accepted several interchangeable answers (`alternative_examples`,
  `choice: one_of`) and none is selected. It is scored against the whole accepted set but
  never trained on, because training needs one target and choosing one would be silently
  picking a label the creator did not pick.
* `pending` / `skipped` -- no usable label.
* `unsupported` -- labelled, but in a form this version does not handle yet: pair checks,
  conditional branches, parameterized skills, optional rolls, special-skill checks, another
  roll system. They are reported with the reason and left for later, never flattened into
  single checks.
* errors -- a record that breaks a required field, an allowed label or uniqueness. Errors
  stop `prepare`; the report names each affected record.

The rules applied here come from the dataset's own `annotation_policy` and from the game's
request contract (`catalog.py`). The source file is read and hashed, never written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import Catalog

SUPPORTED_SCHEMA_VERSIONS = ("1.0",)
#: Which split family a scope feeds: general scenes train and validate, Meridia-specific
#: scenes are held out as the independent test set.
SCOPES = ("general", "game_specific")
SUPPORTED_ROLL_SYSTEMS = ("unidirectional",)
#: The input text each input_mode carries; a record without input_mode is the legacy
#: declared-action presentation (`task_scope.legacy_input_mode`).
ACTION_FIELD = {None: "player_action", "observed_event": "observed_event"}
#: Input fields this version does not render into the prompt. A trainable record carrying
#: one would lose information silently, so it is reported as unsupported instead.
UNRENDERED_INPUTS = ("appearance_state", "character_state")

SECTION_KIND = {
    "examples": "labelled",
    "alternative_examples": "alternatives",
    "pending_examples": "pending",
    "skipped_examples": "skipped",
    "conditional_examples": "conditional",
    "parameterized_examples": "parameterized",
}


@dataclass
class Row:
    id: str
    source: str
    section: str
    scope: str
    lang: str
    input: dict
    reference: dict  # {"roll_required": bool, "options": [[check, ...], ...]}
    target: dict | None  # the one decision to train on; None for eval-only rows
    links: list[str] = field(default_factory=list)
    similarity_text: str = ""  # scene + action in every language, for near-duplicate grouping
    split: str | None = None
    group: str | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id, "source": self.source, "section": self.section, "scope": self.scope,
            "split": self.split, "group": self.group, "lang": self.lang, "input": self.input,
            "reference": self.reference, "target": self.target,
        }


@dataclass
class Report:
    sources: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    unsupported: list[dict] = field(default_factory=list)
    eval_only: list[dict] = field(default_factory=list)
    trainable: list[str] = field(default_factory=list)
    all_ids: dict[str, list[str]] = field(default_factory=dict)  # id -> linked ids
    near_duplicates: list[dict] = field(default_factory=list)
    splits: dict[str, list[str]] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "counts": {
                "trainable": len(self.trainable), "eval_only": len(self.eval_only),
                "pending": len(self.pending), "skipped": len(self.skipped),
                "unsupported": len(self.unsupported), "errors": len(self.errors),
                **{f"split_{k}": len(v) for k, v in self.splits.items()},
            },
            "sources": self.sources, "errors": self.errors, "warnings": self.warnings,
            "unsupported": self.unsupported, "pending": self.pending, "skipped": self.skipped,
            "eval_only": self.eval_only, "near_duplicates": self.near_duplicates,
            "splits": self.splits,
        }


class _Skip(Exception):
    """A record leaves the data for a reported, non-error reason."""

    def __init__(self, bucket: str, reason: str):
        super().__init__(reason)
        self.bucket = bucket
        self.reason = reason


class _Invalid(Exception):
    pass


def _check_entry(entry: Any, catalog: Catalog) -> dict:
    """One annotated check -> {kind, name, difficulty}, or raise why it cannot be used."""
    if not isinstance(entry, dict):
        raise _Invalid("a check entry must be an object")
    if entry.get("name") is None and "skill_selector" in entry:
        raise _Skip("unsupported", "parameterized: skill_selector is not bound to a catalog skill")
    roll_system = entry.get("roll_system")
    if roll_system is None:
        raise _Invalid(f"check {entry.get('name')!r} has no roll_system")
    if roll_system not in SUPPORTED_ROLL_SYSTEMS:
        raise _Skip("unsupported", f"roll_system {roll_system!r}")
    out = {"kind": entry.get("kind"), "name": entry.get("name"), "difficulty": entry.get("difficulty")}
    error = catalog.entry_error(out)
    if error:
        if out["kind"] == "skill" and out["name"] in catalog.special_skills:
            raise _Skip("unsupported", f"special skill check: {error}")
        raise _Invalid(error)
    return out


def _single_checks(annotation: dict, catalog: Catalog) -> list[dict]:
    """The checks of a single-check label (possibly empty for no-roll), or raise."""
    if not isinstance(annotation, dict):
        raise _Invalid("annotation must be an object")
    raw = annotation.get("raw_response")
    if not isinstance(raw, str) or not raw.strip():
        # The raw creator reply is what makes a label a human-reviewed one (policy
        # preserve_raw_response); without it nothing says the creator gave this label.
        raise _Invalid("annotation has no raw_response")
    roll_required = annotation.get("roll_required")
    if not isinstance(roll_required, bool):
        raise _Invalid(f"roll_required must be true or false, got {roll_required!r}")
    if annotation.get("roll_optional"):
        raise _Skip("unsupported", "optional roll (roll_optional: true)")
    checks = annotation.get("checks")
    if not isinstance(checks, list):
        raise _Invalid("checks must be a list")
    if not roll_required:
        if checks:
            raise _Invalid("roll_required is false but checks is not empty")
        return []
    mode = annotation.get("check_mode")
    if mode not in (None, "single", "pair"):
        raise _Invalid(f"check_mode must be single or pair, got {mode!r}")
    if mode == "pair" or len(checks) == 2:
        raise _Skip("unsupported", "pair check (check_mode: pair)")
    if len(checks) != 1:
        raise _Invalid(f"roll_required is true with {len(checks)} checks (single needs exactly 1)")
    return [_check_entry(checks[0], catalog)]


def _input(record: dict, lang: str) -> dict:
    mode = record.get("input_mode")
    if mode not in ACTION_FIELD:
        raise _Skip("unsupported", f"input_mode {mode!r}")
    action_field = ACTION_FIELD[mode]
    other = "observed_event" if action_field == "player_action" else "player_action"
    if other in record:
        raise _Invalid(f"input_mode {mode!r} expects {action_field}, but the record carries {other}")
    for extra in UNRENDERED_INPUTS:
        if record.get(extra) is not None:
            raise _Skip("unsupported", f"carries {extra}, which the prompt does not render yet")
    out = {}
    for key in ("scene", action_field):
        text = (record.get(key) or {}).get(lang) if isinstance(record.get(key), dict) else None
        if not isinstance(text, str) or not text.strip():
            raise _Invalid(f"missing {key}.{lang}")
        out[key] = text.strip()
    state = record.get("runtime_state")
    if state is not None:
        if not isinstance(state, dict):
            raise _Invalid("runtime_state must be an object")
        out["runtime_state"] = state
    return out


def _all_languages(record: dict) -> str:
    """Scene and action text in every language the record carries: two variations that differ
    in the prompt language but match in the other are still one scenario."""
    parts = []
    for key in ("scene", "player_action", "observed_event"):
        value = record.get(key)
        if isinstance(value, dict):
            parts.extend(str(v) for _, v in sorted(value.items()) if isinstance(v, str))
    return " ".join(parts)


def _links(value: Any) -> list[str]:
    """Every `related_example_id` anywhere in a record: the dataset's own marker for two
    entries that are variations of one scenario."""
    found: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "related_example_id" and isinstance(v, str):
                found.append(v)
            else:
                found.extend(_links(v))
    elif isinstance(value, list):
        for v in value:
            found.extend(_links(v))
    return found


def _classify(record: dict, kind: str, catalog: Catalog, lang: str) -> tuple[dict, dict | None]:
    """(input, reference, target) for a record that becomes data, or raise."""
    if kind == "pending":
        raise _Skip("pending", "annotation pending" if record.get("annotation") is None else "in pending_examples")
    if kind == "skipped":
        raise _Skip("skipped", str(record.get("skip_reason_raw") or "skipped"))
    if kind == "conditional":
        raise _Skip("unsupported", "conditional: a branch is selected by runtime state the record leaves unset")
    if kind == "parameterized":
        raise _Skip("unsupported", "parameterized: unbound skill selector")
    if kind == "labelled":
        if record.get("annotation") is None:
            raise _Invalid("examples entry has no annotation (policy: only labelled entries belong in examples)")
        checks = _single_checks(record["annotation"], catalog)
        decision = {"roll_required": bool(checks), "checks": checks}
        return _input(record, lang), {"roll_required": bool(checks), "options": [checks]}, decision
    # alternatives
    annotation = record.get("annotation")
    if not isinstance(annotation, dict) or annotation.get("choice") != "one_of":
        raise _Invalid("alternative_examples entry needs annotation.choice: one_of")
    if not isinstance(annotation.get("raw_response"), str) or not annotation["raw_response"].strip():
        raise _Invalid("annotation has no raw_response")
    if annotation.get("roll_required") is not True:
        raise _Invalid("alternative_examples entry must have roll_required: true")
    if annotation.get("selected_alternative") is not None:
        # The dataset has not defined how a selection is recorded yet; guessing its shape
        # would train on a label nobody confirmed.
        raise _Skip("unsupported", "selected_alternative is set, and its format is not defined yet")
    options = []
    alternatives = annotation.get("alternatives")
    if not isinstance(alternatives, list):
        raise _Invalid("alternatives must be a list")
    for option in alternatives:
        if not isinstance(option, dict):
            raise _Invalid(f"each alternative must be an object, got {option!r}")
        sub = dict(annotation, checks=option.get("checks"), check_mode=option.get("check_mode", annotation.get("check_mode")))
        options.append(_single_checks(sub, catalog))
    if len(options) < 2:
        raise _Invalid("choice one_of needs at least two alternatives")
    return _input(record, lang), {"roll_required": True, "options": options}, None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(sources: list[str], catalog: Catalog, lang: str) -> tuple[list[Row], Report]:
    report = Report()
    rows: list[Row] = []
    seen: dict[str, str] = {}
    for source in sources:
        path = Path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("schema_version")
        info = {
            "path": source, "sha256": _sha256(path), "schema_version": version,
            "scenario_scope": data.get("scenario_scope"), "split": data.get("split"),
            "skill_catalog_blob_sha": (data.get("skill_catalog") or {}).get("blob_sha"),
            "attribute_catalog_blob_sha": (data.get("attribute_catalog") or {}).get("blob_sha"),
        }
        report.sources.append(info)
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            report.errors.append({"source": source, "id": None, "message":
                                  f"schema_version {version!r} is not one of {SUPPORTED_SCHEMA_VERSIONS}"})
            continue
        for key, have in (("skill_catalog_blob_sha", catalog.skill_blob_sha),
                          ("attribute_catalog_blob_sha", catalog.attribute_blob_sha)):
            if info[key] != have:
                report.warnings.append(
                    f"{source}: labelled against {key.split('_blob')[0]} {info[key]}, "
                    f"but the catalog snapshot is {have}; labels are validated against the snapshot")
        for section in data:
            if section.endswith("_examples") or section == "examples":
                if section not in SECTION_KIND:
                    report.errors.append({"source": source, "id": None, "message":
                                          f"unknown section {section!r}: the adapter does not know how to treat it"})
        for section, kind in SECTION_KIND.items():
            records = data.get(section, [])
            if not isinstance(records, list):
                # A section edited to null or {} is not "no records": that would drop them silently.
                report.errors.append({"source": source, "id": None, "message":
                                      f"section {section!r} must be a list, got {type(records).__name__}"})
                continue
            for index, record in enumerate(records):
                where = f"{source}:{section}[{index}]"
                rid = record.get("id") if isinstance(record, dict) else None
                if not isinstance(rid, str) or not rid.strip():
                    report.errors.append({"source": where, "id": None, "message": "record has no id"})
                    continue
                if rid in seen:
                    report.errors.append({"source": where, "id": rid,
                                          "message": f"duplicate id (also at {seen[rid]})"})
                    continue
                seen[rid] = where
                report.all_ids[rid] = _links(record)
                scope = record.get("scenario_scope", data.get("scenario_scope"))
                if scope not in SCOPES:
                    report.errors.append({"source": where, "id": rid,
                                          "message": f"scenario_scope must be one of {SCOPES}, got {scope!r}"})
                    continue
                try:
                    inp, reference, target = _classify(record, kind, catalog, lang)
                except _Skip as skip:
                    getattr(report, skip.bucket).append({"id": rid, "source": where, "reason": skip.reason})
                    continue
                except _Invalid as invalid:
                    report.errors.append({"source": where, "id": rid, "message": str(invalid)})
                    continue
                row = Row(id=rid, source=source, section=section, scope=scope, lang=lang,
                          input=inp, reference=reference, target=target, links=report.all_ids[rid],
                          similarity_text=_all_languages(record))
                rows.append(row)
                if target is None:
                    report.eval_only.append({"id": rid, "source": where,
                                             "reason": f"{len(reference['options'])} accepted alternatives, none selected"})
                else:
                    report.trainable.append(rid)
    _check_duplicate_inputs(rows, report)
    return rows, report


def _check_duplicate_inputs(rows: list[Row], report: Report) -> None:
    by_text: dict[str, str] = {}
    for row in rows:
        key = " ".join(json.dumps(row.input, ensure_ascii=False, sort_keys=True).split()).lower()
        if key in by_text:
            report.errors.append({"source": row.source, "id": row.id,
                                  "message": f"same input as {by_text[key]}"})
        else:
            by_text[key] = row.id
