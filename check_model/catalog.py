"""The label catalog: which skills, attributes and difficulties a check may name.

The catalog is not defined here. It is a snapshot of two MeridiaGame files, the same
two that `dice_rolling_train.json` pins in `skill_catalog` and `attribute_catalog`:

* `Scripts/profile/skill/SkillBank.gd` -- the skill rows. A row is a valid check label
  when `DialogueCheck.is_rollable_skill()` would accept it: it exists and its `initial`
  is not blank. The blank-initial rows are the special skills, which are not check-based.
* `server/check_turn.py` -- `DIFFICULTIES`, `KINDS` and `ATTRIBUTES`, the values the
  game's request contract accepts.

`sync_from_meridia()` regenerates the snapshot from a MeridiaGame checkout and records
each file's git blob SHA, so the adapter can tell when the dataset was labelled against
a different catalog version.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

SKILL_BANK_PATH = "Scripts/profile/skill/SkillBank.gd"
CHECK_TURN_PATH = "server/check_turn.py"
MERIDIA_REPOSITORY = "iwa964/MeridiaGame"

_SKILL_ROW = re.compile(
    r'\{"name": "(?P<name>[^"]+)", "category": "(?P<category>[^"]*)", '
    r'"subcategory": "(?P<subcategory>[^"]*)", "initial": "(?P<initial>[^"]*)"\}'
)


def git_blob_sha(data: bytes) -> str:
    """The SHA git gives a file with these bytes (`git hash-object`)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@dataclass(frozen=True)
class Catalog:
    skills: tuple[dict, ...]
    attributes: tuple[str, ...]
    difficulties: tuple[str, ...]
    kinds: tuple[str, ...]
    source: dict

    @property
    def rollable_skills(self) -> tuple[str, ...]:
        return tuple(s["name"] for s in self.skills if s["initial"].strip())

    @property
    def special_skills(self) -> tuple[str, ...]:
        return tuple(s["name"] for s in self.skills if not s["initial"].strip())

    @property
    def skill_blob_sha(self) -> str:
        return self.source["skill_catalog"]["blob_sha"]

    @property
    def attribute_blob_sha(self) -> str:
        return self.source["attribute_catalog"]["blob_sha"]

    def entry_error(self, entry: dict) -> str | None:
        """Why one check entry {kind, name, difficulty} is not a valid label, or None."""
        kind, name, difficulty = entry.get("kind"), entry.get("name"), entry.get("difficulty")
        if kind not in self.kinds:
            return f"kind must be one of {list(self.kinds)}, got {kind!r}"
        if difficulty not in self.difficulties:
            return f"difficulty must be one of {list(self.difficulties)}, got {difficulty!r}"
        if not isinstance(name, str) or not name:
            return "check has no name"
        if kind == "attribute":
            if name not in self.attributes:
                return f"unknown attribute {name!r} (allowed: {list(self.attributes)})"
            return None
        if name in self.special_skills:
            return f"{name!r} is a special skill (blank initial): not check-based"
        if name not in self.rollable_skills:
            return f"unknown skill {name!r}: not in the SkillBank snapshot"
        return None

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "kinds": list(self.kinds),
            "difficulties": list(self.difficulties),
            "attributes": list(self.attributes),
            "skills": list(self.skills),
        }


def load_catalog(path: str | Path) -> Catalog:
    return parse_catalog(Path(path).read_bytes())


def parse_catalog(raw: bytes) -> Catalog:
    """A catalog from the bytes of its JSON file, so a caller can hash exactly what it parses."""
    data = json.loads(raw.decode("utf-8"))
    return Catalog(
        skills=tuple(data["skills"]),
        attributes=tuple(data["attributes"]),
        difficulties=tuple(data["difficulties"]),
        kinds=tuple(data["kinds"]),
        source=data["source"],
    )


def _tuple_constant(source: str, name: str) -> tuple[str, ...]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise ValueError(f"{CHECK_TURN_PATH} defines no {name}")


def parse_skill_bank(text: str) -> list[dict]:
    rows = [m.groupdict() for m in _SKILL_ROW.finditer(text)]
    # Every `{"name":` line must parse: a row in an unexpected format would otherwise be
    # dropped from the label set without a word.
    expected = text.count('{"name":')
    if len(rows) != expected:
        raise ValueError(
            f"{SKILL_BANK_PATH}: parsed {len(rows)} skill rows but found {expected} "
            '\'{"name":\' entries -- the row format changed; update _SKILL_ROW'
        )
    return rows


def sync_from_meridia(meridia_dir: str | Path) -> Catalog:
    root = Path(meridia_dir)
    try:
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", SKILL_BANK_PATH, CHECK_TURN_PATH],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        dirty = ""  # not a git checkout: the snapshot records no commit, below
    if dirty:
        # The snapshot records HEAD as its commit; bytes HEAD does not hold would make that a lie.
        raise ValueError(f"{root} has uncommitted changes to the catalog sources ({dirty.splitlines()}); "
                         "commit or stash them, then sync")
    skill_bytes = (root / SKILL_BANK_PATH).read_bytes()
    check_bytes = (root / CHECK_TURN_PATH).read_bytes()
    check_source = check_bytes.decode("utf-8")
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return Catalog(
        skills=tuple(parse_skill_bank(skill_bytes.decode("utf-8"))),
        attributes=_tuple_constant(check_source, "ATTRIBUTES"),
        difficulties=_tuple_constant(check_source, "DIFFICULTIES"),
        kinds=_tuple_constant(check_source, "KINDS"),
        source={
            "repository": MERIDIA_REPOSITORY,
            "commit": commit,
            "skill_catalog": {"path": SKILL_BANK_PATH, "blob_sha": git_blob_sha(skill_bytes)},
            "attribute_catalog": {
                "path": CHECK_TURN_PATH,
                "blob_sha": git_blob_sha(check_bytes),
                "definition": "ATTRIBUTES",
            },
        },
    )


def write_catalog(catalog: Catalog, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(catalog.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
