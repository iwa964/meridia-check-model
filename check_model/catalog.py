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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import strictjson

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


_CATALOG_KEYS = ("source", "kinds", "difficulties", "attributes", "skills")


def _catalog_problem(data) -> str | None:
    """Why decoded catalog JSON is not a catalog, or None."""
    if not isinstance(data, dict) or set(data) != set(_CATALOG_KEYS):
        got = sorted(data) if isinstance(data, dict) else type(data).__name__
        return f"expected an object with exactly the keys {list(_CATALOG_KEYS)}, got {got}"
    for key in ("kinds", "difficulties", "attributes"):
        if not (isinstance(data[key], list) and data[key]
                and all(isinstance(v, str) and v.strip() for v in data[key])):
            return f"{key} must be a non-empty list of names"
    if not (isinstance(data["skills"], list) and data["skills"]
            and all(isinstance(r, dict) and isinstance(r.get("name"), str) and r["name"].strip()
                    and isinstance(r.get("initial"), str) for r in data["skills"])):
        return "skills must be a non-empty list of rows, each with a name and an initial (strings)"
    # A name listed twice is ambiguous: a skill with a blank and a set initial would be offered
    # as rollable by the prompt and refused as special by the label check.
    for key, names in (("kinds", data["kinds"]), ("difficulties", data["difficulties"]),
                       ("attributes", data["attributes"]), ("skills", [r["name"] for r in data["skills"]])):
        repeated = sorted(n for n, count in Counter(names).items() if count > 1)
        if repeated:
            return f"{key} lists {repeated} more than once"
    source = data["source"]
    if not (isinstance(source, dict)
            and all(isinstance(source.get(k), dict) and isinstance(source[k].get("blob_sha"), str)
                    for k in ("skill_catalog", "attribute_catalog"))):
        return "source must record skill_catalog.blob_sha and attribute_catalog.blob_sha"
    return None


def parse_catalog(raw: bytes) -> Catalog:
    """A catalog from the bytes of its JSON file, so a caller can hash exactly what it parses.
    Strict: a repeated key (two `skills` lists after a merge, where the last would silently win)
    or a malformed field is a ValueError, not another label set or a KeyError later."""
    data = strictjson.loads(raw.decode("utf-8"))
    problem = _catalog_problem(data)
    if problem:
        raise ValueError(f"not a catalog: {problem}")
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
            value = ast.literal_eval(node.value)
            # ("STR") is the string "STR", and tuple() of it would be ("S", "T", "R").
            if not (isinstance(value, (tuple, list)) and all(isinstance(v, str) for v in value)):
                raise ValueError(f"{CHECK_TURN_PATH}: {name} must be a tuple or list of names, "
                                 f"got {type(value).__name__} {value!r:.60}")
            return tuple(value)
    raise ValueError(f"{CHECK_TURN_PATH} defines no {name}")


#: A skill row's opening however it is spaced: `{"name":`, `{ "name" :`, ...
_ROW_START = re.compile(r'\{\s*"name"\s*:')


def parse_skill_bank(text: str) -> list[dict]:
    rows = [m.groupdict() for m in _SKILL_ROW.finditer(text)]
    # Every row opening must parse: a row in an unexpected format would otherwise be dropped from
    # the label set without a word. Counted spacing-independently, so a reformatted file is caught
    # instead of both counts reading zero.
    expected = len(_ROW_START.findall(text))
    if len(rows) != expected:
        raise ValueError(
            f"{SKILL_BANK_PATH}: parsed {len(rows)} skill rows but found {expected} "
            '\'{"name":\' entries -- the row format changed; update _SKILL_ROW'
        )
    if not rows:
        raise ValueError(f"{SKILL_BANK_PATH}: no skill rows found; the file or its format changed")
    return rows


def _git_output(root: Path, *args: str) -> bytes | None:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def sync_from_meridia(meridia_dir: str | Path) -> Catalog:
    root = Path(meridia_dir)
    head = _git_output(root, "rev-parse", "--verify", "HEAD^{commit}")
    commit = head.decode("ascii").strip() if head else None  # None: not a git checkout
    if commit is None:
        skill_bytes = (root / SKILL_BANK_PATH).read_bytes()
        check_bytes = (root / CHECK_TURN_PATH).read_bytes()
    else:
        dirty = _git_output(root, "status", "--porcelain", "--", SKILL_BANK_PATH, CHECK_TURN_PATH)
        if dirty is None or dirty.strip():
            # The snapshot records HEAD as its commit; the working tree is what the person sees.
            listed = dirty.decode("utf-8", "replace").splitlines() if dirty else "git status failed"
            raise ValueError(f"{root} has uncommitted changes to the catalog sources ({listed}); "
                             "commit or stash them, then sync")
        # Both files from the one commit resolved above, not from the working tree: a checkout
        # or an edit in between cannot mix two revisions under one recorded commit.
        blobs = [_git_output(root, "cat-file", "blob", f"{commit}:{path}")
                 for path in (SKILL_BANK_PATH, CHECK_TURN_PATH)]
        if None in blobs:
            raise ValueError(f"{root}: commit {commit} does not hold {SKILL_BANK_PATH} and {CHECK_TURN_PATH}")
        skill_bytes, check_bytes = blobs
    check_source = check_bytes.decode("utf-8")
    catalog = Catalog(
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
    problem = _catalog_problem(catalog.to_json())  # never write a snapshot parse_catalog refuses
    if problem:
        raise ValueError(f"the synced catalog is not valid: {problem}")
    return catalog


def write_catalog(catalog: Catalog, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(catalog.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
