"""What the model reads and writes.

INPUT (the query a game backend sends) -- the fields the creator was shown when
labelling (`task_scope.presented_input_fields`):

    {"scene": "...", "player_action": "..."}                       # declared action
    {"scene": "...", "observed_event": "...", "runtime_state": {"in_combat": false}}

OUTPUT -- one JSON object, nothing else:

    {"roll_required": false, "checks": []}
    {"roll_required": true, "checks": [{"kind": "skill", "name": "Climbing", "difficulty": "extreme"}]}

`checks` is a list so a later version can emit the game's two-entry pair without a format
change; this version accepts exactly one entry. Entries use the pair-entry shape the game
already validates (`DialogueCheck.validate_entry`: kind, name, difficulty).

`to_game_request()` turns a decision into the request `DialogueCheck.from_request()`
accepts. Its single-check form is `{"skill", "difficulty"}` and rolls skills only, so a
single ATTRIBUTE check has no request form in the game today; it is returned as a
decision with a note, never forced into the skill slot.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from . import strictjson
from .catalog import Catalog

#: Bump when the prompt or target format changes: a model trained on one format is not
#: scored or served with another.
PROMPT_FORMAT_VERSION = "1"
MAX_CHECKS = 1  # single checks only in this version; the game's pair form is 2

ATTRIBUTE_REQUEST_NOTE = (
    "single attribute check: the game's single-check request {skill, difficulty} rolls skills only "
    "(DialogueCheck.from_request), so this decision has no request form yet"
)


#: What each difficulty label means in the game: the roll must come in under this share of the
#: tested value (MeridiaGame `Check.threshold`, pinned by DialogueCheckTest: success = value,
#: hard = 1/2, extreme = 1/5). The catalog carries the labels only, so a label added to
#: check_turn.DIFFICULTIES needs its rule written here before the prompt can describe it.
DIFFICULTY_RULES = {
    "success": "a normal check, rolled under the full value",
    "hard": "under half the value",
    "extreme": "under a fifth of the value",
}


#: The check kinds this code implements end to end: the prompt describes these two, the catalog
#: treats every non-attribute kind as a skill, and to_game_request maps exactly these.
SUPPORTED_KINDS = ("skill", "attribute")


def _difficulty_sentence(difficulties: tuple[str, ...]) -> str:
    unknown = [d for d in difficulties if d not in DIFFICULTY_RULES]
    if unknown or not difficulties:
        raise ValueError(f"catalog difficulties {list(difficulties)}: no rule text for {unknown}; "
                         "add it to prompt.DIFFICULTY_RULES from the game's Check.threshold")
    parts = [f'"{d}" ({DIFFICULTY_RULES[d]})' for d in difficulties]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " or " + parts[-1]


def system_prompt(catalog: Catalog) -> str:
    if set(catalog.kinds) != set(SUPPORTED_KINDS):
        raise ValueError(f"catalog kinds {list(catalog.kinds)}: this code implements exactly "
                         f"{list(SUPPORTED_KINDS)} (prompt, label validation and game request)")
    return (
        "You choose the dice check for one moment in the game Meridia. The game rolls the dice "
        "and resolves the result; you only decide the check.\n\n"
        "The input is a JSON object: the scene, then either the action the player declares "
        "(\"player_action\") or the event the game observed (\"observed_event\"), sometimes with "
        "runtime state.\n\n"
        "Decide whether a check is required. If one is, choose exactly one skill or attribute and "
        "a difficulty: " + _difficulty_sentence(catalog.difficulties) + ".\n\n"
        "Reply with one JSON object and nothing else, in one of these forms:\n"
        '{"roll_required": false, "checks": []}\n'
        '{"roll_required": true, "checks": [{"kind": "skill", "name": "<skill>", "difficulty": "<difficulty>"}]}\n'
        'Use "kind": "attribute" with an attribute name when an attribute is tested instead of a skill.\n\n'
        "Attributes: " + ", ".join(catalog.attributes) + "\n"
        "Skills: " + ", ".join(catalog.rollable_skills)
    )


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def user_message(query: dict) -> str:
    return json.dumps(query, ensure_ascii=False)


def messages(system: str, query: dict, target: dict | None = None) -> list[dict]:
    out = [{"role": "system", "content": system}, {"role": "user", "content": user_message(query)}]
    if target is not None:
        out.append({"role": "assistant", "content": target_text(target)})
    return out


def target_text(decision: dict) -> str:
    """The canonical serialization: fixed key order, one line."""
    checks = [{"kind": c["kind"], "name": c["name"], "difficulty": c["difficulty"]} for c in decision["checks"]]
    return json.dumps({"roll_required": decision["roll_required"], "checks": checks}, ensure_ascii=False)


def query_errors(query: Any) -> list[str]:
    """Why a query is not in the input format, or []."""
    if not isinstance(query, dict):
        return ["query must be a JSON object"]
    errors = []
    if not isinstance(query.get("scene"), str) or not query["scene"].strip():
        errors.append("scene must be a non-empty string")
    actions = [k for k in ("player_action", "observed_event") if k in query]
    if len(actions) != 1:
        errors.append("exactly one of player_action / observed_event is required")
    elif not isinstance(query[actions[0]], str) or not query[actions[0]].strip():
        errors.append(f"{actions[0]} must be a non-empty string")
    if "runtime_state" in query and not isinstance(query["runtime_state"], dict):
        errors.append("runtime_state must be an object")
    unknown = set(query) - {"scene", "player_action", "observed_event", "runtime_state"}
    if unknown:
        errors.append(f"unknown fields {sorted(unknown)}")
    return errors


def parse_decision(text: str, catalog: Catalog) -> tuple[dict | None, list[str]]:
    """(decision, []) for a well-formed reply, else (None, reasons). Strict: the whole reply
    must be the one object, with no extra keys and only catalog labels."""
    try:
        raw = strictjson.loads(text.strip())  # a repeated key is ambiguous, not "the last one wins"
    except ValueError as exc:
        return None, [f"not JSON: {exc}"]
    if not isinstance(raw, dict):
        return None, ["not a JSON object"]
    errors = []
    if set(raw) != {"roll_required", "checks"}:
        errors.append(f"keys must be roll_required and checks, got {sorted(raw)}")
    roll_required = raw.get("roll_required")
    checks = raw.get("checks")
    if not isinstance(roll_required, bool):
        errors.append("roll_required must be true or false")
    if not isinstance(checks, list):
        errors.append("checks must be a list")
    if errors:
        return None, errors
    if not roll_required and checks:
        return None, ["roll_required is false but checks is not empty"]
    if roll_required and not 1 <= len(checks) <= MAX_CHECKS:
        return None, [f"roll_required is true with {len(checks)} checks (this model version emits exactly {MAX_CHECKS})"]
    out = []
    for entry in checks:
        if not isinstance(entry, dict) or set(entry) != {"kind", "name", "difficulty"}:
            return None, ["each check must have exactly kind, name and difficulty"]
        error = catalog.entry_error(entry)
        if error:
            return None, [error]
        out.append({"kind": entry["kind"], "name": entry["name"], "difficulty": entry["difficulty"]})
    return {"roll_required": roll_required, "checks": out}, []


def to_game_request(decision: dict) -> tuple[dict | None, str | None]:
    """(request for DialogueCheck.from_request, note). No roll -> (None, None)."""
    if not decision["roll_required"]:
        return None, None
    (check,) = decision["checks"]
    if check["kind"] == "attribute":
        return None, ATTRIBUTE_REQUEST_NOTE
    return {"skill": check["name"], "difficulty": check["difficulty"]}, None
