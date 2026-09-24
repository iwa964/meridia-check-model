import json

import pytest

from check_model import prompt

CLIMB = {"roll_required": True, "checks": [{"kind": "skill", "name": "Climbing", "difficulty": "extreme"}]}
STR = {"roll_required": True, "checks": [{"kind": "attribute", "name": "STR", "difficulty": "success"}]}
NO_ROLL = {"roll_required": False, "checks": []}


@pytest.mark.parametrize("decision", [CLIMB, STR, NO_ROLL])
def test_target_round_trips(catalog, decision):
    assert prompt.parse_decision(prompt.target_text(decision), catalog) == (decision, [])


@pytest.mark.parametrize("text, expected", [
    ('{"roll_required": true, "checks": [{"kind": "skill", "name": "Climbing", "difficulty": "extreme"}]} ok',
     "not JSON"),
    ('```json\n{"roll_required": false, "checks": []}\n```', "not JSON"),
    ('{"roll_required": false, "checks": [], "reason": "easy"}', "keys must be"),
    ('{"roll_required": "yes", "checks": []}', "roll_required must be"),
    ('{"roll_required": false, "checks": [{"kind": "skill", "name": "Climbing", "difficulty": "hard"}]}',
     "checks is not empty"),
    ('{"roll_required": true, "checks": []}', "0 checks"),
    ('{"roll_required": true, "checks": [{"kind": "skill", "name": "History", "difficulty": "success"}, '
     '{"kind": "skill", "name": "Lang (Meridian)", "difficulty": "success"}]}', "2 checks"),
    ('{"roll_required": true, "checks": [{"kind": "skill", "name": "Design", "difficulty": "success"}]}',
     "unknown skill"),
    ('{"roll_required": true, "checks": [{"kind": "skill", "name": "Climbing"}]}', "exactly kind, name"),
])
def test_parse_is_strict(catalog, text, expected):
    decision, errors = prompt.parse_decision(text, catalog)
    assert decision is None and any(expected in e for e in errors), errors


def test_game_request_shapes():
    assert prompt.to_game_request(CLIMB) == ({"skill": "Climbing", "difficulty": "extreme"}, None)
    assert prompt.to_game_request(NO_ROLL) == (None, None)
    request, note = prompt.to_game_request(STR)
    assert request is None and "attribute" in note


def test_system_prompt_lists_every_label(catalog):
    text = prompt.system_prompt(catalog)
    skills_line = next(line for line in text.splitlines() if line.startswith("Skills: "))
    assert skills_line[len("Skills: "):].split(", ") == list(catalog.rollable_skills)
    assert not any(s in skills_line for s in catalog.special_skills)
    assert all(a in text for a in catalog.attributes)


@pytest.mark.parametrize("query, expected", [
    ({"scene": "s", "player_action": "a"}, []),
    ({"scene": "s", "observed_event": "e", "runtime_state": {"in_combat": False}}, []),
    ({"scene": "s"}, ["exactly one of"]),
    ({"scene": "s", "player_action": "a", "observed_event": "e"}, ["exactly one of"]),
    ({"scene": "", "player_action": "a"}, ["scene"]),
    ({"scene": "s", "player_action": "a", "npc": "x"}, ["unknown fields"]),
])
def test_query_errors(query, expected):
    errors = prompt.query_errors(query)
    assert len(errors) == len(expected) and all(e in got for e, got in zip(expected, errors))


def test_user_message_keeps_unicode():
    assert "攀爬" in prompt.user_message({"scene": "攀爬", "player_action": "a"})
    assert json.loads(prompt.user_message({"scene": "攀爬", "player_action": "a"}))["scene"] == "攀爬"


@pytest.mark.parametrize("reply", [
    '{"roll_required": false, "roll_required": true, "checks": []}',
    '{"roll_required": true, "checks": [{"kind": "skill", "name": "Climbing", "name": "Swimming", '
    '"difficulty": "hard"}]}'])
def test_a_reply_with_a_repeated_key_is_not_a_decision(catalog, reply):
    decision, errors = prompt.parse_decision(reply, catalog)
    assert decision is None and "duplicate key(s)" in errors[0]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_json_constants_are_refused(catalog, constant):
    from check_model import strictjson

    with pytest.raises(ValueError, match=f"{constant} is not valid JSON"):
        strictjson.loads('{"scene": "s", "runtime_state": {"hp": %s}}' % constant)
    decision, errors = prompt.parse_decision('{"roll_required": %s, "checks": []}' % constant, catalog)
    assert decision is None and "not valid JSON" in errors[0]


def test_a_number_too_large_for_a_float_is_refused(catalog):
    from check_model import strictjson

    with pytest.raises(ValueError, match="1e999 is out of range"):
        strictjson.loads('{"scene": "s", "runtime_state": {"x": 1e999}}')
    assert strictjson.loads('{"x": 1.5e300, "n": 123456789012345678901234567890}')["x"] == 1.5e300
