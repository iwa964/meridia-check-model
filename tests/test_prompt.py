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


@pytest.mark.parametrize("state, problem", [
    ({"x": float("nan")}, "runtime_state.x is nan, not a finite number"),
    ({"x": {1, 2}}, "runtime_state.x holds a set, which is not JSON"),
    ({"a": [1, {"b": float("inf")}]}, "runtime_state.a[1].b is inf, not a finite number"),
    ({1: "x"}, "runtime_state has a non-string key 1")])
def test_runtime_state_must_be_plain_finite_json(state, problem):
    assert prompt.query_errors({"scene": "s", "player_action": "a", "runtime_state": state}) == [problem]
    assert prompt.query_errors({"scene": "s", "player_action": "a",
                                "runtime_state": {"hp": 3, "in_combat": False, "tags": ["a"], "x": 0.5}}) == []


def test_duplicate_key_detection_is_linear():
    import time

    from check_model import strictjson

    wide = "{" + ", ".join(f'"k{i}": {i}' for i in range(100_000)) + "}"
    started = time.perf_counter()
    assert len(strictjson.loads(wide)) == 100_000
    assert time.perf_counter() - started < 2.0  # keys.count() per key took ~10 s here
    with pytest.raises(ValueError, match=r"duplicate key\(s\) \['a', 'b'\]"):
        strictjson.loads('{"b": 1, "a": 1, "b": 2, "a": 2}')


def _nested(depth: int) -> dict:
    state = {}
    for _ in range(depth - 1):
        state = {"x": state}
    return state


def test_runtime_state_that_contains_itself_is_refused():
    state = {"hp": 3}
    state["self"] = state
    assert prompt.query_errors({"scene": "s", "player_action": "a", "runtime_state": state}) == [
        "runtime_state.self contains itself (a cycle), which is not JSON"]
    shared = {"hp": 3}  # the same object twice, side by side, is not a cycle
    assert prompt.query_errors({"scene": "s", "player_action": "a",
                                "runtime_state": {"a": shared, "b": [shared, shared]}}) == []


def test_runtime_state_nesting_is_bounded():
    ok = {"scene": "s", "player_action": "a", "runtime_state": _nested(prompt.MAX_JSON_DEPTH)}
    assert prompt.query_errors(ok) == []
    assert prompt.user_message(ok)  # what passes validation renders
    deep = {"scene": "s", "player_action": "a", "runtime_state": _nested(prompt.MAX_JSON_DEPTH + 1)}
    (problem,) = prompt.query_errors(deep)
    assert problem.endswith(f"is nested more than {prompt.MAX_JSON_DEPTH} levels deep")
    far = {"scene": "s", "player_action": "a", "runtime_state": _nested(5000)}
    assert prompt.query_errors(far) == [problem]  # refused, not a RecursionError


def test_json_nested_past_the_recursion_limit_is_a_value_error():
    from check_model import strictjson

    with pytest.raises(ValueError, match="nested more than 200 levels deep"):
        strictjson.loads('{"a":' * 5000 + "1" + "}" * 5000)


def test_a_query_with_non_string_keys_is_a_bad_query_not_a_crash():
    errors = prompt.query_errors({"scene": "s", "player_action": "a", 1: "x", "extra": "y"})
    assert errors == ["query keys must be strings, got [1]", "unknown fields ['extra']"]


def test_json_nesting_is_bounded_whatever_the_decoder_allows():
    from check_model import strictjson

    def nested(depth):
        return '{"a":' * (depth - 1) + "{}" + "}" * (depth - 1)

    assert strictjson.loads(nested(strictjson.MAX_DEPTH))  # at the bound: accepted
    # 300 levels decode fine on Python 3.11 too: the limit is this module's, not the decoder's.
    with pytest.raises(ValueError, match=f"nested more than {strictjson.MAX_DEPTH} levels deep"):
        strictjson.loads(nested(300))
    with pytest.raises(ValueError, match=f"nested more than {strictjson.MAX_DEPTH} levels deep"):
        strictjson.loads("[" * 300 + "]" * 300)
    # Room for the deepest legitimate document: a source record (3 levels) carrying a
    # runtime_state at the query bound.
    assert strictjson.MAX_DEPTH >= 3 + prompt.MAX_JSON_DEPTH


def test_an_integer_too_long_to_render_is_a_bad_query():
    query = {"scene": "s", "player_action": "a", "runtime_state": {"n": 10 ** 5000}}
    assert prompt.query_errors(query) == ["runtime_state.n is an integer too long to render"]
    assert prompt.query_errors({"scene": "s", "player_action": "a", "runtime_state": {"n": 10 ** 40}}) == []


def test_unpaired_surrogates_are_refused():
    from check_model import strictjson

    for text in ('{"scene": "\\ud800"}', '{"\\udfff": 1}', '["ok", ["\\ud800"]]'):
        with pytest.raises(ValueError, match="unpaired surrogate"):
            strictjson.loads(text)
    assert strictjson.loads('{"scene": "\\ud83d\\ude00"}') == {"scene": "\U0001F600"}  # a pair is fine
    lone = "\ud800"
    assert prompt.query_errors({"scene": lone, "player_action": "a"}) == [
        "scene holds an unpaired surrogate, which is not text"]
    assert prompt.query_errors({"scene": "s", "observed_event": lone}) == [
        "observed_event holds an unpaired surrogate, which is not text"]
    assert prompt.query_errors({"scene": "s", "player_action": "a", "runtime_state": {"x": [lone]}}) == [
        "runtime_state.x[0] holds an unpaired surrogate, which is not text"]
    assert prompt.query_errors({"scene": "s", "player_action": "a", "runtime_state": {lone: 1}}) == [
        "runtime_state has a key with an unpaired surrogate, which is not text"]
