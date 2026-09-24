import json

from check_model.evaluate import evaluate_rows, input_fingerprint, score, summarize

CLIMB_EXTREME = {"kind": "skill", "name": "Climbing", "difficulty": "extreme"}
# dice_train_000038's accepted set: normal Economics or normal Mathematics.
ECON = {"kind": "skill", "name": "Economics", "difficulty": "success"}
MATH = {"kind": "skill", "name": "Mathematics", "difficulty": "success"}


def ref(*options, roll=True):
    return {"roll_required": roll, "options": [[o] for o in options] if roll else [[]]}


def decide(check=None):
    return {"roll_required": check is not None, "checks": [check] if check else []}


def test_exact_match():
    assert score(ref(CLIMB_EXTREME), decide(CLIMB_EXTREME)) == {
        "format_valid": True, "roll_required": True, "check": True, "difficulty": True, "exact": True}


def test_fields_are_scored_separately():
    wrong_skill = dict(CLIMB_EXTREME, name="Jumping")
    assert score(ref(CLIMB_EXTREME), decide(wrong_skill)) == {
        "format_valid": True, "roll_required": True, "check": False, "difficulty": True, "exact": False}
    wrong_difficulty = dict(CLIMB_EXTREME, difficulty="hard")
    assert score(ref(CLIMB_EXTREME), decide(wrong_difficulty))["check"] is True
    assert score(ref(CLIMB_EXTREME), decide(wrong_difficulty))["difficulty"] is False


def test_any_accepted_alternative_is_right():
    for pick in (ECON, MATH):
        assert score(ref(ECON, MATH), decide(pick))["exact"] is True
    assert score(ref(ECON, MATH), decide(dict(ECON, name="Business")))["check"] is False


def test_invalid_output_is_wrong_everywhere_it_applies():
    assert score(ref(CLIMB_EXTREME), None) == {
        "format_valid": False, "roll_required": False, "check": False, "difficulty": False, "exact": False}


def test_no_roll_reference_skips_check_fields():
    s = score(ref(roll=False), decide(None))
    assert s["roll_required"] is True and s["check"] is None and s["difficulty"] is None
    s = score(ref(roll=False), decide(CLIMB_EXTREME))
    assert s["roll_required"] is False and s["exact"] is False


def test_missed_roll_misses_check_and_difficulty():
    s = score(ref(CLIMB_EXTREME), decide(None))
    assert s == {"format_valid": True, "roll_required": False, "check": False, "difficulty": False, "exact": False}


def test_summary_counts_only_applicable_rows():
    out = summarize([score(ref(roll=False), decide(None)), score(ref(CLIMB_EXTREME), decide(CLIMB_EXTREME))])
    assert out["roll_required"] == {"correct": 2, "total": 2, "rate": 1.0}
    assert out["check"] == {"correct": 1, "total": 1, "rate": 1.0}


class EchoModel:
    """Answers each query with a fixed reply; a test double for scoring, not a model."""

    def __init__(self, reply):
        self.reply = reply

    def predict_many(self, queries, batch_size=8):
        return [{"decision": self.reply, "errors": [], "raw_output": json.dumps(self.reply),
                 "game_request": None, "note": None} for _ in queries]


def rows():
    return [{"id": f"r{i}", "source": "s", "split": "val", "input": {"scene": "s", "player_action": "a"},
             "reference": ref(CLIMB_EXTREME)} for i in range(3)]


def test_trained_rows_are_excluded_from_held_out_scores(tmp_path):
    metrics = evaluate_rows(EchoModel(decide(CLIMB_EXTREME)), rows(), split="val", train_ids={"r0"},
                            out_dir=tmp_path)
    assert metrics["held_out"] is True and metrics["independent_test"] is False
    assert metrics["excluded_trained_rows"] == ["r0"] and metrics["scored"] == 2
    saved = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert [r["id"] for r in saved] == ["r1", "r2"]
    assert json.loads((tmp_path / "metrics.json").read_text())["fields"]["exact"]["rate"] == 1.0


def test_scoring_training_rows_is_labelled_not_held_out(tmp_path):
    metrics = evaluate_rows(EchoModel(decide(CLIMB_EXTREME)), rows(), split="train", train_ids={"r0", "r1", "r2"},
                            out_dir=tmp_path, include_training_rows=True)
    assert metrics["held_out"] is False and "NOT a held-out result" in metrics["note"]
    assert metrics["scored"] == 3


def test_only_the_test_split_is_independent(tmp_path):
    metrics = evaluate_rows(EchoModel(decide(CLIMB_EXTREME)), rows(), split="test", train_ids=set(), out_dir=tmp_path)
    assert metrics["independent_test"] is True


def test_a_split_whose_rows_were_all_trained_on_claims_nothing(tmp_path):
    metrics = evaluate_rows(EchoModel(decide(CLIMB_EXTREME)), rows(), split="test", train_ids={"r0", "r1", "r2"},
                            out_dir=tmp_path)
    assert metrics["scored"] == 0
    assert metrics["held_out"] is False and metrics["independent_test"] is False
    assert "every test row was used in training" in metrics["note"]


def test_a_renamed_training_row_is_still_excluded(tmp_path):
    renamed = rows()  # r0's input is the one the run trained on, under another id then
    trained = {input_fingerprint(renamed[0]["input"])}
    metrics = evaluate_rows(EchoModel(decide(CLIMB_EXTREME)), renamed[:1], split="val", train_ids={"old_id"},
                            out_dir=tmp_path, train_fingerprints=frozenset(trained))
    assert metrics["scored"] == 0 and metrics["excluded_trained_rows"] == ["r0"]


def _row(rid, group, scene, text=None):
    query = {"scene": scene, "player_action": "I try it."}
    return {"id": rid, "group": group, "input": query, "similarity_text": text or scene,
            "source": "s", "split": "val", "reference": ref(CLIMB_EXTREME)}


def test_rows_sharing_a_scenario_with_a_training_row_are_related():
    from check_model.evaluate import training_relatives

    rows_now = [
        _row("t1", "t1", "A wet stone wall around a courtyard, three metres high."),         # trained
        _row("t1b", "t1", "A wet stone wall around a courtyard, three metres high, at dusk."),  # its new sibling
        _row("x", "x", "A merchant asks fifty copper coins for a used backpack."),           # unrelated
    ]
    related = training_relatives(rows_now, train_ids={"t1"}, train_fingerprints=frozenset(),
                                 train_texts=[], threshold=None)
    assert related == {"t1b"}


def test_a_near_duplicate_of_a_removed_training_row_is_related():
    from check_model.evaluate import training_relatives

    removed = "The rain has just stopped; a slippery three metre stone wall blocks the courtyard."
    rows_now = [
        _row("new", "new", "The rain has just stopped; a slippery three metre stone wall blocks the yard."),
        _row("x", "x", "A merchant asks fifty copper coins for a used backpack at the market."),
    ]
    related = training_relatives(rows_now, train_ids={"gone"}, train_fingerprints=frozenset(),
                                 train_texts=[removed], threshold=0.38)
    assert related == {"new"}


def test_related_rows_are_excluded_and_listed(tmp_path):
    metrics = evaluate_rows(EchoModel(decide(CLIMB_EXTREME)), rows(), split="val", train_ids=set(),
                            out_dir=tmp_path, related_to_training=frozenset({"r1"}))
    assert metrics["excluded_related_rows"] == ["r1"] and metrics["scored"] == 2


def test_explicit_links_to_a_removed_training_row_still_relate():
    from check_model.evaluate import training_relatives

    points_at = dict(_row("a", "a", "Consulting the archive's registers."), links=["gone"])
    pointed_to = _row("b", "b", "An unrelated-looking scene about banners.")
    unrelated = _row("c", "c", "A merchant asks fifty copper coins for a used backpack.")
    related = training_relatives([points_at, pointed_to, unrelated], train_ids={"gone"},
                                 train_fingerprints=frozenset(), train_texts=[], threshold=None,
                                 train_links={"gone": ["b"]})
    assert related == {"a", "b"}  # threshold None: links alone, no similarity needed


def test_a_scenario_member_reached_only_through_a_removed_chain_stays_related():
    from check_model.evaluate import training_relatives

    # Trained 000028 <- unsupported 000029 <- 000047, pending then and trainable now; 000028 has
    # since been removed and 000029 is still in no split file.
    candidate = dict(_row("dice_train_000047", "dice_train_000029", "A later-annotated variation."),
                     links=["dice_train_000029"])
    known = dict(train_ids={"dice_train_000028"}, train_fingerprints=frozenset(), train_texts=[],
                 threshold=None, train_links={"dice_train_000028": ["dice_train_000029"]})
    assert training_relatives([candidate], **known) == set()  # no direct link, no shared current group
    scenario = {"dice_train_000028", "dice_train_000029", "dice_train_000047"}
    assert training_relatives([candidate], **known, train_scenario_ids=scenario) == {"dice_train_000047"}


def test_an_evaluation_emptied_by_related_rows_says_so(tmp_path):
    metrics = evaluate_rows(EchoModel(decide(CLIMB_EXTREME)), rows(), split="val", train_ids={"r0"},
                            out_dir=tmp_path, related_to_training=frozenset({"r1", "r2"}))
    assert metrics["scored"] == 0
    assert metrics["note"] == ("nothing was scored: every val row was used in training (1) "
                               "or belongs to a training row's scenario (2)")


def test_a_row_grouped_with_a_newly_related_row_is_related_too():
    from check_model.evaluate import training_relatives

    # bridge -> the removed training row; candidate -> bridge, so both share bridge's group now.
    bridge = dict(_row("bridge", "bridge", "A bridge scene."), links=["removed"])
    candidate = dict(_row("candidate", "bridge", "A candidate scene."), links=["bridge"])
    unrelated = dict(_row("other", "other", "An unrelated scene."), links=[])
    related = training_relatives([bridge, candidate, unrelated], train_ids={"removed"},
                                 train_fingerprints=frozenset(), train_texts=[], threshold=None,
                                 train_links={"removed": []})
    assert related == {"bridge", "candidate"}
