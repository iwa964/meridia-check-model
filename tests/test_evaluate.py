import json

from check_model.evaluate import evaluate_rows, score, summarize

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
