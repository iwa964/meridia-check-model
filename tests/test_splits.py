from pathlib import Path

from helpers import clone

from check_model.adapter import load_rows
from check_model.splits import assign_splits, similar_pairs

SUBSET = Path(__file__).resolve().parent / "fixtures" / "dice_subset.json"


def split(catalog, paths, *, fraction=0.5, seed=0, threshold=None, extra=()):
    rows, report = load_rows([str(p) for p in paths], catalog, "en")
    assign_splits(rows, report, val_fraction=fraction, split_seed=seed,
                  near_duplicate_threshold=threshold, extra_groups=[list(g) for g in extra])
    return {r.id: r for r in rows}, report


def test_related_example_links_are_read_from_the_record(catalog):
    _, report = split(catalog, [SUBSET])
    # 000028 names 000029 in creator_guidance; 000029 names 000028 at the top level.
    assert report.all_ids["dice_train_000028"] == ["dice_train_000029"]
    assert report.all_ids["dice_train_000029"] == ["dice_train_000028"]


def test_related_example_links_form_one_group(catalog, subset, write_source):
    variation = clone(subset, "dice_train_000002", "dice_train_000950")
    variation["scene"]["en"] = "A variation. " + variation["scene"]["en"]
    variation["related_example_id"] = "dice_train_000033"  # the shape 000029 uses
    subset["examples"].append(variation)
    for seed in range(20):
        rows, _ = split(catalog, [write_source(subset)], seed=seed)
        assert rows["dice_train_000950"].group == rows["dice_train_000033"].group == "dice_train_000033"
        assert rows["dice_train_000950"].split == rows["dice_train_000033"].split


def test_groups_never_straddle_splits(catalog):
    for seed in range(20):
        rows, report = split(catalog, [SUBSET], seed=seed,
                             extra=[("dice_train_000001", "dice_train_000002", "dice_train_000021")])
        assert len({rows[i].split for i in ("dice_train_000001", "dice_train_000002", "dice_train_000021")}) == 1
        assert report.errors == []


def test_eval_only_rows_go_to_validation(catalog):
    for seed in range(20):
        rows, _ = split(catalog, [SUBSET], seed=seed, fraction=0.0)
        assert rows["dice_train_000038"].split == "val"
        assert {r.split for i, r in rows.items() if i != "dice_train_000038"} == {"train"}


def test_game_specific_rows_are_the_test_split(catalog, subset, write_source):
    subset["scenario_scope"] = "game_specific"
    for section in subset.values():
        if isinstance(section, list):
            for r in section:
                r["id"] = r["id"].replace("dice_train", "meridia_test")
    rows, report = split(catalog, [SUBSET, write_source(subset)])
    assert {r.split for i, r in rows.items() if i.startswith("meridia_test")} == {"test"}
    assert "test" not in {r.split for i, r in rows.items() if i.startswith("dice_train")}
    assert report.splits["test"]


def test_a_group_mixing_scopes_is_an_error(catalog, subset, write_source):
    extra = clone(subset, "dice_train_000002", "meridia_test_000001")
    extra["scene"]["en"] += " (game-specific copy)"
    extra["scenario_scope"] = "game_specific"
    subset["examples"].append(extra)
    _, report = split(catalog, [write_source(subset)], extra=[("dice_train_000002", "meridia_test_000001")])
    assert any("spans splits" in e["message"] for e in report.errors)


def test_adding_rows_does_not_move_existing_ones(catalog, subset, write_source):
    before, _ = split(catalog, [SUBSET], fraction=0.4)
    extra = clone(subset, "dice_train_000002", "dice_train_999999")
    extra["scene"]["en"] = "A different scene. " + extra["scene"]["en"]
    subset["examples"].append(extra)
    after, _ = split(catalog, [write_source(subset)], fraction=0.4)
    assert all(after[i].split == r.split for i, r in before.items())


def test_similar_texts_are_paired():
    texts = {"a": "a locked wooden chest and lock-picking tools",
             "b": "a locked wooden chest and lock-picking tools",
             "c": "the tavern is crowded and noisy with music"}
    assert similar_pairs(texts, 0.9) == [("a", "b", 1.0)]


def test_near_duplicates_are_grouped_and_reported(catalog, subset, write_source):
    near = clone(subset, "dice_train_000001", "dice_train_000900")
    near["player_action"]["en"] = "I climb the wall using the gaps between the stones."
    subset["examples"].append(near)
    rows, report = split(catalog, [write_source(subset)], threshold=0.5)
    assert rows["dice_train_000900"].group == rows["dice_train_000001"].group
    assert any({p["a"], p["b"]} == {"dice_train_000001", "dice_train_000900"} for p in report.near_duplicates)


def test_similarity_reads_every_language(catalog, subset, write_source):
    # Same scene in Chinese, reworded in English: only the Chinese text says they are one.
    variant = clone(subset, "dice_train_000001", "dice_train_000901")
    variant["scene"]["en"] = "Wet masonry blocks a courtyard; nobody is chasing and there is no deadline."
    variant["player_action"]["en"] = "Hand over hand I go up and over."
    subset["examples"].append(variant)
    rows, report = split(catalog, [write_source(subset)], threshold=0.4)
    assert rows["dice_train_000901"].group == rows["dice_train_000001"].group
