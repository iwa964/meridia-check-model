from pathlib import Path

import pytest

from check_model.catalog import git_blob_sha, parse_skill_bank, sync_from_meridia

MERIDIA = Path(__file__).resolve().parent.parent.parent / "MeridiaGame"


def test_blob_sha_matches_git():
    # `printf 'hello\n' | git hash-object --stdin`
    assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_rollable_is_nonblank_initial(catalog):
    blank = {s["name"] for s in catalog.skills if not s["initial"].strip()}
    assert set(catalog.special_skills) == blank
    assert not blank & set(catalog.rollable_skills)
    assert len(catalog.rollable_skills) + len(catalog.special_skills) == len(catalog.skills)


@pytest.mark.parametrize("entry, error", [
    ({"kind": "skill", "name": "Climbing", "difficulty": "extreme"}, None),
    ({"kind": "attribute", "name": "STR", "difficulty": "success"}, None),
    # 000017's first attempt: the creator's "maintenance", which the catalog does not have.
    ({"kind": "skill", "name": "maintenance", "difficulty": "success"}, "unknown skill"),
    ({"kind": "skill", "name": "Void Sense", "difficulty": "success"}, "special skill"),
    ({"kind": "skill", "name": "STR", "difficulty": "success"}, "unknown skill"),
    ({"kind": "attribute", "name": "Climbing", "difficulty": "success"}, "unknown attribute"),
    # Stored as "success" (annotation_policy.difficulty_normalization), never as "normal".
    ({"kind": "skill", "name": "Climbing", "difficulty": "normal"}, "difficulty must be"),
    ({"kind": "skil", "name": "Climbing", "difficulty": "hard"}, "kind must be"),
])
def test_entry_error(catalog, entry, error):
    got = catalog.entry_error(entry)
    assert (got is None) if error is None else (error in got)


def test_changed_row_format_is_refused():
    text = ('{"name": "Climbing", "category": "general", "subcategory": "athletics", "initial": "20"},\n'
            '{"name": "Jumping", "category": "general", "initial": "20"},\n')
    with pytest.raises(ValueError, match="row format changed"):
        parse_skill_bank(text)


@pytest.mark.skipif(not (MERIDIA / "Scripts").is_dir(), reason="no MeridiaGame checkout beside this repo")
def test_sync_reproduces_snapshot_when_catalog_unchanged(catalog):
    fresh = sync_from_meridia(MERIDIA)
    if fresh.skill_blob_sha != catalog.skill_blob_sha:
        pytest.skip("MeridiaGame's SkillBank moved on; run sync-catalog")
    assert fresh.skills == catalog.skills
    assert fresh.attributes == catalog.attributes


# Two rows and the three constants exactly as MeridiaGame's SkillBank.gd and check_turn.py carry them.
SKILL_ROWS = ('\t{"name": "Climbing", "category": "general", "subcategory": "athletics", "initial": "20"},\n'
              '\t{"name": "Swimming", "category": "general", "subcategory": "athletics", "initial": "20"},\n')
CONSTANTS = ('DIFFICULTIES = ("success", "hard", "extreme")\nKINDS = ("skill", "attribute")\n'
             'ATTRIBUTES = ("STR", "CON", "SIZE", "DEX", "INT", "EDU", "WIL", "APP")\n')


def fake_meridia(root: Path, constants: str = CONSTANTS) -> Path:
    (root / "Scripts/profile/skill").mkdir(parents=True)
    (root / "server").mkdir()
    (root / "Scripts/profile/skill/SkillBank.gd").write_text(SKILL_ROWS, encoding="utf-8")
    (root / "server/check_turn.py").write_text(constants, encoding="utf-8")
    return root


def git(root: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
                   check=True, capture_output=True)


def test_sync_refuses_uncommitted_catalog_sources(tmp_path):
    root = fake_meridia(tmp_path / "MeridiaGame")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-qm", "catalog")
    assert sync_from_meridia(root).source["commit"]  # clean: HEAD holds exactly these bytes
    (root / "Scripts/profile/skill/SkillBank.gd").write_text(SKILL_ROWS.replace('"20"', '"25"'), encoding="utf-8")
    with pytest.raises(ValueError, match="uncommitted changes to the catalog sources"):
        sync_from_meridia(root)


def test_sync_refuses_a_difficulty_the_prompt_cannot_describe(tmp_path):
    from check_model.__main__ import main

    root = fake_meridia(tmp_path / "MeridiaGame",
                        CONSTANTS.replace('"extreme")', '"extreme", "critical")'))
    out = tmp_path / "catalog.json"
    with pytest.raises(SystemExit, match=r"no rule text for \['critical'\]"):
        main(["sync-catalog", "--meridia", str(root), "--out", str(out)])
    assert not out.exists()


def test_sync_refuses_a_kind_this_code_does_not_implement(tmp_path):
    from check_model.__main__ import main

    root = fake_meridia(tmp_path / "MeridiaGame",
                        CONSTANTS.replace('KINDS = ("skill", "attribute")', 'KINDS = ("skill", "attribute", "save")'))
    out = tmp_path / "catalog.json"
    with pytest.raises(SystemExit, match=r"catalog kinds \['skill', 'attribute', 'save'\]"):
        main(["sync-catalog", "--meridia", str(root), "--out", str(out)])
    assert not out.exists()
