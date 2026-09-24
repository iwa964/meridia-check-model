"""Score predictions field by field, and save them for a person to read.

Each field is scored only where it applies:

* `format_valid` -- every row: the reply parsed as one decision with catalog labels.
* `roll_required` -- every row (each has a creator decision).
* `check` (skill or attribute) and `difficulty` -- only rows whose reference requires a
  roll. A reference may accept several options (`alternative_examples`); a prediction is
  right if it matches any accepted option. `difficulty` is scored against the option whose
  kind and name the prediction chose, or against every accepted option's difficulty when
  it chose none of them, so the two fields are measured separately.
* `exact` -- the whole decision equals an accepted option.

An unparseable reply counts as wrong on every field it applies to.

Rows the run's manifest lists as training rows are excluded from the metrics of any split
and counted, so re-preparing the data after training cannot leak them into validation.
Scoring training rows on purpose (smoke mode, `--split train`) is labelled `held_out:
false`. Only the `test` split -- Meridia-specific scenes -- is `independent_test`;
validation is held out but drawn from the same general scenes as training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def input_fingerprint(query: dict) -> str:
    """A training input's identity independent of its id: a record renamed after training keeps
    its fingerprint, so it is still recognised as trained on."""
    return hashlib.sha256(json.dumps(query, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def training_relatives(all_rows: list[dict], *, train_ids: set[str], train_fingerprints: frozenset[str],
                        train_texts: list[str], threshold: float | None,
                        train_links: dict[str, list[str]] | None = None) -> set[str]:
    """Ids of rows that are not training rows themselves but belong to the same scenario as one:
    they share a current group with a training row (a link or similarity added after training
    joins them), they are explicitly linked to or from a training row (recorded at training,
    so this survives the training row's removal), or their text is a near-duplicate of a
    training row's (which also survives removal). Scoring them would leak the scenario."""
    from .splits import similar_pairs

    trained = {r["id"] for r in all_rows
               if r["id"] in train_ids or input_fingerprint(r["input"]) in train_fingerprints}
    groups = {r["group"] for r in all_rows if r["id"] in trained}
    related = {r["id"] for r in all_rows if r["group"] in groups and r["id"] not in trained}
    linked_from_training = {i for links in (train_links or {}).values() for i in links}
    related |= {r["id"] for r in all_rows if r["id"] not in trained
                and (r["id"] in linked_from_training or set(r.get("links") or []) & set(train_ids))}
    if threshold is not None and train_texts:
        texts = {r["id"]: r.get("similarity_text") or "" for r in all_rows if r["id"] not in trained}
        texts.update({f"\0train{i}": t for i, t in enumerate(train_texts)})
        for a, b, _ in similar_pairs(texts, threshold):
            a_train, b_train = a.startswith("\0train"), b.startswith("\0train")
            if a_train != b_train:
                related.add(b if a_train else a)
    return related


def _key(check: dict) -> tuple[str, str]:
    return check["kind"], check["name"]


def score(reference: dict, decision: dict | None) -> dict:
    valid = decision is not None
    out = {"format_valid": valid, "roll_required": valid and decision["roll_required"] == reference["roll_required"]}
    if not reference["roll_required"]:
        out.update(check=None, difficulty=None, exact=valid and not decision["roll_required"])
        return out
    options = [opt[0] for opt in reference["options"] if len(opt) == 1]
    predicted = decision["checks"][0] if valid and decision["checks"] else None
    if predicted is None:
        out.update(check=False, difficulty=False, exact=False)
        return out
    matching = [o for o in options if _key(o) == _key(predicted)]
    allowed = {o["difficulty"] for o in (matching or options)}
    out.update(check=bool(matching), difficulty=predicted["difficulty"] in allowed,
               exact=any(o == predicted for o in matching))
    return out


FIELDS = ("format_valid", "roll_required", "check", "difficulty", "exact")


def summarize(scored: list[dict]) -> dict:
    out = {}
    for f in FIELDS:
        applicable = [s[f] for s in scored if s[f] is not None]
        correct = sum(1 for v in applicable if v)
        out[f] = {"correct": correct, "total": len(applicable),
                  "rate": round(correct / len(applicable), 4) if applicable else None}
    return out


def evaluate_rows(model, rows: list[dict], *, split: str, train_ids: set[str], out_dir: str | Path,
                  include_training_rows: bool = False, batch_size: int = 8,
                  data_revision: dict | None = None, train_fingerprints: frozenset[str] = frozenset(),
                  related_to_training: frozenset[str] = frozenset()) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def trained(row: dict) -> bool:
        return row["id"] in train_ids or input_fingerprint(row["input"]) in train_fingerprints

    def excluded(row: dict) -> bool:
        return trained(row) or row["id"] in related_to_training

    seen = [r for r in rows if trained(r)]
    related = [r for r in rows if not trained(r) and r["id"] in related_to_training]
    scored_rows = rows if include_training_rows else [r for r in rows if not excluded(r)]
    predictions = model.predict_many([r["input"] for r in scored_rows], batch_size=batch_size)
    records, scores = [], []
    for row, pred in zip(scored_rows, predictions):
        s = score(row["reference"], pred["decision"])
        scores.append(s)
        records.append({
            "id": row["id"], "source": row["source"], "split": row["split"], "input": row["input"],
            "reference": row["reference"], "raw_output": pred["raw_output"], "decision": pred["decision"],
            "format_errors": pred["errors"], "scores": s, "game_request": pred["game_request"],
            "note": pred["note"], "trained_on": trained(row),
        })
    held_out = bool(records) and not any(r["trained_on"] for r in records) and split != "train"
    if not records:
        note = (f"the {split} split is empty; nothing was scored" if not rows else
                f"nothing was scored: every {split} row was used in training")
    elif not held_out:
        note = "NOT a held-out result: scored rows were used in training. This verifies the pipeline only."
    elif split == "val":
        note = ("held-out validation on general scenes, the same distribution as training; "
                "not the independent Meridia-specific test")
    else:
        note = "independent test on Meridia-specific scenes"
    metrics = {
        "split": split,
        "held_out": held_out,
        "independent_test": held_out and split == "test",
        "note": note,
        "scored": len(records),
        "excluded_trained_rows": [] if include_training_rows else sorted(r["id"] for r in seen),
        "excluded_related_rows": [] if include_training_rows else sorted(r["id"] for r in related),
        "data_revision": data_revision,
        "fields": summarize(scores),
    }
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics
