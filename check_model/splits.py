"""Group near-duplicates, then split by group.

* `game_specific` rows are the independent test set. `general` rows are split into train
  and validation.
* Rows that are variations of one scenario share a group, and a group never straddles
  two splits. A group is formed by the dataset's own `related_example_id` links, by
  `extra_groups` in the config, and by text similarity (TF-IDF cosine over scene + action in
  every language the record carries, not just the one the prompt shows) at or above
  `near_duplicate_threshold`. Every pair joined by similarity
  is listed in the report so a person can check it.
* A group is assigned by hashing its key (its smallest id) with `split_seed`, not by
  shuffling, so adding unrelated examples does not move old ones. It is not frozen: a new row
  that joins an existing group and sorts first, or a similarity merge (IDF shifts as data
  grows), can change a group's key and move it. `prepare` warns about every row that moved
  since the last run, and `evaluate` excludes trained rows whatever split they are in now.
* A group holding an eval-only row goes to validation: the row cannot be trained on, and
  training on its siblings would leak it.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter

from .adapter import Report, Row

_TOKEN = re.compile(r"[a-z0-9]+|[一-鿿]")
#: Floating-point slack in a cosine score, far below any difference between two texts.
SCORE_TOLERANCE = 1e-9


def _text(row_input: dict) -> str:
    return " ".join(str(v) for k, v in row_input.items() if isinstance(v, str))


def similar_pairs(texts: dict[str, str], threshold: float) -> list[tuple[str, str, float]]:
    ids = sorted(texts)
    tokens = {i: _TOKEN.findall(texts[i].lower()) for i in ids}
    df = Counter(t for i in ids for t in set(tokens[i]))
    n = len(ids)
    vectors = {}
    for i in ids:
        counts = Counter(tokens[i])
        # Smoothed IDF (never zero): with a raw log(n/df), a term in every row weighs nothing,
        # so in a small corpus two near-identical rows could score 0.
        v = {t: c * (math.log((1 + n) / (1 + df[t])) + 1) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vectors[i] = {t: x / norm for t, x in v.items()}
    out = []
    for a, b in itertools.combinations(ids, 2):
        va, vb = vectors[a], vectors[b]
        score = sum(x * vb.get(t, 0.0) for t, x in va.items())
        # Normalising leaves identical vectors a hair under 1.0 (0.9999999999999998): within
        # SCORE_TOLERANCE of the threshold counts, so a threshold of 1.0 still pairs exact copies.
        if score >= threshold - SCORE_TOLERANCE:
            out.append((a, b, round(min(score, 1.0), 3)))
    return out


class _Groups:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # The smaller id names the group: stable while new rows sort after it.
            self.parent[max(ra, rb)] = min(ra, rb)


def _in_validation(group: str, seed: int | str, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < fraction


def assign_splits(rows: list[Row], report: Report, *, val_fraction: float, split_seed: int | str,
                  near_duplicate_threshold: float | None, extra_groups: list[list[str]]) -> None:
    groups = _Groups()
    known = set(report.all_ids)
    for rid, links in report.all_ids.items():
        groups.find(rid)
        for other in links:
            if other in known:
                groups.union(rid, other)
            else:
                report.warnings.append(f"{rid} links to {other}, which is in no source; the link is ignored")
    for group in extra_groups:
        unknown = [i for i in group if i not in known]
        if unknown:
            report.errors.append({"source": "config data.extra_groups", "id": None,
                                  "message": f"group {group} names ids in no source: {unknown}"})
            continue
        for a, b in zip(group, group[1:]):
            groups.union(a, b)
    if near_duplicate_threshold is not None:
        texts = {r.id: r.similarity_text or _text(r.input) for r in rows}
        for a, b, score in similar_pairs(texts, near_duplicate_threshold):
            groups.union(a, b)
            report.near_duplicates.append({"a": a, "b": b, "similarity": score})

    eval_only_groups = {groups.find(r.id) for r in rows if r.target is None}
    # Every known id per group, unsupported and pending records included: a scenario can be
    # joined through a record that never reaches a split file (trained -> unsupported ->
    # pending), and only this membership still says so once the training row is gone.
    members: dict[str, list[str]] = {}
    for rid in sorted(known):
        members.setdefault(groups.find(rid), []).append(rid)
    for row in rows:
        row.group = groups.find(row.id)
        row.group_members = list(members[row.group])
        if row.scope == "game_specific":
            row.split = "test"
        elif row.group in eval_only_groups or _in_validation(row.group, split_seed, val_fraction):
            row.split = "val"
        else:
            row.split = "train"
    by_group: dict[str, set[str]] = {}
    for row in rows:
        by_group.setdefault(row.group, set()).add(row.split)
    for group, splits in by_group.items():
        if len(splits) > 1:
            # A group mixing general and game-specific rows cannot be kept whole.
            report.errors.append({"source": None, "id": group, "message":
                                  f"group {group} spans splits {sorted(splits)}: a scenario's variations "
                                  "mix general and game_specific scope"})
    report.splits = {s: sorted(r.id for r in rows if r.split == s) for s in ("train", "val", "test")}
