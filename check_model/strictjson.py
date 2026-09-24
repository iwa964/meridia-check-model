"""JSON decoding that refuses a key repeated within one object, at any depth.

`json.loads` keeps only the last of repeated keys, without a word: two `examples` sections left
by a merge lose every record of the first, and a model reply with two `roll_required` fields
reads as whichever came last -- where another parser may read the first.
"""

from __future__ import annotations

import json
from typing import Any


def _unique(pairs: list[tuple[str, Any]]) -> dict:
    keys = [k for k, _ in pairs]
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        raise ValueError(f"duplicate key(s) {repeated} in one object")
    return dict(pairs)


def _no_constant(name: str) -> Any:
    # json.loads accepts NaN and Infinity by default; they are not JSON, and json.dumps would
    # write them straight back into a prompt.
    raise ValueError(f"{name} is not valid JSON")


def loads(text: str) -> Any:
    """json.loads, raising ValueError on a repeated key or a NaN / Infinity constant."""
    return json.loads(text, object_pairs_hook=_unique, parse_constant=_no_constant)
