"""JSON decoding that refuses a key repeated within one object, at any depth.

`json.loads` keeps only the last of repeated keys, without a word: two `examples` sections left
by a merge lose every record of the first, and a model reply with two `roll_required` fields
reads as whichever came last -- where another parser may read the first.
"""

from __future__ import annotations

import json
import math
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


def _finite_float(token: str) -> float:
    # 1e999 is a valid JSON number that Python reads as inf, which json.dumps writes as Infinity.
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"{token} is out of range for a float")
    return value


def loads(text: str) -> Any:
    """json.loads, raising ValueError on a repeated key, a NaN / Infinity constant, or a number
    too large for a finite float."""
    return json.loads(text, object_pairs_hook=_unique, parse_constant=_no_constant,
                      parse_float=_finite_float)
