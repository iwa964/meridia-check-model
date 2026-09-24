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
    seen, repeated = set(), set()
    for key, _ in pairs:  # one pass: keys.count() per key was quadratic in the object's size
        (repeated if key in seen else seen).add(key)
    if repeated:
        raise ValueError(f"duplicate key(s) {sorted(repeated)} in one object")
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
    """json.loads, raising ValueError on a repeated key, a NaN / Infinity constant, a number
    too large for a finite float, or nesting deeper than the decoder can follow."""
    try:
        return json.loads(text, object_pairs_hook=_unique, parse_constant=_no_constant,
                          parse_float=_finite_float)
    except RecursionError:
        # A few thousand nested brackets exhaust the interpreter's recursion limit. That is bad
        # input like any other, not an exception past every caller's ValueError handler.
        raise ValueError("nested too deeply to decode") from None
