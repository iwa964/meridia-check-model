import copy


def record(data, rid):
    for section, value in data.items():
        if isinstance(value, list):
            for r in value:
                if r.get("id") == rid:
                    return r
    raise KeyError(rid)


def clone(data, rid, new_id):
    r = copy.deepcopy(record(data, rid))
    r["id"] = new_id
    return r
