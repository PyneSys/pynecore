"""
@pyne

Pine's typeless ``na`` put where a bool goes is the bool na the script runs with.
Pine v6 refuses it there at compile time; a v5 source compiled two-state and
hand-written Pyne code still spell it, and with the default ``na_bool=False``
the only bool na is ``false``: every element, variable, parameter, ternary arm,
map value, matrix element and UDT field below reads back as ``false``.
"""
from pynecore.core.pine_udt import udt
from pynecore.lib import script, na, array, map, matrix, string, bar_index
from pynecore.types.matrix import Matrix


@udt
class Flag:
    f: bool = na


def is_false(x: bool) -> bool:
    return x == False  # noqa: E712


@script.indicator(title="typeless na bool, two-state")
def main():
    a_new = array.new_bool(1, na)
    a_set = array.new_bool(1, True)
    array.set(a_set, 0, na)
    a_push = array.new_bool(0)
    array.push(a_push, na)
    a_fill = array.new_bool(1, True)
    array.fill(a_fill, na)
    a_unshift = array.new_bool(0)
    array.unshift(a_unshift, na)
    a_insert = array.new_bool(0)
    array.insert(a_insert, 0, na)
    v: bool = na
    t: bool = True if bar_index < 0 else na
    m: dict[str, bool] = map.new()
    map.put(m, "k", na)
    mx: Matrix[bool] = matrix.new(1, 1, na)
    return {
        "new_bool": array.get(a_new, 0),
        "set": array.get(a_set, 0),
        "push": array.get(a_push, 0),
        "fill": array.get(a_fill, 0),
        "unshift": array.get(a_unshift, 0),
        "insert": array.get(a_insert, 0),
        "var": v,
        "ternary": t,
        "map": map.get(m, "k"),
        "matrix": matrix.get(mx, 0, 0),
        "udt": Flag.new().f,
        "param": is_false(na),
        "eq": array.get(a_new, 0) == False,  # noqa: E712
        "tostr": string.tostring(array.get(a_new, 0)),
    }


def __test_typeless_na_where_a_bool_goes_is_false__(csv_reader, runner):
    """Every typeless na in a bool position reads back as false on the first bar"""
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for candle, plot in runner(cr).run_iter():
            for key in ("new_bool", "set", "push", "fill", "unshift", "insert", "var",
                        "ternary", "map", "matrix", "udt"):
                assert plot[key] is False, f"{key} is {plot[key]!r}"
            assert plot["param"] is True
            assert plot["eq"] is True
            assert plot["tostr"] == "false"
            break
