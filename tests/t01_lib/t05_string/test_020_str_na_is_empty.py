"""
@pyne

A na string is the empty string.

MEASURED on TradingView (CAPITALCOM:EURUSD 60, last bar, Pine v6 and v5 alike) with
``string s = na``: ``str.tostring(s)`` is ``""`` of length 0; ``str.format("{0}", s)`` is
``""``; ``str.tostring(array.from("a", s))`` and its ``str.format`` are ``[a, ]``;
``"x" + s + "y"`` is ``xy`` of length 2 and ``"x" + s`` is not na but equals ``"x"``;
``str.tostring(matrix.new<string>(1, 1))`` is ``[]``; ``na(s)``, ``na("")``,
``na(str.repeat("\\n", 0))``, ``na(s + s)``, ``na(str.lower(s))`` and
``na(str.substring("abc", 1, 1))`` are true; ``s == ""``, ``s != "a"``, ``str.upper(s) == ""``
and ``str.contains("abc", s)`` are true, ``str.contains(s, "a")`` is false;
``str.format(str.repeat("\\n", 0) + "{0}", 5)`` is ``5``; ``str.replace("abc", s, "-")`` is
``-abc`` and ``str.replace(s, "a", "-")`` is ``""``; ``str.substring(s, 0)`` is ``""``;
``array.sort`` of ``["b", s, "a"]`` prints ``[, a, b]``.

A typeless ``na`` that goes where a string goes -- a string declaration, a reassignment of
a string variable, a ternary arm next to a string, a string array or map element, a string
parameter, a string-keyed map's key -- is that same string na.
"""
from pynecore.lib import array, bar_index, map, matrix, na, script, string
from pynecore.types import NA


def tail(p: str) -> str:
    return "x" + p + "y"


@script.indicator("Na String Is Empty")
def main():
    s: str = na(str)
    r = string.repeat("\n", 0)
    joined = "x" + s
    sorted_items: list[str] = array.from_items("b", s, "a")
    array.sort(sorted_items)

    t: str = na
    u: str = "q"
    u = na
    w = "up" if bar_index < 0 else na
    a: list[str] = array.new_string()
    array.push(a, na)
    mp: dict[str, str] = map.new()
    map.put(mp, "k", na)
    keyed: dict[str, str] = map.new()
    map.put(keyed, na, "v")

    return {
        "g1": string.format("[{0}] {1}", string.tostring(s), string.length(string.tostring(s))),
        "g2": string.format("[{0}]", string.format("{0}", s)),
        "g3": string.tostring(array.from_items("a", s)),
        "g4": "x" + s + "y",
        "g6": string.format("{0}", array.from_items("a", s)),
        "g7": string.tostring(matrix.new(1, 1, initial_value=NA(str))),
        "h1": string.format("{0} {1} {2}", na(s), na(r), string.length(r)),
        "h2": string.format("{0} {1}", na(joined), joined == "x"),
        "h3": string.format(r + "{0}", 5),
        "h4": string.format("{0} {1} {2}", s == "", na(s + s), string.length("x" + s + "y")),
        "j1": string.format("{0} {1}", na(""), na(string.substring("abc", 1, 1))),
        "j3": string.format("{0} {1} {2}", s != "a", string.contains("abc", s),
                            string.contains(s, "a")),
        "j4": string.format("{0} {1}", na(string.lower(s)), string.upper(s) == ""),
        "l3": string.format("[{0}] [{1}] [{2}]", string.replace("abc", s, "-"),
                            string.replace(s, "a", "-"), string.substring(s, 0)),
        "l5": string.tostring(sorted_items),
        "t1": string.format("{0} {1} {2}", "x" + t + "y", string.length(t), u == ""),
        "t2": string.format("{0} {1}", "x" + w, w == ""),
        "t3": string.format("{0} {1}", string.tostring(a), map.get(mp, "k") == ""),
        "t4": tail(na),
        "t5": string.format("{0} {1} {2} {3}", string.length(na), string.endswith("abc", na),
                            string.lower(na) == "", string.trim(na) == ""),
        "t6": string.format("{0} {1} {2}", map.get(keyed, ""), map.get(keyed, na),
                            map.contains(keyed, na)),
    }


def __test_na_string_is_empty__(runner, dummy_ohlcv_iter):
    """a string na prints, concatenates, compares and searches as the empty string"""
    result = None
    for _candle, plot in runner(dummy_ohlcv_iter).run_iter():
        result = plot
        break
    assert result == {
        "g1": "[] 0",
        "g2": "[]",
        "g3": "[a, ]",
        "g4": "xy",
        "g6": "[a, ]",
        "g7": "[]",
        "h1": "true true 0",
        "h2": "false true",
        "h3": "5",
        "h4": "true true 2",
        "j1": "true true",
        "j3": "true true false",
        "j4": "true true",
        "l3": "[-abc] [] []",
        "l5": "[, a, b]",
        "t1": "xy 0 true",
        "t2": "x true",
        "t3": "[] true",
        "t4": "xy",
        "t5": "0 true true true",
        "t6": "v v true",
    }
