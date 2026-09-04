"""
@pyne

A Pine int is a double at runtime. Every int-typed value a script sees -- the
bar index, an ``input.int``, a cast, a count, a counter, a ``for`` loop
counter, an int series' history -- arrives as a native float, and the numeric
na of an int is the same nan a float has. A Python-native consumer of such a
value (``range()``) is truncated by the pipeline.
"""
from pynecore.lib import script, bar_index, input, array, math, na, ta
from pynecore.core.pine_range import pine_range
from pynecore.types.persistent import Persistent
from pynecore.types.series import Series


@script.indicator(title="float-only int")
def main():
    length = input.int(3)
    count: Persistent[int] = 0
    count += 1
    hist: Series[int] = 2 * 1
    a = array.new_int(0)
    for i in pine_range(0, 2):
        array.push(a, i)
    seen = 0
    for _ in range(array.size(a)):
        seen += 1
    return {
        "bar_index": bar_index,
        "length": length,
        "count": count,
        "hist_prev": hist[1],
        "cast": int(bar_index / 2),
        "floor": math.floor(2.7),
        "size": array.size(a),
        "first": array.get(a, 0),
        "seen": seen,
        "barssince": ta.barssince(bar_index == 2),
        "na_int": na(int),
    }


def __test_every_int_typed_value_is_a_float__(csv_reader, runner):
    """The bar index, inputs, casts, counts and int series all travel as floats"""
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            for key in ("bar_index", "length", "count", "cast", "floor", "size", "first"):
                assert type(plot[key]) is float, f"bar {i}: {key} is {plot[key]!r}"
            assert plot["bar_index"] == i
            assert plot["length"] == 3
            assert plot["count"] == i + 1
            assert plot["cast"] == i // 2
            assert plot["floor"] == 2
            assert plot["size"] == 3
            assert plot["first"] == 0
            # A Pine int fed to range() is truncated by the pipeline
            assert plot["seen"] == 3
            # An int series' history: the value is a float, its na the native nan
            if i == 0:
                assert plot["hist_prev"] != plot["hist_prev"]
            else:
                assert type(plot["hist_prev"]) is float and plot["hist_prev"] == 2
            # An int na is the native nan, like a float na
            assert type(plot["na_int"]) is float and plot["na_int"] != plot["na_int"]
            # An int-returning builtin answers na as the nan, then floats
            if i < 2:
                assert plot["barssince"] != plot["barssince"]
            else:
                assert type(plot["barssince"]) is float and plot["barssince"] == i - 2
            if i >= 5:
                break


def __test_the_loop_counter_is_a_float__():
    """A Pine for loop counts in floats, its bounds are not truncated"""
    assert list(pine_range(0, 3)) == [0.0, 1.0, 2.0, 3.0]
    assert all(type(x) is float for x in pine_range(0, 3))
    assert list(pine_range(3.0, 1.0)) == [3.0, 2.0, 1.0]
    assert list(pine_range(0, 1, 0.5)) == [0.0, 0.5, 1.0]
    assert list(pine_range(float('nan'), 3)) == []
