"""
@pyne

A function instantiated in several call-site contexts converts a value the way each
context needs: an int pushed into a float array is stored as a float, the same int
pushed into an int array by another call of the same function stays an int, and a
typeless ``na`` pushed into a string array is the empty string while the same ``na``
pushed into a float array stays na. A method called through ``method_call`` converts the
same way.
"""
from pynecore.core.pine_method import method, method_call
from pynecore.lib import array, na, script, string


def put(a, v):
    array.push(a, v)
    return a


def put_twice(a, v):
    return put(put(a, v), v)


def put_na(a):
    array.push(a, na)
    return a


@method
def stash(a, v):
    array.push(a, v)
    return a


@script.indicator("Int Into Container Per Context")
def main():
    floats: list[float] = array.new_float(0)
    ints: list[int] = array.new_int(0)
    put(floats, 1)
    put(ints, 1)

    nested_floats: list[float] = array.new_float(0)
    nested_ints: list[int] = array.new_int(0)
    put_twice(nested_floats, 2)
    put_twice(nested_ints, 2)

    strings: list[str] = array.new_string(0)
    na_floats: list[float] = array.new_float(0)
    put_na(strings)
    put_na(na_floats)

    method_floats: list[float] = array.new_float(0)
    method_ints: list[int] = array.new_int(0)
    method_call(stash, method_floats, 3)
    method_call(stash, method_ints, 3)

    return {
        "floats": string.format("{0}", floats),
        "ints": string.format("{0}", ints),
        "nested_floats": string.format("{0}", nested_floats),
        "nested_ints": string.format("{0}", nested_ints),
        "string_na_is_empty": array.get(strings, 0) == "",
        "float_na_is_na": na(array.get(na_floats, 0)),
        "method_floats": string.format("{0}", method_floats),
        "method_ints": string.format("{0}", method_ints),
    }


def __test_int_into_container_per_context__(runner, dummy_ohlcv_iter):
    """each instance of a shared body converts as its own call-site context needs"""
    result = None
    for _candle, plot in runner(dummy_ohlcv_iter).run_iter():
        result = plot
        break
    assert result == {
        "floats": "[1.0]",
        "ints": "[1]",
        "nested_floats": "[2.0, 2.0]",
        "nested_ints": "[2, 2]",
        "string_na_is_empty": True,
        "float_na_is_na": True,
        "method_floats": "[3.0]",
        "method_ints": "[3]",
    }
