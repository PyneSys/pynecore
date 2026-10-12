"""
@pyne

An int that goes into a float-element container is stored as a float, and an overload
over containers is chosen by their static element type.

MEASURED on TradingView (CAPITALCOM:EURUSD 60, last bar): ``array.from(1, 2.0)`` then
``push(2)``, ``set(0, n)`` and ``push(bar_index * 0 + 7)`` formats as ``[3.0, 2.0, 2.0, 7.0]``;
a ``map<string, float>`` put of ``1`` reads back ``1.0``; ``concat`` of ``array.from(5, 6)``
into a float array gives ``[1.5, 5.0, 6.0]``; ``matrix.new<float>(1, 1, -3)`` holds ``-3.0``
and ``set(0, 0, 5)`` / ``add_row`` of an int array ``5.0`` / ``8.0``; ``fill(4)``,
``insert(0, 9)``, ``unshift(8)`` on a float array give ``[8.0, 9.0, 4.0, 4.0]``. The
``array<int>`` / ``array<float>`` overload pair picks FLOAT for an empty float array and for
``array.from(1, 2.0)``, INT for an int array; the matrix pair picks FLOAT for an empty and for
an int-initialized ``matrix<float>``.
"""
from pynecore.lib import array, bar_index, map, matrix, na, script, string
from pynecore.core.overload import overload
from pynecore.types import Matrix


# noinspection PyUnusedLocal
@overload
def f(a: list[int]) -> str:
    return "INT"


# noinspection PyUnusedLocal
@overload
def f(a: list[float]) -> str:
    return "FLOAT"


# noinspection PyUnusedLocal
@overload
def g(m: Matrix[int]) -> str:
    return "INT"


# noinspection PyUnusedLocal
@overload
def g(m: Matrix[float]) -> str:
    return "FLOAT"


@script.indicator("Int Into Float Container")
def main():
    n: int = 3
    a: list[float] = array.from_items(1, 2.0)
    array.push(a, 2)
    array.set(a, 0, n)
    array.push(a, bar_index * 0 + 7)

    mp: dict[str, float] = map.new()
    map.put(mp, 'a', 1)

    b: list[float] = array.from_items(1.5)
    array.concat(b, array.from_items(5, 6))

    m: Matrix[float] = matrix.new(1, 1, -3)
    first = string.format("{0}", matrix.row(m, 0))
    matrix.set(m, 0, 0, 5)
    matrix.add_row(m, 1, array.from_items(8))

    af: list[float] = array.new_float(2, 1)
    array.fill(af, 4)
    array.insert(af, 0, 9)
    array.unshift(af, 8)

    # An int array holding a double-valued Pine int: the head element is a float, the
    # static type is not
    ai: list[int] = array.new_int()
    array.push(ai, string.length("ab"))

    return {
        "e1": string.format("{0}", a),
        "e2": string.format("{0}", array.from_items(map.get(mp, 'a'))),
        "e3": string.format("{0}", b),
        "e4": first,
        "e5": string.format("{0} {1}", matrix.row(m, 0), matrix.row(m, 1)),
        "e6": string.format("{0} {1} {2}", f(array.new_float(0)), f(array.new_int(0)),
                            g(matrix.new(0, 0, na(float)))),
        "e7": string.format("{0} {1} {2}", f(ai), g(m), f(array.from_items(1, 2.0))),
        "e8": string.format("{0}", af),
    }


def __test_int_into_float_container__(runner, dummy_ohlcv_iter):
    """every write path stores the float TradingView stores, and overloads follow the element type"""
    result = None
    for _candle, plot in runner(dummy_ohlcv_iter).run_iter():
        result = plot
        break
    assert result == {
        "e1": "[3.0, 2.0, 2.0, 7.0]",
        "e2": "[1.0]",
        "e3": "[1.5, 5.0, 6.0]",
        "e4": "[-3.0]",
        "e5": "[5.0] [8.0]",
        "e6": "FLOAT INT FLOAT",
        "e7": "INT FLOAT FLOAT",
        "e8": "[8.0, 9.0, 4.0, 4.0]",
    }
