"""Sorting an array of objects by one of their fields (``sort_field``).

Deliberately not a ``@pyne`` module: the locally defined dataclass standing in for a
user UDT must not be run through the Pyne AST transform.

Measured on TradingView (BINANCE:BTCUSDT 1D) with the elements
``(v=3.0, i=30, s="c")``, ``(v=1.0, i=10, s="a")`` and ``(v=na, i=20, s="b")``:

* ``sort(a, order.ascending, sort_field="v")``  -> 10, 30, 20
* ``sort(a, order.descending, sort_field="v")`` -> 20, 30, 10

so the field decides the order while the whole element moves, and na lands where
the field's own type puts it -- last for a number, first for a string. The field
is also the third POSITIONAL parameter: ``array.sort(a, order.ascending, "v")``
compiles and sorts the same way.
"""
from dataclasses import dataclass
from math import nan

from pynecore.lib import array, order
from pynecore.types.na import NA


@dataclass(slots=True)
class _Reading:
    v: float = 0.0
    i: int = 0
    s: str = ""


def __test_helper_readings() -> list[_Reading]:
    return [_Reading(3.0, 30, "c"), _Reading(1.0, 10, "a"), _Reading(nan, 20, "b")]


def __test_sort_field_orders_by_the_field_and_moves_the_element__():
    """The numeric field puts na last ascending and first descending."""
    ascending = __test_helper_readings()
    array.sort(ascending, order.ascending, sort_field="v")
    assert [element.i for element in ascending] == [10, 30, 20]

    descending = __test_helper_readings()
    array.sort(descending, order.descending, sort_field="v")
    assert [element.i for element in descending] == [20, 30, 10]

    positional = __test_helper_readings()
    array.sort(positional, order.ascending, "v")
    assert [element.i for element in positional] == [10, 30, 20]


def __test_sort_field_reads_the_na_rule_off_the_field_type__():
    """A string field sorts na to the FRONT, the opposite end from a numeric one."""
    readings = [_Reading(0.0, 30, "c"), _Reading(0.0, 20, NA(str)), _Reading(0.0, 10, "a")]
    assert array.sort_indices(readings, order.ascending, sort_field="s") == [1, 2, 0]

    array.sort(readings, order.ascending, sort_field="s")
    assert [element.i for element in readings] == [20, 10, 30]


def __test_sort_without_a_field_is_unchanged__():
    """The element itself stays the key when no field is named."""
    values = [30.0, 20.0, 10.0, nan]
    array.sort(values)
    assert values[:3] == [10.0, 20.0, 30.0]
    assert values[3] != values[3]

    assert array.sort_indices([nan, nan, 5.0, 1.0]) == [3, 2, 0, 1]
