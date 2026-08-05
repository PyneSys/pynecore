"""
@pyne

array.fill must never resize the array it fills.

The na-index tolerance made out-of-range fill bounds reachable without raising:
an na `index_from` combined with a bound outside the array used to raise a
TypeError, so the destructive slice assignment underneath was never reached.
These tests pin the size invariant for both the na and the plain-integer bounds.
"""
from math import nan
from typing import Any

from pynecore.lib import array
from pynecore.types.na import NA

# Typed as Any because Pine declares these arguments `int`; na is what the
# guards accept on top of that, not part of the declared parameter type.
NA_INDICES: tuple[Any, ...] = (NA(int), NA(None), nan)


def main():
    """ Dummy main to keep this a valid Pyne script """


def __test_fill_na_from_with_out_of_range_to__():
    """ An na index_from plus a too-large index_to fills what exists, nothing more """
    for na_index in NA_INDICES:
        a = [10, 20, 30, 40]
        array.fill(a, 5, na_index, 99)
        assert a == [5, 5, 5, 5]


def __test_fill_na_from_with_negative_to__():
    """ An na index_from plus a negative index_to must not delete elements """
    for na_index in NA_INDICES:
        a = [10, 20, 30, 40]
        array.fill(a, 5, na_index, -1)
        assert a == [5, 5, 5, 40]


def __test_fill_negative_from__():
    """ A negative index_from counts from the end instead of growing the array """
    a = [10, 20, 30, 40]
    array.fill(a, 5, -2)
    assert a == [10, 20, 5, 5]
    b = [10, 20, 30, 40]
    array.fill(b, 5, -2, 4)
    assert b == [10, 20, 5, 5]


def __test_fill_out_of_range_from__():
    """ An index_from past the end fills nothing """
    a = [10, 20, 30, 40]
    array.fill(a, 5, 99)
    assert a == [10, 20, 30, 40]


def __test_fill_reversed_bounds_is_a_noop__():
    """ index_from greater than index_to fills nothing """
    a = [10, 20, 30, 40]
    array.fill(a, 5, 3, 1)
    assert a == [10, 20, 30, 40]


def __test_fill_through_a_slice_writes_plain_values_into_the_parent__():
    """ Filling a slice view fills the addressed part of the parent, nothing else """
    # Measured on TradingView: fill(slice(a, 0, 2), 5) on [10, 20, 30, 40] leaves
    # the parent [5, 5, 30, 40] and the view [5, 5]. The view's slice assignment
    # used to store the replacement LIST in every addressed slot, so the parent
    # ended up holding nested lists -- [[5, 5], [5, 5], 30, 40].
    a = [10, 20, 30, 40]
    view = array.slice(a, 0, 2)
    array.fill(view, 5)
    assert a == [5, 5, 30, 40]
    assert list(view) == [5, 5]

    # The same through an na bound, which is what makes the whole view reachable
    b = [10, 20, 30, 40]
    tail = array.slice(b, 2, 4)
    array.fill(tail, 7, NA(int), NA(int))
    assert b == [10, 20, 7, 7]


def __test_fill_in_range_bounds_unchanged__():
    """ The ordinary in-range cases keep their existing results """
    a = [10, 20, 30, 40]
    array.fill(a, 5)
    assert a == [5, 5, 5, 5]
    b = [10, 20, 30, 40]
    array.fill(b, 5, 1)
    assert b == [10, 5, 5, 5]
    c = [10, 20, 30, 40]
    array.fill(c, 5, 1, 3)
    assert c == [10, 5, 5, 40]
    d: list[int] = []
    array.fill(d, 5)
    assert d == []
