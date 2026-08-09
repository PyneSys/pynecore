"""
@pyne

array.fill rejects an out-of-range bound instead of clamping it.

Measured on TradingView (FX:EURUSD 240, array [10, 20, 30, 40]): `index_from`
must address an existing element while `index_to` may reach the array size, so
fill(a, 5, 0, 99), fill(a, 5, 0, -1), fill(a, 5, -1, 2) and fill(a, 5, 4, 4) all
halt with RE10045, as does a fill on an empty array. Reversed bounds are a
silent no-op, and no fill ever changes the array size.

PyneCore used to clamp every out-of-range bound into the array, which filled
part of it and kept running where TradingView stops the script.
"""
from math import nan
from typing import Any

import pytest

from pynecore.lib import array
from pynecore.types.na import NA

# Typed as Any because Pine declares these arguments `int`; na is what the
# guards accept on top of that, not part of the declared parameter type.
NA_INDICES: tuple[Any, ...] = (NA(int), NA(None), nan)


def main():
    """ Dummy main to keep this a valid Pyne script """


def __test_fill_out_of_range_to_raises__():
    """ An index_to past the array size is an error, with an na index_from too """
    for na_index in NA_INDICES:
        a = [10, 20, 30, 40]
        with pytest.raises(IndexError):
            array.fill(a, 5, na_index, 99)
        assert a == [10, 20, 30, 40]
    b = [10, 20, 30, 40]
    with pytest.raises(IndexError):
        array.fill(b, 5, 0, 99)
    assert b == [10, 20, 30, 40]


def __test_fill_negative_to_raises__():
    """ A negative index_to is out of range, it does not count from the end """
    for na_index in NA_INDICES:
        a = [10, 20, 30, 40]
        with pytest.raises(IndexError):
            array.fill(a, 5, na_index, -1)
        assert a == [10, 20, 30, 40]


def __test_fill_negative_from_raises__():
    """ A negative index_from is out of range as well """
    a = [10, 20, 30, 40]
    with pytest.raises(IndexError):
        array.fill(a, 5, -2)
    assert a == [10, 20, 30, 40]
    b = [10, 20, 30, 40]
    with pytest.raises(IndexError):
        array.fill(b, 5, -2, 4)
    assert b == [10, 20, 30, 40]


def __test_fill_from_at_or_past_the_size_raises__():
    """ index_from must address an element, so the array size is already too far """
    a = [10, 20, 30, 40]
    with pytest.raises(IndexError):
        array.fill(a, 5, 4, 4)
    assert a == [10, 20, 30, 40]
    b = [10, 20, 30, 40]
    with pytest.raises(IndexError):
        array.fill(b, 5, 99)
    assert b == [10, 20, 30, 40]


def __test_fill_on_empty_array_raises__():
    """ An empty array has no index_from to address, so even the default fill halts """
    empty: list[int] = []
    with pytest.raises(IndexError):
        array.fill(empty, 5)
    assert empty == []


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
    """ The ordinary in-range cases keep their existing results and the array size """
    a = [10, 20, 30, 40]
    array.fill(a, 5)
    assert a == [5, 5, 5, 5]
    b = [10, 20, 30, 40]
    array.fill(b, 5, 1)
    assert b == [10, 5, 5, 5]
    c = [10, 20, 30, 40]
    array.fill(c, 5, 1, 3)
    assert c == [10, 5, 5, 40]
    d = [10, 20, 30, 40]
    array.fill(d, 5, 3, 4)
    assert d == [10, 20, 30, 5]
