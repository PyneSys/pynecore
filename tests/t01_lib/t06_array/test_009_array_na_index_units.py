"""
@pyne

Direct unit tests for na-index tolerance in the array functions.

The runner test (test_008) proves the TradingView-measured behaviour end to end,
but it can only produce one na representation. These tests cover the rest of the
surface: the `NA` instance, the typeless `NA(None)` and the native `nan` must all
be recognised, slice views must be accepted, and a valid integer index -- negative
or out of range -- must behave exactly as before.
"""
from math import nan
from typing import Any

import pytest

from pynecore.lib import array
from pynecore.types.na import NA

# Every na representation that can reach an array function: an NA instance
# (bare `na`, or `int()` of an na float via safe_int), the typeless na, and the
# native nan a float-typed na really is. Typed as Any because Pine declares
# every one of these arguments `int`/`float`; na is what the guards accept on
# top of that, not part of the declared parameter type.
NA_INDICES: tuple[Any, ...] = (NA(int), NA(None), nan)


def main():
    """ Dummy main to keep this a valid Pyne script """


def _is_na(value) -> bool:
    return isinstance(value, NA) or value != value


def __test_get_na_index_returns_na__():
    """ array.get with an na index returns na and leaves the array alone """
    # TV probe: get([10, 20, 30, 40], na) -> NaN, arr: 10,20,30,40
    for na_index in NA_INDICES:
        a = [10, 20, 30, 40]
        assert _is_na(array.get(a, na_index))
        assert a == [10, 20, 30, 40]


def __test_get_na_index_on_empty_array__():
    """ array.get with an na index tolerates an empty array """
    # TV probe: get(array.new_int(), na) -> NaN, size: 0
    for na_index in NA_INDICES:
        assert _is_na(array.get([], na_index))


def __test_get_na_index_element_type__():
    """ The returned na carries the array's element type where it is knowable """
    # A float array yields the native nan (Pine's float na IS an IEEE-754 nan),
    # an int/str/bool array the matching NA instance, an empty array a typeless
    # na. TradingView cannot separate na-string from "" nor na-bool from false,
    # so only the float/int cases are TV-measured; the rest follow the in-file
    # NA(type(first)) idiom.
    float_na = array.get([1.5, 2.5], NA(int))
    assert isinstance(float_na, float) and float_na != float_na
    assert array.get([1, 2], NA(int)) is NA(int)
    assert array.get(["x", "y"], NA(int)) is NA(str)
    assert array.get([True, False], NA(int)) is NA(bool)
    assert array.get([], NA(int)) is NA(None)


def __test_set_na_index_is_noop__():
    """ array.set with an na index changes nothing """
    # TV probe: set([10, 20, 30, 40], na, 99) -> 10,20,30,40 size: 4
    for na_index in NA_INDICES:
        a = [10, 20, 30, 40]
        array.set(a, na_index, 99)
        assert a == [10, 20, 30, 40]
        empty: list[int] = []
        array.set(empty, na_index, 9)
        assert empty == []


def __test_remove_na_index_returns_na__():
    """ array.remove with an na index removes nothing and returns na """
    # TV probe: remove([10, 20, 30, 40], na) -> NaN, arr: 10,20,30,40
    # TV probe: remove(array.new_int(), na)  -> NaN, size: 0
    for na_index in NA_INDICES:
        a = [10, 20, 30, 40]
        assert _is_na(array.remove(a, na_index))
        assert a == [10, 20, 30, 40]
        assert _is_na(array.remove([], na_index))


def __test_insert_na_index_appends__():
    """ array.insert with an na index appends at the end """
    # TV probe: insert([10, 20, 30, 40], na, 77) -> 10,20,30,40,77
    # TV probe: insert(array.new_int(), na, 77)  -> 77, size: 1
    for na_index in NA_INDICES:
        a = [10, 20, 30, 40]
        array.insert(a, na_index, 77)
        assert a == [10, 20, 30, 40, 77]
        empty: list[int] = []
        array.insert(empty, na_index, 77)
        assert empty == [77]


def __test_slice_na_bounds__():
    """ na index_from is 0, na index_to is the array size """
    # TV probe: slice(a, na, 2) -> 10,20 | slice(a, 1, na) -> 20,30,40
    #           slice(a, na, na) -> 10,20,30,40
    for na_index in NA_INDICES:
        a = [10, 20, 30, 40]
        assert list(array.slice(a, na_index, 2)) == [10, 20]
        assert list(array.slice(a, 1, na_index)) == [20, 30, 40]
        assert list(array.slice(a, na_index, na_index)) == [10, 20, 30, 40]


def __test_slice_na_bounds_still_a_live_view__():
    """ An na-bounded slice is the usual live view over the parent array """
    a = [10, 20, 30, 40]
    view = array.slice(a, NA(int), 2)
    array.set(view, 0, -1)
    assert a == [-1, 20, 30, 40]


def __test_get_set_na_index_on_slice_view__():
    """ na index tolerance also holds when the target is a slice view """
    # TradingView's behaviour on a slice view is UNMEASURED; PyneCore routes
    # slice views through the same array.get/array.set, so they must at least
    # not raise.
    for na_index in NA_INDICES:
        a = [10, 20, 30, 40]
        view = array.slice(a, 1, 3)
        assert _is_na(array.get(view, na_index))
        array.set(view, na_index, 5)
        assert a == [10, 20, 30, 40]


def __test_fill_na_bounds__():
    """ na index_from fills from the start, na index_to fills to the end """
    # TV probe: fill(a, 5, na, 2) -> 5,5,30,40 | fill(a, 5, 1, na) -> 10,5,5,5
    for na_index in NA_INDICES:
        a = [10, 20, 30, 40]
        array.fill(a, 5, na_index, 2)
        assert a == [5, 5, 30, 40]
        b = [10, 20, 30, 40]
        array.fill(b, 5, 1, na_index)
        assert b == [10, 5, 5, 5]


def __test_percentrank_na_index_is_index_zero__():
    """ An na index behaves as index 0, not as na """
    # TV probe: percentrank([10, 20, 30, 40], na) -> 0
    # TV probe (reversed array, from the measurement session): percentrank
    # ([40, 30, 20, 10], na) -> 100, the same as index 0.
    for na_index in NA_INDICES:
        assert array.percentrank([10, 20, 30, 40], na_index) == 0.0
        assert array.percentrank([40, 30, 20, 10], na_index) == 100.0


def __test_percentrank_degenerate_array_returns_na__():
    """ An array with no rank denominator yields na instead of halting """
    # Measured on TradingView (FX:EURUSD 240, read at bar_index == 100):
    #   array.percentrank(array.new_int(0), na) -> NaN, script keeps running
    #   array.percentrank(array.new_int(0), 0)  -> NaN
    #   array.percentrank(array.from(5), 0)     -> NaN
    #   array.percentrank(array.from(5), na)    -> NaN
    # The rank formula divides by len - 1, so both cases used to raise
    # ZeroDivisionError / ValueError and halt the script.
    for na_index in NA_INDICES:
        assert _is_na(array.percentrank([], na_index))
        assert _is_na(array.percentrank([5], na_index))
    assert _is_na(array.percentrank([], 0))
    assert _is_na(array.percentrank([5], 0))
    # A two-element array still has a denominator and ranks normally
    # (TV probe: array.percentrank(array.from(10, 20), 1) -> 100)
    assert array.percentrank([10, 20], 1) == 100.0


def __test_percentrank_out_of_range_still_raises__():
    """ Out-of-range is an error for percentrank, negative index included """
    # Measured on TradingView: array.percentrank([10, 20, 30, 40], -1) halts
    # with RE10045, unlike get/set/remove/insert which accept -1.
    with pytest.raises(ValueError):
        array.percentrank([10, 20, 30, 40], -1)
    with pytest.raises(ValueError):
        array.percentrank([5], -1)


def __test_max_min_na_nth_is_zero__():
    """ An na nth behaves as nth = 0 """
    # TV probe: max([10, 20, 30, 40], na) -> 40, min -> 10
    for na_index in NA_INDICES:
        assert array.max([10, 20, 30, 40], na_index) == 40
        assert array.min([10, 20, 30, 40], na_index) == 10
    # nth = 1 still differs, so the na branch is not swallowing a real rank
    assert array.max([10, 20, 30, 40], 1) == 30
    assert array.min([10, 20, 30, 40], 1) == 20


def __test_percentile_na_percentage_returns_na__():
    """ An na percentage yields na instead of raising """
    # UNMEASURED on TradingView: na is returned because it cannot halt a
    # running script, which raising would.
    for na_index in NA_INDICES:
        assert _is_na(array.percentile_nearest_rank([1.0, 2.0, 3.0], na_index))
        assert _is_na(array.percentile_linear_interpolation([1.0, 2.0, 3.0], na_index))


def __test_valid_indices_unchanged__():
    """ Valid integer indices keep their existing behaviour """
    a = [10, 20, 30, 40]
    assert array.get(a, 0) == 10
    assert array.get(a, -1) == 40
    assert list(array.slice(a, 1, 3)) == [20, 30]
    assert array.percentrank(a, 3) == 100.0
    array.set(a, 1, 99)
    assert a == [10, 99, 30, 40]
    assert array.remove(a, 0) == 10
    assert a == [99, 30, 40]
    array.insert(a, 1, 77)
    assert a == [99, 77, 30, 40]


def __test_out_of_range_index_still_raises__():
    """ Out-of-range is NOT na: TradingView halts there too (RE10045) """
    with pytest.raises(IndexError):
        array.get([10, 20, 30, 40], 4)
    with pytest.raises(IndexError):
        array.set([10, 20, 30, 40], 4, 1)
    with pytest.raises(IndexError):
        array.remove([10, 20, 30, 40], 4)
    with pytest.raises(ValueError):
        array.percentrank([10, 20, 30, 40], 4)
