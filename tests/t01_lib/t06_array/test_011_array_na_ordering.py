"""
@pyne

Ordering and standardization with na elements.

Python's own comparisons are all False against na, so `list.sort()` leaves an
array holding na in an arbitrary order and `statistics.mean()` raises on one.
Both surfaces are reachable from ordinary code -- `array.new_int(n)` fills with
na -- so these pin the TradingView-measured answers.
"""
from math import nan

from pynecore.lib import array, order
from pynecore.types.na import NA


def main():
    """ Dummy main to keep this a valid Pyne script """


def _is_na(value) -> bool:
    return isinstance(value, NA) or value != value


def __test_sort_puts_numeric_na_last__():
    """ na sorts as the largest value of a numeric array """
    # Measured on TradingView (FX:EURUSD 240, bar 100):
    #   sort([30, 20, 10, na])             -> 10, 20, 30, na
    #   sort([30, 20, 10, na], descending) -> na, 30, 20, 10
    a = [30.0, 20.0, 10.0, nan]
    array.sort(a)
    assert a[:3] == [10.0, 20.0, 30.0]
    assert _is_na(a[3])

    b = [30.0, 20.0, 10.0, nan]
    array.sort(b, order.descending)
    assert _is_na(b[0])
    assert b[1:] == [30.0, 20.0, 10.0]

    # An NA instance is the same na: array.new_int() fills with exactly this
    c = [5, NA(int), 1]
    array.sort(c)
    assert c[:2] == [1, 5]
    assert _is_na(c[2])


def __test_sort_puts_string_na_first__():
    """ na sorts as the smallest value of a string array """
    # Measured on TradingView: sort(["b", "a", na]) -> na, "a", "b" and the
    # descending form is that reversed -- the opposite end from a numeric array.
    a = ["b", "a", NA(str)]
    array.sort(a)
    assert _is_na(a[0])
    assert a[1:] == ["a", "b"]

    b = ["b", "a", NA(str)]
    array.sort(b, order.descending)
    assert b[:2] == ["b", "a"]
    assert _is_na(b[2])


def __test_sort_indices_follows_the_same_order__():
    """ The indices of na elements go where sort would put the elements """
    # Measured on TradingView:
    #   sort_indices([30, 20, 10, na])              -> 2, 1, 0, 3
    #   sort_indices([na, na, 5, 1])                -> 3, 2, 0, 1
    #   sort_indices([30, 20, 10, na], descending)  -> 3, 0, 1, 2
    #   sort_indices(["b", "a", na])                -> 2, 1, 0
    assert array.sort_indices([30.0, 20.0, 10.0, nan]) == [2, 1, 0, 3]
    assert array.sort_indices([NA(int), NA(int), 5, 1]) == [3, 2, 0, 1]
    assert array.sort_indices([30.0, 20.0, 10.0, nan], order.descending) == [3, 0, 1, 2]
    assert array.sort_indices(["b", "a", NA(str)]) == [2, 1, 0]
    # An all-na array keeps its original order
    assert array.sort_indices([nan, nan, nan]) == [0, 1, 2]


def __test_sort_without_na_is_unchanged__():
    """ An array with no na sorts exactly as before """
    a = [30, 10, 20]
    array.sort(a)
    assert a == [10, 20, 30]
    array.sort(a, order.descending)
    assert a == [30, 20, 10]
    assert array.sort_indices([30, 10, 20]) == [1, 2, 0]


def __test_standardize_skips_na_elements__():
    """ na is left out of the statistics and stays na in the result """
    # Measured on TradingView: standardize([1, 2, 3, na]) gives the z-scores of
    # [1, 2, 3] with NaN in the fourth slot, so the population divisor is 3 and
    # not 4. Feeding the raw array to statistics.mean() used to raise instead.
    result = array.standardize([1.0, 2.0, 3.0, nan])
    assert result[:3] == [-1.224744871391589, 0.0, 1.224744871391589]
    assert _is_na(result[3])


def __test_standardize_does_not_threshold_int_arrays__():
    """ An int array standardizes exactly like the same values typed as float """
    # Measured on TradingView: standardize([1, 2, 6, 7, 8, 9, 10, 11]) yields the
    # continuous z-scores, identical to the float-typed array -- there is no
    # -1/0/1 thresholding anywhere in the function.
    ints = array.standardize([1, 2, 6, 7, 8, 9, 10, 11])
    floats = array.standardize([1.0, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    assert ints == floats
    assert ints[0] == -1.7002089231955175
    assert ints[-1] == 1.2566761606227739


def __test_standardize_degenerate_arrays__():
    """ All-equal, all-na and empty arrays """
    # Measured on TradingView: an all-equal array standardizes to 1.0 for every
    # element (one element included), an all-na array to na, an empty one to [].
    assert array.standardize([5.0, 5.0]) == [1.0, 1.0]
    assert array.standardize([5.0]) == [1.0]
    assert all(_is_na(v) for v in array.standardize([nan, nan, nan]))
    assert array.standardize([]) == []
