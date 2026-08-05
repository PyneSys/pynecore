"""
@pyne

Direct unit tests for na-index tolerance in the matrix functions.

An na never compares true against a bounds check, so it slipped past the guards
in Matrix and reached list indexing, where it halted the script with a TypeError.
The indices come from ordinary arithmetic on warmup bars, so this is reachable
from correct Pine code.
"""
from math import nan
from typing import Any

from pynecore.lib import matrix
from pynecore.types.na import NA

# Every na representation that can reach a matrix function: an NA instance (a
# bare `na`, or `int()` of an na float via safe_int), the typeless na, and the
# native nan a float-typed na really is.
NA_INDICES: tuple[Any, ...] = (NA(int), NA(None), nan)


def main():
    """ Dummy main to keep this a valid Pyne script """


def _is_na(value) -> bool:
    return isinstance(value, NA) or value != value


def __test_get_na_index_returns_na__():
    """ An na row or column index returns na and leaves the matrix alone """
    # Measured on TradingView (FX:EURUSD 240, bar 100, 2x2 float matrix):
    # get(m, na, 0), get(m, 0, na) and get(m, na, na) all return NaN.
    for na_index in NA_INDICES:
        m = matrix.new(2, 2, 7.0)
        assert _is_na(matrix.get(m, na_index, 0))
        assert _is_na(matrix.get(m, 0, na_index))
        assert _is_na(matrix.get(m, na_index, na_index))
        assert m.data == [[7.0, 7.0], [7.0, 7.0]]


def __test_get_na_index_matches_the_element_type__():
    """ The na handed back carries the matrix's element type """
    m = matrix.new(1, 1, "x")
    value = matrix.get(m, NA(int), 0)
    assert isinstance(value, NA) and value.type is str
    # An empty matrix has no knowable element type
    empty = matrix.new(0, 0)
    assert _is_na(matrix.get(empty, NA(int), 0))


def __test_set_na_index_is_a_noop__():
    """ An na row or column index changes neither contents nor size """
    # Measured on TradingView: set(m, na, 0, 5.0) leaves a 2x2 matrix at
    # [[7, 7], [7, 7]] with rows = 2 -- a no-op, not an error.
    for na_index in NA_INDICES:
        m = matrix.new(2, 2, 7.0)
        matrix.set(m, na_index, 0, 5.0)
        matrix.set(m, 0, na_index, 5.0)
        assert m.data == [[7.0, 7.0], [7.0, 7.0]]
        assert matrix.rows(m) == 2 and matrix.columns(m) == 2


def __test_row_and_col_na_index_return_na__():
    """ An na index yields an na array, exactly like an na matrix does """
    # Measured on TradingView: matrix.row(m, na) hands back an na array -- the
    # error only comes later, from the array function called on it (RE10052).
    for na_index in NA_INDICES:
        m = matrix.new(2, 2, 7.0)
        assert _is_na(matrix.row(m, na_index))
        assert _is_na(matrix.col(m, na_index))


def __test_valid_indices_unchanged__():
    """ Valid integer indices keep their existing behaviour """
    m = matrix.new(2, 2, 7.0)
    matrix.set(m, 0, 1, 3.0)
    assert matrix.get(m, 0, 1) == 3.0
    assert matrix.row(m, 0) == [7.0, 3.0]
    assert matrix.col(m, 1) == [3.0, 7.0]


def __test_out_of_range_index_still_raises__():
    """ Out-of-range is NOT na: it stays an error """
    m = matrix.new(2, 2, 7.0)
    for bad in (2, -1):
        try:
            matrix.get(m, bad, 0)
        except IndexError:
            pass
        else:
            raise AssertionError(f"matrix.get accepted an out-of-range row {bad}")
