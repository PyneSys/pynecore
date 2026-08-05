"""
@pyne

Direct unit tests for iterating a matrix.

Pine's ``for row in m`` is a plain Python ``for`` over the matrix, and its
indexed ``for [i, row] in m`` form an ``enumerate()`` over the same iterator, so
a Matrix that is not iterable halted every such script with a TypeError.
"""
from pynecore.lib import array, matrix


def main():
    """ Dummy main to keep this a valid Pyne script """


def __test_for_in_yields_the_rows_in_order__():
    """ Iterating a matrix walks its rows top to bottom """
    m = matrix.new(2, 3, 0.0)
    matrix.set(m, 0, 1, 5.0)
    matrix.set(m, 1, 2, 7.0)

    assert [row for row in m] == [[0.0, 5.0, 0.0], [0.0, 0.0, 7.0]]


def __test_for_in_rows_work_as_arrays__():
    """ A row is an ordinary array, so the array functions accept it """
    m = matrix.new(2, 3, 1.0)
    for row in m:
        assert array.size(row) == 3
        assert array.get(row, 0) == 1.0


def __test_indexed_for_in_enumerates_the_rows__():
    """ The indexed `for [i, row] in m` form walks the same rows """
    m = matrix.new(3, 1, 0.0)
    matrix.set(m, 2, 0, 9.0)

    seen = []
    for i, row in enumerate(m):
        seen.append((i, row[0]))
    assert seen == [(0, 0.0), (1, 0.0), (2, 9.0)]


def __test_for_in_rows_are_not_copies__():
    """ Writing through the loop variable changes the matrix itself """
    # TradingView documents this: for-in directly references the row arrays of
    # a matrix, so a write through the loop variable is visible in the matrix.
    # matrix.row() is the copying one.
    m = matrix.new(2, 2, 0.0)
    for row in m:
        array.set(row, 0, 4.0)
    assert matrix.get(m, 0, 0) == 4.0 and matrix.get(m, 1, 0) == 4.0

    copied = matrix.row(m, 0)
    array.set(copied, 1, 8.0)
    assert matrix.get(m, 0, 1) == 0.0


def __test_empty_matrix_iterates_zero_times__():
    """ A matrix with no rows is iterable, it just yields nothing """
    assert [row for row in matrix.new(0, 0)] == []
    assert [row for row in matrix.new(0, 3)] == []
