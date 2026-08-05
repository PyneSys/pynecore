"""
@pyne

Direct unit tests for float-index tolerance in the matrix functions.

TradingView rejects a float index while compiling, so a compiled script can
never produce one. PyneCore does not always know the type, though, and an
integer carried as a float -- from a division, from math.round -- used to halt
the script with "list indices must be integers or slices, not float".
"""
from typing import Any

from pynecore.lib import matrix


def main():
    """ Dummy main to keep this a valid Pyne script """


def _idx(value: float) -> Any:
    """
    Hand a float over as an index argument.

    Returns ``Any`` because Pine declares these arguments ``int``; the float is
    what the coercion accepts on top of that, not part of the declared type.

    :param value: The float index
    :return: The same value, with its type erased
    """
    return value


def __test_matrix_float_index__():
    """ The element and vector accessors take float-carried integer indices """
    m = matrix.new(_idx(2.0), _idx(2.0), 7.0)
    assert m.data == [[7.0, 7.0], [7.0, 7.0]]
    matrix.set(m, _idx(0.0), _idx(1.0), 3.0)
    assert matrix.get(m, _idx(0.0), _idx(1.0)) == 3.0
    assert matrix.row(m, _idx(0.0)) == [7.0, 3.0]
    assert matrix.col(m, _idx(1.0)) == [3.0, 7.0]


def __test_matrix_float_bounds__():
    """ The area and structure functions take them too """
    m = matrix.new(3, 3, 0.0)
    matrix.fill(m, 1.0, _idx(0.0), _idx(2.0), _idx(0.0), _idx(2.0))
    assert m.data == [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    assert matrix.submatrix(m, _idx(0.0), _idx(2.0),
                            _idx(0.0), _idx(2.0)).data == [[1.0, 1.0], [1.0, 1.0]]
    matrix.swap_rows(m, _idx(0.0), _idx(2.0))
    assert m.data == [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    matrix.swap_columns(m, _idx(0.0), _idx(2.0))
    assert m.data == [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    assert matrix.remove_row(m, _idx(0.0)) == [0.0, 0.0, 0.0]
    assert matrix.remove_col(m, _idx(0.0)) == [0.0, 0.0]


def __test_matrix_float_power__():
    """ A float-carried integer power raises the matrix and leaves it alone """
    m = matrix.new(2, 2, 0.0)
    matrix.set(m, 0, 0, 1.0)
    matrix.set(m, 0, 1, 2.0)
    matrix.set(m, 1, 0, 3.0)
    matrix.set(m, 1, 1, 4.0)
    assert matrix.pow(m, _idx(2.0)).data == [[7.0, 10.0], [15.0, 22.0]]
    assert m.data == [[1.0, 2.0], [3.0, 4.0]]
