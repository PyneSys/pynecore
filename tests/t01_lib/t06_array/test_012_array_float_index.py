"""
@pyne

Direct unit tests for float-index tolerance in the array functions.

TradingView rejects a float index while compiling, so a compiled script can
never produce one. PyneCore does not always know the type, though, and an
integer carried as a float -- from a division, from math.round -- used to halt
the script with "list indices must be integers or slices, not float".
"""
from typing import Any

from pynecore.lib import array


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


def __test_get_set_float_index__():
    """ A float-carried integer addresses the element its value names """
    a = [10, 20, 30, 40]
    assert array.get(a, _idx(2.0)) == 30
    array.set(a, _idx(1.0), 99)
    assert a == [10, 99, 30, 40]


def __test_remove_insert_float_index__():
    """ remove and insert take a float-carried integer too """
    a = [10, 20, 30, 40]
    assert array.remove(a, _idx(1.0)) == 20
    assert a == [10, 30, 40]
    array.insert(a, _idx(1.0), 77)
    assert a == [10, 77, 30, 40]


def __test_fractional_index_truncates__():
    """ A genuinely fractional index truncates toward zero """
    # No valid program produces one -- int() is used for its exactness on the
    # float-carried integers, and this pins what the same coercion does here.
    a = [10, 20, 30, 40]
    assert array.get(a, _idx(2.7)) == 30
    assert array.get(a, _idx(-1.5)) == 40


def __test_percentrank_and_nth_float__():
    """ percentrank's index and max/min's nth accept a float-carried integer """
    a = [10, 20, 30, 40]
    assert array.percentrank(a, _idx(1.0)) == array.percentrank(a, 1)
    assert array.max(a, _idx(1.0)) == 30
    assert array.min(a, _idx(1.0)) == 20


def __test_fill_and_slice_float_bounds__():
    """ fill and slice take float-carried integer bounds """
    a = [10, 20, 30, 40]
    array.fill(a, 5, _idx(1.0), _idx(3.0))
    assert a == [10, 5, 5, 40]
    assert list(array.slice(a, _idx(1.0), _idx(3.0))) == [5, 5]


def __test_new_float_size__():
    """ An array constructor takes a float-carried integer size """
    assert array.new_int(_idx(3.0), 0) == [0, 0, 0]
