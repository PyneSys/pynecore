"""
@pyne

Overload selection on the exact argument type.

Selection took the first registered implementation that matched at all, so an
int argument went to a float parameter whenever that one was declared first,
and an na argument carrying a container type (`matrix<float> m = na` gives
NA(Matrix[float])) matched nothing and died with `No matching implementation
found`. A real matrix was never sampled for its element type the way a list is
either, so overloads differing only there always went to the first one.
"""
from pynecore.lib import matrix, na
from pynecore.types import Matrix

# noinspection PyProtectedMember
from pynecore.core.overload import overload


def main():
    """ Dummy main to keep this a valid Pyne script """


# noinspection PyUnusedLocal
@overload
def _scalar(value: float) -> int:  # type: ignore[no-redef]
    return 1


# noinspection PyUnusedLocal,PyRedeclaration
@overload
def _scalar(value: int) -> int:  # type: ignore[no-redef]
    return 2


# The same pair the other way round: selection must not depend on this order
# noinspection PyUnusedLocal
@overload
def _scalar_reversed(value: int) -> int:  # type: ignore[no-redef]
    return 2


# noinspection PyUnusedLocal,PyRedeclaration
@overload
def _scalar_reversed(value: float) -> int:  # type: ignore[no-redef]
    return 1


# noinspection PyUnusedLocal
@overload
def _only_float(value: float) -> int:  # type: ignore[no-redef]
    return 9


# noinspection PyUnusedLocal
@overload
def _kind(value: Matrix[float]) -> int:  # type: ignore[no-redef]
    return 1


# noinspection PyUnusedLocal,PyRedeclaration
@overload
def _kind(value: Matrix[int]) -> int:  # type: ignore[no-redef]
    return 2


# noinspection PyUnusedLocal,PyRedeclaration
@overload
def _kind(value: list[float]) -> int:  # type: ignore[no-redef]
    return 3


# noinspection PyUnusedLocal,PyRedeclaration
@overload
def _kind(value: list[int]) -> int:  # type: ignore[no-redef]
    return 4


# noinspection PyUnusedLocal,PyRedeclaration
@overload
def _kind(value: list[str]) -> int:  # type: ignore[no-redef]
    return 5


# noinspection PyUnusedLocal
@overload
def _only_float_container(value: list[float]) -> int:  # type: ignore[no-redef]
    return 9


# noinspection PyUnusedLocal
@overload
def _only_float_matrix(value: Matrix[float]) -> int:  # type: ignore[no-redef]
    return 9


# noinspection PyUnusedLocal
@overload
def _nested(value: list[list[int]]) -> int:  # type: ignore[no-redef]
    return 1


# noinspection PyUnusedLocal,PyRedeclaration
@overload
def _nested(value: list[list[str]]) -> int:  # type: ignore[no-redef]
    return 2


# noinspection PyUnusedLocal
@overload
def _only_float_nested(value: list[list[float]]) -> int:  # type: ignore[no-redef]
    return 9


# noinspection PyUnusedLocal
@overload
def _arity(value: tuple[int, str]) -> int:  # type: ignore[no-redef]
    return 1


# noinspection PyUnusedLocal,PyRedeclaration
@overload
def _arity(value: tuple[int]) -> int:  # type: ignore[no-redef]
    return 2


# noinspection PyUnusedLocal
@overload
def _nested_arity(value: list[tuple[int, str]]) -> int:  # type: ignore[no-redef]
    return 1


# noinspection PyUnusedLocal,PyRedeclaration
@overload
def _nested_arity(value: list[tuple[int]]) -> int:  # type: ignore[no-redef]
    return 2


def __test_scalar_takes_its_exact_type__():
    """ An int argument takes the int parameter, not the float one """
    # Measured on TradingView (FX:EURUSD 240): with f(float) and f(int)
    # declared in EITHER order, f(5), f(int var) and f(bar_index) all answer
    # from the int implementation
    assert _scalar(5) == 2
    assert _scalar(5.0) == 1
    assert _scalar_reversed(5) == 2
    assert _scalar_reversed(5.0) == 1


def __test_scalar_widens_when_there_is_no_exact_match__():
    """ Without an int parameter the int argument still fits the float one """
    # Measured on TradingView: a script declaring only f(float) accepts f(int)
    assert _only_float(5) == 9
    assert _only_float(5.0) == 9


def __test_na_scalar_takes_the_type_it_was_declared_with__():
    """ A typed na scalar dispatches on its declared type """
    # Measured on TradingView: `int naI = na` answers from the int overload,
    # `float naF = na` from the float one, and a float na against an int-only
    # parameter is a compile error (CE10123) -- float never narrows to int
    assert _scalar(na(int)) == 2
    assert _only_float(na(int)) == 9


def __test_container_takes_its_exact_element_type__():
    """ A container dispatches on its element type, sampled or declared """
    # Measured on TradingView: array<int>/array<float>/matrix<int> arguments
    # each answer from the overload of their own element type
    assert _kind(matrix.new(2, 2, 1.0)) == 1
    assert _kind(matrix.new(2, 2, 1)) == 2
    assert _kind([1.0, 2.0]) == 3
    assert _kind([1, 2]) == 4
    assert _kind(['a', 'b']) == 5


def __test_na_container_takes_its_declared_element_type__():
    """ An na container dispatches on the subscript it was declared with """
    # Measured on TradingView: `array<int> a = na` answers from the array<int>
    # overload, `matrix<int> m = na` from the matrix<int> one
    assert _kind(na(Matrix[float])) == 1
    assert _kind(na(Matrix[int])) == 2
    assert _kind(na(list[float])) == 3
    assert _kind(na(list[int])) == 4
    assert _kind(na(list[str])) == 5


def __test_na_nested_container_takes_its_declared_element_type__():
    """ An na of a nested container matches on the whole declared subscript """
    # Two independently built generic aliases are equal but not the same
    # object, so the subscripts have to be compared structurally
    assert _nested(na(list[list[int]])) == 1
    assert _nested(na(list[list[str]])) == 2
    assert _only_float_nested(na(list[list[int]])) == 9


def __test_declared_subscripts_of_different_arity_do_not_match__():
    """ A declared subscript matches only one of the same shape """
    # Two subscripts of different length are different types, so the shorter one
    # must not fall into the longer one's overload just because the arguments
    # cannot be zipped
    assert _arity(na(tuple[int, str])) == 1
    assert _arity(na(tuple[int])) == 2
    assert _nested_arity(na(list[tuple[int, str]])) == 1
    assert _nested_arity(na(list[tuple[int]])) == 2


def __test_container_widens_when_there_is_no_exact_match__():
    """ An int container fits a float container parameter, real or na """
    # Measured on TradingView: a script declaring only f(array<float>) accepts
    # an array<int> argument, a matrix<int> one, and their na declarations
    assert _only_float_container([1, 2]) == 9
    assert _only_float_container(na(list[int])) == 9
    assert _only_float_matrix(matrix.new(2, 2, 1)) == 9


def __test_typeless_na_takes_the_first_declared_overload__():
    """ A bare na fits everything, so declaration order decides """
    # Measured on TradingView: f(na) answers from whichever of f(float)/f(int)
    # comes first in the script, for scalars and containers alike
    assert _scalar(na()) == 1
    assert _scalar_reversed(na()) == 2
    assert _kind(na()) == 1


def __test_unrelated_element_type_matches_nothing__():
    """ The sample really discriminates -- a string matrix is not numeric """
    try:
        _kind(matrix.new(2, 2, 'x'))
    except TypeError as e:
        assert 'No matching implementation' in str(e)
    else:
        raise AssertionError('a string matrix matched a numeric overload')


def __test_empty_container_matches_any_element_type__():
    """ With no element to sample, the first matching container wins """
    assert _kind(matrix.new(0, 0)) == 1
    assert _kind([]) == 3
