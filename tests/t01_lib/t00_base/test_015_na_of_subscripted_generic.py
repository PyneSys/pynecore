"""
@pyne

A typed na declaration (`matrix<float> m = na`) reaches the runtime as
``na(Matrix[float])``. A subscripted user Generic is a typing._GenericAlias,
which the old types.GenericAlias membership test did not match, so na fell
through to its predicate face and answered False -- a bool. The declared
variable then held False instead of an na: `na(m)` claimed the matrix was
there, and the first matrix function on it halted the script with
`'bool' object has no attribute ...`.
"""
from math import nan, inf

from pynecore.lib import matrix, na
from pynecore.types import Matrix
from pynecore.types.na import NA


def main():
    """ Dummy main to keep this a valid Pyne script """


def __test_na_of_a_subscripted_user_generic__():
    """ na(Matrix[float]) builds an na, exactly like na(Matrix) does """
    value = na(Matrix[float])
    assert isinstance(value, NA)
    assert na(value) is True
    assert repr(value) == repr(na(Matrix))


def __test_na_of_a_subscripted_builtin_generic__():
    """ The builtin generics that already worked keep working """
    assert isinstance(na(list[float]), NA)
    assert isinstance(na(dict[str, int]), NA)


def __test_na_matrix_stays_usable__():
    """ The matrix functions see a declared na matrix as na, not as a bool """
    m = na(Matrix[float])
    assert na(matrix.copy(m)) is True
    assert na(matrix.get(m, 0, 0)) is True
    assert na(matrix.rows(m)) is True


def __test_the_predicate_face_is_unchanged__():
    """ Values still answer the predicate, not a constructed na """
    assert na(1.5) is False
    assert na(5) is False
    assert na('x') is False
    assert na(nan) is True
    assert na(inf) is True
    assert na(-inf) is True
    assert na(NA(int)) is True


def __test_the_constructor_face_is_unchanged__():
    """ Plain types keep building the na they used to """
    # `is`, not `==`: every comparison on an na is False, and the na of a type
    # is interned
    assert na(int) is NA(int)
    assert na(float) != na(float)  # a float na is the native nan
    assert isinstance(na(), NA) and na().type is None
    # `na` itself is a type too, but it is the sentinel class, not a type to
    # build an na of
    assert na(NA) is True
