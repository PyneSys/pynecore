from math import copysign as _copysign, fmod as _fmod, inf
from typing import Any

from ..types import NA, PyneFloat, PyneInt
from ..types.matrix import Matrix
from ..types.na import na_float, na_int
from ..types.pine_types import pine_int

_NEG_INF = -inf

# Pine's typeless na. ``NA(None)`` is interned, so an identity test finds every one
_NA_NONE = NA(None)


class _ZeroDivisor:
    """
    Stand-in for a zero divisor on the fast path of an inlined division.

    ``SafeDivisionTransformer`` divides by ``b or zero_divisor``: a falsy ``b``
    is replaced by this object before the division can raise. Every numeric
    type defers to the reflected method for an unknown operand, and the nan it
    answers fails the quotient's self-equality test, so the expression falls
    back to :func:`safe_div` with the original operands.
    """
    __slots__ = ()

    def __rtruediv__(self, _: object) -> float:
        return na_float


zero_divisor = _ZeroDivisor()


def safe_div(a: PyneFloat, b: PyneFloat):
    """
    Safe division mimicking Pine Script semantics.

    Pine's `na()` predicate reports inf/-inf/nan as NA, but arithmetic and
    comparisons on those values follow IEEE-754 (e.g. `inf > 40` is true).
    Native floats give exactly that: division by zero returns raw inf/-inf/nan,
    the `na()` predicate (`not isfinite`) reports them as na, and arithmetic
    and comparisons on them follow IEEE-754 natively.

    @param a: The numerator.
    @param b: The denominator.
    @return: a/b, raw inf/-inf/nan on zero denominator, or nan for na inputs.
    """
    try:
        result = a / b
    except ZeroDivisionError:
        if not (a == a):
            return na_float
        # MEASURED: ``1 / x`` of a runtime ``-0.0`` is -inf -- the zero's sign counts
        if a > 0:
            return inf if _copysign(1.0, b) > 0.0 else _NEG_INF
        if a < 0:
            return _NEG_INF if _copysign(1.0, b) > 0.0 else inf
        return na_float
    except TypeError:
        return na_float
    except Exception:
        # A na operand answers na before any division error surfaces
        if not (a == a) or not (b == b):
            return na_float
        raise
    # The quotient is tested instead of the operands: a nan or ``NA`` operand
    # always yields a quotient that fails ``==``, so one test clears the common
    # case. Only a failing quotient needs the operand tests, because a na
    # operand answers the interned ``na_float``, while ``inf / inf`` on two
    # finite-typed operands keeps its own computed nan.
    if result == result:
        return result
    if not (a == a) or not (b == b):  # is_na_arg
        return na_float
    return result


def safe_float(value: PyneFloat) -> float:
    """
    Safe float conversion that returns NA for NA inputs.
    Catches TypeError (thrown by NA values) but allows ValueError to propagate normally.

    @param value: The value to convert to float.
    @return: The float value, or na when the input is na.
    """
    try:
        return float(value)
    except TypeError:
        # NA values throw TypeError, convert these to NA
        return na_float


def safe_mod(a: PyneFloat | PyneInt, b: PyneFloat | PyneInt) -> PyneFloat | PyneInt:
    """
    Pine's ``%``: the truncated remainder, which takes the sign of the dividend.

    MEASURED on TradingView, const and series operands alike: ``-7 % 3`` is ``-1``,
    ``7 % -3`` is ``1`` and ``-7.5 % 2`` is ``-1.5`` -- C's ``fmod``, not Python's
    floored ``%`` (``2``, ``-2``, ``0.5``); a series ``-6.0 % 3`` is ``-0.0``. A zero
    divisor and a na operand give na instead of raising. Two Python ints keep an int
    result, so an int-typed remainder stays an int value.

    :param a: The dividend.
    :param b: The divisor.
    :return: The remainder, or na.
    """
    try:
        if a.__class__ is int and b.__class__ is int:
            r = abs(a) % abs(b)
            return -r if a < 0 else r
        r = _fmod(a, b)
    except (ZeroDivisionError, ValueError, TypeError):
        # A zero divisor, an infinite dividend or an ``NA`` operand
        return na_float
    return r if r == r else na_float


def safe_float_items(items: list[PyneInt] | NA) -> list[float] | NA:
    """
    The elements of an int array as floats, for copying into a float array.

    Pine converts the elements of an ``array<int>`` that ``array.concat`` or
    ``matrix.add_row`` copies into a float container; the transform wraps such
    an operand in this call. The source array is left as it is.

    :param items: The int array, or na
    :return: A new list of the elements as floats (na elements stay na), or the na
    """
    if isinstance(items, NA):
        return items
    return [safe_float(item) for item in items]


def safe_float_matrix(matrix: Matrix[PyneInt] | NA) -> Matrix[float] | NA:
    """
    The elements of an int matrix as floats, for copying into a float matrix.

    The ``matrix.concat`` counterpart of :func:`safe_float_items`.

    :param matrix: The int matrix, or na
    :return: A new matrix of the same shape holding the elements as floats, or the na
    """
    if isinstance(matrix, NA):
        return matrix
    result: Matrix[float] = Matrix(matrix.rows, matrix.cols, na_float)
    result.data = [[safe_float(item) for item in row] for row in matrix.data]
    return result


def implicit_convert(value: Any, kind: str | type | None) -> Any:
    """
    The implicit conversion one instance of a shared function body applies to a value.

    A generic body can put the same value into a float container in one call-site
    context and into an int container in another, so whether Pine converts it is a
    property of the instance. The transform wraps such a value in this call and
    hands it the instance's own vector entry, one of the type pass's conversion
    characters: ``'f'`` converts one int to float, ``'a'`` / ``'m'`` copy an int
    array / matrix as floats, ``'s'`` / ``'n'`` / ``'t'`` turn Pine's typeless na into
    the string na, the numeric na or the text ``na``, ``'A'`` / ``'M'`` / ``'X'`` into
    the na of an array, a map or a matrix, None converts nothing.

    A value that holds the typeless na without spelling it -- a defaulted parameter, a
    variable -- is wrapped in this call with its conversion character as a constant, or
    with the class whose na it becomes where it goes as an object of that class. The
    same body can be reached with a value of its own, so the na conversions leave every
    value but the typeless na as it is.

    :param value: The value on its way into the container
    :param kind: The conversion character, the class whose na the typeless na is, or None
    :return: The value as the container receives it
    """
    if kind is None:
        return value
    if kind == 'f':
        return safe_float(value)
    if kind == 'a':
        return safe_float_items(value)
    if kind == 'm':
        return safe_float_matrix(value)
    if value is not _NA_NONE:
        return value
    if kind == 'n':
        return na_float
    if kind == 't':
        return 'na'
    if kind == 'A':
        return NA(list)
    if kind == 'M':
        return NA(dict)
    if kind == 'X':
        return NA(Matrix)
    if isinstance(kind, type):
        return NA(kind)
    return ''


def native_int(value: PyneInt) -> int | NA:
    """
    Truncate a Pine number to a native Python int for internal consumption.

    This is what ``int()`` means inside a ``@pyne lib`` module: the lib computes
    its lengths, counts and ring indexes in native int and converts back to the
    Pine representation only at its boundary (see :func:`safe_int`). An na
    input stays an na object, so the value keeps propagating as na.

    @param value: The value to truncate.
    @return: The native int, or the typeless na when the input is na.
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        # NA objects throw TypeError; int(nan) throws ValueError; int(inf) OverflowError
        return NA(None)


def native_int_or(value: PyneInt, default: int) -> int:
    """
    Truncate a Pine number to a native Python int, with a fallback for na.

    Same truncation as :func:`native_int`, for the consuming slots that need a
    real integer even when the value is na: TradingView substitutes a concrete
    integer at those slots instead of propagating the na, and which integer it
    substitutes is a property of the slot.

    @param value: The value to truncate.
    @param default: The integer the slot uses when the value is na.
    @return: The native int, or the default when the input is na.
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        # NA objects throw TypeError; int(nan) throws ValueError; int(inf) OverflowError
        return default


def safe_int(value: PyneInt) -> PyneInt:
    """
    Safe int conversion that returns na for na inputs.

    @param value: The value to convert to int.
    @return: The truncated value, or na when the input is na.
    """
    try:
        return pine_int(int(value))
    except (TypeError, ValueError, OverflowError):
        # NA objects throw TypeError; int(nan) throws ValueError; int(inf) OverflowError
        return na_int
