from math import inf

from ..types import NA, PyneFloat, PyneInt
from ..types.na import na_float, na_int
from ..types.pine_types import pine_int

_NEG_INF = -inf


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
        if a > 0:
            return inf
        if a < 0:
            return _NEG_INF
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
