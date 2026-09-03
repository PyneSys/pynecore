from math import inf as _INF

from ..types import NA, PyneFloat, PyneInt
from ..types.na import na_float as _NAN

_NEG_INF = -_INF


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
    if not (a == a) or not (b == b):  # is_na_arg
        return _NAN
    try:
        return a / b
    except ZeroDivisionError:
        if a > 0:
            return _INF
        if a < 0:
            return _NEG_INF
        return _NAN
    except TypeError:
        return _NAN


def safe_float(value: PyneFloat) -> float:
    """
    Safe float conversion that returns NA for NA inputs.
    Catches TypeError (thrown by NA values) but allows ValueError to propagate normally.

    @param value: The value to convert to float.
    @return: The float value, or _NAN if TypeError occurs.
    """
    try:
        return float(value)
    except TypeError:
        # NA values throw TypeError, convert these to NA
        return _NAN


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


def safe_int(value: PyneInt) -> float:
    """
    Safe int conversion that returns na for na inputs.

    A Pine int is a double at runtime, so the truncated value travels as a float.

    @param value: The value to convert to int.
    @return: The truncated value, or na when the input is na.
    """
    try:
        return float(int(value))
    except (TypeError, ValueError, OverflowError):
        # NA objects throw TypeError; int(nan) throws ValueError; int(inf) OverflowError
        return _NAN
