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
    if isinstance(a, NA) or a != a or isinstance(b, NA) or b != b:
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


def safe_int(value: PyneInt) -> int:
    """
    Safe int conversion that returns NA for NA inputs.
    Catches TypeError (thrown by NA values) but allows ValueError to propagate normally.

    @param value: The value to convert to int.
    @return: The int value, or NA(int) if TypeError occurs.
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        # NA objects throw TypeError; int(nan) throws ValueError; int(inf) OverflowError
        return NA(int)
