from ..types import NA, PyneFloat, PyneInt
from ..types.na import na_inf, na_neg_inf, na_nan


def safe_div(a: PyneFloat, b: PyneFloat):
    """
    Safe division mimicking Pine Script semantics.

    Pine's `na()` predicate reports inf/-inf/nan as NA, but arithmetic and
    comparisons on those values follow IEEE-754 (e.g. `inf > 40` is true).
    To satisfy both behaviors we return valued-NA singletons on division by
    zero: `isinstance(x, NA)` stays true (lib-internal NA-skip keeps working),
    while comparisons and arithmetic dispatch to the backing float.

    @param a: The numerator.
    @param b: The denominator.
    @return: a/b, one of the valued-NA singletons on zero denominator, or
             plain NA(float) if inputs are NA.
    """
    if a is NA() or b is NA():
        return NA(float)
    try:
        return a / b
    except ZeroDivisionError:
        if a > 0:
            return na_inf
        if a < 0:
            return na_neg_inf
        return na_nan
    except TypeError:
        return NA(float)


def safe_float(value: PyneFloat) -> float | NA[float]:
    """
    Safe float conversion that returns NA for NA inputs.
    Catches TypeError (thrown by NA values) but allows ValueError to propagate normally.

    @param value: The value to convert to float.
    @return: The float value, or NA(float) if TypeError occurs.
    """
    try:
        return float(value)
    except TypeError:
        # NA values throw TypeError, convert these to NA
        return NA(float)


def safe_int(value: PyneInt) -> int | NA[int]:
    """
    Safe int conversion that returns NA for NA inputs.
    Catches TypeError (thrown by NA values) but allows ValueError to propagate normally.

    @param value: The value to convert to int.
    @return: The int value, or NA(int) if TypeError occurs.
    """
    try:
        return int(value)
    except TypeError:
        # NA values throw TypeError, convert these to NA
        return NA(int)
