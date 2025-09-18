from ..types import NA


def safe_div(a: float | NA[float], b: float | NA[float]):
    """
    Safe division that returns NA for division by zero.
    Mimics Pine Script behavior where division by zero returns NA.

    @param a: The numerator.
    @param b: The denominator.
    @return: The division result, or NA(float) if b is zero or NA.
    """
    if b == 0 or b == 0.0:
        return NA(float)
    if a is NA() or b is NA():
        return NA(float)
    try:
        return a / b
    except (ZeroDivisionError, TypeError):
        return NA(float)


def safe_float(value: float | NA[float]) -> float | NA[float]:
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


def safe_int(value: int | NA[int]) -> int | NA[int]:
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
