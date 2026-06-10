"""
@pyne

Stateful implementations of ``lib.math.random`` and ``lib.math.sum``. They
live in their own small module because the ``@pyne`` marker is module-level
and the host module (``lib/math.py``) must stay untransformed; the host
re-exports the functions, and the layouts travel on the function objects.
"""
# Absolute imports on purpose: the call-site classifier resolves absolute
# imports at transform time, so NA() calls stay direct instead of anchored
from typing import TypeVar

from pynecore.types import NA, Persistent, PyneFloat, PyneInt, Series
from pynecore.core.random import PineRandom as _PineRandom

TFI = TypeVar('TFI', float, int)

__all__ = ['random', 'sum']


# noinspection PyShadowingBuiltins,PyShadowingNames
def random(min: TFI | NA[TFI] = 0, max: TFI | NA[TFI] = 1, seed: PyneInt = NA(int)) -> PyneFloat:
    """
    Returns a random number between two numbers.

    :param min: The minimum number.
    :param max: The maximum number.
    :param seed: The seed for the random number generator.
    :return: A random number between the minimum and maximum numbers.
    """
    prng: Persistent[_PineRandom | None] = None
    if prng is None:  # Lazy init: the PRNG must not be created before the seed is known
        prng = _PineRandom(seed)
    res = prng.random(min, max)
    return res


# noinspection PyShadowingBuiltins
def sum(source: TFI | NA[TFI], length: int) -> PyneFloat | TFI | NA[TFI]:
    """
    Returns the sum of a series over a specified length using Kahan summation.

    :param source: Source series
    :param length: Length of the sum
    :return: The sliding sum of the series
    """
    summ: Persistent[float] = 0.0
    count: Persistent[int] = 0
    compensation: Persistent[float] = 0.0

    if length == 1:  # Shortcut
        return source
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)

    isna = isinstance(source, NA)
    if not isna:
        # NA values are NOT stored in the buffer, only skipped (the read
        # below indexes past them) — keep this assignment conditional
        src: Series[float] = source

    if count < length - 1:
        if not isna:
            count += 1
            # Kahan summation for adding new value
            corrected_value = float(source) - compensation
            new_sum = summ + corrected_value
            compensation = (new_sum - summ) - corrected_value
            summ = new_sum
        return NA(float)
    elif count == length - 1:
        if isna:
            return NA(float)
        count += 1
    else:
        if isna:
            return summ
        # Kahan summation for removing old value (float() compiles to
        # safe_convert.safe_float, returning NA instead of raising on NA)
        old_value = float(src[length])
        corrected_old = -old_value - compensation
        new_sum = summ + corrected_old
        compensation = (new_sum - summ) - corrected_old
        summ = new_sum

    # Kahan summation for adding new value
    corrected_value = float(source) - compensation
    new_sum = summ + corrected_value
    compensation = (new_sum - summ) - corrected_value
    summ = new_sum

    return summ
