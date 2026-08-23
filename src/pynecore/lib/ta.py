"""
@pyne lib
"""
from typing import TypeVar, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from pynecore.types.type_checker import *

import builtins
import math
import heapq

from collections import deque

from ..types import Series, Persistent, NA, PyneFloat, PyneInt, PyneBool, na_float
from ..core.module_property import module_property, module_function_property
from pynecore.core.overload import overload

from ..core import safe_convert
from ..core.series import SeriesImpl as _SeriesImpl
# Pine's absolute comparison tolerance. Only the builtins MEASURED to compare
# tolerantly use it (see core/pine_compare.py); the bit-exact ones must not.
from ..core.pine_compare import (EPSILON as _EPSILON, lower_bound as _tol_lower_bound,
                                 upper_bound as _tol_upper_bound)

# We need to use this kind of import to make transformer work. ``_last_close`` is
# deliberately outside ``lib.__all__``: that list is the public Pine surface, and this
# is a runner internal, like ``_time`` or ``_script``.
# noinspection PyProtectedMember
from pynecore.lib import (open, high, low, close, volume, hl2, hlc3, bar_index, array, session,
                          max_bars_back, math as lib_math, _last_close, _stale_on_gap)

TFIB = TypeVar('TFIB', float, int, bool)
TFI = TypeVar('TFI', float, int)

__all__ = [
    "accdist",
    "alma",
    "atr",
    "barssince",
    "bb",
    "bbw",
    "cci",
    "change",
    "cmo",
    "cog",
    "correlation",
    "cross",
    "crossover",
    "crossunder",
    "cum",
    "dev",
    "dmi",
    "ema",
    "falling",
    "highest",
    "highestbars",
    "hma",
    "iii",
    "kc",
    "kcw",
    "linreg",
    "lowest",
    "lowestbars",
    "macd",
    "max",
    "median",
    "mfi",
    "min",
    "mode",
    "mom",
    "nvi",
    "obv",
    "percentile_linear_interpolation",
    "percentile_nearest_rank",
    "percentrank",
    "pivot_point_levels",
    "pivothigh",
    "pivotlow",
    "pvi",
    "pvt",
    "range",
    "rci",
    "rising",
    "rma",
    "roc",
    "rsi",
    "sar",
    "sma",
    "stdev",
    "stoch",
    "supertrend",
    "swma",
    "tr",
    "tsi",
    "valuewhen",
    "variance",
    "vwap",
    "vwma",
    "wad",
    "wma",
    "wpr",
    "wvad"
]

#
# Helper functions
#

#
# Indicators
#

@module_property
def accdist() -> PyneFloat:
    """
    Accumulation/Distribution index
    A/D = ((Close - Low) - (High - Close)) / (High - Low) * Volume + Previous A/D

    :return: Accumulation/Distribution index
    """
    ad: Persistent[float] = 0.0

    mfm = ((close - low) - (high - close)) / (high - low)
    mfv = mfm * volume
    if mfv == mfv:
        ad += mfv

    return ad


def alma(series: Series[float], length: int, offset: float = 0.85, sigma: float = 6.0, floor=False) \
        -> PyneFloat:
    """
    Calculate the Arnaud Legoux Moving Average (ALMA) of the source series with the given length.

    Fun fact: ALMA means "soul" in latin and Spanish, and Portugese.
              It means "apple" in Hungarian, Finnish, and Estonian.
              It means "take it" in Turkish.
              It means "water" in Arabic.
              It means "apple tree" in Georgian.
              ...

    :param series: The source series
    :param length: The length of the ALMA
    :param offset: The offset of the ALMA
    :param sigma: The sigma value of the ALMA
    :param floor:  Specifies whether the offset calculation is floored before ALMA is calculated. Default value is false
    :return: The ALMA of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if not (series == series):  # is_na_arg
        return na_float
    length = int(length)

    # Use persistent weights to avoid recalculation
    weights: Persistent[list[float]] = []
    norm: Persistent[float] = 0.0

    # Calculate weights only once
    if not weights:
        m = offset * (length - 1) if not floor else math.floor(offset * (length - 1))
        s = length / sigma
        weights = [math.exp(-1 * ((i - m) * (i - m)) / (2 * s * s)) for i in builtins.range(length)]
        weights.reverse()  # This is faster then using backward range or index subtraction
        norm = sum(weights)

    # Vectorized calculation using dot product
    summ = 0.0
    for i, w in enumerate(weights):
        summ += w * series[i]
    return summ / norm


def atr(length: int) -> PyneFloat:
    """
    Calculate Average True Range (ATR) of the source series with the given length.

    :param length: The length of the ATR
    :return: The ATR of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    return rma(tr(True), length)


def barssince(condition: bool) -> PyneInt:
    """
    Calculate the number of bars since the condition was true.

    :param condition: The condition to check
    :return: The number of bars since the condition was true
    """
    counter: Persistent[int] = -1
    if condition:
        counter = 0
    elif counter == -1:
        return NA(int)
    else:
        counter += 1
    return counter


def bb(series: float, length: int, mult: float | int) -> tuple[PyneFloat, PyneFloat, PyneFloat]:
    """
    Calculate the Bollinger Bands (BB) of the source series with the given length and multiplier.

    :param series: The source series
    :param length: The length of the BB
    :param mult: The multiplier of the BB
    :return: The Bollinger Bands (BB) of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    assert mult > 0, "Invalid multiplier, multiplier must be greater than 0!"

    std_dev = stdev(series, length)

    middle = sma(series, length)

    if not (middle == middle):
        return na_float, na_float, na_float
    std_dev *= mult
    return middle, middle + std_dev, middle - std_dev


def bbw(series: float, length: int, mult: float | int) -> PyneFloat:
    """
    Calculate the Bollinger Bands Width (BBW) of the source series with the given length and multiplier.

    :param series: The source series
    :param length: The length of the BBW
    :param mult: The multiplier of the BBW
    :return: The Bollinger Bands Width (BBW) of the source series
    """
    b, h, l = bb(series, length, mult)
    if not (b == b) or b == 0.0:
        return na_float
    return ((h - l) / b) * 100


def cci(source: float, length: int) -> PyneFloat:
    """
    Calculate the Commodity Channel Index (CCI) of the source series with the given length.

    :param source: The source series
    :param length: The length of the CCI
    :return: The Commodity Channel Index (CCI) of the source series
    """
    mean = sma(source, length)
    mdev = dev(source, length, _mean=mean)
    if not (mdev == mdev):
        return na_float
    return (source - mean) / (0.015 * mdev)


def change(source: Series[TFIB], length: int = 1) -> TFIB:
    """
    Calculate a simple change with respect to the given bar offset.

    :param source: The source series
    :param length: The offset in bars
    :return: The change from source to source[length]
    """
    # The difference is exact: TradingView subtracts the raw doubles and does not
    # quantize the result (probe m554, measured at a base small enough for a 1e-15
    # step to be representable -- ``ta.change`` reproduced it bit for bit). Dust
    # below Pine's 1e-10 comparison tolerance is absorbed by the comparison
    # operators, not by this function.
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK
    # Grow the buffer so ``source[length]`` stays addressable for lengths beyond the
    # per-series default max_bars_back (500); otherwise it reads na and the change is na.
    # The resize is monotonic: a series ``length`` that dips low must not shrink the
    # buffer, or the history a later increase needs would already be gone.
    if length > capacity:
        capacity = length
        max_bars_back(source, capacity)

    prev_val = source[length]  # noqa

    if not (source == source):  # is_na_arg
        # type(source) would be the NA class itself — keep the source sentinel's type
        return source
    if not (prev_val == prev_val):
        return NA(type(source))
    if isinstance(source, float):
        return cast(TFIB, source - prev_val)  # noqa
    if isinstance(source, int):
        return cast(TFIB, source - prev_val)  # noqa
    return source != prev_val


def cmo(series: float, length: int) -> PyneFloat:
    """
    Calculate the Chande Momentum Oscillator (CMO) of the source series with the given length.

    The momentum sign test is tolerant: a momentum below the float comparison
    tolerance in magnitude lands in the up bucket, so a sub-tolerance zig-zag
    collapses both sums to zero and the result is na.

    :param series: The source series
    :param length: The length of the CMO
    :return: The Chande Momentum Oscillator (CMO) of the source series
    """
    momentum = change(series)
    if not (momentum == momentum):
        return na_float
    # Tolerant sign test, measured on TradingView (probe m548)
    rising_momentum = momentum >= -_EPSILON
    sum1 = lib_math.sum(momentum if rising_momentum else 0.0, length)
    sum2 = lib_math.sum(0.0 if rising_momentum else -momentum, length)
    total = sum1 + sum2
    if total == 0.0:
        # Both buckets empty (a flat window, or a sub-tolerance zig-zag that
        # cancels): TradingView's 0/0 is nan, so the oscillator is na
        return na_float
    return 100 * (sum1 - sum2) / total


# noinspection PyUnusedLocal,PyShadowingBuiltins
def cog(source: Series[float], length: int) -> PyneFloat:
    """
    Calculate the Center of Gravity (COG) of the source series with the given length.

    :param source: The source series
    :param length: The length of the COG
    :return: The Center of Gravity (COG) of the source series
    """
    # An int-typed Pine value can still carry a fraction (``int / int``); the
    # truncation happens where an integer is required — see ``_check_type``.
    length = int(length)
    count: Persistent[int] = 0
    summ: Persistent[float] = 0.0
    weighted_summ: Persistent[float] = 0.0
    val: Persistent[float] = na_float
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK

    if not (source == source):  # is_na_arg
        # An NA bar leaves the window unchanged; hold the last full value
        # (still NA while warming up)
        return na_float if count < length else val

    # NA values are NOT stored in the buffer, only skipped, so ``src[length]``
    # indexes past NA gaps to the true oldest value still inside the window.
    # Reading the parameter directly would step back ``length`` *bars* and land
    # inside an NA gap, subtracting an NA that poisons ``summ`` forever.
    src: Series[float] = source
    # Grow the na-compacted buffer so ``src[length]`` stays addressable for lengths
    # beyond the per-series default max_bars_back (500); otherwise the window-drop
    # read returns na and poisons ``summ`` permanently. The resize is monotonic: a
    # series ``length`` that dips low must not shrink the buffer, or the history a
    # later increase needs would already be gone.
    if length > capacity:
        capacity = length
        max_bars_back(src, capacity)

    # Warming up phase — only non-NA samples advance the window
    if count < length:
        count += 1
        summ += source
        weighted_summ += source * (length - count)
        if count < length:
            return na_float

    # Normal calculation phase
    else:
        new_summ = summ + source - src[length]
        weighted_summ = weighted_summ + summ - length * src[length]
        summ = new_summ
    val = -weighted_summ / summ - 1.0
    return val


def correlation(source1: Series[float], source2: Series[float], length: int) -> PyneFloat:
    """
    Calculate the correlation of the source series with the given length.

    A covariance within the float comparison tolerance of zero makes the result 0.0, even
    for otherwise strongly correlated sources. The same tolerance decides the degenerate
    cases: a constant source stays na, while ``length == 1`` is 0.0 from the first bar.

    :param source1: The first source series
    :param source2: The second source series
    :param length: The length of the correlation
    :return: The correlation of the source series
    """
    # Measured law (probes m557, 27.8k bars each, real, synthetic and
    # catastrophic-cancellation sources, na gaps, lengths 1/2/3/5/14/20, every
    # displayed bar bit-identical): with the exact rolling-sum machine over both
    # sources, their product and their squares, TradingView computes
    #   cov = sxy / length - mx * my
    #   vx  = max(0, sx2 / length - mx * mx), likewise vy -- the clamp ``variance`` uses
    #   cov / sqrt(vx * vy), ONE square root over the product; two separate roots
    #   differ by one ulp on roughly 40% of the bars.
    assert length > 0, "Length must be greater than 0"
    length = int(length)

    # All five rolling machines must advance on every bar (na bars included), so
    # they run before any early return.
    sx = lib_math.sum(source1, length)
    sy = lib_math.sum(source2, length)
    sxy = lib_math.sum(source1 * source2, length)
    sx2 = lib_math.sum(source1 * source1, length)
    sy2 = lib_math.sum(source2 * source2, length)
    if not (sx == sx and sy == sy and sxy == sxy and sx2 == sx2 and sy2 == sy2):
        return na_float

    mx = sx / length
    my = sy / length
    cov = sxy / length - mx * my
    # Measured on scaled-down sources: a pair correlated at 0.95 still reports 0.0
    # once its covariance is scaled below the tolerance, and the switch sits exactly
    # at that bound (0.0 up to 9.99999e-11, a real value from 1.00001e-10). This is
    # also why ``length == 1`` is 0.0 -- there the covariance cancels exactly.
    if -_EPSILON <= cov <= _EPSILON:
        return 0.0

    # A constant source clamps its variance to zero, so the quotient stays na
    var_product = (builtins.max(0.0, sx2 / length - mx * mx)
                   * builtins.max(0.0, sy2 / length - my * my))
    if var_product == 0.0:
        return na_float
    return cov / math.sqrt(var_product)


# noinspection PyUnusedLocal
def cross(source1: float, source2: float) -> PyneBool:
    """
    Check if the source series crossed over or under the given series.

    :param source1: The first source series
    :param source2: The second source series
    :return: True if the source series crossed over the given series
    """
    # Measured: ``ta.cross`` is NOT ``crossover or crossunder``. It remembers the
    # last TOLERANTLY-unequal relation in a three-state armed flag and fires only on
    # a STRICT jump out of it. Unlike crossover/crossunder, whose armed state starts
    # engaged (they fire on the first jump out of a from-the-start equality plateau),
    # cross starts UNARMED: with no prior strict relation there is no direction to
    # cross, so a from-the-start equality plateau never fires either way. An equality
    # run (|diff| <= EPSILON) preserves the armed direction, so a strict relation
    # before the plateau still arms the jump after it. Probes U/D/E/F/G/H/I on
    # BINANCE:BTCUSDT 30m, 2026-08.
    armed: Persistent[int] = 0  # 0 unarmed, -1 last strictly below, +1 last strictly above
    res = (armed < 0 and source1 > source2) or (armed > 0 and source1 < source2)
    # Only refreshed on bars where both sources are defined; TV compares against the
    # last such bar, so na gaps must not reset the state
    if source1 == source1 and source2 == source2:
        diff = source1 - source2
        if diff < -_EPSILON:
            armed = -1
        elif diff > _EPSILON:
            armed = 1
    return res


# noinspection PyUnusedLocal
def crossover(source1: float, source2: float) -> PyneBool:
    """
    Check if the source series crossed over the given series.

    :param source1: The first source series
    :param source2: The second source series
    :return: True if the source series crossed over the given series
    """
    # Measured EXACT rule (no tolerance, immediate previous bar only): fire when
    # source1 was at or below source2 on the last bar where both were defined and
    # is STRICTLY above now -- a 5e-11 step at the previous bar both fires and
    # blocks, so the comparison is raw, not tolerant. A from-the-start equality
    # plateau AND an equality run entered from above both arm the jump (probes
    # S1-S9 on BINANCE:BTCUSDT 30m, 2026-08); crossover has no armed direction to
    # lose, unlike ta.cross. Pine has no na bool: with no previous relation to
    # compare against there is no cross, so the first defined bar yields false.
    was_le: Persistent[bool] = False
    res = source1 > source2 and was_le
    # Only refreshed on bars where both sources are defined; TV compares against the
    # last such bar, so na gaps must not reset the state
    if source1 == source1 and source2 == source2:
        was_le = source1 <= source2
    return res


# noinspection PyUnusedLocal
def crossunder(source1: float, source2: float) -> PyneBool:
    """
    Check if the source series crossed under the given series.

    :param source1: The first source series
    :param source2: The second source series
    :return: True if the source series crossed under the given series
    """
    # Measured EXACT rule, the mirror of crossover: fire when source1 was at or
    # above source2 on the last defined bar and is STRICTLY below now.
    was_ge: Persistent[bool] = False
    res = source1 < source2 and was_ge
    if source1 == source1 and source2 == source2:
        was_ge = source1 >= source2
    return res


def cum(source: Series[float | int]) -> PyneFloat:
    """
    Calculate the cumulative sum of the source series.

    :param source: The source series
    :return: The cumulative sum of the source series
    """
    if not (source == source):  # is_na_arg
        return na_float
    var: Persistent[float] = 0.0
    var += source
    return var


# The slice + ``oldest`` read on ``source`` looks ill-typed only because
# ``Series[T]`` erases to ``T`` for the IDE; it is a series under the transform.
# noinspection PyUnresolvedReferences,PyTypeChecker
def dev(source: Series[float], length: int, _mean: PyneFloat | None = None) -> PyneFloat:
    """
    Calculate the Mean Absolute Deviation (MAD) of the source series with the given length.

    :param source: The source series
    :param length: The length of the MAD calculation
    :param _mean: The mean value of the source series, if it is already calculated
    :return: The mean absolute deviation of the source series
    """
    # Bit-exact with Pine (measured, probes m556): TradingView computes the plain
    # newest-first loop ``sum(abs(source[i] - sma)) / length`` -- exactly the shape
    # below with the exact rolling-sum sma.
    assert length > 0, "Invalid length, length must be greater than 0!"
    if length == 1:
        return 0.0
    length = int(length)
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK
    # The loop below reads ``source[length - 1]``; grow the buffer so that index
    # stays addressable for lengths beyond the per-series default max_bars_back. The
    # resize is monotonic: a series ``length`` that dips low must not shrink the
    # buffer, or the history a later increase needs would already be gone.
    if length > capacity:
        capacity = length
        max_bars_back(source, capacity)

    mean = _mean if _mean is not None else sma(source, length)
    if not (mean == mean):
        return na_float

    # Newest-first walk over the raw window list: bit-identical to per-element
    # ``source[i]`` reads, but the list walk is several times cheaper -- the
    # ``__getitem__`` call, not the arithmetic, dominates here.
    summ = 0.0
    for y in builtins.reversed(source[0:length].oldest):
        summ += abs(y - mean)

    return summ / length


# noinspection PyPep8Naming
def dmi(diLength: int, adxSmoothing: int) -> tuple[PyneFloat, PyneFloat, PyneFloat]:
    """
    Calculate the Directional Movement Index (DMI) of the source series with the given DI length and ADX smoothing.

    :param diLength: The length of the DI
    :param adxSmoothing: The smoothing of the ADX
    :return: Tuple of three DMI series:
             - Positive Directional Movement (+DI)
             - Negative Directional Movement (-DI)
             - Average Directional Movement Index (ADX)
    """
    assert diLength > 0, "Invalid DI length, DI length must be greater than 0!"
    assert adxSmoothing > 0, "Invalid ADX smoothing, ADX smoothing must be greater than 0!"
    up = change(high)
    down = -change(low)
    # All three rolling averages must be entered on EVERY bar, the first one
    # included. ``tr`` carries the previous close in its own per-call-site slot,
    # so a bar it is not called on is a bar it never sees: entering it first at
    # bar 1 leaves that slot na and the true range there degrades to high - low,
    # which then seeds the average. MEASURED on TradingView (CAPITALCOM:EURUSD 60,
    # 22396 bars): the denominator of ``ta.dmi`` is ``rma`` over a true range that
    # is na on the first bar and exact from the second on -- ``tr()``, not
    # ``tr(true)``, so it differs from ``ta.atr(diLength)`` for the whole run
    # (240 bars still above 1e-12 apart). The directional movements stay na on
    # that bar for the same reason ``ta.change`` is na there.
    input_na = not (up == up) or not (down == down)
    a = rma(tr(), diLength)
    # The directional-movement selection compares TOLERANTLY, like Pine's own
    # operators: MEASURED on TradingView (CAPITALCOM:EURUSD 60), a bar whose high
    # and low moved by the same quantized amount produces no directional movement
    # at all, even though the two differences land ~1e-18 apart in float. A strict
    # ``>`` picks a movement there and the smoothed result stays off for hundreds
    # of bars (1324 of 22396 above 1e-12).
    plus_dm = na_float if input_na else (up if (up - down > _EPSILON and up > _EPSILON) else 0.0)
    minus_dm = na_float if input_na else (down if (down - up > _EPSILON and down > _EPSILON) else 0.0)
    p = rma(plus_dm, diLength)
    m = rma(minus_dm, diLength)
    if not (a == a) or not (p == p) or not (m == m) or a == 0.0:
        return na_float, na_float, na_float
    p = 100 * p / a
    m = 100 * m / a
    summ = p + m
    adx = rma(abs(p - m) / (summ if summ != 0.0 else 1.0), adxSmoothing) * 100
    return p, m, adx


def ema(source: PyneFloat, length: int) -> PyneFloat:
    """
    Calculate the Exponential Moving Average (EMA) of the source series with the given length.

    The average is seeded with :func:`sma` over the same length. na bars are skipped
    whole: the result is na there and the state does not advance, so the average always
    sees the last ``length`` non-na values.

    :param source: The source series
    :param length: The length of the EMA
    :return: The Exponential Moving Average (EMA) of the source series
    """
    # Measured law (probes m558, 27.8k bars each, lengths 2/5/9/14/21/50/200, every
    # displayed bar bit-identical): alpha = 2 / (length + 1) and the step is
    # ``prev + alpha * (source - prev)``. The algebraically equal
    # ``alpha * source + (1 - alpha) * prev`` drifts from TradingView on more than half
    # of the bars, and so does ``alpha * source + prev - alpha * prev``. The seed is the
    # na-compacted sma on the bar where that sma first exists.
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)
    if length == 1:  # Shortcut
        return source

    if not (source == source):  # is_na_arg
        return na_float

    alpha: Persistent[float] = 2 / (length + 1)
    last_val: Persistent[float] = na_float

    # Use SMA at warming stage
    if not (last_val == last_val):
        last_val = sma(source, length)
        return last_val

    # Warmed result
    last_val = last_val + alpha * (source - last_val)
    return last_val


# noinspection PyUnusedLocal
def falling(source: float, length: int) -> bool:
    """
    Test if the source series is now falling for length bars long.

    The fall test is tolerant: a step smaller than the float comparison tolerance
    does not count as falling.

    :param source: The source series
    :param length: The length of the falling test
    :return: True if the source series is falling for length bars long
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)

    last_val: Persistent[float] = na_float
    counter: Persistent[int] = 0

    if not (last_val == last_val):
        last_val = source
        return False

    # Tolerant step test, measured on TradingView (probe m547)
    if last_val - source > _EPSILON:
        counter += 1
    else:
        counter = 0

    last_val = source
    return counter >= length


# noinspection PyUnusedLocal,DuplicatedCode
@overload
def highest(source: Series[float], length: int, _bars: bool = False, _tuple: bool = False, _check_eq: bool = False) \
        -> PyneFloat:
    """
    Calculate the highest value of the source series with the given length.

    :param source: The source series
    :param length: The length of the highest value
    :param _bars: If true, return the number of bars since the highest value, internal use only
    :param _tuple: If true, return a tuple of the highest value and the number of bars since the highest value,
                   internal use only
    :param _check_eq: If true, check for equality too, internal use only
    :return: The highest value of the source series
    """
    # An int-typed Pine value can still carry a fraction (``int / int``); the
    # truncation happens where an integer is required — see ``_check_type``.
    length = int(length)
    _stale_on_gap(source, True)
    capacity: Persistent[int] = 0
    # TradingView sizes this window to the deepest read it has needed so far, and a
    # bar that skips the call reads the slot from one capacity back (see
    # ``_stale_on_gap``), so the capacity is part of the RESULT here, not just of the
    # storage: it decides how old that stale value is. The growth is monotonic, like
    # TradingView's own on-demand resize.
    if length > capacity:
        capacity = length
        max_bars_back(source, capacity)

    last_max: Persistent[float] = na_float
    last_max_index: Persistent[int] = 0
    last_bar: Persistent[int] = -1

    # The kept extreme ages in BARS, not in calls. A conditionally called window only
    # takes a stale slot into account when its rescan fires, so both halves of the
    # measurement have to hold together: with the bar-aged index over a ``length + 1``
    # ring, ``ta.lowest(low, 40)`` inside the else branch of the wild-corpus script
    # "Leledc levels (IS)" reproduces TradingView on all 28302 bars it runs on, while
    # call-aging misses 257 of them. The same law binds the opposite case — a window
    # read from inside a ``for`` loop, called many times on ONE bar (probe
    # ``hiloop_probe``, BINANCE:BTCUSDT 30m): the age advances once per bar no matter
    # how many calls land on it, and each call merely rewrites the newest ring slot.
    # Aging per call instead misses 5991 of 143815 returned values there.
    gap = bar_index - last_bar
    if last_bar >= 0 and gap > 0:
        last_max_index += gap
    last_bar = bar_index

    if last_max < source or not (last_max == last_max) or (_check_eq and last_max == source):
        last_max = source
        last_max_index = 0

    if last_max_index >= length:
        last_max = source
        last_max_index = 0
        for i in builtins.range(1, length):
            s = source[i]
            if s > last_max:
                last_max = s
                last_max_index = i
            elif not _check_eq and s == last_max:
                # For normal highest: update index for equal values
                last_max_index = i
            # For pivot detection (_check_eq=True): don't update index for equal values

    max_index = last_max_index

    if bar_index < length - 1:
        return na_float if not _tuple else (na_float, na_float)  # type: ignore[return-value]

    if _bars:
        return -max_index
    if _tuple:
        return last_max, -max_index  # type: ignore[return-value]
    return last_max


@overload
def highest(length: int) -> PyneFloat:
    return highest(high, length)


# noinspection PyUnusedLocal
@overload
def highestbars(source: Series[float], length: int) -> PyneFloat:
    """
    Calculate the number of bars since the highest value of the source series with the given length.

    :param source: The source series
    :param length: The length of the highest value
    :return: The number of bars since the highest value of the source series
    """
    return highest(source, length, _bars=True)


@overload
def highestbars(length: int) -> PyneFloat:
    return highest(high, length, _bars=True)


def hma(source: float, length: int) -> PyneFloat:
    """
    Calculate the Hull Moving Average (HMA) of the source series with the given length.

    :param source: The source series
    :param length: The length of the HMA
    :return: The Hull Moving Average (HMA) of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if not (source == source):  # is_na_arg
        return na_float
    length = int(length)

    ma_np2 = wma(source, length // 2)
    ma = wma(source, length)
    if not (ma == ma) or not (ma_np2 == ma_np2):
        return na_float
    return wma(2 * ma_np2 - ma, int(length ** 0.5))


@module_property
def iii() -> PyneFloat:
    """
    Intraday Intensity Index.

    :return: Intraday Intensity Index
    """
    return (2 * close - high - low) / ((high - low) * volume)


# noinspection PyPep8Naming
def kc(series: float, length: int, mult: float | int, useTrueRange: bool = True) \
        -> tuple[PyneFloat, PyneFloat, PyneFloat]:
    """
    Calculate the Keltner Channels (KC) of the source series with the given length and multiplier.

    :param series: The source series
    :param length: The length of the KC
    :param mult: The multiplier of the KC
    :param useTrueRange: Specifies whether to use True Range for KC calculation
    :return: The Keltner Channels (KC) of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    assert mult > 0, "Invalid multiplier, multiplier must be greater than 0!"

    base = ema(series, length)
    span = tr(False) if useTrueRange else (high - low)
    range_ma = ema(span, length)
    if not (base == base):
        return na_float, na_float, na_float
    if not (range_ma == range_ma):
        return base, na_float, na_float
    range_ma *= mult
    return base, base + range_ma, base - range_ma


# noinspection PyPep8Naming
def kcw(series: float, length: int, mult: float | int, useTrueRange: bool = True) -> PyneFloat:
    """
    Calculate the Keltner Channels Width (KCW) of the source series with the given length and multiplier.

    :param series: The source series
    :param length: The length of the KCW
    :param mult: The multiplier of the KCW
    :param useTrueRange: Specifies whether to use True Range for KCW calculation
    :return: The Keltner Channels Width (KCW) of the source series
    """
    b, h, l = kc(series, length, mult, useTrueRange)
    if not (b == b) or b == 0.0:
        return na_float
    return (h - l) / b


# The IDE findings here are ``@pyne`` transform artifacts: ``Persistent`` writes look
# unused because they are read on the NEXT bar, and the slice + ``oldest`` read
# on ``src`` look ill-typed because ``Series[T]`` erases to ``T`` for the IDE.
# noinspection PyUnusedLocal,PyUnresolvedReferences,PyTypeChecker
def linreg(source: Series[float], length: int, offset: int) -> PyneFloat:
    """
    Computes the linear regression value of the source series over a given period.

    :param source: Input series
    :param length: Number of bars to calculate regression
    :param offset: Number of bars to shift the result
    :return: Linear regression value
    """
    # TradingView recomputes the whole window every bar: its distance from the
    # exact rational result stays flat over 22k bars (probe m567), while a rolling
    # update of the two sums drifts to 1e-11 on the same data. The x-axis runs
    # 1..length from the oldest bar, and the result is the line evaluated at
    # ``length - offset``.
    # An int-typed Pine value can still carry a fraction (``int / int``); the
    # truncation happens where an integer is required — see ``_check_type``. It
    # precedes both the domain check and the single-bar shortcut, because the
    # regression runs on the truncated length: a 1.5 IS a 1, and a 0.5 IS an
    # invalid 0 rather than a value that passes ``> 0`` and then divides by zero.
    length = int(length)
    offset = int(offset)
    assert length > 0, "Invalid length, must be greater than 0!"
    if length == 1:
        return source

    count: Persistent[int] = 0
    val: Persistent[float] = na_float
    const_len: Persistent[int] = 0
    sum_x: Persistent[float] = 0.0
    denom: Persistent[float] = 0.0
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK

    if not (source == source):  # is_na_arg
        # An NA bar leaves the window unchanged; hold the last full value
        # (still NA while warming up)
        return na_float if count < length else val

    # NA values are NOT stored in the buffer, only skipped, so ``src[i]`` is the
    # i-th most recent non-NA value. Reading the parameter directly would step
    # back whole *bars* and land inside an NA gap, poisoning the sums.
    src: Series[float] = source
    # Grow the na-compacted buffer so the oldest window slot stays addressable for
    # lengths beyond the per-series default max_bars_back; otherwise the window
    # read returns na and the whole regression collapses to na. The resize is
    # monotonic: a series ``length`` that dips low must not shrink the buffer, or
    # the history a later increase needs would already be gone.
    if length > capacity:
        capacity = length
        max_bars_back(src, capacity)

    if count < length:
        count += 1
        if count < length:  # Not enough data yet
            return na_float

    # The x-side sums depend only on ``length``; they are accumulated with the
    # same sequential loop TV runs (so the cached floats are bit-identical to a
    # per-bar recompute) but only when the length changes.
    if const_len != length:
        const_len = length
        sx = 0.0
        sx2 = 0.0
        for i in builtins.range(1, length + 1):
            per = builtins.float(i)
            sx = sx + per
            sx2 = sx2 + per * per
        sum_x = sx
        denom = length * sx2 - sx * sx

    # The y-side sums are a fresh oldest-first walk every bar. The window is
    # taken as a raw list instead of per-element ``src[i]`` reads: the two are
    # bit-identical (the accumulators are independent, so splitting the x and y
    # sums does not change either sequence), but the list walk is several times
    # cheaper -- the ``__getitem__`` call, not the arithmetic, dominates here.
    sum_y = 0.0
    sum_xy = 0.0
    per = 0.0
    for y in src[0:length].oldest:
        per = per + 1.0
        sum_y = sum_y + y
        sum_xy = sum_xy + y * per

    # The line is anchored one step short of the evaluation point: the intercept
    # carries an extra ``+ slope`` and the shift is ``length - 1 - offset``. The
    # algebraically equal ``intercept + slope * (length - offset)`` rounds
    # differently and misses TradingView on a quarter to a half of the bars
    # (probes m567/m568, eleven length/offset configurations over 22k bars each,
    # zero mismatches this way). ``slope * sum_x / length`` is left to right --
    # ``slope * (sum_x / length)`` costs another handful of bars.
    slope = (length * sum_xy - sum_x * sum_y) / denom
    average = sum_y / length
    intercept = average - slope * sum_x / length + slope

    val = intercept + slope * (length - 1 - offset)
    return val


# noinspection PyUnusedLocal,DuplicatedCode
@overload
def lowest(source: Series[float], length: int,
           _bars: bool = False, _tuple: bool = False, _check_eq: bool = False) \
        -> PyneFloat:
    """
    Calculate the lowest value of the source series with the given length.

    :param source: The source series
    :param length: The length of the lowest value
    :param _bars: If true, return the number of bars since the lowest value, internal use only
    :param _tuple: If true, return a tuple of the lowest value and the number of bars since the lowest value,
                   Internal use only
    :param _check_eq: If true, check for equality too, internal use only
    :return: The lowest value of the source series
    """
    # An int-typed Pine value can still carry a fraction (``int / int``); the
    # truncation happens where an integer is required — see ``_check_type``.
    length = int(length)
    _stale_on_gap(source, True)
    capacity: Persistent[int] = 0
    # See ``highest``: the window capacity decides how old the stale value a skipped
    # bar leaves behind is, so it must be the length TradingView sizes it to.
    if length > capacity:
        capacity = length
        max_bars_back(source, capacity)

    last_min: Persistent[float] = na_float
    last_min_index: Persistent[int] = 0
    last_bar: Persistent[int] = -1

    # The kept extreme ages in BARS, not in calls: a bar that skips the call still
    # moves the window on, and a bar that calls it many times moves it only once, so
    # the rescan that lets a stale slot in fires on the same bar TradingView fires it
    # (see ``highest``).
    gap = bar_index - last_bar
    if last_bar >= 0 and gap > 0:
        last_min_index += gap
    last_bar = bar_index

    if last_min > source or not (last_min == last_min) or (_check_eq and last_min == source):
        last_min = source
        last_min_index = 0

    if last_min_index >= length:
        last_min = source
        last_min_index = 0
        for i in builtins.range(1, length):
            s = source[i]
            if s < last_min:
                last_min = s
                last_min_index = i
            elif not _check_eq and s == last_min:
                # For normal lowest: update index for equal values
                last_min_index = i
            # For pivot detection (_check_eq=True): don't update index for equal values

    min_index = last_min_index

    if bar_index < length - 1:
        return na_float if not _tuple else (na_float, NA(int))  # type: ignore[return-value]

    if _bars:
        return -min_index
    if _tuple:
        return last_min, -min_index  # type: ignore[return-value]
    return last_min


@overload
def lowest(length: int) -> PyneFloat:
    return lowest(low, length)


# noinspection PyUnusedLocal
@overload
def lowestbars(source: Series[float], length: int) -> PyneFloat:
    """
    Calculate the number of bars since the lowest value of the source series with the given length.

    :param source: The source series
    :param length: The length of the lowest value
    :return: The number of bars since the lowest value of the source series
    """
    return lowest(source, length, _bars=True)


@overload
def lowestbars(length: int) -> PyneFloat:
    return lowest(low, length, _bars=True)


def macd(source: float, fastlen: int, slowlen: int, siglen: int) \
        -> tuple[PyneFloat, PyneFloat, PyneFloat]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) of the source series with the given
    fast, slow, and signal lengths.

    :param source: The source series
    :param fastlen: The length of the fast EMA
    :param slowlen: The length of the slow EMA
    :param siglen: The length of the signal EMA
    :return: Tuple of three MACD series:
             - MACD Line
             - Signal Line
             - Histogram
    """
    assert fastlen > 0, "Invalid fast length, fast length must be greater than 0!"
    assert slowlen > 0, "Invalid slow length, slow length must be greater than 0!"
    assert siglen > 0, "Invalid signal length, signal length must be greater than 0!"
    fast = ema(source, fastlen)
    slow = ema(source, slowlen)
    if not (fast == fast) or not (slow == slow):
        return na_float, na_float, na_float
    macd_val = fast - slow
    signal = ema(macd_val, siglen)
    if not (signal == signal):
        return macd_val, na_float, na_float
    return macd_val, signal, macd_val - signal


# noinspection PyShadowingBuiltins
def max(source: Series[float]) -> PyneFloat:
    """
    Calculate the maximum value of the source series.

    :param source: The source series
    :return: The maximum value of the source series
    """
    max_val: Persistent[float] = na_float
    if max_val < source or not (max_val == max_val):
        max_val = source
    return max_val


# The IDE findings below are ``@pyne`` transform artifacts: ``Persistent``
# assignments look dead because their value is read on the NEXT bar, and ``src``
# looks possibly-unbound because it is a series whose storage outlives the ``if``
# that feeds it.
# noinspection PyUnusedLocal,PyUnboundLocalVariable
def median(source: Series[TFI], length: int) -> TFI:
    """
    Calculate the median of the source series over a given period.

    :param source: Input series of values
    :param length: Number of bars to calculate over
    :return: The median value or na during warmup
    """
    # Store heaps and window
    heap_low: Persistent[list[TFI]] = []  # Max heap (negative values)
    heap_high: Persistent[list[TFI]] = []  # Min heap
    window: Persistent[list[TFI]] = []  # Recent values for removal
    prev_length: Persistent[int] = 0
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK

    source_na = not (source == source)  # is_na_arg
    if not source_na:
        # The window drops na bars, so the rebuild below has to read a history
        # that drops them too: this second buffer is na-compacted (only non-na
        # bars reach the assignment), so ``src[k]`` is the k-th most recent
        # non-na value. It is fed ahead of every length guard below, because the
        # bars those return early on are still part of the history a later
        # rebuild has to see.
        src: Series[TFI] = source

    # na length is an all-na series here, not an error; see
    # ``percentile_linear_interpolation`` for the measured family law.
    if not (length == length):  # is_na_arg
        # The window is frozen while the length is na; ``-1`` marks it for a
        # rebuild so a valid length coming back does not inherit a short window.
        prev_length = -1
        return NA(cast(type[TFI], type(source)))  # type: ignore
    # An int-typed Pine value can still carry a fraction (``int / int``); the
    # truncation happens where an integer is required — see ``_check_type``. It
    # precedes the domain check, because a 1.5 IS a 1 and a 0.5 IS an invalid 0
    # rather than a value that passes ``> 0`` and then indexes an empty heap.
    length = int(length)
    assert length > 0, "Invalid length, length must be greater than 0!"
    # History is only read on a mid-run length change (window rebuild); keep the
    # buffer large enough for that. The resize is monotonic and floored at the
    # series' own default: a series ``length`` that dips low must not shrink the
    # buffer, or the history a later increase needs would already be gone.
    if length > capacity:
        capacity = length
        max_bars_back(src, capacity)

    if source_na:
        # An na bar is not part of the window at all, so it must not touch the
        # length bookkeeping either: the next non-na bar still has to see the
        # length change this bar may have carried.
        # type(source) would be the NA class itself — keep the source sentinel's type
        return source

    if length != prev_length and prev_length != 0:
        # ``length`` is a series value and changed: rebuild the window and both
        # heaps from the source history, oldest first, without the current bar.
        # Without this a shrinking length would never evict more than one value
        # per bar and the machine would keep answering from the older, wider
        # window. The buffer is na-compacted, so an na only comes back where the
        # history ends — deepest first, which is where the skip belongs.
        window = []
        heap_low = []
        heap_high = []
        for i in builtins.range(length - 1, 0, -1):
            old_value = src[i]
            if old_value == old_value:  # is_na_arg
                window.append(old_value)
                heapq.heappush(heap_low, -old_value)
                heapq.heappush(heap_high, -heapq.heappop(heap_low))
                if len(heap_low) < len(heap_high):
                    heapq.heappush(heap_low, -heapq.heappop(heap_high))
    prev_length = length

    if length == 1:  # Shortcut
        return source

    # Add new value and balance heaps
    value = source
    window.append(value)
    heapq.heappush(heap_low, -value)
    heapq.heappush(heap_high, -heapq.heappop(heap_low))

    if len(heap_low) < len(heap_high):
        heapq.heappush(heap_low, -heapq.heappop(heap_high))

    # Remove old value if window full
    if len(window) > length:
        old = window.pop(0)

        # Remove from correct heap
        if old <= -heap_low[0]:
            heap_low.remove(-old)
            heapq.heapify(heap_low)
        else:
            heap_high.remove(old)
            heapq.heapify(heap_high)

        # Rebalance if needed
        if len(heap_low) < len(heap_high):
            heapq.heappush(heap_low, -heapq.heappop(heap_high))
        elif len(heap_low) > len(heap_high) + 1:
            heapq.heappush(heap_high, -heapq.heappop(heap_low))

    # Return na during warmup
    if len(window) < length:
        return NA(cast(type[TFI], type(source)))  # type: ignore

    # Return median based on heap sizes
    if len(heap_low) > len(heap_high):
        return -heap_low[0]  # Max heap root
    return -heap_low[0] if isinstance(source, int) else (-heap_low[0] + heap_high[0]) / 2  # type: ignore


def mfi(series: float, length: int) -> PyneFloat:
    """
    Calculate the Money Flow Index (MFI) of the source series with the given length.

    The money-flow direction test is tolerant: a price change below the float
    comparison tolerance in magnitude counts as neither inflow nor outflow. With
    both sums zero the result is 100, not na.

    :param series: The source series
    :param length: The length of the MFI
    :return: The Money Flow Index (MFI) of the source series
    """
    # Tolerant direction test measured on TradingView (probes m548/m549); the empty
    # case was measured separately (probe m550) on an exactly flat source, where the
    # tolerance plays no part.
    assert length > 0, "Invalid length, length must be greater than 0!"
    if not (series == series):  # is_na_arg
        return na_float
    length = int(length)

    chg = change(series)
    chg_na = not (chg == chg)
    upper = lib_math.sum(volume * (0.0 if not chg_na and chg <= _EPSILON else series), length)
    lower = lib_math.sum(volume * (0.0 if not chg_na and chg >= -_EPSILON else series), length)
    if not (upper == upper) or not (lower == lower):
        return na_float
    # A side made of pure accumulation dust counts as an exact zero: the rolling
    # sums do not return to zero when their whole window is zero, and Pine's
    # comparison tolerance swallows the residue (measured on probe m566).
    if -_EPSILON <= upper <= _EPSILON:
        upper = 0.0
    if -_EPSILON <= lower <= _EPSILON:
        lower = 0.0
    if lower == 0.0:  # Includes the empty case, where the upper sum is zero too
        return 100.0
    # The money ratio is formed first and the whole result derived from it; the
    # algebraically equal ``100 - 100 * lower / (upper + lower)`` rounds differently
    # and misses TV on 38% of the bars.
    return 100.0 - (100.0 / (1.0 + upper / lower))


# noinspection PyShadowingBuiltins
def min(source: Series[float]) -> PyneFloat:
    """
    Calculate the minimum value of the source series.

    :param source: The source series
    :return: The minimum value of the source series
    """
    min_val: Persistent[float] = na_float
    if min_val > source or not (min_val == min_val):
        min_val = source
    return min_val


def mode(source: Series[TFI], length: int) -> TFI:
    """
    Returns the mode of the series. If there are several values with the same frequency,
    it returns the smallest value.

    :param source: Series of values to process
    :param length: Number of bars (length)
    :return: The most frequently occurring value from the source. If none exists, returns
             the smallest value instead. Returns na during warm-up period.
    """
    if not (length == length):  # is_na_arg
        return cast(TFI, NA(builtins.type(source)))
    # An int-typed Pine value can still carry a fraction (``int / int``); the
    # truncation happens where an integer is required — see ``_check_type``. It
    # precedes both the domain check and the warmup guard, because the whole
    # function runs on the truncated length: a 1.5 IS a 1, and a 0.5 IS an
    # invalid 0 rather than a value that passes ``> 0`` and then returns na.
    length = int(length)
    assert length > 0, "Invalid length, length must be greater than 0!"
    if not (source == source):  # is_na_arg
        return source
    if bar_index < length - 1:
        return cast(TFI, NA(builtins.type(source)))

    # Store values for quick access
    values = [source[i] for i in builtins.range(length) if source[i] == source[i]]
    if not values:
        return cast(TFI, NA(builtins.type(source)))

    # Find mode - sort values to handle equal frequencies
    values.sort()  # Ensure we pick the smallest value when frequencies are equal
    mode_val = values[0]
    current_val = values[0]
    max_freq = curr_freq = 1

    # Single pass through sorted values
    for i in builtins.range(1, len(values)):
        if values[i] == current_val:
            curr_freq += 1
            if curr_freq > max_freq:
                max_freq = curr_freq
                mode_val = current_val
        else:
            current_val = values[i]
            curr_freq = 1

    return mode_val


def mom(source: float, length: int) -> PyneFloat:
    """
    Calculate the Momentum of the source series with the given length.

    :param source: The source series
    :param length: The length of the Momentum
    :return: The Momentum of the source series
    """
    # It is exactly the same as change function
    return change(source, length)


# noinspection PyUnusedLocal
@module_property
def nvi() -> PyneFloat:
    """
    Negative Volume Index.

    :return: Negative Volume Index
    """
    prev_close: Persistent[float] = 0.0
    prev_volume: Persistent[float] = 0.0
    prev_nvi: Persistent[float] = 1.0

    if close == 0.0 or prev_close == 0.0:
        _nvi = prev_nvi
    else:
        _nvi = prev_nvi + ((close - prev_close) / prev_close) * prev_nvi if volume < prev_volume else prev_nvi

    prev_close = close
    prev_volume = volume
    prev_nvi = _nvi

    return _nvi


@module_property
def obv() -> PyneFloat:
    """
    On Balance Volume.

    :return: On Balance Volume
    """
    chg = change(close)
    if not (chg == chg):
        return na_float
    if chg > 0:
        chg = 1.0
    elif chg < 0:
        chg = -1.0
    else:
        chg = 0.0
    return cum(volume * chg)


# noinspection PyUnusedLocal,PyProtectedMember
def percentile_linear_interpolation(source: Series[float], length: int, percentage: int | float) \
        -> PyneFloat:
    """
    Calculates percentile using method of linear interpolation between the two nearest ranks.

    :param source: The source series
    :param length: The length of the percentile
    :param percentage: The percentage of the percentile
    :return: The percentile of the source series
    """
    # Rolling state and its tolerant ordering are ``percentile_nearest_rank``'s;
    # see there for the measurement. Confirmed for this form too (probe m580, four
    # length/percentage configurations, 22k bars each, zero mismatches), where an
    # exact order misses TradingView on up to 44% of the bars.
    window: Persistent[deque[float]] = deque()
    sorted_buf: Persistent[list[float]] = []
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK

    # The percentile machines are the only rolling ``ta`` functions that accept an
    # na length: TradingView returns na for the whole series instead of raising
    # (probe sweep on BINANCE:BTCUSDT 30m -- ``percentile_nearest_rank``,
    # ``percentile_linear_interpolation``, ``median`` and ``mode`` all export an
    # all-na plot, while sma/stdev/wma/linreg/change/highest/percentrank and the
    # rest raise RE10003 "must not be na"). A length of exactly 0 raises RE10001
    # in this family too, so only the na case is let through. A na-length call
    # leaves the machine untouched (measured on a shared loop call site: the
    # next valid-length call continues the very same window).
    if not (length == length):  # is_na_arg
        return na_float
    # An int-typed Pine value can still carry a fraction (``int / int``); the
    # truncation happens where an integer is required — see ``_check_type``. It
    # precedes the domain check, because a 1.5 IS a 1 and a 0.5 IS an invalid 0
    # rather than a value that passes ``> 0`` and then answers na forever.
    length = int(length)
    assert length > 0, "Invalid length, length must be greater than 0!"
    # The underfull top-up below reads up to ``length - 1`` bars of history; keep
    # the buffer large enough for that. Done before the warmup guard so the oldest
    # candles are kept from the first bar on. The resize is monotonic and floored
    # at the series' own default: a series ``length`` that dips low must not
    # shrink the buffer, or the history a later increase needs would already be
    # gone by the time the length comes back up.
    if length > capacity:
        capacity = length
        max_bars_back(source, capacity)

    window.append(source)
    if source == source:
        sorted_buf.insert(_tol_lower_bound(sorted_buf, source), source)
    if len(window) < length:
        # Underfull window — the first bars, a longer length arriving on a
        # shared call-site machine, or a resume after na-length calls:
        # TradingView tops the window up to ``length`` from the source's BAR
        # history (``source[i]`` for ``i = len(window)..length - 1``, na beyond
        # the buffer), prepended as older entries that PERSIST in the window —
        # they are evicted like ordinary pushes, never replaced. See
        # ``percentile_nearest_rank`` for the measurement.
        fills = []
        for i in builtins.range(len(window), length):
            fills.append(source[i])
        for v in fills:
            window.appendleft(v)
        for v in fills:
            if v == v:
                sorted_buf.insert(_tol_lower_bound(sorted_buf, v), v)
    while len(window) > length:
        old = window.popleft()
        if old == old:
            pos = _tol_upper_bound(sorted_buf, old) - 1
            del sorted_buf[pos if pos > 0 else 0]

    if not (source == source):  # is_na_arg
        return na_float

    if bar_index < length - 1:
        return na_float

    # The na elements of the window sort to the virtual end; n is the full
    # window length, na included -- same semantics as the array form. The
    # interpolation position ``pos = length * percentage / 100 + 0.5`` inside is
    # TradingView's own (probe m554: a percentage sweep over a window of four
    # separated values walks the ranks in exact half steps).
    return array._select_linear_interpolation(sorted_buf, length, percentage)


# TradingView advances this machine once per EXECUTION of its call site — loop
# iterations share it (measured: a [5,9] / [5,na,9] length loop reproduces
# every exported value bit-exactly on the shared machine, while per-iteration
# instances miss 83% of the bars) — so the isolation transformer must not give
# loop iterations their own instances. Not a family-wide builtin law:
# ``ta.ema``/``ta.sma`` measured the opposite way (see ``_FAST_SHARED`` in the
# function_isolation transformer).
percentile_linear_interpolation.__pyne_shared_call_site__ = True


# noinspection PyUnusedLocal,PyProtectedMember
def percentile_nearest_rank(source: Series[float], length: int, percentage: int | float) \
        -> PyneFloat:
    """
    Calculates percentile using the nearest rank method.

    :param source: The source series
    :param length: The length of the percentile
    :param percentage: The percentage of the percentile
    :return: The percentile of the source series
    """
    # Rolling state: the chronological window plus its ascending-ordered numeric
    # part, maintained incrementally (insert/remove per bar instead of a full
    # re-sort of the window).
    #
    # Measured law (probes m577-m579, twelve length/percentage configurations over
    # 22k bars each, zero mismatches): the order is the tolerant one, a new value
    # goes in front of the values it ties with and an evicted value takes the last
    # of its ties with it. That is what makes the builtin depend on the order the
    # window was filled in and not only on its contents -- an exact order misses
    # TradingView on up to 43% of the bars once the window holds ties. The window
    # is where the divergence lives: a tie needs the two values to be within the
    # comparison tolerance of each other, which a spacing sweep confirmed at the
    # expected 1e-10 (1e-11 apart still ties, 1e-10 apart no longer does).
    #
    # The machine is shared by every execution of the call site — loop
    # iterations included — and each call advances it once: push the current
    # value, top an underfull window up from the source's bar history, evict
    # while over the CURRENT call's length. Measured (BINANCE:BTCUSDT 30m,
    # min/median/max order statistics on a bar_index-coded source): a loop over
    # lengths [5,9] / [9,5] / [5,na,9] / [14,28,42,84,98,na,na] reproduces
    # every window boundary and rank bit-exactly under this law, while
    # per-iteration instances, a rebuild-on-length-change and every trim-only
    # variant miss by thousands of bars. The top-up entries persist as ordinary
    # window elements (two consecutive top-ups leave two separate history
    # blocks with a one-bar gap between them — observed, not an artifact).
    window: Persistent[deque[float]] = deque()
    sorted_buf: Persistent[list[float]] = []
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK

    # na length is an all-na series here, not an error; see
    # ``percentile_linear_interpolation`` for the measured family law. The
    # machine stays untouched (measured: on a shared loop call site the next
    # valid-length call continues the very same window).
    if not (length == length):  # is_na_arg
        return na_float
    # An int-typed Pine value can still carry a fraction (``int / int``); the
    # truncation happens where an integer is required — see ``_check_type``. It
    # precedes the domain check, because a 1.5 IS a 1 and a 0.5 IS an invalid 0
    # rather than a value that passes ``> 0`` and then answers na forever.
    length = int(length)
    assert length > 0, "Invalid length, length must be greater than 0!"
    # The underfull top-up reads up to ``length - 1`` bars of history; keep the
    # buffer large enough for that. Done before the warmup guard so the oldest
    # candles are kept from the first bar on. The resize is monotonic and floored
    # at the series' own default: a series ``length`` that dips low must not
    # shrink the buffer, or the history a later increase needs would already be
    # gone by the time the length comes back up.
    if length > capacity:
        capacity = length
        max_bars_back(source, capacity)

    window.append(source)
    if source == source:
        sorted_buf.insert(_tol_lower_bound(sorted_buf, source), source)
    if len(window) < length:
        # Top the window up to ``length`` with the source's bar history —
        # ``source[i]`` for ``i = len(window)..length - 1``, na beyond the
        # buffer — prepended as older entries, nearest bar innermost, so the
        # deque stays chronological.
        fills = []
        for i in builtins.range(len(window), length):
            fills.append(source[i])
        for v in fills:
            window.appendleft(v)
        for v in fills:
            if v == v:
                sorted_buf.insert(_tol_lower_bound(sorted_buf, v), v)
    while len(window) > length:
        old = window.popleft()
        if old == old:
            # Tolerant equality is not transitive, so a long enough chain of ties
            # can drift until nothing is left within tolerance of the evicted
            # value; the first slot is then the closest match left.
            pos = _tol_upper_bound(sorted_buf, old) - 1
            del sorted_buf[pos if pos > 0 else 0]

    if not (source == source):  # is_na_arg
        return na_float

    if bar_index < length - 1:
        return na_float

    # The na elements of the window sort to the virtual end; n is the full
    # window length, na included -- same semantics as the array form. The rank
    # ``ceil(percentage * length / 100)`` inside is TradingView's own (probes
    # m552/m554, measured with a percentage sweep over separated values).
    return array._select_nearest_rank(sorted_buf, length, percentage)


# Shared per-call-site machine on TradingView, loop iterations included — see
# the docstring's measured law and ``percentile_linear_interpolation``'s
# matching marker.
percentile_nearest_rank.__pyne_shared_call_site__ = True


def percentrank(source: Series[float], length: int) -> PyneFloat:
    """
    Percent rank is the percents of how many previous values was less than or equal to the current
    value of given series.

    :param source: The source series
    :param length: Number of bars back to include in the calculation
    :return: The percentage of values less than or equal to the current value
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK
    # The final slice reads ``length + 1`` candles of history; a buffer of
    # ``max_bars_back == length`` (capacity ``length + 1``) holds exactly that.
    # Done before the warmup guard so the oldest candles are kept from bar 0. The
    # resize is monotonic: a series ``length`` that dips low must not shrink the
    # buffer, or the slice a later increase needs would run off its end.
    if length > capacity:
        capacity = length
        max_bars_back(source, capacity)
    if not (source == source):  # is_na_arg
        return na_float

    if bar_index < length:
        return na_float

    return array.percentrank(source[:length + 1], 0)  # type: ignore


# noinspection PyUnusedLocal,PyShadowingBuiltins
def pivot_point_levels(type: str, anchor: bool, developing: bool = False) -> list[PyneFloat]:
    """
    Calculate pivot point levels based on the specified calculation type.

    Returns an array of 11 float values representing pivot point levels:
    [P, R1, S1, R2, S2, R3, S3, R4, S4, R5, S5]

    :param type: Pivot calculation type: "Traditional", "Fibonacci", "Woodie",
                 "Classic", "DM", or "Camarilla"
    :param anchor: Condition that triggers recalculation of pivot levels
                   (e.g., timeframe.change("D"))
    :param developing: If true, values recalculate on each bar using current OHLC;
                      if false (default), values remain constant until the next anchor
    :return: Array of 11 float values: [P, R1, S1, R2, S2, R3, S3, R4, S4, R5, S5]
             Not all types support all levels - unsupported ones return na_float
    """
    # Persistent state for anchor-based calculation
    # These store the COMPLETED previous period's values (used when developing=False)
    prev_period_high: Persistent[float] = na_float
    prev_period_low: Persistent[float] = na_float
    prev_period_close: Persistent[float] = na_float
    prev_period_open: Persistent[float] = na_float

    # These accumulate values for the CURRENT period (will become prev_period on next anchor)
    curr_period_high: Persistent[float] = na_float
    curr_period_low: Persistent[float] = na_float
    curr_period_open: Persistent[float] = na_float
    is_first_bar_of_period: Persistent[bool] = True

    levels: Persistent[list[PyneFloat]] = [na_float] * 11
    had_anchor: Persistent[bool] = False

    # Normalize type to lowercase for case-insensitive comparison
    type_lower = type.lower() if isinstance(type, str) else ""

    # On anchor, finalize the previous period and start a new one
    if anchor:
        # Save the accumulated current period values as the previous period
        prev_period_high = curr_period_high
        prev_period_low = curr_period_low
        prev_period_close = close[1] if bar_index > 0 else close  # Last close of prev period
        prev_period_open = curr_period_open

        # Reset current period accumulators for the new period
        curr_period_high = high
        curr_period_low = low
        curr_period_open = open
        is_first_bar_of_period = False
        had_anchor = True
    else:
        # Accumulate OHLC extremes for current period
        if is_first_bar_of_period or not (curr_period_high == curr_period_high):
            curr_period_high = high
            curr_period_low = low
            curr_period_open = open
            is_first_bar_of_period = False
        else:
            if high == high and high > curr_period_high:
                curr_period_high = high
            if low == low and low < curr_period_low:
                curr_period_low = low

    # If no anchor has occurred yet, return all NA values
    if not had_anchor:
        return [na_float] * 11

    # Determine which OHLC values to use
    if developing:
        # Use current accumulated values for developing mode
        h = curr_period_high
        l = curr_period_low
        c = close  # Current close for developing
        o = curr_period_open
    else:
        # Use previous period's OHLC (fixed after anchor)
        h = prev_period_high
        l = prev_period_low
        c = prev_period_close
        o = prev_period_open

    # Check for NA values
    if not (h == h) or not (l == l) or not (c == c):
        return [na_float] * 11

    # Calculate range
    rng = h - l

    # Calculate levels based on type
    if type_lower in ("traditional", "classic"):
        # Traditional/Classic Pivot Points
        p = (h + l + c) / 3
        r1 = 2 * p - l
        s1 = 2 * p - h
        r2 = p + rng
        s2 = p - rng
        r3 = r1 + rng
        s3 = s1 - rng
        levels = [p, r1, s1, r2, s2, r3, s3, na_float, na_float, na_float, na_float]

    elif type_lower == "fibonacci":
        # Fibonacci Pivot Points
        p = (h + l + c) / 3
        r1 = p + 0.382 * rng
        s1 = p - 0.382 * rng
        r2 = p + 0.618 * rng
        s2 = p - 0.618 * rng
        r3 = p + 1.000 * rng
        s3 = p - 1.000 * rng
        levels = [p, r1, s1, r2, s2, r3, s3, na_float, na_float, na_float, na_float]

    elif type_lower == "woodie":
        # Woodie Pivot Points
        # Note: Woodie uses current period's OPEN (not prev period's close) for the "close" component
        # This makes Woodie more responsive to current price action
        woodie_c = curr_period_open if not developing else close
        if not (woodie_c == woodie_c):
            return [na_float] * 11
        p = (h + l + 2 * woodie_c) / 4
        r1 = 2 * p - l
        s1 = 2 * p - h
        r2 = p + rng
        s2 = p - rng
        r3 = r1 + rng
        s3 = s1 - rng
        levels = [p, r1, s1, r2, s2, r3, s3, na_float, na_float, na_float, na_float]

    elif type_lower == "dm":
        # DeMark Pivot Points
        if not (o == o):
            return [na_float] * 11

        if c < o:
            x = h + 2 * l + c
        elif c > o:
            x = 2 * h + l + c
        else:  # c == o
            x = h + l + 2 * c

        p = x / 4
        r1 = x / 2 - l
        s1 = x / 2 - h
        # DM only has P, R1, S1
        levels = [p, r1, s1, na_float, na_float, na_float, na_float,
                  na_float, na_float, na_float, na_float]

    elif type_lower == "camarilla":
        # Camarilla Pivot Points
        p = (h + l + c) / 3
        r1 = c + rng * 1.1 / 12
        s1 = c - rng * 1.1 / 12
        r2 = c + rng * 1.1 / 6
        s2 = c - rng * 1.1 / 6
        r3 = c + rng * 1.1 / 4
        s3 = c - rng * 1.1 / 4
        r4 = c + rng * 1.1 / 2
        s4 = c - rng * 1.1 / 2
        # Camarilla has P, R1-R4, S1-S4 (no R5, S5)
        levels = [p, r1, s1, r2, s2, r3, s3, r4, s4, na_float, na_float]

    else:
        # Unknown type - return all NA
        levels = [na_float] * 11

    return levels


@overload
def pivothigh(source: float, leftbars: int, rightbars: int) -> PyneFloat:
    """
    This function returns price of the pivot high point. It returns 'NaN', if there was no pivot high point.

    :param source: The source series
    :param leftbars: Left strength
    :param rightbars: Right strength.
    :return: Price of the pivot high point, or NaN if no pivot
    """
    # An int-typed Pine value can still carry a fraction (``int / int``); the
    # truncation happens where an integer is required — see ``_check_type``. The
    # strength checks run on the truncated values, since those are the ones the
    # pivot window is built from: a 0.5 strength is an invalid 0, not a legal side.
    leftbars = int(leftbars)
    rightbars = int(rightbars)
    assert leftbars > 0, "Invalid leftbars, leftbars must be greater than 0!"
    assert rightbars > 0, "Invalid rightbars, rightbars must be greater than 0!"

    if not (source == source):  # is_na_arg
        return na_float

    pivotrange = leftbars + rightbars + 1
    ph, pi = cast(tuple[float, int], highest(source, pivotrange, _tuple=True, _check_eq=True))

    if pi == -rightbars:
        return ph

    return na_float


@overload
def pivothigh(leftbars: int, rightbars: int) -> PyneFloat:
    """
    This function returns price of the pivot high point. It returns 'NaN', if there was no pivot high point.

    :param leftbars: Left strength
    :param rightbars: Right strength.
    :return: Price of the pivot high point, or NaN if no pivot
    """
    try:
        return pivothigh(safe_convert.safe_float(high), leftbars, rightbars)  # type: ignore
    except TypeError:
        if not (high == high):
            return na_float
        else:
            raise


@overload
def pivotlow(source: float, leftbars: int, rightbars: int) -> PyneFloat:
    """
    This function returns price of the pivot low point. It returns 'NaN', if there was no pivot low point.

    :param source: The source series
    :param leftbars: Left strength
    :param rightbars: Right strength.
    :return: Price of the pivot low point, or NaN if no pivot
    """
    # An int-typed Pine value can still carry a fraction (``int / int``); the
    # truncation happens where an integer is required — see ``_check_type``. The
    # strength checks run on the truncated values, since those are the ones the
    # pivot window is built from: a 0.5 strength is an invalid 0, not a legal side.
    leftbars = int(leftbars)
    rightbars = int(rightbars)
    assert leftbars > 0, "Invalid leftbars, leftbars must be greater than 0!"
    assert rightbars > 0, "Invalid rightbars, rightbars must be greater than 0!"

    if not (source == source):  # is_na_arg
        return na_float

    pivotrange = leftbars + rightbars + 1
    pl, pi = cast(tuple[float, int], lowest(source, pivotrange, _tuple=True, _check_eq=True))
    if pi == -rightbars:
        return pl

    return na_float


@overload
def pivotlow(leftbars: int, rightbars: int) -> PyneFloat:
    """
    This function returns price of the pivot low point. It returns 'NaN', if there was no pivot low point.

    :param leftbars: Left strength
    :param rightbars: Right strength.
    :return: Price of the pivot low point, or NaN if no pivot
    """
    try:
        return pivotlow(safe_convert.safe_float(low), leftbars, rightbars)  # type: ignore
    except TypeError:
        if not (low == low):
            return na_float
        else:
            raise


# noinspection PyUnusedLocal
@module_property
def pvi() -> PyneFloat:
    """
    Positive Volume Index.

    :return: Positive Volume Index
    """
    prev_close: Persistent[float] = 0.0
    prev_volume: Persistent[float] = 0.0
    prev_pvi: Persistent[float] = 1.0

    _pvi = prev_pvi + ((close - prev_close) / prev_close) * prev_pvi if volume > prev_volume else prev_pvi
    # na() predicate semantics: a division by zero (prev_close warmup 0.0) gives
    # inf, which is na on TV — not just nan, so the guard must be isfinite-based.
    if isinstance(_pvi, NA) or not math.isfinite(_pvi):
        _pvi = prev_pvi

    prev_close = close
    prev_volume = volume
    prev_pvi = _pvi

    return _pvi


# noinspection PyUnusedLocal
@module_property
def pvt() -> PyneFloat:
    """
    Price Volume Trend.

    :return: Price Volume Trend
    """
    prev_close: Persistent[float] = na_float
    chg = close - prev_close
    res = cum((chg / prev_close) * volume)
    prev_close = close
    return res


# noinspection PyShadowingBuiltins
def range(source: Series[float], length: int) -> PyneFloat:
    """
    Returns the difference between the max and min values in a series.

    :param source: The source series
    :param length: Number of bars
    :return: The range of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if not (source == source):  # is_na_arg
        return na_float
    length = int(length)

    return highest(source, length) - lowest(source, length)


def rci(source: Series[float], length: int) -> PyneFloat:
    """
    Calculate Rank Correlation Index (RCI).

    :param source: Series of values to calculate RCI for
    :param length: Length of RCI calculation period
    :return: RCI value between -100 and 100, or na during warmup
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK
    # The slice below reads ``length`` candles of history; grow the source
    # buffer to fit it (the per-series default may be smaller). Done before the
    # warmup guard so the oldest candles are kept from the first bar on. The resize
    # is monotonic: a series ``length`` that dips low must not shrink the buffer, or
    # the slice a later increase needs would run off its end.
    if length > capacity:
        capacity = length
        max_bars_back(source, capacity)
    if not (source == source):  # is_na_arg
        return na_float

    if bar_index < length:
        return na_float

    # Collect values for performance (newest-first window)
    values = cast(list[float], source[:length])  # type: ignore

    # TV-exact pipeline (validated bit-for-bit on 333k synthetic windows across
    # lengths 2..14, 27.7k real BTCUSDT windows including ties, and dedicated
    # tie-threshold probes): values are grouped into rank-ties by ascending
    # sweep — a group collects every value closer than 1e-10 to the group's
    # smallest member (transitive, min-anchored clustering, NOT pairwise
    # epsilon comparison). Groups get 0-based average ranks (descending by
    # value), then moment-form variances (E[r^2] - mean^2), covariance from
    # deviation products, and the final (100*cov)/(sd_x*sd_y) scaling.
    # Every operation order below matters for bit-exactness — do not reorder.
    n = float(length)
    order = sorted(builtins.range(length), key=lambda k: values[k])
    ranks: list[float] = [0.0] * length
    pos = 0
    i = 0
    while i < length:
        anchor = values[order[i]]
        j = i
        while j < length and values[order[j]] - anchor < 1e-10:
            j += 1
        avg_rank = (length - 1) - (pos + pos + (j - i) - 1) / 2.0
        for t in builtins.range(i, j):
            ranks[order[t]] = avg_rank
        pos += j - i
        i = j
    sum_x = sum_y = sum_x2 = sum_y2 = 0.0
    for i in builtins.range(length):
        y = ranks[i]
        x = float(i)
        sum_x += x
        sum_y += y
        sum_x2 += x * x
        sum_y2 += y * y
    mean_x = sum_x / n
    mean_y = sum_y / n
    var_x = sum_x2 / n - mean_x * mean_x
    var_y = sum_y2 / n - mean_y * mean_y
    if var_x <= 0 or var_y <= 0:
        return na_float
    cov = 0.0
    for i in builtins.range(length):
        cov += (float(i) - mean_x) * (ranks[i] - mean_y)
    cov /= n
    return (100.0 * cov) / (math.sqrt(var_x) * math.sqrt(var_y))


# noinspection PyUnusedLocal
def rising(source: float, length: int) -> bool:
    """
    Test if the source series is now rising for length bars long.

    The rise test is tolerant: a step smaller than the float comparison tolerance
    does not count as rising, unlike ``ta.crossover`` or ``ta.highest``, which are
    exact.

    :param source: The source series
    :param length: The length of the rising test
    :return: True if the source series is rising for length bars long
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)

    last_val: Persistent[float] = na_float
    counter: Persistent[int] = 0

    if not (last_val == last_val):
        last_val = source
        return False

    # Tolerant step test, measured on TradingView (probe m547)
    if source - last_val > _EPSILON:
        counter += 1
    else:
        counter = 0

    last_val = source
    return counter >= length


def rma(source: PyneFloat, length: int) -> PyneFloat:
    """
    Calculate the RMA (Running Moving Average, Wilder's smoothing) of the source series with
    the given length.

    Seed, warmup bar and na handling are :func:`ema`'s.

    :param source: The source series
    :param length: The length of the RMA
    :return: The RMA of the source series
    """
    # Measured law (probes m558): the step is ``(prev * (length - 1) + source) / length``.
    # This is NOT ``ema`` with alpha = 1 / length -- both alpha shapes drift from
    # TradingView on the majority of bars, so rma runs its own machine.
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)
    if length == 1:  # Shortcut
        return source

    if not (source == source):  # is_na_arg
        return na_float

    last_val: Persistent[float] = na_float

    # Use SMA at warming stage
    if not (last_val == last_val):
        last_val = sma(source, length)
        return last_val

    # Warmed result
    last_val = (last_val * (length - 1) + source) / length
    return last_val


def roc(source: Series[float], length: int) -> PyneFloat:
    """
    Calculate the Rate of Change (ROC) of the source series with the given length.

    :param source: The source series
    :param length: The length of the ROC
    :return: The Rate of Change (ROC) of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if not (source == source):  # is_na_arg
        return na_float
    length = int(length)
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK
    # Grow the buffer so ``source[length]`` stays addressable for lengths beyond the
    # per-series default max_bars_back (500); otherwise it reads na and the roc is na.
    # The resize is monotonic: a series ``length`` that dips low must not shrink the
    # buffer, or the history a later increase needs would already be gone.
    if length > capacity:
        capacity = length
        max_bars_back(source, capacity)

    prev_val = source[length]
    chg = change(source, length)

    if not (prev_val == prev_val):
        return na_float

    return 100 * chg / prev_val


# noinspection PyUnusedLocal
def rsi(source: float, length: int) -> PyneFloat:
    """
    Calculate the Relative Strength Index (RSI) of the source series with the given length.

    :param source: The source series
    :param length: The length of the RSI
    :return: The Relative Strength Index (RSI) of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if not (source == source):  # is_na_arg
        return na_float

    prev_src: Persistent[float] = na_float
    if not (prev_src == prev_src):
        prev_src = source
        return na_float

    rma_u = rma(builtins.max(source - prev_src, 0.0), length)
    rma_d = rma(builtins.max(prev_src - source, 0.0), length)
    prev_src = source

    # MEASURED (probe ``rsiconst_probe``, BINANCE:BTCUSDT 30m, 28519 bars): the
    # documented ``100 - 100/(1 + rma_u/rma_d)`` is only what the reference
    # snippet computes — the native machine first ZEROES each side under the
    # language's 1e-10 absolute tolerance, and a zeroed side SATURATES the
    # result instead of dividing. A weekly ``fixnan`` source held constant for
    # days decays ``rma_d`` to ~1e-11 while ``rma_u`` stays at ~10, where the
    # quotient form returns 99.99999999915303 and TradingView returns exactly
    # 100. The down side is tested FIRST, so a doubly zeroed pair (both sides
    # dormant) is 100, not 0 and not na — measured on 322 such bars.
    if rma_d <= 1e-10:
        return 100.0
    if rma_u <= 1e-10:
        return 0.0

    return 100 - 100 / (1 + rma_u / rma_d)


# noinspection PyShadowingBuiltins,PyUnusedLocal,PyShadowingNames
def sar(start: float = 0.02, inc: float = 0.02, max: float = 0.2) -> PyneFloat:
    """
    Parabolic SAR (Stop and Reverse) - method devised by J. Welles Wilder, Jr.,
    to find potential reversals in the market price direction of traded goods.

    :param start: Starting value for acceleration factor
    :param inc: Acceleration factor increment
    :param max: Maximum acceleration factor value
    :return: SAR value for current bar
    """
    # The comparisons below stay exact. They cannot be measured directly -- the builtin
    # takes no source, so a sub-tolerance series cannot be fed into it -- but they were
    # bounded instead: over 49k bars of BTCUSDT 30m and EURUSD 1m the smallest non-zero
    # |sar - low| / |sar - high| margin was 2.7e-8, with zero margins in the (0, 1e-10)
    # band (probe m555). Exact ties do occur (72 bars) and decide the same way under
    # either rule.
    assert 0 < start <= max, "Start must be positive and not greater than max!"
    assert inc > 0, "Increment must be positive!"
    assert max <= 0.5, "Maximum cannot exceed 0.5!"

    if bar_index == 0:
        return na_float

    # Persistent states
    pos_long: Persistent[bool] = True  # Current position (long/short)
    af: Persistent[float] = start  # Current acceleration factor
    sar_val: Persistent[float] = na_float  # Current SAR value
    ep: Persistent[float] = na_float  # Extreme point

    # Unlike ``tr``, the previous bars here are read from this function's OWN window,
    # which advances per CALL -- measured (probe m571): a ta.sar() inside an `if` is na
    # on TradingView for the whole run, even when only every 100th bar is skipped,
    # while a gated ta.atr() in the same block keeps producing values. The first gated
    # call has no history yet, so ``high[1]`` is na and the recurrence below carries
    # that na forward for good, which is exactly what TradingView shows. Reading the
    # runner's global ``lib._last_*`` windows instead would invent values there.

    # Initialize on second bar
    if bar_index == 1:
        if high[1] > high:
            pos_long = False
            sar_val = high[1]  # short start
            ep = low  # EP is current low
        else:
            pos_long = True
            sar_val = low[1]  # long start
            ep = high  # EP is current high
        return sar_val

    # Calculate next SAR value
    next_sar = sar_val + af * (ep - sar_val)

    # Trend-dependent logic
    if pos_long:
        # Long trend
        if low <= next_sar:  # Reverse to short
            pos_long = False
            af = start
            next_sar = ep  # Start from previous EP (Wilder method)
            # Clip to current and previous 2 candle highs
            next_sar = builtins.max(
                next_sar,
                high,
                high[1],
                high[2] if high[2] == high[2] else high[1]
            )
            ep = low  # New EP
        else:
            # Continue long
            next_sar = builtins.min(
                next_sar,
                low[1],
                low[2] if low[2] == low[2] else low[1]
            )
            if high > ep:  # New peak
                ep = high
                af = builtins.min(af + inc, max)
    else:
        # Short trend
        if high >= next_sar:  # Reverse to long
            pos_long = True
            af = start
            next_sar = ep  # Start from previous EP (Wilder method)
            # Clip to current and previous 2 candle lows
            next_sar = builtins.min(
                next_sar,
                low,
                low[1],
                low[2] if low[2] == low[2] else low[1]
            )
            ep = high  # New EP
        else:
            # Continue short
            next_sar = builtins.max(
                next_sar,
                high[1],
                high[2] if high[2] == high[2] else high[1]
            )
            if low < ep:  # New trough
                ep = low
                af = builtins.min(af + inc, max)

    sar_val = next_sar
    return sar_val


def sma(source: Series[float], length: int) -> PyneFloat:
    """
    Calculate Simple Moving Average (SMA)

    :param source: The source series
    :param length: The length of the moving average
    :return: The Simple Moving Average (SMA)
    """
    # The divisor is the TRUNCATED length, not the argument: Pine's ``int / int``
    # keeps its fraction while staying int-typed, and ``ta.sma(close, R / 8)``
    # with R = 14 is ``ta.sma(close, 1)`` on TradingView, not a division by 1.75.
    # ``lib_math.sum`` truncates its own window the same way.
    length = int(length)
    # No decimal rounding here: ``lib_math.sum`` reproduces Pine's compensated
    # accumulator bit-for-bit, so the plain quotient IS TradingView's value.
    # Rounding to 15 decimals sits above the ulp for typical price magnitudes
    # and would perturb the last bits (measured: it breaks the match on 17451
    # of 22282 EURUSD 60m bars).
    return lib_math.sum(source, length) / length


def stdev(source: float, length: int, biased=True) -> PyneFloat:
    """
    Calculate the standard deviation of the source series with the given length.

    It is the square root of :func:`variance`, whose clamp applies here too, so
    cancellation regimes give 0.0, not na.

    :param source: The source series
    :param length: The length of the standard deviation
    :param biased: Specifies whether the biased or unbiased standard deviation is calculated
    :return: The standard deviation of the source series
    """
    # Measured (probes m556, 27.8k bars per configuration, zero mismatches):
    # TradingView's stdev is exactly sqrt of its variance on both paths.
    try:
        return math.sqrt(variance(source, length, biased))
    except TypeError:
        return na_float


# noinspection PyShadowingNames
def stoch(source: float | Series[float], high: float | Series[float], low: float | Series[float],
          length: int) -> PyneFloat:
    """
    Calculate the Stochastic Oscillator of the source series with the given length.

    The result is not clamped to ``[0, 100]``: a source that leaves the window range
    of ``high``/``low`` scales freely beyond it, and rounding can push an in-range
    source a hair above 100.

    :param source: The source series
    :param high: Series of high values
    :param low: Series of low values
    :param length: The length of the Stochastic Oscillator
    :return: The Stochastic Oscillator of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if not (source == source and high == high and low == low):
        return na_float
    length = int(length)

    highs: Series[float] = high
    lows: Series[float] = low
    hmax = highest(highs, length)
    lmin = lowest(lows, length)

    if bar_index < length - 1:
        return na_float

    # Measured law (probes m561): the bare ratio with the multiplication done
    # FIRST, and without any clamping. Feeding a source three ranges below the
    # window low and three above it, TradingView reported the full -300..+400
    # span, and every one of the 22289 bars matched ``(100 * dl) / hl`` bit for
    # bit. That leading multiplication also rounds before the division, so a
    # source sitting exactly on the window high can come out as
    # 100.00000000000001 — clamping to 100.0 was the last divergence left in
    # the stochastic RSI chain.
    return 100 * (source - lmin) / (hmax - lmin)  # type: ignore


# noinspection PyUnusedLocal,PyShadowingNames,PyPep8Naming
def supertrend(factor: float | int, atrPeriod: int) -> tuple[PyneFloat, PyneInt]:
    """
    Calculate Supertrend indicator.

    :param factor: ATR multiplier
    :param atrPeriod: ATR period length
    :return: Tuple of (supertrend value, direction). Direction: 1=down, -1=up
    """
    # The band comparisons below stay exact, bounded the same way as ``sar``: over 49k
    # bars the smallest non-zero band-to-band margin was 1.5e-9 and the smallest
    # close-to-band margin 2.0e-5, with nothing in the (0, 1e-10) band (probe m555). A
    # sub-tolerance perturbation of a band could not flip a direction decision either,
    # since those margins stay four decades above it.
    assert atrPeriod > 0, "Invalid ATR period, must be greater than 0!"

    # Store persistent state
    prev_lower: Persistent[float] = na_float
    prev_upper: Persistent[float] = na_float
    prev_close: Persistent[float] = na_float
    prev_direction: Persistent[int] = NA(int)
    prev_supertrend: Persistent[float] = na_float

    # Calculate base values
    src = hl2
    atr_val = atr(atrPeriod)

    # This is a strange bug in Pine Script, but we need to replicate it
    if bar_index == 0:
        return 0.0, 1

    if not (src == src) or not (atr_val == atr_val):
        return na_float, prev_direction if prev_direction == prev_direction else 1

    # Calculate bands
    upper = src + factor * atr_val
    lower = src - factor * atr_val

    # First value initialization
    if not (prev_direction == prev_direction):
        direction = 1
        supertrend = upper
        prev_direction = direction
        prev_supertrend = supertrend
        prev_lower = lower
        prev_upper = upper
        prev_close = close
        return supertrend, direction

    # Adjust bands based on previous values
    if lower > prev_lower or prev_close < prev_lower:
        curr_lower = lower
    else:
        curr_lower = prev_lower

    if upper < prev_upper or prev_close > prev_upper:
        curr_upper = upper
    else:
        curr_upper = prev_upper

    # Calculate direction
    if prev_supertrend == prev_upper:
        direction = -1 if close > curr_upper else 1
    else:
        direction = 1 if close < curr_lower else -1

    # Calculate supertrend value
    supertrend = curr_upper if direction == 1 else curr_lower

    # Store values for next iteration
    prev_direction = direction
    prev_supertrend = supertrend
    prev_lower = curr_lower
    prev_upper = curr_upper
    prev_close = close

    return supertrend, direction


def swma(source: Series[float]) -> PyneFloat:
    """
    Symmetrically weighted moving average with fixed length: 4. Weights: [1/6, 2/6, 2/6, 1/6].

    :param source: The source series
    :return: The SWWMA of the source series
    """
    if not (source == source):  # is_na_arg
        return na_float

    return (source + 2 * source[1] + 2 * source[2] + source[3]) / 6


# noinspection PyUnusedLocal
@module_function_property
def tr(handle_na: bool = False) -> PyneFloat:
    """
    Calculate True Range (TR)

    :param handle_na: If true, and previous day's close is NaN then tr would be calculated as
                      current day high-low. Otherwise (if false) tr would return NaN in such cases
    :return: True Range (TR)
    """
    # The runner's window, not this function's own state: TradingView reads close[1]
    # here, which advances on every bar, while a tr() inside a conditional branch runs
    # only on some of them. Measured on the "Follow Line Indicator" corpus script,
    # whose atr() calls sit in `if` branches: the accumulating rma stays per-call-site
    # and call-gated, only the previous close is global.
    prev_close = _last_close

    if not (prev_close == prev_close):
        val = (high - low) if handle_na else na_float
    else:
        val = builtins.max(high - low, abs(high - prev_close), abs(low - prev_close))

    return val  # type: ignore


def tsi(source: Series[float], short_length: int, long_length: int) -> PyneFloat:
    """
    True strength index. It uses moving averages of the underlying momentum
    of a financial instrument.

    :param source: Source series
    :param short_length: Short length
    :param long_length: Long length
    :return: True strength index between -1 and 1
    """
    assert short_length > 0, "Invalid short length, must be greater than 0!"
    assert long_length > 0, "Invalid long length, must be greater than 0!"
    if not (source == source):  # is_na_arg
        return na_float

    # Calculate momentum
    momentum = change(source)
    if not (momentum == momentum):
        return na_float

    # First smooth both momentum and abs(momentum)
    momentum_ema = ema(momentum, long_length)
    abs_momentum_ema = ema(abs(momentum), long_length)

    if not (momentum_ema == momentum_ema) or not (abs_momentum_ema == abs_momentum_ema):
        return na_float

    # Second smooth
    tsi_value = ema(momentum_ema, short_length)
    abs_value = ema(abs_momentum_ema, short_length)

    if not (abs_value == abs_value):
        return na_float

    return tsi_value / abs_value


def variance(source: Series[float],
             length: int,
             biased: bool = True) -> PyneFloat:
    """
    Calculate the rolling variance of the source series.

    The result is clamped at zero: under catastrophic cancellation the raw expression
    goes negative and comes back as 0.0. An unbiased variance with ``length == 1`` is na.

    :param source: The source series.
    :param length: The length of the rolling window.
    :param biased: If True, calculates biased variance; otherwise, calculates unbiased variance.
    :return: The variance of the source series.
    """
    # Measured law (probes m556, 27.8k bars each, synthetic full-mantissa,
    # catastrophic-cancellation and real close sources, lengths 1/2/5/9/14/20, every
    # displayed bar bit-identical): with p = math.sum(source, length),
    # q = math.sum(source * source, length) and m = p / length, TradingView computes
    #   biased:   max(0, q / length - m * m)
    #   unbiased: max(0, q / (length - 1) - p * m / (length - 1))
    # The unbiased subtraction is distributed over the division exactly as written --
    # visible whenever ``length - 1`` is not a power of two.
    assert length > 0, "Invalid length, must be > 0!"
    length = int(length)

    # Both rolling machines must advance on every bar (na bars included), so
    # they run before any early return.
    p = lib_math.sum(source, length)
    q = lib_math.sum(source * source, length)
    if not (p == p) or not (q == q):
        return na_float
    if not biased and length == 1:
        return na_float

    m = p / length
    if biased:
        var = q / length - m * m
    else:
        var = q / (length - 1) - p * m / (length - 1)
    return builtins.max(0.0, var)


def valuewhen(condition: bool, source: float, occurrence: int) -> PyneFloat:
    """
    Returns the value of the source series when the condition is true for the given occurrence.

    :param condition: The condition series
    :param source: The source series
    :param occurrence: The occurrence of the condition
    :return: The value of the source series when the condition is true for the given occurrence
    """
    # An int-typed Pine value can still carry a fraction (``int / int``); the
    # truncation happens where an integer is required — see ``_check_type``. The
    # domain check runs on the truncated value, which is the occurrence actually
    # looked up: a -0.5 IS occurrence 0, not an out-of-domain argument.
    occurrence = int(occurrence)
    assert occurrence >= 0, "Invalid occurrence, must be >= 0!"

    # The remembered value survives every bar the condition is false, so a na
    # source on such a bar must not blank the result. A na source ON a condition
    # bar is recorded verbatim: it still consumes an occurrence and is returned
    # as na. Both measured on TradingView (CAPITALCOM:EURUSD 30m, condition
    # ``bar_index % 3 == 0`` over a source na on even bars): occurrence 1 read
    # from a condition bar whose source was na came back na while occurrence 2
    # reached past it to the older value, and a always-defined source held its
    # last value across the false bars.
    values: Persistent[deque[PyneFloat]] = deque(maxlen=occurrence + 1)

    if condition:
        values.append(source)

    if len(values) == occurrence + 1:
        return values[0]
    return na_float


# noinspection PyUnusedLocal
@module_function_property
def vwap(source: Series[float] | None = None, anchor: bool | None = None,
         stdev_mult: float | None = None) -> PyneFloat | tuple[PyneFloat, PyneFloat, PyneFloat]:
    """
    Volume weighted average price.

    Referenced bare (``ta.vwap``) this is the variable form: the VWAP of ``hlc3``
    anchored to the session. Passing an explicit ``source`` selects the function
    form ``ta.vwap(source)``.

    :param source: The source series; defaults to ``hlc3`` for the bare variable form
    :param anchor: The condition that triggers the reset of VWAP calculation
    :param stdev_mult: If specified, the function will calculate the standard deviation bands based on the main VWAP
    :return: The VWAP value or tuple of (vwap, upper_band, lower_band) if stdev_mult is specified
    """
    src = hlc3 if source is None else source
    if not (src == src):
        return na_float if stdev_mult is None else (na_float, na_float, na_float)

    # Persistent variables for calculation
    sum_vol: Persistent[float] = 0.0
    sum_pv: Persistent[float] = 0.0
    sum_ppv: Persistent[float] = 0.0
    had_anchor: Persistent[bool] = False

    if anchor is None:
        anchor = session.isfirstbar

    # Reset calculations if anchor condition is met
    if anchor is not None and anchor:
        sum_vol = volume
        sum_pv = src * volume
        sum_ppv = 0.0
        had_anchor = True
    # Only accumulate after first anchor
    elif had_anchor:
        sum_vol += volume
        sum_pv += src * volume
    else:  # There was no anchor yet
        return na_float if stdev_mult is None else (na_float, na_float, na_float)

    # Calculate VWAP
    vwap_value = sum_pv / sum_vol
    if not (vwap_value == vwap_value):
        return na_float if stdev_mult is None else (na_float, na_float, na_float)

    # If stdev_mult is specified, calculate bands
    if had_anchor and stdev_mult is not None:
        sum_ppv += src * src * volume
        std = math.sqrt(builtins.max(0.0, sum_ppv / sum_vol - vwap_value * vwap_value))
        band_width = std * stdev_mult
        # Return tuple of (vwap, upper_band, lower_band)
        return vwap_value, vwap_value + band_width, vwap_value - band_width

    return vwap_value


def vwma(source: float, length: int) -> PyneFloat:
    return sma(source * volume, length) / sma(volume, length)


# noinspection PyUnusedLocal
@module_property
def wad() -> PyneFloat:
    """
    Williams Accumulation/Distribution.

    :return: Williams Accumulation/Distribution
    """
    prev_close: Persistent[float] = na_float
    true_high = builtins.max(high, prev_close)
    true_low = builtins.min(low, prev_close)
    momentum = close - prev_close
    gain = (close - true_low) if momentum > 0.0 else ((close - true_high) if momentum < 0.0 else 0.0)
    prev_close = close
    return cum(gain)


# The IDE findings here are ``@pyne`` transform artifacts: ``Persistent`` writes look
# unused because they are read on the NEXT bar, and the slice + ``oldest`` read
# on ``ff`` look ill-typed because ``Series[T]`` erases to ``T`` for the IDE.
# noinspection PyUnusedLocal,PyUnresolvedReferences,PyTypeChecker
def wma(source: Series[float], length: int) -> PyneFloat:
    """
    Calculate the Weighted Moving Average (WMA) of the source series with the given length.

    Unlike the other rolling averages this one does not compact the window: an na bar
    carries the previous value forward, so the average runs over the last ``length``
    bars, not over the last ``length`` non-na values. The na bar itself returns na.

    :param source: The source series
    :param length: The length of the WMA
    :return: The WMA of the source series
    """
    # Measured law (probes m560): Pine re-sums the whole window on every bar, OLDEST
    # first, weighting ``source[i]`` with ``length - i``. The order matters: summing
    # newest-first, or weighting with ``(length - i) * length`` the way the reference
    # pseudocode does, drifts on more than half of the bars. Being a full re-sum this
    # cannot be kept incremental -- an O(1) rolling update gives different last bits.
    # The forward-filled window was measured on scattered, consecutive and leading
    # gaps at every length.
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)

    count: Persistent[int] = 0
    last: Persistent[float] = na_float
    const_len: Persistent[int] = 0
    norm: Persistent[float] = 0.0
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK

    source_na = not (source == source)  # is_na_arg
    if not source_na:
        count += 1
        last = builtins.float(source)

    # The forward-filled window must advance on every bar, na ones included.
    ff: Series[float] = last
    # Grow the buffer so the deepest read stays addressable for lengths beyond the
    # per-series default max_bars_back (500), which would otherwise return na. The
    # resize is monotonic and floored at the series' own default: a series ``length``
    # that dips low must not shrink the buffer, or the window a later increase needs
    # would already be gone (and the oldest-first walk would run off its end).
    if length > capacity:
        capacity = length
        max_bars_back(ff, capacity)

    if source_na or count < length:
        return na_float

    # The weight sum depends only on ``length``; accumulated with the same
    # sequential loop TV runs, but only when the length changes.
    if const_len != length:
        const_len = length
        n = 0.0
        for i in builtins.range(1, length + 1):
            n = n + builtins.float(i)
        norm = n

    # Fresh oldest-first walk every bar over the raw window list: bit-identical
    # to per-element ``ff[i]`` reads (the two accumulators are independent, so
    # hoisting the weight sum does not change either sequence), but several
    # times cheaper -- the ``__getitem__`` call dominates, not the arithmetic.
    summ = 0.0
    weight = 0.0
    for y in ff[0:length].oldest:
        weight = weight + 1.0
        summ = summ + y * weight

    return summ / norm


def wpr(length: int) -> PyneFloat:
    """
    Williams %R indicator.

    :param length: Length of the indicator
    :return: Williams %R value
    """
    assert length > 0, "Invalid length, must be greater than 0!"
    length = int(length)

    if length == 1:
        return close

    hmax = highest(high, length)
    lmin = lowest(low, length)

    return 100 * (close - hmax) / (hmax - lmin)


@module_property
def wvad() -> PyneFloat:
    """
    Weighted Volume Accumulation/Distribution.

    :return: Weighted Volume Accumulation/Distribution
    """
    return (close - open) / (high - low) * volume
