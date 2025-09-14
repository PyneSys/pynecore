"""
@pyne
"""
from typing import TypeVar, cast

import builtins
import math
import heapq

from collections import deque

from ..types import Series, Persistent, NA
from ..core.module_property import module_property
from pynecore.core.overload import overload

from ..core import safe_convert

# We need to use this kind of import to make transformer work
from pynecore.lib import open, high, low, close, volume, hl2, bar_index, array, session, math as lib_math

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

EPSILON = 1e-14


def _avgrank(values: list[float], val_to_compare: float) -> float:
    """
    Helper function to calculate rank with proper tie handling

    :param values: The list of values
    :param val_to_compare: The value to compare
    :return: The average rank of the value
    """
    rank_sum = 0.0
    tie_count = 0

    # Count values less than current and ties
    for vi in values:
        diff = vi - val_to_compare
        if diff > EPSILON:
            rank_sum += 1
        elif abs(diff) < EPSILON:
            tie_count += 1

    # Return average rank for ties
    return rank_sum + (tie_count - 1) / 2.0 + 1


#
# Indicators
#

@module_property
def accdist() -> float | NA[float]:
    """
    Accumulation/Distribution index
    A/D = ((Close - Low) - (High - Close)) / (High - Low) * Volume + Previous A/D

    :return: Accumulation/Distribution index
    """
    ad: Persistent[float] = 0.0

    mfm = ((close - low) - (high - close)) / (high - low)
    mfv = mfm * volume
    if not isinstance(mfv, NA):
        ad += mfv

    return ad


def alma(source: Series[float], length: int, offset: float = 0.85, sigma: float = 6.0, floor=False) \
        -> float | Series[float] | NA[float]:
    """
    Calculate the Arnaud Legoux Moving Average (ALMA) of the source series with the given length.

    Fun fact: ALMA means "soul" in latin and Spanish, and Portugese.
              It means "apple" in Hungarian, Finnish, and Estonian.
              It means "take it" in Turkish.
              It means "water" in Arabic.
              It means "apple tree" in Georgian.
              ...

    :param source: The source series
    :param length: The length of the ALMA
    :param offset: The offset of the ALMA
    :param sigma: The sigma value of the ALMA
    :param floor:  Specifies whether the offset calculation is floored before ALMA is calculated. Default value is false
    :return: The ALMA of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA):
        return NA(float)
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
        summ += w * source[i]
    return summ / norm


def atr(length: int) -> float | NA[float]:
    """
    Calculate Average True Range (ATR) of the source series with the given length.

    :param length: The length of the ATR
    :return: The ATR of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    return rma(tr(True), length)


def barssince(condition: bool) -> int | NA[int]:
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


def bb(source: float, length: int, mult: float | int) -> tuple[float | NA[float], float | NA[float], float | NA[float]]:
    """
    Calculate the Bollinger Bands (BB) of the source series with the given length and multiplier.

    :param source: The source series
    :param length: The length of the BB
    :param mult: The multiplier of the BB
    :return: The Bollinger Bands (BB) of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    assert mult > 0, "Invalid multiplier, multiplier must be greater than 0!"

    std_dev = stdev(source, length)

    middle = sma(source, length)

    if isinstance(middle, NA):
        return NA(float), NA(float), NA(float)
    std_dev *= mult
    return middle, middle + std_dev, middle - std_dev


def bbw(source: float, length: int, mult: float | int) -> float | NA[float]:
    """
    Calculate the Bollinger Bands Width (BBW) of the source series with the given length and multiplier.

    :param source: The source series
    :param length: The length of the BBW
    :param mult: The multiplier of the BBW
    :return: The Bollinger Bands Width (BBW) of the source series
    """
    b, h, l = bb(source, length, mult)
    if isinstance(b, NA) or b == 0.0:
        return NA(float)
    return ((h - l) / b) * 100


def cci(source: float, length: int) -> float | NA[float]:
    """
    Calculate the Commodity Channel Index (CCI) of the source series with the given length.

    :param source: The source series
    :param length: The length of the CCI
    :return: The Commodity Channel Index (CCI) of the source series
    """
    mean = sma(source, length)
    mdev = dev(source, length, _mean=mean)
    if isinstance(mdev, NA):
        return NA(float)
    return (source - mean) / (0.015 * mdev)


def change(source: Series[TFIB], length: int = 1) -> TFIB | NA[TFIB]:
    """
    Calculate a simple change with respect to the given bar offset.

    :param source: The source series
    :param length: The offset in bars
    :return: The change from source to source[length]
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)

    # We need to round to prevent problems caused by floating point precision
    if isinstance(source, (float, int)):
        source = round(source, 14)
    prev_val = source[length]

    if isinstance(source, NA) or isinstance(prev_val, NA):
        return NA(cast(type[TFIB], type(source)))  # type: ignore
    if isinstance(source, (float, int)):
        diff = round(source - prev_val, 14)
        return cast(TFIB, diff)
    return source != prev_val


def cmo(source: float, length: int) -> float | NA[float]:
    """
    Calculate the Chande Momentum Oscillator (CMO) of the source series with the given length.

    :param source: The source series
    :param length: The length of the CMO
    :return: The Chande Momentum Oscillator (CMO) of the source series
    """
    momentum = change(source)
    if isinstance(momentum, NA):
        return momentum
    sum1 = lib_math.sum(momentum if momentum >= 0.0 else 0.0, length)
    sum2 = lib_math.sum(0.0 if momentum >= 0.0 else -momentum, length)
    return 100 * (sum1 - sum2) / (sum1 + sum2)


# noinspection PyUnusedLocal,PyShadowingBuiltins
def cog(source: Series[float], length: int) -> float | NA[float]:
    """
    Calculate the Center of Gravity (COG) of the source series with the given length.

    :param source: The source series
    :param length: The length of the COG
    :return: The Center of Gravity (COG) of the source series
    """
    if isinstance(source, NA):
        return NA(float)

    count: Persistent[int] = 0
    summ: Persistent[float] = 0.0
    weighted_summ: Persistent[float] = 0.0
    val: Persistent[float] = NA(float)

    # Warming up phase
    if count < length:
        count += 1
        summ += source
        weighted_summ += source * (length - count)
        if count < length:
            return NA(float)

    # Normal calculation phase
    else:
        new_summ = summ + source - source[length]
        weighted_summ = weighted_summ + summ - length * source[length]
        summ = new_summ
    val = -weighted_summ / summ - 1.0
    return val


def correlation(source1: Series[float], source2: Series[float], length: int) -> float | NA:
    """
    Calculate the correlation of the source series with the given length.

    NOTE: It is about 7 digits accurate to the result of Pine Script's correlation function. There
          are a lot of floating point operations, and even the order matters. I cannot found a
          better matching calculation.

    :param source1: The first source series
    :param source2: The second source series
    :param length: The length of the correlation
    :return: The correlation of the source series
    """
    assert length > 0, "Length must be greater than 0"
    if isinstance(source1, NA) or isinstance(source2, NA):
        return NA(float)
    length = int(length)

    sum_x: Persistent[float] = 0.0
    sum_y: Persistent[float] = 0.0
    sum_xy: Persistent[float] = 0.0
    sum_x2: Persistent[float] = 0.0
    sum_y2: Persistent[float] = 0.0
    count: Persistent[int] = 0

    if count < length:
        sum_x += source1
        sum_y += source2
        sum_xy += source1 * source2
        sum_x2 += source1 * source1
        sum_y2 += source2 * source2
        count += 1
        if count < length:
            return NA(float)
    else:
        x_toremove = source1[length]
        y_toremove = source2[length]

        sum_x += source1 - source1[length]
        sum_y += source2 - source2[length]

        sum_xy += (source1 * source2) - (x_toremove * y_toremove)
        sum_x2 += (source1 * source1) - (x_toremove * x_toremove)
        sum_y2 += (source2 * source2) - (y_toremove * y_toremove)
    try:
        numerator = (length * sum_xy) - (sum_x * sum_y)
        denominator = math.sqrt((length * sum_x2 - sum_x * sum_x) * (length * sum_y2 - sum_y * sum_y))
        return numerator / denominator
    except ValueError:
        return NA(float)


def cross(source1: float, source2: float) -> bool | NA[bool]:
    """
    Check if the source series crossed over or under the given series.

    :param source1: The first source series
    :param source2: The second source series
    :return: True if the source series crossed over the given series
    """
    return crossover(source1, source2) or crossunder(source1, source2)


# noinspection PyUnusedLocal
def crossover(source1: float, source2: float) -> bool | NA[bool]:
    """
    Check if the source series crossed over the given series.

    :param source1: The first source series
    :param source2: The second source series
    :return: True if the source series crossed over the given series
    """
    l1_lte_l2: Persistent[bool] = NA(bool)
    res = source1 > source2 and l1_lte_l2
    l1_lte_l2 = source1 <= source2
    return res


# noinspection PyUnusedLocal
def crossunder(source1: float, source2: float) -> bool | NA[bool]:
    """
    Check if the source series crossed under the given series.

    :param source1: The first source series
    :param source2: The second source series
    :return: True if the source series crossed under the given series
    """
    l1_gte_l2: Persistent[bool] = NA(bool)
    res = source1 < source2 and l1_gte_l2
    l1_gte_l2 = source1 >= source2
    return res


def cum(source: Series[float | int]) -> float | NA:
    """
    Calculate the cumulative sum of the source series.

    :param source: The source series
    :return: The cumulative sum of the source series
    """
    if isinstance(source, NA):
        return NA(float)
    var: Persistent[float] = 0.0
    var += source
    return var


def dev(source: Series[float], length: int, _mean: float | None = None) -> float | NA:
    """
    Calculate the Mean Absolute Deviation (MAD) of the source series with the given length.

    :param source: The source series
    :param length: The length of the MAD calculation
    :param _mean: The mean value of the source series, if it is already calculated
    :return: The mean absolute deviation of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if length == 1:
        return 0.0
    length = int(length)

    mean = _mean if _mean is not None else sma(source, length)
    if isinstance(mean, NA):
        return NA(float)

    summ = 0.0
    for i in builtins.range(length):
        summ += abs(source[i] - mean)

    return summ / length


# noinspection PyPep8Naming
def dmi(diLength: int, adxSmoothing: int) -> tuple[float | NA, float | NA, float | NA]:
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
    if isinstance(up, NA) or isinstance(down, NA):
        return NA(float), NA(float), NA(float)
    a = atr(diLength)
    plus_dm = up if (up > down and up > 0.0) else 0.0
    minus_dm = down if (down > up and down > 0.0) else 0.0
    p = rma(plus_dm, diLength)
    m = rma(minus_dm, diLength)
    if isinstance(a, NA) or isinstance(p, NA) or isinstance(m, NA) or a == 0.0:
        return NA(float), NA(float), NA(float)
    p = 100 * p / a
    m = 100 * m / a
    summ = p + m
    adx = rma(abs(p - m) / (summ if summ != 0.0 else 1.0), adxSmoothing) * 100
    return p, m, adx


def ema(source: float, length: int, _alpha: float | None = None) -> float | NA[float]:
    """
    Calculate the Exponential Moving Average (EMA) of the source series with the given length.

    :param source: The source series
    :param length: The length of the EMA
    :param _alpha: The alpha value for EMA calculation (it is a private argument)
    :return:
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)
    if length == 1:  # Shortcut
        return source

    if isinstance(source, NA):
        return NA(float)

    alpha: Persistent[float] = _alpha or (2 / (length + 1))
    last_val: Persistent[float | NA] = NA(float)

    # Use SMA at warming stage
    if isinstance(last_val, NA):
        last_val = sma(source, length)
        return cast(float | NA[float], last_val)

    # Warmed result
    last_val = alpha * source + (1 - alpha) * last_val
    return last_val


# noinspection PyUnusedLocal
def falling(source: float, length: int) -> bool:
    """
    Test if the source series is now falling for length bars long.

    :param source: The source series
    :param length: The length of the falling test
    :return: True if the source series is falling for length bars long
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)

    last_val: Persistent[float | NA[float]] = NA(float)
    counter: Persistent[int] = 0

    if isinstance(last_val, NA):
        last_val = source
        return False

    if source < last_val:
        counter += 1
    else:
        counter = 0

    last_val = source
    return counter >= length


# noinspection PyUnusedLocal
@overload
def highest(source: Series[float], length: int, _bars: bool = False, _tuple: bool = False, _check_eq: bool = False) \
        -> float | tuple[float | NA[float], float | NA[float]] | NA[float]:
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
    last_max: Persistent[float | NA] = NA(float)
    last_max_index: Persistent[int] = 0

    if last_max < source or isinstance(last_max, NA) or (_check_eq and last_max == source):
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
    last_max_index += 1

    if bar_index < length - 1:
        return NA(float) if not _tuple else (NA(float), NA(float))

    if _bars:
        return -max_index
    if _tuple:
        return cast(float | tuple[float | NA[float], float | NA[float]], (last_max, -max_index))
    return cast(float | NA[float], last_max)


@overload
def highest(length: int) -> float | NA[float]:
    return highest(high, length)


# noinspection PyUnusedLocal
@overload
def highestbars(source: Series[float], length: int) -> float | NA[float]:
    """
    Calculate the number of bars since the highest value of the source series with the given length.

    :param source: The source series
    :param length: The length of the highest value
    :return: The number of bars since the highest value of the source series
    """
    return highest(source, length, _bars=True)


@overload
def highestbars(length: int) -> float | NA[float]:
    return highest(high, length, _bars=True)


def hma(source: float, length: int) -> float | NA[float]:
    """
    Calculate the Hull Moving Average (HMA) of the source series with the given length.

    :param source: The source series
    :param length: The length of the HMA
    :return: The Hull Moving Average (HMA) of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA):
        return NA(float)
    length = int(length)

    ma_np2 = wma(source, length // 2)
    ma = wma(source, length)
    if isinstance(ma, NA) or isinstance(ma_np2, NA):
        return NA(float)
    return wma(2 * ma_np2 - ma, int(length ** 0.5))


@module_property
def iii() -> float | Series[float] | NA[float]:
    """
    Intraday Intensity Index.

    :return: Intraday Intensity Index
    """
    return (2 * close - high - low) / ((high - low) * volume)


# noinspection PyPep8Naming
def kc(series: float, length: int, mult: float | int, useTrueRange: bool = True) \
        -> tuple[float | NA[float], float | NA[float], float | NA[float]]:
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
    if isinstance(base, NA):
        return NA(float), NA(float), NA(float)
    if isinstance(range_ma, NA):
        return base, NA(float), NA(float)
    range_ma *= mult
    return base, base + range_ma, base - range_ma


# noinspection PyPep8Naming
def kcw(series: float, length: int, mult: float | int, useTrueRange: bool = True) -> float | NA[float]:
    """
    Calculate the Keltner Channels Width (KCW) of the source series with the given length and multiplier.

    :param series: The source series
    :param length: The length of the KCW
    :param mult: The multiplier of the KCW
    :param useTrueRange: Specifies whether to use True Range for KCW calculation
    :return: The Keltner Channels Width (KCW) of the source series
    """
    b, h, l = kc(series, length, mult, useTrueRange)
    if isinstance(b, NA) or b == 0.0:
        return NA(float)
    return (h - l) / b


def linreg(source: Series[float], length: int, offset: int) -> float | Series[float] | NA[float]:
    """
    Computes the linear regression value of the source series over a given period.

    :param source: Input series
    :param length: Number of bars to calculate regression
    :param offset: Number of bars to shift the result
    :return: Linear regression value
    """
    if isinstance(source, NA):
        return NA(float)

    assert length > 0, "Invalid length, must be greater than 0!"
    if length == 1:
        return source
    length = int(length)
    window_size = length

    # Precomputed constants for x-coordinates
    sum_x = window_size * (window_size - 1) / 2.0
    sum_x2 = (window_size - 1) * window_size * (2 * window_size - 1) / 6.0

    # Persistent state variables
    bar_count: Persistent[int] = 0
    sum_y: Persistent[float] = 0.0  # Sum of source values in the window
    sum_xy: Persistent[float] = 0.0  # Weighted sum: sum((window_size - 1 - i) * source[i])

    # Warm-up phase: accumulate values until the window is full
    if bar_count < window_size:
        prev_sum_y = sum_y
        sum_y = prev_sum_y + source
        sum_xy = (window_size - 1) * source + sum_xy - prev_sum_y
        bar_count += 1

        # Return NA until we have enough data
        if bar_count < window_size:
            return NA(float)
    else:
        # Rolling update: remove the oldest value when the window is full
        dropped_value = source[window_size]
        prev_sum_y = sum_y
        sum_y = prev_sum_y + source - dropped_value
        sum_xy = (window_size - 1) * source + sum_xy - prev_sum_y + dropped_value

    # Compute slope and intercept
    denominator = window_size * sum_x2 - sum_x * sum_x
    slope = (window_size * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / window_size

    # Compute final regression value
    return intercept + slope * ((window_size - 1) - offset)


# noinspection PyUnusedLocal
@overload
def lowest(source: Series[float], length: int,
           _bars: bool = False, _tuple: bool = False, _check_eq: bool = False) \
        -> float | tuple[float | NA[float], float | NA[float]] | NA[float]:
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
    last_min: Persistent[float | NA[float]] = NA(float)
    last_min_index: Persistent[int] = 0

    if last_min > source or isinstance(last_min, NA) or (_check_eq and last_min == source):
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
    last_min_index += 1

    if bar_index < length - 1:
        return NA(float) if not _tuple else (NA(float), NA(int))

    if _bars:
        return -min_index
    if _tuple:
        return cast(float | tuple[float | NA[float], float | NA[float]], (last_min, -min_index))
    return cast(float | NA[float], last_min)


@overload
def lowest(length: int) -> float | NA:
    return lowest(low, length)


# noinspection PyUnusedLocal
@overload
def lowestbars(source: Series[float], length: int) -> float | NA[float]:
    """
    Calculate the number of bars since the lowest value of the source series with the given length.

    :param source: The source series
    :param length: The length of the lowest value
    :return: The number of bars since the lowest value of the source series
    """
    return lowest(source, length, _bars=True)


@overload
def lowestbars(length: int) -> float | NA[float]:
    return lowest(low, length, _bars=True)


def macd(source: float, fastlen: int, slowlen: int, siglen: int) \
        -> tuple[float | NA[float], float | NA[float], float | NA[float]]:
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
    if isinstance(fast, NA) or isinstance(slow, NA):
        return NA(float), NA(float), NA(float)
    macd_val = fast - slow
    signal = ema(macd_val, siglen)
    if isinstance(signal, NA):
        return macd_val, NA(float), NA(float)
    return macd_val, signal, macd_val - signal


# noinspection PyShadowingBuiltins
def max(source: Series[float]) -> float | NA[float]:
    """
    Calculate the maximum value of the source series.

    :param source: The source series
    :return: The maximum value of the source series
    """
    max_val: Persistent[float | NA] = NA(float)
    if max_val < source or isinstance(max_val, NA):
        max_val = source
    return cast(float | NA[float], max_val)


def median(source: Series[TFI], length: int) -> TFI | NA[TFI] | Series[TFI]:
    """
    Calculate the median of the source series over a given period.

    :param source: Input series of values
    :param length: Number of bars to calculate over
    :return: The median value or na during warmup
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if length == 1:  # Shortcut
        return source
    length = int(length)

    if isinstance(source, NA):
        return NA(cast(type[TFI], type(source)))  # type: ignore

    # Store heaps and window
    heap_low: Persistent[list[TFI]] = []  # Max heap (negative values)
    heap_high: Persistent[list[TFI]] = []  # Min heap
    window: Persistent[list[TFI]] = []  # Recent values for removal

    # Add new value and balance heaps
    value = source
    window.append(cast(TFI, value))
    heapq.heappush(heap_low, -cast(TFI, value))
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


def mfi(source: float, length: int) -> float | NA[float]:
    """
    Calculate the Money Flow Index (MFI) of the source series with the given length.

    :param source: The source series
    :param length: The length of the MFI
    :return: The Money Flow Index (MFI) of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA):
        return NA(float)
    length = int(length)

    chg = change(source)
    upper = lib_math.sum(volume * (0.0 if not isinstance(chg, NA) and chg <= 0 else source), length)
    lower = lib_math.sum(volume * (0.0 if not isinstance(chg, NA) and chg >= 0 else source), length)
    if isinstance(upper, NA) or isinstance(lower, NA):
        return NA(float)
    return 100.0 - (100 * lower / (upper + lower))


# noinspection PyShadowingBuiltins
def min(source: Series[float]) -> float | NA:
    """
    Calculate the minimum value of the source series.

    :param source: The source series
    :return: The minimum value of the source series
    """
    min_val: Persistent[float | NA] = NA(float)
    if min_val > source or isinstance(min_val, NA):
        min_val = source
    return cast(float | NA[float], min_val)


def mode(source: Series[TFI], length: int) -> TFI | NA:
    """
    Returns the mode of the series. If there are several values with the same frequency,
    it returns the smallest value.

    :param source: Series of values to process
    :param length: Number of bars (length)
    :return: The most frequently occurring value from the source. If none exists, returns
             the smallest value instead. Returns na during warm-up period.
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA) or bar_index < length - 1:
        return NA(float)
    length = int(length)

    # Store values for quick access
    values = [source[i] for i in builtins.range(length) if source[i] is not NA(float)]
    if not values:
        return NA(float)

    # Find mode - sort values to handle equal frequencies
    values.sort()  # Ensure we pick smallest value when frequencies are equal
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


def mom(source: float, length: int) -> float | NA:
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
def nvi() -> float | NA[float] | Series[float]:
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
def obv() -> float | NA[float]:
    """
    On Balance Volume.

    :return: On Balance Volume
    """
    chg = change(close)
    if isinstance(chg, NA):
        return NA(float)
    if chg > 0:
        chg = 1.0
    elif chg < 0:
        chg = -1.0
    else:
        chg = 0.0
    return cum(volume * chg)


def percentile_linear_interpolation(source: Series[float], length: int, percentage: int | float) \
        -> float | NA[float] | Series[float]:
    """
    Calculates percentile using method of linear interpolation between the two nearest ranks.

    :param source: The source series
    :param length: The length of the percentile
    :param percentage: The percentage of the percentile
    :return: The percentile of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA):
        return NA(float)
    length = int(length)

    if bar_index < length - 1:
        return NA(float)

    return array.percentile_linear_interpolation(source[:length], percentage)  # type: ignore


def percentile_nearest_rank(source: Series[float], length: int, percentage: int | float) \
        -> float | NA[float] | Series[float]:
    """
    Calculates percentile using the nearest rank method.

    :param source: The source series
    :param length: The length of the percentile
    :param percentage: The percentage of the percentile
    :return: The percentile of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA):
        return NA(float)
    length = int(length)

    if bar_index < length - 1:
        return NA(float)

    return array.percentile_nearest_rank(source[:length], percentage)  # type: ignore


def percentrank(source: Series[float], length: int) -> float | NA[float] | Series[float]:
    """
    Percent rank is the percents of how many previous values was less than or equal to the current
    value of given series.

    :param source: The source series
    :param length: Number of bars back to include in the calculation
    :return: The percentage of values less than or equal to the current value
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA) or bar_index < length:
        return NA(float)
    length = int(length)

    return array.percentrank(source[:length + 1], 0)  # type: ignore


@overload
def pivothigh(source: float, leftbars: int, rightbars: int) -> float | NA[float]:
    """
    This function returns price of the pivot high point. It returns 'NaN', if there was no pivot high point.

    :param source: The source series
    :param leftbars: Left strength
    :param rightbars: Right strength.
    :return: Price of the pivot high point, or NaN if no pivot
    """
    assert leftbars > 0, "Invalid leftbars, leftbars must be greater than 0!"
    assert rightbars > 0, "Invalid rightbars, rightbars must be greater than 0!"

    if isinstance(source, NA):
        return NA(float)

    pivotrange = leftbars + rightbars + 1
    ph, pi = cast(tuple[float, int], highest(source, pivotrange, _tuple=True, _check_eq=True))

    if pi == -rightbars:
        return ph

    return NA(float)


@overload
def pivothigh(leftbars: int, rightbars: int) -> float | NA[float]:
    """
    This function returns price of the pivot high point. It returns 'NaN', if there was no pivot high point.

    :param leftbars: Left strength
    :param rightbars: Right strength.
    :return: Price of the pivot high point, or NaN if no pivot
    """
    try:
        return pivothigh(safe_convert.safe_float(high), leftbars, rightbars)  # type: ignore
    except TypeError:
        if isinstance(high, NA):
            return NA(float)
        else:
            raise


@overload
def pivotlow(source: float, leftbars: int, rightbars: int) -> float | NA[float]:
    """
    This function returns price of the pivot low point. It returns 'NaN', if there was no pivot low point.

    :param source: The source series
    :param leftbars: Left strength
    :param rightbars: Right strength.
    :return: Price of the pivot low point, or NaN if no pivot
    """
    assert leftbars > 0, "Invalid leftbars, leftbars must be greater than 0!"
    assert rightbars > 0, "Invalid rightbars, rightbars must be greater than 0!"

    if isinstance(source, NA):
        return NA(float)

    pivotrange = leftbars + rightbars + 1
    pl, pi = cast(tuple[float, int], lowest(source, pivotrange, _tuple=True, _check_eq=True))
    if pi == -rightbars:
        return pl

    return NA(float)


@overload
def pivotlow(leftbars: int, rightbars: int) -> float | NA[float]:
    """
    This function returns price of the pivot low point. It returns 'NaN', if there was no pivot low point.

    :param leftbars: Left strength
    :param rightbars: Right strength.
    :return: Price of the pivot low point, or NaN if no pivot
    """
    try:
        return pivotlow(safe_convert.safe_float(low), leftbars, rightbars)  # type: ignore
    except TypeError:
        if isinstance(low, NA):
            return NA(float)
        else:
            raise


# noinspection PyUnusedLocal
@module_property
def pvi() -> float | NA[float] | Series[float]:
    """
    Positive Volume Index.

    :return: Positive Volume Index
    """
    prev_close: Persistent[float] = 0.0
    prev_volume: Persistent[float] = 0.0
    prev_pvi: Persistent[float] = 1.0

    _pvi = prev_pvi + ((close - prev_close) / prev_close) * prev_pvi if volume > prev_volume else prev_pvi
    if isinstance(_pvi, NA):
        _pvi = prev_pvi

    prev_close = close
    prev_volume = volume
    prev_pvi = _pvi

    return _pvi


# noinspection PyUnusedLocal
@module_property
def pvt() -> float | NA[float]:
    """
    Price Volume Trend.

    :return: Price Volume Trend
    """
    prev_close: Persistent[float] = NA(float)
    chg = close - prev_close
    res = cum((chg / prev_close) * volume)
    prev_close = close
    return res


# noinspection PyShadowingBuiltins
def range(source: Series[float], length: int) -> float | NA[float]:
    """
    Returns the difference between the max and min values in a series.

    :param source: The source series
    :param length: Number of bars
    :return: The range of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA):
        return NA(float)
    length = int(length)

    return highest(source, length) - lowest(source, length)


def rci(source: Series[float], length: int) -> float | NA[float]:
    """
    Calculate Rank Correlation Index (RCI).

    :param source: Series of values to calculate RCI for
    :param length: Length of RCI calculation period
    :return: RCI value between -100 and 100, or na during warmup
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)

    if isinstance(source, NA) or bar_index < length:
        return NA(float)

    # Collect values for performance
    try:
        values = cast(list[float], source[:length])  # type: ignore
        # log.warning("values: {0}", values)
    except IndexError:
        return NA(float)

    # Calculate sums for correlation
    sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = 0.0

    for i in builtins.range(length):
        x = i + 1  # Time rank (newest value gets highest rank)
        y = _avgrank(values, values[i])  # Data rank  # type: ignore

        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x
        sum_y2 += y * y

    # Calculate correlation coefficient
    n = length
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return (numerator / denominator) * 100


# noinspection PyUnusedLocal
def rising(source: float, length: int) -> bool:
    """
    Test if the source series is now rising for length bars long.

    :param source: The source series
    :param length: The length of the rising test
    :return: True if the source series is rising for length bars long
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)

    last_val: Persistent[float | NA] = NA(float)
    counter: Persistent[int] = 0

    if isinstance(last_val, NA):
        last_val = source
        return False

    if source > last_val:
        counter += 1
    else:
        counter = 0

    last_val = source
    return counter >= length


def rma(source: float, length: int) -> float | NA[float]:
    """
    Calculate the RMA (Running Moving Average) of the source series with the given length.

    :param source: The source series
    :param length: The length of the RMA
    :return: The RMA of the source series
    """
    return ema(source, length, 1 / length)


def roc(source: Series[float], length: int) -> float | NA[float]:
    """
    Calculate the Rate of Change (ROC) of the source series with the given length.

    :param source: The source series
    :param length: The length of the ROC
    :return: The Rate of Change (ROC) of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA):
        return NA(float)
    length = int(length)

    prev_val = source[length]
    chg = change(source, length)

    if isinstance(prev_val, NA):
        return NA(float)

    return 100 * chg / prev_val


# noinspection PyUnusedLocal
def rsi(source: float, length: int) -> float | NA[float]:
    """
    Calculate the Relative Strength Index (RSI) of the source series with the given length.

    :param source: The source series
    :param length: The length of the RSI
    :return: The Relative Strength Index (RSI) of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA):
        return NA(float)

    prev_src: Persistent[float | NA[float]] = NA(float)
    if isinstance(prev_src, NA):
        prev_src = source
        return NA(float)

    rma_u = rma(builtins.max(source - prev_src, 0.0), length)
    rma_d = rma(builtins.max(prev_src - source, 0.0), length)
    prev_src = source

    return 100 - 100 / (1 + rma_u / rma_d)


# noinspection PyShadowingBuiltins,PyUnusedLocal,PyShadowingNames
def sar(start: float = 0.02, inc: float = 0.02, max: float = 0.2) -> float | NA[float]:
    """
    Parabolic SAR (Stop and Reverse) - method devised by J. Welles Wilder, Jr.,
    to find potential reversals in the market price direction of traded goods.

    :param start: Starting value for acceleration factor
    :param inc: Acceleration factor increment
    :param max: Maximum acceleration factor value
    :return: SAR value for current bar
    """
    assert 0 < start <= max, "Start must be positive and not greater than max!"
    assert inc > 0, "Increment must be positive!"
    assert max <= 0.5, "Maximum cannot exceed 0.5!"

    if bar_index == 0:
        return NA(float)

    # Persistent states
    pos_long: Persistent[bool] = True  # Current position (long/short)
    af: Persistent[float] = start  # Current acceleration factor
    sar_val: Persistent[float] = NA(float)  # Current SAR value
    ep: Persistent[float] = NA(float)  # Extreme point

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
                high[2] if not isinstance(high[2], NA) else high[1]
            )
            ep = low  # New EP
        else:
            # Continue long
            next_sar = builtins.min(
                next_sar,
                low[1],
                low[2] if not isinstance(low[2], NA) else low[1]
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
                low[2] if not isinstance(low[2], NA) else low[1]
            )
            ep = high  # New EP
        else:
            # Continue short
            next_sar = builtins.max(
                next_sar,
                high[1],
                high[2] if not isinstance(high[2], NA) else high[1]
            )
            if low < ep:  # New trough
                ep = low
                af = builtins.min(af + inc, max)

    sar_val = next_sar
    return cast(float | NA[float], sar_val)


def sma(source: Series[float], length: int) -> float | NA[float]:
    """
    Calculate Simple Moving Average (SMA)

    :param source: The source series
    :param length: The length of the moving average
    :return: The Simple Moving Average (SMA)
    """
    # Round is necessary to solve precision issues
    return round(lib_math.sum(source, length) / length, 15)


def stdev(source: float, length: int, biased=True) -> float | NA[float]:
    """
    Calculate the standard deviation of the source series with the given length.

    :param source: The source series
    :param length: The length of the standard deviation
    :param biased: Specifies whether the biased or unbiased standard deviation is calculated
    :return: The standard deviation of the source series
    """
    try:
        return math.sqrt(variance(source, length, biased))
    except TypeError:
        return NA(float)


# noinspection PyShadowingNames
def stoch(source: float | Series[float], high: float | Series[float], low: float | Series[float],
          length: int) -> float | NA[float]:
    """
    Calculate the Stochastic Oscillator of the source series with the given length.

    :param source: The source series
    :param high: Series of high values
    :param low: Series of low values
    :param length: The length of the Stochastic Oscillator
    :return: The Stochastic Oscillator of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA) or isinstance(high, NA) or isinstance(low, NA):
        return NA(float)
    length = int(length)

    highs: Series[float] = high
    lows: Series[float] = low
    hmax = highest(highs, length)
    lmin = lowest(lows, length)

    if bar_index < length - 1:
        return NA(float)

    dl_diff = source - lmin
    hl_diff = hmax - lmin
    if dl_diff < 0.0:
        k = 0.0
    else:
        k = 100 * dl_diff / hl_diff
        k = 100.0 if k > 100.0 else 0.0 if k < 0.0 else k
    return k  # type: ignore


# noinspection PyUnusedLocal,PyShadowingNames
def supertrend(factor: float | int, atr_period: int) -> tuple[float | NA, int | NA]:
    """
    Calculate Supertrend indicator.

    :param factor: ATR multiplier
    :param atr_period: ATR period length
    :return: Tuple of (supertrend value, direction). Direction: 1=down, -1=up
    """
    assert atr_period > 0, "Invalid ATR period, must be greater than 0!"

    # Store persistent state
    prev_lower: Persistent[float | NA[float]] = NA(float)
    prev_upper: Persistent[float | NA[float]] = NA(float)
    prev_close: Persistent[float | NA[float]] = NA(float)
    prev_direction: Persistent[int | NA[int]] = NA(int)
    prev_supertrend: Persistent[float | NA[float]] = NA(float)

    # Calculate base values
    src = hl2
    atr_val = atr(atr_period)

    # This is a strange bug in Pine Script, but we need to replicate it
    if bar_index == 0:
        return 0.0, 1

    if isinstance(src, NA) or isinstance(atr_val, NA):
        return NA(float), prev_direction if not isinstance(prev_direction, NA) else 1

    # Calculate bands
    upper = src + factor * atr_val
    lower = src - factor * atr_val

    # First value initialization
    if isinstance(prev_direction, NA):
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


def swma(source: Series[float]) -> float | NA[float] | Series[float]:
    """
    Symmetrically weighted moving average with fixed length: 4. Weights: [1/6, 2/6, 2/6, 1/6].

    :param source: The source series
    :return: The SWWMA of the source series
    """
    if isinstance(source, NA):
        return NA(float)

    return (source + 2 * source[1] + 2 * source[2] + source[3]) / 6


# noinspection PyUnusedLocal
@module_property
def tr(handle_na: bool = False) -> float | NA[float] | Series[float]:
    """
    Calculate True Range (TR)

    :param handle_na: If true, and previous day's close is NaN then tr would be calculated as
                      current day high-low. Otherwise (if false) tr would return NaN in such cases
    :return: True Range (TR)
    """
    prev_close: Persistent[float | NA[float]] = NA(float)

    if isinstance(prev_close, NA):
        val = (high - low) if handle_na else NA(float)
    else:
        val = builtins.max(high - low, abs(high - prev_close), abs(low - prev_close))

    prev_close = close
    return val  # type: ignore


def tsi(source: Series[float], short_length: int, long_length: int) -> float | NA[float]:
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
    if isinstance(source, NA):
        return NA(float)

    # Calculate momentum
    momentum = change(source)
    if isinstance(momentum, NA):
        return NA(float)

    # First smooth both momentum and abs(momentum)
    momentum_ema = ema(momentum, long_length)
    abs_momentum_ema = ema(abs(momentum), long_length)

    if isinstance(momentum_ema, NA) or isinstance(abs_momentum_ema, NA):
        return NA(float)

    # Second smooth
    tsi_value = ema(momentum_ema, short_length)
    abs_value = ema(abs_momentum_ema, short_length)

    if isinstance(abs_value, NA):
        return NA(float)

    return tsi_value / abs_value


def variance(source: Series[float],
             length: int,
             biased: bool = True) -> float | NA[float]:
    """
    Calculate the rolling variance of the source series.

    :param source: The source series.
    :param length: The length of the rolling window.
    :param biased: If True, calculates biased variance; otherwise, calculates unbiased variance.
    :return: The variance of the source series.
    """
    assert length > 0, "Invalid length, must be > 0!"
    length = int(length)
    if length == 1:
        return 0.0
    if isinstance(source, NA):
        return NA(float)

    count: Persistent[int] = 0

    sum_val: Persistent[float] = 0.0
    sum_val_c: Persistent[float] = 0.0
    sum_sq: Persistent[float] = 0.0
    sum_sq_c: Persistent[float] = 0.0

    # Always add new value with Kahan summation
    y = source - sum_val_c
    t = sum_val + y
    sum_val_c = (t - sum_val) - y
    sum_val = t

    # Kahan summation for squared value
    sq = source * source
    y = sq - sum_sq_c
    t = sum_sq + y
    sum_sq_c = (t - sum_sq) - y
    sum_sq = t

    if count < length:
        count += 1
        if count < length:
            return NA(float)
    else:
        count += 1

        # Remove old value with Kahan summation
        old_value = source[length]
        y = -old_value - sum_val_c
        t = sum_val + y
        sum_val_c = (t - sum_val) - y  # noqa - it is persistent
        sum_val = t

        # Remove old squared value with Kahan summation
        sq_old = old_value * old_value
        y = -sq_old - sum_sq_c
        t = sum_sq + y
        sum_sq_c = (t - sum_sq) - y  # noqa - it is persistent
        sum_sq = t

    # Calculate variance
    if biased:
        # Biased variance: divide by n
        mean_val = sum_val / length
        squares = sum_sq / length
        var = squares - mean_val * mean_val
    else:
        # Unbiased variance: divide by n-1
        mean_val = sum_val / length
        squares = sum_sq / (length - 1)
        var = squares - (length / (length - 1)) * mean_val * mean_val

    return var


def valuewhen(condition: bool, source: float, occurrence: int) -> float | NA[float]:
    """
    Returns the value of the source series when the condition is true for the given occurrence.

    :param condition: The condition series
    :param source: The source series
    :param occurrence: The occurrence of the condition
    :return: The value of the source series when the condition is true for the given occurrence
    """
    assert occurrence >= 0, "Invalid occurrence, must be >= 0!"
    if isinstance(source, NA):
        return NA(float)

    values: Persistent[deque[float]] = deque(maxlen=occurrence + 1)

    if condition:
        values.append(source)

    if len(values) == occurrence + 1:
        return values[0]
    return NA(float)


# noinspection PyUnusedLocal
def vwap(source: Series[float], anchor: bool | None = None, stdev_mult: float | None = None) -> \
        float | NA | tuple[float | NA, float | NA, float | NA]:
    """
    Volume weighted average price.

    :param source: The source series
    :param anchor: The condition that triggers the reset of VWAP calculation
    :param stdev_mult: If specified, the function will calculate the standard deviation bands based on the main VWAP
    :return: The VWAP value or tuple of (vwap, upper_band, lower_band) if stdev_mult is specified
    """
    if isinstance(source, NA):
        return NA(float) if stdev_mult is None else (NA(float), NA(float), NA(float))

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
        sum_pv = source * volume
        sum_ppv = 0.0
        had_anchor = True
    # Only accumulate after first anchor
    elif had_anchor:
        sum_vol += volume
        sum_pv += source * volume
    else:  # There was no anchor yet
        return NA(float) if stdev_mult is None else (NA(float), NA(float), NA(float))

    # Calculate VWAP
    vwap_value = sum_pv / sum_vol
    if isinstance(vwap_value, NA):
        return NA(float) if stdev_mult is None else (NA(float), NA(float), NA(float))

    # If stdev_mult is specified, calculate bands
    if had_anchor and stdev_mult is not None:
        sum_ppv += source * source * volume
        std = math.sqrt(builtins.max(0.0, sum_ppv / sum_vol - vwap_value * vwap_value))
        band_width = std * stdev_mult
        # Return tuple of (vwap, upper_band, lower_band)
        return vwap_value, vwap_value + band_width, vwap_value - band_width

    return vwap_value


def vwma(source: float, length: int) -> float | NA[float]:
    return sma(source * volume, length) / sma(volume, length)


# noinspection PyUnusedLocal
@module_property
def wad() -> float | NA[float]:
    """
    Williams Accumulation/Distribution.

    :return: Williams Accumulation/Distribution
    """
    prev_close: Persistent[float | NA[float]] = NA(float)
    true_high = builtins.max(high, prev_close)
    true_low = builtins.min(low, prev_close)
    momentum = close - prev_close
    gain = (close - true_low) if momentum > 0.0 else ((close - true_high) if momentum < 0.0 else 0.0)
    prev_close = close
    return cum(gain)


def wma(source: Series[float], length: int) -> float | NA[float]:
    """
    Calculate the Weighted Moving Average (WMA) of the source series with the given length.

    :param source: The source series
    :param length: The length of the WMA
    :return: The WMA of the source series
    """
    assert length > 0, "Invalid length, length must be greater than 0!"
    if isinstance(source, NA):
        return NA(float)
    length = int(length)

    # Calculate denominator only once
    denom: Persistent[float] = length * (length + 1) / 2

    count: Persistent[int] = 0
    summ: Persistent[float] = 0.0
    weighted_summ: Persistent[float] = 0.0

    # Warming up phase
    if count < length:
        count += 1
        summ += source
        weighted_summ += source * count
        if count < length:
            return NA(float)

    # Normal calculation phase
    else:
        old_summ = summ
        # Substract the oldest value and add the newest value
        summ -= source[length] - source
        # Substract the oldest weighted value and add the newest weighted value
        weighted_summ -= old_summ - length * source

    val = weighted_summ / denom
    return val  # type: ignore


def wpr(length: int) -> float | NA[float] | Series[float]:
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
def wvad() -> float | NA[float] | Series[float]:
    """
    Weighted Volume Accumulation/Distribution.

    :return: Weighted Volume Accumulation/Distribution
    """
    return (close - open) / (high - low) * volume
