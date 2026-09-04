from typing import TypeVar, cast, overload
import builtins
import math

from ..core import fdlibm, pine_math
from ..types.na import NA, na_float, na_int
from ..types import PyneFloat, PyneInt

from . import syminfo
from ._math_stateful import random, sum

TFI = TypeVar('TFI', float, int)

__all__ = [
    'e', 'pi', 'phi', 'rphi',
    'abs', 'acos', 'asin', 'atan', 'avg', 'ceil', 'cos', 'exp', 'floor',
    'log', 'log10', 'max', 'min', 'pow', 'random', 'round', 'round_to_mintick',
    'sign', 'sin', 'sqrt', 'sum', 'tan', 'todegrees', 'toradians'
]

# Constants
e = math.e
pi = math.pi
phi = (1 + math.sqrt(5)) / 2
rphi = 1 / phi

# `0.5 - 1e-10` as an exact fraction, the tie threshold round() compares against
_ROUND_TIE_SCALE = 10 ** 10
_ROUND_TIE_HALF = 5 * 10 ** 9 - 1
# round() uses a precision above this as this
_ROUND_MAX_PRECISION = 16.0
# From here up every double is an integer
_ROUND_INT_LIMIT = 2.0 ** 52

# noinspection PyShadowingBuiltins
def abs(number: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the absolute value of a number.

    :param number: A number.
    :return: The absolute value of the number.
    """
    if not (number == number):  # is_na_arg
        return na_float
    return builtins.abs(number)


def acos(angle: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the arc cosine of a value.

    :param angle: A value.
    :return: The arc cosine of the value.
    """
    if not (angle == angle):  # is_na_arg
        return na_float
    # TV's JVM has no JIT intrinsic for acos/asin: their runtime is StrictMath (fdlibm)
    return fdlibm.acos(angle)


def asin(angle: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the arc sine of a value.

    :param angle: A value.
    :return: The arc sine of the value.
    """
    if not (angle == angle):  # is_na_arg
        return na_float
    return fdlibm.asin(angle)


def atan(angle: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the arc tangent of a value.

    :param angle: A value.
    :return: The arc tangent of the value.
    """
    if not (angle == angle):  # is_na_arg
        return na_float
    return math.atan(angle)


def avg(*numbers: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the average of the numbers.

    :param numbers: Numbers.
    :return: The average of the numbers.
    """
    # Measured law (probes m569-m576): TradingView runs two different sums here.
    # Up to two arguments the terms are added plainly -- a compensated sum was
    # rejected on 4188 bars where the two disagree, with zero counter-examples.
    # From three arguments on, the terms go through a Kahan compensated sum whose
    # pending correction is flushed back into the total before the division; that
    # flush was confirmed on every bar where it decides the last bit, and a plain
    # sum misses TradingView on up to a quarter of the bars once the terms differ
    # in magnitude.
    assert numbers, "At least one number is necessary!"

    count = len(numbers)
    for n in numbers:
        if not (n == n):
            return na_float

    if count <= 2:
        summ = 0.0
        for n in numbers:
            summ = summ + n
        return summ / count

    summ = 0.0
    compensation = 0.0
    for n in numbers:
        y = n - compensation
        t = summ + y
        compensation = (t - summ) - y
        summ = t

    return (summ - compensation) / count


def ceil(number: TFI | NA[TFI]) -> PyneInt:
    """
    Returns the smallest integer greater than or equal to a number.

    :param number: A number.
    :return: The smallest integer greater than or equal to the number.
    """
    if not (number == number):  # is_na_arg
        return na_int
    # A Pine int is a double at runtime: the integral result travels as a float
    return float(math.ceil(number))


def cos(angle: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the cosine of an angle.

    :param angle: An angle in radians.
    :return: The cosine of the angle.
    """
    if not (angle == angle):  # is_na_arg
        return na_float
    # TV's JIT evaluates runtime cos/sin/exp with the Intel-LIBM intrinsics,
    # which pine_math ports bit-exactly (parse-time constants fold with
    # fdlibm instead -- see transformers/const_fold.py)
    return pine_math.cos(angle)


def exp(number: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns e raised to the power of a number.

    :param number: A number.
    :return: e raised to the power of the number.
    """
    if not (number == number):  # is_na_arg
        return na_float
    return pine_math.exp(number)


def floor(number: TFI | NA[TFI]) -> PyneInt:
    """
    Returns the largest integer less than or equal to a number.

    :param number: A number.
    :return: The largest integer less than or equal to the number.
    """
    if not (number == number):  # is_na_arg
        return na_int
    # int() truncates toward zero; Pine's floor is a true floor (floor(-1.2) == -2).
    # A Pine int is a double at runtime: the integral result travels as a float
    return float(math.floor(number))


def log(number: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the natural logarithm of a number.

    :param number: A number.
    :return: The natural logarithm of the number.
    """
    if not (number == number):  # is_na_arg
        return na_float
    return pine_math.log(number)


def log10(number: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the base-10 logarithm of a number.

    :param number: A number.
    :return: The base-10 logarithm of the number.
    """
    if not (number == number):  # is_na_arg
        return na_float
    return math.log10(number)


def _na_of_operands(numbers: tuple[TFI | NA[TFI], ...]) -> PyneFloat:
    """
    Return the na matching the operands' numeric contract: the typeless na when
    no operand carries a type at all, the numeric na otherwise.

    Pine's int is a static type only, so an int-typed na and a float-typed na
    are the same native nan at runtime; only the typeless ``na`` (an ``NA``
    with no type) is distinct, and it must stay typeless so it does not push
    a contract on the caller.
    """
    for n in numbers:
        if n == n or not isinstance(n, NA) or n.type is not None:
            return na_float
    return NA(None)


# noinspection PyShadowingBuiltins
@overload
def max(*numbers: int) -> PyneInt: ...
# noinspection PyShadowingBuiltins
@overload
def max(*numbers: TFI | NA[TFI]) -> PyneFloat: ...


# noinspection PyShadowingBuiltins
def max(*numbers: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the largest number.

    :param numbers: Numbers.
    :return: The largest number.
    """
    assert numbers, "At least one number is necessary!"

    # The na scan is a plain loop, not ``any(... for n in numbers)``: the generator
    # object the comprehension allocates on every call is pure overhead here, and
    # these two are among the most-called builtins in a script (a rolling
    # min/max loop reaches them once per window element per bar).
    for n in numbers:
        if not (n == n):  # is_na_arg
            return _na_of_operands(numbers)

    return builtins.max(cast(list[TFI], numbers))


# noinspection PyShadowingBuiltins
@overload
def min(*numbers: int) -> PyneInt: ...
# noinspection PyShadowingBuiltins
@overload
def min(*numbers: TFI | NA[TFI]) -> PyneFloat: ...


# noinspection PyShadowingBuiltins
def min(*numbers: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the smallest number.

    :param numbers: Numbers.
    :return: The smallest number.
    """
    assert numbers, "At least one number is necessary!"

    # Plain loop instead of ``any(... for n in numbers)`` -- see :func:`max`.
    for n in numbers:
        if not (n == n):  # is_na_arg
            return _na_of_operands(numbers)

    return builtins.min(cast(list[TFI], numbers))


# noinspection PyShadowingBuiltins
def pow(base: TFI | NA[TFI], exponent: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns a number raised to the power of another number.

    :param base: The base number.
    :param exponent: The exponent number.
    :return: The base number raised to the power of the exponent number.
    """
    if isinstance(base, NA) or isinstance(exponent, NA):
        return na_float
    if base != base or exponent != exponent:
        return na_float

    b = cast(float, base)
    # MEASURED (BINANCE:BTCUSDT@30, 8000 bars, base in [0.3, 1.3]): TradingView
    # answers these four exponents with the shortcut result exactly -- every bar
    # of ``pow(x, 2) - x * x``, ``pow(x, 0.5) - sqrt(x)``, ``pow(x, 1) - x`` and
    # ``pow(x, 0) - 1`` was zero. The platform ``pow()`` is not: it disagrees
    # with ``x * x`` on 8 of those bars and with ``sqrt(x)`` on 5, which a
    # recursive script carries into its output (Signal Moving Average [LuxAlgo]).
    # Only these hold -- ``pow(x, -1)`` is NOT ``1 / x`` on TradingView (3 bars),
    # and ``pow(x, 3)`` is not ``x * x * x`` on 2149 of them.
    if exponent == 2:
        return b * b
    if exponent == 1:
        return b
    if exponent == 0:
        return 1.0
    if exponent == 0.5 and b >= 0.0:
        return math.sqrt(b)

    return b ** cast(float, exponent)


# noinspection PyShadowingBuiltins
@overload
def round(number: TFI | NA[TFI]) -> PyneInt: ...
# noinspection PyShadowingBuiltins
@overload
def round(number: TFI | NA[TFI], precision: PyneInt) -> PyneFloat: ...


# noinspection PyShadowingBuiltins
def round(number: TFI | NA[TFI], precision: PyneInt = na_int) -> PyneFloat:
    """
    Returns a number rounded to a specified number of decimal places, ties going
    away from zero.

    Without a precision, or with a precision of zero or less, the number is
    rounded to an integer. A positive precision keeps the integer part of the
    number and rounds only its fractional part to ``precision`` decimals; a
    fractional precision is used as it is, and a precision above 16 acts as 16.

    :param number: A number.
    :param precision: The number of decimal places to round to.
    :return: The rounded number.
    """
    # The two overloads are told apart by the PRESENCE of a precision, never by
    # its Python type: an int-TYPED Pine expression can arrive as a float
    # (``math.round(x, 4 / 2)``), and an ``isinstance(precision, int)`` test
    # silently took that for "no precision given" and dropped the rounding.
    has_precision = precision == precision  # is_na_arg (inverted)
    if not (number == number):  # is_na_arg
        # No precision means the int contract (first overload), so an int-typed na
        return na_float if has_precision else na_int
    if not math.isfinite(number):
        # Pine has no non-finite values (1/0 is na); the precision overload keeps
        # builtins.round() behavior (returns the float unchanged), but the
        # one-argument overload must honor its int contract, so it yields an int na
        return cast(float, number) if has_precision else na_int
    negative = number < 0.0
    magnitude = -number if negative else number
    if not has_precision or precision <= 0.0:
        # Half away from zero, compared EXACTLY: a value below the tie rounds down
        # however close it sits (452523.49999999994 -> 452523 at 5.8e-11), an exact
        # tie goes away from zero (2.5 -> 3, -2.5 -> -3). MEASURED on
        # BINANCE:BTCUSDT@30 and FX:EURUSD@60 (probes roundprobe, mr2, mr3, mr10).
        # A negative precision does NOT round to tens or hundreds: it is the same
        # integer rounding (1234.5678 @ -2 -> 1235, 2.5 @ -3 -> 3, mr2/mr3).
        # From 2**52 up every double is an integer, and ``magnitude + 0.5`` would
        # round to even there (2**52 + 1 -> 2**52 + 2); below it the sum is exact.
        if magnitude >= _ROUND_INT_LIMIT:
            return cast(float, number)
        units = math.floor(magnitude + 0.5)
        # A Pine int is a double at runtime: the integral result travels as a float
        return float(-units if negative else units)
    # TV splits the number into its integer part and its fraction and scales ONLY
    # the fraction: ``i + round(f * scale) / scale``. That is why a fractional
    # precision gives 2.34567 @ 0.25 -> 2.5623413251903493 (= 2 + 10**-0.25)
    # rather than anything near the input. The scale is the reciprocal of
    # ``10**-precision``, not ``10**precision`` -- the two differ by 1 ulp at
    # precision 0.75, 1.75, 2.5 and 3.5 and the reciprocal reproduces every
    # measurement; the very same double is the divisor. MEASURED on FX:EURUSD@60
    # (probes mr1-mr14: 448/448 at precision <= 15, plus mrp1/mrp2 for the tie
    # rule and the precision cap below).
    # Above 16 the precision is clamped: every measured value at 16.001 .. 400 is
    # bit-identical to the one at 16 (mrp2), while at 15.8 .. 16 TV's own last
    # bit is not fully modelled (67/78 there) -- that noise is inherited by the
    # clamped range.
    p = precision if precision <= _ROUND_MAX_PRECISION else _ROUND_MAX_PRECISION
    scale = 1.0 / math.pow(10.0, -p)
    fraction = math.fmod(magnitude, 1.0)
    scaled = fraction * scale
    units = math.floor(scaled)
    offset = scaled - units - 0.5
    # The tie is decided on the EXACT product of the two doubles with the same
    # 1e-10 tolerance the relational operators and round_to_mintick carry: the
    # double product cannot tell 13.887175 @ 5 (exact product 9.7e-11 short, TV
    # rounds up) from 63.986275 @ 5 (1.02e-10 short, TV rounds down) -- both
    # scale to the same 88717.4999999999 -- yet the exact product does (mrp1:
    # 59/59 where the double product misses 4), and the threshold sits at
    # 1e-10 to within 1e-16 (0.266741911488056 @ 0.75 up at 9.99981e-11,
    # 0.14999999999 @ 1 down at 1.0000e-10). Only a near-tie is worth the exact
    # product; the window covers the double's own error (8 ulp) at any magnitude.
    slack = 1e-9 + scaled * 1.8e-15
    if -slack < offset < slack:
        numerator, denominator = fraction.as_integer_ratio()
        scale_numerator, scale_denominator = scale.as_integer_ratio()
        numerator *= scale_numerator
        denominator *= scale_denominator
        remainder = numerator - units * denominator
        if remainder * _ROUND_TIE_SCALE >= _ROUND_TIE_HALF * denominator:
            units += 1
    elif offset > 0.0:
        units += 1
    value = magnitude - fraction + units / scale
    # ``0.0 - value``: a negative fraction rounded away to nothing is 0, not -0
    # (-7.12675179601e-05 @ 2.5 -> 0 in mr10; TV prints -0 as "0" anyway).
    return 0.0 - value if negative else value


@overload
def round_to_mintick(number: float | int) -> float: ...
@overload
def round_to_mintick(number: PyneFloat | PyneInt) -> PyneFloat: ...

def round_to_mintick(number: PyneFloat | PyneInt) -> PyneFloat:
    """
    Returns value rounded to symbol's mintick, ties going away from zero.
    """
    if not (number == number):  # is_na_arg
        return na_float
    # `mintick = minmove / pricescale` (Pine syminfo), so the position on the tick
    # grid is `number * pricescale / minmove` -- reconstructed from the int pair, not
    # divided by the double mintick, which `minmove != 1` symbols (QM1!: mintick=0.025,
    # pricescale=1000, minmove=25) need anyway and which the tie rule below requires.
    # TV decides a tie on the EXACT value of that product with the same 1e-10 tolerance
    # `round()` carries. MEASURED on BINANCE:BTCUSDT 30m: 19999.585, whose product falls
    # 8.7e-11 short of the tie, rounds UP, while 16384.245 at 1.0e-10 short rounds DOWN
    # -- yet both products are the very same exactly representable `.5` as a double, so
    # no double-precision model can separate them. Negatives are symmetric (-1.075 ->
    # -1.08, -94130.045 -> -94130.04).
    minmove = syminfo.minmove
    pricescale = syminfo.pricescale
    negative = number < 0.0
    magnitude = -number if negative else number
    scaled = magnitude * pricescale / minmove
    units = builtins.int(scaled)
    offset = scaled - units - 0.5
    # Only a near-tie is worth the exact product; elsewhere the double decides. The
    # window covers the double's own error (8 ulp) at any magnitude.
    slack = 1e-6 + scaled * 1.8e-15
    if -slack < offset < slack:
        numerator, denominator = float(magnitude).as_integer_ratio()
        # The tick pair is a Pine int (a double); the exact product needs the native ints
        numerator *= builtins.int(pricescale)
        denominator *= builtins.int(minmove)
        units, remainder = divmod(numerator, denominator)
        if remainder * _ROUND_TIE_SCALE >= _ROUND_TIE_HALF * denominator:
            units += 1
    elif offset > 0.0:
        units += 1
    return (-units if negative else units) * minmove / pricescale


def sign(number: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the sign of a number.

    :param number: A number.
    :return: The sign of the number.
    """
    if not (number == number):  # is_na_arg
        return na_float
    if number == 0.0:
        return 0.0
    if number > 0.0:
        return 1.0
    return -1.0


def sin(angle: float | int | NA) -> PyneFloat:
    """
    Returns the sine of an angle.

    :param angle: An angle in radians.
    :return: The sine of the angle.
    """
    if not (angle == angle):  # is_na_arg
        return na_float
    return pine_math.sin(angle)


def sqrt(number: float | int | NA) -> PyneFloat:
    """
    Returns the square root of a number.

    :param number: A number.
    :return: The square root of the number.
    """
    if not (number == number):  # is_na_arg
        return na_float
    try:
        return math.sqrt(number)
    except ValueError:
        return na_float


def tan(angle: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the tangent of an angle.

    :param angle: An angle in radians.
    :return: The tangent of the angle.
    """
    if not (angle == angle):  # is_na_arg
        return na_float
    return math.tan(angle)


def todegrees(radians: TFI | NA[TFI]) -> PyneFloat:
    """
    Converts an angle from radians to degrees.

    :param radians: An angle in radians.
    :return: The angle in degrees.
    """
    if not (radians == radians):  # is_na_arg
        return na_float
    return math.degrees(radians)


def toradians(degrees: TFI | NA[TFI]) -> PyneFloat:
    """
    Converts an angle from degrees to radians.

    :param degrees: An angle in degrees.
    :return: The angle in radians.
    """
    if not (degrees == degrees):  # is_na_arg
        return na_float
    return math.radians(degrees)
