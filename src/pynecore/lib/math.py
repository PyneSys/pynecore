from typing import TypeVar, cast, overload
import builtins
import math

from ..core import fdlibm, pine_math
from ..types.na import NA, na_float
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
        return NA(int)
    return math.ceil(number)


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
        return NA(int)
    # int() truncates toward zero; Pine's floor is a true floor (floor(-1.2) == -2)
    return math.floor(number)


def log(number: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the natural logarithm of a number.

    :param number: A number.
    :return: The natural logarithm of the number.
    """
    if not (number == number):  # is_na_arg
        return na_float
    return math.log(number)


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
    Return the na matching the operands' numeric contract: na_float when any
    type-carrying operand is float-like, NA(int) when the type-carrying operands
    are all int-like, the typeless na when no operand carries a type at all.
    Typeless na operands are neutral — they must not push an int contract to float.
    """
    saw_typed = False
    for n in numbers:
        if n != n:
            # A native nan is a float-typed na by definition
            return na_float
        if isinstance(n, NA):
            if n.type is None:
                continue
            saw_typed = True
            if n.type is not int:
                return na_float
        else:
            saw_typed = True
            if not isinstance(n, int):
                return na_float
    return NA(int) if saw_typed else NA(None)


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
def round(number: TFI | NA[TFI], precision: PyneInt = NA(int)) -> PyneFloat:
    """
    Returns a number rounded to a specified number of decimal places.

    :param number: A number.
    :param precision: The number of decimal places to round to.
    :return: The rounded number.
    """
    if not (number == number):  # is_na_arg
        # No precision means the int contract (first overload), so an int-typed na
        return na_float if isinstance(precision, int) else NA(int)
    if not math.isfinite(number):
        # Pine has no non-finite values (1/0 is na); the precision overload keeps
        # builtins.round() behavior (returns the float unchanged), but the
        # one-argument overload must honor its int contract, so it yields an int na
        return cast(float, number) if isinstance(precision, int) else NA(int)
    # TV rounds the EXACT binary value of the double scaled by 10**precision, with
    # ties going away from zero and a 1e-10 absolute tolerance on that scaled value
    # -- the same slack its relational operators carry. Measured on BINANCE:BTCUSDT
    # 30m (roundprobe, 28397 bars x 5 series columns at precision 2/3/5): all
    # 141985 values reproduce, and no tolerance between 0 and 5e-11 or above 2e-10
    # does. The tolerance is what separates 118152.265 -> 118152.27 (short 5.8e-11
    # of the tie) from 100019.7405 -> 100019.740 (short 1.2e-10); both scale to an
    # exactly representable .5, so no double-precision model can tell them apart.
    # Known limit: at precision 8 on single-digit values the tolerance no longer
    # separates the ties (26697 of 28397) -- that regime is unmodelled.
    p = precision if isinstance(precision, int) else 0
    if p > 308 or p < -308:
        # 10.0 ** p overflows; no attainable rounding changes a finite double here
        return cast(float, number)
    negative = number < 0.0
    numerator, denominator = float(abs(number)).as_integer_ratio()
    if p >= 0:
        numerator *= 10 ** p
    else:
        denominator *= 10 ** -p
    units, remainder = divmod(numerator, denominator)
    if remainder * _ROUND_TIE_SCALE >= _ROUND_TIE_HALF * denominator:
        units += 1
    if not isinstance(precision, int):
        return -units if negative else units
    value = units / 10.0 ** p if p >= 0 else units * 10.0 ** -p
    return -value if negative else value


@overload
def round_to_mintick(number: float | int) -> float: ...
@overload
def round_to_mintick(number: PyneFloat | PyneInt) -> PyneFloat: ...

def round_to_mintick(number: PyneFloat | PyneInt) -> PyneFloat:
    """
    Returns value rounded to symbol's mintick with ties rounding up.
    """
    if not (number == number):  # is_na_arg
        return na_float
    # `mintick = minmove / pricescale` (Pine syminfo). Reconstruct via int math so
    # `minmove=1` paths stay bit-identical to the old formula, while `minmove != 1`
    # symbols (e.g. QM1!: mintick=0.025, pricescale=1000, minmove=25) round correctly.
    return int(number / syminfo.mintick + 0.5) * syminfo.minmove / syminfo.pricescale


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
