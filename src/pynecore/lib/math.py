from typing import TypeVar, cast, overload
import builtins
import math
from decimal import Decimal, ROUND_HALF_UP, localcontext

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

# noinspection PyShadowingBuiltins
def abs(number: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the absolute value of a number.

    :param number: A number.
    :return: The absolute value of the number.
    """
    if (isinstance(number, NA) or number != number):
        return na_float
    return builtins.abs(number)


def acos(value: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the arc cosine of a value.

    :param value: A value.
    :return: The arc cosine of the value.
    """
    if (isinstance(value, NA) or value != value):
        return na_float
    return math.acos(value)


def asin(value: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the arc sine of a value.

    :param value: A value.
    :return: The arc sine of the value.
    """
    if (isinstance(value, NA) or value != value):
        return na_float
    return math.asin(value)


def atan(value: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the arc tangent of a value.

    :param value: A value.
    :return: The arc tangent of the value.
    """
    if (isinstance(value, NA) or value != value):
        return na_float
    return math.atan(value)


def avg(*numbers: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the average of the numbers.

    :param numbers: Numbers.
    :return: The average of the numbers.
    """
    assert numbers, "At least one number is necessary!"

    if any((isinstance(n, NA) or n != n) for n in numbers):
        return na_float

    return builtins.sum(n for n in numbers) / len(numbers)


def ceil(number: TFI | NA[TFI]) -> PyneInt:
    """
    Returns the smallest integer greater than or equal to a number.

    :param number: A number.
    :return: The smallest integer greater than or equal to the number.
    """
    if (isinstance(number, NA) or number != number):
        return NA(int)
    return math.ceil(number)


def cos(angle: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the cosine of an angle.

    :param angle: An angle in radians.
    :return: The cosine of the angle.
    """
    if (isinstance(angle, NA) or angle != angle):
        return na_float
    return math.cos(angle)


def exp(number: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns e raised to the power of a number.

    :param number: A number.
    :return: e raised to the power of the number.
    """
    if (isinstance(number, NA) or number != number):
        return na_float
    return math.exp(number)


def floor(number: TFI | NA[TFI]) -> PyneInt:
    """
    Returns the largest integer less than or equal to a number.

    :param number: A number.
    :return: The largest integer less than or equal to the number.
    """
    if (isinstance(number, NA) or number != number):
        return NA(int)
    # int() truncates toward zero; Pine's floor is a true floor (floor(-1.2) == -2)
    return math.floor(number)


def log(number: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the natural logarithm of a number.

    :param number: A number.
    :return: The natural logarithm of the number.
    """
    if (isinstance(number, NA) or number != number):
        return na_float
    return math.log(number)


def log10(number: TFI | NA[TFI]) -> PyneFloat:
    """
    Returns the base-10 logarithm of a number.

    :param number: A number.
    :return: The base-10 logarithm of the number.
    """
    if (isinstance(number, NA) or number != number):
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

    if any((isinstance(n, NA) or n != n) for n in numbers):
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

    if any((isinstance(n, NA) or n != n) for n in numbers):
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
    if (isinstance(base, NA) or base != base) or (isinstance(exponent, NA) or exponent != exponent):
        return na_float

    return base ** exponent


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
    if (isinstance(number, NA) or number != number):
        # No precision means the int contract (first overload), so an int-typed na
        return na_float if isinstance(precision, int) else NA(int)
    # TV-measured: ties round away from zero on the decimal (shortest-repr) value,
    # not half-even on the binary double (round(2.5) == 3, round(-2.5) == -3,
    # round(2.675, 2) == 2.68 — builtins.round gives 2, -2 and 2.67 there)
    if not math.isfinite(number):
        # Pine has no non-finite values (1/0 is na); the precision overload keeps
        # builtins.round() behavior (returns the float unchanged), but the
        # one-argument overload must honor its int contract, so it yields an int na
        return number if isinstance(precision, int) else NA(int)
    decimal_number = Decimal(repr(number))
    if not isinstance(precision, int):
        return int(decimal_number.to_integral_value(rounding=ROUND_HALF_UP))
    # quantize() fails when the result needs more digits than the context precision
    # (default 28), so size a local context from the magnitude and the requested
    # precision (e.g. round(1e30, 2), round(x, 100))
    with localcontext() as ctx:
        ctx.prec = builtins.max(1, decimal_number.adjusted() + precision + 2)
        return float(decimal_number.quantize(Decimal(1).scaleb(-precision),
                                             rounding=ROUND_HALF_UP))


@overload
def round_to_mintick(number: float | int) -> float: ...
@overload
def round_to_mintick(number: PyneFloat | PyneInt) -> PyneFloat: ...

def round_to_mintick(number: PyneFloat | PyneInt) -> PyneFloat:
    """
    Returns value rounded to symbol's mintick with ties rounding up.
    """
    if (isinstance(number, NA) or number != number):
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
    if (isinstance(number, NA) or number != number):
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
    if (isinstance(angle, NA) or angle != angle):
        return na_float
    return math.sin(angle)


def sqrt(number: float | int | NA) -> PyneFloat:
    """
    Returns the square root of a number.

    :param number: A number.
    :return: The square root of the number.
    """
    if (isinstance(number, NA) or number != number):
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
    if (isinstance(angle, NA) or angle != angle):
        return na_float
    return math.tan(angle)


def todegrees(angle: TFI | NA[TFI]) -> PyneFloat:
    """
    Converts an angle from radians to degrees.

    :param angle: An angle in radians.
    :return: The angle in degrees.
    """
    if (isinstance(angle, NA) or angle != angle):
        return na_float
    return math.degrees(angle)


def toradians(angle: TFI | NA[TFI]) -> PyneFloat:
    """
    Converts an angle from degrees to radians.

    :param angle: An angle in degrees.
    :return: The angle in radians.
    """
    if (isinstance(angle, NA) or angle != angle):
        return na_float
    return math.radians(angle)
