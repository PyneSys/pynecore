from typing import Any
import re

from functools import lru_cache
from math import isinf, copysign as _copysign

from datetime import datetime, UTC
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, ROUND_HALF_EVEN, localcontext

from ..core.safe_convert import native_int_or as _native_int_or
from ..types.matrix import Matrix as _Matrix
from ..types.na import NA, na_float, na_int
from ..types.pine_types import PyneFloat, PyneInt, PyneStr, PyneBool, pine_int
from ..utils.sequence_view import SequenceView as _SequenceView

from ..types.format import Format
from . import format as _format
from . import syminfo as _syminfo
from .. import lib
from ..core import safe_convert

from pynecore.core.datetime import parse_timezone as _parse_timezone

__all__ = ['contains', 'endswith', 'format', 'format_time', 'length', 'lower', 'match', 'pos', 'repeat', 'replace',
           'replace_all', 'split', 'startswith', 'substring', 'tonumber', 'tostring', 'trim', 'upper']


#
# Private helper functions
#

# Enough digits for the integer part of any finite double (max ~1.8e308).
_MAX_INT_DIGITS = 320
# Below 2**53 every integral double is exact, so the double's exact value and its shortest
# digits round alike at any decimal place, ties included
_EXACT_INT = 9007199254740992.0
# str.tostring without a format prints plain decimals only strictly between these
# magnitudes, and switches to exponential notation outside them (measured: 9.99e20 and
# 1.5e-21 are plain, 1e21 and 1e-21 are not)
_EXP_HIGH = 1e21
_EXP_LOW = 1e-21
# MessageFormat's number style, used by ``{0}`` and ``{0,number}``
_MESSAGE_NUMBER = '#,##0.###'


def _decimal_format_quantize(digits: Decimal, value: float, quantum: Decimal) -> Decimal:
    """
    Round shortest digits at ``quantum`` the way Java's ``DecimalFormat`` does.

    ``DecimalFormat`` rounds the digits ``Double.toString`` produced, half-even, but a tie
    on those digits is decided on the double's exact value: 2.675 (really 2.67499...) goes
    down, 0.0025 (really 0.00250000000000000005) goes up, an exact 0.25 goes to even.

    One exception is Java's own: when the lone digit 5 sits right below the rounding place
    of a value under 1e-3, ``DigitList`` reads it from the exponential ``5.0E-11`` spelling
    before dropping trailing zeros, sees a 5 followed by a 0 instead of a tie, and rounds
    down -- ``5e-11`` with ten decimals and ``0.0005`` with three print ``0``, while
    ``0.05`` with one prints ``0.1`` (measured on TradingView and reproduced on JDK 8).

    :param digits: The shortest round-trip digits of ``value``
    :param value: The double the digits stand for, non-negative and finite
    :param quantum: The place to round at, as a power of ten
    :return: The rounded digits, carrying ``quantum``'s exponent
    """
    down = digits.quantize(quantum, rounding=ROUND_DOWN)
    tail = (digits - down) * 2
    if tail > quantum:
        return down + quantum
    if tail < quantum or (value < 1e-3 and down.is_zero()):
        return down
    exact = Decimal(value)
    if exact > digits:
        return down + quantum
    if exact < digits:
        return down
    return digits.quantize(quantum, rounding=ROUND_HALF_EVEN)


def _round_digits(value: float, decimals: int, decimal_format: bool) -> str:
    """
    Round a positive value to the requested number of decimals, the way TradingView does.

    Both formatters start from the digits Java's ``Double.toString`` produces -- the
    shortest decimal that reads back as the same double -- not from the double's exact
    binary expansion: ``str.tostring(0.1, '#.####################')`` is ``0.1``, not
    ``0.10000000000000000555``. While that representation stays in plain notation
    (``|value| >= 1e-3``) it never carries more than 16 fraction digits, so a mask asking
    for more gets zeros; below 1e-3 ``Double.toString`` switches to exponential notation
    and the cap is gone (measured: ``1e-17`` prints all 17 decimals).

    ``str.tostring`` with a mask then rounds those digits half-up, while ``str.format`` is
    Java's DecimalFormat (see :func:`_decimal_format_quantize`), which breaks a tie on the
    exact value -- 2.675 is really 2.67499..., so it rounds down where ``str.tostring``
    rounds up.

    :param value: The value to round (must be non-negative and finite)
    :param decimals: Number of fraction digits the mask allows
    :param decimal_format: True for ``str.format()``, False for ``str.tostring()``
    :return: The rounded value in plain decimal notation
    """
    if value >= 1e-3:
        decimals = min(decimals, 16)
    d = Decimal(repr(value))
    quantum = Decimal(1).scaleb(-decimals)
    with localcontext() as ctx:
        ctx.prec = _MAX_INT_DIGITS + decimals
        if decimal_format:
            return f"{_decimal_format_quantize(d, value, quantum):f}"
        return f"{d.quantize(quantum, rounding=ROUND_HALF_UP):f}"


def _group_digits(digits: str, size: int) -> str:
    """
    Insert the grouping separator into an integer digit string.

    :param digits: The integer digits, without sign
    :param size: Digits per group, from the pattern's last ``,`` to its decimal point
    :return: The digits with ``,`` between the groups
    """
    if size <= 0 or len(digits) <= size:
        return digits
    head = len(digits) % size or size
    return ','.join([digits[:head]] + [digits[i:i + size] for i in range(head, len(digits), size)])


def _grouping_size(int_pattern: str) -> int:
    """
    The grouping size of a DecimalFormat integer pattern: the digits after its last ``,``.

    :param int_pattern: The digit pattern before the decimal point
    :return: The group size, 0 when the pattern does not group
    """
    comma = int_pattern.rfind(',')
    return 0 if comma < 0 else len(int_pattern) - comma - 1


def _format_default(value: float) -> str:
    """
    Format a finite or non-finite double the way ``str.tostring(value)`` does without a format.

    This is not the ``'#.##########'`` mask: it is Java's ``DecimalFormat`` with ten optional
    fraction digits (ties on the exact value, :func:`_decimal_format_quantize`, where the
    mask rounds half-up: ``3.5e-10`` prints ``0.0000000003`` here and ``0.0000000004`` with
    the mask), and it switches to ``0.##########E0`` outside ``(1e-21, 1e21)`` (``1E21``,
    ``1.2345678901E-25``). A result without a non-zero digit carries no sign.

    :param value: The value to format
    :return: The formatted string
    """
    if not (value == value) or isinf(value):
        return "NaN"
    magnitude = abs(value)
    if magnitude == 0.0:
        return "0"
    sign = '-' if value < 0 else ''
    if _EXP_LOW < magnitude < _EXP_HIGH:
        text = _round_digits_exact(magnitude, 10) if magnitude < _EXACT_INT else None
        if text is None:
            text = _round_digits(magnitude, 10, True)
        text = text.rstrip('0').rstrip('.') if '.' in text else text
        return "0" if text == "0" else sign + text

    digits = Decimal(repr(magnitude))
    exponent = digits.adjusted()
    with localcontext() as ctx:
        ctx.prec = 40
        rounded = _decimal_format_quantize(digits, magnitude, Decimal(1).scaleb(exponent - 10))
        if rounded.adjusted() > exponent:
            # 9.99999999999e21 rounds up to 10.0000000000e21, printed as 1E22
            exponent += 1
        mantissa = f"{rounded.scaleb(-exponent).normalize():f}"
    return f"{sign}{mantissa}E{exponent}"


def _java_double_str(value: float) -> str:
    """
    Java's ``Double.toString``: what ``str.format`` prints for a float array element.

    Plain decimals with at least one fraction digit for ``1e-3 <= |x| < 1e7``, computerized
    scientific notation otherwise (``1.0E21``, ``5.0E-4``), ``-0.0`` for negative zero.

    :param value: The value to format
    :return: The formatted string
    """
    if not (value == value):
        return "NaN"
    if isinf(value):
        return "-Infinity" if value < 0 else "Infinity"
    if value == 0.0:
        return "-0.0" if _copysign(1.0, value) < 0.0 else "0.0"
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e7:
        return repr(value)
    digits, exponent = Decimal(repr(magnitude)).as_tuple()[1:]
    text = ''.join(map(str, digits))
    # A finite Decimal's exponent is an int; int() narrows the NaN/Infinity markers of its type
    point = len(text) + int(exponent) - 1
    fraction = text[1:].rstrip('0') or '0'
    return f"{'-' if value < 0 else ''}{text[0]}.{fraction}E{point}"


@lru_cache(maxsize=128)
def _split_number_pattern(pattern: str) -> tuple[str, str, str]:
    """
    Cut a Java DecimalFormat subpattern into literal prefix, digit pattern and literal suffix.

    Only an unquoted ``#``, ``0``, ``.`` or ``,`` belongs to the digit pattern; in the
    affixes ``'#'`` is a literal hash and ``''`` a literal quote, so ``'#'.##`` prints
    ``#1.23`` instead of opening the digits one character early.

    :param pattern: One subpattern (the part between two ``;`` separators)
    :return: The prefix, the digit pattern and the suffix, all quotes resolved
    """
    chars: list[str] = []
    is_digit: list[bool] = []
    in_quote = False
    i = 0
    n = len(pattern)

    while i < n:
        ch = pattern[i]
        if ch == "'":
            if i + 1 < n and pattern[i + 1] == "'":
                chars.append("'")
                is_digit.append(False)
                i += 1
            else:
                in_quote = not in_quote
                i += 1
                continue
        else:
            chars.append(ch)
            is_digit.append(not in_quote and ch in '#0.,')
        i += 1

    if True not in is_digit:
        return ''.join(chars), '', ''
    lo = is_digit.index(True)
    hi = len(is_digit) - 1 - is_digit[::-1].index(True)
    return ''.join(chars[:lo]), ''.join(chars[lo:hi + 1]), ''.join(chars[hi + 1:])


def _round_digits_fast(value: float, decimals: int, pad: bool) -> str | None:
    """
    Round a non-negative double half-up on its shortest round-trip digits, without ``Decimal``.

    ``repr(value)`` already IS the digit string ``Decimal(repr(value))`` would carry -- the
    shortest decimal that reads back as the same double, the same thing Java's
    ``Double.toString`` produces -- so the half-up decision and the carry can be done with
    plain integer arithmetic on those digits. The 16-fraction-digit cap at ``|value| >= 1e-3``
    is applied exactly as in :func:`_round_digits`.

    The caller must pass a non-negative finite double. A ``repr`` in exponential notation
    (``|value| < 1e-4`` or ``>= 1e16``) and negative zero are not handled here and answer
    ``None``, which routes the value to the general path.

    :param value: The value to round (non-negative and finite)
    :param decimals: Number of fraction digits the mask allows
    :param pad: True to fill the unused fraction places with zeros. A caller whose mask has
                no required decimal place strips them again right away, and asks for False
    :return: The rounded value with the allowed number of fraction digits, or ``None`` when
             this routine does not cover the value
    """
    r = repr(value)
    if 'e' in r or r[0] == '-':
        return None
    dot = r.find('.')
    if dot < 0:
        int_part, frac = r, ''
    else:
        int_part = r[:dot]
        frac = r[dot + 1:]
    if decimals > 16 and value >= 1e-3:
        decimals = 16

    n_frac = len(frac)
    if n_frac <= decimals:
        # The shortest digits already fit the mask: only zero padding is needed
        if decimals == 0:
            return int_part
        if not pad:
            return r
        return int_part + '.' + frac + '0' * (decimals - n_frac)

    keep = (int_part + frac)[:len(int_part) + decimals]
    if frac[decimals] >= '5':
        # Half-up on the digit string; zfill restores the leading zeros int() dropped
        keep = str(int(keep) + 1).zfill(len(keep))
    if decimals == 0:
        return keep
    return keep[:-decimals] + '.' + keep[-decimals:]


def _round_digits_exact(value: float, decimals: int) -> str | None:
    """
    Round a non-negative finite double the way Java's DecimalFormat does, without ``Decimal``.

    This is the ``str.format`` rounding of :func:`_round_digits`. When the shortest digits
    already fit the mask they are the answer, zero padded. Otherwise the rounding place lies
    inside them, and there the shortest digits and the exact value round alike, ties
    included (a tie on the shortest digits is decided on the exact value: 2.675, which is
    really 2.67499..., rounds down), so Python's fixed-point float formatting, correctly
    rounded from the exact value, yields the same digits. The 16-fraction-digit cap at
    ``|value| >= 1e-3`` applies here as well.

    Values below 1e-3, where Java's ``DigitList`` has its own tie rule (see
    :func:`_decimal_format_quantize`), and negative zero are not covered and answer ``None``.

    :param value: The value to round (non-negative, finite and below 2**53)
    :param decimals: Number of fraction digits the mask allows
    :return: The rounded value with exactly the allowed number of fraction digits, or
             ``None`` when this routine does not cover the value
    """
    if value < 1e-3 and (value != 0.0 or _copysign(1.0, value) < 0.0):
        return None
    r = repr(value)
    if decimals > 16 and value >= 1e-3:
        decimals = 16
    int_part, _, frac = r.partition('.')
    if len(frac) <= decimals:
        if decimals == 0:
            return int_part
        return int_part + '.' + frac.ljust(decimals, '0')
    return f"{value:.{decimals}f}"


# One side of a parsed number pattern: literal prefix, literal suffix, the number of
# required integer digits ('0' before the point), the number of required fraction digits
# ('0' after it), the total number of fraction places the mask allows and the grouping
# size (0 when the pattern has no ',').
_PatternSide = tuple[str, str, int, int, int, int]


@lru_cache(maxsize=128)
def _parse_precision(precision: str) -> tuple[_PatternSide, _PatternSide] | None:
    """
    Parse a Java DecimalFormat number pattern into the two sides the formatter needs.

    The pattern is split on ``;`` into a positive and an optional negative subpattern, each
    of which is cut into affixes and a digit pattern by :func:`_split_number_pattern`. Without
    a negative subpattern the negative side reuses the positive one with a ``-`` in front of
    its prefix, which is what Java does. The result is cached per pattern string, because Pine
    masks are compile-time constants re-applied on every bar.

    A subpattern with more than one decimal point has no parsed form. The general path
    looks only at the subpattern the value's sign selects, so such a pattern is left to it
    whole: it formats through the well-formed side and raises on the other.

    :param precision: The whole pattern, negative subpattern included
    :return: The side used for non-negative values and the side used for negative ones,
             or None when either subpattern has more than one decimal point
    """
    subpatterns = precision.split(';')
    sides: list[_PatternSide] = []
    for chosen, sign_prefix in ((subpatterns[0], ''),
                                (subpatterns[1], '') if len(subpatterns) > 1 else (subpatterns[0], '-')):
        affix, digits, suffix = _split_number_pattern(chosen)
        before, dot, after = digits.partition('.')
        if dot and '.' in after:
            return None
        sides.append((sign_prefix + affix, suffix,
                      before.count('0'), after.count('0'), len(after), _grouping_size(before)))
    return sides[0], sides[1]


@lru_cache(maxsize=32)
def _mintick_decimals(tick_size: float) -> int:
    """
    The number of fraction digits of a tick size, counted on its plain decimal spelling.

    A tick in exponential notation is expanded first. Cached per tick value, so a symbol
    switch picks up its own tick without re-deriving it on every call.

    :param tick_size: The symbol's minimum tick
    :return: The fraction digit count, 0 for a whole-number tick
    """
    tick_str = str(tick_size)
    if 'e' in tick_str or 'E' in tick_str:
        tick_decimal = f"{tick_size:.20f}".rstrip('0')
        return len(tick_decimal.split('.')[1]) if '.' in tick_decimal else 0
    if '.' in tick_str:
        return len(tick_str.rstrip('0').split('.')[1])
    return 0


@lru_cache(maxsize=32)
def _mintick_mask(tick_size: float) -> str:
    """
    The digit mask ``str.tostring(x, format.mintick)`` prints a symbol's ticks with.

    One required fraction digit per fraction digit of the tick size: with a 0.00001 tick 1.1
    prints ``1.10000`` and 0 prints ``0.00000`` (measured).

    :param tick_size: The symbol's minimum tick
    :return: The mask, ``'0'`` for a whole-number tick
    """
    decimals = _mintick_decimals(tick_size)
    return '0.' + '0' * decimals if decimals else '0'


# The tick count format.mintick rounds to is a Java long: a larger count saturates
_MAX_TICKS = 9.223372036854775807e18

# format.volume suffixes, largest first, with the magnitude each one starts at
_VOLUME_UNITS = ((1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K'))


def _round_to_mintick(value: float, tick_size: float) -> float:
    """
    Round a finite value to a whole number of ticks the way ``format.mintick`` does.

    The tick count rounds half away from zero (-0.5 ticks is -1 tick, measured) and is held
    in a Java long, so a count beyond 2**63 saturates: with a 0.00001 tick both 1e16 and 1e21
    print ``92233720368547.77000`` on TradingView. A quotient that overflows to infinity
    saturates the same way.

    :param value: The value to round
    :param tick_size: The symbol's minimum tick, positive
    :return: The value on the tick grid
    """
    ticks = value / tick_size
    magnitude = abs(ticks)
    if magnitude >= _MAX_TICKS:
        count = _MAX_TICKS
    else:
        # The fraction is split off exactly: adding 0.5 first would round in the addition
        # (0.49999999999999994 ticks would become 1, and an odd count above 2**52 would
        # move to the next even one)
        count = magnitude // 1.0
        if magnitude - count >= 0.5:
            count += 1.0
    return _copysign(count, ticks) * tick_size


def _format_parsed(value: float, sides: tuple[_PatternSide, _PatternSide],
                   decimal_format: bool = False) -> str | None:
    """
    Format a finite double with an already parsed pattern.

    Applies the same digit handling as the general path: optional places lose their trailing
    zeros, required places are truncated-then-padded, required integer digits are zero filled,
    a mask with required decimals drops a zero integer part, and a ``,`` in the integer part
    groups the integer digits. The digits come from :func:`_round_digits_exact` for
    ``str.format`` and from :func:`_round_digits_fast` for ``str.tostring``. ``str.tostring``
    prints a result without a non-zero digit unsigned (``-0.001`` with ``'#.##'`` is ``0``),
    ``str.format`` keeps the sign (``-0``). Negative zero, ``str.format`` above 2**53, and for
    ``str.tostring`` a value whose ``repr`` is exponential, are not covered: the answer is
    ``None`` so the caller can fall back.

    :param value: A finite double, of either sign
    :param sides: The parsed pattern from :func:`_parse_precision`
    :param decimal_format: True for the ``str.format()`` rounding, False for ``str.tostring()``
    :return: The formatted string, or ``None`` when the value is not covered
    """
    negative = value < 0
    if negative:
        prefix, suffix, required_before, required_decimals, max_decimals, group = sides[1]
        value = -value
    else:
        prefix, suffix, required_before, required_decimals, max_decimals, group = sides[0]

    if decimal_format:
        if value >= _EXACT_INT:
            return None
        result = _round_digits_exact(value, max_decimals)
        if result is None:
            return None
    else:
        result = _round_digits_fast(value, max_decimals, required_decimals > 0)
        if result is None:
            return None

    if max_decimals > 0:
        if required_decimals == 0:
            result = result.rstrip('0').rstrip('.')
        else:
            int_text, _, decimal_part = result.partition('.')
            if len(decimal_part) > required_decimals:
                decimal_part = decimal_part[:required_decimals].rstrip('0')
            if decimal_part:
                result = int_text + '.' + decimal_part.ljust(required_decimals, '0')
            else:
                result = int_text

    if negative and not decimal_format and not result.strip('0.'):
        # Rounded to zero: str.tostring formats it through the positive subpattern
        return _format_parsed(value, (sides[0], sides[0]))

    if required_before > 0:
        int_text, dot, decimal_part = result.partition('.')
        result = int_text.zfill(required_before) + dot + decimal_part
    elif required_decimals > 0 and result.startswith('0.'):
        result = result[1:]

    if group:
        int_text, dot, decimal_part = result.partition('.')
        result = _group_digits(int_text, group) + dot + decimal_part

    return prefix + result + suffix


# noinspection PyProtectedMember
def _format_number(value: float | int | NA, fmt_type: str = '', precision: str = '#.###',
                   *, decimal_format: bool = False) -> str:
    """
    Format a number according to Pine rules.

    A fast path handles the shapes a running script hits millions of times -- a finite
    double with a digit mask, from ``str.tostring`` or ``str.format``, and ``str.tostring``
    with ``format.mintick`` -- by parsing the mask once per pattern string and rounding
    without ``Decimal``: on the digits of ``repr`` for ``str.tostring``, with the float
    formatter for ``str.format``. Every input it does not provably cover goes to
    :func:`_format_number_general`, which stays the semantic reference.

    :param value: Value to format
    :param fmt_type: Format type (integer, currency, percent, mintick, volume, price, inherit)
    :param precision: Custom precision format string (like '#.##')
    :param decimal_format: True for ``str.format()`` (Java DecimalFormat engine),
                           False for ``str.tostring()`` (chart number formatter)
    :return: Formatted string
    """
    # The range tests also reject nan and both infinities, which never reach the digit masks
    if decimal_format:
        if type(value) is float and not fmt_type and type(precision) is str \
                and -_EXACT_INT < value < _EXACT_INT:
            sides = _parse_precision(precision)
            if sides is not None:
                result = _format_parsed(value, sides, True)
                if result is not None:
                    return result
    elif type(value) is float and -1e16 < value < 1e16:
        # str.tostring prints zero unsigned; adding 0.0 turns -0.0 into 0.0
        value += 0.0
        if not fmt_type:
            # Only a plain str is a cache key the parser can take; anything else keeps the
            # general path's own error
            sides = _parse_precision(precision) if type(precision) is str else None
            if sides is not None:
                result = _format_parsed(value, sides)
                if result is not None:
                    return result
        elif fmt_type == _format.mintick:
            tick_size = _syminfo.mintick
            if type(tick_size) is float and 0.0 < tick_size < 1e16:
                sides = _parse_precision(_mintick_mask(tick_size))
                if sides is not None:
                    result = _format_parsed(_round_to_mintick(value, tick_size) + 0.0, sides)
                    if result is not None:
                        return result
    return _format_number_general(value, fmt_type, precision, decimal_format=decimal_format)


# noinspection PyProtectedMember
def _format_number_general(value: float | int | NA, fmt_type: str = '', precision: str = '#.###',
                           *, decimal_format: bool = False) -> str:
    """
    Format a number according to Pine rules.

    Format strings use # for optional digits and 0 for required digits:
    - #.## -> removes trailing zeros after decimal
    - #.00 -> keeps trailing zeros
    - # -> rounds to integer
    - 000.00 -> adds leading zeros, keeps trailing zeros
    - #,##0.00 -> groups the integer digits

    Special formats:
    - integer: rounds to integer, grouped
    - currency: $X.XX format
    - percent: ``str.tostring`` prints the value with two decimals and a '%'; ``str.format``
      multiplies by 100 and rounds to an integer
    - mintick: rounds to whole ticks, printed with the tick's decimals
    - volume: ``str.tostring`` adds K/M/B/T suffixes with up to three decimals
    - price: same as currency
    - inherit: uses script's precision

    ``str.tostring`` prints a result without a non-zero digit unsigned.

    :param value: Value to format
    :param fmt_type: Format type (integer, currency, percent, mintick, volume, price, inherit)
    :param precision: Custom precision format string (like '#.##')
    :param decimal_format: True for ``str.format()`` (Java DecimalFormat engine),
                           False for ``str.tostring()`` (chart number formatter)
    :return: Formatted string
    """
    if value is None or not (value == value):  # None, NA object or native nan
        # DecimalFormat prints NaN bare; the chart formatter keeps percent's '%'
        return "NaN%" if fmt_type == _format.percent and not decimal_format else "NaN"

    if isinf(value):
        # Infinity must never reach the digit patterns -- round() raises
        # OverflowError and Decimal raises InvalidOperation on it. TV's two number
        # formatters disagree on what to print, so they are handled separately.
        if not decimal_format:
            # str.tostring() uses the chart's number formatter: "NaN" for every
            # format but format.mintick, which stringifies the raw double. Percent
            # still appends '%', and volume its magnitude suffix, which for an
            # infinite magnitude is always 'T'.
            if fmt_type == _format.mintick:
                return "-Infinity" if value < 0 else "Infinity"
            if fmt_type == _format.percent:
                return "NaN%"
            if fmt_type == _format.volume:
                return "NaNT"
            return "NaN"
        # str.format() is Java DecimalFormat: the infinity symbol carries the
        # pattern's prefix and suffix ('$∞', '∞%', '+∞'), unlike NaN which is bare.
        if fmt_type in ('price', 'currency'):
            prefix, suffix = ('-$' if value < 0 else '$'), ''
        elif fmt_type == _format.percent:
            prefix, suffix = ('-' if value < 0 else ''), '%'
        elif fmt_type:
            prefix, suffix = ('-' if value < 0 else ''), ''
        else:
            subpatterns = precision.split(';')
            negative_subpattern = value < 0 and len(subpatterns) > 1
            chosen = subpatterns[1] if negative_subpattern else subpatterns[0]
            affix, _, suffix = _split_number_pattern(chosen)
            prefix = ('-' if value < 0 and not negative_subpattern else '') + affix
        return prefix + '∞' + suffix

    if not decimal_format:
        # str.tostring prints zero unsigned; adding 0.0 turns -0.0 into 0.0
        value += 0.0

    # Handle special formats first
    if fmt_type == _format.mintick:
        tick_size = _syminfo.mintick
        if not decimal_format:
            value = _round_to_mintick(value, tick_size) + 0.0
            precision = _mintick_mask(tick_size)
        else:
            value = round(value / tick_size) * tick_size  # type: ignore
            precision = '#.' + '#' * _mintick_decimals(tick_size)

    elif fmt_type == _format.inherit:
        assert lib._script is not None and lib._script.precision is not None
        precision = '#.' + '#' * int(lib._script.precision)

    elif fmt_type == _format.volume and not decimal_format:
        # The magnitude of the raw value picks the suffix, then the quotient keeps up to three
        # decimals: 1234.5678 is 1.235K, 999999.9 is 1000K, 999.5 is 1000 (measured)
        magnitude = abs(value)
        for divisor, unit in _VOLUME_UNITS:
            if magnitude >= divisor:
                return _format_number(value / divisor, precision='#.###') + unit
        return _format_number(value, precision='#')

    elif fmt_type == _format.volume:
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.2f}K"
        return str(int(value))

    elif fmt_type == 'integer':
        # MessageFormat's integer style is DecimalFormat '#,##0'
        return _format_number(value, precision='#,##0', decimal_format=decimal_format)

    elif fmt_type == _format.percent:
        if not decimal_format:
            # The chart formatter does not scale: 0.5 is 0.50% (measured)
            return _format_number(value, precision='0.00') + '%'
        # MessageFormat's percent style is DecimalFormat '#,##0%', which scales by 100
        return _format_number(value * 100, precision='#,##0', decimal_format=True) + '%'

    # Convert to Decimal for precise handling
    d = Decimal(str(value))

    if fmt_type == 'price' or fmt_type == 'currency':
        # Format as currency with 2 decimals
        d = d.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        return f"${d:,.2f}"

    # Pine number patterns follow Java DecimalFormat: an optional negative
    # subpattern after ';', and literal prefix/suffix (e.g. '+', '-', '$', 'R')
    # around the digit pattern. Pick the subpattern by sign, format the
    # magnitude with the bare digit pattern, and re-attach the affixes.
    prefix = ''
    pattern = precision
    subpatterns = precision.split(';')
    # DecimalFormat formats -0.0 as a negative number: its sign goes to the negative prefix
    # (str.tostring has already turned it into 0.0)
    negative = value < 0 or (value == 0 and _copysign(1.0, value) < 0.0)
    if negative:
        value = -value
        if len(subpatterns) > 1:
            chosen = subpatterns[1]
        else:
            chosen = subpatterns[0]
            prefix = '-'
    else:
        chosen = subpatterns[0]

    affix, precision, suffix = _split_number_pattern(chosen)
    prefix += affix

    # Parse format string
    if '.' in precision:
        before, after = precision.split('.')
    else:
        before, after = precision, ''

    # Count required digits before decimal
    required_before = len([c for c in before if c == '0'])

    # Count required vs optional digits after decimal
    required_decimals = len([c for c in after if c == '0'])
    max_decimals = len(after)

    # Format the number
    if max_decimals == 0:
        # Integer format
        result = _round_digits(value, 0, decimal_format)
    else:
        # Float format
        formatted = _round_digits(value, max_decimals, decimal_format)
        if required_decimals == 0:
            # Remove trailing zeros if all places are optional (#)
            formatted = formatted.rstrip('0').rstrip('.')
        else:
            # Keep required number of decimal places
            decimal_part = formatted.split('.')[1]
            if len(decimal_part) > required_decimals:
                # Remove trailing zeros after required places
                decimal_part = decimal_part[:required_decimals].rstrip('0')
                if decimal_part:
                    formatted = f"{formatted.split('.')[0]}.{decimal_part}"
                else:
                    formatted = formatted.split('.')[0]

        # Handle required decimal places
        if '.' in formatted and required_decimals > 0:
            decimal_part = formatted.split('.')[1]
            while len(decimal_part) < required_decimals:
                decimal_part += '0'
            result = f"{formatted.split('.')[0]}.{decimal_part}"
        else:
            result = formatted

    if negative and not decimal_format and not result.strip('0.'):
        # Rounded to zero: str.tostring formats it through the positive subpattern
        return _format_number_general(value, precision=pattern)

    # Handle leading zeros if needed (the sign, if any, is carried by ``prefix``)
    if required_before > 0:
        int_part = result.split('.')[0]
        while len(int_part) < required_before:
            int_part = '0' + int_part
        if '.' in result:
            result = f"{int_part}.{result.split('.')[1]}"
        else:
            result = int_part
    elif required_decimals > 0 and result.startswith('0.'):
        # A mask with required decimals drops the integer part when it is zero and the
        # pattern does not demand a digit there ('#.0' formats 0.5 as '.5'), the way Java's
        # DecimalFormat does. An all-'#' mask keeps the zero ('#.#' gives '0.5').
        result = result[1:]

    group = _grouping_size(before)
    if group:
        int_part, dot, decimal_part = result.partition('.')
        result = _group_digits(int_part, group) + dot + decimal_part

    return prefix + result + suffix


def _format_array_item(value: Any) -> str:
    """
    Format one array element of a ``str.format`` argument.

    MessageFormat prints an array through the elements' own ``toString``, so a float element
    is Java's ``Double.toString`` (``1.0``, ``1.0E21``, ``5.0E-4``), an int element its plain
    digits, and na ``NaN`` (measured).

    :param value: The element
    :return: The formatted element
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return _java_double_str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, NA) or value is None:
        return "NaN"
    return str(value)


def _format_value(value: Any) -> str:
    """
    Format a ``str.format`` argument that has no ``number`` style.

    A number goes through MessageFormat's default number style ``'#,##0.###'`` whatever its
    Python type, so ``7`` and ``7.0`` print alike and both group their thousands.

    :param value: The argument
    :return: The formatted argument
    """
    if isinstance(value, (list, _SequenceView)):
        return f"[{', '.join([_format_array_item(x) for x in value])}]"
    elif isinstance(value, str):
        return value
    elif isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, (int, float)):
        return _format_number(safe_convert.safe_float(value), precision=_MESSAGE_NUMBER,
                              decimal_format=True)
    elif isinstance(value, NA) or value is None:
        return "NaN"
    return str(value)


@lru_cache(maxsize=32)
def _datatime_fmt_tv2py(fmt: str) -> str:
    """
    Convert Pine format to Python format

    :param fmt: Pine format string
    :return: Python format string
    """
    # Handle escaped parts first
    escaped_parts = {}
    i = 0
    while "'" in fmt:
        start = fmt.find("'")
        end = fmt.find("'", start + 1)
        if end == -1:
            break
        key = f"__ESC{i}__"
        escaped_parts[key] = fmt[start + 1:end]
        fmt = fmt[:start] + key + fmt[end + 1:]
        i += 1

    # Format mapping
    mapping = {
        'yyyy': '%Y',  # Year
        'yy': '%y',
        'MM': '%m',  # Month
        'dd': '%d',  # Day
        'HH': '%H',  # Hour
        'hh': '%I',  # Hour (12)
        'mm': '%M',  # Minute
        'ss': '%S',  # Second
        'SSS': '%f',  # Milliseconds
        'aa': '%p',  # AM/PM
        'A': '%p',  # AM/PM
        'E': '%a',  # Weekday abbr
        'EEE': '%a',  # Weekday abbr
        'EEEE': '%A',  # Weekday
        'MMM': '%b',  # Month abbr
        'MMMM': '%B',  # Month name
        'z': '@',  # Timezone name (temp)
        'Z': '%z',  # Timezone
        '@': '%Z',  # Timezone name
    }

    # Sort by length for proper replacement
    patterns = sorted(mapping.keys(), key=len, reverse=True)

    # Replace patterns
    result = fmt
    for pattern in patterns:
        result = result.replace(pattern, mapping[pattern])

    # Restore escaped parts
    for key, value in escaped_parts.items():
        result = result.replace(key, value)

    return result


#
# Exported functions
#

# noinspection PyShadowingBuiltins
def contains(source: str | NA[str], str: str | NA[str]) -> PyneBool:
    """
    Returns true if the source string contains the str substring, false otherwise.

    :param source: Source string
    :param str: Substring to search for
    :return: True if the source string contains the str substring, na if either
        argument is na.
    """
    # na-propagation (Pine): a na source/substring yields na, never a search.
    # Without this guard ``str in na`` would loop forever, since NA.__getitem__
    # returns self for every index and the ``in`` operator falls back to the
    # sequence protocol.
    if isinstance(source, NA) or source is None or isinstance(str, NA) or str is None:
        return NA(bool)
    return str in source


# noinspection PyShadowingBuiltins
def endswith(source: str, str: str) -> bool:
    """
    Returns true if the source string ends with the substring specified in str, false otherwise.

    :param source: Source string
    :param str: Substring to search for
    :return: True if the source string ends with the substring specified in str, false otherwise.
    """
    return source.endswith(str)


# The four segments a Java MessageFormat pattern is cut into: the literal text
# between placeholders, then the argument index, the format type and the format
# style inside one.
_SEG_RAW, _SEG_INDEX, _SEG_TYPE, _SEG_STYLE = 0, 1, 2, 3

# A style naming one of these is a keyword, matched case-insensitively and
# ignoring surrounding space; anything else is a DecimalFormat pattern taken
# verbatim, where a leading or trailing space is a literal affix.
_NUMBER_STYLES = ('integer', 'currency', _format.percent, _format.mintick,
                  _format.volume, _format.price, _format.inherit)

# Java's Integer.parseInt accepts digits and an optional sign, but no surrounding
# space -- ``{0 ,number,#.#}`` is an error on TradingView, not index 0.
_ARG_INDEX_RE = re.compile(r'\+?\d+\Z')


@lru_cache(maxsize=128)
def _parse_format_pattern(pattern: str) -> tuple[str | tuple[str, str, str], ...]:
    """
    Split a ``str.format`` pattern into literal text and placeholders.

    A ``str`` piece is literal output; a 3-tuple is a placeholder's raw index,
    type and style segments. The scan is Java ``MessageFormat.applyPattern``:

    - In literal text ``''`` is one quote, a lone ``'`` toggles quoting, and while
      quoted both braces are literal -- ``'{'`` prints ``{``, ``'{{ticker}}'``
      prints ``{{ticker}}``. An unterminated quote makes the rest literal.
    - An unquoted ``}`` outside a placeholder is literal too; only ``{`` opens one.
    - Inside a placeholder quotes are copied through to the segment (the style is
      a DecimalFormat pattern with its own quoting), ``,`` moves to the next
      segment, nested braces nest, and a space is dropped only ahead of the type.

    The results are cached because Pine format patterns are compile-time constants
    re-formatted on every bar.

    :param pattern: The ``str.format`` pattern
    :return: Literal strings and (index, type, style) placeholder segments in order
    :raises ValueError: If the pattern ends inside a placeholder
    """
    parts: list[str | tuple[str, str, str]] = []
    seg = ['', '', '', '']
    part = _SEG_RAW
    in_quote = False
    brace_depth = 0
    i = 0
    n = len(pattern)

    while i < n:
        ch = pattern[i]
        if part == _SEG_RAW:
            if ch == "'":
                if i + 1 < n and pattern[i + 1] == "'":
                    seg[_SEG_RAW] += ch
                    i += 1
                else:
                    in_quote = not in_quote
            elif ch == '{' and not in_quote:
                if seg[_SEG_RAW]:
                    parts.append(seg[_SEG_RAW])
                    seg[_SEG_RAW] = ''
                seg[_SEG_INDEX] = seg[_SEG_TYPE] = seg[_SEG_STYLE] = ''
                part = _SEG_INDEX
            else:
                seg[_SEG_RAW] += ch
        elif in_quote:
            seg[part] += ch
            if ch == "'":
                in_quote = False
        elif ch == ',':
            if part < _SEG_STYLE:
                part += 1
            else:
                seg[part] += ch
        elif ch == '{':
            brace_depth += 1
            seg[part] += ch
        elif ch == '}':
            if brace_depth:
                brace_depth -= 1
                seg[part] += ch
            else:
                parts.append((seg[_SEG_INDEX], seg[_SEG_TYPE], seg[_SEG_STYLE]))
                part = _SEG_RAW
        elif ch == "'":
            in_quote = True
            seg[part] += ch
        elif not (ch == ' ' and part == _SEG_TYPE and not seg[_SEG_TYPE]):
            seg[part] += ch
        i += 1

    # An unclosed placeholder is an error, except with a brace still open inside
    # it -- Java drops that one silently instead of reporting it.
    if part != _SEG_RAW and not brace_depth:
        raise ValueError("Format pattern ends inside a placeholder")
    if seg[_SEG_RAW]:
        parts.append(seg[_SEG_RAW])
    return tuple(parts)


# noinspection PyPep8Naming,PyShadowingBuiltins
def format(formatString: str, *args: Any) -> str:
    """
    Converts the formatting string and value(s) into a formatted string.
    Supports:
    - Basic placeholders: {0}, {1}, etc
    - Number formats: {0,number,integer}, {0,number,currency}, {0,number,percent}
    - Custom precision: {0,number,#.#}
    - Pine-style array formatting: [item1, item2] without quotes for strings
    - Single quotes escape braces: '{' is a literal brace, '' is a literal quote

    :param formatString: Format pattern
    :param args: Values to format
    :return: Formatted string, or na if the format pattern is na
    :raises ValueError: If the pattern is malformed
    """
    # na-propagation (Pine): a na format pattern yields na (e.g. the TV
    # Technical Ratings idiom ``str.repeat("\n", 0) + "{0}"`` — the repeat
    # returns na, the concat propagates it, and str.format passes it through).
    if isinstance(formatString, NA) or formatString is None:
        return NA(str)

    out: list[str] = []
    for piece in _parse_format_pattern(formatString):
        if isinstance(piece, str):
            out.append(piece)
            continue

        index, arg_type, style = piece
        if not _ARG_INDEX_RE.match(index):
            raise ValueError(f"Invalid argument index: {index}")
        position = int(index)
        if position >= len(args):
            # An index with no argument behind it is echoed as its own placeholder
            # instead of failing -- a script that misnumbers one keeps running.
            out.append(f"{{{position}}}")
            continue
        value = args[position]

        if isinstance(value, NA) or value is None:
            out.append("NaN")
        elif arg_type.strip().lower() != 'number':
            out.append(_format_value(value))
        else:
            # The keyword lookup trims and lowercases, but a DecimalFormat pattern
            # is passed on untouched: '{0, number, #.#}' formats with " #.#" and
            # so prints a leading space, while '{0, number, integer}' does not.
            keyword = style.strip().lower()
            number = safe_convert.safe_float(value)
            if keyword in _NUMBER_STYLES:
                out.append(_format_number(number, decimal_format=True,
                                          fmt_type=keyword, precision=style))
            elif keyword:
                out.append(_format_number(number, decimal_format=True, precision=style))
            else:
                out.append(_format_number(number, decimal_format=True, precision=_MESSAGE_NUMBER))

    return ''.join(out)


# noinspection PyProtectedMember,PyShadowingNames,PyShadowingBuiltins
def format_time(time: int | NA[int], format: str | None = None,
                timezone: str | None = None) -> str:
    """
    Format timestamp according to format string and timezone

    :param time: UNIX timestamp in milliseconds
    :param format: Format string (Pine format)
    :param timezone: Timezone string (UTC±HHMM, GMT±HHMM or IANA name)
    :return: Formatted time string, or na when ``time`` is na
    """
    # na timestamp formats to na (Pine na-propagation)
    if time is None or not (time == time):  # is_na_arg
        return NA(str)

    # Default format
    fmt = format if format else "yyyy-MM-ddTHH:mm:ssZ"

    # Convert timestamp to datetime
    dt = datetime.fromtimestamp(time / 1000, UTC)

    # Convert timezone using _parse_timezone
    dt = dt.astimezone(_parse_timezone(timezone or _syminfo.timezone))

    # Convert format and apply
    py_fmt = _datatime_fmt_tv2py(fmt)
    return dt.strftime(py_fmt)


def length(string: str) -> PyneInt:
    """
    Returns an integer corresponding to the amount of chars in that string.

    :param string: String to get the length of
    :return: Amount of chars in the string
    """
    return pine_int(len(string))


def lower(source: str) -> str:
    """
    Returns a new string with all letters converted to lowercase.

    :param source: Source string
    :return: A new string with all letters converted to lowercase.
    """
    return source.lower()


# Matches a ``\z`` escape that is a real escape (an even number of preceding
# backslashes) rather than a literal ``\\z``. Captures the leading backslash
# pairs so they are preserved in the replacement.
_PINE_END_ANCHOR_RE = re.compile(r'(?<!\\)((?:\\\\)*)\\z')


def _pine_regex_to_python(regex: str) -> str:
    """
    Adapt a Pine (Java-flavoured) regular expression to Python's ``re`` syntax.

    Pine's ``str.match`` uses Java regex semantics, which accept ``\\z`` (end of
    input); Python's ``re`` has no ``\\z`` and raises ``PatternError: bad escape``
    for it. Java's ``\\z`` maps to Python's ``\\Z`` (end of string). Only a real
    escape is rewritten — a literal ``\\\\z`` is left untouched.

    :param regex: The Pine regular expression.
    :return: An equivalent Python regular expression.
    """
    return _PINE_END_ANCHOR_RE.sub(lambda m: m.group(1) + r'\Z', regex)


def match(source: str, regex: str) -> str:
    """
    Returns the new substring of the source string if it matches a regex regular expression, an empty string otherwise.

    :param source: Source string
    :param regex: Regular expression
    :return: New substring of the source string if it matches a regex regular expression, an empty string otherwise.
    """
    m = re.match(_pine_regex_to_python(regex), source)
    if m is None:
        return ""
    return m.group()


# noinspection PyShadowingBuiltins
def pos(source: str, str: str) -> PyneInt:
    """
    Returns the position of the first occurrence of the str string in the source string, 'na' otherwise.

    :param source: Source string
    :param str: Subtring to search for
    :return: Position of the first occurrence of the str string in the source string, 'na' otherwise.
    """
    res = source.find(str)
    if res == -1:
        return na_int
    return pine_int(res)


# noinspection PyShadowingNames
def repeat(source: str, repeat: int, separator: str = '') -> PyneStr:
    """
    Returns a new string consisting of the source string repeated the specified number of times,
    separated by the separator string.

    :param source: Source string
    :param repeat: Number of times to repeat the source string
    :param separator: Separator string
    :return: New string consisting of the source string repeated the specified number of times,
             separated by the separator string. A na source or repeat count — and a repeat
             count of zero — yields na; a na separator behaves as ''.
    """
    # na-propagation (Pine): without this guard ``[source] * repeat`` with a na
    # count evaluates through ``NA.__rmul__`` to na, and ``str.join(na)`` falls
    # back to the sequence protocol, which never terminates (NA.__getitem__
    # returns self for every index).
    if isinstance(source, NA) or source is None or repeat is None or not (repeat == repeat):
        return NA(str)
    if repeat <= 0:
        return NA(str)
    if isinstance(separator, NA) or separator is None:
        separator = ''
    return separator.join([source] * int(repeat))


def replace(source: str, target: str, replacement: str, occurrence=0) -> PyneStr:
    """
    Replaces the nth occurrence of target string with the replacement string in the source string.

    :param source: Source string
    :param target: Target string
    :param replacement: Replacement string
    :param occurrence: Occurrence to replace
    :return: New string with the nth occurrence of target string replaced with the replacement
             string, or na if source or target is na.
    """
    # na-propagation (Pine): a na source would reach ``source.find`` through
    # ``NA.__getattr__``, which answers na to every attribute and turns the scan below
    # into a silent non-result — same trap as in ``repeat``/``contains``.
    if isinstance(source, NA) or source is None or isinstance(target, NA) or target is None:
        return NA(str)
    # A negative occurrence is a compile error on TradingView (CE10039), so it can only
    # arrive from hand-written Pyne code; leaving the source untouched is the quiet option.
    if occurrence < 0:
        return source
    # An na occurrence is the first one. Measured: replace("aaa", "a", "-", na) == "-aa"
    # and replace("abc", "", "-", na) == "-abc".
    occurrence = _native_int_or(occurrence, 0)
    if not target:
        # An empty target is an insertion point rather than a match: the replacement lands
        # at the nth character position, clamped to the end of the source. Measured:
        # replace("abc", "", "-", 2) == "ab-c" and replace("abc", "", "-", 4) == "abc-".
        index = min(occurrence, len(source))
    else:
        # Occurrences are enumerated by an overlapping left-to-right scan, so "aa" occurs
        # at index 0 AND 1 in "aaa" — a split-based walk would only see the first one.
        # Measured: replace("aaa", "aa", "-", 1) == "a-".
        index = -1
        for _ in range(occurrence + 1):
            index = source.find(target, index + 1)
            if index < 0:
                return source
    return source[:index] + replacement + source[index + len(target):]


def replace_all(source: str, target: str, replacement: str) -> str:
    """
    Replaces each occurrence of the target string in the source string with the replacement string.

    :param source: The source string
    :param target: Target string
    :param replacement: Replacement string
    :return: New string with each occurrence of the target string replaced with the replacement string.
    """
    return source.replace(target, replacement)


def split(string: str, separator: str) -> list[str]:
    """
    Divides a string into an array of substrings and returns its array id.

    :param string: String to split
    :param separator: Separator
    :return: Array of substrings
    """
    if not separator:
        # An empty separator splits into individual characters, but an empty source still
        # yields one empty piece, exactly like every other separator does. Python raises
        # "empty separator" for both. Measured: split("abc", "") == ["a", "b", "c"] and
        # split("", "") == [""].
        return list(string) or [""]
    return string.split(separator)


# noinspection PyShadowingBuiltins
def startswith(source: str, str: str) -> bool:
    """
    Returns true if the source string starts with the str substring, false otherwise.

    :param source: The source string
    :param str: The substring to search for
    :return: True if the source string starts with the str substring, false otherwise
    """
    return source.startswith(str)


def substring(source: str, begin_pos: int, end_pos: int | None = None) -> PyneStr:
    """
    Returns a substring of the source string starting at the specified position and ending at the specified position.

    :param source: The source string
    :param begin_pos: The starting position
    :param end_pos: The ending position
    :return: The substring of the source string starting at the specified position and ending at
             the specified position, or na when the source is na
    """
    # na-propagation (Pine): an na source has no substring, and no length either --
    # the end position defaults to ``len(source)``, so the source has to be resolved
    # before any position is.
    if isinstance(source, NA) or source is None:
        return NA(str)
    # Pine's int is a static type only: an int-TYPED expression can carry a
    # fractional value (``14 / 8``), so the positions are truncated where they
    # are CONSUMED -- ahead of the range checks and of the empty-slice test,
    # both of which must see the same integer the slice below uses. Measured on
    # TradingView, the two positions answer na differently: substring("hello", na)
    # is "hello" -- an na start is 0 -- while substring("hello", 3, na) is "lo", so
    # an na end reaches the end of the source the way an omitted one does.
    begin_pos = _native_int_or(begin_pos, 0)
    assert begin_pos >= 0, "Positions must be >= 0!"
    if end_pos is not None:
        end_pos = _native_int_or(end_pos, len(source))
        assert end_pos >= begin_pos, "End position must be >= begin position!"
    if begin_pos == end_pos:
        return ""
    if end_pos is None:
        end_pos = len(source)
    return source[begin_pos:end_pos]


def tonumber(string: str) -> PyneFloat:
    """
    Converts a value represented in string to its "float" equivalent, or `na` if the conversion is not possible.

    :param string: Value to convert
    :return: Float equivalent of the value or `na` if the conversion is not possible.
    """
    try:
        return float(string)
    except ValueError:
        return na_float


# noinspection PyShadowingBuiltins,PyShadowingNames
def tostring(value: int | float | str | bool | NA | list | _SequenceView | _Matrix,
             format: str | Format | None = None) -> str:
    """
    Convert value to string with optional formatting.

    Without a format a number prints with up to ten decimals, in exponential notation outside
    ``(1e-21, 1e21)``. An array prints as ``[a, b]`` and a matrix as one such line per row,
    every element formatted like a single value with the same format.

    :param value: Value to convert (number, string, boolean, na, array or matrix)
    :param format: Format string like '#.##' or Format instance, None for the default
    :return: String representation
    """
    if type(value) is float:
        # The hot case: a native double (na included) needs no conversion and none of the
        # type tests below apply to it
        if format is None:
            return _format_default(value)
        if isinstance(format, Format):
            return _format_number(value, fmt_type=format)
        return _format_number(value, precision=format)
    if isinstance(value, NA) or value is None:
        return "NaN"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        number = safe_convert.safe_float(value)
        if format is None:
            return _format_default(number)
        if isinstance(format, Format):
            return _format_number(number, fmt_type=format)
        return _format_number(number, precision=format)
    if isinstance(value, (list, _SequenceView)):
        return f"[{', '.join([tostring(item, format) for item in value])}]"
    if isinstance(value, _Matrix):
        return '\n'.join([f"[{', '.join([tostring(item, format) for item in row])}]"
                          for row in value.data])
    return str(value)  # noqa: it may be reachable if it is used with unsupported types


def trim(source: str) -> str:
    """
    Removes leading and trailing whitespaces from the source string.

    :param source: Source string
    :return: Source string without leading and trailing whitespaces.
    """
    return source.strip()


def upper(source: str) -> str:
    """
    Returns a new string with all letters converted to uppercase.

    :param source: Source string
    :return: A new string with all letters converted to uppercase.
    """
    return source.upper()
