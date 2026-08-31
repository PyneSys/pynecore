from typing import Any
import re

from functools import lru_cache
from math import isinf

from datetime import datetime, UTC
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN, localcontext

from ..types.na import NA, na_float
from ..types.pine_types import PyneFloat, PyneInt, PyneStr, PyneBool

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

    ``str.tostring`` then rounds those digits half-up, while ``str.format`` is Java's
    DecimalFormat, which breaks a tie on the exact value -- 2.675 is really 2.67499...,
    so it rounds down where ``str.tostring`` rounds up.

    :param value: The value to round (must be non-negative and finite)
    :param decimals: Number of fraction digits the mask allows
    :param decimal_format: True for ``str.format()``, False for ``str.tostring()``
    :return: The rounded value in plain decimal notation
    """
    if value >= 1e-3:
        decimals = min(decimals, 16)
    if decimal_format:
        d = Decimal(value)  # The double's exact value decides the ties
        rounding = ROUND_HALF_EVEN
    else:
        d = Decimal(repr(value))  # Shortest round-trip digits, like Double.toString
        rounding = ROUND_HALF_UP
    with localcontext() as ctx:
        ctx.prec = _MAX_INT_DIGITS + decimals
        return f"{d.quantize(Decimal(1).scaleb(-decimals), rounding=rounding):f}"


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


# noinspection PyProtectedMember
def _format_number(value: float | int | NA, fmt_type: str = '', precision: str = '#.###',
                   *, decimal_format: bool = False) -> str:
    """
    Format a number according to Pine rules.

    Format strings use # for optional digits and 0 for required digits:
    - #.## -> removes trailing zeros after decimal
    - #.00 -> keeps trailing zeros
    - # -> rounds to integer
    - 000.00 -> adds leading zeros, keeps trailing zeros

    Special formats:
    - integer: rounds to integer
    - currency: $X.XX format
    - percent: adds % and multiplies by 100
    - mintick: rounds to symbol's mintick
    - volume: adds K/M/B suffixes
    - price: same as currency
    - inherit: uses script's precision

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

    # Handle special formats first
    if fmt_type == _format.mintick:
        tick_size = _syminfo.mintick
        value = round(value / tick_size) * tick_size  # type: ignore
        # Get decimal places from mintick
        tick_str = str(tick_size)
        if 'e' in tick_str or 'E' in tick_str:
            # Handle scientific notation
            # Convert to decimal format to count decimal places
            tick_decimal = f"{tick_size:.20f}".rstrip('0')
            if '.' in tick_decimal:
                dec_str = tick_decimal.split('.')[1]
            else:
                dec_str = ''
        else:
            # Handle regular decimal notation
            if '.' in tick_str:
                dec_str = tick_str.rstrip('0').split('.')[1]
            else:
                dec_str = ''
        precision = '#.' + '#' * len(dec_str)

    elif fmt_type == _format.inherit:
        assert lib._script is not None and lib._script.precision is not None
        precision = '#.' + '#' * int(lib._script.precision)

    elif fmt_type == _format.volume:
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.2f}K"
        return str(int(value))

    elif fmt_type == 'integer':
        return str(int(round(value)))

    elif fmt_type == _format.percent:
        return f"{value * 100:.0f}%"

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
    subpatterns = precision.split(';')
    if value < 0:
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

    return prefix + result + suffix


def _format_value(value: Any, _no_format_numbers=False) -> str:
    """ Format a value in Pine-compatible way """
    if isinstance(value, list):
        res = f"[{', '.join(_format_value(x, _no_format_numbers=True) for x in value)}]"
        return res
    elif isinstance(value, str):
        return value
    elif isinstance(value, float):
        # Use default formatting for floats
        return _format_number(value, decimal_format=True) if not _no_format_numbers else str(value)
    elif isinstance(value, bool):
        return str(value).lower()
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
                out.append(_format_number(number, decimal_format=True))

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
    if isinstance(time, NA) or time is None:
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


def length(string: str) -> int:
    """
    Returns an integer corresponding to the amount of chars in that string.

    :param string: String to get the length of
    :return: Amount of chars in the string
    """
    return len(string)


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
        return NA(int)
    return res


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
    if isinstance(source, NA) or source is None or isinstance(repeat, NA) or repeat is None:
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
    if not target:
        # An empty target is an insertion point rather than a match: the replacement lands
        # at the nth character position, clamped to the end of the source. Measured:
        # replace("abc", "", "-", 2) == "ab-c" and replace("abc", "", "-", 4) == "abc-".
        index = min(int(occurrence), len(source))
    else:
        # Occurrences are enumerated by an overlapping left-to-right scan, so "aa" occurs
        # at index 0 AND 1 in "aaa" — a split-based walk would only see the first one.
        # Measured: replace("aaa", "aa", "-", 1) == "a-".
        index = -1
        for _ in range(int(occurrence) + 1):
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


def substring(source: str, begin_pos: int, end_pos: int | None = None) -> str:
    """
    Returns a substring of the source string starting at the specified position and ending at the specified position.

    :param source: The source string
    :param begin_pos: The starting position
    :param end_pos: The ending position
    :return: The substring of the source string starting at the specified position and ending at the specified position
    """
    # Pine's int is a static type only: an int-TYPED expression can carry a
    # fractional value (``14 / 8``), so the positions are truncated where they
    # are CONSUMED -- ahead of the range checks and of the empty-slice test,
    # both of which must see the same integer the slice below uses.
    begin_pos = int(begin_pos)
    assert begin_pos >= 0, "Positions must be >= 0!"
    if end_pos is not None:
        end_pos = int(end_pos)
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
def tostring(value: int | float | str | bool | NA, format: str | Format = '#.##########') -> str:
    """
    Convert value to string with optional formatting.

    :param value: Value to convert (number, string, boolean or na)
    :param format: Format string like '#.##' or Format instance
    :return: String representation
    """
    if isinstance(value, NA) or value is None:
        return "NaN"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(format, Format):
            return _format_number(safe_convert.safe_float(value), fmt_type=format)
        return _format_number(safe_convert.safe_float(value), precision=format)
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
