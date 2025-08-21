from typing import Any
import re

from functools import lru_cache

from datetime import datetime, UTC
from decimal import Decimal, ROUND_HALF_UP

from ..types.na import NA

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

# noinspection PyProtectedMember
def _format_number(value: float, fmt_type: str = '', precision: str = '#.###') -> str:
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
    :return: Formatted string
    """
    if isinstance(value, NA) or value is None:
        return "NaN"

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
        result = str(int(round(value)))
    else:
        # Float format
        formatted = f"{value:.{max_decimals}f}"
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

    # Handle leading zeros if needed
    if required_before > 0:
        int_part = result.split('.')[0].lstrip('-')  # Handle negative numbers
        is_negative = value < 0
        while len(int_part) < required_before:
            int_part = '0' + int_part
        if is_negative:
            int_part = '-' + int_part
        if '.' in result:
            result = f"{int_part}.{result.split('.')[1]}"
        else:
            result = int_part

    return result


def _format_value(value: Any, _no_format_numbers=False) -> str:
    """ Format a value in Pine-compatible way """
    if isinstance(value, list):
        res = f"[{', '.join(_format_value(x, _no_format_numbers=True) for x in value)}]"
        return res
    elif isinstance(value, str):
        return value
    elif isinstance(value, float):
        # Use default formatting for floats
        return _format_number(value) if not _no_format_numbers else str(value)
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

def contains(source: str, str_: str) -> bool:
    """
    Returns true if the source string contains the str substring, false otherwise.

    :param source: Source string
    :param str_: Substring to search for
    :return: True if the source string contains the str substring, false otherwise.
    """
    return str_ in source


def endswith(source: str, str_: str) -> bool:
    """
    Returns true if the source string ends with the substring specified in str, false otherwise.

    :param source: Source string
    :param str_: Substring to search for
    :return: True if the source string ends with the substring specified in str, false otherwise.
    """
    return source.endswith(str_)


# noinspection PyPep8Naming,PyShadowingBuiltins
def format(formatString: str, *args: Any) -> str:
    """
    Converts the formatting string and value(s) into a formatted string.
    Supports:
    - Basic placeholders: {0}, {1}, etc
    - Number formats: {0,number,integer}, {0,number,currency}, {0,number,percent}
    - Custom precision: {0,number,#.#}
    - Pine-style array formatting: [item1, item2] without quotes for strings
    - Special quote handling: single quotes are removed if unpaired, converted to single quote if paired

    :param formatString: Format pattern
    :param args: Values to format
    :return: Formatted string
    """

    # Pre-process quotes before handling placeholders
    def process_quotes(s: str) -> str:
        result = []
        i = 0
        while i < len(s):
            if s[i] == "'":
                # Look ahead for another quote
                if i + 1 < len(s) and s[i + 1] == "'":
                    result.append("'")
                    i += 2  # Skip both quotes
                else:
                    # Single quote - remove it
                    i += 1
                continue
            result.append(s[i])
            i += 1
        return ''.join(result)

    formatString = process_quotes(formatString)

    # noinspection PyShadowingNames
    def replace_field(match: re.Match) -> str:
        field = match.group(1).strip()
        parts = [p.strip() for p in field.split(',')]

        try:
            index = int(parts[0])
            value = args[index]
        except (ValueError, IndexError):
            raise ValueError(f"Invalid argument index: {parts[0]}")

        if isinstance(value, NA) or value is None:
            return "NaN"

        if len(parts) >= 2 and parts[1] == 'number':
            if len(parts) >= 3:
                return _format_number(safe_convert.safe_float(value),
                                      fmt_type=parts[2] if parts[2] in (
                                          'integer', 'currency',
                                          _format.percent,
                                          _format.mintick,
                                          _format.volume,
                                          _format.price,
                                          _format.inherit
                                      ) else '',
                                      precision=parts[2])
            return _format_number(safe_convert.safe_float(value))

        return _format_value(value)

    # Validate and balance curly braces
    stack = []
    in_placeholder = False
    for c in formatString:
        if c == '{' and not in_placeholder:
            stack.append(c)
            in_placeholder = True
        elif c == '}' and in_placeholder:
            if not stack:
                raise ValueError("Unmatched closing brace '}'")
            stack.pop()
            in_placeholder = False
        elif c == '}' and not in_placeholder:
            raise ValueError("Unmatched closing brace '}'")

    if stack:
        raise ValueError("Unmatched opening brace '{'")

    return re.sub(r'\{([^}]+)}', replace_field, formatString)


# noinspection PyProtectedMember
def format_time(time: int, fmt: str | None = None, tz: str | None = None) -> str:
    """
    Format timestamp according to format string and timezone

    :param time: UNIX timestamp in milliseconds
    :param fmt: Format string (Pine format)
    :param tz: Timezone string (UTC±HHMM, GMT±HHMM or IANA name)
    :return: Formatted time string
    """
    # Default format
    if not fmt:
        fmt = "yyyy-MM-ddTHH:mm:ssZ"

    # Convert timestamp to datetime
    dt = datetime.fromtimestamp(time / 1000, UTC)

    # Convert timezone using _parse_timezone
    dt = dt.astimezone(_parse_timezone(tz or _syminfo.timezone))

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


def match(source: str, regex: str) -> str:
    """
    Returns the new substring of the source string if it matches a regex regular expression, an empty string otherwise.

    :param source: Source string
    :param regex: Regular expression
    :return: New substring of the source string if it matches a regex regular expression, an empty string otherwise.
    """
    m = re.match(regex, source)
    if m is None:
        return ""
    return m.group()


def pos(source: str, str_: str) -> int | NA[int]:
    """
    Returns the position of the first occurrence of the str string in the source string, 'na' otherwise.

    :param source: Source string
    :param str_: Subtring to search for
    :return: Position of the first occurrence of the str string in the source string, 'na' otherwise.
    """
    res = source.find(str_)
    if res == -1:
        return NA(int)
    return res


# noinspection PyShadowingNames
def repeat(source: str, repeat: int, separator: str = '') -> str:
    """
    Returns a new string consisting of the source string repeated the specified number of times,
    separated by the separator string.

    :param source: Source string
    :param repeat: Number of times to repeat the source string
    :param separator: Separator string
    :return: New string consisting of the source string repeated the specified number of times,
             separated by the separator string.
    """
    return separator.join([source] * repeat)


def replace(source: str, target: str, replacement: str, occurence=0) -> str:
    """
    Replaces the nth occurence of target string with the replacement string in the source string.

    :param source: Source string
    :param target: Target string
    :param replacement: Replacement string
    :param occurence: Occurence to replace
    :return: New string with the nth occurence of target string replaced with the replacement string.
    """
    if occurence == 0:
        return source.replace(target, replacement, 1)
    a = source.split(target)
    p1 = target.join(a[:occurence])
    p2 = target.join(a[occurence:])
    if p2 == "":
        return source
    return p1 + replacement + p2


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
    return string.split(separator)


def startswith(source: str, str_: str) -> bool:
    """
    Returns true if the source string starts with the str substring, false otherwise.

    :param source: The source string
    :param str_: The substring to search for
    :return: True if the source string starts with the str substring, false otherwise
    """
    return source.startswith(str_)


def substring(source: str, begin_pos: int, end_pos: int | None = None) -> str:
    """
    Returns a substring of the source string starting at the specified position and ending at the specified position.

    :param source: The source string
    :param begin_pos: The starting position
    :param end_pos: The ending position
    :return: The substring of the source string starting at the specified position and ending at the specified position
    """
    assert begin_pos >= 0, "Positions must be >= 0!"
    if end_pos is not None:
        assert end_pos >= begin_pos, "End position must be >= begin position!"
    if begin_pos == end_pos:
        return ""
    if end_pos is None:
        end_pos = len(source)
    return source[begin_pos:end_pos]


def tonumber(string: str) -> float | NA[float]:
    """
    Converts a value represented in string to its "float" equivalent, or `na` if the conversion is not possible.

    :param string: Value to convert
    :return: Float equivalent of the value or `na` if the conversion is not possible.
    """
    try:
        return float(string)
    except ValueError:
        return NA(float)


# noinspection PyShadowingBuiltins,PyShadowingNames
def tostring(value: int | float | str | bool | NA, fmt: str | Format = '#.##########') -> str:
    """
    Convert value to string with optional formatting.
    Replicates Pine's str.tostring function.

    :param value: Value to convert (number, string, boolean or na)
    :param fmt: Format string like '#.##' or Format instance
    :return: String representation
    """
    if isinstance(value, NA) or value is None:
        return "NaN"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(fmt, Format):
            return _format_number(safe_convert.safe_float(value), fmt_type=fmt)
        return _format_number(safe_convert.safe_float(value), precision=fmt)
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
