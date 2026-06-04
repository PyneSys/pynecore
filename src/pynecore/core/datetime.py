import re
import sys
from zoneinfo import ZoneInfo
from datetime import datetime, UTC
from functools import cache, lru_cache
from ..lib import syminfo

# Standard formats for non-ISO dates
# %b = abbreviated month (Jan, Feb), %B = full month (January, February)
STANDARD_FORMATS = [
    "%d %b %Y %H:%M:%S %z",  # "20 Feb 2020 15:30:00 +0200"
    "%d %b %Y %H:%M %z",  # "01 Jan 2018 00:00 +0000"
    "%d %B %Y %H:%M:%S %z",  # "20 February 2020 15:30:00 +0200"
    "%d %B %Y %H:%M %z",  # "1 January 2018 00:00 +0000"
]

# Pine Script specific formats (without timezone)
# %b = abbreviated month (Jan, Feb), %B = full month (January, February)
PINE_FORMATS = [
    "%b %d %Y %H:%M:%S",  # "Feb 01 2020 22:10:05"
    "%d %b %Y %H:%M:%S",  # "04 Dec 1995 00:12:00"
    "%d %b %Y %H:%M",  # "01 Jan 2018 00:00"
    "%b %d %Y",  # "Feb 01 2020"
    "%d %b %Y",  # "04 Dec 1995"
    "%B %d %Y %H:%M:%S",  # "February 01 2020 22:10:05"
    "%d %B %Y %H:%M:%S",  # "04 December 1995 00:12:00"
    "%d %B %Y %H:%M",  # "01 January 2018 00:00"
    "%B %d %Y",  # "February 01 2020"
    "%d %B %Y",  # "04 December 1995"
    "%Y-%m-%d"  # "2020-02-20"
]


def normalize_timezone(datestring: str) -> str:
    """
    Normalize timezone format to be compatible with Python's datetime.
    Converts formats like "+00:00" to "+0000"

    :param datestring: Input date string
    :return: Normalized date string
    """
    tz_match = re.search(r'([+-])(\d{2}):(\d{2})(?:\s|$)', datestring)
    if tz_match:
        sign, hours, minutes = tz_match.groups()
        new_tz = f"{sign}{hours}{minutes}"
        return datestring[:tz_match.start()] + new_tz + datestring[tz_match.end():]
    return datestring


# Matches UTC/GMT±HHMM offset forms with optional colon: "UTC-5", "GMT+0530", "+05:30"
_OFFSET_RE = re.compile(r'^(UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2})?)?$')


class TimezoneNotFoundError(ValueError):
    """
    Raised when a timezone string cannot be resolved to a ``ZoneInfo``.

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers keep
    working, but is a distinct type so callers (e.g. :func:`pynecore.lib.time`)
    can surface it as an actionable error instead of silently degrading to ``na``.
    """


@cache
def _timezone_db_available() -> bool:
    """
    Return whether an IANA timezone database is reachable on this system.

    Probes a canonical zone name. On Windows without the ``tzdata`` package and
    without a system zoneinfo database, even standard names fail to resolve.

    :return: True if standard IANA names can be resolved
    """
    try:
        ZoneInfo("America/New_York")
        return True
    except Exception:  # noqa - any failure means the database is unusable
        return False


def _missing_timezone_message(timezone: str) -> str:
    """
    Build an actionable error message for an unresolved timezone when the IANA
    database is missing.

    :param timezone: The timezone string that could not be resolved
    :return: Multi-line, platform-aware error message
    """
    lines = [
        f"Timezone {timezone!r} could not be resolved: the IANA timezone database "
        "is not available on this system.",
        "",
        "Install it with:",
        "    pip install tzdata",
    ]
    if sys.platform.startswith("win"):
        lines += [
            "",
            "Windows has no built-in timezone database, so the 'tzdata' package is "
            "required. PyneCore's [cli] and [all] installs include it automatically.",
        ]
    return "\n".join(lines)


@lru_cache(maxsize=128)
def parse_timezone(timezone: str | None) -> ZoneInfo:
    """
    Parse timezone string into ZoneInfo object. Supports:
    - IANA timezone names (e.g. "America/New_York")
    - UTC±HHMM format (e.g. "UTC-5", "UTC+0530")
    - GMT±HHMM format (e.g. "GMT-5", "GMT+0530")
    - Raw offset (e.g. "+0530", "-05:00")

    :param timezone: Timezone string
    :return: ZoneInfo object
    :raises ValueError: If timezone format is invalid
    """
    if not timezone and syminfo.timezone:
        timezone = syminfo.timezone

    # Try as IANA timezone first
    try:
        try:
            return ZoneInfo(timezone)  # type: ignore
        except TypeError:  # timezone is None and syminfo has none -> default to UTC
            return ZoneInfo('UTC')
    except KeyError:
        # ZoneInfoNotFoundError is a KeyError subclass: the name is not in the IANA
        # database. UTC/GMT±HHMM offset forms are parsed below; any other name is an
        # IANA name whose lookup genuinely failed.
        pass

    # ZoneInfo raised KeyError, which only happens for a string key (None/NA raise
    # TypeError, handled above), so timezone is a str from here on.
    assert isinstance(timezone, str)

    # Parse UTC/GMT±HHMM offset format with optional colon
    match = _OFFSET_RE.match(timezone)
    if match is None:
        # Not an offset form -> the timezone name could not be resolved. The most
        # common cause is a missing IANA database (Windows ships none by default).
        if not _timezone_db_available():
            raise TimezoneNotFoundError(_missing_timezone_message(timezone))
        raise TimezoneNotFoundError(
            f"Unknown timezone {timezone!r}. Use a valid IANA name "
            "(e.g. 'America/New_York') or a UTC/GMT±HHMM offset (e.g. 'UTC-5', 'GMT+0530')."
        )

    prefix, sign, hours, minutes = match.groups()
    offset = int(hours)
    if minutes:
        offset += int(minutes) / 60

    # UTC/GMT+X maps to Etc/GMT-X and vice versa
    # Special case: offset 0 should use UTC directly
    if offset == 0:
        return ZoneInfo("UTC")
    zone = f"Etc/GMT{'-' if sign == '+' else '+'}{int(abs(offset))}"
    return ZoneInfo(zone)


def parse_datestring(datestring: str) -> datetime:
    """
    Parse date string using multiple formats.
    Handles ISO 8601 with microseconds and timezone offsets.
    If no time is supplied, "00:00" is used.
    If no timezone is supplied, GMT+0 is used.

    :param datestring: Date string to parse
    :return: Parsed datetime object
    :raises ValueError: If the date format is invalid
    """
    datestring = datestring.strip()
    if not datestring:
        return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    # Try parsing ISO 8601 style dates WITH TIME first (handles both T and space separator)
    iso_match = re.match(
        r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?)'  # datetime part
        r'([+-]\d{2}:\d{2})?$',  # timezone part
        datestring
    )
    if iso_match:
        dt_part, tz_part = iso_match.groups()
        if tz_part:
            datestring = normalize_timezone(datestring)
            dt_str = datestring.replace(' ', 'T')  # Normalize to T for parsing
            try:
                return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f%z")
            except ValueError:
                return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S%z")
        else:
            try:
                dt = datetime.strptime(dt_part.replace(' ', 'T'), "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                dt = datetime.strptime(dt_part.replace(' ', 'T'), "%Y-%m-%dT%H:%M:%S")
            return dt.replace(tzinfo=UTC)

    # Try parsing ISO 8601 DATE ONLY format (YYYY-MM-DD) before timezone extraction
    # This prevents the timezone regex from incorrectly matching date parts like -09 in 2025-01-09
    iso_date_match = re.match(r'^\d{4}-\d{2}-\d{2}$', datestring)
    if iso_date_match:
        dt = datetime.strptime(datestring, "%Y-%m-%d")
        # Use exchange timezone (from syminfo) when no timezone is specified
        default_tz = parse_timezone(None)  # This will return syminfo.timezone
        return dt.replace(tzinfo=default_tz)

    # Extract timezone if present at the end for other formats
    # The regex requires whitespace before timezone to avoid matching date parts
    tz_match = re.search(r'\s+((?:UTC|GMT)?[+-]\d{1,2}(?::?\d{2})?)\s*$', datestring)
    if tz_match:
        tz = parse_timezone(
            f"UTC{tz_match.group(1)}" if not tz_match.group(1).startswith(('UTC', 'GMT')) else tz_match.group(1))
        datestring = datestring[:tz_match.start()].strip()
    else:
        # Use exchange timezone (from syminfo) when no timezone is specified
        tz = parse_timezone(None)  # This will return syminfo.timezone

    # Try standard formats (with timezone)
    if tz_match:
        normalized = normalize_timezone(f"{datestring} {tz_match.group(1)}")
        for fmt in STANDARD_FORMATS:
            try:
                return datetime.strptime(normalized, fmt)
            except ValueError:
                continue

    # Try Pine formats (without timezone)
    for fmt in PINE_FORMATS:
        try:
            dt = datetime.strptime(datestring, fmt)
            return dt.replace(tzinfo=tz)
        except ValueError:
            continue

    raise ValueError(
        f"Invalid date format: {datestring}\n"
        "Supported formats:\n"
        "- ISO Style: '2020-02-20T15:30:00+02:00', '2025-01-01 01:23:45-05:00'\n"
        "- With fraction: '2024-08-01T04:38:47.731215+00:00'\n"
        "- RFC Style: '20 Feb 2020 15:30:00 GMT+0200', '1 January 2018 00:00 +0000'\n"
        "- Simple Pine: 'Feb 01 2020 22:10:05', '1 January 2018', '2020-02-20'"
    )
