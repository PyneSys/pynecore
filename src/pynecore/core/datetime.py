import re
import sys
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone as dt_timezone, tzinfo, UTC
from functools import cache, lru_cache

# Standard formats for non-ISO dates
# %b = abbreviated month (Jan, Feb), %B = full month (January, February)
STANDARD_FORMATS = [
    "%d %b %Y %H:%M:%S %z",  # "20 Feb 2020 15:30:00 +0200"
    "%d %b %Y %H:%M %z",  # "01 Jan 2018 00:00 +0000"
    "%d %B %Y %H:%M:%S %z",  # "20 February 2020 15:30:00 +0200"
    "%d %B %Y %H:%M %z",  # "1 January 2018 00:00 +0000"
    "%Y-%m-%d %H:%M:%S %z",  # "2021-01-01 00:00:00 +0000"
    "%Y-%m-%d %H:%M %z",  # "2021-01-01 00:00 +0000"
    "%m %d %Y %H:%M:%S %z",  # "05 12 2000 10:20:30 +0000" (month-first)
    "%m %d %Y %H:%M %z",  # "01 1 2000 00:00 +0000" (month-first)
    "%m %d %Y %z",  # "01 1 2000 +0000" (month-first)
]

# Pine Script specific formats (without timezone)
# %b = abbreviated month (Jan, Feb), %B = full month (January, February)
# Numeric dates are MONTH-FIRST (MM-DD-YYYY) with '-', '/', '.' or ' ' separators:
# TradingView parses "03-04-2023" (and "05 12 2000") as March 4 / May 12 and
# rejects a day-first "13-04-2023" ("31 1 2000") outright ("timestamp(s):
# unrecognized datetime format"), so there is intentionally no day-first
# fallback here.
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
    "%b %Y",  # "Jan 2025" (day defaults to the 1st)
    "%B %Y",  # "January 2025" (day defaults to the 1st)
    "%Y-%m-%d",  # "2020-02-20"
    "%Y-%m-%d %H:%M:%S",  # "2021-01-01 00:00:00"
    "%Y-%m-%d %H:%M",  # "2021-01-01 00:00"
    "%m %d %Y %H:%M:%S",  # "05 12 2000 10:20:30"
    "%m %d %Y %H:%M",  # "01 1 2000 00:00"
    "%m %d %Y",  # "05 12 2000", "3 4 2023"
    "%m-%d-%Y %H:%M:%S",  # "03-04-2023 10:20:30"
    "%m-%d-%Y %H:%M",  # "03-04-2023 10:20"
    "%m-%d-%Y",  # "03-04-2023", "3-4-2023"
    "%m/%d/%Y %H:%M:%S",  # "03/04/2023 10:20:30"
    "%m/%d/%Y %H:%M",  # "03/04/2023 10:20"
    "%m/%d/%Y",  # "03/04/2023"
    "%m.%d.%Y %H:%M:%S",  # "03.04.2023 10:20:30"
    "%m.%d.%Y %H:%M",  # "03.04.2023 10:20"
    "%m.%d.%Y",  # "03.04.2023"
]

# Zone names a date string may END with, as their UTC offsets: the RFC 2822 names
# plus "Z", in any letter case. Any other trailing name ("CET", a military letter
# like "A", an IANA name) is rejected.
_ZONE_NAME_OFFSETS = {
    "UT": "+0000", "UTC": "+0000", "GMT": "+0000", "Z": "+0000",
    "EST": "-0500", "EDT": "-0400", "CST": "-0600", "CDT": "-0500",
    "MST": "-0700", "MDT": "-0600", "PST": "-0800", "PDT": "-0700",
}

_MONTH_NAMES = frozenset((
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december",
))


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
def _parse_timezone_cached(timezone: str) -> ZoneInfo:
    """
    Parse a concrete, non-empty timezone string into a ZoneInfo object.

    Kept separate from :func:`parse_timezone` so the cache is only ever keyed on
    an explicit timezone string. The ``None`` -> exchange-timezone fallback must
    NOT be cached: it resolves against the mutable ``syminfo.timezone`` global,
    so a cached ``None`` entry would leak one script's timezone into the next run
    in the same process.

    :param timezone: Concrete timezone string (IANA name or UTC/GMT±HHMM offset)
    :return: ZoneInfo object
    :raises TimezoneNotFoundError: If the timezone cannot be resolved
    """
    # Try as IANA timezone first
    try:
        return ZoneInfo(timezone)
    except KeyError:
        # ZoneInfoNotFoundError is a KeyError subclass: the name is not in the IANA
        # database. UTC/GMT±HHMM offset forms are parsed below; any other name is an
        # IANA name whose lookup genuinely failed.
        pass

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


# Lazily bound to the ``lib.syminfo`` module on first use. Importing it at module
# top would create a datetime <-> lib import cycle (lib pulls in timeframe, which
# imports parse_timezone), so the reference is fetched once on the first fallback
# call and reused -- keeping the hot path a plain attribute read with no per-call
# import cost.
_syminfo = None


def parse_timezone(timezone: str | None) -> ZoneInfo:
    """
    Parse timezone string into ZoneInfo object. Supports:
    - IANA timezone names (e.g. "America/New_York")
    - UTC±HHMM format (e.g. "UTC-5", "UTC+0530")
    - GMT±HHMM format (e.g. "GMT-5", "GMT+0530")
    - Raw offset (e.g. "+0530", "-05:00")

    When ``timezone`` is falsy the exchange timezone (``syminfo.timezone``) is
    used, defaulting to UTC when that is unset too. This fallback value is read on
    every call -- never cached -- so changing the active symbol's timezone takes
    effect immediately instead of returning a previous run's cached zone.

    :param timezone: Timezone string, or None to use the exchange timezone
    :return: ZoneInfo object
    :raises TimezoneNotFoundError: If the timezone cannot be resolved
    """
    if not timezone:
        global _syminfo
        if _syminfo is None:
            from ..lib import syminfo
            _syminfo = syminfo
        timezone = _syminfo.timezone or 'UTC'
    return _parse_timezone_cached(timezone)


def parse_datestring(datestring: str) -> datetime:
    """
    Parse date string using multiple formats.
    Handles ISO 8601 with microseconds and timezone offsets.
    If no time is supplied, "00:00" is used.
    If no timezone is supplied, GMT+0 is used, whatever the exchange timezone is.
    Words in front of the date (a weekday, a zone name) are skipped; a zone name
    after the date is honoured.

    :param datestring: Date string to parse
    :return: Parsed datetime object
    :raises ValueError: If the date format is invalid
    """
    datestring = datestring.strip()
    if not datestring:
        return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    # Leading words carry no meaning -- measured on NASDAQ:AAPL: "UTC 01 Jan 2022
    # 00:00", "EST ...", "America/New_York ...", "Sat, ...", "Foo ..." and "hello
    # world ..." all resolve to 2022-01-01 00:00 UTC, and "UTC 01 Jan 2022 00:00
    # +0300" to 21:00 UTC the day before. A word is digit-free and unsigned
    # ("UTC01 Jan 2022" is rejected); a month name starts the date itself. After
    # such a word the ISO date/time form is rejected ("UTC 2022-01-01T05:00") while
    # a bare ISO date is not ("UTC 2022-01-01").
    words = 0
    while True:
        lead = re.match(r'([^\s\d+-]\S*)\s+', datestring)
        if lead is None or lead.group(1).lower() in _MONTH_NAMES:
            break
        datestring = datestring[lead.end():]
        words += 1

    # A trailing zone name is its offset -- measured on NASDAQ:AAPL:
    # "01 Jan 2022 00:00 EST" / "est" resolves to 05:00 UTC, "PDT" to 07:00 UTC,
    # "UT" and "Z" to 00:00 UTC, while "CET", "A" and "America/New_York" are
    # rejected.
    zone = re.search(r'\s+([A-Za-z]+)$', datestring)
    if zone is not None and (offset := _ZONE_NAME_OFFSETS.get(zone.group(1).upper())):
        datestring = f"{datestring[:zone.start()]} UTC{offset}"

    # Try parsing ISO 8601 style dates WITH TIME first. The date and the time may
    # be separated by "T", a space or a colon, and the hour needs no zero padding;
    # seconds are optional. TradingView accepts every one of those spellings --
    # measured: "2021-01-13:05:00", "2021-01-13:5:00", "2021-01-13:05:00:00" and
    # "2021-01-13T05:00" all resolve to 2021-01-13 05:00 UTC, while a missing
    # separator ("2021-01-1305:00"), a letter one ("2021-01-13x05:00") and a
    # minute-less time ("2021-01-13:05") are rejected outright. The offset is
    # four digits with an optional colon, glued to the time -- measured:
    # "2026-08-01T11:00:00-0500" and "...-05:00" both resolve to 16:00 UTC and
    # "...+0530" to 05:30 UTC, while an hour-only offset ("-05", "-5") and "Z"
    # are rejected.
    iso_match = re.match(
        r'(\d{4}-\d{2}-\d{2})'  # date part
        r'[T\s:]'  # date/time separator
        r'(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)'  # time part
        r'([+-]\d{2}:?\d{2})?$',  # timezone part
        datestring
    )
    if iso_match and not words:
        date_part, time_part, tz_part = iso_match.groups()
        dt_str = f"{date_part}T{time_part}"
        if tz_part:
            dt_str = normalize_timezone(dt_str + tz_part)
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M%z"):
                try:
                    return datetime.strptime(dt_str, fmt)
                except ValueError:
                    continue
        else:
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
                try:
                    return datetime.strptime(dt_str, fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue

    # Try parsing ISO 8601 DATE ONLY format (YYYY-MM-DD) before timezone extraction
    # This prevents the timezone regex from incorrectly matching date parts like -09 in 2025-01-09
    iso_date_match = re.match(r'^\d{4}-\d{2}-\d{2}$', datestring)
    if iso_date_match:
        return datetime.strptime(datestring, "%Y-%m-%d").replace(tzinfo=UTC)

    # Year-only and year-month dates, where TradingView fills the missing components
    # in with the start of the period -- measured: "2025" -> 2025-01-01 00:00 and
    # "2025-06" / "2025-6" / "2025/06" / "2025.06" -> 2025-06-01 00:00. Only the
    # year-first spellings are accepted: a leading month ("06 2025") is rejected by
    # TradingView outright, so it must keep raising here too.
    partial_match = re.match(r'^(\d{4})(?:[-/.](\d{1,2}))?$', datestring)
    if partial_match:
        year, month = partial_match.groups()
        return datetime(int(year), int(month) if month else 1, 1, tzinfo=UTC)

    # Extract timezone if present at the end for other formats
    # The regex requires whitespace before timezone to avoid matching date parts
    tz_match = re.search(r'\s+((UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?)\s*$', datestring)
    tz: tzinfo | None
    if tz_match:
        _, prefix, sign, hours, minutes = tz_match.groups()
        datestring = datestring[:tz_match.start()].strip()
        offset = timedelta(hours=int(hours), minutes=int(minutes or 0))
        tz = dt_timezone(-offset if sign == '-' else offset)
        # A date spelled with a month NAME and no time ignores its zone, and refuses
        # a bare offset -- measured on NASDAQ:AAPL: "01 Jan 2022 GMT+3", "01 Jan
        # 2022 PST" and "Jan 01 2022 EST" all resolve to midnight UTC, "01 Jan 2022
        # +0300" is rejected, while the numeric "2022-01-01 EST" and "01-02-2022
        # EST" resolve to 05:00 UTC.
        if ':' not in datestring and re.search(r'[A-Za-z]', datestring):
            tz = UTC if prefix else None
    else:
        # No timezone means UTC, not the exchange timezone -- measured on
        # NASDAQ:AAPL (America/New_York): "01 Jan 2022 00:00", "2022-01-01",
        # "2022", "Jan 2022" and "01-02-2022" all resolve to midnight UTC.
        tz = UTC

    # Try standard formats (with timezone)
    if tz_match and tz is not None:
        normalized = normalize_timezone(f"{datestring} {tz_match.group(1)}")
        for fmt in STANDARD_FORMATS:
            try:
                return datetime.strptime(normalized, fmt)
            except ValueError:
                continue

    # Try Pine formats (without timezone)
    for fmt in PINE_FORMATS if tz is not None else ():
        try:
            dt = datetime.strptime(datestring, fmt)
            return dt.replace(tzinfo=tz)
        except ValueError:
            continue

    raise ValueError(
        f"Invalid date format: {datestring}\n"
        "Supported formats:\n"
        "- ISO Style: '2020-02-20T15:30:00+02:00', '2025-01-01 01:23:45-05:00',"
        " '2021-01-13:05:00'\n"
        "- With fraction: '2024-08-01T04:38:47.731215+00:00'\n"
        "- RFC Style: '20 Feb 2020 15:30:00 GMT+0200', '1 January 2018 00:00 +0000'\n"
        "- Simple Pine: 'Feb 01 2020 22:10:05', '1 January 2018', '2020-02-20'\n"
        "- Numeric, month first: '01-01-2023', '03/04/2023', '03.04.2023 10:20:30'\n"
        "- Partial, year first: '2025', '2025-06', '2025.06', 'Jan 2025'"
    )


# The Gregorian calendar repeats exactly every 400 years
GREGORIAN_CYCLE_DAYS = 146097

# First day of the Gregorian calendar in TradingView's (Java's) hybrid calendar
# -- 1582-10-15 -- counted in days from the Unix epoch
GREGORIAN_CUTOVER_DAY = -141427

# Anchor making julian_civil_days() agree with civil_days() on the cutover:
# the Julian date 1582-10-05 is the same day as the Gregorian 1582-10-15
_JULIAN_EPOCH_DAY = -719470


def civil_days(year: int, month: int, day: int) -> int:
    """
    Days from the Unix epoch for a proleptic Gregorian date.

    ``day`` enters linearly, so out-of-range values roll over like Pine's do.

    :param year: Year (unbounded, may be zero or negative)
    :param month: Month, 1-12
    :param day: Day of month
    :return: Whole days from 1970-01-01
    """
    y = year - (month <= 2)
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (month - 3 if month > 2 else month + 9) + 2) // 5 + day - 1
    return era * GREGORIAN_CYCLE_DAYS + yoe * 365 + yoe // 4 - yoe // 100 + doy - 719468


def julian_civil_days(year: int, month: int, day: int) -> int:
    """
    Days from the Unix epoch for a Julian-calendar date.

    Same shape as :func:`civil_days` without the century rule: every fourth
    year is a leap year.

    :param year: Year (unbounded, may be zero or negative)
    :param month: Month, 1-12
    :param day: Day of month
    :return: Whole days from 1970-01-01
    """
    y = year - (month <= 2)
    doy = (153 * (month - 3 if month > 2 else month + 9) + 2) // 5 + day - 1
    return y * 365 + y // 4 + doy + _JULIAN_EPOCH_DAY
