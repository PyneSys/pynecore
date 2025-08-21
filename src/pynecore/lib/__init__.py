"""
Builtin library of Pyne
"""
from typing import TYPE_CHECKING, TypeAlias, Any, Callable
from types import GenericAlias

if TYPE_CHECKING:
    from ..types.session import SessionInfo

import sys

from functools import lru_cache
from datetime import datetime, UTC

from pynecore.types.source import Source

from ..core.module_property import module_property
from ..core.script import script, input

from ..types.series import Series
from ..types.na import NA
from . import syminfo  # This should be imported before core.datetime to avoid circular import!
from . import barstate, string, log, math, plot, hline, linefill, alert
from . import timeframe as timeframe_module
from . import session as session_module

from pynecore.core.overload import overload
from pynecore.core.datetime import parse_datestring as _parse_datestring, parse_timezone as _parse_timezone
from ..core.resampler import Resampler

__all__ = [
    # Other modules
    'syminfo', 'barstate', 'string', 'log', 'math', 'plot',

    # Variables
    'bar_index', 'last_bar_index', 'last_bar_time',
    'open', 'high', 'low', 'close', 'volume',
    'hl2', 'hlc3', 'ohlc4', 'hlcc4',

    # Functions / objects
    'input', 'script',

    'max_bars_back',

    'timestamp',

    'plotchar', 'plotarrow', 'plotbar', 'plotcandle', 'plotshape', 'barcolor', 'bgcolor',
    'fill', 'linefill',

    'alertcondition',

    'fixnan', 'nz',

    # Module properties
    'dayofmonth', 'dayofweek', 'hour', 'minute', 'month', 'second', 'weekofyear', 'year',
    'time', 'time_close', 'na',
]

#
# Constants
#

# For better type hints
TimezoneStr: TypeAlias = str  # e.g. "UTC-5", "GMT+0530", "America/New_York"
DateStr: TypeAlias = str  # e.g. "2020-02-20", "20 Feb 2020"

#
# Module variables
#

bar_index: Series[int] = 0
last_bar_index: Series[int] = 0  # This always points to the bar_index

open: Series[float] | NA[float] = Source("open")  # noqa (shadowing built-in name (open) intentionally)
high: Series[float] | NA[float] = Source("high")
low: Series[float] | NA[float] = Source("low")
close: Series[float] | NA[float] = Source("close")
volume: Series[float] | NA[float] = Source("volume")

hl2: Series[float] | NA[float] = Source("hl2")
hlc3: Series[float] | NA[float] = Source("hlc3")
ohlc4: Series[float] | NA[float] = Source("ohlc4")
hlcc4: Series[float] | NA[float] = Source("hlcc4")

# Store time as integer as in Pine Scripts timestamp format
_time: int = 0
last_bar_time: int = 0

# Datetime object in the exchange timezone
_datetime: datetime = datetime.fromtimestamp(0, UTC)

# Script settings from `script.indicator`, `script.strategy` or `script.library`
_script: script | None = None

# Stores data to polot
_plot_data: dict[str, Any] = {}

# Lib semaphore - to prevent lib`s main function to do things it must not (plot, strategy things, etc.)
_lib_semaphore = False

#
# Callable modules
#

if TYPE_CHECKING:
    from hline import hline
    from plot import plot
    from alert import alert


#
# Functions
#

# noinspection PyUnusedLocal
def max_bars_back(var: Any, num: int) -> None:
    """
    Function sets the maximum number of bars that is available for historical reference of a given
    built-in or user variable.

    :param var: Series variable identifier for which history buffer should be resized.
    :param num: History buffer size which is the number of bars to keep.
    """


### Date / Time ###

# noinspection PyShadowingNames
def _get_dt(time: int | None = None, timezone: str | None = None) -> datetime:
    """ Get datetime object from time and timezone """
    dt = _datetime if time is None else datetime.fromtimestamp(time / 1000, UTC)
    assert dt is not None
    return dt.astimezone(_parse_timezone(timezone))


@lru_cache(maxsize=1024)
@overload
def timestamp(date_string: DateStr) -> int:  # It is more pythonic, but not supported by Pine Script
    """
    Parse date string and return UNIX timestamp in milliseconds

    Multiple calling formats supported:
    - timestamp("2020-02-20T15:30:00+02:00")  # ISO 8601
    - timestamp("20 Feb 2020 15:30:00 GMT+0200")  # RFC 2822
    - timestamp("Feb 01 2020 22:10:05")       # Pine format
    - timestamp("2011-10-10T14:48:00")        # Pine format without timezone

    :param date_string: Date string in Pine Script format
    :return: UNIX timestamp in milliseconds
    """
    dt = _parse_datestring(date_string)
    return int(dt.timestamp() * 1000)


# noinspection PyPep8Naming
@overload
def timestamp(dateString: DateStr) -> int:
    """
    Parse date string and return UNIX timestamp in milliseconds

    Multiple calling formats supported:
    - timestamp("2020-02-20T15:30:00+02:00")  # ISO 8601
    - timestamp("20 Feb 2020 15:30:00 GMT+0200")  # RFC 2822
    - timestamp("Feb 01 2020 22:10:05")       # Pine format
    - timestamp("2011-10-10T14:48:00")        # Pine format without timezone
    - timestamp("UTC-5", 2020, 2, 20, 15, 30) # With timezone

    :param dateString: Date string in Pine Script format
    :return: UNIX timestamp in milliseconds
    """
    return timestamp(date_string=dateString)


# noinspection PyShadowingNames
@overload
def timestamp(timezone: TimezoneStr | None, year: int | float, month: int | float, day: int | float,
              hour: int | float = 0, minute: int | float = 0, second: int | float = 0) -> int:
    """
    Create timestamp from date/time components with timezone:
    - timestamp("UTC-5", 2020, 2, 20, 15, 30)
    - timestamp("GMT+0530", 2020, 2, 20, 15, 30)

    :param timezone: Timezone string
    :param year: Year
    :param month: Month
    :param day: Day
    :param hour: Hour
    :param minute: Minute
    :param second: Second
    :return: UNIX timestamp in milliseconds
    """
    tz = _parse_timezone(timezone)
    dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=tz)
    return int(dt.timestamp() * 1000)


# noinspection PyShadowingNames
@overload
def timestamp(year: int | float, month: int | float, day: int | float, hour: int | float = 0,
              minute: int | float = 0, second: int | float = 0) -> int:
    """
    Create timestamp from date/time components:
    - timestamp(2020, 2, 20, 15, 30)          # From components
    - timestamp(2020, 2, 20, 15, 30, 0)       # With seconds

    :param year: Year
    :param month: Month
    :param day: Day
    :param hour: Hour
    :param minute: Minute
    :param second: Second
    :return: UNIX timestamp in milliseconds
    """
    return timestamp(None, year=year, month=month, day=day, hour=hour, minute=minute, second=second)


### Plotting ###

# TODO: implement creating plot metadata to be able to plot in a different module

def barcolor(*_, **__):
    ...


def bgcolor(*_, **__):
    ...


def fill(*_, **__):
    ...


def plotarrow(*_, **__):
    ...


def plotbar(*_, **__):
    ...


def plotcandle(*_, **__):
    ...


def plotchar(series: Any, title: str | None = None, *_, **__):
    plot(series, title)


def plotshape(*_, **__):
    ...


### Alert ###

def alertcondition(*_, **__):
    """
    Define alert condition. Currently implemented as no-op.

    In the future this could be used to define alert conditions
    that can be triggered based on boolean expressions.
    """
    if bar_index == 0:  # Only check if it is the first bar for performance reasons
        # Check if it is called from the main function
        if sys._getframe(1).f_code.co_name != 'main':  # noqa
            raise RuntimeError("The alertcondition function can only be called from the main function!")


### Other ###

__persistent_last_not_nan__: Any = NA(None)
__persistent_function_vars__ = {'fixnan': ['__persistent_last_not_nan__']}


def fixnan(source: Any) -> Any:
    """
    Fix NA values by replacing them with the last non-NA value

    :param source: The source value
    :return: The source value if it is not NA, otherwise the last non-NA value
    """
    global __persistent_last_not_nan__
    __persistent_last_not_nan__ = source if not isinstance(source, NA) else __persistent_last_not_nan__
    return __persistent_last_not_nan__


def is_na(source: Any = None) -> bool | NA:
    """
    Check if the source is NA
    """
    if source is None:
        return NA(None)
    # If the source is a type or GenericAlias (like list[float]), return NA of that type
    if isinstance(source, (type, GenericAlias)) and source is not NA:
        return NA(source)
    return isinstance(source, NA) or source is NA


# In Pine Script, na is both a property and a function
na: Callable[[Any], bool | NA] | Any = is_na


def nz(source: Any, replacement: Any = 0) -> Any:
    """
    Replace NA values with a replacement value or 0 if not specified

    :param source: The source value
    :param replacement: The replacement value, default is 0
    :return: The source value if it is not NA, otherwise the replacement value
    """
    if isinstance(source, NA):
        return replacement
    return source


#
# Module properties
#

### Date / Time ###

# noinspection PyShadowingNames
@module_property
def dayofmonth(time: int | None = None, timezone: str | None = None) -> int:
    """
    Day of the month

    :param time: The time to get the day of the month from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The day of the month
    """
    return _get_dt(time, timezone).day


# noinspection PyShadowingNames
@module_property
def dayofweek(time: int | None = None, timezone: str | None = None) -> int:
    """
    Day of the week

    :param time: The time to get the day of the week from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The day of the week, 1 is Sunday, 2 is Monday, ..., 7 is Saturday
    """
    res = _get_dt(time, timezone).weekday() + 2
    if res == 8:
        res = 1
    return res


# noinspection PyShadowingNames
@module_property
def hour(time: int | None = None, timezone: str | None = None) -> int:
    """
    Hour of the day

    :param time: The time to get the hour of the day from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The hour of the day
    """
    return _get_dt(time, timezone).hour


# noinspection PyShadowingNames
@module_property
def minute(time: int | None = None, timezone: str | None = None) -> int:
    """
    Minute of the hour

    :param time: The time to get the minute of the hour from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The minute of the hour
    """
    return _get_dt(time, timezone).minute


# noinspection PyShadowingNames
@module_property
def month(time: int | None = None, timezone: str | None = None) -> int:
    """
    Month of the year

    :param time: The time to get the month of the year from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The month of the year
    """
    return _get_dt(time, timezone).month


# noinspection PyShadowingNames
@module_property
def second(time: int | None = None, timezone: str | None = None) -> int:
    """
    Second of the minute

    :param time: The time to get the second of the minute from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The second of the minute
    """
    return _get_dt(time, timezone).second


### Session parsing and validation helpers ###

def _parse_session_string(session: str, timezone: str | None = None) -> 'SessionInfo':
    """
    Parse a session string into a SessionInfo object.

    :param session: Session string (e.g., "0930-1600", "0930-1600:23456", "0000-0000:1234567")
    :param timezone: Timezone string, defaults to exchange timezone if None
    :return: SessionInfo object
    :raises ValueError: If session string is invalid
    """
    from ..types.session import SessionInfo
    from datetime import time as dt_time

    if not session or session.strip() == "":
        raise ValueError("Session string cannot be empty")

    # Use exchange timezone if not specified
    if timezone is None:
        # Use a safe default if syminfo.timezone is not available
        timezone = getattr(syminfo, 'timezone', 'UTC')
        # Handle NA values
        if hasattr(timezone, '__class__') and 'NA' in timezone.__class__.__name__:
            timezone = 'UTC'

    # Split session and days if present
    if ':' in session:
        time_part, days_part = session.split(':', 1)
    else:
        time_part = session
        # Default days in Pine Script v5 is all days (1234567)
        days_part = "1234567"

    # Parse time part (HHMM-HHMM format)
    if '-' not in time_part:
        raise ValueError(f"Invalid session format: {session}. Expected HHMM-HHMM format")

    start_str, end_str = time_part.split('-', 1)

    if len(start_str) != 4 or len(end_str) != 4:
        raise ValueError(f"Invalid time format in session: {session}. Expected HHMM-HHMM")

    try:
        start_hour = int(start_str[:2])
        start_minute = int(start_str[2:])
        end_hour = int(end_str[:2])
        end_minute = int(end_str[2:])

        # Validate time values
        if not (0 <= start_hour <= 23 and 0 <= start_minute <= 59):
            raise ValueError(f"Invalid start time: {start_str}")
        if not (0 <= end_hour <= 23 and 0 <= end_minute <= 59):
            raise ValueError(f"Invalid end time: {end_str}")

        start_time = dt_time(start_hour, start_minute)
        end_time = dt_time(end_hour, end_minute)

    except ValueError as e:
        raise ValueError(f"Invalid time values in session: {session}") from e

    # Parse days (1=Sunday, 2=Monday, ..., 7=Saturday)
    try:
        days = set()
        for day_char in days_part:
            day_num = int(day_char)
            if not 1 <= day_num <= 7:
                raise ValueError(f"Invalid day: {day_num}")
            days.add(day_num)
    except ValueError as e:
        raise ValueError(f"Invalid days specification: {days_part}") from e

    return SessionInfo(
        start_time=start_time,
        end_time=end_time,
        days=days,
        timezone=timezone
    )


def _is_bar_in_session(bar_time_ms: int, session_info: 'SessionInfo', timeframe: str) -> bool:
    """
    Check if a bar time falls within the specified session.

    :param bar_time_ms: Bar time in milliseconds (UNIX timestamp)
    :param session_info: Session information
    :param timeframe: Timeframe string for calculating bar duration
    :return: True if bar is within session, False otherwise
    """
    from datetime import datetime, timedelta

    # Convert bar time to datetime in session timezone
    bar_dt = datetime.fromtimestamp(bar_time_ms / 1000)
    session_tz = _parse_timezone(session_info.timezone)
    bar_dt_local = bar_dt.astimezone(session_tz)

    # Get the day of week in TradingView format (1=Sunday, 2=Monday, ..., 7=Saturday)
    # Python weekday: 0=Monday, 6=Sunday
    python_weekday = bar_dt_local.weekday()
    tv_weekday = (python_weekday + 2) % 7
    if tv_weekday == 0:
        tv_weekday = 7

    # Check if the day is in the session days
    if tv_weekday not in session_info.days:
        return False

    # Get bar time components
    bar_time = bar_dt_local.time()

    # Get timeframe duration for checking bar overlap
    try:
        tf_seconds = timeframe_module.in_seconds(timeframe)
    except (ValueError, AssertionError):
        # If timeframe is invalid, assume 1-minute bars
        tf_seconds = 60

    # Calculate bar end time
    bar_end_dt = bar_dt_local + timedelta(seconds=tf_seconds)
    bar_end_time = bar_end_dt.time()

    # Handle overnight sessions
    if session_info.is_overnight:
        # Session spans midnight (e.g., 22:00-06:00)
        # Bar is in session if it starts after session start OR ends before session end
        in_session = (bar_time >= session_info.start_time or
                      bar_end_time <= session_info.end_time)
    else:
        # Normal session within same day
        # Bar is in session if it overlaps with the session time range
        # Bar overlaps if: bar_start < session_end AND bar_end > session_start
        in_session = (bar_time < session_info.end_time and
                      bar_end_time > session_info.start_time)

    return in_session


@module_property
def time(timeframe: str | None = None, session: str | None = None, timezone: str | None = None) -> int | NA[int]:
    """
    The time function returns the UNIX time of the current bar for the specified timeframe
    and session or NA if the time point is out of session.

    Usage examples:
    - time() - Current bar time
    - time("60") - Current 1-hour bar start time
    - time("1D", "0930-1600") - Daily bar time if within session
    - time("60", "0930-1600:23456", "America/New_York") - With timezone

    :param timeframe: The timeframe to get the time for (e.g., "D", "60", "240").
                     If None, returns current bar time.
    :param session: Session specification string (e.g., "0930-1600", "0000-0000:23456").
                   Format: "HHMM-HHMM" or "HHMM-HHMM:days" where days are 1234567 (1=Sun, 7=Sat)
    :param timezone: Timezone for the session (e.g., "GMT+2", "America/New_York").
                    If None, uses exchange timezone.
    :return: UNIX time in milliseconds or NA if bar is outside session or invalid parameters
    """
    if timeframe is None:
        return _time

    # Get resampler for the requested timeframe
    try:
        resampler = Resampler.get_resampler(timeframe)
    except ValueError:
        # Invalid timeframe
        return NA(int)

    # Get the current bar time for the requested timeframe
    current_time_ms = _time
    bar_time = resampler.get_bar_time(current_time_ms)

    if session is None:
        # No session specified, return the bar time
        return bar_time

    # Parse session string
    try:
        session_info = _parse_session_string(session, timezone)
    except ValueError:
        # Invalid session string
        return NA(int)

    # Check if the bar is within the session
    try:
        if _is_bar_in_session(bar_time, session_info, timeframe):
            return bar_time
        else:
            return NA(int)
    except Exception:  # noqa
        # Error during session validation
        return NA(int)


@module_property
def timenow():
    """
    Current time in UNIX format. It is the number of milliseconds that have elapsed since 00:00:00 UTC, 1 January 1970.

    :return: Current time in milliseconds
    """
    # Get current UTC time and convert to milliseconds since Unix epoch
    return int(datetime.now(UTC).timestamp() * 1000)


@module_property
def time_close(timeframe: str | None = None, session: str | None = None, timezone: str | None = None) -> int | NA[int]:
    """
    The time_close function returns the UNIX time of the current bar's close for the specified timeframe
    and session or NA if the time point is outside the session.

    Usage examples:
    - time_close() - Current bar close time
    - time_close("60") - Current 1-hour bar close time
    - time_close("1D", "0930-1600") - Daily bar close time if within session
    - time_close("60", "0930-1600:23456", "America/New_York") - With timezone

    :param timeframe: The timeframe to get the close time for (e.g., "D", "60", "240").
                     If None, returns current bar close time.
    :param session: Session specification string (e.g., "0930-1600", "0000-0000:23456").
                   Format: "HHMM-HHMM" or "HHMM-HHMM:days" where days are 1234567 (1=Sun, 7=Sat)
    :param timezone: Timezone for the session (e.g., "GMT+2", "America/New_York").
                    If None, uses exchange timezone.
    :return: UNIX time in milliseconds of bar close or NA if bar is outside session or invalid parameters
    """
    if timeframe is None:
        return _time

    # Get resampler for the requested timeframe
    try:
        resampler = Resampler.get_resampler(timeframe)
    except ValueError:
        # Invalid timeframe
        return NA(int)

    # Get the current bar time for the requested timeframe
    current_time_ms = _time
    bar_start_time = resampler.get_bar_time(current_time_ms)

    # Calculate bar close time by adding timeframe duration
    try:
        tf_seconds = timeframe_module.in_seconds(timeframe)
        bar_close_time = bar_start_time + (tf_seconds * 1000)  # Convert to milliseconds
    except (ValueError, AssertionError):
        return NA(int)

    if session is None:
        # No session specified, return the bar close time
        return bar_close_time

    # Parse session string
    try:
        session_info = _parse_session_string(session, timezone)
    except ValueError:
        # Invalid session string
        return NA(int)

    # Check if the bar is within the session (using bar start time for session validation)
    try:
        if _is_bar_in_session(bar_start_time, session_info, timeframe):
            return bar_close_time
        else:
            return NA(int)
    except Exception:  # noqa
        # Error during session validation
        return NA(int)


# noinspection PyShadowingNames
@module_property
def weekofyear(time: int | None = None, timezone: str | None = None) -> int:
    """
    Week of the year

    :param time: The time to get the week of the year from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The week of the year
    """
    return _get_dt(time, timezone).isocalendar()[1]


# noinspection PyShadowingNames
@module_property
def year(time: int | None = None, timezone: str | None = None) -> int:
    """
    Year

    :param time: The time to get the year from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The year
    """
    return _get_dt(time, timezone).year
