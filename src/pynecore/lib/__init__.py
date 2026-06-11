"""
Builtin library of Pyne
"""
from typing import TYPE_CHECKING, TypeAlias, Any, Callable
from types import GenericAlias

if TYPE_CHECKING:
    from pynecore.types.type_checker import *
    from ..types.session import SessionInfo

import sys
import math as _math

from functools import lru_cache
from datetime import datetime, timedelta, time as dt_time, UTC

from pynecore.types.source import Source

from ..core.module_property import module_property
from ..core.script import script, input

from ..types.na import NA
from ..types import Series, PyneInt
from . import syminfo  # This should be imported before core.datetime to avoid circular import!
from . import barstate, string, log, math, plot, hline, linefill, alert, dayofweek
from .plot import plot as _plot
from . import timeframe as timeframe_module
from . import session as session_module
from ._fixnan import fixnan

from pynecore.core.overload import overload
from pynecore.core.datetime import parse_datestring as _parse_datestring, parse_timezone as _parse_timezone, \
    TimezoneNotFoundError
from ..core.resampler import Resampler

__all__ = [
    # Other modules
    'syminfo', 'barstate', 'string', 'log', 'math', 'plot',

    # Variables
    'bar_index', 'last_bar_index', 'last_bar_time',
    'open', 'high', 'low', 'close', 'volume',
    'bid', 'ask',
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
    'time', 'time_close', 'time_tradingday', 'timenow', 'na',
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

open: float = Source("open")  # noqa (shadowing built-in name (open) intentionally)
high: float = Source("high")
low: float = Source("low")
close: float = Source("close")
volume: float = Source("volume")

bid: float = Source("bid")
ask: float = Source("ask")

hl2: float = Source("hl2")
hlc3: float = Source("hlc3")
ohlc4: float = Source("ohlc4")
hlcc4: float = Source("hlcc4")

# Store time as integer as in Pine Scripts timestamp format
_time: int = 0
last_bar_time: int = 0

# Datetime object in the exchange timezone
_datetime: datetime = datetime.fromtimestamp(0, UTC)

# Script settings from `script.indicator`, `script.strategy` or `script.library`
_script: script = None  # type: ignore[assignment]

# Stores data to polot
_plot_data: dict[str, Any] = {}

# Extra fields from CSV data (beyond OHLCV), populated each bar by ScriptRunner
extra_fields: dict[str, Any] = {}

# Lib semaphore - to prevent lib`s main function to do things it must not (plot, strategy things, etc.)
_lib_semaphore = False

# Live trading mode flag — set by run.py when --live is specified
_is_live = False

# Strategy suppression — prevents strategy order placement during historical phase in live mode
_strategy_suppressed = False

#
# Function-and-namespace modules — the IDE-facing rebinding; at runtime the AST
# transformer routes ``hline(...)``-style calls to the module's self-named function
#

if TYPE_CHECKING:
    from .hline import hline
    from .plot import plot
    from .alert import alert


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
def _get_dt(time: int | None = None, timezone: str | None = None) -> datetime | NA[datetime]:
    """ Get datetime object from time and timezone """
    if isinstance(time, NA):
        return time
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
    _plot(series, title)


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

def is_na(source: Any = None) -> bool | NA:
    """
    Check if the source is NA.

    Pine treats inf/-inf/nan floats as "na" for na() predicate purposes,
    even though they participate in arithmetic/comparisons as normal IEEE-754
    values. This matches that dual behavior.
    """
    if source is None:
        return NA(None)
    # If the source is a type or GenericAlias (like list[float]), return NA of that type
    if isinstance(source, (type, GenericAlias)) and source is not NA:
        return NA(source)
    if isinstance(source, float):
        if _math.isnan(source) or _math.isinf(source):
            return True
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


def _intraday_session_args(timeframe: str) -> tuple:
    """
    Build the ``(tz, session_starts)`` arguments for :meth:`Resampler.get_bar_time`
    that anchor an intraday ``timeframe`` to the exchange session open, the way
    TradingView aligns intraday HTF bars. Returns an empty tuple for daily/weekly/
    monthly timeframes and when no session is known, keeping the pure UTC
    clock-floor. Anchoring is a no-op for on-hour / 24-7 markets, so it is always
    safe to pass for intraday.

    :param timeframe: The requested timeframe string (already validated).
    :return: ``(tz, session_starts)`` for intraday with a session, else ``()``.
    """
    # noinspection PyProtectedMember
    session_starts = getattr(syminfo, '_session_starts', None)
    if not session_starts:
        return ()
    # noinspection PyProtectedMember
    modifier, _ = timeframe_module._process_tf(timeframe)
    if modifier not in ('S', ''):
        return ()
    tz_name = getattr(syminfo, 'timezone', None)
    tz = _parse_timezone(tz_name) if tz_name else None
    return tz, session_starts


@module_property
def time(timeframe: str | None = None, session: str | int | None = None,
         timezone: str | None = None, bars_back: int = 0) -> PyneInt:
    """
    The time function returns the UNIX time of the current bar for the specified timeframe
    and session or NA if the time point is out of session.

    Usage examples:
    - time() - Current bar time
    - time("60") - Current 1-hour bar start time
    - time("1D", "0930-1600") - Daily bar time if within session
    - time("60", "0930-1600:23456", "America/New_York") - With timezone
    - time("60", -1) - Expected start time of the next 1-hour bar

    :param timeframe: The timeframe to get the time for (e.g., "D", "60", "240").
                     An empty string selects the chart's timeframe.
                     If None, returns current bar time.
    :param session: Session specification string (e.g., "0930-1600", "0000-0000:23456").
                   Format: "HHMM-HHMM" or "HHMM-HHMM:days" where days are 1234567 (1=Sun, 7=Sat).
                   An int value here is treated as ``bars_back`` (Pine's
                   ``time(timeframe, bars_back)`` overload).
    :param timezone: Timezone for the session (e.g., "GMT+2", "America/New_York").
                    If None, uses exchange timezone.
    :param bars_back: Bar offset on the chart's timeframe: positive values refer to past
                     bars, negative values to the expected times of future bars. The offset
                     is computed on a continuous time grid (exact for 24/7 markets).
    :return: UNIX time in milliseconds or NA if bar is outside session or invalid parameters
    """
    # Pine overload: time(timeframe, bars_back) -- an int second argument is a bar offset
    if isinstance(session, int) and not isinstance(session, bool):
        bars_back = session
        session = None

    if timeframe is None:
        return _time

    # An empty string selects the chart's timeframe
    if timeframe == '':
        timeframe = str(syminfo.period)

    # Get resampler for the requested timeframe
    try:
        resampler = Resampler.get_resampler(timeframe)
    except ValueError:
        # Invalid timeframe
        return NA(int)

    # Get the current bar time for the requested timeframe
    current_time_ms = _time
    if bars_back:
        try:
            current_time_ms -= bars_back * timeframe_module.in_seconds(str(syminfo.period)) * 1000
        except (ValueError, AssertionError):
            return NA(int)
    bar_time = resampler.get_bar_time(current_time_ms, *_intraday_session_args(timeframe))

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
    except TimezoneNotFoundError:
        # A missing/unresolvable timezone is a configuration error: surface it with
        # the actionable message instead of silently treating every bar as closed.
        raise
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


# ``time_tradingday`` cache. The strategy engine calls the property on every bar
# (intraday risk day-rollover), so the result is memoized per bar, keyed by the
# identity of ``_datetime`` — the function's actual input. Every bar installs a
# fresh (immutable) datetime object, so an identity hit guarantees an identical
# result; anything that swaps ``_datetime`` (including tests driving it
# directly) misses the memo and recomputes. NOT keyed by calendar date, which
# would be wrong for overnight sessions where bars before/after the session
# open on the same date belong to different trading days. The session-structure
# table is rebuilt whenever ``syminfo._opening_hours`` is replaced
# (``_set_lib_syminfo_properties`` always assigns a fresh list) or
# ``syminfo.period`` changes.
_ttd_memo_dt: datetime | None = None
_ttd_memo_result: int = 0
_ttd_session_hours: list | None = None
_ttd_session_period: str | None = None
_ttd_overnight_by_wd: dict[int, list[dt_time]] = {}
_ttd_period_delta: timedelta = timedelta()
_EPOCH_ORDINAL = 719163  # date(1970, 1, 1).toordinal()


# noinspection PyProtectedMember
@module_property
def time_tradingday() -> PyneInt:
    """
    The beginning time of the trading day the current bar belongs to, as a UNIX
    timestamp in milliseconds. It is 00:00 UTC of the calendar date — expressed in
    the exchange timezone — on which the bar's trading session ends.

    For symbols whose session crosses midnight (e.g. forex and futures overnight
    sessions) a bar that reaches into the session start belongs to the next calendar
    day's trading day, matching TradingView — including the boundary bar whose window
    merely contains the open (a 17:00-18:00 bar for a 17:05 open). For symbols whose
    session stays within a single calendar day (stocks, 24/7 crypto) it is simply
    00:00 UTC of the bar's exchange-timezone date.

    :return: UNIX time in milliseconds of 00:00 UTC on the trading day's date
    """
    global _ttd_memo_dt, _ttd_memo_result, _ttd_session_hours, _ttd_session_period, \
        _ttd_overnight_by_wd, _ttd_period_delta

    opening_hours = syminfo._opening_hours
    period = syminfo.period
    if opening_hours is not _ttd_session_hours or period != _ttd_session_period:
        # Session structure changed — rebuild the per-weekday table of overnight
        # session opens (the only entries that can roll the trading day: ones that
        # end at or before their own start and do not begin exactly at midnight).
        overnight: dict[int, list[dt_time]] = {}
        for day, start, end in opening_hours:
            starts_at_midnight = start.hour == 0 and start.minute == 0 and start.second == 0
            if end <= start and not starts_at_midnight:
                overnight.setdefault(day, []).append(start)
        _ttd_overnight_by_wd = overnight
        _ttd_period_delta = timedelta(seconds=timeframe_module.in_seconds(period))
        _ttd_session_hours = opening_hours
        _ttd_session_period = period
        _ttd_memo_dt = None

    if _datetime is _ttd_memo_dt:
        return _ttd_memo_result

    local_dt = _datetime  # already expressed in the exchange timezone
    trade_date = local_dt.date()

    # Roll into the next trading day when the bar overlaps the evening portion of an
    # overnight session. A bar whose window merely *contains* the session open — e.g.
    # a 17:00-18:00 bar when the session opens at 17:05 — already belongs to the new
    # trading day, matching TradingView and ``session.isfirstbar_regular``. Comparing
    # the bar's *end* against the open captures that boundary bar; comparing only the
    # bar's start would leave it in the previous day whenever the open does not land
    # exactly on a bar boundary.
    overnight_starts = _ttd_overnight_by_wd.get(local_dt.weekday())
    if overnight_starts:
        bar_end = local_dt + _ttd_period_delta
        for start in overnight_starts:
            session_open = local_dt.replace(
                hour=start.hour, minute=start.minute, second=start.second, microsecond=0)
            if bar_end > session_open:
                trade_date += timedelta(days=1)
                break

    # 00:00 UTC of the trading day's date — pure ordinal arithmetic (UTC has no
    # DST, so this is exactly ``datetime(y, m, d, tzinfo=UTC).timestamp() * 1000``).
    result = (trade_date.toordinal() - _EPOCH_ORDINAL) * 86_400_000
    _ttd_memo_dt = local_dt
    _ttd_memo_result = result
    return result


@module_property
def time_close(timeframe: str | None = None, session: str | int | None = None,
               timezone: str | None = None, bars_back: int = 0) -> PyneInt:
    """
    The time_close function returns the UNIX time of the current bar's close for the specified timeframe
    and session or NA if the time point is outside the session.

    Usage examples:
    - time_close() - Current bar close time
    - time_close("60") - Current 1-hour bar close time
    - time_close("1D", "0930-1600") - Daily bar close time if within session
    - time_close("60", "0930-1600:23456", "America/New_York") - With timezone
    - time_close("60", -1) - Expected close time of the next 1-hour bar

    :param timeframe: The timeframe to get the close time for (e.g., "D", "60", "240").
                     An empty string selects the chart's timeframe.
                     If None, returns current bar close time.
    :param session: Session specification string (e.g., "0930-1600", "0000-0000:23456").
                   Format: "HHMM-HHMM" or "HHMM-HHMM:days" where days are 1234567 (1=Sun, 7=Sat).
                   An int value here is treated as ``bars_back`` (Pine's
                   ``time_close(timeframe, bars_back)`` overload).
    :param timezone: Timezone for the session (e.g., "GMT+2", "America/New_York").
                    If None, uses exchange timezone.
    :param bars_back: Bar offset on the chart's timeframe: positive values refer to past
                     bars, negative values to the expected times of future bars. The offset
                     is computed on a continuous time grid (exact for 24/7 markets).
    :return: UNIX time in milliseconds of bar close or NA if bar is outside session or invalid parameters
    """
    # Pine overload: time_close(timeframe, bars_back) -- an int second argument is a bar offset
    if isinstance(session, int) and not isinstance(session, bool):
        bars_back = session
        session = None

    if timeframe is None:
        # Close time of the current chart bar
        try:
            return _time + timeframe_module.in_seconds(str(syminfo.period)) * 1000
        except (ValueError, AssertionError):
            return NA(int)

    # An empty string selects the chart's timeframe
    if timeframe == '':
        timeframe = str(syminfo.period)

    # Get resampler for the requested timeframe
    try:
        resampler = Resampler.get_resampler(timeframe)
    except ValueError:
        # Invalid timeframe
        return NA(int)

    # Get the current bar time for the requested timeframe
    current_time_ms = _time
    if bars_back:
        try:
            current_time_ms -= bars_back * timeframe_module.in_seconds(str(syminfo.period)) * 1000
        except (ValueError, AssertionError):
            return NA(int)
    bar_start_time = resampler.get_bar_time(current_time_ms, *_intraday_session_args(timeframe))

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
    except TimezoneNotFoundError:
        # A missing/unresolvable timezone is a configuration error: surface it with
        # the actionable message instead of silently treating every bar as closed.
        raise
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
