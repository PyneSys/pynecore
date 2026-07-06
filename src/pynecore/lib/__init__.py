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

from datetime import datetime, timedelta, time as dt_time, date, UTC

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
from ..core.resampler import (
    Resampler, ObservedDayCounter as _ObservedDayCounter,
    grid_mode as _grid_mode, overnight_opens as _overnight_opens,
    overnight_starts_by_weekday as _overnight_starts_by_weekday,
    close_table_by_weekday as _close_table_by_weekday,
    trading_day as _trading_day, trading_day_open_sec as _trading_day_open_sec,
    observed_week_key as _observed_week_key,
)

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

# Chart (main-series) timeframe, propagated into request.security children so
# ``timeframe.main_period`` there reports the chart TF instead of the context's
# own period. ``None`` on the chart side, where ``_script`` carries it directly.
_main_timeframe: str | None = None

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
    # Pine accepts out-of-range components and rolls them over (e.g. hour 26 ->
    # next day + 2h, month 13 -> next January). Normalize the month into the
    # year, then carry day/hour/minute/second through timedelta so the wall
    # clock overflows before the timezone conversion.
    y = int(year)
    m = int(month)
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    dt = datetime(y, m, 1, tzinfo=tz) + timedelta(
        days=int(day) - 1, hours=int(hour), minutes=int(minute), seconds=int(second)
    )
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
    TradingView aligns intraday HTF bars. Anchoring is a no-op for on-hour / 24-7
    markets, so it is always safe to pass for intraday.

    Daily/weekly/monthly timeframes get ``(tz,)`` instead: their calendar floor
    must run in the exchange timezone (TradingView day/week/month boundaries are
    exchange-local), not in the machine's local time.

    :param timeframe: The requested timeframe string (already validated).
    :return: ``(tz, session_starts)`` for intraday with a session, ``(tz,)`` for
             daily/weekly/monthly, else ``()``.
    """
    tz_name = getattr(syminfo, 'timezone', None)
    tz = _parse_timezone(tz_name) if tz_name else None
    # noinspection PyProtectedMember
    modifier, _ = timeframe_module._process_tf(timeframe)
    if modifier not in ('S', ''):
        return (tz,) if tz is not None else ()
    # noinspection PyProtectedMember
    session_starts = getattr(syminfo, '_session_starts', None)
    if not session_starts:
        return (tz,) if tz is not None else ()
    return tz, session_starts


# Multi-period (nD/nW/nM) scheduled-grid tracker. TradingView counts scheduled
# trading days per exchange calendar with a year-reset counter (see the
# ``core.resampler`` module docs). 'calendar' (24/7) and 'weekday' (FX) grids
# are pure arithmetic; 'observed' symbols (exchange-listed) count the actual
# trading days streamed through the chart, which realizes TradingView's
# holiday calendar. The tracker is fed from ``_set_lib_properties`` with a
# single integer compare per bar (``_dg_next_roll``); the heavy path runs once
# per trading day. It only activates for 'observed' symbols on charts of at
# most daily resolution — everything else resolves arithmetically on demand.
_dg_next_roll: float = 0.0  # epoch-sec threshold of the next possible day roll
_dg_mode: str = ''
_dg_eff: int = 0  # bar open -> last instant offset in seconds (intraday charts)
_dg_tz = None
_dg_overnight: dict[int, dt_time] = {}
_dg_template: list | None = None  # identity guard, like the _ttd machinery
_dg_day: date | None = None  # current trading day
_dg_counter: '_ObservedDayCounter | None' = None  # year-reset day counter (+ fold)
_dg_last_ts: float = 0.0  # previous bar open (epoch sec) — fed to the fold detector
_dg_day_starts: dict[int, dict[int, int]] = {}  # year -> {ordinal: bar-open ms}
_dg_ord_by_day: dict[date, int] = {}  # date -> ordinal (current + previous year)
_dg_week_first: dict[tuple[int, int], int] = {}  # (monday-year, week ordinal) -> ms
_dg_month_first: dict[tuple[int, int], int] = {}  # (year, month) -> first bar ms


def _dg_reset() -> None:
    """Reset the scheduled-grid tracker (new run / new script)."""
    global _dg_next_roll, _dg_mode, _dg_eff, _dg_template, _dg_day, \
        _dg_counter, _dg_last_ts
    _dg_next_roll = 0.0
    _dg_mode = ''
    _dg_eff = 0
    _dg_template = None
    _dg_day = None
    _dg_counter = None
    _dg_last_ts = 0.0
    _dg_day_starts.clear()
    _dg_ord_by_day.clear()
    _dg_week_first.clear()
    _dg_month_first.clear()


# noinspection PyProtectedMember
def _dg_on_roll(ts: float) -> None:
    """
    Advance the observed-day tracker to the bar at ``ts`` (epoch seconds).

    Only called when a bar reaches ``_dg_next_roll`` — i.e. at most once per
    trading day, plus once at configuration time. A bar belongs to the trading
    day its *last* instant falls into (``_dg_eff`` offset): on intraday charts
    the bar containing the session open starts the new day even when its own
    timestamp precedes the open.

    :param ts: Current bar open in epoch seconds
    """
    global _dg_next_roll, _dg_mode, _dg_eff, _dg_tz, _dg_overnight, \
        _dg_template, _dg_day, _dg_counter

    opening_hours = syminfo._opening_hours
    if opening_hours is not _dg_template:
        # (Re)configure from the symbol template
        _dg_template = opening_hours
        _dg_mode = _grid_mode(getattr(syminfo, 'type', None), opening_hours)
        tz_name = getattr(syminfo, 'timezone', None)
        _dg_tz = _parse_timezone(tz_name) if tz_name else None
        _dg_overnight = _overnight_opens(opening_hours, syminfo._session_starts)
        try:
            chart_sec = timeframe_module.in_seconds(str(syminfo.period))
            chart_mod, _ = timeframe_module._process_tf(str(syminfo.period))
        except (ValueError, AssertionError):
            chart_sec = 0
            chart_mod = None
        if _dg_mode != 'observed' or not 0 < chart_sec <= 86_400:
            # Arithmetic grids need no tracking, and day counting needs a
            # stream of at most daily bars
            _dg_next_roll = _math.inf
            return
        _dg_eff = chart_sec - 1 if chart_mod in ('', 'S') else 0
        # Intraday charts carry per-bar end instants for the holiday half-day
        # fold; a daily chart stream is already folded.
        _dg_counter = _ObservedDayCounter(
            _dg_tz, opening_hours, fold=chart_mod in ('', 'S'))
        _dg_day = None
        _dg_day_starts.clear()
        _dg_ord_by_day.clear()
        _dg_week_first.clear()
        _dg_month_first.clear()

    assert _dg_counter is not None
    eff = ts + _dg_eff
    td = _trading_day(eff, _dg_tz, _dg_overnight)
    prev = _dg_day
    if td != prev:
        if prev is not None and td.year != prev.year:
            # Keep only the current and previous year's records
            for y in [y for y in _dg_day_starts if y < td.year - 1]:
                del _dg_day_starts[y]
            for d in [d for d in _dg_ord_by_day if d.year < td.year - 1]:
                del _dg_ord_by_day[d]
            for k in [k for k in _dg_week_first if k[0] < td.year - 1]:
                del _dg_week_first[k]
            for k in [k for k in _dg_month_first if k[0] < td.year - 1]:
                del _dg_month_first[k]
        # Feed the previous day's last bar end so the fold can tell whether it
        # closed early, then advance the year-reset counter.
        if _dg_last_ts:
            _dg_counter.note_bar_end(int(_dg_last_ts) + _dg_eff + 1)
        ordinal = _dg_counter.ordinal(td)
        _dg_day = td

        ms = int(ts * 1000)
        # setdefault: a folded holiday half-day shares the early-close day's
        # ordinal and must not overwrite that period's first session open.
        _dg_day_starts.setdefault(td.year, {}).setdefault(ordinal, ms)
        _dg_ord_by_day[td] = ordinal
        wy, week = _observed_week_key(td)
        _dg_week_first.setdefault((wy, week), ms)
        _dg_month_first.setdefault((td.year, td.month), ms)

    # Next possible roll: the first chart bar whose span reaches a scheduled
    # session open. Scheduled opens exist even on holidays — a threshold on a
    # dataless day is harmless, the next real bar recomputes its trading day
    # from scratch.
    for i in range(1, 8):
        open_sec = _trading_day_open_sec(
            td + timedelta(days=i), _dg_tz, syminfo._session_starts, _dg_overnight)
        if open_sec > eff:
            _dg_next_roll = open_sec - _dg_eff
            break
    else:
        _dg_next_roll = ts + 86_400


def _dwm_change_key(timeframe: str, modifier: str, multiplier: int) -> int:
    """
    Period identity of the current bar on a multi-period (nD/nW/nM) grid —
    the ``timeframe.change`` helper. Kept here so the transformed
    ``_timeframe_change`` module only makes single-attribute ``lib.*`` calls.

    :param timeframe: The requested timeframe string
    :param modifier: 'D', 'W' or 'M' (from ``_process_tf``)
    :param multiplier: Period multiplier (> 1)
    :return: The period's opening time in milliseconds
    """
    return _dwm_bar_time(
        Resampler.get_resampler(timeframe), modifier, multiplier, _time)


def _chart_span_off_ms() -> int:
    """
    Offset from a chart bar's open to its last instant, in milliseconds.

    A chart bar belongs to the D/W/M period its *last* instant falls into: the
    bar containing a session open is the new trading day's first bar even when
    its own timestamp precedes the open (e.g. a 17:05 session open on a
    240-minute grid — the 17:00 bar starts the new day). D/W/M chart bars are
    session-aligned by construction, so only intraday charts need the offset.

    :return: ``chart bar span - 1`` for intraday chart periods, else 0
    """
    try:
        # noinspection PyProtectedMember
        chart_mod, _ = timeframe_module._process_tf(str(syminfo.period))
        if chart_mod in ('', 'S'):
            return timeframe_module.in_seconds(str(syminfo.period)) * 1000 - 1
    except (ValueError, AssertionError):
        pass
    return 0


def _dg_trading_day():
    """
    The current bar's trading day (``datetime.date``).

    Uses the tracker's record when it is active ('observed' symbols on at most
    daily charts); otherwise derives it from ``_datetime`` — the bar's
    exchange-local datetime — advanced to the bar's last instant
    (:func:`_chart_span_off_ms`), with the overnight roll (a bar reaching its
    weekday's overnight open belongs to the next calendar day). The tracker's
    configuration pass runs on the first bar of every run, so
    ``_dg_overnight`` is populated whenever real data is streaming.

    :return: Trading day of the current bar
    """
    if _dg_day is not None:
        return _dg_day
    dt_loc = _datetime
    off = _chart_span_off_ms()
    if off:
        dt_loc = dt_loc + timedelta(milliseconds=off)
    d = dt_loc.date()
    if _dg_overnight:
        t0 = _dg_overnight.get(dt_loc.weekday())
        if t0 is not None and dt_loc.time() >= t0:
            d += timedelta(days=1)
    return d


# noinspection PyProtectedMember
def _dwm_bar_time(resampler: Resampler, modifier: str, multiplier: int,
                  current_time_ms: int) -> int:
    """
    Multi-period (nD/nW/nM) bar open time on the scheduled grid.

    'calendar'/'weekday' symbols resolve arithmetically; 'observed' symbols
    look up the tracker's records and fall back to the weekday grid for
    timestamps outside the tracked window (pre-data or future times).

    ``current_time_ms`` is a chart bar open; the bar is resolved by its *last*
    instant (:func:`_chart_span_off_ms`), so the bar containing a session open
    counts as the new trading day's first bar.

    :param resampler: Resampler of the requested timeframe
    :param modifier: 'D', 'W' or 'M'
    :param multiplier: Period multiplier (> 1)
    :param current_time_ms: Chart bar open to resolve, in milliseconds
    :return: Bar opening time in milliseconds
    """
    eff_ms = current_time_ms + _chart_span_off_ms()
    if _dg_mode != 'observed' or _dg_next_roll == _math.inf:
        # Pure arithmetic — also the fallback when the tracker is inactive
        # (chart resolution above daily)
        return resampler.get_bar_time(
            eff_ms, _dg_tz, syminfo._session_starts,
            syminfo._opening_hours, _dg_mode or None)

    if current_time_ms == _time and _dg_day is not None:
        td = _dg_day
    else:
        td = _trading_day(eff_ms // 1000, _dg_tz, _dg_overnight)

    if modifier == 'D':
        ordinal = _dg_ord_by_day.get(td)
        if ordinal is not None:
            base = (ordinal // multiplier) * multiplier
            days = _dg_day_starts.get(td.year)
            if days is not None:
                for i in range(base, ordinal + 1):
                    ms = days.get(i)
                    if ms is not None:
                        return ms
    elif modifier == 'W':
        wy, week = _observed_week_key(td)
        base = (week // multiplier) * multiplier
        for i in range(base, week + 1):
            ms = _dg_week_first.get((wy, i))
            if ms is not None:
                return ms
    else:  # 'M'
        m0 = ((td.month - 1) // multiplier) * multiplier + 1
        for m in range(m0, td.month + 1):
            ms = _dg_month_first.get((td.year, m))
            if ms is not None:
                return ms

    # Outside the tracked window — weekday-grid approximation
    return resampler.get_bar_time(
        eff_ms, _dg_tz, syminfo._session_starts,
        syminfo._opening_hours, 'weekday')


# Single-day ("D"/"1D") bar open cache: TradingView daily bars open at the
# trading day's session open (FX Monday opens Sunday 17:00, TSE 09:00), not at
# the calendar-midnight floor. Rebuilt when the session template is replaced
# (identity guard, like the ``_ttd``/``_tdc`` machinery).
_dbt_guard: tuple | None = None  # (opening_hours, session_starts) identities
_dbt_tz = None
_dbt_on: dict = {}
_dbt_starts: list | None = None


# noinspection PyProtectedMember
def _d_bar_time(current_time_ms: int) -> int:
    """
    Daily ("D") bar open time: the session open of the bar's trading day.

    The bar is resolved by its *last* instant (:func:`_chart_span_off_ms`), so
    the chart bar containing a session open counts as the new trading day's
    first bar. Falls back to the trading day's local midnight when no session
    template is known.

    :param current_time_ms: Chart bar open to resolve, in milliseconds
    :return: Bar opening time in milliseconds
    """
    global _dbt_guard, _dbt_tz, _dbt_on, _dbt_starts
    oh = syminfo._opening_hours
    ss = syminfo._session_starts
    if _dbt_guard is None or _dbt_guard[0] is not oh or _dbt_guard[1] is not ss:
        tz_name = getattr(syminfo, 'timezone', None)
        _dbt_tz = _parse_timezone(tz_name) if tz_name else None
        _dbt_on = _overnight_opens(oh or None, ss or None)
        _dbt_starts = ss or None
        _dbt_guard = (oh, ss)
    eff_sec = (current_time_ms + _chart_span_off_ms()) // 1000
    td = _trading_day(eff_sec, _dbt_tz, _dbt_on)
    return _trading_day_open_sec(td, _dbt_tz, _dbt_starts, _dbt_on) * 1000


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
    # noinspection PyProtectedMember
    modifier, multiplier = timeframe_module._process_tf(timeframe)
    if modifier in ('D', 'W', 'M') and multiplier > 1:
        # noinspection PyProtectedMember
        if (modifier, multiplier) == timeframe_module._process_tf(str(syminfo.period)):
            # The chart's own bars are the requested grid
            bar_time = current_time_ms
        else:
            bar_time = _dwm_bar_time(resampler, modifier, multiplier, current_time_ms)
    elif modifier == 'D':
        # noinspection PyProtectedMember
        if ('D', 1) == timeframe_module._process_tf(str(syminfo.period)):
            bar_time = current_time_ms
        else:
            # TradingView daily bars open at the trading day's session open
            # (the previous evening for overnight markets), not at midnight
            bar_time = _d_bar_time(current_time_ms)
    else:
        bar_time = resampler.get_bar_time(current_time_ms, *_intraday_session_args(timeframe))

    if session is None:
        # No session specified, return the bar time
        return bar_time
    if not isinstance(session, str):
        # A bool slips past the int(bars_back) overload guard (bool is an int):
        # it is not a valid session specification.
        return NA(int)

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
        # session opens (the only entries that can roll the trading day).
        _ttd_overnight_by_wd = _overnight_starts_by_weekday(opening_hours)
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


# Trading-day close cap for ``time_close``. TradingView closes a bar at
# ``min(bar open + timeframe span, end of the bar's trading day)``: the last —
# possibly shortened — bar of the day closes when the trading day ends, while
# intra-day gaps (lunch breaks) and continuous overnight sessions do not cap.
# ``_tdc_by_wd`` maps a trading day's weekday to its closing instant as a
# (time-of-day, calendar-day offset from the trading-day date) pair; rebuilt
# whenever ``syminfo._opening_hours`` is replaced (identity guard, like the
# ``_ttd`` machinery above).
_tdc_hours: list | None = None  # identity guard
_tdc_by_wd: dict[int, tuple[dt_time, int]] = {}  # weekday -> (end tod, +days)
_tdc_overnight_by_wd: dict[int, list[dt_time]] = {}
_tdc_tz = None


def _tdc_rebuild(opening_hours: list) -> None:
    """
    Rebuild the per-weekday trading-day close table from ``opening_hours``.

    Each interval's end instant is assigned to the trading day it closes —
    rolled to the next day when the instant lies inside an overnight session —
    and the latest end per trading day wins (the lunch-break morning end loses
    to the afternoon close).

    :param opening_hours: ``syminfo._opening_hours`` (``SymInfoInterval`` list)
    """
    global _tdc_hours, _tdc_by_wd, _tdc_overnight_by_wd, _tdc_tz

    _tdc_overnight_by_wd = _overnight_starts_by_weekday(opening_hours)
    _tdc_by_wd = _close_table_by_weekday(opening_hours, _tdc_overnight_by_wd)
    tz_name = getattr(syminfo, 'timezone', None)
    _tdc_tz = _parse_timezone(tz_name) if tz_name else None
    _tdc_hours = opening_hours


# noinspection PyProtectedMember
def _tdc_cap_ms(bar_open_ms: int, bar_close_ms: int) -> int:
    """
    Cap a computed bar close at the end of the bar's trading day.

    :param bar_open_ms: Bar opening time (UNIX ms)
    :param bar_close_ms: Uncapped close, i.e. open + timeframe span (UNIX ms)
    :return: ``min(bar_close_ms, trading day end)``; ``bar_close_ms`` unchanged
             when no session template is known or none ends on the bar's day
    """
    opening_hours = syminfo._opening_hours
    if not opening_hours:
        return bar_close_ms
    if opening_hours is not _tdc_hours:
        _tdc_rebuild(opening_hours)
    if not _tdc_by_wd:
        return bar_close_ms

    # The current chart bar (the hot path) reuses the runner-installed local
    # datetime instead of converting again.
    dt_local = _datetime if bar_open_ms == _time \
        else datetime.fromtimestamp(bar_open_ms / 1000, tz=_tdc_tz)
    trade_date = dt_local.date()

    # Overnight roll: a bar whose window reaches into a session opening this
    # calendar day and crossing midnight belongs to the next trading day
    # (same rule as ``time_tradingday``).
    opens = _tdc_overnight_by_wd.get(dt_local.weekday())
    if opens:
        for o in opens:
            session_open = dt_local.replace(
                hour=o.hour, minute=o.minute, second=o.second, microsecond=0)
            if bar_close_ms > session_open.timestamp() * 1000:
                trade_date += timedelta(days=1)
                break

    entry = _tdc_by_wd.get(trade_date.weekday())
    if entry is None:
        return bar_close_ms
    end_tod, offset = entry
    end_date = trade_date + timedelta(days=offset)
    day_end_ms = int(datetime(
        end_date.year, end_date.month, end_date.day,
        end_tod.hour, end_tod.minute, end_tod.second,
        tzinfo=_tdc_tz,
    ).timestamp() * 1000)
    if day_end_ms <= bar_open_ms:  # degenerate template — never close before the open
        return bar_close_ms
    return min(bar_close_ms, day_end_ms)


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
        # Close time of the current chart bar — capped at the trading-day end,
        # because the last bar of a session may be shortened
        try:
            close_ms = _time + timeframe_module.in_seconds(str(syminfo.period)) * 1000
            # noinspection PyProtectedMember
            chart_mod, chart_mult = timeframe_module._process_tf(str(syminfo.period))
        except (ValueError, AssertionError):
            return NA(int)
        if chart_mod in ('', 'S') or (chart_mod == 'D' and chart_mult == 1):
            close_ms = _tdc_cap_ms(_time, close_ms)
        return close_ms

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
    # noinspection PyProtectedMember
    modifier, multiplier = timeframe_module._process_tf(timeframe)
    if modifier in ('D', 'W', 'M') and multiplier > 1:
        # noinspection PyProtectedMember
        if (modifier, multiplier) == timeframe_module._process_tf(str(syminfo.period)):
            bar_start_time = current_time_ms
        else:
            bar_start_time = _dwm_bar_time(resampler, modifier, multiplier, current_time_ms)
    elif modifier == 'D':
        # noinspection PyProtectedMember
        if ('D', 1) == timeframe_module._process_tf(str(syminfo.period)):
            bar_start_time = current_time_ms
        else:
            # TradingView daily bars open at the trading day's session open
            # (the previous evening for overnight markets), not at midnight
            bar_start_time = _d_bar_time(current_time_ms)
    else:
        bar_start_time = resampler.get_bar_time(current_time_ms, *_intraday_session_args(timeframe))

    # Calculate bar close time by adding timeframe duration
    try:
        tf_seconds = timeframe_module.in_seconds(timeframe)
        bar_close_time = bar_start_time + (tf_seconds * 1000)  # Convert to milliseconds
    except (ValueError, AssertionError):
        return NA(int)

    if modifier in ('', 'S') or (modifier == 'D' and multiplier == 1):
        # TradingView closes the (possibly shortened) last bar of the day at
        # the trading-day end; weekly/monthly and multi-period close times
        # are not session-capped.
        bar_close_time = _tdc_cap_ms(bar_start_time, bar_close_time)

    if session is None:
        # No session specified, return the bar close time
        return bar_close_time
    if not isinstance(session, str):
        # A bool slips past the int(bars_back) overload guard (bool is an int):
        # it is not a valid session specification.
        return NA(int)

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
