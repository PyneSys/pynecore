"""
Builtin library of Pyne
"""
from typing import TYPE_CHECKING, TypeAlias, Any, TypeVar, overload as _typing_overload

if TYPE_CHECKING:
    from pynecore.types.type_checker import *
    from ..types.session import SessionInfo

import sys
import math as _math

from functools import lru_cache as _lru_cache
from datetime import datetime, timedelta, time as dt_time, date, UTC, \
    timezone as _fixed_timezone, tzinfo as _tzinfo, MINYEAR as _MINYEAR, MAXYEAR as _MAXYEAR

from pynecore.types.source import Source

from ..core.module_property import module_property, module_function_property
from ..core.series import SeriesImpl as _SeriesImpl
from ..core.script import script, input

from ..types.na import NA, na_int
from ..types import Series, PyneInt
from ..types.pine_types import pine_int
from ..types.plot_meta import PlotMeta
from . import syminfo  # This should be imported before core.datetime to avoid circular import!
from . import barstate, string, log, math, plot, hline, linefill, alert, dayofweek
from .plot import plot as _plot
from ..types.hline import HLine
from ..types.plot import Plot
from . import timeframe as timeframe_module
from . import session as session_module
from ._fixnan import fixnan

from pynecore.core.overload import overload
from pynecore.core.safe_convert import native_int_or as _native_int_or
from pynecore.core.datetime import parse_datestring as _parse_datestring, parse_timezone as _parse_timezone, \
    TimezoneNotFoundError, civil_days as _civil_days, julian_civil_days as _julian_civil_days, \
    GREGORIAN_CUTOVER_DAY as _GREGORIAN_CUTOVER_DAY, GREGORIAN_CYCLE_DAYS as _GREGORIAN_CYCLE_DAYS
from ..core.security import BarCalendar as _BarCalendar, actual_bar_close as _actual_bar_close, \
    dwm_period_end as _dwm_period_end
from ..core.resampler import (
    Resampler, ObservedDayCounter as _ObservedDayCounter,
    grid_mode as _grid_mode, overnight_opens as _overnight_opens,
    overnight_starts_by_weekday as _overnight_starts_by_weekday,
    close_table_by_weekday as _close_table_by_weekday,
    trading_day as _trading_day, trading_day_open_sec as _trading_day_open_sec,
    scheduled_day_open_sec as _scheduled_day_open_sec,
    observed_week_key as _observed_week_key,
)

# The interned typeless na: bare ``na`` in compiled scripts evaluates through
# ``is_na()`` on every bar, so its result must be a constant, not an allocation
_na_none: NA = NA(None)

# One full Gregorian leap cycle in milliseconds -- the period ``timestamp()``
# folds by to reach dates datetime cannot represent -- and the span it folds
# into: the 400 years starting at 2000-01-01
_GREGORIAN_CYCLE_MS: int = _GREGORIAN_CYCLE_DAYS * 86_400_000
_CYCLE_ANCHOR_DAY: int = _civil_days(2000, 1, 1)
_MIN_DATETIME_DAY: int = _civil_days(_MINYEAR, 1, 1)
_MAX_DATETIME_DAY: int = _civil_days(_MAXYEAR, 12, 31)

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

    '__dividends_tickerid', '__earnings_tickerid', '__splits_tickerid',

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

# A Pine int is a double at runtime: the runner publishes these as floats
bar_index: Series[int] = 0.0
last_bar_index: Series[int] = 0.0  # This always points to the bar_index

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

# Previous bar's close, published by the runner on every bar. A ta.* machine keeps
# its own state in the function, which advances per CALL — so one sitting inside an
# `if` branch cannot recover the close of a bar it did not run on. TradingView reads
# close[1] there, a global series that advances regardless, and this is that series'
# one-deep window. ``_last_close_bar`` is the bar it was rolled for: the live path
# republishes the same bar on every tick and must not roll the window with it.
# Not every builtin wants this: ``ta.sar`` deliberately keeps its own call-gated
# window, because TradingView's own sar dies on na when a call is skipped.
_last_close: float = _math.nan
_last_close_bar: int | None = None

# Store time as integer as in Pine Scripts timestamp format
_time: int = 0
# Open time (ms) of the chart bar AFTER the current one, 0 when none is known
# (the last historical bar, and every live bar). The historical loop already
# reads one bar ahead for ``barstate.islast``; HTF security confirmation uses
# the same peek to see an exchange's EARLY CLOSE, which no session schedule
# carries (see ``core/security.py::_get_confirmed_time``).
_next_time: int = 0
last_bar_time: PyneInt = 0.0

# Open times (ms) of the chart's recent bars, bar ``i`` in slot ``i % _BAR_OPENS_SIZE``.
# ``time(tf, bars_back)`` is evaluated on the chart bar ``bars_back`` bars back, whose
# open a gap in the data or between sessions keeps off the nominal grid. The runner
# fills the slot of every bar it publishes, so no slot within reach is stale.
_BAR_OPENS_SIZE = _SeriesImpl.MAXIMUM_MAX_BARS_BACK + 1
_bar_opens: list[int] = [0] * _BAR_OPENS_SIZE

# Datetime object in the exchange timezone
_datetime: datetime = datetime.fromtimestamp(0, UTC)

# Script settings from `script.indicator`, `script.strategy` or `script.library`
_script: script = None  # type: ignore[assignment]

# Chart (main-series) timeframe, propagated into request.security children so
# ``timeframe.main_period`` there reports the chart TF instead of the context's
# own period. ``None`` on the chart side, where ``_script`` carries it directly.
_main_timeframe: str | None = None

#: Timeframe declared by the running script (``indicator(..., timeframe='W')``).
#: Published by the ``@script.*`` decorator -- i.e. while the script module body is
#: still executing -- because the security transformer's module-level
#: ``__security_contexts__`` dict evaluates ``timeframe.period`` right there, and a
#: script running on a higher timeframe must already see THAT timeframe. ``None``
#: for a plain chart-timeframe script.
_script_timeframe: str | None = None

# Stores data to polot
_plot_data: dict[str, Any] = {}

# Plot-family registration state
_plot_meta: dict[str, PlotMeta] = {}  # id -> meta, insertion order = registration order
_plot_meta_new: list[PlotMeta] = []  # pending metas, drained only by the viz writer
_viz_dyn: dict[str, Any] = {}  # per-bar dynamic channels, cleared with _plot_data
_viz_seq: dict[str, int] = {}  # per-bar ordinal counters for bgcolor/barcolor/fill/hline

# Extra fields from CSV data (beyond OHLCV), populated each bar by ScriptRunner
extra_fields: dict[str, Any] = {}

# Lib semaphore - to prevent lib`s main function to do things it must not (plot, strategy things, etc.)
_lib_semaphore = False

# Security-child flag — True in a ``request.security`` worker process. The child
# re-runs the WHOLE script at the context's timeframe, while TradingView only ever
# evaluates the REQUESTED EXPRESSION there, so chart-level guards must not fire.
_in_security = False

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
    from .dayofweek import dayofweek


#
# Functions
#

# noinspection PyUnusedLocal,unused-parameter
def max_bars_back(var: Any, num: int) -> None:
    """
    Function sets the maximum number of bars that is available for historical reference of a given
    built-in or user variable.

    :param var: Series variable identifier for which history buffer should be resized.
    :param num: History buffer size which is the number of bars to keep.
    """


### Date / Time ###

# noinspection PyShadowingNames
def _get_dt(time: int | float | None = None, timezone: str | None = None) -> datetime | None:
    """ Get datetime object from time and timezone, None for an na time """
    if time is not None and not (time == time):  # is_na_arg
        return None
    dt = _datetime if time is None else datetime.fromtimestamp(time / 1000, UTC)
    assert dt is not None
    return dt.astimezone(_parse_timezone(timezone))


@overload
def timestamp(date_string: DateStr) -> PyneInt:  # It is more pythonic, but not supported by Pine Script
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
    return pine_int(int(dt.timestamp() * 1000))


# noinspection PyPep8Naming
@overload
def timestamp(dateString: DateStr) -> PyneInt:
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
              hour: int | float = 0, minute: int | float = 0, second: int | float = 0) -> PyneInt:
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
    # year, then carry the day through timedelta so the wall clock overflows
    # before the timezone conversion.
    # TradingView substitutes 0 for an na component instead of propagating the na
    # (measured: an na year/month/hour/second each lands on the year-0 / month-0 /
    # hour-0 / second-0 timestamp, never on na), so every component truncates
    # through the na-tolerant conversion.
    y = _native_int_or(year, 0)
    m = _native_int_or(month, 0)
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    d = _native_int_or(day, 0)
    # The clock components roll over into days too, so they are carried into the
    # day count before every calendar decision below. The wall clock is not moved
    # by the carry (the timedelta at the end shifts wall time, not absolute time),
    # but a rollover that leaves datetime's range -- hour 24 on the last
    # representable day -- now folds like any other out-of-range date instead of
    # raising OverflowError.
    clock_seconds = (_native_int_or(hour, 0) * 3600 + _native_int_or(minute, 0) * 60
                     + _native_int_or(second, 0))
    day_carry, second_of_day = divmod(clock_seconds, 86400)
    d += day_carry
    # TradingView runs the Julian calendar before 1582-10-15 and the Gregorian
    # one from that day on, the same hybrid Java's GregorianCalendar keeps, and
    # reads the ten dates the switch skipped as Julian too (measured:
    # timestamp(1582, 10, 5, 0, 0) == timestamp(1582, 10, 15, 0, 0), and
    # timestamp(1, 1, 1, 0, 0) is two days before the proleptic Gregorian
    # instant). datetime only knows the proleptic Gregorian calendar, so the
    # difference is carried as whole days on the wall clock -- exact, because
    # no zone observes DST that far back.
    gregorian_day = _civil_days(y, m, d)
    calendar_shift = 0
    if gregorian_day < _GREGORIAN_CUTOVER_DAY:
        calendar_shift = _julian_civil_days(y, m, d) - gregorian_day
    # The date may also fall outside datetime's 1..9999 years, which Pine does
    # not limit at all: TradingView keeps counting in both directions and
    # clamps nothing (measured: timestamp(9999, 31, 12, 23, 59) ==
    # timestamp(10001, 7, 12, 23, 59) == 253450598340000, and
    # timestamp(1000000, 1, 1, 0, 0) == 31494784780800000). The Gregorian
    # calendar repeats exactly every 400 years, so an unrepresentable date is
    # folded into 2000..2399 by whole cycles and their length added back to the
    # result. Month, day and weekday are preserved, so a named zone's DST rules
    # land on the same wall clock as the unfolded date would. Representable
    # dates keep their own year, and with it the zone's real offset history.
    target_day = gregorian_day + calendar_shift
    cycles = 0
    if not (_MINYEAR <= y <= _MAXYEAR and _MIN_DATETIME_DAY <= target_day <= _MAX_DATETIME_DAY):
        cycles = (target_day - _CYCLE_ANCHOR_DAY) // _GREGORIAN_CYCLE_DAYS
        y -= cycles * 400
    dt = datetime(y, m, 1, tzinfo=tz) + timedelta(
        days=d - 1 + calendar_shift, seconds=second_of_day
    )
    return pine_int(int(dt.timestamp() * 1000) + cycles * _GREGORIAN_CYCLE_MS)


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

def _uniq_title(title: str) -> str:
    """Return a title unique against the current bar's ``_plot_data`` keys."""
    c = 0
    t = title
    while t in _plot_data:
        t = title + ' ' + str(c)
        c += 1
    return t


def _auto_viz_title(seq_key: str, base: str) -> str:
    """
    Return a per-bar-stable default title for an untitled ``bgcolor``/``barcolor``/``fill``.

    Numbered by call order among untitled records of the same kind (``base``,
    ``base 1``, ``base 2``, ...), mirroring the id sequence so it stays stable
    across bars. ``seq_key`` must differ from the id counter keys.
    """
    n = _viz_seq.get(seq_key, 0)
    _viz_seq[seq_key] = n + 1
    return base if n == 0 else f'{base} {n}'


# noinspection PyProtectedMember,PyShadowingBuiltins
def plotshape(series: Any, title: str | None = None, style: Any = None, location: Any = None,
              color: Any = None, offset: int = 0, text: str | None = None, textcolor: Any = None,
              editable: bool = True, size: Any = None, show_last: int | None = None,
              display: Any = None, format: str | None = None, precision: int | None = None,
              force_overlay: bool = False) -> None:
    """
    Plot a shape marker on bars where ``series`` is true.

    :param series: Marker is drawn on bars where this value is true (na propagates)
    :param title: Plot title
    :param style: Shape style (``shape.*``); default ``shape.xcross``
    :param location: Marker location (``location.*``); default ``location.abovebar``
    :param color: Marker color
    :param offset: Horizontal shift in bars
    :param text: Text displayed with the marker
    :param textcolor: Color of the marker text
    :param editable: If true, the plot style is editable in the Format dialog
    :param size: Marker size (``size.*``); default ``size.auto``
    :param show_last: If set, only the last ``show_last`` markers are drawn
    :param display: Controls where the plot is displayed
    :param format: Formatting of the displayed values
    :param precision: Number of decimal places for the displayed values
    :param force_overlay: If true, the plot displays on the main chart pane
    """
    if _lib_semaphore:
        return
    if bar_index == 0:
        if sys._getframe(1).f_code.co_name != 'main':  # noqa
            raise RuntimeError("The plotshape function can only be called from the main function!")
    t = _uniq_title('Shapes' if title is None else title)
    # TradingView exports whatever the series holds, not a truthiness flag: a
    # numeric series marks the bar AND carries its value into the exported
    # column (measured: ``plotshape(cond ? high : na, location=location.abovebar)``
    # exports the price). Only a genuine bool is serialized, as 0/1.
    _plot_data[t] = int(series) if isinstance(series, bool) else series
    meta = _plot_meta.get(t)
    if meta is None:
        meta = PlotMeta(id=t, kind='shape', title=t, style=style, location=location, color=color,
                        offset=offset, text=text, textcolor=textcolor, editable=editable,
                        size=size, show_last=show_last, display=display, format=format,
                        precision=precision, force_overlay=force_overlay)
        _plot_meta[t] = meta
        _plot_meta_new.append(meta)
    if not meta.dynamic:
        if (color is not None and color is not meta.color) or \
                (textcolor is not None and textcolor is not meta.textcolor):
            meta.dynamic = True
            # The static meta record is already out — re-queue an updated one.
            _plot_meta_new.append(meta)
    if meta.dynamic:
        # Once dynamic, record every bar so reverts to the static colors are emitted
        _viz_dyn[t] = (color, textcolor)


# noinspection PyProtectedMember,PyShadowingBuiltins
def plotchar(series: Any, title: str | None = None, char: str | None = None, location: Any = None,
             color: Any = None, offset: int = 0, text: str | None = None, textcolor: Any = None,
             editable: bool = True, size: Any = None, show_last: int | None = None,
             display: Any = None, format: str | None = None, precision: int | None = None,
             force_overlay: bool = False) -> None:
    """
    Plot a character marker on bars where ``series`` is true.

    :param series: The value plotted (stored raw)
    :param title: Plot title
    :param char: The character to draw; default '◆'
    :param location: Marker location (``location.*``); default ``location.abovebar``
    :param color: Marker color
    :param offset: Horizontal shift in bars
    :param text: Text displayed with the marker
    :param textcolor: Color of the marker text
    :param editable: If true, the plot style is editable in the Format dialog
    :param size: Marker size (``size.*``); default ``size.auto``
    :param show_last: If set, only the last ``show_last`` markers are drawn
    :param display: Controls where the plot is displayed
    :param format: Formatting of the displayed values
    :param precision: Number of decimal places for the displayed values
    :param force_overlay: If true, the plot displays on the main chart pane
    """
    if _lib_semaphore:
        return
    if bar_index == 0:
        if sys._getframe(1).f_code.co_name != 'main':  # noqa
            raise RuntimeError("The plotchar function can only be called from the main function!")
    t = _uniq_title('Chars' if title is None else title)
    _plot_data[t] = int(series) if isinstance(series, bool) else series
    meta = _plot_meta.get(t)
    if meta is None:
        meta = PlotMeta(id=t, kind='char', title=t, char=char, location=location, color=color,
                        offset=offset, text=text, textcolor=textcolor, editable=editable,
                        size=size, show_last=show_last, display=display, format=format,
                        precision=precision, force_overlay=force_overlay)
        _plot_meta[t] = meta
        _plot_meta_new.append(meta)
    if not meta.dynamic:
        if (color is not None and color is not meta.color) or \
                (textcolor is not None and textcolor is not meta.textcolor):
            meta.dynamic = True
            # The static meta record is already out — re-queue an updated one.
            _plot_meta_new.append(meta)
    if meta.dynamic:
        # Once dynamic, record every bar so reverts to the static colors are emitted
        _viz_dyn[t] = (color, textcolor)


# noinspection PyProtectedMember,PyShadowingBuiltins
def plotarrow(series: Any, title: str | None = None, colorup: Any = None, colordown: Any = None,
              offset: int = 0, minheight: int = 5, maxheight: int = 100, editable: bool = True,
              show_last: int | None = None, display: Any = None, format: str | None = None,
              precision: int | None = None, force_overlay: bool = False) -> None:
    """
    Plot up/down arrows sized by the magnitude of ``series``.

    :param series: Arrow direction/length; positive draws up, negative draws down
    :param title: Plot title
    :param colorup: Color of up arrows
    :param colordown: Color of down arrows
    :param offset: Horizontal shift in bars
    :param minheight: Minimum arrow height in pixels
    :param maxheight: Maximum arrow height in pixels
    :param editable: If true, the plot style is editable in the Format dialog
    :param show_last: If set, only the last ``show_last`` arrows are drawn
    :param display: Controls where the plot is displayed
    :param format: Formatting of the displayed values
    :param precision: Number of decimal places for the displayed values
    :param force_overlay: If true, the plot displays on the main chart pane
    """
    if _lib_semaphore:
        return
    if bar_index == 0:
        if sys._getframe(1).f_code.co_name != 'main':  # noqa
            raise RuntimeError("The plotarrow function can only be called from the main function!")
    t = _uniq_title('Arrows' if title is None else title)
    _plot_data[t] = int(series) if isinstance(series, bool) else series
    meta = _plot_meta.get(t)
    if meta is None:
        meta = PlotMeta(id=t, kind='arrow', title=t, colorup=colorup, colordown=colordown,
                        offset=offset, minheight=minheight, maxheight=maxheight, editable=editable,
                        show_last=show_last, display=display, format=format, precision=precision,
                        force_overlay=force_overlay)
        _plot_meta[t] = meta
        _plot_meta_new.append(meta)
    if not meta.dynamic:
        if (colorup is not None and colorup is not meta.colorup) or \
                (colordown is not None and colordown is not meta.colordown):
            meta.dynamic = True
            # The static meta record is already out — re-queue an updated one.
            _plot_meta_new.append(meta)
    if meta.dynamic:
        # Once dynamic, record every bar so reverts to the static colors are emitted
        _viz_dyn[t] = (colorup, colordown)


# noinspection PyProtectedMember,PyShadowingBuiltins,shadowing-names
def plotcandle(open: Any, high: Any, low: Any, close: Any, title: str | None = None,
               color: Any = None, wickcolor: Any = None, editable: bool = True,
               show_last: int | None = None, bordercolor: Any = None, display: Any = None,
               format: str | None = None, precision: int | None = None,
               force_overlay: bool = False) -> None:
    """
    Plot OHLC candles from the four supplied series.

    :param open: Open value of the candle
    :param high: High value of the candle
    :param low: Low value of the candle
    :param close: Close value of the candle
    :param title: Plot title
    :param color: Body color
    :param wickcolor: Wick color
    :param editable: If true, the plot style is editable in the Format dialog
    :param show_last: If set, only the last ``show_last`` candles are drawn
    :param bordercolor: Border color
    :param display: Controls where the plot is displayed
    :param format: Formatting of the displayed values
    :param precision: Number of decimal places for the displayed values
    :param force_overlay: If true, the plot displays on the main chart pane
    """
    if _lib_semaphore:
        return
    if bar_index == 0:
        if sys._getframe(1).f_code.co_name != 'main':  # noqa
            raise RuntimeError("The plotcandle function can only be called from the main function!")
    base = 'Candles' if title is None else title
    c = 0
    t = base
    while f"{t} (open)" in _plot_data:
        t = base + ' ' + str(c)
        c += 1
    _plot_data[f"{t} (open)"] = open
    _plot_data[f"{t} (high)"] = high
    _plot_data[f"{t} (low)"] = low
    _plot_data[f"{t} (close)"] = close
    meta = _plot_meta.get(t)
    if meta is None:
        meta = PlotMeta(id=t, kind='candle', title=t, color=color, wickcolor=wickcolor,
                        bordercolor=bordercolor, editable=editable, show_last=show_last,
                        display=display, format=format, precision=precision,
                        force_overlay=force_overlay)
        _plot_meta[t] = meta
        _plot_meta_new.append(meta)
    if not meta.dynamic:
        if (color is not None and color is not meta.color) or \
                (wickcolor is not None and wickcolor is not meta.wickcolor) or \
                (bordercolor is not None and bordercolor is not meta.bordercolor):
            meta.dynamic = True
            # The static meta record is already out — re-queue an updated one.
            _plot_meta_new.append(meta)
    if meta.dynamic:
        # Once dynamic, record every bar so reverts to the static colors are emitted
        _viz_dyn[t] = (color, wickcolor, bordercolor)


# noinspection PyProtectedMember,PyShadowingBuiltins,shadowing-names
def plotbar(open: Any, high: Any, low: Any, close: Any, title: str | None = None, color: Any = None,
            editable: bool = True, show_last: int | None = None, display: Any = None,
            format: str | None = None, precision: int | None = None,
            force_overlay: bool = False) -> None:
    """
    Plot OHLC bars from the four supplied series.

    :param open: Open value of the bar
    :param high: High value of the bar
    :param low: Low value of the bar
    :param close: Close value of the bar
    :param title: Plot title
    :param color: Bar color
    :param editable: If true, the plot style is editable in the Format dialog
    :param show_last: If set, only the last ``show_last`` bars are drawn
    :param display: Controls where the plot is displayed
    :param format: Formatting of the displayed values
    :param precision: Number of decimal places for the displayed values
    :param force_overlay: If true, the plot displays on the main chart pane
    """
    if _lib_semaphore:
        return
    if bar_index == 0:
        if sys._getframe(1).f_code.co_name != 'main':  # noqa
            raise RuntimeError("The plotbar function can only be called from the main function!")
    base = 'Bars' if title is None else title
    c = 0
    t = base
    while f"{t} (open)" in _plot_data:
        t = base + ' ' + str(c)
        c += 1
    _plot_data[f"{t} (open)"] = open
    _plot_data[f"{t} (high)"] = high
    _plot_data[f"{t} (low)"] = low
    _plot_data[f"{t} (close)"] = close
    meta = _plot_meta.get(t)
    if meta is None:
        meta = PlotMeta(id=t, kind='bar', title=t, color=color, editable=editable,
                        show_last=show_last, display=display, format=format, precision=precision,
                        force_overlay=force_overlay)
        _plot_meta[t] = meta
        _plot_meta_new.append(meta)
    if meta.dynamic:
        # Once dynamic, record every bar so a return to the static color is emitted
        _viz_dyn[t] = color
    elif color is not None and color is not meta.color:
        _viz_dyn[t] = color
        meta.dynamic = True
        # The static meta record is already out — re-queue an updated one.
        _plot_meta_new.append(meta)


# noinspection PyProtectedMember
def bgcolor(color: Any = None, offset: int = 0, editable: bool = True, show_last: int | None = None,
            title: str | None = None, display: Any = None, force_overlay: bool = False) -> None:
    """
    Fill the background of bars with ``color``.

    :param color: Background color for the current bar (na leaves the bar unpainted)
    :param offset: Horizontal shift in bars
    :param editable: If true, the fill is editable in the Format dialog
    :param show_last: If set, only the last ``show_last`` bars are painted
    :param title: Plot title
    :param display: Controls where the fill is displayed
    :param force_overlay: If true, the fill displays on the main chart pane
    """
    if _lib_semaphore:
        return
    if bar_index == 0:
        if sys._getframe(1).f_code.co_name != 'main':  # noqa
            raise RuntimeError("The bgcolor function can only be called from the main function!")
    if title is None:
        title = _auto_viz_title('bgcolor:title', 'Background color')
    n = _viz_seq.get('bgcolor', 0)
    _viz_seq['bgcolor'] = n + 1
    key = f'bgcolor#{n}'
    meta = _plot_meta.get(key)
    if meta is None:
        meta = PlotMeta(id=key, kind='bgcolor', title=title, offset=offset, editable=editable,
                        show_last=show_last, display=display, force_overlay=force_overlay,
                        dynamic=True)
        _plot_meta[key] = meta
        _plot_meta_new.append(meta)
    # Record every bar (na/None -> null "off") so paint/unpaint transitions are emitted
    _viz_dyn[key] = color


# noinspection PyProtectedMember
def barcolor(color: Any = None, offset: int = 0, editable: bool = True, show_last: int | None = None,
             title: str | None = None, display: Any = None) -> None:
    """
    Color the price bars with ``color``.

    :param color: Bar color for the current bar (na leaves the bar unchanged)
    :param offset: Horizontal shift in bars
    :param editable: If true, the coloring is editable in the Format dialog
    :param show_last: If set, only the last ``show_last`` bars are colored
    :param title: Plot title
    :param display: Controls where the coloring is displayed
    """
    if _lib_semaphore:
        return
    if bar_index == 0:
        if sys._getframe(1).f_code.co_name != 'main':  # noqa
            raise RuntimeError("The barcolor function can only be called from the main function!")
    if title is None:
        title = _auto_viz_title('barcolor:title', 'Bar color')
    n = _viz_seq.get('barcolor', 0)
    _viz_seq['barcolor'] = n + 1
    key = f'barcolor#{n}'
    meta = _plot_meta.get(key)
    if meta is None:
        meta = PlotMeta(id=key, kind='barcolor', title=title, offset=offset, editable=editable,
                        show_last=show_last, display=display, dynamic=True)
        _plot_meta[key] = meta
        _plot_meta_new.append(meta)
    # Record every bar (na/None -> null "off") so color/unpaint transitions are emitted
    _viz_dyn[key] = color


# Positional parameter orders of Pine's three ``fill`` overloads; ``fill()`` maps
# ``*args`` onto one of these depending on the runtime shape of the call.
_FILL_PLOT_PARAMS = ('plot1', 'plot2', 'color', 'title', 'editable', 'show_last',
                     'fillgaps', 'display')
_FILL_HLINE_PARAMS = ('hline1', 'hline2', 'color', 'title', 'editable', 'fillgaps', 'display')
_FILL_GRADIENT_PARAMS = ('plot1', 'plot2', 'top_value', 'bottom_value', 'top_color',
                         'bottom_color', 'title', 'display', 'fillgaps', 'editable')


# noinspection PyProtectedMember,incorrect-docstring
def fill(*args: Any, **kwargs: Any) -> None:
    """
    Fill the area between two plots or two hlines.

    Three call shapes are accepted:

    - ``fill(plot1, plot2, color, title, editable, show_last, fillgaps, display)``
    - ``fill(hline1, hline2, color, title, editable, fillgaps, display)``
    - ``fill(plot1, plot2, top_value, bottom_value, top_color, bottom_color, title,
      display, fillgaps, editable)`` — vertical gradient

    Positional arguments are bound to the hline shape when the first argument is an
    ``hline``, to the gradient shape when the third argument is a numeric ``top_value``
    rather than a color, and to the plot shape otherwise.

    :param plot1: First plot object (``hline1`` for the hline shape)
    :param plot2: Second plot object (``hline2`` for the hline shape)
    :param color: Solid fill color
    :param title: Plot title
    :param editable: If true, the fill is editable in the Format dialog
    :param show_last: If set, only the last ``show_last`` bars are filled (plot shape only)
    :param fillgaps: If true, the fill continues across gaps (na values)
    :param display: Controls where the fill is displayed
    :param top_value: Value mapped to ``top_color`` in gradient mode
    :param bottom_value: Value mapped to ``bottom_color`` in gradient mode
    :param top_color: Color at ``top_value`` in gradient mode
    :param bottom_color: Color at ``bottom_value`` in gradient mode
    """
    if _lib_semaphore:
        return
    if bar_index == 0:
        if sys._getframe(1).f_code.co_name != 'main':  # noqa
            raise RuntimeError("The fill function can only be called from the main function!")
    if args:
        if isinstance(args[0], HLine):
            names = _FILL_HLINE_PARAMS
        else:
            a2 = args[2] if len(args) > 2 else None
            # Gradient shape: the third positional is ``top_value`` — a number (or an na
            # value followed by a gradient color where the plot shape would have a bool).
            if (isinstance(a2, (int, float)) and not isinstance(a2, bool)) or \
                    (isinstance(a2, NA) and len(args) > 4 and not isinstance(args[4], bool)):
                names = _FILL_GRADIENT_PARAMS
            else:
                names = _FILL_PLOT_PARAMS
        if len(args) > len(names):
            raise TypeError(f"fill() takes at most {len(names)} positional arguments")
        for name, value in zip(names, args):
            if name in kwargs:
                raise TypeError(f"fill() got multiple values for argument '{name}'")
            kwargs[name] = value
    plot1: Plot | HLine | None = kwargs.get('plot1') if 'plot1' in kwargs else kwargs.get('hline1')
    plot2: Plot | HLine | None = kwargs.get('plot2') if 'plot2' in kwargs else kwargs.get('hline2')
    color = kwargs.get('color')
    title = kwargs.get('title')
    if title is None:
        title = _auto_viz_title('fill:title', 'Plots Background')
    editable = kwargs.get('editable', True)
    show_last = kwargs.get('show_last')
    fillgaps = kwargs.get('fillgaps', False)
    display = kwargs.get('display')
    top_value = kwargs.get('top_value')
    bottom_value = kwargs.get('bottom_value')
    top_color = kwargs.get('top_color')
    bottom_color = kwargs.get('bottom_color')
    n = _viz_seq.get('fill', 0)
    _viz_seq['fill'] = n + 1
    key = f'fill#{n}'
    meta = _plot_meta.get(key)
    created = meta is None
    if meta is None:
        id1 = None if plot1 is None else plot1.id
        id2 = None if plot2 is None else plot2.id
        if isinstance(plot1, HLine):
            meta = PlotMeta(id=key, kind='fill', title=title, color=color, editable=editable,
                            show_last=show_last, fillgaps=fillgaps, display=display,
                            hline1=id1, hline2=id2)
        else:
            meta = PlotMeta(id=key, kind='fill', title=title, color=color, editable=editable,
                            show_last=show_last, fillgaps=fillgaps, display=display,
                            plot1=id1, plot2=id2)
        _plot_meta[key] = meta
        _plot_meta_new.append(meta)
    if top_color is not None or bottom_color is not None \
            or top_value is not None or bottom_value is not None:
        _viz_dyn[key] = (top_value, bottom_value, top_color, bottom_color)
        if not meta.dynamic:
            meta.dynamic = True
            if not created:
                # Already emitted as static — re-queue so an updated meta record
                # (dynamic: true) precedes this bar's color delta.
                _plot_meta_new.append(meta)
    elif meta.dynamic:
        # Once dynamic, record every bar so a return to the static color is emitted
        _viz_dyn[key] = color
    elif color is not None and color is not meta.color:
        _viz_dyn[key] = color
        meta.dynamic = True
        # The static meta record is already out — re-queue an updated one.
        _plot_meta_new.append(meta)


### Alert ###

def alertcondition(*_, **__):
    """
    Define alert condition. Currently implemented as no-op.

    In the future this could be used to define alert conditions
    that can be triggered based on boolean expressions.
    """
    if _lib_semaphore:
        return
    if bar_index == 0:  # Only check if it is the first bar for performance reasons
        # Check if it is called from the main function
        if sys._getframe(1).f_code.co_name != 'main':  # noqa
            raise RuntimeError("The alertcondition function can only be called from the main function!")


### Other ###

def is_na(x: Any = None) -> bool | NA:
    """
    Check if the value is NA.

    inf/-inf/nan floats all count as na here, even though they participate in
    arithmetic and comparisons as normal IEEE-754 values.

    :param x: The value to test, or a type to build an NA sentinel of
    """
    # The parameter is named ``x`` because Pine accepts ``na(x = close)`` as a
    # named argument, and the compiler emits Pine's own keyword verbatim.
    # The branches are ordered by how often each face is called: a script tests
    # float values on every bar, na sentinels far less often, and builds an na
    # OF a type only where one is declared.
    if isinstance(x, float):
        return not _math.isfinite(x)
    if isinstance(x, NA):
        return True
    if x is None:
        return _na_none
    # A type or a subscripted generic builds an na of that type. The generic is
    # matched on ``__origin__`` because that is what both kinds carry:
    # ``list[float]`` is a types.GenericAlias while a subscripted user Generic
    # like ``Matrix[float]`` is a typing._GenericAlias -- and the second one is
    # what a ``matrix<float> m = na`` declaration produces.
    if x is not NA and (isinstance(x, type) or getattr(x, '__origin__', None) is not None):
        # na.pyi deliberately types NA(x) as x itself (so na sentinels flow as
        # values in user scripts), which contradicts the honest annotation here
        return NA(x)  # pyright: ignore[reportReturnType]
    return x is NA


# In Pine Script, na is both a property and a function; any narrower type than
# Any produces false positives on one of its three faces (bare value, na(x)
# predicate, na(type) constructor). The bare-VALUE face never reaches this
# object in transformed code: ModulePropertyTransformer rewrites a
# value-position ``na`` to ``lib._na_none`` (the interned typeless NA), so this
# name only ever gets CALLED — and stays a plain function, the fastest callable
# there is. Calling lib functions with this object from plain Python (outside
# the @pyne transform) is therefore NOT a supported na-value spelling.
na: Any = is_na

_T = TypeVar('_T')


# The static face of nz: the result is the source's own type with its na stripped.
# Spelled through the typing alias on purpose: a decorator named ``overload`` would make
# the lib type registry read these as a runtime overload group
@_typing_overload
def nz(source: NA[_T] | _T) -> _T: ...


@_typing_overload
def nz(source: NA[_T] | _T, replacement: _T) -> _T: ...


def nz(source: Any, replacement: Any = 0) -> Any:
    """
    Replace NA values with a replacement value or 0 if not specified

    Uses the na() predicate semantics for floats: inf/-inf/nan are all na, so
    ``nz(inf, -5)`` is ``-5``.

    :param source: The source value
    :param replacement: The replacement value, default is 0
    :return: The source value if it is not NA, otherwise the replacement value
    """
    if isinstance(source, float):
        return source if _math.isfinite(source) else replacement
    if isinstance(source, NA):
        # nz on a bool answers a bool (TV: ``nz(na_bool)`` is false)
        return bool(replacement) if source.type is bool else replacement
    return source


# Prefix of TradingView's corporate-action data feeds. The three
# ``__*_tickerid()`` helpers below are undocumented Pine built-ins that map a
# regular ticker identifier onto such a feed, so that ``request.security()``
# can read dividend/earnings/split events as a daily series.
_CORPORATE_ACTION_PREFIX = 'ESD_FACTSET'


def _corporate_action_tickerid(tickerid: str, feed: str) -> str:
    """
    Build a corporate-action feed identifier from a ticker identifier.

    TV-measured (2026-07-27): ``NASDAQ:AAPL`` becomes
    ``ESD_FACTSET:NASDAQ;AAPL;DIVIDENDS`` — a purely syntactic rewrite, applied
    to any exchange/symbol pair without validating either.

    :param tickerid: A ``EXCHANGE:SYMBOL`` ticker identifier
    :param feed: The feed name (``DIVIDENDS``, ``EARNINGS``, ``SPLITS``)
    :return: The corporate-action feed identifier
    """
    exchange, _, symbol = str(tickerid).partition(':')
    return f"{_CORPORATE_ACTION_PREFIX}:{exchange};{symbol};{feed}"


def __dividends_tickerid(tickerid: str) -> str:
    """
    Ticker identifier of the dividends feed of a symbol.

    :param tickerid: A ``EXCHANGE:SYMBOL`` ticker identifier
    :return: The dividends feed identifier
    """
    return _corporate_action_tickerid(tickerid, 'DIVIDENDS')


def __earnings_tickerid(tickerid: str) -> str:
    """
    Ticker identifier of the earnings feed of a symbol.

    :param tickerid: A ``EXCHANGE:SYMBOL`` ticker identifier
    :return: The earnings feed identifier
    """
    return _corporate_action_tickerid(tickerid, 'EARNINGS')


def __splits_tickerid(tickerid: str) -> str:
    """
    Ticker identifier of the splits feed of a symbol.

    :param tickerid: A ``EXCHANGE:SYMBOL`` ticker identifier
    :return: The splits feed identifier
    """
    return _corporate_action_tickerid(tickerid, 'SPLITS')


#
# Module properties
#

### Date / Time ###

# noinspection PyShadowingNames
@module_function_property
def dayofmonth(time: int | float | None = None, timezone: str | None = None) -> PyneInt:
    """
    Day of the month

    :param time: The time to get the day of the month from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The day of the month
    """
    dt = _get_dt(time, timezone)
    return na_int if dt is None else pine_int(dt.day)


# noinspection PyShadowingNames
@module_function_property
def hour(time: int | float | None = None, timezone: str | None = None) -> PyneInt:
    """
    Hour of the day

    :param time: The time to get the hour of the day from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The hour of the day
    """
    dt = _get_dt(time, timezone)
    return na_int if dt is None else pine_int(dt.hour)


# noinspection PyShadowingNames
@module_function_property
def minute(time: int | float | None = None, timezone: str | None = None) -> PyneInt:
    """
    Minute of the hour

    :param time: The time to get the minute of the hour from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The minute of the hour
    """
    dt = _get_dt(time, timezone)
    return na_int if dt is None else pine_int(dt.minute)


# noinspection PyShadowingNames
@module_function_property
def month(time: int | float | None = None, timezone: str | None = None) -> PyneInt:
    """
    Month of the year

    :param time: The time to get the month of the year from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The month of the year
    """
    dt = _get_dt(time, timezone)
    return na_int if dt is None else pine_int(dt.month)


# noinspection PyShadowingNames
@module_function_property
def second(time: int | float | None = None, timezone: str | None = None) -> PyneInt:
    """
    Second of the minute

    :param time: The time to get the second of the minute from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The second of the minute
    """
    dt = _get_dt(time, timezone)
    return na_int if dt is None else pine_int(dt.second)


### Session parsing and validation helpers ###

_MINUTES_PER_DAY = 24 * 60
# Session day numbers run 1 = Sunday ... 7 = Saturday
_ALL_SESSION_DAYS = frozenset(range(1, 8))
_WEEKDAY_SESSION_DAYS = frozenset(range(2, 7))
_SESSION_DIGITS = frozenset('0123456789')
_SESSION_DAY_DIGITS = frozenset('1234567')


def _parse_session_string(session: str, timezone: str | None = None) -> tuple['SessionInfo', ...]:
    """
    Parse a session string into one SessionInfo per time range.

    Grammar: ``range[,range...][:days]``, several such sections joined by ``|``. A
    range is ``HHMM-HHMM``; days are digits 1 (Sunday) .. 7 (Saturday) naming the day
    each run's last minute falls on. A lone range without days runs every day; in any
    other form a section without days is the default section, which runs on the
    weekdays (2..6) no other section names, and a section naming a day replaces the
    earlier ones on that day. Surrounding whitespace is ignored.

    A string that is empty, blank or does not start with a digit -- a session name
    such as "regular" -- is the symbol's own session: its opening hours read in the
    given timezone (see :func:`_symbol_session_infos`). "24x7" is the all-day session.

    :param session: Session string (e.g., "0930-1600", "0930-1600:23456",
                    "0400-0700,0900-1300:23456", "0930-1600|1000-1300:7")
    :param timezone: Timezone string, defaults to exchange timezone if None
    :return: One SessionInfo per time range and section, in the order they were written
    :raises ValueError: If session string is invalid
    """
    # The exchange fallback is resolved here rather than inside the cached parse so
    # a later symbol change takes effect instead of returning the previous run's
    # timezone -- the same split ``core.datetime.parse_timezone`` uses.
    if timezone is None:
        # Use a safe default if syminfo.timezone is not available
        timezone = getattr(syminfo, 'timezone', 'UTC')
        # Handle NA values
        if hasattr(timezone, '__class__') and 'NA' in timezone.__class__.__name__:
            timezone = 'UTC'
    session_infos = _parse_session_string_cached(session, timezone)
    if session_infos is None:
        return _symbol_session_infos(timezone)
    return session_infos


@_lru_cache(maxsize=128)
def _parse_session_string_cached(session: str, timezone: str) -> 'tuple[SessionInfo, ...] | None':
    """
    Parse a fully resolved session specification, memoized on its arguments.

    A script's session strings are constants, so every bar re-derives the same
    ranges: ``time(tf, session)`` alone reached this parser ~50 times per bar on a
    multi-context strategy. The result is a tuple of frozen ``SessionInfo`` and
    callers only read it, so one instance can be shared by every caller.

    :param session: Session string (grammar in :func:`_parse_session_string`)
    :param timezone: Timezone string, already resolved to a concrete zone
    :return: One SessionInfo per time range, in the order they were written, or
             ``None`` when the string selects the symbol's own session
    :raises ValueError: If session string is invalid
    """
    from ..types.session import SessionInfo

    spec = session.strip()
    # MEASURED (TradingView, CAPITALCOM:AAPL and BTCUSD 10-minute charts, 2026-09-25):
    # "", " ", "invalid", "regular", "abc-def", "invalid:23456", "abc:23456",
    # "a0930-1600", ",1100-1400", "|1100-1400", "-1400", ":23456" and "x" all ran on the
    # symbol's session, read in the timezone argument when one was given. A leading
    # digit starts a specification, which halts the script when it is malformed ("0930",
    # "0930-16", "1100-", "0930-1600:8", ...); the script is kept running with na here.
    # "extended" is the symbol's extended-hours session, which the symbol data does not
    # carry, so it resolves to the regular one.
    if not spec or spec[0] not in _SESSION_DIGITS:
        return None
    if spec == '24x7':
        # MEASURED (same charts): "24x7" is "0000-0000" on every bar, AAPL included
        return (SessionInfo(start_time=dt_time(0), end_time=dt_time(0),
                            days=_ALL_SESSION_DAYS, timezone=timezone),)

    if ':' not in spec and ',' not in spec and '|' not in spec:
        # MEASURED (CAPITALCOM:BTCUSD, "UTC"): a lone range such as "1100-1400", also
        # with surrounding whitespace, runs on all seven days, while "1100-1400,",
        # "1100-1400:", "1100-1400|" and "1100-1400,1500-1600" run Monday to Friday.
        return tuple(
            SessionInfo(start_time=start_time, end_time=end_time,
                        days=_ALL_SESSION_DAYS, timezone=timezone)
            for start_time, end_time in _parse_session_ranges(spec, session)
        )

    sections = spec.split('|')
    if not sections[-1]:
        # A trailing separator ends the list ("1100-1400|" is "1100-1400")
        sections.pop()
    parsed: list[tuple[tuple[tuple[dt_time, dt_time], ...], frozenset[int] | None]] = []
    has_default = False
    for section in sections:
        parts = section.split(':')
        if len(parts) > 2:
            raise ValueError(f"Invalid session section {section!r} in session: {session}")
        ranges = _parse_session_ranges(parts[0], session)
        if len(parts) == 1 or not parts[1]:
            # MEASURED: "0930-1600|1700-1800" and "1100-1400:|1200-1300" halt with
            # "duplicated default section"
            if has_default:
                raise ValueError(f"Duplicated default section in session: {session}")
            has_default = True
            parsed.append((ranges, None))
        else:
            parsed.append((ranges, _parse_session_days(parts[1], session)))

    # MEASURED (CAPITALCOM:BTCUSD, "UTC"): a later section naming a day replaces the
    # earlier ones on that day ("1100-1400:2|1200-1300:2" ran 12:00-13:00 on Mondays
    # only), and the default section takes the weekdays nobody named -- "1100-1400|
    # 1200-1300:7" ran 11:00-14:00 Monday to Friday, 12:00-13:00 on Saturday and
    # nothing on Sunday.
    own_days: list[frozenset[int]] = [_WEEKDAY_SESSION_DAYS] * len(parsed)
    claimed: frozenset[int] = frozenset()
    for index in range(len(parsed) - 1, -1, -1):
        days = parsed[index][1]
        if days is not None:
            own_days[index] = days - claimed
            claimed |= days
    for index, (_, days) in enumerate(parsed):
        if days is None:
            own_days[index] = _WEEKDAY_SESSION_DAYS - claimed

    return tuple(
        SessionInfo(start_time=start_time, end_time=end_time, days=own_days[index],
                    timezone=timezone)
        for index, (ranges, _) in enumerate(parsed) if own_days[index]
        for start_time, end_time in ranges
    )


def _parse_session_ranges(text: str, session: str) -> tuple[tuple[dt_time, dt_time], ...]:
    """
    Parse the comma-separated ranges of one session section.

    :param text: The ranges part of the section; empty in an empty section
    :param session: The whole session string, for error messages
    :return: ``(start, end)`` time-of-day pairs; an end at or before the start is on
             the next day, equal endpoints span the whole day
    :raises ValueError: If a range is malformed or out of range
    """
    if not text:
        return ()
    entries = text.split(',')
    if not entries[-1]:
        # A trailing comma ends the list; an empty entry anywhere else is malformed
        entries.pop()
    ranges: list[tuple[dt_time, dt_time]] = []
    for entry in entries:
        # MEASURED: "0930-16", "930-1600", "09300-1600", "0930-1600a", "0930-abcd",
        # "1100 -1400" and an empty entry all halt the script
        if (len(entry) != 9 or entry[4] != '-'
                or not _SESSION_DIGITS.issuperset(entry[:4] + entry[5:])):
            raise ValueError(f"Invalid session range {entry!r} in session: {session}")
        start_minutes = int(entry[:2]) * 60 + int(entry[2:4])
        end_minutes = int(entry[5:7]) * 60 + int(entry[7:])
        # MEASURED (CAPITALCOM:BTCUSD, "UTC", start and end hours up to 99, minutes up
        # to 99): both endpoints are plain minute counts taken modulo one day --
        # "0930-2500" ran 09:30 -> 01:00, "0930-3400" 09:30 -> 10:00 of the same day,
        # "0960-1600" 10:00-16:00, "2500-1200" 01:00-12:00. Any end is accepted after a
        # start before 24:00. A start at 24:00 or later is accepted only when, moved
        # back one day, it opens a run of at most a day ending at the given end (00:00
        # read as 24:00): "2500-2500", "2500-0101", "4759-4759" and "2400-0000" run,
        # "2500-0100", "2500-2600", "2400-2500", "4700-2300" and "4800-4800" halt.
        if start_minutes >= _MINUTES_PER_DAY:
            end_of_run = end_minutes or _MINUTES_PER_DAY
            if not (start_minutes - _MINUTES_PER_DAY < end_of_run <= start_minutes
                    < 2 * _MINUTES_PER_DAY):
                raise ValueError(f"Invalid session range {entry!r} in session: {session}")
        start_minutes %= _MINUTES_PER_DAY
        end_minutes %= _MINUTES_PER_DAY
        ranges.append((dt_time(*divmod(start_minutes, 60)), dt_time(*divmod(end_minutes, 60))))
    return tuple(ranges)


def _parse_session_days(text: str, session: str) -> frozenset[int]:
    """
    Parse the days of one session section.

    :param text: Day digits, 1 (Sunday) .. 7 (Saturday), in any order
    :param session: The whole session string, for error messages
    :return: The day numbers
    :raises ValueError: If a character is not a day digit
    """
    # MEASURED: ":0", ":8", ":9", ":abc", ":23456a", ":23456," and ":2345 6" halt the
    # script with "Invalid days specification"; a repeated digit is accepted
    if not _SESSION_DAY_DIGITS.issuperset(text):
        raise ValueError(f"Invalid days specification {text!r} in session: {session}")
    return frozenset(int(day_char) for day_char in text)


# The symbol's own session per timezone, rebuilt when ``syminfo._opening_hours`` is
# replaced (identity guard, like the ``_ttd``/``_tdc`` machinery).
_ssi_hours: list | None = None
_ssi_by_tz: dict[str, 'tuple[SessionInfo, ...]'] = {}


# The schedule is lib's own ``syminfo`` module state, installed by the script runner
# noinspection PyProtectedMember
def _symbol_session_infos(timezone: str) -> tuple['SessionInfo', ...]:
    """
    The symbol's own session: its opening hours as a session specification.

    Each opening-hours interval is one run opening on its weekday, closing on the next
    day when its end is at or before its start, named by the day its last minute falls
    on. An end with seconds is the last instant inside the session, so a ``23:59:59``
    end is midnight. The wall-clock times are read in ``timezone``, which may differ
    from the exchange timezone. A symbol without opening hours is a continuous market.

    :param timezone: Timezone to read the session in
    :return: One SessionInfo per distinct range
    """
    global _ssi_hours
    opening_hours = syminfo._opening_hours
    if opening_hours is not _ssi_hours:
        _ssi_by_tz.clear()
        _ssi_hours = opening_hours
    session_infos = _ssi_by_tz.get(timezone)
    if session_infos is not None:
        return session_infos

    from ..types.session import SessionInfo

    # MEASURED (TradingView, 10-minute charts, 2026-09-25): time(tf, "") and time(tf,
    # "invalid") equal the plain time(tf) and time_close(tf) on every bar for "45", "60",
    # "240" and timeframe_bars_back 3 on CAPITALCOM:AAPL (20058 bars), BTCUSD (20501) and
    # GOLD (20429). "D" reports the symbol's runs -- AAPL 09:30-16:00, BTCUSD 17:00 ->
    # 17:00, GOLD 18:00 -> 17:00 New York -- and with the "UTC" argument the same wall
    # clocks read in UTC (AAPL 09:30-16:00 UTC).
    days_by_range: dict[tuple[int, int], set[int]] = {}
    for day, start, end in opening_hours or ():
        start_minutes = start.hour * 60 + start.minute
        end_minutes = (end.hour * 60 + end.minute
                       + (1 if end.second or end.microsecond else 0)) % _MINUTES_PER_DAY
        last_day = day if end_minutes == 0 or end_minutes > start_minutes else day + 1
        days_by_range.setdefault((start_minutes, end_minutes), set()).add(
            (last_day + 2) % 7 or 7)
    if not days_by_range:
        days_by_range[(0, 0)] = set(_ALL_SESSION_DAYS)
    session_infos = tuple(
        SessionInfo(start_time=dt_time(*divmod(start_minutes, 60)),
                    end_time=dt_time(*divmod(end_minutes, 60)),
                    days=frozenset(days), timezone=timezone)
        for (start_minutes, end_minutes), days in sorted(days_by_range.items())
    )
    _ssi_by_tz[timezone] = session_infos
    return session_infos


def _is_bar_in_session(bar_time_ms: int, session_infos: 'tuple[SessionInfo, ...]') -> bool:
    """
    Check if a bar's opening time falls within any of the specified session ranges.

    :param bar_time_ms: Bar time in milliseconds (UNIX timestamp)
    :param session_infos: Session ranges of one session specification -- every range
                          shares the same day set and timezone, and the bar is in
                          session as soon as one of them contains it
    :return: True if bar is within session, False otherwise
    """
    return _session_occurrence(bar_time_ms, session_infos) is not None


def _intraday_session_wall_ms(wall: datetime) -> tuple[int, timedelta]:
    """
    Resolve a session endpoint's wall clock the way the intraday session mask does.

    A wall clock inside a spring gap takes the offset AFTER the change and one a
    fall-back repeats takes its FIRST reading -- on both nights that is the larger
    of the two offsets the date offers for it.

    :param wall: The endpoint, timezone-aware, as written in the session string
    :return: The instant in milliseconds and the offset it was resolved with
    """
    offset = wall.utcoffset()
    other = wall.replace(fold=1).utcoffset()
    assert offset is not None and other is not None
    if other > offset:
        offset = other
    instant = wall.replace(tzinfo=_fixed_timezone(offset))
    return int(instant.timestamp() * 1000), offset


def _intraday_session_bounds(day: date, close_day: date, session_info: 'SessionInfo',
                             tz: _tzinfo) -> tuple[int, int]:
    """
    Open and close of one session run as the INTRADAY session mask draws it.

    MEASURED (BINANCE:BTCUSDT@30, ``time(timeframe.period, session, tz)``, nine
    session shapes in New York and five in London over the 2025 spring and fall
    changes and the 2026 spring one): each endpoint is its own wall clock,
    resolved by :func:`_intraday_session_wall_ms` -- "2200-0230" ran 22:00 EST ->
    02:30 read as EDT, "1700-0300" 17:00 EDT -> 03:00 EST, "1800-0100" closed at
    the first 01:00 of the fall-back night, and "0130-0230" is empty on the gap
    date. The one exception is a run CLOSING on the very wall clock the change
    happens at (02:00 in New York both ways, 01:00 in London in spring and 02:00
    in the fall): the whole run, its open included, then takes the offset after
    the change, keeping its nominal length -- "1700-0200" ran 16:00 EST -> 01:00
    EST in spring and 18:00 EDT -> 03:00 EDT in the fall, London's "2000-0100"
    19:00 GMT -> 00:00 GMT.

    :param day: Opening date of the run, in the session's timezone
    :param close_day: Closing date of the run
    :param session_info: The session range
    :param tz: The session's timezone
    :return: ``(open_ms, close_ms)``
    """
    open_wall = datetime.combine(day, session_info.start_time, tzinfo=tz)
    close_wall = datetime.combine(close_day, session_info.end_time, tzinfo=tz)
    start_ms, _ = _intraday_session_wall_ms(open_wall)
    end_ms, close_offset = _intraday_session_wall_ms(close_wall)

    # The minute before the close tells whether the close IS the changing wall
    # clock: it has to exist (a close deeper inside the gap has a gap minute
    # before it) and to carry another offset than the close does.
    before = close_wall - timedelta(minutes=1)
    before_offset = before.utcoffset()
    before_late = before.replace(fold=1).utcoffset()
    assert before_offset is not None and before_late is not None
    if before_late <= before_offset != close_offset:
        start_ms = int(open_wall.replace(tzinfo=_fixed_timezone(close_offset)).timestamp()
                       * 1000)
    return start_ms, end_ms


@_lru_cache(maxsize=512)
def _session_occurrences_opening_on(day: date,
                                    session_infos: 'tuple[SessionInfo, ...]',
                                    intraday: bool = False
                                    ) -> tuple[tuple[int, int], ...]:
    """
    Every occurrence of a session specification that OPENS on a calendar date.

    An occurrence is one concrete run of a session range: the "0300-1200" range
    on a given date, or -- for an overnight range -- the run that opens on this
    date and closes on the next one.

    The opening endpoint is plain wall clock on its own date and the closing one
    is the end of the run's LAST MINUTE, so a run crossing a daylight-saving
    change keeps its nominal clock times and changes length in real time.
    MEASURED (BINANCE:BTCUSDT@30, New York, daily requests over both 2025
    changes): "1900-0400" ran 19:00 EST -> 04:00 EDT across the spring change
    (eight hours) and ten hours across the fall-back, "0100-0500" three and
    five, "0200-1000" seven and eight, "0130-0230" one and two. The sessions
    closing at 02:00 -- the hour the clock jumps at -- kept their nominal span
    on both nights instead: 01:59 is the LAST minute of "1700-0200", and on the
    fall-back date that is the first, still-EDT 01:59, an hour before the
    unambiguous 02:00 EST the closing wall clock alone would name.

    Those are the bounds a DAILY request reports and gates on. On a change night
    TradingView's INTRADAY session mask draws the same run differently whenever
    an endpoint lands on the changing hour, so an intraday request takes its
    bounds from :func:`_intraday_session_bounds` instead.

    KNOWN DIVERGENCE: on the EVENING BEFORE the fall change, where nothing shifts
    at all, TradingView appends an extra hour to the intraday close of every run
    with an endpoint on the changing wall clock, returning na from ``time()`` on
    its first bar and a value from ``time_close()`` on the very same bar
    (2025-11-01 06:00 for "1700-0200", and 14:00 for "0200-1000"). No single
    occurrence can reproduce a bar that is in and out of session at once; it
    costs one 30-minute bar a year. The measurement is recorded in
    ``docs/overview/compatibility.md`` ("Known TradingView quirks not reproduced").

    :param day: Opening date of the runs, in the session's timezone
    :param session_infos: Session ranges of one session specification
    :param intraday: Whether the runs are drawn by the intraday session mask
    :return: ``(open_ms, close_ms)`` pairs, ordered by opening time
    """
    from datetime import datetime, timedelta

    if not session_infos:
        return ()
    tz = _parse_timezone(session_infos[0].timezone)
    occurrences: list[tuple[int, int]] = []
    for session_info in session_infos:
        start_minutes = session_info.start_time.hour * 60 + session_info.start_time.minute
        end_minutes = session_info.end_time.hour * 60 + session_info.end_time.minute
        # An end at or before the start closes on the next date. Equal endpoints
        # are Pine's all-day session ("0000-0000", the default of input.session),
        # which spans the full 24 hours and closes on the next date too.
        close_day = day + timedelta(days=1) if end_minutes <= start_minutes else day
        if intraday:
            start_ms, end_ms = _intraday_session_bounds(day, close_day, session_info, tz)
            if end_ms <= start_ms:
                continue
        else:
            start_ms = int(datetime.combine(day, session_info.start_time,
                                            tzinfo=tz).timestamp() * 1000)
            # The close is the end of the session's LAST MINUTE rather than the
            # closing wall clock itself, which only differs when the change falls
            # inside that minute: a session closing at 02:00 on a fall-back date
            # runs to the FIRST 01:59, an hour before the unambiguous 02:00.
            last_minute = datetime.combine(close_day, session_info.end_time,
                                           tzinfo=tz) - timedelta(minutes=1)
            end_ms = int(last_minute.timestamp() * 1000) + 60_000
        # The day mask names the weekday the occurrence's LAST minute falls on,
        # not its opening one. MEASURED (BINANCE:BTCUSDT@60): "1700-0200:23456"
        # ran Sunday 17:00 -> Monday 02:00 up to Thursday 17:00 -> Friday 02:00,
        # and "1700-1700:23456" ran Sunday 17:00 -> Monday 17:00 up to Thursday
        # -> Friday, both leaving the Friday evening out. "0000-0000:23456" --
        # whose close lands exactly on midnight -- still ran Monday through
        # Friday, which the last minute gets right and the closing instant would
        # not.
        last_dt = datetime.fromtimestamp((end_ms - 60_000) / 1000, tz)
        tv_weekday = (last_dt.weekday() + 2) % 7 or 7
        if tv_weekday in session_info.days:
            occurrences.append((start_ms, end_ms))
    occurrences.sort()
    return tuple(occurrences)


@_lru_cache(maxsize=256)
def _session_occurrence(bar_time_ms: int,
                        session_infos: 'tuple[SessionInfo, ...]',
                        intraday: bool = False) -> tuple[int, int] | None:
    """
    Bounds of the single session occurrence a chart bar falls into.

    MEASURED LAW (BINANCE:BTCUSDT at 30 minutes, 4 hours and daily): the bar
    belongs to an occurrence only when its OPENING time lies in ``[open, close)``
    -- a bar merely overlapping the run is NOT in session. On the 30-minute
    chart "0945-1015" ran on the 10:00 bar alone (never the straddling 09:30
    one), "0915-0945" on 09:30 alone (never 09:00), "1545-1615" on 16:00 alone,
    and "1550-1555" -- a run entirely inside one bar -- on no bar at all. The
    same holds where the bar is longer than the run: at 4 hours "0930-1600" ran
    on the 12:00 bar only and "0100-0300" on none, and the daily bar (opening
    00:00) was in "2200-0200" but in neither "0930-1600" nor "1200-1300".
    The latest containing occurrence wins when ranges overlap.

    :param bar_time_ms: Chart bar open in milliseconds
    :param session_infos: Session ranges of one session specification
    :param intraday: Whether the runs are drawn by the intraday session mask
    :return: ``(open_ms, close_ms)`` of the occurrence, or ``None`` when the bar
             is outside every range
    """
    from datetime import datetime, timedelta

    if not session_infos:
        return None
    tz = _parse_timezone(session_infos[0].timezone)
    bar_dt = datetime.fromtimestamp(bar_time_ms / 1000, tz)

    best: tuple[int, int] | None = None
    # An overnight range that opened yesterday still covers this bar, so both
    # dates have to be offered to the containment test. The intraday mask can
    # also pull TOMORROW's run an hour ahead of its own midnight (see
    # :func:`_intraday_session_bounds`).
    for day_offset in (-1, 0, 1) if intraday else (-1, 0):
        day = (bar_dt + timedelta(days=day_offset)).date()
        for start_ms, end_ms in _session_occurrences_opening_on(day, session_infos, intraday):
            if start_ms <= bar_time_ms < end_ms:
                if best is None or start_ms > best[0]:
                    best = (start_ms, end_ms)
    return best


@_lru_cache(maxsize=256)
def _previous_session_occurrence(before_ms: int,
                                 session_infos: 'tuple[SessionInfo, ...]',
                                 intraday: bool = False) -> tuple[int, int] | None:
    """
    The latest session occurrence opening strictly before an instant.

    Walking ``timeframe_bars_back`` over session bars leaves the current
    occurrence, so the series has to be continued backwards one run at a time.

    :param before_ms: Instant the occurrence has to open before, in milliseconds
    :param session_infos: Session ranges of one session specification
    :param intraday: Whether the runs are drawn by the intraday session mask
    :return: ``(open_ms, close_ms)``, or ``None`` when no day in the previous
             two months runs the session
    """
    from datetime import datetime, timedelta

    if not session_infos:
        return None
    tz = _parse_timezone(session_infos[0].timezone)
    day = datetime.fromtimestamp(before_ms / 1000, tz).date()
    for day_offset in range(60):
        best: tuple[int, int] | None = None
        for occurrence in _session_occurrences_opening_on(day - timedelta(days=day_offset),
                                                          session_infos, intraday):
            if occurrence[0] < before_ms and (best is None or occurrence[0] > best[0]):
                best = occurrence
        if best is not None:
            return best
    return None


@_lru_cache(maxsize=256)
def _next_session_occurrence(after_ms: int,
                             session_infos: 'tuple[SessionInfo, ...]',
                             intraday: bool = False) -> tuple[int, int] | None:
    """
    The earliest session occurrence opening strictly after an instant.

    :param after_ms: Instant the occurrence has to open after, in milliseconds
    :param session_infos: Session ranges of one session specification
    :param intraday: Whether the runs are drawn by the intraday session mask
    :return: ``(open_ms, close_ms)``, or ``None`` when no day in the next two
             months runs the session
    """
    if not session_infos:
        return None
    tz = _parse_timezone(session_infos[0].timezone)
    day = datetime.fromtimestamp(after_ms / 1000, tz).date()
    best: tuple[int, int] | None = None
    for day_offset in range(61):
        # The intraday mask can pull a run an hour ahead of its own midnight, so the
        # date after the first one with a match may still hold an earlier run
        found = best is not None
        for occurrence in _session_occurrences_opening_on(day + timedelta(days=day_offset),
                                                          session_infos, intraday):
            if occurrence[0] > after_ms and (best is None or occurrence[0] < best[0]):
                best = occurrence
        if found:
            return best
    return best


@_lru_cache(maxsize=512)
def _session_day_bounds(day: date,
                        session_infos: 'tuple[SessionInfo, ...]') -> tuple[int, int] | None:
    """
    Bounds of the daily session bar of one trading day.

    A trading day is every run whose last minute falls on its date -- the day the day
    mask names -- and its daily bar spans from the first of them to the last, so the
    gaps between the ranges of a multi-range session lie inside it.

    :param day: Date of the trading day, in the session's timezone
    :param session_infos: Session ranges of one session specification
    :return: ``(open_ms, close_ms)``, or ``None`` when the session does not run that day
    """
    # MEASURED (TradingView, CAPITALCOM:BTCUSD 10-minute chart, "UTC", 2026-09-25):
    # time("D", s) and time_close("D", s) of "1100-1400,1500-1600" ran 11:00 -> 16:00 on
    # every bar of that span, 14:xx included, "2200-0200,1100-1400" Sunday 22:00 -> Monday
    # 14:00 and "2500-1200,1300-1400" 01:00 -> 14:00, while their "60" buckets stayed na
    # in the gaps.
    tz = _parse_timezone(session_infos[0].timezone)
    bounds: tuple[int, int] | None = None
    for opening_day in (day - timedelta(days=1), day):
        for start_ms, end_ms in _session_occurrences_opening_on(opening_day, session_infos):
            if datetime.fromtimestamp((end_ms - 60_000) / 1000, tz).date() != day:
                continue
            if bounds is None:
                bounds = (start_ms, end_ms)
            else:
                bounds = (min(bounds[0], start_ms), max(bounds[1], end_ms))
    return bounds


@_lru_cache(maxsize=256)
def _session_day_occurrence(bar_time_ms: int,
                            session_infos: 'tuple[SessionInfo, ...]') -> tuple[int, int] | None:
    """
    Bounds of the daily session bar a chart bar falls into.

    The bar belongs to a trading day when its opening time lies in the day's
    ``[open, close)`` (see :func:`_session_day_bounds`); the latest opening day wins
    when two overlap.

    :param bar_time_ms: Chart bar open in milliseconds
    :param session_infos: Session ranges of one session specification
    :return: ``(open_ms, close_ms)``, or ``None`` when the bar is outside every day
    """
    if not session_infos:
        return None
    tz = _parse_timezone(session_infos[0].timezone)
    bar_day = datetime.fromtimestamp(bar_time_ms / 1000, tz).date()
    best: tuple[int, int] | None = None
    # A trading day closing tomorrow can open today (an overnight range)
    for day in (bar_day, bar_day + timedelta(days=1)):
        bounds = _session_day_bounds(day, session_infos)
        if (bounds is not None and bounds[0] <= bar_time_ms < bounds[1]
                and (best is None or bounds[0] > best[0])):
            best = bounds
    return best


@_lru_cache(maxsize=256)
def _previous_session_day(before_ms: int,
                          session_infos: 'tuple[SessionInfo, ...]') -> tuple[int, int] | None:
    """
    The latest daily session bar opening strictly before an instant.

    :param before_ms: Instant the day has to open before, in milliseconds
    :param session_infos: Session ranges of one session specification
    :return: ``(open_ms, close_ms)``, or ``None`` when no day in the previous two
             months runs the session
    """
    if not session_infos:
        return None
    tz = _parse_timezone(session_infos[0].timezone)
    day = datetime.fromtimestamp(before_ms / 1000, tz).date() + timedelta(days=1)
    for day_offset in range(62):
        bounds = _session_day_bounds(day - timedelta(days=day_offset), session_infos)
        if bounds is not None and bounds[0] < before_ms:
            return bounds
    return None


@_lru_cache(maxsize=256)
def _next_session_day(after_ms: int,
                      session_infos: 'tuple[SessionInfo, ...]') -> tuple[int, int] | None:
    """
    The earliest daily session bar opening strictly after an instant.

    :param after_ms: Instant the day has to open after, in milliseconds
    :param session_infos: Session ranges of one session specification
    :return: ``(open_ms, close_ms)``, or ``None`` when no day in the next two
             months runs the session
    """
    if not session_infos:
        return None
    tz = _parse_timezone(session_infos[0].timezone)
    day = datetime.fromtimestamp(after_ms / 1000, tz).date()
    for day_offset in range(62):
        bounds = _session_day_bounds(day + timedelta(days=day_offset), session_infos)
        if bounds is not None and bounds[0] > after_ms:
            return bounds
    return None


@_lru_cache(maxsize=64)
def _session_period_anchor(period_date: date,
                           session_infos: 'tuple[SessionInfo, ...]') -> int | None:
    """
    Opening time of the first session occurrence belonging to a calendar period.

    Weekly and monthly session bars open at the period's first session open, so
    a period whose first days are masked out (a Saturday-starting month against
    a ``:23456`` session) walks forward to the first day the session runs on. An
    occurrence belongs to the day its LAST MINUTE falls on -- the same rule the
    day mask uses -- so an overnight session anchors the period on the EVENING
    BEFORE its first session day, while one closing exactly at midnight still
    anchors on the day itself. MEASURED (BINANCE:BTCUSDT@30, New York): the
    weekly bar of "1700-0200:23456" opened Sunday 17:00, and its monthly bar
    opened 2024-12-31 17:00 for January 2025 and 2025-02-02 17:00 for February;
    "0000-0000:23456" opened Monday 00:00 weekly, 2025-01-01 00:00 for January
    and 2025-02-03 00:00 for February -- the first two February days being a
    masked-out weekend.

    :param period_date: First calendar date of the period, in exchange time
    :param session_infos: Session ranges of one session specification
    :return: Opening time in milliseconds, or ``None`` when no day in the next
             six weeks runs the session
    """
    from datetime import datetime, timedelta

    if not session_infos:
        return None
    tz = _parse_timezone(session_infos[0].timezone)
    # The opening date is not derivable from the period's first date by clock
    # arithmetic -- a run closing at midnight ends on its own opening date, an
    # overnight one on the next -- so the runs themselves are enumerated and the
    # earliest one whose last minute lands inside the period wins. Only the day
    # before the period can reach into it, no range being longer than a day.
    for day_offset in range(-1, 42):
        day = period_date + timedelta(days=day_offset)
        best: int | None = None
        for start_ms, end_ms in _session_occurrences_opening_on(day, session_infos):
            last_dt = datetime.fromtimestamp((end_ms - 60_000) / 1000, tz)
            if last_dt.date() < period_date:
                continue
            if best is None or start_ms < best:
                best = start_ms
        if best is not None:
            return best
    return None


def _period_start_date(period_date: date, modifier: str, steps: int) -> date:
    """
    First calendar date of the period ``steps`` periods away from another one.

    :param period_date: First calendar date of the reference period
    :param modifier: Period modifier, ``'W'`` or ``'M'``
    :param steps: Periods to move, negative walks back
    :return: First calendar date of the requested period
    """
    if modifier == 'W':
        return period_date + timedelta(days=7 * steps)
    month_index = period_date.month - 1 + steps
    return period_date.replace(year=period_date.year + month_index // 12,
                               month=month_index % 12 + 1, day=1)


def _session_bucket_count(occurrence: tuple[int, int], step_ms: int) -> int:
    """
    Number of intraday session bars an occurrence is tiled into.

    :param occurrence: ``(open_ms, close_ms)`` of the occurrence
    :param step_ms: Requested bar length in milliseconds
    :return: Bucket count, at least one -- the last bucket may be cut short
    """
    return max(1, -(-(occurrence[1] - occurrence[0]) // step_ms))


def _session_bar_bounds(chart_time_ms: int, session_infos: 'tuple[SessionInfo, ...]',
                        modifier: str, multiplier: int,
                        bar_start_ms: int, bar_close_ms: int,
                        steps: int = 0) -> tuple[int, int] | None:
    """
    Open and close of the session bar ``time``/``time_close`` report.

    MEASURED LAW (BINANCE:BTCUSDT@30, ``0300-1200``/``1700-0200``/``0930-1600``
    New York and ``0900-1130``/``0930-1600`` exchange time): a session does not
    merely filter the requested timeframe's grid, it replaces that grid with a
    series of session bars.

    - The na gate always tests the CURRENT CHART BAR, whatever the requested
      timeframe is: ``"D"``, ``"60"`` and the chart's own period each left
      exactly the same 10647 of 28391 bars defined.
    - A daily request reports the session's own bounds -- 08:00 open and 17:00
      close for that day's occurrence -- not the calendar day's. A multi-range
      session's trading day runs from its first range's open to its last range's
      close (:func:`_session_day_bounds`).
    - An intraday request tiles the occurrence: buckets are counted from the
      session open, so an off-grid session shifts the whole series, and the
      final bucket is truncated at the session close.
    - Weekly and monthly requests are never na: their session bars open at the
      period's first session open. On an intraday chart they close at the NEXT
      period's first session open (weekly close landed on the following Monday's
      open, not on Friday's close), so consecutive bars tile the whole timeline;
      on a daily, weekly or monthly chart they close at the end of the period's
      last session run.

    Multi-period daily/weekly/monthly requests ("3D", "2W") keep the plain grid:
    their session anchoring is unmeasured.

    :param chart_time_ms: Chart bar open the call is evaluated on, in milliseconds
    :param session_infos: Session ranges of one session specification
    :param modifier: Requested timeframe modifier ('', 'S', 'D', 'W' or 'M')
    :param multiplier: Requested timeframe multiplier
    :param bar_start_ms: Plain grid bar open of the requested timeframe
    :param bar_close_ms: Plain grid bar close of the requested timeframe
    :param steps: Session bars to walk back, the positive ``timeframe_bars_back``; a
                  negative count walks forward to a bar that has not opened yet
    :return: ``(open_ms, close_ms)``, or ``None`` when the bar is out of session
    """
    if modifier in ('W', 'M'):
        if multiplier == 1:
            exchange_tz = _parse_timezone(getattr(syminfo, 'timezone', None) or 'UTC')
            # The walk starts from the calendar period of the CHART bar's date and the
            # offset is applied to its result: the requested grid's bar opens at the
            # session open of the period's first trading day, which can fall on the
            # calendar day before the period. MEASURED (TradingView, CAPITALCOM:BTCUSD
            # 10-minute chart, 2026-09-25): time("W", "") opens Sunday 17:00, with the
            # run closing on the week's Monday.
            chart_date = datetime.fromtimestamp(chart_time_ms / 1000, exchange_tz).date()
            period_date = (chart_date - timedelta(days=chart_date.weekday()) if modifier == 'W'
                           else chart_date.replace(day=1))
            # A session anchor can fall on either side of its calendar period's
            # first date -- an overnight session opens the evening BEFORE its
            # first session day -- so the period holding the chart bar is walked
            # to instead of assumed.
            for _ in range(4):
                open_ms = _session_period_anchor(period_date, session_infos)
                if open_ms is None:
                    break
                if chart_time_ms < open_ms:
                    period_date = _period_start_date(period_date, modifier, -1)
                    continue
                next_ms = _session_period_anchor(_period_start_date(period_date, modifier, 1),
                                                 session_infos)
                if next_ms is not None and chart_time_ms >= next_ms:
                    period_date = _period_start_date(period_date, modifier, 1)
                    continue
                break
            period_date = _period_start_date(period_date, modifier, -steps)
            open_ms = _session_period_anchor(period_date, session_infos)
            close_ms = _session_period_anchor(_period_start_date(period_date, modifier, 1),
                                              session_infos)
            if open_ms is not None and close_ms is not None:
                if _chart_modifier() not in ('', 'S'):
                    # MEASURED (TradingView, 2026-09-25, CAPITALCOM:EURUSD on D, 2D, W and
                    # M, GOLD, AAPL and BTCUSD on D): on a daily, weekly or monthly chart
                    # time_close("W"/"M", session) is the end of the period's last run --
                    # Friday 17:00 for "" on EURUSD, Sunday 23:59 for "0000-2359" -- while
                    # the 10- and 60-minute charts report the next period's first session
                    # open.
                    last_run = _previous_session_occurrence(close_ms, session_infos)
                    if last_run is not None and last_run[0] >= open_ms:
                        close_ms = last_run[1]
                return open_ms, close_ms
        return bar_start_ms, bar_close_ms

    if modifier not in ('', 'S', 'D') or (modifier == 'D' and multiplier != 1):
        return bar_start_ms, bar_close_ms

    if modifier == 'D':
        # A daily request reports whole trading days, gaps between ranges included
        occurrence = _session_day_occurrence(chart_time_ms, session_infos)
        if occurrence is None:
            if steps == 0:
                return None
            if steps > 0:
                # MEASURED: out of session a daily request is NOT na once an offset
                # is given, it reports the occurrence the offset lands on counted
                # from the one the chart bar is heading into -- which is one step
                # further than the last one that ran.
                occurrence = _previous_session_day(chart_time_ms + 1, session_infos)
                steps -= 1
            else:
                # A day that has not opened yet is counted from the one the chart
                # bar is heading into
                occurrence = _next_session_day(chart_time_ms, session_infos)
                steps += 1
        while occurrence is not None and steps > 0:
            occurrence = _previous_session_day(occurrence[0], session_infos)
            steps -= 1
        while occurrence is not None and steps < 0:
            occurrence = _next_session_day(occurrence[0], session_infos)
            steps += 1
        return occurrence

    occurrence = _session_occurrence(chart_time_ms, session_infos, True)
    if occurrence is None and steps == 0:
        return None

    # An intraday request tiles the occurrence itself instead of filtering the
    # plain grid: the first bucket opens at the session open even when that is
    # off the grid, and the last one is cut short at the session close. MEASURED
    # (BINANCE:BTCUSDT@30, requested "60"): "0930-1600" ran 09:30-10:30,
    # 10:30-11:30, ... and closed 15:30-16:00, while "0900-1130" closed its
    # 11:00 bucket at 11:30, not at 12:00.
    step_ms = (multiplier if modifier == 'S' else multiplier * 60) * 1000
    if occurrence is not None:
        index = (chart_time_ms - occurrence[0]) // step_ms - steps
    elif steps < 0:
        # A bar that has not opened yet is counted from the run the chart bar is
        # heading into
        occurrence = _next_session_occurrence(chart_time_ms, session_infos, True)
        if occurrence is None:
            return None
        index = -steps - 1
    else:
        # MEASURED (TradingView, 2026-09-25, "UTC" sessions, requested "60"):
        # out of session the value of the run's LAST bucket holds -- the walk
        # starts from the last bucket of the run that has already closed and
        # reports the bucket it lands on. CAPITALCOM:BTCUSD@30 "0900-1130" gave
        # 10:00 for offset 1 and the previous day's 11:00 for offset 3 on every
        # bar from 11:30 to the next 09:00; "0930-1600" offset 2 gave 13:30 from
        # 16:00 on, on BTCUSD@30 and on every out-of-session bar of AAPL@10.
        occurrence = _previous_session_occurrence(chart_time_ms + 1, session_infos, True)
        if occurrence is None:
            return None
        index = _session_bucket_count(occurrence, step_ms) - 1 - steps
    while index < 0:
        previous = _previous_session_occurrence(occurrence[0], session_infos, True)
        if previous is None:
            index = 0
            break
        occurrence = previous
        index += _session_bucket_count(occurrence, step_ms)
    while index >= (count := _session_bucket_count(occurrence, step_ms)):
        following = _next_session_occurrence(occurrence[0], session_infos, True)
        if following is None:
            index = count - 1
            break
        occurrence = following
        index -= count
    start_ms = occurrence[0] + index * step_ms
    return start_ms, min(start_ms + step_ms, occurrence[1])


def _session_grid_args() -> tuple:
    """
    Build the session arguments of :meth:`Resampler.get_bar_time` for an intraday or a
    single-period weekly/monthly timeframe.

    An intraday grid is anchored to the exchange session open, the way TradingView
    aligns intraday HTF bars; anchoring is a no-op for on-hour / 24-7 markets. A weekly
    or monthly bar is the trading week (month) and opens at the session open of its
    first trading day -- for an overnight market the previous evening (an FX week opens
    Sunday 17:00 New York), not at local midnight.

    Without a session template the calendar floor still runs in the exchange timezone
    (TradingView week/month boundaries are exchange-local), not in the machine's local
    time.

    :return: ``(tz, session_starts, opening_hours)`` with a session template,
             ``(tz,)`` without one, ``()`` when the timezone is unknown too.
    """
    tz_name = getattr(syminfo, 'timezone', None)
    tz = _parse_timezone(tz_name) if tz_name else None
    session_starts = getattr(syminfo, '_session_starts', None)
    if not session_starts:
        return (tz,) if tz is not None else ()
    return tz, session_starts, getattr(syminfo, '_opening_hours', None) or None


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
            run_tf = timeframe_module._current_period()
            chart_sec = timeframe_module._in_seconds(run_tf)
            chart_mod, _ = timeframe_module._process_tf(run_tf)
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


# noinspection PyProtectedMember
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
        run_tf = timeframe_module._current_period()
        chart_mod, _ = timeframe_module._process_tf(run_tf)
        if chart_mod in ('', 'S'):
            return timeframe_module._in_seconds(run_tf) * 1000 - 1
    except (ValueError, AssertionError):
        pass
    return 0


# noinspection PyProtectedMember
def _chart_modifier() -> str | None:
    """
    Timeframe modifier of the bars the script runs on.

    :return: '', 'S', 'D', 'W' or 'M'; ``None`` when the period cannot be parsed
    """
    try:
        return timeframe_module._process_tf(timeframe_module._current_period())[0]
    except (ValueError, AssertionError):
        return None


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
# (bar open, span offset) -> the previous scheduled trading day's open, which a
# ``timeframe_bars_back`` walk asks for on every chart bar of the same period
_ptd_cache: dict[tuple[int, int], int] = {}


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
    oh = syminfo._opening_hours
    ss = syminfo._session_starts
    if _dbt_guard is None or _dbt_guard[0] is not oh or _dbt_guard[1] is not ss:
        _dbt_rebuild(oh, ss)
    eff_sec = (current_time_ms + _chart_span_off_ms()) // 1000
    td = _trading_day(eff_sec, _dbt_tz, _dbt_on)
    return _trading_day_open_sec(td, _dbt_tz, _dbt_starts, _dbt_on) * 1000


def _dbt_rebuild(opening_hours: list | None, session_starts: list | None) -> None:
    """
    Rebuild the daily-open cache from the session template.

    :param opening_hours: ``syminfo._opening_hours``
    :param session_starts: ``syminfo._session_starts``
    """
    global _dbt_guard, _dbt_tz, _dbt_on, _dbt_starts
    tz_name = getattr(syminfo, 'timezone', None)
    _dbt_tz = _parse_timezone(tz_name) if tz_name else None
    _dbt_on = _overnight_opens(opening_hours or None, session_starts or None)
    _dbt_starts = session_starts or None
    _dbt_guard = (opening_hours, session_starts)
    _ptd_cache.clear()


# noinspection PyProtectedMember
def _previous_trading_day_open_ms(open_ms: int, span_off_ms: int) -> int:
    """
    Session open of the scheduled trading day before the one a bar belongs to.

    Days the session template schedules no open for (the weekend of a Monday to Friday
    market) are skipped; holidays are not, the template does not know them. Without a
    template the instant before ``open_ms`` is returned.

    :param open_ms: Open of the bar, in milliseconds
    :param span_off_ms: Offset from the bar's open to its last instant, which decides its
                        trading day (:func:`_chart_span_off_ms`)
    :return: The previous scheduled trading day's session open, in milliseconds
    """
    oh = syminfo._opening_hours
    ss = syminfo._session_starts
    if _dbt_guard is None or _dbt_guard[0] is not oh or _dbt_guard[1] is not ss:
        _dbt_rebuild(oh, ss)
    key = (open_ms, span_off_ms)
    previous_open_ms = _ptd_cache.get(key)
    if previous_open_ms is not None:
        return previous_open_ms
    previous_open_ms = open_ms - 1
    if _dbt_starts:
        day = _trading_day((open_ms + span_off_ms) // 1000, _dbt_tz, _dbt_on)
        for _ in range(7):
            day -= timedelta(days=1)
            open_sec = _scheduled_day_open_sec(day, _dbt_tz, _dbt_starts, _dbt_on)
            if open_sec is not None:
                previous_open_ms = open_sec * 1000
                break
    if len(_ptd_cache) >= 1024:
        _ptd_cache.clear()
    _ptd_cache[key] = previous_open_ms
    return previous_open_ms


def _requested_bar_time(resampler: Resampler, modifier: str, multiplier: int,
                        current_time_ms: int, steps: int) -> int:
    """
    Bar open time on a requested timeframe's grid, stepped back by whole grid bars.

    :param resampler: Resampler of the requested timeframe
    :param modifier: Timeframe modifier ('', 'S', 'D', 'W' or 'M')
    :param multiplier: Timeframe multiplier
    :param current_time_ms: Chart bar open to resolve, in milliseconds
    :param steps: Whole grid bars to step back; zero or less resolves without stepping
    :return: Bar opening time in milliseconds
    """
    dwm = multiplier > 1 and modifier in ('D', 'W', 'M')
    daily = multiplier == 1 and modifier == 'D'
    if steps <= 0 and modifier in ('D', 'W', 'M'):
        # noinspection PyProtectedMember
        if (modifier, multiplier) == timeframe_module._process_tf(
                timeframe_module._current_period()):
            # The script's own bars are the requested grid
            return current_time_ms
    while True:
        if dwm:
            bar_time = _dwm_bar_time(resampler, modifier, multiplier, current_time_ms)
        elif daily:
            # TradingView daily bars open at the trading day's session open
            # (the previous evening for overnight markets), not at midnight
            bar_time = _d_bar_time(current_time_ms)
        else:
            # Intraday bars on the session-anchored grid; weekly and monthly bars at their
            # first trading day's session open. MEASURED (TradingView, 2026-09-25,
            # CAPITALCOM:EURUSD on 60/240/D/W/M, BTCUSD@60, GOLD@60/D, AAPL@60/D/W): a
            # week opens at its Monday trading day's open (EURUSD Sunday 17:00, AAPL Monday
            # 09:30 even on a holiday Monday), a month at its first scheduled day's open
            # (EURUSD January 2016: Thursday 2015-12-31 17:00, BTCUSD February 2026:
            # Saturday 01-31 17:00).
            bar_time = resampler.get_bar_time(current_time_ms, *_session_grid_args())
        if steps <= 0:
            return bar_time
        steps -= 1
        # The walk resolves a probe inside the previous bar rather than subtracting a
        # nominal bar length, because a month is not a fixed span: taking _in_seconds('M')
        # (30.4375 days) off a bar early in the month reaches the month BEFORE the intended
        # one. An intraday bar is left one instant before its open. A D, W or M bar is left
        # through the scheduled trading day before its first one, because the instant
        # before its open can lie in a gap that resolves forward again: AAPL's 09:29 is
        # still the Monday that opens at 09:30, GOLD's 17:59 after the 17:00 close is
        # already the next trading day, and the FX Sunday before the 17:00 open has no
        # trading day at all. MEASURED (TradingView, 2026-09-25, CAPITALCOM:AAPL, EURUSD,
        # GOLD and BTCUSD on 60 and D charts): time("D", timeframe_bars_back=1) is the
        # previous trading day of the session template, a template day without data (a
        # holiday) included -- AAPL's Tuesday 2017-05-30 steps to Memorial Day 09:30.
        if modifier in ('', 'S'):
            current_time_ms = bar_time - 1
        else:
            # ``_dwm_bar_time`` and ``_d_bar_time`` resolve a chart bar by its own last
            # instant, so the probe carries the same span back
            span_off_ms = _chart_span_off_ms() if dwm or daily else 0
            current_time_ms = _previous_trading_day_open_ms(bar_time, span_off_ms) - span_off_ms


# noinspection PyProtectedMember
def _chart_bar_open(bars_back: int | float) -> int | None:
    """
    Open of the chart bar a ``bars_back`` offset of ``time()`` or ``time_close()`` names.

    A positive offset names a bar the chart already has. A negative one names a bar that
    has not opened yet, whose expected open walks the chart's timeframe over the symbol's
    session schedule.

    :param bars_back: Chart bars back, negative for a future bar
    :return: Open time in milliseconds, or ``None`` when the chart has no bar that far back
    """
    # MEASURED (TradingView, CAPITALCOM:EURUSD@60, 2026-09-25): time(tf, bars_back) is
    # time(tf) evaluated on the chart bar ``bars_back`` bars back -- time("", 1) equals
    # time[1] on all 23246 bars, across weekends and missing bars, and is na while the
    # chart has fewer bars behind it. A future bar's open follows the schedule alone: it
    # skips the weekend but knows neither holidays nor the bars the feed leaves out.
    # time("", -1) was Sunday 17:00 on the Friday 16:00 bar, 18:00 on a Sunday 17:00 bar
    # followed by 19:00, and 17:00 on the 16:00 bar before Christmas Day.
    offset = int(bars_back)
    if offset < 0:
        run_tf = timeframe_module._current_period()
        modifier, multiplier = timeframe_module._process_tf(run_tf)
        nominal_ms = _time - offset * timeframe_module._in_seconds(run_tf) * 1000
        bounds = _session_bar_bounds(
            _time, _symbol_session_infos(getattr(syminfo, 'timezone', None) or 'UTC'),
            modifier, multiplier, nominal_ms, nominal_ms, offset)
        # A grid the session walk does not tile (nD, nW, nM) steps the nominal bar
        # length, and so does a schedule with no run in reach
        return nominal_ms if bounds is None else bounds[0]
    index = int(bar_index) - offset
    if index < 0 or offset >= _BAR_OPENS_SIZE:
        return None
    return _bar_opens[index % _BAR_OPENS_SIZE]


# noinspection PyProtectedMember
@module_function_property
def time(timeframe: str | None = None, session: str | int | None = None,
         timezone: str | None = None, bars_back: int = 0,
         timeframe_bars_back: int = 0) -> PyneInt:
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
                     An empty string or ``na`` selects the chart's timeframe; an intraday
                     timeframe on a daily, weekly or monthly chart resolves as "D".
                     Weekly and monthly bars open at the session open of their first
                     trading day. If None, returns current bar time.
    :param session: Session specification string (e.g., "0930-1600", "0000-0000:23456").
                   Format: "HHMM-HHMM" or "HHMM-HHMM:days" where days are 1234567 (1=Sun, 7=Sat);
                   hours past 24 run into the next day ("0930-2500" ends at 01:00). An
                   empty string or one not starting with a digit (e.g. "regular") selects
                   the symbol's own session; a malformed specification gives na.
                   An int value here is treated as ``bars_back`` (Pine's
                   ``time(timeframe, bars_back)`` overload).
    :param timezone: Timezone for the session (e.g., "GMT+2", "America/New_York").
                    If None, uses exchange timezone.
    :param bars_back: Bar offset on the chart's timeframe: the call is evaluated on the
                     chart bar this many bars back, and is na while the chart has no bar
                     that far back. A negative value evaluates it on the expected open of a
                     future chart bar, walked over the symbol's session schedule.
    :param timeframe_bars_back: Bar offset on the requested ``timeframe`` instead of the
                     chart's, applied on top of ``bars_back``. Positive values walk the
                     requested grid one bar at a time, so an uneven grid such as a monthly
                     one steps exactly, an intraday one skips the time between the
                     symbol's sessions and a daily one the days its session template does
                     not schedule. Negative values refer to bars that have not opened yet:
                     an intraday grid is walked forward over the symbol's sessions, a
                     daily, weekly or monthly one steps its nominal bar length.
    :return: UNIX time in milliseconds or NA if bar is outside session or invalid parameters
    """
    # Pine overload: time(timeframe, bars_back) -- a numeric second argument is a bar
    # offset (a Pine int arrives as a float; bool is not a number here)
    if isinstance(session, (int, float)) and not isinstance(session, bool):
        bars_back = session
        session = None

    if timeframe is None:
        return pine_int(_time)

    # An empty or na timeframe selects the timeframe the script runs on. MEASURED
    # (TradingView, CAPITALCOM:EURUSD@60, 2026-09-25): time(na) == time("") on every bar.
    if not timeframe:
        timeframe = timeframe_module._current_period()

    # Get resampler for the requested timeframe
    try:
        resampler = Resampler.get_resampler(timeframe)
    except ValueError:
        # Invalid timeframe
        return na_int

    modifier, multiplier = timeframe_module._process_tf(timeframe)
    if modifier in ('', 'S') and _chart_modifier() in ('D', 'W', 'M'):
        # A daily, weekly or monthly chart resolves an intraday request as "D". MEASURED
        # (TradingView, CAPITALCOM:EURUSD on D/W/M, 2026-09-25): time() and time_close()
        # of "1", "30", "60" and "240", and of "60" with a session, equal those of "D" on
        # every bar -- on a W or M chart the trading day the chart bar opens with.
        timeframe, modifier, multiplier = 'D', 'D', 1
        resampler = Resampler.get_resampler(timeframe)

    # The chart bar the call is evaluated on
    current_time_ms = _time
    intraday = modifier in ('', 'S')
    if bars_back or (timeframe_bars_back < 0 and not intraday):
        try:
            if bars_back:
                chart_bar_ms = _chart_bar_open(bars_back)
                if chart_bar_ms is None:
                    return na_int
                current_time_ms = chart_bar_ms
            if timeframe_bars_back < 0 and not intraday:
                # A daily, weekly or monthly bar that has not opened yet steps its nominal
                # length
                current_time_ms -= timeframe_bars_back * timeframe_module._in_seconds(timeframe) * 1000
        except (ValueError, AssertionError):
            return na_int
    if session is None and timeframe_bars_back and intraday:
        # An offset walks the intraday grid over the symbol's session runs, skipping the
        # time between them in both directions. MEASURED (TradingView, 2026-09-25):
        # time("60", timeframe_bars_back=3) equals time("60", "", timeframe_bars_back=3) on
        # every bar of CAPITALCOM:AAPL, BTCUSD and GOLD at 10 minutes, and on EURUSD@60 the
        # offset 1 steps from the Sunday 17:00 bar back to Friday 16:00, while
        # time("60", bars_back=1, timeframe_bars_back=-1) steps from the Friday 16:00 bar
        # forward to Sunday 17:00.
        session = ''
    bar_time = _requested_bar_time(resampler, modifier, multiplier,
                                   current_time_ms, timeframe_bars_back)

    if session is None:
        # No session specified, return the bar time
        return pine_int(bar_time)
    if not isinstance(session, str):
        # A bool slips past the int(bars_back) overload guard (bool is an int):
        # it is not a valid session specification.
        return na_int

    # Parse session string
    try:
        session_infos = _parse_session_string(session, timezone)
    except ValueError:
        # Invalid session string
        return na_int

    # Resolve the session bar this call reports (see _session_bar_bounds)
    try:
        steps = timeframe_bars_back if intraday else max(timeframe_bars_back, 0)
        bounds = _session_bar_bounds(current_time_ms, session_infos, modifier, multiplier,
                                     bar_time, bar_time, steps)
        if bounds is None:
            return na_int
        return pine_int(bounds[0])
    except TimezoneNotFoundError:
        # A missing/unresolvable timezone is a configuration error: surface it with
        # the actionable message instead of silently treating every bar as closed.
        raise
    except Exception:  # noqa
        # Error during session validation
        return na_int


# Pinned ``timenow`` for a bounded historical replay, Unix milliseconds; ``0``
# means "read the clock". A backtest that reads the real clock is NOT
# reproducible: a script gating its entries on ``time >= timenow - N days``
# measures a different bar set on every run, so its result silently changes
# from one day to the next and can never be matched against a stored reference
# again (measured on two wild-corpus strategies that scored an exact match on
# their reference's capture day and diverged the day after, with no code change
# in between). ``pyne run`` therefore pins the value to the last bar of the
# data it replays -- the instant that run's world ends -- while live runs leave
# it at ``0``. The environment variable carries the pin into the ``request.security``
# subprocesses, which import this module fresh under the ``spawn`` start method
# and read the variable in ``security_process``.
_timenow_ms: int = 0


@module_property
def timenow() -> PyneInt:
    """
    Current time in UNIX format. It is the number of milliseconds that have elapsed since 00:00:00 UTC, 1 January 1970.

    On a bounded historical replay this is the timestamp of the data's last bar,
    which keeps a backtest reproducible; a live run reads the system clock.

    :return: Current time in milliseconds
    """
    if _timenow_ms:
        return pine_int(_timenow_ms)
    # Get current UTC time and convert to milliseconds since Unix epoch
    return pine_int(int(datetime.now(UTC).timestamp() * 1000))


# ``time_tradingday`` cache. The strategy engine calls the property on every bar
# (intraday risk day-rollover), so the result is memoized per bar, keyed by the
# identity of ``_datetime`` — the function's actual input. Every bar installs a
# fresh (immutable) datetime object, so an identity hit guarantees an identical
# result; anything that swaps ``_datetime`` (including tests driving it
# directly) misses the memo and recomputes. NOT keyed by calendar date, which
# would be wrong for overnight sessions where bars before/after the session
# open on the same date belong to different trading days. The session-structure
# table is rebuilt whenever ``syminfo._opening_hours`` is replaced
# (``_set_lib_syminfo_properties`` always assigns a fresh list) or the
# timeframe the script runs on changes.
_ttd_memo_dt: datetime | None = None
_ttd_memo_result: int = 0
_ttd_session_hours: list | None = None
_ttd_session_period: str | None = None
_ttd_overnight_by_wd: dict[int, list[dt_time]] = {}
_ttd_period_delta: timedelta = timedelta()
_EPOCH_ORDINAL = 719163  # date(1970, 1, 1).toordinal()


# noinspection PyProtectedMember
@module_function_property
def time_tradingday() -> PyneInt:
    """
    The beginning time of the trading day the current bar belongs to, as a UNIX
    timestamp in milliseconds. It is 00:00 UTC of the calendar date — expressed in
    the exchange timezone — on which the bar's trading session ends.

    For symbols whose session crosses midnight (e.g. forex and futures overnight
    sessions) a bar that reaches into the session start belongs to the next calendar
    day's trading day — including the boundary bar whose window merely contains the
    open (a 17:00-18:00 bar for a 17:05 open). For symbols whose session stays within
    a single calendar day (stocks, 24/7 crypto) it is simply 00:00 UTC of the bar's
    exchange-timezone date.

    :return: UNIX time in milliseconds of 00:00 UTC on the trading day's date
    """
    global _ttd_memo_dt, _ttd_memo_result, _ttd_session_hours, _ttd_session_period, \
        _ttd_overnight_by_wd, _ttd_period_delta

    opening_hours = syminfo._opening_hours
    period = timeframe_module._current_period()
    if opening_hours is not _ttd_session_hours or period != _ttd_session_period:
        # Session structure changed — rebuild the per-weekday table of overnight
        # session opens (the only entries that can roll the trading day).
        _ttd_overnight_by_wd = _overnight_starts_by_weekday(opening_hours)
        _ttd_period_delta = timedelta(seconds=timeframe_module._in_seconds(period))
        _ttd_session_hours = opening_hours
        _ttd_session_period = period
        _ttd_memo_dt = None

    if _datetime is _ttd_memo_dt:
        return pine_int(_ttd_memo_result)

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
    return pine_int(result)


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
    day_end_ms = _tdc_day_end_ms(trade_date)

    if day_end_ms is None or day_end_ms <= bar_open_ms:
        # Overnight roll: once the calendar date's own trading day is over (or the
        # date has none), a bar whose window reaches into a session opening this
        # calendar day and crossing midnight belongs to the next trading day (same
        # rule as ``time_tradingday``). A bar opening while the day still runs stays
        # in it and is cut at its end, even when the uncapped close reaches past the
        # overnight open. MEASURED (TradingView, CAPITALCOM:BTCUSD@60, 2026-09-25): on
        # the 2026-03-08 DST change the "240" bar opening 14:00 closes at the 17:00
        # roll, not at 18:00.
        opens = _tdc_overnight_by_wd.get(dt_local.weekday())
        if opens:
            for o in opens:
                session_open = dt_local.replace(
                    hour=o.hour, minute=o.minute, second=o.second, microsecond=0)
                if bar_close_ms > session_open.timestamp() * 1000:
                    day_end_ms = _tdc_day_end_ms(trade_date + timedelta(days=1))
                    break

    if day_end_ms is None or day_end_ms <= bar_open_ms:
        # No session ends the bar's trading day, or a degenerate template — never
        # close before the open
        return bar_close_ms
    return min(bar_close_ms, day_end_ms)


def _tdc_day_end_ms(trade_date: date) -> int | None:
    """
    Scheduled end of trading day ``trade_date`` from the close table.

    :param trade_date: The trading day's date
    :return: The day's end (UNIX ms), ``None`` when no session ends that day
    """
    entry = _tdc_by_wd.get(trade_date.weekday())
    if entry is None:
        return None
    end_tod, offset = entry
    end_date = trade_date + timedelta(days=offset)
    return int(datetime(
        end_date.year, end_date.month, end_date.day,
        end_tod.hour, end_tod.minute, end_tod.second,
        tzinfo=_tdc_tz,
    ).timestamp() * 1000)


# Scheduled calendar of the chart symbol, used for D/W/M bar closes. Rebuilt
# when the session template is replaced (identity guard, like the ``_tdc`` and
# ``_dbt`` machinery). ``_dwc_cache`` memoises the probe-driven close per bar,
# because ``time_close`` is called on every bar with the same arguments.
_dwc_guard: tuple | None = None
_dwc_cal: '_BarCalendar | None' = None
_dwc_cache: dict[tuple[int, str, bool], int] = {}


# noinspection PyProtectedMember
def _dwm_close_ms(bar_open_ms: int, timeframe: str, to_next_open: bool = False) -> int:
    """
    Scheduled close of the D/W/M bar opening at ``bar_open_ms``.

    A daily, weekly or monthly bar closes at the end of the last scheduled
    trading day inside its calendar period — never at ``open + nominal span``,
    because a month has no fixed length and the last session of a week or month
    ends before the next period opens. Measured on TradingView (CAPITALCOM:GOLD):
    a monthly bar's ``time_close - time`` varies per month and a weekly one is a
    constant 4d23h, both landing on the last trading day's session end.

    With ``to_next_open`` the bar closes when the next period of its grid opens
    instead: that is where a period longer than one trading day ends when it is
    requested from an intraday chart.

    :param bar_open_ms: Bar opening time (UNIX ms)
    :param timeframe: The bar's timeframe string ('D', 'W' or 'M' modifier)
    :param to_next_open: Close at the next period's open instead of at the last
                         scheduled session end inside the period
    :return: The bar's close instant (UNIX ms)
    """
    global _dwc_guard, _dwc_cal
    oh = syminfo._opening_hours
    ss = syminfo._session_starts
    # Effective-dated session corrections (half-days, holidays) decide the last
    # trading day of a period just as much as the template does.
    corr = getattr(syminfo, 'session_corrections', None) or None
    tz_name = getattr(syminfo, 'timezone', None)
    if (_dwc_guard is None or _dwc_guard[0] is not oh or _dwc_guard[1] is not ss
            or _dwc_guard[2] is not corr or _dwc_guard[3] != tz_name):
        _dwc_cal = _BarCalendar(
            tz=_parse_timezone(tz_name) if tz_name else None,
            opening_hours=tuple(oh or ()),
            session_starts=tuple(ss or ()),
            corrections=corr,
            grid_mode=_grid_mode(syminfo.type, oh or None),
        )
        _dwc_guard = (oh, ss, corr, tz_name)
        _dwc_cache.clear()

    key = (bar_open_ms, timeframe, to_next_open)
    cached = _dwc_cache.get(key)
    if cached is not None:
        return cached
    if len(_dwc_cache) >= 1024:
        _dwc_cache.clear()
    assert _dwc_cal is not None
    if to_next_open:
        # MEASURED (TradingView, 2026-09-25, 60 charts of CAPITALCOM:EURUSD, BTCUSD,
        # GOLD and AAPL): time_close() of "2D", "3D", "W", "2W", "M", "3M" and "12M" is
        # the next period's time() on every bar -- the EURUSD week of Sunday 2025-11-16
        # 17:00 closes Sunday 11-23 17:00, a GOLD week Sunday 18:00, an AAPL week
        # Monday 09:30 even when that Monday is a holiday -- while "D" closes at its
        # own session end (Friday 17:00 / 16:00). On EURUSD D, 2D, W and M charts
        # every period closes at its last session end instead.
        close_ms = _dwm_period_end(bar_open_ms, _dwc_cal, timeframe)
    else:
        close_ms = _actual_bar_close(bar_open_ms, 0, _dwc_cal, timeframe)
    _dwc_cache[key] = close_ms
    return close_ms


# noinspection PyProtectedMember
@module_function_property
def time_close(timeframe: str | None = None, session: str | int | None = None,
               timezone: str | None = None, bars_back: int = 0,
               timeframe_bars_back: int = 0) -> PyneInt:
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
                     An empty string or ``na`` selects the chart's timeframe; an intraday
                     timeframe on a daily, weekly or monthly chart resolves as "D".
                     A daily bar closes at the end of its trading day; a longer period
                     (nD, W, M) closes at the end of its last trading day on a daily,
                     weekly or monthly chart, and when the next period opens on an
                     intraday chart. If None, returns current bar close time.
    :param session: Session specification string (e.g., "0930-1600", "0000-0000:23456").
                   Format: "HHMM-HHMM" or "HHMM-HHMM:days" where days are 1234567 (1=Sun, 7=Sat);
                   hours past 24 run into the next day ("0930-2500" ends at 01:00). An
                   empty string or one not starting with a digit (e.g. "regular") selects
                   the symbol's own session; a malformed specification gives na.
                   An int value here is treated as ``bars_back`` (Pine's
                   ``time_close(timeframe, bars_back)`` overload).
    :param timezone: Timezone for the session (e.g., "GMT+2", "America/New_York").
                    If None, uses exchange timezone.
    :param bars_back: Bar offset on the chart's timeframe: the call is evaluated on the
                     chart bar this many bars back, and is na while the chart has no bar
                     that far back. A negative value evaluates it on the expected open of a
                     future chart bar, walked over the symbol's session schedule.
    :param timeframe_bars_back: Bar offset on the requested ``timeframe`` instead of the
                     chart's, applied on top of ``bars_back``. Positive values walk the
                     requested grid one bar at a time, so an uneven grid such as a monthly
                     one steps exactly, an intraday one skips the time between the
                     symbol's sessions and a daily one the days its session template does
                     not schedule. Negative values refer to bars that have not opened yet:
                     an intraday grid is walked forward over the symbol's sessions, a
                     daily, weekly or monthly one steps its nominal bar length.
    :return: UNIX time in milliseconds of bar close or NA if bar is outside session or invalid parameters
    """
    # Pine overload: time_close(timeframe, bars_back) -- a numeric second argument is a
    # bar offset (a Pine int arrives as a float; bool is not a number here)
    if isinstance(session, (int, float)) and not isinstance(session, bool):
        bars_back = session
        session = None

    if timeframe is None:
        # Close time of the current chart bar — capped at the trading-day end,
        # because the last bar of a session may be shortened
        try:
            run_tf = timeframe_module._current_period()
            chart_mod, _chart_mult = timeframe_module._process_tf(run_tf)
            if chart_mod in ('D', 'W', 'M'):
                close_ms = _dwm_close_ms(_time, run_tf)
            else:
                close_ms = _time + timeframe_module._in_seconds(run_tf) * 1000
                close_ms = _tdc_cap_ms(_time, close_ms)
        except (ValueError, AssertionError):
            return na_int
        return pine_int(close_ms)

    # An empty or na timeframe selects the timeframe the script runs on. MEASURED
    # (TradingView, CAPITALCOM:EURUSD@60, 2026-09-25): time_close(na) == time_close("") on
    # every bar.
    if not timeframe:
        timeframe = timeframe_module._current_period()

    # Get resampler for the requested timeframe
    try:
        resampler = Resampler.get_resampler(timeframe)
    except ValueError:
        # Invalid timeframe
        return na_int

    modifier, multiplier = timeframe_module._process_tf(timeframe)
    chart_modifier = _chart_modifier()
    if modifier in ('', 'S') and chart_modifier in ('D', 'W', 'M'):
        # A daily, weekly or monthly chart resolves an intraday request as "D" (see time())
        timeframe, modifier, multiplier = 'D', 'D', 1
        resampler = Resampler.get_resampler(timeframe)

    # The chart bar the call is evaluated on
    current_time_ms = _time
    intraday = modifier in ('', 'S')
    if bars_back or (timeframe_bars_back < 0 and not intraday):
        try:
            if bars_back:
                chart_bar_ms = _chart_bar_open(bars_back)
                if chart_bar_ms is None:
                    return na_int
                current_time_ms = chart_bar_ms
            if timeframe_bars_back < 0 and not intraday:
                # A daily, weekly or monthly bar that has not opened yet steps its nominal
                # length
                current_time_ms -= timeframe_bars_back * timeframe_module._in_seconds(timeframe) * 1000
        except (ValueError, AssertionError):
            return na_int
    if session is None and timeframe_bars_back and intraday:
        # An offset walks the symbol's session runs (see time())
        session = ''
    bar_start_time = _requested_bar_time(resampler, modifier, multiplier,
                                         current_time_ms, timeframe_bars_back)

    # Calculate the bar close time: D/W/M periods close at the end of their last
    # scheduled trading day -- except on an intraday chart, where a period longer than
    # one trading day closes when the next period opens -- and intraday bars at the
    # (possibly shortened) day end.
    try:
        if modifier in ('D', 'W', 'M'):
            bar_close_time = _dwm_close_ms(
                bar_start_time, timeframe,
                chart_modifier in ('', 'S') and (modifier != 'D' or multiplier > 1))
        else:
            tf_seconds = timeframe_module._in_seconds(timeframe)
            bar_close_time = _tdc_cap_ms(bar_start_time,
                                         bar_start_time + (tf_seconds * 1000))
    except (ValueError, AssertionError):
        return na_int

    if session is None:
        # No session specified, return the bar close time
        return pine_int(bar_close_time)
    if not isinstance(session, str):
        # A bool slips past the int(bars_back) overload guard (bool is an int):
        # it is not a valid session specification.
        return na_int

    # Parse session string
    try:
        session_infos = _parse_session_string(session, timezone)
    except ValueError:
        # Invalid session string
        return na_int

    # Resolve the session bar this call reports (see _session_bar_bounds)
    try:
        steps = timeframe_bars_back if intraday else max(timeframe_bars_back, 0)
        bounds = _session_bar_bounds(current_time_ms, session_infos, modifier, multiplier,
                                     bar_start_time, bar_close_time, steps)
        if bounds is None:
            return na_int
        return pine_int(bounds[1])
    except TimezoneNotFoundError:
        # A missing/unresolvable timezone is a configuration error: surface it with
        # the actionable message instead of silently treating every bar as closed.
        raise
    except Exception:  # noqa
        # Error during session validation
        return na_int


# noinspection PyShadowingNames
@module_function_property
def weekofyear(time: int | float | None = None, timezone: str | None = None) -> PyneInt:
    """
    Week of the year

    :param time: The time to get the week of the year from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The week of the year
    """
    dt = _get_dt(time, timezone)
    return na_int if dt is None else pine_int(dt.isocalendar()[1])


# noinspection PyShadowingNames
@module_function_property
def year(time: int | float | None = None, timezone: str | None = None) -> PyneInt:
    """
    Year

    :param time: The time to get the year from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The year
    """
    dt = _get_dt(time, timezone)
    return na_int if dt is None else pine_int(dt.year)
