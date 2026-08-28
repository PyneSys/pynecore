"""
Builtin library of Pyne
"""
from typing import TYPE_CHECKING, TypeAlias, Any

if TYPE_CHECKING:
    from pynecore.types.type_checker import *
    from ..types.session import SessionInfo

import sys
import math as _math

from functools import lru_cache as _lru_cache
from datetime import datetime, timedelta, time as dt_time, date, UTC, \
    MINYEAR as _MINYEAR, MAXYEAR as _MAXYEAR

from pynecore.types.source import Source

from ..core.module_property import module_property, module_function_property
from ..core.script import script, input

from ..types.na import NA
from ..types import Series, PyneInt
from ..types.plot_meta import PlotMeta
from . import syminfo  # This should be imported before core.datetime to avoid circular import!
from . import barstate, string, log, math, plot, hline, linefill, alert, dayofweek
from .plot import plot as _plot
from ..types.hline import HLine
from . import timeframe as timeframe_module
from . import session as session_module
from ._fixnan import fixnan

from pynecore.core.overload import overload
from pynecore.core.datetime import parse_datestring as _parse_datestring, parse_timezone as _parse_timezone, \
    TimezoneNotFoundError, civil_days as _civil_days, julian_civil_days as _julian_civil_days, \
    GREGORIAN_CUTOVER_DAY as _GREGORIAN_CUTOVER_DAY, GREGORIAN_CYCLE_DAYS as _GREGORIAN_CYCLE_DAYS
from ..core.resampler import (
    Resampler, ObservedDayCounter as _ObservedDayCounter,
    grid_mode as _grid_mode, overnight_opens as _overnight_opens,
    overnight_starts_by_weekday as _overnight_starts_by_weekday,
    close_table_by_weekday as _close_table_by_weekday,
    trading_day as _trading_day, trading_day_open_sec as _trading_day_open_sec,
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
    # year, then carry the day through timedelta so the wall clock overflows
    # before the timezone conversion.
    y = int(year)
    m = int(month)
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    d = int(day)
    # The clock components roll over into days too, so they are carried into the
    # day count before every calendar decision below. The wall clock is not moved
    # by the carry (the timedelta at the end shifts wall time, not absolute time),
    # but a rollover that leaves datetime's range -- hour 24 on the last
    # representable day -- now folds like any other out-of-range date instead of
    # raising OverflowError.
    clock_seconds = int(hour) * 3600 + int(minute) * 60 + int(second)
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
    return int(dt.timestamp() * 1000) + cycles * _GREGORIAN_CYCLE_MS


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
    plot1 = kwargs.get('plot1') if 'plot1' in kwargs else kwargs.get('hline1')
    plot2 = kwargs.get('plot2') if 'plot2' in kwargs else kwargs.get('hline2')
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
        if isinstance(plot1, HLine):
            meta = PlotMeta(id=key, kind='fill', title=title, color=color, editable=editable,
                            show_last=show_last, fillgaps=fillgaps, display=display,
                            hline1=plot1.id if plot1 is not None else None,
                            hline2=plot2.id if plot2 is not None else None)
        else:
            meta = PlotMeta(id=key, kind='fill', title=title, color=color, editable=editable,
                            show_last=show_last, fillgaps=fillgaps, display=display,
                            plot1=plot1.id if plot1 is not None else None,
                            plot2=plot2.id if plot2 is not None else None)
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
        return replacement
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
def dayofmonth(time: int | None = None, timezone: str | None = None) -> int:
    """
    Day of the month

    :param time: The time to get the day of the month from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The day of the month
    """
    return _get_dt(time, timezone).day


# noinspection PyShadowingNames
@module_function_property
def hour(time: int | None = None, timezone: str | None = None) -> int:
    """
    Hour of the day

    :param time: The time to get the hour of the day from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The hour of the day
    """
    return _get_dt(time, timezone).hour


# noinspection PyShadowingNames
@module_function_property
def minute(time: int | None = None, timezone: str | None = None) -> int:
    """
    Minute of the hour

    :param time: The time to get the minute of the hour from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The minute of the hour
    """
    return _get_dt(time, timezone).minute


# noinspection PyShadowingNames
@module_function_property
def month(time: int | None = None, timezone: str | None = None) -> int:
    """
    Month of the year

    :param time: The time to get the month of the year from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The month of the year
    """
    return _get_dt(time, timezone).month


# noinspection PyShadowingNames
@module_function_property
def second(time: int | None = None, timezone: str | None = None) -> int:
    """
    Second of the minute

    :param time: The time to get the second of the minute from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The second of the minute
    """
    return _get_dt(time, timezone).second


### Session parsing and validation helpers ###

_MINUTES_PER_DAY = 24 * 60


def _parse_session_string(session: str, timezone: str | None = None) -> tuple['SessionInfo', ...]:
    """
    Parse a session string into one SessionInfo per time range.

    A session may list several comma-separated ranges ("0400-0700,0900-1300"); the
    optional ``:days`` suffix applies to the whole specification, so every range
    shares the same day set and timezone.

    :param session: Session string (e.g., "0930-1600", "0930-1600:23456",
                    "0400-0700,0900-1300:23456", "0000-0000:1234567")
    :param timezone: Timezone string, defaults to exchange timezone if None
    :return: One SessionInfo per time range, in the order they were written
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
    return _parse_session_string_cached(session, timezone)


@_lru_cache(maxsize=128)
def _parse_session_string_cached(session: str, timezone: str) -> tuple['SessionInfo', ...]:
    """
    Parse a fully resolved session specification, memoized on its arguments.

    A script's session strings are constants, so every bar re-derives the same
    ranges: ``time(tf, session)`` alone reached this parser ~50 times per bar on a
    multi-context strategy. The result is a tuple of frozen ``SessionInfo`` and
    callers only read it, so one instance can be shared by every caller.

    :param session: Session string, never empty
    :param timezone: Timezone string, already resolved to a concrete zone
    :return: One SessionInfo per time range, in the order they were written
    :raises ValueError: If session string is invalid
    """
    from ..types.session import SessionInfo

    if not session or session.strip() == "":
        raise ValueError("Session string cannot be empty")

    # Split session and days if present
    if ':' in session:
        time_part, days_part = session.split(':', 1)
    else:
        time_part = session
        # Default days in Pine Script v5 is all days (1234567)
        days_part = "1234567"

    # Parse time part (one or more comma-separated HHMM-HHMM ranges)
    ranges: list[tuple[dt_time, dt_time]] = []
    for range_part in time_part.split(','):
        if '-' not in range_part:
            raise ValueError(f"Invalid session format: {session}. Expected HHMM-HHMM format")

        start_str, end_str = range_part.split('-', 1)

        if len(start_str) != 4 or len(end_str) != 4:
            raise ValueError(f"Invalid time format in session: {session}. Expected HHMM-HHMM")

        try:
            start_hour = int(start_str[:2])
            start_minute = int(start_str[2:])
            end_hour = int(end_str[:2])
            end_minute = int(end_str[2:])

            # Validate time values -- hour 24 is a legal "end of day" hour ("0000-2400"
            # is the usual all-day session), hour 25 and above is rejected.
            if not (0 <= start_hour <= 24 and 0 <= start_minute <= 59):
                raise ValueError(f"Invalid start time: {start_str}")
            if not (0 <= end_hour <= 24 and 0 <= end_minute <= 59):
                raise ValueError(f"Invalid end time: {end_str}")

            # The start is wrapped into the day, the end is not: measured on
            # TradingView, "2430-1200" runs 00:30-12:00 while "0930-2430" runs to 00:30
            # of the NEXT day. An end at or before the start is the next day's, and a
            # range covering 24 hours or more is the whole day -- encoded as equal
            # endpoints, which :func:`_is_bar_in_session` reads as the all-day session.
            start_minutes = (start_hour * 60 + start_minute) % _MINUTES_PER_DAY
            end_minutes = end_hour * 60 + end_minute
            if end_minutes <= start_minutes:
                end_minutes += _MINUTES_PER_DAY
            if end_minutes - start_minutes >= _MINUTES_PER_DAY:
                end_minutes = start_minutes
            else:
                end_minutes %= _MINUTES_PER_DAY

            ranges.append((dt_time(*divmod(start_minutes, 60)),
                           dt_time(*divmod(end_minutes, 60))))

        except ValueError as e:
            raise ValueError(f"Invalid time values in session: {session}") from e

    # Parse days (1=Sunday, 2=Monday, ..., 7=Saturday)
    try:
        day_nums = set()
        for day_char in days_part:
            day_num = int(day_char)
            if not 1 <= day_num <= 7:
                raise ValueError(f"Invalid day: {day_num}")
            day_nums.add(day_num)
    except ValueError as e:
        raise ValueError(f"Invalid days specification: {days_part}") from e
    days = frozenset(day_nums)

    return tuple(
        SessionInfo(start_time=start_time, end_time=end_time, days=days, timezone=timezone)
        for start_time, end_time in ranges
    )


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


@_lru_cache(maxsize=512)
def _session_occurrences_opening_on(day: date,
                                    session_infos: 'tuple[SessionInfo, ...]'
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

    KNOWN DIVERGENCE: on a change night TradingView's INTRADAY session mask
    contradicts its OWN daily bounds whenever an endpoint lands on the changing
    hour, and it does so in three mutually inconsistent ways. MEASURED (same
    symbol, thirteen session shapes at 30 minutes over both 2025 changes and the
    2026 spring one):

    - A run closing at 02:00 keeps the pre-change offset on both endpoints in the
      daily bounds and the POST-change one intraday, so the whole occurrence sits
      an hour earlier in spring and an hour later in the fall: daily
      2025-03-08 17:00 EST -> 03:00 EDT against intraday 16:00 EST -> 01:00 EST.
    - An endpoint inside the spring gap is resolved with the pre-change offset
      daily and the post-change one intraday, which shortens "1700-0300" and
      "0200-1000" by an hour and leaves "0130-0230" with NO intraday occurrence
      at all on the gap date.
    - On the EVENING BEFORE the fall change, where nothing shifts at all,
      TradingView appends an extra hour to the intraday close of every run with
      an 02:00 endpoint, returning na from ``time()`` and a value from
      ``time_close()`` on the very same bar (2025-11-01 06:00, and 14:00 for
      "0200-1000").

    The last one no single occurrence can reproduce and the other two contradict
    the daily request, so the measured daily bounds win. The cost, of 28396 bars:
    14 for "1700-0200", "1000-0200" and "2200-0200" each, 6 for "0200-1000", 4
    for "0130-0230" and "1700-0300", and 0 for every session -- "1700-0100",
    "1900-0400", "0100-0500" among them -- that keeps clear of the changing hour.

    :param day: Opening date of the runs, in the session's timezone
    :param session_infos: Session ranges of one session specification
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
                        session_infos: 'tuple[SessionInfo, ...]') -> tuple[int, int] | None:
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
    # dates have to be offered to the containment test.
    for day_offset in (-1, 0):
        day = (bar_dt + timedelta(days=day_offset)).date()
        for start_ms, end_ms in _session_occurrences_opening_on(day, session_infos):
            if start_ms <= bar_time_ms < end_ms:
                if best is None or start_ms > best[0]:
                    best = (start_ms, end_ms)
    return best


@_lru_cache(maxsize=256)
def _previous_session_occurrence(before_ms: int,
                                 session_infos: 'tuple[SessionInfo, ...]'
                                 ) -> tuple[int, int] | None:
    """
    The latest session occurrence opening strictly before an instant.

    Walking ``timeframe_bars_back`` over session bars leaves the current
    occurrence, so the series has to be continued backwards one run at a time.

    :param before_ms: Instant the occurrence has to open before, in milliseconds
    :param session_infos: Session ranges of one session specification
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
                                                          session_infos):
            if occurrence[0] < before_ms and (best is None or occurrence[0] > best[0]):
                best = occurrence
        if best is not None:
            return best
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
    from datetime import timedelta

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
      close for that day's occurrence -- not the calendar day's.
    - An intraday request tiles the occurrence: buckets are counted from the
      session open, so an off-grid session shifts the whole series, and the
      final bucket is truncated at the session close.
    - Weekly and monthly requests are never na: their session bars run from the
      period's first session open to the NEXT period's first session open
      (weekly close landed on the following Monday's open, not on Friday's
      close), so consecutive bars tile the whole timeline.

    Multi-period daily/weekly/monthly requests ("3D", "2W") keep the plain grid:
    their session anchoring is unmeasured.

    :param chart_time_ms: Chart bar open the call is evaluated on, in milliseconds
    :param session_infos: Session ranges of one session specification
    :param modifier: Requested timeframe modifier ('', 'S', 'D', 'W' or 'M')
    :param multiplier: Requested timeframe multiplier
    :param bar_start_ms: Plain grid bar open of the requested timeframe
    :param bar_close_ms: Plain grid bar close of the requested timeframe
    :param steps: Session bars to walk back, the positive ``timeframe_bars_back``
    :return: ``(open_ms, close_ms)``, or ``None`` when the bar is out of session
    """
    from datetime import datetime

    if modifier in ('W', 'M'):
        if multiplier == 1:
            exchange_tz = _parse_timezone(getattr(syminfo, 'timezone', None) or 'UTC')
            # ``bar_start_ms`` has already walked ``steps`` periods back, so the
            # period the CHART bar sits in is recovered before the walk below and
            # the offset is re-applied to its result.
            period_date = _period_start_date(
                datetime.fromtimestamp(bar_start_ms / 1000, exchange_tz).date(),
                modifier, steps)
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
                return open_ms, close_ms
        return bar_start_ms, bar_close_ms

    if modifier not in ('', 'S', 'D') or (modifier == 'D' and multiplier != 1):
        return bar_start_ms, bar_close_ms

    occurrence = _session_occurrence(chart_time_ms, session_infos)
    if occurrence is None and steps <= 0:
        return None

    if modifier == 'D':
        if occurrence is None:
            # MEASURED: out of session a daily request is NOT na once an offset
            # is given, it reports the occurrence the offset lands on counted
            # from the one the chart bar is heading into -- which is one step
            # further than the last one that ran.
            occurrence = _previous_session_occurrence(chart_time_ms + 1, session_infos)
            steps -= 1
        while occurrence is not None and steps > 0:
            occurrence = _previous_session_occurrence(occurrence[0], session_infos)
            steps -= 1
        return occurrence

    # An intraday request tiles the occurrence itself instead of filtering the
    # plain grid: the first bucket opens at the session open even when that is
    # off the grid, and the last one is cut short at the session close. MEASURED
    # (BINANCE:BTCUSDT@30, requested "60"): "0930-1600" ran 09:30-10:30,
    # 10:30-11:30, ... and closed 15:30-16:00, while "0900-1130" closed its
    # 11:00 bucket at 11:30, not at 12:00.
    step_ms = (multiplier if modifier == 'S' else multiplier * 60) * 1000
    in_session = occurrence is not None
    if occurrence is not None:
        index = (chart_time_ms - occurrence[0]) // step_ms - steps
    else:
        # MEASURED (both "0930-1600" with seven buckets and "0900-1130" with
        # three, requested "60", offsets 1..8): out of session the walk starts
        # from the LAST bucket of the run that has already closed, and whichever
        # run it lands in is reported by its OPENING bucket -- the within-run
        # position of a bar that is not in the run does not carry over.
        occurrence = _previous_session_occurrence(chart_time_ms + 1, session_infos)
        if occurrence is None:
            return None
        index = _session_bucket_count(occurrence, step_ms) - 1 - steps
    while index < 0:
        previous = _previous_session_occurrence(occurrence[0], session_infos)
        if previous is None:
            index = 0
            break
        occurrence = previous
        index += _session_bucket_count(occurrence, step_ms)
    if not in_session:
        index = 0
    start_ms = occurrence[0] + index * step_ms
    return start_ms, min(start_ms + step_ms, occurrence[1])


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


def _requested_bar_time(resampler: Resampler, timeframe: str, modifier: str, multiplier: int,
                        current_time_ms: int, steps: int) -> int:
    """
    Bar open time on a requested timeframe's grid, stepped back by whole grid bars.

    :param resampler: Resampler of the requested timeframe
    :param timeframe: The requested timeframe
    :param modifier: Timeframe modifier ('', 'S', 'D', 'W' or 'M')
    :param multiplier: Timeframe multiplier
    :param current_time_ms: Chart bar open to resolve, in milliseconds
    :param steps: Whole grid bars to step back; zero or less resolves without stepping
    :return: Bar opening time in milliseconds
    """
    dwm = multiplier > 1 and modifier in ('D', 'W', 'M')
    daily = multiplier == 1 and modifier == 'D'
    if (dwm or daily) and steps <= 0:
        # noinspection PyProtectedMember
        if (modifier, multiplier) == timeframe_module._process_tf(str(syminfo.period)):
            # The chart's own bars are the requested grid
            return current_time_ms
    while True:
        if dwm:
            bar_time = _dwm_bar_time(resampler, modifier, multiplier, current_time_ms)
        elif daily:
            # TradingView daily bars open at the trading day's session open
            # (the previous evening for overnight markets), not at midnight
            bar_time = _d_bar_time(current_time_ms)
        else:
            bar_time = resampler.get_bar_time(current_time_ms, *_intraday_session_args(timeframe))
        if steps <= 0:
            return bar_time
        steps -= 1
        # The walk lands one instant before a resolved bar open rather than subtracting a
        # nominal bar length, because a month is not a fixed span: taking in_seconds('M')
        # (30.4375 days) off a bar early in the month reaches the month BEFORE the intended
        # one. ``_dwm_bar_time`` and ``_d_bar_time`` resolve a chart bar by its own last
        # instant, so the probe carries the same span back.
        current_time_ms = bar_time - 1 - (_chart_span_off_ms() if dwm or daily else 0)


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
    :param timeframe_bars_back: Bar offset on the requested ``timeframe`` instead of the
                     chart's, applied on top of ``bars_back``. Positive values walk the
                     requested grid one bar at a time, so an uneven grid such as a monthly
                     one steps exactly; negative values refer to bars that do not exist yet
                     and use the nominal bar length.
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
    # noinspection PyProtectedMember
    modifier, multiplier = timeframe_module._process_tf(timeframe)
    if bars_back or timeframe_bars_back < 0:
        try:
            if bars_back:
                current_time_ms -= bars_back * timeframe_module.in_seconds(str(syminfo.period)) * 1000
            if timeframe_bars_back < 0:
                # A future bar has no grid to walk yet, so its nominal length is used
                current_time_ms -= timeframe_bars_back * timeframe_module.in_seconds(timeframe) * 1000
        except (ValueError, AssertionError):
            return NA(int)
    bar_time = _requested_bar_time(resampler, timeframe, modifier, multiplier,
                                   current_time_ms, timeframe_bars_back)

    if session is None:
        # No session specified, return the bar time
        return bar_time
    if not isinstance(session, str):
        # A bool slips past the int(bars_back) overload guard (bool is an int):
        # it is not a valid session specification.
        return NA(int)

    # Parse session string
    try:
        session_infos = _parse_session_string(session, timezone)
    except ValueError:
        # Invalid session string
        return NA(int)

    # Resolve the session bar this call reports (see _session_bar_bounds)
    try:
        bounds = _session_bar_bounds(current_time_ms, session_infos, modifier, multiplier,
                                     bar_time, bar_time, max(timeframe_bars_back, 0))
        if bounds is None:
            return NA(int)
        return bounds[0]
    except TimezoneNotFoundError:
        # A missing/unresolvable timezone is a configuration error: surface it with
        # the actionable message instead of silently treating every bar as closed.
        raise
    except Exception:  # noqa
        # Error during session validation
        return NA(int)


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
def timenow():
    """
    Current time in UNIX format. It is the number of milliseconds that have elapsed since 00:00:00 UTC, 1 January 1970.

    On a bounded historical replay this is the timestamp of the data's last bar,
    which keeps a backtest reproducible; a live run reads the system clock.

    :return: Current time in milliseconds
    """
    if _timenow_ms:
        return _timenow_ms
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
    :param timeframe_bars_back: Bar offset on the requested ``timeframe`` instead of the
                     chart's, applied on top of ``bars_back``. Positive values walk the
                     requested grid one bar at a time, so an uneven grid such as a monthly
                     one steps exactly; negative values refer to bars that do not exist yet
                     and use the nominal bar length.
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
    # noinspection PyProtectedMember
    modifier, multiplier = timeframe_module._process_tf(timeframe)
    if bars_back or timeframe_bars_back < 0:
        try:
            if bars_back:
                current_time_ms -= bars_back * timeframe_module.in_seconds(str(syminfo.period)) * 1000
            if timeframe_bars_back < 0:
                # A future bar has no grid to walk yet, so its nominal length is used
                current_time_ms -= timeframe_bars_back * timeframe_module.in_seconds(timeframe) * 1000
        except (ValueError, AssertionError):
            return NA(int)
    bar_start_time = _requested_bar_time(resampler, timeframe, modifier, multiplier,
                                         current_time_ms, timeframe_bars_back)

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
        session_infos = _parse_session_string(session, timezone)
    except ValueError:
        # Invalid session string
        return NA(int)

    # Resolve the session bar this call reports (see _session_bar_bounds)
    try:
        bounds = _session_bar_bounds(current_time_ms, session_infos, modifier, multiplier,
                                     bar_start_time, bar_close_time,
                                     max(timeframe_bars_back, 0))
        if bounds is None:
            return NA(int)
        return bounds[1]
    except TimezoneNotFoundError:
        # A missing/unresolvable timezone is a configuration error: surface it with
        # the actionable message instead of silently treating every bar as closed.
        raise
    except Exception:  # noqa
        # Error during session validation
        return NA(int)


# noinspection PyShadowingNames
@module_function_property
def weekofyear(time: int | None = None, timezone: str | None = None) -> int:
    """
    Week of the year

    :param time: The time to get the week of the year from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The week of the year
    """
    return _get_dt(time, timezone).isocalendar()[1]


# noinspection PyShadowingNames
@module_function_property
def year(time: int | None = None, timezone: str | None = None) -> int:
    """
    Year

    :param time: The time to get the year from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The year
    """
    return _get_dt(time, timezone).year
