from datetime import datetime, timedelta
from typing import Iterator

from ..types.session import Session

from ..core.module_property import module_property
from ..core.syminfo import SymInfoInterval

from . import syminfo
from . import timeframe
from .. import lib

__all__ = [
    "regular",
    "extended",
    "isfirstbar_regular",
    "isfirstbar",
    "islastbar_regular",
    "islastbar",
    "ismarket",
    "ispremarket",
    "ispostmarket"
]

#
# Constants
#

regular = Session('regular')
extended = Session('extended')


#
# Helpers
#

# MEASURED (TradingView, NASDAQ:AAPL 60-minute charts on the regular 09:30-16:00 and the
# extended 04:00-20:00 hours, 18000 bars each, 2026-09-27; CAPITALCOM:EURUSD 60-minute
# chart, 2025-01):
# - isfirstbar is the bar holding the session open (the 17:00 bar of a 17:05 open, the
#   04:00 bar of the extended chart) and islastbar the bar holding the session close
#   (the 15:30 bar of a 16:00 close, the 12:30 bar of a 13:00 early close, the 19:00 bar
#   of the extended chart). Both follow the hours the bars are cut from: on the
#   extended chart they are the extended hours and their early closes (17:00).
# - The regular hours on the extended chart are read by the bar OPEN: the extended
#   chart's bars are cut from 04:00, so isfirstbar_regular is the 10:00 bar, not the
#   09:00 bar that contains the 09:30 open, while islastbar_regular is the 15:00 bar
#   holding the close (12:00 on an early-close day). ismarket is a bar opening inside
#   the regular hours, ispremarket one opening before the day's regular open, and
#   ispostmarket one opening at or after the regular close (the 13:00 bar of an
#   early-close day). On the regular chart every bar is a market bar.


def _day_windows(hours: list[SymInfoInterval],
                 corrections: dict | None) -> Iterator[tuple[datetime, datetime]]:
    """
    The ``[open, close)`` windows of a weekly template that may hold the current bar.

    The windows of the bar's calendar day, plus the previous day's overnight ones (an
    interval ending at or before its start closes on the next day: a 00:00-00:00 day
    runs to the next midnight, a 17:00-17:00 market opens Sunday evening for Monday).
    A date listed in ``corrections`` trades on the listed hours instead of its weekday's.

    :param hours: Weekly ``SymInfoInterval`` template
    :param corrections: Single-day exceptions keyed by exchange-local date, or ``None``
    :return: Iterator of window bounds in the bar's timezone
    """
    now = lib._datetime
    today = now.date()
    weekday = today.weekday()
    prev_weekday = (weekday - 1) % 7
    today_hours: list[SymInfoInterval] | tuple[SymInfoInterval, ...] = hours
    prev_hours: list[SymInfoInterval] | tuple[SymInfoInterval, ...] = hours
    if corrections:
        today_hours = corrections.get(today, hours)
        prev_hours = corrections.get(today - timedelta(days=1), hours)
    for day, start, end in prev_hours:
        if day != prev_weekday or start < end:
            continue
        open_dt = now.replace(hour=start.hour, minute=start.minute, second=start.second,
                              microsecond=0) - timedelta(days=1)
        yield open_dt, now.replace(hour=end.hour, minute=end.minute, second=end.second,
                                   microsecond=0)
    for day, start, end in today_hours:
        if day != weekday:
            continue
        open_dt = now.replace(hour=start.hour, minute=start.minute, second=start.second,
                              microsecond=0)
        close_dt = now.replace(hour=end.hour, minute=end.minute, second=end.second,
                               microsecond=0)
        if close_dt <= open_dt:
            close_dt += timedelta(days=1)
        yield open_dt, close_dt


def _bar_span() -> timedelta:
    """The chart bar's nominal length"""
    return timedelta(seconds=timeframe.in_seconds(syminfo.period))


def _own_marks(starts: bool) -> Iterator[datetime]:
    """
    The session opens (or closes) of the hours the bars follow on the bar's day.

    The template's ``session_starts`` / ``session_ends`` name the real boundaries: a
    provider that splits a run at midnight marks neither side of the split. A date
    with a single-day exception takes the exception's own bounds instead, a closed
    day having none.

    :param starts: The opens, otherwise the closes
    :return: Iterator of the marks in the bar's timezone
    """
    now = lib._datetime
    corrections = getattr(syminfo, 'session_corrections', None)
    correction = corrections.get(now.date()) if corrections else None
    if correction is not None:
        marks = [iv.start if starts else iv.end for iv in correction]
    else:
        weekday = now.weekday()
        marks = [m.time for m in (syminfo._session_starts if starts else syminfo._session_ends)
                 if m.day == weekday]
    for mark in marks:
        yield now.replace(hour=mark.hour, minute=mark.minute, second=mark.second,
                          microsecond=mark.microsecond)


def _holds_own_open() -> bool:
    """Whether the current bar holds a session open of the hours the bars follow"""
    bar_open = lib._datetime
    bar_end = bar_open + _bar_span()
    return any(bar_open <= open_dt < bar_end for open_dt in _own_marks(True))


def _holds_own_close() -> bool:
    """Whether the current bar holds a session close of the hours the bars follow"""
    bar_open = lib._datetime
    bar_end = bar_open + _bar_span()
    for close_dt in _own_marks(False):
        # A close at or before the bar's open is the day boundary (00:00 is 24:00 on a
        # round-the-clock market): it closes the next day
        if close_dt <= bar_open:
            close_dt += timedelta(days=1)
        if bar_open < close_dt <= bar_end:
            return True
    return False


def _holds_close(hours: list[SymInfoInterval], corrections: dict | None) -> bool:
    """Whether the current bar holds a session close of the template"""
    bar_open = lib._datetime
    bar_end = bar_open + _bar_span()
    return any(bar_open < close_dt <= bar_end for _open_dt, close_dt in _day_windows(hours, corrections))


def _opens_in(hours: list[SymInfoInterval], corrections: dict | None) -> bool:
    """Whether the current bar opens inside a window of the template"""
    bar_open = lib._datetime
    return any(open_dt <= bar_open < close_dt for open_dt, close_dt in _day_windows(hours, corrections))


def _opens_before(hours: list[SymInfoInterval], corrections: dict | None) -> bool:
    """Whether the current bar opens before every window of the template"""
    bar_open = lib._datetime
    windows = list(_day_windows(hours, corrections))
    return bool(windows) and all(bar_open < open_dt for open_dt, _close_dt in windows)


def _is_extended() -> bool:
    """Whether the bars follow the extended hours"""
    return syminfo.session == extended


def _regular_corrections() -> dict | None:
    """The regular hours' single-day exceptions"""
    return getattr(syminfo, '_regular_corrections', None) or None


#
# Module properties
#

# noinspection PyProtectedMember
@module_property
def isfirstbar_regular() -> bool:
    """
    Check if the current candle is the first of the regular trading session.

    On bars of the extended hours it is the first bar opening inside the regular hours.

    :return: True if the current candle is the first of the regular trading session
    """
    if _is_extended():
        bar_open = lib._datetime
        span = _bar_span()
        return any(open_dt <= bar_open < open_dt + span and bar_open < close_dt
                   for open_dt, close_dt in _day_windows(syminfo._regular_hours,
                                                         _regular_corrections()))
    return _holds_own_open()


# noinspection PyProtectedMember
@module_property
def isfirstbar() -> bool:
    """
    Check if the current candle is the first of the trading session.

    On bars of the extended hours only the first bar of the pre-market is the first.

    :return: True if the current candle is the first of the trading session
    """
    return _holds_own_open()


# noinspection PyProtectedMember
@module_property
def islastbar_regular() -> bool:
    """
    Check if the current candle is the last of the regular trading session.

    :return: True if the current candle is the last of the regular trading session
    """
    if _is_extended():
        return _holds_close(syminfo._regular_hours, _regular_corrections())
    return _holds_own_close()


# noinspection PyProtectedMember
@module_property
def islastbar() -> bool:
    """
    Check if the current candle is the last of the trading session.

    On bars of the extended hours only the last bar of the post-market is the last.

    :return: True if the current candle is the last of the trading session
    """
    return _holds_own_close()


# noinspection PyProtectedMember
@module_property
def ismarket() -> bool:
    """
    Check if the current candle belongs to the regular trading hours.

    Every bar of a regular-hours chart does; on bars of the extended hours the bar has
    to open inside the regular hours.

    :return: True if the current candle belongs to the regular trading hours
    """
    if _is_extended():
        return _opens_in(syminfo._regular_hours, _regular_corrections())
    return True


# noinspection PyProtectedMember
@module_property
def ispremarket() -> bool:
    """
    Check if the current candle belongs to the pre-market.

    Only bars of the extended hours can: the bar opens before the day's regular open.

    :return: True if the current candle belongs to the pre-market
    """
    return _is_extended() and _opens_before(syminfo._regular_hours, _regular_corrections())


# noinspection PyProtectedMember
@module_property
def ispostmarket() -> bool:
    """
    Check if the current candle belongs to the post-market.

    Only bars of the extended hours can: the bar opens at or after the day's regular
    close (and not inside the regular hours).

    :return: True if the current candle belongs to the post-market
    """
    if not _is_extended():
        return False
    corrections = _regular_corrections()
    return (not _opens_in(syminfo._regular_hours, corrections)
            and not _opens_before(syminfo._regular_hours, corrections))
