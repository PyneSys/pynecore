from datetime import datetime, timedelta

from ..types.session import Session

from ..core.module_property import module_property

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
# Functions
#

def _is_in_session(opening_hours, dt: datetime, tf_sec: int) -> bool:
    """
    Check if a candle overlaps any interval in the given opening_hours list.

    Pure function: takes the calendar data explicitly so live-runtime callers
    (framework live_runner, broker plugin watchdogs) do not have to depend on
    the global ``lib.syminfo`` state. The list contains ``SymInfoInterval``
    tuples ``(day, start_time, end_time)`` where ``day`` is Python weekday
    (0=Mon..6=Sun) and the times are in the symbol's source timezone.

    Slot-aware: returns True when the ``[dt, dt+tf_sec)`` candle window has a
    positive-duration overlap with a session. For point-in-time "is the market
    open right now?" decisions, use :func:`_is_point_in_session` instead.

    Overnight sessions are handled by inspecting both the current weekday
    interval and any previous-weekday interval whose end crosses midnight,
    so a post-midnight candle still matches when the calendar encodes the
    overnight wrap under its source-side start weekday.

    :param opening_hours: Iterable of ``SymInfoInterval``-like tuples.
    :param dt: Start datetime of the candle (must be in the same timezone the
               opening_hours times are expressed in).
    :param tf_sec: Timeframe in seconds.
    :return: True if the candle overlaps any session, else False.
    """
    candle_start = dt
    candle_end = dt + timedelta(seconds=tf_sec)
    weekday = dt.weekday()
    prev_weekday = (weekday - 1) % 7

    for day, ss, se in opening_hours:
        # Overnight interval owned by the previous weekday wraps into today.
        if day == prev_weekday and se < ss:
            ssdt = dt.replace(hour=ss.hour, minute=ss.minute, second=ss.second,
                              microsecond=0) - timedelta(days=1)
            sedt = ssdt.replace(hour=se.hour, minute=se.minute, second=se.second,
                                microsecond=0) + timedelta(days=1)
            if candle_end > ssdt and candle_start < sedt:
                return True
            continue
        if day != weekday:
            continue
        ssdt = dt.replace(hour=ss.hour, minute=ss.minute, second=ss.second, microsecond=0)
        sedt = dt.replace(hour=se.hour, minute=se.minute, second=se.second, microsecond=0)
        if sedt < ssdt:  # Overnight session that started today
            sedt += timedelta(days=1)
        if candle_end > ssdt and candle_start < sedt:
            return True

    return False


def _is_point_in_session(opening_hours, dt: datetime) -> bool:
    """
    Check if a single instant ``dt`` falls inside any opening_hours interval.

    Point-in-time variant of :func:`_is_in_session`: no timeframe span is
    applied, so the result reflects "is the market open at this exact
    moment?". Use this for reconnect/watchdog gates that consult wall-clock
    now; use :func:`_is_in_session` for slot-aware decisions (bar synth,
    REST recovery of a specific missing slot).

    Overnight sessions are handled by inspecting both the current weekday
    interval and any previous-weekday interval whose end crosses midnight.

    :param opening_hours: Iterable of ``SymInfoInterval``-like tuples.
    :param dt: Instant to check (must be in the same timezone the
               opening_hours times are expressed in).
    :return: True if ``dt`` is inside an open session, else False.
    """
    weekday = dt.weekday()
    prev_weekday = (weekday - 1) % 7
    for day, ss, se in opening_hours:
        # Overnight interval owned by the previous weekday wraps into today.
        if day == prev_weekday and se < ss:
            ssdt = dt.replace(hour=ss.hour, minute=ss.minute, second=ss.second,
                              microsecond=0) - timedelta(days=1)
            sedt = ssdt.replace(hour=se.hour, minute=se.minute, second=se.second,
                                microsecond=0) + timedelta(days=1)
            if ssdt <= dt < sedt:
                return True
            continue
        if day != weekday:
            continue
        ssdt = dt.replace(hour=ss.hour, minute=ss.minute, second=ss.second, microsecond=0)
        sedt = dt.replace(hour=se.hour, minute=se.minute, second=se.second, microsecond=0)
        if sedt < ssdt:  # Overnight session that started today
            sedt += timedelta(days=1)
        if ssdt <= dt < sedt:
            return True
    return False


# noinspection PyProtectedMember
def _check_session(dt: datetime, tf_sec: int) -> bool:
    """
    Check if candle overlaps with any trading session.

    :param dt: Start datetime of the candle
    :param tf_sec: Timeframe in seconds
    :return: True if candle overlaps with any session
    """
    return _is_in_session(syminfo._opening_hours, dt, tf_sec)


#
# Module properties
#

# noinspection PyProtectedMember
@module_property
def isfirstbar_regular() -> bool:
    """
    Check if the current candle is the first of the trading session.
    The result is the same whether extended session information is used or not.

    :return: True if the current candle is the first of the trading session
    """
    tf_sec = timeframe.in_seconds(syminfo.period)
    for ss in syminfo._session_starts:
        if ss.day == lib._datetime.weekday():
            ssdt = lib._datetime.replace(hour=ss.time.hour, minute=ss.time.minute, second=ss.time.second,
                                         microsecond=ss.time.microsecond)
            if lib._datetime <= ssdt < lib._datetime + timedelta(seconds=tf_sec):
                return True
    return False


# noinspection PyProtectedMember
# TODO: implement this when extended session will be supported
@module_property
def isfirstbar() -> bool:
    """
    Check if the current candle is the first of the trading session.
    If extended session information is used, only returns true on the first bar of the pre-market bars.
    NOTE: extended session information is not yet supported.

    :return: True if the current candle is the first of the trading session
    """
    # TODO: support pre market sessions
    tf_sec = timeframe.in_seconds(syminfo.period)
    for ss in syminfo._session_starts:
        if ss.day == lib._datetime.weekday():
            ssdt = lib._datetime.replace(hour=ss.time.hour, minute=ss.time.minute, second=ss.time.second,
                                         microsecond=ss.time.microsecond)
            if lib._datetime <= ssdt < lib._datetime + timedelta(seconds=tf_sec):
                return True
    return False


# noinspection PyProtectedMember
@module_property
def islastbar_regular() -> bool:
    """
    Check if the current candle is the last of the trading session.
    The result is the same whether extended session information is used or not.

    :return: True if the current candle is the last of the trading session
    """
    tf_sec = timeframe.in_seconds(syminfo.period)
    for se in syminfo._session_ends:
        if se.day == lib._datetime.weekday():
            sedt = lib._datetime.replace(hour=se.time.hour, minute=se.time.minute, second=se.time.second,
                                         microsecond=se.time.microsecond)
            # A session end at/before the bar's start denotes the day boundary
            # (00:00 = 24:00 for 24/7 markets); roll it to the next day so it
            # lands on the closing bar's end instead of the day's own start.
            if sedt <= lib._datetime:
                sedt += timedelta(days=1)
            if lib._datetime < sedt <= lib._datetime + timedelta(seconds=tf_sec):
                return True
    return False


# noinspection PyProtectedMember
@module_property
def islastbar() -> bool:
    """
    Check if the current candle is the last of the trading session.
    If extended session information is used, only returns true on the last bar of the post-market bars.
    NOTE: extended session information is not yet supported.

    :return: True if the current candle is the last of the trading session
    """
    tf_sec = timeframe.in_seconds(syminfo.period)
    for se in syminfo._session_ends:
        if se.day == lib._datetime.weekday():
            sedt = lib._datetime.replace(hour=se.time.hour, minute=se.time.minute, second=se.time.second,
                                         microsecond=se.time.microsecond)
            # A session end at/before the bar's start denotes the day boundary
            # (00:00 = 24:00 for 24/7 markets); roll it to the next day so it
            # lands on the closing bar's end instead of the day's own start.
            if sedt <= lib._datetime:
                sedt += timedelta(days=1)
            if lib._datetime < sedt <= lib._datetime + timedelta(seconds=tf_sec):
                return True
    return False


# noinspection PyProtectedMember
@module_property
def ismarket() -> bool:
    """
    Check if the current candle is within a trading session.

    :return:  True if the current candle is within a trading session
    """
    tf_sec = timeframe.in_seconds(syminfo.period)
    return _check_session(lib._datetime, tf_sec)


@module_property
def ispremarket() -> bool:
    """
    Check if the current candle is within the pre-market session.
    It is not yet implemented.

    :return: It is always False at the moment
    """
    # TODO: implement this
    return False


@module_property
def ispostmarket() -> bool:
    """
    Check if the current candle is within the post-market session.
    It is not yet implemented.

    :return: It is always False at the moment
    """
    # TODO: implement this
    return False
