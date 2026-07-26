from typing import Iterable

from datetime import datetime, timedelta

from .syminfo import SymInfoInterval

__all__ = ["is_in_session", "is_point_in_session"]


def is_in_session(opening_hours: Iterable[SymInfoInterval], dt: datetime, tf_sec: int) -> bool:
    """
    Check if a candle overlaps any interval in the given opening_hours list.

    Pure function: takes the calendar data explicitly so live-runtime callers
    (framework live_runner, broker plugin watchdogs) do not have to depend on
    the global ``lib.syminfo`` state. The list contains ``SymInfoInterval``
    tuples ``(day, start_time, end_time)`` where ``day`` is Python weekday
    (0=Mon..6=Sun) and the times are in the symbol's source timezone.

    Slot-aware: returns True when the ``[dt, dt+tf_sec)`` candle window has a
    positive-duration overlap with a session. For point-in-time "is the market
    open right now?" decisions, use :func:`is_point_in_session` instead.

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


def is_point_in_session(opening_hours: Iterable[SymInfoInterval], dt: datetime) -> bool:
    """
    Check if a single instant ``dt`` falls inside any opening_hours interval.

    Point-in-time variant of :func:`is_in_session`: no timeframe span is
    applied, so the result reflects "is the market open at this exact
    moment?". Use this for reconnect/watchdog gates that consult wall-clock
    now; use :func:`is_in_session` for slot-aware decisions (bar synth,
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
