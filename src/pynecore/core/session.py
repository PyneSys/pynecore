from typing import Iterable, Iterator, Mapping

from datetime import date, datetime, timedelta

from .syminfo import SymInfoInterval

__all__ = ["is_in_session", "is_point_in_session"]

SessionCorrections = Mapping[date, tuple[SymInfoInterval, ...]]


def _session_windows(
        opening_hours: Iterable[SymInfoInterval],
        corrections: SessionCorrections | None,
        dt: datetime,
) -> Iterator[tuple[datetime, datetime]]:
    """
    Yield the ``[start, end)`` session windows that can contain ``dt``'s calendar day.

    The weekly template is looked up for ``dt``'s weekday plus the previous
    weekday's overnight intervals (a session that crosses midnight is encoded
    under its start weekday, so a post-midnight instant still belongs to it).
    A date listed in ``corrections`` trades on its own hours instead of its
    weekday's: the correction of ``dt``'s date replaces today's intervals and
    the correction of the previous date replaces the overnight source. An
    empty correction tuple means the market did not trade that day at all.

    :param opening_hours: Weekly ``SymInfoInterval`` template.
    :param corrections: ``SymInfo.session_corrections``, or ``None``.
    :param dt: Instant whose calendar day selects the intervals; its wall
               clock is only used as the day anchor.
    :return: Iterator of ``(start, end)`` aware/naive datetimes in ``dt``'s zone.
    """
    weekday = dt.weekday()
    prev_weekday = (weekday - 1) % 7
    today_hours: Iterable[SymInfoInterval] = opening_hours
    prev_hours: Iterable[SymInfoInterval] = opening_hours
    if corrections:
        today = dt.date()
        today_hours = corrections.get(today, opening_hours)
        prev_hours = corrections.get(today - timedelta(days=1), opening_hours)
    for day, ss, se in prev_hours:
        # Overnight interval owned by the previous weekday wraps into today.
        if day != prev_weekday or not se < ss:
            continue
        ssdt = dt.replace(hour=ss.hour, minute=ss.minute, second=ss.second,
                          microsecond=0) - timedelta(days=1)
        sedt = ssdt.replace(hour=se.hour, minute=se.minute, second=se.second,
                            microsecond=0) + timedelta(days=1)
        yield ssdt, sedt
    for day, ss, se in today_hours:
        if day != weekday:
            continue
        ssdt = dt.replace(hour=ss.hour, minute=ss.minute, second=ss.second, microsecond=0)
        sedt = dt.replace(hour=se.hour, minute=se.minute, second=se.second, microsecond=0)
        if sedt < ssdt:  # Overnight session that started today
            sedt += timedelta(days=1)
        yield ssdt, sedt


def is_in_session(
        opening_hours: Iterable[SymInfoInterval],
        dt: datetime,
        tf_sec: int,
        corrections: SessionCorrections | None = None,
) -> bool:
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
    :param corrections: Date-specific overrides of the weekly template
                        (``SymInfo.session_corrections``): exchange holidays
                        and early closes, keyed by the exchange-local date.
    :return: True if the candle overlaps any session, else False.
    """
    candle_start = dt
    candle_end = dt + timedelta(seconds=tf_sec)
    for ssdt, sedt in _session_windows(opening_hours, corrections, dt):
        if candle_end > ssdt and candle_start < sedt:
            return True
    return False


def is_point_in_session(
        opening_hours: Iterable[SymInfoInterval],
        dt: datetime,
        corrections: SessionCorrections | None = None,
) -> bool:
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
    :param corrections: Date-specific overrides of the weekly template
                        (``SymInfo.session_corrections``).
    :return: True if ``dt`` is inside an open session, else False.
    """
    for ssdt, sedt in _session_windows(opening_hours, corrections, dt):
        if ssdt <= dt < sedt:
            return True
    return False
