"""
Date-specific session corrections in the live session checks.

``SymInfo.session_corrections`` overrides one calendar date's trading hours
(an exchange holiday, an early close). The live-runtime checks in
:mod:`pynecore.core.session` decide bar synthesis, feed-stale alarms and
reconnect gating from the calendar, so a correction must close the market
for them exactly as it does for the backtest bar calendar.
"""
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from pynecore.core.session import is_in_session, is_point_in_session
from pynecore.core.syminfo import SymInfoInterval

_TZ = ZoneInfo("UTC")

# 24/7 template rendered as single-day segments (the broker plugins' shape).
_ALL_WEEK = [SymInfoInterval(day=d, start=time(0, 0), end=time(23, 59, 59)) for d in range(7)]
# Friday 2026-09-25 22:00 .. Saturday 2026-09-26 04:00 closed.
_CORRECTIONS = {
    date(2026, 9, 25): (SymInfoInterval(day=4, start=time(0, 0), end=time(22, 0)),),
    date(2026, 9, 26): (SymInfoInterval(day=5, start=time(4, 0), end=time(23, 59, 59)),),
}


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=_TZ)


def __test_point_check_honours_closed_window__():
    """The holiday window is closed, the rest of both days stays open."""
    assert is_point_in_session(_ALL_WEEK, _at(25, 21, 59), _CORRECTIONS)
    assert not is_point_in_session(_ALL_WEEK, _at(25, 22, 0), _CORRECTIONS)
    assert not is_point_in_session(_ALL_WEEK, _at(26, 3, 59), _CORRECTIONS)
    assert is_point_in_session(_ALL_WEEK, _at(26, 4, 0), _CORRECTIONS)
    # Without corrections the same instants are open: the template is 24/7.
    assert is_point_in_session(_ALL_WEEK, _at(25, 22, 0))
    assert is_point_in_session(_ALL_WEEK, _at(26, 3, 59), {})


def __test_slot_check_honours_closed_window__():
    """A one-minute slot inside the window is closed; a straddling slot is open."""
    assert not is_in_session(_ALL_WEEK, _at(25, 22, 0), 60, _CORRECTIONS)
    assert not is_in_session(_ALL_WEEK, _at(26, 0, 0), 60, _CORRECTIONS)
    assert is_in_session(_ALL_WEEK, _at(25, 21, 59), 60, _CORRECTIONS)
    assert is_in_session(_ALL_WEEK, _at(25, 21, 59), 120, _CORRECTIONS)
    assert is_in_session(_ALL_WEEK, _at(26, 4, 0), 60, _CORRECTIONS)
    assert is_in_session(_ALL_WEEK, _at(25, 22, 0), 60)


def __test_empty_correction_closes_the_whole_day__():
    """An empty tuple means no trading on that date; other dates are untouched."""
    corrections = {date(2026, 9, 25): ()}
    assert not is_point_in_session(_ALL_WEEK, _at(25, 12, 0), corrections)
    assert not is_in_session(_ALL_WEEK, _at(25, 23, 59), 60, corrections)
    assert is_point_in_session(_ALL_WEEK, _at(24, 23, 59), corrections)
    assert is_point_in_session(_ALL_WEEK, _at(26, 0, 0), corrections)


def __test_overnight_correction_of_previous_date_applies_after_midnight__():
    """A corrected overnight session is read from the date it STARTED on."""
    # Weekly template: Mon-Fri 20:00 -> 02:00 overnight.
    template = [SymInfoInterval(day=d, start=time(20, 0), end=time(2, 0)) for d in range(5)]
    # Thursday 2026-09-24 closes early at 23:00 instead of running to Friday 02:00.
    corrections = {date(2026, 9, 24): (SymInfoInterval(day=3, start=time(20, 0), end=time(23, 0)),)}
    assert is_point_in_session(template, _at(25, 1, 0))
    assert not is_point_in_session(template, _at(25, 1, 0), corrections)
    assert not is_in_session(template, _at(25, 0, 0), 3600, corrections)
    assert is_in_session(template, _at(24, 22, 30), 3600, corrections)
    # A correction that keeps the overnight leg still wraps into the next date.
    kept = {date(2026, 9, 24): (SymInfoInterval(day=3, start=time(21, 0), end=time(1, 0)),)}
    assert is_point_in_session(template, _at(25, 0, 30), kept)
    assert not is_point_in_session(template, _at(25, 1, 30), kept)
