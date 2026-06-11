"""
Multi-period (nD/nW/nM) scheduled grid (issue #65).

TradingView counts scheduled trading days per exchange calendar with a
year-reset counter and stamps each period with its first scheduled day's
session open (see the ``core.resampler`` module docs). These tests cover the
grid helpers, the mode-aware ``Resampler.get_bar_time`` multi-period branch,
the observed-day counter, chart-bar containment, and the security-side
multi-period confirmation pointer.
"""
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from pynecore.core.resampler import (
    Resampler, ObservedDayCounter, grid_mode, overnight_opens, trading_day,
    trading_day_open_sec, weekday_ordinal, weekday_from_ordinal, first_monday,
)
from pynecore.core.security import SecurityState, _get_confirmed_time
from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


def _ms(tz: ZoneInfo, y: int, mo: int, d: int, h: int = 0, mi: int = 0, s: int = 0) -> int:
    return int(datetime(y, mo, d, h, mi, s, tzinfo=tz).timestamp() * 1000)


def _sec(tz: ZoneInfo, y: int, mo: int, d: int, h: int = 0, mi: int = 0, s: int = 0) -> int:
    return int(datetime(y, mo, d, h, mi, s, tzinfo=tz).timestamp())


def _cme_template() -> tuple[list[SymInfoInterval], list[SymInfoSession]]:
    """CME-style overnight session: opens Sun-Thu 17:00, closes next day 17:00."""
    t17 = time(17, 0)
    days = [6, 0, 1, 2, 3]
    hours = [SymInfoInterval(day=d, start=t17, end=t17) for d in days]
    starts = [SymInfoSession(day=d, time=t17) for d in days]
    return hours, starts


def __test_weekday_ordinal_roundtrip__():
    """weekday_from_ordinal inverts weekday_ordinal across leap and common years."""
    for year in (2020, 2022, 2024, 2025):
        expected = 0
        d = date(year, 1, 1)
        while d.year == year:
            if d.weekday() < 5:
                assert weekday_ordinal(d) == expected
                assert weekday_from_ordinal(year, expected) == d
                expected += 1
            d = date.fromordinal(d.toordinal() + 1)


def __test_overnight_opens_rules__():
    """Overnight = end at or before start, except an open at exactly midnight."""
    hours, starts = _cme_template()
    on = overnight_opens(hours, starts)
    assert on == {6: time(17, 0), 0: time(17, 0), 1: time(17, 0),
                  2: time(17, 0), 3: time(17, 0)}
    assert 4 not in on  # no Friday-evening session

    # A full-day session opening at midnight does not roll the trading day
    midnight = [SymInfoInterval(day=0, start=time(0, 0), end=time(0, 0))]
    assert overnight_opens(midnight, None) == {}

    # Split representation (day part + evening part) detects the evening part
    split = [
        SymInfoInterval(day=0, start=time(0, 0), end=time(16, 59, 50)),
        SymInfoInterval(day=0, start=time(17, 5), end=time(0, 0)),
    ]
    assert overnight_opens(split, None) == {0: time(17, 5)}

    # Fallback from session_starts alone: evening opens (>= 12:00) roll
    assert overnight_opens(None, [SymInfoSession(day=6, time=time(17, 0))]) \
        == {6: time(17, 0)}
    assert overnight_opens(None, [SymInfoSession(day=0, time=time(9, 30))]) == {}


def __test_trading_day_roll__():
    """A bar at or after its weekday's overnight open belongs to the next day."""
    on = overnight_opens(*_cme_template())
    assert trading_day(_sec(_NY, 2025, 6, 8, 17, 0), _NY, on) == date(2025, 6, 9)
    assert trading_day(_sec(_NY, 2025, 6, 8, 16, 59), _NY, on) == date(2025, 6, 8)
    assert trading_day(_sec(_NY, 2025, 6, 9, 10, 0), _NY, on) == date(2025, 6, 9)
    assert trading_day(_sec(_NY, 2025, 6, 9, 17, 0), _NY, on) == date(2025, 6, 10)
    # Friday evening has no session -> no roll
    assert trading_day(_sec(_NY, 2025, 6, 13, 18, 0), _NY, on) == date(2025, 6, 13)


def __test_trading_day_open_sec__():
    """Trading-day opens are scheduled instants — synthetic on dataless days."""
    hours, starts = _cme_template()
    on = overnight_opens(hours, starts)
    # Monday's trading day opens Sunday evening
    assert trading_day_open_sec(date(2025, 6, 9), _NY, starts, on) \
        == _sec(_NY, 2025, 6, 8, 17, 0)
    # Friday's opens Thursday evening
    assert trading_day_open_sec(date(2025, 6, 13), _NY, starts, on) \
        == _sec(_NY, 2025, 6, 12, 17, 0)
    # A holiday Monday (MLK 2025) still has its scheduled open
    assert trading_day_open_sec(date(2025, 1, 20), _NY, starts, on) \
        == _sec(_NY, 2025, 1, 19, 17, 0)
    # No template entry -> local midnight fallback
    assert trading_day_open_sec(date(2025, 6, 14), _NY, starts, on) \
        == _sec(_NY, 2025, 6, 14)
    assert trading_day_open_sec(date(2025, 6, 9), _UTC, None, {}) \
        == _sec(_UTC, 2025, 6, 9)


def __test_grid_mode_classification__():
    """7-day sessions and crypto types are 'calendar'; forex 'weekday'; rest 'observed'."""
    hours, _ = _cme_template()
    assert grid_mode('futures', hours) == 'observed'
    assert grid_mode('stock', hours) == 'observed'
    assert grid_mode('forex', hours) == 'weekday'
    assert grid_mode('crypto', hours) == 'calendar'
    full_week = [SymInfoInterval(day=d, start=time(0, 0), end=time(0, 0))
                 for d in range(7)]
    assert grid_mode('futures', full_week) == 'calendar'


def __test_get_bar_time_weekday_5d__():
    """Weekday 5D grid: year-reset slots stamped at (synthetic) session opens."""
    hours, starts = _cme_template()
    r = Resampler.get_resampler('5D')

    def bar(y, mo, d, h, mi=0):
        return r.get_bar_time(_ms(_NY, y, mo, d, h, mi), _NY, starts, hours, 'weekday')

    # Jan 1 2025 (Wednesday) is weekday ordinal 0 -> the year's first slot,
    # stamped at Tuesday Dec 31 17:00 even though markets were closed
    first_slot = _ms(_NY, 2024, 12, 31, 17, 0)
    assert bar(2025, 1, 2, 12) == first_slot
    assert bar(2025, 1, 3, 18) == first_slot  # Friday evening stays in slot
    # Ordinal 5 (Wed Jan 8) starts the second slot
    assert bar(2025, 1, 8, 12) == _ms(_NY, 2025, 1, 7, 17, 0)
    # Year reset: Dec 31 2025 is ordinal 260 (a slot start), Jan 1 2026 ordinal 0
    assert bar(2025, 12, 31, 12) == _ms(_NY, 2025, 12, 30, 17, 0)
    assert bar(2026, 1, 1, 12) == _ms(_NY, 2025, 12, 31, 17, 0)


def __test_get_bar_time_calendar_5d__():
    """Calendar 5D grid: pure 5-calendar-day groups from Jan 1, midnight stamps."""
    r = Resampler.get_resampler('5D')

    def bar(y, mo, d, h=0):
        return r.get_bar_time(_ms(_UTC, y, mo, d, h), _UTC, None, None, 'calendar')

    assert bar(2025, 1, 1) == _ms(_UTC, 2025, 1, 1)
    assert bar(2025, 1, 5, 23) == _ms(_UTC, 2025, 1, 1)
    assert bar(2025, 1, 6) == _ms(_UTC, 2025, 1, 6)
    # Year reset: Dec 31 (ordinal 364) belongs to the Dec 27 slot, Jan 1 restarts
    assert bar(2025, 12, 31) == _ms(_UTC, 2025, 12, 27)
    assert bar(2026, 1, 1) == _ms(_UTC, 2026, 1, 1)


def __test_get_bar_time_week_month__():
    """nW counts weeks from the year's first Monday; nM groups in-year months."""
    r3w = Resampler.get_resampler('3W')
    # A week belongs to its Monday's year: Jan 1-3 2025 sit in the week of
    # Mon Dec 30 2024, anchored to 2024's weekly grid (slot Dec 23 2024)
    assert r3w.get_bar_time(_ms(_UTC, 2025, 1, 2), _UTC, None, None, 'calendar') \
        == _ms(_UTC, 2024, 12, 23)
    # First Monday of 2025 is Jan 6 -> week 0 -> slot Jan 6
    assert first_monday(2025) == date(2025, 1, 6)
    assert r3w.get_bar_time(_ms(_UTC, 2025, 1, 8), _UTC, None, None, 'calendar') \
        == _ms(_UTC, 2025, 1, 6)

    r3m = Resampler.get_resampler('3M')
    # Q2 2028 starts Sat Apr 1: weekday mode bumps the stamp to Mon Apr 3
    assert r3m.get_bar_time(_ms(_UTC, 2028, 5, 15), _UTC, None, None, 'weekday') \
        == _ms(_UTC, 2028, 4, 3)
    assert r3m.get_bar_time(_ms(_UTC, 2028, 5, 15), _UTC, None, None, 'calendar') \
        == _ms(_UTC, 2028, 4, 1)


def __test_observed_day_counter__():
    """Observed counting: weekday seed, +1 per present day, year reset."""
    c = ObservedDayCounter()
    # Thu Jan 9 2025 is the year's 7th weekday -> seed ordinal 6
    assert c.ordinal(date(2025, 1, 9)) == 6
    assert c.ordinal(date(2025, 1, 9)) == 6  # same day is idempotent
    assert c.ordinal(date(2025, 1, 10)) == 7
    # A holiday gap consumes no slot: the next present day is simply +1
    assert c.ordinal(date(2025, 1, 14)) == 8
    # Year change resets the counter
    assert c.ordinal(date(2026, 1, 2)) == 0


def __test_chart_bar_containment__():
    """A chart bar belongs to the day its LAST instant falls into."""
    # CAPITALCOM-style late open (Mon-Thu 17:05): the 240-minute bar stamped
    # 17:00 contains the 17:05 open, so its last instant is already the next
    # trading day while its own timestamp is not
    split = []
    for d in range(4):
        split.append(SymInfoInterval(day=d, start=time(0, 0), end=time(16, 59, 50)))
        split.append(SymInfoInterval(day=d, start=time(17, 5), end=time(0, 0)))
    on = overnight_opens(split, None)
    bar_open = _sec(_NY, 2025, 6, 10, 17, 0)  # Tuesday 17:00 ET
    assert trading_day(bar_open, _NY, on) == date(2025, 6, 10)
    assert trading_day(bar_open + 4 * 3600 - 1, _NY, on) == date(2025, 6, 11)


def _multiperiod_state(opens: list[int], chart_off: int = 0) -> SecurityState:
    state = SecurityState(
        sec_id='s', timeframe='5D', gaps_on=False, same_timeframe=False,
        resampler=Resampler.get_resampler('5D'), tz=_UTC,
    )
    state.bar_opens = opens
    state.chart_off = chart_off
    return state


def __test_confirmed_time_multiperiod_pointer__():
    """Multi-period securities confirm by walking the child's actual bar opens."""
    day = 86_400_000
    opens = [1_000 * day, 1_007 * day, 1_014 * day + day]  # third bar starts a day late
    state = _multiperiod_state(opens, chart_off=4 * 3_600_000 - 1)

    # Inside bar 0: nothing is confirmed yet
    assert _get_confirmed_time(state, 1_002 * day) == 0
    # Entering bar 1 confirms bar 0
    assert _get_confirmed_time(state, 1_007 * day) == opens[0]
    state.last_confirmed = opens[0]
    assert _get_confirmed_time(state, 1_010 * day) == opens[0]
    # Containment: the chart bar OPENING just before the (shifted) bar-2 open
    # already contains it -> bar 1 confirmed on that bar
    assert _get_confirmed_time(state, 1_015 * day - 4 * 3_600_000 + 1) == opens[1]
    state.last_confirmed = opens[1]
    # Inside the last bar with no fallback grid: stays
    assert _get_confirmed_time(state, 1_016 * day) == opens[1]


def __test_confirmed_time_multiperiod_first_bar__():
    """The first chart bar confirms all child bars before its own period."""
    day = 86_400_000
    opens = [1_000 * day, 1_007 * day, 1_014 * day]
    state = _multiperiod_state(opens)
    assert _get_confirmed_time(state, 1_015 * day) == opens[1]


def __test_confirmed_time_multiperiod_past_end__():
    """Past the child's last bar, the arithmetic grid decides the final close."""
    opens = [_ms(_UTC, 2025, 1, 1), _ms(_UTC, 2025, 1, 6)]  # calendar 5D slots
    state = _multiperiod_state(opens)
    state.sec_grid_args = (_UTC, None, None, 'calendar')

    # Walk into the last bar first
    assert _get_confirmed_time(state, _ms(_UTC, 2025, 1, 6)) == opens[0]
    state.last_confirmed = opens[0]
    # Still inside the Jan 6 slot
    assert _get_confirmed_time(state, _ms(_UTC, 2025, 1, 10)) == opens[0]
    # The Jan 11 slot starts -> the last child bar is confirmed
    assert _get_confirmed_time(state, _ms(_UTC, 2025, 1, 11)) == opens[1]
