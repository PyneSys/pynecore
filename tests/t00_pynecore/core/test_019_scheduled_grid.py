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


def __test_observed_day_counter_fold__():
    """Holiday half-day fold: the day after an early close shares its ordinal.

    Detected from data alone (no calendar): a trading day is an early close when
    its last bar ends before the scheduled session close; the next calendar day
    then folds into it instead of advancing. A folded day is itself not a trigger
    (chain-stop).
    """
    tz = ZoneInfo("America/Chicago")
    t17, t16 = time(17, 0), time(16, 0)
    # CME-style: opens Sun-Thu 17:00, closes next day 16:00 (trading-day close)
    hours = [SymInfoInterval(day=d, start=t17, end=t16) for d in (6, 0, 1, 2, 3)]

    def close_sec(d: date, h: int, mi: int) -> int:
        return int(datetime(d.year, d.month, d.day, h, mi, tzinfo=tz).timestamp())

    mon, tue = date(2025, 6, 23), date(2025, 6, 24)
    wed, thu = date(2025, 6, 25), date(2025, 6, 26)

    c = ObservedDayCounter(tz, hours, fold=True)
    o_mon = c.ordinal(mon, close_sec(mon, 16, 0))   # normal: ends at the close
    o_tue = c.ordinal(tue, close_sec(tue, 13, 0))   # early close (13:00)
    assert o_tue == o_mon + 1                        # Tuesday itself advances
    o_wed = c.ordinal(wed, close_sec(wed, 13, 0))   # folds into early-close Tue
    assert o_wed == o_tue                            # shares Tuesday's ordinal
    o_thu = c.ordinal(thu, close_sec(thu, 16, 0))   # chain-stop: Wed was folded
    assert o_thu == o_wed + 1

    # Control: with folding off the same sequence advances every day
    nf = ObservedDayCounter(tz, hours, fold=False)
    n_mon = nf.ordinal(mon, close_sec(mon, 16, 0))
    n_tue = nf.ordinal(tue, close_sec(tue, 13, 0))
    n_wed = nf.ordinal(wed, close_sec(wed, 13, 0))
    assert (n_tue, n_wed) == (n_mon + 1, n_mon + 2)


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
    state.bar_opens_multiperiod = True
    state.chart_off = chart_off
    return state


def _single_period_state(tf: str, opens: 'list[int] | None',
                         chart_off: int = 30 * 60 * 1000 - 1) -> SecurityState:
    """Single-period (1D/1W/1M) state: grid + clamp when ``opens`` is loaded."""
    state = SecurityState(
        sec_id='s', timeframe=tf, gaps_on=False, same_timeframe=False,
        resampler=Resampler.get_resampler(tf), tz=_UTC,
    )
    state.bar_opens = opens
    state.bar_opens_multiperiod = False
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


def __test_confirmed_time_multiperiod_closing_bar__():
    """The chart bar closing exactly on a child-bar boundary already confirms.

    TV ``lookahead_off`` merge rule: the period's last chart bar (whose close
    instant equals the next child bar's open) carries the confirmation — not
    the next period's first bar.
    """
    day = 86_400_000
    opens = [1_000 * day, 1_007 * day, 1_014 * day]
    state = _multiperiod_state(opens, chart_off=4 * 3_600_000 - 1)
    # The 4h chart bar opening at 1_007*day - 4h closes exactly at the bar-1
    # open → bar 0 is confirmed on this bar.
    assert _get_confirmed_time(state, 1_007 * day - 4 * 3_600_000) == opens[0]


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


def __test_confirmed_time_single_period_sparse_forward_fill__():
    """Sparse single-period daily confirms at the CALENDAR close and holds.

    A monthly-cadence series queried at ``1D`` carries a bar only on scattered
    days (Jan 31, Feb 28, Mar 31). TradingView confirms each daily bar at its
    own calendar close (the next midnight) and forward-fills it across the gap;
    the grid+clamp path reproduces this. Between real bars the target does not
    move, so the chart-side ``target > last_confirmed`` gate keeps ``new_period``
    False — ``gaps_off`` holds the value while ``gaps_on`` emits ``na``. (The
    extent-walk used for multi-period would instead delay Jan 31 all the way to
    the next real bar Feb 28, lagging a full gap — the bug this path fixes.)
    """
    opens = [_ms(_UTC, 2025, 1, 31), _ms(_UTC, 2025, 2, 28), _ms(_UTC, 2025, 3, 31)]
    state = _single_period_state('1D', opens)

    # Jan 31's last 30m bar closes at Feb 1 00:00 -> Jan 31 confirmed here.
    jan31_last = _ms(_UTC, 2025, 2, 1) - 30 * 60 * 1000
    assert _get_confirmed_time(state, jan31_last) == opens[0]
    state.last_confirmed = opens[0]
    # Mid-gap chart bars (no real child bar) hold Jan 31 -> new_period stays False.
    assert _get_confirmed_time(state, _ms(_UTC, 2025, 2, 14)) == opens[0]
    assert _get_confirmed_time(state, _ms(_UTC, 2025, 2, 27)) == opens[0]
    # Feb 28's last 30m bar closes at Mar 1 00:00 -> Feb 28 confirmed on its own
    # calendar close, NOT delayed to the next real bar (Mar 31).
    feb28_last = _ms(_UTC, 2025, 3, 1) - 30 * 60 * 1000
    assert _get_confirmed_time(state, feb28_last) == opens[1]


def __test_confirmed_time_single_period_dense_matches_bare_grid__():
    """Dense daily: the clamp is a no-op — identical to the bare arithmetic grid.

    A real bar on every calendar day means every grid target coincides with an
    open, so grid+clamp must agree bar-for-bar with the ``bar_opens is None``
    arithmetic path. This is the zero-regression guarantee for the common case.
    """
    opens = [_ms(_UTC, 2025, 3, d) for d in range(1, 28)]  # bar every day in March
    clamped = _single_period_state('1D', opens)
    bare = _single_period_state('1D', None)  # bar_opens None -> bare grid

    t = _ms(_UTC, 2025, 3, 2)
    end = _ms(_UTC, 2025, 3, 27)
    step = 6 * 3_600_000  # 6h cadence crosses every daily boundary
    while t <= end:
        a = _get_confirmed_time(clamped, t)
        b = _get_confirmed_time(bare, t)
        assert a == b, f"clamp {a} != bare grid {b} at {t}"
        for st in (clamped, bare):
            if a > st.last_confirmed:
                st.last_confirmed = a
        t += step


def __test_confirmed_time_single_period_past_end_holds_last__():
    """Past the last sparse bar the value holds (forward-fill), never goes na."""
    opens = [_ms(_UTC, 2025, 1, 31), _ms(_UTC, 2025, 2, 28)]
    state = _single_period_state('1D', opens)

    feb28_last = _ms(_UTC, 2025, 3, 1) - 30 * 60 * 1000
    assert _get_confirmed_time(state, feb28_last) == opens[1]
    state.last_confirmed = opens[1]
    # Weeks past the last child bar: still Feb 28 (held), not na.
    assert _get_confirmed_time(state, _ms(_UTC, 2025, 3, 20)) == opens[1]
    assert _get_confirmed_time(state, _ms(_UTC, 2025, 4, 10)) == opens[1]


def __test_confirmed_time_single_period_before_first_bar__():
    """Before the first real child open nothing is confirmed (stays na)."""
    opens = [_ms(_UTC, 2025, 2, 10)]
    state = _single_period_state('1D', opens)

    # Chart bars before the Feb 10 child bar exists -> no confirmation yet.
    assert _get_confirmed_time(state, _ms(_UTC, 2025, 2, 3)) == 0
    assert _get_confirmed_time(state, _ms(_UTC, 2025, 2, 9)) == 0
    # Feb 10's last 30m bar closes Feb 11 00:00 -> confirmed.
    feb10_last = _ms(_UTC, 2025, 2, 11) - 30 * 60 * 1000
    assert _get_confirmed_time(state, feb10_last) == opens[0]
