"""
Session-anchored intraday HTF alignment (issue #63).

TradingView anchors intraday HTF bars to the session open (a 09:30 open at 1H
gives 09:30, 10:30, 11:30…), while the historical default is a pure UTC-epoch
clock-floor. These tests cover the anchor helper, the session-aware
``Resampler.get_bar_time``, the ``needs_anchor`` decision, the chart-side
``_get_confirmed_time`` advance, and DST stepping.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo

from pynecore.core.resampler import Resampler, _session_anchor_sec
from pynecore.core.security import (
    SecurityState, _get_confirmed_time, _needs_session_anchor,
)
from pynecore.core.syminfo import SymInfoSession


_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


def _ms(tz: ZoneInfo, y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=tz).timestamp() * 1000)


def _sec(tz: ZoneInfo, y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=tz).timestamp())


def _rth_0930() -> list[SymInfoSession]:
    """RTH equity opening 09:30 every weekday (Python weekday 0=Mon..4=Fri)."""
    return [SymInfoSession(day=d, time=time(9, 30, 0)) for d in range(5)]


def __test_session_anchor_contains_bar__(log):
    """Anchor returns the open of the session that CONTAINS the bar timestamp."""
    ss = _rth_0930()
    # Wednesday 2024-01-17, NY.
    open_sec = _sec(_NY, 2024, 1, 17, 9, 30)

    # Just after the open → today's open.
    assert _session_anchor_sec(_sec(_NY, 2024, 1, 17, 9, 31), _NY, ss) == open_sec
    # Mid-session → still today's open.
    assert _session_anchor_sec(_sec(_NY, 2024, 1, 17, 14, 0), _NY, ss) == open_sec
    # Just before the open (pre-market) belongs to a prior session, not today's.
    assert _session_anchor_sec(_sec(_NY, 2024, 1, 17, 9, 0), _NY, ss) < open_sec


def __test_session_anchor_overnight__(log):
    """An after-midnight bar anchors to the previous evening's session open."""
    # Overnight session opening 18:00 every weekday.
    ss = [SymInfoSession(day=d, time=time(18, 0, 0)) for d in range(7)]
    # Wednesday 02:00 belongs to Tuesday 18:00.
    got = _session_anchor_sec(_sec(_NY, 2024, 1, 17, 2, 0), _NY, ss)
    assert got == _sec(_NY, 2024, 1, 16, 18, 0)


def __test_session_anchor_no_session_falls_back__(log):
    """No applicable session → 0, which reproduces the plain clock-floor."""
    assert _session_anchor_sec(_sec(_UTC, 2024, 1, 17, 9, 31), _UTC, []) == 0


def __test_get_bar_time_anchored_to_0930__(log):
    """60m bars anchor to 09:30 → 09:30, 10:30, 11:30 (TradingView grid)."""
    ss = _rth_0930()
    r = Resampler.get_resampler("60")
    o0930 = _ms(_NY, 2024, 1, 17, 9, 30)
    o1030 = _ms(_NY, 2024, 1, 17, 10, 30)

    for mi in (30, 45, 59):  # 09:30..09:59 → 09:30 bar
        assert r.get_bar_time(_ms(_NY, 2024, 1, 17, 9, mi), _NY, ss) == o0930
    assert r.get_bar_time(_ms(_NY, 2024, 1, 17, 10, 0), _NY, ss) == o0930
    assert r.get_bar_time(_ms(_NY, 2024, 1, 17, 10, 29), _NY, ss) == o0930
    assert r.get_bar_time(_ms(_NY, 2024, 1, 17, 10, 30), _NY, ss) == o1030


def __test_get_bar_time_ongrid_is_noop__(log):
    """On-grid opens (00:00) make the anchored grid identical to clock-floor."""
    ss = [SymInfoSession(day=d, time=time(0, 0, 0)) for d in range(7)]
    r = Resampler.get_resampler("60")
    for mi in (0, 30, 59):
        for h in (0, 1, 12, 23):
            ts = _ms(_NY, 2024, 1, 17, h, mi)
            assert r.get_bar_time(ts, _NY, ss) == r.get_bar_time(ts, _NY)


def __test_get_bar_time_default_unchanged__(log):
    """Without session_starts the result is the unchanged pure clock-floor."""
    r = Resampler.get_resampler("60")
    ts = _ms(_UTC, 2024, 1, 17, 9, 30)
    assert r.get_bar_time(ts) == _ms(_UTC, 2024, 1, 17, 9, 0)


def __test_needs_anchor_is_per_timeframe__(log):
    """A 09:30 open is off-grid at 60m but on-grid at 30m."""
    ss = _rth_0930()
    assert _needs_session_anchor(ss, _NY, "60") is True
    assert _needs_session_anchor(ss, _NY, "30") is False  # 09:30 lands on the 30m grid
    # 00:00 open is on-grid for both.
    ss0 = [SymInfoSession(day=d, time=time(0, 0, 0)) for d in range(7)]
    assert _needs_session_anchor(ss0, _NY, "60") is False
    # Daily/weekly/monthly are never session-anchored here.
    assert _needs_session_anchor(ss, _NY, "1D") is False


def __test_get_confirmed_time_session_anchored__(log):
    """
    The chart-side OFF/historical advance confirms HTF periods on the
    session-anchored boundaries (10:30, 11:30) — not the clock-floor (10:00…).
    Each HTF period is confirmed on its own LAST chart bar (the one whose
    close lands on the period boundary), per TV's historical ``lookahead_off``
    merge rule.
    """
    ss = _rth_0930()
    state = SecurityState(
        sec_id="s60",
        timeframe="60",
        gaps_on=False,
        same_timeframe=False,
        resampler=Resampler.get_resampler("60"),
        tz=_NY,
        session_starts=ss,
        session_tz=_NY,
        chart_off=30 * 60 * 1000 - 1,  # 30m chart bars
    )

    o0930 = _ms(_NY, 2024, 1, 17, 9, 30)
    o1030 = _ms(_NY, 2024, 1, 17, 10, 30)
    # Seed past the previous session so the first bar's target (a prior-day
    # period) does not register as a fresh confirmation.
    state.last_confirmed = o0930 - 1

    # 30m chart bars from the 09:30 open.
    chart_bars = [
        _ms(_NY, 2024, 1, 17, 9, 30),
        _ms(_NY, 2024, 1, 17, 10, 0),    # closes at 10:30 → confirms 09:30
        _ms(_NY, 2024, 1, 17, 10, 30),
        _ms(_NY, 2024, 1, 17, 11, 0),    # closes at 11:30 → confirms 10:30
        _ms(_NY, 2024, 1, 17, 11, 30),
    ]
    confirmed: list[int | None] = []
    for ct in chart_bars:
        target = _get_confirmed_time(state, ct)
        if target > state.last_confirmed:
            state.last_confirmed = target
            confirmed.append(target)
        else:
            confirmed.append(None)

    # Inside the 09:30 period → no fresh confirmation.
    assert confirmed[0] is None
    # The 10:00 bar closes exactly at the 10:30 boundary → confirms 09:30.
    assert confirmed[1] == o0930
    assert confirmed[2] is None
    # The 11:00 bar closes at 11:30 → confirms 10:30.
    assert confirmed[3] == o1030
    assert confirmed[4] is None


def __test_dst_overnight_absolute_stepping__(log):
    """
    Across a DST spring-forward, an off-grid overnight session steps in absolute
    epoch seconds, so consecutive 60m boundaries stay exactly 3600s apart even
    though the wall clock jumps an hour.
    """
    # Overnight session opening 18:30 (off-grid for 60m), spanning the
    # 2024-03-10 02:00→03:00 spring-forward in New York.
    ss = [SymInfoSession(day=d, time=time(18, 30, 0)) for d in range(7)]
    r = Resampler.get_resampler("60")
    anchor = _sec(_NY, 2024, 3, 9, 18, 30)

    # Bars before and after the transition, both within the same overnight session.
    before = _ms(_NY, 2024, 3, 10, 1, 45)   # 01:45 EST (pre-jump)
    after = _ms(_NY, 2024, 3, 10, 4, 45)    # 04:45 EDT (post-jump)
    b_start = r.get_bar_time(before, _NY, ss) // 1000
    a_start = r.get_bar_time(after, _NY, ss) // 1000

    # Both boundaries are anchor + k*3600 (absolute-time grid, DST-safe).
    assert (b_start - anchor) % 3600 == 0
    assert (a_start - anchor) % 3600 == 0
    # And the wall clock advanced across the gap (a_start is later than b_start).
    assert a_start > b_start
