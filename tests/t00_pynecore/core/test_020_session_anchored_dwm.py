"""
Session-anchored single-period D/W/M grid.

TradingView opens a daily bar at its trading day's session open, not at the
exchange timezone's midnight — measured on CAPITALCOM:EURUSD (daily bars at
17:00 New York) and NASDAQ:AAPL (09:30 New York). ``Resampler.get_bar_time``
only honoured that for MULTI-period (nD/nW/nM) timeframes, so a plain
``request.security(syminfo.tickerid, "D", ...)`` on any market that does not
open at midnight read the wrong period: on EURUSD@30 its daily ``close``
matched TradingView on 299 of 5493 bars, and a corpus strategy built on the
daily ATR diverged. With the fix all 5493 match.

Because a period runs open-to-open, ``trading_day`` (a calendar-date answer
derived from opens alone) needs correcting at both ends of the window: an
instant before the day's own open belongs to the previous scheduled day, and
one at or after its close belongs to the next — the latter is what closes the
last trading day of an overnight market's week, which has no open of its own.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo

from pynecore.core.resampler import Resampler, trading_day_end_sec, scheduled_day_open_sec
from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

_NY = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


def _ms(tz: ZoneInfo, y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=tz).timestamp() * 1000)


def _fx() -> tuple[list[SymInfoInterval], list[SymInfoSession]]:
    """FX week: opens Sunday 17:00, each session runs 24h, closes Friday 17:00."""
    t17 = time(17, 0)
    days = [6, 0, 1, 2, 3]
    return ([SymInfoInterval(day=d, start=t17, end=t17) for d in days],
            [SymInfoSession(day=d, time=t17) for d in days])


def _rth() -> tuple[list[SymInfoInterval], list[SymInfoSession]]:
    """US equity regular session: Mon-Fri 09:30-16:00."""
    return ([SymInfoInterval(day=d, start=time(9, 30), end=time(16, 0)) for d in range(5)],
            [SymInfoSession(day=d, time=time(9, 30)) for d in range(5)])


def _c247() -> tuple[list[SymInfoInterval], list[SymInfoSession]]:
    """Round-the-clock crypto: every day 00:00-00:00."""
    return ([SymInfoInterval(day=d, start=time(0, 0), end=time(0, 0)) for d in range(7)],
            [SymInfoSession(day=d, time=time(0, 0)) for d in range(7)])


def __test_daily_opens_at_fx_session_open__(log):
    """An FX daily bar opens the previous evening at 17:00, not at midnight"""
    oh, ss = _fx()
    r = Resampler.get_resampler("D")
    # Wednesday morning belongs to the day that opened Tuesday 17:00.
    assert r.get_bar_time(_ms(_NY, 2025, 6, 11, 6, 0), _NY, ss, oh) \
        == _ms(_NY, 2025, 6, 10, 17, 0)
    # The 17:00 open itself starts the next daily bar.
    assert r.get_bar_time(_ms(_NY, 2025, 6, 11, 17, 0), _NY, ss, oh) \
        == _ms(_NY, 2025, 6, 11, 17, 0)


def __test_daily_opens_at_equity_session_open__(log):
    """An equity daily bar opens 09:30; a pre-open bar is still the day before"""
    oh, ss = _rth()
    r = Resampler.get_resampler("D")
    assert r.get_bar_time(_ms(_NY, 2025, 6, 11, 10, 0), _NY, ss, oh) \
        == _ms(_NY, 2025, 6, 11, 9, 30)
    assert r.get_bar_time(_ms(_NY, 2025, 6, 11, 9, 29), _NY, ss, oh) \
        == _ms(_NY, 2025, 6, 10, 9, 30)
    # Monday pre-open walks back over the weekend to Friday's session.
    assert r.get_bar_time(_ms(_NY, 2025, 6, 9, 8, 0), _NY, ss, oh) \
        == _ms(_NY, 2025, 6, 6, 9, 30)


def __test_period_ends_at_session_close__(log):
    """At the day's close the period is over — the closing bar confirms it"""
    oh, ss = _rth()
    r = Resampler.get_resampler("D")
    # 16:00 ends Wednesday's period, so it no longer resolves to Wednesday.
    assert r.get_bar_time(_ms(_NY, 2025, 6, 11, 16, 0), _NY, ss, oh) \
        > _ms(_NY, 2025, 6, 11, 9, 30)
    # FX: Friday 17:00 closes the week even though nothing opens on a Friday.
    fx_oh, fx_ss = _fx()
    assert r.get_bar_time(_ms(_NY, 2025, 6, 13, 17, 0), _NY, fx_ss, fx_oh) \
        > _ms(_NY, 2025, 6, 12, 17, 0)
    assert r.get_bar_time(_ms(_NY, 2025, 6, 13, 16, 30), _NY, fx_ss, fx_oh) \
        == _ms(_NY, 2025, 6, 12, 17, 0)


def __test_round_the_clock_keeps_midnight_grid__(log):
    """A 24/7 market's daily bars stay on midnight — the fix is a no-op there"""
    oh, ss = _c247()
    r = Resampler.get_resampler("D")
    for hour in (0, 6, 13, 23):
        assert r.get_bar_time(_ms(_UTC, 2025, 6, 11, hour), _UTC, ss, oh) \
            == _ms(_UTC, 2025, 6, 11), hour
    # And identical to the sessionless clock-floor path.
    assert r.get_bar_time(_ms(_UTC, 2025, 6, 11, 13), _UTC, ss, oh) \
        == r.get_bar_time(_ms(_UTC, 2025, 6, 11, 13), _UTC)


def __test_weekly_opens_with_the_week__(log):
    """A weekly bar opens at the session open of its week's first trading day"""
    oh, ss = _fx()
    r = Resampler.get_resampler("W")
    # Week of Mon 2025-06-09 opens Sunday 2025-06-08 17:00.
    assert r.get_bar_time(_ms(_NY, 2025, 6, 11, 6, 0), _NY, ss, oh) \
        == _ms(_NY, 2025, 6, 8, 17, 0)


def __test_trading_day_end_selection__(log):
    """Day end comes from rolling and bounded sessions; 24h intervals have none"""
    fx_oh, _ = _fx()
    # Friday closes the FX week at 17:00 (end of the interval that opened Thu).
    assert trading_day_end_sec(_d(2025, 6, 13), _NY, fx_oh) == _sec(_NY, 2025, 6, 13, 17, 0)
    rth_oh, _ = _rth()
    assert trading_day_end_sec(_d(2025, 6, 11), _NY, rth_oh) == _sec(_NY, 2025, 6, 11, 16, 0)
    c_oh, _ = _c247()
    assert trading_day_end_sec(_d(2025, 6, 11), _NY, c_oh) is None
    assert trading_day_end_sec(_d(2025, 6, 11), _NY, None) is None


def __test_scheduled_day_open_is_none_off_schedule__(log):
    """An unscheduled day reports no open instead of a synthesised midnight"""
    _, ss = _rth()
    on: dict[int, time] = {}
    assert scheduled_day_open_sec(_d(2025, 6, 11), _NY, ss, on) \
        == _sec(_NY, 2025, 6, 11, 9, 30)
    assert scheduled_day_open_sec(_d(2025, 6, 14), _NY, ss, on) is None  # Saturday


def _d(y: int, mo: int, day: int):
    from datetime import date
    return date(y, mo, day)


def _sec(tz: ZoneInfo, y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=tz).timestamp())
