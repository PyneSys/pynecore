"""
Daily HTF confirmation on an EARLY-CLOSING exchange session.

A US half-day (Black Friday, Christmas Eve) ends the regular session hours
before the schedule in ``opening_hours`` says it does. TradingView owns the real
holiday calendar and still confirms the daily bar on that day's LAST chart bar;
PyneCore's arithmetic session grid does not know about half-days, so the value
lagged a full trading day on exactly those bars.

The chart's own bar grid realizes the calendar: when the NEXT chart bar already
belongs to a later period, this bar is the period's last one. MEASURED on
``AMEX:SPY@5`` requesting ``"1D"`` — 2025-11-28 and 2025-12-24 were the only two
bars out of 20207 where the daily ``ta.rma(ta.tr, 14)`` lagged TradingView.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo

from pynecore.core.resampler import Resampler
from pynecore.core.security import SecurityState, _get_confirmed_time
from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

_NY = ZoneInfo("America/New_York")
_BAR_MS = 5 * 60 * 1000


def _ms(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=_NY).timestamp() * 1000)


def _rth() -> tuple[list[SymInfoSession], list[SymInfoInterval]]:
    """US equity RTH: 09:30-16:00 every weekday (0=Mon..4=Fri)."""
    starts = [SymInfoSession(day=d, time=time(9, 30)) for d in range(5)]
    hours = [SymInfoInterval(day=d, start=time(9, 30), end=time(16, 0))
             for d in range(5)]
    return starts, hours


def _daily_state(opens: list[int]) -> SecurityState:
    starts, hours = _rth()
    state = SecurityState(
        sec_id="d1",
        timeframe="1D",
        gaps_on=False,
        same_timeframe=False,
        resampler=Resampler.get_resampler("1D"),
        tz=_NY,
        session_starts=starts,
        session_tz=_NY,
        session_opening_hours=hours,
        chart_off=_BAR_MS - 1,  # 5m chart bars
    )
    state.bar_opens = opens
    state.bar_opens_multiperiod = False
    state.bar_ptr = -1
    return state


def _confirm(state: SecurityState, bars: list[int]) -> list[int]:
    """Target time for each chart bar, with the next bar's open as the peek."""
    out = []
    for i, ct in enumerate(bars):
        nxt = bars[i + 1] if i + 1 < len(bars) else 0
        out.append(_get_confirmed_time(state, ct, nxt))
    return out


def __test_full_session_confirms_on_its_last_bar__(log):
    """A regular session already confirms on its last bar — the peek is a no-op."""
    # 2025-11-26 (Wed) and 2025-12-01 (Mon), both full 09:30-16:00 sessions.
    d26, d01 = _ms(2025, 11, 26, 9, 30), _ms(2025, 12, 1, 9, 30)
    state = _daily_state([d26, d01])
    bars = [_ms(2025, 11, 26, 15, 50), _ms(2025, 11, 26, 15, 55), d01]
    got = _confirm(state, bars)

    assert got[0] < d26          # mid-session: the day is still open
    assert got[1] == d26         # 15:55 closes at 16:00 -> the day is confirmed
    assert got[2] == d26         # next day's first bar keeps it


def __test_early_close_confirms_on_its_last_bar__(log):
    """A 13:00 half-day confirms the daily bar on the 12:55 bar, like TradingView."""
    # 2025-11-28 (Black Friday, closes 13:00) followed by 2025-12-01.
    d28, d01 = _ms(2025, 11, 28, 9, 30), _ms(2025, 12, 1, 9, 30)
    state = _daily_state([d28, d01])
    bars = [_ms(2025, 11, 28, 12, 50), _ms(2025, 11, 28, 12, 55), d01]
    got = _confirm(state, bars)

    assert got[0] < d28          # mid-session
    assert got[1] == d28         # the exchange closed here -> confirm the day
    assert got[2] == d28


def __test_no_peek_keeps_the_schedule__(log):
    """Without a next bar (live, last historical bar) the session schedule rules."""
    d28 = _ms(2025, 11, 28, 9, 30)
    state = _daily_state([d28])
    # The half-day's last bar, with nothing behind it: the scheduled 16:00 close
    # has not been reached, so nothing new is confirmed.
    assert _get_confirmed_time(state, _ms(2025, 11, 28, 12, 55), 0) < d28
