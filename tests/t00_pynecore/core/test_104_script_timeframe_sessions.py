"""
Session-bounded symbols in :class:`ScriptTimeframe`.

Two things a nominal period length gets wrong on a symbol that trades only part
of the day, both checked here against the class directly:

* the final complete daily bar -- the session ends at 17:00, so the day IS
  complete even though the civil day runs to midnight;
* ``last_bar_index`` -- a market holiday has no bar, so a scheduled-grid count
  reports one script bar more than the feed contains.
"""
from datetime import datetime, time, UTC
from zoneinfo import ZoneInfo

from pynecore.core.script_timeframe import ScriptTimeframe
from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
from pynecore.types.ohlcv import OHLCV

_TZ = ZoneInfo("UTC")
_HOUR_MS = 3600000


def _syminfo() -> SymInfo:
    """A Monday-Friday 09:00-17:00 symbol on a 60-minute chart."""
    return SymInfo(
        prefix="PYTEST", description="Session Symbol", ticker="SESS", currency="USD",
        period="60", type="stock", mintick=0.01, pricescale=100, minmove=1,
        pointvalue=1, mincontract=1, timezone="UTC", volumetype="base",
        taker_fee=0.0, maker_fee=0.0,
        opening_hours=[SymInfoInterval(day=i, start=time(9, 0), end=time(17, 0))
                       for i in range(5)],
        session_starts=[SymInfoSession(day=i, time=time(9, 0)) for i in range(5)],
        session_ends=[SymInfoSession(day=i, time=time(17, 0)) for i in range(5)],
    )


def _candle(time_ms: int) -> OHLCV:
    """One flat chart bar at ``time_ms``."""
    return OHLCV(time_ms, 100.0, 101.0, 99.0, 100.5, 1.0)


def _session_bars(days: list[int]) -> list[int]:
    """Open times of the 09:00..16:00 hourly bars of the given January 2024 days."""
    out = []
    for day in days:
        base = int(datetime(2024, 1, day, 9, 0, tzinfo=UTC).timestamp()) * 1000
        for hour in range(8):
            out.append(base + hour * _HOUR_MS)
    return out


def __test_last_session_day_completes__():
    """The Friday bar closing at 17:00 completes Friday's daily bar."""
    times = _session_bars([15, 16, 17, 18, 19])
    stf = ScriptTimeframe("1D", True, "60", _syminfo(), _TZ)
    stf.prepare(times[0], times[-1], iter(times))

    assert stf.last_bar_index == 4
    assert stf.last_bar_time == stf.period_start(times[-1])

    closed = []
    for index, time_ms in enumerate(times):
        candle = _candle(time_ms)
        htf = stf.feed(candle, index == len(times) - 1)
        if htf is not None:
            closed.append(htf.timestamp)
    assert len(closed) == 5
    assert closed[-1] == stf.last_bar_time
    assert stf.is_last


def __test_last_bar_index_skips_a_holiday__():
    """A missing trading day is one script bar fewer, not one the grid schedules."""
    times = _session_bars([15, 16, 18, 19])       # 2024-01-17 is a holiday
    stf = ScriptTimeframe("1D", True, "60", _syminfo(), _TZ)
    stf.prepare(times[0], times[-1], iter(times))
    assert stf.last_bar_index == 3

    # Without a replayable feed the count falls back to the scheduled grid,
    # which cannot know the holiday and reports the scheduled bar too.
    scheduled = ScriptTimeframe("1D", True, "60", _syminfo(), _TZ)
    scheduled.prepare(times[0], times[-1])
    assert scheduled.last_bar_index is not None
    assert scheduled.last_bar_index >= stf.last_bar_index


def __test_unknown_feed_end_still_has_a_last_bar__():
    """A live-style feed with no known end closes its open period on the last bar."""
    times = _session_bars([15, 16])
    stf = ScriptTimeframe("1D", False, "60", _syminfo(), _TZ)
    stf.prepare(times[0], None)
    assert stf.last_bar_index is None

    closed = []
    for index, time_ms in enumerate(times):
        htf = stf.feed(_candle(time_ms), index == len(times) - 1)
        if htf is not None:
            closed.append(htf.timestamp)
    assert len(closed) == 2
    assert stf.is_last
