"""
@pyne
"""
from pynecore.lib import script, plot, time, time_close


@script.indicator(title="Trading-day close", shorttitle="tdc")
def main():
    plot(time, "t")
    plot(time_close, "tc")
    plot(time_close("60"), "tc60")
    plot(time_close("240"), "tc240")
    plot(time_close("D"), "tcD")
    plot(time("D"), "tD")


def _bars(isos):
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV
    out = []
    for iso in isos:
        ts = int(datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp())
        out.append(OHLCV(timestamp=ts, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
    return out


def _ms(tz_name, y, mo, d, h, mi=0, s=0):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return int(datetime(y, mo, d, h, mi, s, tzinfo=ZoneInfo(tz_name)).timestamp() * 1000)


def __test_time_close_trading_day_end__(runner, log):
    """ The (shortened) last bar of an exchange day closes at the trading-day end """
    from datetime import time as dt_time
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    # TSE-style template (TradingView reports 7203 as a single 09:00-15:31
    # interval; the 15:30 closing-auction bar is shortened to one minute).
    days = (0, 1, 2, 3, 4)
    override = {
        "period": "30",
        "type": "stock",
        "timezone": "Asia/Tokyo",
        "opening_hours": [SymInfoInterval(day=d, start=dt_time(9, 0), end=dt_time(15, 31)) for d in days],
        "session_starts": [SymInfoSession(day=d, time=dt_time(9, 0)) for d in days],
        "session_ends": [SymInfoSession(day=d, time=dt_time(15, 31)) for d in days],
    }

    # Wednesday 2025-01-08, bars in UTC (JST = UTC+9)
    bars = _bars([
        "2025-01-08T00:00:00",  # 09:00 JST
        "2025-01-08T02:30:00",  # 11:30 JST
        "2025-01-08T06:00:00",  # 15:00 JST
        "2025-01-08T06:30:00",  # 15:30 JST (shortened auction bar)
    ])
    tz = "Asia/Tokyo"
    expected = {
        # Mid-day bars close at open + span, the auction bar at the day end
        "tc": [_ms(tz, 2025, 1, 8, 9, 30), _ms(tz, 2025, 1, 8, 12, 0),
               _ms(tz, 2025, 1, 8, 15, 30), _ms(tz, 2025, 1, 8, 15, 31)],
        # The last hourly bar (15:00-16:00 by the clock) is day-end capped
        "tc60": [_ms(tz, 2025, 1, 8, 10, 0), _ms(tz, 2025, 1, 8, 12, 0),
                 _ms(tz, 2025, 1, 8, 15, 31), _ms(tz, 2025, 1, 8, 15, 31)],
        "tcD": [_ms(tz, 2025, 1, 8, 15, 31)] * 4,
        # Daily bars open at the session open, not at midnight
        "tD": [_ms(tz, 2025, 1, 8, 9, 0)] * 4,
    }

    results = {k: [] for k in expected}
    for _candle, plots in runner(iter(bars), syminfo_override=override).run_iter():
        for k in results:
            results[k].append(plots[k])
    assert results == expected, f"exchange-day mismatch: {results} != {expected}"

    _check_overnight_session(runner)
    _check_24_7(runner)


def _check_overnight_session(runner):
    """ Overnight (FX) sessions: no midnight cap, day ends at 17:00, daily opens 17:00 """
    from datetime import time as dt_time
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    ny_days = (0, 1, 2, 3, 4, 6)
    override = {
        "period": "60",
        "type": "forex",
        "timezone": "America/New_York",
        "opening_hours": [SymInfoInterval(day=d, start=dt_time(17, 0), end=dt_time(17, 0)) for d in ny_days],
        "session_starts": [SymInfoSession(day=d, time=dt_time(17, 0)) for d in ny_days],
        "session_ends": [SymInfoSession(day=d, time=dt_time(17, 0)) for d in ny_days],
    }

    # Sun 2025-01-05 .. Tue 2025-01-07, bars in UTC (NY = EST = UTC-5)
    bars = _bars([
        "2025-01-05T22:00:00",  # Sun 17:00 NY — session open, Monday's trading day
        "2025-01-06T14:00:00",  # Mon 09:00 NY
        "2025-01-06T21:00:00",  # Mon 16:00 NY — last hour of Monday's trading day
        "2025-01-06T22:00:00",  # Mon 17:00 NY — Tuesday's trading day
        "2025-01-07T02:00:00",  # Mon 21:00 NY — 4H bar crossing midnight
    ])
    tz = "America/New_York"
    expected = {
        "tc": [_ms(tz, 2025, 1, 5, 18, 0), _ms(tz, 2025, 1, 6, 10, 0),
               _ms(tz, 2025, 1, 6, 17, 0), _ms(tz, 2025, 1, 6, 18, 0),
               _ms(tz, 2025, 1, 6, 22, 0)],
        # 4H bars anchor to the 17:00 session open; the 21:00 bar runs into the
        # next calendar day uncapped (continuous overnight session)
        "tc240": [_ms(tz, 2025, 1, 5, 21, 0), _ms(tz, 2025, 1, 6, 13, 0),
                  _ms(tz, 2025, 1, 6, 17, 0), _ms(tz, 2025, 1, 6, 21, 0),
                  _ms(tz, 2025, 1, 7, 1, 0)],
        "tcD": [_ms(tz, 2025, 1, 6, 17, 0), _ms(tz, 2025, 1, 6, 17, 0),
                _ms(tz, 2025, 1, 6, 17, 0), _ms(tz, 2025, 1, 7, 17, 0),
                _ms(tz, 2025, 1, 7, 17, 0)],
        # Daily bars open at the previous evening's session open
        "tD": [_ms(tz, 2025, 1, 5, 17, 0), _ms(tz, 2025, 1, 5, 17, 0),
               _ms(tz, 2025, 1, 5, 17, 0), _ms(tz, 2025, 1, 6, 17, 0),
               _ms(tz, 2025, 1, 6, 17, 0)],
    }

    results = {k: [] for k in expected}
    for _candle, plots in runner(iter(bars), syminfo_override=override).run_iter():
        for k in results:
            results[k].append(plots[k])
    assert results == expected, f"overnight-session mismatch: {results} != {expected}"


def _check_24_7(runner):
    """ 24/7 markets: the trading-day end is the next midnight — nothing changes """
    from datetime import time as dt_time
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    override = {
        "period": "60",
        "type": "crypto",
        "timezone": "Etc/UTC",
        "opening_hours": [SymInfoInterval(day=d, start=dt_time(0, 0), end=dt_time(0, 0)) for d in range(7)],
        "session_starts": [SymInfoSession(day=d, time=dt_time(0, 0)) for d in range(7)],
        "session_ends": [SymInfoSession(day=d, time=dt_time(0, 0)) for d in range(7)],
    }

    bars = _bars([
        "2025-01-08T20:00:00",
        "2025-01-08T23:00:00",
    ])
    tz = "Etc/UTC"
    expected = {
        "tc": [_ms(tz, 2025, 1, 8, 21, 0), _ms(tz, 2025, 1, 9, 0, 0)],
        "tc240": [_ms(tz, 2025, 1, 9, 0, 0), _ms(tz, 2025, 1, 9, 0, 0)],
        "tcD": [_ms(tz, 2025, 1, 9, 0, 0), _ms(tz, 2025, 1, 9, 0, 0)],
        "tD": [_ms(tz, 2025, 1, 8, 0, 0), _ms(tz, 2025, 1, 8, 0, 0)],
    }

    results = {k: [] for k in expected}
    for _candle, plots in runner(iter(bars), syminfo_override=override).run_iter():
        for k in results:
            results[k].append(plots[k])
    assert results == expected, f"mismatch: {results} != {expected}"
