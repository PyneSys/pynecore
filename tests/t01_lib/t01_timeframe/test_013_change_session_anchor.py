"""
@pyne
"""
from pynecore.lib import script, timeframe, plot


@script.indicator(title="Timeframe Change Session Anchor", shorttitle="tf_change_anchor")
def main():
    plot(1 if timeframe.change("30") else 0, "chg30")
    plot(1 if timeframe.change("60") else 0, "chg60")
    plot(1 if timeframe.change("240") else 0, "chg240")


def __test_helper_syminfo_override():
    """ CME-style session: Sun..Thu 17:00 to next day 16:00 America/Chicago """
    from datetime import time
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    days = [6, 0, 1, 2, 3]
    return dict(
        prefix="CME_MINI", ticker="NQ1!", period="5", type="futures", timezone="America/Chicago",
        opening_hours=[SymInfoInterval(d, time(17), time(16)) for d in days],
        session_starts=[SymInfoSession(d, time(17)) for d in days],
        session_ends=[SymInfoSession((d + 1) % 7, time(16)) for d in days],
    )


def __test_helper_bars(start_ct, end_ct):
    """ Flat 5 minute bars with the daily 16:00-17:00 CT maintenance break left out """
    from datetime import timedelta, timezone
    from pynecore.types.ohlcv import OHLCV

    bars, t = [], start_ct
    while t < end_ct:
        if not 16 <= t.hour < 17:
            ms = int(t.astimezone(timezone.utc).timestamp() * 1000)
            bars.append(OHLCV(timestamp=ms, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
        t += timedelta(minutes=5)
    return bars


def __test_helper_changes(runner, start_ct):
    """ Bar times (CT) where timeframe.change("30") is true """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ct = ZoneInfo("America/Chicago")
    end_ct = datetime(2026, 9, 22, 19, 0, tzinfo=ct)
    data = __test_helper_bars(start_ct, end_ct)
    return [datetime.fromtimestamp(candle.timestamp / 1000, ct)
            for candle, _plot in runner(data, __test_helper_syminfo_override()).run_iter()
            if _plot["chg30"]]


def __test_change_anchor_data_starts_at_session_open__(runner):
    """ timeframe.change("30") is on the session grid when the data starts at a session open """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    start = datetime(2026, 9, 21, 17, 0, tzinfo=ZoneInfo("America/Chicago"))
    changes = __test_helper_changes(runner, start)
    assert changes
    assert all(t.minute in (0, 30) for t in changes), [t.strftime("%a %H:%M") for t in changes]


def __test_change_anchor_data_starts_mid_session__(runner):
    """ timeframe.change("30") is on the session grid when the data starts mid-session """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    start = datetime(2026, 9, 21, 11, 10, tzinfo=ZoneInfo("America/Chicago"))
    changes = __test_helper_changes(runner, start)

    # The first partial session is anchored to its session start, not to the first bar
    first_session = [t.strftime("%H:%M") for t in changes if t.day == 21][:3]
    assert first_session == ["11:30", "12:00", "12:30"], first_session

    # The anchor moves to the next session open even though a virtual candle of the
    # break already lands on the previous grid
    next_session = [t.strftime("%H:%M") for t in changes if t.day == 22 and t.hour >= 17]
    assert next_session == ["17:00", "17:30", "18:00", "18:30"], next_session

    assert all(t.minute in (0, 30) for t in changes), [t.strftime("%a %H:%M") for t in changes]


def __test_helper_flat_bars(times):
    """ Flat bars opening at the given aware datetimes """
    from datetime import timezone
    from pynecore.types.ohlcv import OHLCV

    return [OHLCV(timestamp=int(t.astimezone(timezone.utc).timestamp() * 1000),
                  open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0) for t in times]


def __test_change_anchor_off_hour_session_open__(runner):
    """ timeframe.change("240") fires inside a 09:30 session on an hourly chart """
    from datetime import datetime, time, timedelta
    from zoneinfo import ZoneInfo
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    ny = ZoneInfo("America/New_York")
    days = [0, 1, 2, 3, 4]
    syminfo_override = dict(
        prefix="NASDAQ", ticker="AAPL", period="60", type="stock", timezone="America/New_York",
        opening_hours=[SymInfoInterval(d, time(9, 30), time(16)) for d in days],
        session_starts=[SymInfoSession(d, time(9, 30)) for d in days],
        session_ends=[SymInfoSession(d, time(16)) for d in days],
    )
    times = [datetime(2026, 9, day, 9, 30, tzinfo=ny) + timedelta(hours=h)
             for day in (21, 22) for h in range(7)]
    changes = [datetime.fromtimestamp(candle.timestamp / 1000, ny).strftime("%a %H:%M")
               for candle, _plot in runner(__test_helper_flat_bars(times), syminfo_override).run_iter()
               if _plot["chg240"]]
    assert changes == ["Mon 13:30", "Tue 09:30", "Tue 13:30"], changes


def __test_change_anchor_non_multiple_chart_timeframe__(runner):
    """ timeframe.change("60") fires on the first 45 minute bar of every hour """
    from datetime import datetime, time, timedelta, timezone
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    days = range(7)
    syminfo_override = dict(
        prefix="TEST", ticker="TEST", period="45", type="crypto", timezone="UTC",
        opening_hours=[SymInfoInterval(d, time(0), time(0)) for d in days],
        session_starts=[SymInfoSession(d, time(0)) for d in days],
        session_ends=[SymInfoSession(d, time(0)) for d in days],
    )
    times = [datetime(2026, 9, 21, tzinfo=timezone.utc) + timedelta(minutes=45 * i)
             for i in range(8)]
    changes = [datetime.fromtimestamp(candle.timestamp / 1000, timezone.utc).strftime("%H:%M")
               for candle, _plot in runner(__test_helper_flat_bars(times), syminfo_override).run_iter()
               if _plot["chg60"]]
    assert changes == ["01:30", "02:15", "03:00", "04:30", "05:15"], changes
