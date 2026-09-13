"""
@pyne

A weekly script on a DAILY chart.

A D/W/M chart bar has no fixed arithmetic span, so the period-completion rule
cannot be "open + span reaches the period end". The chart bar's own close comes
from the trading schedule instead, which for a 24/7 symbol is its civil period
end -- so the Sunday bar closes the week it belongs to and the week is executed
there, not one bar late on the next Monday.
"""
from datetime import datetime, UTC

from pynecore.lib import script, close, high, low, open as open_price, bar_index, \
    last_bar_index, barstate, na
from pynecore.types.ohlcv import OHLCV

# 2024-01-01 is a Monday
__test_helper_start_ms = 1704067200000
__test_helper_day_ms = 86400000


@script.indicator("Script Timeframe DWM Chart", timeframe="1W", timeframe_gaps=True)
def main():
    return {
        "bar_index": bar_index,
        "last_bar_index": last_bar_index,
        "islast": 1.0 if barstate.islast else 0.0,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
    }


def __test_helper_daily_bars(count: int):
    out = []
    for i in range(count):
        p = 100.0 + i
        out.append(OHLCV(__test_helper_start_ms + i * __test_helper_day_ms,
                         p, p + 1.0, p - 1.0, p + 0.5, 2.0))
    return out


def __test_weekly_bars_close_on_the_sunday_bar__(runner):
    """28 daily bars = exactly 4 complete weeks, executed on their Sunday bars."""
    bars = __test_helper_daily_bars(28)
    r = runner(iter(bars), syminfo_override={"period": "1D"},
               last_bar_index=len(bars) - 1, last_bar_time=bars[-1].timestamp)
    rows = [(datetime.fromtimestamp(candle.timestamp / 1000, UTC), dict(plot))
            for candle, plot in r.run_iter()]

    executed = [(when, plot) for when, plot in rows if not na(plot["bar_index"])]
    assert len(executed) == 4
    # Every execution lands on a Sunday -- the last chart bar of its week.
    assert [when.weekday() for when, _ in executed] == [6, 6, 6, 6]
    assert [plot["bar_index"] for _, plot in executed] == [0.0, 1.0, 2.0, 3.0]
    assert all(plot["last_bar_index"] == 3.0 for _, plot in executed)
    # The fourth week completes inside the feed, so ``islast`` fires on it.
    assert [plot["islast"] for _, plot in executed] == [0.0, 0.0, 0.0, 1.0]

    # First week: 2024-01-01 .. 2024-01-07 (bars 0..6)
    first = executed[0][1]
    assert first["open"] == bars[0].open
    assert first["high"] == max(b.high for b in bars[:7])
    assert first["low"] == min(b.low for b in bars[:7])
    assert first["close"] == bars[6].close


def __test_open_week_is_na_and_never_anticipated__(runner):
    """A partial trailing week produces nothing; no value precedes its close."""
    bars = __test_helper_daily_bars(31)  # 4 full weeks + 3 days
    r = runner(iter(bars), syminfo_override={"period": "1D"},
               last_bar_index=len(bars) - 1, last_bar_time=bars[-1].timestamp)
    rows = [(datetime.fromtimestamp(candle.timestamp / 1000, UTC), dict(plot))
            for candle, plot in r.run_iter()]

    executed = [when for when, plot in rows if not na(plot["bar_index"])]
    assert len(executed) == 4
    assert executed[-1] == datetime(2024, 1, 28, tzinfo=UTC)
    # Everything after the last Sunday stays na -- the fifth week is still open.
    tail = [plot for when, plot in rows if when > datetime(2024, 1, 28, tzinfo=UTC)]
    assert len(tail) == 3
    assert all(na(plot["close"]) for plot in tail)
