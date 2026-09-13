"""
@pyne

Script-level higher timeframe with ``timeframe_gaps=False``.

Chart bars inside an open higher-timeframe period repeat the last CONFIRMED
higher-timeframe value instead of plotting ``na``. ``last_bar_index`` /
``barstate.islast`` describe the HTF series, not the chart's.
"""
from datetime import datetime, UTC

from pynecore.lib import script, close, bar_index, last_bar_index, barstate
from pynecore.types.ohlcv import OHLCV

#: Hourly chart bars, UTC, starting 2024-01-01 00:00. Three full days plus five
#: hours of a fourth (incomplete) day.
__test_helper_start_ms = 1704067200000
__test_helper_hour_ms = 3600000
__test_helper_bars = 24 * 3 + 5


@script.indicator("Script Timeframe No Gaps", timeframe="1D", timeframe_gaps=False)
def main():
    return {
        "bar_index": bar_index,
        "last_bar_index": last_bar_index,
        "islast": 1.0 if barstate.islast else 0.0,
        "close": close,
    }


def __test_helper_run(runner):
    """Run the script and return ``[(utc datetime, plot dict)]``."""
    bars = []
    for i in range(__test_helper_bars):
        p = 100.0 + i
        bars.append(OHLCV(__test_helper_start_ms + i * __test_helper_hour_ms,
                          p, p + 0.5, p - 0.5, p + 0.25, 2.0))
    r = runner(iter(bars), syminfo_override={"period": "60"},
               last_bar_index=len(bars) - 1, last_bar_time=bars[-1].timestamp)
    out = []
    for candle, plot in r.run_iter():
        out.append((datetime.fromtimestamp(candle.timestamp / 1000, UTC), dict(plot)))
    return out


def __test_no_gaps_forward_fills_the_last_confirmed_value__(runner):
    """Every chart bar carries the last COMPLETED day's value."""
    rows = __test_helper_run(runner)

    day_closes = [100.0 + 23 + 0.25, 100.0 + 47 + 0.25, 100.0 + 71 + 0.25]
    for dt, plot in rows:
        # Days close on the 23:00 hourly bar, so a bar at or after day N's 23:00
        # and before day N+1's shows day N's value.
        day = (dt - datetime(2024, 1, 1, 23, 0, tzinfo=UTC)).days
        assert plot["close"] == day_closes[day], dt.isoformat()
        assert plot["bar_index"] == float(day)

    # No na anywhere: the first row is the first execution itself.
    assert rows[0][0] == datetime(2024, 1, 1, 23, 0, tzinfo=UTC)
    assert len(rows) == __test_helper_bars - 23


def __test_no_gaps_never_shows_a_value_before_its_period_closed__(runner):
    """Forward fill repeats the PAST, it never anticipates the open period."""
    rows = __test_helper_run(runner)
    for dt, plot in rows:
        hour = int((dt.timestamp() * 1000 - __test_helper_start_ms)
                   // __test_helper_hour_ms)
        assert plot["close"] <= 100.25 + hour, dt.isoformat()


def __test_last_bar_index_counts_completed_htf_bars__(runner):
    """The script's last bar is the last day that COMPLETES inside the feed."""
    rows = __test_helper_run(runner)

    # Three completed days -> indices 0..2, whatever the chart's own bar count is.
    assert all(plot["last_bar_index"] == 2.0 for _, plot in rows)

    lasts = [dt for dt, plot in rows if plot["islast"] == 1.0]
    assert lasts[0] == datetime(2024, 1, 3, 23, 0, tzinfo=UTC), \
        "islast fires on the last COMPLETED day, not on the chart's last bar"
