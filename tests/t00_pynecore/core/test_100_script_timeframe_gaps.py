"""
@pyne

Script-level higher timeframe with ``timeframe_gaps=True`` (Pine's default).

The body runs once per COMPLETED higher-timeframe bar, on the chart bar whose
close instant reaches the period end, and every other chart bar plots ``na``.
Series state (``bar_index``, ``[1]`` history, ``ta.*``) advances per HTF bar.
"""
from datetime import datetime, UTC

from pynecore import Series
from pynecore.lib import script, close, high, low, open, volume, bar_index, timeframe, ta
from pynecore.types.na import isna_num
from pynecore.types.ohlcv import OHLCV

#: Hourly chart bars, UTC, starting 2024-01-01 00:00. Three full days plus five
#: hours of a fourth (incomplete) day.
__test_helper_start_ms = 1704067200000
__test_helper_hour_ms = 3600000
__test_helper_bars = 24 * 3 + 5


@script.indicator("Script Timeframe Gaps", timeframe="1D", timeframe_gaps=True)
def main():
    c: Series[float] = close
    return {
        "bar_index": bar_index,
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "prev_close": c[1],
        "sma2": ta.sma(close, 2),
        "period": timeframe.period,
        "main_period": timeframe.main_period,
        "multiplier": timeframe.multiplier,
        "isdaily": 1.0 if timeframe.isdaily else 0.0,
        "isintraday": 1.0 if timeframe.isintraday else 0.0,
    }


def __test_helper_bars_iter():
    """Rising hourly candles: close of hour ``i`` is ``100.25 + i``."""
    out = []
    for i in range(__test_helper_bars):
        p = 100.0 + i
        out.append(OHLCV(__test_helper_start_ms + i * __test_helper_hour_ms,
                         p, p + 0.5, p - 0.5, p + 0.25, 2.0))
    return out


def __test_helper_run(runner):
    """Run the script and return ``[(utc datetime, plot dict)]``."""
    bars = __test_helper_bars_iter()
    # The shared syminfo fixture is already a 24/7 UTC crypto symbol; only the
    # chart timeframe has to become hourly.
    r = runner(iter(bars), syminfo_override={"period": "60"},
               last_bar_index=len(bars) - 1, last_bar_time=bars[-1].timestamp)
    out = []
    for candle, plot in r.run_iter():
        out.append((datetime.fromtimestamp(candle.timestamp / 1000, UTC), dict(plot)))
    return out


def __test_script_timeframe_runs_once_per_htf_bar__(runner):
    """One execution per completed day, on that day's LAST hourly bar."""
    rows = __test_helper_run(runner)

    # One output row per chart bar from the first execution on; nothing before it
    # (no plot column exists yet, exactly as TradingView drops its leading all-na
    # rows).
    assert rows[0][0] == datetime(2024, 1, 1, 23, 0, tzinfo=UTC)
    assert len(rows) == __test_helper_bars - 23

    executed = [(dt, p) for dt, p in rows if not isna_num(p["bar_index"])]
    assert [dt.isoformat() for dt, _ in executed] == [
        "2024-01-01T23:00:00+00:00",
        "2024-01-02T23:00:00+00:00",
        "2024-01-03T23:00:00+00:00",
    ], "the day's value lands on the chart bar that closes it"

    # The still-developing 4th day produces nothing.
    assert executed[-1][0] < datetime(2024, 1, 4, tzinfo=UTC)

    # bar_index counts HTF bars, not chart bars.
    assert [p["bar_index"] for _, p in executed] == [0.0, 1.0, 2.0]


def __test_script_timeframe_sees_aggregated_htf_ohlcv__(runner):
    """open/high/low/close/volume are the DAY's, built from the hourly bars."""
    rows = __test_helper_run(runner)
    executed = [p for _, p in rows if not isna_num(p["bar_index"])]

    day0 = executed[0]
    assert day0["open"] == 100.0                  # first hour's open
    assert day0["high"] == 100.0 + 23 + 0.5       # highest hourly high
    assert day0["low"] == 100.0 - 0.5             # lowest hourly low
    assert day0["close"] == 100.0 + 23 + 0.25     # last hour's close
    assert day0["volume"] == 24 * 2.0             # summed hourly volume

    day1 = executed[1]
    assert day1["open"] == 124.0
    assert day1["close"] == 100.0 + 47 + 0.25


def __test_script_timeframe_series_advance_per_htf_bar__(runner):
    """``[1]`` and ``ta.*`` step one HTF bar, not one chart bar."""
    rows = __test_helper_run(runner)
    executed = [p for _, p in rows if not isna_num(p["bar_index"])]

    assert isna_num(executed[0]["prev_close"])
    assert executed[1]["prev_close"] == executed[0]["close"]
    assert executed[2]["prev_close"] == executed[1]["close"]

    assert isna_num(executed[0]["sma2"])
    assert executed[1]["sma2"] == (executed[0]["close"] + executed[1]["close"]) / 2
    assert executed[2]["sma2"] == (executed[1]["close"] + executed[2]["close"]) / 2


def __test_script_timeframe_gaps_are_na_and_carry_no_lookahead__(runner):
    """Chart bars inside an open day plot ``na`` -- never the day's future value."""
    rows = __test_helper_run(runner)

    gaps = [(dt, p) for dt, p in rows if isna_num(p["bar_index"])]
    assert gaps, "there must be gap bars between the daily executions"
    for dt, plot in gaps:
        for key, value in plot.items():
            assert isna_num(value), f"{dt.isoformat()} {key} is not na"

    # No lookahead: on every chart bar the last value seen is the one of a day
    # that has already CLOSED.
    last_close = None
    for dt, plot in rows:
        if not isna_num(plot["bar_index"]):
            last_close = plot["close"]
            # The executing bar is the day's last hour: the day is over.
            assert dt.hour == 23
        if last_close is not None:
            # 100.25 + i is the close of hour i; a day's close can only be known
            # once that day's last hour has been seen.
            hour = int((dt.timestamp() * 1000 - __test_helper_start_ms)
                       // __test_helper_hour_ms)
            assert last_close <= 100.25 + hour


def __test_timeframe_builtins_report_the_script_timeframe__(runner):
    """Every ``timeframe.*`` builtin describes the timeframe the script runs on.

    MEASURED on TradingView (CAPITALCOM:GOLD@60 with ``timeframe='W'``):
    ``timeframe.period`` and ``timeframe.main_period`` both report 604800s (weekly),
    ``timeframe.isweekly`` is true, ``timeframe.isintraday`` false and
    ``timeframe.multiplier`` 1 -- none of them the chart's own 60 minutes.
    """
    rows = __test_helper_run(runner)
    executed = [p for _, p in rows if not isna_num(p["bar_index"])]
    assert [p["period"] for p in executed] == ["1D", "1D", "1D"]
    assert [p["main_period"] for p in executed] == ["1D", "1D", "1D"]
    assert all(p["multiplier"] == 1.0 for p in executed)
    assert all(p["isdaily"] == 1.0 for p in executed)
    assert all(p["isintraday"] == 0.0 for p in executed)
