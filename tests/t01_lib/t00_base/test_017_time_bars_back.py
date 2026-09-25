"""
@pyne

``time()`` and ``time_close()`` with ``bars_back`` read the chart's own bar history.

A positive offset evaluates the call on the chart bar that many bars back, so a bar
missing from the data is skipped, and the offset is na until the chart has that many
bars behind it. A negative offset is a bar that has not opened yet: it follows the
session schedule, which knows nothing of the bars the feed will leave out. Measured
on TradingView (CAPITALCOM:EURUSD 60m): ``time("", 1)`` equals ``time[1]`` on every
bar, and ``time("", -1)`` names the scheduled next bar even when the data skips it.
"""
from pynecore.lib import script, plot, time, time_close


@script.indicator(title="time bars_back", shorttitle="tbb")
def main():
    plot(time(), "t")
    plot(time("", 1), "b1")
    plot(time("", 2), "b2")
    plot(time_close("", 1), "c1")
    plot(time("", -1), "f1")


__test_helper_HOUR_MS = 3_600_000


def __test_bars_back_reads_the_chart_bar_history__(runner, log):
    """ A gap in the data is stepped over backwards, but not forwards """
    from datetime import datetime, UTC, time as dt_time
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    hour = __test_helper_HOUR_MS
    override = {
        "period": "60",
        "type": "crypto",
        "timezone": "Etc/UTC",
        "opening_hours": [SymInfoInterval(day=d, start=dt_time(0, 0), end=dt_time(0, 0))
                          for d in range(7)],
        "session_starts": [SymInfoSession(day=d, time=dt_time(0, 0)) for d in range(7)],
        "session_ends": [SymInfoSession(day=d, time=dt_time(0, 0)) for d in range(7)],
    }

    start = int(datetime(2025, 1, 6, tzinfo=UTC).timestamp()) * 1000
    # The 03:00 bar is missing from the data
    bars = [OHLCV(timestamp=start + h * hour, open=1.0, high=1.0, low=1.0,
                  close=1.0, volume=1.0)
            for h in (0, 1, 2, 4, 5)]

    # ``run_iter`` reuses one plot dict per bar, so the values are copied inside the loop
    rows = [dict(plots) for _candle, plots in
            runner(iter(bars), syminfo_override=override).run_iter()]

    def at(h: int) -> float:
        return float(start + h * hour)

    def is_na(value: float) -> bool:
        return value != value

    assert [row["t"] for row in rows] == [at(0), at(1), at(2), at(4), at(5)]
    # 04:00 looks back to 02:00, the bar before it in the data
    assert is_na(rows[0]["b1"])
    assert [row["b1"] for row in rows[1:]] == [at(0), at(1), at(2), at(4)]
    assert is_na(rows[0]["b2"]) and is_na(rows[1]["b2"])
    assert [row["b2"] for row in rows[2:]] == [at(0), at(1), at(2)]
    assert [row["c1"] for row in rows[1:]] == [at(1), at(2), at(3), at(5)]
    # 02:00 looks ahead to the scheduled 03:00 bar, which the data never brings
    assert [row["f1"] for row in rows] == [at(1), at(2), at(3), at(5), at(6)]
