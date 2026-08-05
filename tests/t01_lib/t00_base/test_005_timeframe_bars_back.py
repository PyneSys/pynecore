"""
@pyne

Regression: ``time`` and ``time_close`` take Pine's fifth parameter,
``timeframe_bars_back``.

A compiled Pine script emits Pine's own keyword for a named argument, so a missing
parameter raises ``TypeError`` and halts the script. Unlike ``bars_back``, which
steps on the CHART's timeframe, ``timeframe_bars_back`` steps on the REQUESTED
timeframe -- measured on TradingView (CAPITALCOM:EURUSD 60m): with
``timeframe="240"`` one step is 4 hours, while a ``bars_back=1`` step of one chart
hour only moves the 4-hour bar when it crosses a boundary.
"""
from pynecore.lib import script, plot, time, time_close

_HOUR_MS = 3_600_000
_DAY_MS = 86_400_000


@script.indicator(title="timeframe_bars_back", shorttitle="tbb")
def main():
    plot(time(timeframe="240"), "t0")
    plot(time(timeframe="240", timeframe_bars_back=1), "t1")
    plot(time(timeframe="240", timeframe_bars_back=2), "t2")
    plot(time_close(timeframe="240"), "c0")
    plot(time_close(timeframe="240", timeframe_bars_back=1), "c1")
    plot(time(timeframe="1M"), "m0")
    plot(time(timeframe="1M", timeframe_bars_back=1), "m1")
    plot(time(timeframe="1M", timeframe_bars_back=2), "m2")


def __test_timeframe_bars_back_steps_the_requested_timeframe__(runner, log):
    """ One ``timeframe_bars_back`` step on "240" moves the bar back by 4 hours """
    from datetime import datetime, UTC, time as dt_time
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    # A true 24/7 template: no trading-day end to cap ``time_close`` against
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
    bars = [OHLCV(timestamp=start + i * _HOUR_MS, open=1.0, high=1.0, low=1.0,
                  close=1.0, volume=1.0)
            for i in range(24)]

    # ``run_iter`` reuses one plot dict per bar, so the values must be read inside
    # the loop, not collected as dict references
    steps: list[tuple[int, int, int]] = []
    for _candle, plots in runner(iter(bars), syminfo_override=override).run_iter():
        steps.append((plots["t0"] - plots["t1"], plots["t0"] - plots["t2"],
                      plots["c0"] - plots["c1"]))

    assert len(steps) == 24
    for one_step, two_steps, close_step in steps:
        assert one_step == 4 * _HOUR_MS
        assert two_steps == 8 * _HOUR_MS
        assert close_step == 4 * _HOUR_MS


def __test_monthly_timeframe_bars_back_walks_calendar_months__(runner, log):
    """ A monthly step lands on the previous month's first day, whatever its length """
    from datetime import datetime, UTC, time as dt_time
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    override = {
        "period": "1D",
        "type": "crypto",
        "timezone": "Etc/UTC",
        "opening_hours": [SymInfoInterval(day=d, start=dt_time(0, 0), end=dt_time(0, 0))
                          for d in range(7)],
        "session_starts": [SymInfoSession(day=d, time=dt_time(0, 0)) for d in range(7)],
        "session_ends": [SymInfoSession(day=d, time=dt_time(0, 0)) for d in range(7)],
    }

    # Across a 28-day February and a 30-day April, the two spans a nominal month length
    # (30.4375 days) overshoots on the first days of March and of May
    start = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp()) * 1000
    bars = [OHLCV(timestamp=start + i * _DAY_MS, open=1.0, high=1.0, low=1.0,
                  close=1.0, volume=1.0)
            for i in range(181)]

    def month_start(year: int, month: int) -> int:
        return int(datetime(year, month, 1, tzinfo=UTC).timestamp()) * 1000

    seen: list[tuple[int, int, int]] = []
    for _candle, plots in runner(iter(bars), syminfo_override=override).run_iter():
        seen.append((plots["m0"], plots["m1"], plots["m2"]))

    assert len(seen) == 181
    for i, (this_month, one_back, two_back) in enumerate(seen):
        bar_date = datetime.fromtimestamp((start + i * _DAY_MS) / 1000, UTC)
        month = bar_date.month
        prev = (2026, month - 1) if month > 1 else (2025, 12)
        prev2 = (2026, month - 2) if month > 2 else (2025, 12 + month - 2)
        assert this_month == month_start(2026, month)
        assert one_back == month_start(*prev)
        assert two_back == month_start(*prev2)
