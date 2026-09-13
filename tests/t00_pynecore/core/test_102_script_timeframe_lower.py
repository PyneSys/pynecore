"""
@pyne

A script timeframe LOWER than the chart's is refused; an EQUAL one is a plain run.

A finer script timeframe is a real TradingView feature -- probed on
CAPITALCOM:GOLD@60 with ``timeframe="15"``: the body runs on the 15-minute grid
(``bar_index`` steps 3, 7, ... 81545, i.e. four executions per chart bar),
``timeframe.period`` answers 900 seconds and the chart bar carries the LAST
intrabar execution. It needs an intrabar feed the runner does not have, so
PyneCore refuses it instead of silently running on the chart bars. A timeframe
equal to the chart's is simply the chart context, with no higher-timeframe
machinery at all.
"""
from datetime import datetime, UTC

import pytest

from pynecore.lib import script, close, bar_index, timeframe
from pynecore.types.ohlcv import OHLCV

__test_helper_start_ms = 1704067200000
__test_helper_minute_ms = 60000


@script.indicator("Script Timeframe Lower", timeframe="5")
def main():
    return {
        "bar_index": bar_index,
        "close": close,
        "period": timeframe.period,
    }


def __test_helper_bars(span_ms: int, count: int):
    out = []
    for i in range(count):
        p = 100.0 + i
        out.append(OHLCV(__test_helper_start_ms + i * span_ms, p, p + 0.5, p - 0.5,
                         p + 0.25, 2.0))
    return out


def __test_lower_script_timeframe_is_refused__(runner):
    """A 5-minute script on a 15-minute chart has no feed to run on."""
    bars = __test_helper_bars(15 * __test_helper_minute_ms, 10)
    with pytest.raises(ValueError) as exc:
        runner(iter(bars), syminfo_override={"period": "15"},
               last_bar_index=len(bars) - 1, last_bar_time=bars[-1].timestamp)
    message = str(exc.value)
    assert "'5'" in message and "'15'" in message
    assert "lower than the chart timeframe" in message
    assert "not supported" in message


def __test_equal_script_timeframe_runs_on_chart_bars__(runner):
    """Same timeframe as the chart: one execution per chart bar, no gaps."""
    bars = __test_helper_bars(5 * __test_helper_minute_ms, 6)
    r = runner(iter(bars), syminfo_override={"period": "5"},
               last_bar_index=len(bars) - 1, last_bar_time=bars[-1].timestamp)
    rows = [(datetime.fromtimestamp(candle.timestamp / 1000, UTC), dict(plot))
            for candle, plot in r.run_iter()]

    assert len(rows) == len(bars)
    assert [plot["bar_index"] for _, plot in rows] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert [plot["close"] for _, plot in rows] == [b.close for b in bars]
    # ``timeframe.period`` still reports the declared timeframe -- which here IS
    # the chart's.
    assert all(plot["period"] == "5" for _, plot in rows)
