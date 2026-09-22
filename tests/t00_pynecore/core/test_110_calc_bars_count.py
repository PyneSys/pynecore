"""
@pyne
"""
from pynecore.lib import bar_index, last_bar_index, plot, script, ta


@script.indicator(title="Calc Bars Count", shorttitle="CBC", calc_bars_count=10)
def main():
    plot(bar_index, "bi")
    plot(last_bar_index, "lbi")
    plot(ta.highest(4), "hi")


def __test_helper_chart_bars():
    """Thirty rising hourly bars."""
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(30):
        c = 100.0 + i
        bars.append(OHLCV(timestamp=1_735_689_600_000 + i * 3_600_000,
                          open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1.0))
    return bars


def __test_history_begins_at_the_first_calculated_bar__(runner, log):
    """The script sees the last ``calc_bars_count`` bars as its whole history

    MEASURED on TradingView (BINANCE:BTCUSDT@30, a 30162-bar chart with
    ``calc_bars_count`` 500 and 1500): bar_index is 0 on the first calculated
    bar, last_bar_index is N - 1, and a windowed builtin warms up from there.
    """
    import os
    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    bars = __test_helper_chart_bars()
    rows = []
    for candle, plots in runner(bars, last_bar_index=len(bars) - 1).run_iter():
        rows.append((candle.timestamp, dict(plots)))

    assert [ts for ts, _ in rows] == [bar.timestamp for bar in bars[-10:]]
    assert [row["bi"] for _, row in rows] == [float(i) for i in range(10)]
    assert {row["lbi"] for _, row in rows} == {9.0}
    highs = [row["hi"] for _, row in rows]
    assert all(h != h for h in highs[:3])
    assert highs[3:] == [bar.high for bar in bars[-7:]]
    log.info("ten bars calculated, indexed from zero, ta.highest(4) warm on the fourth")
