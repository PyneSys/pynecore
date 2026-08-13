"""
@pyne
"""
from pynecore.lib import plot, script, timeframe


@script.indicator(title="Timeframe Change Empty", shorttitle="tf_change_empty")
def main():
    # An empty timeframe string is the chart's own timeframe, so these two agree
    # on every bar (measured against TradingView over a full history)
    plot(1 if timeframe.change("") else 0, "empty")
    plot(1 if timeframe.change(timeframe.period) else 0, "period")


def __test_change_empty_timeframe_is_chart_timeframe__(csv_reader, runner, log):
    """ timeframe.change("") equals timeframe.change(timeframe.period) """
    from pathlib import Path

    syminfo_path = Path(__file__).parent / "data" / "timeframe.toml"
    bars = 0
    changes = 0
    with csv_reader('timeframe.csv', subdir="data") as cr:
        for i, (_candle, _plot) in enumerate(runner(cr, syminfo_path=syminfo_path).run_iter()):
            bars += 1
            changes += 1 if _plot['empty'] else 0
            assert _plot['empty'] == _plot['period'], \
                f"bar {i}: empty={_plot['empty']} != period={_plot['period']}"

    assert bars > 0, "no bars were run"
    log.info("Empty timeframe matched the chart timeframe on %d bars (%d changes)",
             bars, changes)
