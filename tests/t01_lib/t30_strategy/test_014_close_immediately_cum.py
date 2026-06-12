"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
import pytest

from pynecore.lib import bar_index, script, strategy


@script.strategy("Close Immediately Cumulative", overlay=True)
def main():
    if bar_index % 20 == 5:
        strategy.entry('L', strategy.long)
    if bar_index % 20 == 15 and strategy.opentrades > 0:
        strategy.close_all('End of Session', immediately=True)


# noinspection PyShadowingNames
def __test_close_immediately_cum__(csv_reader, runner):
    """ Same-tick (immediately=True) closes must book cumulative stats.

    The regular settle runs inside process_orders(); a strategy.close_all
    (immediately=True) fill lands in new_closed_trades after it, and without
    its own settle the next bar's clear() dropped it with cum_profit never
    booked — TradingView chains these trades into the cumulative like any
    other close (verified on a TV trade export). """
    with csv_reader('ohlcv.csv', subdir="data") as cr:
        r = runner(cr)
        total = 0.0
        closed = 0
        for candle, plot, new_closed_trades in r.run_iter():
            for trade in new_closed_trades:
                total += trade.profit
                assert trade.cum_profit == pytest.approx(total)
                closed += 1
        assert closed >= 2  # the data must actually produce same-tick closes
