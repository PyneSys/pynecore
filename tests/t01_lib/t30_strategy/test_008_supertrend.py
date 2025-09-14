"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
from pynecore.lib import input, script, strategy, ta


@script.strategy("Supertrend Strategy", overlay=True, default_qty_type=strategy.percent_of_equity, default_qty_value=15)
def main(
        atrPeriod=input(10, "ATR Length"),
        factor=input.float(3.0, "Factor", step=0.01)
):
    _, direction = ta.supertrend(factor, atrPeriod)

    if ta.change(direction) < 0:
        strategy.entry('My Long Entry Id', strategy.long)

    if ta.change(direction) > 0:
        strategy.entry('My Short Entry Id', strategy.short)


# noinspection PyShadowingNames
def __test_supertrend__(csv_reader, runner, strat_equity_comparator):
    """ Supertrend Strategy """
    with csv_reader('ohlcv.csv', subdir="data") as cr, \
            csv_reader('supertrend_trades.csv', subdir="data") as cr_equity:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern"))
        equity_iter = iter(cr_equity)
        for i, (candle, plot, new_closed_trades) in enumerate(r.run_iter()):
            for trade in new_closed_trades:
                good_entry = next(equity_iter)
                good_exit = next(equity_iter)
                strat_equity_comparator(trade, good_entry.extra_fields, good_exit.extra_fields)
