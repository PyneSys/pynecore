"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
from pynecore.lib import close, input, open, script, strategy


@script.strategy("BarUpDn Strategy", overlay=True, default_qty_type=strategy.percent_of_equity, default_qty_value=10)
def main(
    maxIdLossPcnt=input.float(1, "Max intraday loss (%)")
):
    strategy.risk.max_intraday_loss(maxIdLossPcnt, strategy.percent_of_equity)
    if close > open and open > close[1]:
        strategy.entry('BarUp', strategy.long)
    if close < open and open < close[1]:
        strategy.entry('BarDn', strategy.short)


# noinspection PyShadowingNames
def __test_barupdn__(csv_reader, runner, strat_equity_comparator):
    """ BarUpDn Strategy """
    with csv_reader('ohlcv.csv', subdir="data") as cr, \
            csv_reader('barupdn_trades.csv', subdir="data") as cr_equity:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern"))
        equity_iter = iter(cr_equity)
        for i, (candle, plot, new_closed_trades) in enumerate(r.run_iter()):
            for trade in new_closed_trades:
                good_entry = next(equity_iter)
                good_exit = next(equity_iter)
                strat_equity_comparator(trade, good_entry.extra_fields, good_exit.extra_fields)