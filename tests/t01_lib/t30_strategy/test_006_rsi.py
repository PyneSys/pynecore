"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
from pynecore.lib import close, input, na, script, strategy, ta


@script.strategy("RSI Strategy", overlay=True)
def main(
        length=input(14, "Length"),
        overSold=input(30, "Oversold"),
        overBought=input(70, "Overbought")
):
    price = close
    vrsi = ta.rsi(price, length)
    co = ta.crossover(vrsi, overSold)
    cu = ta.crossunder(vrsi, overBought)
    if not na(vrsi):
        if co:
            strategy.entry('RsiLE', strategy.long, comment='RsiLE')
        if cu:
            strategy.entry('RsiSE', strategy.short, comment='RsiSE')


# noinspection PyShadowingNames
def __test_rsi__(csv_reader, runner, strat_equity_comparator):
    """ RSI Strategy """
    with csv_reader('ohlcv.csv', subdir="data") as cr, \
            csv_reader('rsi_trades.csv', subdir="data") as cr_equity:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern"))
        equity_iter = iter(cr_equity)
        for i, (candle, plot, new_closed_trades) in enumerate(r.run_iter()):
            for trade in new_closed_trades:
                good_entry = next(equity_iter)
                good_exit = next(equity_iter)
                strat_equity_comparator(trade, good_entry.extra_fields, good_exit.extra_fields)
