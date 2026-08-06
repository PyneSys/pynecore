"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
from pynecore.lib import close, input, script, strategy, ta


@script.strategy("MACD Strategy", overlay=True)
def main(
        fastLength=input(12, "Fast length"),
        slowlength=input(26, "Slow length"),
        MACDLength=input(9, "MACD length")
):
    MACD = ta.ema(close, fastLength) - ta.ema(close, slowlength)
    aMACD = ta.ema(MACD, MACDLength)
    delta = MACD - aMACD
    if ta.crossover(delta, 0):
        strategy.entry('MacdLE', strategy.long, comment='MacdLE')
    if ta.crossunder(delta, 0):
        strategy.entry('MacdSE', strategy.short, comment='MacdSE')


# noinspection PyShadowingNames
def __test_macd__(csv_reader, runner, strat_equity_comparator):
    """ MACD Strategy """
    with csv_reader('ohlcv.csv', subdir="data") as cr, \
            csv_reader('macd_trades.csv', subdir="data") as cr_equity:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern"))
        with strat_equity_comparator(cr_equity) as reference:
            for candle, plot, new_closed_trades in r.run_iter():
                for trade in new_closed_trades:
                    reference.compare(trade)
