"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
from pynecore.lib import close, input, script, strategy, ta


@script.strategy("Bollinger Bands Strategy", overlay=True)
def main(
    length=input.int(20, minval=1),
    mult=input.float(2.0, minval=0.001, maxval=50)
):
    source = close
    basis = ta.sma(source, length)
    dev = mult * ta.stdev(source, length)
    upper = basis + dev
    lower = basis - dev
    buyEntry = ta.crossover(source, lower)
    sellEntry = ta.crossunder(source, upper)
    if ta.crossover(source, lower):
        strategy.entry('BBandLE', strategy.long, stop=lower, oca_name='BollingerBands', oca_type=strategy.oca.cancel, comment='BBandLE')
    else:
        strategy.cancel(id='BBandLE')
    if ta.crossunder(source, upper):
        strategy.entry('BBandSE', strategy.short, stop=upper, oca_name='BollingerBands', oca_type=strategy.oca.cancel, comment='BBandSE')
    else:
        strategy.cancel(id='BBandSE')


# noinspection PyShadowingNames
def __test_bollinger__(csv_reader, runner, strat_equity_comparator):
    """ Bollinger Bands Strategy """
    with csv_reader('ohlcv.csv', subdir="data") as cr, \
            csv_reader('bollinger_trades.csv', subdir="data") as cr_equity:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern"))
        with strat_equity_comparator(cr_equity) as reference:
            for candle, plot, new_closed_trades in r.run_iter():
                for trade in new_closed_trades:
                    reference.compare(trade)
