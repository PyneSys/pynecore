"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
from pynecore.lib import close, display, high, input, low, nz, script, strategy, syminfo, ta
from pynecore.types import Series


@script.strategy(title="Keltner Channels Strategy", overlay=True)
def main(
        length=input.int(20, minval=1),
        mult=input.float(2.0, "Multiplier"),
        src: Series[float] = input(close, title="Source"),
        exp=input(True, "Use Exponential MA", display=display.data_window),
        BandsStyle=input.string("Average True Range", options=("Average True Range", "True Range", "Range"),
                                title="Bands Style", display=display.data_window),
        atrlength=input(10, "ATR Length", display=display.data_window)
):
    def esma(source, length):
        s = ta.sma(source, length)
        e = ta.ema(source, length)
        return e if exp else s

    ma = esma(src, length)
    rangema = ta.tr(True) if BandsStyle == 'True Range' else ta.atr(
        atrlength) if BandsStyle == 'Average True Range' else ta.rma(high - low, length)
    upper = ma + rangema * mult
    lower = ma - rangema * mult
    crossUpper = ta.crossover(src, upper)
    crossLower = ta.crossunder(src, lower)
    bprice: Series[float] = 0.0
    bprice = high + syminfo.mintick if crossUpper else nz(bprice[1])
    sprice: Series[float] = 0.0
    sprice = low - syminfo.mintick if crossLower else nz(sprice[1])
    crossBcond: Series[bool] = False
    crossBcond = True if crossUpper else crossBcond[1]
    crossScond: Series[bool] = False
    crossScond = True if crossLower else crossScond[1]
    cancelBcond = crossBcond and (src < ma or high >= bprice)
    cancelScond = crossScond and (src > ma or low <= sprice)
    if cancelBcond:
        strategy.cancel('KltChLE')
    if crossUpper:
        strategy.entry('KltChLE', strategy.long, stop=bprice, comment='KltChLE')
    if cancelScond:
        strategy.cancel('KltChSE')
    if crossLower:
        strategy.entry('KltChSE', strategy.short, stop=sprice, comment='KltChSE')


# noinspection PyShadowingNames
def __test_keltnerchannel__(csv_reader, runner, strat_equity_comparator):
    """ Keltner Channels Strategy """
    with csv_reader('ohlcv.csv', subdir="data") as cr, \
            csv_reader('keltnerchannel_trades.csv', subdir="data") as cr_equity:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern"))
        equity_iter = iter(cr_equity)
        for i, (candle, plot, new_closed_trades) in enumerate(r.run_iter()):
            for trade in new_closed_trades:
                good_entry = next(equity_iter)
                good_exit = next(equity_iter)
                strat_equity_comparator(trade, good_entry.extra_fields, good_exit.extra_fields)
