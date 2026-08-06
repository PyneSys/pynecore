"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
from pynecore.lib import bar_index, bgcolor, color, hline, input, na, plot, script, strategy, string

SLIPPAGE_TICKS: int = 15


@script.strategy("Slippage Test - Direction Change", overlay=True, initial_capital=10000, slippage=SLIPPAGE_TICKS, commission_type=strategy.commission.percent, commission_value=0.1)
def main(
    flipInterval=input.int(10, "Bars Between Direction Changes", minval=5),
    startBar=input.int(20, "Start Trading at Bar", minval=1)
):

    barsSinceStart = bar_index - startBar
    shouldTrade = bar_index >= startBar and barsSinceStart % flipInterval == 0
    tradeNumber = barsSinceStart / flipInterval

    isLongTrade = tradeNumber % 2 == 0

    if shouldTrade:
        if isLongTrade:
            strategy.entry('Long', strategy.long, comment='Flip to Long #' + string.tostring(tradeNumber))
        else:
            strategy.entry('Short', strategy.short, comment='Flip to Short #' + string.tostring(tradeNumber))

    bgcolor(color.new(color.green, 90) if shouldTrade and isLongTrade else na, title='Long Entry')
    bgcolor(color.new(color.red, 90) if shouldTrade and (not isLongTrade) else na, title='Short Entry')

    plot(strategy.position_size, 'Position Size', color.purple, 2, plot.style_stepline)
    hline(0, 'Zero Line', color.gray, hline.style_dashed)


# noinspection PyShadowingNames
def __test_slippage_direction__(csv_reader, runner, strat_equity_comparator):
    """ Slippage Test - Direction Change """
    with csv_reader('slippage_direction_ohlcv.csv', subdir="data") as cr, \
            csv_reader('slippage_direction_trades.csv', subdir="data") as cr_equity:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern", type="forex"))
        with strat_equity_comparator(cr_equity) as reference:
            for candle, plot, new_closed_trades in r.run_iter():
                for trade in new_closed_trades:
                    reference.compare(trade)
