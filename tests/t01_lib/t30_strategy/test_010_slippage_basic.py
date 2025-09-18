"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
from pynecore.lib import bar_index, bgcolor, color, input, na, script, strategy

SLIPPAGE_TICKS: int = 10


@script.strategy("Slippage Test - Basic Market Orders", overlay=True, initial_capital=10000, slippage=SLIPPAGE_TICKS, commission_type=strategy.commission.percent, commission_value=0.1)
def main(
    testMode=input.string("Both", "Test Mode", options=("Long Only", "Short Only", "Both")),
    entryBar=input.int(10, "Entry at Bar", minval=1),
    exitBar=input.int(20, "Exit at Bar", minval=1)
):

    if bar_index == entryBar:
        if testMode == 'Long Only' or testMode == 'Both':
            strategy.entry('Long', strategy.long, comment='Long Entry')
        elif testMode == 'Short Only':
            strategy.entry('Short', strategy.short, comment='Short Entry')

    if bar_index == exitBar:
        if testMode == 'Long Only' or testMode == 'Both':
            strategy.close('Long', comment='Long Exit')
        elif testMode == 'Short Only':
            strategy.close('Short', comment='Short Exit')

    if testMode == 'Both' and bar_index == entryBar + 30:
        strategy.entry('Short', strategy.short, comment='Short Entry')

    if testMode == 'Both' and bar_index == exitBar + 30:
        strategy.close('Short', comment='Short Exit')

    bgcolor(color.new(color.green, 90) if bar_index == entryBar else na, title='Entry Bar')
    bgcolor(color.new(color.red, 90) if bar_index == exitBar else na, title='Exit Bar')
    bgcolor(color.new(color.orange, 90) if bar_index == entryBar + 30 and testMode == 'Both' else na, title='Second Entry')
    bgcolor(color.new(color.purple, 90) if bar_index == exitBar + 30 and testMode == 'Both' else na, title='Second Exit')


# noinspection PyShadowingNames
def __test_slippage_basic__(csv_reader, runner, strat_equity_comparator):
    """ Slippage Test - Basic Market Orders """
    with csv_reader('slippage_basic_ohlcv.csv', subdir="data") as cr, \
            csv_reader('slippage_basic_trades.csv', subdir="data") as cr_equity:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern", type="forex"))
        equity_iter = iter(cr_equity)
        for i, (candle, plot, new_closed_trades) in enumerate(r.run_iter()):
            for trade in new_closed_trades:
                good_entry = next(equity_iter)
                good_exit = next(equity_iter)
                strat_equity_comparator(trade, good_entry.extra_fields, good_exit.extra_fields)