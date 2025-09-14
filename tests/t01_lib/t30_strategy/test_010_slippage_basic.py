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
def __test_slippage_basic__(csv_reader, runner):
    """ Slippage Test - Basic Market Orders """
    with csv_reader('ohlcv.csv', subdir="data") as cr:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern"))
        for i, result in enumerate(r.run_iter()):
            if len(result) == 3:  # Strategy
                candle, plot, new_closed_trades = result
            else:  # Indicator
                candle, plot = result
            # Basic validation test - just ensure it runs without errors
            pass