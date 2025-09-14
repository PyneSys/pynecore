"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
from pynecore.lib import (
    bar_index, barstate, bgcolor, color, input, na, plot, position, script,
    strategy, string, table
)
from pynecore.types import Persistent, Table

SLIPPAGE_TICKS: int = 5
PYRAMIDING_LIMIT: int = 3


@script.strategy("Slippage Test - Pyramiding", overlay=True, initial_capital=10000, pyramiding=PYRAMIDING_LIMIT, slippage=SLIPPAGE_TICKS, commission_type=strategy.commission.percent, commission_value=0.1, default_qty_type=strategy.percent_of_equity, default_qty_value=10)
def main(
    entrySpacing=input.int(5, "Bars Between Entries", minval=1),
    startBar=input.int(20, "First Entry Bar", minval=1)
):

    if bar_index >= startBar and bar_index <= startBar + entrySpacing * 2:
        if (bar_index - startBar) % entrySpacing == 0:
            entryNum = (bar_index - startBar) / entrySpacing + 1
            strategy.entry('Long ' + string.tostring(entryNum), strategy.long, comment='Entry #' + string.tostring(entryNum))

    if bar_index == startBar + entrySpacing * 3 + 5:
        strategy.close_all(comment='Close All Positions')

    isEntryBar = bar_index >= startBar and bar_index <= startBar + entrySpacing * 2 and ((bar_index - startBar) % entrySpacing == 0)
    isExitBar = bar_index == startBar + entrySpacing * 3 + 5

    bgcolor(color.new(color.green, 90) if isEntryBar else na, title='Entry Bar')
    bgcolor(color.new(color.red, 90) if isExitBar else na, title='Exit Bar')

    plot(strategy.position_avg_price, 'Avg Entry Price', color.orange, 2, plot.style_stepline)

    infoTable: Persistent[Table] = table.new(position.top_right, 2, 9, border_width=1)

    if barstate.islast:
        table.cell(infoTable, 0, 0, 'Slippage:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 0, string.tostring(SLIPPAGE_TICKS) + ' ticks', bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 1, 'Pyramiding:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 1, string.tostring(PYRAMIDING_LIMIT), bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 2, 'Open Trades:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 2, string.tostring(strategy.opentrades), bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 3, 'Position Size:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 3, string.tostring(strategy.position_size), bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 4, 'Avg Entry:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 4, string.tostring(strategy.position_avg_price, '#.####'), bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 5, 'Total Equity:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 5, string.tostring(strategy.equity, '#.##'), bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 6, 'Open P&L:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 6, string.tostring(strategy.openprofit, '#.##'), bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 7, 'Net Profit:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 7, string.tostring(strategy.netprofit, '#.##'), bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 8, 'Note:', bgcolor=color.yellow, text_color=color.black)
        table.cell(infoTable, 1, 8, 'Each entry has slippage', bgcolor=color.yellow, text_color=color.black)


# noinspection PyShadowingNames
def __test_slippage_pyramiding__(csv_reader, runner):
    """ Slippage Test - Pyramiding """
    with csv_reader('ohlcv.csv', subdir="data") as cr:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern"))
        for i, result in enumerate(r.run_iter()):
            if len(result) == 3:  # Strategy
                candle, plot, new_closed_trades = result
            else:  # Indicator
                candle, plot = result
            # Basic validation test - just ensure it runs without errors
            pass