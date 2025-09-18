"""
@pyne

This code was compiled by PyneComp — the Pine Script to Python compiler.
Accessible via PyneSys: https://pynesys.io
Run with open-source PyneCore: https://pynecore.org
"""
from pynecore.lib import (
    barstate, bgcolor, close, color, input, na, plot, position, script,
    strategy, string, ta, table
)
from pynecore.types import Persistent, Table

SLIPPAGE_TICKS: int = 20


@script.strategy("Slippage Test - Limit Orders (No Slippage)", overlay=True, initial_capital=10000, slippage=SLIPPAGE_TICKS, commission_type=strategy.commission.percent, commission_value=0.1)
def main(
    limitOffset=input.float(0.5, "Limit Offset %", minval=0.1, step=0.1),
    stopOffset=input.float(1.0, "Stop Offset %", minval=0.1, step=0.1),
    useMarketOrders=input.bool(False, "Use Market Orders (for comparison)")
):

    fastMA = ta.sma(close, 10)
    slowMA = ta.sma(close, 20)

    longSignal = ta.crossover(fastMA, slowMA)
    shortSignal = ta.crossunder(fastMA, slowMA)

    if longSignal:
        if useMarketOrders:
            strategy.entry('Long Market', strategy.long, comment='Market Long')
        else:
            limitPrice = close * (1 - limitOffset / 100)
            strategy.entry('Long Limit', strategy.long, limit=limitPrice, comment='Limit Long @ ' + string.tostring(limitPrice, '#.##'))

    if shortSignal:
        if useMarketOrders:
            strategy.entry('Short Market', strategy.short, comment='Market Short')
        else:
            limitPrice = close * (1 + limitOffset / 100)
            strategy.entry('Short Limit', strategy.short, limit=limitPrice, comment='Limit Short @ ' + string.tostring(limitPrice, '#.##'))

    if strategy.position_size > 0:
        stopPrice = strategy.position_avg_price * (1 - stopOffset / 100)
        strategy.exit('Long Stop', from_entry='Long Limit', stop=stopPrice, comment='Stop Exit')
        strategy.exit('Long Stop Market', from_entry='Long Market', stop=stopPrice, comment='Stop Exit')

    if strategy.position_size < 0:
        stopPrice = strategy.position_avg_price * (1 + stopOffset / 100)
        strategy.exit('Short Stop', from_entry='Short Limit', stop=stopPrice, comment='Stop Exit')
        strategy.exit('Short Stop Market', from_entry='Short Market', stop=stopPrice, comment='Stop Exit')

    plot(fastMA, 'Fast MA', color.blue, 2)
    plot(slowMA, 'Slow MA', color.red, 2)

    bgcolor(color.new(color.green, 90) if longSignal else na, title='Long Signal')
    bgcolor(color.new(color.red, 90) if shortSignal else na, title='Short Signal')

    infoTable: Persistent[Table] = table.new(position.top_right, 2, 7, border_width=1)

    if barstate.islast:
        table.cell(infoTable, 0, 0, 'Slippage Setting:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 0, string.tostring(SLIPPAGE_TICKS) + ' ticks', bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 1, 'Order Type:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 1, 'MARKET' if useMarketOrders else 'LIMIT', bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 2, 'Position:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 2, string.tostring(strategy.position_size), bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 3, 'Avg Entry:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 3, string.tostring(strategy.position_avg_price, '#.####'), bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 4, 'Total Trades:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 4, string.tostring(strategy.closedtrades), bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 5, 'Net Profit:', bgcolor=color.gray, text_color=color.white)
        table.cell(infoTable, 1, 5, string.tostring(strategy.netprofit, '#.##'), bgcolor=color.gray, text_color=color.white)

        table.cell(infoTable, 0, 6, 'Note:', bgcolor=color.yellow, text_color=color.black)
        table.cell(infoTable, 1, 6, 'Limit orders should NOT have slippage', bgcolor=color.yellow, text_color=color.black)


# noinspection PyShadowingNames
def __test_slippage_limit__(csv_reader, runner, strat_equity_comparator):
    """ Slippage Test - Limit Orders """
    with csv_reader('slippage_limit_ohlcv.csv', subdir="data") as cr, \
            csv_reader('slippage_limit_trades.csv', subdir="data") as cr_equity:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern", type="forex"))
        equity_iter = iter(cr_equity)
        for i, (candle, plot, new_closed_trades) in enumerate(r.run_iter()):
            for trade in new_closed_trades:
                good_entry = next(equity_iter)
                good_exit = next(equity_iter)
                strat_equity_comparator(trade, good_entry.extra_fields, good_exit.extra_fields)