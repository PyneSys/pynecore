"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    close, display, format, open, plot, script, strategy, string
)

@script.strategy('LIVE2 stop place+cancel', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    isGreen = close > open
    isRed = close < open
    stopPrice = close[1] * 1.05
    if isGreen:
        strategy.entry("S", strategy.long, stop=stopPrice, comment="stop@" + string.tostring(stopPrice, format.mintick))

    if isRed:
        strategy.cancel("S")

    PROTECT_PCT = -0.1
    openPnlPct = (strategy.opentrades.profit_percent(strategy.opentrades - 1) if strategy.position_size != 0 else 0.0)
    if strategy.position_size != 0 and openPnlPct < PROTECT_PCT:
        strategy.close_all(comment="PROTECT")

    plot(strategy.position_size, "pos", display=display.data_window)