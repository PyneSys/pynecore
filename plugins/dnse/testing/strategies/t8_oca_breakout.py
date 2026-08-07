"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    close, display, format, open, plot, script, strategy, string
)

@script.strategy('T8 OCA breakout group', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    isGreen = close > open
    isRed = close < open
    upStop = close[1] * 1.05
    dnStop = close[1] * 0.95

    if isGreen:
        strategy.entry("Up", strategy.long, stop=upStop, oca_name="brk", oca_type=strategy.oca.cancel, comment="buy-stop@" + string.tostring(upStop, format.mintick))
        strategy.entry("Dn", strategy.short, stop=dnStop, oca_name="brk", oca_type=strategy.oca.cancel, comment="sell-stop@" + string.tostring(dnStop, format.mintick))

    if isRed:
        strategy.cancel("Up")
        strategy.cancel("Dn")

    PROTECT_PCT = -0.1
    openPnlPct = (strategy.opentrades.profit_percent(strategy.opentrades - 1) if strategy.position_size != 0 else 0.0)
    if strategy.position_size != 0 and openPnlPct < PROTECT_PCT:
        strategy.close_all(comment="PROTECT")

    plot(strategy.position_size, "pos", display=display.data_window)