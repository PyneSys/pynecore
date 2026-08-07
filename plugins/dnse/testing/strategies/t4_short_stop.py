"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    close, display, format, high, low, plot, script, strategy, string
)

@script.strategy('T4 Short, stop entry below prior low', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    TAKE_PCT = 0.02
    flat = strategy.position_size == 0
    short_ = strategy.position_size < 0
    if flat:
        strategy.entry("S", strategy.short, stop=low[1], comment="E-stop@" + string.tostring(low[1], format.mintick))

    if short_:
        strategy.exit("X", from_entry="S", stop=high[1], comment_loss="SL@" + string.tostring(high[1], format.mintick))
        if close <= strategy.position_avg_price * (1 - TAKE_PCT):
            strategy.close("S", comment="TP@" + string.tostring(close, format.mintick))

    PROTECT_PCT = -0.1
    openPnlPct = (strategy.opentrades.profit_percent(strategy.opentrades - 1) if strategy.position_size != 0 else 0.0)
    if strategy.position_size != 0 and openPnlPct < PROTECT_PCT:
        strategy.close_all(comment="PROTECT")

    plot(strategy.position_size, "pos", display=display.data_window)
    plot(strategy.position_avg_price, "avg", display=display.data_window)