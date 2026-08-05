"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    close, display, format, high, low, plot, script, strategy, string
)

@script.strategy('T6 Short, stop+limit entry and bracketed exit', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    TAKE_PCT = 0.02
    flat = strategy.position_size == 0
    short_ = strategy.position_size < 0
    if flat:
        strategy.entry("S", strategy.short, stop=low[1], limit=low[1] * 0.999, comment="E-sl@" + string.tostring(low[1], format.mintick))

    if short_:
        strategy.exit("X", from_entry="S", stop=high[1], limit=strategy.position_avg_price * 0.99, comment_loss="SL@" + string.tostring(high[1], format.mintick), comment_profit="TPlim@" + string.tostring(strategy.position_avg_price * 0.99, format.mintick))
        if close <= strategy.position_avg_price * (1 - TAKE_PCT):
            strategy.close("S", comment="TP@" + string.tostring(close, format.mintick))

    PROTECT_PCT = -0.5
    openPnlPct = 0.0
    if strategy.position_size > 0:
        openPnlPct = (close - strategy.position_avg_price) / strategy.position_avg_price * 100
    elif strategy.position_size < 0:
        openPnlPct = (strategy.position_avg_price - close) / strategy.position_avg_price * 100

    if strategy.position_size != 0 and openPnlPct < PROTECT_PCT:
        strategy.close_all(comment="PROTECT")

    plot(strategy.position_size, "pos", display=display.data_window)
    plot(strategy.position_avg_price, "avg", display=display.data_window)