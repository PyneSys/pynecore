"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    close, display, format, open, plot, script, strategy, string
)

@script.strategy('T1 Long, market entry on green candle', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    TAKE_PCT = 0.02
    isGreen = close > open
    isRed = close < open
    flat = strategy.position_size == 0
    long_ = strategy.position_size > 0
    if flat and isGreen:
        strategy.entry("L", strategy.long, comment="E-mkt@" + string.tostring(close, format.mintick))

    if long_ and close >= strategy.position_avg_price * (1 + TAKE_PCT):
        strategy.close("L", comment="TP@" + string.tostring(close, format.mintick))
    elif long_ and isRed:
        strategy.close("L", comment="X-red@" + string.tostring(close, format.mintick))

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