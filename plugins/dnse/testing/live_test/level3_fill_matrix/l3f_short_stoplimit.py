"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, barstate, display, format, high, low, na, plot, script, strategy, string
)
from pynecore.types import PersistentSeries, Series

@script.strategy('L3f short stop-limit', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    traded: PersistentSeries[bool] = False
    closing: PersistentSeries[bool] = False

    if barstate.isrealtime and (not traded):
        strategy.entry("E", strategy.short, stop=low[1], limit=low[1] * 0.999, comment="STOPLIMIT short")
        traded = True

    slLevel: PersistentSeries[float] = na(float)

    if strategy.position_size < 0:
        if na(slLevel):
            slLevel = high[1]
        strategy.exit("X", from_entry="E", stop=slLevel, comment_loss="SL@" + string.tostring(slLevel, format.mintick))

    if traded and strategy.position_size != 0 and (not closing):
        strategy.close("E", comment="FLATTEN")
        closing = True

    PROTECT_PCT = -0.1
    openPnlPct = (strategy.opentrades.profit_percent(strategy.opentrades - 1) if strategy.position_size != 0 else 0.0)
    if strategy.position_size != 0 and openPnlPct < PROTECT_PCT:
        strategy.close_all(comment="PROTECT")

    plot(strategy.position_size, "pos", display=display.data_window)
    plot(strategy.position_avg_price, "avg", display=display.data_window)