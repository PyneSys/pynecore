"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, barstate, close, display, format, na, plot, script, strategy, string
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE2b fill+bracket+flatten', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    traded: PersistentSeries[bool] = False
    closing: PersistentSeries[bool] = False
    tpLevel: PersistentSeries[float] = na(float)
    slLevel: PersistentSeries[float] = na(float)

    if barstate.isrealtime and (not traded):
        tpLevel = close * 1.002
        slLevel = close * 0.998
        strategy.entry("E", strategy.long, comment="MKT 1-lot")
        strategy.exit("X", from_entry="E", limit=tpLevel, stop=slLevel, oca_name="brk", oca_type=strategy.oca.cancel, comment_profit="TP@" + string.tostring(tpLevel, format.mintick), comment_loss="SL@" + string.tostring(slLevel, format.mintick))
        traded = True

    DOUBLE_CHECK_PCT = -0.3
    openPnlPct = (strategy.opentrades.profit_percent(strategy.opentrades - 1) if strategy.position_size != 0 else 0.0)
    if strategy.position_size != 0 and openPnlPct < DOUBLE_CHECK_PCT:
        strategy.cancel_all()
        strategy.close_all(comment="DOUBLECHECK")

    if traded and strategy.position_size != 0 and (not closing):
        strategy.close("E", comment="FLATTEN")
        strategy.cancel_all()
        closing = True

    plot(strategy.position_size, "pos", display=display.data_window)
    plot(tpLevel, "tp", display=display.data_window)
    plot(slLevel, "sl", display=display.data_window)