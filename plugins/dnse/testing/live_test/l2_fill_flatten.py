"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    barstate, display, plot, script, strategy
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE2 fill+flatten', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    traded: PersistentSeries[bool] = False
    closing: PersistentSeries[bool] = False

    if barstate.isrealtime and (not traded):
        strategy.entry("E", strategy.long, comment="MKT 1-lot")
        traded = True

    if traded and strategy.position_size != 0 and (not closing):
        strategy.close("E", comment="FLATTEN")
        closing = True

    plot(strategy.position_size, "pos", display=display.data_window)