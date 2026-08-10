"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    close, display, format, plot, script, strategy, string
)
from pynecore.types import PersistentSeries, Series

@script.strategy('N0 Limit then Cancel', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, calc_on_every_tick=False, process_orders_on_close=False)
def main():
    placed: PersistentSeries[bool] = False
    cancelled: PersistentSeries[bool] = False

    limitPx = close[1] * 0.95
    if not placed:
        strategy.entry("L", strategy.long, qty=1, limit=limitPx, comment="LIMIT@" + string.tostring(limitPx, format.mintick))
        placed = True
    elif placed and (not cancelled):
        strategy.cancel("L", comment="CANCEL")
        cancelled = True

    plot(strategy.position_size, "pos", display=display.data_window)
    plot(limitPx, "limit", display=display.data_window)