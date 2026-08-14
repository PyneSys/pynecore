"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, bar_index, close, display, format, input, log, na, plot, script, strategy, string, time, timestamp
)
from pynecore.types import PersistentSeries, Series

@script.strategy('DUAL A', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    winStart=input.time(timestamp("2030-01-01T00:00:00+07:00"), "Trade window START"),
    winEnd=input.time(timestamp("2030-01-01T23:59:00+07:00"), "Trade window END")
):
    placedBar: PersistentSeries[int] = na(int)
    lvl: PersistentSeries[float] = na(float)
    done: PersistentSeries[bool] = False

    if time >= winStart:
        log.info("[DUAL-A] bar={0} close={1} placed={2} done={3} pos={4}", bar_index, string.tostring(close, format.mintick), ("no" if na(placedBar) else "yes"), ("yes" if done else "no"), strategy.position_size)

    if time >= winStart and time <= winEnd and na(placedBar) and (not done):
        lvl = close * 0.95
        strategy.entry("DUALA", strategy.long, limit=lvl, comment="DUALA")
        placedBar = bar_index
        log.info("[DUAL-A] PLACE id=DUALA limit={0}", string.tostring(lvl, format.mintick))

    if (not na(placedBar)) and bar_index > placedBar and (not done):
        strategy.cancel("DUALA")
        done = True
        log.info("[DUAL-A] CANCEL step done (cancel by id)")

    if strategy.position_size != 0:
        log.error("[DUAL-A] !!! UNEXPECTED FILL pos={0}", strategy.position_size)
        strategy.cancel_all()
        strategy.close_all(comment="SAFETY")

    plot(strategy.position_size, "pos", display=display.data_window)