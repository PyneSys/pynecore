"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, bar_index, close, display, format, input, log, na, plot, script, strategy, string, time, timestamp
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE crossed stop', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    winStart=input.time(timestamp("2030-01-01T00:00:00+07:00"), "Trade window START"),
    winEnd=input.time(timestamp("2030-01-01T23:59:00+07:00"), "Trade window END"),
    crossPct=input.float(1.0, "Trigger below market %", minval=0.1, maxval=5.0)
):
    FILL_TIMEOUT_BARS = 3

    placedBar: PersistentSeries[int] = na(int)
    flattening: PersistentSeries[bool] = False
    done: PersistentSeries[bool] = False
    announced: PersistentSeries[bool] = False

    started = time >= winStart
    canPlace = started and time <= winEnd
    pending = not na(placedBar)
    if started and (not announced):
        log.info("[XS] === F10 crossed-stop-at-placement — buy-stop trigger -{0}% BELOW " + "market. Oracle: fills at next open. Post-fix live: [BROKER] must show " + "'crossed stop' -> marketable NORMAL LO -> fill. ===", crossPct)
        announced = True

    if started:
        log.info("[XS] bar={0} close={1} pos={2} pending={3}", bar_index, string.tostring(close, format.mintick), strategy.position_size, ("yes" if pending else "no"))

    if canPlace and (not pending) and (not done) and (not flattening):
        trig = close * (1 - crossPct / 100)
        strategy.entry("F10", strategy.long, stop=trig, comment="F10")
        placedBar = bar_index
        log.info("[XS] PLACE buy-stop trigger={0} while close={1} — CROSSED at placement: " + "Pine fills IMMEDIATELY; a resting order here = the #34 gap", string.tostring(trig, format.mintick), string.tostring(close, format.mintick))

    if pending and (not flattening) and strategy.position_size > 0:
        log.info("[XS] FILLED pos={0} avg={1} (bars-to-fill={2}, expect 1) -> flatten", strategy.position_size, string.tostring(strategy.position_avg_price, format.mintick), bar_index - placedBar)
        strategy.close_all(comment="F10-flat")
        flattening = True

    if flattening and strategy.position_size == 0:
        log.info("[XS] === F10 DONE — filled and flat. Verify at the venue: flat, nothing " + "working. ===")
        flattening = False
        placedBar = na
        done = True

    if pending and (not flattening) and strategy.position_size == 0 and bar_index - placedBar >= FILL_TIMEOUT_BARS:
        log.error("[XS] !!! NO FILL after {0} bars — crossed stop did NOT execute " + "(#34 gap demonstrated). Cancelling.", FILL_TIMEOUT_BARS)
        strategy.cancel_all()
        placedBar = na
        done = True

    plot(strategy.position_size, "pos", display=display.data_window)