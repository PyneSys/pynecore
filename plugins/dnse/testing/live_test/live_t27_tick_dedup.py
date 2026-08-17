"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, bar_index, close, dayofweek, display, format, hour, input, log, minute, na, open, plot, script, strategy, string, time, timestamp
)
from pynecore.types import PersistentSeries, Series

try:
    from pynecore.core import _var_cache as _vcm
except ImportError:
    _vcm = None

__var_deps__ = {0: frozenset()}
__num_cache_slots__ = 1

@script.strategy('LIVE T27 tick dedup', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=True, process_orders_on_close=False)
def main(
    winStart=input.time(timestamp("2030-01-01T00:00:00+07:00"), "Trade window START"),
    winEnd=input.time(timestamp("2030-01-01T23:59:00+07:00"), "Trade window END"),
    holdBars=input.int(3, "Bars to hold before cancel", minval=1, maxval=30)
):
    _vc = _vcm._data if _vcm else None
    _vb = _vcm._build if _vcm else None
    placedBar: PersistentSeries[int] = na(int)
    lvlEntry: PersistentSeries[float] = na(float)
    cancelled: PersistentSeries[bool] = False
    execCount: PersistentSeries[int] = 0

    def phaseOf():
        dow = dayofweek(time, "Asia/Ho_Chi_Minh")
        hm = hour(time, "Asia/Ho_Chi_Minh") * 100 + minute(time, "Asia/Ho_Chi_Minh")
        return ("CLOSED" if dow == dayofweek.saturday or dow == dayofweek.sunday else ("CLOSED" if hm < 845 else ("ATO" if hm < 900 else ("POST-ATO" if hm < 915 else ("CONT-AM" if hm < 1130 else ("LUNCH" if hm < 1300 else ("CONT-PM" if hm < 1430 else ("ATC" if hm < 1445 else "CLOSED"))))))))


    phase = _vc[0][int(bar_index)] if _vc is not None and _vc[0] is not None else phaseOf()
    if _vb is not None and _vb[0] is not None: _vb[0].append(phase)
    execCount += 1
    if time >= winStart and time <= winEnd:
        if na(placedBar) and (not cancelled) and close > open:
            lvlEntry = close * 0.95
            strategy.entry("T27", strategy.long, limit=lvlEntry, comment="T27")
            placedBar = bar_index
            log.info("[T27] PLACE phase={0} limit={1} exec#{2} — calc_on_every_tick=true: this " + "body reruns per tick, yet the venue must record EXACTLY ONE placement", phase, string.tostring(lvlEntry, format.mintick), execCount)
        if not na(placedBar):
            log.info("[T27] tick phase={0} bar={1} exec#{2} placedBar={3} — every extra venue " + "placement after the first is a FAIL", phase, bar_index, execCount, placedBar)

    if (not na(placedBar)) and (not cancelled) and bar_index >= placedBar + holdBars:
        strategy.cancel("T27")
        cancelled = True
        log.info("[T27] CANCEL phase={0} after {1} bars, exec#{2} — venue must show ONE cancel; " + "book clean of ours after. DONE", phase, holdBars, execCount)

    if strategy.position_size != 0:
        log.error("[T27] !!! UNEXPECTED FILL phase={0} pos={1} — flatten", phase, strategy.position_size)
        strategy.cancel_all()
        strategy.close_all(comment="SAFETY-FLATTEN")

    plot(execCount, "exec_count", display=display.data_window)
    plot(strategy.position_size, "pos", display=display.data_window)