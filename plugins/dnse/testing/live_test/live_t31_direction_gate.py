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

@script.strategy('LIVE T31 direction gate', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    winStart=input.time(timestamp("2030-01-01T00:00:00+07:00"), "Trade window START"),
    winEnd=input.time(timestamp("2030-01-01T23:59:00+07:00"), "Trade window END"),
    startState=input.int(0, "Start at state (0=short-blocked 1=long-control)", minval=0, maxval=1)
):
    _vc = _vcm._data if _vcm else None
    _vb = _vcm._build if _vcm else None
    strategy.risk.allow_entry_in(strategy.direction.long)

    state: PersistentSeries[int] = startState
    placedBar: PersistentSeries[int] = na(int)
    lvlEntry: PersistentSeries[float] = na(float)
    announced: PersistentSeries[bool] = False

    def phaseOf():
        dow = dayofweek(time, "Asia/Ho_Chi_Minh")
        hm = hour(time, "Asia/Ho_Chi_Minh") * 100 + minute(time, "Asia/Ho_Chi_Minh")
        return ("CLOSED" if dow == dayofweek.saturday or dow == dayofweek.sunday else ("CLOSED" if hm < 845 else ("ATO" if hm < 900 else ("POST-ATO" if hm < 915 else ("CONT-AM" if hm < 1130 else ("LUNCH" if hm < 1300 else ("CONT-PM" if hm < 1430 else ("ATC" if hm < 1445 else "CLOSED"))))))))


    phase = _vc[0][int(bar_index)] if _vc is not None and _vc[0] is not None else phaseOf()
    if _vb is not None and _vb[0] is not None: _vb[0].append(phase)
    pending = not na(placedBar)
    if time >= winStart:
        if not announced:
            log.info("[T31] === DIRECTION GATE — allow_entry_in(long). startState={0} ===", startState)
            announced = True
        log.info("[T31] bar={0} phase={1} state={2} pending={3}", bar_index, phase, state, ("yes" if pending else "no"))
        if time <= winEnd and (not pending) and state < 2:
            if state == 0:
                lvlEntry = close * 1.05
                strategy.entry("T31s", strategy.short, limit=lvlEntry, comment="T31s")
                placedBar = bar_index
                log.info("[T31] state0 SHORT attempt phase={0} limit={1} — the gate must swallow " + "this: ZERO [BROKER] lines, NOTHING at the venue. Any placement is a FAIL", phase, string.tostring(lvlEntry, format.mintick))
            elif state == 1 and close > open:
                lvlEntry = close * 0.95
                strategy.entry("T31l", strategy.long, limit=lvlEntry, comment="T31l")
                placedBar = bar_index
                log.info("[T31] state1 LONG control phase={0} limit={1} — must place normally " + "(proves the gate is directional, not total)", phase, string.tostring(lvlEntry, format.mintick))
        if pending and bar_index > placedBar:
            if state == 0:
                log.info("[T31] state0 VERIFY phase={0} — confirm venue shows nothing of T31s", phase)
            elif state == 1:
                strategy.cancel("T31l")
                log.info("[T31] state1 CANCEL T31l. DONE — verify venue clean", phase)
            placedBar = na
            state += 1
            if state == 2:
                log.info("[T31] === DONE ===")

    if strategy.position_size != 0:
        log.error("[T31] !!! UNEXPECTED FILL phase={0} pos={1} — flatten", phase, strategy.position_size)
        strategy.cancel_all()
        strategy.close_all(comment="SAFETY-FLATTEN")

    plot(state, "state", display=display.data_window)