"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, bar_index, close, dayofweek, display, format, hour, input, log, math, minute, na, open, plot, script, strategy, string, syminfo, time, timestamp
)
from pynecore.types import PersistentSeries, Series

try:
    from pynecore.core import _var_cache as _vcm
except ImportError:
    _vcm = None

__var_deps__ = {0: frozenset()}
__num_cache_slots__ = 1

@script.strategy('LIVE staged param matrix', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    winStart=input.time(timestamp("2030-01-01T00:00:00+07:00"), "Trade window START"),
    winEnd=input.time(timestamp("2030-01-01T23:59:00+07:00"), "Trade window END"),
    startState=input.int(0, "Start at state (0=T19 .. 13=T32)", minval=0, maxval=13)
):
    _vc = _vcm._data if _vcm else None
    _vb = _vcm._build if _vcm else None
    strategy.risk.max_position_size(2)

    state: PersistentSeries[int] = startState
    placedBar: PersistentSeries[int] = na(int)
    stageStep: PersistentSeries[int] = 0
    lvlEntry: PersistentSeries[float] = na(float)
    lvlStop: PersistentSeries[float] = na(float)
    lvlTP: PersistentSeries[float] = na(float)
    announced: PersistentSeries[bool] = False

    def phaseOf():
        dow = dayofweek(time, "Asia/Ho_Chi_Minh")
        hm = hour(time, "Asia/Ho_Chi_Minh") * 100 + minute(time, "Asia/Ho_Chi_Minh")
        return ("CLOSED" if dow == dayofweek.saturday or dow == dayofweek.sunday else ("CLOSED" if hm < 845 else ("ATO" if hm < 900 else ("POST-ATO" if hm < 915 else ("CONT-AM" if hm < 1130 else ("LUNCH" if hm < 1300 else ("CONT-PM" if hm < 1430 else ("ATC" if hm < 1445 else "CLOSED"))))))))


    phase = _vc[0][int(bar_index)] if _vc is not None and _vc[0] is not None else phaseOf()
    if _vb is not None and _vb[0] is not None: _vb[0].append(phase)
    isGreen = close > open
    pending = not na(placedBar)
    started = time >= winStart
    canPlace = started and time <= winEnd

    contOK = phase == "CONT-AM" or phase == "CONT-PM"
    if started:
        if not announced:
            log.info("[PRM] === PARAM MATRIX — window open. T19-T26/T30 continuous, T16 hold/restart, " + "T14 ATC, T32 ATO. 1 contract unless flagged, >=4.5% away. startState={0} ===", startState)
            announced = True
        log.info("[PRM] bar={0} close={1} candle={2} phase={3} state={4} step={5} pending={6} pos={7}", bar_index, string.tostring(close, format.mintick), ("GREEN" if isGreen else "RED"), phase, state, stageStep, ("yes" if pending else "no"), strategy.position_size)
        if canPlace and (not pending) and state < 14 and (state >= 12 or contOK):
            if state == 0 and isGreen:
                lvlStop = close * 1.05
                lvlTP = close * 0.95
                strategy.entry("T19", strategy.long, stop=lvlStop, limit=lvlTP, comment="T19")
                placedBar = bar_index
                log.info("[PRM] T19 PLACE phase={0} buy STOP-LIMIT trigger={1} limit={2} — KEY (#14): " + "venue must show exactly ONE conditional stop-limit, NOT an OCO pair", phase, string.tostring(lvlStop, format.mintick), string.tostring(lvlTP, format.mintick))
            elif state == 1:
                lvlStop = close * 0.95
                strategy.entry("T20", strategy.short, stop=lvlStop, comment="T20")
                placedBar = bar_index
                log.info("[PRM] T20 PLACE phase={0} SHORT sell-STOP {1} (-5%) — first-ever SELL " + "trigger direction on the conditional book (#13 evidence): reject loudly, " + "emulate, or silent drop? Watch [BROKER]", phase, string.tostring(lvlStop, format.mintick))
            elif state == 2 and isGreen:
                lvlEntry = close * 0.95
                strategy.entry("T21", strategy.long, limit=lvlEntry, comment="T21")
                placedBar = bar_index
                log.info("[PRM] T21 PLACE phase={0} qty OMITTED limit={1} — venue qty must equal " + "default_qty_value=1 (sizing path)", phase, string.tostring(lvlEntry, format.mintick))
            elif state == 3 and isGreen:
                lvlEntry = close * 0.95
                profitTicks = math.round(close * 0.1 / syminfo.mintick)
                lossTicks = math.round(close * 0.01 / syminfo.mintick)
                lvlTP = lvlEntry + profitTicks * syminfo.mintick
                lvlStop = lvlEntry - lossTicks * syminfo.mintick
                strategy.entry("T22", strategy.long, limit=lvlEntry, comment="T22")
                strategy.exit("X22", from_entry="T22", profit=profitTicks, loss=lossTicks, comment_profit="X22tp", comment_loss="X22sl")
                placedBar = bar_index
                log.info("[PRM] T22 PLACE phase={0} entry={1} + exit(profit={2}t loss={3}t) — KEY: " + "venue legs must rest at EXACTLY tp={4} sl={5} (entry +/- N*mintick)", phase, string.tostring(lvlEntry, format.mintick), profitTicks, lossTicks, string.tostring(lvlTP, format.mintick), string.tostring(lvlStop, format.mintick))
            elif state == 4 and isGreen:
                lvlEntry = close * 0.95
                trailPts = math.round(close * 0.02 / syminfo.mintick)
                strategy.entry("T23", strategy.long, limit=lvlEntry, comment="T23")
                strategy.exit("X23", from_entry="T23", trail_points=trailPts, trail_offset=trailPts, comment_trailing="X23tr")
                placedBar = bar_index
                log.info("[PRM] T23 PLACE phase={0} entry={1} + exit(trail_points={2} ONLY) — " + "NEGATIVE PROBE: trailing is engine-emulated; NOTHING of X23 may reach the " + "venue pre-activation. PASS = T23 is the ONLY working order", phase, string.tostring(lvlEntry, format.mintick), trailPts)
            elif state == 5 and isGreen:
                lvlEntry = close * 0.95
                lvlStop = close * 0.94
                strategy.entry("T24", strategy.long, qty=2, limit=lvlEntry, comment="T24")
                strategy.exit("X24", from_entry="T24", qty_percent=50, stop=lvlStop, comment_loss="X24")
                placedBar = bar_index
                log.info("[PRM] T24 PLACE phase={0} entry qty=2 limit={1} + exit(qty_percent=50 " + "stop={2}) — KEY: the exit leg must rest at venue qty 1 (50% of 2)", phase, string.tostring(lvlEntry, format.mintick), string.tostring(lvlStop, format.mintick))
            elif state == 6 and isGreen:
                lvlEntry = close * 0.95
                lvlStop = close * 0.94
                lvlTP = close * 0.935
                strategy.entry("T25", strategy.long, qty=2, limit=lvlEntry, comment="T25")
                strategy.exit("X25a", from_entry="T25", qty=1, stop=lvlStop, comment_loss="X25a")
                strategy.exit("X25b", from_entry="T25", qty=1, stop=lvlTP, comment_loss="X25b")
                placedBar = bar_index
                log.info("[PRM] T25 PLACE phase={0} entry qty=2 + TWO exits qty=1 each (stops {1} / " + "{2}) — both must rest, combined qty <= entry qty; then cancel all three", phase, string.tostring(lvlStop, format.mintick), string.tostring(lvlTP, format.mintick))
            elif state == 7:
                strategy.close("GHOST", comment="T26a")
                placedBar = bar_index
                log.info("[PRM] T26a close('GHOST') phase={0} with NO position and NO order — " + "PASS = clean no-op, ZERO [BROKER] activity on this bar", phase)
            elif state == 8 and isGreen:
                lvlEntry = close * 0.95
                strategy.entry("T26", strategy.long, limit=lvlEntry, comment="T26")
                placedBar = bar_index
                log.info("[PRM] T26b PLACE phase={0} limit={1} — NEXT bar calls strategy.close('T26') " + "while it RESTS: close() targets POSITIONS and must NOT cancel the resting " + "entry", phase, string.tostring(lvlEntry, format.mintick))
            elif state == 9 and isGreen:
                lvlEntry = close * 0.95
                strategy.entry("T30", strategy.long, qty=5, limit=lvlEntry, comment="T30")
                placedBar = bar_index
                log.info("[PRM] T30 PLACE phase={0} qty=5 limit={1} with risk.max_position_size(2) — " + "KEY: engine trims at placement; venue order must rest at qty 2, not 5", phase, string.tostring(lvlEntry, format.mintick))
            elif state == 10 and isGreen:
                lvlEntry = close * 0.95
                strategy.entry("T16", strategy.long, limit=lvlEntry, comment="T16")
                placedBar = bar_index
                log.info("[PRM] T16a PLACE-AND-HOLD phase={0} limit={1} — this state NEVER advances. " + "Kill the engine now (run_t16_restart.sh drives this), then relaunch with " + "startState=11", phase, string.tostring(lvlEntry, format.mintick))
            elif state == 12 and (phase == "CONT-PM" or phase == "ATC"):
                lvlEntry = close * 0.95
                lvlStop = close * 0.94
                strategy.entry("T14", strategy.long, limit=lvlEntry, comment="T14")
                strategy.entry("T14r", strategy.long, limit=lvlStop, comment="T14r")
                placedBar = bar_index
                log.info("[PRM] T14 PLACE phase={0} T14 limit={1} + rider T14r limit={2} — cancel of " + "T14 fires only when phase==ATC (venue must REFUSE: CANNOT_CANCEL_..._ATC); " + "T14r is never cancelled and must EXPIRE at the close. Grade both terminal " + "states from the venue AFTER the session", phase, string.tostring(lvlEntry, format.mintick), string.tostring(lvlStop, format.mintick))
            elif state == 13 and (phase == "ATO" or phase == "POST-ATO"):
                lvlEntry = close * 0.95
                lvlStop = close * 1.05
                strategy.entry("T32a", strategy.long, limit=lvlEntry, comment="T32a")
                strategy.entry("T32b", strategy.long, stop=lvlStop, comment="T32b")
                placedBar = bar_index
                log.info("[PRM] T32 PLACE phase={0} (FIRST-EVER ATO measurement) T32a limit={1} + " + "T32b cond stop={2} — record per-type accept/refuse " + "(CAN_NOT_PLACE_..._ON_ATO_SESSION); survivors cancelled once phase " + "leaves ATO", phase, string.tostring(lvlEntry, format.mintick), string.tostring(lvlStop, format.mintick))
        if pending and bar_index > placedBar and (state >= 12 or contOK):
            if state == 8 and stageStep == 0:
                strategy.close("T26", comment="T26close")
                stageStep = 1
                placedBar = bar_index
                log.info("[PRM] T26b CLOSE ATTEMPT phase={0} — strategy.close('T26') with zero " + "position: expect NO venue cancel; T26 must STILL be resting after this bar", phase)
            elif state == 10:
                log.info("[PRM] T16a HOLDING phase={0} — order resting, awaiting kill", phase)
            elif state == 12 and stageStep == 0:
                if phase == "ATC":
                    strategy.cancel("T14")
                    stageStep = 1
                    placedBar = bar_index
                    log.info("[PRM] T14 CANCEL-IN-ATC phase={0} — venue must REFUSE " + "(CANNOT_CANCEL_THE_ORDER_IN_THE_ATC_SESSION); plugin must surface + " + "park, NOT crash. T14/T14r then EXPIRE at the close", phase)
                else:
                    log.info("[PRM] T14 WAITING for ATC phase={0} — order resting through CONT-PM", phase)
            elif state == 12 and stageStep == 1:
                log.info("[PRM] T14 DONE (attempt made) phase={0} — grade terminal states from the " + "venue after 14:45", phase)
                placedBar = na
                stageStep = 0
                state = 14
            elif state == 13 and stageStep == 0:
                if phase != "ATO" and phase != "POST-ATO":
                    strategy.cancel_all()
                    stageStep = 1
                    placedBar = bar_index
                    log.info("[PRM] T32 CANCEL_ALL phase={0} — sweeping whatever the ATO accepted", phase)
                else:
                    log.info("[PRM] T32 WAITING for continuous phase={0}", phase)
            else:
                if state == 0:
                    strategy.cancel("T19")
                    log.info("[PRM] T19 CANCEL")
                elif state == 1:
                    strategy.cancel("T20")
                    log.info("[PRM] T20 CANCEL")
                elif state == 2:
                    strategy.cancel("T21")
                    log.info("[PRM] T21 CANCEL")
                elif state == 3:
                    strategy.cancel("T22")
                    strategy.cancel("X22")
                    log.info("[PRM] T22 CANCEL id=T22 and id=X22 (both, explicitly)")
                elif state == 4:
                    strategy.cancel("T23")
                    log.info("[PRM] T23 CANCEL id=T23 — X23 must never have existed at the venue")
                elif state == 5:
                    strategy.cancel("T24")
                    strategy.cancel("X24")
                    log.info("[PRM] T24 CANCEL id=T24 and id=X24")
                elif state == 6:
                    strategy.cancel("T25")
                    strategy.cancel("X25a")
                    strategy.cancel("X25b")
                    log.info("[PRM] T25 CANCEL all three, explicitly")
                elif state == 7:
                    log.info("[PRM] T26a DONE — verify the [BROKER] silence on the previous bar")
                elif state == 8:
                    strategy.cancel("T26")
                    log.info("[PRM] T26b CANCEL id=T26 (explicit, after the close() no-op)")
                elif state == 9:
                    strategy.cancel("T30")
                    log.info("[PRM] T30 CANCEL")
                elif state == 11:
                    strategy.cancel_all()
                    log.info("[PRM] T16b POST-RESTART cancel_all — the adopted/quarantined T16 " + "must be reachable; venue must be clean after")
                elif state == 13:
                    log.info("[PRM] T32 DONE")
                log.info("[PRM] STAGE DONE phase={0} — state {1}->{2}", phase, state, state + 1)
                placedBar = na
                stageStep = 0
                state += 1
                if state == 14:
                    log.info("[PRM] === PARAM MATRIX DONE. Verify at the venue that NOTHING of ours " + "is still working, then stop the run. ===")
        if state == 11 and (not pending) and started and contOK:
            if stageStep < 2:
                stageStep += 1
                log.info("[PRM] T16b OBSERVE bar {0}/2 phase={1} — watch [BROKER] startup: the " + "resting T16 must be adopted or quarantined EXPLICITLY, never silent", stageStep, phase)
            else:
                placedBar = bar_index - 1
                stageStep = 0
        if strategy.position_size != 0:
            log.error("[PRM] !!! UNEXPECTED FILL phase={0}: pos={1} avg={2} — cancelling all and " + "flattening", phase, strategy.position_size, string.tostring(strategy.position_avg_price, format.mintick))
            strategy.cancel_all()
            strategy.close_all(comment="SAFETY-FLATTEN")

    plot(state, "state", display=display.data_window)
    plot(strategy.position_size, "pos", display=display.data_window)
    plot(lvlEntry, "entry_lvl", display=display.data_window)
    plot(lvlStop, "stop_lvl", display=display.data_window)