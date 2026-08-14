"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, bar_index, close, display, format, input, log, na, open, plot, script, strategy, string, time, timestamp
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE staged place+cancel', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    winStart=input.time(timestamp("2030-01-01T00:00:00+07:00"), "Trade window START"),
    winEnd=input.time(timestamp("2030-01-01T23:59:00+07:00"), "Trade window END"),
    startState=input.int(0, "Start at state (0=T1 .. 8=T9)", minval=0, maxval=8)
):
    state: PersistentSeries[int] = startState
    placedBar: PersistentSeries[int] = na(int)
    stageStep: PersistentSeries[int] = 0
    lvlEntry: PersistentSeries[float] = na(float)
    lvlStop: PersistentSeries[float] = na(float)
    lvlTP: PersistentSeries[float] = na(float)
    lvlAmend: PersistentSeries[float] = na(float)
    announced: PersistentSeries[bool] = False

    isGreen = close > open
    pending = not na(placedBar)
    started = time >= winStart
    canPlace = started and time <= winEnd
    if started:
        if not announced:
            log.info("[L1] === STAGED PLACE/CANCEL v2 — window open. Tests T1-T9, 1 contract " + "each, every order >=4.5% away so none can fill. startState={0} ===", startState)
            log.info("[L1] plan: T1/T2 limit place+cancel | T3 +exit(stop) | T4 OCA cancel-entry-only " + "| T5 native-OCO cancel-entry-only | T6 amend NORMAL | T7 amend STOP (#18 500) " + "| T8 cancel_all both books | T9 strategy.order()")
            announced = True
        log.info("[L1] bar={0} close={1} candle={2} state={3} step={4} pending={5} pos={6}", bar_index, string.tostring(close, format.mintick), ("GREEN" if isGreen else "RED"), state, stageStep, ("yes" if pending else "no"), strategy.position_size)
        if canPlace and (not pending) and state < 9:
            if state == 0 and isGreen:
                lvlEntry = close * 0.95
                strategy.entry("T1", strategy.long, limit=lvlEntry, comment="T1")
                placedBar = bar_index
                log.info("[L1] TEST 1 PLACE id=T1 LONG limit={0} (-5%) — expect one NORMAL LO, filled=0", string.tostring(lvlEntry, format.mintick))
            elif state == 1:
                lvlEntry = close * 1.05
                strategy.entry("T2", strategy.short, limit=lvlEntry, comment="T2")
                placedBar = bar_index
                log.info("[L1] TEST 2 PLACE id=T2 SHORT limit={0} (+5%)", string.tostring(lvlEntry, format.mintick))
            elif state == 2 and isGreen:
                lvlEntry = close * 0.95
                lvlStop = close * 0.94
                strategy.entry("T3", strategy.long, limit=lvlEntry, comment="T3")
                strategy.exit("X3", from_entry="T3", stop=lvlStop, comment_loss="X3")
                placedBar = bar_index
                log.info("[L1] TEST 3 PLACE id=T3 limit={0} + X3 stop={1} — exit reaches the venue " + "pre-fill (measured); both cancelled explicitly next bar", string.tostring(lvlEntry, format.mintick), string.tostring(lvlStop, format.mintick))
            elif state == 3:
                lvlEntry = close * 1.05
                lvlStop = close * 1.055
                strategy.entry("T4", strategy.short, limit=lvlEntry, oca_name="t4", oca_type=strategy.oca.cancel, comment="T4")
                strategy.exit("X4", from_entry="T4", stop=lvlStop, oca_name="t4", comment_loss="X4")
                placedBar = bar_index
                log.info("[L1] TEST 4 PLACE id=T4 limit={0} + X4 stop={1} (oca) — next bar cancels " + "ENTRY ONLY; the cascade must remove X4 (verified 2026-08-14)", string.tostring(lvlEntry, format.mintick), string.tostring(lvlStop, format.mintick))
            elif state == 4 and isGreen:
                lvlEntry = close * 0.95
                lvlTP = close * 1.05
                lvlStop = close * 0.94
                strategy.entry("T5", strategy.long, limit=lvlEntry, comment="T5")
                strategy.exit("X5", from_entry="T5", limit=lvlTP, stop=lvlStop, comment_profit="X5tp", comment_loss="X5sl")
                placedBar = bar_index
                log.info("[L1] TEST 5 PLACE id=T5 limit={0} + X5 tp={1}/sl={2} -> NATIVE OCO — first " + "Level-1 use of the OCO umbrella book. Next bar cancels ENTRY ONLY: does the " + "cascade clear the OCO leg too? (only STOP was verified this morning)", string.tostring(lvlEntry, format.mintick), string.tostring(lvlTP, format.mintick), string.tostring(lvlStop, format.mintick))
            elif state == 5 and isGreen:
                lvlEntry = close * 0.95
                lvlAmend = close * 0.955
                strategy.entry("T6", strategy.long, limit=lvlEntry, comment="T6")
                placedBar = bar_index
                log.info("[L1] TEST 6 PLACE id=T6 LONG limit={0} (-5%) — NEXT bar re-issues at {1} " + "(-4.5%): an AMEND on the NORMAL book (worked 2026-08-10; re-verify on 6.8.5)", string.tostring(lvlEntry, format.mintick), string.tostring(lvlAmend, format.mintick))
            elif state == 6:
                lvlStop = close * 1.05
                lvlAmend = close * 1.045
                strategy.entry("T7", strategy.long, stop=lvlStop, comment="T7")
                placedBar = bar_index
                log.info("[L1] TEST 7 PLACE id=T7 buy-STOP {0} (+5%) — NEXT bar re-issues at {1} " + "(+4.5%): an AMEND on a CONDITIONAL. DNSE 500s (#18); expect a [BROKER] " + "park+verify WARNING, not a crash — then prove it still cancels", string.tostring(lvlStop, format.mintick), string.tostring(lvlAmend, format.mintick))
            elif state == 7 and isGreen:
                lvlEntry = close * 0.95
                lvlStop = close * 1.05
                strategy.entry("T8a", strategy.long, limit=lvlEntry, comment="T8a")
                strategy.entry("T8b", strategy.long, stop=lvlStop, comment="T8b")
                placedBar = bar_index
                log.info("[L1] TEST 8 PLACE T8a limit={0} AND T8b stop={1} — two ids across BOTH " + "books; next bar strategy.cancel_all(), never before fired live", string.tostring(lvlEntry, format.mintick), string.tostring(lvlStop, format.mintick))
            elif state == 8 and isGreen:
                lvlEntry = close * 0.95
                strategy.order("T9", strategy.long, limit=lvlEntry, comment="T9")
                placedBar = bar_index
                log.info("[L1] TEST 9 PLACE via strategy.order() LONG limit={0} — order() has never " + "touched the broker; expect identical NORMAL-LO routing to entry()", string.tostring(lvlEntry, format.mintick))
        if pending and bar_index > placedBar:
            if state == 5 and stageStep == 0:
                strategy.entry("T6", strategy.long, limit=lvlAmend, comment="T6amend")
                stageStep = 1
                placedBar = bar_index
                log.info("[L1] TEST 6 AMEND id=T6 -> limit {0} — expect [BROKER] 'modifying ...' " + "then the venue accepting the new price", string.tostring(lvlAmend, format.mintick))
            elif state == 6 and stageStep == 0:
                strategy.entry("T7", strategy.long, stop=lvlAmend, comment="T7amend")
                stageStep = 1
                placedBar = bar_index
                log.info("[L1] TEST 7 AMEND id=T7 -> stop {0} — expect code=REMOTE_SERVER_ERROR " + "http=500 -> park+verify (#18). KEY: the run must continue, and the order " + "must still cancel next bar", string.tostring(lvlAmend, format.mintick))
            else:
                if state == 0:
                    strategy.cancel("T1")
                    log.info("[L1] TEST 1 CANCEL id=T1")
                elif state == 1:
                    strategy.cancel("T2")
                    log.info("[L1] TEST 2 CANCEL id=T2")
                elif state == 2:
                    strategy.cancel("T3")
                    strategy.cancel("X3")
                    log.info("[L1] TEST 3 CANCEL id=T3 and id=X3 (both, explicitly)")
                elif state == 3:
                    strategy.cancel("T4")
                    log.info("[L1] TEST 4 CANCEL id=T4 ONLY — cascade must also CANCEL X4")
                elif state == 4:
                    strategy.cancel("T5")
                    log.info("[L1] TEST 5 CANCEL id=T5 ONLY — KEY OBSERVATION: does the cascade " + "clear the OCO leg X5? Venue must show NOTHING of T5/X5 working after")
                elif state == 5:
                    strategy.cancel("T6")
                    log.info("[L1] TEST 6 CANCEL id=T6 (post-amend)")
                elif state == 6:
                    strategy.cancel("T7")
                    log.info("[L1] TEST 7 CANCEL id=T7 (post-500) — proves a parked amend does not " + "wedge the order: it must still cancel cleanly")
                elif state == 7:
                    strategy.cancel_all()
                    log.info("[L1] TEST 8 strategy.cancel_all() — BOTH T8a (NORMAL) and T8b (STOP) " + "must go; venue book must be clear of ours")
                elif state == 8:
                    strategy.cancel("T9")
                    log.info("[L1] TEST 9 CANCEL id=T9")
                log.info("[L1] TEST {0} DONE — state {1}->{2}", state + 1, state, state + 1)
                placedBar = na
                stageStep = 0
                state += 1
                if state == 9:
                    log.info("[L1] === ALL 9 TESTS DONE. Verify at the venue that NOTHING of ours " + "is still working, then stop the run. ===")
        if strategy.position_size != 0:
            log.error("[L1] !!! UNEXPECTED FILL: pos={0} avg={1} — cancelling all and flattening", strategy.position_size, string.tostring(strategy.position_avg_price, format.mintick))
            strategy.cancel_all()
            strategy.close_all(comment="SAFETY-FLATTEN")

    plot(state, "state", display=display.data_window)
    plot(strategy.position_size, "pos", display=display.data_window)
    plot(lvlEntry, "entry_lvl", display=display.data_window)
    plot(lvlStop, "stop_lvl", display=display.data_window)