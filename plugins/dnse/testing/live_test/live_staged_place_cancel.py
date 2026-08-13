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
    winEnd=input.time(timestamp("2030-01-01T23:59:00+07:00"), "Trade window END")
):
    state: PersistentSeries[int] = 0
    placedBar: PersistentSeries[int] = na(int)
    lvlEntry: PersistentSeries[float] = na(float)
    lvlStop: PersistentSeries[float] = na(float)
    announced: PersistentSeries[bool] = False

    isGreen = close > open
    pending = not na(placedBar)
    started = time >= winStart
    canPlace = started and time <= winEnd
    if started:
        if not announced:
            log.info("[L1] === STAGED PLACE/CANCEL — window open. 4 tests, 1 contract each, " + "every order >=5% away so none can fill. ===")
            log.info("[L1] plan: T1 long lim-5% | T2 short lim+5% | T3 long lim-5%+exit(stop-6%) " + "| T4 short lim+5%+exit(stop+5.5%) in one OCA, cancel ENTRY only")
            announced = True
        log.info("[L1] bar={0} close={1} candle={2} state={3} pending={4} pos={5} canPlace={6}", bar_index, string.tostring(close, format.mintick), ("GREEN" if isGreen else "RED"), state, ("yes" if pending else "no"), strategy.position_size, ("yes" if canPlace else "no"))
        if canPlace and (not pending) and state < 4:
            if state == 0 and isGreen:
                lvlEntry = close * 0.95
                strategy.entry("T1", strategy.long, limit=lvlEntry, comment="T1 long lim@" + string.tostring(lvlEntry, format.mintick))
                placedBar = bar_index
                log.info("[L1] TEST 1 PLACE id=T1 LONG limit={0} (close {1}, -5%) — expect: " + "one NORMAL LO resting, filled=0, int order id", string.tostring(lvlEntry, format.mintick), string.tostring(close, format.mintick))
            elif state == 1:
                lvlEntry = close * 1.05
                strategy.entry("T2", strategy.short, limit=lvlEntry, comment="T2 short lim@" + string.tostring(lvlEntry, format.mintick))
                placedBar = bar_index
                log.info("[L1] TEST 2 PLACE id=T2 SHORT limit={0} (close {1}, +5%) — expect: " + "one NORMAL LO resting, filled=0", string.tostring(lvlEntry, format.mintick), string.tostring(close, format.mintick))
            elif state == 2 and isGreen:
                lvlEntry = close * 0.95
                lvlStop = close * 0.94
                strategy.entry("T3", strategy.long, limit=lvlEntry, comment="T3 long lim@" + string.tostring(lvlEntry, format.mintick))
                strategy.exit("X3", from_entry="T3", stop=lvlStop, comment_loss="T3 SL@" + string.tostring(lvlStop, format.mintick))
                placedBar = bar_index
                log.info("[L1] TEST 3 PLACE id=T3 LONG limit={0} + exit X3 stop={1} (-6%) — " + "expect: entry rests; the exit likely NEVER reaches DNSE " + "(bracket parked pending_entry until the entry fills)", string.tostring(lvlEntry, format.mintick), string.tostring(lvlStop, format.mintick))
            elif state == 3:
                lvlEntry = close * 1.05
                lvlStop = close * 1.055
                strategy.entry("T4", strategy.short, limit=lvlEntry, oca_name="t4", oca_type=strategy.oca.cancel, comment="T4 short lim@" + string.tostring(lvlEntry, format.mintick))
                strategy.exit("X4", from_entry="T4", stop=lvlStop, oca_name="t4", comment_loss="T4 SL@" + string.tostring(lvlStop, format.mintick))
                placedBar = bar_index
                log.info("[L1] TEST 4 PLACE id=T4 SHORT limit={0} + exit X4 stop={1} (+5.5%), " + "oca_name=t4 — next bar cancels ONLY T4, to see whether X4 goes with it", string.tostring(lvlEntry, format.mintick), string.tostring(lvlStop, format.mintick))
        if pending and bar_index > placedBar:
            if state == 0:
                strategy.cancel("T1")
                log.info("[L1] TEST 1 CANCEL id=T1 — expect: [BROKER] event CANCELLED, book clear")
            elif state == 1:
                strategy.cancel("T2")
                log.info("[L1] TEST 2 CANCEL id=T2 — expect: [BROKER] event CANCELLED, book clear")
            elif state == 2:
                strategy.cancel("T3")
                strategy.cancel("X3")
                log.info("[L1] TEST 3 CANCEL id=T3 and id=X3 (both) — expect: entry CANCELLED; " + "X3 may be a no-op if it never reached the venue")
            elif state == 3:
                strategy.cancel("T4")
                log.info("[L1] TEST 4 CANCEL id=T4 ONLY (X4 deliberately left alone) — " + "KEY OBSERVATION: does X4 also disappear from DNSE?")
            log.info("[L1] TEST {0} DONE — entry_lvl={1} stop_lvl={2}; state {3}->{4}", state + 1, string.tostring(lvlEntry, format.mintick), ("n/a" if na(lvlStop) else string.tostring(lvlStop, format.mintick)), state, state + 1)
            placedBar = na
            state += 1
            if state == 4:
                log.info("[L1] === ALL 4 TESTS DONE. Verify at the venue that NOTHING of ours " + "is still working, then stop the run. ===")
        if strategy.position_size != 0:
            log.error("[L1] !!! UNEXPECTED FILL: pos={0} avg={1} — cancelling all and flattening", strategy.position_size, string.tostring(strategy.position_avg_price, format.mintick))
            strategy.cancel_all()
            strategy.close_all(comment="SAFETY-FLATTEN")

    plot(state, "state", display=display.data_window)
    plot(strategy.position_size, "pos", display=display.data_window)
    plot(lvlEntry, "entry_lvl", display=display.data_window)
    plot(lvlStop, "stop_lvl", display=display.data_window)