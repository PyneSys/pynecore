"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, bar_index, close, display, format, input, log, na, plot, script, strategy, string, time, timestamp
)
from pynecore.types import PersistentSeries, Series

@script.strategy('BINANCE staged pyramid + scale-out', overlay=True, pyramiding=3, initial_capital=10000, default_qty_type=strategy.fixed, default_qty_value=0.001, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    winStart=input.time(timestamp("2030-01-01T00:00:00+00:00"), "Trade window START"),
    winEnd=input.time(timestamp("2030-01-01T23:59:00+00:00"), "Trade window END"),
    startState=input.int(0, "Start at state (0=BP1 .. 3=BP4)", minval=0, maxval=3)
):
    STEP_TIMEOUT_BARS = 8

    state: PersistentSeries[int] = startState
    markBar: PersistentSeries[int] = na(int)
    legs: PersistentSeries[int] = 0
    announced: PersistentSeries[bool] = False
    tpLevel: PersistentSeries[float] = na(float)
    posMark: PersistentSeries[float] = na(float)

    started = time >= winStart
    canPlace = started and time <= winEnd
    pos = strategy.position_size
    if started:
        if not announced:
            log.info("[BP] === PYRAMID + SCALE-OUT — 3-unit stack then 3 different partial teardowns. startState={0} ===", startState)
            announced = True
        log.info("[BP] bar={0} close={1} state={2} legs={3} pos={4} avg={5}", bar_index, string.tostring(close, format.mintick), state, legs, string.tostring(pos, "#.#####"), string.tostring(strategy.position_avg_price, format.mintick))

    if canPlace and state == 0 and legs < 3 and pos == legs * 0.001:
        legs += 1
        entryId = "E" + string.tostring(legs)
        strategy.entry(entryId, strategy.long, comment="BP1 leg " + string.tostring(legs))
        markBar = bar_index
        log.info("[BP] BP1 PLACE market entry {0} (leg {1}/3) — expect pos to reach {2}", entryId, legs, string.tostring(legs * 0.001, "#.#####"))

    if state == 0 and legs == 3 and pos >= 0.003:
        log.info("[BP] BP1 DONE — 3-unit stack built, pos={0} avg={1}; ledger must show 3 fills", string.tostring(pos, "#.#####"), string.tostring(strategy.position_avg_price, format.mintick))
        state = 1
        markBar = na

    if canPlace and state == 1 and na(markBar):
        tpLevel = close * 1.0003
        strategy.exit("X1", from_entry="E1", qty=0.001, limit=tpLevel, comment_profit="BP2 tp")
        markBar = bar_index
        posMark = pos
        log.info("[BP] BP2 PLACE partial exit X1 from E1 qty=0.001 limit={0} — exit qty < pos({1}); " + "the other 2 units must stay open", string.tostring(tpLevel, format.mintick), string.tostring(pos, "#.#####"))

    if state == 1 and (not na(markBar)) and (not na(posMark)) and pos < posMark:
        log.info("[BP] BP2 DONE — partial exit filled, pos={0} (2 units remain)", string.tostring(pos, "#.#####"))
        state = 2
        markBar = na

    if state == 1 and (not na(markBar)) and bar_index - markBar >= STEP_TIMEOUT_BARS:
        strategy.cancel("X1")
        log.info("[BP] BP2 TIMEOUT after {0} bars — TP never filled; cancelling and advancing", STEP_TIMEOUT_BARS)
        state = 2
        markBar = na

    if canPlace and state == 2 and na(markBar):
        strategy.close("E2", qty=0.001, comment="BP3 partial close")
        markBar = bar_index
        posMark = pos
        log.info("[BP] BP3 PLACE market close of E2 qty=0.001 — partial close against a " + "multi-entry position (FIFO attribution)")

    if state == 2 and (not na(markBar)) and (not na(posMark)) and pos < posMark:
        log.info("[BP] BP3 DONE — partial close filled, pos={0} (1 unit remains)", string.tostring(pos, "#.#####"))
        state = 3
        markBar = na

    if state == 2 and (not na(markBar)) and bar_index - markBar >= STEP_TIMEOUT_BARS:
        log.error("[BP] BP3 TIMEOUT after {0} bars — partial close never reduced the position; advancing to close_all", STEP_TIMEOUT_BARS)
        state = 3
        markBar = na

    if canPlace and state == 3 and na(markBar) and pos > 0:
        strategy.close_all(comment="BP4 close_all")
        markBar = bar_index
        log.info("[BP] BP4 PLACE close_all — tearing down the remainder ({0})", string.tostring(pos, "#.#####"))

    if state == 3 and (not na(markBar)) and pos == 0:
        log.info("[BP] BP4 DONE — flat. === ALL PYRAMID CASES DONE. Verify at the venue: " + "flat, nothing working, ledger net_base back to 0. ===")
        state = 4
        markBar = na

    plot(state, "state", display=display.data_window)
    plot(pos, "pos", display=display.data_window)
    plot(legs, "legs", display=display.data_window)