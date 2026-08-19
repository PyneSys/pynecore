"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, bar_index, close, display, format, high, input, log, low, na, plot, script, strategy, string, time, timestamp
)
from pynecore.types import PersistentSeries, Series

@script.strategy('STAGED FILL (L3+oracle)', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    winStart=input.time(timestamp("2030-01-01T00:00:00+07:00"), "Trade window START"),
    winEnd=input.time(timestamp("2030-01-01T23:59:00+07:00"), "Trade window END"),
    startState=input.int(0, "Start at state (0=F1 .. 7=F8)", minval=0, maxval=7),
    operatorCloses=input.bool(True, "Operator closes the position manually")
):
    FILL_TIMEOUT_BARS = 6
    MAX_RETRIES = 1
    CLOSE_WAIT_BARS = 5

    state: PersistentSeries[int] = startState
    placedBar: PersistentSeries[int] = na(int)
    retries: PersistentSeries[int] = 0
    flattening: PersistentSeries[bool] = False
    waitBars: PersistentSeries[int] = 0
    halted: PersistentSeries[bool] = False
    protLvl: PersistentSeries[float] = na(float)
    announced: PersistentSeries[bool] = False

    isLongStage = state == 0 or state == 2 or state == 4 or state == 6
    started = time >= winStart
    canPlace = started and time <= winEnd
    pending = not na(placedBar)
    if started:
        if not announced:
            log.info("[F] === STAGED FILL — window open. 8 fill cases, 1 contract each, " + "flatten on sight, protection armed at placement. startState={0} ===", startState)
            announced = True
        log.info("[F] bar={0} close={1} state={2} pending={3} flattening={4} retries={5} pos={6}", bar_index, string.tostring(close, format.mintick), state, ("yes" if pending else "no"), ("yes" if flattening else "no"), retries, strategy.position_size)

    if canPlace and (not pending) and (not flattening) and state < 8:
        protLvl = (low[1] if isLongStage else high[1])
        if state == 0:
            strategy.entry("E", strategy.long, comment="F1 mkt long")
            log.info("[F] F1 PLACE long MARKET — expect marketable LO at the band edge, fills next bar")
        elif state == 1:
            strategy.entry("E", strategy.short, comment="F2 mkt short")
            log.info("[F] F2 PLACE short MARKET")
        elif state == 2:
            strategy.entry("E", strategy.long, stop=high[1], comment="F3 stop long")
            log.info("[F] F3 PLACE long STOP trigger={0} — fills on a break of the prior high", string.tostring(high[1], format.mintick))
        elif state == 3:
            strategy.entry("E", strategy.short, stop=low[1], comment="F4 stop short")
            log.info("[F] F4 PLACE short STOP trigger={0}", string.tostring(low[1], format.mintick))
        elif state == 4:
            strategy.entry("E", strategy.long, stop=high[1], limit=high[1] * 1.0002, comment="F5 stoplim long")
            log.info("[F] F5 PLACE long STOP-LIMIT stop={0} limit={1} — #14: must go out as ONE " + "STOP carrying a limit, not an OCO pair", string.tostring(high[1], format.mintick), string.tostring(high[1] * 1.0002, format.mintick))
        elif state == 5:
            strategy.entry("E", strategy.short, stop=low[1], limit=low[1] * 0.9998, comment="F6 stoplim short")
            log.info("[F] F6 PLACE short STOP-LIMIT stop={0} limit={1}", string.tostring(low[1], format.mintick), string.tostring(low[1] * 0.9998, format.mintick))
        elif state == 6:
            strategy.entry("E", strategy.long, stop=high[1], oca_name="f", oca_type=strategy.oca.cancel, comment="F7 oca up-near")
            strategy.entry("B", strategy.short, stop=low[1] * 0.95, oca_name="f", oca_type=strategy.oca.cancel, comment="F7 oca dn-far")
            log.info("[F] F7 PLACE OCA: near up-stop {0} (fills) + far dn-stop {1} — on the " + "fill the FAR leg must be CANCELLED (one-cancels-other on a real fill)", string.tostring(high[1], format.mintick), string.tostring(low[1] * 0.95, format.mintick))
        elif state == 7:
            strategy.entry("E", strategy.short, stop=low[1], oca_name="f", oca_type=strategy.oca.cancel, comment="F8 oca dn-near")
            strategy.entry("B", strategy.long, stop=high[1] * 1.05, oca_name="f", oca_type=strategy.oca.cancel, comment="F8 oca up-far")
            log.info("[F] F8 PLACE OCA mirror: near dn-stop {0} + far up-stop {1}", string.tostring(low[1], format.mintick), string.tostring(high[1] * 1.05, format.mintick))
        strategy.exit("P", from_entry="E", stop=protLvl, comment_loss="P SL@" + string.tostring(protLvl, format.mintick))
        placedBar = bar_index

    if pending and (not flattening) and strategy.position_size != 0:
        strategy.cancel("B")
        flattening = True
        waitBars = 0
        if operatorCloses:
            log.info("[F] F{0} FILLED pos={1} avg={2}  >>> OPERATOR: CLOSE THIS POSITION " + "NOW (DNSE app) <<<  the strategy will NOT close it — it waits for " + "flat, then advances", state + 1, strategy.position_size, string.tostring(strategy.position_avg_price, format.mintick))
        else:
            log.info("[F] F{0} FILLED pos={1} avg={2} -> ORACLE mode: strategy closes itself", state + 1, strategy.position_size, string.tostring(strategy.position_avg_price, format.mintick))
            strategy.close("E", comment="FLATTEN")

    if flattening and strategy.position_size != 0 and operatorCloses and (not halted):
        waitBars += 1
        if waitBars >= CLOSE_WAIT_BARS:
            log.error("[F] F{0} HALTED — still pos={1} after {2} bars. Either the position " + "is still open (CLOSE IT), or you closed it and the ENGINE DID NOT " + "DETECT the external close — that is a FINDING, record it. The " + "strategy will not close on a possibly-stale view.", state + 1, strategy.position_size, CLOSE_WAIT_BARS)
            halted = True
            strategy.cancel_all()
        else:
            log.info("[F] F{0} waiting for the manual close ({1}/{2}) — engine still " + "sees pos={3}", state + 1, waitBars, CLOSE_WAIT_BARS, strategy.position_size)

    if flattening and strategy.position_size == 0:
        strategy.cancel("P")
        log.info("[F] F{0} DONE — flat again; state {1}->{2}", state + 1, state, state + 1)
        placedBar = na
        flattening = False
        retries = 0
        state += 1
        if state == 8:
            log.info("[F] === ALL 8 FILL CASES DONE. Verify at the venue: flat, nothing " + "working. ===")

    if pending and (not flattening) and strategy.position_size == 0 and bar_index - placedBar >= FILL_TIMEOUT_BARS:
        strategy.cancel("E")
        strategy.cancel("B")
        strategy.cancel("P")
        if retries < MAX_RETRIES:
            retries += 1
            placedBar = na
            log.info("[F] F{0} TIMEOUT after {1} bars — cancelled all, RETRY {2}/{3} with " + "fresh levels next bar", state + 1, FILL_TIMEOUT_BARS, retries, MAX_RETRIES)
        else:
            log.info("[F] F{0} SKIP — no fill after retry (quiet tape); state {1}->{2}", state + 1, state, state + 1)
            placedBar = na
            retries = 0
            state += 1

    DOUBLE_CHECK_PCT = -0.3
    openPnlPct = (strategy.opentrades.profit_percent(strategy.opentrades - 1) if strategy.position_size != 0 else 0.0)
    if strategy.position_size != 0 and openPnlPct < DOUBLE_CHECK_PCT:
        strategy.cancel_all()
        if operatorCloses:
            log.error("[F] !!! DOUBLE-CHECK at {0}% — all orders cancelled. >>> OPERATOR: " + "CLOSE THE POSITION NOW <<< (the strategy does not auto-close in live " + "mode: a close on a stale view could REVERSE the position)", string.tostring(openPnlPct, "#.##"))
        else:
            log.error("[F] !!! DOUBLE-CHECK at {0}% — cancel_all + close_all (oracle)", string.tostring(openPnlPct, "#.##"))
            strategy.close_all(comment="DOUBLECHECK")

    plot(state, "state", display=display.data_window)
    plot(strategy.position_size, "pos", display=display.data_window)