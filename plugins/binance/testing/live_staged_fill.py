"""
@pyne

Binance testnet staged FILL test — ported from the DNSE suite's
``live_staged_fill`` (F1–F8), long-only spot subset. Backtest mode over a past
window IS the oracle; the live run is graded against it.

Cases (0.001 BTC each; protection SL armed at placement, flatten on sight):

* BF1 (state 0) — market long fill -> flatten          (DNSE F1)
* BF3 (state 1) — stop-entry long, trigger prior high  (DNSE F3)
* BF5 (state 2) — stop-LIMIT long (#14 analogue: one STOP_LOSS_LIMIT order,
  not an OCO pair)                                     (DNSE F5)
* BF7 (state 3) — entry OCA pair long/long: near up-stop fills, far up-stop
  must be CANCELLED on the fill. KEY: entry OCA has NO venue link on Binance;
  this measures whether the engine's sibling-cancel runs under
  ``oca_cancel = NATIVE``                              (DNSE F7)
* BF9 (state 4) — market fill -> NATIVE OCO bracket (tight TP +0.05% / far SL
  -3%) -> TP leg fills -> venue must auto-cancel the SL sibling (spot
  orderList one-cancels-other on a REAL fill). No DNSE equivalent — this is
  the Live-B2 case from the plan.

DNSE F2/F4/F6/F8 (short side) are N/A on spot.
"""

from pynecore.lib import (
    bar_index, close, display, format, high, input, log, na, plot, script,
    strategy, string, time, timestamp
)
from pynecore.types import PersistentSeries


@script.strategy('BINANCE staged fill', overlay=True, pyramiding=0,
                 initial_capital=10000, default_qty_type=strategy.fixed,
                 default_qty_value=0.001, calc_on_every_tick=False,
                 process_orders_on_close=False)
def main(
    winStart=input.time(timestamp("2030-01-01T00:00:00+00:00"), "Trade window START"),
    winEnd=input.time(timestamp("2030-01-01T23:59:00+00:00"), "Trade window END"),
    startState=input.int(0, "Start at state (0=BF1 .. 4=BF9)", minval=0, maxval=4)
):
    FILL_TIMEOUT_BARS = 6
    MAX_RETRIES = 1

    state: PersistentSeries[int] = startState
    placedBar: PersistentSeries[int] = na(int)
    retries: PersistentSeries[int] = 0
    flattening: PersistentSeries[bool] = False
    bracketed: PersistentSeries[bool] = False
    protLvl: PersistentSeries[float] = na(float)
    announced: PersistentSeries[bool] = False

    started = time >= winStart
    canPlace = started and time <= winEnd
    pending = not na(placedBar)
    if started:
        if not announced:
            log.info("[BF] === BINANCE STAGED FILL — 5 long-only cases, 0.001 BTC each, "
                     + "flatten on sight. startState={0} ===", startState)
            announced = True
        log.info("[BF] bar={0} close={1} state={2} pending={3} flattening={4} retries={5} pos={6}",
                 bar_index, string.tostring(close, format.mintick), state,
                 ("yes" if pending else "no"), ("yes" if flattening else "no"),
                 retries, strategy.position_size)

    if canPlace and (not pending) and (not flattening) and state < 5:
        if state == 0:
            strategy.entry("E", strategy.long, comment="BF1 mkt long")
            strategy.exit("P", from_entry="E", stop=close * 0.97, comment_loss="P")
            log.info("[BF] BF1 PLACE long MARKET — fills next bar; protection SL -3%")
        elif state == 1:
            strategy.entry("E", strategy.long, stop=high[1], comment="BF3 stop long")
            strategy.exit("P", from_entry="E", stop=close * 0.97, comment_loss="P")
            log.info("[BF] BF3 PLACE long STOP trigger={0} — fills on a break of the prior high",
                     string.tostring(high[1], format.mintick))
        elif state == 2:
            strategy.entry("E", strategy.long, stop=high[1], limit=high[1] * 1.0002,
                           comment="BF5 stoplim long")
            strategy.exit("P", from_entry="E", stop=close * 0.97, comment_loss="P")
            log.info("[BF] BF5 PLACE long STOP-LIMIT stop={0} limit={1} — must go out as ONE "
                     + "STOP_LOSS_LIMIT, not a pair", string.tostring(high[1], format.mintick),
                     string.tostring(high[1] * 1.0002, format.mintick))
        elif state == 3:
            strategy.entry("E", strategy.long, stop=high[1], oca_name="f",
                           oca_type=strategy.oca.cancel, comment="BF7 oca near")
            strategy.entry("B", strategy.long, stop=high[1] * 1.05, oca_name="f",
                           oca_type=strategy.oca.cancel, comment="BF7 oca far")
            strategy.exit("P", from_entry="E", stop=close * 0.97, comment_loss="P")
            log.info("[BF] BF7 PLACE entry-OCA long/long: near up-stop {0} (fills) + far "
                     + "up-stop {1} — on the fill the FAR leg must be CANCELLED (engine "
                     + "sibling-cancel; no venue link for entry groups)",
                     string.tostring(high[1], format.mintick),
                     string.tostring(high[1] * 1.05, format.mintick))
        elif state == 4:
            strategy.entry("E", strategy.long, comment="BF9 mkt for bracket")
            log.info("[BF] BF9 PLACE long MARKET — on the fill a NATIVE OCO bracket is armed: "
                     + "tight TP +0.05%% / far SL -3%%; the TP leg must fill and the venue "
                     + "must auto-cancel the SL sibling")
        placedBar = bar_index

    if state == 4 and strategy.position_size > 0 and not bracketed:
        strategy.exit("P", from_entry="E", limit=close * 1.0005, stop=close * 0.97,
                      comment_profit="Ptp", comment_loss="Psl")
        bracketed = True
        log.info("[BF] BF9 BRACKET armed: tp={0} sl={1} -> native OCO orderList",
                 string.tostring(close * 1.0005, format.mintick),
                 string.tostring(close * 0.97, format.mintick))

    if pending and (not flattening) and strategy.position_size != 0 and state != 4:
        log.info("[BF] BF state={0} FILLED pos={1} avg={2} -> FLATTEN now (close E, cancel P "
                 + "and B explicitly)", state, strategy.position_size,
                 string.tostring(strategy.position_avg_price, format.mintick))
        strategy.close("E", comment="FLATTEN")
        strategy.cancel("P")
        strategy.cancel("B")
        flattening = True

    if state == 4 and bracketed and strategy.position_size == 0:
        log.info("[BF] BF9 DONE — bracket resolved (TP fill expected; verify the SL sibling "
                 + "was CANCELLED at the venue). ALL CASES DONE.")
        placedBar = na
        bracketed = False
        state += 1

    if flattening and strategy.position_size == 0:
        log.info("[BF] state {0}->{1} — flat again", state, state + 1)
        placedBar = na
        flattening = False
        retries = 0
        state += 1

    if (pending and (not flattening) and strategy.position_size == 0
            and bar_index - placedBar >= FILL_TIMEOUT_BARS and state < 4):
        strategy.cancel("E")
        strategy.cancel("B")
        strategy.cancel("P")
        if retries < MAX_RETRIES:
            retries += 1
            placedBar = na
            log.info("[BF] state={0} TIMEOUT after {1} bars — cancelled all, RETRY {2}/{3}",
                     state, FILL_TIMEOUT_BARS, retries, MAX_RETRIES)
        else:
            log.info("[BF] state={0} SKIP — no fill after retry; state {1}->{2}",
                     state, state, state + 1)
            placedBar = na
            retries = 0
            state += 1

    plot(state, "state", display=display.data_window)
    plot(strategy.position_size, "pos", display=display.data_window)
