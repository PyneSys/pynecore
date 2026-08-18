"""
@pyne

Binance testnet staged place+cancel — no-fill probe, ported from the DNSE
suite's ``live_staged_place_cancel`` (T1–T13) with the SPOT adaptations:

* LONG-ONLY: spot cannot short (``short_selling`` unsupported), so DNSE's
  short cases become second long variants (state 1) or keep their structure
  with long direction (states 3, 9, 11).
* States 2/3/4 additionally MEASURE whether a pre-fill ``strategy.exit``
  reaches a spot venue at all (a resting sell needs base inventory — the
  testnet account holds 1 BTC of foreign baseline).
* State 9 (oca.cancel x3) is the KEY honesty check for the plugin's
  ``oca_cancel = NATIVE`` declaration: entry OCA groups have NO venue-side
  link on Binance (only exit OCO lists do).

Every order 0.001 BTC, >=4% from market: rests, cannot fill.
States: 0=B1 1=B2 2=B3 3=B4 4=B5 5=B6 6=B7 7=B8 8=B9 9=B11 10=B12 11=B13.
"""

from pynecore.lib import (
    bar_index, close, display, format, input, log, na, plot, script, strategy,
    string, time, timestamp
)
from pynecore.types import PersistentSeries


@script.strategy('BINANCE staged place+cancel', overlay=True, pyramiding=0,
                 initial_capital=10000, default_qty_type=strategy.fixed,
                 default_qty_value=0.001, calc_on_every_tick=False,
                 process_orders_on_close=False)
def main(
    winStart=input.time(timestamp("2030-01-01T00:00:00+00:00"), "Trade window START"),
    winEnd=input.time(timestamp("2030-01-01T23:59:00+00:00"), "Trade window END"),
    startState=input.int(0, "Start at state (0=B1 .. 11=B13)", minval=0, maxval=11)
):
    state: PersistentSeries[int] = startState
    placedBar: PersistentSeries[int] = na(int)
    stageStep: PersistentSeries[int] = 0
    lvlEntry: PersistentSeries[float] = na(float)
    lvlStop: PersistentSeries[float] = na(float)
    lvlTP: PersistentSeries[float] = na(float)
    lvlAmend: PersistentSeries[float] = na(float)
    announced: PersistentSeries[bool] = False

    pending = not na(placedBar)
    started = time >= winStart
    canPlace = started and time <= winEnd
    if started:
        if not announced:
            log.info("[B1] === BINANCE STAGED PLACE/CANCEL — long-only spot port of DNSE T1-T13. "
                     + "qty 0.001 BTC, every order >=4% away. startState={0} ===", startState)
            announced = True
        log.info("[B1] bar={0} close={1} state={2} step={3} pending={4} pos={5}",
                 bar_index, string.tostring(close, format.mintick), state, stageStep,
                 ("yes" if pending else "no"), strategy.position_size)
        if canPlace and (not pending) and state < 12:
            if state == 0:
                lvlEntry = close * 0.95
                strategy.entry("B1", strategy.long, limit=lvlEntry, comment="B1")
                placedBar = bar_index
                log.info("[B1] TEST B1 PLACE long limit={0} (-5%) — expect one LIMIT GTC, filled=0",
                         string.tostring(lvlEntry, format.mintick))
            elif state == 1:
                lvlEntry = close * 0.96
                strategy.entry("B2", strategy.long, limit=lvlEntry, comment="B2")
                placedBar = bar_index
                log.info("[B1] TEST B2 PLACE long limit={0} (-4%) — spot is long-only; this replaces "
                         + "DNSE's short-side T2", string.tostring(lvlEntry, format.mintick))
            elif state == 2:
                lvlEntry = close * 0.95
                lvlStop = close * 0.94
                strategy.entry("B3", strategy.long, limit=lvlEntry, comment="B3")
                strategy.exit("X3", from_entry="B3", stop=lvlStop, comment_loss="X3")
                placedBar = bar_index
                log.info("[B1] TEST B3 PLACE limit={0} + X3 stop={1} — MEASURE: does the pre-fill "
                         + "exit reach the spot venue? then cancel both explicitly",
                         string.tostring(lvlEntry, format.mintick),
                         string.tostring(lvlStop, format.mintick))
            elif state == 3:
                lvlEntry = close * 0.95
                lvlStop = close * 0.94
                strategy.entry("B4", strategy.long, limit=lvlEntry, oca_name="b4",
                               oca_type=strategy.oca.cancel, comment="B4")
                strategy.exit("X4", from_entry="B4", stop=lvlStop, oca_name="b4",
                              comment_loss="X4")
                placedBar = bar_index
                log.info("[B1] TEST B4 PLACE limit={0} + X4 stop={1} (oca) — next bar cancels "
                         + "ENTRY ONLY; observe whether X4 remains (DNSE: orphan expected, #19)",
                         string.tostring(lvlEntry, format.mintick),
                         string.tostring(lvlStop, format.mintick))
            elif state == 4:
                lvlEntry = close * 0.95
                lvlTP = close * 1.05
                lvlStop = close * 0.94
                strategy.entry("B5", strategy.long, limit=lvlEntry, comment="B5")
                strategy.exit("X5", from_entry="B5", limit=lvlTP, stop=lvlStop,
                              comment_profit="X5tp", comment_loss="X5sl")
                placedBar = bar_index
                log.info("[B1] TEST B5 PLACE limit={0} + X5 tp={1}/sl={2} — the tp+sl pair routes "
                         + "to the NATIVE spot OCO (orderList). Next bar cancels ENTRY ONLY; "
                         + "observe the OCO legs", string.tostring(lvlEntry, format.mintick),
                         string.tostring(lvlTP, format.mintick),
                         string.tostring(lvlStop, format.mintick))
            elif state == 5:
                lvlEntry = close * 0.95
                lvlAmend = close * 0.955
                strategy.entry("B6", strategy.long, limit=lvlEntry, comment="B6")
                placedBar = bar_index
                log.info("[B1] TEST B6 PLACE long limit={0} (-5%) — NEXT bar re-issues at {1} "
                         + "(-4.5%): plugin amend is cancel+replace (SOFTWARE); expect a fresh "
                         + "venue order id at the new price",
                         string.tostring(lvlEntry, format.mintick),
                         string.tostring(lvlAmend, format.mintick))
            elif state == 6:
                lvlStop = close * 1.05
                lvlAmend = close * 1.045
                strategy.entry("B7", strategy.long, stop=lvlStop, comment="B7")
                placedBar = bar_index
                log.info("[B1] TEST B7 PLACE buy-STOP {0} (+5%) — NEXT bar re-issues at {1} "
                         + "(+4.5%): amend on a STOP_LOSS_LIMIT (DNSE 500'd here, #18; Binance "
                         + "should cancel+replace cleanly)",
                         string.tostring(lvlStop, format.mintick),
                         string.tostring(lvlAmend, format.mintick))
            elif state == 7:
                lvlEntry = close * 0.95
                lvlStop = close * 1.05
                strategy.entry("B8a", strategy.long, limit=lvlEntry, comment="B8a")
                strategy.entry("B8b", strategy.long, stop=lvlStop, comment="B8b")
                placedBar = bar_index
                log.info("[B1] TEST B8 PLACE B8a limit={0} AND B8b stop={1} — next bar "
                         + "strategy.cancel_all(): exercises the native bulk cancel + the "
                         + "expected-cancel sink", string.tostring(lvlEntry, format.mintick),
                         string.tostring(lvlStop, format.mintick))
            elif state == 8:
                lvlEntry = close * 0.95
                strategy.order("B9", strategy.long, limit=lvlEntry, comment="B9")
                placedBar = bar_index
                log.info("[B1] TEST B9 PLACE via strategy.order() long limit={0} — expect "
                         + "identical LIMIT routing to entry()",
                         string.tostring(lvlEntry, format.mintick))
            elif state == 9:
                lvlEntry = close * 0.95
                lvlAmend = close * 0.94
                lvlStop = close * 1.05
                strategy.entry("Ga", strategy.long, limit=lvlEntry, oca_name="g11",
                               oca_type=strategy.oca.cancel, comment="Ga")
                strategy.entry("Gb", strategy.long, limit=lvlAmend, oca_name="g11",
                               oca_type=strategy.oca.cancel, comment="Gb")
                strategy.entry("Gc", strategy.long, stop=lvlStop, oca_name="g11",
                               oca_type=strategy.oca.cancel, comment="Gc")
                placedBar = bar_index
                log.info("[B1] TEST B11 PLACE oca.cancel x3: Ga limit={0} + Gb limit={1} + Gc "
                         + "stop={2} — next bar cancels ONLY Ga. KEY: Gb and Gc must REMAIN "
                         + "(OCA fires on FILL, not cancel; entry groups have NO venue link "
                         + "on Binance)", string.tostring(lvlEntry, format.mintick),
                         string.tostring(lvlAmend, format.mintick),
                         string.tostring(lvlStop, format.mintick))
            elif state == 10:
                lvlEntry = close * 0.95
                lvlStop = close * 1.05
                strategy.entry("Ra", strategy.long, limit=lvlEntry, oca_name="g12",
                               oca_type=strategy.oca.reduce, comment="Ra")
                strategy.entry("Rb", strategy.long, stop=lvlStop, oca_name="g12",
                               oca_type=strategy.oca.reduce, comment="Rb")
                placedBar = bar_index
                log.info("[B1] TEST B12 PLACE oca.reduce x2: Ra limit={0} + Rb stop={1} — BOTH "
                         + "must rest FULL qty at the venue; next bar cancel_all sweeps them",
                         string.tostring(lvlEntry, format.mintick),
                         string.tostring(lvlStop, format.mintick))
            elif state == 11:
                lvlEntry = close * 0.95
                lvlAmend = close * 0.94
                strategy.entry("Na", strategy.long, limit=lvlEntry, oca_name="g13",
                               oca_type=strategy.oca.none, comment="Na")
                strategy.entry("Nb", strategy.long, limit=lvlAmend, oca_name="g13",
                               oca_type=strategy.oca.none, comment="Nb")
                placedBar = bar_index
                log.info("[B1] TEST B13 PLACE oca.none (SHARED name g13): Na limit={0} + Nb "
                         + "limit={1} — next bar cancels ONLY Na; Nb must be untouched",
                         string.tostring(lvlEntry, format.mintick),
                         string.tostring(lvlAmend, format.mintick))
        if pending and bar_index > placedBar:
            if state == 5 and stageStep == 0:
                strategy.entry("B6", strategy.long, limit=lvlAmend, comment="B6amend")
                stageStep = 1
                placedBar = bar_index
                log.info("[B1] TEST B6 AMEND -> limit {0} — expect cancel+replace, new venue id",
                         string.tostring(lvlAmend, format.mintick))
            elif state == 6 and stageStep == 0:
                strategy.entry("B7", strategy.long, stop=lvlAmend, comment="B7amend")
                stageStep = 1
                placedBar = bar_index
                log.info("[B1] TEST B7 AMEND -> stop {0} — cancel+replace on a conditional; the "
                         + "run must continue and the order must still cancel next bar",
                         string.tostring(lvlAmend, format.mintick))
            elif state == 9 and stageStep == 0:
                strategy.cancel("Ga")
                stageStep = 1
                placedBar = bar_index
                log.info("[B1] TEST B11 CANCEL Ga ONLY — Gb and Gc must still be resting")
            elif state == 11 and stageStep == 0:
                strategy.cancel("Na")
                stageStep = 1
                placedBar = bar_index
                log.info("[B1] TEST B13 CANCEL Na ONLY — Nb must be untouched (oca.none)")
            else:
                if state == 0:
                    strategy.cancel("B1")
                    log.info("[B1] TEST B1 CANCEL")
                elif state == 1:
                    strategy.cancel("B2")
                    log.info("[B1] TEST B2 CANCEL")
                elif state == 2:
                    strategy.cancel("B3")
                    strategy.cancel("X3")
                    log.info("[B1] TEST B3 CANCEL B3 and X3 (both, explicitly)")
                elif state == 3:
                    strategy.cancel("B4")
                    log.info("[B1] TEST B4 CANCEL B4 ONLY — observe X4 at the venue")
                elif state == 4:
                    strategy.cancel("B5")
                    log.info("[B1] TEST B5 CANCEL B5 ONLY — observe the X5 OCO legs")
                elif state == 5:
                    strategy.cancel("B6")
                    log.info("[B1] TEST B6 CANCEL (post-amend)")
                elif state == 6:
                    strategy.cancel("B7")
                    log.info("[B1] TEST B7 CANCEL (post-amend)")
                elif state == 7:
                    strategy.cancel_all()
                    log.info("[B1] TEST B8 strategy.cancel_all() — both must go; book clear")
                elif state == 8:
                    strategy.cancel("B9")
                    log.info("[B1] TEST B9 CANCEL")
                elif state == 9:
                    strategy.cancel("Gb")
                    strategy.cancel("Gc")
                    log.info("[B1] TEST B11 CANCEL Gb and Gc (the survivors)")
                elif state == 10:
                    strategy.cancel_all()
                    log.info("[B1] TEST B12 cancel_all — Ra and Rb must both go")
                elif state == 11:
                    strategy.cancel("Nb")
                    log.info("[B1] TEST B13 CANCEL Nb (the survivor)")
                log.info("[B1] TEST state {0}->{1}", state, state + 1)
                placedBar = na
                stageStep = 0
                state += 1
                if state == 12:
                    log.info("[B1] === ALL TESTS DONE (B1-B9 + B11-B13). Verify at the venue "
                             + "that NOTHING of ours is still working, then stop the run. ===")
        if strategy.position_size != 0:
            log.error("[B1] !!! UNEXPECTED FILL: pos={0} — cancelling all and flattening",
                      strategy.position_size)
            strategy.cancel_all()
            strategy.close_all(comment="SAFETY-FLATTEN")

    plot(state, "state", display=display.data_window)
    plot(strategy.position_size, "pos", display=display.data_window)
