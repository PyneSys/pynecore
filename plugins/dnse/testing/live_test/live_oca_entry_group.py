"""
@pyne

This code was compiled by Pine2Pyne — the Pine Script to PyneCore's Python compiler.
"""

from pynecore.lib import (
    NA, bar_index, close, display, format, high, input, log, na, plot, script, strategy, string, time, timestamp
)
from pynecore.types import PersistentSeries, Series

@script.strategy('LIVE OCA entry group', overlay=True, pyramiding=0, initial_capital=500000000, default_qty_type=strategy.fixed, default_qty_value=1, margin_long=18.48, margin_short=18.48, slippage=1, calc_on_every_tick=False, process_orders_on_close=False)
def main(
    winStart=input.time(timestamp("2030-01-01T00:00:00+07:00"), "Trade window START"),
    winEnd=input.time(timestamp("2030-01-01T23:59:00+07:00"), "Trade window END"),
    farAwayPct=input.float(5.0, "Far leg distance %", minval=0.1, maxval=10.0),
    operatorCloses=input.bool(True, "Operator closes the position manually")
):
    NOFILL_TIMEOUT_BARS = 10

    placedBar: PersistentSeries[int] = na(int)
    filledBar: PersistentSeries[int] = na(int)
    done: PersistentSeries[bool] = False
    announced: PersistentSeries[bool] = False

    started = time >= winStart
    canPlace = started and time <= winEnd
    pending = not na(placedBar)
    if started and (not announced):
        log.info("[OCA] === F11 OCA entry-group cancel — near buy-stop + far sell-stop " + "-{0}%, NO explicit far cancel. Oracle: far cancelled on near fill. ===", farAwayPct)
        announced = True

    if started:
        log.info("[OCA] bar={0} close={1} pos={2} pending={3} filledBar={4}", bar_index, string.tostring(close, format.mintick), strategy.position_size, ("yes" if pending else "no"), filledBar)

    if canPlace and (not pending) and (not done):
        nearTrig = high[1]
        farTrig = close * (1 - farAwayPct / 100)
        strategy.entry("NEAR", strategy.long, stop=nearTrig, oca_name="g33", oca_type=strategy.oca.cancel, comment="F11near")
        strategy.entry("FAR", strategy.short, stop=farTrig, oca_name="g33", oca_type=strategy.oca.cancel, comment="F11far")
        strategy.exit("P", from_entry="NEAR", stop=close * 0.997, comment_loss="F11-protect")
        placedBar = bar_index
        log.info("[OCA] PLACE near buy-stop={0} + far sell-stop={1} (oca.cancel 'g33') — " + "on the near FILL the far leg must be CANCELLED by the engine cascade; " + "watch [BROKER] for the cascade cancel, then verify at the venue", string.tostring(nearTrig, format.mintick), string.tostring(farTrig, format.mintick))

    if pending and na(filledBar) and strategy.position_size > 0:
        filledBar = bar_index
        if operatorCloses:
            log.info("[OCA] NEAR FILLED pos={0} avg={1}  >>> OPERATOR: CLOSE THIS POSITION " + "NOW (DNSE app) <<<  this script cancels only the protective exit; " + "the FAR leg is left untouched because its venue state IS the " + "measurement", strategy.position_size, string.tostring(strategy.position_avg_price, format.mintick))
        else:
            log.info("[OCA] NEAR FILLED pos={0} avg={1} -> ORACLE mode: strategy closes itself", strategy.position_size, string.tostring(strategy.position_avg_price, format.mintick))
            strategy.close_all(comment="F11-flat")

    if pending and na(filledBar) and bar_index - placedBar >= NOFILL_TIMEOUT_BARS and (not done):
        log.info("[OCA] NO FILL after {0} bars — nothing to measure; cancelling NEAR, FAR " + "and the protective exit, run over", NOFILL_TIMEOUT_BARS)
        strategy.cancel_all()
        done = True

    if strategy.position_size < 0:
        strategy.cancel_all()
        if operatorCloses:
            log.error("[OCA] !!! POSITION FLIPPED SHORT pos={0} — the FAR leg FILLED after " + "the near fill: cascade ABSENT (#33 regression). Orders cancelled. " + ">>> OPERATOR: CLOSE THE POSITION NOW <<<", strategy.position_size)
        else:
            log.error("[OCA] !!! POSITION FLIPPED SHORT pos={0} — cascade ABSENT (#33 " + "regression). Flattening (oracle).", strategy.position_size)
            strategy.close_all(comment="F11-flipguard")
        done = True

    if (not na(filledBar)) and strategy.position_size == 0 and (not done):
        strategy.cancel("P")
        done = True
        log.error("[OCA] === F11 DONE — GRADE NOW at the venue: FAR must read Canceled " + "(engine cascade). If FAR is still WORKING the cascade failed (#33 " + "regression) — record it, then cancel FAR manually. ===")

    plot(strategy.position_size, "pos", display=display.data_window)