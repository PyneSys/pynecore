"""
@pyne

Binance testnet staged NO-FILL probe (Live-B1).

Every 5 bars: place a far-below limit buy (30% under market — cannot fill),
cancel it 2 bars later. Exercises place -> open -> cancel round-trips on the
venue without ever holding a position. Grade from the VENUE record
(fetch_order / the run's broker store), never the run log alone.

Run (1m for fast bars):
    .venv/bin/python plugins/binance/tools/l0_gate.py           # MUST exit 0
    .venv/bin/pyne run plugins/binance/testing/probe_nofill.py \
        binance_broker:BTC/USDT@1 --broker
"""
from pynecore.lib import bar_index, barstate, close, log, script, strategy


@script.strategy("Binance no-fill probe", overlay=True)
def main():
    if not barstate.isrealtime:
        return
    phase = bar_index % 5
    if phase == 1:
        log.info("PROBE place: far limit buy @ {0}", close * 0.7)
        strategy.entry("Probe", strategy.long, qty=0.001, limit=close * 0.7)
    elif phase == 3:
        log.info("PROBE cancel")
        strategy.cancel("Probe")
