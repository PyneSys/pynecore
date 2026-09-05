"""
@pyne

A ta.tr() inside a conditional branch still measures against the real previous bar.

Measured on TradingView with the "Follow Line Indicator" corpus script, whose
`atr(ATRperiod)` calls sit inside `if BBSignal == ...` blocks: a re-simulation of
the script matches TradingView on all 28466 comparable bars only when the rma
accumulator is per-call-site AND call-gated while the previous close is read from
the global series, which advances on every bar. The same simulation with a
call-gated previous close reproduces PyneCore's old output instead.

``tr`` used to keep the previous close in its own ``Persistent``, so a call inside
an `if` branch measured the gap to the previous CALL and the whole indicator ran
0.6% matched. The previous close now comes from ``lib._last_close``, published by
the runner every bar, and ``tr`` carries no state at all -- while ``atr``'s rma
stays per-call-site and call-gated, which is what TradingView does.
"""
from pynecore.lib import script, ta, bar_index

# Flat bars (high == low == close) so tr collapses to |close - previous close|,
# on a ladder whose steps all differ -- the gap to the previous bar and the gap to
# the previous even bar can then never coincide by accident.
__test_helper_CLOSES = (0.0, 10.0, 30.0, 60.0, 100.0, 150.0, 210.0, 280.0)


@script.indicator(title="Conditional tr")
def main():
    every = ta.tr(True)
    every_atr = ta.atr(3)
    gated = -1.0
    gated_atr = -1.0
    if bar_index % 2 == 0:
        gated = ta.tr(True)
        gated_atr = ta.atr(3)
    return {"every": every, "gated": gated,
            "every_atr": every_atr, "gated_atr": gated_atr}


def __test_helper_rows():
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV

    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    return [OHLCV(timestamp=base_ts + bar * 1800, open=close, high=close,
                  low=close, close=close, volume=10.0)
            for bar, close in enumerate(__test_helper_CLOSES)]


def __test_gated_tr_uses_the_previous_bar__(runner):
    """ tr() called on even bars only still reads the bar before, not the call before """
    seen = 0
    for bar, (_candle, plot) in enumerate(runner(iter(__test_helper_rows())).run_iter()):
        if bar % 2:
            assert plot["gated"] == -1.0, f"bar {bar}: the branch must not run"
            continue
        assert plot["gated"] == plot["every"], (
            f"bar {bar}: gated tr {plot['gated']} != every-bar tr {plot['every']}")
        seen += 1
    assert seen == 4


def __test_tr_is_the_gap_to_the_previous_bar__(runner):
    """ The flat bars make tr the step of the close ladder, na-handled on bar 0 """
    for bar, (_candle, plot) in enumerate(runner(iter(__test_helper_rows())).run_iter()):
        expected = 0.0 if bar == 0 else __test_helper_CLOSES[bar] - __test_helper_CLOSES[bar - 1]
        assert plot["every"] == expected, f"bar {bar}: tr {plot['every']} != {expected}"


def __test_gated_atr_accumulator_stays_call_gated__(runner):
    """ atr's rma still advances per call: a length of 3 needs 3 CALLS to leave na """
    for bar, (_candle, plot) in enumerate(runner(iter(__test_helper_rows())).run_iter()):
        every_atr = plot["every_atr"]
        assert (every_atr != every_atr) == (bar < 2), (
            f"bar {bar}: every-bar atr(3) should be na only before bar 2, got {every_atr}")
        if bar % 2:
            continue
        gated_atr = plot["gated_atr"]
        # Fed on bars 0, 2, 4 -- so it is still na on bar 2, where the every-bar
        # one already has its three values.
        assert (gated_atr != gated_atr) == (bar < 4), (
            f"bar {bar}: gated atr(3) should be na only before bar 4, got {gated_atr}")
