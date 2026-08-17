"""
@pyne

Stateful ta builtin variables are engine-level series, not per-reference machines.

Measured on TradingView (probes m570/m572, BINANCE:BTCUSDT 30m, 28505+ bars): a
builtin *variable* read inside an ``if`` — ``ta.nvi``, ``ta.obv``, ``ta.pvt``,
``ta.wad``, ``ta.accdist``, ``ta.pvi``, ``ta.vwap`` — agrees with an unconditional
read of the same variable on every gated bar. Pine keeps one engine series per
builtin variable; only *function*-form builtins gate their state per call site
(see ``test_365_conditional_tr_prev_close`` and
``test_366_conditional_prev_bar_prices``).

The ``TaVariableHoistTransformer`` implements this: every referenced stateful
builtin variable is evaluated once, unconditionally, at the top of ``main``, and
all reference sites — gated branches and module-level helper functions alike —
read that cached value. The unconditional ``every_*`` plots and the distinct-value
guard are the controls: without them a series stuck at its seed would satisfy the
equality vacuously.
"""
from pynecore.lib import script, ta, bar_index

# (close, volume): volume alternates around the previous bar so nvi/pvi update on
# real bars, and close moves both ways so obv accumulates in both directions
BARS = ((100.0, 50.0), (102.0, 40.0), (101.0, 60.0), (105.0, 30.0), (104.0, 70.0),
        (108.0, 20.0), (107.0, 90.0), (110.0, 35.0), (109.0, 80.0), (113.0, 25.0),
        (112.0, 95.0), (116.0, 45.0), (114.0, 85.0), (118.0, 15.0), (117.0, 75.0),
        (121.0, 55.0), (119.0, 65.0), (124.0, 10.0), (122.0, 100.0), (127.0, 42.0))


def helper_obv() -> float:
    return ta.obv


@script.indicator(title="Conditional builtin variables")
def main():
    every_nvi = ta.nvi
    every_obv = ta.obv
    half_nvi = -1.0
    if bar_index % 2 == 0:
        half_nvi = ta.nvi
    sparse_obv = -1.0
    if bar_index % 3 != 0:
        sparse_obv = helper_obv()
    return {"every_nvi": every_nvi, "every_obv": every_obv,
            "half_nvi": half_nvi, "sparse_obv": sparse_obv}


def _rows():
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV

    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    return [OHLCV(timestamp=base_ts + bar * 1800, open=c, high=c + 1.0, low=c - 1.0,
                  close=c, volume=v)
            for bar, (c, v) in enumerate(BARS)]


def __test_gated_variable_reads_match_the_unconditional_series__(runner):
    """ A gated builtin-variable read equals the unconditional one on every gated bar """
    nvi_values = set()
    for bar, (_candle, plot) in enumerate(runner(iter(_rows())).run_iter()):
        nvi_values.add(plot["every_nvi"])
        if bar % 2 == 0:
            assert plot["half_nvi"] == plot["every_nvi"], \
                f"bar {bar}: gated nvi {plot['half_nvi']} != {plot['every_nvi']}"
        else:
            assert plot["half_nvi"] == -1.0, f"bar {bar}: the branch must not run"
    assert len(nvi_values) >= 3, "nvi must actually move, or the equality proves nothing"


def __test_helper_function_read_matches_too__(runner):
    """ A read through a gated module-level helper sees the same engine series """
    obv_values = set()
    for bar, (_candle, plot) in enumerate(runner(iter(_rows())).run_iter()):
        every = plot["every_obv"]
        obv_values.add(every)
        if bar % 3 != 0:
            gated = plot["sparse_obv"]
            both_na = every != every and gated != gated
            assert both_na or gated == every, \
                f"bar {bar}: helper obv {gated} != {every}"
        else:
            assert plot["sparse_obv"] == -1.0, f"bar {bar}: the branch must not run"
    assert len(obv_values) >= 3, "obv must actually move, or the equality proves nothing"
