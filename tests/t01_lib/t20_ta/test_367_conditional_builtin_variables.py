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

Function scope splits them (probes m573/m574): ``nvi``/``obv``/``pvi``/``pvt``/
``wad`` stay engine-global inside user functions too, while ``vwap`` and
``accdist`` get a per-instance machine there (they agree with the global only on
1 and ~594 of 14255 gated bars — for vwap the session-anchor bars where both
reset).

The ``TaVariableHoistTransformer`` implements this: every referenced stateful
builtin variable is evaluated once, unconditionally, at the top of ``main``, and
the reference sites read that cached value — engine-global variables everywhere,
``vwap``/``accdist`` only directly in ``main``'s body, so inside a helper those
keep their own call-gated machine. The unconditional ``every_*`` plots and the
distinct-value guards are the controls: without them a series stuck at its seed
would satisfy the equalities vacuously.
"""
from pynecore.lib import script, ta, bar_index

# (close, volume): volume alternates around the previous bar so nvi/pvi update on
# real bars, and close moves both ways so obv accumulates in both directions
__test_helper_BARS = ((100.0, 50.0), (102.0, 40.0), (101.0, 60.0), (105.0, 30.0), (104.0, 70.0),
        (108.0, 20.0), (107.0, 90.0), (110.0, 35.0), (109.0, 80.0), (113.0, 25.0),
        (112.0, 95.0), (116.0, 45.0), (114.0, 85.0), (118.0, 15.0), (117.0, 75.0),
        (121.0, 55.0), (119.0, 65.0), (124.0, 10.0), (122.0, 100.0), (127.0, 42.0))


def helper_obv() -> float:
    return ta.obv


def helper_accdist() -> float:
    return ta.accdist


@script.indicator(title="Conditional builtin variables")
def main():
    every_nvi = ta.nvi
    every_obv = ta.obv
    every_acc = ta.accdist
    half_nvi = -1.0
    if bar_index % 2 == 0:
        half_nvi = ta.nvi
    sparse_obv = -1.0
    if bar_index % 3 != 0:
        sparse_obv = helper_obv()
    half_acc = -1.0
    fn_acc = -1.0
    if bar_index % 2 == 0:
        half_acc = ta.accdist
        fn_acc = helper_accdist()
    return {"every_nvi": every_nvi, "every_obv": every_obv, "every_acc": every_acc,
            "half_nvi": half_nvi, "sparse_obv": sparse_obv,
            "half_acc": half_acc, "fn_acc": fn_acc}


def __test_helper_rows():
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV

    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    # close is off-center in the bar range, so accdist's money-flow term is nonzero
    return [OHLCV(timestamp=base_ts + bar * 1800, open=c, high=c + 1.0, low=c - 2.0,
                  close=c, volume=v)
            for bar, (c, v) in enumerate(__test_helper_BARS)]


def __test_gated_variable_reads_match_the_unconditional_series__(runner):
    """ A gated builtin-variable read equals the unconditional one on every gated bar """
    nvi_values = set()
    for bar, (_candle, plot) in enumerate(runner(iter(__test_helper_rows())).run_iter()):
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
    for bar, (_candle, plot) in enumerate(runner(iter(__test_helper_rows())).run_iter()):
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


def __test_accdist_is_global_in_main_but_instanced_in_a_helper__(runner):
    """ vwap/accdist law: main-body reads track the engine, helper reads do not """
    acc_values = set()
    fn_mismatch = 0
    for bar, (_candle, plot) in enumerate(runner(iter(__test_helper_rows())).run_iter()):
        every = plot["every_acc"]
        acc_values.add(every)
        if bar % 2 == 0:
            assert plot["half_acc"] == every, \
                f"bar {bar}: main-body accdist {plot['half_acc']} != {every}"
            if plot["fn_acc"] != every:
                fn_mismatch += 1
        else:
            assert plot["half_acc"] == -1.0 and plot["fn_acc"] == -1.0, \
                f"bar {bar}: the branch must not run"
    assert len(acc_values) >= 3, "accdist must actually move, or the equality proves nothing"
    assert fn_mismatch > 0, \
        "the helper's accdist must run its own gated machine and diverge from the engine series"
