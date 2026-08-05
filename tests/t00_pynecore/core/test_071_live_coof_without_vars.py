"""
@pyne

calc_on_order_fills in live simulation must re-execute the body on a fill even
when the script has no root variable slots. The live branch used to call the
COOF loop only when the variable snapshot had slots to restore, so a strategy
without a single ``var`` silently ran once per bar -- unlike the historical path
and unlike TradingView.
"""
import sys
import itertools

from pynecore.lib import plot, script, strategy, bar_index

# A module-level global is outside the slot scheme, so nothing rolls it back --
# it counts body executions, not bars.
_execs: list[int] = []


@script.strategy(
    "Live COOF Without Vars",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    calc_on_order_fills=True,
)
def main():
    _execs.append(bar_index)

    # Placed on the first live bar: warmup executions are suppressed, so an
    # order issued there would never reach the simulator
    if bar_index == 1:
        strategy.entry('Long', strategy.long)

    plot(len(_execs), 'total_execs')


def _make_ohlcv(ts, close=100.0, is_closed=True):
    from pynecore.types.ohlcv import OHLCV
    return OHLCV(timestamp=ts, open=close, high=close + 1, low=close - 1,
                 close=close, volume=1000.0, is_closed=is_closed)


def _create_live_runner(script_path, module_key, syminfo, ohlcv_iter):
    """Helper: set live mode flags, clean module cache, create ScriptRunner."""
    from pynecore.core.script_runner import ScriptRunner
    from pynecore import lib

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)

    setattr(lib, '_is_live', True)
    setattr(lib, '_strategy_suppressed', True)
    return ScriptRunner(script_path, ohlcv_iter, syminfo)


def _chain_live(historical, live):
    """Chain historical OHLCV with LIVE_TRANSITION sentinel and live OHLCV."""
    from pynecore.core.script_runner import LIVE_TRANSITION
    return itertools.chain(historical, [LIVE_TRANSITION], live)


def __test_live_coof_reexecutes_without_var_slots__(script_path, module_key, syminfo):
    """ A fill re-runs the body in live simulation even with no var slots """
    _execs.clear()

    historical = [_make_ohlcv(0, 100.0)]
    live = [
        _make_ohlcv(60, 101.0),   # entry queued here
        _make_ohlcv(120, 102.0),  # entry fills -> body re-executes
        _make_ohlcv(180, 103.0),
    ]

    runner = _create_live_runner(
        script_path, module_key, syminfo,
        _chain_live(historical, live),
    )
    results = [dict(plot_data) for _candle, plot_data, _trades in runner.run_iter()]

    assert len(results) == 4

    # Bar 2 fills the entry, so its body runs twice -- the discarded run and the
    # real one. Every other bar runs once.
    assert [r['total_execs'] for r in results] == [1, 2, 4, 5]

    # The re-execution is what lets the script see its own fill on the fill bar
    assert results[2]['total_execs'] - results[1]['total_execs'] == 2
