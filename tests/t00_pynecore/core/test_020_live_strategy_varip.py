"""
@pyne

Tests that strategies handle varip correctly in live mode:
- Default (calc_on_every_tick=False): script runs only on bar close, var == varip
- calc_on_every_tick=True: script runs every tick, var rolls back, varip persists
"""
import sys
import itertools

from pynecore.lib import barstate, plot, script, strategy
from pynecore.types import Persistent, IBPersistent


@script.strategy(
    "Live Strategy varip Test",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    var_count: Persistent[int] = 0
    varip_count: IBPersistent[int] = 0
    var_count += 1
    varip_count += 1
    plot(var_count, 'var')
    plot(varip_count, 'varip')
    plot(1 if barstate.isnew else 0, 'isnew')


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


def __test_strategy_default_var_equals_varip__(script_path, module_key, syminfo):
    """Default strategy: script runs only at bar close, var and varip are identical."""
    historical = [_make_ohlcv(i * 60, 100.0 + i) for i in range(3)]
    live = [
        _make_ohlcv(3 * 60, is_closed=False, close=104.0),   # bar open
        _make_ohlcv(3 * 60, is_closed=False, close=104.5),   # intra-bar
        _make_ohlcv(3 * 60, is_closed=True, close=105.0),    # bar close
    ]

    runner = _create_live_runner(
        script_path, module_key, syminfo,
        _chain_live(historical, live),
    )

    results = []
    for candle, plot_data, _trades in runner.run_iter():
        results.append(dict(plot_data))

    assert len(results) == 4, f"Expected 4 bars, got {len(results)}"

    # Historical: var and varip both increment by 1
    for i in range(3):
        assert results[i]['var'] == i + 1
        assert results[i]['varip'] == i + 1

    # Live bar: strategy runs ONCE (bar close only), so var == varip == 4
    assert results[3]['var'] == 4, f"var should be 4, got {results[3]['var']}"
    assert results[3]['varip'] == 4, f"varip should be 4, got {results[3]['varip']}"
    assert results[3]['isnew'] == 1, "barstate.isnew should be True at bar close"


def __test_strategy_calc_on_every_tick__(script_path, module_key, syminfo):
    """Strategy with calc_on_every_tick: var rolls back, varip persists like indicator."""
    historical = [_make_ohlcv(i * 60, 100.0 + i) for i in range(3)]
    live = [
        _make_ohlcv(3 * 60, is_closed=False, close=104.0),   # bar open
        _make_ohlcv(3 * 60, is_closed=False, close=104.5),   # intra-bar
        _make_ohlcv(3 * 60, is_closed=True, close=105.0),    # bar close
    ]

    runner = _create_live_runner(
        script_path, module_key, syminfo,
        _chain_live(historical, live),
    )
    runner.script.calc_on_every_tick = True

    results = []
    for candle, plot_data, _trades in runner.run_iter():
        results.append(dict(plot_data))

    assert len(results) == 4, f"Expected 4 bars, got {len(results)}"

    # Historical: identical
    for i in range(3):
        assert results[i]['var'] == i + 1
        assert results[i]['varip'] == i + 1

    # Live bar: 3 executions (open, intra, close)
    # var: rolled back each time, final = 3 + 1 = 4
    # varip: accumulated across 3 ticks = 3 + 3 = 6
    assert results[3]['var'] == 4, f"var should be 4, got {results[3]['var']}"
    assert results[3]['varip'] == 6, f"varip should be 6, got {results[3]['varip']}"


def __test_strategy_default_barstate_isnew_on_every_live_bar__(script_path, module_key, syminfo):
    """Default strategy: barstate.isnew is True on every live bar close (strategy runs once)."""
    historical = [_make_ohlcv(0, 100.0)]
    live = [
        # First live bar: intra-bar ticks then close
        _make_ohlcv(60, is_closed=False, close=101.0),
        _make_ohlcv(60, is_closed=True, close=102.0),
        # Second live bar: direct close (no intra-bar)
        _make_ohlcv(120, is_closed=True, close=103.0),
        # Third live bar: with intra-bar ticks
        _make_ohlcv(180, is_closed=False, close=104.0),
        _make_ohlcv(180, is_closed=False, close=104.5),
        _make_ohlcv(180, is_closed=True, close=105.0),
    ]

    runner = _create_live_runner(
        script_path, module_key, syminfo,
        _chain_live(historical, live),
    )

    results = []
    for candle, plot_data, _trades in runner.run_iter():
        results.append(dict(plot_data))

    assert len(results) == 4, f"Expected 4 bars, got {len(results)}"

    # Live bars (1-3): strategy runs once per bar, so isnew should always be True
    for i in range(1, 4):
        assert results[i]['isnew'] == 1, \
            f"Bar {i}: barstate.isnew should be True, got {results[i]['isnew']}"
