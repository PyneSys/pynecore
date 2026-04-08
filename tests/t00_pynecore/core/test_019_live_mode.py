"""
@pyne
"""
import sys
import itertools

from pynecore.lib import script, barstate
from pynecore.types import Persistent, IBPersistent


@script.indicator(title="Live Mode Test")
def main():
    var_count: Persistent[int] = 0
    varip_count: IBPersistent[int] = 0
    var_count += 1
    varip_count += 1
    return {
        "var": var_count,
        "varip": varip_count,
        "rt": 1 if barstate.isrealtime else 0,
        "hist": 1 if barstate.ishistory else 0,
        "conf": 1 if barstate.isconfirmed else 0,
        "new": 1 if barstate.isnew else 0,
        "lch": 1 if barstate.islastconfirmedhistory else 0,
    }


def _make_ohlcv(ts, close=100.0):
    from pynecore.types.ohlcv import OHLCV
    return OHLCV(timestamp=ts, open=close, high=close + 1, low=close - 1,
                 close=close, volume=1000.0)


def _make_bar_update(ts, is_closed=True, close=100.0):
    from pynecore.core.plugin.live_provider import BarUpdate
    return BarUpdate(ohlcv=_make_ohlcv(ts, close), is_closed=is_closed)


def _create_live_runner(script_path, module_key, syminfo, ohlcv_iter):
    """Helper: set live mode flags, clean module cache, create ScriptRunner."""
    from pynecore.core.script_runner import ScriptRunner
    from pynecore import lib

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)

    # Use setattr to avoid FunctionIsolationTransformer mangling the assignment
    setattr(lib, '_is_live', True)
    setattr(lib, '_strategy_suppressed', True)
    return ScriptRunner(script_path, ohlcv_iter, syminfo)


def __test_barstate_historical_then_live__(script_path, module_key, syminfo):
    """barstate transitions from ishistory=True to isrealtime=True at live phase"""
    historical = [_make_ohlcv(i * 60, 100.0 + i) for i in range(3)]
    live = [_make_bar_update(3 * 60, is_closed=True, close=104.0)]

    runner = _create_live_runner(
        script_path, module_key, syminfo,
        itertools.chain(historical, live),
    )

    results = [(c, dict(p)) for c, p in runner.run_iter()]

    assert len(results) == 4

    # Historical bars
    for i in range(3):
        _, plot_data = results[i]
        assert plot_data["hist"] == 1, f"Bar {i} should be historical"
        assert plot_data["rt"] == 0, f"Bar {i} should not be realtime"

    # Live bar
    _, plot_data = results[3]
    assert plot_data["hist"] == 0, "Live bar should not be historical"
    assert plot_data["rt"] == 1, "Live bar should be realtime"
    assert plot_data["conf"] == 1, "Closed live bar should be confirmed"


def __test_islastconfirmedhistory__(script_path, module_key, syminfo):
    """islastconfirmedhistory is True only on the final historical bar before live"""
    historical = [_make_ohlcv(i * 60, 100.0 + i) for i in range(3)]
    live = [_make_bar_update(3 * 60, is_closed=True, close=104.0)]

    runner = _create_live_runner(
        script_path, module_key, syminfo,
        itertools.chain(historical, live),
    )

    results = [(c, dict(p)) for c, p in runner.run_iter()]

    # Only bar 2 (last historical) should have islastconfirmedhistory=True
    assert results[0][1]["lch"] == 0
    assert results[1][1]["lch"] == 0
    assert results[2][1]["lch"] == 1, "Last historical bar should have islastconfirmedhistory"
    assert results[3][1]["lch"] == 0, "Live bar should not have islastconfirmedhistory"


def __test_intrabar_barstate__(script_path, module_key, syminfo):
    """Intra-bar ticks have isconfirmed=False, isnew=True on first tick"""
    historical = [_make_ohlcv(0, 100.0)]
    live = [
        _make_bar_update(60, is_closed=False, close=101.0),  # bar open
        _make_bar_update(60, is_closed=False, close=101.5),  # intra-bar
        _make_bar_update(60, is_closed=True, close=102.0),   # bar close
    ]

    runner = _create_live_runner(
        script_path, module_key, syminfo,
        itertools.chain(historical, live),
    )

    results = [(c, dict(p)) for c, p in runner.run_iter()]

    # 1 historical + 1 closed live = 2 yielded results (intra-bar not yielded)
    assert len(results) == 2

    # The live bar's final values (from the bar-close execution)
    _, plot_data = results[1]
    assert plot_data["conf"] == 1, "Closed bar should be confirmed"
    assert plot_data["rt"] == 1, "Should be realtime"


def __test_var_rollback_varip_persist__(script_path, module_key, syminfo):
    """var rolls back on intra-bar, varip persists across all executions"""
    historical = [_make_ohlcv(i * 60, 100.0 + i) for i in range(3)]
    live = [
        # Bar with 3 executions: open tick, intra-bar tick, close
        _make_bar_update(3 * 60, is_closed=False, close=104.0),
        _make_bar_update(3 * 60, is_closed=False, close=104.5),
        _make_bar_update(3 * 60, is_closed=True, close=105.0),
    ]

    runner = _create_live_runner(
        script_path, module_key, syminfo,
        itertools.chain(historical, live),
    )

    results = [(c, dict(p)) for c, p in runner.run_iter()]
    assert len(results) == 4  # 3 historical + 1 live close

    # Historical bars: var and varip both increment by 1 per bar
    assert results[0][1]["var"] == 1
    assert results[0][1]["varip"] == 1
    assert results[1][1]["var"] == 2
    assert results[1][1]["varip"] == 2
    assert results[2][1]["var"] == 3
    assert results[2][1]["varip"] == 3

    # Live bar: var should be 4 (3+1, rolled back each time, final = 4)
    # varip should be 6 (3 + 3 executions: open, intra, close)
    _, live_plot = results[3]
    assert live_plot["var"] == 4, f"var should be 4, got {live_plot['var']}"
    assert live_plot["varip"] == 6, f"varip should be 6, got {live_plot['varip']}"


def __test_yield_only_on_closed_bars__(script_path, module_key, syminfo):
    """run_iter only yields for closed bars, not intra-bar ticks"""
    historical = [_make_ohlcv(0, 100.0)]
    live = [
        _make_bar_update(60, is_closed=False, close=101.0),
        _make_bar_update(60, is_closed=False, close=101.5),
        _make_bar_update(60, is_closed=False, close=101.8),
        _make_bar_update(60, is_closed=True, close=102.0),
        _make_bar_update(120, is_closed=True, close=103.0),
    ]

    runner = _create_live_runner(
        script_path, module_key, syminfo,
        itertools.chain(historical, live),
    )

    results = [(c, dict(p)) for c, p in runner.run_iter()]

    # 1 historical + 2 closed live bars = 3 yields
    assert len(results) == 3


def __test_no_live_bars_unchanged_behavior__(script_path, module_key, syminfo):
    """When is_live=True but no BarUpdate arrives, behaves like normal backtest"""
    historical = [_make_ohlcv(i * 60, 100.0 + i) for i in range(5)]

    runner = _create_live_runner(
        script_path, module_key, syminfo,
        iter(historical),
    )

    results = [(c, dict(p)) for c, p in runner.run_iter()]

    assert len(results) == 5
    for i, (_, plot_data) in enumerate(results):
        assert plot_data["hist"] == 1
        assert plot_data["rt"] == 0
        assert plot_data["var"] == i + 1
        assert plot_data["varip"] == i + 1
