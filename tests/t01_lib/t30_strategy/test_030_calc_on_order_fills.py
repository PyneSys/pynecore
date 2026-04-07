"""
@pyne

calc_on_order_fills end-to-end test.

Tests that when calc_on_order_fills=True:
- var variables are rolled back before each re-execution (increments by exactly 1 per bar)
- varip (IBPersistent) variables accumulate across ALL executions including re-executions
- The re-execution loop fires on bars where orders fill
"""
from pynecore.lib import close, plot, script, strategy
from pynecore.types import IBPersistent, Persistent


@script.strategy(
    "COOF Test",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    calc_on_order_fills=True,
)
def main():
    var_exec: Persistent[int] = 0
    varip_exec: IBPersistent[int] = 0

    var_exec += 1
    varip_exec += 1

    if strategy.position_size == 0:
        strategy.entry('Long', strategy.long)

    plot(var_exec, 'var_exec')
    plot(varip_exec, 'varip_exec')


def _make_syminfo(period: str = '5'):
    """Create a minimal SymInfo for testing."""
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def _make_ohlcv(num_bars: int, base_ts: int = 1704067200, period: int = 300):
    """Create simple flat OHLCV bars."""
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(
            timestamp=base_ts + i * period,
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0
        )
        for i in range(num_bars)
    ]


# noinspection PyShadowingNames
def __test_coof_var_rollback_varip_accumulates__(script_path, module_key):
    """
    calc_on_order_fills: var is rolled back, varip accumulates across re-executions.

    Execution flow:
    - Bar 0: entry('Long') signal, no pending orders → 1 execution
    - Bar 1: market order fills → re-execution + bar-close main = 2 main() calls
    - Bar 2+: no new fills → 1 execution per bar

    var_exec: always bar_index+1 (rollback ensures re-execution doesn't add extra)
    varip_exec: bar_index+1 + number_of_reexecutions (accumulated from fill bar)

    On bar 1, varip gets +1 extra from re-execution. After that, var_exec and
    varip_exec increment at the same rate (+1/bar), but varip stays 1 ahead.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='5')
    ohlcv = _make_ohlcv(num_bars=5)

    runner = ScriptRunner(
        Path(script_path), iter(ohlcv), syminfo,
    )

    results = []
    for candle, plot_data, trades in runner.run_iter():
        results.append(dict(plot_data))

    assert len(results) == 5, f"Expected 5 bars, got {len(results)}"

    # Bar 0: no pending orders, single execution
    assert results[0]['var_exec'] == 1
    assert results[0]['varip_exec'] == 1

    # Bar 1: fill happens → re-execution. Re-execution produces duplicate plot keys.
    # The bar-close main() values get ' 0' suffix due to plot key collision.
    # var_exec: re-exec sees restored(1)+1=2, bar-close sees restored(1)+1=2 → same
    # varip_exec: re-exec sees 1+1=2, bar-close sees 2+1=3 (NOT rolled back)
    assert results[1]['var_exec'] == 2, \
        f"Bar 1 var_exec: expected 2, got {results[1]['var_exec']}"
    assert results[1].get('varip_exec 0', results[1]['varip_exec']) == 3, \
        f"Bar 1 varip_exec (bar-close): expected 3, got {results[1]}"

    # Bar 2+: no fills, single execution per bar. Both increment by 1.
    # var_exec: 3, 4, 5 (committed state from prev bar + 1)
    # varip_exec: 4, 5, 6 (stays 1 ahead of var_exec due to bar 1's extra re-execution)
    for i in range(2, 5):
        expected_var = i + 1
        expected_varip = i + 2  # 1 extra from bar 1 re-execution
        assert results[i]['var_exec'] == expected_var, \
            f"Bar {i} var_exec: expected {expected_var}, got {results[i]['var_exec']}"
        assert results[i]['varip_exec'] == expected_varip, \
            f"Bar {i} varip_exec: expected {expected_varip}, got {results[i]['varip_exec']}"

    # Key invariant: varip_exec is always exactly 1 more than var_exec (one re-execution happened)
    for i in range(2, 5):
        assert results[i]['varip_exec'] - results[i]['var_exec'] == 1, \
            f"Bar {i}: varip should be 1 ahead of var due to COOF re-execution"


# noinspection PyShadowingNames
def __test_coof_disabled_no_reexecution__(script_path, module_key):
    """
    Without calc_on_order_fills, no re-execution occurs — var and varip behave identically.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(module_key, None)
    sys.modules.pop(Path(script_path).stem, None)

    syminfo = _make_syminfo(period='5')
    ohlcv = _make_ohlcv(num_bars=5)

    runner = ScriptRunner(
        Path(script_path), iter(ohlcv), syminfo,
    )

    # Override calc_on_order_fills to False
    runner.script.calc_on_order_fills = False

    results = []
    for candle, plot_data, trades in runner.run_iter():
        results.append(dict(plot_data))

    assert len(results) == 5

    # Without COOF: var and varip both increment by 1 per bar, no re-execution
    for i in range(5):
        expected = i + 1
        assert results[i]['var_exec'] == expected, \
            f"Bar {i} var_exec: expected {expected}, got {results[i]['var_exec']}"
        assert results[i]['varip_exec'] == expected, \
            f"Bar {i} varip_exec: expected {expected}, got {results[i]['varip_exec']}"
