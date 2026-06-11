"""
calc_on_order_fills regression: a state-carrying ``@script.library`` function
in the SAME module as the strategy main must not collide with main's root
vector (root keys are qualified per function). A collision detaches main's
root, so ``instance_state.reset()`` and ``RootVarSnapshot`` silently stop
covering main and the COOF var rollback is lost (var_exec would read
1, 2, 4, 5, 6 instead of 1..5).
"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'


def _make_syminfo(period: str = '5'):
    """Create a minimal SymInfo for testing."""
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
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


def __test_coof_same_module_library_var_rollback__():
    """ var rollback keeps working with a same-module library registration """
    from pynecore.core import script as script_core
    from pynecore.core.script_runner import ScriptRunner

    # The library registration is process-global: restore it so other runner
    # tests in the session do not run this script's library entry per bar
    saved_libraries = list(script_core._registered_libraries)
    try:
        runner = ScriptRunner(
            DATA_DIR / 'coof_same_module_lib.py', iter(_make_ohlcv(5)), _make_syminfo(),
        )
        results = [dict(plot_data) for _candle, plot_data, _trades in runner.run_iter()]
    finally:
        script_core._registered_libraries[:] = saved_libraries
        sys.modules.pop('coof_same_module_lib', None)

    # Bar 1 fills the market order -> COOF re-execution; the rollback must keep
    # var_exec at exactly bar_index+1 on every bar
    assert [r['var_exec'] for r in results] == [1, 2, 3, 4, 5]
