"""
A library's bool semantics cross the export boundary with it.

The three-state bool of a Pine v4/v5 library is a property of ITS source. The
mode is process-wide, and every entry point sets it -- but an exported function
is called later, from the importing script's body, where the caller's mode is
in effect. A v6 caller then flattened the library's ``na(bool)`` to False and
its ``na(state)`` branch was never taken.

The crossing swaps the mode for the duration of the call, at both places the
export proxy is unwrapped: the anchored binding (``instance_state._bind_target``)
and the method binding (``pine_method._bound_method``).
"""
import sys
from pathlib import Path

from pynecore.types.na import set_bool_na

DATA_DIR = Path(__file__).parent / 'data'


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='5', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def _make_ohlcv(num_bars: int):
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=1_704_067_200_000 + i * 300_000,
                  open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0)
            for i in range(num_bars)]


def __test_an_exported_function_keeps_its_own_bool_mode__():
    """A v6 script calling a v4/v5 library gets the library's three-state bool"""
    from pynecore.core import script as script_core
    from pynecore.core.script_runner import ScriptRunner

    saved_libraries = list(script_core._registered_libraries)
    sys.path.insert(0, str(DATA_DIR))
    try:
        runner = ScriptRunner(DATA_DIR / 'bool_na_export_script.py', iter(_make_ohlcv(3)),
                              _make_syminfo())
        results = [dict(plot_data) for _candle, plot_data in runner.run_iter()]
    finally:
        sys.path.remove(str(DATA_DIR))
        script_core._registered_libraries[:] = saved_libraries
        sys.modules.pop('bool_na_export_lib', None)
        sys.modules.pop('bool_na_export_script', None)
        set_bool_na(False)

    # The library's own na(bool) is the na object, on every bar
    assert [r['lib_na'] for r in results] == [1.0, 1.0, 1.0]
    # ... so its ``if na(state)`` branch runs and the argument reaches state
    assert [r['state'] for r in results] == [1.0, 1.0, 1.0]
    # The caller keeps the v6 bool it declared
    assert [r['main_na'] for r in results] == [0.0, 0.0, 0.0]
