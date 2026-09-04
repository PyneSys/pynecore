"""
The bool na mode is process-wide and re-applied on every entry: an imported
library's entry runs under the script's choice even when another script's run
switched the mode off in between.
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


def __test_the_library_entry_runs_in_the_scripts_mode__():
    """The mode is set before the library mains, not only before main"""
    from pynecore.core import script as script_core
    from pynecore.core.script_runner import ScriptRunner

    saved_libraries = list(script_core._registered_libraries)
    try:
        runner = ScriptRunner(DATA_DIR / 'bool_na_lib_entry.py', iter(_make_ohlcv(3)),
                              _make_syminfo())
        results = []
        for _candle, plot_data in runner.run_iter():
            results.append(dict(plot_data))
            # Another script's run, interleaved between the bars, leaves the
            # two-state mode behind
            set_bool_na(False)
    finally:
        script_core._registered_libraries[:] = saved_libraries
        sys.modules.pop('bool_na_lib_entry_lib', None)
        sys.modules.pop('bool_na_lib_entry', None)
        set_bool_na(False)

    assert [r['lib_seen'] for r in results] == [1.0, 1.0, 1.0]
    assert [r['main_seen'] for r in results] == [1.0, 1.0, 1.0]
    assert [r['udt_seen'] for r in results] == [1.0, 1.0, 1.0]
