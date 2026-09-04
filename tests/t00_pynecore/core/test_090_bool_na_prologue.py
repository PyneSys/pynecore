"""
The bool na choice is re-applied after EVERY top-level import of the module,
not only after the leading block: an import further down pulls in a library
whose own prologue would otherwise outlive this module's choice, and a UDT
default built after that import would come out as the two-state false.
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


def __test_the_choice_follows_every_import__():
    """A UDT default built after a later library import is still the bool na"""
    from pynecore.core import script as script_core
    from pynecore.core.script_runner import ScriptRunner

    saved_libraries = list(script_core._registered_libraries)
    try:
        runner = ScriptRunner(DATA_DIR / 'bool_na_prologue_script.py', iter(_make_ohlcv(2)),
                              _make_syminfo())
        results = [dict(plot_data) for _candle, plot_data in runner.run_iter()]
    finally:
        script_core._registered_libraries[:] = saved_libraries
        sys.modules.pop('bool_na_prologue_lib', None)
        sys.modules.pop('bool_na_prologue_script', None)
        set_bool_na(False)

    assert [r['udt_seen'] for r in results] == [1.0, 1.0]
