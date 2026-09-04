"""
A library's UDT bool na default follows the CALLING script's bool choice: the
default is built per construction, not frozen at the library's import under
the library's own choice (see DynamicDefaultTransformer).
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


def _run(script_name: str, lib_name: str) -> list[dict]:
    from pynecore.core import script as script_core
    from pynecore.core.script_runner import ScriptRunner

    saved_libraries = list(script_core._registered_libraries)
    try:
        runner = ScriptRunner(DATA_DIR / script_name, iter(_make_ohlcv(2)), _make_syminfo())
        return [dict(plot_data) for _candle, plot_data in runner.run_iter()]
    finally:
        script_core._registered_libraries[:] = saved_libraries
        sys.modules.pop(lib_name, None)
        sys.modules.pop(script_name[:-3], None)
        set_bool_na(False)


def __test_a_three_state_caller_gets_the_bool_na_from_a_two_state_library__():
    results = _run('bool_na_udt_script3.py', 'bool_na_udt_lib')
    assert [r['lib_udt_na'] for r in results] == [1.0, 1.0]


def __test_a_two_state_caller_gets_false_from_a_three_state_library__():
    results = _run('bool_na_udt_script2.py', 'bool_na_udt_lib3')
    assert [r['lib_udt_false'] for r in results] == [1.0, 1.0]
