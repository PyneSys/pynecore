"""
A stateful export reached from a loop keeps one step per bar across a bool-mode
crossing.

Once a module asks for the three-state bool, every export binding is handed to
the caller wrapped in ``pine_export.in_module_bool_mode``. A loop site rolls the
builtins behind its binding back on every pass of the same bar, and a per-bar
redefinition carries the binding's state over: both have to reach the binding
through the wrapper. Through an opaque wrapper each pass of the loop advanced the
library's ``ta.sma`` / ``ta.stdev`` once more, so the values ran ahead of the
same functions called without a module boundary.
"""
import math
import sys
from pathlib import Path

from pynecore.types.na import isna_num, set_bool_na

DATA_DIR = Path(__file__).parent / 'data'
NUM_BARS = 40
LENGTH = 5


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
    """A varying close, so the window position shows in sma and stdev"""
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=1_704_067_200_000 + i * 300_000,
                  open=100.0, high=110.0, low=90.0,
                  close=100.0 + 5.0 * math.sin(i / 3.0), volume=1000.0)
            for i in range(num_bars)]


def _plain(value):
    return None if isna_num(value) else value


def __test_a_looped_export_steps_once_per_bar_across_bool_modes__():
    """The library's export, looped in the caller and in the library, matches the local copy"""
    from pynecore.core import script as script_core
    from pynecore.core.script_runner import ScriptRunner

    saved_libraries = list(script_core._registered_libraries)
    sys.path.insert(0, str(DATA_DIR))
    try:
        runner = ScriptRunner(DATA_DIR / 'bool_mode_loop_script.py',
                              iter(_make_ohlcv(NUM_BARS)), _make_syminfo())
        results = [dict(plot_data) for _candle, plot_data in runner.run_iter()]
    finally:
        sys.path.remove(str(DATA_DIR))
        script_core._registered_libraries[:] = saved_libraries
        sys.modules.pop('bool_mode_loop_lib', None)
        sys.modules.pop('bool_mode_loop_script', None)
        set_bool_na(False)

    assert len(results) == NUM_BARS
    for name in ('outer', 'inner'):
        got = [_plain(r[name]) for r in results]
        expected = [_plain(r[name + '_ref']) for r in results]
        # One step per bar: na until the window is full
        assert got[:LENGTH - 1] == [None] * (LENGTH - 1), (name, got[:LENGTH])
        assert None not in got[LENGTH - 1:], name
        assert got == expected, name
