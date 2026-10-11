"""
A window call reached by an instance created with no vector runs the ring.

A helper whose window site resolves differently per calling context reads its
``_series_length`` from its instance vector (``__state__[k][j]``). An instance
created with no vector -- a cross-module call into a library export that relays
its own entry -- keeps the all-None template, so the window call receives
``_series_length=None``. The plain functions test it for truth; the overload
groups (``highest``/``lowest`` and their ``*bars`` forms) must accept it too,
or a valid script dies with ``TypeError: No matching implementation`` the first
time such a branch runs (wild-corpus library ``enhanced_ta`` v14,
``customseries(source, "high", length)`` reached through ``ma``).
"""
import sys
from pathlib import Path

__test_helper_DATA_DIR = Path(__file__).parent / 'data'


def __test_helper_make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period="30", type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


def __test_helper_make_ohlcv():
    from pynecore.types.ohlcv import OHLCV
    closes = (100.0, 102.0, 101.0, 105.0, 104.0, 108.0, 107.0, 110.0, 109.0, 113.0,
              112.0, 116.0, 114.0, 118.0, 117.0, 121.0, 119.0, 124.0, 122.0, 127.0)
    return [
        OHLCV(timestamp=1_704_067_200 + bar * 1800, open=c, high=c + 1.0 + bar % 3,
              low=c - 2.0 - bar % 4, close=c, volume=100.0)
        for bar, c in enumerate(closes)
    ]


def __test_helper_run_script() -> list[dict]:
    from pynecore.core import script as script_core
    from pynecore.core.script_runner import ScriptRunner

    # noinspection PyProtectedMember
    saved_libraries = list(script_core._registered_libraries)
    try:
        runner = ScriptRunner(
            __test_helper_DATA_DIR / 'vectorless_window_script.py',
            iter(__test_helper_make_ohlcv()), __test_helper_make_syminfo(),
        )
        return [dict(plot_data) for _candle, plot_data in runner.run_iter()]
    finally:
        # noinspection PyProtectedMember
        script_core._registered_libraries[:] = saved_libraries
        sys.modules.pop('vectorless_window_lib', None)
        sys.modules.pop('vectorless_window_script', None)


def __test_vectorless_instance_runs_every_window_as_the_ring__():
    """ Every window reached with ``_series_length=None`` matches the direct const call """
    results = __test_helper_run_script()
    assert len(results) == len(__test_helper_make_ohlcv())
    compared = 0
    for bar, plot in enumerate(results):
        for name in ("hi", "lo", "hib", "lob", "lr"):
            got, want = plot[name], plot[f"{name}_direct"]
            if got != got and want != want:
                continue
            assert got == want, f"bar {bar}: {name} {got} != direct {want}"
            compared += 1
    assert compared > 50, "the windows must produce values, or the equality proves nothing"


def __test_every_series_length_overload_accepts_the_vectorless_entry__():
    """
    Every overloaded implementation taking ``_series_length`` matches the three
    values the isolation pass can hand it: True, False and the None of a
    vectorless instance
    """
    import importlib
    import pkgutil

    import pynecore.lib
    # noinspection PyProtectedMember
    from pynecore.core.overload import _check_type, _registry

    for info in pkgutil.iter_modules(pynecore.lib.__path__, 'pynecore.lib.'):
        importlib.import_module(info.name)

    checked = []
    for qualname, impls in _registry.items():
        if not qualname.startswith('pynecore.lib.'):
            continue
        for impl in impls:
            hint = impl.type_hints.get('_series_length')
            if hint is None:
                continue
            for value in (True, False, None):
                assert _check_type(value, hint), \
                    f"{qualname}{impl.sig}: _series_length={value!r} matches no implementation"
            checked.append(qualname)
    assert {'pynecore.lib.ta.highest', 'pynecore.lib.ta.lowest', 'pynecore.lib.ta.highestbars',
            'pynecore.lib.ta.lowestbars'} <= set(checked)
