"""
A library's export surface is built once per RUN, not once per bar.

A library ``main`` is a per-bar entry point, and Pine's exports live inside it,
so as written every bar allocates a fresh function object per export and runs
the ``export`` decorator over it. MEASURED: those objects are never called —
a call site anchors on the module-level ``Exported`` proxy, whose identity is
stable, so the binding unwraps the bar-0 callable once and keeps it for the
whole run.

``ExportOnceTransformer`` puts the definitions under a latch that is a
``Persistent`` slot of ``main``. Keyed to the ROOT rather than to the module,
a rerun (a fresh root vector) rebuilds them — which is what the second half of
this file pins, together with the values staying correct in both run modes.
"""
import sys
from pathlib import Path
from typing import Any, Callable

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


#: Closes of the fixture bars; the smoothed export averages three of them
CLOSES = (100.0, 104.0, 109.0, 115.0, 122.0, 130.0, 139.0, 149.0)


def _make_ohlcv():
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=1_704_067_200_000 + i * 300_000,
                  open=c, high=c + 1.0, low=c - 1.0, close=c, volume=1000.0)
            for i, c in enumerate(CLOSES)]


def _expected_smoothed() -> list[float]:
    """Three-bar simple average of the fixture closes, na before warm-up."""
    return [float('nan') if i < 2 else sum(CLOSES[i - 2:i + 1]) / 3.0
            for i in range(len(CLOSES))]


class _Counter:
    """Counts how often an ``Exported`` proxy is (re)bound to a callable."""

    def __init__(self):
        from pynecore.core import pine_export
        self.module = pine_export
        # The plain function behind the method, so the wrapper below can pass
        # the receiver explicitly
        self.original: Callable[..., Any] = vars(pine_export.Exported)['set']
        self.count = 0

    def __enter__(self):
        counter = self
        original: Callable[..., Any] = self.original

        def counted_set(proxy, client, na_bool=False):
            counter.count += 1
            return original(proxy, client, na_bool)

        self.module.Exported.set = counted_set
        return self

    def __exit__(self, *_exc):
        self.module.Exported.set = self.original
        return False


def _run(script_name: str, runs: int = 1) -> list[list[dict]]:
    """Run a data script the given number of times, keeping the modules loaded.

    :param script_name: File name under ``data``.
    :param runs: How many times to drive the same modules.
    :return: One plot-data list per run.
    """
    from pynecore.core import script as script_core
    from pynecore.core.script_runner import ScriptRunner

    # noinspection PyProtectedMember
    saved_libraries = list(script_core._registered_libraries)
    sys.path.insert(0, str(DATA_DIR))
    try:
        results = []
        for _ in range(runs):
            runner = ScriptRunner(DATA_DIR / script_name, iter(_make_ohlcv()),
                                  _make_syminfo())
            results.append([dict(plot_data) for _candle, plot_data in runner.run_iter()])
        return results
    finally:
        sys.path.remove(str(DATA_DIR))
        # noinspection PyProtectedMember
        script_core._registered_libraries[:] = saved_libraries
        sys.modules.pop('export_once_lib', None)
        sys.modules.pop('export_once_script', None)


def __test_an_imported_export_is_defined_once_per_run__():
    """Two exports, one definition each — no matter how many bars run"""
    with _Counter() as counter:
        results = _run('export_once_script.py')[0]

    assert len(results) == len(CLOSES)
    assert counter.count == 2, \
        f"the two exports were (re)defined {counter.count} times, not twice"

    assert [r['scaled'] for r in results] == [c * 3.0 for c in CLOSES]
    expected = _expected_smoothed()
    for bar, row in enumerate(results):
        got, want = row['smoothed'], expected[bar]
        if want != want:  # na before the sma has three bars
            assert got != got, f"bar {bar}: {got} is not na"
        else:
            assert abs(got - want) < 1e-12, f"bar {bar}: {got} != {want}"


def __test_a_library_run_as_a_study_keeps_its_own_calls_working__():
    """The main body calls its own export on every bar, not only the first"""
    with _Counter() as counter:
        results = _run('export_once_lib.py')[0]

    assert counter.count == 2, \
        f"the two exports were (re)defined {counter.count} times, not twice"
    # The definitions stopped being rebuilt, so a later bar reads the name
    # from the module-level proxy the ``global`` binding points at
    assert [r['own_scaled'] for r in results] == [6.0] * len(CLOSES)
    assert [r['runs'] for r in results] == list(range(1, len(CLOSES) + 1))
    expected = _expected_smoothed()
    for bar, row in enumerate(results):
        got, want = row['own_smoothed'], expected[bar]
        if want != want:
            assert got != got, f"bar {bar}: {got} is not na"
        else:
            assert abs(got - want) < 1e-12, f"bar {bar}: {got} != {want}"


def __test_a_rerun_rebuilds_the_exports__():
    """The latch is a slot of the root, so a fresh root defines them again"""
    with _Counter() as counter:
        runs = _run('export_once_script.py', runs=2)

    assert counter.count == 4, \
        f"a second run rebound {counter.count - 2} exports, expected 2"
    assert runs[0] == runs[1], "the second run must reproduce the first"
