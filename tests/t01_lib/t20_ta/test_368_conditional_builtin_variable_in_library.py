"""
Builtin-variable reads and the library boundary: ``export`` is the dividing
line, in both run modes.

Measured on TradingView (probes m575-m578, a private library whose exports
return ``ta.nvi``/``ta.vwap``, BINANCE:BTCUSDT 30m, 28527+ bars):

- an EXPORTED function's read is a per-call-site gated machine: called
  unconditionally it agrees with the unconditional global read on every bar,
  called inside an ``if`` it diverges (2/14264 agreement — the seed region),
  and two differently gated call sites of the same export diverge from each
  other too; the signature is identical when the library runs as a study;
- the library's global scope and its NON-exported functions follow the script
  laws: gated reads there are engine-global (14264/14264).

The ``TaVariableHoistTransformer`` implements this by skipping ``@export``
functions and the module-level functions of a hand-written library, while
rewiring the library main's body and its non-exported nested functions. Both
run modes are pinned here: through a real import, and running the library
module directly as the script.

The library registration is process-global (see
``test_031_coof_same_module_library``), so the tests restore it and drop the
imported modules afterwards.
"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'


def _make_syminfo():
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


def _make_ohlcv():
    from pynecore.types.ohlcv import OHLCV
    # close and volume both alternate around the previous bar so nvi keeps moving
    bars = ((100.0, 50.0), (102.0, 40.0), (101.0, 60.0), (105.0, 30.0), (104.0, 70.0),
            (108.0, 20.0), (107.0, 90.0), (110.0, 35.0), (109.0, 80.0), (113.0, 25.0),
            (112.0, 95.0), (116.0, 45.0), (114.0, 85.0), (118.0, 15.0), (117.0, 75.0),
            (121.0, 55.0), (119.0, 65.0), (124.0, 10.0), (122.0, 100.0), (127.0, 42.0))
    return [
        OHLCV(timestamp=1_704_067_200 + bar * 1800, open=c, high=c + 1.0, low=c - 2.0,
              close=c, volume=v)
        for bar, (c, v) in enumerate(bars)
    ]


def _expected_nvi() -> list[float]:
    # Straight re-computation of ta.nvi over the fixture bars: one advancement
    # per bar — this is what pins that the library main runs exactly once per
    # bar when the module is the script (it is registered as a lib main too)
    values = []
    nvi = 1.0
    prev_close = prev_volume = 0.0
    for row in _make_ohlcv():
        if prev_close != 0.0 and row.volume < prev_volume:
            nvi += (row.close - prev_close) / prev_close * nvi
        values.append(nvi)
        prev_close, prev_volume = row.close, row.volume
    return values


def _run_script(script_name: str) -> list[dict]:
    from pynecore.core import script as script_core
    from pynecore.core.script_runner import ScriptRunner

    # noinspection PyProtectedMember
    saved_libraries = list(script_core._registered_libraries)
    try:
        runner = ScriptRunner(
            DATA_DIR / script_name, iter(_make_ohlcv()), _make_syminfo(),
        )
        return [dict(plot_data) for _candle, plot_data in runner.run_iter()]
    finally:
        # noinspection PyProtectedMember
        script_core._registered_libraries[:] = saved_libraries
        sys.modules.pop('gated_builtin_lib', None)
        sys.modules.pop('gated_builtin_lib_script', None)


def __test_imported_library_helper_is_a_per_call_site_gated_machine__():
    """ Unconditional lib call tracks the engine; a gated lib call runs its own machine """
    results = _run_script('gated_builtin_lib_script.py')

    nvi_values = set()
    gated_mismatch = 0
    for bar, plot in enumerate(results):
        every = plot["every_nvi"]
        nvi_values.add(every)
        assert plot["all_lib"] == every, \
            f"bar {bar}: unconditional library nvi {plot['all_lib']} != {every}"
        if bar % 2 == 0:
            if plot["gated"] != every:
                gated_mismatch += 1
        else:
            assert plot["gated"] == -1.0, f"bar {bar}: the branch must not run"
    assert len(nvi_values) >= 3, "nvi must actually move, or the equality proves nothing"
    assert gated_mismatch > 0, \
        "the gated library call must run its own call-gated machine and diverge (m575/m576)"


def __test_library_run_as_a_study_follows_the_same_laws__():
    """ Study mode: main-body and non-export reads track the engine, an export does not """
    results = _run_script('gated_builtin_lib.py')

    nvi_values = set()
    export_mismatch = 0
    expected = _expected_nvi()
    for bar, plot in enumerate(results):
        every = plot["every_nvi"]
        nvi_values.add(every)
        assert every == expected[bar], \
            f"bar {bar}: nvi {every} != recomputed {expected[bar]} — advanced more than once?"
        assert plot["runs"] == bar + 1, \
            f"bar {bar}: main ran {plot['runs'] - bar} time(s) this bar — it is also a " \
            f"registered library entry and must not run twice"
        if bar % 2 == 0:
            assert plot["gated_direct"] == every, \
                f"bar {bar}: main-body nvi {plot['gated_direct']} != {every}"
            assert plot["gated_local"] == every, \
                f"bar {bar}: non-exported function nvi {plot['gated_local']} != {every}"
            if plot["gated_export"] != every:
                export_mismatch += 1
        else:
            assert plot["gated_direct"] == -1.0 and plot["gated_local"] == -1.0 \
                   and plot["gated_export"] == -1.0, \
                f"bar {bar}: the branch must not run"
    assert len(nvi_values) >= 3, "nvi must actually move, or the equality proves nothing"
    assert export_mismatch > 0, \
        "the exported function must run its own call-gated machine in study mode too (m577)"
