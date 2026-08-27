"""
@pyne

Regression test for the ``timenow`` pin that makes a backtest reproducible.

``timenow`` is the real system clock, exactly like TradingView. A script that
gates its entries on it — ``time >= timenow - N days``, or "only today's bars" —
therefore measures a DIFFERENT bar set on every run, and a stored reference can
never be matched again once a day has passed. ``pyne run`` pins the value to the
last bar of the window it replays (see ``run._pin_timenow``), so the same data
always yields the same run.
"""
import sys
from pathlib import Path

from pynecore.lib import plot, script, timenow


@script.indicator(title="Timenow Pin", shorttitle="tnp")
def main():
    plot(timenow, "now")


# Every timestamp here is Unix MILLISECONDS.
_TS0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 5-minute grid
_STEP = 300_000  # conftest syminfo period is "5" (5-minute bars)


def _bars(n):
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=_TS0 + i * _STEP, open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0)
            for i in range(n)]


def _make_runner(script_path, module_key, syminfo, bars, **kwargs):
    from pynecore.core.script_runner import ScriptRunner
    sys.modules.pop(module_key, None)
    sys.modules.pop(Path(script_path).stem, None)
    return ScriptRunner(script_path, bars, syminfo, **kwargs)


# noinspection PyProtectedMember
def __test_timenow_pinned_to_the_last_bar__(script_path, module_key, syminfo):
    """With the pin set, every bar of the run reads the same anchored instant."""
    from pynecore import lib

    final_ms = _TS0 + 3 * _STEP
    previous = lib._timenow_ms
    lib._timenow_ms = final_ms
    try:
        r = _make_runner(script_path, module_key, syminfo, _bars(4))
        rows = [dict(_plot) for _, _plot in r.run_iter()]
    finally:
        lib._timenow_ms = previous

    assert [row["now"] for row in rows] == [final_ms] * 4


# noinspection PyProtectedMember
def __test_timenow_reads_the_clock_without_a_pin__(script_path, module_key, syminfo):
    """No pin (live semantics): the system clock, not the bar's timestamp."""
    from datetime import datetime, UTC
    from pynecore import lib

    previous = lib._timenow_ms
    lib._timenow_ms = 0
    try:
        before = int(datetime.now(UTC).timestamp() * 1000)
        r = _make_runner(script_path, module_key, syminfo, _bars(3))
        rows = [dict(_plot) for _, _plot in r.run_iter()]
        after = int(datetime.now(UTC).timestamp() * 1000)
    finally:
        lib._timenow_ms = previous

    for row in rows:
        assert before <= row["now"] <= after, (
            f"Unpinned timenow {row['now']} outside the run's wall-clock window"
        )
