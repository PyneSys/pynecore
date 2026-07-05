"""
@pyne
"""
import sys
from pathlib import Path

from pynecore.lib import chart, last_bar_time, plot, script, time


@script.indicator(title="Last Bar Time Anchor", shorttitle="lbta")
def main():
    plot(last_bar_time, "lbt")
    plot(chart.right_visible_bar_time, "rvbt")
    plot(chart.left_visible_bar_time, "lvbt")
    plot(1 if time == chart.right_visible_bar_time else 0, "vis")


_TS0 = 1735689600  # 2025-01-01T00:00:00 UTC, aligned to the 5-minute grid
_STEP = 300  # conftest syminfo period is "5" (5-minute bars)
_VISIBLE_SPAN_MS = 20 * _STEP * 1000  # chart._visible_bars bars back from the right edge


def _bars(n):
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=_TS0 + i * _STEP, open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0)
            for i in range(n)]


def _make_runner(script_path, module_key, syminfo, bars, **kwargs):
    from pynecore.core.script_runner import ScriptRunner
    sys.modules.pop(module_key, None)
    sys.modules.pop(Path(script_path).stem, None)
    return ScriptRunner(script_path, bars, syminfo, **kwargs)


def __test_last_bar_time_anchored__(script_path, module_key, syminfo):
    """Historical anchor: with ``last_bar_time`` passed to the runner (as ``pyne run``
    does from the data window), every bar sees the run's FINAL bar time, the visible
    range hangs off it, and ``time == chart.right_visible_bar_time`` is true only on
    the final bar — Pine's fixed viewport on historical bars."""
    final_ms = (_TS0 + 3 * _STEP) * 1000
    r = _make_runner(script_path, module_key, syminfo, _bars(4),
                     last_bar_index=3, last_bar_time=final_ms)

    rows = [dict(_plot) for _, _plot in r.run_iter()]
    assert [row["lbt"] for row in rows] == [final_ms] * 4
    assert [row["rvbt"] for row in rows] == [final_ms] * 4
    assert [row["lvbt"] for row in rows] == [final_ms - _VISIBLE_SPAN_MS] * 4
    assert [row["vis"] for row in rows] == [0, 0, 0, 1]


def __test_last_bar_time_tracking_default__(script_path, module_key, syminfo):
    """No anchor (``last_bar_time=None`` — live semantics): ``last_bar_time`` tracks
    the current bar, so the current bar is always the right edge of the viewport."""
    r = _make_runner(script_path, module_key, syminfo, _bars(3))

    rows = [dict(_plot) for _, _plot in r.run_iter()]
    times = [(_TS0 + i * _STEP) * 1000 for i in range(3)]
    assert [row["lbt"] for row in rows] == times
    assert [row["rvbt"] for row in rows] == times
    assert [row["vis"] for row in rows] == [1, 1, 1]
