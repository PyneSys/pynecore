"""
@pyne

The historical->live transition re-executes the last warmup bar whenever the
first live update carries its timestamp -- which is the normal case, because
``download_ohlcv`` returns the still-open current bar. That warmup execution is
discarded, so its drawings have to go with it.

The live branches snapshot the drawings only when a NEW bar opens, so without a
baseline captured at the end of the warmup this bar's every execution piled up
in the registry and ate the script's ``max_lines_count`` budget.
"""
import sys
import itertools

from pynecore.lib import array, line, script, bar_index, close


@script.indicator(title="Live Transition Drawing Rollback", max_lines_count=500)
def main():
    # ``close`` as the second endpoint makes every execution of the same bar
    # distinguishable, so a test can tell a replaced run from a skipped one
    line.new(bar_index, 0.0, bar_index, close)
    return {"lines": array.size(line.all)}


def _make_ohlcv(ts, close=100.0, is_closed=True):
    from pynecore.types.ohlcv import OHLCV
    return OHLCV(timestamp=ts, open=close, high=close + 1, low=close - 1,
                 close=close, volume=1000.0, is_closed=is_closed)


def _create_live_runner(script_path, module_key, syminfo, ohlcv_iter):
    """Helper: set live mode flags, clean module cache, create ScriptRunner."""
    from pynecore.core.script_runner import ScriptRunner
    from pynecore import lib

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)

    setattr(lib, '_is_live', True)
    setattr(lib, '_strategy_suppressed', True)
    return ScriptRunner(script_path, ohlcv_iter, syminfo)


def _chain_live(historical, live):
    """Chain historical OHLCV with LIVE_TRANSITION sentinel and live OHLCV."""
    from pynecore.core.script_runner import LIVE_TRANSITION
    return itertools.chain(historical, [LIVE_TRANSITION], live)


def __test_live_continuation_of_warmup_bar_drops_its_drawings__(
        script_path, module_key, syminfo):
    """ Live ticks continuing the last warmup bar leave one drawing, not one per run """
    from pynecore.core import viz

    historical = [
        _make_ohlcv(0, 100.0),
        # The still-open current bar, returned by the history download and then
        # continued by the live feed under the same timestamp
        _make_ohlcv(60, 101.0, is_closed=False),
    ]
    live = [
        _make_ohlcv(60, 101.5, is_closed=False),
        _make_ohlcv(60, 102.0, is_closed=True),
    ]

    try:
        runner = _create_live_runner(
            script_path, module_key, syminfo,
            _chain_live(historical, live),
        )
        results = [dict(plot_data) for _candle, plot_data in runner.run_iter()]
    finally:
        viz.reset_state()

    # Warmup bar 0, warmup bar 60, then the closed live bar 60
    assert len(results) == 3

    # Bar 60 runs three times in total (warmup + two live updates) but is one
    # bar, so it must not leave more than the one line of its final run
    assert [r['lines'] for r in results] == [1, 2, 2]


def __test_update_after_bar_close_replaces_that_bars_run__(
        script_path, module_key, syminfo):
    """ A non-closed update repeating a closed bar's timestamp replaces its run """
    from pynecore.core import viz
    from pynecore.lib import line as line_mod

    historical = [_make_ohlcv(0, 100.0), _make_ohlcv(60, 101.0)]
    live = [
        _make_ohlcv(120, 102.0, is_closed=True),
        # Providers keep emitting intra-bar updates under the closed bar's
        # timestamp until the next one opens; only duplicate CLOSED bars are
        # filtered out upstream, so this reaches the runner
        _make_ohlcv(120, 102.5, is_closed=False),
        _make_ohlcv(180, 103.0, is_closed=True),
    ]

    try:
        runner = _create_live_runner(
            script_path, module_key, syminfo,
            _chain_live(historical, live),
        )
        results = [dict(plot_data) for _candle, plot_data in runner.run_iter()]
        # noinspection PyProtectedMember
        drawn = [(ln.x1, ln.y2) for ln in line_mod._registry]
    finally:
        viz.reset_state()

    # One line per bar: bar 2's post-close update re-executes that bar instead
    # of stacking a second line on top of the run it already committed
    assert [r['lines'] for r in results] == [1, 2, 3, 4]
    # ...and the surviving line of bar 2 is the LAST run's, so the update did
    # re-execute the body rather than being ignored
    assert drawn == [(0, 100.0), (1, 101.0), (2, 102.5), (3, 103.0)]


def __test_new_live_bar_keeps_its_own_drawing__(script_path, module_key, syminfo):
    """ A live update opening a fresh bar is not a re-execution -- its drawing stays """
    from pynecore.core import viz

    historical = [_make_ohlcv(0, 100.0), _make_ohlcv(60, 101.0)]
    live = [_make_ohlcv(120, 102.0, is_closed=True)]

    try:
        runner = _create_live_runner(
            script_path, module_key, syminfo,
            _chain_live(historical, live),
        )
        results = [dict(plot_data) for _candle, plot_data in runner.run_iter()]
    finally:
        viz.reset_state()

    assert [r['lines'] for r in results] == [1, 2, 3]
