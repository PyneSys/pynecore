"""
@pyne

The last warmup bar is executed once more when the live feed continues it under
the same timestamp, and that warmup run has to be discarded -- variables and
drawings alike. Without calc_on_every_tick the rollback used to be skipped
entirely (it hung off ``run_on_every_tick``) and no variable snapshot even
existed, so the bar was counted twice and every later bar stayed off by one.
"""
import sys
import itertools

from pynecore.lib import array, line, plot, script, strategy, bar_index
from pynecore.types import Persistent

# A module-level global is outside the slot scheme, so nothing rolls it back --
# it counts body executions, not bars.
_execs: list[int] = []


@script.strategy(
    "Live Transition No Every Tick",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    max_lines_count=500,
)
def main():
    counter: Persistent[int] = 0
    counter += 1
    _execs.append(bar_index)
    line.new(bar_index, 0.0, bar_index, 1.0)
    plot(counter, 'var')
    plot(array.size(line.all), 'lines')
    plot(len(_execs), 'total_execs')


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


def _run(script_path, module_key, syminfo, every_tick):
    """Run four bars where the live feed continues the last warmup bar."""
    from pynecore.core import viz
    _execs.clear()

    historical = [_make_ohlcv(0, 100.0), _make_ohlcv(60, 101.0)]
    live = [
        _make_ohlcv(60, 101.5, is_closed=False),  # continues the warmup bar
        _make_ohlcv(60, 102.0, is_closed=True),
        _make_ohlcv(120, 103.0, is_closed=True),
        _make_ohlcv(180, 104.0, is_closed=True),
    ]

    try:
        runner = _create_live_runner(
            script_path, module_key, syminfo, _chain_live(historical, live))
        runner.script.calc_on_every_tick = every_tick
        results = []
        for _candle, plot_data, _trades in runner.run_iter():
            results.append(dict(plot_data))
    finally:
        viz.reset_state()
    return results


def __test_warmup_bar_continued_live_counts_once__(script_path, module_key, syminfo):
    """ The continued warmup bar is one bar, whether or not the script runs on ticks """
    for every_tick in (False, True):
        results = _run(script_path, module_key, syminfo, every_tick)

        # Warmup bars 0 and 60, the closed live bar 60, then live bars 120, 180
        assert len(results) == 5, f"every_tick={every_tick}"

        # Four distinct bars, so four increments and four lines -- bar 60 must
        # not count twice just because the live feed continued it
        assert [r['var'] for r in results] == [1, 2, 2, 3, 4], f"every_tick={every_tick}"
        assert [r['lines'] for r in results] == [1, 2, 2, 3, 4], f"every_tick={every_tick}"


def __test_every_tick_runs_the_continued_bar_more_often__(script_path, module_key, syminfo):
    """ The rolled-back state is identical although the tick path executes more """
    without_ticks = _run(script_path, module_key, syminfo, False)
    with_ticks = _run(script_path, module_key, syminfo, True)

    # Bar 60 runs twice without tick execution (warmup + live close) and three
    # times with it (warmup + open tick + live close)
    assert [r['total_execs'] for r in without_ticks] == [1, 2, 3, 4, 5]
    assert [r['total_execs'] for r in with_ticks] == [1, 2, 4, 5, 6]
