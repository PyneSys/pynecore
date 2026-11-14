"""
@pyne

``run_iter(yield_ticks=True)`` yields an item for every intra-bar update of the
forming bar in live mode, carrying the update itself (``is_closed=False``) and the
results of the body's run on it. The bar's close is still yielded once, as without
the switch.
"""
import sys
import itertools

from pynecore.lib import barstate, close, plot, script
from pynecore.types import IBPersistent


@script.indicator("Live Yield Ticks", overlay=True)
def main():
    runs: IBPersistent[int] = 0
    if barstate.isrealtime:
        runs += 1
    plot(close, 'close')
    plot(runs, 'runs')


def __test_helper_bar(ts, c, closed=True):
    from pynecore.types.ohlcv import OHLCV
    return OHLCV(timestamp=ts, open=100.0, high=max(100.0, c), low=min(100.0, c), close=c,
                 volume=1.0, is_closed=closed)


def __test_helper_run(script_path, module_key, syminfo, yield_ticks):
    from pynecore.core.script_runner import ScriptRunner, LIVE_TRANSITION

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)

    historical = [__test_helper_bar(i * 60, 100.0) for i in range(3)]
    live = [
        __test_helper_bar(180, 101.0, closed=False),
        __test_helper_bar(180, 102.0, closed=False),
        __test_helper_bar(180, 103.0),
    ]
    runner = ScriptRunner(script_path, itertools.chain(historical, [LIVE_TRANSITION], live),
                          syminfo, live=True)
    return [(candle, dict(plot_data))
            for candle, plot_data in runner.run_iter(yield_ticks=yield_ticks)]


def __test_forming_bar_yielded_per_update__(script_path, module_key, syminfo):
    """Each intra-bar update is yielded with its own values, then the close"""
    results = __test_helper_run(script_path, module_key, syminfo, yield_ticks=True)

    assert len(results) == 6
    ticks = results[3:]
    assert [candle.is_closed for candle, _ in ticks] == [False, False, True]
    assert [plots['close'] for _, plots in ticks] == [101.0, 102.0, 103.0]
    assert [plots['runs'] for _, plots in ticks] == [1, 2, 3]


def __test_closed_bars_only_by_default__(script_path, module_key, syminfo):
    """Without the switch only closed bars are yielded"""
    results = __test_helper_run(script_path, module_key, syminfo, yield_ticks=False)

    assert len(results) == 4
    assert all(candle.is_closed for candle, _ in results)
    assert results[3][1]['close'] == 103.0

