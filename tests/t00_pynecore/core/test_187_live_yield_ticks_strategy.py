"""
@pyne

``run_iter(yield_ticks=True)`` with a strategy that runs on the bar close only:
every intra-bar update is still yielded, so the forming candle can be drawn, but
with empty plot data and trade list, since nothing was calculated on it.
"""
import sys
import itertools

from pynecore.lib import close, plot, script, strategy


@script.strategy("Live Yield Ticks Strategy", overlay=True, initial_capital=100000)
def main():
    plot(close, 'close')
    plot(strategy.position_size, 'pos')


def __test_helper_bar(ts, c, closed=True):
    from pynecore.types.ohlcv import OHLCV
    return OHLCV(timestamp=ts, open=100.0, high=max(100.0, c), low=min(100.0, c), close=c,
                 volume=1.0, is_closed=closed)


def __test_update_without_body_run_carries_only_the_candle__(script_path, module_key, syminfo):
    """Each update is yielded with its candle; plots arrive with the bar close"""
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
    results = list(runner.run_iter(yield_ticks=True))

    ticks = results[3:]
    assert [candle.close for candle, _plots, _trades in ticks] == [101.0, 102.0, 103.0]
    assert [candle.is_closed for candle, _plots, _trades in ticks] == [False, False, True]
    assert [plots for _candle, plots, _trades in ticks[:2]] == [{}, {}]
    assert [trades for _candle, _plots, trades in ticks[:2]] == [[], []]
    assert ticks[2][1]['close'] == 103.0
