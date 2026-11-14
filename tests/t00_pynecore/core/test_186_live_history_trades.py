"""
@pyne

``live_history_trades``: live paper trading can let the strategy trade on the
warmup bars and carry that position into the live phase. Off by default, a live
run starts flat. The last warmup bar places no orders while its period is still
open: the live feed goes on forming it.
"""
import sys
import time
import itertools

import pytest

from pynecore.lib import bar_index, barstate, plot, script, strategy


@script.strategy(
    "Live History Trades",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if barstate.ishistory and bar_index == 1:
        strategy.entry("A", strategy.long)
    if barstate.ishistory and bar_index == 2:
        strategy.close("A")
    if barstate.islastconfirmedhistory:
        strategy.entry("B", strategy.long)
    plot(strategy.closedtrades, 'closed')
    plot(strategy.position_size, 'pos')


def __test_helper_bar(ts, c, closed=True):
    from pynecore.types.ohlcv import OHLCV
    return OHLCV(timestamp=ts, open=c, high=c, low=c, close=c, volume=1.0, is_closed=closed)


def __test_helper_runner(script_path, module_key, syminfo, feed, **kwargs):
    from pynecore.core.script_runner import ScriptRunner

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)
    return ScriptRunner(script_path, feed, syminfo, **kwargs)


def __test_helper_closed_history_feed():
    """Old bars, so the last warmup bar's period is over; the live bar is a new one"""
    from pynecore.core.script_runner import LIVE_TRANSITION

    period_ms = 5 * 60 * 1000
    historical = [__test_helper_bar(i * period_ms, 100.0 + i) for i in range(5)]
    live = [
        __test_helper_bar(5 * period_ms, 110.0, closed=False),
        __test_helper_bar(5 * period_ms, 111.0),
    ]
    return itertools.chain(historical, [LIVE_TRANSITION], live)


def __test_helper_last_bars(results):
    return results[-1][1]


def __test_history_trades_off_by_default__(script_path, module_key, syminfo):
    """A live run without the switch starts flat: no trade from the warmup bars"""
    runner = __test_helper_runner(script_path, module_key, syminfo,
                                  __test_helper_closed_history_feed(), live=True)
    results = list(runner.run_iter())

    plots = __test_helper_last_bars(results)
    assert plots['closed'] == 0
    assert plots['pos'] == 0


def __test_history_trades_continue_live__(script_path, module_key, syminfo):
    """With the switch the warmup trades and its last bar's entry fills on the live bar"""
    runner = __test_helper_runner(script_path, module_key, syminfo,
                                  __test_helper_closed_history_feed(), live=True,
                                  live_history_trades=True)
    results = list(runner.run_iter())

    plots = __test_helper_last_bars(results)
    assert plots['closed'] == 1
    assert plots['pos'] == 1
    trade = runner.script.position.open_trades[0]
    assert trade.entry_id == "B"
    # Placed at the last warmup bar's close, filled at the live bar's open
    assert trade.entry_price == pytest.approx(110.0)


def __test_forming_last_warmup_bar_places_no_order__(script_path, module_key, syminfo):
    """The downloaded last bar is still forming: its run places no order"""
    from pynecore.core.script_runner import LIVE_TRANSITION

    period_ms = 5 * 60 * 1000
    forming_ts = int(time.time() * 1000) - 60 * 1000
    historical = [__test_helper_bar(forming_ts - (4 - i) * period_ms, 100.0 + i)
                  for i in range(5)]
    live = [
        __test_helper_bar(forming_ts, 110.0, closed=False),
        __test_helper_bar(forming_ts, 111.0),
    ]
    feed = itertools.chain(historical, [LIVE_TRANSITION], live)
    runner = __test_helper_runner(script_path, module_key, syminfo, feed, live=True,
                                  live_history_trades=True)
    # The yielded trade list is the runner's per-bar buffer: copy it on every item
    results = [(c, p, list(t)) for c, p, t in runner.run_iter()]

    plots = __test_helper_last_bars(results)
    # The earlier warmup bars still traded
    assert plots['closed'] == 1
    assert plots['pos'] == 0
    # The suppressed bar does not repeat the previous bar's close
    assert sum(len(t) for _, _, t in results) == 1


def __test_catchup_close_replaces_partial_last_history_bar__(script_path, module_key, syminfo):
    """The catch-up drain's true close of the partial last history bar is the same bar"""
    from pynecore.core.script_runner import LIVE_TRANSITION

    period_ms = 5 * 60 * 1000
    historical = [__test_helper_bar(i * period_ms, 100.0 + i) for i in range(4)]
    historical.append(__test_helper_bar(4 * period_ms, 104.0, closed=False))
    catchup = [__test_helper_bar(4 * period_ms, 105.0)]
    live = [
        __test_helper_bar(5 * period_ms, 110.0, closed=False),
        __test_helper_bar(5 * period_ms, 111.0),
    ]
    feed = itertools.chain(historical, catchup, [LIVE_TRANSITION], live)
    runner = __test_helper_runner(script_path, module_key, syminfo, feed, live=True,
                                  live_history_trades=True)
    results = list(runner.run_iter())

    timestamps = [c.timestamp for c, _, _ in results]
    assert len(timestamps) == len(set(timestamps))
    trade = runner.script.position.open_trades[0]
    assert trade.entry_id == "B"
    assert trade.entry_bar_index == 5
    assert trade.entry_price == pytest.approx(110.0)


def __test_period_ended_during_warmup_keeps_partial_bar_suppressed__(script_path, module_key,
                                                                     syminfo):
    """A bar partial at download stays partial when its period ends before the warmup reaches it"""
    from pynecore.core.script_runner import LIVE_TRANSITION
    from pynecore.types.ohlcv import OHLCV

    period_ms = 5 * 60 * 1000
    # The period is over by now, but the history was downloaded one minute into it
    partial_ts = int(time.time() * 1000) - 2 * period_ms
    historical = [__test_helper_bar(partial_ts - (4 - i) * period_ms, 100.0 + i)
                  for i in range(4)]
    historical.append(OHLCV(timestamp=partial_ts, open=100.0, high=104.0, low=100.0,
                            close=104.0, volume=1.0))
    live = [
        OHLCV(timestamp=partial_ts, open=100.0, high=105.0, low=100.0, close=105.0, volume=2.0),
        __test_helper_bar(partial_ts + period_ms, 110.0, closed=False),
        __test_helper_bar(partial_ts + period_ms, 111.0),
    ]
    feed = itertools.chain(historical, [LIVE_TRANSITION], live)
    runner = __test_helper_runner(script_path, module_key, syminfo, feed, live=True,
                                  live_history_trades=True,
                                  history_as_of=partial_ts + 60 * 1000)
    results = list(runner.run_iter())

    # The partial run's entry would fill at the bar's own open, before it was placed
    assert not runner.script.position.open_trades
    plots = __test_helper_last_bars(results)
    assert plots['closed'] == 1
    assert plots['pos'] == 0


def __test_history_trades_rejected_outside_live_paper__(script_path, module_key, syminfo):
    """Backtests have nothing to switch, broker runs must not trade on history"""
    with pytest.raises(ValueError):
        __test_helper_runner(script_path, module_key, syminfo, iter([]),
                             live_history_trades=True)
