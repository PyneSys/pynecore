"""
@pyne

Paper trading in live mode with ``calc_on_every_tick``: an order placed on one
tick of the forming bar fills on the next tick, at a price that comes after it,
and the body sees the fill on the tick it happened on. A live update carries the
forming bar so far, so its open is the bar open and its high/low reach back to
it; none of that may fill an order placed later in the bar.
"""
import sys
import itertools

import pytest

from pynecore.lib import barstate, close, plot, script, strategy
from pynecore.types import IBPersistent


@script.strategy(
    "Live Paper Tick Fills",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    calc_on_every_tick=True,
)
def main():
    ticks: IBPersistent[int] = 0
    first_tick_in_position: IBPersistent[int] = -1
    entered: IBPersistent[bool] = False
    if barstate.isrealtime:
        ticks += 1
        if strategy.position_size > 0 and first_tick_in_position < 0:
            first_tick_in_position = ticks
        if close >= 110 and not entered:
            entered = True
            strategy.entry("L", strategy.long)
            strategy.exit("X", "L", limit=115, stop=108)
    plot(ticks, 'ticks')
    plot(first_tick_in_position, 'first_tick_in_position')
    plot(strategy.position_size, 'pos')


def __test_helper_bar(ts, o, h, l, c, closed=True):
    from pynecore.types.ohlcv import OHLCV
    return OHLCV(timestamp=ts, open=o, high=h, low=l, close=c, volume=1.0, is_closed=closed)


def __test_helper_live_bar(ts, prices, closing):
    """Cumulative live updates of one bar: each carries the bar so far."""
    updates = []
    o = prices[0]
    hi = lo = o
    for p in prices:
        hi, lo = max(hi, p), min(lo, p)
        updates.append(__test_helper_bar(ts, o, hi, lo, p, closed=False))
    hi, lo = max(hi, closing), min(lo, closing)
    updates.append(__test_helper_bar(ts, o, hi, lo, closing))
    return updates


def __test_helper_run(script_path, module_key, syminfo, live, on_close=False):
    from pynecore.core.script_runner import ScriptRunner, LIVE_TRANSITION

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)

    historical = [__test_helper_bar(i * 60, 100.0, 100.0, 100.0, 100.0) for i in range(3)]
    # The bars sit on a timeline of their own, far in the past: on the system clock
    # the live bar would be a catch-up bar, which does not trade. The clock reads the
    # end of the live bar the updates belong to, as when the feed delivers on time.
    runner = ScriptRunner(script_path, itertools.chain(historical, [LIVE_TRANSITION], live),
                          syminfo, live=True, live_clock=lambda: live[-1].timestamp + 60)
    runner.script.process_orders_on_close = on_close
    results = []
    for _candle, plot_data, trades in runner.run_iter():
        results.append((dict(plot_data), list(trades)))
    return runner, results


def __test_market_entry_fills_on_next_tick__(script_path, module_key, syminfo):
    """The entry placed at 110 fills at the next tick (111), not at the bar open"""
    live = __test_helper_live_bar(180, [100.0, 102.0, 105.0, 110.0, 111.0], 112.0)
    runner, results = __test_helper_run(script_path, module_key, syminfo, live)

    plots, closed = results[3]
    assert plots['ticks'] == 6
    # Placed on tick 4, filled on tick 5, which is the first run in position
    assert plots['first_tick_in_position'] == 5
    assert plots['pos'] == 1
    assert not closed
    trade = runner.script.position.open_trades[0]
    assert trade.entry_price == pytest.approx(111.0)


def __test_exit_limit_fills_on_extreme_between_ticks__(script_path, module_key, syminfo):
    """Entry and limit exit close on later ticks of one bar; the bar yields the trade"""
    live = __test_helper_live_bar(180, [100.0, 105.0, 110.0, 111.0], 113.0)
    # The feed skipped the trades up to 116: the closing update reports the new
    # high while its own price is back at 113
    live[-1] = live[-1]._replace(high=116.0)
    _runner, results = __test_helper_run(script_path, module_key, syminfo, live)

    plots, closed = results[3]
    assert plots['pos'] == 0
    assert len(closed) == 1
    assert closed[0].entry_price == pytest.approx(111.0)
    # The price passed the 115 limit on its way to 116: it fills at its own level
    assert closed[0].exit_price == pytest.approx(115.0)


def __test_limit_stepped_over_between_ticks_fills_at_tick_price__(
        script_path, module_key, syminfo):
    """The price steps from 111 to 116 past the 115 limit: it fills at 116"""
    live = __test_helper_live_bar(180, [100.0, 105.0, 110.0, 111.0, 116.0], 114.0)
    _runner, results = __test_helper_run(script_path, module_key, syminfo, live)

    _plots, closed = results[3]
    assert len(closed) == 1
    assert closed[0].entry_price == pytest.approx(111.0)
    # No price between the two ticks: the next price after the limit is 116
    assert closed[0].exit_price == pytest.approx(116.0)


def __test_stop_stepped_over_between_ticks_fills_at_tick_price__(
        script_path, module_key, syminfo):
    """The price steps from 111 to 107 past the 108 stop: it fills at 107"""
    live = __test_helper_live_bar(180, [100.0, 105.0, 110.0, 111.0, 107.0], 107.0)
    _runner, results = __test_helper_run(script_path, module_key, syminfo, live)

    _plots, closed = results[3]
    assert len(closed) == 1
    assert closed[0].entry_price == pytest.approx(111.0)
    # 107 is above the bar's low (100) the cumulative update still carries: the
    # stop must not be filled from that old low, only from the tick itself
    assert closed[0].exit_price == pytest.approx(107.0)


def __test_entry_on_closing_update_fills_at_close_with_on_close__(
        script_path, module_key, syminfo):
    """process_orders_on_close fills the entry the closing update placed at that close"""
    live = __test_helper_live_bar(180, [100.0, 102.0, 105.0], 110.0)
    runner, _results = __test_helper_run(script_path, module_key, syminfo, live, on_close=True)

    trades = runner.script.position.open_trades
    assert len(trades) == 1
    assert trades[0].entry_price == pytest.approx(110.0)


def __test_entry_on_closing_update_waits_without_on_close__(
        script_path, module_key, syminfo):
    """Without process_orders_on_close the entry placed on the closing update stays queued"""
    live = __test_helper_live_bar(180, [100.0, 102.0, 105.0], 110.0)
    runner, _results = __test_helper_run(script_path, module_key, syminfo, live)

    assert not runner.script.position.open_trades
