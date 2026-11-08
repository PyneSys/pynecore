"""
@pyne
"""
from pynecore.lib import bar_index, close, input, script, strategy


@script.strategy("Trade recorder", initial_capital=10000,
                 default_qty_type=strategy.fixed, default_qty_value=1)
def main(mode=input.int(0), every=input.int(20)):
    entry_bar = strategy.position_size == 0 and bar_index % every == 5
    if entry_bar:
        strategy.entry("L", strategy.long)
    if entry_bar and mode == 0:
        strategy.exit("LX", "L", stop=close - 2.0, limit=close + 4.0)
    if entry_bar and mode == 1:
        strategy.exit("LX", "L", loss=200000, profit=400000)
    if mode == 2 and strategy.position_size > 0:
        strategy.exit("LX", "L", stop=strategy.position_avg_price - 2.0,
                      limit=strategy.position_avg_price + 4.0)
    if mode == 3 and strategy.position_size > 0 and close < strategy.position_avg_price - 0.5:
        # The first stop of an unprotected trade, filled at the close it is placed on
        strategy.exit("LX", "L", stop=close + 0.25)
    if mode == 3 and bar_index % every == 4:
        strategy.close("L")


def __test_helper_bars(spread=1.5):
    """A sine wave of hourly bars that reaches 2 below and 4 above most entries."""
    import math
    from pynecore.types.ohlcv import OHLCV
    start = 1_735_689_600_000
    bars = []
    for i in range(600):
        mid = 100.0 + 6.0 * math.sin(i / 6.0)
        bars.append(OHLCV(timestamp=start + i * 3_600_000, open=mid, high=mid + spread,
                          low=mid - spread, close=mid + 0.2, volume=1.0))
    return bars


def __test_helper_record(runner, mode, calc_on_order_fills=False, spread=1.5,
                         process_orders_on_close=False):
    from pynecore.analysis import TradeRecorder
    bars = __test_helper_bars(spread)
    r = runner(iter(bars), inputs={'mode': mode}, last_bar_index=len(bars) - 1,
               last_bar_time=bars[-1].timestamp)
    r.script.calc_on_order_fills = calc_on_order_fills
    r.script.process_orders_on_close = process_orders_on_close
    position = r.script.position
    recorder = TradeRecorder(position, r.syminfo)
    for _ in r.run_iter(copy_results=False):
        recorder.on_bar()
    return recorder.trades, position


def __test_stop_placed_with_the_entry__(runner):
    """An exit placed together with the entry gives the trade its initial stop."""
    trades, position = __test_helper_record(runner, 0)
    assert len(trades) == position.closed_trades_count > 5
    assert all(t.initial_stop is not None for t in trades)
    stopped = [t for t in trades if t.exit_price == t.initial_stop]
    assert stopped
    for t in stopped:
        assert abs(t.r + 1.0) < 1e-9
    assert abs(sum(t.profit for t in trades) - position.netprofit) < 1e-9


def __test_tick_offset_stop__(runner):
    """A ``loss`` tick offset resolves to a level measured from the entry fill."""
    trades, _ = __test_helper_record(runner, 1)
    assert trades
    for t in trades:
        assert t.initial_stop is not None
        assert abs(t.entry_price - 2.0 - t.initial_stop) < 1e-6


def __test_stop_placed_after_the_fill__(runner):
    """An exit first placed while the trade is open becomes its initial stop."""
    trades, _ = __test_helper_record(runner, 2)
    assert trades
    for t in trades:
        assert t.initial_stop is not None
        assert abs(t.entry_price - 2.0 - t.initial_stop) < 1e-6
        assert t.r is not None


def __test_stop_placed_and_filled_on_the_entry_bar__(runner):
    """A stop a calc_on_order_fills re-execution places and fills within one bar counts."""
    # Every bar reaches 2.5 below its open: the stop fills on the entry's own bar
    trades, _ = __test_helper_record(runner, 2, calc_on_order_fills=True, spread=2.5)
    assert any(t.entry_bar_index == t.exit_bar_index for t in trades)
    for t in trades:
        assert t.initial_stop is not None
        assert abs(t.entry_price - 2.0 - t.initial_stop) < 1e-6


def __test_first_stop_filled_on_a_later_bar__(runner):
    """An unprotected trade's first stop counts when it fills on the bar it is placed on."""
    trades, _ = __test_helper_record(runner, 3, process_orders_on_close=True)
    stopped = [t for t in trades if t.exit_comment == 'LX']
    assert len(stopped) > 3
    for t in stopped:
        assert t.initial_stop is not None
        assert t.initial_stop < t.entry_price
        assert t.r is not None

