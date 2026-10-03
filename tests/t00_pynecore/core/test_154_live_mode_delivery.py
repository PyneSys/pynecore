"""
@pyne
"""
from pynecore.lib import script, strategy, close, plot


@script.strategy("Live mode delivery", initial_capital=1000000)
def main():
    # pyramiding=1: once the position is open, the repeated entry is ignored
    strategy.entry("L", strategy.long, qty=1)
    plot(close, "close")
    plot(strategy.position_size, "size")


def __test_helper_bar(i, is_closed=True):
    from pynecore.types.ohlcv import OHLCV
    price = 100.0 + i
    return OHLCV(timestamp=i * 60_000, open=price, high=price + 1, low=price - 1,
                 close=price, volume=1000.0, is_closed=is_closed)


def __test_helper_runner(script_path, module_key, syminfo, ohlcv_iter):
    import sys
    from pynecore.core.script_runner import ScriptRunner

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)
    return ScriptRunner(script_path, ohlcv_iter, syminfo, live=True)


def __test_closed_live_bar_is_delivered_before_the_next_item__(script_path, module_key, syminfo):
    """ A closed live bar's result arrives before the runner asks the feed for more """
    from pynecore.core.script_runner import LIVE_TRANSITION

    events = []

    def feed():
        for i in range(3):
            events.append(("feed", i))
            yield __test_helper_bar(i)
        yield LIVE_TRANSITION
        for i in range(3, 6):
            events.append(("feed", i))
            yield __test_helper_bar(i)

    runner = __test_helper_runner(script_path, module_key, syminfo, feed())
    for candle, _plot, _trades in runner.run_iter():
        events.append(("result", candle.timestamp // 60_000))

    live_events = events[events.index(("feed", 3)):]
    assert live_events == [("feed", 3), ("result", 3), ("feed", 4), ("result", 4),
                           ("feed", 5), ("result", 5)]


def __test_strategy_orders_start_at_the_live_transition__(script_path, module_key, syminfo):
    """ Warmup bars build state only; the first order is placed on the first live bar """
    import itertools
    from pynecore.core.script_runner import LIVE_TRANSITION

    historical = [__test_helper_bar(i) for i in range(3)]
    live = [__test_helper_bar(i) for i in range(3, 6)]
    runner = __test_helper_runner(script_path, module_key, syminfo,
                                  itertools.chain(historical, [LIVE_TRANSITION], live))

    sizes = [plot_data["size"] for _c, plot_data, _t in runner.run_iter()]

    # The entry placed at the first live close fills at the next bar's open
    assert sizes == [0, 0, 0, 0, 1, 1]


def __test_live_flags_are_reset_after_the_run__(script_path, module_key, syminfo):
    """ Live mode belongs to its run: a later run of another runner starts historical """
    import itertools
    from pynecore import lib
    from pynecore.core.script_runner import LIVE_TRANSITION

    runner = __test_helper_runner(
        script_path, module_key, syminfo,
        itertools.chain([__test_helper_bar(0)], [LIVE_TRANSITION], [__test_helper_bar(1)]))
    for _row in runner.run_iter():
        assert getattr(lib, '_is_live')

    assert not getattr(lib, '_is_live')
    assert not getattr(lib, '_strategy_suppressed')
