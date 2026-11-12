"""
@pyne
"""
from pynecore.lib import script, close, open, plot


@script.strategy("Live unclosed bar plot ids", calc_on_every_tick=True)
def main():
    plot(close)
    plot(open)


def __test_helper_bar(i, is_closed=True):
    from pynecore.types.ohlcv import OHLCV
    price = 100.0 + i
    return OHLCV(timestamp=i * 60_000, open=price, high=price + 1, low=price - 1,
                 close=price, volume=1000.0, is_closed=is_closed)


def __test_next_bar_keeps_its_plot_ids_when_a_close_never_comes__(script_path, module_key,
                                                                   syminfo):
    """ A bar opened after one whose close never came plots under its own titles """
    import sys
    from pynecore import lib
    from pynecore.core.script_runner import LIVE_TRANSITION, ScriptRunner

    def feed():
        for i in range(3):
            yield __test_helper_bar(i)
        yield LIVE_TRANSITION
        yield __test_helper_bar(3, is_closed=False)
        yield __test_helper_bar(3, is_closed=False)
        # Bar 3 never closes: bar 4 opens on the next update
        yield __test_helper_bar(4, is_closed=False)
        yield __test_helper_bar(4)

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)
    runner = ScriptRunner(script_path, feed(), syminfo, live=True)
    rows = []
    for candle, plot_data, _trades in runner.run_iter():
        rows.append((candle.timestamp // 60_000, dict(plot_data), list(lib._plot_meta)))

    assert rows[-1] == (4, {"Plot": 104.0, "Plot 0": 104.0}, ["Plot", "Plot 0"])
