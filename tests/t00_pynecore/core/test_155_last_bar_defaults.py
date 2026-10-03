"""
@pyne
"""
from pynecore.lib import script, plot, last_bar_index, last_bar_time


@script.indicator("Last bar defaults")
def main():
    plot(last_bar_index, "lbi")
    plot(last_bar_time, "lbt")


def __test_helper_bars():
    from pynecore.types.ohlcv import OHLCV
    bars = []
    for i in range(5):
        price = 100.0 + i
        bars.append(OHLCV(timestamp=i * 60_000, open=price, high=price + 1, low=price - 1,
                          close=price, volume=1000.0))
    return bars


def __test_helper_run(script_path, module_key, syminfo, ohlcv_iter, **kwargs):
    import sys
    from pynecore.core.script_runner import ScriptRunner

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)
    runner = ScriptRunner(script_path, ohlcv_iter, syminfo, **kwargs)
    return [(plot_data["lbi"], plot_data["lbt"]) for _c, plot_data in runner.run_iter()]


def __test_list_feed_knows_its_last_bar__(script_path, module_key, syminfo):
    """ A list feed fixes both values to its final bar on every bar """
    rows = __test_helper_run(script_path, module_key, syminfo, __test_helper_bars())
    assert rows == [(4, 240_000)] * 5


def __test_stream_feed_tracks_the_current_bar__(script_path, module_key, syminfo):
    """ A feed of unknown length reports the current bar, as on a realtime bar """
    rows = __test_helper_run(script_path, module_key, syminfo, iter(__test_helper_bars()))
    assert rows == [(i, i * 60_000) for i in range(5)]


def __test_explicit_values_win__(script_path, module_key, syminfo):
    """ Explicit values are used as given, for a list and for a stream """
    expected = [(9, 540_000)] * 5
    for feed in (__test_helper_bars(), iter(__test_helper_bars())):
        rows = __test_helper_run(script_path, module_key, syminfo, feed,
                                 last_bar_index=9, last_bar_time=540_000)
        assert rows == expected
