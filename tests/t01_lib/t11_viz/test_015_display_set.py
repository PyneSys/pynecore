"""
@pyne

``display.*`` is a set of chart places, combined with ``+`` (union) and ``-``
(difference), and the visual-data stream names the places a value covers.

The expected values were measured on TradingView: the compiled script metadata
(``metaInfo.defaults.styles.<plot>.display``) carries the bitmask of each plot.
"""
from pynecore.lib import script, plot, display, close, open, high, low, input


@script.indicator("Display set", overlay=True)
def main(flag=input.bool(True, "flag", display=display.data_window + display.status_line)):
    plot(close if flag else open, "hidden_from_pane", display=display.all - display.pane)
    plot(open, "status_and_window", display=display.data_window + display.status_line)
    plot(high, "screener", display=display.pine_screener)
    plot(low, "nowhere", display=display.none)


def __test_display_bits_match_tradingview__():
    """each constant carries TradingView's bit value; all is every bit"""
    from pynecore.types.display import Display
    assert int(display.none) == 0
    assert int(display.pane) == 1
    assert int(display.data_window) == 2
    assert int(display.price_scale) == 4
    assert int(display.status_line) == 8
    assert int(display.pine_screener) == 16
    assert int(display.all) == 0xFFFFFFFF
    assert isinstance(display.all - display.pane, Display)
    assert isinstance(display.pane + display.status_line, Display)


def __test_display_arithmetic_is_set_arithmetic__():
    """+ adds a place at most once, - removes only a place that is present"""
    assert int(display.data_window + display.pane) == 3
    assert int(display.data_window + display.status_line) == 10
    assert int(display.all - display.pane) == 0xFFFFFFFE
    assert int(display.all - display.pane - display.pane) == 0xFFFFFFFE
    assert int(display.pane + display.pane) == 1
    assert int(display.none + display.pane) == 1
    assert int(display.pane - display.data_window) == 1
    assert int(display.all + display.pane) == 0xFFFFFFFF
    assert int(display.none - display.pane) == 0
    assert int(display.pane - display.pane) == 0
    assert int(display.all - display.status_line - display.data_window) == 0xFFFFFFF5
    assert int(display.all - display.pine_screener) == 0xFFFFFFEF
    assert int(display.pane + display.data_window + display.price_scale + display.status_line) == 15


def __test_display_places_names_every_place__():
    """the stream lists the places in bit order, never a single colliding name"""
    from pynecore.core.viz import display_places
    every = ["pane", "data_window", "price_scale", "status_line", "pine_screener"]
    assert display_places(None) == every
    assert display_places(display.all) == every
    assert display_places(display.none) == []
    assert display_places(display.data_window + display.pane) == ["pane", "data_window"]
    assert display_places(display.data_window + display.status_line) == [
        "data_window", "status_line"]
    assert display_places(display.all - display.pane) == every[1:]


def __test_display_combinations_reach_plot_meta__(runner, tmp_path):
    """a combined display survives the script run and serializes to its places"""
    from pynecore import lib
    from pynecore.core import viz
    from pynecore.types.ohlcv import OHLCV

    bars = [OHLCV(timestamp=1_704_067_200_000 + i * 300_000, open=100.0, high=101.0,
                  low=99.0, close=100.5, volume=1.0) for i in range(2)]
    for _ in runner(iter(bars), plot_path=tmp_path / 'plots.csv').run_iter():
        pass

    meta = lib._plot_meta
    assert viz.serialize_meta(meta["hidden_from_pane"])["display"] == [
        "data_window", "price_scale", "status_line", "pine_screener"]
    assert viz.serialize_meta(meta["status_and_window"])["display"] == [
        "data_window", "status_line"]
    assert viz.serialize_meta(meta["screener"])["display"] == ["pine_screener"]
    assert viz.serialize_meta(meta["nowhere"])["display"] == []

    flag = next(i for i in main.script.inputs.values() if i.title == "flag")
    assert int(flag.display) == 10
