"""
@pyne
"""
from pynecore.lib import script, plot, color, close, open, bar_index


@script.indicator("DynColor", "dc", overlay=True)
def main():
    # Dynamic color: alternates every bar. Bar 0's color becomes the plot's
    # static meta.color, so only the *divergent* bars land in lib._viz_dyn.
    plot(close, "dyn", color=color.red if bar_index % 2 == 0 else color.lime)
    # Static color: the same color object on every bar -> never in lib._viz_dyn.
    plot(open, "stat", color=color.blue)


def __test_dynamic_color__(runner):
    """Dynamic per-bar colors land in lib._viz_dyn; static colors never do.

    A plot whose color alternates each bar is compared against a fixed-color plot. Bar 0's
    color becomes the static ``meta.color`` and is elided. Once the plot becomes dynamic
    (the first divergence, bar 1), every subsequent bar records its current color -- including
    the bars that return to the static color -- so a reader can reconstruct the color for any
    bar. The static plot never appears in ``lib._viz_dyn`` and keeps ``dynamic`` False.
    """
    from pynecore import lib
    from pynecore.types.ohlcv import OHLCV

    base = 1704067200
    rows = [(100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i) for i in range(6)]
    bars = [OHLCV(timestamp=base + i * 300, open=o, high=h, low=l, close=c, volume=100.0)
            for i, (o, h, l, c) in enumerate(rows)]

    dyn_snaps: list[dict] = []
    for _candle, _plot in runner(iter(bars)).run_iter():
        dyn_snaps.append(dict(lib._viz_dyn))

    assert len(dyn_snaps) == 6
    for i, snap in enumerate(dyn_snaps):
        # The static plot is elided on every bar.
        assert "stat" not in snap
        if i == 0:
            # Bar 0's color is registered as the static meta.color -> elided.
            assert "dyn" not in snap
        else:
            # Once dynamic, every bar records its current color (including the
            # revert to the static color on even bars).
            assert "dyn" in snap
            assert snap["dyn"] is (color.red if i % 2 == 0 else color.lime)

    assert lib._plot_meta["dyn"].dynamic is True
    assert lib._plot_meta["stat"].dynamic is False
