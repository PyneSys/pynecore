"""
@pyne
"""
from pynecore.lib import (
    script, plot, fill, hline, bgcolor, barcolor,
    color, close, open, bar_index, na,
)


@script.indicator("BgFillHline", "bfh", overlay=True)
def main():
    p1 = plot(close, "p1", color=color.red)
    p2 = plot(open, "p2", color=color.blue)
    # Solid fill between two plots.
    fill(p1, p2, color=color.new(color.green, 80), title="pf")
    # Two hlines and a fill between them.
    h1 = hline(100.0, "h1", color=color.gray)
    h2 = hline(50.0, "h2", color=color.gray)
    fill(h1, h2, color=color.new(color.red, 90), title="hf")
    # Gradient fill: top/bottom value+color instead of a single color.
    fill(p1, p2, top_value=110.0, bottom_value=90.0,
         top_color=color.green, bottom_color=color.red, title="grad")
    # Background & bar coloring: na leaves the bar unpainted.
    bgcolor(color.new(color.blue, 85) if bar_index % 2 == 0 else na, title="bg")
    barcolor(color.new(color.red, 0) if close > open else na, title="bc")


def __test_bg_fill_hline__(runner):
    """bgcolor/barcolor/fill/hline register ordinal-id metas and per-bar dynamic channels.

    Ordinal ids (fill#0..2, hline#0..1, bgcolor#0, barcolor#0) are stable across bars; fill
    metas reference their plot/hline ids; the gradient fill and bgcolor/barcolor are dynamic.
    Per bar, the na-guarded background/bar colors and the solid/gradient fill colors appear in
    ``lib._viz_dyn`` exactly when active.
    """
    from pynecore import lib
    from pynecore.types.na import NA
    from pynecore.types.ohlcv import OHLCV

    base = 1704067200
    rows = [
        (100.0, 101.0, 99.0, 100.5),  # up
        (101.0, 101.5, 100.0, 100.5),  # down
        (100.5, 102.0, 100.0, 101.5),  # up
        (101.5, 102.0, 101.0, 101.0),  # down
        (101.0, 102.0, 100.0, 101.5),  # up
        (101.5, 102.5, 101.0, 101.0),  # down
    ]
    bars = [OHLCV(timestamp=base + i * 300, open=o, high=h, low=l, close=c, volume=100.0)
            for i, (o, h, l, c) in enumerate(rows)]

    dyn_snaps: list[dict] = []
    for _candle, _plot in runner(iter(bars)).run_iter():
        dyn_snaps.append(dict(lib._viz_dyn))

    assert len(dyn_snaps) == 6

    meta = lib._plot_meta
    # Ordinal ids are stable and all registered.
    assert set(meta) == {"p1", "p2", "fill#0", "fill#1", "fill#2",
                         "hline#0", "hline#1", "bgcolor#0", "barcolor#0"}

    # Solid plot-pair fill references the two plot ids.
    assert meta["fill#0"].kind == "fill"
    assert meta["fill#0"].plot1 == "p1"
    assert meta["fill#0"].plot2 == "p2"
    assert meta["fill#0"].hline1 is None

    # Hline-pair fill references the two hline ids.
    assert meta["fill#1"].kind == "fill"
    assert meta["fill#1"].hline1 == "hline#0"
    assert meta["fill#1"].hline2 == "hline#1"
    assert meta["fill#1"].plot1 is None

    # Gradient fill is dynamic and still references the plot ids.
    assert meta["fill#2"].kind == "fill"
    assert meta["fill#2"].dynamic is True
    assert meta["fill#2"].plot1 == "p1"

    # Hline metas.
    assert meta["hline#0"].kind == "hline"
    assert meta["hline#0"].price == 100.0
    assert meta["hline#0"].color is color.gray
    assert meta["hline#1"].price == 50.0

    # bgcolor/barcolor are always dynamic.
    assert meta["bgcolor#0"].kind == "bgcolor"
    assert meta["bgcolor#0"].dynamic is True
    assert meta["barcolor#0"].kind == "barcolor"
    assert meta["barcolor#0"].dynamic is True

    for i, (o, _h, _l, c) in enumerate(rows):
        snap = dyn_snaps[i]
        # Gradient fill is recorded every bar (identity check bypassed).
        assert "fill#2" in snap
        # Solid fill uses a fresh color object each bar -> recorded from bar 1 on
        # (bar 0's color becomes the static meta.color).
        assert ("fill#0" in snap) == (i > 0)
        # bgcolor/barcolor record every bar so the na "off" transition is emitted;
        # na bars carry an NA value (serialized to null), painted bars a Color.
        assert "bgcolor#0" in snap
        assert isinstance(snap["bgcolor#0"], NA) == (i % 2 != 0)
        assert "barcolor#0" in snap
        assert isinstance(snap["barcolor#0"], NA) == (not (c > o))
