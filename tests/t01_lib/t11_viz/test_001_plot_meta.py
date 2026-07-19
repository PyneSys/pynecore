"""
@pyne
"""
from pynecore.lib import script, plot, color, display, close, open, bar_index


@script.indicator("PlotMeta", "pm", overlay=True)
def main():
    # A heavily styled plot: every non-default option is exercised so the
    # registered PlotMeta can be asserted field-by-field after the run.
    plot(close, "styled", color=color.red, linewidth=3, style=plot.style_stepline,
         offset=2, trackprice=True, histbase=100.0, display=display.pane, force_overlay=True)
    # A plain plot to verify defaults (color None, linewidth 1, style None).
    plot(open, "plain")
    # A conditional plot that only first fires on bar 3: its meta must be lazily
    # registered exactly once (on bar 3) and its value must be absent before that.
    if bar_index >= 3:
        plot(close, "late", color=color.green)


def __test_plot_meta__(runner):
    """Styled plot() calls register a complete PlotMeta; conditional plots register lazily.

    A single styled plot exercises color/linewidth/style/offset/trackprice/histbase/
    display/force_overlay; a plain plot verifies defaults; and a plot first reached on
    bar 3 verifies lazy register-once behaviour (meta appears only after it fires, and its
    per-bar value is absent on bars 0-2).
    """
    from pynecore import lib
    from pynecore.core import viz
    from pynecore.types.ohlcv import OHLCV

    base = 1704067200  # 2024-01-01 00:00:00 UTC
    rows = [
        (100.0, 101.0, 99.0, 100.5),
        (100.5, 101.5, 99.5, 101.0),
        (101.0, 102.0, 100.0, 101.5),
        (101.5, 102.5, 100.5, 102.0),
        (102.0, 103.0, 101.0, 102.5),
        (102.5, 103.5, 101.5, 103.0),
    ]
    bars = [OHLCV(timestamp=base + i * 300, open=o, high=h, low=l, close=c, volume=100.0)
            for i, (o, h, l, c) in enumerate(rows)]

    per_bar: list[dict] = []
    for _candle, _plot in runner(iter(bars)).run_iter():
        per_bar.append(dict(_plot))

    assert len(per_bar) == 6

    # Lazy registration: "late" absent on bars 0-2, present with the bar's close after.
    for i in range(3):
        assert "late" not in per_bar[i]
    for i in range(3, 6):
        assert "late" in per_bar[i]
        assert per_bar[i]["late"] == rows[i][3]

    # The always-present plots carry the right per-bar values.
    for i in range(6):
        assert per_bar[i]["styled"] == rows[i][3]
        assert per_bar[i]["plain"] == rows[i][0]

    meta = lib._plot_meta
    assert set(meta) == {"styled", "plain", "late"}

    styled = meta["styled"]
    assert styled.kind == "plot"
    assert styled.title == "styled"
    assert styled.color is color.red
    assert styled.linewidth == 3
    assert styled.style is plot.style_stepline
    assert styled.offset == 2
    assert styled.trackprice is True
    assert styled.histbase == 100.0
    assert styled.display is display.pane
    assert styled.force_overlay is True
    # color.red is the same object on every bar -> stays static, not dynamic.
    assert styled.dynamic is False

    # Serialized meta applies enum-name mapping and drops defaults.
    sm = viz.serialize_meta(styled)
    assert sm["kind"] == "plot"
    assert sm["style"] == "stepline"
    assert sm["display"] == "pane"
    assert sm["linewidth"] == 3
    assert sm["offset"] == 2
    assert sm["trackprice"] is True
    assert sm["histbase"] == 100.0
    assert sm["force_overlay"] is True

    plain = meta["plain"]
    assert plain.kind == "plot"
    assert plain.color is None
    assert plain.linewidth == 1
    assert plain.style is None
    assert plain.dynamic is False
    # A plain plot serializes to the Pine default style/display.
    assert viz.serialize_meta(plain)["style"] == "line"
    assert viz.serialize_meta(plain)["display"] == "all"

    late = meta["late"]
    assert late.kind == "plot"
    assert late.color is color.green
