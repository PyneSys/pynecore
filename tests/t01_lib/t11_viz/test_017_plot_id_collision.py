"""
@pyne
"""
from pynecore.lib import (
    script, plot, plotcandle, plotbar, bgcolor, color, close, open, high, low,
)


@script.indicator("Plot id collision", "pic", overlay=True)
def main():
    # A candle claims "X" before the plot of the same title.
    plotcandle(open, high, low, close, "X", color=color.green)
    plot(close, "X", color=color.red)
    # A plot claims "Y" before the candle of the same title.
    plot(close, "Y", color=color.blue)
    plotcandle(open, high, low, close, "Y", color=color.orange)
    # A plot title equal to one of the bar's channel keys.
    plot(high, "Z (open)", color=color.purple)
    plotbar(open, high, low, close, "Z", color=color.teal)
    # A plot title equal to the next bgcolor id.
    plot(low, "bgcolor#0", color=color.gray)
    bgcolor(color.yellow)


def __test_plot_id_collision__(runner):
    """Plot-family calls of one bar never share a meta id or a data key.

    Every call registers its own PlotMeta with its own kind and color, the later
    caller of a contested name is renamed the way duplicate plot titles are, and
    no data key is overwritten by another call.
    """
    from pynecore import lib
    from pynecore.types.ohlcv import OHLCV

    base = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    rows = [
        (100.0, 101.0, 99.0, 100.5),
        (101.0, 101.5, 100.0, 100.5),
        (100.5, 102.0, 100.0, 101.5),
    ]
    bars = [OHLCV(timestamp=base + i * 300_000, open=o, high=h, low=l, close=c, volume=100.0)
            for i, (o, h, l, c) in enumerate(rows)]

    per_bar: list[dict] = []
    for _candle, _plot in runner(iter(bars)).run_iter():
        per_bar.append(dict(_plot))

    assert len(per_bar) == 3
    for i, (o, h, l, c) in enumerate(rows):
        rec = per_bar[i]
        assert rec["X (open)"] == o
        assert rec["X (close)"] == c
        assert rec["X 0"] == c
        assert rec["Y"] == c
        assert rec["Y 0 (open)"] == o
        assert rec["Y 0 (close)"] == c
        assert rec["Z (open)"] == h
        assert rec["Z 0 (open)"] == o
        assert rec["Z 0 (close)"] == c
        assert rec["bgcolor#0"] == l
        assert "X" not in rec
        assert "Y (open)" not in rec
        assert "Z (close)" not in rec

    meta = lib._plot_meta
    expected = {
        "X": ("candle", color.green),
        "X 0": ("plot", color.red),
        "Y": ("plot", color.blue),
        "Y 0": ("candle", color.orange),
        "Z (open)": ("plot", color.purple),
        "Z 0": ("bar", color.teal),
        "bgcolor#0": ("plot", color.gray),
    }
    for pid, (kind, col) in expected.items():
        assert meta[pid].kind == kind, pid
        assert meta[pid].color is col, pid
        assert not meta[pid].dynamic, pid
    assert meta["bgcolor#1"].kind == "bgcolor"
    assert sum(1 for m in meta.values() if m.kind == "bgcolor") == 1
