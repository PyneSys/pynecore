"""
@pyne
"""
from pynecore.lib import (
    script, plotshape, plotchar, plotarrow, plotcandle, plotbar,
    shape, location, color, close, open, high, low,
)


@script.indicator("Family", "fam", overlay=True)
def main():
    # plotshape stores int(bool(series)); an NA series propagates as NA.
    plotshape(close > open, "sh", style=shape.triangleup, location=location.belowbar,
              color=color.red)
    # close[1] is NA on bar 0 -> stored as NA; a real value afterwards -> int(bool()).
    plotshape(close[1], "shna")
    # plotchar stores the raw series value.
    plotchar(close, "ch", char="X", location=location.top, color=color.blue)
    # plotarrow stores the raw series value.
    plotarrow(close - open, "ar", colorup=color.green, colordown=color.red)
    # plotcandle / plotbar each expand to four "<title> (open|high|low|close)" keys.
    plotcandle(open, high, low, close, "cd", color=color.green)
    plotbar(open, high, low, close, "br", color=color.orange)


def __test_plot_family__(runner):
    """plotshape/plotchar/plotarrow/plotcandle/plotbar populate _plot_data and register metas.

    Verifies per-bar values (plotshape 0/1 with NA propagation, plotchar/plotarrow raw
    series, plotcandle/plotbar four OHLC keys) and that each family registers a PlotMeta with
    the right kind and style/char/location/color fields.
    """
    from pynecore import lib
    from pynecore.types.na import NA
    from pynecore.types.ohlcv import OHLCV

    base = 1704067200
    rows = [
        (100.0, 101.0, 99.0, 100.5),  # close > open -> 1
        (101.0, 101.5, 100.0, 100.5),  # close < open -> 0
        (100.5, 102.0, 100.0, 101.5),  # close > open -> 1
        (101.5, 102.0, 101.0, 101.0),  # close < open -> 0
    ]
    bars = [OHLCV(timestamp=base + i * 300, open=o, high=h, low=l, close=c, volume=100.0)
            for i, (o, h, l, c) in enumerate(rows)]

    per_bar: list[dict] = []
    for _candle, _plot in runner(iter(bars)).run_iter():
        per_bar.append(dict(_plot))

    assert len(per_bar) == 4
    for i, (o, h, l, c) in enumerate(rows):
        rec = per_bar[i]
        # plotshape: 0/1 based on the boolean series.
        assert rec["sh"] == (1 if c > o else 0)
        # plotshape NA propagation: close[1] is NA on bar 0.
        if i == 0:
            assert isinstance(rec["shna"], NA)
        else:
            assert rec["shna"] in (0, 1)
        # plotchar / plotarrow store the raw series value.
        assert rec["ch"] == c
        assert rec["ar"] == c - o
        # plotcandle / plotbar expand to four OHLC keys.
        assert rec["cd (open)"] == o
        assert rec["cd (high)"] == h
        assert rec["cd (low)"] == l
        assert rec["cd (close)"] == c
        assert rec["br (open)"] == o
        assert rec["br (high)"] == h
        assert rec["br (low)"] == l
        assert rec["br (close)"] == c

    meta = lib._plot_meta
    assert meta["sh"].kind == "shape"
    assert meta["sh"].style is shape.triangleup
    assert meta["sh"].location is location.belowbar
    assert meta["sh"].color is color.red

    assert meta["ch"].kind == "char"
    assert meta["ch"].char == "X"
    assert meta["ch"].location is location.top

    assert meta["ar"].kind == "arrow"
    assert meta["ar"].colorup is color.green
    assert meta["ar"].colordown is color.red

    assert meta["cd"].kind == "candle"
    assert meta["cd"].color is color.green

    assert meta["br"].kind == "bar"
    assert meta["br"].color is color.orange
