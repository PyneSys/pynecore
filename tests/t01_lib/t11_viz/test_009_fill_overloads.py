"""
@pyne
"""
from pynecore.lib import (
    script, plot, fill, hline, color, close, open,
)


@script.indicator("FillOverloads", "fo", overlay=True)
def main():
    p1 = plot(close, "p1", color=color.red)
    p2 = plot(open, "p2", color=color.blue)
    # Plot-pair solid fill, fully positional (Pine overload 1: 6th positional is show_last).
    fill(p1, p2, color.green, "pf", True, 10, False)
    h1 = hline(100.0, "h1", color=color.gray)
    h2 = hline(50.0, "h2", color=color.gray)
    # Hline-pair fill, fully positional (Pine overload 2: 6th positional is fillgaps).
    fill(h1, h2, color.red, "hf", True, True)
    # Gradient fill, fully positional (Pine overload 3: values then colors).
    fill(p1, p2, 110.0, 90.0, color.green, color.red, "grad")


def __test_fill_positional_overloads__(runner, tmp_path):
    """Positional ``fill`` calls bind per Pine's three overload shapes.

    The plot-pair shape binds its 6th positional to ``show_last``, the hline-pair shape binds
    its 6th positional to ``fillgaps``, and the gradient shape binds values/colors in Pine's
    order. The NDJSON stream serializes all three metas without error and carries the gradient
    color channel.
    """
    import json
    from pynecore import lib
    from pynecore.core.viz import color_str
    from pynecore.types.ohlcv import OHLCV

    base = 1704067200
    bars = [OHLCV(timestamp=base + i * 300, open=100.0, high=101.0, low=99.0,
                  close=100.5, volume=100.0) for i in range(3)]

    out = tmp_path / "viz.ndjson"
    list(runner(iter(bars), viz_path=out).run_iter())

    meta = lib._plot_meta
    assert {"fill#0", "fill#1", "fill#2"} <= set(meta)

    # Plot-pair solid shape: 6th positional went to show_last, not fillgaps.
    assert meta["fill#0"].plot1 == "p1"
    assert meta["fill#0"].plot2 == "p2"
    assert meta["fill#0"].title == "pf"
    assert meta["fill#0"].color is color.green
    assert meta["fill#0"].show_last == 10
    assert meta["fill#0"].fillgaps is False

    # Hline-pair shape: 6th positional went to fillgaps (there is no show_last).
    assert meta["fill#1"].hline1 == "hline#0"
    assert meta["fill#1"].hline2 == "hline#1"
    assert meta["fill#1"].title == "hf"
    assert meta["fill#1"].color is color.red
    assert meta["fill#1"].fillgaps is True
    assert meta["fill#1"].show_last is None

    # Gradient shape: values and colors bound in Pine's order; the fill is dynamic.
    assert meta["fill#2"].plot1 == "p1"
    assert meta["fill#2"].plot2 == "p2"
    assert meta["fill#2"].title == "grad"
    assert meta["fill#2"].dynamic is True

    # The whole stream serialized (no color_str crash on misbound floats).
    lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    metas = {rec["id"]: rec for rec in lines if rec["t"] == "meta"}
    assert metas["fill#2"].get("dynamic") is True
    first_bar = next(rec for rec in lines if rec["t"] == "bar")
    assert first_bar["c"]["fill#2"] == [110.0, 90.0, color_str(color.green), color_str(color.red)]


def __test_table_merge_serialization__():
    """Merged table cells carry their merge range in the serialized cell dict."""
    import json
    from pynecore.lib import position
    from pynecore.types.table import Table
    from pynecore.core.viz import _table_dict

    tbl = Table(position=position.top_right, columns=2, rows=2)
    tbl.get_cell(0, 0).text = "A"
    tbl.get_cell(1, 1).text = "B"
    tbl.merge_cells(0, 0, 1, 0)
    tbl.vid = 1

    d = _table_dict(tbl)
    by_pos = {(c["col"], c["row"]): c for c in d["cells"]}
    assert by_pos[(0, 0)]["merge"] == [0, 0, 1, 0]
    assert by_pos[(1, 0)]["merge"] == [0, 0, 1, 0]
    assert "merge" not in by_pos[(1, 1)]
    json.dumps(d)  # remains JSON-serializable
