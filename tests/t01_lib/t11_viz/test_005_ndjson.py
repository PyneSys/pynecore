"""
@pyne
"""
from pynecore.lib import script, plot, color, close, open, bar_index


@script.indicator("Ndjson", "nd", overlay=True)
def main():
    # "c" alternates color per bar (dynamic); "o" is a fixed color (static).
    plot(close, "c", color=color.red if bar_index % 2 == 0 else color.lime)
    plot(open, "o", color=color.blue)


def __test_ndjson__(runner, tmp_path):
    """viz_path writes a well-formed NDJSON stream with only-on-change color encoding.

    Every line parses as JSON; the header is first (journal false), meta records precede the
    bar records, the drawings snapshot precedes the terminal ``end`` record, per-bar ``v``
    values match the plotted series, and the dynamic color channel is emitted only on the bar
    it actually changes (the static color never appears).
    """
    import json
    from pynecore.core.viz import color_str
    from pynecore.types.ohlcv import OHLCV

    base = 1704067200
    rows = [(100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i) for i in range(4)]
    bars = [OHLCV(timestamp=base + i * 300, open=o, high=h, low=l, close=c, volume=100.0)
            for i, (o, h, l, c) in enumerate(rows)]

    out = tmp_path / "viz.ndjson"
    list(runner(iter(bars), viz_path=out).run_iter())

    lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 4

    # Header first, journal disabled, script/syminfo context present.
    hdr = lines[0]
    assert hdr["t"] == "hdr"
    assert hdr["journal"] is False
    assert hdr["script"]["title"] == "Ndjson"
    assert "syminfo" in hdr

    # End last, drawings snapshot immediately before it.
    assert lines[-1]["t"] == "end"
    assert lines[-1]["bars"] == 4
    assert lines[-2]["t"] == "drawings"

    kinds = [rec["t"] for rec in lines]
    meta_idx = [i for i, t in enumerate(kinds) if t == "meta"]
    bar_idx = [i for i, t in enumerate(kinds) if t == "bar"]
    assert len(bar_idx) == 4
    # Both plots registered on bar 0 -> their metas precede every bar record. "c" only turns
    # dynamic on bar 1, so its meta is re-emitted (dynamic: true) between bar 0 and bar 1.
    assert len(meta_idx) == 3
    assert meta_idx[1] < min(bar_idx)
    reemit = lines[meta_idx[2]]
    assert reemit["id"] == "c"
    assert reemit["dynamic"] is True
    assert bar_idx[0] < meta_idx[2] < bar_idx[1]
    # The initial "c" meta went out as static.
    first_c = next(lines[i] for i in meta_idx if lines[i]["id"] == "c")
    assert "dynamic" not in first_c

    metas = {lines[i]["id"]: lines[i] for i in meta_idx}  # later record wins per id
    assert set(metas) == {"c", "o"}
    assert metas["c"]["kind"] == "plot"
    assert metas["o"]["kind"] == "plot"

    bar_recs = [lines[i] for i in bar_idx]
    for i, rec in enumerate(bar_recs):
        assert rec["i"] == i
        assert rec["time"] == (base + i * 300) * 1000
        assert rec["v"]["c"] == rows[i][3]
        assert rec["v"]["o"] == rows[i][0]

    # Only-on-change: bar 0 registers red as the static color (no "c"); from bar 1 on the
    # color alternates and every change is emitted -- including the revert to red on bar 2 --
    # so a reader can reconstruct every bar. The static "o" never appears.
    c_bars = [i for i, rec in enumerate(bar_recs) if "c" in rec and "c" in rec["c"]]
    assert c_bars == [1, 2, 3]
    assert bar_recs[1]["c"]["c"] == color_str(color.lime)
    assert bar_recs[2]["c"]["c"] == color_str(color.red)
    assert bar_recs[3]["c"]["c"] == color_str(color.lime)
    assert all("o" not in rec.get("c", {}) for rec in bar_recs)
