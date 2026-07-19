"""
@pyne
"""
from pynecore.lib import script, bgcolor, color, close, open, bar_index, na


@script.indicator("BgOff", "bgo", overlay=True)
def main():
    # Background painted on even bars, unpainted (na) on odd bars.
    bgcolor(color.new(color.blue, 0) if bar_index % 2 == 0 else na, title="bg")


def __test_bgcolor_offtransition__(runner, tmp_path):
    """bgcolor paint -> unpaint -> paint round-trips through the NDJSON stream.

    An na (unpainted) bar must emit an explicit ``null`` for the channel so a reader carrying
    the last emitted color forward does not keep painting; the following painted bar re-emits
    the color. Reconstructing per-bar colors from the ``c`` deltas must recover the exact
    alternating paint/unpaint pattern.
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
    bar_recs = [r for r in lines if r["t"] == "bar"]
    assert len(bar_recs) == 4

    # Reconstruct the channel by carrying the last emitted value forward.
    reconstructed: list = []
    cur = None
    for rec in bar_recs:
        c = rec.get("c", {})
        if "bgcolor#0" in c:
            cur = c["bgcolor#0"]
        reconstructed.append(cur)

    blue = color_str(color.new(color.blue, 0))
    # Even bars painted blue, odd bars explicitly off (null).
    assert reconstructed == [blue, None, blue, None]
