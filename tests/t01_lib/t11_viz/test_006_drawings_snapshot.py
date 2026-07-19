"""
@pyne
"""
from pynecore.lib import (
    script, line, label, box, table, polyline, linefill, chart, position,
    color, bar_index,
)


@script.indicator("Draw", "dr", overlay=True)
def main():
    if bar_index == 0:
        l1 = line.new(0, 100.0, 5, 110.0, color=color.red)
        l2 = line.copy(l1)  # copied line: fresh vid, same coordinates
        label.new(2, 105.0, "hi")
        box.new(1, 110.0, 4, 100.0)
        tbl = table.new(position.top_right, 2, 2)
        table.cell(tbl, 0, 0, "A")
        table.cell(tbl, 1, 1, "B")
        polyline.new([chart.point.from_index(0, 100.0), chart.point.from_index(1, 105.0)])
        linefill.new(l1, l2, color.new(color.blue, 80))
        ldel = line.new(0, 1.0, 1, 2.0)
        line.delete(ldel)


def __test_drawings_snapshot__(runner):
    """runner.drawings() reflects the final live drawing registries.

    Verifies object counts per family, globally unique vids, that a copied line is present
    (distinct vid, same coordinates), that a deleted line is absent, that a table carries its
    cells, and that a linefill embeds the current state of both referenced lines.
    """
    from pynecore.types.ohlcv import OHLCV

    base = 1704067200
    bars = [OHLCV(timestamp=base + i * 300, open=100.0, high=101.0, low=99.0,
                  close=100.5, volume=100.0) for i in range(3)]

    r = runner(iter(bars))
    list(r.run_iter())
    snap = r.drawings()

    # The original and its copy survive; the deleted throwaway line does not.
    assert len(snap["lines"]) == 2
    assert len(snap["labels"]) == 1
    assert len(snap["boxes"]) == 1
    assert len(snap["tables"]) == 1
    assert len(snap["polylines"]) == 1
    assert len(snap["linefills"]) == 1

    # Every vid is unique across all families.
    all_ids = [d["id"] for fam in snap.values() if isinstance(fam, list) for d in fam]
    assert len(all_ids) == len(set(all_ids))

    # The deleted throwaway line (coords 0,1 -> 1,2) is absent.
    assert all(not (ln["x1"] == 0 and ln["y1"] == 1.0 and ln["y2"] == 2.0)
               for ln in snap["lines"])

    # The original and its copy share coordinates but have distinct vids.
    originals = [ln for ln in snap["lines"] if ln["x1"] == 0 and ln["y1"] == 100.0
                 and ln["x2"] == 5 and ln["y2"] == 110.0]
    assert len(originals) == 2
    assert len({ln["id"] for ln in originals}) == 2

    # Table carries its two cells.
    cells = snap["tables"][0]["cells"]
    assert len(cells) == 2
    assert {c["text"] for c in cells} == {"A", "B"}

    # Linefill embeds both referenced lines' states, keyed by their vids.
    lf = snap["linefills"][0]
    assert lf["line1"] == lf["line1_state"]["id"]
    assert lf["line2"] == lf["line2_state"]["id"]
    line_ids = {ln["id"] for ln in snap["lines"]}
    assert lf["line1"] in line_ids
    assert lf["line2"] in line_ids
