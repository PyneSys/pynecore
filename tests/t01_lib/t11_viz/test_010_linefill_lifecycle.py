"""
@pyne
"""
from pynecore.lib import script, line, linefill, color, bar_index
from pynecore.types import Persistent, Line


@script.indicator("LineFill lifecycle", "lfl", overlay=True)
def main():
    # Persistent: the lines are created once (bar 0) and reused every bar.
    l1: Persistent[Line] = line.new(0, 100.0, 5, 110.0, color=color.red)
    l2: Persistent[Line] = line.new(0, 90.0, 5, 95.0, color=color.green)
    l3: Persistent[Line] = line.new(0, 80.0, 5, 85.0)
    if bar_index == 0:
        linefill.new(l1, l2, color.new(color.blue, 80))
        linefill.new(l2, l3, color.new(color.purple, 70))
    if bar_index == 1:
        # Pine allows only one linefill per pair: this replaces the blue fill.
        linefill.new(l1, l2, color.new(color.orange, 60))
        # Deleting a line deletes every linefill referencing it.
        line.delete(l3)


def __test_linefill_replace_and_cascade__(runner, tmp_path):
    """linefill.new() on the same pair replaces the old fill; line.delete cascades.

    Bar 0 creates three lines and two fills. Bar 1 re-fills the (l1, l2) pair,
    replacing the first fill, and deletes l3, which must cascade-delete the fill
    referencing it. The final snapshot must contain exactly one linefill, and the
    journal must record delete events for both removed linefill vids.
    """
    import json
    from pynecore.types.ohlcv import OHLCV

    base = 1704067200
    bars = [OHLCV(timestamp=base + i * 300, open=100.0, high=101.0, low=99.0,
                  close=100.5, volume=100.0) for i in range(3)]

    out = tmp_path / "lifecycle.ndjson"
    r = runner(iter(bars), viz_path=out, viz_journal=True)
    list(r.run_iter())

    snap = r.drawings()
    # l1 + l2 survive; l3 was deleted.
    assert len(snap["lines"]) == 2
    # Only the replacement fill on the (l1, l2) pair is live.
    assert len(snap["linefills"]) == 1
    lf = snap["linefills"][0]
    line_ids = {ln["id"] for ln in snap["lines"]}
    assert lf["line1"] in line_ids
    assert lf["line2"] in line_ids

    records = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    evs = [rec for rec in records if rec["t"] == "ev" and rec["obj"] == "linefill"]
    creates = {ev["id"] for ev in evs if ev["op"] == "create"}
    deletes = {ev["id"] for ev in evs if ev["op"] == "delete"}
    # Three fills were created; the replaced one and the cascaded one were deleted.
    assert len(creates) == 3
    assert len(deletes) == 2
    assert deletes < creates
    assert creates - deletes == {lf["id"]}
    # Both deletions happen on bar 1.
    assert all(ev["i"] == 1 for ev in evs if ev["op"] == "delete")
