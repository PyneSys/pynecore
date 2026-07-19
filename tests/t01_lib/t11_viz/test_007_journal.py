"""
@pyne
"""
from pynecore.lib import script, line, color, bar_index
from pynecore.types import Persistent, Line


@script.indicator("Journal", "jr", overlay=True)
def main():
    # Persistent: the line is created once (bar 0) and reused every bar.
    ln: Persistent[Line] = line.new(0, 100.0, 0, 100.0, color=color.red)
    # Move the far endpoint on bars 1-3 -> one update event per bar.
    if 1 <= bar_index <= 3:
        line.set_xy2(ln, bar_index, 100.0 + bar_index)
    # Remove it on bar 4 -> one delete event.
    if bar_index == 4:
        line.delete(ln)


def __test_journal__(runner, tmp_path):
    """Journal mode emits exactly one create, N updates and one delete for a moving line.

    A persistent line is created on bar 0, moved on bars 1-3 and deleted on bar 4. The NDJSON
    ``ev`` records must contain a single create, three updates and one delete for that vid; the
    header must flag journal=true; and the ``viz_events`` callback must receive the same events
    the file records.
    """
    import json
    from pynecore.types.ohlcv import OHLCV

    base = 1704067200
    bars = [OHLCV(timestamp=base + i * 300, open=100.0, high=101.0, low=99.0,
                  close=100.5, volume=100.0) for i in range(6)]

    out = tmp_path / "journal.ndjson"
    r = runner(iter(bars), viz_path=out, viz_journal=True)

    batches: list[list[dict]] = []
    r.viz_events = lambda evs: batches.append(list(evs))

    list(r.run_iter())

    lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    assert lines[0]["t"] == "hdr"
    assert lines[0]["journal"] is True

    evs = [rec for rec in lines if rec["t"] == "ev"]
    # Exactly one line object is journaled.
    vids = {ev["id"] for ev in evs}
    assert len(vids) == 1
    vid = vids.pop()
    assert all(ev["obj"] == "line" for ev in evs)

    ops = [ev["op"] for ev in evs]
    assert ops.count("create") == 1
    assert ops.count("update") == 3
    assert ops.count("delete") == 1

    # The create carries the initial state; the delete carries no state payload.
    create = next(ev for ev in evs if ev["op"] == "create")
    assert create["i"] == 0
    assert create["s"]["id"] == vid
    delete = next(ev for ev in evs if ev["op"] == "delete")
    assert delete["i"] == 4
    assert "s" not in delete

    # The callback received the same events, batched per bar.
    cb_evs = [ev for batch in batches for ev in batch]
    assert [ev["op"] for ev in cb_evs] == ops
    # Non-empty batches correspond to bars 0-4 (bar 5 produced no events).
    assert [b[0]["i"] for b in batches if b] == [0, 1, 2, 3, 4]
