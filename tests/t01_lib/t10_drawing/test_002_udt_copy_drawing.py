"""
@pyne
"""
from pynecore.core.pine_udt import udt_copy
from pynecore.lib import script, label, line, box, bar_index, high, low


@script.indicator("udt_copy drawing", "udtcopy", overlay=True, max_labels_count=5)
def main():
    # ``udt_copy`` is exactly what PyneComp emits for Pine's method form ``l.copy()``.
    # Every bar creates one label with ``label.new`` and one with a copy; the copy's
    # ``y`` is offset by one so the surviving labels' coordinates spell out the exact
    # creation order of the two routes.
    src = label.new(bar_index, bar_index * 2.0, "n")
    clone = udt_copy(src)
    label.set_text(clone, "c")
    label.set_y(clone, bar_index * 2.0 + 1.0)

    if bar_index == 0:
        ln = line.new(0, high, 5, low)
        ln_clone = udt_copy(ln)
        line.set_x2(ln_clone, 9)
        bx = box.new(0, high, 5, low)
        bx_clone = udt_copy(bx)
        box.set_right(bx_clone, 9)


def __test_udt_copy_registers_drawings_and_counts_toward_the_cap__(runner):
    """A drawing copied with ``udt_copy`` is a live, independent, registered object.

    Asserts that a copied label, line and box each reach the drawings snapshot with
    their own id, carry only their own mutations, and share one eviction pool with
    the objects created by ``label.new``.
    """
    # Measured on TradingView with max_labels_count=5 and one label.new plus one
    # .copy() per bar over 6 bars: 12 objects were created and the newest ones
    # survived, evicted in strict interleaved creation order with no distinction
    # between the two creation routes (the all-label.new control behaved
    # identically). The unified pool and the ordering are what is encoded below.
    #
    # The survivor COUNT is not TradingView's: its drawing GC is bar-granular and
    # overshoots the declared limit (it kept 6), while PyneCore trims to the cap on
    # every insert. That divergence belongs to the eviction machinery itself, not
    # to copying, and is shared with label.new.
    from pynecore.types.ohlcv import OHLCV

    base = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [OHLCV(timestamp=base + i * 300_000, open=100.0, high=101.0, low=99.0,
                  close=100.5, volume=100.0) for i in range(6)]

    r = runner(iter(bars))
    list(r.run_iter())
    snap = r.drawings()

    labels = snap["labels"]
    # 12 labels created (6 bars x new+copy), max_labels_count=5 enforced.
    assert len(labels) == 5
    assert len({lb["id"] for lb in labels}) == 5  # no duplicated vid
    # ``y`` is the creation ordinal: the 5 newest of 0..11 survived, in order.
    assert [lb["y"] for lb in labels] == [7.0, 8.0, 9.0, 10.0, 11.0]
    # Odd ordinals are copies, even ones are label.new: the two routes interleave in
    # one pool, and each label carries only its own mutation. An orphaned copy would
    # leave nothing but even ordinals here.
    assert [lb["text"] for lb in labels] == ["c", "n", "c", "n", "c"]

    lines = snap["lines"]
    assert len(lines) == 2
    assert len({ln["id"] for ln in lines}) == 2
    # Only the copy's endpoint moved.
    assert sorted(ln["x2"] for ln in lines) == [5, 9]

    boxes = snap["boxes"]
    assert len(boxes) == 2
    assert len({bx["id"] for bx in boxes}) == 2
    assert sorted(bx["right"] for bx in boxes) == [5, 9]
