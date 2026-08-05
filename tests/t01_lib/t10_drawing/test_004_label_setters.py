"""
@pyne
"""
from pynecore.lib import script, label, chart, xloc, text, font, bar_index, high


@script.indicator("label setters", "lblset", overlay=True, max_labels_count=10)
def main():
    if bar_index == 0:
        # Text attributes
        lbl_fmt = label.new(bar_index, high, "fmt")
        label.set_text_font_family(lbl_fmt, font.family_monospace)
        label.set_text_formatting(lbl_fmt, text.format_bold)

        # ``set_point`` takes the x coordinate belonging to the label's own xloc,
        # so the same chart point moves a bar_index label and a bar_time label to
        # different x values
        point = chart.point.new(1_700_000_000_000, 42, 123.5)
        lbl_idx = label.new(bar_index, high, "idx")
        label.set_point(lbl_idx, point)
        lbl_time = label.new(1_704_067_200_000, high, "time", xloc=xloc.bar_time)
        label.set_point(lbl_time, point)

        # ``set_xloc`` moves the label and switches its x-coordinate space at once
        lbl_xloc = label.new(bar_index, high, "xloc")
        label.set_xloc(lbl_xloc, 1_700_000_000_000, xloc.bar_time)


def __test_label_setters__(runner):
    """The four label setters write the fields their Pine counterparts do.

    ``set_text_font_family`` and ``set_text_formatting`` are plain attribute writes,
    while ``set_point`` picks ``point.index`` or ``point.time`` by the label's current
    ``xloc`` -- the same rule ``label.new()``, ``line.set_first_point()`` and
    ``box.set_bottom_right_point()`` follow -- and ``set_xloc`` sets ``x`` and ``xloc``
    together, matching Pine's three-argument form.
    """
    from pynecore.types.label import Label
    from pynecore.types.na import NA
    from pynecore.types.ohlcv import OHLCV
    from pynecore.types.chart import ChartPoint
    from pynecore.lib import (label as label_mod, xloc as xloc_mod, text as text_mod,
                              font as font_mod)

    base = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [OHLCV(timestamp=base + i * 300_000, open=100.0, high=101.0, low=99.0,
                  close=100.5, volume=100.0) for i in range(3)]

    r = runner(iter(bars))
    list(r.run_iter())
    labels = {lb["text"]: lb for lb in r.drawings()["labels"]}
    assert len(labels) == 4

    assert labels["fmt"]["text_font_family"] == "monospace"
    assert labels["fmt"]["text_formatting"] == "bold"

    assert labels["idx"]["x"] == 42
    assert labels["idx"]["y"] == 123.5
    assert labels["time"]["x"] == 1_700_000_000_000
    assert labels["time"]["y"] == 123.5

    assert labels["xloc"]["x"] == 1_700_000_000_000
    assert labels["xloc"]["xloc"] == "bar_time"

    # An na label is a no-op for every setter, like its siblings
    na_label = NA(Label)
    label_mod.set_text_font_family(na_label, font_mod.family_monospace)
    label_mod.set_text_formatting(na_label, text_mod.format_bold)
    label_mod.set_point(na_label, ChartPoint(index=1, time=2, price=3.0))
    label_mod.set_xloc(na_label, 1, xloc_mod.bar_time)
