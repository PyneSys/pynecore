"""
@pyne
"""
from pynecore.lib import (
    script, plot, chart, line, box, label, polyline, bar_index, close, high, low, na
)


@script.indicator(title="ChartPoint", shorttitle="cp")
def main():
    # --- chart.point constructors -------------------------------------------------
    fi = chart.point.from_index(bar_index, close)
    ok_fi = 1 if (fi.index == bar_index and fi.price == close and na(fi.time)) else 0

    ft = chart.point.from_time(1700000000000, close)
    ok_ft = 1 if (ft.time == 1700000000000 and ft.price == close and na(ft.index)) else 0

    nw = chart.point.now(close)
    ok_now = 1 if (nw.index == bar_index and nw.price == close and not na(nw.time)) else 0

    nu = chart.point.new(1700000000000, 42, close)
    ok_new = 1 if (nu.time == 1700000000000 and nu.index == 42 and nu.price == close) else 0

    cp = chart.point.copy(nu)
    ok_copy = 1 if (cp.index == nu.index and cp.time == nu.time and cp.price == nu.price) else 0

    # --- drawing functions accept the Pine keyword parameter names ----------------
    p1 = chart.point.from_index(bar_index, high)
    p2 = chart.point.from_index(bar_index, low)

    ln = line.new(first_point=p1, second_point=p2)
    ok_line = 1 if (line.get_x1(ln) == bar_index and line.get_y1(ln) == high
                    and line.get_y2(ln) == low) else 0

    bx = box.new(top_left=p1, bottom_right=p2)
    ok_box = 1 if (box.get_left(bx) == bar_index and box.get_top(bx) == high
                   and box.get_bottom(bx) == low) else 0

    # Mixed form: first point positional, second point by Pine keyword name
    ln_mixed = line.new(p1, second_point=p2)
    ok_line_mixed = 1 if line.get_y2(ln_mixed) == low else 0
    bx_mixed = box.new(p1, bottom_right=p2)
    ok_box_mixed = 1 if box.get_bottom(bx_mixed) == low else 0

    lb = label.new(point=p1, text="kw")
    ok_label_kw = 1 if (label.get_x(lb) == bar_index and label.get_text(lb) == "kw") else 0

    # Positional chart.point form: the second positional argument is the label text
    lb2 = label.new(p1, "pos")
    ok_label_pos = 1 if label.get_text(lb2) == "pos" else 0

    pl = polyline.new([p1, p2])
    ok_poly = 1 if not na(pl) else 0

    plot(ok_fi, "ok_fi")
    plot(ok_ft, "ok_ft")
    plot(ok_now, "ok_now")
    plot(ok_new, "ok_new")
    plot(ok_copy, "ok_copy")
    plot(ok_line, "ok_line")
    plot(ok_box, "ok_box")
    plot(ok_line_mixed, "ok_line_mixed")
    plot(ok_box_mixed, "ok_box_mixed")
    plot(ok_label_kw, "ok_label_kw")
    plot(ok_label_pos, "ok_label_pos")
    plot(ok_poly, "ok_poly")


def __test_chart_point__(csv_reader, runner, dict_comparator):
    """chart.point constructors and drawing functions accepting Pine keyword names.

    Every plotted column is an identity check that evaluates to 1, so the reference CSV
    holds all ones: it verifies the five chart.point constructors (including the ``na``
    coordinate left by ``from_index`` / ``from_time``) and that ``line`` / ``box`` /
    ``label`` accept the Pine keyword parameter names plus the positional point + text form.
    """
    with csv_reader('chart_point.csv', subdir="data") as cr:
        for candle, _plot in runner(cr).run_iter():
            dict_comparator(_plot, candle.extra_fields)
