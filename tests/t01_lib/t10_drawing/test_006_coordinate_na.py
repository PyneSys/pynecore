"""
@pyne

A drawing's coordinates are normalized where the script hands them over.

``line.new`` always did: an x lands on the bar grid, a y keeps its value, and
anything that is not a number is na. The SETTERS did not, so a
``line.set_xy1(id, na, na)`` stored the universal na object in a field the
getters read back as a number -- ``line.get_x1`` then raised instead of
answering na. The normalization now lives in one place for line, box and label
(``lib/_drawing.py``), so what a getter returns cannot depend on which setter
put the value there.
"""
from pynecore.lib import script, box, chart, label, line, na, xloc


@script.indicator("drawing coordinate na", "dcna", overlay=True,
                  max_lines_count=10, max_boxes_count=10, max_labels_count=10)
def main():
    pass


def __test_a_line_setter_normalizes_like_the_constructor__():
    """An na coordinate through a setter reads back as na, not as an error"""
    ln = line.new(10, 1.0, 20, 2.0)
    assert line.get_x1(ln) == 10 and line.get_x2(ln) == 20

    line.set_xy1(ln, na, na)
    line.set_xy2(ln, na, na)
    x1, y1 = line.get_x1(ln), line.get_y1(ln)
    assert x1 != x1 and y1 != y1
    x2, y2 = line.get_x2(ln), line.get_y2(ln)
    assert x2 != x2 and y2 != y2

    # A fractional bar index is truncated on the way in, like in the constructor
    line.set_x1(ln, 12.75)
    line.set_x2(ln, 20.9)
    assert line.get_x1(ln) == 12 and line.get_x2(ln) == 20
    assert type(line.get_x1(ln)) is float

    # ... and the same through the pair and the xloc setters
    line.set_xy1(ln, 3.5, 1.5)
    assert line.get_x1(ln) == 3 and line.get_y1(ln) == 1.5
    line.set_xloc(ln, 4.25, na, xloc.bar_time)
    x2 = line.get_x2(ln)
    assert line.get_x1(ln) == 4 and x2 != x2


def __test_a_line_point_setter_normalizes__():
    """A chart point without the active xloc's coordinate leaves na behind"""
    ln = line.new(10, 1.0, 20, 2.0)
    # from_index carries no time, and the line reads times in bar_time xloc
    line.set_xloc(ln, 10, 20, xloc.bar_time)
    line.set_first_point(ln, chart.point.from_index(7, 1.0))
    x1 = line.get_x1(ln)
    assert x1 != x1


def __test_a_box_setter_normalizes__():
    """The box coordinates take the same boundary as the line's"""
    bx = box.new(10, 2.0, 20, 1.0)
    assert box.get_left(bx) == 10 and box.get_right(bx) == 20

    box.set_lefttop(bx, na, na)
    left, top = box.get_left(bx), box.get_top(bx)
    assert left != left and top != top

    box.set_rightbottom(bx, 30.75, 0.5)
    assert box.get_right(bx) == 30 and box.get_bottom(bx) == 0.5
    assert type(box.get_right(bx)) is float

    box.set_left(bx, 5.9)
    assert box.get_left(bx) == 5


def __test_a_label_setter_normalizes__():
    """And so does the label's single point"""
    lbl = label.new(10, 1.0, "x")
    assert label.get_x(lbl) == 10

    label.set_xy(lbl, na, na)
    x, y = label.get_x(lbl), label.get_y(lbl)
    assert x != x and y != y

    label.set_x(lbl, 8.9)
    assert label.get_x(lbl) == 8
    assert type(label.get_x(lbl)) is float
