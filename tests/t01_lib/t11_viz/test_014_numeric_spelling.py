"""
@pyne

Every numeric field of the visual-data stream is spelled by its Pine type.

A Pine int is a double at run time while a literal stays a Python int, so one field could
reach the serializer as ``2`` or ``2.0`` depending on where the value came from. The stream
must not show that: a Pine int field prints as an integer, a Pine float field as a float.
"""
import json

from pynecore.core.viz import (
    _box_dict, _encode_color_channel, _label_dict, _line_dict, _point_dict, _scrub, _table_dict,
    serialize_meta,
)
from pynecore.lib import color, position
from pynecore.types.box import Box
from pynecore.types.chart import ChartPoint
from pynecore.types.label import Label
from pynecore.types.line import Line
from pynecore.types.na import NA
from pynecore.types.plot_meta import PlotMeta
from pynecore.types.table import Table


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


def __test_helper_dump(d: dict) -> str:
    return json.dumps(d, separators=(',', ':'))


def __test_drawing_fields_ignore_the_python_class__():
    """the same drawing built from ints and from doubles serializes identically"""
    from_ints = Line(x1=3, y1=100, x2=4, y2=101, width=2, vid=1)
    from_floats = Line(x1=3.0, y1=100.0, x2=4.0, y2=101.0, width=2.0, vid=1)
    assert __test_helper_dump(_line_dict(from_ints)) == __test_helper_dump(_line_dict(from_floats))
    d = _line_dict(from_ints)
    assert (type(d["x1"]), type(d["y1"]), type(d["width"])) == (int, float, int)

    label_a = _label_dict(Label(x=1, y=100, vid=2))
    label_b = _label_dict(Label(x=1.0, y=100.0, vid=2))
    assert __test_helper_dump(label_a) == __test_helper_dump(label_b)
    assert label_a["y"] == 100.0 and type(label_a["y"]) is float

    box_a = _box_dict(Box(left=1, top=102, right=2, bottom=101, border_width=2, vid=3))
    box_b = _box_dict(Box(left=1.0, top=102.0, right=2.0, bottom=101.0, border_width=2.0, vid=3))
    assert __test_helper_dump(box_a) == __test_helper_dump(box_b)

    point = _point_dict(ChartPoint(index=2, time=1600000600000, price=100))
    assert point == {"index": 2, "time": 1600000600000, "price": 100.0}
    assert type(point["price"]) is float


def __test_table_fields_ignore_the_python_class__():
    """table sizes and cell sizes keep one spelling each"""
    a = Table(position=position.top_right, columns=2, rows=2, frame_width=1, border_width=2)
    b = Table(position=position.top_right, columns=2.0, rows=2.0, frame_width=1.0, border_width=2.0)
    a.vid = b.vid = 4
    a.get_cell(0, 0).width = 5
    b.get_cell(0, 0).width = 5.0
    assert __test_helper_dump(_table_dict(a)) == __test_helper_dump(_table_dict(b))


def __test_meta_fields_ignore_the_python_class__():
    """plot linewidth is an int field, hline price a float field"""
    lit = serialize_meta(PlotMeta(id="p", kind="plot", linewidth=2))
    inp = serialize_meta(PlotMeta(id="p", kind="plot", linewidth=2.0))
    assert lit == inp and type(lit["linewidth"]) is int
    h_lit = serialize_meta(PlotMeta(id="h", kind="hline", price=50, linewidth=1))
    h_inp = serialize_meta(PlotMeta(id="h", kind="hline", price=50.0, linewidth=1.0))
    assert __test_helper_dump(h_lit) == __test_helper_dump(h_inp)
    assert type(h_lit["price"]) is float


def __test_gradient_fill_values_ignore_the_python_class__():
    """a gradient fill's top/bottom values are float fields in the color channel"""
    lit = _encode_color_channel((70, 30, color.red, color.blue))
    inp = _encode_color_channel((70.0, 30.0, color.red, color.blue))
    assert __test_helper_dump({"c": lit}) == __test_helper_dump({"c": inp})
    assert (type(lit[0]), type(lit[1])) == (float, float)
    assert _scrub(_encode_color_channel((NA(float), 30, None, color.blue)))[:3] == [None, 30.0, None]
