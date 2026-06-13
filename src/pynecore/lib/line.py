from copy import copy as _copy
from typing import overload

from ..core.module_property import module_property
from ..types.chart import ChartPoint
from ..types.line import LineEnum, Line
from ..types.na import NA, na_int, na_float
from ..lib import xloc as _xloc, extend as _extend, color as _color

_registry: list[Line] = []

style_arrow_both = LineEnum('ab')
style_arrow_left = LineEnum('al')
style_arrow_right = LineEnum('ar')
style_dashed = LineEnum('dsh')
style_dotted = LineEnum('dot')
style_solid = LineEnum('sol')


@overload
def new(first_point: ChartPoint, second_point: ChartPoint, xloc: _xloc.XLoc = _xloc.bar_index,
        extend: _extend.Extend = _extend.none, color: _color.Color = _color.blue,
        style: LineEnum = style_solid, width: int = 1, force_overlay: bool = False) -> Line: ...


@overload
def new(x1: int | float, y1: float, x2: int | float, y2: float,
        xloc: _xloc.XLoc = _xloc.bar_index, extend: _extend.Extend = _extend.none,
        color: _color.Color = _color.blue, style: LineEnum = style_solid,
        width: int = 1, force_overlay: bool = False) -> Line: ...


def new(x1: ChartPoint | int | float | None = None, y1: ChartPoint | float | None = None,
        x2: int | float | None = None, y2: float | None = None,
        xloc: _xloc.XLoc = _xloc.bar_index, extend: _extend.Extend = _extend.none,
        color: _color.Color = _color.blue, style: LineEnum = style_solid,
        width: int = 1, force_overlay: bool = False,
        first_point: ChartPoint | None = None, second_point: ChartPoint | None = None) -> Line:
    """
    Creates a new line object.

    Two call shapes are accepted (Pine-compatible):
    - ``line.new(first_point, second_point, ...)`` where both arguments are ``chart.point`` objects.
    - ``line.new(x1, y1, x2, y2, ...)`` where ``x1`` / ``x2`` are bar index
      (``xloc.bar_index``) or bar UNIX time in milliseconds (``xloc.bar_time``),
      and ``y1`` / ``y2`` are prices. Float ``x1`` / ``x2`` are truncated to int
      to mirror Pine's implicit float-to-int conversion on ``series int`` parameters.

    :param x1: Bar index / bar time of the first point (coordinate form), or the first
               ``chart.point`` (point form, when passed as the first positional argument)
    :param y1: Price of the first point (coordinate form), or the second ``chart.point`` when
               the first positional argument is a ``chart.point``
    :param x2: Bar index / bar time of the second point (coordinate form only)
    :param y2: Price of the second point (coordinate form only)
    :param xloc: Possible values: ``xloc.bar_index`` and ``xloc.bar_time``
    :param extend: If ``extend=extend.none``, draws segment from (x1, y1) to (x2, y2)
    :param color: Line color
    :param style: Line style. Possible values: ``line.style_solid``, ``line.style_dotted``,
                  ``line.style_dashed``, ``line.style_arrow_left``, ``line.style_arrow_right``,
                  ``line.style_arrow_both``
    :param width: Line width in pixels
    :param force_overlay: If true, the drawing will display on the main chart pane
    :param first_point: First ``chart.point`` (point form, keyword equivalent of the first positional)
    :param second_point: Second ``chart.point`` (point form, keyword equivalent of the second positional)
    :return: A line object
    """
    if first_point is not None:
        x1 = first_point
    if second_point is not None:
        y1 = second_point
    if isinstance(x1, ChartPoint):
        second = y1 if isinstance(y1, ChartPoint) else x1
        if xloc == _xloc.bar_time:
            x1_val, y1_val = x1.time, x1.price
            x2_val, y2_val = second.time, second.price
        else:
            x1_val, y1_val = x1.index, x1.price
            x2_val, y2_val = second.index, second.price
    else:
        x1_val = int(x1) if isinstance(x1, (int, float)) else na_int
        y1_val = y1 if isinstance(y1, (int, float)) else na_float
        x2_val = int(x2) if isinstance(x2, (int, float)) else na_int
        y2_val = y2 if isinstance(y2, (int, float)) else na_float

    line_obj = Line(
        x1=x1_val,
        y1=y1_val,
        x2=x2_val,
        y2=y2_val,
        xloc=xloc,
        extend=extend,
        color=color,
        style=style or style_solid,
        width=width,
        force_overlay=force_overlay
    )
    _registry.append(line_obj)
    return line_obj


# noinspection PyShadowingBuiltins
@module_property
def all() -> list[Line]:
    """Returns all line objects"""
    return _registry


# noinspection PyShadowingBuiltins
def delete(id):
    """Delete line object"""
    if isinstance(id, NA):
        return
    try:
        _registry.remove(id)
    except ValueError:
        pass


# noinspection PyShadowingBuiltins
def copy(id):
    """Copy line object"""
    if isinstance(id, NA):
        return NA(Line)
    return _copy(id)


# noinspection PyShadowingBuiltins
def get_x1(id: Line) -> int | NA:
    """
    Returns UNIX time or bar index (depending on the last xloc value set) of the first point of the line.

    :param id: Line object
    :return: UNIX timestamp (in milliseconds) or bar index
    """
    if isinstance(id, NA):
        return NA(int)
    return id.x1


# noinspection PyShadowingBuiltins
def get_y1(id: Line) -> float | NA:
    """
    Returns price of the first point of the line.

    :param id: Line object
    :return: Price of the first point
    """
    if isinstance(id, NA):
        return NA(float)
    return id.y1


# noinspection PyShadowingBuiltins
def get_x2(id: Line) -> int | NA:
    """
    Returns UNIX time or bar index (depending on the last xloc value set) of the second point of the line.

    :param id: Line object
    :return: UNIX timestamp (in milliseconds) or bar index
    """
    if isinstance(id, NA):
        return NA(int)
    return id.x2


# noinspection PyShadowingBuiltins
def get_y2(id: Line) -> float | NA:
    """
    Returns price of the second point of the line.

    :param id: Line object
    :return: Price of the second point
    """
    if isinstance(id, NA):
        return NA(float)
    return id.y2


# noinspection PyShadowingBuiltins
def set_color(id: Line, color: _color.Color) -> None:
    """
    Sets the line color

    :param id: Line object
    :param color: New line color
    """
    if isinstance(id, NA):
        return
    id.color = color


# noinspection PyShadowingBuiltins
def set_extend(id: Line, extend: _extend.Extend) -> None:
    """
    Sets extending type of this line object

    :param id: Line object
    :param extend: New extending type
    """
    if isinstance(id, NA):
        return
    id.extend = extend


# noinspection PyShadowingBuiltins
def set_first_point(id: Line, point: ChartPoint) -> None:
    """
    Sets the first point of the id line to point

    :param id: A line object
    :param point: A chart.point object
    """
    if isinstance(id, NA):
        return
    if id.xloc == _xloc.bar_time:
        id.x1, id.y1 = point.time, point.price
    else:  # xloc.bar_index
        id.x1, id.y1 = point.index, point.price


# noinspection PyShadowingBuiltins
def set_second_point(id: Line, point: ChartPoint) -> None:
    """
    Sets the second point of the id line to point

    :param id: A line object
    :param point: A chart.point object
    """
    if isinstance(id, NA):
        return
    if id.xloc == _xloc.bar_time:
        id.x2, id.y2 = point.time, point.price
    else:  # xloc.bar_index
        id.x2, id.y2 = point.index, point.price


# noinspection PyShadowingBuiltins
def set_style(id: Line, style: LineEnum) -> None:
    """
    Sets the line style

    :param id: Line object
    :param style: New line style
    """
    if isinstance(id, NA):
        return
    id.style = style


# noinspection PyShadowingBuiltins
def set_width(id: Line, width: int) -> None:
    """
    Sets the line width

    :param id: Line object
    :param width: New line width in pixels
    """
    if isinstance(id, NA):
        return
    id.width = width


# noinspection PyShadowingBuiltins
def set_x1(id: Line, x: int) -> None:
    """
    Sets bar index or bar time (depending on the xloc) of the first point

    :param id: Line object
    :param x: Bar index or bar time
    """
    if isinstance(id, NA):
        return
    id.x1 = x


# noinspection PyShadowingBuiltins
def set_x2(id: Line, x: int) -> None:
    """
    Sets bar index or bar time (depending on the xloc) of the second point

    :param id: Line object
    :param x: Bar index or bar time
    """
    if isinstance(id, NA):
        return
    id.x2 = x


# noinspection PyShadowingBuiltins
def set_xloc(id: Line, x1: int, x2: int, xloc: _xloc.XLoc) -> None:
    """
    Sets x-location and new bar index/time values

    :param id: Line object
    :param x1: Bar index or bar time of the first point
    :param x2: Bar index or bar time of the second point
    :param xloc: New x-location value
    """
    if isinstance(id, NA):
        return
    id.x1 = x1
    id.x2 = x2
    id.xloc = xloc


# noinspection PyShadowingBuiltins
def set_xy1(id: Line, x: int, y: float) -> None:
    """
    Sets bar index/time and price of the first point

    :param id: Line object
    :param x: Bar index or bar time
    :param y: Price
    """
    if isinstance(id, NA):
        return
    id.x1 = x
    id.y1 = y


# noinspection PyShadowingBuiltins
def set_xy2(id: Line, x: int, y: float) -> None:
    """
    Sets bar index/time and price of the second point

    :param id: Line object
    :param x: Bar index or bar time
    :param y: Price
    """
    if isinstance(id, NA):
        return
    id.x2 = x
    id.y2 = y


# noinspection PyShadowingBuiltins
def set_y1(id: Line, y: float) -> None:
    """
    Sets price of the first point

    :param id: Line object
    :param y: Price
    """
    if isinstance(id, NA):
        return
    id.y1 = y


# noinspection PyShadowingBuiltins
def set_y2(id: Line, y: float) -> None:
    """
    Sets price of the second point

    :param id: Line object
    :param y: Price
    """
    if isinstance(id, NA):
        return
    id.y2 = y


# noinspection PyShadowingBuiltins
def get_price(id: Line, x: int) -> float | NA:
    """
    Returns the price level of a line at a given bar index.

    :param id: Line object
    :param x: Bar index for which price is required
    :return: Price value of line 'id' at bar index 'x'
    :raises RuntimeError: If the line was created with xloc.bar_time instead of xloc.bar_index
    """
    if isinstance(id, NA):
        return NA(float)
    
    # Check if line was created with xloc.bar_index
    if id.xloc != _xloc.bar_index:
        raise RuntimeError("line.get_price() can only be called for lines created using 'xloc.bar_index'")
    
    # Line is considered to have been created using 'extend=extend.both'
    # Calculate the slope of the line
    x1, y1 = id.x1, id.y1
    x2, y2 = id.x2, id.y2
    
    # Handle vertical line case (x1 == x2)
    if x1 == x2:
        # For vertical lines, return y1 for any x
        return y1
    
    # Calculate price using linear interpolation/extrapolation
    # Formula: y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    slope = (y2 - y1) / (x2 - x1)
    price = y1 + slope * (x - x1)
    
    return price
