from copy import copy as _copy
from typing import Any, overload

from ..core.module_property import module_property

from ..types.base import next_vid
from ..types.box import Box
from ..types.na import NA, na_int, na_float
from ..types.chart import ChartPoint
from ..lib import (color as _color, extend as _extend, xloc as _xloc, size as _size, line as _line,
                   text as _text, font as _font)
from .. import lib

_registry: dict[Box, None] = {}


@overload
def new(top_left: ChartPoint, bottom_right: ChartPoint, border_color: _color.Color = _color.blue,
        border_width: int = 1, border_style: _line.LineEnum = _line.style_solid,
        extend: _extend.Extend = _extend.none, xloc: _xloc.XLoc = _xloc.bar_index,
        bgcolor: _color.Color = _color.blue, text: str = "", text_size: _size.Size = _size.auto,
        text_color: _color.Color = _color.black, text_halign: _text.AlignEnum = _text.align_center,
        text_valign: _text.AlignEnum = _text.align_center, text_wrap: _text.WrapEnum = _text.wrap_none,
        text_font_family: _font.FontFamilyEnum = _font.family_default, force_overlay: bool = False,
        text_formatting: _text.FormatEnum = _text.format_none) -> Box: ...


@overload
def new(left: int | float, top: float, right: int | float, bottom: float,
        border_color: _color.Color = _color.blue, border_width: int = 1,
        border_style: _line.LineEnum = _line.style_solid, extend: _extend.Extend = _extend.none,
        xloc: _xloc.XLoc = _xloc.bar_index, bgcolor: _color.Color = _color.blue, text: str = "",
        text_size: _size.Size = _size.auto, text_color: _color.Color = _color.black,
        text_halign: _text.AlignEnum = _text.align_center, text_valign: _text.AlignEnum = _text.align_center,
        text_wrap: _text.WrapEnum = _text.wrap_none, text_font_family: _font.FontFamilyEnum = _font.family_default,
        force_overlay: bool = False, text_formatting: _text.FormatEnum = _text.format_none) -> Box: ...


# Positional parameter orders of the two Pine call shapes; ``new()`` maps ``*args``
# onto one of these depending on whether the first positional is a ``chart.point``.
_COMMON_PARAMS = ('border_color', 'border_width', 'border_style', 'extend', 'xloc', 'bgcolor',
                  'text', 'text_size', 'text_color', 'text_halign', 'text_valign', 'text_wrap',
                  'text_font_family', 'force_overlay', 'text_formatting')
_POINT_PARAMS = ('top_left', 'bottom_right') + _COMMON_PARAMS
_COORD_PARAMS = ('left', 'top', 'right', 'bottom') + _COMMON_PARAMS


# noinspection PyProtectedMember
def new(*args: Any, **kwargs: Any) -> Box:
    """
    Creates a new box object.

    Two call shapes are accepted (Pine-compatible):
    - ``box.new(top_left, bottom_right, ...)`` where both arguments are ``chart.point`` objects
      (the top-left and bottom-right corners).
    - ``box.new(left, top, right, bottom, ...)`` where ``left`` / ``right`` are bar index
      (``xloc.bar_index``) or bar UNIX time in milliseconds (``xloc.bar_time``),
      and ``top`` / ``bottom`` are prices. Float ``left`` / ``right`` are truncated
      to int to mirror Pine's implicit float-to-int conversion on ``series int`` parameters.

    :param left: Bar index / bar time of the left border (coordinate form)
    :param top: Price of the top border (coordinate form)
    :param right: Bar index / bar time of the right border (coordinate form only)
    :param bottom: Price of the bottom border (coordinate form only)
    :param border_color: Color of the four borders
    :param border_width: Width of the four borders, in pixels
    :param border_style: Style of the four borders
    :param extend: When ``extend.none`` is used, the horizontal borders start at the left border and end at the right border
    :param xloc: Determines whether ``left`` / ``right`` are a bar index or a bar time value
    :param bgcolor: Background color of the box
    :param text: The text to be displayed inside the box
    :param text_size: Size of the box's text
    :param text_color: The color of the text
    :param text_halign: The horizontal alignment of the box's text
    :param text_valign: The vertical alignment of the box's text
    :param text_wrap: Whether to wrap text. Wrapped text starts a new line
    :param text_font_family: The font family of the text
    :param force_overlay: If true, the drawing will display on the main chart pane
    :param text_formatting: The formatting of the displayed text
    :param top_left: Top-left corner ``chart.point`` (point form, keyword equivalent of the first positional)
    :param bottom_right: Bottom-right corner ``chart.point`` (point form, keyword equivalent of the second positional)
    :return: A box object
    """
    if args:
        names = _POINT_PARAMS if isinstance(args[0], ChartPoint) else _COORD_PARAMS
        if len(args) > len(names):
            raise TypeError(f"box.new() takes at most {len(names)} positional arguments")
        for name, value in zip(names, args):
            if name in kwargs:
                raise TypeError(f"box.new() got multiple values for argument '{name}'")
            kwargs[name] = value
    top_left = kwargs.get('top_left')
    bottom_right = kwargs.get('bottom_right')
    border_color = kwargs.get('border_color', _color.blue)
    border_width = kwargs.get('border_width', 1)
    border_style = kwargs.get('border_style', _line.style_solid)
    extend = kwargs.get('extend', _extend.none)
    xloc = kwargs.get('xloc', _xloc.bar_index)
    bgcolor = kwargs.get('bgcolor', _color.blue)
    text = kwargs.get('text', "")
    text_size = kwargs.get('text_size', _size.auto)
    text_color = kwargs.get('text_color', _color.black)
    text_halign = kwargs.get('text_halign', _text.align_center)
    text_valign = kwargs.get('text_valign', _text.align_center)
    text_wrap = kwargs.get('text_wrap', _text.wrap_none)
    text_font_family = kwargs.get('text_font_family', _font.family_default)
    force_overlay = kwargs.get('force_overlay', False)
    text_formatting = kwargs.get('text_formatting', _text.format_none)
    if isinstance(top_left, ChartPoint):
        bottom_right_point = bottom_right if isinstance(bottom_right, ChartPoint) else top_left
        if xloc == _xloc.bar_time:
            left_val, top_val = top_left.time, top_left.price
            right_val, bottom_val = bottom_right_point.time, bottom_right_point.price
        else:
            left_val, top_val = top_left.index, top_left.price
            right_val, bottom_val = bottom_right_point.index, bottom_right_point.price
    else:
        left = kwargs.get('left')
        top = kwargs.get('top')
        right = kwargs.get('right')
        bottom = kwargs.get('bottom')
        left_val = int(left) if isinstance(left, (int, float)) else na_int
        top_val = top if isinstance(top, (int, float)) else na_float
        right_val = int(right) if isinstance(right, (int, float)) else na_int
        bottom_val = bottom if isinstance(bottom, (int, float)) else na_float

    box = Box(
        left=left_val,
        top=top_val,
        right=right_val,
        bottom=bottom_val,
        border_color=border_color,
        border_width=border_width,
        border_style=border_style,
        extend=extend,
        xloc=xloc,
        bgcolor=bgcolor,
        text=text,
        text_size=text_size,
        text_color=text_color,
        text_halign=text_halign,
        text_valign=text_valign,
        text_wrap=text_wrap,
        text_font_family=text_font_family,
        text_formatting=text_formatting,
        force_overlay=force_overlay,
    )
    box.vid = next_vid()
    _registry[box] = None
    # Enforce Pine's max_boxes_count cap: drop the oldest box (FIFO) past the limit.
    # A security child never sets ``lib._script``; fall back to TV's hard maximum
    # (500) there, otherwise the registry grows without bound (the child re-runs
    # main() for every bar of its own series, accumulating every drawing ever made).
    if len(_registry) > (lib._script.max_boxes_count if lib._script is not None else 500):
        del _registry[next(iter(_registry))]
    return box


# noinspection PyShadowingBuiltins
@module_property
def all() -> list[Box]:
    """Returns all box objects"""
    return list(_registry)


# noinspection PyShadowingBuiltins
def delete(id):
    if isinstance(id, NA):
        return
    _registry.pop(id, None)


# noinspection PyShadowingBuiltins,PyProtectedMember
def copy(id):
    if isinstance(id, NA):
        return NA(Box)
    clone = _copy(id)
    clone.vid = next_vid()
    _registry[clone] = None
    if len(_registry) > (lib._script.max_boxes_count if lib._script is not None else 500):
        del _registry[next(iter(_registry))]
    return clone


# Setter methods

# noinspection PyShadowingBuiltins
def set_bgcolor(id: Box, color: _color.Color) -> None:
    """Sets the background color of the box."""
    if isinstance(id, NA):
        return
    id.bgcolor = color


# noinspection PyShadowingBuiltins
def set_border_color(id: Box, color: _color.Color) -> None:
    """Sets the border color of the box."""
    if isinstance(id, NA):
        return
    id.border_color = color


# noinspection PyShadowingBuiltins
def set_border_style(id: Box, style: _line.LineEnum) -> None:
    """Sets the border style of the box."""
    if isinstance(id, NA):
        return
    id.border_style = style


# noinspection PyShadowingBuiltins
def set_border_width(id: Box, width: int) -> None:
    """Sets the border width of the box."""
    if isinstance(id, NA):
        return
    id.border_width = width


# noinspection PyShadowingBuiltins
def set_bottom(id: Box, bottom: float) -> None:
    """Sets the bottom coordinate of the box."""
    if isinstance(id, NA):
        return
    id.bottom = bottom


# noinspection PyShadowingBuiltins
def set_bottom_right_point(id: Box, point: ChartPoint) -> None:
    """Sets the bottom-right corner location of the box to point."""
    if isinstance(id, NA):
        return
    if id.xloc == _xloc.bar_time:
        id.right = point.time
    else:
        id.right = point.index
    id.bottom = point.price


# noinspection PyShadowingBuiltins
def set_extend(id: Box, extend: _extend.Extend) -> None:
    """Sets extending type of the border of this box object."""
    if isinstance(id, NA):
        return
    id.extend = extend


# noinspection PyShadowingBuiltins
def set_left(id: Box, left: int) -> None:
    """Sets the left coordinate of the box."""
    if isinstance(id, NA):
        return
    id.left = left


# noinspection PyShadowingBuiltins
def set_lefttop(id: Box, left: int, top: float) -> None:
    """Sets the left and top coordinates of the box."""
    if isinstance(id, NA):
        return
    id.left = left
    id.top = top


# noinspection PyShadowingBuiltins
def set_right(id: Box, right: int) -> None:
    """Sets the right coordinate of the box."""
    if isinstance(id, NA):
        return
    id.right = right


# noinspection PyShadowingBuiltins
def set_rightbottom(id: Box, right: int, bottom: float) -> None:
    """Sets the right and bottom coordinates of the box."""
    if isinstance(id, NA):
        return
    id.right = right
    id.bottom = bottom


# noinspection PyShadowingBuiltins


def set_text(id: Box, text: str) -> None:
    """Sets the text in the box."""
    if isinstance(id, NA):
        return
    id.text = text


# noinspection PyShadowingBuiltins
def set_text_color(id: Box, text_color: _color.Color) -> None:
    """Sets the color of the text inside the box."""
    if isinstance(id, NA):
        return
    id.text_color = text_color


# noinspection PyShadowingBuiltins
def set_text_font_family(id: Box, text_font_family: _font.FontFamilyEnum) -> None:
    """Sets the font family of the text inside the box."""
    if isinstance(id, NA):
        return
    id.text_font_family = text_font_family


# noinspection PyShadowingBuiltins
def set_text_formatting(id: Box, text_formatting: _text.FormatEnum) -> None:
    """Sets the formatting attributes the drawing applies to displayed text."""
    if isinstance(id, NA):
        return
    id.text_formatting = text_formatting


# noinspection PyShadowingBuiltins
def set_text_halign(id: Box, text_halign: _text.AlignEnum) -> None:
    """Sets the horizontal alignment of the box's text."""
    if isinstance(id, NA):
        return
    id.text_halign = text_halign


# noinspection PyShadowingBuiltins
def set_text_size(id: Box, text_size: _size.Size) -> None:
    """Sets the size of the box's text."""
    if isinstance(id, NA):
        return
    id.text_size = text_size


# noinspection PyShadowingBuiltins
def set_text_valign(id: Box, text_valign: _text.AlignEnum) -> None:
    """Sets the vertical alignment of the box's text."""
    if isinstance(id, NA):
        return
    id.text_valign = text_valign


# noinspection PyShadowingBuiltins
def set_text_wrap(id: Box, text_wrap: _text.WrapEnum) -> None:
    """Sets the mode of wrapping of the text inside the box."""
    if isinstance(id, NA):
        return
    id.text_wrap = text_wrap


# noinspection PyShadowingBuiltins
def set_top(id: Box, top: float) -> None:
    """Sets the top coordinate of the box."""
    if isinstance(id, NA):
        return
    id.top = top


# noinspection PyShadowingBuiltins
def set_top_left_point(id: Box, point: ChartPoint) -> None:
    """Sets the top-left corner location of the box to point."""
    if isinstance(id, NA):
        return
    if id.xloc == _xloc.bar_time:
        id.left = point.time
    else:
        id.left = point.index
    id.top = point.price


# noinspection PyShadowingBuiltins
def set_xloc(id: Box, left: int, right: int, xloc: _xloc.XLoc) -> None:
    """Sets the left and right borders of a box and updates its xloc property."""
    if isinstance(id, NA):
        return
    id.left = left
    id.right = right
    id.xloc = xloc


# Getter methods

# noinspection PyShadowingBuiltins
def get_bottom(id: Box) -> float | NA:
    """Returns the price value of the bottom border of the box."""
    if isinstance(id, NA):
        return NA(float)
    return id.bottom


# noinspection PyShadowingBuiltins
def get_left(id: Box) -> int | NA:
    """
    Returns the bar index or the UNIX time (depending on the last value used for 'xloc') of the
    left border of the box.
    """
    if isinstance(id, NA):
        return NA(int)
    return id.left


# noinspection PyShadowingBuiltins
def get_right(id: Box) -> int | NA:
    """
    Returns the bar index or the UNIX time (depending on the last value used for 'xloc') of the
    right border of the box.
    """
    if isinstance(id, NA):
        return NA(int)
    return id.right


# noinspection PyShadowingBuiltins
def get_top(id: Box) -> float | NA:
    """Returns the price value of the top border of the box."""
    if isinstance(id, NA):
        return NA(float)
    return id.top
