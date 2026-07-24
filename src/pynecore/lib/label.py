from copy import copy as _copy
from typing import Any, overload

from ..core.module_property import module_property
from ..types.chart import ChartPoint
from ..types.base import next_vid
from ..types.label import LabelStyleEnum, Label
from ..types.na import NA, na_int, na_float
from ..types.pine_types import PyneFloat, PyneInt, PyneStr
from ..lib import xloc as _xloc, yloc as _yloc, color as _color, size as _size, text as _text, font as _font
from .. import lib

_registry: dict[Label, None] = {}

# Label style constants
style_none = LabelStyleEnum('n')
style_xcross = LabelStyleEnum('xcr')
style_cross = LabelStyleEnum('cr')
style_triangleup = LabelStyleEnum('tup')
style_triangledown = LabelStyleEnum('tdn')
style_flag = LabelStyleEnum('flg')
style_circle = LabelStyleEnum('cir')
style_arrowup = LabelStyleEnum('aup')
style_arrowdown = LabelStyleEnum('adn')
style_label_up = LabelStyleEnum('lup')
style_label_down = LabelStyleEnum('ldn')
style_label_left = LabelStyleEnum('llf')
style_label_right = LabelStyleEnum('lrg')
style_label_lower_left = LabelStyleEnum('llwlf')
style_label_lower_right = LabelStyleEnum('llwrg')
style_label_upper_left = LabelStyleEnum('luplf')
style_label_upper_right = LabelStyleEnum('luprg')
style_label_center = LabelStyleEnum('lcn')
style_square = LabelStyleEnum('sq')
style_diamond = LabelStyleEnum('dia')
style_text_outline = LabelStyleEnum('to')


@overload
def new(point: ChartPoint, text: str = "", xloc: _xloc.XLoc = _xloc.bar_index,
        yloc: _yloc.YLoc = _yloc.price, color: _color.Color = _color.blue,
        style: LabelStyleEnum = style_label_down, textcolor: _color.Color = _color.white,
        size: _size.Size = _size.normal, textalign: _text.AlignEnum = _text.align_center,
        tooltip: str = "", text_font_family: _font.FontFamilyEnum = _font.family_default,
        force_overlay: bool = False, text_formatting: _text.FormatEnum = _text.format_none) -> Label: ...


@overload
def new(x: int | float, y: int | float, text: str = "", xloc: _xloc.XLoc = _xloc.bar_index,
        yloc: _yloc.YLoc = _yloc.price, color: _color.Color = _color.blue,
        style: LabelStyleEnum = style_label_down, textcolor: _color.Color = _color.white,
        size: _size.Size = _size.normal, textalign: _text.AlignEnum = _text.align_center,
        tooltip: str = "", text_font_family: _font.FontFamilyEnum = _font.family_default,
        force_overlay: bool = False, text_formatting: _text.FormatEnum = _text.format_none) -> Label: ...


# Positional parameter orders of the two Pine call shapes; ``new()`` maps ``*args``
# onto one of these depending on whether the first positional is a ``chart.point``.
_COMMON_PARAMS = ('xloc', 'yloc', 'color', 'style', 'textcolor', 'size', 'textalign',
                  'tooltip', 'text_font_family', 'force_overlay', 'text_formatting')
_POINT_PARAMS = ('point', 'text') + _COMMON_PARAMS
_COORD_PARAMS = ('x', 'y', 'text') + _COMMON_PARAMS


# noinspection PyProtectedMember
def new(*args: Any, **kwargs: Any) -> Label:
    """
    Creates a new label object.

    Two call shapes are accepted (Pine-compatible):
    - ``label.new(point, text="", ...)`` where ``point`` is a ``chart.point`` object. In this
      form the second positional argument is the label ``text``.
    - ``label.new(x, y, text="", ...)`` where ``x`` is bar index (``xloc.bar_index``) or
      bar UNIX time in milliseconds (``xloc.bar_time``), and ``y`` is the price.
      A float ``x`` is truncated to int to mirror Pine's implicit float-to-int
      conversion on ``series int`` parameters.

    :param x: Bar index / bar time of the label position (coordinate form)
    :param y: Price of the label position (coordinate form)
    :param text: Label text
    :param xloc: Possible values: ``xloc.bar_index`` and ``xloc.bar_time``
    :param yloc: Possible values are ``yloc.price``, ``yloc.abovebar``, ``yloc.belowbar``
    :param color: Color of the label border and arrow
    :param style: Label style
    :param textcolor: Text color
    :param size: Size of the label
    :param textalign: Label text alignment
    :param tooltip: Hover to see tooltip label
    :param text_font_family: The font family of the text
    :param force_overlay: If true, the drawing will display on the main chart pane
    :param text_formatting: The formatting of the displayed text
    :param point: ``chart.point`` object (point form, keyword equivalent of the first positional)
    :return: A label object
    """
    if args:
        names = _POINT_PARAMS if isinstance(args[0], ChartPoint) else _COORD_PARAMS
        if len(args) > len(names):
            raise TypeError(f"label.new() takes at most {len(names)} positional arguments")
        for name, value in zip(names, args):
            if name in kwargs:
                raise TypeError(f"label.new() got multiple values for argument '{name}'")
            kwargs[name] = value
    point = kwargs.get('point')
    text = kwargs.get('text', "")
    xloc = kwargs.get('xloc', _xloc.bar_index)
    yloc = kwargs.get('yloc', _yloc.price)
    color = kwargs.get('color', _color.blue)
    style = kwargs.get('style', style_label_down)
    textcolor = kwargs.get('textcolor', _color.white)
    size = kwargs.get('size', _size.normal)
    textalign = kwargs.get('textalign', _text.align_center)
    tooltip = kwargs.get('tooltip', "")
    text_font_family = kwargs.get('text_font_family', _font.family_default)
    force_overlay = kwargs.get('force_overlay', False)
    text_formatting = kwargs.get('text_formatting', _text.format_none)
    if isinstance(point, ChartPoint):
        if xloc == _xloc.bar_time:
            x_val, y_val = point.time, point.price
        else:
            x_val, y_val = point.index, point.price
    else:
        x = kwargs.get('x')
        y = kwargs.get('y')
        x_val = int(x) if isinstance(x, (int, float)) and x == x else na_int
        y_val = y if isinstance(y, (int, float)) else na_float

    label_obj = Label(
        x=x_val,
        y=y_val,
        text=text,
        xloc=xloc,
        yloc=yloc or _yloc.price,
        color=color,
        style=style or style_label_down,
        textcolor=textcolor,
        size=size or _size.normal,
        textalign=textalign or _text.align_center,
        tooltip=tooltip,
        text_font_family=text_font_family or _font.family_default,
        force_overlay=force_overlay,
        text_formatting=text_formatting or _text.format_none
    )
    label_obj.vid = next_vid()
    _registry[label_obj] = None
    # Enforce Pine's max_labels_count cap: drop the oldest label (FIFO) past the limit.
    # A security child never sets ``lib._script``; fall back to TV's hard maximum
    # (500) there, otherwise the registry grows without bound (the child re-runs
    # main() for every bar of its own series, accumulating every drawing ever made).
    if len(_registry) > (lib._script.max_labels_count if lib._script is not None else 500):
        del _registry[next(iter(_registry))]
    return label_obj


# noinspection PyShadowingBuiltins
@module_property
def all() -> list[Label]:
    """Returns all label objects"""
    return list(_registry)


# noinspection PyShadowingBuiltins
def delete(id):
    """Delete label object"""
    if isinstance(id, NA):
        return
    _registry.pop(id, None)


# noinspection PyShadowingBuiltins,PyProtectedMember
def copy(id):
    """Copy label object"""
    if isinstance(id, NA):
        return NA(Label)
    clone = _copy(id)
    clone.vid = next_vid()
    _registry[clone] = None
    if len(_registry) > (lib._script.max_labels_count if lib._script is not None else 500):
        del _registry[next(iter(_registry))]
    return clone


# noinspection PyShadowingBuiltins
def get_text(id: Label) -> PyneStr:
    """
    Returns the text of the label.

    :param id: Label object
    :return: Label text
    """
    if isinstance(id, NA):
        return NA(str)
    return id.text


# noinspection PyShadowingBuiltins
def set_text(id: Label, text: str) -> None:
    """
    Sets the label text

    :param id: Label object
    :param text: New label text
    """
    if isinstance(id, NA):
        return
    id.text = text


# noinspection PyShadowingBuiltins
def set_color(id: Label, color: _color.Color) -> None:
    """
    Sets the label color

    :param id: Label object
    :param color: New label color
    """
    if isinstance(id, NA):
        return
    id.color = color


# noinspection PyShadowingBuiltins
def set_style(id: Label, style: LabelStyleEnum) -> None:
    """
    Sets the label style

    :param id: Label object
    :param style: New label style
    """
    if isinstance(id, NA):
        return
    id.style = style


# noinspection PyShadowingBuiltins
def set_textcolor(id: Label, textcolor: _color.Color) -> None:
    """
    Sets the label text color

    :param id: Label object
    :param textcolor: New text color
    """
    if isinstance(id, NA):
        return
    id.textcolor = textcolor


# noinspection PyShadowingBuiltins
def set_size(id: Label, size: _size.Size) -> None:
    """
    Sets the label size

    :param id: Label object
    :param size: New label size
    """
    if isinstance(id, NA):
        return
    id.size = size


# noinspection PyShadowingBuiltins
def set_textalign(id: Label, textalign: _text.AlignEnum) -> None:
    """
    Sets the label text alignment

    :param id: Label object
    :param textalign: New text alignment
    """
    if isinstance(id, NA):
        return
    id.textalign = textalign


# noinspection PyShadowingBuiltins
def set_tooltip(id: Label, tooltip: str) -> None:
    """
    Sets the label tooltip

    :param id: Label object
    :param tooltip: New tooltip text
    """
    if isinstance(id, NA):
        return
    id.tooltip = tooltip


# noinspection PyShadowingBuiltins
def set_x(id: Label, x: int) -> None:
    """
    Sets bar index or bar time (depending on the xloc) of the label

    :param id: Label object
    :param x: Bar index or bar time
    """
    if isinstance(id, NA):
        return
    id.x = x


# noinspection PyShadowingBuiltins
def set_y(id: Label, y: int | float) -> None:
    """
    Sets price of the label

    :param id: Label object
    :param y: Price
    """
    if isinstance(id, NA):
        return
    id.y = y


# noinspection PyShadowingBuiltins
def set_xy(id: Label, x: int, y: int | float) -> None:
    """
    Sets bar index/time and price of the label

    :param id: Label object
    :param x: Bar index or bar time
    :param y: Price
    """
    if isinstance(id, NA):
        return
    id.x = x
    id.y = y


# noinspection PyShadowingBuiltins
def set_yloc(id: Label, yloc: _yloc.YLoc) -> None:
    """
    Sets the y-location of the label

    :param id: Label object
    :param yloc: New y-location value
    """
    if isinstance(id, NA):
        return
    id.yloc = yloc


# noinspection PyShadowingBuiltins
def get_x(id: Label) -> PyneInt:
    """
    Returns bar index or UNIX time (depending on the xloc value) of the label.

    :param id: Label object
    :return: Bar index or UNIX timestamp (in milliseconds)
    """
    if isinstance(id, NA):
        return NA(int)
    return id.x


# noinspection PyShadowingBuiltins
def get_y(id: Label) -> PyneInt | PyneFloat:
    """
    Returns price of the label.

    :param id: Label object
    :return: Price of the label
    """
    if isinstance(id, NA):
        return na_float
    return id.y
