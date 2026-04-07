from typing import TypeVar, Any, cast

import builtins

import math
import statistics

from ..utils.sequence_view import SequenceView

from ..types.na import NA
from ..types import PyneFloat, PyneInt, PyneStr, PyneBool
from ..types.color import Color
from ..types.box import Box
from ..types.line import Line, LineEnum
from ..types.label import Label, LabelStyleEnum
from ..types.linefill import LineFill
from ..types.chart import ChartPoint
from ..core.overload import overload
from . import order as _order
from ..lib import (box as _box, line as _line, label as _label, linefill as _linefill,
                   color as _color, extend as _extend, xloc as _xloc, yloc as _yloc,
                   size as _size, text as _text, font as _font)

T = TypeVar('T')
Number = TypeVar('Number', int, float)

__all__ = [
    'abs',
    'avg',
    'binary_search',
    'binary_search_leftmost',
    'binary_search_rightmost',
    'clear',
    'concat',
    'copy',
    'covariance',
    'every',
    'fill',
    'first',
    'from_items',
    'get',
    'includes',
    'indexof',
    'insert',
    'join',
    'last',
    'lastindexof',
    'max',
    'median',
    'min',
    'mode',
    'new',
    'new_bool',
    'new_box',
    'new_color',
    'new_float',
    'new_int',
    'new_label',
    'new_line',
    'new_linefill',
    'new_string',
    'percentile_linear_interpolation',
    'percentile_nearest_rank',
    'percentrank',
    'pop',
    'push',
    'range',
    'remove',
    'reverse',
    'set',
    'shift',
    'size',
    'slice',
    'some',
    'sort',
    'sort_indices',
    'standardize',
    'stdev',
    'sum',
    'unshift',
    'variance',
]


# noinspection PyShadowingBuiltins
def abs(id: list[int | float]) -> list[int | float]:
    """
    Returns an array containing the absolute value of each element in the original array.

    :param id: Input array
    :return: Array containing the absolute value of each element in the original array
    """
    return [builtins.abs(v) for v in id]


# noinspection PyShadowingBuiltins
def avg(id: list[Number]) -> float:
    """
    Returns the average value of the elements in the array.

    :param id: Input array
    :return: Average value of the elements in the array
    """
    return builtins.sum(id) / len(id)


# noinspection PyShadowingBuiltins
def binary_search(id: list[Any], val: Any) -> int:
    """
    Returns the index of the specified value in the sorted array using binary search.
    If the value is not found, returns -1.
    The array to search must be sorted in ascending order.

    :param id: Input array
    :param val: Value to search for
    :return: Index of the specified value in the sorted array, or -1 if not found
    """
    low = 0
    high = len(id) - 1
    while low <= high:
        mid = (low + high) // 2
        if id[mid] == val:
            return mid
        else:
            if val < id[mid]:
                high = mid - 1
            else:
                low = mid + 1
    return -1


# noinspection PyShadowingBuiltins
def binary_search_leftmost(id: list[Any], val: Any) -> int:
    """
    Returns the index of the specified value in the sorted array using binary search.
    If the value is not found, returns the index of the leftmost element greater than the value.
    The array to search must be sorted in ascending order.

    :param id: Input array
    :param val: Value to search for
    :return: Index of the specified value in the sorted array, or the index of the leftmost element
             greater than the value
    """
    low = 0
    high = len(id) - 1
    while low <= high:
        mid = (low + high) // 2
        if id[mid] == val:
            return mid
        else:
            if val < id[mid]:
                high = mid - 1
            else:
                low = mid + 1
    return low - 1


# noinspection PyShadowingBuiltins
def binary_search_rightmost(id: list[Any], val: Any) -> int:
    """
    Returns the index of the specified value in the sorted array using binary search.
    If the value is not found, returns the index of the rightmost element less than the value.
    The array to search must be sorted in ascending order.

    :param id: Input array
    :param val: Value to search for
    :return: Index of the specified value in the sorted array, or the index of the rightmost element less than the value
    """
    low = 0
    high = len(id) - 1
    while low <= high:
        mid = (low + high) // 2
        if id[mid] == val:
            return mid
        else:
            if val < id[mid]:
                high = mid - 1
            else:
                low = mid + 1
    return high + 1


# noinspection PyShadowingBuiltins
def clear(id: list[Any]) -> None:
    """
    Removes all elements from the array.

    :param id: Input array
    """
    id.clear()


# noinspection PyShadowingBuiltins
def concat(id1: list[T], id2: list[T]) -> list[T]:
    """
    Concatenates two arrays into a single array.

    :param id1: First array
    :param id2: Second array
    :return: Array containing the elements of both input arrays
    """
    id1.extend(id2)
    return id1


# noinspection PyShadowingBuiltins
def copy(id: list[T]) -> list[T]:
    """
    Returns a shallow copy of the array.

    :param id: Input array
    :return: Shallow copy of the array
    """
    return list(id)


# noinspection PyShadowingBuiltins
def covariance(id1: list[Number], id2: list[Number], biased: bool = True) -> float:
    """
    Returns the covariance between the elements in the two arrays.

    :param id1: First input array
    :param id2: Second input array
    :param biased: If True, calculates the biased covariance. If False, calculates the unbiased covariance.
    :return: Covariance between the elements in the two arrays
    """
    assert len(id1) == len(id2), "Input arrays must have the same length!"
    mean1 = statistics.mean(id1)
    mean2 = statistics.mean(id2)
    length = len(id1)
    summ = 0.0
    for v1, v2 in zip(id1, id2):
        summ += (v1 - mean1) * (v2 - mean2)
    return summ / ((length - 1) if not biased else length)


# noinspection PyShadowingBuiltins
def every(id: list[Any]) -> bool:
    """
    Returns true if all elements of the id array are true, false otherwise.

    :param id: Input array
    :return: True if all elements of the id array are true, false otherwise
    """
    return all(id)


# noinspection PyShadowingBuiltins
def fill(id: list[T], value: T, index_from: int = 0, index_to: int | NA = NA(int)) -> None:
    """
    Fills the elements in the array with the specified value.

    :param id: Input array
    :param value: Value to fill
    :param index_from: Index to start filling from
    :param index_to: Index to stop filling at
    """
    if isinstance(index_to, NA):
        index_to = len(id)
    id[index_from:index_to] = [value] * (index_to - index_from)


# noinspection PyShadowingBuiltins
def first(id: list[T]) -> T:
    """
    Returns the first element in the array.

    :param id: Input array
    :return: First element in the array
    """
    if len(id) == 0:
        raise RuntimeError("Cannot get first element of an empty array!")
    return id[0]


# noinspection PyShadowingBuiltins
def from_items(*items: T) -> list[T]:
    """
    Returns an array containing the specified elements.
    NOTE: this is `array.from()` in Pine Script, but `from` is a reserved keyword in Python

    :param items: Elements to include in the array
    :return: Array containing the specified elements
    """
    return list(items)


# noinspection PyShadowingBuiltins
def get(id: list[T] | SequenceView[T], index: int) -> T:
    """
    Returns the element at the specified index in the array.

    :param id: Input array
    :param index: Index of the element to return
    :return: Element at the specified index in the array
    """
    return cast(T, id[index])  # noqa


# noinspection PyShadowingBuiltins
def includes(id: list[T], value: T) -> bool:
    """
    Returns true if the array contains the specified value, false otherwise.

    :param id: Input array
    :param value: Value to search for
    :return: True if the array contains the specified value, false otherwise
    """
    return value in id


# noinspection PyShadowingBuiltins
def indexof(id: list[T], value: T) -> int:
    """
    Returns the index of the first occurrence of the specified value in the array.

    :param id: Input array
    :param value: Value to search for
    :return: Index of the first occurrence of the specified value in the array
    """
    try:
        return id.index(value)
    except ValueError:
        return -1


# noinspection PyShadowingBuiltins
def insert(id: list[T], index: int, value: T) -> None:
    """
    Inserts the specified value at the specified index in the array.

    :param id: Input array
    :param index: Index to insert the value at
    :param value: Value to insert
    """
    id.insert(index, value)


# noinspection PyShadowingBuiltins
def join(id: list, separator: str) -> str:
    """
    Concatenates the elements in the array into a single string, separated by the specified separator.

    :param id: Input array
    :param separator: Separator to use
    :return: String containing the concatenated elements
    """
    sa = [str(i) for i in id]  # Ensure all elements are strings
    return separator.join(sa)


# noinspection PyShadowingBuiltins
def last(id: list[T]) -> T:
    """
    Returns the last element in the array.

    :param id: Input array
    :return: Last element in the array
    """
    if len(id) == 0:
        raise RuntimeError("Cannot get last element of an empty array!")
    return id[-1]


# noinspection PyShadowingBuiltins
def lastindexof(id: list[T], value: T) -> int:
    """
    Returns the index of the last occurrence of the specified value in the array.

    :param id: Input array
    :param value: Value to search for
    :return: Index of the last occurrence of the specified value in the array
    """
    try:
        return len(id) - 1 - id[::-1].index(value)
    except ValueError:
        return -1


# noinspection PyShadowingBuiltins
def max(id: list[Number]) -> Number:
    """
    Returns the maximum value in the array.

    :param id: Input array
    :return: Maximum value in the array
    """
    return builtins.max(id)


# noinspection PyShadowingBuiltins
def median(id: list[Number]) -> float:
    """
    Returns the median value of the elements in the array.

    :param id: Input array
    :return: Median value of the elements in the array
    """
    return statistics.median(id)


# noinspection PyShadowingBuiltins
def min(id: list[Number]) -> float:
    """
    Returns the minimum value in the array.

    :param id: Input array
    :return: Minimum value in the array
    """
    return builtins.min(id)


# noinspection PyShadowingBuiltins
def mode(id: list[T]) -> T:
    """
    Returns the most frequently occurring element in the array.

    :param id: Input array
    :return: Most frequently occurring element in the array
    """
    return statistics.mode(id)


# noinspection PyShadowingNames
@overload
def new_box(top_left: ChartPoint, bottom_right: ChartPoint, border_color: _color.Color = _color.blue,
            border_width: int = 1, border_style: LineEnum = _line.style_solid,
            extend: _extend.Extend = _extend.none, xloc: _xloc.XLoc = _xloc.bar_index,
            bgcolor: _color.Color = _color.blue, text: str = "", text_size: _size.Size = _size.auto,
            text_color: _color.Color = _color.black, text_halign: _text.AlignEnum = _text.align_center,
            text_valign: _text.AlignEnum = _text.align_center, text_wrap: _text.WrapEnum = _text.wrap_none,
            text_font_family: _font.FontFamilyEnum = _font.family_default, force_overlay: bool = False,
            text_formatting: _text.FormatEnum = _text.format_none) -> Box:
    """
    Creates a new box object using ChartPoint objects.

    :param top_left: chart.point object that specifies the top-left corner location
    :param bottom_right: chart.point object that specifies the bottom-right corner location
    :param border_color: Color of the four borders
    :param border_width: Width of the four borders, in pixels
    :param border_style: Style of the four borders
    :param extend: When extend.none is used, the horizontal borders start at the left border and end at the right border
    :param xloc: Determines whether the arguments to 'left' and 'right' are a bar index or a time value
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
    :return: A box object that can be used with other array functions
    """
    return _box.new(
        top_left=top_left,
        bottom_right=bottom_right,
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
        force_overlay=force_overlay,
        text_formatting=text_formatting
    )


# noinspection PyShadowingNames
@overload
def new_box(left: int, top: float, right: int, bottom: float, border_color: _color.Color = _color.blue,
            border_width: int = 1, border_style: LineEnum = _line.style_solid,
            extend: _extend.Extend = _extend.none, xloc: _xloc.XLoc = _xloc.bar_index,
            bgcolor: _color.Color = _color.blue, text: str = "", text_size: _size.Size = _size.auto,
            text_color: _color.Color = _color.black, text_halign: _text.AlignEnum = _text.align_center,
            text_valign: _text.AlignEnum = _text.align_center, text_wrap: _text.WrapEnum = _text.wrap_none,
            text_font_family: _font.FontFamilyEnum = _font.family_default, force_overlay: bool = False,
            text_formatting: _text.FormatEnum = _text.format_none) -> Box:
    """
    Creates a new box object using coordinate values.

    :param left: Bar index (if xloc = xloc.bar_index) or bar UNIX time (if xloc = xloc.bar_time) of the left border
    :param top: Price of the top border
    :param right: Bar index (if xloc = xloc.bar_index) or bar UNIX time (if xloc = xloc.bar_time) of the right border
    :param bottom: Price of the bottom border
    :param border_color: Color of the four borders
    :param border_width: Width of the four borders, in pixels
    :param border_style: Style of the four borders
    :param extend: When extend.none is used, the horizontal borders start at the left border and end at the right border
    :param xloc: Determines whether the arguments to 'left' and 'right' are a bar index or a time value
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
    :return: A box object that can be used with other array functions
    """
    return _box.new(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
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
        force_overlay=force_overlay,
        text_formatting=text_formatting
    )


# noinspection PyShadowingNames
@overload
def new_line(first_point: ChartPoint, second_point: ChartPoint, xloc: _xloc.XLoc = _xloc.bar_index,
             extend: _extend.Extend = _extend.none, color: _color.Color = _color.blue,
             style: LineEnum = _line.style_solid, width: int = 1, force_overlay: bool = False) -> Line:
    """
    Creates a new line object using ChartPoint objects.

    :param first_point: chart.point object that specifies the line's starting coordinate
    :param second_point: chart.point object that specifies the line's ending coordinate
    :param xloc: See description of x1 argument. Possible values: xloc.bar_index and xloc.bar_time
    :param extend: If extend=extend.none, draws segment starting at point (x1, y1) and ending at point (x2, y2)
    :param color: Line color
    :param style: Line style. Possible values: line.style_solid, line.style_dotted, line.style_dashed,
                  line.style_arrow_left, line.style_arrow_right, line.style_arrow_both
    :param width: Line width in pixels
    :param force_overlay: If true, the drawing will display on the main chart pane
    :return: A line object that can be used with other array functions
    """
    return _line.new(
        first_point=first_point,
        second_point=second_point,
        xloc=xloc,
        extend=extend,
        color=color,
        style=style,
        width=width,
        force_overlay=force_overlay
    )


# noinspection PyShadowingNames
@overload
def new_line(x1: int, y1: float, x2: int, y2: float, xloc: _xloc.XLoc = _xloc.bar_index,
             extend: _extend.Extend = _extend.none, color: _color.Color = _color.blue,
             style: LineEnum = _line.style_solid, width: int = 1, force_overlay: bool = False) -> Line:
    """
    Creates a new line object using coordinate values.

    :param x1: Bar index (if xloc = xloc.bar_index) or bar UNIX time (if xloc = xloc.bar_time) of the first point
    :param y1: Price of the first point of the line
    :param x2: Bar index (if xloc = xloc.bar_index) or bar UNIX time (if xloc = xloc.bar_time) of the second point
    :param y2: Price of the second point of the line
    :param xloc: See description of x1 argument. Possible values: xloc.bar_index and xloc.bar_time
    :param extend: If extend=extend.none, draws segment starting at point (x1, y1) and ending at point (x2, y2)
    :param color: Line color
    :param style: Line style. Possible values: line.style_solid, line.style_dotted,
                  line.style_dashed, line.style_arrow_left, line.style_arrow_right, line.style_arrow_both
    :param width: Line width in pixels
    :param force_overlay: If true, the drawing will display on the main chart pane
    :return: A line object that can be used with other array functions
    """
    return _line.new(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        xloc=xloc,
        extend=extend,
        color=color,
        style=style,
        width=width,
        force_overlay=force_overlay
    )


# noinspection PyShadowingNames
@overload
def new_label(point: ChartPoint, text: str = "", xloc: _xloc.XLoc = _xloc.bar_index,
              yloc: _yloc.YLoc = _yloc.price, color: _color.Color = _color.blue,
              style: LabelStyleEnum = _label.style_label_down, textcolor: _color.Color = _color.white,
              size: _size.Size = _size.normal, textalign: _text.AlignEnum = _text.align_center,
              tooltip: str = "", text_font_family: _font.FontFamilyEnum = _font.family_default,
              force_overlay: bool = False, text_formatting: _text.FormatEnum = _text.format_none) -> Label:
    """
    Creates a new label object using a ChartPoint.

    :param point: chart.point object that specifies the label position
    :param text: Label text
    :param xloc: See description of x argument. Possible values: xloc.bar_index and xloc.bar_time
    :param yloc: Possible values are yloc.price, yloc.abovebar, yloc.belowbar
    :param color: Color of the label border and arrow
    :param style: Label style
    :param textcolor: Color of the label text
    :param size: Size of the label text
    :param textalign: Alignment of the label text
    :param tooltip: Tooltip text displayed when hovering over the label
    :param text_font_family: Font family of the label text
    :param force_overlay: If true, the drawing will display on the main chart pane
    :param text_formatting: Text formatting options
    :return: A label object that can be used with other array functions
    """
    return _label.new(
        point=point,
        text=text,
        xloc=xloc,
        yloc=yloc,
        color=color,
        style=style,
        textcolor=textcolor,
        size=size,
        textalign=textalign,
        tooltip=tooltip,
        text_font_family=text_font_family,
        force_overlay=force_overlay,
        text_formatting=text_formatting
    )


# noinspection PyShadowingNames
@overload
def new_label(x: int, y: int | float, text: str = "", xloc: _xloc.XLoc = _xloc.bar_index,
              yloc: _yloc.YLoc = _yloc.price, color: _color.Color = _color.blue,
              style: LabelStyleEnum = _label.style_label_down, textcolor: _color.Color = _color.white,
              size: _size.Size = _size.normal, textalign: _text.AlignEnum = _text.align_center,
              tooltip: str = "", text_font_family: _font.FontFamilyEnum = _font.family_default,
              force_overlay: bool = False, text_formatting: _text.FormatEnum = _text.format_none) -> Label:
    """
    Creates a new label object using coordinate values.

    :param x: Bar index (if xloc = xloc.bar_index) or bar UNIX time (if xloc = xloc.bar_time) of the label position
    :param y: Price of the label position
    :param text: Label text
    :param xloc: See description of x argument. Possible values: xloc.bar_index and xloc.bar_time
    :param yloc: Possible values are yloc.price, yloc.abovebar, yloc.belowbar
    :param color: Color of the label border and arrow
    :param style: Label style
    :param textcolor: Color of the label text
    :param size: Size of the label text
    :param textalign: Alignment of the label text
    :param tooltip: Tooltip text displayed when hovering over the label
    :param text_font_family: Font family of the label text
    :param force_overlay: If true, the drawing will display on the main chart pane
    :param text_formatting: Text formatting options
    :return: A label object that can be used with other array functions
    """
    return _label.new(
        x=x,
        y=y,
        text=text,
        xloc=xloc,
        yloc=yloc,
        color=color,
        style=style,
        textcolor=textcolor,
        size=size,
        textalign=textalign,
        tooltip=tooltip,
        text_font_family=text_font_family,
        force_overlay=force_overlay,
        text_formatting=text_formatting
    )


# noinspection PyShadowingNames
def new_linefill(line1: Line, line2: Line, color: _color.Color) -> LineFill | NA[LineFill]:
    """
    Creates a new linefill object and displays it on the chart, filling the space between line1 and
    line2 with the color specified in color.

    :param line1: First line object
    :param line2: Second line object
    :param color: The color used to fill the space between the lines
    :return: The ID of a linefill object that can be used with other array functions, or NA if any line is NA
    """
    return _linefill.new(line1=line1, line2=line2, color=color)


# noinspection PyShadowingNames
def new(size: int = 0, initial_value: T | NA[T] = NA(T)) -> list[T | NA[T]]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    assert size >= 0, "Size must be >=0!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new_bool(size: int = 0, initial_value: bool | NA = NA(bool)) -> list[PyneBool]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    assert size >= 0, "Size must be >=0!"
    assert isinstance(initial_value, (bool, NA)), "Initial value must be bool!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new_color(size: int = 0, initial_value: Color | NA = NA(Color)) -> list[Color | NA[Color]]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    assert size >= 0, "Size must be >=0!"
    assert isinstance(initial_value, (Color, NA)), "Initial value must be Color!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new_float(size: int = 0, initial_value: float | int | NA = NA(float)) -> list[PyneFloat]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    assert size >= 0, "Size must be >=0!"
    assert isinstance(initial_value, (float, int, NA)), "Initial value must be float!"
    if isinstance(initial_value, int):
        initial_value = float(initial_value)
    return [initial_value] * size


# noinspection PyShadowingNames
def new_int(size: int = 0, initial_value: int | NA = NA(int)) -> list[PyneInt]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    assert size >= 0, "Size must be >=0!"
    assert isinstance(initial_value, (int, NA)), "Initial value must be int!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new_string(size: int = 0, initial_value: str | NA = NA(str)) -> list[PyneStr]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    assert size >= 0, "Size must be >=0!"
    assert isinstance(initial_value, (str, NA)), "Initial value must be str!"
    return [initial_value] * size


# noinspection PyShadowingBuiltins,PyShadowingNames
def percentile_linear_interpolation(id: list[float], percentage: float) -> float:
    """
    Calculate the percentile value using linear interpolation, following TradingView's logic.

    :param id: List of numeric values
    :param percentage: Percentile (0-100, not 0-1)
    :return: Interpolated value at the given percentile
    :raises ValueError: If arr is empty or percentage is not in [0, 100]
    """
    if not id:
        raise ValueError("Input array is empty")
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")

    sorted_arr = sorted(id)
    n = len(sorted_arr)

    # Calculate position in 1-indexed system
    pos = n * percentage / 100.0

    # Special cases for extreme percentiles
    if pos < 1:
        return sorted_arr[0]
    if pos >= n:
        return sorted_arr[-1]

    # If pos is an integer, average the two adjacent values
    if pos.is_integer():
        pos_int = int(pos)
        return (sorted_arr[pos_int - 1] + sorted_arr[pos_int]) / 2.0

    # For non-integer positions, perform linear interpolation
    lower_index = int(pos)  # floor of pos
    frac = pos - lower_index
    return sorted_arr[lower_index] + frac * (sorted_arr[lower_index + 1] - sorted_arr[lower_index])


# noinspection PyShadowingBuiltins,PyShadowingNames
def percentile_nearest_rank(id: list[float], percentage: float) -> float:
    """
    Calculate the nearest rank percentile without interpolation.

    :param id: List of numeric values
    :param percentage: Percentile (0-100)
    :return: The value at the nearest rank for the specified percentile
    :raises ValueError: If arr is empty or percentage is not between 0 and 100
    """
    if not id:
        raise ValueError("Input array is empty")
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")

    sorted_arr = sorted(id)
    n = len(sorted_arr)
    if percentage == 0:
        return sorted_arr[0]

    # Calculate the rank using the ceiling function as per the nearest rank method
    rank = math.ceil(percentage * n / 100)
    # Clamp rank to be within the valid range [1, n]
    rank = builtins.max(1, builtins.min(rank, n))
    # Adjust for 0-indexed array: return the (rank-1)th element
    return sorted_arr[rank - 1]


# noinspection PyShadowingBuiltins,PyShadowingNames
def percentrank(id: list[Number], index: int) -> float:
    """
    Returns the percentile rank of the element at the specified index.
    The percentile rank is the percentage of values less than or equal to the value at index.

    :param id: Input array
    :param index: Index of the element to calculate rank for
    :return: Percentile rank (0-100)
    :raises ValueError: If input array is empty or index is out of range
    """
    if not id:
        raise ValueError("Input array is empty")

    if not 0 <= index < len(id):
        raise ValueError("Index out of range")

    # Get value at index
    value = id[index]

    # Count elements less than or equal
    count = builtins.sum(1 for x in id if x <= value)

    # Calculate percentage
    return (count - 1) * 100 / (len(id) - 1)


# noinspection PyShadowingBuiltins
def pop(id: list[T]) -> T:
    """
    Removes the last element from the array and returns it.

    :param id: Input array
    :return: Last element from the array
    """
    return id.pop()


# noinspection PyShadowingBuiltins
def push(id: list[T], value: T) -> None:
    """
    Appends the specified value to the end of the array.

    :param id: Input array
    :param value: Value to append
    """
    id.append(value)


# noinspection PyShadowingBuiltins
def range(id: list[Number]) -> Number:
    """
    Returns the range of the elements in the array.

    :param id: Input array
    :return: Range of the elements in the array
    """
    return cast(Number, max(id) - min(id))  # noqa


# noinspection PyShadowingBuiltins
def remove(id: list[T], index: int) -> T:
    """
    Removes the element at the specified index from the array.

    :param id: Input array
    :param index: Index of the element to remove
    :return: The removed element
    """
    return id.pop(index)


# noinspection PyShadowingBuiltins
def reverse(id: list[T]) -> None:
    """
    Reverses the order of the elements in the array.

    :param id: Input array
    """
    id.reverse()


# noinspection PyShadowingBuiltins
def set(id: list[T] | SequenceView[T], index: int, value: T) -> None:
    """
    Sets the value of the element at the specified index in the array.

    :param id: Input array
    :param index: Index of the element to set
    :param value: Value to set
    """
    id[index] = value


# noinspection PyShadowingBuiltins
def shift(id: list[T]) -> T:
    """
    Removes the first element from the array and returns it.

    :param id: Input array
    :return: First element from the array
    """
    return id.pop(0)


# noinspection PyShadowingBuiltins
def size(id: list[Any] | SequenceView[Any]) -> int:
    """
    Returns the number of elements in the array.

    :param id: Input array
    :return: Number of elements in the array
    """
    return len(id)


# noinspection PyShadowingBuiltins
def slice(id: list[T], index_from: int, index_to: int) -> SequenceView[T]:
    """
    The function creates a slice from an existing array. If an object from the slice changes, the
    changes are applied to both the new and the original arrays.

    :param id: Input array
    :param index_from: Index to start the sub-array from
    :param index_to: Index to end the sub-array at
    :return: Slice view of the original array
    """
    return SequenceView(id)[int(index_from):int(index_to)]  # type: ignore


# noinspection PyShadowingBuiltins
def some(id: list[Any]) -> bool:
    """
    Returns true if at least one element of the id array is true, false otherwise.

    :param id: Input array
    :return: True if at least one element of the id array is true, false otherwise
    """
    return any(id)


# noinspection PyShadowingBuiltins
def sort(id: list[int | float | str], order: _order.Order = _order.ascending) -> None:
    """
    Sorts the elements in the array in ascending or descending order.

    :param id: Input array
    :param order: Order to sort the elements in
    """
    id.sort(reverse=order == _order.descending)


# noinspection PyShadowingBuiltins
def sort_indices(id: list[T], order: _order.Order = _order.ascending) -> list[int]:
    """
    Returns an array of indices which, when used to index the original array, will access its elements
    in their sorted order. It does not modify the original array.

    :param id: Input array
    :param order: Order to sort the elements in
    :return: Array of indices to access the elements in their sorted order
    """
    indices: list[int] = sorted(builtins.range(len(id)), key=id.__getitem__)  # type: ignore
    if order == _order.descending:
        indices.reverse()
    return indices


# noinspection PyShadowingBuiltins,PyShadowingNames
def standardize(id: list[float | int]) -> list[float | int]:
    """
    Standardizes the input array in a Pine Script-like manner:
      1) Uses a left-to-right summation for the mean (population mean).
      2) Uses a second pass for summing squared differences (population variance).
      3) Computes the population standard deviation (divisor = N).
      4) Returns the z-score for each element.
         - If all input elements are integers, it applies thresholding:
             z < -1 -> -1,
             z > 1  -> 1,
             otherwise 0
         - If any element is float, the result is the continuous z-score value.
    This version is bit-by-bit compatible with Pine Script's `standardize()` function.

    :param id: A list of numeric values (int or float).
    :return: A list containing the standardized values.
    """
    n = len(id)
    if n == 0:
        # You can decide how you want to handle the empty list.
        return []

    mean = statistics.mean(id)
    stdev = math.sqrt(statistics.mean([(v - mean) ** 2 for v in id]))
    z_scores = [(v - mean) / stdev for v in id]

    # If all values are integers, apply the thresholding to get -1, 0, or 1.
    if all(isinstance(v, int) for v in id):
        # Pine Script-style integer thresholding
        return [
            -1 if z < -1 else
            1 if z > 1 else
            0
            for z in z_scores
        ]

    # Otherwise, return the continuous z-score values.
    return z_scores


# noinspection PyShadowingBuiltins
def stdev(id: list[Number], biased: bool = True) -> float:
    """
    Returns the standard deviation of the elements in the array.

    :param id: Input array
    :param biased: If True, calculates the biased standard deviation. If False, calculates the
                   unbiased standard deviation.
    :return: Standard deviation of the elements in the array
    """
    a = cast(list[Number], [i for i in id if not isinstance(i, NA)])
    if len(a) < 2:
        return 0.0
    if not biased:
        return statistics.stdev(a)
    mean = statistics.mean(a)
    return math.sqrt(statistics.mean([(v - mean) ** 2 for v in a]))


# noinspection PyShadowingBuiltins
def sum(id: list[float | int]) -> float | int:
    """
    Returns the sum of the elements in the array.

    :param id: Input array
    :return: Sum of the elements in the array
    """
    return builtins.sum(id)


# noinspection PyShadowingBuiltins
def unshift(id: list[T], value: T) -> None:
    """
    Prepends the specified value to the beginning of the array.

    :param id: Input array
    :param value: Value to prepend
    """
    id.insert(0, value)


# noinspection PyShadowingBuiltins
def variance(id: list[Number], biased: bool = True) -> float:
    """
    Returns the variance of the elements in the array.

    :param id: Input array
    :param biased: If True, calculates the biased variance. If False, calculates the unbiased variance.
    :return: Variance of the elements in the array
    """
    if not biased:
        return statistics.variance(id)

    length = len(id)
    mean = statistics.mean(id)
    summ = 0.0
    for v in id:
        summ += (v - mean) ** 2
    return summ / length
