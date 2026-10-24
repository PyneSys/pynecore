from ..types.na import NA, na_float, na_int
from ..types.pine_types import PyneInt, pine_int
from ..types.color import Color
from ..types.label import Label
from ..types.table import Table
from ..types.box import Box
from ..types.line import Line
from ..types.linefill import LineFill

# Pine's ``bool()`` cast is the conversion a number passed to a lib function's
# ``bool`` parameter gets, so both share the one definition
from .safe_convert import safe_bool as cast_bool


def cast_color(x: Color | NA) -> Color:
    """
    Casts `na` to Color
    :param x: The value to convert
    :return: The casted value
    """
    return NA(Color) if isinstance(x, NA) else x


def cast_label(x: Label | NA) -> Label:
    """
    Casts `na` to Label

    :param x: The value to convert
    :return: The casted value
    """
    return NA(Label) if isinstance(x, NA) else x


def cast_table(x: Table | NA) -> Table:
    """
    Casts `na` to Table

    :param x: The value to convert
    :return: The casted value
    """
    return NA(Table) if isinstance(x, NA) else x


def cast_box(x: Box | NA) -> Box:
    """
    Casts `na` to Box

    :param x: The value to convert
    :return: The casted value
    """
    return NA(Box) if isinstance(x, NA) else x


def cast_int(x: int | float | NA) -> PyneInt:
    """
    Casts na or truncates float value to int

    :param x: The value to convert
    :return: The casted value
    """
    if not (x == x):  # NA object or native nan
        return na_int
    return pine_int(int(x))


def cast_line(x: Line | NA) -> Line:
    """
    Casts `na` to Line

    :param x: The value to convert
    :return: The casted value
    """
    return NA(Line) if isinstance(x, NA) else x


def cast_float(x: float | int | NA) -> float:
    """
    Casts `na` to float

    :param x: The value to convert
    :return: The casted value
    """
    if not (x == x):  # NA object or native nan
        return na_float
    return float(x)


def cast_string(x: str | NA) -> str:
    """
    Casts `na` to string

    :param x: The value to convert
    :return: The casted value
    """
    return NA(str) if isinstance(x, NA) else x


def cast_linefill(x: LineFill | NA) -> LineFill:
    """
    Casts `na` to LineFill

    :param x: The value to convert
    :return: The casted value
    """
    return NA(LineFill) if isinstance(x, NA) else x
