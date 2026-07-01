from ..types.na import NA

from ..types.color import Color
from ..types.label import Label
from ..types.table import Table
from ..types.box import Box
from ..types.line import Line
from ..types.linefill import LineFill


def cast_color(x: Color | NA) -> Color | NA[Color]:
    """
    Casts `na` to Color
    :param x: The value to convert
    :return: The casted value
    """
    return NA(Color) if isinstance(x, NA) else x


def cast_label(x: Label | NA) -> Label | NA[Label]:
    """
    Casts `na` to Label

    :param x: The value to convert
    :return: The casted value
    """
    return NA(Label) if isinstance(x, NA) else x


def cast_table(x: Table | NA) -> Table | NA[Table]:
    """
    Casts `na` to Table

    :param x: The value to convert
    :return: The casted value
    """
    return NA(Table) if isinstance(x, NA) else x


def cast_bool(x: bool | int | float | NA) -> bool:
    """
    Converts the x value to a bool value

    :param x: The value to convert
    :return: The casted value
    """
    if isinstance(x, NA):
        return False
    return not not x


def cast_box(x: Box | NA) -> Box | NA[Box]:
    """
    Casts `na` to Box

    :param x: The value to convert
    :return: The casted value
    """
    return NA(Box) if isinstance(x, NA) else x


def cast_int(x: int | float | NA) -> int | NA[int]:
    """
    Casts na or truncates float value to int

    :param x: The value to convert
    :return: The casted value
    """
    if isinstance(x, NA):
        return NA(int)
    return int(x)


def cast_line(x: Line | NA) -> Line | NA[Line]:
    """
    Casts `na` to Line

    :param x: The value to convert
    :return: The casted value
    """
    return NA(Line) if isinstance(x, NA) else x


def cast_float(x: float | int | NA) -> float | NA[float]:
    """
    Casts `na` to float

    :param x: The value to convert
    :return: The casted value
    """
    if isinstance(x, NA):
        return NA(float)
    return float(x)


def cast_string(x: str | NA) -> str | NA[str]:
    """
    Casts `na` to string

    :param x: The value to convert
    :return: The casted value
    """
    return NA(str) if isinstance(x, NA) else x


def cast_linefill(x: LineFill | NA) -> LineFill | NA[LineFill]:
    """
    Casts `na` to LineFill

    :param x: The value to convert
    :return: The casted value
    """
    return NA(LineFill) if isinstance(x, NA) else x
