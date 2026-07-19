from ..types.base import next_vid
from ..types.linefill import LineFill
from ..types.line import Line
from ..types.na import NA
from ..lib import color as _color

_registry: list[LineFill] = []


# noinspection PyShadowingBuiltins
def new(line1: Line, line2: Line, color: _color.Color) -> LineFill | NA[LineFill]:
    """
    Creates a new linefill object and displays it on the chart, filling the space between line1 and
    line2 with the color specified in color.

    :param line1: First line object
    :param line2: Second line object
    :param color: The color used to fill the space between the lines
    :return: The ID of a linefill object that can be passed to other linefill.*() functions
    """
    if isinstance(line1, NA) or isinstance(line2, NA):
        return NA(LineFill)

    # Pine allows only one linefill per pair of lines: a successive ``linefill.new()``
    # call for the same pair replaces the previous fill.
    for existing in _registry:
        if ((existing.line1 is line1 and existing.line2 is line2)
                or (existing.line1 is line2 and existing.line2 is line1)):
            _registry.remove(existing)
            break

    linefill_obj = LineFill(
        line1=line1,
        line2=line2,
        color=color
    )
    linefill_obj.vid = next_vid()
    _registry.append(linefill_obj)
    return linefill_obj


# noinspection PyShadowingBuiltins
def delete(id: LineFill) -> None:
    """
    Deletes the specified linefill object. If it has already been deleted, does nothing.

    :param id: A linefill object
    """
    if isinstance(id, NA):
        return
    try:
        _registry.remove(id)
    except ValueError:
        pass


# noinspection PyShadowingBuiltins
def get_line1(id: LineFill) -> Line | NA:
    """
    Returns the ID of the first line used in the id linefill.

    :param id: A linefill object
    :return: First line object
    """
    if isinstance(id, NA):
        return NA(Line)
    return id.line1


# noinspection PyShadowingBuiltins
def get_line2(id: LineFill) -> Line | NA:
    """
    Returns the ID of the second line used in the id linefill.

    :param id: A linefill object
    :return: Second line object
    """
    if isinstance(id, NA):
        return NA(Line)
    return id.line2


# noinspection PyShadowingBuiltins
def set_color(id: LineFill, color: _color.Color) -> None:
    """
    The function sets the color of the linefill object passed to it.

    :param id: A linefill object
    :param color: The color of the linefill object
    """
    if isinstance(id, NA):
        return
    id.color = color
