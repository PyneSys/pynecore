from ..core.module_property import module_property
from ..types.base import next_vid
from ..types.chart import ChartPoint
from ..types.polyline import Polyline
from ..types.na import NA
from ..lib import xloc as _xloc, color as _color
from ..types.line import LineEnum
from .. import lib

_registry: dict[Polyline, None] = {}

# Line style constants (same as in line.py)
style_arrow_both = LineEnum()
style_arrow_left = LineEnum()
style_arrow_right = LineEnum()
style_dashed = LineEnum()
style_dotted = LineEnum()
style_solid = LineEnum()


# noinspection PyProtectedMember
def new(points: list[ChartPoint], curved: bool = False, closed: bool = False,
        xloc: _xloc.XLoc = _xloc.bar_index, line_color: _color.Color = _color.blue,
        fill_color: _color.Color | None = None, line_style: LineEnum = style_solid,
        line_width: int = 1, force_overlay: bool = False) -> Polyline | NA[Polyline]:
    """
    Creates a new polyline instance and displays it on the chart, sequentially connecting all of the
    points in the points array with line segments.

    :param points: An array of chart.point objects for the drawing to sequentially connect
    :param curved: If true, the drawing will connect all points using curved line segments
    :param closed: If true, the drawing will connect the first point to the last point, resulting in a closed polyline
    :param xloc: Determines the field of the chart.point objects that the polyline will use for its x-coordinates
    :param line_color: The color of the line segments
    :param fill_color: The fill color of the polyline
    :param line_style: The style of the polyline
    :param line_width: The width of the line segments, expressed in pixels
    :param force_overlay: If true, the drawing will display on the main chart pane
    :return: The ID of a new polyline object
    """
    if not points or len(points) == 0:
        return NA(Polyline)

    # Check if any points are NA
    for point in points:
        if isinstance(point, NA):
            return NA(Polyline)

    polyline_obj = Polyline(
        points=points,
        curved=curved,
        closed=closed,
        xloc=xloc,
        line_color=line_color,
        fill_color=fill_color,
        line_style=line_style,
        line_width=line_width,
        force_overlay=force_overlay
    )
    polyline_obj.vid = next_vid()
    _registry[polyline_obj] = None
    # Enforce Pine's max_polylines_count cap: drop the oldest polyline (FIFO) past the limit.
    # A security child never sets ``lib._script``; fall back to TV's hard maximum
    # (500) there, otherwise the registry grows without bound (the child re-runs
    # main() for every bar of its own series, accumulating every drawing ever made).
    if len(_registry) > (lib._script.max_polylines_count if lib._script is not None else 500):
        del _registry[next(iter(_registry))]
    return polyline_obj


# noinspection PyShadowingBuiltins
def delete(id: Polyline) -> None:
    """
    Deletes the specified polyline object. It has no effect if the id doesn't exist.

    :param id: The polyline ID to delete
    """
    if isinstance(id, NA):
        return
    _registry.pop(id, None)


# noinspection PyShadowingBuiltins
@module_property
def all() -> list[Polyline]:
    """
    Returns an array containing all current polyline instances drawn by the script.

    :return: Array of all polyline objects
    """
    return list(_registry)
