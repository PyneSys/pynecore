from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..lib import color as _color, xloc as _xloc
from ..types.line import LineEnum

if TYPE_CHECKING:
    from .chart import ChartPoint


@dataclass(slots=True, eq=False)
class Polyline:
    # Array of chart.point objects for the drawing to sequentially connect
    points: list["ChartPoint"]
    
    # If true, the drawing will connect all points using curved line segments
    curved: bool = False
    
    # If true, the drawing will connect the first point to the last point, resulting in a closed polyline
    closed: bool = False
    
    # Determines the field of the chart.point objects that the polyline will use for its x-coordinates
    xloc: Optional[_xloc.XLoc] = None
    
    # The color of the line segments
    line_color: Optional[_color.Color] = None
    
    # The fill color of the polyline
    fill_color: Optional[_color.Color] = None
    
    # The style of the polyline
    line_style: Optional[LineEnum] = None
    
    # The width of the line segments, expressed in pixels
    line_width: int = 1
    
    # If true, the drawing will display on the main chart pane
    force_overlay: bool = False

    vid: int = -1