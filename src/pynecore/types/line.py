from dataclasses import dataclass
from typing import Optional

from ..lib import color as _color, extend as _extend, xloc as _xloc
from .base import StrLiteral
from .na import NA


class LineEnum(StrLiteral):
    ...


@dataclass(slots=True)
class Line:
    # Required parameters - coordinates (``na`` when set from a chart.point that lacks the
    # corresponding coordinate for the active xloc)
    x1: int | NA  # Bar index or UNIX time
    y1: float | NA  # Price of the first point
    x2: int | NA  # Bar index or UNIX time
    y2: float | NA  # Price of the second point

    # Optional parameters with defaults
    xloc: Optional[_xloc.XLoc] = None
    extend: Optional[_extend.Extend] = None
    color: Optional[_color.Color] = None
    style: Optional[LineEnum] = None
    width: int = 1

    force_overlay: bool = False
