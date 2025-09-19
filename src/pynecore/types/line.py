from dataclasses import dataclass
from typing import Optional

from ..lib import color as _color, extend as _extend, xloc as _xloc
from .base import IntEnum


class LineEnum(IntEnum):
    __slots__ = ()


@dataclass(slots=True)
class Line:
    # Required parameters - coordinates
    x1: int  # Bar index or UNIX time
    y1: float  # Price of the first point
    x2: int  # Bar index or UNIX time
    y2: float  # Price of the second point

    # Optional parameters with defaults
    xloc: Optional[_xloc.XLoc] = None
    extend: Optional[_extend.Extend] = None
    color: Optional[_color.Color] = None
    style: Optional[LineEnum] = None
    width: int = 1

    force_overlay: bool = False
