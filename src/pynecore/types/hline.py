from dataclasses import dataclass
from typing import Optional, Union

from .base import IntEnum
from ..lib import color as _color, display as _display


class HLineEnum(IntEnum):
    __slots__ = ()


@dataclass(slots=True)
class HLine:
    # Required parameter
    price: Union[int, float]  # Price value at which the object will be rendered

    # Optional parameters
    title: Optional[str] = None  # Title of the object
    color: Optional[_color.Color] = None  # Color of the rendered line
    linestyle: Optional[HLineEnum] = None  # Style of the rendered line
    linewidth: int = 1  # Width of the rendered line
    editable: bool = True  # If true then hline style will be editable in Format dialog
    display: Optional[_display.Display] = None  # Controls where the hline is displayed
