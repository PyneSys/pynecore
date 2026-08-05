from dataclasses import dataclass

from ..lib import color as _color
from .line import Line
from .base import Drawing


@dataclass(slots=True, eq=False)
class LineFill(Drawing):
    line1: Line  # First line object
    line2: Line  # Second line object
    color: _color.Color  # Fill color

    vid: int = -1
