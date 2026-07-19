from dataclasses import dataclass
from typing import Optional

from ..lib import color as _color, xloc as _xloc, yloc as _yloc, size as _size, text as _text, font as _font
from .base import StrLiteral
from .na import NA


class LabelStyleEnum(StrLiteral):
    ...


@dataclass(slots=True, eq=False)
class Label:
    # Required parameters (``na`` when set from a chart.point that lacks the corresponding
    # coordinate for the active xloc)
    x: int | NA  # Bar index or UNIX time
    y: int | float | NA  # Price of the label position
    text: str = ""  # Label text
    
    # Optional parameters with defaults
    xloc: Optional[_xloc.XLoc] = None
    yloc: Optional[_yloc.YLoc] = None
    color: Optional[_color.Color] = None
    style: Optional[LabelStyleEnum] = None
    textcolor: Optional[_color.Color] = None
    size: Optional[_size.Size] = None
    textalign: Optional[_text.AlignEnum] = None
    tooltip: str = ""
    text_font_family: Optional[_font.FontFamilyEnum] = None
    force_overlay: bool = False
    text_formatting: Optional[_text.FormatEnum] = None

    vid: int = -1
