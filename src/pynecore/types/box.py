from typing import Optional
from dataclasses import dataclass

from ..lib import (color as _color, extend as _extend, xloc as _xloc, size as _size, line as _line,
                   text as _text, font as _font)


@dataclass(slots=True)
class Box:
    # Required parameters - coordinates
    left: int  # Bar index or UNIX time
    top: float  # Price of the top border
    right: int  # Bar index or UNIX time
    bottom: float  # Price of the bottom border

    # Optional parameters with defaults
    border_color: Optional[_color.Color] = None
    border_width: int = 1
    border_style: Optional[_line.LineEnum] = None
    extend: Optional[_extend.Extend] = None
    xloc: Optional[_xloc.XLoc] = None
    bgcolor: Optional[_color.Color] = None
    text: str = ""
    text_size: Optional[_size.Size] = None
    text_color: Optional[_color.Color] = None
    text_halign: Optional[_text.AlignEnum] = None
    text_valign: Optional[_text.AlignEnum] = None
    text_wrap: Optional[_text.WrapEnum] = None
    text_font_family: Optional[_font.FontFamilyEnum] = None
    text_formatting: Optional[_text.FormatEnum] = None

    force_overlay: bool = False
