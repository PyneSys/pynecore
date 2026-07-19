from dataclasses import dataclass
from typing import Any

from .color import Color


@dataclass(slots=True)
class PlotMeta:
    """
    Static (registered-once) metadata for a plot-family output.

    Covers every plot family (``plot``, ``plotshape``, ``plotchar``, ``plotarrow``,
    ``plotcandle``, ``plotbar``, ``bgcolor``, ``barcolor``, ``hline``, ``fill``); the
    serializer drops ``None`` fields and applies kind-specific defaults. Per-bar values
    (the series data and dynamic color channels) live elsewhere; this object only holds
    what is fixed for the whole run.
    """
    id: str
    kind: str  # 'plot'|'shape'|'char'|'arrow'|'candle'|'bar'|'bgcolor'|'barcolor'|'hline'|'fill'

    title: str | None = None
    color: Color | None = None
    linewidth: int = 1
    style: Any = None
    trackprice: bool = False
    histbase: float = 0.0
    offset: int = 0
    join: bool = False
    editable: bool = True
    show_last: int | None = None
    display: Any = None
    format: str | None = None
    precision: int | None = None
    force_overlay: bool = False

    char: str | None = None
    location: Any = None
    size: Any = None
    text: str | None = None
    textcolor: Color | None = None

    colorup: Color | None = None
    colordown: Color | None = None
    minheight: int | None = None
    maxheight: int | None = None

    wickcolor: Color | None = None
    bordercolor: Color | None = None

    price: float | None = None
    linestyle: Any = None

    plot1: str | None = None
    plot2: str | None = None
    hline1: str | None = None
    hline2: str | None = None

    fillgaps: bool = False

    dynamic: bool = False
