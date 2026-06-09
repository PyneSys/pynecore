from typing import Any

from ..core.callable_module import CallableModule
from ..types.plot import Plot
from ..types.plot import PlotEnum


class PlotModule(CallableModule):
    style_area: PlotEnum
    style_areabr: PlotEnum
    style_circles: PlotEnum
    style_columns: PlotEnum
    style_cross: PlotEnum
    style_histogram: PlotEnum
    style_line: PlotEnum
    style_linebr: PlotEnum
    style_stepline: PlotEnum
    style_steplinebr: PlotEnum
    style_stepline_diamond: PlotEnum
    linestyle_solid: PlotEnum
    linestyle_dashed: PlotEnum
    linestyle_dotted: PlotEnum

    def __call__(self, series: Any, title: str | None = None, *args, **kwargs) -> Plot: ...

    def new(self): ...


plot: PlotModule = PlotModule(__name__)
