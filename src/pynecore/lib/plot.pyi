from typing import Any

from ..types.plot import Plot
from ..types.plot import PlotEnum


# IDE-facing view of the function-and-namespace module: user code reads the
# constants and calls the bare name; the AST transformer resolves both at runtime.
class PlotModule:
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


plot: PlotModule
