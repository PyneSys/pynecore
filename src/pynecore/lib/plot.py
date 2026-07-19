from typing import Any
import sys

from ..types.plot import PlotEnum, Plot
from ..types.plot_meta import PlotMeta


#
# Constants
#

style_area = PlotEnum()
style_areabr = PlotEnum()
style_circles = PlotEnum()
style_columns = PlotEnum()
style_cross = PlotEnum()
style_histogram = PlotEnum()
style_line = PlotEnum()
style_linebr = PlotEnum()
style_stepline = PlotEnum()
style_steplinebr = PlotEnum()
style_stepline_diamond = PlotEnum()

linestyle_solid = PlotEnum()
linestyle_dashed = PlotEnum()
linestyle_dotted = PlotEnum()


#
# Module function
#

# noinspection PyProtectedMember,PyShadowingBuiltins
def plot(series: Any, title: str | None = None, color: Any = None, linewidth: int = 1,
         style: Any = None, trackprice: bool = False, histbase: float = 0.0, offset: int = 0,
         join: bool = False, editable: bool = True, show_last: int | None = None,
         display: Any = None, format: str | None = None, precision: int | None = None,
         force_overlay: bool = False, *_, **__):
    """
    Plot a series on the chart.

    :param series: The value to plot on every bar
    :param title: The title of the plot; if several plots share a title a number is appended
    :param color: Plot color; when it varies per bar it is recorded as a dynamic channel
    :param linewidth: Width of the plotted line in pixels
    :param style: Plot style (``plot.style_*``); ``None`` means Pine's ``style_line``
    :param trackprice: If true, a horizontal price line is shown at the last value
    :param histbase: Reference value for ``style_histogram``, ``style_columns`` and ``style_area``
    :param offset: Horizontal shift of the plot, in bars
    :param join: If true, ``style_circles`` / ``style_cross`` points are joined with lines
    :param editable: If true, the plot style is editable in the Format dialog
    :param show_last: If set, only the last ``show_last`` bars are plotted
    :param display: Controls where the plot is displayed
    :param format: Formatting of the plotted values (``format.price``, ``format.volume`` etc.)
    :param precision: Number of decimal places for the plotted values
    :param force_overlay: If true, the plot displays on the main chart pane
    :return: A Plot object, used to reference the plot in other functions
    """
    from .. import lib
    if lib._lib_semaphore:
        return Plot('')

    if lib.bar_index == 0:  # Only check if it is the first bar for performance reasons
        # Check if it is called from the main function
        if sys._getframe(1).f_code.co_name != 'main':  # noqa
            raise RuntimeError("The plot function can only be called from the main function!")

    # Ensure unique title
    title: str = 'Plot' if title is None else title
    # Handle duplicate titles
    c = 0
    t: str = title
    while t in lib._plot_data:
        t = title + ' ' + str(c)
        c += 1

    lib._plot_data[t] = series
    meta = lib._plot_meta.get(t)
    if meta is None:
        meta = PlotMeta(id=t, kind='plot', title=t, color=color, linewidth=linewidth, style=style,
                        trackprice=trackprice, histbase=histbase, offset=offset, join=join,
                        editable=editable, show_last=show_last, display=display, format=format,
                        precision=precision, force_overlay=force_overlay)
        lib._plot_meta[t] = meta
        lib._plot_meta_new.append(meta)
    if meta.dynamic:
        # Once dynamic, record every bar so a return to the static color is emitted
        lib._viz_dyn[t] = color
    elif color is not None and color is not meta.color:
        lib._viz_dyn[t] = color
        meta.dynamic = True
        # The static meta record is already out — re-queue an updated one.
        lib._plot_meta_new.append(meta)

    return Plot(t)
