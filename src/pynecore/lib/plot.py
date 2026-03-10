from typing import Any
import sys

from ..core.callable_module import CallableModule
from ..types.plot import PlotEnum, Plot


#
# Module object
#

class PlotModule(CallableModule):
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


#
# Callable module function
#

# noinspection PyProtectedMember
def plot(series: Any, title: str | None = None, *_, **__):
    """
    Plot series, by default a CSV is generated, but this can be extended

    :param series: The value to plot in every bar
    :param title: The title of the plot, if multiple plots are created with the same title, a
                  number will be appended
    :return: The a Plot object, can be used to reference the plot in other functions
    """
    from .. import lib
    if lib._lib_semaphore:
        return Plot()

    if lib.bar_index == 0:  # Only check if it is the first bar for performance reasons
        # Check if it is called from the main function
        if sys._getframe(2).f_code.co_name not in ('main', 'plotchar'):  # noqa
            raise RuntimeError("The plot function can only be called from the main function!")

    # Ensure unique title
    if title is None:
        title = 'Plot'
    # Handle duplicate titles
    c = 0
    t = title
    while t in lib._plot_data:
        t = title + ' ' + str(c)
        c += 1
    title = t

    # Store plot data
    lib._plot_data[title] = series

    return Plot()


#
# Module initialization
#

PlotModule(__name__)
