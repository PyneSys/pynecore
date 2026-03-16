"""
Horizontal line

This is a callable module, so the modul itself is both a function and a namespace
"""
from __future__ import annotations
from ..core.callable_module import CallableModule

from ..types.hline import HLineEnum, HLine

from . import color as _color, display as _display


#
# Module object
#

class HLineModule(CallableModule):
    #
    # Constants
    #

    style_solid = HLineEnum('style_solid')
    style_dotted = HLineEnum('style_dotted')
    style_dashed = HLineEnum('style_dashed')


#
# Callable module function
#

def hline(
        price: float,
        title: str = "",
        color: _color.Color = _color.blue,
        linestyle: HLineEnum = HLineModule.style_solid,
        linewidth: int = 1,
        editable: bool = True,
        display: _display.Display = _display.all
) -> HLine:
    """
    Renders a horizontal line at a given fixed price level.

    :param price: Price value at which the object will be rendered. Required argument.
    :param title: Title of the object
    :param color: Color of the rendered line. Must be a constant value (not an expression)
    :param linestyle: Style of the rendered line. Possible values are: hline.style_solid, hline.style_dotted, hline.style_dashed
    :param linewidth: Width of the rendered line. Default value is 1
    :param editable: If true then hline style will be editable in Format dialog. Default is true
    :param display: Controls where the hline is displayed. Possible values are: display.none, display.all. Default is display.all
    :return: An hline object, that can be used in fill
    """
    return HLine(
        price=price,
        title=title or None,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        editable=editable,
        display=display
    )


#
# Module initialization
#

HLineModule(__name__)
