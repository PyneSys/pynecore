"""
Horizontal line

The module is both a function (``hline(...)``) and a namespace (``hline.style_solid``);
call sites are routed to :func:`hline` by the module property AST transformer.
"""
from ..types.hline import HLineEnum, HLine
from ..types.plot_meta import PlotMeta

from . import color as _color, display as _display
from .. import lib


#
# Constants
#

style_solid = HLineEnum('style_solid')
style_dotted = HLineEnum('style_dotted')
style_dashed = HLineEnum('style_dashed')


#
# Module function
#

def hline(
        price: float,
        title: str = "",
        color: _color.Color = _color.blue,
        linestyle: HLineEnum = style_solid,
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
    n = lib._viz_seq.get('hline', 0)
    lib._viz_seq['hline'] = n + 1
    hid = f'hline#{n}'
    t = title or None
    if hid not in lib._plot_meta:
        meta = PlotMeta(id=hid, kind='hline', title=t, price=price, color=color,
                        linestyle=linestyle, linewidth=linewidth, editable=editable,
                        display=display)
        lib._plot_meta[hid] = meta
        lib._plot_meta_new.append(meta)
    return HLine(
        price=price,
        title=t,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        editable=editable,
        display=display,
        id=hid
    )
