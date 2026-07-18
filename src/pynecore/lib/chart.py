from copy import copy as _copy

from ..types.color import Color
from ..types.chart import ChartPoint
from ..types.na import NA
from ..core.module_property import module_property

from .. import lib
from .timeframe import in_seconds

__all__ = [
    'bg_color',
    'fg_color',
    'is_heikinashi',
    'is_kagi',
    'is_linebreak',
    'is_pnf',
    'is_range',
    'is_renko',
    'is_standard',
    'left_visible_bar_time',
    'right_visible_bar_time',
    'point'
]

_visible_bars = 20

bg_color = Color('#000000')
fg_color = Color('#FFFFFF')

is_heikinashi = False
is_kagi = False
is_linebreak = False
is_pnf = False
is_range = False
is_renko = False
is_standard = True


# The visible range is anchored to the chart's LAST bar (a chart scrolled to
# its right edge — TradingView's default viewport). ``lib.last_bar_time`` is
# fixed to the final bar on historical runs and tracks the realtime bar live,
# so ``time == chart.right_visible_bar_time`` is true once per historical run
# (on the final bar) and on every realtime bar — matching TV.
@module_property
def left_visible_bar_time() -> int:
    return lib.last_bar_time - int(in_seconds(lib.syminfo.period) * 1000) * _visible_bars


@module_property
def right_visible_bar_time() -> int:
    return lib.last_bar_time


# noinspection PyShadowingBuiltins
class _ChartPoint:
    """
    The ``chart.point`` constructor namespace.

    A chart point holds the coordinates a drawing object uses to position itself: a bar
    ``index``, a UNIX ``time`` (in milliseconds), and a ``price``. The ``xloc`` of the
    consuming drawing decides which x-coordinate field is read (``index`` for
    ``xloc.bar_index``, ``time`` for ``xloc.bar_time``).
    """

    @staticmethod
    def new(time: int, index: int, price: float) -> ChartPoint:
        """
        Creates a new ``chart.point`` from explicit time, index, and price values.

        :param time: The x-coordinate as a UNIX time value, in milliseconds
        :param index: The x-coordinate as a bar index value
        :param price: The y-coordinate
        :return: A new chart.point object
        """
        return ChartPoint(index=index, time=time, price=price)

    # noinspection PyProtectedMember
    @staticmethod
    def now(price: float) -> ChartPoint:
        """
        Creates a new ``chart.point`` at the current bar, using its index and time.

        :param price: The y-coordinate
        :return: A new chart.point object with the current bar's index and time
        """
        return ChartPoint(
            index=int(lib.bar_index),
            time=int(lib._datetime.timestamp() * 1000),
            price=price
        )

    @staticmethod
    def from_index(index: int, price: float) -> ChartPoint:
        """
        Creates a new ``chart.point`` from a bar index and price. The ``time`` field is ``na``.

        :param index: The x-coordinate as a bar index value
        :param price: The y-coordinate
        :return: A new chart.point object whose ``time`` field is ``na``
        """
        return ChartPoint(index=index, time=NA(int), price=price)

    @staticmethod
    def from_time(time: int, price: float) -> ChartPoint:
        """
        Creates a new ``chart.point`` from a UNIX time and price. The ``index`` field is ``na``.

        :param time: The x-coordinate as a UNIX time value, in milliseconds
        :param price: The y-coordinate
        :return: A new chart.point object whose ``index`` field is ``na``
        """
        return ChartPoint(index=NA(int), time=time, price=price)

    @staticmethod
    def copy(id: ChartPoint) -> ChartPoint:
        """
        Returns a copy of the ``chart.point`` object.

        :param id: The chart.point object to copy
        :return: A new chart.point object with the same field values
        """
        if isinstance(id, NA):
            return NA(ChartPoint)
        return _copy(id)


point = _ChartPoint()
