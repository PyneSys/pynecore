from dataclasses import dataclass

from .pine_types import PyneFloat, PyneInt


@dataclass(slots=True)
class ChartPoint:
    # The x-coordinate of the point, expressed as a bar index value (``na`` when the point
    # was created from a time only, e.g. ``chart.point.from_time``)
    index: PyneInt

    # The x-coordinate of the point, expressed as a UNIX time value, in milliseconds (``na``
    # when the point was created from an index only, e.g. ``chart.point.from_index``)
    time: PyneInt

    # The y-coordinate of the point
    price: PyneFloat
