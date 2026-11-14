from dataclasses import dataclass
from typing import ClassVar

from .pine_types import PyneFloat, PyneInt


@dataclass(slots=True, eq=False)
class ChartPoint:
    # The x-coordinate of the point, expressed as a bar index value (``na`` when the point
    # was created from a time only, e.g. ``chart.point.from_time``)
    index: PyneInt

    # The x-coordinate of the point, expressed as a UNIX time value, in milliseconds (``na``
    # when the point was created from an index only, e.g. ``chart.point.from_index``)
    time: PyneInt

    # The y-coordinate of the point
    price: PyneFloat

    # Its fields are read directly: touching one on an na point is an error
    __pyne_object__: ClassVar[bool] = True
