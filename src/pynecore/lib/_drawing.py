"""
Coordinate normalization shared by the drawing libraries.

A drawing's coordinates are Pine values the script hands over: an x is a bar
index or a bar time, a y is a price. Both arrive at the library boundary in
whatever shape the script had them in — an int, a float with a fractional
part, or ``na`` in any of its forms — and the object stores what the getters
read back later, so the normalization belongs here, on the way in, not in
every getter and every consumer of one.

An x lands on the bar grid: Pine truncates it, and the stored value is a
float like every other Pine number. A y keeps its value untouched. Anything
that is not a number is ``na``.
"""
from typing import Any

from ..types.na import na_int, na_float

__all__ = ['bar_coord', 'price']


def bar_coord(value: Any) -> float:
    """Normalize a drawing's x coordinate (bar index or bar time).

    :param value: The coordinate the script passed.
    :return: The truncated coordinate, or ``na`` when it is not a number.
    """
    return float(int(value)) if isinstance(value, (int, float)) and value == value else na_int


def price(value: Any) -> float:
    """Normalize a drawing's y coordinate (a price).

    :param value: The coordinate the script passed.
    :return: The price unchanged, or ``na`` when it is not a number.
    """
    return value if isinstance(value, (int, float)) else na_float
