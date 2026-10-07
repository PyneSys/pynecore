# cython: language_level=3
"""Compiled color creation with the same mutable Color type as the Python fallback.

Alpha arithmetic follows ``types.color._alpha`` operation for operation. The build
disables FMA contraction and fast-math so fractional transparency rounds identically.
``lib.color.new`` is stateless and already uses a plain call-site route; this twin
needs no instance binding either.
"""
from libc.math cimport floor

from ..types.color import Color
from ..types.na import NA


def new(color, transp=0):
    """Return a fresh color with unchanged RGB and the requested transparency.

    :param color: A Color, a hexadecimal color string or an na color
    :param transp: Transparency percentage, clipped into 0-100
    :return: A fresh Color, or an na color if either argument is na
    """
    cdef double transparency
    cdef int alpha
    cdef object result

    if isinstance(color, NA) or not (transp == transp):
        return NA(Color)
    if isinstance(color, str):
        color = Color(color)
    if transp <= 0:
        alpha = 255
    elif transp >= 100:
        alpha = 0
    else:
        transparency = transp
        alpha = <int>floor((1.0 - transparency / 100.0) * 255.0 + 0.5)
    result = object.__new__(Color)
    result.value = (color.value & 0xFFFFFF00) | alpha
    return result
