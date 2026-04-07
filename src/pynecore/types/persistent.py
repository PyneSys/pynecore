from typing import TypeVar, TypeAlias
from .na import NA
from .series import Series

T = TypeVar('T')

# Public type alias that allows both T and NA[T] or Series[T]
Persistent: TypeAlias = T | NA[T] | Series[T]
