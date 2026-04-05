from typing import TypeVar, TypeAlias
from pynecore.types.na import NA
from .series import Series

T = TypeVar('T')

IBPersistent: TypeAlias = T | NA[T] | Series[T]
IBPersistentSeries: TypeAlias = T | NA[T] | Series[T]
