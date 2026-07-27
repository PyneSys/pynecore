from __future__ import annotations
from typing import TypeVar, Generic

__all__ = (
    'Series',
    'PersistentSeries',
)

T = TypeVar('T')


class Series(Generic[T]):
    """
    Runtime placeholder for the Pine ``series`` type: it passes the value
    through unchanged, so the annotation costs nothing at runtime.

    The actual series behavior — history and ``[n]`` indexing — is implemented
    by the AST transformers and the ``SeriesImpl`` class. What users see on
    hover comes from ``series.pyi``, not from here.
    """

    def __new__(cls, val: T) -> T:
        return val


# This is only for the AST transformer to mark a variable as Pine Script like persistent series
PersistentSeries = Series
