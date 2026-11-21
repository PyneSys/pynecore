"""
Pine qualifier markers for overload implementations.

Pine lets two overloads of a function differ only by a parameter's qualifier:
``replace(simple string this, ...)`` next to ``replace(string this, ...)``. In
Python both would have the same signature, so PyneComp marks the qualified
parameter as ``Simple[str]`` (or ``Const[str]``) in exactly such a group, and
the type inference reads the marker to pick the implementation Pine picks for
the call's argument.

At run time a marker is nothing: ``Simple[str]`` IS ``str``, so the overload
dispatcher, ``get_type_hints`` and every other runtime reader see the plain
type.
"""
from typing import Any

__all__ = (
    'Simple',
    'Const',
)


class Simple:
    """
    Marks a parameter Pine declares ``simple``; ``Simple[T]`` is ``T`` at run time.

    What users see on hover comes from ``qualifier.pyi``, not from here.
    """

    def __class_getitem__(cls, item: Any) -> Any:
        return item


class Const:
    """
    Marks a parameter Pine declares ``const``; ``Const[T]`` is ``T`` at run time.

    What users see on hover comes from ``qualifier.pyi``, not from here.
    """

    def __class_getitem__(cls, item: Any) -> Any:
        return item
