"""
Pine Script type checker compatibility layer.

In Pine Script (and PyneCore), every variable is implicitly a Series — meaning even a plain
``float`` supports historical indexing like ``close[1]`` (the previous bar's value). At runtime
the AST transformer handles this, but IDEs (PyCharm, etc.) only see the static types. Since
``Series[float]`` is a union of ``float | _SeriesType[float] | NA[float]``, strict type checkers
complain that the ``float`` member has no ``__getitem__``.

This module solves it by re-exporting ``float``, ``int``, and ``bool`` as enhanced subclasses
**only under TYPE_CHECKING** (i.e., only visible to the IDE, zero runtime cost). The enhanced
versions inherit from their builtins and add ``__getitem__``, so every member of the Series union
becomes indexable in the IDE's eyes.

Usage in @pyne files (via the TypeCheckingStripperTransformer, these are removed at runtime)::

    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from pynecore.types.type_checker import *

Or import via ``pine_types`` which re-exports these automatically::

    from pynecore.types import PyneFloat  # already includes the enhanced float

Pyright/basedpyright: set ``"defineConstant": {"TYPECHECKER": "pyright"}`` in pyrightconfig.json
PyCharm: set ``TYPECHECKER=pycharm`` environment variable
"""
import builtins

TYPECHECKER: str

if TYPECHECKER == "pycharm":
    # noinspection PyPep8Naming,PyRedeclaration,PyShadowingBuiltins
    class float(builtins.float):
        def __getitem__(self, index: builtins.int) -> builtins.float: ...

    # noinspection PyPep8Naming,PyRedeclaration,PyShadowingBuiltins
    class int(builtins.int):
        def __getitem__(self, index: builtins.int) -> builtins.int: ...

    # noinspection PyPep8Naming,PyRedeclaration,PyShadowingBuiltins
    class bool(builtins.bool):  # type: ignore
        def __getitem__(self, index: builtins.int) -> builtins.bool: ...

else:
    float = builtins.float
    int = builtins.int
    bool = builtins.bool
