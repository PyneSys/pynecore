"""
Snapshot and rollback of the drawing registries.

A bar's body can run more than once: ``calc_on_order_fills`` replays it after
every fill, and in live mode every intra-bar tick replays it. Only the last run
counts, so everything a discarded run drew has to be gone before the next one
starts -- otherwise the drawings pile up in the registry, fill the
``max_lines_count`` budget and evict live ones.
"""
from dataclasses import fields as dataclass_fields
from operator import attrgetter, is_not
from typing import Any

from ..lib import box as _box, label as _label, line as _line
from ..lib import linefill as _linefill, polyline as _polyline, table as _table

__all__ = ['DrawingSnapshot']

_accessor_cache: dict[type, tuple[Any, tuple[str, ...]]] = {}


def _accessors(obj: Any) -> tuple[Any, tuple[str, ...]]:
    """One C-level getter for all dataclass fields of a drawing type, and the
    field names in the same order. Every drawing type has several fields, so
    the getter always returns a tuple."""
    cls = type(obj)
    entry = _accessor_cache.get(cls)
    if entry is None:
        names = tuple(f.name for f in dataclass_fields(obj))
        entry = _accessor_cache[cls] = (attrgetter(*names), names)
    return entry


# Insertion order carries meaning: the registries evict their oldest entry once
# the script's max_*_count is reached, so a restore has to put the entries back
# in order, not just as a set.
_REGISTRIES = (_line._registry, _label._registry, _box._registry,
               _table._registry, _polyline._registry, _linefill._registry)


class DrawingSnapshot:
    """
    Snapshot/restore of every drawing registry and of the drawings themselves.

    Field values are written back into the SAME objects, so a handle still held
    by a script variable keeps addressing its chart object -- replacing the
    object would detach the variable from the registry.

    A table keeps its cells in a dict that the ``table.cell*`` functions change
    in place, so the dict and every cell's fields are saved as well.
    """

    __slots__ = ('_registries', '_cells', '_states')

    def __init__(self) -> None:
        self._registries: list[tuple[Any, Any]] = []
        self._cells: list[tuple[dict, dict]] = []
        self._states: list[tuple[Any, tuple[Any, tuple[str, ...]], tuple]] = []

    def save(self) -> None:
        """Snapshot every registry and the field values of every live drawing."""
        self._registries = [(registry, registry.copy()) for registry in _REGISTRIES]
        self._cells = [(table.cells, table.cells.copy()) for table in _table._registry]
        self._states = [
            (obj, accessors, accessors[0](obj))
            for registry in _REGISTRIES for obj in registry
            for accessors in (_accessors(obj),)
        ]
        self._states.extend(
            (cell, accessors, accessors[0](cell))
            for cells, _saved in self._cells for cell in cells.values()
            for accessors in (_accessors(cell),)
        )

    def restore(self) -> None:
        """Put every registry and every drawing back to the snapshot."""
        for registry, saved in self._registries:
            if isinstance(registry, dict):
                registry.clear()
                registry.update(saved)
            else:
                registry[:] = saved
        for cells, saved_cells in self._cells:
            cells.clear()
            cells.update(saved_cells)
        for obj, (getter, names), saved_fields in self._states:
            # Identity, not equality: ``1``, ``1.0`` and ``True`` compare equal
            if any(map(is_not, getter(obj), saved_fields)):
                for name, value in zip(names, saved_fields):
                    setattr(obj, name, value)
