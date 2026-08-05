"""
Snapshot and rollback of the drawing registries.

A bar's body can run more than once: ``calc_on_order_fills`` replays it after
every fill, and in live mode every intra-bar tick replays it. Only the last run
counts, so everything a discarded run drew has to be gone before the next one
starts -- otherwise the drawings pile up in the registry, fill the
``max_lines_count`` budget and evict live ones.
"""
from dataclasses import fields as dataclass_fields
from typing import Any

from ..lib import box as _box, label as _label, line as _line
from ..lib import linefill as _linefill, polyline as _polyline, table as _table

__all__ = ['DrawingSnapshot']

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
    """

    __slots__ = ('_registries', '_states')

    def __init__(self) -> None:
        self._registries: list[tuple[Any, Any]] = []
        self._states: list[tuple[Any, tuple[tuple[str, Any], ...]]] = []

    def save(self) -> None:
        """Snapshot every registry and the field values of every live drawing."""
        self._registries = [(registry, registry.copy()) for registry in _REGISTRIES]
        self._states = [
            (obj, tuple((f.name, getattr(obj, f.name)) for f in dataclass_fields(obj)))
            for registry in _REGISTRIES for obj in registry
        ]

    def restore(self) -> None:
        """Put every registry and every drawing back to the snapshot."""
        for registry, saved in self._registries:
            if isinstance(registry, dict):
                registry.clear()
                registry.update(saved)
            else:
                registry[:] = saved
        for obj, saved_fields in self._states:
            for name, value in saved_fields:
                setattr(obj, name, value)
