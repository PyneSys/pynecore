from .base import StrLiteral

ADOPTED_STARTUP_ENTRY_ID = "__adopted_startup__"
"""Synthetic FIFO parent-trade id seeded by startup adoption.

When a fresh process restarts over an existing broker position and the real
Pine parent entry id cannot be recovered (no bracket, or a pyramided
multi-parent position), the sync engine seeds the adopted size under this
synthetic id. The id deliberately does NOT match any real ``strategy.entry``
id, so both the close-quantity clamp and the ``strategy.close(id)`` binding
must treat an open FIFO that carries it as untracked exposure: a keyed
``strategy.close(id)`` that misses every faithful id must still be allowed to
flatten the adopted position rather than be dropped.
"""


class QtyType(StrLiteral):
    ...


class Direction(StrLiteral):
    ...


class Commission(StrLiteral):
    ...


class Oca(StrLiteral):
    ...
