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

ADOPTED_STARTUP_EXTRA_KEY = "adopted_startup"
"""``OrderRow.extras`` flag marking a store row a plugin synthesized at startup
for an *untracked* live venue leg (one this run has no durable journal for).

Such rows exist purely so the normal close/exit paths have a confirmed
``position`` row to route a DELETE/opposite-close against — they are NOT a
product of THIS run's own orders. Startup run-ownership reconstruction
(:meth:`OrderSyncEngine._durable_owned_signed_size`) must therefore exclude
them: on a one-way (netting) account two runs share one venue net, and a leg
this run merely adopted for bookkeeping belongs to another run. Counting it as
owned would re-inflate the ownership clamp and let the run copy a foreign
run's exposure into ``_position`` — the very cross-run double count the clamp
exists to prevent.
"""


class QtyType(StrLiteral):
    ...


class Direction(StrLiteral):
    ...


class Commission(StrLiteral):
    ...


class Oca(StrLiteral):
    ...
