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

JOURNAL_EXPOSURE_RETIRED_EXTRA_KEY = "journal_exposure_retired"
"""``OrderRow.extras`` counter of entry exposure already closed back on the venue.

An entry row's ``filled_qty`` is a MONOTONE cumulative-execution watermark — the
PUSH / reconcile / recovery de-dup paths all compare the venue's cumulative
``executedVolume`` against it, so a partial close of the position must never
decrement it. Plugins whose close fills do not land as separate journal rows
(cTrader books a close under the venue's own close order id, which is never a
row of ours) instead accumulate the closed quantity here, on the entry row the
close reduced. Run-ownership reconstruction
(:meth:`OrderSyncEngine._durable_owned_signed_size`) subtracts it, clamped into
``[0, filled_qty]``, so the owned net reflects the venue's remaining exposure
while the watermark semantics of ``filled_qty`` stay intact.
"""


class QtyType(StrLiteral):
    ...


class Direction(StrLiteral):
    ...


class Commission(StrLiteral):
    ...


class Oca(StrLiteral):
    ...
