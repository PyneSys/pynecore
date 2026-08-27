"""Pure cancel-disposition classification (#55).

A cancel's outcome may only come from a POSITIVE venue observation — a
read-back (detail or history row) whose status and fill answer "cancelled
with no fill?" directly. Nothing here does I/O: the broker feeds observations
in, these functions map them onto :class:`CancelDispositionOutcome`.

Guards pinned by ``tests/test_cancel_disposition.py``:

- **G1**: ``CANCEL_CONFIRMED`` requires a terminal status AND ``filled_qty == 0``.
- **G5**: a WORKING read-back (``New``/``PendingNew``/…) is ``UNKNOWN`` — never
  ``STILL_OPEN`` (which per models.py means "a fresh cancel *succeeded*" and
  makes the engine fire the entry-stop market) and never ``TOO_LATE_TO_CANCEL``.
- **G6**: ``filled_qty > 0`` outranks the status on EVERY read path — a
  ``Canceled``/``Rejected``/``Expired`` row carrying a partial fill classifies
  ``ALREADY_FILLED`` (the engine must restore legs onto the position), never a
  confirmed-cancel-class outcome the engine would fire the market on.
"""
from typing import Iterable

from pynecore.core.broker.models import CancelDispositionOutcome, OrderStatus

#: Model-level terminal statuses. Deliberately duplicated from
#: ``broker._TERMINAL_STATUSES`` so this module needs no broker import (pure,
#: no I/O); the two sets are asserted equal by a test anchor.
TERMINAL_STATUSES = frozenset({
    OrderStatus.FILLED, OrderStatus.CANCELLED,
    OrderStatus.REJECTED, OrderStatus.EXPIRED,
})


def classify_readback(status: OrderStatus,
                      filled_qty: float) -> CancelDispositionOutcome:
    """Map one venue read-back onto a cancel disposition (G1/G5/G6)."""
    if status not in TERMINAL_STATUSES:
        return CancelDispositionOutcome.UNKNOWN            # G5: still working
    if status is OrderStatus.FILLED or filled_qty > 0:
        return CancelDispositionOutcome.ALREADY_FILLED     # G6: fill outranks status
    if status is OrderStatus.REJECTED:
        return CancelDispositionOutcome.TOO_LATE_TO_CANCEL  # terminal, provably no fill
    return CancelDispositionOutcome.CANCEL_CONFIRMED       # CANCELLED/EXPIRED, fill 0


def aggregate(outcomes: Iterable[CancelDispositionOutcome]
              ) -> CancelDispositionOutcome:
    """Fold per-id dispositions into the envelope's answer.

    capitalcom's conservative order (execution.py:1775-1781): any
    ``ALREADY_FILLED`` wins (the engine must restore legs onto a position that
    now exists), then any ``UNKNOWN`` keeps the retry loop armed, and only a
    unanimously confirmed-cancel class collapses to ``CANCEL_CONFIRMED``.
    Empty means nothing was observed — ``UNKNOWN``, never a fabricated verdict.
    """
    folded = list(outcomes)
    if not folded:
        return CancelDispositionOutcome.UNKNOWN
    if CancelDispositionOutcome.ALREADY_FILLED in folded:
        return CancelDispositionOutcome.ALREADY_FILLED
    if CancelDispositionOutcome.UNKNOWN in folded:
        return CancelDispositionOutcome.UNKNOWN
    return CancelDispositionOutcome.CANCEL_CONFIRMED
