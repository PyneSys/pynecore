"""In-run resolution of dispatches parked with an unknown disposition.

A venue write whose reply was lost (timeout, dropped connection, gateway
error) leaves its persist-first row in
:data:`~pynecore.core.broker.store_helpers.STATE_DISPOSITION_UNKNOWN` and the
engine's dispatch parked. The engine's own verification only promotes an
order that ``get_open_orders`` echoes back, and the fill path only unparks an
order that executes. A dispatch that never reached the venue matches neither,
so it would stay parked for the rest of the run: the strategy's entry never
goes out again and every exit that depends on it is skipped, sync after sync.

Releasing such a dispatch is the same rule on every venue; only the question
"what does the venue know about this client order id" is venue-specific. This
module owns the rule, and a plugin supplies the lookup:

* the venue still has the order working or has executed some of it
  (:attr:`ParkedOutcome.LANDED`): nothing to release, the engine verification
  or the fill path unparks it. An order the venue no longer lists, whose
  execution the plugin's stream can therefore never deliver, the plugin hands
  over with :meth:`~pynecore.core.plugin.broker.BrokerPlugin.request_execution_hold`:
  the engine confirms the row on it and books its execution from the venue;
* the venue has the order and refused it without a fill
  (:attr:`ParkedOutcome.REFUSED`): rejected at once;
* the venue conclusively has no such order (:attr:`ParkedOutcome.ABSENT`):
  rejected once the row is old enough and the answer was the same on enough
  consecutive lookups;
* the venue could not be read (:attr:`ParkedOutcome.INCONCLUSIVE`): no
  evidence either way, the miss streak stays as it was.

A rejected row is closed through the journal and gets a ``'rejected'`` plugin
resolution, which the engine consumes on its next sync: it drops the parked
envelope and dispatches the intent again. A rejection made from a conclusive
absence is an inference, not the venue's answer: the row is flagged
:data:`~pynecore.core.broker.store_helpers.EXTRAS_KEY_REJECTED_ON_ABSENCE`, so a
reader that needs proof that nothing of the dispatch executed (a held
partial-bracket close, a cancel probe) does not take it for one
(:func:`~pynecore.core.broker.store_helpers.is_proven_refusal`).

A dispatch an earlier process of the run left pending is resolved the same
way. Its row was adopted at startup
(:meth:`~pynecore.core.broker.storage.RunContext.is_inherited_dispatch`), and
the plugin's startup recovery could not tell yet whether the send landed: the
row may still be ``submitted`` (a crash between the persist-first write and
the venue's answer), and it may have no park, because the park lived only in
the dispatch that crashed. Nothing of the running process would ever ask the
venue about it again, while the script emitting the same intent makes the
engine dispatch it again under the same client order id. On a venue that does not
refuse a duplicate id itself (any idempotency but
``CapabilityLevel.NATIVE``), a first send that did land then opens the
position twice. Three pieces close this:

* the resolver looks inherited rows up in every pending state, not only in
  :data:`~pynecore.core.broker.store_helpers.STATE_DISPOSITION_UNKNOWN`
  (:func:`awaits_venue_answer`);
* on such a venue the engine gives every inherited entry row without a park
  one at startup (:func:`park_inherited_dispatches`), so a landed working
  order is promoted by the engine's verification and a rejection reaches the
  engine as for any other park;
* on such a venue the engine does not dispatch an entry while an inherited
  dispatch of the same intent is unresolved
  (:func:`inherited_dispatch_in_flight`). The hold lasts until the plugin's
  resolver pass answers; a plugin that never runs one keeps the entry held
  for the whole run, and the engine reminds of it periodically in the log.
"""
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from time import time as epoch_time
from typing import TYPE_CHECKING

from pynecore.core.broker.journal import DispatchJournal, ReconcileOutcome
from pynecore.core.broker.store_helpers import (
    ENTRY_KIND_POSITION,
    ENTRY_KIND_WORKING,
    EXTRAS_KEY_REJECTED_ON_ABSENCE,
    PENDING_DISPATCH_STATES,
    find_pending_dispatch,
)

if TYPE_CHECKING:
    from pynecore.core.broker.storage import OrderRow, RunContext

__all__ = [
    'ParkedDispatchResolver',
    'ParkedOutcome',
    'ParkedVerdict',
    'awaits_venue_answer',
    'inherited_dispatch_in_flight',
    'park_inherited_dispatches',
]

#: Audit event written when a parked dispatch is released as rejected.
PARKED_DISPATCH_REJECTED_EVENT = 'parked_dispatch_rejected'


class ParkedOutcome(Enum):
    """What the venue knows about a parked dispatch's client order id."""
    INCONCLUSIVE = 'inconclusive'
    ABSENT = 'absent'
    REFUSED = 'refused'
    LANDED = 'landed'


@dataclass(frozen=True, slots=True)
class ParkedVerdict:
    """A plugin's answer for one parked dispatch.

    :ivar outcome: What the venue knows about the order.
    :ivar order_id: The venue's order id when it has the order.
    :ivar status: The venue's own name of the order's status when it has the
        order; it goes into the audit event and the log line of a rejection.
    """
    outcome: ParkedOutcome
    order_id: str | None = None
    status: str = ''


def awaits_venue_answer(store_ctx: 'RunContext', row: 'OrderRow') -> bool:
    """Whether a live row's dispatch waits for the resolver's venue lookup.

    A :data:`~pynecore.core.broker.store_helpers.STATE_DISPOSITION_UNKNOWN`
    row does, and so does an inherited dispatch
    (:meth:`~pynecore.core.broker.storage.RunContext.is_inherited_dispatch`)
    in any pending state. A ``submitted`` row of this process does not: it is
    mid-flight, or the engine retries it itself.

    The rule is :meth:`~pynecore.core.broker.storage.RunContext.awaits_venue_answer`.

    :param store_ctx: The run's store.
    :param row: A live row without a venue order id.
    :return: Whether the row is the resolver's to look up.
    """
    return store_ctx.awaits_venue_answer(row)


def park_inherited_dispatches(store_ctx: 'RunContext') -> list[str]:
    """Park every inherited entry dispatch that is still pending and has no park.

    An inherited dispatch whose send crashed before the venue answered has no
    park: the engine parks a dispatch only when the plugin reports the
    unknown disposition, which the crash cut off. Parked here, it is treated
    like a park the earlier process left: the engine's verification promotes
    it once the venue lists the order as working, and the resolver's
    ``'rejected'`` resolution makes the engine drop its envelope and
    dispatch the intent again under a fresh client order id. A row that
    already has a park keeps it untouched, resolution included.

    Only entry rows (``extras['kind']`` a position or a working order) are
    parked; the park is dated at the row's own dispatch. The parks are one
    write: a failure parks none, and the next call parks them all.

    :param store_ctx: The run's store.
    :return: The client order ids parked.
    """
    inherited = store_ctx.inherited_dispatch_ids()
    if not inherited:
        return []
    _envelopes, parks = store_ctx.replay()
    parked: list[str] = []
    with store_ctx.transaction():
        for row in find_pending_dispatch(store_ctx):
            coid = row.client_order_id
            if (coid not in inherited or coid in parks or not row.intent_key
                    or (row.extras or {}).get('kind') not in (ENTRY_KIND_POSITION,
                                                              ENTRY_KIND_WORKING)):
                continue
            store_ctx.record_park(coid, row.intent_key, parked_ts_ms=row.created_ts_ms)
            parked.append(coid)
    return parked


def inherited_dispatch_in_flight(store_ctx: 'RunContext', intent_key: str) -> str | None:
    """Return the inherited dispatch of an intent whose outcome is still open.

    An inherited dispatch (:meth:`~pynecore.core.broker.storage.RunContext.is_inherited_dispatch`)
    is open while its row is live in a
    :data:`~pynecore.core.broker.store_helpers.PENDING_DISPATCH_STATES`
    state, and also once the resolver rejected it until the engine consumed
    that ``'rejected'`` resolution: the engine drops the envelope only then,
    and a dispatch before it would reuse the rejected client order id.

    :param store_ctx: The run's store.
    :param intent_key: The intent's key.
    :return: The open dispatch's client order id, or ``None``.
    """
    inherited = store_ctx.inherited_dispatch_ids()
    if not inherited:
        return None
    parks = None
    for coid in sorted(inherited):
        row = store_ctx.get_order(coid)
        if row is None or row.intent_key != intent_key:
            continue
        if row.closed_ts_ms is None and row.state in PENDING_DISPATCH_STATES:
            return coid
        if parks is None:
            parks = store_ctx.replay()[1]
        park = parks.get(coid)
        if park is not None and park.resolution == 'rejected':
            return coid
    return None


class ParkedDispatchResolver:
    """Release the parked dispatches the venue never accepted.

    One instance lives as long as its plugin's connection state; the miss
    streaks are in memory only, so a restart counts them afresh.

    :param venue: Venue name for the log lines.
    :param log: The plugin's logger.
    :param grace_s: Age of the persist-first row before the first lookup. It
        covers the venue's own indexing lag of a write that did land moments
        before its reply was lost.
    :param absent_after_s: Age of the row before a conclusive absence may
        reject it. A venue whose order history is indexed late needs more than
        ``grace_s``; ``None`` uses ``grace_s``.
    :param misses_to_reject: Consecutive conclusive absences needed.
    :param lookup_interval_s: Shortest time between two lookups of one row.
    """

    def __init__(
            self, *, venue: str,
            log: logging.Logger,
            grace_s: float = 30.0,
            absent_after_s: float | None = None,
            misses_to_reject: int = 3,
            lookup_interval_s: float = 0.0,
            clock: Callable[[], float] = epoch_time,
    ) -> None:
        self._venue = venue
        self._log = log
        self._grace_s = grace_s
        self._absent_after_s = grace_s if absent_after_s is None else absent_after_s
        self._misses_to_reject = misses_to_reject
        self._lookup_interval_s = lookup_interval_s
        self._clock = clock
        #: Consecutive conclusive absences per client order id.
        self.misses: dict[str, int] = {}
        self._last_lookup: dict[str, float] = {}

    async def resolve(
            self, store_ctx: 'RunContext | None',
            lookup: 'Callable[[OrderRow], Awaitable[ParkedVerdict]]',
    ) -> list[str]:
        """Run one pass over the parked dispatches.

        Rows without a venue order id are looked at when they are in
        :data:`~pynecore.core.broker.store_helpers.STATE_DISPOSITION_UNKNOWN`,
        or when an earlier process left them pending
        (:meth:`~pynecore.core.broker.storage.RunContext.is_inherited_dispatch`).
        A ``submitted`` row of this process is mid-flight or waits for the
        engine's own retry, and a row that got its venue reference is the
        engine's.

        :param store_ctx: The run's store; ``None`` without persistence.
        :param lookup: Asks the venue about one parked row. It must report
            :attr:`ParkedOutcome.ABSENT` only when the venue was read
            completely; a failed or truncated read is
            :attr:`ParkedOutcome.INCONCLUSIVE`.
        :return: The client order ids rejected in this pass.
        """
        if store_ctx is None:
            return []
        rejected: list[str] = []
        seen: set[str] = set()
        for row in list(find_pending_dispatch(store_ctx)):
            if row.exchange_order_id or not awaits_venue_answer(store_ctx, row):
                continue
            coid = row.client_order_id
            seen.add(coid)
            now = self._clock()
            age_s = now - row.created_ts_ms / 1000.0
            if age_s < self._grace_s:
                continue
            if now - self._last_lookup.get(coid, float('-inf')) < self._lookup_interval_s:
                continue
            self._last_lookup[coid] = now
            verdict = await lookup(row)
            if verdict.outcome is ParkedOutcome.INCONCLUSIVE:
                continue
            if verdict.outcome is ParkedOutcome.LANDED:
                self.misses.pop(coid, None)
                continue
            if verdict.outcome is ParkedOutcome.ABSENT:
                self.misses[coid] = self.misses.get(coid, 0) + 1
                if (self.misses[coid] < self._misses_to_reject
                        or age_s < self._absent_after_s):
                    continue
            if self._reject(store_ctx, row, verdict):
                rejected.append(coid)
        for coid in [c for c in self.misses if c not in seen]:
            del self.misses[coid]
        for coid in [c for c in self._last_lookup if c not in seen]:
            del self._last_lookup[coid]
        return rejected

    def _reject(self, store_ctx: 'RunContext', row: 'OrderRow', verdict: ParkedVerdict) -> bool:
        """Close a parked row the venue never accepted and tell the engine.

        The verdict was formed while the venue lookup was awaited, and in the
        meantime the fill path or a reconcile pass may have confirmed the
        order, or a dispatch of this process may have taken an inherited row
        over. The row is therefore read again inside the write transaction,
        and the rejection applies only while it is still a live, unfilled row
        without a venue reference, in the state the lookup was asked about,
        that still waits for the resolver. The row's close, its audit event
        and the engine's ``'rejected'`` resolution commit together: a failed
        write leaves the row parked and eligible for the next pass.

        The envelope anchor and the park row stay: the engine's resolution
        consumer drops them once it has cleared its in-memory intent, so the
        two sides retire together. An inherited row without a park (the
        engine parks inherited rows only where a duplicate send is unsafe,
        see :func:`park_inherited_dispatches`) is only closed; the engine
        has nothing of it to retire.

        :return: Whether the row was rejected.
        """
        coid = row.client_order_id
        found = verdict.outcome is ParkedOutcome.REFUSED
        with store_ctx.transaction():
            current = store_ctx.get_order(coid)
            if (current is None or current.closed_ts_ms is not None
                    or current.state != row.state
                    or not awaits_venue_answer(store_ctx, current)
                    or current.exchange_order_id or current.filled_qty > 0.0):
                return False
            DispatchJournal(store_ctx).apply_reconcile_outcome(
                coid,
                ReconcileOutcome(
                    kind='terminal_close',
                    reason='missing_pending_grace_expired',
                    new_state='rejected',
                    audit_event=PARKED_DISPATCH_REJECTED_EVENT,
                    close_row=True,
                    audit_payload={'order_id': verdict.order_id or '',
                                   'order_status': verdict.status, 'found': found},
                    exchange_order_id=verdict.order_id or None,
                    extras_patch=None if found else {EXTRAS_KEY_REJECTED_ON_ABSENCE: True},
                ),
            )
            store_ctx.record_resolution(coid, 'rejected')
        self.misses.pop(coid, None)
        self._last_lookup.pop(coid, None)
        if found:
            why = f"venue reports {verdict.status}"
        else:
            why = f"not found on the venue after {self._misses_to_reject} consecutive lookups"
        self._log.warning(
            "%s %s dispatch %s (intent %s) resolved as rejected: %s — "
            "the intent may be dispatched again",
            self._venue,
            'inherited' if store_ctx.is_inherited_dispatch(coid) else 'parked',
            coid, row.intent_key, why,
        )
        return True
