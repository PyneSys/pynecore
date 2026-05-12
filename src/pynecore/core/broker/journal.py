"""
Crash-safe dispatch journal for the broker-runtime layer.

Broker plugins currently encode their own persist-first state machine
for every ``execute_*`` call: write the order row, log the audit
event, POST to the exchange, parse the response, mirror the server
reference, confirm-readback, finalize the row. The Capital.com plugin
has six methods that re-implement the same nine-step pattern with
exchange-specific endpoints, response shapes, and reject codes
sprinkled between them.

:class:`DispatchJournal` owns the persist-first state machine. The
plugin provides typed hooks for the parts that genuinely differ
between exchanges: request shape, response parsing, reject
classification. The journal coordinates writes against the
:class:`~pynecore.core.broker.storage.RunContext` through the typed
helpers in :mod:`pynecore.core.broker.store_helpers`, so the canonical
``extras`` schema and the order-of-writes invariants live in exactly
one place.

This module is the M1 proof-of-shape — entry dispatch only, no
brackets, no modify. It is shipped *alongside* the existing plugin
state machine so the parity test can compare both paths byte-for-byte
before any production code path is rewritten. The replacement plan is
documented in ``docs/pynecore/plugin-system/broker/broker-plugin-responsibility-review.md``.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import time as epoch_time
from typing import Literal, Protocol, TYPE_CHECKING

from pynecore.core.broker.exceptions import (
    ExchangeOrderRejectedError,
    OrderDispositionUnknownError,
)
from pynecore.core.broker.models import (
    EntryIntent,
    ExchangeOrder,
    OrderStatus,
    OrderType,
)
from pynecore.core.broker.store_helpers import (
    create_entry_order_row,
    find_pending_dispatch,
    mark_confirmed_with_fill,
    mark_disposition_unknown,
    mark_rejected,
    record_server_ref,
)

if TYPE_CHECKING:
    from pynecore.core.broker.storage import OrderRow, RunContext

__all__ = [
    'DispatchJournal',
    'EntryDispatchHooks',
    'SubmitOutcome',
    'ConfirmOutcome',
    'ResumeOutcome',
    'ResumeStatus',
]


# === Hook return types =====================================================

@dataclass(frozen=True)
class SubmitOutcome:
    """Plugin's successful ``submit()`` result.

    On any failure mode (timeout, missing reference, synchronous
    reject) the hook raises an appropriate
    :class:`~pynecore.core.broker.exceptions.BrokerError` subclass
    instead of returning. The journal converts those raises to the
    matching persisted state.

    :ivar server_ref: Exchange-allocated reference for the submission
        (``dealReference`` for Capital.com, ``orderLinkId`` echo for
        Bybit, etc.). Persisted into ``order_refs`` under
        ``ref_type='deal_reference'``.
    :ivar raw: Verbatim exchange response, attached to the
        ``deal_reference_seen`` audit event for forensics. ``None``
        when the plugin does not want to expose the raw body.
    """
    server_ref: str
    raw: dict | None = None


@dataclass(frozen=True)
class ConfirmOutcome:
    """Plugin's successful ``confirm_submission()`` result.

    On a confirm-REJECTED outcome the hook raises
    :class:`~pynecore.core.broker.exceptions.ExchangeOrderRejectedError`
    (or a subclass) so the journal can persist the rejection. On a
    confirm-timeout it raises
    :class:`~pynecore.core.broker.exceptions.OrderDispositionUnknownError`.

    :ivar exchange_id: Exchange-allocated id for the resulting order
        / position (``dealId``, ``orderId``, ...). ``None`` when the
        exchange returns no id at confirm time — the journal still
        advances state but skips the ``deal_id`` ref.
    :ivar is_filled: ``True`` only for MARKET-side fills that confirm
        as OPEN. LIMIT / STOP submissions confirm as ACCEPTED but not
        filled; their fills arrive later via the activity stream.
    :ivar filled_qty: Confirmed fill quantity. Ignored unless
        ``is_filled`` is ``True``.
    :ivar fill_price: Confirm-side fill price. Persisted into
        ``extras['confirm_level']`` only when strictly positive (a
        zero/negative level is a no-quote artefact and would corrupt
        the recovery fallback).
    :ivar raw: Verbatim confirm response, attached to the
        ``confirmed`` audit event for forensics.
    """
    exchange_id: str | None
    is_filled: bool
    filled_qty: float = 0.0
    fill_price: float | None = None
    raw: dict | None = None


ResumeStatus = Literal['confirmed', 'rejected', 'still_unknown']


@dataclass(frozen=True)
class ResumeOutcome:
    """Plugin's verdict on a pending dispatch found at restart.

    :ivar status: ``'confirmed'`` when the plugin verified the
        submission landed (exchange shows the order / position).
        ``'rejected'`` when the plugin verified it did not land
        (activity stream, snapshot, or confirm GET says so).
        ``'still_unknown'`` when the plugin cannot decide yet — the
        journal leaves the row as-is and the engine's pending-
        verification reconciler tries again on the next sync.
    :ivar exchange_id: Same semantics as :class:`ConfirmOutcome`.
    :ivar is_filled: Same semantics as :class:`ConfirmOutcome`.
    :ivar filled_qty: Same semantics as :class:`ConfirmOutcome`.
    :ivar fill_price: Same semantics as :class:`ConfirmOutcome`.
    :ivar reject_reason: Free-form reject reason, written to the audit
        event when ``status == 'rejected'``.
    """
    status: ResumeStatus
    exchange_id: str | None = None
    is_filled: bool = False
    filled_qty: float = 0.0
    fill_price: float | None = None
    reject_reason: str | None = None


# === Hook protocol =========================================================

class EntryDispatchHooks(Protocol):
    """Plugin-supplied callbacks for an entry dispatch.

    A fresh hook instance is constructed by the plugin per dispatch.
    The journal owns persistence and state transitions; hooks own the
    exchange wire format, response parsing, and rejection
    classification.

    All four methods are mandatory. The ``async`` methods may raise
    any :class:`~pynecore.core.broker.exceptions.BrokerError`
    subclass; the journal catches them and persists the matching
    state before re-raising.
    """

    async def submit(
            self, *, coid: str, intent: EntryIntent, qty: float,
    ) -> SubmitOutcome:
        """Submit the order to the exchange.

        Implementations issue exactly one REST / WS call and return a
        :class:`SubmitOutcome` on success, OR raise
        :class:`~pynecore.core.broker.exceptions.OrderDispositionUnknownError`
        on network timeout / missing server reference, OR raise
        :class:`~pynecore.core.broker.exceptions.ExchangeOrderRejectedError`
        when the POST itself produces a definitive synchronous reject
        (e.g. a 4xx with a parseable reason and no server reference to
        confirm against). The journal converts both raises into the
        matching terminal / pending state. Implementations MUST NOT
        write any persistence — the journal owns that.

        The preferred pattern remains "defer rejection to
        :meth:`confirm_submission`" because most exchanges do issue a
        server reference even for soon-to-be-rejected orders, and the
        confirm phase carries richer reason data. Raise
        :class:`ExchangeOrderRejectedError` here only when the POST
        response itself is the final word.
        """
        ...

    async def confirm_submission(
            self, *, coid: str, intent: EntryIntent, server_ref: str,
    ) -> ConfirmOutcome:
        """Read back the confirmation for a recorded server reference.

        Implementations issue exactly one REST / WS call and return a
        :class:`ConfirmOutcome` on success, OR raise
        :class:`~pynecore.core.broker.exceptions.ExchangeOrderRejectedError`
        on synchronous reject, OR
        :class:`~pynecore.core.broker.exceptions.OrderDispositionUnknownError`
        on confirm-timeout / unparseable response. They MUST NOT
        write any persistence — the journal owns that.

        For exchanges that have no separate confirm step (server
        echoes the full result on POST), implement this as a pure
        function that synthesises a :class:`ConfirmOutcome` from data
        the plugin cached during ``submit``.
        """
        ...

    def exchange_order_from_state(
            self, *, row: 'OrderRow', intent: EntryIntent,
    ) -> ExchangeOrder:
        """Build the :class:`ExchangeOrder` the engine expects.

        The journal calls this *after* :class:`ConfirmOutcome` has
        been persisted, so the row carries the final fill state.
        Pure function — no I/O, no persistence writes.
        """
        ...

    async def resume_pending_dispatch(
            self, *, row: 'OrderRow', refs: Mapping[str, str],
    ) -> ResumeOutcome:
        """Decide the disposition of a pre-restart pending row.

        Called once per ``find_pending_dispatch`` hit during
        :meth:`DispatchJournal.recover_pending`. The plugin checks
        the exchange's authoritative view (activity stream, snapshot,
        confirm GET) and returns the corresponding
        :class:`ResumeOutcome`.

        :param row: The persisted row, including its ``extras`` dict.
        :param refs: Mapping of ``ref_type`` → ``ref_value`` already
            recorded for this COID (``'deal_reference'`` is the
            typical entry; ``'deal_id'`` may also be present when the
            crash happened after server-ref-seen but before
            confirmed).
        """
        ...


# === Journal ===============================================================

@dataclass
class DispatchJournal:
    """Persist-first orchestrator for a single dispatch lifecycle.

    Constructed once per :class:`~pynecore.core.broker.storage.RunContext`.
    Reused across dispatches — the instance is thread-safe to the same
    extent the underlying ``RunContext`` is (single writer assumption).

    :param store: The active run context.
    """
    store: 'RunContext'

    # --- Entry path --------------------------------------------------------

    async def run_entry(
            self,
            *,
            coid: str,
            intent: EntryIntent,
            qty: float,
            kind: str,
            hooks: EntryDispatchHooks,
            audit_payload: dict | None = None,
    ) -> list[ExchangeOrder]:
        """Run an entry dispatch from initial persist through confirm.

        The state-machine is :data:`STATE_SUBMITTED` → optionally
        :data:`STATE_SERVER_REF_SEEN` → :data:`STATE_CONFIRMED` (or
        :data:`STATE_REJECTED` / :data:`STATE_DISPOSITION_UNKNOWN` on
        the failure paths). Every state advance happens through a
        store helper, so the on-disk schema is identical to what the
        legacy plugin path writes — the parity test relies on that.

        :param coid: The dispatch's canonical client-order-id (already
            derived from ``envelope.client_order_id(KIND_ENTRY)``).
        :param intent: The :class:`EntryIntent` being dispatched.
        :param qty: Quantity to submit, already quantized to the
            broker's lot step.
        :param kind: :data:`ENTRY_KIND_POSITION` for MARKET orders or
            :data:`ENTRY_KIND_WORKING` for LIMIT / STOP. Decides the
            ``extras['kind']`` value.
        :param hooks: Plugin-supplied callbacks (see
            :class:`EntryDispatchHooks`).
        :param audit_payload: Extra fields merged into the
            ``dispatch_submitted`` audit event payload. Optional;
            typically the plugin's endpoint + body so forensics can
            reconstruct the exact request.
        :return: A one-element list with the resulting
            :class:`ExchangeOrder` — matches the
            :meth:`BrokerPlugin.execute_entry` signature.
        :raises OrderDispositionUnknownError: If ``submit`` reported
            an ambiguous outcome.
        :raises ExchangeOrderRejectedError: If ``confirm_submission``
            reported a definitive reject.
        """
        # (1) PERSIST submitted row + audit event.
        create_entry_order_row(
            self.store,
            coid=coid,
            symbol=intent.symbol,
            side=intent.side,
            qty=qty,
            intent_key=intent.intent_key,
            pine_entry_id=intent.pine_id,
            kind=kind,
            order_type=intent.order_type.value,
        )
        submit_payload = {'kind': kind, 'order_type': intent.order_type.value}
        if audit_payload:
            submit_payload.update(audit_payload)
        self.store.log_event(
            'dispatch_submitted',
            client_order_id=coid,
            intent_key=intent.intent_key,
            payload=submit_payload,
        )

        # (2) SUBMIT — network errors raise OrderDispositionUnknownError.
        # A synchronous reject (4xx with a reason string) raises
        # ExchangeOrderRejectedError; the canonical pattern is to defer
        # rejection to the confirm phase, but a plugin may legitimately
        # raise here when the POST itself definitively rejects (e.g. a
        # well-formed exchange error response with no server reference
        # to confirm against). The journal terminates the row safely
        # either way — leaving it ``submitted`` would make it look
        # pending and recovery would retry an order the exchange already
        # rejected.
        try:
            submit = await hooks.submit(coid=coid, intent=intent, qty=qty)
        except OrderDispositionUnknownError as exc:
            mark_disposition_unknown(self.store, coid=coid)
            self.store.log_event(
                'disposition_unknown',
                client_order_id=coid,
                intent_key=intent.intent_key,
                payload={'phase': 'submit', 'reason': str(exc)},
            )
            raise
        except ExchangeOrderRejectedError as exc:
            mark_rejected(self.store, coid=coid)
            self.store.log_event(
                'rejected',
                client_order_id=coid,
                intent_key=intent.intent_key,
                payload={'phase': 'submit', 'reason': str(exc)},
            )
            raise

        # (3) PERSIST server reference + advance state.
        record_server_ref(
            self.store,
            coid=coid,
            deal_reference=submit.server_ref,
            kind=kind,
            order_type=intent.order_type.value,
        )
        self.store.log_event(
            'deal_reference_seen',
            client_order_id=coid,
            payload={'deal_reference': submit.server_ref},
        )

        # (4) CONFIRM — reject raises ExchangeOrderRejectedError,
        #     timeout raises OrderDispositionUnknownError.
        try:
            confirm = await hooks.confirm_submission(
                coid=coid, intent=intent, server_ref=submit.server_ref,
            )
        except ExchangeOrderRejectedError as exc:
            mark_rejected(self.store, coid=coid)
            self.store.log_event(
                'rejected',
                client_order_id=coid,
                intent_key=intent.intent_key,
                payload={'reason': str(exc)},
            )
            raise
        except OrderDispositionUnknownError as exc:
            mark_disposition_unknown(self.store, coid=coid)
            self.store.log_event(
                'disposition_unknown',
                client_order_id=coid,
                intent_key=intent.intent_key,
                payload={'phase': 'confirm', 'reason': str(exc)},
            )
            raise

        # (5) PERSIST confirmed + fill (if any).
        mark_confirmed_with_fill(
            self.store,
            coid=coid,
            exchange_id=confirm.exchange_id,
            is_filled=confirm.is_filled,
            filled_qty=confirm.filled_qty,
            fill_price=confirm.fill_price,
        )
        self.store.log_event(
            'confirmed',
            client_order_id=coid,
            exchange_order_id=confirm.exchange_id,
            intent_key=intent.intent_key,
            payload={
                'is_filled': confirm.is_filled,
                'fill_price': confirm.fill_price,
            },
        )

        # (6) Return the ExchangeOrder built from final row state.
        row = self.store.get_order(coid)
        if row is None:
            raise RuntimeError(
                f"DispatchJournal.run_entry: row vanished after confirm "
                f"(coid={coid!r})"
            )
        return [hooks.exchange_order_from_state(row=row, intent=intent)]

    # --- Recovery path -----------------------------------------------------

    async def recover_pending(
            self,
            hooks_for: 'PendingHooksProvider',
    ) -> list['PendingResolution']:
        """Replay every pending row through the plugin's resume hook.

        Called once after :class:`~pynecore.core.broker.storage.BrokerStore.open_run`
        and before the first sync-engine iteration. The plugin
        supplies a callable that builds the per-row hook from the
        stored ``extras`` / refs; this keeps the journal free of
        plugin-specific construction logic.

        :param hooks_for: Callable that returns either an
            :class:`EntryDispatchHooks` for an entry row, or
            ``None`` if the row's ``extras['kind']`` is not handled
            by the entry journal (other kinds — bracket legs etc. —
            land in their own journals later).
        :return: One :class:`PendingResolution` per processed row,
            useful for diagnostics / tests.
        """
        resolutions: list[PendingResolution] = []
        for row in find_pending_dispatch(self.store):
            hooks = hooks_for(row)
            if hooks is None:
                resolutions.append(PendingResolution(
                    coid=row.client_order_id,
                    status='skipped',
                    reason='unhandled_kind',
                ))
                continue
            refs = _collect_refs_for(self.store, coid=row.client_order_id)
            outcome = await hooks.resume_pending_dispatch(row=row, refs=refs)
            resolutions.append(self._apply_resume_outcome(row, outcome))
        return resolutions

    def _apply_resume_outcome(
            self,
            row: 'OrderRow',
            outcome: ResumeOutcome,
    ) -> 'PendingResolution':
        """Persist the recovery verdict and return a diagnostic record."""
        if outcome.status == 'confirmed':
            mark_confirmed_with_fill(
                self.store,
                coid=row.client_order_id,
                exchange_id=outcome.exchange_id,
                is_filled=outcome.is_filled,
                filled_qty=outcome.filled_qty,
                fill_price=outcome.fill_price,
            )
            self.store.log_event(
                'recovered_confirmed',
                client_order_id=row.client_order_id,
                exchange_order_id=outcome.exchange_id,
                intent_key=row.intent_key,
                payload={
                    'is_filled': outcome.is_filled,
                    'fill_price': outcome.fill_price,
                    'prior_state': row.state,
                },
            )
        elif outcome.status == 'rejected':
            mark_rejected(self.store, coid=row.client_order_id)
            self.store.log_event(
                'recovered_rejected',
                client_order_id=row.client_order_id,
                intent_key=row.intent_key,
                payload={
                    'reason': outcome.reject_reason,
                    'prior_state': row.state,
                },
            )
        else:
            # still_unknown — keep the row; engine reconciler retries.
            self.store.log_event(
                'recovery_pending',
                client_order_id=row.client_order_id,
                intent_key=row.intent_key,
                payload={'prior_state': row.state},
            )
        return PendingResolution(
            coid=row.client_order_id,
            status=outcome.status,
            reason=outcome.reject_reason,
        )


# === Recovery support types ================================================

@dataclass(frozen=True)
class PendingResolution:
    """One row's recovery outcome, returned for inspection / tests.

    :ivar coid: The recovered row's client-order-id.
    :ivar status: ``'confirmed'`` / ``'rejected'`` / ``'still_unknown'``
        (echo of :class:`ResumeOutcome.status`) or ``'skipped'`` when
        the row's ``kind`` was not handled by this journal.
    :ivar reason: Free-form note. For ``'rejected'`` carries the
        plugin's reject reason; for ``'skipped'`` describes why the
        row was bypassed.
    """
    coid: str
    status: Literal['confirmed', 'rejected', 'still_unknown', 'skipped']
    reason: str | None = None


class PendingHooksProvider(Protocol):
    """Callable that maps a pending row to its recovery hook.

    The plugin implements this to decide which hook flavour applies
    to each ``extras['kind']`` — only ``'position'`` / ``'working'``
    are entry rows; future kinds (``'tp'``, ``'sl'``, ...) belong to
    their own journals.
    """

    def __call__(self, row: 'OrderRow') -> EntryDispatchHooks | None: ...


# === Private helpers =======================================================

def _collect_refs_for(store: 'RunContext', *, coid: str) -> Mapping[str, str]:
    """Materialise the full ``order_refs`` map for one COID.

    Required for the narrow crash window inside
    :func:`~pynecore.core.broker.store_helpers.record_server_ref`:
    between the ``add_ref('deal_reference', ...)`` commit and the
    subsequent ``upsert_order(extras={...})`` commit the
    ``deal_reference`` lives *only* in ``order_refs``. A resume hook
    that relied solely on ``row.extras`` would miss it and treat the
    already-submitted order as never posted, causing a duplicate POST
    on the next restart.

    The result is a plain ``dict``; multiple refs of the same type are
    not expected per COID and the last one wins if they ever occur.
    """
    return dict(store.iter_refs_for_coid(coid))
