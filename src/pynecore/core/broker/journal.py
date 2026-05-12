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
from typing import Any, Literal, Protocol, TYPE_CHECKING

from pynecore.core.broker.exceptions import (
    ExchangeOrderRejectedError,
    OrderDispositionUnknownError,
)
from pynecore.core.broker.models import (
    CancelIntent,
    CloseIntent,
    EntryIntent,
    ExchangeOrder,
    ExitIntent,
    OrderStatus,
    OrderType,
)
from pynecore.core.broker.store_helpers import (
    KIND_CANCEL,
    KIND_FULL_CLOSE,
    KIND_MODIFY_ENTRY,
    KIND_MODIFY_EXIT,
    KIND_PARTIAL_CLOSE,
    create_cancel_command_row,
    create_close_target_row,
    create_entry_order_row,
    create_modify_entry_row,
    create_modify_exit_row,
    find_pending_dispatch,
    mark_cancel_completed,
    mark_close_completed,
    mark_closing,
    mark_confirmed_with_fill,
    mark_disposition_unknown,
    mark_modify_completed,
    mark_rejected,
    record_close_server_ref,
    record_server_ref,
)

if TYPE_CHECKING:
    from pynecore.core.broker.storage import OrderRow, RunContext

__all__ = [
    'DispatchJournal',
    'EntryDispatchHooks',
    'CloseDispatchHooks',
    'CancelDispatchHooks',
    'ModifyEntryDispatchHooks',
    'ModifyExitDispatchHooks',
    'SubmitOutcome',
    'ConfirmOutcome',
    'ResumeOutcome',
    'ResumeStatus',
    'CloseOutcome',
    'CancelOutcome',
    'CancelReasonPath',
    'ModifyEntryOutcome',
    'ModifyExitOutcome',
    'ModifyExitStatus',
    'PendingResolution',
    'PendingHooksProvider',
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


CancelReasonPath = Literal['deleted', 'already_gone', 'noop']
ModifyExitStatus = Literal['ACCEPTED', 'REJECTED']


@dataclass(frozen=True)
class CloseOutcome:
    """Plugin's successful close-dispatch result.

    Returned from the plugin's ``submit_full_close`` or
    ``submit_partial_close`` hook. On a confirm-REJECTED outcome the
    hook raises
    :class:`~pynecore.core.broker.exceptions.ExchangeOrderRejectedError`;
    on a network / disposition-unknown outcome it raises
    :class:`~pynecore.core.broker.exceptions.OrderDispositionUnknownError`.

    :ivar mode: ``'full'`` for a full-close DELETE chain, ``'partial'``
        for the partial-close emulated POST. The journal also receives
        the ``kind`` argument up-front; the field is echoed here so a
        single outcome shape covers both branches.
    :ivar applied_targets: Exchange ``dealId`` strings the dispatch
        actually touched. Full close: every target the DELETE chain
        completed against. Partial close: a single-element list with
        the newly-opened opposite leg's ``dealId``, or empty if the
        POST returned no id.
    :ivar deal_reference: Server-allocated POST reference. Only the
        partial-close branch carries one; full-close returns ``None``.
    :ivar exchange_id: Single representative exchange id for the
        :class:`ExchangeOrder` the engine receives. For full close
        this is the first target's ``dealId``; for partial close this
        is the new opposite leg's ``dealId``.
    :ivar filled_qty: Quantity reported as filled by the broker
        response (or synthesised from the intent for the full-close
        synchronous flow).
    :ivar fill_price: Confirm-side price when known. ``None`` when
        the broker did not echo a fill price.
    :ivar raw: Verbatim broker response for forensics.
    """
    mode: Literal['full', 'partial']
    applied_targets: list[str]
    deal_reference: str | None = None
    exchange_id: str | None = None
    filled_qty: float = 0.0
    fill_price: float | None = None
    raw: dict | None = None


@dataclass(frozen=True)
class CancelOutcome:
    """Plugin's successful cancel-dispatch result.

    :ivar succeeded: ``True`` once the per-target sweep finished
        (including the benign already-gone path). ``False`` is
        currently unused — cancel failures raise rather than return.
    :ivar reason_path: Why the cancel resolved this way:

        - ``'deleted'`` — at least one target was actively swept by
          the dispatch.
        - ``'already_gone'`` — every target had already vanished from
          the broker; nothing was DELETEd.
        - ``'noop'`` — no targets matched the intent at all.

    :ivar cleared_legs: Number of bracket / working-order legs the
        dispatch swept (``len(applied_target_coids)``).
    :ivar applied_target_coids: Plugin-side COIDs the dispatch closed.
        Persisted into ``extras['applied_target_coids']`` so recovery
        can reason about which targets the per-target loop actually
        reached before any crash.
    :ivar raw: Verbatim broker response (or aggregated responses) for
        forensics.
    """
    succeeded: bool
    reason_path: CancelReasonPath
    cleared_legs: int
    applied_target_coids: list[str]
    raw: dict | None = None


@dataclass(frozen=True)
class ModifyEntryOutcome:
    """Plugin's successful working-order amend result.

    :ivar server_ref: ``dealReference`` of the amend PUT. Persisted
        under ``order_refs['deal_reference']`` so recovery can verify
        the change landed via a confirm GET.
    :ivar new_level: Echoed back from the confirm response — the
        broker's view of the amended level. Compared against the
        intent's requested level on recovery to detect drift.
    :ivar raw: Verbatim confirm response for forensics.
    """
    server_ref: str
    new_level: float
    raw: dict | None = None


@dataclass(frozen=True)
class ModifyExitOutcome:
    """Plugin's successful position bracket amend result.

    :ivar server_ref: ``dealReference`` of the amend PUT.
    :ivar deal_status: ``'ACCEPTED'`` for happy path,
        ``'REJECTED'`` is converted to
        :class:`ExchangeOrderRejectedError` at the hook boundary so
        the journal can persist the rejection; the field exists so
        forensics see the broker's exact verdict.
    :ivar rejected_reason: Free-form reject reason copied from the
        confirm response when ``deal_status == 'REJECTED'``.
    :ivar post_put_state: Mapping of the broker-echoed levels the
        plugin's ``mirror_bracket_legs`` callback needs to materialise
        the synthetic leg rows post-success. Keys are plugin-defined
        (e.g. ``'profit_level'``, ``'stop_level'``,
        ``'trailing_stop'``).
    :ivar raw: Verbatim confirm response for forensics.
    """
    server_ref: str
    deal_status: ModifyExitStatus
    rejected_reason: str | None = None
    post_put_state: Mapping[str, Any] = field(default_factory=dict)
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
    :ivar recovery_path: Plugin-defined category for the resolution
        route — e.g. ``'stored_ref'``, ``'activity_single_match'``,
        ``'ttl_fallback_snapshot'``, ``'confirm_get_direct'``. Merged
        into the ``recovered_*`` / ``recovery_pending`` audit event
        payload when set. The Core does not validate the value; the
        plugin owns the taxonomy.
    :ivar recovery_context: Structured plugin diagnostic that goes
        alongside ``recovery_path`` in the audit event (e.g.
        ``{'matched_snapshot': 'working', 'activity_count': 1}``).
    """
    status: ResumeStatus
    exchange_id: str | None = None
    is_filled: bool = False
    filled_qty: float = 0.0
    fill_price: float | None = None
    reject_reason: str | None = None
    recovery_path: str | None = None
    recovery_context: Mapping[str, Any] | None = None


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


class CloseDispatchHooks(Protocol):
    """Plugin-supplied callbacks for a close dispatch.

    Exactly one of :meth:`submit_full_close` and
    :meth:`submit_partial_close` is invoked per dispatch, decided by
    the ``kind`` argument the journal receives. The other method is
    not required to do anything meaningful — implementations typically
    raise ``RuntimeError`` from the unused one as a defensive marker.
    """

    async def submit_full_close(
            self, *, coid: str, intent: CloseIntent,
            targets: list['OrderRow'],
    ) -> CloseOutcome:
        """DELETE every target position. Synchronous fill.

        ``targets`` are the live position rows the dispatch must
        close. Implementations issue one DELETE per target and return
        a :class:`CloseOutcome` with ``mode='full'`` and
        ``applied_targets`` listing the ``dealId`` of every
        successfully DELETEd position. On a benign already-gone race
        (404) targets may be omitted from ``applied_targets``; the
        recovery contract treats vanished targets as confirmed.

        Implementations MUST NOT mutate the close command row's state
        — only the journal does that. They MAY mutate the *target*
        rows (e.g. ``store.set_order_state(target_coid, 'closing')``)
        because those rows live outside the journal's command-row
        scope.

        Network / timeout errors raise
        :class:`OrderDispositionUnknownError`; explicit broker rejects
        raise :class:`ExchangeOrderRejectedError`.
        """
        ...

    async def submit_partial_close(
            self, *, coid: str, intent: CloseIntent,
    ) -> CloseOutcome:
        """Emulated partial close via opposite-direction POST.

        Implementations issue a single POST (Capital.com has no native
        partial-close endpoint), record the ``dealReference``, and
        reconcile pre/post position snapshots to detect any race
        against an unrelated opposite-side opening. Returns a
        :class:`CloseOutcome` with ``mode='partial'``,
        ``deal_reference`` populated, ``exchange_id`` set to the new
        opposite-leg ``dealId``, and ``applied_targets`` listing that
        single ``dealId``. The hook itself raises
        :class:`BrokerManualInterventionError` on an unresolved race;
        the journal does not catch that — it propagates to the engine.
        """
        ...

    def exchange_order_from_state(
            self, *, row: 'OrderRow', intent: CloseIntent,
            outcome: CloseOutcome,
    ) -> ExchangeOrder:
        """Build the :class:`ExchangeOrder` the engine expects.

        Called once the command row has reached its terminal state
        (``closing`` for full, ``confirmed`` for partial). Pure
        function — no I/O.
        """
        ...


class CancelDispatchHooks(Protocol):
    """Plugin-supplied callbacks for a cancel dispatch.

    Cancel has no submit/confirm split — the plugin sweeps all targets
    in a single call and returns the outcome. The journal owns the
    command-row state transitions; the plugin owns the per-target
    REST operations and the target-row mutations.
    """

    async def submit_cancel(
            self, *, coid: str, intent: CancelIntent,
            targets: list['OrderRow'],
    ) -> CancelOutcome:
        """Sweep every target. Returns a single :class:`CancelOutcome`.

        Implementations issue the per-target REST calls (DELETE for
        working orders, PUT-null for bracket legs) and mark the
        target rows closed via ``store.close_order(target_coid)``.
        Benign already-gone (404) responses are absorbed without
        raising — the resulting :attr:`CancelOutcome.reason_path`
        reflects whether any actual DELETE happened.

        Implementations MUST NOT mutate the cancel command row's
        state — only the journal does that.
        """
        ...

    def exchange_order_from_state(
            self, *, row: 'OrderRow', intent: CancelIntent,
            outcome: CancelOutcome,
    ) -> ExchangeOrder:
        """Build the synthetic :class:`ExchangeOrder` for the cancel.

        Cancel does not produce an exchange order per se, but the
        engine signature expects one. The hook synthesises a
        :class:`OrderStatus.CANCELLED` order so the caller can plumb
        the outcome through unchanged channels.
        """
        ...


class ModifyEntryDispatchHooks(Protocol):
    """Plugin-supplied callbacks for a working-order amend dispatch."""

    async def submit_amend(
            self, *, coid: str, target_coid: str,
            old_intent: EntryIntent, new_intent: EntryIntent,
    ) -> ModifyEntryOutcome:
        """PUT the new level and confirm.

        Returns a :class:`ModifyEntryOutcome` with the broker-echoed
        ``new_level``. On reject raises
        :class:`ExchangeOrderRejectedError`; on timeout raises
        :class:`OrderDispositionUnknownError`. The amend target row
        (the working order itself) is mutated by the hook via
        ``store.upsert_order(target_coid, ...)`` because it lives
        outside the journal's command-row scope.
        """
        ...

    def exchange_order_from_state(
            self, *, row: 'OrderRow', new_intent: EntryIntent,
            outcome: ModifyEntryOutcome,
    ) -> list[ExchangeOrder]:
        """Build the :class:`ExchangeOrder` list for the engine.

        The engine's :meth:`modify_entry` signature returns a list of
        orders; for atomic amends this is a one-element list pointing
        at the same target as before. Pure function.
        """
        ...


class ModifyExitDispatchHooks(Protocol):
    """Plugin-supplied callbacks for a position bracket amend dispatch.

    Modify-exit is the most complex dispatch — the plugin's
    ``prepare()`` logic decides the new TP / SL / trailing levels and
    seeds any newly-added bracket leg rows in
    ``disposition_unknown``. The journal then drives the entry-row
    audit trail and the ``mirror_bracket_legs`` callback that
    materialises the synthetic legs on success.
    """

    async def submit_amend(
            self, *, coid: str, target_coid: str,
            old_intent: ExitIntent, new_intent: ExitIntent,
    ) -> ModifyExitOutcome:
        """PUT the new bracket and confirm.

        On the happy path returns a :class:`ModifyExitOutcome` with
        ``deal_status='ACCEPTED'`` and ``post_put_state`` filled.
        On reject raises :class:`ExchangeOrderRejectedError`; on
        timeout raises :class:`OrderDispositionUnknownError`. Before
        raising, the hook flips any leg rows it pre-seeded into the
        appropriate disposition-unknown state and persists the
        attempted target levels under the leg rows' extras — those
        side-channel writes are part of the hook's responsibility.
        """
        ...

    def mirror_bracket_legs(
            self, *, target_row: 'OrderRow', new_intent: ExitIntent,
            outcome: ModifyExitOutcome,
    ) -> None:
        """Materialise synthetic TP / SL leg rows after success.

        Invoked by the journal only on the happy path (after the
        entry-side command row has transitioned to ``confirmed``).
        On any ambiguous / reject path the journal does NOT call
        this hook; the disposition-unknown leg seeds the
        ``submit_amend`` hook already wrote remain the source of
        truth for recovery. Pure plugin-side write — uses
        ``store.upsert_order(leg_coid, ...)`` directly because the
        synthetic leg rows are outside the journal's command-row
        scope for M4.
        """
        ...

    def exchange_order_from_state(
            self, *, row: 'OrderRow', new_intent: ExitIntent,
            outcome: ModifyExitOutcome,
    ) -> list[ExchangeOrder]:
        """Build the engine-facing :class:`ExchangeOrder` list.

        Returns one :class:`ExchangeOrder` per active bracket leg
        (TP / SL) reflecting the post-amend levels. Pure function.
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

    # --- Close path --------------------------------------------------------

    async def run_close(
            self,
            *,
            coid: str,
            intent: CloseIntent,
            kind: str,
            targets: list['OrderRow'],
            hooks: CloseDispatchHooks,
            audit_payload: dict | None = None,
    ) -> ExchangeOrder:
        """Run a close dispatch.

        Routes between full-close (DELETE chain) and partial-close
        (emulated POST) based on ``kind``. Each branch persists the
        command row, calls the matching hook, and finalises the row.
        Target-row mutations live in the hook, since they are outside
        the command-row state-machine the journal owns.

        :param coid: Close dispatch COID.
        :param intent: The :class:`CloseIntent` being dispatched.
        :param kind: :data:`KIND_FULL_CLOSE` or
            :data:`KIND_PARTIAL_CLOSE`.
        :param targets: Live position rows the dispatch should close.
            For partial close this is the pre-existing rows (used by
            the hook to derive the pre-snapshot delta), for full close
            this drives the DELETE loop.
        :param hooks: Plugin callbacks.
        :param audit_payload: Optional extras to merge into the
            initial ``dispatch_submitted`` event payload.
        """
        if kind == KIND_FULL_CLOSE:
            return await self._run_full_close(
                coid=coid, intent=intent, targets=targets,
                hooks=hooks, audit_payload=audit_payload,
            )
        if kind == KIND_PARTIAL_CLOSE:
            return await self._run_partial_close(
                coid=coid, intent=intent,
                hooks=hooks, audit_payload=audit_payload,
            )
        raise ValueError(
            f"DispatchJournal.run_close: kind must be one of "
            f"{{KIND_FULL_CLOSE, KIND_PARTIAL_CLOSE}}, got {kind!r}"
        )

    async def _run_full_close(
            self,
            *,
            coid: str,
            intent: CloseIntent,
            targets: list['OrderRow'],
            hooks: CloseDispatchHooks,
            audit_payload: dict | None,
    ) -> ExchangeOrder:
        # (1) PERSIST command row + audit event.
        target_ids = [r.exchange_order_id for r in targets]
        create_close_target_row(
            self.store,
            coid=coid,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            intent_key=intent.intent_key,
            kind=KIND_FULL_CLOSE,
            extra_payload={'targets': list(target_ids)},
        )
        submit_payload: dict[str, Any] = {
            'kind': KIND_FULL_CLOSE,
            'targets': target_ids,
        }
        if audit_payload:
            submit_payload.update(audit_payload)
        self.store.log_event(
            'dispatch_submitted',
            client_order_id=coid,
            intent_key=intent.intent_key,
            payload=submit_payload,
        )

        # (2) SUBMIT — per-target DELETE chain inside the hook.
        try:
            outcome = await hooks.submit_full_close(
                coid=coid, intent=intent, targets=targets,
            )
        except OrderDispositionUnknownError as exc:
            mark_disposition_unknown(self.store, coid=coid)
            self.store.log_event(
                'disposition_unknown',
                client_order_id=coid,
                intent_key=intent.intent_key,
                payload={'phase': 'full_close_delete', 'reason': str(exc)},
            )
            raise
        except ExchangeOrderRejectedError as exc:
            mark_rejected(self.store, coid=coid)
            self.store.log_event(
                'rejected',
                client_order_id=coid,
                intent_key=intent.intent_key,
                payload={'phase': 'full_close_delete', 'reason': str(exc)},
            )
            raise

        # (3) PERSIST closing state + targets.
        mark_closing(
            self.store,
            coid=coid,
            kind=KIND_FULL_CLOSE,
            targets=outcome.applied_targets,
        )
        self.store.log_event(
            'close_dispatched',
            client_order_id=coid,
            intent_key=intent.intent_key,
            payload={'mode': 'full', 'applied_targets': outcome.applied_targets},
        )

        # (4) Return the synthetic ExchangeOrder built by the hook.
        row = self.store.get_order(coid)
        if row is None:
            raise RuntimeError(
                f"DispatchJournal._run_full_close: row vanished after closing "
                f"(coid={coid!r})"
            )
        return hooks.exchange_order_from_state(
            row=row, intent=intent, outcome=outcome,
        )

    async def _run_partial_close(
            self,
            *,
            coid: str,
            intent: CloseIntent,
            hooks: CloseDispatchHooks,
            audit_payload: dict | None,
    ) -> ExchangeOrder:
        # (1) PERSIST command row + audit event.
        create_close_target_row(
            self.store,
            coid=coid,
            symbol=intent.symbol,
            side=intent.side,
            qty=intent.qty,
            intent_key=intent.intent_key,
            kind=KIND_PARTIAL_CLOSE,
        )
        submit_payload: dict[str, Any] = {'kind': KIND_PARTIAL_CLOSE}
        if audit_payload:
            submit_payload.update(audit_payload)
        self.store.log_event(
            'dispatch_submitted',
            client_order_id=coid,
            intent_key=intent.intent_key,
            payload=submit_payload,
        )

        # (2) SUBMIT — opposite-direction POST + race detection inside the hook.
        try:
            outcome = await hooks.submit_partial_close(
                coid=coid, intent=intent,
            )
        except OrderDispositionUnknownError as exc:
            mark_disposition_unknown(self.store, coid=coid)
            self.store.log_event(
                'disposition_unknown',
                client_order_id=coid,
                intent_key=intent.intent_key,
                payload={'phase': 'partial_close_post', 'reason': str(exc)},
            )
            raise
        except ExchangeOrderRejectedError as exc:
            mark_rejected(self.store, coid=coid)
            self.store.log_event(
                'rejected',
                client_order_id=coid,
                intent_key=intent.intent_key,
                payload={'phase': 'partial_close_post', 'reason': str(exc)},
            )
            raise

        # (3) PERSIST server ref (if any).
        if outcome.deal_reference is not None:
            record_close_server_ref(
                self.store,
                coid=coid,
                deal_reference=outcome.deal_reference,
                kind=KIND_PARTIAL_CLOSE,
            )
            self.store.log_event(
                'deal_reference_seen',
                client_order_id=coid,
                payload={'deal_reference': outcome.deal_reference},
            )

        # (4) PERSIST completion. The helper only advances state;
        # ``close_order`` is issued *after* the ``confirmed`` event so
        # the audit order is consistent.
        mark_close_completed(
            self.store,
            coid=coid,
            kind=KIND_PARTIAL_CLOSE,
        )
        self.store.log_event(
            'confirmed',
            client_order_id=coid,
            exchange_order_id=outcome.exchange_id,
            intent_key=intent.intent_key,
            payload={
                'mode': 'partial',
                'applied_targets': outcome.applied_targets,
                'fill_price': outcome.fill_price,
            },
        )
        self.store.close_order(coid)

        # (5) Return the ExchangeOrder built by the hook.
        row = self.store.get_order(coid)
        if row is None:
            raise RuntimeError(
                f"DispatchJournal._run_partial_close: row vanished after confirm "
                f"(coid={coid!r})"
            )
        return hooks.exchange_order_from_state(
            row=row, intent=intent, outcome=outcome,
        )

    # --- Cancel path -------------------------------------------------------

    async def run_cancel(
            self,
            *,
            coid: str,
            intent: CancelIntent,
            targets: list['OrderRow'],
            hooks: CancelDispatchHooks,
            audit_payload: dict | None = None,
    ) -> ExchangeOrder:
        """Run a cancel dispatch.

        The journal persists the command row, calls the hook for the
        per-target sweep, and finalises the row with the
        ``reason_path`` from the outcome. The per-target REST calls
        and target-row mutations are owned by the hook.

        :param coid: Cancel dispatch COID.
        :param intent: The :class:`CancelIntent` being dispatched.
        :param targets: Live rows the dispatch should cancel.
        :param hooks: Plugin callbacks.
        :param audit_payload: Optional extras for the
            ``dispatch_submitted`` event payload.
        """
        target_coids = [r.client_order_id for r in targets]
        agg_qty = sum(max(0.0, r.qty - r.filled_qty) for r in targets)
        primary_side = targets[0].side if targets else 'buy'

        # (1) PERSIST command row + audit event.
        create_cancel_command_row(
            self.store,
            coid=coid,
            symbol=intent.symbol,
            side=primary_side,
            qty=agg_qty,
            intent_key=intent.intent_key,
            pine_entry_id=intent.pine_id,
            from_entry=intent.from_entry,
            target_coids=target_coids,
        )
        submit_payload: dict[str, Any] = {
            'kind': KIND_CANCEL,
            'target_coids': target_coids,
        }
        if audit_payload:
            submit_payload.update(audit_payload)
        self.store.log_event(
            'dispatch_submitted',
            client_order_id=coid,
            intent_key=intent.intent_key,
            payload=submit_payload,
        )

        # (2) SUBMIT — per-target sweep inside the hook.
        try:
            outcome = await hooks.submit_cancel(
                coid=coid, intent=intent, targets=targets,
            )
        except OrderDispositionUnknownError as exc:
            mark_disposition_unknown(self.store, coid=coid)
            self.store.log_event(
                'disposition_unknown',
                client_order_id=coid,
                intent_key=intent.intent_key,
                payload={'phase': 'cancel_sweep', 'reason': str(exc)},
            )
            raise
        except ExchangeOrderRejectedError as exc:
            mark_rejected(self.store, coid=coid)
            self.store.log_event(
                'rejected',
                client_order_id=coid,
                intent_key=intent.intent_key,
                payload={'phase': 'cancel_sweep', 'reason': str(exc)},
            )
            raise

        # (3) PERSIST completion with reason_path.
        mark_cancel_completed(
            self.store,
            coid=coid,
            reason_path=outcome.reason_path,
            extra_payload={
                'applied_target_coids': outcome.applied_target_coids,
            },
        )
        self.store.log_event(
            'confirmed',
            client_order_id=coid,
            intent_key=intent.intent_key,
            payload={
                'reason_path': outcome.reason_path,
                'cleared_legs': outcome.cleared_legs,
                'applied_target_coids': outcome.applied_target_coids,
            },
        )
        self.store.close_order(coid)

        # (4) Return the synthetic ExchangeOrder built by the hook.
        row = self.store.get_order(coid)
        if row is None:
            raise RuntimeError(
                f"DispatchJournal.run_cancel: row vanished after confirm "
                f"(coid={coid!r})"
            )
        return hooks.exchange_order_from_state(
            row=row, intent=intent, outcome=outcome,
        )

    # --- Modify entry path -------------------------------------------------

    async def run_modify_entry(
            self,
            *,
            coid: str,
            target_coid: str,
            old_intent: EntryIntent,
            new_intent: EntryIntent,
            qty: float,
            hooks: ModifyEntryDispatchHooks,
            audit_payload: dict | None = None,
    ) -> list[ExchangeOrder]:
        """Run an atomic working-order amend dispatch.

        :param coid: COID of the amend command row.
        :param target_coid: COID of the working order being amended.
        :param old_intent: Intent before the amend (for audit context).
        :param new_intent: Intent the broker should land.
        :param qty: Order quantity (unchanged across the amend).
        :param hooks: Plugin callbacks.
        :param audit_payload: Optional extras for the
            ``dispatch_submitted`` event payload.
        """
        # (1) PERSIST command row + audit event.
        new_level = float(new_intent.limit if new_intent.limit is not None
                          else new_intent.stop or 0.0)
        create_modify_entry_row(
            self.store,
            coid=coid,
            target_coid=target_coid,
            symbol=new_intent.symbol,
            side=new_intent.side,
            qty=qty,
            intent_key=new_intent.intent_key,
            new_level=new_level,
            pine_entry_id=new_intent.pine_id,
        )
        submit_payload: dict[str, Any] = {
            'kind': KIND_MODIFY_ENTRY,
            'target_coid': target_coid,
            'new_level': new_level,
        }
        if audit_payload:
            submit_payload.update(audit_payload)
        self.store.log_event(
            'dispatch_submitted',
            client_order_id=coid,
            intent_key=new_intent.intent_key,
            payload=submit_payload,
        )

        # (2) SUBMIT — PUT + confirm inside the hook.
        try:
            outcome = await hooks.submit_amend(
                coid=coid, target_coid=target_coid,
                old_intent=old_intent, new_intent=new_intent,
            )
        except OrderDispositionUnknownError as exc:
            mark_disposition_unknown(self.store, coid=coid)
            self.store.log_event(
                'disposition_unknown',
                client_order_id=coid,
                intent_key=new_intent.intent_key,
                payload={'phase': 'modify_entry_put', 'reason': str(exc)},
            )
            raise
        except ExchangeOrderRejectedError as exc:
            mark_rejected(self.store, coid=coid)
            self.store.log_event(
                'rejected',
                client_order_id=coid,
                intent_key=new_intent.intent_key,
                payload={'phase': 'modify_entry_put', 'reason': str(exc)},
            )
            raise

        # (3) PERSIST server ref + completion.
        self.store.add_ref(coid, 'deal_reference', outcome.server_ref)
        self.store.log_event(
            'deal_reference_seen',
            client_order_id=coid,
            payload={'deal_reference': outcome.server_ref},
        )
        mark_modify_completed(
            self.store,
            coid=coid,
            extra_payload={'echoed_level': outcome.new_level},
        )
        self.store.log_event(
            'confirmed',
            client_order_id=coid,
            intent_key=new_intent.intent_key,
            payload={'new_level': outcome.new_level},
        )
        self.store.close_order(coid)

        # (4) Return the engine-facing list.
        row = self.store.get_order(coid)
        if row is None:
            raise RuntimeError(
                f"DispatchJournal.run_modify_entry: row vanished after confirm "
                f"(coid={coid!r})"
            )
        return hooks.exchange_order_from_state(
            row=row, new_intent=new_intent, outcome=outcome,
        )

    # --- Modify exit path --------------------------------------------------

    async def run_modify_exit(
            self,
            *,
            coid: str,
            target_coid: str,
            target_row: 'OrderRow',
            old_intent: ExitIntent,
            new_intent: ExitIntent,
            qty: float,
            hooks: ModifyExitDispatchHooks,
            audit_payload: dict | None = None,
    ) -> list[ExchangeOrder]:
        """Run a position bracket amend dispatch.

        The journal owns the entry-side command row state machine and
        invokes :meth:`mirror_bracket_legs` on the happy path. The
        synthetic leg rows themselves are written by the hook (both
        the disposition-unknown seeds in the ambiguous path and the
        confirmed leg rows in the mirror path).

        :param coid: COID of the amend command row.
        :param target_coid: COID of the entry row representing the
            position being amended.
        :param target_row: Live row of the target entry (passed to the
            mirror callback).
        :param old_intent: Intent before the amend.
        :param new_intent: Intent the broker should land.
        :param qty: Position quantity (unchanged across the amend).
        :param hooks: Plugin callbacks.
        :param audit_payload: Optional extras for the
            ``dispatch_submitted`` event payload.
        """
        # (1) PERSIST command row + audit event.
        create_modify_exit_row(
            self.store,
            coid=coid,
            target_coid=target_coid,
            symbol=new_intent.symbol,
            side=new_intent.side,
            qty=qty,
            intent_key=new_intent.intent_key,
            new_tp=new_intent.tp_price,
            new_sl=new_intent.sl_price,
            new_trail=new_intent.trail_offset,
            pine_entry_id=new_intent.pine_id,
            from_entry=new_intent.from_entry,
        )
        submit_payload: dict[str, Any] = {
            'kind': KIND_MODIFY_EXIT,
            'target_coid': target_coid,
            'new_tp': new_intent.tp_price,
            'new_sl': new_intent.sl_price,
            'new_trail': new_intent.trail_offset,
        }
        if audit_payload:
            submit_payload.update(audit_payload)
        self.store.log_event(
            'dispatch_submitted',
            client_order_id=coid,
            intent_key=new_intent.intent_key,
            payload=submit_payload,
        )

        # (2) SUBMIT — PUT + confirm inside the hook; ambiguous-path
        # leg seeding is the hook's responsibility before re-raising.
        try:
            outcome = await hooks.submit_amend(
                coid=coid, target_coid=target_coid,
                old_intent=old_intent, new_intent=new_intent,
            )
        except OrderDispositionUnknownError as exc:
            mark_disposition_unknown(self.store, coid=coid)
            self.store.log_event(
                'disposition_unknown',
                client_order_id=coid,
                intent_key=new_intent.intent_key,
                payload={'phase': 'modify_exit_put', 'reason': str(exc)},
            )
            raise
        except ExchangeOrderRejectedError as exc:
            mark_rejected(self.store, coid=coid)
            self.store.log_event(
                'rejected',
                client_order_id=coid,
                intent_key=new_intent.intent_key,
                payload={'phase': 'modify_exit_put', 'reason': str(exc)},
            )
            raise

        # (3) PERSIST server ref + completion.
        self.store.add_ref(coid, 'deal_reference', outcome.server_ref)
        self.store.log_event(
            'deal_reference_seen',
            client_order_id=coid,
            payload={'deal_reference': outcome.server_ref},
        )
        mark_modify_completed(
            self.store,
            coid=coid,
            extra_payload={'post_put_state': dict(outcome.post_put_state)},
        )
        self.store.log_event(
            'confirmed',
            client_order_id=coid,
            intent_key=new_intent.intent_key,
            payload={
                'deal_status': outcome.deal_status,
                'post_put_state': dict(outcome.post_put_state),
            },
        )
        self.store.close_order(coid)

        # (4) MIRROR bracket legs (happy path only).
        hooks.mirror_bracket_legs(
            target_row=target_row, new_intent=new_intent, outcome=outcome,
        )

        # (5) Return the engine-facing list.
        row = self.store.get_order(coid)
        if row is None:
            raise RuntimeError(
                f"DispatchJournal.run_modify_exit: row vanished after confirm "
                f"(coid={coid!r})"
            )
        return hooks.exchange_order_from_state(
            row=row, new_intent=new_intent, outcome=outcome,
        )

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
            payload: dict[str, Any] = {
                'is_filled': outcome.is_filled,
                'fill_price': outcome.fill_price,
                'prior_state': row.state,
            }
            _merge_recovery_diagnostics(payload, outcome)
            self.store.log_event(
                'recovered_confirmed',
                client_order_id=row.client_order_id,
                exchange_order_id=outcome.exchange_id,
                intent_key=row.intent_key,
                payload=payload,
            )
        elif outcome.status == 'rejected':
            mark_rejected(self.store, coid=row.client_order_id)
            payload = {
                'reason': outcome.reject_reason,
                'prior_state': row.state,
            }
            _merge_recovery_diagnostics(payload, outcome)
            self.store.log_event(
                'recovered_rejected',
                client_order_id=row.client_order_id,
                intent_key=row.intent_key,
                payload=payload,
            )
        else:
            # still_unknown — keep the row; engine reconciler retries.
            payload = {'prior_state': row.state}
            _merge_recovery_diagnostics(payload, outcome)
            self.store.log_event(
                'recovery_pending',
                client_order_id=row.client_order_id,
                intent_key=row.intent_key,
                payload=payload,
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

def _merge_recovery_diagnostics(
        payload: dict[str, Any], outcome: ResumeOutcome,
) -> None:
    """Inject ``recovery_path`` / ``recovery_context`` into an event payload.

    Both fields are optional plugin-supplied diagnostics; when ``None``
    they are not written at all so the on-disk audit shape stays
    minimal for hooks that do not bother annotating the route.
    """
    if outcome.recovery_path is not None:
        payload['recovery_path'] = outcome.recovery_path
    if outcome.recovery_context is not None:
        payload['recovery_context'] = dict(outcome.recovery_context)


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
