"""Generic bot-owned-order disappearance tracking for broker plugins.

Detecting bot-owned orders that vanish behind the engine's back (manual
close on the broker UI, broker-side liquidation, silent cancel) is a
plugin responsibility, but its core state machine is venue-agnostic and
was implemented twice (Capital.com and cTrader). This module extracts
that core:

- **Persisted stamp / clear / grace state machine.** A row whose broker
  counterpart has vanished from every tracked namespace gets a
  ``missing_pending_since`` stamp in its ``extras``; the stamp is cleared
  the moment any tracked ref reappears. Only a stamp older than the grace
  window triggers any irreversible action — a fill in flight can flicker
  out of every snapshot for one poll. Stamp and clear both re-read the
  live row and touch *only* the ``missing_pending_since`` key, so a
  breadcrumb written concurrently by another broker thread is never
  clobbered.
- **Typed ``(namespace, ref)`` keys.** A venue stores bot orders in
  several resource namespaces (working orders, open positions,
  position-attached brackets); refs are tracked per namespace so an
  order-id can never collide with a position-id.
- **Per-namespace snapshot completeness.** A namespace whose fetch failed
  this pass is reported as ``None``: rows tracked in it are neither
  stamped nor cleared — an incomplete snapshot must never look like a
  complete absence.
- **Declarative grace-expiry classification.** At grace expiry the venue
  hook ``confirm_missing`` re-verifies the disappearance (e.g. against
  the deal history) and returns a :class:`MissingConfirmation` — it does
  NOT mutate. The tracker applies the outcome in ONE serialized store
  transaction: discovered fill slice + terminal state + sibling
  retirement together, so a partial-fill-then-cancel can never lose its
  fill slice, and a crash cannot leave a half-applied resolution. Every
  post-apply artefact (the cancelled event, the policy hook, the
  execution registration) is built from the row re-read *after* the fill
  slice was booked, so none of them can carry stale pre-fill quantities.
- **Stamp-version guard.** The row can come back (or be re-stamped)
  while the async confirmation runs; the apply transaction re-checks the
  original ``missing_pending_since`` value and drops a stale outcome.
- **Fail-closed on unpriced fills.** A discovered fill slice is only
  booked when it carries a strictly positive price and a strictly
  positive persisted quantity delta. An unpriced fill would be dropped by
  the engine's ``record_fill`` yet still advance the store's
  ``filled_qty`` and seed the plugin's dedup channel — so the tracker
  keeps the stamp and defers instead of concluding on unpriced evidence.
- **Dual signal.** A confirmed unexpected cancel is booked as a terminal
  close AND yielded as a synthesised ``cancelled`` :class:`OrderEvent`
  (the engine's router cleans its tracking — essential under the
  non-halting policies), while the configured ``on_unexpected_cancel``
  policy separately decides the operational reaction. ``stop``
  QUARANTINES through the ``request_quarantine`` hook as an external
  intervention: the engine stops writing to the venue altogether, but the
  process — and this tracker's ingestion loop — stays alive;
  raising here instead would tear down the very event stream the
  quarantine invariant requires to keep running. Only the explicit
  ``halt`` policy (or a quarantining policy with no hook wired — the
  fail-safe fallback) arms the process-exiting halt. Persistence and
  policy application run BEFORE the event is yielded, so neither depends
  on the consumer pulling another element from the generator; a pending
  halt survives an abandoned generator and re-raises on the next
  :meth:`DisappearanceTracker.observe` call. The halt is delivered
  exactly once — each raise site consumes it.
- **Cancels before an operator resume are accepted.** A disappearance under
  investigation when the operator resumed the run, or a row's first absence
  after the resume (:func:`accept_missing_before_resume`), retires its row on
  a CANCELLED verdict with a ``cancel_accepted_by_resume`` audit event and no
  policy. The first absence is the one the presence feed stamps before any
  complete snapshot shows the row again, or, for a row the plugin stamps
  itself, a stamp it writes within :data:`RESUME_FIRST_ABSENCE_WINDOW_S` of
  the resume. A row the venue lists among its open orders after the resume
  (the sync engine's first open-orders read, :func:`note_seen_after_resume`)
  was still there then: its later cancel is judged as usual. A resume
  applied again after a restart keeps what the earlier process settled and
  restarts the window only for the rows it never judged
  (:func:`renew_resume_first_absence`). The sync engine gives a
  ``cancelled`` event the plugin reports itself the same verdict
  (:func:`resume_accepts_cancel`).
- **Venue-driven cancels are not interventions.** A confirmation that
  names a :data:`~pynecore.core.broker.models.VENUE_DRIVEN_CANCEL_REASONS`
  cause (the venue terminated the order itself, e.g. a reduce-only squeeze
  or an OCA sibling trigger whose push was lost) retires the row and
  yields the ``cancelled`` event carrying that reason, but never applies
  the ``on_unexpected_cancel`` policy: nobody else touched the order, so
  the engine trims an exit leg (its stop-coverage guard may re-arm it) or
  retires an entry / close order, exactly as for the same cancel pushed on
  the private stream.
"""
import logging
import math
import time
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Set,
)
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeGuard

from pynecore.core.broker.exceptions import UnexpectedCancelError
from pynecore.core.broker.journal import (
    DispatchJournal,
    ReconcileOutcome,
    ReconcileReason,
)
from pynecore.core.broker.models import (
    VENUE_DRIVEN_CANCEL_REASONS,
    ExchangeOrder,
    LegType,
    OrderEvent,
    OrderStatus,
    OrderType,
    QuarantineKind,
)

if TYPE_CHECKING:
    from pynecore.core.broker.storage import OrderRow, RunContext

__all__ = [
    'MISSING_ACCEPTED_BY_RESUME_EXTRA',
    'MISSING_PENDING_EXTRA',
    'RESUME_FIRST_ABSENCE_WINDOW_S',
    'UNEXPECTED_CANCEL_POLICIES',
    'DisappearanceTracker',
    'MissingConfirmation',
    'MissingResolution',
    'accept_missing_before_resume',
    'note_seen_after_resume',
    'renew_resume_first_absence',
    'resolve_unexpected_cancel_policy',
    'resume_accepts_cancel',
]

logger = logging.getLogger(__name__)

#: The persisted observation breadcrumb key in ``OrderRow.extras``. The
#: same key both reference plugins already use, so a refactor onto the
#: tracker inherits stamps persisted by earlier plugin versions.
MISSING_PENDING_EXTRA = 'missing_pending_since'

#: ``OrderRow.extras`` key holding the ``missing_pending_since`` stamp an
#: operator resume accepted: the disappearance under that stamp happened before
#: the resume, so a CANCELLED verdict on it retires the row without the
#: ``on_unexpected_cancel`` policy.
MISSING_ACCEPTED_BY_RESUME_EXTRA = 'missing_accepted_by_resume'

#: :data:`MISSING_ACCEPTED_BY_RESUME_EXTRA` value of a row that was live and not
#: stamped at an operator resume: its first absence after the resume, when no
#: complete snapshot showed it in between, is accepted too. A restarted
#: process cannot tell an order cancelled before the resume (while it was down)
#: from one cancelled right after it, and the operator accepted the venue as it
#: stood; the first complete snapshot that shows the row clears the mark, and so
#: does an open-orders read that lists it (:func:`note_seen_after_resume`).
_RESUME_ACCEPTS_NEXT_ABSENCE = 'next_absence'

#: ``OrderRow.extras`` key holding the resume time (unix seconds) of a row
#: marked :data:`_RESUME_ACCEPTS_NEXT_ABSENCE`, which bounds the acceptance of a
#: stamp the tracker did not write itself.
_MISSING_RESUMED_AT_EXTRA = 'missing_resumed_at'

#: Seconds after an operator resume within which a ``missing_pending_since``
#: stamp that a plugin wrote itself (not through the presence feed) still marks
#: the first absence the resume accepted. Such a plugin stamps on its own
#: evidence (a working order that left the venue with no fill in its history),
#: so the tracker never sees a complete snapshot show the row: the window covers
#: the restarted process's connect and its first reconcile passes, including a
#: few retries of an inconclusive history read. A later stamp is a fresh absence
#: and is judged as usual.
RESUME_FIRST_ABSENCE_WINDOW_S = 300.0

#: Valid ``on_unexpected_cancel`` policies (see ``BrokerDefaults``).
UNEXPECTED_CANCEL_POLICIES = (
    'stop', 're_place', 'ignore', 'halt',
)

#: Float comparison slack for fill quantities, matching the reconcile
#: paths in the reference plugins.
_QTY_EPS = 1e-9


def _is_fully_filled(row: 'OrderRow') -> bool:
    """``True`` when the row's cumulative fill consumed its full quantity."""
    return row.qty > 0.0 and row.filled_qty >= row.qty - _QTY_EPS


def _is_stamp(value: Any) -> TypeGuard[float]:
    """Whether an extras value is a stamp (unix seconds)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _accepted_by_resume(row: 'OrderRow', since: float) -> bool:
    """Whether an operator resume accepted the disappearance stamped at ``since``."""
    return (row.extras or {}).get(MISSING_ACCEPTED_BY_RESUME_EXTRA) == since


def _without_resume_mark(extras: Mapping[str, Any]) -> dict[str, Any]:
    """``extras`` without the resume acceptance and its resume time."""
    return {k: v for k, v in extras.items()
            if k not in (MISSING_ACCEPTED_BY_RESUME_EXTRA, _MISSING_RESUMED_AT_EXTRA)}


def accept_missing_before_resume(
        store_ctx: 'RunContext', *, resumed_at: float | None = None,
) -> int:
    """Accept every disappearance that happened before an operator resume.

    The operator resumed the run and so accepted the venue as it stands,
    including an order someone cancelled before the resume. A stamped row
    remembers the stamp it was accepted under; a row not stamped yet (the
    cancel happened while no process observed the venue) has its first
    absence after the resume accepted: the one the presence feed stamps before
    a complete snapshot shows the row again, or a stamp the plugin writes
    itself within :data:`RESUME_FIRST_ABSENCE_WINDOW_S` of ``resumed_at``. A
    later CANCELLED verdict for an accepted stamp retires the row with a
    ``cancel_accepted_by_resume`` audit event and no policy. A row that a
    complete snapshot shows again, or whose stamp the plugin cleared, loses
    the acceptance: when it vanishes later, it is stamped anew and judged as
    usual.

    :param store_ctx: The resumed run's journal.
    :param resumed_at: The time the resume was applied (unix seconds); now
        when omitted.
    :return: The number of live rows marked.
    """
    if resumed_at is None:
        resumed_at = time.time()
    accepted = 0
    with store_ctx.transaction():
        for row in store_ctx.iter_live_orders():
            extras = _without_resume_mark(row.extras or {})
            since = extras.get(MISSING_PENDING_EXTRA)
            if _is_stamp(since):
                extras[MISSING_ACCEPTED_BY_RESUME_EXTRA] = since
            else:
                extras[MISSING_ACCEPTED_BY_RESUME_EXTRA] = _RESUME_ACCEPTS_NEXT_ABSENCE
                extras[_MISSING_RESUMED_AT_EXTRA] = resumed_at
            store_ctx.upsert_order(row.client_order_id, extras=extras)
            accepted += 1
    return accepted


def note_seen_after_resume(store_ctx: 'RunContext', client_order_ids: Iterable[str]) -> int:
    """Drop the first-absence acceptance of rows the venue showed after the resume.

    A row marked for its first absence after an operator resume
    (:func:`accept_missing_before_resume`) was still on the venue when a read
    made after the resume listed it, so whatever removes it later happened
    after the resume and is judged as usual. A stamped absence the resume
    accepted is the tracker's to settle and is left alone.

    :param store_ctx: The resumed run's journal.
    :param client_order_ids: Client order ids of the orders the venue listed.
    :return: The number of rows whose acceptance was dropped.
    """
    seen = set(client_order_ids)
    if not seen:
        return 0
    dropped = 0
    with store_ctx.transaction():
        for row in store_ctx.iter_live_orders():
            if row.client_order_id not in seen:
                continue
            extras = row.extras or {}
            if (extras.get(MISSING_ACCEPTED_BY_RESUME_EXTRA) != _RESUME_ACCEPTS_NEXT_ABSENCE
                    or _is_stamp(extras.get(MISSING_PENDING_EXTRA))):
                continue
            store_ctx.upsert_order(row.client_order_id, extras=_without_resume_mark(extras))
            dropped += 1
    return dropped


def resume_accepts_cancel(row: 'OrderRow', *, now_ts: float | None = None) -> bool:
    """Whether an operator resume accepted the cancel of ``row`` reported now.

    For a ``cancelled`` event a plugin reports itself instead of through the
    tracker (an activity catch-up replaying the venue's history after the
    restart): it gets the verdict the tracker gives the same row's absence. An
    absence stamp the resume accepted is this cancel. A row still marked for
    its first absence after the resume (no complete snapshot showed it since)
    accepts a cancel reported within :data:`RESUME_FIRST_ABSENCE_WINDOW_S` of
    the resume, the window of a stamp a plugin writes itself; a later one is
    a fresh absence.

    :param row: The cancelled order's live journal row.
    :param now_ts: The time the cancel is reported (unix seconds); now when
        omitted.
    """
    extras = row.extras or {}
    mark = extras.get(MISSING_ACCEPTED_BY_RESUME_EXTRA)
    since = extras.get(MISSING_PENDING_EXTRA)
    if mark == _RESUME_ACCEPTS_NEXT_ABSENCE:
        resumed_at = extras.get(_MISSING_RESUMED_AT_EXTRA)
        if not _is_stamp(resumed_at):
            return False
        if _is_stamp(since):
            absent_since = since
        else:
            absent_since = time.time() if now_ts is None else now_ts
        return absent_since <= resumed_at + RESUME_FIRST_ABSENCE_WINDOW_S
    return _is_stamp(mark) and mark == since


def renew_resume_first_absence(
        store_ctx: 'RunContext', *, resumed_at: float | None = None,
) -> int:
    """Restart the first-absence window of the rows a resume still has to judge.

    For a resume applied again by a process that restarted before the first
    one completed it (:func:`accept_missing_before_resume` ran in that first
    process). A row still marked for its first absence and not stamped since
    was judged by no process after the resume: whether it vanished before the
    resume or while no process ran cannot be told apart, so its acceptance
    window starts again at ``resumed_at``. Every other row keeps what the
    earlier process settled: a row it saw on the venue lost its acceptance, a
    stamp it accepted stays accepted, and a stamp it wrote keeps the window of
    the first application.

    :param store_ctx: The resumed run's journal.
    :param resumed_at: The time this process applies the resume (unix
        seconds); now when omitted.
    :return: The number of live rows whose window restarted.
    """
    if resumed_at is None:
        resumed_at = time.time()
    renewed = 0
    with store_ctx.transaction():
        for row in store_ctx.iter_live_orders():
            extras = dict(row.extras or {})
            if (extras.get(MISSING_ACCEPTED_BY_RESUME_EXTRA) != _RESUME_ACCEPTS_NEXT_ABSENCE
                    or _is_stamp(extras.get(MISSING_PENDING_EXTRA))):
                continue
            extras[_MISSING_RESUMED_AT_EXTRA] = resumed_at
            store_ctx.upsert_order(row.client_order_id, extras=extras)
            renewed += 1
    return renewed


def resolve_unexpected_cancel_policy(
        policy: str,
        *,
        reason: str,
        context: dict[str, Any],
        request_quarantine: Callable[..., bool | None] | None,
        log_event: Callable[..., None] | None,
        client_order_id: str | None,
        exchange_order_id: str | None,
        audit_prefix: str = 'unexpected_cancel',
) -> UnexpectedCancelError | None:
    """Resolve the ``on_unexpected_cancel`` policy for a confirmed
    unexpected cancel — the mapping shared by the grace-window tracker path
    (:meth:`DisappearanceTracker._apply_policy`) and the sync engine's
    WS-push handler
    (:meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._apply_unexpected_cancel_policy`).

    Applies the ``ignore`` / ``re_place`` audit-only logging and the
    ``stop`` quarantine latch, and RETURNS the halt the caller must arm its
    own way (the tracker parks it on :attr:`pending_halt`; the engine
    records + raises it) — or ``None`` when the outcome needed no halt
    (ignored, re-placed, or quarantined). An unexpected cancel is someone
    else acting on a bot-owned order, so ``stop`` latches an
    :attr:`~pynecore.core.broker.models.QuarantineKind.EXTERNAL_INTERVENTION`
    quarantine: the bot leaves the venue exactly as the operator left it.
    The fail-closed rule lives here: a ``stop`` whose ``request_quarantine``
    sink is missing or raises falls back to the halt, never to continued
    trading.

    :param policy: The configured ``on_unexpected_cancel`` policy name
        (one of :data:`UNEXPECTED_CANCEL_POLICIES`).
    :param reason: Human-readable description of the confirmed cancel,
        carried into the halt / quarantine request.
    :param context: Structured payload passed to the quarantine sink.
    :param request_quarantine: Quarantine latch sink; ``None`` (an unwired
        sink under a quarantining policy) triggers the fail-closed halt. A sink
        that returns ``False`` already recorded this intervention under its
        held latch (a re-report after a restart): no audit event is written.
    :param log_event: Audit sink (``RunContext.log_event``); ``None`` skips
        audit persistence (a store-less engine on the push path).
    :param client_order_id: The affected row's COID, stamped on audit events.
    :param exchange_order_id: The affected row's venue ref, stamped on
        audit events.
    :param audit_prefix: Prefix of the audit event kinds
        (``<prefix>_ignored`` / ``_re_place`` / ``_quarantine``); other kinds
        of external intervention (a moved SL/TP, a manual close) name
        themselves.
    """
    def _log(kind: str) -> None:
        if log_event is not None:
            log_event(
                kind,
                client_order_id=client_order_id,
                exchange_order_id=exchange_order_id,
            )

    if policy == 'ignore':
        _log(f'{audit_prefix}_ignored')
        return None
    if policy == 're_place':
        _log(f'{audit_prefix}_re_place')
        return None
    quarantined = False
    if policy == 'stop' and request_quarantine is not None:
        # noinspection PyBroadException
        try:
            noted = request_quarantine(
                reason, context, kind=QuarantineKind.EXTERNAL_INTERVENTION,
            )
        except Exception:
            logger.exception(
                "request_quarantine hook failed for %r; falling back "
                "to the process-exiting halt", client_order_id,
            )
        else:
            quarantined = True
            if noted is not False:
                _log(f'{audit_prefix}_quarantine')
    if not quarantined:
        # 'halt', or a quarantining policy whose hook is missing / raised:
        # arm the process-exiting manual-intervention signal.
        return UnexpectedCancelError(reason, context=context)
    return None


class MissingResolution(StrEnum):
    """Grace-expiry classification returned by the ``confirm_missing`` hook.

    - :data:`STILL_PRESENT` — the row's broker counterpart is verifiably
      back (or the earlier absence was a snapshot artifact): clear the
      stamp, no further action.
    - :data:`INCONCLUSIVE` — the verification could not complete (e.g.
      history transport down, pagination hole): keep the stamp and wait;
      never conclude a cancel from missing evidence.
    - :data:`FILLED` — the ref vanished because it (partially or fully)
      filled: the stamp premise is false; any discovered fill slice is
      applied and the row stays live for the venue's normal promotion
      path.
    - :data:`CLOSED` — the row filled and its position was then closed
      (native TP/SL fired, external close): retire the row (and its
      position siblings) as a natural close — never a synthetic cancel —
      and yield the closing fills the venue evidence proved
      (:attr:`MissingConfirmation.close_events`), so the engine books the
      close it missed instead of finding an unexplained flat venue.
    - :data:`CANCELLED` — verified external cancel: terminal-close the
      row, emit the synthesised ``cancelled`` event and apply the
      ``on_unexpected_cancel`` policy.
    """
    STILL_PRESENT = 'still_present'
    INCONCLUSIVE = 'inconclusive'
    FILLED = 'filled'
    CLOSED = 'closed'
    CANCELLED = 'cancelled'


@dataclass(frozen=True, slots=True)
class MissingConfirmation:
    """Declarative result of a grace-expiry re-verification.

    The ``confirm_missing`` hook builds this from venue evidence (deal
    history, activity log, order-status probe) and mutates NOTHING — the
    tracker applies the whole outcome in one store transaction so it is
    either fully booked or fully dropped.

    Any terminal resolution may carry a discovered fill slice: a working
    order that partially filled and was then externally cancelled
    resolves as :data:`MissingResolution.CANCELLED` *with* fill data, and
    the tracker books the slice in the same transaction as the terminal
    close. A fill slice is only booked when ``fill_price`` is strictly
    positive; an unpriced slice makes the tracker defer (keep the stamp)
    rather than book exposure the engine would silently drop.

    :ivar resolution: The classification — see :class:`MissingResolution`.
        A plain string equal to one of the enum values is coerced.
    :ivar cumulative_filled_qty: The order's proven cumulative filled
        quantity, when the verification discovered fill progress. The
        tracker clamps it into ``[row.filled_qty, row.qty]`` (monotonic,
        never overstating the order's own size) and books the delta.
    :ivar fill_price: Volume-weighted price of the discovered executions
        (the price the delta is booked at). Must be strictly positive for
        the slice to be booked.
    :ivar fill_fee: Summed commission of the discovered executions.
    :ivar execution_ids: Venue execution/deal ids backing the evidence.
        Passed to the ``register_executions`` hook after a fill slice was
        actually booked or the row was retired on a terminal resolution,
        so the plugin can seed its duplicate-fill channel; a lone id is
        also stamped as the fill event's ``fill_id``.
    :ivar position_ref: Venue position id the fill belongs to. Selects
        the ``working_promoted_position`` audit reason and scopes the
        sibling retirement on :data:`MissingResolution.CLOSED`.
    :ivar extras_patch: Plugin extras merged into the row alongside a
        discovered fill (e.g. ``{'position_id': ...}``).
    :ivar executed_ts: Broker-reported execution time (unix seconds) for
        the fill event; falls back to the observation time.
    :ivar cancel_reason: On :data:`MissingResolution.CANCELLED`, the venue's
        own cancel cause when the evidence names one of
        :data:`~pynecore.core.broker.models.VENUE_DRIVEN_CANCEL_REASONS`. Such
        a cancel retires the row and is yielded with this reason, but the
        ``on_unexpected_cancel`` policy is not applied — the venue, not an
        operator, terminated the order. ``None`` means an unexplained cancel.
    :ivar close_events: On :data:`MissingResolution.CLOSED`, the closing
        fills of the position the evidence proved and the event stream never
        delivered, as close-leg :class:`OrderEvent` objects attributed to the
        row's entry. They are yielded only when the row is actually retired,
        after the retirement and the ``register_executions`` hook, so the
        engine books the close before its venue-vs-book check would read the
        flat venue as an unexplained flatten. Ignored on every other
        resolution.
    """
    resolution: MissingResolution
    cumulative_filled_qty: float | None = None
    fill_price: float | None = None
    fill_fee: float = 0.0
    execution_ids: tuple[str, ...] = ()
    position_ref: str | None = None
    extras_patch: Mapping[str, Any] | None = None
    executed_ts: float | None = None
    cancel_reason: str | None = None
    close_events: tuple[OrderEvent, ...] = ()

    def __post_init__(self) -> None:
        # Coerce a plain string (or reject an invalid value at construction
        # time) so the apply path can rely on identity checks against the
        # enum members and never fall through to the CANCELLED mutation.
        if not isinstance(self.resolution, MissingResolution):
            object.__setattr__(
                self, 'resolution', MissingResolution(self.resolution),
            )


class DisappearanceTracker:
    """Venue-agnostic disappearance state machine over the BrokerStore.

    One instance per plugin. The plugin's snapshot loop calls
    :meth:`observe` once per pass with the per-namespace present-ref
    sets; the tracker owns stamping, clearing, grace accounting, the
    grace-expiry confirmation protocol and the dual-signal delivery. All
    venue knowledge enters through the constructor hooks.

    :param store_ctx: The plugin's open :class:`RunContext`.
    :param grace_s: Seconds a stamp must age before ``confirm_missing``
        runs. Venue-tuned (``max(5, 5 × poll cadence)`` on the reference
        plugins).
    :param policy: The configured ``on_unexpected_cancel`` policy —
        one of :data:`UNEXPECTED_CANCEL_POLICIES`.
    :param tracked_refs: Maps a live row to its ``{(namespace, ref)}``
        set. An empty set exempts the row from tracking (e.g. bracket
        legs with no broker id of their own).
    :param confirm_missing: Async grace-expiry verifier — returns a
        :class:`MissingConfirmation`, mutates nothing.
    :param is_exempt: Optional extra exemption (e.g. rows already flagged
        as naturally closed). Checked on every pass, both phases.
    :param request_quarantine: Sink that latches the engine's quarantine
        state (``(reason, context, *, kind)``); the runner wires it to
        ``OrderSyncEngine.record_quarantine``. Used by the ``stop`` policy.
        When it is missing (or raises), the policy falls back to arming the
        process-exiting halt — fail-safe, never fail-open into continued
        trading.
    :param sibling_coids: Maps a :data:`MissingResolution.CLOSED` row to
        the client-order-ids of live sibling rows sharing its position
        (a netting account merges pyramid entries onto one position id);
        they are retired in the same transaction.
    :param register_executions: Called with the confirmation's
        ``execution_ids`` after a fill slice was actually booked or the
        row was retired on a terminal resolution, so the plugin can seed
        its duplicate-fill channel before any replayed push event. A
        raising hook is logged, not propagated.
    :param cancelled_event_factory: Overrides the synthesised
        ``cancelled`` event construction (venues that key the event on
        something other than ``row.exchange_order_id``). The default
        builder is entry-shaped (``LegType.ENTRY``, ``reduce_only=False``,
        ``OrderType.MARKET``); a non-entry row MUST supply a custom factory.
    :param fill_event_factory: Overrides the recovered-fill event
        construction. Receives ``(row, confirmation, cumulative,
        fill_qty, now_ts)``. The default builder is entry-shaped as
        above; a non-entry row MUST supply a custom factory.
    :param retire_cancelled_row: Retires the row of a confirmed
        :data:`MissingResolution.CANCELLED` in place of the default
        terminal close, inside the same store transaction. For a venue
        whose entry rows keep owning the exposure a partially filled
        order left behind — the cancel ended only the unfilled remainder,
        so closing the row would drop that ownership. The tracker still
        writes the audit event and yields the cancelled event; it clears
        the stamp of a row the hook kept live, so the row is not
        re-confirmed.
    """

    def __init__(
            self,
            store_ctx: 'RunContext',
            *,
            grace_s: float,
            policy: str,
            tracked_refs: Callable[['OrderRow'], Set[tuple[str, str]]],
            confirm_missing: Callable[['OrderRow'], Awaitable[MissingConfirmation]],
            is_exempt: Callable[['OrderRow'], bool] | None = None,
            request_quarantine: Callable[..., bool | None] | None = None,
            sibling_coids: Callable[
                ['OrderRow', MissingConfirmation], Iterable[str]] | None = None,
            register_executions: Callable[
                ['OrderRow', tuple[str, ...]], None] | None = None,
            cancelled_event_factory: Callable[
                ['OrderRow', float], OrderEvent] | None = None,
            fill_event_factory: Callable[
                ['OrderRow', MissingConfirmation, float, float, float],
                OrderEvent] | None = None,
            retire_cancelled_row: Callable[['OrderRow'], None] | None = None,
    ) -> None:
        if policy not in UNEXPECTED_CANCEL_POLICIES:
            raise ValueError(
                f"DisappearanceTracker: unknown policy {policy!r}; "
                f"expected one of {UNEXPECTED_CANCEL_POLICIES}"
            )
        self._store = store_ctx
        self._grace_s = grace_s
        self._policy = policy
        self._tracked_refs = tracked_refs
        self._confirm_missing = confirm_missing
        self._is_exempt = is_exempt
        self._request_quarantine = request_quarantine
        self._sibling_coids = sibling_coids
        self._register_executions = register_executions
        self._cancelled_event_factory = cancelled_event_factory
        self._fill_event_factory = fill_event_factory
        self._retire_cancelled_row = retire_cancelled_row
        #: Halt decided by a halting policy but not yet delivered to the
        #: consumer. Survives an abandoned generator: re-raised at the top
        #: of the next :meth:`observe` call. Consumed (cleared) by whichever
        #: raise site delivers it, so it fires exactly once.
        self._pending_halt: UnexpectedCancelError | None = None
        #: ``(coid, event_kind)`` pairs already warned about a deferred
        #: grace re-check, so a sustained anomaly logs once per row per
        #: reason (an inconclusive re-check must not mute a later unpriced
        #: fill on the same row), not once per cadence.
        self._warned_deferred: set[tuple[str, str]] = set()

    @property
    def pending_halt(self) -> UnexpectedCancelError | None:
        """The undelivered halt decided by a halting policy, if any.

        Observational only — delivery happens by the :class:`UnexpectedCancelError`
        raised from :meth:`observe`, which consumes it. A plugin that both
        consumes the generator and reads this property must not act on the
        property independently, or it would double-handle the same halt.
        """
        return self._pending_halt

    def _take_pending_halt(self) -> UnexpectedCancelError | None:
        """Consume the pending halt so it is delivered exactly once."""
        halt = self._pending_halt
        self._pending_halt = None
        return halt

    async def observe(
            self,
            present: Mapping[str, Set[str] | None],
            now_ts: float,
    ) -> AsyncIterator[OrderEvent]:
        """Run one observation pass and yield the recovered events.

        Phase 1 stamps / clears every tracked live row against
        ``present``; phase 2 runs the grace-expiry confirmation protocol
        on rows whose stamp has aged past the grace window.

        :param present: Per-namespace sets of the refs visible in this
            pass's snapshot. ``None`` (or a missing key) marks a
            namespace whose fetch FAILED — rows tracked in it are
            neither stamped nor cleared this pass.
        :param now_ts: The observation timestamp (unix seconds); also
            the value stamped into ``missing_pending_since``.
        :raises UnexpectedCancelError: After the ``cancelled`` event of a
            row whose policy is halting — or immediately, when a halt
            decided on an earlier (abandoned) pass is still undelivered.
        """
        halt = self._take_pending_halt()
        if halt is not None:
            raise halt

        self.observe_presence(present, now_ts)

        for row in list(self._store.iter_live_orders()):
            if self._is_exempt is not None and self._is_exempt(row):
                continue
            extras = row.extras or {}
            since = extras.get(MISSING_PENDING_EXTRA)
            if not isinstance(since, (int, float)):
                continue
            since_ts = float(since)
            if (now_ts - since_ts) < self._grace_s:
                continue
            confirmation = await self._confirm_missing(row)
            events, applied, register_ids, applied_row = self._apply_confirmation(
                row, since_ts, confirmation, now_ts,
            )
            # The post-commit hooks must see the row as it stands after the
            # fill slice was booked — a stale pre-fill row would let a hook
            # miss the recovered quantity / promotion metadata.
            hook_row = applied_row if applied_row is not None else row
            if applied and register_ids and self._register_executions is not None:
                # noinspection PyBroadException
                try:
                    self._register_executions(hook_row, register_ids)
                except Exception:
                    # A committed terminal row must never be stranded
                    # without its event / halt by a dedup-seeding failure.
                    logger.exception(
                        "register_executions hook failed for %r",
                        row.client_order_id,
                    )
            if (applied
                    and confirmation.resolution is MissingResolution.CANCELLED
                    and confirmation.cancel_reason not in VENUE_DRIVEN_CANCEL_REASONS
                    and not _is_fully_filled(hook_row)
                    and not _accepted_by_resume(row, since_ts)):
                # A fully filled row was retired as a benign post-fill
                # close by ``_apply_confirmation`` (same predicate), a
                # venue-driven cancel is nobody's intervention, and a cancel
                # an operator resume accepted is no longer one — the
                # unexpected-cancel policy must not fire for any of them.
                self._apply_policy(hook_row)
            for event in events:
                yield event
            halt = self._take_pending_halt()
            if halt is not None:
                raise halt

    # --- Phase 1: stamp / clear against the present-sets --------------------

    def observe_presence(
            self,
            present: Mapping[str, Set[str] | None],
            now_ts: float,
    ) -> None:
        """Run phase 1 only: stamp / clear every tracked live row.

        For venues whose snapshot-reconcile pass owns the presence diff
        (stamping from inside the same walk that books fills) while a
        separate later pass drives the grace protocol via
        :meth:`observe`. :meth:`observe` runs this itself, so calling
        both against the same snapshot is safe — stamp and clear are
        idempotent per pass.
        """
        live_coids: set[str] = set()
        for row in list(self._store.iter_live_orders()):
            live_coids.add(row.client_order_id)
            if self._is_exempt is not None and self._is_exempt(row):
                continue
            self._observe_presence(row, present, now_ts)

        # Drop throttle keys for rows that went terminal via another path
        # (a PUSH event) and left the live set without reaching a resolve —
        # else the in-memory set grows unbounded on a long-running instance.
        if self._warned_deferred:
            self._warned_deferred = {
                k for k in self._warned_deferred if k[0] in live_coids
            }

    def _observe_presence(
            self,
            row: 'OrderRow',
            present: Mapping[str, Set[str] | None],
            now_ts: float,
    ) -> None:
        refs = self._tracked_refs(row)
        self._settle_resume_mark(row, now_ts, presence_tracked=bool(refs))
        if not refs:
            return
        visible = False
        all_fetched = True
        for namespace, ref in refs:
            ns_refs = present.get(namespace)
            if ns_refs is None:
                all_fetched = False
                continue
            if ref in ns_refs:
                visible = True
                break
        extras = row.extras or {}
        if visible:
            if MISSING_PENDING_EXTRA in extras or MISSING_ACCEPTED_BY_RESUME_EXTRA in extras:
                self._clear_stamp(row.client_order_id)
            return
        if not all_fetched:
            # Incomplete snapshot: absence is unproven — neither stamp
            # nor clear.
            return
        if MISSING_PENDING_EXTRA not in extras:
            self._stamp(row.client_order_id, now_ts)

    def _stamp(self, coid: str, now_ts: float) -> None:
        """Stamp ``missing_pending_since`` atomically, preserving extras.

        Re-reads the live row and rewrites only the one key so a breadcrumb
        another broker thread wrote between the phase-1 snapshot and here is
        not clobbered; a stamp added concurrently is left untouched.
        """
        with self._store.transaction():
            fresh = self._store.get_order(coid)
            if fresh is None or fresh.closed_ts_ms is not None:
                return
            extras = dict(fresh.extras or {})
            if MISSING_PENDING_EXTRA in extras:
                return
            extras[MISSING_PENDING_EXTRA] = now_ts
            if extras.get(MISSING_ACCEPTED_BY_RESUME_EXTRA) == _RESUME_ACCEPTS_NEXT_ABSENCE:
                extras[MISSING_ACCEPTED_BY_RESUME_EXTRA] = now_ts
                extras.pop(_MISSING_RESUMED_AT_EXTRA, None)
            self._store.upsert_order(coid, extras=extras)

    def _settle_resume_mark(
            self, row: 'OrderRow', now_ts: float, *, presence_tracked: bool,
    ) -> None:
        """Bind an operator resume's acceptance to a stamp the plugin wrote itself.

        The presence feed converts the first-absence mark when it stamps the
        row (:meth:`_stamp`) and drops it when a complete snapshot shows the
        row. A plugin that stamps a row on its own evidence bypasses both, so
        the mark is settled here instead: a stamp written within
        :data:`RESUME_FIRST_ABSENCE_WINDOW_S` of the resume becomes the
        accepted one; a later stamp, a mark of a row the presence feed does not
        track that is still unstamped past the window, and an accepted stamp the
        plugin cleared or replaced all lose the acceptance. Re-reads the live
        row and rewrites only the mark keys.
        """
        if MISSING_ACCEPTED_BY_RESUME_EXTRA not in (row.extras or {}):
            return
        with self._store.transaction():
            fresh = self._store.get_order(row.client_order_id)
            if fresh is None or fresh.closed_ts_ms is not None:
                return
            extras = fresh.extras or {}
            mark = extras.get(MISSING_ACCEPTED_BY_RESUME_EXTRA)
            if mark is None:
                return
            since = extras.get(MISSING_PENDING_EXTRA)
            settled: float | None = None
            if mark == _RESUME_ACCEPTS_NEXT_ABSENCE:
                resumed_at = extras.get(_MISSING_RESUMED_AT_EXTRA)
                if _is_stamp(resumed_at):
                    deadline = resumed_at + RESUME_FIRST_ABSENCE_WINDOW_S
                    if _is_stamp(since):
                        settled = since if since <= deadline else None
                    elif presence_tracked or now_ts <= deadline:
                        return
            elif _is_stamp(since) and mark == since:
                return
            merged = _without_resume_mark(extras)
            if settled is not None:
                merged[MISSING_ACCEPTED_BY_RESUME_EXTRA] = settled
            self._store.upsert_order(fresh.client_order_id, extras=merged)

    def _clear_stamp(self, coid: str) -> None:
        """Remove ``missing_pending_since`` (and its resume acceptance) atomically."""
        stamp_keys = (MISSING_PENDING_EXTRA, MISSING_ACCEPTED_BY_RESUME_EXTRA,
                      _MISSING_RESUMED_AT_EXTRA)
        with self._store.transaction():
            fresh = self._store.get_order(coid)
            if fresh is not None:
                extras = fresh.extras or {}
                if any(key in extras for key in stamp_keys):
                    merged = {k: v for k, v in extras.items() if k not in stamp_keys}
                    self._store.upsert_order(coid, extras=merged)
        self._forget_deferred(coid)

    # --- Phase 2: grace-expiry confirmation apply ----------------------------

    def _apply_confirmation(
            self,
            row: 'OrderRow',
            since: float,
            confirmation: MissingConfirmation,
            now_ts: float,
    ) -> tuple[list[OrderEvent], bool, tuple[str, ...], 'OrderRow | None']:
        """Apply one confirmation atomically under the stamp-version guard.

        Returns ``(events, applied, register_ids, applied_row)``. ``applied``
        is ``False`` when the guard dropped a stale outcome (the row came
        back, was re-stamped or reached a terminal state while the
        confirmation ran) or when a fill slice was deferred for lack of a
        price. ``register_ids`` is non-empty only when a fill slice was
        actually booked OR the row reached a terminal resolution (CLOSED /
        CANCELLED) in the same transaction — the retirement was concluded
        FROM that execution evidence, so a replayed push copy of it must
        already be suppressed. A deferred or dropped outcome never seeds
        the dedup channel: the evidence may still need to book later.
        ``applied_row`` is the row re-read *after* the fill slice was
        booked (``None`` on a dropped or no-op outcome); the caller feeds
        it to the post-commit hooks so none of them sees a stale pre-fill
        quantity.
        """
        resolution = confirmation.resolution
        events: list[OrderEvent] = []
        with self._store.transaction():
            fresh = self._store.get_order(row.client_order_id)
            if fresh is None or fresh.closed_ts_ms is not None:
                return [], False, (), None
            stamp = (fresh.extras or {}).get(MISSING_PENDING_EXTRA)
            if stamp != since:
                logger.info(
                    "disappearance confirmation for %r dropped: stamp "
                    "changed during verification (%r -> %r)",
                    row.client_order_id, since, stamp,
                )
                return [], False, (), None

            if resolution is MissingResolution.INCONCLUSIVE:
                self._warn_deferred(
                    fresh, 'missing_pending_recheck_inconclusive',
                    "grace-expired row %r left un-retired: disappearance "
                    "re-check inconclusive — deferring rather than "
                    "concluding a false cancel",
                )
                return [], True, (), None

            if resolution is MissingResolution.STILL_PRESENT:
                self._clear_stamp(fresh.client_order_id)
                return [], True, (), None

            # Discovered fill slice (pure computation, no writes yet).
            cumulative = self._clamped_cumulative(fresh, confirmation)
            new_qty = 0.0 if cumulative is None else cumulative - fresh.filled_qty
            has_new_fill = new_qty > _QTY_EPS
            if has_new_fill and not (
                    confirmation.fill_price is not None
                    and math.isfinite(confirmation.fill_price)
                    and confirmation.fill_price > 0.0):
                # Fail-closed: a fill we cannot price (missing, non-finite,
                # or non-positive) would be dropped by the engine's
                # record_fill yet still advance the store and seed dedup.
                # Keep the stamp and defer.
                self._warn_deferred(
                    fresh, 'missing_pending_fill_unpriced',
                    "grace-expired row %r reported fill progress without a "
                    "usable price — deferring rather than booking an "
                    "unpriced fill",
                )
                return [], False, (), None

            register_ids: tuple[str, ...] = ()
            fill_event: OrderEvent | None = None
            updated = fresh
            if has_new_fill and cumulative is not None:
                self._book_fill(fresh, confirmation, cumulative)
                updated = self._store.get_order(fresh.client_order_id) or fresh
                fill_event = self._build_fill_event(
                    updated, confirmation, cumulative, new_qty, now_ts,
                )
                register_ids = confirmation.execution_ids
            elif confirmation.position_ref is not None and cumulative is not None:
                # Working->position promotion without fresh quantity (the
                # fill was already booked): flip state + extras, emit and
                # register nothing.
                self._book_fill(fresh, confirmation, cumulative)
                updated = self._store.get_order(fresh.client_order_id) or fresh
            elif confirmation.extras_patch:
                # Metadata-only patch: merge extras WITHOUT manufacturing a
                # kind='filled' journal outcome for a zero-delta write.
                merged = dict(fresh.extras or {})
                merged.update(confirmation.extras_patch)
                self._store.upsert_order(fresh.client_order_id, extras=merged)
                updated = self._store.get_order(fresh.client_order_id) or fresh

            if resolution is MissingResolution.FILLED:
                # The stamp premise is false — the ref vanished because it
                # filled. The row stays live; the venue's normal snapshot
                # promotion path owns it from here.
                self._clear_stamp(updated.client_order_id)
                if fill_event is not None:
                    events.append(fill_event)
                return events, True, register_ids, updated

            if resolution is MissingResolution.CLOSED:
                # Filled-then-closed: no entry fill event is emitted for
                # exposure that no longer exists — any discovered fill
                # progress was persisted above for bookkeeping — but the
                # closing fills the evidence proved are yielded, or the
                # engine would read the flat venue as an unexplained
                # flatten. The retirement is concluded FROM the
                # confirmation's execution evidence, so its ids are
                # registered even without a fresh booked slice — a replayed
                # push copy of an already-counted deal must not resurface as
                # a new fill after the retire.
                register_ids = confirmation.execution_ids
                DispatchJournal(self._store).apply_reconcile_outcome(
                    updated.client_order_id,
                    ReconcileOutcome(
                        kind='terminal_close',
                        reason='bracket_natural_close_followup',
                        new_state='closed',
                        audit_event='reconcile_filled_then_closed_retired',
                        close_row=True,
                        audit_payload={
                            'position_ref': confirmation.position_ref,
                            'missing_since': since,
                            'execution_ids': list(confirmation.execution_ids),
                        },
                        exchange_order_id=(confirmation.position_ref
                                           or updated.exchange_order_id),
                    ),
                )
                if self._sibling_coids is not None:
                    for sibling in self._sibling_coids(updated, confirmation):
                        self._store.close_order(sibling)
                self._forget_deferred(updated.client_order_id)
                return list(confirmation.close_events), True, register_ids, updated

            if resolution is MissingResolution.CANCELLED:
                # Terminal like CLOSED above: register the backing ids so
                # a replayed copy of the evidence cannot re-book after the
                # retire.
                register_ids = confirmation.execution_ids
                if _is_fully_filled(updated):
                    # A fully filled row's ref vanishing is NOT an external
                    # cancel: the full quantity already executed and was
                    # booked, so there was nothing working left to cancel.
                    # Venues can still surface a late CANCELLED for such an
                    # order — e.g. Capital.com netting an in-flight entry
                    # into an opposite deal's close and reporting the
                    # consumed order CANCELLED minutes later. Applying the
                    # ``on_unexpected_cancel`` policy here would quarantine
                    # a healthy run over a bookkeeping echo. Retire the row
                    # as a benign post-fill close instead: no cancelled
                    # event, and :meth:`observe` skips ``_apply_policy`` on
                    # the same predicate.
                    DispatchJournal(self._store).apply_reconcile_outcome(
                        updated.client_order_id,
                        ReconcileOutcome(
                            kind='terminal_close',
                            reason='missing_pending_grace_expired',
                            new_state='closed',
                            audit_event='reconcile_cancel_after_full_fill_retired',
                            close_row=True,
                            audit_payload={'missing_since': since,
                                           'grace': self._grace_s,
                                           'filled_qty': updated.filled_qty,
                                           'qty': updated.qty},
                            exchange_order_id=updated.exchange_order_id,
                        ),
                    )
                    if fill_event is not None:
                        events.append(fill_event)
                    self._forget_deferred(updated.client_order_id)
                    return events, True, register_ids, updated
                if confirmation.cancel_reason in VENUE_DRIVEN_CANCEL_REASONS:
                    # The venue terminated the order itself and its push was
                    # lost: retire the row and hand the engine the reason, so
                    # it trims the leg instead of treating the cancel as an
                    # intervention.
                    self._retire_cancelled(
                        updated,
                        ReconcileOutcome(
                            kind='terminal_close',
                            reason='missing_pending_grace_expired',
                            new_state='closed',
                            audit_event='venue_driven_cancel_recovered',
                            close_row=True,
                            audit_payload={'missing_since': since,
                                           'grace': self._grace_s,
                                           'cancel_reason': confirmation.cancel_reason},
                            exchange_order_id=updated.exchange_order_id,
                        ),
                    )
                    if fill_event is not None:
                        events.append(fill_event)
                    events.append(self._build_cancelled_event(
                        updated, now_ts, confirmation.cancel_reason,
                    ))
                    self._forget_deferred(updated.client_order_id)
                    return events, True, register_ids, updated
                accepted = _accepted_by_resume(fresh, since)
                if accepted:
                    logger.warning(
                        "bot-owned order %r (ref %r) was cancelled before the operator "
                        "resume — the resume accepted it, no intervention",
                        updated.client_order_id, updated.exchange_order_id,
                    )
                self._retire_cancelled(
                    updated,
                    ReconcileOutcome(
                        kind='terminal_close',
                        reason='missing_pending_grace_expired',
                        new_state='rejected',
                        audit_event=('cancel_accepted_by_resume' if accepted
                                     else 'unexpected_cancel'),
                        close_row=True,
                        audit_payload={'missing_since': since,
                                       'grace': self._grace_s},
                        exchange_order_id=updated.exchange_order_id,
                    ),
                )
                if fill_event is not None:
                    events.append(fill_event)
                events.append(self._build_cancelled_event(updated, now_ts))
                self._forget_deferred(updated.client_order_id)
                return events, True, register_ids, updated

            raise AssertionError(f"unhandled resolution {resolution!r}")

    def _retire_cancelled(self, row: 'OrderRow', outcome: ReconcileOutcome) -> None:
        """Retire a confirmed-cancelled row, through the plugin hook when wired.

        Runs inside the caller's transaction. Without the hook the journal
        closes the row; with it the audit event is written here and the hook
        decides the row's fate, and a row it kept live loses its stamp.
        """
        if self._retire_cancelled_row is None:
            DispatchJournal(self._store).apply_reconcile_outcome(
                row.client_order_id, outcome,
            )
            return
        self._store.log_event(
            outcome.audit_event,
            client_order_id=row.client_order_id,
            exchange_order_id=outcome.exchange_order_id,
            payload=(dict(outcome.audit_payload)
                     if outcome.audit_payload is not None else None),
        )
        self._retire_cancelled_row(row)
        fresh = self._store.get_order(row.client_order_id)
        if fresh is not None and fresh.closed_ts_ms is None:
            self._clear_stamp(row.client_order_id)

    @staticmethod
    def _clamped_cumulative(
            fresh: 'OrderRow', confirmation: MissingConfirmation,
    ) -> float | None:
        """Clamp the confirmed cumulative into ``[fresh.filled_qty, fresh.qty]``.

        Monotonic (never regresses below what is already booked) and never
        overstates the order's own size. ``None`` when the confirmation
        carried no fill quantity.
        """
        if confirmation.cumulative_filled_qty is None:
            return None
        return min(
            fresh.qty,
            max(fresh.filled_qty, confirmation.cumulative_filled_qty),
        )

    def _book_fill(
            self,
            fresh: 'OrderRow',
            confirmation: MissingConfirmation,
            cumulative: float,
    ) -> None:
        """Persist the discovered fill slice inside the caller's transaction."""
        reason: ReconcileReason = (
            'working_promoted_position'
            if confirmation.position_ref is not None
            else 'partial_fill_progress'
        )
        DispatchJournal(self._store).apply_reconcile_outcome(
            fresh.client_order_id,
            ReconcileOutcome(
                kind='filled',
                reason=reason,
                new_state='confirmed',
                audit_event='reconcile_missing_fill_recovered',
                filled_qty=cumulative,
                extras_patch=confirmation.extras_patch,
                audit_payload={
                    'cumulative': cumulative,
                    'previous': fresh.filled_qty,
                    'execution_ids': list(confirmation.execution_ids),
                },
                exchange_order_id=(confirmation.position_ref
                                   or fresh.exchange_order_id),
            ),
        )

    def _build_fill_event(
            self,
            updated: 'OrderRow',
            confirmation: MissingConfirmation,
            cumulative: float,
            fill_qty: float,
            now_ts: float,
    ) -> OrderEvent:
        if self._fill_event_factory is not None:
            return self._fill_event_factory(
                updated, confirmation, cumulative, fill_qty, now_ts,
            )
        return self._default_fill_event(
            updated, confirmation, cumulative, fill_qty, now_ts,
        )

    def _warn_deferred(self, row: 'OrderRow', event_kind: str, msg: str) -> None:
        """Log once per row per reason that a grace-expired retire deferred."""
        coid = row.client_order_id
        key = (coid, event_kind)
        if key in self._warned_deferred:
            return
        logger.warning(msg, coid)
        self._store.log_event(
            event_kind,
            client_order_id=coid,
            exchange_order_id=row.exchange_order_id,
        )
        # Arm the throttle only after the audit event is persisted, so a
        # failed log_event does not mute a later genuine re-warning.
        self._warned_deferred.add(key)

    def _forget_deferred(self, coid: str) -> None:
        """Drop every deferred-warning throttle for a row (it resolved)."""
        self._warned_deferred = {
            k for k in self._warned_deferred if k[0] != coid
        }

    # --- Policy ---------------------------------------------------------------

    def _apply_policy(self, row: 'OrderRow') -> None:
        """Apply the ``on_unexpected_cancel`` policy for a confirmed cancel.

        Runs BEFORE the cancelled event is yielded: the quarantine / halt
        decision must not depend on the consumer pulling further elements.
        ``stop`` latches the engine's external-intervention quarantine
        through the ``request_quarantine`` hook — the process (and this
        observation loop) keeps running; a missing or raising hook falls
        back to arming the halt, never to continued trading. ``halt``
        always arms :attr:`pending_halt`; it is raised after the row's
        events were yielded (or at the top of the next pass, when the
        generator was abandoned).
        """
        reason = (
            f"Bot-owned order disappeared unexpectedly: "
            f"coid={row.client_order_id!r} "
            f"ref={row.exchange_order_id!r}"
        )
        context = {
            'client_order_id': row.client_order_id,
            'exchange_order_id': row.exchange_order_id,
            'symbol': row.symbol,
            'policy': self._policy,
        }
        halt = resolve_unexpected_cancel_policy(
            self._policy,
            reason=reason,
            context=context,
            request_quarantine=self._request_quarantine,
            log_event=self._store.log_event,
            client_order_id=row.client_order_id,
            exchange_order_id=row.exchange_order_id,
        )
        if halt is not None:
            self._pending_halt = halt

    # --- Default event builders ------------------------------------------------

    def _build_cancelled_event(
            self, row: 'OrderRow', now_ts: float, cancel_reason: str | None = None,
    ) -> OrderEvent:
        if self._cancelled_event_factory is not None:
            event = self._cancelled_event_factory(row, now_ts)
        else:
            event = OrderEvent(
                order=ExchangeOrder(
                    id=row.exchange_order_id or '', symbol=row.symbol,
                    side=row.side, order_type=OrderType.MARKET,
                    qty=row.qty, filled_qty=row.filled_qty,
                    remaining_qty=max(0.0, row.qty - row.filled_qty),
                    price=None, stop_price=None, average_fill_price=None,
                    status=OrderStatus.CANCELLED, timestamp=now_ts, fee=0.0,
                    fee_currency='', reduce_only=False,
                    client_order_id=row.client_order_id,
                ),
                event_type='cancelled',
                fill_price=None, fill_qty=None, timestamp=now_ts,
                pine_id=row.pine_entry_id, from_entry=row.from_entry,
                leg_type=LegType.ENTRY,
            )
        # Stamp the tracker-origin marker on both the default and the
        # plugin-supplied event: the policy already ran in
        # :meth:`_apply_policy` before this event is emitted, so the sync
        # engine's WS-push handler must NOT re-apply it.
        event.from_disappearance_tracker = True
        event.cancel_reason = cancel_reason
        return event

    @staticmethod
    def _default_fill_event(
            row: 'OrderRow',
            confirmation: MissingConfirmation,
            cumulative: float,
            fill_qty: float,
            now_ts: float,
    ) -> OrderEvent:
        full = cumulative >= row.qty - _QTY_EPS
        timestamp = confirmation.executed_ts or now_ts
        # A lone backing execution id is a stable duplicate-fill key; an
        # aggregate over several executions has none — the plugin's own
        # dedup channel (via ``register_executions``) must cover it.
        fill_id = (confirmation.execution_ids[0]
                   if len(confirmation.execution_ids) == 1 else None)
        return OrderEvent(
            order=ExchangeOrder(
                id=(confirmation.position_ref
                    or row.exchange_order_id or ''),
                symbol=row.symbol, side=row.side,
                order_type=OrderType.MARKET,
                qty=row.qty, filled_qty=cumulative,
                remaining_qty=max(0.0, row.qty - cumulative),
                price=None, stop_price=None,
                average_fill_price=confirmation.fill_price,
                status=(OrderStatus.FILLED if full
                        else OrderStatus.PARTIALLY_FILLED),
                timestamp=timestamp, fee=confirmation.fill_fee,
                fee_currency='', reduce_only=False,
                client_order_id=row.client_order_id,
            ),
            event_type='filled' if full else 'partial',
            fill_price=confirmation.fill_price, fill_qty=fill_qty,
            timestamp=timestamp,
            pine_id=row.pine_entry_id, from_entry=row.from_entry,
            leg_type=LegType.ENTRY, fee=confirmation.fill_fee,
            fill_id=fill_id,
        )
