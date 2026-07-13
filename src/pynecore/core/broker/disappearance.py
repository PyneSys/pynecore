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
  out of every snapshot for one poll.
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
  fill slice, and a crash cannot leave a half-applied resolution.
- **Stamp-version guard.** The row can come back (or be re-stamped)
  while the async confirmation runs; the apply transaction re-checks the
  original ``missing_pending_since`` value and drops a stale outcome.
- **Dual signal.** A confirmed unexpected cancel is booked as a terminal
  close AND yielded as a synthesised ``cancelled`` :class:`OrderEvent`
  (the engine's router cleans its tracking — essential under the
  non-halting policies), while the configured ``on_unexpected_cancel``
  policy separately decides whether the bot also halts. Persistence and
  policy application run BEFORE the event is yielded, so neither depends
  on the consumer pulling another element from the generator; a pending
  halt survives an abandoned generator and re-raises on the next
  :meth:`DisappearanceTracker.observe` call.
"""
import logging
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
from typing import TYPE_CHECKING, Any

from pynecore.core.broker.exceptions import UnexpectedCancelError
from pynecore.core.broker.journal import (
    DispatchJournal,
    ReconcileOutcome,
    ReconcileReason,
)
from pynecore.core.broker.models import (
    ExchangeOrder,
    LegType,
    OrderEvent,
    OrderStatus,
    OrderType,
)

if TYPE_CHECKING:
    from pynecore.core.broker.storage import OrderRow, RunContext

__all__ = [
    'MISSING_PENDING_EXTRA',
    'UNEXPECTED_CANCEL_POLICIES',
    'DisappearanceTracker',
    'MissingConfirmation',
    'MissingResolution',
]

logger = logging.getLogger(__name__)

#: The persisted observation breadcrumb key in ``OrderRow.extras``. The
#: same key both reference plugins already use, so a refactor onto the
#: tracker inherits stamps persisted by earlier plugin versions.
MISSING_PENDING_EXTRA = 'missing_pending_since'

#: Valid ``on_unexpected_cancel`` policies (see ``BrokerDefaults``).
UNEXPECTED_CANCEL_POLICIES = ('stop', 'stop_and_cancel', 're_place', 'ignore')

#: Float comparison slack for fill quantities, matching the reconcile
#: paths in the reference plugins.
_QTY_EPS = 1e-9


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
      position siblings) as a natural close — never a synthetic cancel,
      and no fill event for exposure that no longer exists.
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
    close.

    :ivar resolution: The classification — see :class:`MissingResolution`.
    :ivar cumulative_filled_qty: The order's proven cumulative filled
        quantity, when the verification discovered fill progress. The
        tracker clamps it into ``[row.filled_qty, row.qty]`` (monotonic,
        never overstating the order's own size) and books the delta.
    :ivar fill_price: Volume-weighted price of the discovered executions
        (the price the delta is booked at).
    :ivar fill_fee: Summed commission of the discovered executions.
    :ivar execution_ids: Venue execution/deal ids backing the evidence.
        Passed to the ``register_executions`` hook after a successful
        apply so the plugin can seed its duplicate-fill channel; a lone
        id is also stamped as the fill event's ``fill_id``.
    :ivar position_ref: Venue position id the fill belongs to. Selects
        the ``working_promoted_position`` audit reason and scopes the
        sibling retirement on :data:`MissingResolution.CLOSED`.
    :ivar extras_patch: Plugin extras merged into the row alongside a
        discovered fill (e.g. ``{'position_id': ...}``).
    :ivar executed_ts: Broker-reported execution time (unix seconds) for
        the fill event; falls back to the observation time.
    """
    resolution: MissingResolution
    cumulative_filled_qty: float | None = None
    fill_price: float | None = None
    fill_fee: float = 0.0
    execution_ids: tuple[str, ...] = ()
    position_ref: str | None = None
    extras_patch: Mapping[str, Any] | None = None
    executed_ts: float | None = None


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
    :param cancel_siblings: Async best-effort cancel sweep over the
        origin row's sibling orders; required by (and only used for) the
        ``stop_and_cancel`` policy.
    :param sibling_coids: Maps a :data:`MissingResolution.CLOSED` row to
        the client-order-ids of live sibling rows sharing its position
        (a netting account merges pyramid entries onto one position id);
        they are retired in the same transaction.
    :param register_executions: Called with the confirmation's
        ``execution_ids`` after a successful apply, so the plugin can
        seed its duplicate-fill channel before any replayed push event.
    :param cancelled_event_factory: Overrides the synthesised
        ``cancelled`` event construction (venues that key the event on
        something other than ``row.exchange_order_id``).
    :param fill_event_factory: Overrides the recovered-fill event
        construction. Receives ``(row, confirmation, cumulative,
        fill_qty, now_ts)``.
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
            cancel_siblings: Callable[['OrderRow'], Awaitable[None]] | None = None,
            sibling_coids: Callable[
                ['OrderRow', MissingConfirmation], Iterable[str]] | None = None,
            register_executions: Callable[
                ['OrderRow', tuple[str, ...]], None] | None = None,
            cancelled_event_factory: Callable[
                ['OrderRow', float], OrderEvent] | None = None,
            fill_event_factory: Callable[
                ['OrderRow', MissingConfirmation, float, float, float],
                OrderEvent] | None = None,
    ) -> None:
        if policy not in UNEXPECTED_CANCEL_POLICIES:
            raise ValueError(
                f"DisappearanceTracker: unknown policy {policy!r}; "
                f"expected one of {UNEXPECTED_CANCEL_POLICIES}"
            )
        if policy == 'stop_and_cancel' and cancel_siblings is None:
            raise ValueError(
                "DisappearanceTracker: policy 'stop_and_cancel' requires "
                "a cancel_siblings hook"
            )
        self._store = store_ctx
        self._grace_s = grace_s
        self._policy = policy
        self._tracked_refs = tracked_refs
        self._confirm_missing = confirm_missing
        self._is_exempt = is_exempt
        self._cancel_siblings = cancel_siblings
        self._sibling_coids = sibling_coids
        self._register_executions = register_executions
        self._cancelled_event_factory = cancelled_event_factory
        self._fill_event_factory = fill_event_factory
        #: Halt decided by a halting policy but not yet delivered to the
        #: consumer. Survives an abandoned generator: re-raised at the top
        #: of the next :meth:`observe` call.
        self._pending_halt: UnexpectedCancelError | None = None
        #: Rows already warned about an inconclusive grace re-check, so a
        #: sustained transport outage logs once per row, not per cadence.
        self._warned_inconclusive: set[str] = set()

    @property
    def pending_halt(self) -> UnexpectedCancelError | None:
        """The undelivered halt decided by a halting policy, if any."""
        return self._pending_halt

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
        if self._pending_halt is not None:
            raise self._pending_halt

        for row in list(self._store.iter_live_orders()):
            if self._is_exempt is not None and self._is_exempt(row):
                continue
            self._observe_presence(row, present, now_ts)

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
            events, applied = self._apply_confirmation(
                row, since_ts, confirmation, now_ts,
            )
            if applied and confirmation.execution_ids and \
                    self._register_executions is not None:
                self._register_executions(row, confirmation.execution_ids)
            if applied and confirmation.resolution is MissingResolution.CANCELLED:
                await self._apply_policy(row)
            for event in events:
                yield event
            if self._pending_halt is not None:
                raise self._pending_halt

    # --- Phase 1: stamp / clear against the present-sets --------------------

    def _observe_presence(
            self,
            row: 'OrderRow',
            present: Mapping[str, Set[str] | None],
            now_ts: float,
    ) -> None:
        refs = self._tracked_refs(row)
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
            if MISSING_PENDING_EXTRA in extras:
                self._clear_stamp(row)
            return
        if not all_fetched:
            # Incomplete snapshot: absence is unproven — neither stamp
            # nor clear.
            return
        if MISSING_PENDING_EXTRA not in extras:
            patched = dict(extras)
            patched[MISSING_PENDING_EXTRA] = now_ts
            self._store.upsert_order(row.client_order_id, extras=patched)

    def _clear_stamp(self, row: 'OrderRow') -> None:
        patched = {k: v for k, v in (row.extras or {}).items()
                   if k != MISSING_PENDING_EXTRA}
        self._store.upsert_order(row.client_order_id, extras=patched)
        self._warned_inconclusive.discard(row.client_order_id)

    # --- Phase 2: grace-expiry confirmation apply ----------------------------

    def _apply_confirmation(
            self,
            row: 'OrderRow',
            since: float,
            confirmation: MissingConfirmation,
            now_ts: float,
    ) -> tuple[list[OrderEvent], bool]:
        """Apply one confirmation atomically under the stamp-version guard.

        Returns ``(events, applied)`` — ``applied`` is ``False`` when the
        guard dropped a stale outcome (the row came back, was re-stamped
        or reached a terminal state while the confirmation ran).
        """
        resolution = confirmation.resolution
        if resolution is MissingResolution.INCONCLUSIVE:
            self._warn_inconclusive(row)
            return [], True

        events: list[OrderEvent] = []
        with self._store.transaction():
            fresh = self._store.get_order(row.client_order_id)
            if fresh is None or fresh.closed_ts_ms is not None:
                return [], False
            stamp = (fresh.extras or {}).get(MISSING_PENDING_EXTRA)
            if stamp != since:
                logger.info(
                    "disappearance confirmation for %r dropped: stamp "
                    "changed during verification (%r -> %r)",
                    row.client_order_id, since, stamp,
                )
                return [], False

            if resolution is MissingResolution.STILL_PRESENT:
                self._clear_stamp(fresh)
                return [], True

            fill_event = self._apply_discovered_fill(
                fresh, confirmation, now_ts,
            )

            if resolution is MissingResolution.FILLED:
                # The stamp premise is false — the ref vanished because it
                # filled. The row stays live; the venue's normal snapshot
                # promotion path owns it from here.
                latest = self._store.get_order(fresh.client_order_id)
                if latest is not None:
                    self._clear_stamp(latest)
                if fill_event is not None:
                    events.append(fill_event)
                return events, True

            if resolution is MissingResolution.CLOSED:
                # Filled-then-closed: the exposure no longer exists, so no
                # fill event is emitted for it (the engine's position
                # reconcile owns the size side) — but any discovered fill
                # progress was persisted above for bookkeeping.
                DispatchJournal(self._store).apply_reconcile_outcome(
                    fresh.client_order_id,
                    ReconcileOutcome(
                        kind='terminal_close',
                        reason='bracket_natural_close_followup',
                        new_state='closed',
                        audit_event='reconcile_filled_then_closed_retired',
                        close_row=True,
                        audit_payload={
                            'position_ref': confirmation.position_ref,
                            'missing_since': since,
                        },
                        exchange_order_id=(confirmation.position_ref
                                           or fresh.exchange_order_id),
                    ),
                )
                if self._sibling_coids is not None:
                    for sibling in self._sibling_coids(fresh, confirmation):
                        self._store.close_order(sibling)
                self._warned_inconclusive.discard(fresh.client_order_id)
                return [], True

            # CANCELLED — verified external cancel.
            DispatchJournal(self._store).apply_reconcile_outcome(
                fresh.client_order_id,
                ReconcileOutcome(
                    kind='terminal_close',
                    reason='missing_pending_grace_expired',
                    new_state='rejected',
                    audit_event='unexpected_cancel',
                    close_row=True,
                    audit_payload={'missing_since': since,
                                   'grace': self._grace_s},
                    exchange_order_id=fresh.exchange_order_id,
                ),
            )
            if fill_event is not None:
                events.append(fill_event)
            events.append(self._build_cancelled_event(fresh, now_ts))
            self._warned_inconclusive.discard(fresh.client_order_id)
            return events, True

    def _apply_discovered_fill(
            self,
            fresh: 'OrderRow',
            confirmation: MissingConfirmation,
            now_ts: float,
    ) -> OrderEvent | None:
        """Persist a fill slice discovered during confirmation.

        Runs inside the caller's transaction. Returns the fill event when
        new quantity was booked, ``None`` otherwise. The cumulative is
        clamped into ``[fresh.filled_qty, fresh.qty]`` — monotonic, never
        overstating the order's own size.
        """
        if confirmation.cumulative_filled_qty is None:
            return None
        cumulative = min(
            fresh.qty,
            max(fresh.filled_qty, confirmation.cumulative_filled_qty),
        )
        new_qty = cumulative - fresh.filled_qty
        if new_qty <= _QTY_EPS and not confirmation.extras_patch:
            return None
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
        if new_qty <= _QTY_EPS:
            return None
        if self._fill_event_factory is not None:
            return self._fill_event_factory(
                fresh, confirmation, cumulative, new_qty, now_ts,
            )
        return self._default_fill_event(
            fresh, confirmation, cumulative, new_qty, now_ts,
        )

    def _warn_inconclusive(self, row: 'OrderRow') -> None:
        coid = row.client_order_id
        if coid in self._warned_inconclusive:
            return
        self._warned_inconclusive.add(coid)
        logger.warning(
            "grace-expired row %r left un-retired: disappearance re-check "
            "inconclusive — deferring rather than concluding a false cancel",
            coid,
        )
        self._store.log_event(
            'missing_pending_recheck_inconclusive',
            client_order_id=coid,
            exchange_order_id=row.exchange_order_id,
        )

    # --- Policy ---------------------------------------------------------------

    async def _apply_policy(self, row: 'OrderRow') -> None:
        """Apply the ``on_unexpected_cancel`` policy for a confirmed cancel.

        Runs BEFORE the cancelled event is yielded: the sweep and the
        halt decision must not depend on the consumer pulling further
        elements. A halting policy only *arms* :attr:`pending_halt` here;
        the raise happens after the row's events were yielded (or at the
        top of the next pass, when the generator was abandoned).
        """
        if self._policy == 'ignore':
            self._store.log_event(
                'unexpected_cancel_ignored',
                client_order_id=row.client_order_id,
                exchange_order_id=row.exchange_order_id,
            )
            return
        if self._policy == 're_place':
            self._store.log_event(
                'unexpected_cancel_re_place',
                client_order_id=row.client_order_id,
                exchange_order_id=row.exchange_order_id,
            )
            return
        if self._policy == 'stop_and_cancel' and self._cancel_siblings is not None:
            await self._cancel_siblings(row)
        self._pending_halt = UnexpectedCancelError(
            f"Bot-owned order disappeared unexpectedly: "
            f"coid={row.client_order_id!r} "
            f"ref={row.exchange_order_id!r}",
            context={
                'client_order_id': row.client_order_id,
                'exchange_order_id': row.exchange_order_id,
                'symbol': row.symbol,
                'policy': self._policy,
            },
        )

    # --- Default event builders ------------------------------------------------

    def _build_cancelled_event(
            self, row: 'OrderRow', now_ts: float,
    ) -> OrderEvent:
        if self._cancelled_event_factory is not None:
            return self._cancelled_event_factory(row, now_ts)
        return OrderEvent(
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
