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
  policy separately decides the operational reaction. ``stop`` /
  ``stop_and_cancel`` QUARANTINE through the ``request_quarantine`` hook:
  trading stops (the engine blocks new / exposure-increasing dispatch)
  but the process — and this tracker's ingestion loop — stays alive;
  raising here instead would tear down the very event stream the
  quarantine invariant requires to keep running. Only the explicit
  ``halt`` policy (or a quarantining policy with no hook wired — the
  fail-safe fallback) arms the process-exiting halt. Persistence and
  policy application run BEFORE the event is yielded, so neither depends
  on the consumer pulling another element from the generator; a pending
  halt survives an abandoned generator and re-raises on the next
  :meth:`DisappearanceTracker.observe` call. The halt is delivered
  exactly once — each raise site consumes it.
"""
import logging
import math
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
    'resolve_unexpected_cancel_policy',
]

logger = logging.getLogger(__name__)

#: The persisted observation breadcrumb key in ``OrderRow.extras``. The
#: same key both reference plugins already use, so a refactor onto the
#: tracker inherits stamps persisted by earlier plugin versions.
MISSING_PENDING_EXTRA = 'missing_pending_since'

#: Valid ``on_unexpected_cancel`` policies (see ``BrokerDefaults``).
UNEXPECTED_CANCEL_POLICIES = (
    'stop', 'stop_and_cancel', 're_place', 'ignore', 'halt',
)

#: Float comparison slack for fill quantities, matching the reconcile
#: paths in the reference plugins.
_QTY_EPS = 1e-9


def resolve_unexpected_cancel_policy(
        policy: str,
        *,
        reason: str,
        context: dict[str, Any],
        request_quarantine: Callable[[str, dict[str, Any]], None] | None,
        log_event: Callable[..., None] | None,
        client_order_id: str | None,
        exchange_order_id: str | None,
) -> UnexpectedCancelError | None:
    """Resolve the ``on_unexpected_cancel`` policy for a confirmed
    unexpected cancel — the mapping shared by the grace-window tracker path
    (:meth:`DisappearanceTracker._apply_policy`) and the sync engine's
    WS-push handler
    (:meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._apply_unexpected_cancel_policy`).

    Applies the ``ignore`` / ``re_place`` audit-only logging and the
    ``stop`` / ``stop_and_cancel`` quarantine latch, and RETURNS the halt
    the caller must arm its own way (the tracker parks it on
    :attr:`pending_halt`; the engine records + raises it) — or ``None``
    when the outcome needed no halt (ignored, re-placed, or quarantined).
    The fail-closed rule lives here: a ``stop`` / ``stop_and_cancel``
    whose ``request_quarantine`` sink is missing or raises falls back to
    the halt, never to continued trading. The caller owns the
    ``stop_and_cancel`` sibling sweep — it is async and shaped differently
    per path.

    :param request_quarantine: Quarantine latch sink; ``None`` (an unwired
        sink under a quarantining policy) triggers the fail-closed halt.
    :param log_event: Audit sink (``RunContext.log_event``); ``None`` skips
        audit persistence (a store-less engine on the push path).
    """
    def _log(kind: str) -> None:
        if log_event is not None:
            log_event(
                kind,
                client_order_id=client_order_id,
                exchange_order_id=exchange_order_id,
            )

    if policy == 'ignore':
        _log('unexpected_cancel_ignored')
        return None
    if policy == 're_place':
        _log('unexpected_cancel_re_place')
        return None
    quarantined = False
    if (policy in ('stop', 'stop_and_cancel')
            and request_quarantine is not None):
        # noinspection PyBroadException
        try:
            request_quarantine(reason, context)
        except Exception:
            logger.exception(
                "request_quarantine hook failed for %r; falling back "
                "to the process-exiting halt", client_order_id,
            )
        else:
            quarantined = True
            _log('unexpected_cancel_quarantine')
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
    """
    resolution: MissingResolution
    cumulative_filled_qty: float | None = None
    fill_price: float | None = None
    fill_fee: float = 0.0
    execution_ids: tuple[str, ...] = ()
    position_ref: str | None = None
    extras_patch: Mapping[str, Any] | None = None
    executed_ts: float | None = None

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
    :param cancel_siblings: Async best-effort cancel sweep over the
        origin row's sibling orders; required by (and only used for) the
        ``stop_and_cancel`` policy. Best-effort — a raising sweep is
        logged and does not swallow the armed quarantine / halt.
    :param request_quarantine: Sink that latches the engine's quarantine
        state (``(reason, context)``); the runner wires it to
        ``OrderSyncEngine.record_quarantine``. Used by the ``stop`` and
        ``stop_and_cancel`` policies. When it is missing (or raises),
        those policies fall back to arming the process-exiting halt —
        fail-safe, never fail-open into continued trading.
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
            request_quarantine: Callable[[str, dict[str, Any]], None] | None = None,
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
        self._request_quarantine = request_quarantine
        self._sibling_coids = sibling_coids
        self._register_executions = register_executions
        self._cancelled_event_factory = cancelled_event_factory
        self._fill_event_factory = fill_event_factory
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
            # fill slice was booked — a stale pre-fill row would let a sweep
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
            if applied and confirmation.resolution is MissingResolution.CANCELLED:
                await self._apply_policy(hook_row)
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
            self._store.upsert_order(coid, extras=extras)

    def _clear_stamp(self, coid: str) -> None:
        """Remove ``missing_pending_since`` atomically, preserving extras."""
        with self._store.transaction():
            fresh = self._store.get_order(coid)
            if fresh is not None:
                extras = fresh.extras or {}
                if MISSING_PENDING_EXTRA in extras:
                    merged = {k: v for k, v in extras.items()
                              if k != MISSING_PENDING_EXTRA}
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
        it to the post-commit hooks so a sweep never sees a stale pre-fill
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
                # Filled-then-closed: the exposure no longer exists, so no
                # fill event is emitted for it (the engine's position
                # reconcile owns the size side) — but any discovered fill
                # progress was persisted above for bookkeeping. The
                # retirement is concluded FROM the confirmation's execution
                # evidence, so its ids are registered even without a fresh
                # booked slice — a replayed push copy of an already-counted
                # deal must not resurface as a new fill after the retire.
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
                return [], True, register_ids, updated

            if resolution is MissingResolution.CANCELLED:
                # Terminal like CLOSED above: register the backing ids so
                # a replayed copy of the evidence cannot re-book after the
                # retire.
                register_ids = confirmation.execution_ids
                DispatchJournal(self._store).apply_reconcile_outcome(
                    updated.client_order_id,
                    ReconcileOutcome(
                        kind='terminal_close',
                        reason='missing_pending_grace_expired',
                        new_state='rejected',
                        audit_event='unexpected_cancel',
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

    async def _apply_policy(self, row: 'OrderRow') -> None:
        """Apply the ``on_unexpected_cancel`` policy for a confirmed cancel.

        Runs BEFORE the cancelled event is yielded: the sweep and the
        quarantine / halt decision must not depend on the consumer pulling
        further elements. ``stop`` and ``stop_and_cancel`` latch the
        engine's quarantine through the ``request_quarantine`` hook — the
        process (and this observation loop) keeps running; a missing or
        raising hook falls back to arming the halt, never to continued
        trading. ``halt`` always arms :attr:`pending_halt`. Either signal
        is armed BEFORE the best-effort sweep, so a raising sweep can
        never swallow it; a pending halt is raised after the row's events
        were yielded (or at the top of the next pass, when the generator
        was abandoned).
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
        if self._policy == 'stop_and_cancel' and self._cancel_siblings is not None:
            # noinspection PyBroadException
            try:
                await self._cancel_siblings(row)
            except Exception:
                logger.exception(
                    "cancel_siblings sweep failed for %r; quarantine/halt "
                    "stays armed",
                    row.client_order_id,
                )
                self._store.log_event(
                    'unexpected_cancel_sweep_failed',
                    client_order_id=row.client_order_id,
                    exchange_order_id=row.exchange_order_id,
                )

    # --- Default event builders ------------------------------------------------

    def _build_cancelled_event(
            self, row: 'OrderRow', now_ts: float,
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
