"""
:class:`OrderSyncEngine` — the bridge between the Pine Script order book and
a :class:`~pynecore.core.plugin.broker.BrokerPlugin`.

On each bar the engine:

1. Drains any :class:`OrderEvent` objects that the broker posted
   asynchronously (via :meth:`on_order_event`), routing fills to the
   :class:`~pynecore.core.broker.position.BrokerPosition` and unfreezing
   tick-based exits once their entry fill price is known.
2. Builds intents from the position's pending order dicts.
3. Runs the interceptor chain to let extensions reject or amend intents.
4. Diffs the resulting intent set against the previously-active one and
   dispatches the **new**, **modified** and **removed** intents to the
   plugin — tick-deferred exits are held back until the referenced entry
   has filled.
5. Every ``reconcile_every_n_syncs`` calls (optional) performs a read-side
   state reconciliation with the exchange.

The engine is synchronous; the broker plugin is async. :meth:`_run_async`
bridges the two, using ``run_coroutine_threadsafe`` on a background event
loop in live mode and ``asyncio.run`` for single-shot unit tests.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import queue
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pynecore.core.broker.exceptions import (
    BracketAttachAfterFillRejectedError,
    BrokerManualInterventionError,
    ExchangeConnectionError,
    OrderDispositionUnknownError,
    OrderSkippedByPlugin,
)
from pynecore.core.broker.intent_builder import build_intents
from pynecore.lib.log import (
    broker_info as _blog_info,
    broker_warning as _blog_warning,
    broker_error as _blog_error,
)
from pynecore.core.broker.models import (
    BrokerEvent,
    CancelIntent,
    CapabilityLevel,
    CloseIntent,
    DispatchEnvelope,
    EntryIntent,
    ExchangePosition,
    ExitIntent,
    InterceptorResult,
    LegPartialRepairedEvent,
    LegRepairFailedEvent,
    LegType,
    ManualInterventionRequiredEvent,
    OcaPartialFillPolicy,
    OcaType,
    OrderEvent,
)
from pynecore.core.broker.storage import (
    EnvelopeRecord,
    PendingRecord,
    RunContext,
)
from pynecore.types.na import na_float

if TYPE_CHECKING:
    from pynecore.core.broker.position import BrokerPosition
    from pynecore.core.plugin.broker import BrokerPlugin

__all__ = ['OrderSyncEngine']

_log = logging.getLogger(__name__)

Intent = EntryIntent | ExitIntent | CloseIntent


class OrderSyncEngine:
    """Translate Pine orders to broker calls and route fills back.

    :param broker: The concrete :class:`BrokerPlugin` instance to drive.
    :param position: The live :class:`BrokerPosition` this engine updates.
    :param symbol: The trading symbol (as the plugin expects it).
    :param run_tag: 4-char base36 session tag (see
        :meth:`~pynecore.core.broker.run_identity.RunIdentity.make_run_tag`) — seeds
        every :class:`DispatchEnvelope` this engine builds, so restarting the
        same script under the same config regenerates the same
        ``client_order_id`` values and the exchange dedups duplicates.
    :param event_loop: A running ``asyncio`` loop on which to execute the
        broker's coroutines. Pass ``None`` for unit tests — each broker call
        will then spin up a transient loop via ``asyncio.run``.
    :param execute_timeout: Seconds to wait for any single ``execute_*``
        coroutine when bridging from a background loop.
    :param reconcile_every_n_syncs: If non-zero, perform a read-side
        reconciliation every N :meth:`sync` calls.
    :param mintick: Symbol minimum tick — used to resolve tick-based exits
        (``profit=`` / ``loss=`` / ``trail_points=``) into absolute prices.
    :param oca_partial_fill_policy: How OCA-cancel groups react to partial
        fills (see :class:`OcaPartialFillPolicy`). Defaults to
        :data:`OcaPartialFillPolicy.FILL_CANCELS` — matches the Pine
        backtester, which treats the first touch as the winning leg.
    :param broker_event_sink: Optional callable invoked for structured
        broker-side :class:`BrokerEvent` objects (bracket repairs, overfill
        guards, ...). ``None`` disables emission — useful in tests and
        single-shot backtests; production wires the runner's observability
        bus here.
    :param store_ctx: Optional :class:`RunContext` from the unified
        :class:`BrokerStore`. When provided the engine persists envelope
        identity and parked-verification entries through it; on construction
        the context is replayed (``SELECT``-ed from SQLite) so a restarted
        process re-uses the same ``client_order_id`` for every live intent
        and matches up parked dispatches against ``get_open_orders`` on the
        next sync. Pass ``None`` for unit tests and single-shot backtests
        where restart safety is not required. The context also carries the
        ``run_tag`` the engine needs — but the engine still takes ``run_tag``
        explicitly so test paths that do not use a storage context can run.
    """

    def __init__(
            self,
            broker: 'BrokerPlugin',
            position: 'BrokerPosition',
            symbol: str,
            *,
            run_tag: str,
            event_loop: asyncio.AbstractEventLoop | None = None,
            execute_timeout: float = 30.0,
            reconcile_every_n_syncs: int = 0,
            mintick: float = 0.01,
            oca_partial_fill_policy: OcaPartialFillPolicy = OcaPartialFillPolicy.FILL_CANCELS,
            broker_event_sink: Callable[[BrokerEvent], None] | None = None,
            store_ctx: RunContext | None = None,
    ) -> None:
        self._broker = broker
        self._position = position
        self._symbol = symbol
        self._run_tag = run_tag
        self._loop = event_loop
        self._timeout = execute_timeout
        self._reconcile_every = reconcile_every_n_syncs
        self._mintick = mintick
        self._oca_partial_policy = oca_partial_fill_policy
        self._broker_event_sink = broker_event_sink
        self._store_ctx = store_ctx
        # Capabilities are declared once at plugin startup — cache the lookup
        # so the cascade-cancel fast path does not pay a method call per event.
        # Only the NATIVE level suppresses the engine fallback path:
        # PARTIAL_NATIVE keeps the engine running because the exchange only
        # owns part of the semantics, and we cannot safely guess which part.
        # Over-cancel / over-amend is idempotent; under-cancel leaves an open
        # exposure on the book.
        caps = broker.get_capabilities()
        self._oca_cancel_native = caps.oca_cancel is CapabilityLevel.NATIVE
        self._tp_sl_bracket_native = caps.tp_sl_bracket is CapabilityLevel.NATIVE

        self._active_intents: dict[str, Intent] = {}
        self._order_mapping: dict[str, list[str]] = {}
        self._envelopes: dict[str, DispatchEnvelope] = {}
        self._pending_verification: dict[str, DispatchEnvelope] = {}
        # Keyed by ``ExitIntent.intent_key`` (``f"{pine_id}\x00{from_entry}"``)
        # so pyramiding entries with multiple tick-deferred exits per
        # ``from_entry`` each get their own slot.
        self._deferred_exits: dict[str, ExitIntent] = {}
        self._event_queue: queue.Queue[OrderEvent] = queue.Queue()
        self._exchange_position: ExchangePosition | None = None
        self._interceptors: list[Callable[[Intent], InterceptorResult]] = []
        self._sync_count = 0
        self._current_bar_ts_ms: int = 0
        # OCA groups already processed inside the current :meth:`sync` pass.
        # Cleared at the start of every sync so a fresh bar re-enables cascade,
        # but kept stable within the pass so two fills in the same group do
        # not emit duplicate CancelIntents.
        self._cancelled_oca_groups_this_sync: set[str] = set()

        # Keys that received an empty :attr:`_order_mapping` adoption marker
        # from :meth:`_consume_plugin_resolutions` during the current sync.
        # The marker tells :meth:`_diff_and_dispatch` "this intent is already
        # live, just adopt it" — but if no Pine intent shows up under the same
        # key in the same sync, the marker becomes a stale trap that would
        # silently absorb a *future* same-key dispatch into the gone-position's
        # slot. End-of-sync cleanup drops markers that no intent claimed.
        self._attached_adoption_keys: set[str] = set()

        # Per-run dedup for ``'attached'`` resolutions. The persisted row
        # is intentionally NOT deleted on the first attached consume —
        # otherwise a late ``'rejected'`` write (per-leg bracket
        # resolvers, cross-poll re-evaluation) would update zero rows
        # and the engine would never learn the leg is missing. Keeping
        # the row alive lets the sticky-rejected SQL flip it; this set
        # prevents re-processing the same attached row on every sync
        # while it lives. Cleared on ``'rejected'`` flip-back so a
        # future re-park (different coid) is processed correctly.
        self._consumed_attached_coids: set[str] = set()

        # Pre-modify intent rollback table — keyed by ``intent_key``, set
        # only when ``_dispatch_modify`` parks a timed-out amend. The
        # current ``_diff_and_dispatch`` flow promotes ``_active_intents[key]``
        # to the NEW intent immediately after ``_dispatch_modify`` returns
        # (line 1328), so by the time a parked modify resolves as
        # ``'rejected'`` the engine no longer remembers the pre-modify
        # state — leaving ``_active_intents[key]`` set to the NEW intent
        # would make the next diff observe Pine == active and never retry
        # the amend, leaving the original exchange order indefinitely
        # stale. Restoring from this dict on the rejected path forces
        # the next diff to see a Pine-vs-active delta again and re-emit
        # ``modify_*``.
        #
        # Limitation: in-memory only — a restart between park and
        # resolution loses the snapshot, so the post-restart rejected
        # path falls back to the previous Round 19 behaviour (preserve
        # promoted active, accept the slim chance of a stale amend).
        # Persisting the snapshot on the ``pending_verifications`` row
        # would close that gap but requires an :class:`Intent`
        # serialisation contract; not worth the surface area until a
        # real restart-during-park incident motivates it.
        self._modify_old_intents: dict[str, Intent] = {}

        # Manual-intervention halt flag. Once set (via :meth:`_record_halt`),
        # every subsequent :meth:`sync` returns early without dispatching or
        # draining events — the strategy must be restarted after the operator
        # resolves the broker-side ambiguity. Plugins signal the halt by
        # raising :class:`BrokerManualInterventionError` from any ``execute_*``
        # or ``watch_orders``; the engine catches once, emits a
        # :class:`ManualInterventionRequiredEvent`, and re-raises so the
        # runner performs a graceful stop.
        self._halted: bool = False
        self._halted_reason: str | None = None
        self._halted_intent_key: str | None = None
        self._halted_context: dict = {}

        # Cross-restart recovery anchors. The state store persists envelope
        # identity and parked-verification entries; replay rebuilds these
        # *anchor* dicts (intent objects are not persisted — they are rebuilt
        # from the Pine order book on the first post-restart sync). The first
        # _build_envelope / _verify_pending_dispatches call for an anchored key
        # promotes the anchor into the live in-memory state and clears it.
        self._persisted_envelope_anchors: dict[str, EnvelopeRecord] = {}
        self._persisted_pending_anchors: dict[str, PendingRecord] = {}
        if store_ctx is not None:
            envelopes, pending = store_ctx.replay()
            self._persisted_envelope_anchors = dict(envelopes)
            self._persisted_pending_anchors = dict(pending)
            if envelopes or pending:
                _log.info(
                    "broker state replay: %d envelope(s), %d pending verification(s)",
                    len(envelopes), len(pending),
                )

    # === Public API ===

    @property
    def active_intents(self) -> dict[str, Intent]:
        return self._active_intents

    @property
    def deferred_exits(self) -> dict[str, ExitIntent]:
        return self._deferred_exits

    @property
    def order_mapping(self) -> dict[str, list[str]]:
        return self._order_mapping

    @property
    def pending_verification(self) -> dict[str, DispatchEnvelope]:
        """Envelopes whose exchange-side disposition is still unknown."""
        return self._pending_verification

    @property
    def exchange_position(self) -> ExchangePosition | None:
        """Latest position snapshot returned by the broker, if any."""
        return self._exchange_position

    def register_interceptor(
            self, fn: Callable[[Intent], InterceptorResult],
    ) -> None:
        """Add an interceptor that may reject or amend intents before dispatch."""
        self._interceptors.append(fn)

    def on_order_event(self, event: OrderEvent) -> None:
        """Queue a broker :class:`OrderEvent` for processing on the next sync.

        Called from the :meth:`run_event_stream` background task or by
        tests injecting synthetic events.
        """
        self._event_queue.put(event)

    async def run_event_stream(self) -> None:
        """Drain :meth:`BrokerPlugin.watch_orders` into the event queue.

        Meant to run as a long-lived task on the shared live-provider event
        loop. If the plugin does not implement WebSocket streaming, the
        method logs and returns — the engine then relies on
        :meth:`reconcile` for fill detection.

        Each incoming event is logged immediately on arrival — the actual
        :meth:`_route_event` processing is deferred to the next
        :meth:`_drain_events` call (i.e. the next bar's :meth:`sync`), so
        a log emitted only at drain time would falsely tag the fill with
        the *next* bar's ``bar_index``.  Logging here, on the broker event
        loop, captures the moment the broker actually observed the
        transition.
        """
        try:
            stream = self._broker.watch_orders()
        except NotImplementedError:
            _log.info(
                "broker does not implement watch_orders; "
                "reconcile() will poll for fills instead",
            )
            return
        try:
            async for event in stream:
                _blog_info("event %s", event)
                self._event_queue.put(event)
        except NotImplementedError:
            _log.info(
                "broker does not implement watch_orders; "
                "reconcile() will poll for fills instead",
            )
            return
        except asyncio.CancelledError:
            raise
        except BrokerManualInterventionError as e:
            self._record_halt(e)
            raise
        except Exception:  # pragma: no cover — defensive
            _log.exception("watch_orders stream terminated with an error")
            raise

    @property
    def halted(self) -> bool:
        """``True`` once :meth:`_record_halt` has latched a manual-intervention halt."""
        return self._halted

    def raise_if_halted(self) -> None:
        """Re-raise the latched halt as :class:`BrokerManualInterventionError`.

        Cheap, drain-free check meant for the script runner's tick loop:
        an async halt set on the broker event-loop thread (e.g. from
        :meth:`run_event_stream` reacting to an
        :class:`UnexpectedCancelError`) should surface in the runner thread
        on the very next tick — *not* one full bar later when
        :meth:`apply_async_events` runs at bar close.
        """
        if self._halted:
            raise BrokerManualInterventionError(
                self._halted_reason or "manual intervention required",
                intent_key=self._halted_intent_key,
                context=dict(self._halted_context),
            )

    def apply_async_events(self) -> None:
        """Drain any async-arrived broker events into the position state.

        Call this from the script runner BEFORE running the user script on
        each bar.  Without it, fills observed asynchronously between bars
        only become visible to ``position.size`` when the next bar's
        :meth:`sync` runs (i.e. AFTER that bar's script has executed),
        leaving the script's view of the position one bar stale.

        Also propagates an async-recorded halt (e.g. an
        :class:`UnexpectedCancelError` from ``run_event_stream``) so the
        bar loop exits via its ``finally`` block instead of running the
        script with stale state.
        """
        self.raise_if_halted()
        self._drain_events()

    def sync(self, bar_ts_ms: int) -> None:
        """Run one diff/dispatch cycle.

        Reads the Pine order book from ``position.entry_orders`` and
        ``position.exit_orders``, resolves tick-deferred exits where the
        referenced entry price is now known, and dispatches whatever
        changed to the broker plugin.

        :param bar_ts_ms: Current bar open timestamp in milliseconds — seeds
            every :class:`DispatchEnvelope` built in this cycle. The caller
            (typically the script runner) sources this from ``lib._time``.
        """
        # Surface a latched halt before any state mutation so an async halt
        # triggered from ``run_event_stream`` (e.g. an
        # :class:`UnexpectedCancelError` observed by the polling plugin)
        # exits the bar loop via its ``finally`` block instead of letting
        # the engine silently keep iterating.
        self.raise_if_halted()
        self._current_bar_ts_ms = bar_ts_ms
        self._cancelled_oca_groups_this_sync.clear()
        # Drain again here in case events arrived between
        # ``apply_async_events`` (start of this bar) and now.  ``sync`` is
        # also called from contexts that don't pre-drain (e.g. tests, the
        # backtest path with broker mode), so this remains the safety net.
        self._drain_events()
        try:
            self._verify_pending_dispatches()
        except ExchangeConnectionError as e:
            _blog_warning(
                "sync skipped after pending dispatch verification connection error: %s",
                e,
            )
            return

        raw = build_intents(
            self._position.entry_orders,
            self._position.exit_orders,
            self._symbol,
        )
        resolved = [self._resolve_ticks(i) for i in raw]
        final = self._apply_interceptors(resolved)

        dispatchable: list[Intent] = []
        new_deferred: dict[str, ExitIntent] = {}
        for i in final:
            if isinstance(i, ExitIntent) and i.has_unresolved_ticks:
                new_deferred[i.intent_key] = i
            else:
                dispatchable.append(i)
        self._deferred_exits = new_deferred

        self._diff_and_dispatch(dispatchable)
        self._cleanup_unused_adoption_markers()

        self._sync_count += 1
        if self._reconcile_every and self._sync_count % self._reconcile_every == 0:
            try:
                self.reconcile()
            except ExchangeConnectionError as e:
                _blog_warning(
                    "periodic reconcile skipped after connection error: %s",
                    e,
                )

    def _verify_pending_dispatches(self) -> None:
        """Match parked timeouts against the exchange's open-orders view.

        When a plugin raises :class:`OrderDispositionUnknownError` the sync
        engine cannot tell whether the order landed on the exchange; it parks
        the envelope here. Every subsequent :meth:`sync` calls this method
        first: it queries ``get_open_orders`` and, for each pending
        ``client_order_id`` that now appears on the exchange, promotes the
        envelope back into ``_order_mapping`` without re-dispatching.

        After a restart the persisted parked entries are also matched here —
        the in-memory envelope is gone, but the persisted ``key`` is enough to
        attach the recovered exchange order to the right ``_order_mapping``
        slot.

        For dispatches whose disposition the broker cannot expose through
        ``get_open_orders`` (e.g. position-attached brackets on Capital.com,
        which never show up there once attached), the plugin writes a
        resolution into the persisted park row via
        :meth:`~pynecore.core.broker.storage.RunContext.record_resolution`.
        This method consumes those resolutions first: ``'attached'`` clears
        the park (the dispatch is live, leave the active intent alone),
        ``'rejected'`` clears the park *and* drops the active intent so the
        next sync re-dispatches the original Pine intent.

        A pending entry that does *not* show up stays parked — the engine
        deliberately does not re-dispatch because the original may still land
        (slow network round-trip). The user can inspect
        :attr:`pending_verification` to surface stuck entries.
        """
        if self._store_ctx is not None:
            self._consume_plugin_resolutions()
        if not self._pending_verification and not self._persisted_pending_anchors:
            return
        orders = self._run_async(self._broker.get_open_orders(self._symbol))
        by_coid = {o.client_order_id: o for o in orders if o.client_order_id}
        for coid in list(self._pending_verification):
            order = by_coid.get(coid)
            if order is None:
                continue
            envelope = self._pending_verification.pop(coid)
            key = envelope.intent.intent_key
            current = self._order_mapping.setdefault(key, [])
            if order.id not in current:
                current.append(order.id)
            if self._store_ctx is not None:
                self._store_ctx.record_unpark(coid)
            _log.info(
                "recovered pending dispatch %s -> exchange order %s "
                "for intent %s", coid, order.id, key,
            )
        for coid in list(self._persisted_pending_anchors):
            order = by_coid.get(coid)
            if order is None:
                continue
            anchor = self._persisted_pending_anchors.pop(coid)
            current = self._order_mapping.setdefault(anchor.key, [])
            if order.id not in current:
                current.append(order.id)
            if self._store_ctx is not None:
                self._store_ctx.record_unpark(coid)
            _log.info(
                "recovered persisted pending dispatch %s -> exchange order %s "
                "for intent %s", coid, order.id, anchor.key,
            )

    def _consume_plugin_resolutions(self) -> None:
        """Apply plugin-driven resolutions written via ``record_resolution``.

        Brokers whose protective brackets are position-attributes (e.g.
        Capital.com native TP/SL) cannot expose them through
        ``get_open_orders``; the plugin's snapshot recovery determines the
        outcome and records it on the persisted park row. This method
        consumes those rows on every sync — see
        :meth:`_verify_pending_dispatches` for the contract.

        ``attached``: the dispatch landed; clear the in-memory park and
        leave ``_active_intents`` alone so the engine keeps treating the
        intent as live. The persisted ``pending_verifications`` row is
        intentionally NOT deleted here: a per-leg resolver (or a slow
        plugin re-evaluating the bracket on a later poll) may still
        flip the row to ``'rejected'`` via the sticky-rejected SQL in
        :meth:`RunContext.record_resolution`. Deleting eagerly would
        force that late UPDATE to find zero rows and the engine would
        never learn the leg is missing. The row is reaped instead by
        :meth:`_drop_envelope` (cancel / fill cleanup / rejected flip)
        or by :meth:`_cleanup_unused_adoption_markers` when no Pine
        intent claimed the marker.

        ``rejected``: the dispatch did not land; clear the park *and*
        drop the matching ``_active_intents`` / ``_order_mapping`` /
        envelope entries (in-memory + persisted) so
        :meth:`_diff_and_dispatch` re-dispatches the same Pine intent on
        the next sync with a fresh envelope (the bracket goes out again,
        restoring protection).
        """
        assert self._store_ctx is not None
        # Group resolutions by intent_key so a single key resolves
        # deterministically even when the snapshot contains multiple
        # rows under it. Multiple rows can occur when a bracket
        # resolver writes per-leg (Round 14 plugin-side aggregation
        # is the typical path, but other plugins or edge replay
        # sequences may still produce per-leg writes), or when a
        # restart finds older 'attached' rows alongside a newer
        # 'rejected' row that arrived during the same retry storm.
        # Within a single key, **'rejected' dominates**: any rejected
        # write means at least one expected exchange-side artefact is
        # confirmed missing, so the intent must be re-dispatched —
        # processing a stale 'attached' row last would otherwise
        # restore the adoption marker the rejected branch had cleared
        # and the engine would silently adopt an unverified dispatch
        # on the next ``_diff_and_dispatch``. The same precedence
        # rule already lives at the storage layer
        # (:meth:`RunContext.record_resolution`'s sticky-rejected SQL);
        # this loop mirrors it for the snapshot we just fetched.
        records_by_key: dict[str, list] = {}
        for record in self._store_ctx.iter_pending_resolutions():
            if record.resolution not in ('attached', 'rejected'):
                _log.error(
                    "ignoring unknown plugin resolution %r for coid=%s "
                    "intent=%s",
                    record.resolution, record.coid, record.key,
                )
                continue
            records_by_key.setdefault(record.key, []).append(record)

        for key, records in records_by_key.items():
            has_rejected = any(r.resolution == 'rejected' for r in records)
            if has_rejected:
                # ``dispatch_kind == 'modify'`` on ANY participating row
                # is enough to flag the whole group as a modify-rejected
                # event: only one parked record per (run_id, coid) ever
                # exists, and re-park overwrites the kind, so an
                # ``'attached'`` survivor row in the same group either
                # came from a prior new-dispatch attached consume that
                # was later flipped to rejected (still 'new') or from a
                # genuine modify amend bookkeeping. The conservative
                # treatment for the mixed case is the modify path:
                # preserving an already-live original order can never
                # produce a duplicate, while clearing it on a real
                # modify-rejected definitely can.
                kind_is_modify = any(
                    r.dispatch_kind == 'modify' for r in records
                )
                for record in records:
                    self._pending_verification.pop(record.coid, None)
                    self._persisted_pending_anchors.pop(record.coid, None)
                    # Drop any prior attached-consume dedup so a future
                    # re-park (different coid, same key) is processed
                    # correctly. Belt-and-braces — the new park gets a
                    # fresh coid which would not collide regardless.
                    self._consumed_attached_coids.discard(record.coid)
                    _log.warning(
                        "plugin-resolved pending dispatch %s as %s "
                        "for intent %s (kind=%s); rejected wins for key "
                        "— %s",
                        record.coid, record.resolution, key,
                        record.dispatch_kind,
                        ("scheduling re-dispatch"
                         if not kind_is_modify
                         else "preserving original order, "
                              "scheduling modify retry"),
                    )
                if not kind_is_modify:
                    # New-dispatch rejected: no original order exists
                    # on the exchange, so clear everything and let the
                    # next ``_diff_and_dispatch`` re-issue the Pine
                    # intent via ``execute_*``.
                    self._active_intents.pop(key, None)
                    self._order_mapping.pop(key, None)
                    # Defensive: a 'new' rejected record should never
                    # have left a modify rollback snapshot on the same
                    # key (snapshots are only stashed for kind='modify'
                    # parks), but drop any stale entry to avoid
                    # resurrecting it on a later modify rollback.
                    self._modify_old_intents.pop(key, None)
                else:
                    # Modify-rejected: the ORIGINAL exchange order is
                    # still live (the parked dispatch was an amend that
                    # the broker did NOT apply). Clearing
                    # ``_active_intents`` / ``_order_mapping`` here
                    # would make ``_diff_and_dispatch`` treat the Pine
                    # intent as brand new and call ``execute_*``,
                    # placing a SECOND order alongside the still-live
                    # original. Restore ``_active_intents[key]`` from
                    # the pre-modify snapshot captured at park time
                    # (:meth:`_park_pending`); without this restoration
                    # the slot still holds the NEW intent that
                    # :meth:`_diff_and_dispatch` promoted right after
                    # the parked ``_dispatch_modify`` returned, and the
                    # next diff observes Pine == active so the amend
                    # is silently dropped — leaving the original
                    # exchange order indefinitely on the OLD parameters.
                    # ``_order_mapping[key]`` is intentionally kept (it
                    # still references the live original order id).
                    restored = self._modify_old_intents.pop(key, None)
                    if restored is not None:
                        self._active_intents[key] = restored
                    else:
                        # Post-restart: the in-memory snapshot did not
                        # survive the process bounce. Without recovery,
                        # both ``_active_intents`` and ``_order_mapping``
                        # stay empty for this key, and
                        # ``_diff_and_dispatch`` issues a fresh
                        # ``execute_*`` alongside the still-live original
                        # — duplicating the order. Recover the exchange
                        # order IDs from the persisted park row (v4
                        # ``order_ids`` column) and seed
                        # ``_order_mapping`` so the cross-restart
                        # adoption path picks the intent up without
                        # re-dispatching. The adoption sets
                        # ``_active_intents[key]`` to the current Pine
                        # intent, which is the NEW (unapplied) intent —
                        # the exchange keeps the OLD parameters. A
                        # subsequent Pine parameter change triggers a
                        # normal modify retry; if Pine stays unchanged
                        # the desync persists as a documented limitation.
                        #
                        # The ``modify`` row is the authoritative source —
                        # ``_park_pending`` snapshots the live
                        # ``_order_mapping[key]`` only on the modify park.
                        # An older ``new``/``attached`` sibling row from
                        # the initial dispatch carries ``order_ids=[]``
                        # (parked before the mapping existed); SQL
                        # returns the group unordered, so picking
                        # ``records[0]`` could land on that empty
                        # snapshot and skip the recovery, duplicating
                        # the still-live original on the next dispatch.
                        recovered_ids: list[str] = []
                        for rec in records:
                            if (rec.dispatch_kind == 'modify'
                                    and rec.order_ids):
                                recovered_ids = list(rec.order_ids)
                                break
                        if not recovered_ids:
                            for rec in records:
                                if rec.order_ids:
                                    recovered_ids = list(rec.order_ids)
                                    break
                        if recovered_ids:
                            self._order_mapping[key] = recovered_ids
                            # Track for end-of-sync cleanup. Without this,
                            # if the first post-restart Pine pass no longer
                            # contains this intent (strategy cancelled it,
                            # position closed while the bot was down),
                            # :meth:`_diff_and_dispatch` has no
                            # ``_active_intents`` entry to remove and the
                            # recovered mapping silently adopts the next
                            # same-key dispatch — skipping the broker call
                            # entirely. Mirror the attached-path behaviour:
                            # the cleanup pass drops markers that no Pine
                            # intent claimed.
                            self._attached_adoption_keys.add(key)
                # The replayed envelope anchor is in-memory only —
                # ``_drop_envelope`` only touches ``_envelopes`` + the DB
                # row, so without this pop a same-sync rebuild would pin
                # the new envelope from the rejected dispatch's anchor
                # and reuse its ``client_order_id``. Brokers that retain
                # idempotency state for rejected submissions would then
                # dedupe the retry instead of accepting it as a new
                # order. Applies to both kinds: a fresh modify retry
                # also needs a fresh COID.
                self._persisted_envelope_anchors.pop(key, None)
                # ``_drop_envelope`` calls ``record_complete(key)`` which
                # already DELETEs every pending_verifications row sharing
                # this intent_key — no separate ``record_unpark`` needed.
                self._drop_envelope(key)
                # Drop any in-flight attached-adoption marker placed by
                # an earlier sync (or this loop iteration before the
                # grouping rewrite). Without this the next call to
                # :meth:`_cleanup_unused_adoption_markers` would not
                # touch it (the `_order_mapping` entry has just been
                # popped above for kind='new', or is still set for
                # kind='modify'), but a future same-sync attached
                # consume would re-add it from the now-already-cleaned
                # state. Belt-and-braces consistency.
                self._attached_adoption_keys.discard(key)
                continue

            # All resolutions for this key are 'attached' — install
            # the adoption marker once (vs. once per coid) and process
            # each row's per-coid bookkeeping.
            #
            # If any record under this key was a parked modify, drop the
            # rollback snapshot stashed by :meth:`_park_pending`. The
            # amend actually went through (despite the timeout) — the
            # promoted-new ``_active_intents[key]`` is correct, and
            # leaving the snapshot in place would make a *future*
            # genuine modify rollback to the wrong (now-superseded)
            # state if its first attempt also times out and is then
            # rejected.
            if any(r.dispatch_kind == 'modify' for r in records):
                self._modify_old_intents.pop(key, None)
            for record in records:
                coid = record.coid
                if coid in self._consumed_attached_coids:
                    # Already processed this attached resolution earlier
                    # in the same run. The row stays alive (so a late
                    # ``'rejected'`` flip can still be observed); the
                    # in-memory state is unchanged on subsequent syncs
                    # until either cleanup, retire, or a flip arrives.
                    continue
                self._consumed_attached_coids.add(coid)
                self._pending_verification.pop(coid, None)
                self._persisted_pending_anchors.pop(coid, None)
                _log.info(
                    "plugin-resolved pending dispatch %s as attached "
                    "for intent %s; keeping active intent", coid, key,
                )
            # After a restart ``_active_intents`` is empty (intents are
            # rebuilt from the Pine order book on the first post-restart
            # sync), so the upcoming :meth:`_diff_and_dispatch` would
            # dispatch this key again unless it sees an existing
            # :attr:`_order_mapping` slot. Mirror the same adoption
            # signal :meth:`_verify_pending_dispatches` uses for
            # recovered ``get_open_orders`` matches: a present (possibly
            # empty) mapping is the "already live, just adopt it" marker.
            # Capital.com's bracket legs do not surface real exchange
            # order ids on this path (their ``id`` is synthesised from
            # the parent ``dealId`` and is never returned by
            # ``get_open_orders``), so the empty-list shape matches how
            # those plugins already populate the slot.
            self._order_mapping.setdefault(key, [])
            # Track the marker so end-of-sync cleanup can drop it if no
            # Pine intent claimed it (e.g. the position has since closed
            # and the strategy moved on). Without this cleanup the empty
            # list would silently adopt a *future* same-key dispatch
            # and skip the broker call entirely.
            self._attached_adoption_keys.add(key)

    def _cleanup_unused_adoption_markers(self) -> None:
        """Drop adoption markers that no Pine intent claimed.

        :meth:`_consume_plugin_resolutions` seeds a slot in
        :attr:`_order_mapping` for two distinct cases:

        - ``'attached'`` resolution: an empty list, signalling "this
          intent is already live, just adopt it on the next diff".
        - Modify-rejected post-restart recovery: the original exchange
          order ids carried over from the persisted park row, so the
          next diff adopts the live original instead of dispatching a
          duplicate ``execute_*`` alongside it.

        Both shapes assume Pine still wants this intent. If the upcoming
        :meth:`_diff_and_dispatch` does not register an
        ``_active_intents[key]`` entry (strategy cancelled the intent,
        position closed while the bot was down), the slot becomes a
        stale trap: a *future* sync producing a fresh intent at the same
        key would hit the adoption branch and skip the broker dispatch
        entirely — a silent loss of order. The same logic applies to the
        envelope anchor: keeping it would pin the new intent to the old
        ``client_order_id``, which the broker may dedupe against the
        previously-attached order. The persisted ``envelopes`` and
        ``pending_verifications`` rows are deleted alongside the
        in-memory state; otherwise a *future restart* would replay the
        anchor and the same staleness would resurface.

        Cleanup runs once per sync, after :meth:`_diff_and_dispatch`. Only
        markers placed *this* sync are tracked (the set is cleared here),
        so legitimate adopters from earlier syncs are not affected.
        """
        for key in list(self._attached_adoption_keys):
            # Set membership already guarantees this key was seeded by
            # :meth:`_consume_plugin_resolutions` (either the attached
            # adoption path with an empty mapping, or the modify-rejected
            # post-restart recovery path with non-empty recovered ids).
            # Both shapes are stale traps if no Pine intent claimed them
            # this sync — drop them uniformly without inspecting the
            # mapping value.
            if (key not in self._active_intents
                    and key in self._order_mapping):
                self._order_mapping.pop(key, None)
                self._persisted_envelope_anchors.pop(key, None)
                # ``_drop_envelope`` removes the in-memory ``_envelopes``
                # entry (defensive — adoption never populated it for
                # this key, but cancel/retire paths use the same call)
                # and persists the cleanup via
                # :meth:`RunContext.record_complete`, which DELETEs the
                # ``envelopes`` row AND every ``pending_verifications``
                # row sharing this intent_key. Without that DELETE a
                # future restart would replay the stale anchor through
                # :meth:`_build_envelope` and reuse the old
                # ``client_order_id`` for a genuinely fresh order.
                self._drop_envelope(key)
                _log.info(
                    "cleared stale attached-adoption marker for intent %s "
                    "(no Pine intent claimed it this sync)", key,
                )
        self._attached_adoption_keys.clear()

    def reconcile(self) -> None:
        """Read-side position reconciliation with the exchange.

        The exchange is authoritative for position state. ``get_position`` is
        compared against ``self._position``: at startup the engine adopts the
        exchange size unconditionally; on periodic passes it only acts on
        shrink-to-zero (external flatten). No orders are ever **sent** from a
        reconciliation pass — that would risk duplicate entries.

        **What this method does NOT do:** it does not diff
        ``_order_mapping`` against ``get_open_orders``. Detecting bot-owned
        orders that disappear from the exchange (manual close from the broker
        UI, broker-side liquidation, exchange-side cancel) is a
        :class:`~pynecore.core.plugin.broker.BrokerPlugin` responsibility,
        because the relevant resource namespace is broker-specific (working
        orders vs open positions vs position-attached brackets vs child
        orders), and ``get_open_orders`` only sees one of those namespaces on
        most brokers. Plugins detect disappearance via their own internal
        snapshot loop and signal the engine through ``watch_orders`` —
        either by emitting a synthesised ``cancelled`` :class:`OrderEvent`
        (which the engine's ``_route_event`` cleans out of
        ``_order_mapping``) or by raising
        :class:`~pynecore.core.broker.exceptions.UnexpectedCancelError` for a
        graceful halt. See the Capital.com plugin's
        ``_reconcile_snapshot`` + ``_emit_unexpected_cancellations`` for the
        reference implementation.
        """
        exch_pos = self._run_async(self._broker.get_position(self._symbol))
        self._exchange_position = exch_pos
        if exch_pos is not None:
            self._position.openprofit = float(exch_pos.unrealized_pnl)
        elif self._position.size == 0.0:
            self._position.openprofit = 0.0
        # The exchange is the single source of truth for position state.
        # ``get_position`` returns ``None`` when no row exists for the symbol,
        # which is functionally a flat position — fold both branches into one
        # ``new_size`` comparison.
        new_size = exch_pos.size if exch_pos is not None else 0.0

        # Periodic reconcile (``sync_count > 0``) only acts on **shrink-to-zero**
        # transitions: the exchange went flat while we still think we hold a
        # position (manual web-UI close, broker liquidation, …). Any other
        # mismatch — including ``new_size > internal`` — could be a fill that
        # raced /positions ahead of the activity stream; adopting it would
        # double-count the moment the matching ``record_fill`` finally drains.
        # The startup call (``sync_count == 0``) still adopts unconditionally
        # so a fresh process restart over an existing exchange position does
        # not double-enter on the first bar.
        is_startup = self._sync_count == 0

        if is_startup and new_size != self._position.size:
            _blog_warning(
                "position size mismatch (exchange=%s, internal=%s) — "
                "adopting exchange",
                new_size, self._position.size,
            )
            self._position.size = new_size
            self._position.sign = (
                1.0 if new_size > 0.0
                else (-1.0 if new_size < 0.0 else 0.0)
            )
            if new_size == 0.0:
                self._position.avg_price = na_float
                self._position.open_trades.clear()
                self._position.openprofit = 0.0
                self._position.open_commission = 0.0
            else:
                self._position.avg_price = (
                    exch_pos.entry_price if exch_pos is not None else na_float
                )
        elif not is_startup and new_size == 0.0 and self._position.size != 0.0:
            # Skip while bot-initiated work is in flight — a close we
            # dispatched ourselves will flatten /positions seconds before
            # the matching ``OrderEvent`` reaches the queue. Pre-empting the
            # event here would zero the position; the closing fill (which
            # arrives with a non-zero ``signed_delta``) would then enter
            # ``record_fill``'s ``Opening`` branch and be miscounted as a
            # fresh entry in the opposite direction.
            if self._active_intents:
                return
            # External flatten detected — wipe ALL trade state so a re-entry
            # on the next bar starts from a clean slate. Leaving stale
            # ``open_trades`` would corrupt P&L bookkeeping the moment the
            # next ``record_fill`` runs (FIFO close against trades that no
            # longer exist on the broker).
            _blog_warning(
                "exchange shows flat, internal=%s — external close detected, "
                "clearing position state",
                self._position.size,
            )
            self._position.size = 0.0
            self._position.sign = 0.0
            self._position.avg_price = na_float
            self._position.open_trades.clear()
            self._position.openprofit = 0.0
            self._position.open_commission = 0.0

    # === Event routing ===

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                return
            self._route_event(event)

    def _route_event(self, event: OrderEvent) -> None:
        # Generic ``event %s`` arrival logging happens in
        # :meth:`run_event_stream` so the timestamp + ``bar_index`` reflect
        # when the broker observed the transition, not when this drain
        # pulled it off the queue.  Only intent-key-specific lines (which
        # need engine state) are emitted here.
        t = event.event_type
        if t in ('filled', 'partial'):
            self._position.record_fill(event)
            _blog_info(
                "position size=%s sign=%s avg=%s (after %s pine=%r)",
                self._position.size, self._position.sign,
                self._position.avg_price, t, event.pine_id,
            )
            if event.leg_type == LegType.ENTRY and event.pine_id:
                self._resolve_deferred_for_entry(event.pine_id)
                self._amend_bracket_qty_for_entry_fill(event)
            self._cascade_oca_cancel(event)
            # When a closing leg fully closes the position, drop the
            # entry + matching exit intents and clear Pine's order
            # dicts. Pine's ``strategy.exit`` is unconditional in most
            # scripts; the simulator gates it via ``open_trades`` —
            # broker mode needs equivalent gating, otherwise the next
            # bar's ``sync()`` rebuilds the same exit and dispatches a
            # pointless ``modify_exit`` against a position that no
            # longer exists on the broker (which on Capital.com fails
            # because the bracket-attached entry row is naturally
            # closed).
            if (event.leg_type in (LegType.TAKE_PROFIT, LegType.STOP_LOSS,
                                    LegType.TRAILING_STOP, LegType.CLOSE)
                    and self._position.size == 0):
                self._cleanup_closed_position(event)
        elif t == 'cancelled':
            key = self._find_key_for_order_id(event.order.id)
            if key is not None:
                _blog_error(
                    "unexpected cancel for intent %s (%s)",
                    key, event,
                )
                self._order_mapping.pop(key, None)
                self._active_intents.pop(key, None)
                self._drop_envelope(key)
            else:
                _blog_info(
                    "external cancel observed (%s)", event,
                )
        elif t == 'rejected':
            key = self._find_key_for_order_id(event.order.id)
            if key is not None:
                _blog_warning(
                    "order rejected for intent %s (%s)",
                    key, event,
                )
                self._order_mapping.pop(key, None)
                self._active_intents.pop(key, None)
                self._drop_envelope(key)
            else:
                _blog_warning(
                    "order rejected (%s)", event,
                )

    def _drop_envelope(self, key: str) -> None:
        """Remove envelope state for ``key`` and persist a ``complete`` marker.

        Called from every site that retires an intent (cancel dispatch,
        unexpected cancel event, reject event). The persisted marker lets the
        replay path skip the envelope and any still-pending verifications
        attached to the same ``key`` — keeping the JSONL self-compacting.
        """
        self._envelopes.pop(key, None)
        if self._store_ctx is not None:
            self._store_ctx.record_complete(key)

    def _find_key_for_order_id(self, order_id: str) -> str | None:
        for key, ids in self._order_mapping.items():
            if order_id in ids:
                return key
        return None

    def _resolve_deferred_for_entry(self, entry_id: str) -> None:
        """An entry fill unblocks every exit that references it via ticks.

        Pyramiding can attach multiple tick-deferred exits (different
        ``pine_id``) to the same ``from_entry``; all of them must be
        resolved on the entry's fill, not just one. Iterating with a
        snapshot of the keys lets the loop mutate ``_deferred_exits``
        safely.
        """
        matches = [
            (key, intent)
            for key, intent in self._deferred_exits.items()
            if intent.from_entry == entry_id
        ]
        for key, deferred in matches:
            del self._deferred_exits[key]
            resolved = self._resolve_ticks(deferred)
            if resolved.has_unresolved_ticks:
                self._deferred_exits[key] = deferred
                continue
            try:
                self._dispatch_new(resolved)
            except OrderSkippedByPlugin as e:
                _blog_warning("%s", e)
                continue
            self._active_intents[resolved.intent_key] = resolved

    # === OCA cascade cancel ===

    def _cascade_oca_cancel(self, event: OrderEvent) -> None:
        """Cancel OCA-cancel siblings of a freshly filled intent.

        Pine semantics: an ``oca_type='cancel'`` group keeps exactly one live
        leg at a time. The Pine backtester enforces this at fill time; this
        method is the live-trading equivalent — without it, a fill on leg A
        leaves leg B open until the next bar's diff pass, and a same-bar
        reversal may fill B too.

        The cascade is **suppressed** when:

        - The plugin declared ``oca_cancel = CapabilityLevel.NATIVE`` — the
          exchange registers and cancels the group natively.
        - The filled intent has no OCA group, or its type is not ``cancel``.
          (``reduce`` groups amend quantities on fill; that belongs to the
          partial-fill qty-amend workstream, not here.)
        - The partial-fill policy is :data:`OcaPartialFillPolicy.FULL_FILL_ONLY`
          and the event is ``partial``.
        - The group was already processed in this sync — prevents a
          double-fill (e.g. TP and entry both filling on the same bar) from
          emitting duplicate cancels.
        """
        if self._oca_cancel_native:
            return
        if event.event_type == 'partial' and (
                self._oca_partial_policy is OcaPartialFillPolicy.FULL_FILL_ONLY
        ):
            return

        filled_key = self._filled_intent_key(event)
        if filled_key is None:
            return
        filled_intent = self._active_intents.get(filled_key)
        if filled_intent is None:
            return
        oca_name = getattr(filled_intent, 'oca_name', None)
        oca_type = getattr(filled_intent, 'oca_type', None)
        if not oca_name or oca_type != OcaType.CANCEL.value:
            return
        if oca_name in self._cancelled_oca_groups_this_sync:
            return
        self._cancelled_oca_groups_this_sync.add(oca_name)

        siblings = [
            (key, intent)
            for key, intent in list(self._active_intents.items())
            if key != filled_key
            and getattr(intent, 'oca_name', None) == oca_name
            and getattr(intent, 'oca_type', None) == OcaType.CANCEL.value
        ]
        for key, intent in siblings:
            _log.info(
                "OCA cascade cancel: fill on %s cancels sibling %s in group %r",
                filled_key, key, oca_name,
            )
            self._active_intents.pop(key, None)
            self._remove_pine_order_for_intent(intent)
            self._dispatch_cancel(intent)

    def _remove_pine_order_for_intent(self, intent: Intent) -> None:
        """Delete the Pine-side :class:`Order` backing ``intent``.

        Mirrors :meth:`SimPosition._cancel_oca_group` for the live path: once
        an OCA-cancel sibling is cancelled exchange-side, the Pine-level order
        book must drop it too — otherwise the next :meth:`sync` rebuilds an
        intent from the stale entry and re-dispatches onto the now-cancelled
        exchange state.
        """
        entry_orders = getattr(self._position, 'entry_orders', None)
        exit_orders = getattr(self._position, 'exit_orders', None)
        if isinstance(intent, EntryIntent) and entry_orders is not None:
            entry_orders.pop(intent.pine_id, None)
        elif isinstance(intent, ExitIntent) and exit_orders is not None:
            exit_orders.pop((intent.pine_id, intent.from_entry), None)

    def _cleanup_closed_position(self, event: OrderEvent) -> None:
        """Drop tracking for an entry fully closed by a TP/SL/CLOSE fill.

        Identifies the closed entry's ``pine_id`` from the event:
        ``event.from_entry`` is set on bracket-leg fills emitted by
        plugins that own a separate exit order (Bybit, Binance USDM);
        ``event.pine_id`` carries it on plugins where the closing
        activity references the entry's own exchange id (Capital.com's
        position-attached bracket). Falling back across both fields
        keeps the cleanup correct on every plugin.

        Cleans:

        - ``_active_intents`` — entry intent keyed by ``pine_id``, every
          exit intent whose ``from_entry`` matches the closed entry.
        - ``_order_mapping`` and envelope state for the dropped keys.
        - ``position.entry_orders`` (single-key by ``pine_id``) and
          ``position.exit_orders`` (composite ``(exit_id, from_entry)``;
          every key whose ``from_entry`` matches is dropped).
        """
        closed_entry_id = event.from_entry or event.pine_id
        if not closed_entry_id:
            return
        # Entry intent + its mapping/envelope.
        self._active_intents.pop(closed_entry_id, None)
        self._order_mapping.pop(closed_entry_id, None)
        self._drop_envelope(closed_entry_id)
        # Every exit intent that points at this entry.
        for key in list(self._active_intents.keys()):
            intent = self._active_intents[key]
            if (isinstance(intent, ExitIntent)
                    and intent.from_entry == closed_entry_id):
                self._active_intents.pop(key, None)
                self._order_mapping.pop(key, None)
                self._drop_envelope(key)
        # Pine-side order dicts.
        entry_orders = getattr(self._position, 'entry_orders', None)
        exit_orders = getattr(self._position, 'exit_orders', None)
        if entry_orders is not None:
            entry_orders.pop(closed_entry_id, None)
        if exit_orders is not None:
            for ex_key in list(exit_orders.keys()):
                if isinstance(ex_key, tuple) and ex_key[1] == closed_entry_id:
                    exit_orders.pop(ex_key, None)

    def _filled_intent_key(self, event: OrderEvent) -> str | None:
        """Resolve a fill event to the ``intent_key`` of the owning intent.

        Exits track identity as ``(pine_id, from_entry)``; entries / closes
        as just ``pine_id``. An event coming from a plugin that did not tag
        the Pine identity cannot be routed and the method returns ``None``.
        """
        if event.pine_id is None:
            return None
        if event.leg_type in (LegType.TAKE_PROFIT, LegType.STOP_LOSS):
            if event.from_entry is None:
                return None
            return f"{event.pine_id}\0{event.from_entry}"
        return event.pine_id

    # === Partial entry fill → bracket qty amend (WS5, Option A) ===

    def _amend_bracket_qty_for_entry_fill(self, event: OrderEvent) -> None:
        """Track partial entry fills with an incremental bracket qty amend.

        Canonical semantics (Option A): the bracket's qty follows the entry's
        cumulative ``filled_qty`` — every partial fill dispatches a
        :meth:`BrokerPlugin.modify_exit` with ``new_qty = filled_qty``. This
        mirrors the Pine backtester, where exits exist against the actually
        filled entry portion; it also guarantees that if the entry ends with
        unfilled remainder (cancel/expire), the bracket is not over-sized.

        Suppressed when:

        - The plugin declared ``tp_sl_bracket = CapabilityLevel.NATIVE`` —
          the exchange tracks partial entry fills natively (Bybit V5
          attached TP/SL, Capital.com position-attribute bracket).
        - No bracket is active for ``event.pine_id`` (plain entry, no exit).
        - The current ExitIntent already matches the target qty — avoids
          redundant dispatch churn.

        Over-fill guard: if ``event.order.filled_qty`` exceeds the entry
        intent's intended qty (exchange rounding or adversarial event), the
        amend is capped at the intended qty and a
        :class:`LegRepairFailedEvent` is emitted so the runner can surface
        the anomaly.
        """
        if self._tp_sl_bracket_native:
            return
        pine_id = event.pine_id
        if pine_id is None:
            return

        filled_qty = event.order.filled_qty
        if filled_qty <= 0.0:
            return

        bracket_key: str | None = None
        bracket_intent: ExitIntent | None = None
        for key, intent in self._active_intents.items():
            if isinstance(intent, ExitIntent) and intent.from_entry == pine_id:
                bracket_key = key
                bracket_intent = intent
                break
        if bracket_key is None or bracket_intent is None:
            return

        entry_intent = self._active_intents.get(pine_id)
        target_qty = filled_qty
        overfill = False
        if isinstance(entry_intent, EntryIntent) and filled_qty > entry_intent.qty:
            target_qty = entry_intent.qty
            overfill = True

        if target_qty == bracket_intent.qty:
            if overfill:
                self._emit_overfill_event(
                    bracket_intent, entry_intent, filled_qty,
                )
            return

        old_qty = bracket_intent.qty
        new_intent = dataclasses.replace(bracket_intent, qty=target_qty)
        self._dispatch_modify(bracket_intent, new_intent)
        self._active_intents[bracket_key] = new_intent
        self._sync_pine_exit_qty(new_intent, target_qty)

        self._emit_broker_event(LegPartialRepairedEvent(
            pine_id=new_intent.pine_id,
            from_entry=new_intent.from_entry,
            leg='bracket',
            generation=0,
            old_qty=old_qty,
            new_qty=target_qty,
        ))
        if overfill:
            self._emit_overfill_event(new_intent, entry_intent, filled_qty)

    def _sync_pine_exit_qty(self, bracket: ExitIntent, new_qty: float) -> None:
        """Mutate the Pine-side exit :class:`Order` to match the amended qty.

        Without this, the next :meth:`sync` rebuilds the ExitIntent from the
        unchanged ``pos.exit_orders[(exit_id, from_entry)]`` (whose ``size``
        still equals the original full qty), the diff engine sees a mismatch
        against the amended active intent, and emits a *second* ``modify_exit``
        back to the original qty — undoing the partial-fill cascade we just did.
        """
        exit_orders = getattr(self._position, 'exit_orders', None)
        if exit_orders is None:
            return
        order = exit_orders.get((bracket.pine_id, bracket.from_entry))
        if order is None:
            return
        sign = 1.0 if order.size >= 0.0 else -1.0
        order.size = sign * new_qty
        order.sign = sign if new_qty > 0.0 else 0.0

    def _emit_overfill_event(
            self,
            bracket: ExitIntent,
            entry: 'Intent | None',
            filled_qty: float,
    ) -> None:
        entry_qty = entry.qty if isinstance(entry, EntryIntent) else None
        self._emit_broker_event(LegRepairFailedEvent(
            pine_id=bracket.pine_id,
            from_entry=bracket.from_entry,
            leg='bracket',
            reason=(
                f"overfill detected: filled_qty={filled_qty} exceeds "
                f"entry qty={entry_qty}"
            ),
            action_taken='capped',
        ))

    def _emit_broker_event(self, event: BrokerEvent) -> None:
        """Forward a structured broker event to the registered sink, if any."""
        if self._broker_event_sink is None:
            _log.info("broker event (no sink): %r", event)
            return
        try:
            self._broker_event_sink(event)
        except Exception:  # pragma: no cover — defensive
            _log.exception("broker_event_sink raised for event %r", event)

    def _record_halt(self, error: BrokerManualInterventionError) -> None:
        """Record a manual-intervention halt and emit the observability event.

        Idempotent — the first call latches the halt state and emits one
        :class:`ManualInterventionRequiredEvent`; subsequent calls (e.g. from
        a second dispatch path that also raised) are no-ops. After this the
        engine's :meth:`sync` returns early on every invocation until the
        strategy restarts.
        """
        if self._halted:
            return
        self._halted = True
        self._halted_reason = error.reason
        self._halted_intent_key = error.intent_key
        self._halted_context = dict(error.context)
        _blog_error(
            "sync engine halted by BrokerManualInterventionError: %s "
            "(intent_key=%s, context=%r)",
            error.reason, error.intent_key, error.context,
        )
        self._emit_broker_event(ManualInterventionRequiredEvent(
            reason=error.reason,
            intent_key=error.intent_key,
            context=dict(error.context),
        ))

    # === Tick resolution ===

    def _resolve_ticks(self, intent: Intent) -> Intent:
        if not isinstance(intent, ExitIntent) or not intent.has_unresolved_ticks:
            return intent
        entry_price, entry_sign = self._find_entry_reference(intent.from_entry)
        if entry_price is None:
            return intent
        return self._ticks_to_prices(intent, entry_price, entry_sign)

    def _find_entry_reference(
            self, from_entry: str,
    ) -> tuple[float | None, float]:
        for trade in self._position.open_trades:
            if trade.entry_id == from_entry:
                return trade.entry_price, trade.sign
        return None, 0.0

    def _ticks_to_prices(
            self, intent: ExitIntent, entry_price: float, entry_sign: float,
    ) -> ExitIntent:
        tp_price = intent.tp_price
        sl_price = intent.sl_price
        trail_price = intent.trail_price
        if intent.profit_ticks is not None:
            tp_price = entry_price + entry_sign * intent.profit_ticks * self._mintick
        if intent.loss_ticks is not None:
            sl_price = entry_price - entry_sign * intent.loss_ticks * self._mintick
        if intent.trail_points_ticks is not None:
            trail_price = (
                entry_price + entry_sign * intent.trail_points_ticks * self._mintick
            )
        return dataclasses.replace(
            intent,
            tp_price=tp_price,
            sl_price=sl_price,
            trail_price=trail_price,
            profit_ticks=None,
            loss_ticks=None,
            trail_points_ticks=None,
        )

    # === Interceptor chain ===

    def _apply_interceptors(self, intents: list[Intent]) -> list[Intent]:
        if not self._interceptors:
            return intents
        out: list[Intent] = []
        for intent in intents:
            current = intent
            rejected = False
            for fn in self._interceptors:
                result = fn(current)
                if result.rejected:
                    rejected = True
                    _log.info(
                        "intent %s rejected by interceptor: %s",
                        current.intent_key, result.reject_reason,
                    )
                    break
                current = self._apply_modifications(current, result)
            if not rejected:
                out.append(current)
        return out

    @staticmethod
    def _apply_modifications(
            intent: Intent, result: InterceptorResult,
    ) -> Intent:
        mods: dict[str, Any] = {}
        if result.modified_qty is not None:
            mods['qty'] = result.modified_qty
        if result.modified_limit is not None:
            if isinstance(intent, ExitIntent):
                mods['tp_price'] = result.modified_limit
            elif isinstance(intent, EntryIntent):
                mods['limit'] = result.modified_limit
        if result.modified_stop is not None:
            if isinstance(intent, ExitIntent):
                mods['sl_price'] = result.modified_stop
            elif isinstance(intent, EntryIntent):
                mods['stop'] = result.modified_stop
        return dataclasses.replace(intent, **mods) if mods else intent

    # === Diff + dispatch ===

    def _diff_and_dispatch(self, intents: list[Intent]) -> None:
        new_map: dict[str, Intent] = {i.intent_key: i for i in intents}

        # Pine semantic: when an entry intent fails to dispatch in this same
        # sync (e.g. plugin reports qty below venue minimum), a bracket exit
        # that references it via ``from_entry`` is a silent no-op — same as
        # Pine's own simulator returning at the missing-entry check
        # (``strategy/__init__.py``). We only short-circuit brackets whose
        # parent we just observed skipping; brackets that reference an
        # already-filled position (no entry intent in this sync) keep the
        # existing dispatch behaviour.
        skipped_entry_ids_this_sync: set[str] = set()

        for key in list(self._active_intents):
            if key not in new_map:
                old = self._active_intents.pop(key)
                # A still-parked modify for this key (Pine cancels while
                # the previous amend's resolution is pending) is being
                # superseded — its rollback snapshot would target an
                # intent the strategy no longer wants. Drop the snapshot
                # so a late ``'rejected'`` resolution does not resurrect
                # a cancelled key into ``_active_intents``.
                self._modify_old_intents.pop(key, None)
                self._dispatch_cancel(old)

        for key, intent in new_map.items():
            if key not in self._active_intents:
                if key in self._order_mapping:
                    # Cross-restart adoption: the persisted state recovered an
                    # exchange-side order for this intent (via
                    # _verify_pending_dispatches). Re-dispatching here would
                    # duplicate the order — instead, adopt the existing
                    # mapping and pin the envelope from the persisted anchor
                    # so subsequent modifies emit the same client_order_id.
                    self._build_envelope(intent)
                    self._active_intents[key] = intent
                else:
                    if (isinstance(intent, ExitIntent)
                            and intent.from_entry is not None
                            and intent.from_entry in skipped_entry_ids_this_sync):
                        # Parent entry was just skipped — drop the bracket
                        # silently. Re-evaluated next bar.
                        continue
                    try:
                        self._dispatch_new(intent)
                    except OrderSkippedByPlugin as e:
                        # Plugin declined (e.g. qty below venue minimum).
                        # Don't register — the intent is re-evaluated next
                        # bar so a later sizing change can still trade.
                        _blog_warning("%s", e)
                        if isinstance(intent, EntryIntent):
                            skipped_entry_ids_this_sync.add(intent.pine_id)
                        continue
                    self._active_intents[key] = intent
            elif intent != self._active_intents[key]:
                try:
                    self._dispatch_modify(self._active_intents[key], intent)
                except OrderSkippedByPlugin as e:
                    # The cancel+re-execute fallback inside _dispatch_modify
                    # cancelled the old order before the plugin declined the
                    # new one. The exchange now has nothing for this key, so
                    # drop it from active too.
                    _blog_warning("%s", e)
                    self._active_intents.pop(key, None)
                    continue
                self._active_intents[key] = intent
            # else: unchanged — skip

    def _build_envelope(self, intent: Intent) -> DispatchEnvelope:
        """Wrap an intent in a :class:`DispatchEnvelope`.

        The first envelope for a given ``intent_key`` is pinned on creation
        (bar_ts_ms, retry_seq frozen). Subsequent modifies re-use the same
        anchor so the ``client_order_id`` stays stable across amend cycles —
        that stability is what lets the exchange recognise a retry as a
        duplicate rather than a new order.

        After a restart, the anchor for an existing ``intent_key`` is
        reconstructed from the persisted :class:`BrokerStore` / :class:`RunContext`
        instead of being recomputed from ``_current_bar_ts_ms`` — the latter
        would yield a new ``client_order_id`` and break exchange-side dedup.
        """
        existing = self._envelopes.get(intent.intent_key)
        if existing is not None:
            return DispatchEnvelope(
                intent=intent,
                run_tag=existing.run_tag,
                bar_ts_ms=existing.bar_ts_ms,
                retry_seq=existing.retry_seq,
            )
        anchor = self._persisted_envelope_anchors.pop(intent.intent_key, None)
        if anchor is not None:
            envelope = DispatchEnvelope(
                intent=intent,
                run_tag=self._run_tag,
                bar_ts_ms=anchor.bar_ts_ms,
                retry_seq=anchor.retry_seq,
            )
            self._envelopes[intent.intent_key] = envelope
            return envelope
        envelope = DispatchEnvelope(
            intent=intent,
            run_tag=self._run_tag,
            bar_ts_ms=self._current_bar_ts_ms,
            retry_seq=0,
        )
        self._envelopes[intent.intent_key] = envelope
        if self._store_ctx is not None:
            self._store_ctx.record_envelope(
                key=intent.intent_key,
                bar_ts_ms=envelope.bar_ts_ms,
                retry_seq=envelope.retry_seq,
            )
        return envelope

    def _build_cancel_envelope(self, cancel: CancelIntent) -> DispatchEnvelope:
        return DispatchEnvelope(
            intent=cancel,
            run_tag=self._run_tag,
            bar_ts_ms=self._current_bar_ts_ms,
            retry_seq=0,
        )

    def _park_pending(
            self, envelope: DispatchEnvelope, error: OrderDispositionUnknownError,
            *, kind: str = 'new', old_intent: Intent | None = None,
    ) -> None:
        """Stash a dispatch whose exchange disposition the plugin could not confirm.

        :meth:`_verify_pending_dispatches` reruns ``get_open_orders`` on each
        subsequent sync and promotes the envelope back to
        ``_order_mapping`` once the order shows up.

        :param kind: ``'new'`` for ``execute_*`` parks, ``'modify'`` for
            ``modify_*`` parks. Persisted on the pending row so a later
            ``'rejected'`` resolution can decide whether the original
            exchange order is still live (and only the amend failed).
        :param old_intent: Only set for ``kind='modify'``. The pre-modify
            ``_active_intents[key]`` snapshot, captured BEFORE
            ``_diff_and_dispatch`` promotes the slot to the new intent
            (line 1328). Stashed in :attr:`_modify_old_intents` so a
            later ``'rejected'`` resolution can restore the slot and
            force the next diff to re-emit the amend — without this the
            promoted-new active matches Pine and the diff stays silent
            even though the exchange still holds the OLD order
            unmodified.
        """
        coid = error.client_order_id
        self._pending_verification[coid] = envelope
        # Re-parking the same coid in this engine instance must reset
        # the in-memory ``_consumed_attached_coids`` dedup. Without
        # this, if an earlier 'attached' resolution already marked the
        # coid as consumed and the row has since been re-parked
        # (record_park resets ``resolution`` to NULL on conflict), a
        # later 'attached' write on the fresh park would be skipped by
        # :meth:`_consume_plugin_resolutions` — leaving
        # ``_pending_verification`` stuck for brokers whose orders
        # never appear in ``get_open_orders`` (e.g. Capital.com
        # position-attached brackets).
        self._consumed_attached_coids.discard(coid)
        if kind == 'modify' and old_intent is not None:
            # First-park-wins: a chained modify (Pine flips parameters
            # while a previous amend is still parked) overwrites the
            # NEW intent in ``_active_intents`` but the EXCHANGE may
            # still be on the OLDEST pre-park state if every park ends
            # up rejected. The earliest captured snapshot is the safest
            # restoration target. ``setdefault`` preserves it; an
            # ``'attached'`` resolution clears the entry so a later
            # genuinely-fresh modify gets a clean snapshot.
            self._modify_old_intents.setdefault(
                envelope.intent.intent_key, old_intent,
            )
        if self._store_ctx is not None:
            self._store_ctx.record_park(
                coid=coid,
                key=envelope.intent.intent_key,
                kind=kind,
                order_ids=self._order_mapping.get(
                    envelope.intent.intent_key, [],
                ),
            )
        _log.warning(
            "dispatch for %s ended with unknown disposition "
            "(client_order_id=%s, kind=%s); will verify on next sync: %s",
            envelope.intent.intent_key, coid, kind, error,
        )

    def _dispatch_new(self, intent: Intent) -> None:
        envelope = self._build_envelope(intent)
        _blog_info("dispatching %s", intent)
        try:
            if isinstance(intent, EntryIntent):
                orders = self._run_async(self._broker.execute_entry(envelope))
                self._order_mapping[intent.intent_key] = [o.id for o in orders]
            elif isinstance(intent, ExitIntent):
                orders = self._run_async(self._broker.execute_exit(envelope))
                self._order_mapping[intent.intent_key] = [o.id for o in orders]
            elif isinstance(intent, CloseIntent):
                order = self._run_async(self._broker.execute_close(envelope))
                self._order_mapping[intent.intent_key] = [order.id]
            _blog_info(
                "dispatched %s -> %s",
                intent, self._order_mapping.get(intent.intent_key),
            )
        except OrderDispositionUnknownError as e:
            _blog_warning(
                "dispatch parked (unknown disposition) for %s: %s",
                intent, e,
            )
            self._park_pending(envelope, e)
        except BracketAttachAfterFillRejectedError as e:
            self._handle_bracket_attach_after_fill_reject(intent, e)
        except BrokerManualInterventionError as e:
            _blog_error(
                "dispatch halted (manual intervention) for %s: %s",
                intent, e,
            )
            self._record_halt(e)
            raise
        except OrderSkippedByPlugin:
            # Caller (_diff_and_dispatch / _resolve_deferred_for_entry) is
            # responsible for the warning + active-intents bookkeeping —
            # don't mislabel this as a dispatch failure.
            raise
        except Exception as e:
            _blog_error(
                "dispatch failed for %s: %s: %s",
                intent, type(e).__name__, e,
            )
            raise

    def _handle_bracket_attach_after_fill_reject(
            self, intent: Intent, e: BracketAttachAfterFillRejectedError,
    ) -> None:
        """Recover from a bracket attach reject after a parent fill committed.

        The plugin already rolled back the persisted bracket leg rows;
        the parent ENTRY/EXIT fill is on the exchange but has no
        protective TP/SL. Halting here would leave the unprotected
        position exposed indefinitely — instead synthesise a defensive
        :class:`CloseIntent` and dispatch it immediately to flatten the
        position.

        The original intent is then surfaced as
        :class:`OrderSkippedByPlugin` so the caller drops it from
        ``_active_intents``: the position it was bracketing no longer
        exists, and re-evaluating next bar lets the strategy rebuild
        from the actual current state instead of replaying the same
        rejected bracket.

        If the defensive close itself fails in an unrecoverable way
        (anything other than transient park / plugin skip), escalate
        to :class:`BrokerManualInterventionError` — at that point the
        position is open AND we couldn't auto-close it, an operator
        must intervene.
        """
        _blog_error(
            "bracket attach rejected after parent fill for %s; "
            "issuing defensive market close "
            "(deal_id=%s, side=%s, qty=%s): %s",
            intent, e.position_deal_id, e.position_side, e.qty, e,
        )
        close_side = "sell" if e.position_side == "buy" else "buy"
        defensive_pine_id = f"__pyne_defensive_close__{e.position_coid}"
        close_intent = CloseIntent(
            pine_id=defensive_pine_id,
            symbol=e.symbol,
            side=close_side,
            qty=e.qty,
            immediately=True,
            comment=f"defensive close after bracket attach reject: {e}",
        )
        try:
            self._dispatch_new(close_intent)
        except (OrderDispositionUnknownError, OrderSkippedByPlugin):
            # Defensive close parked (timeout) or skipped by plugin —
            # the next reconcile / next bar will resolve. At worst the
            # position remains open until then, no worse than not
            # issuing the close at all.
            pass
        except BrokerManualInterventionError as halt:
            # Inner _dispatch_new already recorded the halt before
            # propagating (see the BrokerManualInterventionError branch
            # in _dispatch_new) — re-raise verbatim.
            raise halt
        except Exception as nested:
            # Unexpected failure — escalate. We record the halt here
            # ourselves because this exception is being raised from a
            # call site (the outer _dispatch_new's
            # BracketAttachAfterFillRejectedError branch) that does
            # NOT pass back through the BrokerManualInterventionError
            # except in _dispatch_new, so _record_halt would otherwise
            # be skipped.
            halt = BrokerManualInterventionError(
                f"Defensive close after bracket attach reject failed for "
                f"{intent.intent_key}: {nested}",
                intent_key=intent.intent_key,
                context={
                    'position_deal_id': e.position_deal_id,
                    'position_coid': e.position_coid,
                    'symbol': e.symbol,
                    'qty': e.qty,
                    'cause': str(e),
                },
            )
            self._record_halt(halt)
            raise halt from nested
        raise OrderSkippedByPlugin(
            f"Bracket attach rejected after entry fill — parent position "
            f"closed defensively (deal_id={e.position_deal_id}); "
            f"intent re-evaluation deferred to next bar",
            intent_key=intent.intent_key,
            reason="bracket_reject_defensive_close",
            context={
                'position_deal_id': e.position_deal_id,
                'position_coid': e.position_coid,
                'symbol': e.symbol,
                'qty': e.qty,
            },
        ) from e

    def _dispatch_modify(self, old: Intent, new: Intent) -> None:
        old_env = self._build_envelope(old)
        new_env = self._build_envelope(new)
        _blog_info("modifying %s -> %s", old, new)
        try:
            if isinstance(new, EntryIntent) and isinstance(old, EntryIntent):
                orders = self._run_async(self._broker.modify_entry(old_env, new_env))
                self._order_mapping[new.intent_key] = [o.id for o in orders]
            elif isinstance(new, ExitIntent) and isinstance(old, ExitIntent):
                orders = self._run_async(self._broker.modify_exit(old_env, new_env))
                self._order_mapping[new.intent_key] = [o.id for o in orders]
            else:
                # CloseIntent or mismatched kinds — cancel + re-execute.
                self._dispatch_cancel(old)
                self._dispatch_new(new)
        except OrderDispositionUnknownError as e:
            _blog_warning(
                "modify parked (unknown disposition) for %s: %s", new, e,
            )
            self._park_pending(new_env, e, kind='modify', old_intent=old)
        except BrokerManualInterventionError as e:
            _blog_error(
                "modify halted (manual intervention) for %s: %s", new, e,
            )
            self._record_halt(e)
            raise
        except OrderSkippedByPlugin:
            # Inner _dispatch_new (the cancel+re-execute fallback) declined.
            # _diff_and_dispatch handles the active-intents pop + warning.
            raise
        except Exception as e:
            _blog_error(
                "modify failed for %s: %s: %s", new, type(e).__name__, e,
            )
            raise

    def _dispatch_cancel(self, old: Intent) -> None:
        if isinstance(old, EntryIntent):
            cancel = CancelIntent(pine_id=old.pine_id, symbol=self._symbol)
        elif isinstance(old, ExitIntent):
            cancel = CancelIntent(
                pine_id=old.pine_id,
                symbol=self._symbol,
                from_entry=old.from_entry,
            )
        else:
            # CloseIntent is immediate market — nothing to cancel.
            self._order_mapping.pop(old.intent_key, None)
            self._drop_envelope(old.intent_key)
            return
        cancel_envelope = self._build_cancel_envelope(cancel)
        _blog_info("cancelling %s", cancel)
        try:
            self._run_async(self._broker.execute_cancel(cancel_envelope))
        except OrderDispositionUnknownError as e:
            # A timed-out cancel leaves the exchange-side order in ambiguous
            # state. The next reconcile() pass observes whether the order is
            # still live; if so, a subsequent cancel attempt hits the same
            # deterministic id and the exchange treats it idempotently.
            _blog_warning(
                "cancel dispatch for %s timed out (coid=%s); "
                "next reconcile will verify: %s",
                old.intent_key, e.client_order_id, e,
            )
        except BrokerManualInterventionError as e:
            self._record_halt(e)
            raise
        self._order_mapping.pop(old.intent_key, None)
        self._drop_envelope(old.intent_key)

    # === Async bridge ===

    def _run_async(self, coro):
        """Run a broker coroutine synchronously from the engine's thread.

        In production the engine shares an event loop with the live
        provider; calls hop to that loop via ``run_coroutine_threadsafe``.
        In unit tests no loop is supplied — the coroutine is driven to
        completion by a transient ``asyncio.run``.
        """
        if self._loop is None:
            return asyncio.run(coro)
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(
            timeout=self._timeout,
        )
