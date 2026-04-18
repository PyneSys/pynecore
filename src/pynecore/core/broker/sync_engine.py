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

from pynecore.core.broker.exceptions import OrderDispositionUnknownError
from pynecore.core.broker.intent_builder import build_intents
from pynecore.core.broker.models import (
    BrokerEvent,
    CancelIntent,
    CloseIntent,
    DispatchEnvelope,
    EntryIntent,
    ExitIntent,
    InterceptorResult,
    LegPartialRepairedEvent,
    LegRepairFailedEvent,
    LegType,
    OcaPartialFillPolicy,
    OcaType,
    OrderEvent,
)
from pynecore.core.broker.state_store import (
    EnvelopeRecord,
    PendingRecord,
    StateStore,
)

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
    :param run_tag: 4-char base36 session tag (see :func:`make_run_tag`) — seeds
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
    :param state_store: Optional :class:`StateStore` for cross-restart
        recovery. When provided the engine persists envelope identity and
        parked-verification entries; on construction it replays the file so
        a restarted process re-uses the same ``client_order_id`` for every
        live intent and matches up parked dispatches against
        ``get_open_orders`` on the next sync. Pass ``None`` for unit tests
        and single-shot backtests where restart safety is not required.
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
            state_store: StateStore | None = None,
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
        self._state_store = state_store
        # Capabilities are declared once at plugin startup — cache the lookup
        # so the cascade-cancel fast path does not pay a method call per event.
        caps = broker.get_capabilities()
        self._oca_cancel_native = bool(getattr(caps, 'oca_cancel_native', False))
        self._tp_sl_bracket_native = bool(
            getattr(caps, 'tp_sl_bracket_native', False),
        )

        self._active_intents: dict[str, Intent] = {}
        self._order_mapping: dict[str, list[str]] = {}
        self._envelopes: dict[str, DispatchEnvelope] = {}
        self._pending_verification: dict[str, DispatchEnvelope] = {}
        self._deferred_exits: dict[str, ExitIntent] = {}
        self._event_queue: queue.Queue[OrderEvent] = queue.Queue()
        self._interceptors: list[Callable[[Intent], InterceptorResult]] = []
        self._sync_count = 0
        self._current_bar_ts_ms: int = 0
        # OCA groups already processed inside the current :meth:`sync` pass.
        # Cleared at the start of every sync so a fresh bar re-enables cascade,
        # but kept stable within the pass so two fills in the same group do
        # not emit duplicate CancelIntents.
        self._cancelled_oca_groups_this_sync: set[str] = set()

        # Cross-restart recovery anchors. The state store persists envelope
        # identity and parked-verification entries; replay rebuilds these
        # *anchor* dicts (intent objects are not persisted — they are rebuilt
        # from the Pine order book on the first post-restart sync). The first
        # _build_envelope / _verify_pending_dispatches call for an anchored key
        # promotes the anchor into the live in-memory state and clears it.
        self._persisted_envelope_anchors: dict[str, EnvelopeRecord] = {}
        self._persisted_pending_anchors: dict[str, PendingRecord] = {}
        if state_store is not None:
            envelopes, pending = state_store.replay()
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
                self._event_queue.put(event)
        except NotImplementedError:
            _log.info(
                "broker does not implement watch_orders; "
                "reconcile() will poll for fills instead",
            )
            return
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover — defensive
            _log.exception("watch_orders stream terminated with an error")
            raise

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
        self._current_bar_ts_ms = bar_ts_ms
        self._cancelled_oca_groups_this_sync.clear()
        self._drain_events()
        self._verify_pending_dispatches()

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
                new_deferred[i.from_entry] = i
            else:
                dispatchable.append(i)
        self._deferred_exits = new_deferred

        self._diff_and_dispatch(dispatchable)

        self._sync_count += 1
        if self._reconcile_every and self._sync_count % self._reconcile_every == 0:
            self.reconcile()

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

        A pending entry that does *not* show up stays parked — the engine
        deliberately does not re-dispatch because the original may still land
        (slow network round-trip). The user can inspect
        :attr:`pending_verification` to surface stuck entries.
        """
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
            if self._state_store is not None:
                self._state_store.record_unpark(coid)
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
            if self._state_store is not None:
                self._state_store.record_unpark(coid)
            _log.info(
                "recovered persisted pending dispatch %s -> exchange order %s "
                "for intent %s", coid, order.id, anchor.key,
            )

    def reconcile(self) -> None:
        """Read-side state reconciliation with the exchange.

        The exchange is authoritative for state. Any mismatch between our
        tracking and ``get_open_orders`` / ``get_position`` is logged and
        the local tracking is overwritten. No orders are ever **sent**
        from a reconciliation pass — that would risk duplicate entries.
        """
        orders = self._run_async(self._broker.get_open_orders(self._symbol))
        tracked_ids: set[str] = set()
        for ids in self._order_mapping.values():
            tracked_ids.update(ids)
        exchange_ids = {o.id for o in orders}
        stale = tracked_ids - exchange_ids
        if stale:
            _log.warning(
                "tracked orders missing from exchange: %s", stale,
            )
        untracked = exchange_ids - tracked_ids
        if untracked:
            _log.info(
                "unknown orders on exchange (not bot-owned): %s", untracked,
            )

        exch_pos = self._run_async(self._broker.get_position(self._symbol))
        if exch_pos is not None and exch_pos.size != self._position.size:
            _log.warning(
                "position size mismatch (exchange=%s, internal=%s) — "
                "adopting exchange",
                exch_pos.size, self._position.size,
            )
            self._position.size = exch_pos.size
            self._position.sign = (
                1.0 if exch_pos.size > 0.0
                else (-1.0 if exch_pos.size < 0.0 else 0.0)
            )
            self._position.avg_price = exch_pos.entry_price

    # === Event routing ===

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                return
            self._route_event(event)

    def _route_event(self, event: OrderEvent) -> None:
        t = event.event_type
        if t in ('filled', 'partial'):
            self._position.record_fill(event)
            if event.leg_type == LegType.ENTRY and event.pine_id:
                self._resolve_deferred_for_entry(event.pine_id)
                self._amend_bracket_qty_for_entry_fill(event)
            self._cascade_oca_cancel(event)
        elif t == 'cancelled':
            key = self._find_key_for_order_id(event.order.id)
            if key is not None:
                _log.error(
                    "unexpected cancel for intent %s (exchange order %s)",
                    key, event.order.id,
                )
                self._order_mapping.pop(key, None)
                self._active_intents.pop(key, None)
                self._drop_envelope(key)
        elif t == 'rejected':
            key = self._find_key_for_order_id(event.order.id)
            if key is not None:
                _log.warning(
                    "order rejected for intent %s (exchange order %s)",
                    key, event.order.id,
                )
                self._order_mapping.pop(key, None)
                self._active_intents.pop(key, None)
                self._drop_envelope(key)

    def _drop_envelope(self, key: str) -> None:
        """Remove envelope state for ``key`` and persist a ``complete`` marker.

        Called from every site that retires an intent (cancel dispatch,
        unexpected cancel event, reject event). The persisted marker lets the
        replay path skip the envelope and any still-pending verifications
        attached to the same ``key`` — keeping the JSONL self-compacting.
        """
        self._envelopes.pop(key, None)
        if self._state_store is not None:
            self._state_store.record_complete(key)

    def _find_key_for_order_id(self, order_id: str) -> str | None:
        for key, ids in self._order_mapping.items():
            if order_id in ids:
                return key
        return None

    def _resolve_deferred_for_entry(self, entry_id: str) -> None:
        """An entry fill unblocks any exit that references it via ticks."""
        deferred = self._deferred_exits.pop(entry_id, None)
        if deferred is None:
            return
        resolved = self._resolve_ticks(deferred)
        if resolved.has_unresolved_ticks:
            self._deferred_exits[entry_id] = deferred
            return
        self._dispatch_new(resolved)
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

        - The plugin declared ``oca_cancel_native=True`` — the exchange
          registers and cancels the group natively.
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
            exit_orders.pop(intent.from_entry, None)

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

        - The plugin declared ``tp_sl_bracket_native=True`` — the exchange
          tracks partial entry fills natively (Bybit V5 attached TP/SL).
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
        unchanged ``pos.exit_orders[from_entry]`` (whose ``size`` still equals
        the original full qty), the diff engine sees a mismatch against the
        amended active intent, and emits a *second* ``modify_exit`` back to
        the original qty — undoing the partial-fill cascade we just did.
        """
        exit_orders = getattr(self._position, 'exit_orders', None)
        if exit_orders is None:
            return
        order = exit_orders.get(bracket.from_entry)
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

        for key in list(self._active_intents):
            if key not in new_map:
                old = self._active_intents.pop(key)
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
                    self._dispatch_new(intent)
                    self._active_intents[key] = intent
            elif intent != self._active_intents[key]:
                self._dispatch_modify(self._active_intents[key], intent)
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
        reconstructed from the persisted :class:`StateStore` instead of being
        recomputed from ``_current_bar_ts_ms`` — the latter would yield a new
        ``client_order_id`` and break exchange-side dedup.
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
        if self._state_store is not None:
            self._state_store.record_envelope(
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
    ) -> None:
        """Stash a dispatch whose exchange disposition the plugin could not confirm.

        :meth:`_verify_pending_dispatches` reruns ``get_open_orders`` on each
        subsequent sync and promotes the envelope back to
        ``_order_mapping`` once the order shows up.
        """
        self._pending_verification[error.client_order_id] = envelope
        if self._state_store is not None:
            self._state_store.record_park(
                coid=error.client_order_id,
                key=envelope.intent.intent_key,
            )
        _log.warning(
            "dispatch for %s ended with unknown disposition "
            "(client_order_id=%s); will verify on next sync: %s",
            envelope.intent.intent_key, error.client_order_id, error,
        )

    def _dispatch_new(self, intent: Intent) -> None:
        envelope = self._build_envelope(intent)
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
        except OrderDispositionUnknownError as e:
            self._park_pending(envelope, e)

    def _dispatch_modify(self, old: Intent, new: Intent) -> None:
        old_env = self._build_envelope(old)
        new_env = self._build_envelope(new)
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
            self._park_pending(new_env, e)

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
        try:
            self._run_async(self._broker.execute_cancel(cancel_envelope))
        except OrderDispositionUnknownError as e:
            # A timed-out cancel leaves the exchange-side order in ambiguous
            # state. The next reconcile() pass observes whether the order is
            # still live; if so, a subsequent cancel attempt hits the same
            # deterministic id and the exchange treats it idempotently.
            _log.warning(
                "cancel dispatch for %s timed out "
                "(client_order_id=%s); next reconcile will verify: %s",
                old.intent_key, e.client_order_id, e,
            )
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
