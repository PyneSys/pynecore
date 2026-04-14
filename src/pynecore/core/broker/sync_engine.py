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

from pynecore.core.broker.intent_builder import build_intents
from pynecore.core.broker.models import (
    CancelIntent,
    CloseIntent,
    EntryIntent,
    ExitIntent,
    InterceptorResult,
    LegType,
    OrderEvent,
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
    :param event_loop: A running ``asyncio`` loop on which to execute the
        broker's coroutines. Pass ``None`` for unit tests — each broker call
        will then spin up a transient loop via ``asyncio.run``.
    :param execute_timeout: Seconds to wait for any single ``execute_*``
        coroutine when bridging from a background loop.
    :param reconcile_every_n_syncs: If non-zero, perform a read-side
        reconciliation every N :meth:`sync` calls.
    :param mintick: Symbol minimum tick — used to resolve tick-based exits
        (``profit=`` / ``loss=`` / ``trail_points=``) into absolute prices.
    """

    def __init__(
            self,
            broker: 'BrokerPlugin',
            position: 'BrokerPosition',
            symbol: str,
            *,
            event_loop: asyncio.AbstractEventLoop | None = None,
            execute_timeout: float = 30.0,
            reconcile_every_n_syncs: int = 0,
            mintick: float = 0.01,
    ) -> None:
        self._broker = broker
        self._position = position
        self._symbol = symbol
        self._loop = event_loop
        self._timeout = execute_timeout
        self._reconcile_every = reconcile_every_n_syncs
        self._mintick = mintick

        self._active_intents: dict[str, Intent] = {}
        self._order_mapping: dict[str, list[str]] = {}
        self._deferred_exits: dict[str, ExitIntent] = {}
        self._event_queue: queue.Queue[OrderEvent] = queue.Queue()
        self._interceptors: list[Callable[[Intent], InterceptorResult]] = []
        self._sync_count = 0

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

    def sync(self) -> None:
        """Run one diff/dispatch cycle.

        Reads the Pine order book from ``position.entry_orders`` and
        ``position.exit_orders``, resolves tick-deferred exits where the
        referenced entry price is now known, and dispatches whatever
        changed to the broker plugin.
        """
        self._drain_events()

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
        elif t == 'cancelled':
            key = self._find_key_for_order_id(event.order.id)
            if key is not None:
                _log.error(
                    "unexpected cancel for intent %s (exchange order %s)",
                    key, event.order.id,
                )
                self._order_mapping.pop(key, None)
                self._active_intents.pop(key, None)
        elif t == 'rejected':
            key = self._find_key_for_order_id(event.order.id)
            if key is not None:
                _log.warning(
                    "order rejected for intent %s (exchange order %s)",
                    key, event.order.id,
                )
                self._order_mapping.pop(key, None)
                self._active_intents.pop(key, None)

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
                self._dispatch_new(intent)
                self._active_intents[key] = intent
            elif intent != self._active_intents[key]:
                self._dispatch_modify(self._active_intents[key], intent)
                self._active_intents[key] = intent
            # else: unchanged — skip

    def _dispatch_new(self, intent: Intent) -> None:
        if isinstance(intent, EntryIntent):
            orders = self._run_async(self._broker.execute_entry(intent))
            self._order_mapping[intent.intent_key] = [o.id for o in orders]
        elif isinstance(intent, ExitIntent):
            orders = self._run_async(self._broker.execute_exit(intent))
            self._order_mapping[intent.intent_key] = [o.id for o in orders]
        elif isinstance(intent, CloseIntent):
            order = self._run_async(self._broker.execute_close(intent))
            self._order_mapping[intent.intent_key] = [order.id]

    def _dispatch_modify(self, old: Intent, new: Intent) -> None:
        if isinstance(new, EntryIntent) and isinstance(old, EntryIntent):
            orders = self._run_async(self._broker.modify_entry(old, new))
            self._order_mapping[new.intent_key] = [o.id for o in orders]
        elif isinstance(new, ExitIntent) and isinstance(old, ExitIntent):
            orders = self._run_async(self._broker.modify_exit(old, new))
            self._order_mapping[new.intent_key] = [o.id for o in orders]
        else:
            # CloseIntent or mismatched kinds — cancel + re-execute.
            self._dispatch_cancel(old)
            self._dispatch_new(new)

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
            return
        self._run_async(self._broker.execute_cancel(cancel))
        self._order_mapping.pop(old.intent_key, None)

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
