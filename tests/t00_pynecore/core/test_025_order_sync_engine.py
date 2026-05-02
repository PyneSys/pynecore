"""
Tests for :class:`OrderSyncEngine` — the diff/dispatch/event-routing core.

A :class:`MockBroker` implements just the async surface the engine uses,
recording every call so assertions can check which intent ended up where.
A stubbed :attr:`lib._script.initial_capital` keeps
:class:`BrokerPosition.equity` well-defined.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from pynecore import lib
from pynecore.core.broker.exceptions import OrderDispositionUnknownError
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.core.broker.models import (
    BrokerEvent,
    DispatchEnvelope,
    ExchangeOrder,
    ExchangePosition,
    ExchangeCapabilities,
    LegPartialRepairedEvent,
    LegRepairFailedEvent,
    OcaPartialFillPolicy,
    OrderEvent,
    OrderStatus,
    OrderType,
    LegType,
    InterceptorResult,
)
from pynecore.lib.strategy import (
    Order,
    _order_type_entry,
    _order_type_close,
    oca as _oca,
)


SYMBOL = "BTCUSDT"
RUN_TAG = "test"
BAR_TS = 1_700_000_000_000


@pytest.fixture(autouse=True)
def _stub_script():
    prev = lib._script
    lib._script = SimpleNamespace(initial_capital=1_000_000.0)
    try:
        yield
    finally:
        lib._script = prev


# === Mock broker ===


@dataclass
class MockBroker:
    """Duck-typed stand-in for :class:`BrokerPlugin`. Records all calls.

    Each call captures the full :class:`DispatchEnvelope` the sync engine
    sends so tests can inspect both the wrapped intent and the allocated
    ``client_order_id``.
    """
    entry_calls: list[DispatchEnvelope] = field(default_factory=list)
    exit_calls: list[DispatchEnvelope] = field(default_factory=list)
    close_calls: list[DispatchEnvelope] = field(default_factory=list)
    cancel_calls: list[DispatchEnvelope] = field(default_factory=list)
    modify_entry_calls: list[tuple[DispatchEnvelope, DispatchEnvelope]] = field(
        default_factory=list,
    )
    modify_exit_calls: list[tuple[DispatchEnvelope, DispatchEnvelope]] = field(
        default_factory=list,
    )
    open_orders: list[ExchangeOrder] = field(default_factory=list)
    position: ExchangePosition | None = None
    streamed_events: list[OrderEvent] = field(default_factory=list)
    watch_orders_impl: str = "generator"  # "generator" | "not_implemented"
    raise_on_next_entry: Exception | None = None
    raise_on_next_exit: Exception | None = None
    raise_on_next_cancel: Exception | None = None
    capabilities: ExchangeCapabilities = field(default_factory=ExchangeCapabilities)
    _next_id: int = 0

    def get_capabilities(self) -> ExchangeCapabilities:
        return self.capabilities

    def _mk_order(self, envelope: DispatchEnvelope, kind: str) -> ExchangeOrder:
        self._next_id += 1
        intent = envelope.intent
        return ExchangeOrder(
            id=f"xchg-{self._next_id}",
            symbol=getattr(intent, 'symbol', SYMBOL),
            side=getattr(intent, 'side', 'buy'),
            order_type=OrderType.MARKET,
            qty=getattr(intent, 'qty', 0.0),
            filled_qty=0.0,
            remaining_qty=getattr(intent, 'qty', 0.0),
            price=None,
            stop_price=None,
            average_fill_price=None,
            status=OrderStatus.OPEN,
            timestamp=0.0,
            fee=0.0,
            fee_currency="",
            client_order_id=envelope.client_order_id(kind),
        )

    async def execute_entry(self, envelope):
        self.entry_calls.append(envelope)
        if self.raise_on_next_entry is not None:
            err = self.raise_on_next_entry
            self.raise_on_next_entry = None
            raise err
        return [self._mk_order(envelope, 'e')]

    async def execute_exit(self, envelope):
        self.exit_calls.append(envelope)
        if self.raise_on_next_exit is not None:
            err = self.raise_on_next_exit
            self.raise_on_next_exit = None
            raise err
        return [self._mk_order(envelope, 't')]

    async def execute_close(self, envelope):
        self.close_calls.append(envelope)
        return self._mk_order(envelope, 'c')

    async def execute_cancel(self, envelope):
        self.cancel_calls.append(envelope)
        if self.raise_on_next_cancel is not None:
            err = self.raise_on_next_cancel
            self.raise_on_next_cancel = None
            raise err
        return True

    async def modify_entry(self, old, new):
        self.modify_entry_calls.append((old, new))
        return [self._mk_order(new, 'e')]

    async def modify_exit(self, old, new):
        self.modify_exit_calls.append((old, new))
        return [self._mk_order(new, 't')]

    async def get_open_orders(self, symbol=None):
        return list(self.open_orders)

    async def get_position(self, symbol):
        return self.position

    def watch_orders(self):
        if self.watch_orders_impl == "not_implemented":
            raise NotImplementedError

        async def _gen():
            for event in self.streamed_events:
                yield event

        return _gen()


# === Helpers ===


def _entry_order(order_id, size, **kw) -> Order:
    return Order(order_id, size, order_type=_order_type_entry, **kw)


def _exit_order(from_entry, size, exit_id, **kw) -> Order:
    return Order(from_entry, size, order_type=_order_type_close, exit_id=exit_id, **kw)


def _mk_engine(broker, mintick: float = 1.0) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        mintick=mintick,
    )
    return engine, pos


def _sync(engine: OrderSyncEngine, *, bar_ts: int = BAR_TS) -> None:
    engine.sync(bar_ts)


def _fill_event(side: str, qty: float, price: float, *,
                pine_id: str, leg: LegType = LegType.ENTRY,
                xchg_id: str = "xchg-1") -> OrderEvent:
    exch = ExchangeOrder(
        id=xchg_id, symbol=SYMBOL, side=side,
        order_type=OrderType.MARKET, qty=qty, filled_qty=qty,
        remaining_qty=0.0, price=None, stop_price=None,
        average_fill_price=price, status=OrderStatus.FILLED,
        timestamp=0.0, fee=0.0, fee_currency="",
    )
    return OrderEvent(
        order=exch, event_type='filled', fill_price=price,
        fill_qty=qty, timestamp=0.0, pine_id=pine_id, leg_type=leg,
    )


# === Diff / dispatch ===


def __test_new_entry_dispatches_execute_entry__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)

    assert len(b.entry_calls) == 1
    assert b.entry_calls[0].intent.pine_id == "L"
    assert b.entry_calls[0].intent.limit == 50_000.0
    assert engine.active_intents.keys() == {"L"}
    assert engine.order_mapping["L"] == ["xchg-1"]


def __test_unchanged_entry_is_not_redispatched__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)
    engine.sync(BAR_TS)

    assert len(b.entry_calls) == 1  # only once


def __test_modified_entry_dispatches_modify_entry__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)

    # Replace with a different limit price
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=49_500.0)
    engine.sync(BAR_TS)

    assert len(b.modify_entry_calls) == 1
    old, new = b.modify_entry_calls[0]
    assert old.intent.limit == 50_000.0 and new.intent.limit == 49_500.0
    # Envelope identity is pinned on first dispatch and preserved on modify —
    # that is what makes the exchange treat the amend as idempotent.
    assert old.bar_ts_ms == new.bar_ts_ms == BAR_TS
    assert old.run_tag == new.run_tag == RUN_TAG


def __test_removed_entry_dispatches_cancel__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)

    del pos.entry_orders["L"]
    engine.sync(BAR_TS)

    assert len(b.cancel_calls) == 1
    assert b.cancel_calls[0].intent.pine_id == "L"
    assert b.cancel_calls[0].intent.from_entry is None
    assert "L" not in engine.active_intents


def __test_close_intent_dispatches_execute_close__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.exit_orders["L"] = Order(
        "L", -1.0, order_type=_order_type_close,
        exit_id="Close entry(s) order L",
    )

    engine.sync(BAR_TS)

    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.pine_id == "L"
    assert b.close_calls[0].intent.side == "sell"


def __test_exit_with_prices_dispatches_execute_exit__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.exit_orders["L"] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )

    engine.sync(BAR_TS)

    assert len(b.exit_calls) == 1
    assert b.exit_calls[0].intent.tp_price == 60_000.0
    assert b.exit_calls[0].intent.sl_price == 45_000.0


# === Tick deferral + resolution ===


def __test_exit_with_ticks_without_entry_is_deferred__():
    b = MockBroker()
    engine, pos = _mk_engine(b, mintick=1.0)
    pos.exit_orders["L"] = _exit_order(
        "L", -1.0, "TP", profit_ticks=100.0, loss_ticks=50.0,
    )

    engine.sync(BAR_TS)

    # Exit never reaches the plugin while ticks are unresolved.
    assert b.exit_calls == []
    assert "L" in engine.deferred_exits
    assert "TP\0L" not in engine.active_intents


def __test_entry_fill_resolves_deferred_exit__():
    b = MockBroker()
    engine, pos = _mk_engine(b, mintick=1.0)
    pos.exit_orders["L"] = _exit_order(
        "L", -1.0, "TP", profit_ticks=100.0, loss_ticks=50.0,
    )
    engine.sync(BAR_TS)  # defers it

    engine.on_order_event(_fill_event(
        "buy", qty=1.0, price=50_000.0, pine_id="L", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS)  # drains the event, resolves ticks, dispatches

    assert len(b.exit_calls) == 1
    resolved = b.exit_calls[0].intent
    # Long entry (sign=+1): TP above, SL below.
    assert resolved.tp_price == 50_100.0
    assert resolved.sl_price == 49_950.0
    assert resolved.profit_ticks is None
    assert resolved.loss_ticks is None
    assert "L" not in engine.deferred_exits


def __test_short_entry_fill_reverses_tick_direction__():
    b = MockBroker()
    engine, pos = _mk_engine(b, mintick=1.0)
    pos.exit_orders["S"] = _exit_order(
        "S", 1.0, "TP", profit_ticks=100.0, loss_ticks=50.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "sell", qty=1.0, price=50_000.0, pine_id="S", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS)

    resolved = b.exit_calls[0].intent
    # Short (sign=-1): TP below entry, SL above entry.
    assert resolved.tp_price == 49_900.0
    assert resolved.sl_price == 50_050.0


# === Interceptor ===


def __test_interceptor_rejects_intent__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    def veto(_intent) -> InterceptorResult:
        return InterceptorResult(intent=_intent, rejected=True, reject_reason="no")

    engine.register_interceptor(veto)
    engine.sync(BAR_TS)

    assert b.entry_calls == []
    assert engine.active_intents == {}


def __test_interceptor_modifies_qty__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    def half(_intent):
        return InterceptorResult(intent=_intent, modified_qty=_intent.qty * 0.5)

    engine.register_interceptor(half)
    engine.sync(BAR_TS)

    assert b.entry_calls[0].intent.qty == 0.5


# === Reconciliation ===


# === run_event_stream (async bridge) ===


def __test_run_event_stream_queues_all_events__():
    b = MockBroker()
    b.streamed_events = [
        _fill_event("buy", qty=1.0, price=50_000.0,
                    pine_id="L", leg=LegType.ENTRY, xchg_id="x1"),
        _fill_event("sell", qty=1.0, price=50_500.0,
                    pine_id="L", leg=LegType.CLOSE, xchg_id="x2"),
    ]
    engine, pos = _mk_engine(b)

    asyncio.run(engine.run_event_stream())

    # Drain via the public path (sync) — verifies integration with record_fill.
    pos.avg_price = 50_000.0  # make equity finite for Trade bookkeeping
    engine.sync(BAR_TS)

    assert len(pos.closed_trades) == 0 or len(pos.closed_trades) == 1
    # We at least confirm the events flowed end-to-end by checking records
    assert len(pos.open_trades) + len(pos.closed_trades) >= 1


def __test_run_event_stream_handles_not_implemented__():
    b = MockBroker()
    b.watch_orders_impl = "not_implemented"
    engine, pos = _mk_engine(b)

    # Should return cleanly, not raise.
    asyncio.run(engine.run_event_stream())


def __test_run_event_stream_handles_async_gen_not_implemented__():
    """A plugin's ``watch_orders`` may raise NotImplementedError from the
    generator body rather than from the outer call — the engine must treat
    both the same way."""
    b = MockBroker()

    def _raise_in_body():
        async def _gen():
            raise NotImplementedError
            yield  # pragma: no cover — unreachable

        return _gen()

    b.watch_orders = _raise_in_body  # type: ignore[method-assign]
    engine, pos = _mk_engine(b)

    asyncio.run(engine.run_event_stream())


# === Reconciliation ===


# === Idempotency: client_order_id allocation + unknown-disposition recovery ===


def __test_dispatch_passes_deterministic_client_order_id__():
    """Plugins receive a canonical ``client_order_id`` via the envelope."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)

    env = b.entry_calls[0]
    coid = env.client_order_id('e')
    # Deterministic prefix built from RUN_TAG + hash(pine_id="L") + BAR_TS.
    assert coid.startswith(RUN_TAG + "-")
    assert coid.endswith("-e0")
    assert len(coid) <= 30


def __test_retry_within_same_bar_reuses_client_order_id__():
    """A second dispatch attempt in the same bar yields the same CO-ID so the
    exchange can dedup the duplicate."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)
    coid_first = b.entry_calls[0].client_order_id('e')

    # Simulate a second engine building the same envelope for the same logical
    # intent on the same bar — same inputs must produce the same CO-ID.
    engine2, pos2 = _mk_engine(MockBroker())
    pos2.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine2.sync(BAR_TS)
    coid_second = engine2._envelopes["L"].client_order_id('e')  # type: ignore[attr-defined]

    assert coid_first == coid_second


def _preview_entry_coid(pine_id: str, *, limit: float, bar_ts: int = BAR_TS) -> str:
    """Learn the ``client_order_id`` the engine will allocate for a given entry."""
    noop = MockBroker()
    engine, pos = _mk_engine(noop)
    pos.entry_orders[pine_id] = _entry_order(pine_id, 1.0, limit=limit)
    engine.sync(bar_ts)
    return noop.entry_calls[0].client_order_id('e')


def __test_unknown_disposition_parks_pending__():
    """A timed-out dispatch is parked on ``pending_verification``, not retried."""
    expected_coid = _preview_entry_coid("L", limit=50_000.0)

    b = MockBroker()
    b.raise_on_next_entry = OrderDispositionUnknownError(
        "simulated timeout", client_order_id=expected_coid,
    )
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)

    assert expected_coid in engine.pending_verification
    # The engine did call execute_entry exactly once — no auto-retry.
    assert len(b.entry_calls) == 1


def __test_verify_pending_promotes_matched_order__():
    """``_verify_pending_dispatches`` matches a pending CO-ID against open orders."""
    expected_coid = _preview_entry_coid("L", limit=50_000.0)

    b = MockBroker()
    b.raise_on_next_entry = OrderDispositionUnknownError(
        "simulated timeout", client_order_id=expected_coid,
    )
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)
    assert expected_coid in engine.pending_verification

    # The order actually did land; surface it on get_open_orders.
    b.open_orders = [
        ExchangeOrder(
            id="xchg-42", symbol=SYMBOL, side="buy",
            order_type=OrderType.LIMIT, qty=1.0, filled_qty=0.0,
            remaining_qty=1.0, price=50_000.0, stop_price=None,
            average_fill_price=None, status=OrderStatus.OPEN,
            timestamp=0.0, fee=0.0, fee_currency="",
            client_order_id=expected_coid,
        ),
    ]

    engine.sync(BAR_TS)

    assert expected_coid not in engine.pending_verification
    assert engine.order_mapping["L"] == ["xchg-42"]


def __test_verify_pending_keeps_pending_when_not_found__():
    """If ``get_open_orders`` does not surface the CO-ID, the pending stays."""
    expected_coid = _preview_entry_coid("L", limit=50_000.0)

    b = MockBroker()
    b.raise_on_next_entry = OrderDispositionUnknownError(
        "simulated timeout", client_order_id=expected_coid,
    )
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)
    # Second sync: exchange has no matching order; pending stays parked.
    engine.sync(BAR_TS)

    assert expected_coid in engine.pending_verification


def __test_reconcile_adopts_exchange_position_size__():
    b = MockBroker()
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=2.0, entry_price=50_000.0,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    pos.size = 1.0  # local tracking disagrees

    engine.reconcile()

    assert pos.size == 2.0
    assert pos.avg_price == 50_000.0


def __test_reconcile_clears_position_when_exchange_flat__():
    """User manually closes via web UI: ``get_position`` returns ``None``.

    The exchange is the source of truth — when it shows no position, the
    engine must drop ``position.size`` to 0 even if the local view still
    thinks there is an open position. Without this, a phantom adoption
    (or a real position closed externally during operation) leaves Pine
    forever convinced the bot is in a trade and blocks new entries.
    """
    b = MockBroker()
    b.position = None  # exchange flat — no row at all
    engine, pos = _mk_engine(b)
    pos.size = 100.0  # adopted earlier; user closed manually since
    pos.sign = 1.0
    pos.avg_price = 1.17

    engine.reconcile()

    assert pos.size == 0.0
    assert pos.sign == 0.0
    from pynecore.types.na import na_float
    assert pos.avg_price is na_float


def __test_reconcile_clears_open_trades_when_exchange_flat__():
    """When the exchange goes flat externally, open_trades MUST be wiped.

    Otherwise a re-entry on the next bar would mix new fills with stale
    trade rows and corrupt P&L bookkeeping.
    """
    from pynecore.lib.strategy import Trade
    b = MockBroker()
    b.position = None
    engine, pos = _mk_engine(b)
    pos.size = 100.0
    pos.sign = 1.0
    pos.avg_price = 1.17
    pos.open_trades.append(Trade(
        size=100.0, entry_id="L", entry_bar_index=0, entry_time=0,
        entry_price=1.17, commission=0.0, entry_comment=None,
        entry_equity=1_000_000.0,
    ))
    pos.openprofit = 5.0
    pos.open_commission = 0.5

    engine.reconcile()

    assert pos.size == 0.0
    assert pos.open_trades == []
    assert pos.openprofit == 0.0
    assert pos.open_commission == 0.0


def __test_reconcile_no_change_when_sizes_match__():
    """No mutation when exchange and internal already agree."""
    b = MockBroker()
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=100.0, entry_price=1.17,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    pos.size = 100.0
    pos.sign = 1.0
    pos.avg_price = 1.17

    engine.reconcile()

    assert pos.size == 100.0
    assert pos.avg_price == 1.17


def __test_periodic_reconcile_clears_state_on_external_flatten__():
    """Mid-operation reconcile after the user flattens via web UI.

    Pre-condition: bot has dispatched + filled an entry, internal mirrors
    the exchange. Then the user closes manually (exchange returns ``None``)
    and the next sync's reconcile must wipe internal state so Pine sees
    ``position_size == 0`` and can re-enter on a future bar.
    """
    from pynecore.lib.strategy import Trade
    b = MockBroker()
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=100.0, entry_price=1.17,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    pos.size = 100.0
    pos.sign = 1.0
    pos.avg_price = 1.17
    pos.open_trades.append(Trade(
        size=100.0, entry_id="L", entry_bar_index=0, entry_time=0,
        entry_price=1.17, commission=0.0, entry_comment=None,
        entry_equity=1_000_000.0,
    ))

    engine.reconcile()  # startup — agreement, no change
    assert pos.size == 100.0

    # User flattens externally; next periodic reconcile sees /positions empty.
    b.position = None
    engine._sync_count = 1  # simulate post-startup periodic call

    engine.reconcile()

    assert pos.size == 0.0
    assert pos.open_trades == []


def __test_periodic_reconcile_does_not_adopt_size_increase__():
    """Mid-operation reconcile MUST NOT adopt a size increase the engine
    has not yet seen via ``record_fill``.

    Race scenario: a market entry the engine just dispatched fills
    *between* the activity poll and the engine's own /positions read, so
    /positions briefly shows a position the matching ``OrderEvent`` has
    not yet drained into ``BrokerPosition``. Adopting that here would
    double-count the size when the event eventually arrives.
    """
    b = MockBroker()
    # Exchange "ahead" of internal — fill in flight.
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=100.0, entry_price=1.17,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    pos.size = 0.0  # event has not yet drained
    engine._sync_count = 1  # post-startup

    engine.reconcile()

    assert pos.size == 0.0  # untouched — record_fill will own this update


def __test_periodic_reconcile_skips_clear_while_close_in_flight__():
    """Reconcile must not clear the position while a bot-dispatched close
    is in flight.

    Race scenario: bar N dispatches ``execute_close``; the broker flattens
    /positions seconds before the matching ``OrderEvent`` reaches the
    queue. If reconcile zeros the position now, the closing fill (when it
    finally drains) would arrive with ``size == 0`` and enter
    :meth:`BrokerPosition.record_fill`'s "Opening" branch — counted as a
    fresh entry in the opposite direction.
    """
    from pynecore.core.broker.models import CloseIntent
    from pynecore.lib.strategy import Trade
    b = MockBroker()
    b.position = None  # exchange has flattened — close hit
    engine, pos = _mk_engine(b)
    pos.size = 1.0
    pos.sign = 1.0
    pos.avg_price = 50_000.0
    pos.open_trades.append(Trade(
        size=1.0, entry_id="L", entry_bar_index=0, entry_time=0,
        entry_price=50_000.0, commission=0.0, entry_comment=None,
        entry_equity=1_000_000.0,
    ))
    engine._sync_count = 1
    # Simulate a CloseIntent we dispatched but whose fill event has not
    # yet drained.
    engine._active_intents["L"] = CloseIntent(
        pine_id="L", symbol=SYMBOL, side="sell", qty=1.0,
    )

    engine.reconcile()

    assert pos.size == 1.0  # left alone for record_fill to own
    assert len(pos.open_trades) == 1


# === OCA cascade cancel ===
#
# The engine must cancel OCA-cancel siblings the moment a fill event arrives,
# not wait for the next bar's diff pass. These tests exercise the full event
# → sync → cascade path with both entry-side and exit-side fills.


def _mk_engine_with_policy(
        broker: MockBroker,
        *,
        policy: OcaPartialFillPolicy = OcaPartialFillPolicy.FILL_CANCELS,
) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        oca_partial_fill_policy=policy,
    )
    return engine, pos


def _oca_entry(order_id: str, size: float, *, oca_name: str,
               oca_type, limit: float | None = None) -> Order:
    return Order(
        order_id, size, order_type=_order_type_entry,
        limit=limit, oca_name=oca_name, oca_type=oca_type,
    )


def __test_fill_cascades_cancel_to_oca_siblings__():
    """Full fill on A triggers an immediate cancel dispatch for sibling B."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.cancel, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.cancel, limit=49_000.0,
    )
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 2
    assert set(engine.active_intents) == {"A", "B"}

    # A fills — must emit a cancel for B on the next sync's drain.
    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 60_000)

    assert len(b.cancel_calls) == 1
    assert b.cancel_calls[0].intent.pine_id == "B"
    assert "B" not in engine.active_intents
    # Pine-side cleanup mirrors SimPosition._cancel_oca_group.
    assert "B" not in pos.entry_orders


def __test_partial_fill_cascades_under_fill_cancels_policy__():
    """Default policy treats a partial fill as a committed win for the leg."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.cancel, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.cancel, limit=49_000.0,
    )
    engine.sync(BAR_TS)

    partial = _fill_event("buy", 0.4, 50_000.0, pine_id="A", leg=LegType.ENTRY)
    partial.event_type = 'partial'
    engine.on_order_event(partial)
    engine.sync(BAR_TS + 60_000)

    assert len(b.cancel_calls) == 1
    assert b.cancel_calls[0].intent.pine_id == "B"


def __test_partial_fill_does_not_cascade_under_full_fill_only_policy__():
    """FULL_FILL_ONLY keeps siblings live until the leg is fully filled."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(
        b, policy=OcaPartialFillPolicy.FULL_FILL_ONLY,
    )
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.cancel, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.cancel, limit=49_000.0,
    )
    engine.sync(BAR_TS)

    partial = _fill_event("buy", 0.4, 50_000.0, pine_id="A", leg=LegType.ENTRY)
    partial.event_type = 'partial'
    engine.on_order_event(partial)
    engine.sync(BAR_TS + 60_000)

    assert b.cancel_calls == []
    assert "B" in engine.active_intents

    # Full fill then arrives — cascade must trigger now.
    engine.on_order_event(_fill_event(
        "buy", 0.6, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 120_000)

    assert len(b.cancel_calls) == 1
    assert b.cancel_calls[0].intent.pine_id == "B"


def __test_native_oca_cancel_suppresses_cascade__():
    """When the exchange owns the OCA group, the sync engine stays hands-off."""
    b = MockBroker(capabilities=ExchangeCapabilities(oca_cancel_native=True))
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.cancel, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.cancel, limit=49_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 60_000)

    assert b.cancel_calls == []
    # Exchange takes care of B; engine's active_intents still reflect both
    # until the plugin surfaces a separate cancelled event for B.
    assert "B" in engine.active_intents


def __test_two_fills_same_group_same_sync_emit_one_cancel__():
    """Per-group dedup inside a single sync pass prevents duplicate cancels."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.cancel, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.cancel, limit=49_000.0,
    )
    pos.entry_orders["C"] = _oca_entry(
        "C", 1.0, oca_name="G", oca_type=_oca.cancel, limit=48_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    # A second spurious fill on the same group (e.g. a partial followed by a
    # full fill reported separately) must not re-trigger the cascade.
    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 60_000)

    # Two siblings cancelled, but only on the first fill — the second is no-op.
    assert len(b.cancel_calls) == 2
    assert {c.intent.pine_id for c in b.cancel_calls} == {"B", "C"}


def __test_non_cancel_oca_does_not_cascade__():
    """OCA-reduce groups stay alive on fill (partial-fill qty-amend is WS5)."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.reduce, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.reduce, limit=49_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 60_000)

    assert b.cancel_calls == []
    assert "B" in engine.active_intents


def __test_standalone_fill_without_oca_group_is_quiet__():
    """Fills on non-OCA intents never touch the cascade path."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _entry_order("A", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 60_000)

    assert b.cancel_calls == []


# === Partial entry fill → bracket qty amend (WS5, Option A) ===


def _partial_entry_event(*, pine_id: str, fill_delta: float,
                         cumulative_filled: float, order_qty: float,
                         price: float, xchg_id: str = "xchg-1") -> OrderEvent:
    """Build an ``event_type='partial'`` entry fill with cumulative tracking.

    ``fill_delta`` is what the plugin reports this tick; ``cumulative_filled``
    is the running total on the exchange-side order (what the sync engine
    reads via ``event.order.filled_qty``).
    """
    exch = ExchangeOrder(
        id=xchg_id, symbol=SYMBOL, side="buy",
        order_type=OrderType.LIMIT, qty=order_qty,
        filled_qty=cumulative_filled,
        remaining_qty=order_qty - cumulative_filled,
        price=price, stop_price=None, average_fill_price=price,
        status=OrderStatus.PARTIALLY_FILLED,
        timestamp=0.0, fee=0.0, fee_currency="",
    )
    return OrderEvent(
        order=exch, event_type='partial', fill_price=price,
        fill_qty=fill_delta, timestamp=0.0,
        pine_id=pine_id, leg_type=LegType.ENTRY,
    )


def _mk_engine_with_sink(
        broker: MockBroker, sink: list[BrokerEvent],
) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        broker_event_sink=sink.append,
    )
    return engine, pos


def __test_partial_entry_fill_amends_bracket_qty__():
    """A 40% partial entry fill scales the bracket down to 0.4."""
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, pos = _mk_engine_with_sink(b, events)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders["L"] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)
    assert len(b.exit_calls) == 1
    assert b.exit_calls[0].intent.qty == 1.0

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.4, cumulative_filled=0.4,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)

    assert len(b.modify_exit_calls) == 1
    old, new = b.modify_exit_calls[0]
    assert old.intent.qty == 1.0
    assert new.intent.qty == 0.4
    assert engine.active_intents["TP\0L"].qty == 0.4

    repair_events = [e for e in events if isinstance(e, LegPartialRepairedEvent)]
    assert len(repair_events) == 1
    assert repair_events[0].old_qty == 1.0
    assert repair_events[0].new_qty == 0.4


def __test_subsequent_partial_fill_emits_another_amend__():
    """Each partial fill with a new cumulative qty triggers a fresh amend."""
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, pos = _mk_engine_with_sink(b, events)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders["L"] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.3, cumulative_filled=0.3,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)
    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.4, cumulative_filled=0.7,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 120_000)

    assert len(b.modify_exit_calls) == 2
    assert b.modify_exit_calls[0][1].intent.qty == 0.3
    assert b.modify_exit_calls[1][1].intent.qty == 0.7


def __test_native_bracket_skips_partial_amend__():
    """tp_sl_bracket_native=True — the plugin/exchange tracks partial fills."""
    b = MockBroker(
        capabilities=ExchangeCapabilities(tp_sl_bracket_native=True),
    )
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders["L"] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.4, cumulative_filled=0.4,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)

    assert b.modify_exit_calls == []
    # Bracket intent untouched: still the original 1.0 qty.
    assert engine.active_intents["TP\0L"].qty == 1.0


def __test_partial_fill_without_bracket_is_quiet__():
    """Entry without a paired exit → no amend, no event."""
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, pos = _mk_engine_with_sink(b, events)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.5, cumulative_filled=0.5,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)

    assert b.modify_exit_calls == []
    assert events == []


def __test_overfill_is_capped_and_emits_leg_repair_failed__():
    """filled_qty > entry_intent.qty → cap at entry qty + LegRepairFailedEvent.

    The bracket was originally dispatched at 1.0; the cap lands it at 1.0
    again, so no second modify_exit is needed — the critical outcome is the
    :class:`LegRepairFailedEvent` surfacing the exchange anomaly.
    """
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, pos = _mk_engine_with_sink(b, events)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders["L"] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=1.2, cumulative_filled=1.2,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)

    # Bracket qty stays at entry qty (cap), no redundant modify_exit.
    assert engine.active_intents["TP\0L"].qty == 1.0
    assert b.modify_exit_calls == []

    overfill = [e for e in events if isinstance(e, LegRepairFailedEvent)]
    assert len(overfill) == 1
    assert "overfill" in overfill[0].reason.lower()
    assert overfill[0].action_taken == 'capped'


def __test_overfill_after_partial_caps_at_entry_qty__():
    """0.4 partial amends to 0.4; follow-up 1.2 cumulative caps back at 1.0."""
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, pos = _mk_engine_with_sink(b, events)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders["L"] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.4, cumulative_filled=0.4,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)

    # Second event over-reports 1.2 cumulative — cap at 1.0.
    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.8, cumulative_filled=1.2,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 120_000)

    # Two amends: 1.0 → 0.4 (first partial), 0.4 → 1.0 (second, capped).
    assert len(b.modify_exit_calls) == 2
    assert b.modify_exit_calls[0][1].intent.qty == 0.4
    assert b.modify_exit_calls[1][1].intent.qty == 1.0
    assert engine.active_intents["TP\0L"].qty == 1.0
    # The second amend carries the overfill flag.
    overfill = [e for e in events if isinstance(e, LegRepairFailedEvent)]
    assert len(overfill) == 1
    assert overfill[0].action_taken == 'capped'


# === Natural close cleanup ===========================================
# When a TP/SL/TRAILING_STOP/CLOSE leg fully closes the position
# (BrokerPosition.size hits 0), the engine must drop the entry +
# matching exit intents from ``_active_intents`` AND clear Pine's
# ``entry_orders`` / ``exit_orders`` dicts. Pine's ``strategy.exit``
# is unconditional in most scripts; only the simulator gates it via
# open trades. Without this cleanup the next bar's ``sync()`` rebuilds
# the same exit intent from the still-present dict entry and dispatches
# a pointless ``modify_exit`` against a position that no longer exists
# on the broker — which on Capital.com fails because the entry row is
# gone.


def _closing_fill_event(side: str, qty: float, price: float, *,
                        pine_id: str, from_entry: str,
                        leg: LegType = LegType.STOP_LOSS,
                        xchg_id: str = "xchg-close") -> OrderEvent:
    exch = ExchangeOrder(
        id=xchg_id, symbol=SYMBOL, side=side,
        order_type=OrderType.MARKET, qty=qty, filled_qty=qty,
        remaining_qty=0.0, price=None, stop_price=None,
        average_fill_price=price, status=OrderStatus.FILLED,
        timestamp=0.0, fee=0.0, fee_currency="",
    )
    return OrderEvent(
        order=exch, event_type='filled', fill_price=price,
        fill_qty=qty, timestamp=0.0,
        pine_id=pine_id, from_entry=from_entry, leg_type=leg,
    )


def __test_natural_close_cleans_entry_and_exit_intents__():
    """SL fill that brings position size to 0 must wipe the entry
    intent, the exit intent, and the matching Pine-side dict entries —
    otherwise Pine re-emits a stale exit on the next bar and the engine
    fires a pointless ``modify_exit`` against a closed position.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders["L"] = _exit_order(
        "L", -1.0, "Bracket", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)
    assert "L" in engine.active_intents
    assert "Bracket\0L" in engine.active_intents

    # Entry fills — position opens, intents stay in tracking.
    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="L", leg=LegType.ENTRY,
    ))
    engine._drain_events()
    assert pos.size == 1.0

    # SL fires — position closes; cleanup must run.
    engine.on_order_event(_closing_fill_event(
        "sell", 1.0, 45_000.0,
        pine_id="L", from_entry="L", leg=LegType.STOP_LOSS,
    ))
    engine._drain_events()

    assert pos.size == 0.0, "SL fill must reduce position to flat"
    assert "L" not in engine.active_intents, (
        "entry intent must be dropped after natural close"
    )
    assert "Bracket\0L" not in engine.active_intents, (
        "exit intent must be dropped after natural close"
    )
    assert "L" not in pos.entry_orders, (
        "Pine entry_orders[L] must be cleared so next bar does not "
        "re-emit a modify against the closed position"
    )
    assert "L" not in pos.exit_orders, (
        "Pine exit_orders[L] must be cleared so next bar does not "
        "re-emit a stale Bracket exit"
    )


def __test_natural_close_partial_fill_does_not_cleanup__():
    """A partial closing fill that does NOT bring size to 0 must keep
    the entry/exit intents intact so the remainder can still close.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 2.0, limit=50_000.0)
    pos.exit_orders["L"] = _exit_order(
        "L", -2.0, "Bracket", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "buy", 2.0, 50_000.0, pine_id="L", leg=LegType.ENTRY,
    ))
    engine._drain_events()
    assert pos.size == 2.0

    # Partial SL fill — size goes 2 → 1, not 0.
    partial = _closing_fill_event(
        "sell", 1.0, 45_000.0,
        pine_id="L", from_entry="L", leg=LegType.STOP_LOSS,
    )
    partial.event_type = 'partial'
    engine.on_order_event(partial)
    engine._drain_events()

    assert pos.size == 1.0
    assert "L" in engine.active_intents, "entry intent must survive partial close"
    assert "Bracket\0L" in engine.active_intents, "exit intent must survive partial close"
    assert "L" in pos.entry_orders
    assert "L" in pos.exit_orders
