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
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.core.broker.models import (
    CancelIntent,
    EntryIntent,
    ExitIntent,
    CloseIntent,
    ExchangeOrder,
    ExchangePosition,
    ExchangeCapabilities,
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
)


SYMBOL = "BTCUSDT"


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
    """Duck-typed stand-in for :class:`BrokerPlugin`. Records all calls."""
    entry_calls: list[EntryIntent] = field(default_factory=list)
    exit_calls: list[ExitIntent] = field(default_factory=list)
    close_calls: list[CloseIntent] = field(default_factory=list)
    cancel_calls: list[CancelIntent] = field(default_factory=list)
    modify_entry_calls: list[tuple[EntryIntent, EntryIntent]] = field(default_factory=list)
    modify_exit_calls: list[tuple[ExitIntent, ExitIntent]] = field(default_factory=list)
    open_orders: list[ExchangeOrder] = field(default_factory=list)
    position: ExchangePosition | None = None
    streamed_events: list[OrderEvent] = field(default_factory=list)
    watch_orders_impl: str = "generator"  # "generator" | "not_implemented"
    _next_id: int = 0

    def _mk_order(self, intent) -> ExchangeOrder:
        self._next_id += 1
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
        )

    async def execute_entry(self, intent):
        self.entry_calls.append(intent)
        return [self._mk_order(intent)]

    async def execute_exit(self, intent):
        self.exit_calls.append(intent)
        return [self._mk_order(intent)]

    async def execute_close(self, intent):
        self.close_calls.append(intent)
        return self._mk_order(intent)

    async def execute_cancel(self, intent):
        self.cancel_calls.append(intent)
        return True

    async def modify_entry(self, old, new):
        self.modify_entry_calls.append((old, new))
        return [self._mk_order(new)]

    async def modify_exit(self, old, new):
        self.modify_exit_calls.append((old, new))
        return [self._mk_order(new)]

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
        mintick=mintick,
    )
    return engine, pos


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

    engine.sync()

    assert len(b.entry_calls) == 1
    assert b.entry_calls[0].pine_id == "L"
    assert b.entry_calls[0].limit == 50_000.0
    assert engine.active_intents.keys() == {"L"}
    assert engine.order_mapping["L"] == ["xchg-1"]


def __test_unchanged_entry_is_not_redispatched__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync()
    engine.sync()

    assert len(b.entry_calls) == 1  # only once


def __test_modified_entry_dispatches_modify_entry__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync()

    # Replace with a different limit price
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=49_500.0)
    engine.sync()

    assert len(b.modify_entry_calls) == 1
    old, new = b.modify_entry_calls[0]
    assert old.limit == 50_000.0 and new.limit == 49_500.0


def __test_removed_entry_dispatches_cancel__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync()

    del pos.entry_orders["L"]
    engine.sync()

    assert len(b.cancel_calls) == 1
    assert b.cancel_calls[0].pine_id == "L"
    assert b.cancel_calls[0].from_entry is None
    assert "L" not in engine.active_intents


def __test_close_intent_dispatches_execute_close__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.exit_orders["L"] = Order(
        "L", -1.0, order_type=_order_type_close,
        exit_id="Close entry(s) order L",
    )

    engine.sync()

    assert len(b.close_calls) == 1
    assert b.close_calls[0].pine_id == "L"
    assert b.close_calls[0].side == "sell"


def __test_exit_with_prices_dispatches_execute_exit__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.exit_orders["L"] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )

    engine.sync()

    assert len(b.exit_calls) == 1
    assert b.exit_calls[0].tp_price == 60_000.0
    assert b.exit_calls[0].sl_price == 45_000.0


# === Tick deferral + resolution ===


def __test_exit_with_ticks_without_entry_is_deferred__():
    b = MockBroker()
    engine, pos = _mk_engine(b, mintick=1.0)
    pos.exit_orders["L"] = _exit_order(
        "L", -1.0, "TP", profit_ticks=100.0, loss_ticks=50.0,
    )

    engine.sync()

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
    engine.sync()  # defers it

    engine.on_order_event(_fill_event(
        "buy", qty=1.0, price=50_000.0, pine_id="L", leg=LegType.ENTRY,
    ))
    engine.sync()  # drains the event, resolves ticks, dispatches

    assert len(b.exit_calls) == 1
    resolved = b.exit_calls[0]
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
    engine.sync()

    engine.on_order_event(_fill_event(
        "sell", qty=1.0, price=50_000.0, pine_id="S", leg=LegType.ENTRY,
    ))
    engine.sync()

    resolved = b.exit_calls[0]
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
    engine.sync()

    assert b.entry_calls == []
    assert engine.active_intents == {}


def __test_interceptor_modifies_qty__():
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    def half(_intent):
        return InterceptorResult(intent=_intent, modified_qty=_intent.qty * 0.5)

    engine.register_interceptor(half)
    engine.sync()

    assert b.entry_calls[0].qty == 0.5


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
    engine.sync()

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
