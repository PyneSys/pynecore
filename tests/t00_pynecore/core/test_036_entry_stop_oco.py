"""
Integration tests for the both-set entry STOP-leg OCO wiring in
:class:`OrderSyncEngine`.

A both-set Pine entry (``strategy.entry(limit=, stop=)``) is realised as a
native LIMIT working order plus a software price-watch on the STOP side. These
tests drive the engine's entry-stop methods with a fake broker that returns a
configurable cancel disposition, verifying:

* arming on dispatch,
* the OCA: a stop cross cancels the native LIMIT and (only on a confirmed
  cancel) fires a MARKET order with a DISTINCT client-order-id,
* the hard gate: ALREADY_FILLED / UNKNOWN never fire the market,
* the native LIMIT filling first retires the watch (limit wins).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pynecore.core.broker.models import (
    CancelDispositionOutcome,
    CapabilityLevel,
    EntryIntent,
    ExchangeCapabilities,
    ExchangeOrder,
    LegType,
    OrderEvent,
    OrderStatus,
    OrderType,
)
from pynecore.core.broker.idempotency import KIND_ENTRY, KIND_ENTRY_STOP
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.store_helpers import (
    ENTRY_STOP_STATE_ARMED,
    ENTRY_STOP_STATE_CANCEL_PENDING,
)
from pynecore.core.broker.sync_engine import OrderSyncEngine


SYMBOL = "EURUSD"


@dataclass
class _FakeBroker:
    capabilities: ExchangeCapabilities
    cancel_outcome: CancelDispositionOutcome = (
        CancelDispositionOutcome.CANCEL_CONFIRMED
    )
    positions: dict = field(default_factory=dict)
    entries: list = field(default_factory=list)
    cancel_outcome_calls: list = field(default_factory=list)

    def get_capabilities(self) -> ExchangeCapabilities:
        return self.capabilities

    async def execute_entry(self, envelope) -> list[ExchangeOrder]:
        intent = envelope.intent
        self.entries.append((intent, envelope))
        return [ExchangeOrder(
            id=f"ex-{intent.pine_id}-{intent.order_type.value}",
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            qty=intent.qty,
            filled_qty=intent.qty if intent.order_type == OrderType.MARKET else 0.0,
            remaining_qty=0.0,
            price=intent.limit,
            stop_price=intent.stop,
            average_fill_price=intent.limit or 1.18,
            status=(
                OrderStatus.FILLED if intent.order_type == OrderType.MARKET
                else OrderStatus.OPEN
            ),
            timestamp=0.0,
            fee=0.0,
            fee_currency="USD",
        )]

    async def execute_cancel_with_outcome(
            self, envelope,
    ) -> CancelDispositionOutcome:
        self.cancel_outcome_calls.append(envelope.intent)
        return self.cancel_outcome

    async def execute_cancel(self, envelope) -> bool:
        return True

    async def get_position(self, symbol):
        return self.positions.get(symbol)

    async def get_open_orders(self, symbol=None):
        return []

    async def get_balance(self):
        return {}


def _caps(**over) -> ExchangeCapabilities:
    # MARKET and LIMIT entries are always available — there is no dedicated
    # capability field for them. Only the native stop entry has a field, and
    # the both-set OCO path does not even need it (the stop side is software).
    base = dict(stop_order=CapabilityLevel.NATIVE)
    base.update(over)
    return ExchangeCapabilities(**base)


def _engine(broker) -> OrderSyncEngine:
    return OrderSyncEngine(
        broker, BrokerPosition(), SYMBOL, run_tag="tts0", mintick=0.0001,
    )


def _both_set_entry() -> EntryIntent:
    # Long both-set: native LIMIT below (pullback), software STOP above (rise).
    return EntryIntent(
        pine_id="Long", symbol=SYMBOL, side="buy", qty=1.0,
        order_type=OrderType.LIMIT, limit=1.16, stop=1.18,
    )


def _arm(eng: OrderSyncEngine, intent: EntryIntent):
    """Run the native-LIMIT dispatch + watch arm exactly as ``_dispatch_new``
    does for a both-set entry."""
    eng._active_intents[intent.intent_key] = intent
    envelope = eng._build_envelope(intent)
    eng._run_async(eng._broker.execute_entry(envelope))
    eng._arm_entry_stop_watch(intent, envelope)
    return envelope


def __test_arm_registers_watch_with_stop_level__():
    broker = _FakeBroker(_caps())
    eng = _engine(broker)
    _arm(eng, _both_set_entry())
    watch = eng._entry_stop_engine.get_watch("Long")
    assert watch is not None
    assert watch.state == ENTRY_STOP_STATE_ARMED
    assert watch.stop_level == 1.18
    assert watch.side == "buy"


def __test_stop_cross_confirmed_cancel_fires_market__():
    broker = _FakeBroker(_caps(),
                         cancel_outcome=CancelDispositionOutcome.CANCEL_CONFIRMED)
    eng = _engine(broker)
    intent = _both_set_entry()
    envelope = _arm(eng, intent)
    broker.entries.clear()  # drop the native-LIMIT dispatch record

    # Price rises across the stop level -> cancel LIMIT (confirmed) -> market.
    eng._drive_entry_stop_triggers(last_price=1.185)

    assert len(broker.cancel_outcome_calls) == 1
    # Exactly one entry dispatched: the stop-fired MARKET.
    assert len(broker.entries) == 1
    market_intent, market_env = broker.entries[0]
    assert market_intent.order_type == OrderType.MARKET
    assert market_intent.stop_fired_market is True
    # The market uses a DISTINCT client-order-id from the native LIMIT leg.
    assert (market_env.client_order_id(KIND_ENTRY_STOP)
            != envelope.client_order_id(KIND_ENTRY))
    # Watch settled terminal (stop won) -> evicted.
    assert eng._entry_stop_engine.get_watch("Long") is None


def __test_no_cross_keeps_watch_armed_and_no_market__():
    broker = _FakeBroker(_caps())
    eng = _engine(broker)
    _arm(eng, _both_set_entry())
    broker.entries.clear()

    eng._drive_entry_stop_triggers(last_price=1.17)  # below the 1.18 stop

    assert broker.cancel_outcome_calls == []
    assert broker.entries == []
    watch = eng._entry_stop_engine.get_watch("Long")
    assert watch is not None and watch.state == ENTRY_STOP_STATE_ARMED


def __test_already_filled_cancel_means_limit_won_no_market__():
    # Hard gate: the LIMIT filled while we tried to cancel it -> limit won the
    # OCO, the market must NEVER fire.
    broker = _FakeBroker(_caps(),
                         cancel_outcome=CancelDispositionOutcome.ALREADY_FILLED)
    eng = _engine(broker)
    _arm(eng, _both_set_entry())
    broker.entries.clear()

    eng._drive_entry_stop_triggers(last_price=1.185)

    assert len(broker.cancel_outcome_calls) == 1
    assert broker.entries == []  # no market fired
    assert eng._entry_stop_engine.get_watch("Long") is None  # limit_won terminal


def __test_unknown_cancel_withholds_market_and_retries__():
    # Hard gate: UNKNOWN disposition -> stay cancel_pending, NO market. The
    # next tick re-drives the (idempotent) cancel gate.
    broker = _FakeBroker(_caps(),
                         cancel_outcome=CancelDispositionOutcome.UNKNOWN)
    eng = _engine(broker)
    _arm(eng, _both_set_entry())
    broker.entries.clear()

    eng._drive_entry_stop_triggers(last_price=1.185)
    assert broker.entries == []  # no market under UNKNOWN
    watch = eng._entry_stop_engine.get_watch("Long")
    assert watch is not None and watch.state == ENTRY_STOP_STATE_CANCEL_PENDING

    # Next tick: broker now confirms the cancel -> market fires.
    broker.cancel_outcome = CancelDispositionOutcome.CANCEL_CONFIRMED
    eng._drive_entry_stop_triggers(last_price=1.185)
    assert len(broker.entries) == 1
    assert broker.entries[0][0].order_type == OrderType.MARKET
    assert eng._entry_stop_engine.get_watch("Long") is None


def __test_native_limit_fill_retires_watch__():
    # The native LIMIT leg fills first: the watch retires (limit wins) and no
    # market may ever fire.
    broker = _FakeBroker(_caps())
    eng = _engine(broker)
    _arm(eng, _both_set_entry())
    broker.entries.clear()

    fill = OrderEvent(
        order=ExchangeOrder(
            id="ex-Long-limit", symbol=SYMBOL, side="buy",
            order_type=OrderType.LIMIT, qty=1.0, filled_qty=1.0,
            remaining_qty=0.0, price=1.16, stop_price=None,
            average_fill_price=1.16, status=OrderStatus.FILLED,
            timestamp=0.0, fee=0.0, fee_currency="USD",
        ),
        event_type="filled", fill_price=1.16, fill_qty=1.0, timestamp=0.0,
        pine_id="Long", from_entry=None, leg_type=LegType.ENTRY,
    )
    eng._retire_entry_stop_watch_on_fill(fill)
    assert eng._entry_stop_engine.get_watch("Long") is None

    # A later price cross has no armed watch -> no market.
    eng._drive_entry_stop_triggers(last_price=1.185)
    assert broker.entries == []
