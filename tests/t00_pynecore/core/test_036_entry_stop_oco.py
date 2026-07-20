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

from concurrent.futures import Future
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
from pynecore.core.broker.exceptions import ExchangeRateLimitError
from pynecore.core.broker.idempotency import KIND_ENTRY, KIND_ENTRY_STOP
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.store_helpers import (
    ENTRY_STOP_STATE_ARMED,
    ENTRY_STOP_STATE_CANCEL_PENDING,
)
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.lib.strategy import Order, _order_type_entry


SYMBOL = "EURUSD"


@dataclass
class _FakeBroker:
    client_order_id_max_len = 30  # BrokerPlugin contract attribute
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

    async def modify_entry(self, old_env, new_env) -> list[ExchangeOrder]:
        intent = new_env.intent
        self.entries.append((intent, new_env))
        return [ExchangeOrder(
            id=f"ex-{intent.pine_id}-{intent.order_type.value}",
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            qty=intent.qty,
            filled_qty=0.0,
            remaining_qty=0.0,
            price=intent.limit,
            stop_price=intent.stop,
            average_fill_price=intent.limit or intent.stop or 1.18,
            status=OrderStatus.OPEN,
            timestamp=0.0,
            fee=0.0,
            fee_currency="USD",
        )]

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


def _stop_only_entry() -> EntryIntent:
    # Long stop-only: a NATIVE STOP entry (rise), no limit leg. This must NEVER
    # carry a software entry-stop watch — the watch's KIND_ENTRY leg assumes a
    # native LIMIT, and a watch here would double-arm against the native STOP.
    return EntryIntent(
        pine_id="Long", symbol=SYMBOL, side="buy", qty=1.0,
        order_type=OrderType.STOP, limit=None, stop=1.18,
    )


def _limit_only_entry() -> EntryIntent:
    # Long limit-only: a plain native LIMIT entry, no stop leg, no watch.
    return EntryIntent(
        pine_id="Long", symbol=SYMBOL, side="buy", qty=1.0,
        order_type=OrderType.LIMIT, limit=1.16, stop=None,
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
    """Arming a both-set entry registers an ARMED watch at the stop level and side."""
    broker = _FakeBroker(_caps())
    eng = _engine(broker)
    _arm(eng, _both_set_entry())
    watch = eng._entry_stop_engine.get_watch("Long")
    assert watch is not None
    assert watch.state == ENTRY_STOP_STATE_ARMED
    assert watch.stop_level == 1.18
    assert watch.side == "buy"


def __test_stop_cross_confirmed_cancel_fires_market__():
    """A stop cross with a confirmed LIMIT cancel fires a MARKET with a distinct id."""
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


def __test_stop_cross_under_quarantine_cancels_limit_never_fires_market__():
    """Under quarantine the stop-cross still cancels the native LIMIT
    (risk-reducing) but the stop-fired MARKET — a new-exposure dispatch that
    bypasses ``_dispatch_new`` — is blocked and the watch settles with no
    position opened; a later tick must not re-fire it."""
    broker = _FakeBroker(_caps(),
                         cancel_outcome=CancelDispositionOutcome.CANCEL_CONFIRMED)
    eng = _engine(broker)
    _arm(eng, _both_set_entry())
    broker.entries.clear()
    eng.record_quarantine("external cancel detected")

    eng._drive_entry_stop_triggers(last_price=1.185)

    assert len(broker.cancel_outcome_calls) == 1  # LIMIT cancel still ran
    assert broker.entries == []                   # no market dispatched
    assert eng._entry_stop_engine.get_watch("Long") is None  # settled
    eng._drive_entry_stop_triggers(last_price=1.19)
    assert broker.entries == []                   # dropped, never re-fired


def __test_stale_read_defers_stop_fired_market_then_fires_when_reads_recover__():
    """A cycle whose broker read failed must not fire the stop MARKET.

    The stop-fired MARKET bypasses ``_dispatch_new`` and POSTs ``execute_entry``
    directly, so gating ``_diff_and_dispatch`` alone would let it open new
    exposure on a position view the engine could not refresh. Unlike the
    quarantine gate this only *defers*: the watch stays live and the latched,
    idempotent state machine fires on the next healthy sync."""
    broker = _FakeBroker(_caps(),
                         cancel_outcome=CancelDispositionOutcome.CANCEL_CONFIRMED)
    pos = BrokerPosition()
    eng = OrderSyncEngine(
        broker, pos, SYMBOL, run_tag="tts0", mintick=0.0001,
    )
    _arm(eng, _both_set_entry())
    broker.entries.clear()
    # Keep the entry live in the Pine order book, otherwise the diff would
    # legitimately cancel it and retire the watch before the stop can fire.
    pos.entry_orders["Long"] = Order(
        "Long", 1.0, order_type=_order_type_entry, limit=1.16, stop=1.18,
    )

    # A retained read that has already failed with a venue 429 — collected by
    # the pre-dispatch guard, which then disallows exposure for this cycle.
    stale = Future()
    stale.set_exception(ExchangeRateLimitError("error.too-many.requests",
                                              retry_after=30.0))
    eng._inflight_read = stale

    eng.sync(1_700_000_000_000, last_price=1.185)

    assert broker.entries == [], "stop MARKET fired on a stale position view"
    watch = eng._entry_stop_engine.get_watch("Long")
    assert watch is not None, "the watch must be deferred, not settled"

    # Reads healthy again -> the deferred cross fires on the next sync.
    eng._read_backoff_until = 0.0
    eng.sync(1_700_000_060_000, last_price=1.185)

    assert len(broker.entries) == 1
    market_intent, _ = broker.entries[0]
    assert market_intent.order_type == OrderType.MARKET
    assert market_intent.stop_fired_market is True


def __test_market_pending_retry_under_quarantine_settles_without_post__():
    """A ``stop_market_pending`` re-entry (restart / retry after an unknown
    disposition — the market coid is already persisted) is settled by the
    quarantine gate without a second POST; a prior landed POST's fill would
    still book, but the retry loop ends."""
    broker = _FakeBroker(_caps())
    eng = _engine(broker)
    intent = _both_set_entry()
    envelope = _arm(eng, intent)
    broker.entries.clear()
    # Drive the watch to stop_market_pending exactly as the fire path does
    # after a confirmed LIMIT cancel, without POSTing the market yet.
    eng._entry_stop_engine.begin_cancel("Long")
    eng._entry_stop_engine.confirm_limit_cancelled_fire_market(
        "Long", market_coid=envelope.client_order_id(KIND_ENTRY_STOP),
    )
    eng.record_quarantine("external cancel detected")

    eng._drive_entry_stop_triggers(last_price=None)  # price-independent state

    assert broker.entries == []                       # no POST under quarantine
    assert eng._entry_stop_engine.get_watch("Long") is None  # settled


def __test_no_cross_keeps_watch_armed_and_no_market__():
    """A price below the stop level leaves the watch ARMED and fires no market."""
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
    """An ALREADY_FILLED cancel means the limit won the OCO, so no market fires."""
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
    """An UNKNOWN cancel withholds the market and stays cancel_pending until confirmed."""
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
    """A native LIMIT fill retires the watch so a later stop cross fires no market."""
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


# === modify keeps the software watch in lockstep with the both-set state =====

def __test_arm_skips_stop_only_entry__():
    """Arming a stop-only entry registers no software watch."""
    # Direct guard: a stop-only entry (native STOP, no limit) must NOT arm a
    # software watch — only a both-set entry (limit AND stop) does.
    broker = _FakeBroker(_caps())
    eng = _engine(broker)
    intent = _stop_only_entry()
    eng._active_intents[intent.intent_key] = intent
    envelope = eng._build_envelope(intent)
    eng._arm_entry_stop_watch(intent, envelope)
    assert eng._entry_stop_engine.get_watch("Long") is None


def __test_modify_limit_only_to_stop_only_does_not_arm_watch__():
    """Modifying a limit-only entry to stop-only arms no software watch."""
    # Regression: modifying a limit-only entry to stop-only (native STOP) must
    # NOT arm a software watch alongside the native STOP. Keying the modify arm
    # on ``new.stop is not None`` alone (the pre-fix bug) would double-arm here.
    broker = _FakeBroker(_caps())
    eng = _engine(broker)
    old = _limit_only_entry()
    eng._active_intents[old.intent_key] = old
    eng._run_async(eng._broker.execute_entry(eng._build_envelope(old)))
    assert eng._entry_stop_engine.get_watch("Long") is None

    new = _stop_only_entry()
    eng._dispatch_modify(old, new)

    # No software watch armed -> a later cross fires no extra market.
    assert eng._entry_stop_engine.get_watch("Long") is None
    broker.entries.clear()
    eng._drive_entry_stop_triggers(last_price=1.185)
    assert broker.entries == []


def __test_modify_both_set_to_stop_only_retires_watch__():
    """Demoting a both-set entry to stop-only retires the watch so no market fires."""
    # Regression: demoting a both-set entry to stop-only removes the software
    # STOP leg; the watch must retire so no market ever fires (the native STOP
    # now owns the trigger).
    broker = _FakeBroker(_caps())
    eng = _engine(broker)
    old = _both_set_entry()
    _arm(eng, old)
    assert eng._entry_stop_engine.get_watch("Long") is not None

    new = _stop_only_entry()
    eng._dispatch_modify(old, new)

    watch = eng._entry_stop_engine.get_watch("Long")
    assert watch is None or watch.state not in (
        ENTRY_STOP_STATE_ARMED, ENTRY_STOP_STATE_CANCEL_PENDING,
    )
    broker.entries.clear()
    eng._drive_entry_stop_triggers(last_price=1.185)
    assert broker.entries == []  # software watch no longer fires


def __test_modify_both_set_amends_watch_stop_level__():
    """A both-set modify with a new stop level amends the armed watch in place."""
    # A both-set -> both-set modify with a new stop level amends the existing
    # watch in place (no re-arm, no retire); the armed level tracks the change.
    broker = _FakeBroker(_caps())
    eng = _engine(broker)
    old = _both_set_entry()
    _arm(eng, old)

    new = EntryIntent(
        pine_id="Long", symbol=SYMBOL, side="buy", qty=1.0,
        order_type=OrderType.LIMIT, limit=1.16, stop=1.20,
    )
    eng._active_intents[new.intent_key] = new
    eng._dispatch_modify(old, new)

    watch = eng._entry_stop_engine.get_watch("Long")
    assert watch is not None
    assert watch.state == ENTRY_STOP_STATE_ARMED
    assert watch.stop_level == 1.20
