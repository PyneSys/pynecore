"""
Integration tests for Pine stop-limit orders in :class:`OrderSyncEngine`.

A Pine ``strategy.entry`` / ``strategy.order`` with both ``stop`` and
``limit`` is ONE stop-limit order (measured on TradingView, BINANCE:BTCUSDT
30m): nothing fills before the stop triggers, then it is a plain LIMIT at the
limit price — a marketable one fills at once, any other rests at the limit; a
stop already passed at placement is a LIMIT immediately.

On a venue without a native stop-limit order the engine emulates it: nothing
rests on the venue before the trigger, and the trigger sends a LIMIT. A venue
that declares ``stop_limit_order`` NATIVE gets the order as it is.
"""
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore import lib

from pynecore.core.broker.exceptions import (
    ExchangeOrderRejectedError,
    ExchangeRateLimitError,
    OrderDispositionUnknownError,
)
from pynecore.core.broker.idempotency import KIND_ENTRY, KIND_ENTRY_TRIGGERED
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
    Quote,
)
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.store_helpers import (
    ENTRY_KIND_WORKING,
    LEGACY_EXTRAS_KEY_OCO_LIMIT_COID,
    LEGACY_EXTRAS_KEY_OCO_MARKET_COID,
    LEGACY_OCO_STATE_MARKET_PENDING,
    STOP_LIMIT_STATE_ARMED,
    STOP_LIMIT_STATE_LIMIT_PENDING,
    STOP_LIMIT_STATE_TRIGGERED,
    create_entry_order_row,
    create_stop_limit_watch_row,
    update_stop_limit_watch,
)
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.lib.strategy import Order, _order_type_entry, _order_type_normal

SYMBOL = "EURUSD"
BAR = 1_700_000_000_000
MINUTE = 60_000


@pytest.fixture(autouse=True)
def _script_stub():
    """A booked fill reads the script's ``initial_capital``."""
    prev = lib._script
    lib._script = SimpleNamespace(initial_capital=1_000_000.0)
    try:
        yield
    finally:
        lib._script = prev


@dataclass
class _FakeBroker:
    client_order_id_max_len = 30  # BrokerPlugin contract attribute
    capabilities: ExchangeCapabilities = field(
        default_factory=lambda: ExchangeCapabilities(short_selling=CapabilityLevel.NATIVE))
    entries: list = field(default_factory=list)
    cancels: list = field(default_factory=list)
    outcome_cancels: list = field(default_factory=list)
    modifies: list = field(default_factory=list)
    cancel_outcome: CancelDispositionOutcome = CancelDispositionOutcome.CANCEL_CONFIRMED
    quote: Quote | None = None
    quote_reads: int = 0
    reject_next_entries: list = field(default_factory=list)
    store_ctx: RunContext | None = None

    def get_capabilities(self) -> ExchangeCapabilities:
        return self.capabilities

    def _order(self, intent: EntryIntent, n: int) -> ExchangeOrder:
        market = intent.order_type is OrderType.MARKET
        return ExchangeOrder(
            id=f"ex-{n}", symbol=intent.symbol, side=intent.side,
            order_type=intent.order_type, qty=intent.qty,
            filled_qty=intent.qty if market else 0.0,
            remaining_qty=0.0 if market else intent.qty,
            price=intent.limit, stop_price=intent.stop,
            average_fill_price=intent.limit or 1.18,
            status=OrderStatus.FILLED if market else OrderStatus.OPEN,
            timestamp=0.0, fee=0.0, fee_currency="USD",
        )

    async def execute_entry(self, envelope) -> list[ExchangeOrder]:
        intent = envelope.intent
        self.entries.append((intent, envelope))
        coid = envelope.client_order_id(
            KIND_ENTRY_TRIGGERED if intent.triggered_entry else KIND_ENTRY)
        if self.store_ctx is not None:
            create_entry_order_row(
                self.store_ctx, coid=coid, symbol=intent.symbol, side=intent.side,
                qty=intent.qty, intent_key=intent.intent_key,
                pine_entry_id=intent.pine_id, kind=ENTRY_KIND_WORKING,
                order_type=intent.order_type.value)
        if self.reject_next_entries:
            error = self.reject_next_entries.pop(0)
            if self.store_ctx is not None:
                self.store_ctx.upsert_order(coid, state='rejected')
                self.store_ctx.close_order(coid)
            raise error
        order = self._order(intent, len(self.entries))
        if self.store_ctx is not None:
            self.store_ctx.upsert_order(coid, state='confirmed', exchange_order_id=order.id)
        return [order]

    async def modify_entry(self, old_env, new_env) -> list[ExchangeOrder]:
        self.modifies.append((old_env.intent, new_env.intent))
        return [self._order(new_env.intent, 100 + len(self.modifies))]

    async def execute_cancel(self, envelope) -> bool:
        self.cancels.append(envelope.intent)
        return True

    async def execute_cancel_with_outcome(self, envelope) -> CancelDispositionOutcome:
        self.outcome_cancels.append(envelope.intent)
        return self.cancel_outcome

    async def get_quote(self, symbol: str) -> Quote | None:
        self.quote_reads += 1
        return self.quote

    async def get_position(self, symbol):
        return None

    async def get_open_orders(self, symbol=None):
        return []

    async def get_balance(self):
        return {}


def _native_caps() -> ExchangeCapabilities:
    return ExchangeCapabilities(short_selling=CapabilityLevel.NATIVE,
                                stop_limit_order=CapabilityLevel.NATIVE)


def _engine(broker: _FakeBroker, ctx: RunContext | None = None
            ) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    run_tag = ctx.run_tag if ctx is not None else "tts0"
    eng = OrderSyncEngine(broker, pos, SYMBOL, run_tag=run_tag, mintick=0.0001,
                          store_ctx=ctx)
    return eng, pos


def _place(pos: BrokerPosition, *, limit: float = 1.16, stop: float = 1.18,
           size: float = 1.0, order_type=_order_type_entry) -> None:
    pos.entry_orders["Long"] = Order("Long", size, order_type=order_type,
                                     limit=limit, stop=stop)


def _fill(order_id: str, qty: float, price: float, *, total: float = 1.0) -> OrderEvent:
    return OrderEvent(
        order=ExchangeOrder(
            id=order_id, symbol=SYMBOL, side="buy", order_type=OrderType.LIMIT,
            qty=total, filled_qty=qty, remaining_qty=total - qty, price=price,
            stop_price=None, average_fill_price=price,
            status=OrderStatus.FILLED if qty >= total else OrderStatus.PARTIALLY_FILLED,
            timestamp=0.0, fee=0.0, fee_currency="USD",
        ),
        event_type="filled" if qty >= total else "partial",
        fill_price=price, fill_qty=qty, timestamp=0.0,
        pine_id="Long", from_entry=None, leg_type=LegType.ENTRY,
    )


def _identity() -> RunIdentity:
    return RunIdentity(strategy_id='stop-limit', symbol=SYMBOL, timeframe='1',
                       account_id='test-account', label=None)


# === Software venue: before the trigger ===================================

@pytest.mark.parametrize("order_type", [_order_type_entry, _order_type_normal],
                         ids=["strategy.entry", "strategy.order"])
def __test_nothing_rests_on_the_venue_before_the_trigger__(order_type):
    """The stop-limit order is armed, not placed: no LIMIT can fill on a pullback."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos, order_type=order_type)

    eng.sync(BAR, last_price=1.17)
    eng.sync(BAR + MINUTE, last_price=1.15)  # a pullback through the limit

    assert broker.entries == []
    watch = eng._stop_limit_engine.get_watch("Long")
    assert watch is not None and watch.state == STOP_LIMIT_STATE_ARMED
    assert (watch.stop, watch.limit) == (1.18, 1.16)


# === Software venue: the trigger ==========================================

@pytest.mark.parametrize("order_type", [_order_type_entry, _order_type_normal],
                         ids=["strategy.entry", "strategy.order"])
def __test_the_trigger_sends_a_limit_at_the_limit_price__(order_type):
    """A stop cross sends a plain LIMIT at the limit price, never a market order."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos, order_type=order_type)
    eng.sync(BAR, last_price=1.17)

    eng.sync(BAR + MINUTE, last_price=1.185)

    assert len(broker.entries) == 1
    intent, envelope = broker.entries[0]
    assert intent.order_type is OrderType.LIMIT
    assert (intent.limit, intent.stop) == (1.16, None)
    assert intent.triggered_entry
    assert intent.is_strategy_order is (order_type is _order_type_normal)
    assert envelope.client_order_id(KIND_ENTRY_TRIGGERED) != envelope.client_order_id(KIND_ENTRY)
    watch = eng._stop_limit_engine.get_watch("Long")
    assert watch is not None and watch.state == STOP_LIMIT_STATE_TRIGGERED
    assert eng.order_mapping["Long"] == ["ex-1"]


def __test_a_non_marketable_triggered_limit_keeps_resting__():
    """A triggered LIMIT below the market rests; later syncs never send it again."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos, limit=1.16, stop=1.18)
    eng.sync(BAR, last_price=1.185)
    for minute in range(1, 4):
        eng.sync(BAR + minute * MINUTE, last_price=1.19)
    assert len(broker.entries) == 1
    watch = eng._stop_limit_engine.get_watch("Long")
    assert watch is not None and watch.state == STOP_LIMIT_STATE_TRIGGERED


def __test_a_marketable_triggered_limit_fills_and_ends_the_watch__():
    """A LIMIT above the trigger fills at once; the fill ends the order."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos, limit=1.20, stop=1.18)
    eng.sync(BAR, last_price=1.17)
    eng.sync(BAR + MINUTE, last_price=1.181)
    intent, _ = broker.entries[0]
    assert (intent.order_type, intent.limit) == (OrderType.LIMIT, 1.20)

    eng.on_order_event(_fill("ex-1", 1.0, 1.181))
    del pos.entry_orders["Long"]
    eng.sync(BAR + 2 * MINUTE, last_price=1.19)

    assert eng._stop_limit_engine.get_watch("Long") is None
    assert len(broker.entries) == 1


def __test_a_stop_already_passed_is_a_limit_at_once__():
    """Placed with the price beyond the stop, the order triggers on its first sync."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos, limit=1.16, stop=1.18)
    eng.sync(BAR, last_price=1.19)
    assert len(broker.entries) == 1
    assert broker.entries[0][0].order_type is OrderType.LIMIT


def __test_a_sell_stop_limit_triggers_on_a_fall__():
    """A sell stop-limit triggers when the price falls to the stop."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    pos.entry_orders["Short"] = Order("Short", -1.0, order_type=_order_type_entry,
                                      limit=1.17, stop=1.15)
    eng.sync(BAR, last_price=1.16)
    assert broker.entries == []
    eng.sync(BAR + MINUTE, last_price=1.15)
    intent, _ = broker.entries[0]
    assert (intent.side, intent.order_type, intent.limit) == ("sell", OrderType.LIMIT, 1.17)


def __test_a_trigger_under_quarantine_opens_nothing__():
    """Under quarantine the triggered order is dropped, never sent."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos)
    eng.sync(BAR, last_price=1.17)
    eng.record_quarantine("external cancel detected")
    eng._drive_stop_limit_watches(last_price=1.19)
    assert broker.entries == []
    assert eng._stop_limit_engine.get_watch("Long") is None


def __test_a_stale_read_defers_the_trigger__():
    """A cycle whose broker read failed sends no LIMIT; the next healthy one does."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos)
    eng.sync(BAR, last_price=1.17)
    stale: Future = Future()
    stale.set_exception(ExchangeRateLimitError("error.too-many.requests", retry_after=30.0))
    eng._inflight_read = stale

    eng.sync(BAR + MINUTE, last_price=1.185)
    assert broker.entries == []
    assert eng._stop_limit_engine.get_watch("Long") is not None

    eng._read_backoff_until = 0.0
    eng.sync(BAR + 2 * MINUTE, last_price=1.185)
    assert len(broker.entries) == 1
    assert broker.entries[0][0].order_type is OrderType.LIMIT


# === The invariant: no entry fill while armed =============================

def __test_a_fill_while_armed_is_booked_and_no_limit_follows__(tmp_path: Path):
    """A fill of the Pine id while nothing of the bot rests is booked as the
    order's fill; the watch takes that order as its LIMIT, so the trigger
    never sends a second order. The breach is journaled."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = store.open_run(_identity(), script_source='// stop-limit')
        broker = _FakeBroker()
        eng, pos = _engine(broker, ctx)
        _place(pos)
        eng.sync(BAR, last_price=1.17)

        eng.on_order_event(_fill("foreign-1", 0.4, 1.16))
        eng.sync(BAR + MINUTE, last_price=1.17)
        watch = eng._stop_limit_engine.get_watch("Long")
        assert watch is not None and watch.state == STOP_LIMIT_STATE_TRIGGERED
        assert eng.order_mapping["Long"] == ["foreign-1"]
        assert pos.size == pytest.approx(0.4)

        eng.sync(BAR + 2 * MINUTE, last_price=1.19)
        assert broker.entries == []
        events = list(ctx.iter_events_by_kind_since('stop_limit_fill_while_armed', 0))
        assert len(events) == 1
        ctx.close()


# === Cancel and modify ====================================================

def __test_cancelling_an_armed_order_touches_no_venue_order__():
    """Before the trigger there is nothing to cancel on the venue."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos)
    eng.sync(BAR, last_price=1.17)
    del pos.entry_orders["Long"]
    eng.sync(BAR + MINUTE, last_price=1.19)
    assert broker.cancels == [] and broker.entries == []
    assert eng._stop_limit_engine.get_watch("Long") is None
    assert "Long" not in eng.active_intents


def __test_cancelling_a_triggered_order_cancels_its_limit__():
    """After the trigger the LIMIT rests on the venue and is cancelled there."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos)
    eng.sync(BAR, last_price=1.19)
    del pos.entry_orders["Long"]
    eng.sync(BAR + MINUTE, last_price=1.19)
    assert [c.pine_id for c in broker.cancels] == ["Long"]
    assert eng._stop_limit_engine.get_watch("Long") is None


def __test_amending_an_armed_order_amends_the_watch__():
    """A changed stop / limit before the trigger only moves the watch."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos, limit=1.16, stop=1.18)
    eng.sync(BAR, last_price=1.17)
    _place(pos, limit=1.165, stop=1.19)
    eng.sync(BAR + MINUTE, last_price=1.185)
    watch = eng._stop_limit_engine.get_watch("Long")
    assert watch is not None and (watch.stop, watch.limit) == (1.19, 1.165)
    assert broker.entries == [] and broker.modifies == []
    eng.sync(BAR + 2 * MINUTE, last_price=1.19)
    assert broker.entries[0][0].limit == 1.165


def __test_amending_a_triggered_order_moves_only_its_limit__():
    """After the trigger a new limit amends the venue LIMIT; a new stop has no effect."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos, limit=1.16, stop=1.18)
    eng.sync(BAR, last_price=1.19)

    _place(pos, limit=1.16, stop=1.25)
    eng.sync(BAR + MINUTE, last_price=1.19)
    assert broker.modifies == []

    _place(pos, limit=1.17, stop=1.25)
    eng.sync(BAR + 2 * MINUTE, last_price=1.19)
    assert len(broker.modifies) == 1
    old, new = broker.modifies[0]
    assert (old.order_type, old.limit, old.stop) == (OrderType.LIMIT, 1.16, None)
    assert (new.order_type, new.limit, new.stop) == (OrderType.LIMIT, 1.17, None)
    assert len(broker.entries) == 1


def __test_turning_into_a_plain_limit_drops_the_watch__():
    """An armed stop-limit amended to a plain LIMIT is placed as that LIMIT."""
    broker = _FakeBroker()
    eng, pos = _engine(broker)
    _place(pos)
    eng.sync(BAR, last_price=1.17)
    pos.entry_orders["Long"] = Order("Long", 1.0, order_type=_order_type_entry, limit=1.16)
    eng.sync(BAR + MINUTE, last_price=1.17)
    assert eng._stop_limit_engine.get_watch("Long") is None
    assert [(i.order_type, i.limit, i.stop) for i, _ in broker.entries] == [
        (OrderType.LIMIT, 1.16, None)]


# === Restart ==============================================================

def __test_a_stop_crossed_while_away_triggers_after_the_restart__(tmp_path: Path):
    """The bars the restart replays are judged by their range."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = store.open_run(_identity(), script_source='// stop-limit')
        broker = _FakeBroker(store_ctx=ctx)
        eng, pos = _engine(broker, ctx)
        _place(pos)
        eng.sync(BAR, last_price=1.17)
        ctx.close()

        ctx = store.open_run(_identity(), script_source='// stop-limit')
        broker = _FakeBroker(store_ctx=ctx)
        eng, pos = _engine(broker, ctx)
        _place(pos)
        eng.observe_history_bar(BAR, 1.30, 1.10)
        eng.observe_history_bar(BAR + MINUTE, 1.179, 1.15)
        watch = eng._stop_limit_engine.get_watch("Long")
        assert watch is not None and not watch.crossed_offline
        eng.observe_history_bar(BAR + 2 * MINUTE, 1.182, 1.15)

        eng.sync(BAR + 3 * MINUTE, last_price=1.17)  # back below the stop
        assert len(broker.entries) == 1
        assert broker.entries[0][0].order_type is OrderType.LIMIT
        ctx.close()


def __test_an_armed_watch_survives_a_restart_without_a_second_row__(tmp_path: Path):
    """The re-emitted Pine order re-uses the persisted watch."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = store.open_run(_identity(), script_source='// stop-limit')
        broker = _FakeBroker(store_ctx=ctx)
        eng, pos = _engine(broker, ctx)
        _place(pos)
        eng.sync(BAR, last_price=1.17)
        coid = eng._stop_limit_engine.get_watch("Long").coid  # type: ignore[union-attr]
        ctx.close()

        ctx = store.open_run(_identity(), script_source='// stop-limit')
        broker = _FakeBroker(store_ctx=ctx)
        eng, pos = _engine(broker, ctx)
        _place(pos)
        eng.sync(BAR + MINUTE, last_price=1.17)
        watch = eng._stop_limit_engine.get_watch("Long")
        assert watch is not None and watch.coid == coid
        assert broker.entries == []
        eng.sync(BAR + 2 * MINUTE, last_price=1.19)
        assert len(broker.entries) == 1
        ctx.close()


class _Crash(BaseException):
    """The process dies inside the venue call."""


@pytest.mark.parametrize('outcome', ['unresolved', 'landed', 'refused'])
def __test_a_limit_lost_in_flight_is_sent_again_only_if_it_never_landed__(
        tmp_path: Path, outcome: str,
) -> None:
    """The process died inside the LIMIT send; the venue does not refuse a
    duplicate id, so the restart verifies the earlier send first."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        before = store.open_run(_identity(), script_source='// stop-limit')
        broker = _FakeBroker(
            capabilities=ExchangeCapabilities(short_selling=CapabilityLevel.NATIVE,
                                              idempotency=CapabilityLevel.PARTIAL_NATIVE),
            store_ctx=before)
        eng, pos = _engine(broker, before)
        _place(pos)
        eng.sync(BAR, last_price=1.17)

        async def crash(envelope) -> list[ExchangeOrder]:
            intent = envelope.intent
            coid = envelope.client_order_id(KIND_ENTRY_TRIGGERED)
            create_entry_order_row(
                before, coid=coid, symbol=intent.symbol, side=intent.side, qty=intent.qty,
                intent_key=intent.intent_key, pine_entry_id=intent.pine_id,
                kind=ENTRY_KIND_WORKING, order_type='limit')
            broker.entries.append((intent, envelope))
            raise _Crash()

        broker.execute_entry = crash  # type: ignore[method-assign]
        with pytest.raises(_Crash):
            eng._drive_stop_limit_watches(last_price=1.19)
        limit_coid = broker.entries[-1][1].client_order_id(KIND_ENTRY_TRIGGERED)
        before.close()

        ctx = store.open_run(_identity(), script_source='// stop-limit')
        broker = _FakeBroker(
            capabilities=ExchangeCapabilities(short_selling=CapabilityLevel.NATIVE,
                                              idempotency=CapabilityLevel.PARTIAL_NATIVE),
            store_ctx=ctx)
        eng, pos = _engine(broker, ctx)
        watch = eng._stop_limit_engine.get_watch("Long")
        assert watch is not None and watch.state == STOP_LIMIT_STATE_LIMIT_PENDING
        assert watch.order_coid == limit_coid

        ctx.set_order_state(limit_coid, 'disposition_unknown')
        eng._drive_stop_limit_watches(last_price=None)
        assert broker.entries == []

        if outcome == 'landed':
            ctx.upsert_order(limit_coid, state='confirmed', exchange_order_id='ex-landed')
        elif outcome == 'refused':
            ctx.upsert_order(limit_coid, state='rejected')
            ctx.close_order(limit_coid)
        eng._drive_stop_limit_watches(last_price=None)

        watch = eng._stop_limit_engine.get_watch("Long")
        if outcome == 'unresolved':
            assert broker.entries == []
            assert watch is not None and watch.state == STOP_LIMIT_STATE_LIMIT_PENDING
        elif outcome == 'landed':
            assert broker.entries == []
            assert watch is not None and watch.state == STOP_LIMIT_STATE_TRIGGERED
            assert eng.order_mapping["Long"] == ['ex-landed']
        else:
            assert [(i.order_type, i.triggered_entry) for i, _ in broker.entries] == [
                (OrderType.LIMIT, True)]
            assert watch is not None and watch.state == STOP_LIMIT_STATE_TRIGGERED
        ctx.close()


def __test_an_unanswered_limit_is_not_sent_again_in_process__(tmp_path: Path):
    """A lost answer parks the send; the next tick does not send it a second time."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = store.open_run(_identity(), script_source='// stop-limit')
        broker = _FakeBroker(
            capabilities=ExchangeCapabilities(short_selling=CapabilityLevel.NATIVE,
                                              idempotency=CapabilityLevel.PARTIAL_NATIVE),
            store_ctx=ctx)
        eng, pos = _engine(broker, ctx)
        _place(pos)
        eng.sync(BAR, last_price=1.17)

        async def lost_answer(envelope) -> list[ExchangeOrder]:
            intent = envelope.intent
            coid = envelope.client_order_id(KIND_ENTRY_TRIGGERED)
            create_entry_order_row(
                ctx, coid=coid, symbol=intent.symbol, side=intent.side, qty=intent.qty,
                intent_key=intent.intent_key, pine_entry_id=intent.pine_id,
                kind=ENTRY_KIND_WORKING, order_type='limit')
            broker.entries.append((intent, envelope))
            ctx.set_order_state(coid, 'disposition_unknown')
            raise OrderDispositionUnknownError('reply lost', client_order_id=coid)

        broker.execute_entry = lost_answer  # type: ignore[method-assign]
        eng._drive_stop_limit_watches(last_price=1.19)
        eng._drive_stop_limit_watches(last_price=1.20)

        assert len(broker.entries) == 1
        watch = eng._stop_limit_engine.get_watch("Long")
        assert watch is not None and watch.state == STOP_LIMIT_STATE_LIMIT_PENDING
        ctx.close()


# === Legacy OCO rows ======================================================

def _legacy_row(ctx: RunContext, *, state: str | None = None,
                market_coid: str | None = None) -> None:
    create_stop_limit_watch_row(
        ctx, coid="legacy-watch", symbol=SYMBOL, side="buy", qty=1.0,
        intent_key="Long", pine_entry_id="Long", stop=1.18, limit=1.16,
        armed_bar_ts_ms=BAR)
    patch: dict = {LEGACY_EXTRAS_KEY_OCO_LIMIT_COID: "legacy-limit"}
    if market_coid is not None:
        patch[LEGACY_EXTRAS_KEY_OCO_MARKET_COID] = market_coid
    update_stop_limit_watch(ctx, coid="legacy-watch", extras_patch=patch)
    if state is not None:
        row = ctx.get_order("legacy-watch")
        assert row is not None
        ctx.upsert_order("legacy-watch", extras={**(row.extras or {}),
                                                 "entry_stop_state": state})


@pytest.mark.parametrize("outcome,closed", [
    (CancelDispositionOutcome.CANCEL_CONFIRMED, True),
    (CancelDispositionOutcome.ALREADY_FILLED, True),
    (CancelDispositionOutcome.UNKNOWN, False),
])
def __test_a_legacy_oco_row_cancels_its_resting_limit_and_closes__(
        tmp_path: Path, outcome, closed,
):
    """An earlier version's resting LIMIT is cancelled at restart and the row
    closed; an unknown outcome retries the next sync."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = store.open_run(_identity(), script_source='// stop-limit')
        _legacy_row(ctx)
        ctx.close()

        ctx = store.open_run(_identity(), script_source='// stop-limit')
        broker = _FakeBroker(cancel_outcome=outcome, store_ctx=ctx)
        eng, pos = _engine(broker, ctx)
        eng._order_mapping["Long"] = ["ex-legacy-limit"]
        eng._retire_legacy_oco_watches()

        assert [c.pine_id for c in broker.outcome_cancels] == ["Long"]
        row = ctx.get_order("legacy-watch")
        assert row is not None and (row.closed_ts_ms is not None) is closed
        assert (eng._stop_limit_engine.get_watch("Long") is None) is closed
        assert broker.entries == []
        # A cancelled LIMIT releases the order for a stop-limit watch; a filled
        # one stays mapped, its fill is the order's fill.
        released = outcome is CancelDispositionOutcome.CANCEL_CONFIRMED
        assert ("Long" not in eng.order_mapping) is released
        if not closed:
            eng.sync(BAR + 2 * MINUTE, last_price=1.17)
            assert len(broker.outcome_cancels) == 2
        ctx.close()


def __test_a_retired_legacy_row_hands_over_to_a_stop_limit_watch__(tmp_path: Path):
    """Once the legacy LIMIT is gone, the Pine order is armed as a stop-limit."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = store.open_run(_identity(), script_source='// stop-limit')
        _legacy_row(ctx)
        ctx.close()

        ctx = store.open_run(_identity(), script_source='// stop-limit')
        broker = _FakeBroker(store_ctx=ctx)
        eng, pos = _engine(broker, ctx)
        eng.sync(BAR + MINUTE, last_price=1.17)
        _place(pos)
        eng.sync(BAR + 2 * MINUTE, last_price=1.17)
        watch = eng._stop_limit_engine.get_watch("Long")
        assert watch is not None and not watch.legacy
        assert watch.state == STOP_LIMIT_STATE_ARMED
        assert broker.entries == []
        ctx.close()


def __test_a_legacy_market_pending_row_closes_once_answered__(tmp_path: Path):
    """The OCO model's stop-fired MARKET is not sent again; the row closes
    once that send is no longer unanswered."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = store.open_run(_identity(), script_source='// stop-limit')
        create_entry_order_row(
            ctx, coid="legacy-market", symbol=SYMBOL, side="buy", qty=1.0,
            intent_key="Long", pine_entry_id="Long", kind=ENTRY_KIND_WORKING,
            order_type='market')
        ctx.set_order_state("legacy-market", 'disposition_unknown')
        _legacy_row(ctx, state=LEGACY_OCO_STATE_MARKET_PENDING, market_coid="legacy-market")
        ctx.close()

        ctx = store.open_run(_identity(), script_source='// stop-limit')
        broker = _FakeBroker(store_ctx=ctx)
        eng, pos = _engine(broker, ctx)
        eng._retire_legacy_oco_watches()
        assert eng._stop_limit_engine.get_watch("Long") is not None
        ctx.upsert_order("legacy-market", state='confirmed', exchange_order_id='ex-m')
        eng._retire_legacy_oco_watches()
        assert eng._stop_limit_engine.get_watch("Long") is None
        assert broker.entries == [] and broker.outcome_cancels == []
        ctx.close()


# === Native venue =========================================================

def __test_a_native_venue_gets_the_stop_limit_order_as_it_is__():
    """NATIVE: one entry with both prices, no software watch."""
    broker = _FakeBroker(capabilities=_native_caps())
    eng, pos = _engine(broker)
    _place(pos)
    eng.sync(BAR, last_price=1.17)
    assert len(broker.entries) == 1
    intent, _ = broker.entries[0]
    assert (intent.order_type, intent.limit, intent.stop) == (OrderType.LIMIT, 1.16, 1.18)
    assert not intent.triggered_entry
    assert eng._stop_limit_engine.get_watch("Long") is None


def __test_a_native_stop_limit_refused_as_crossed_is_sent_as_its_limit__():
    """A stop already passed is triggered: the order goes out as its plain LIMIT,
    never at market."""
    broker = _FakeBroker(capabilities=_native_caps())
    broker.reject_next_entries.append(ExchangeOrderRejectedError(
        "retCode=110092 expect Rising", level_crossed=True))
    eng, pos = _engine(broker)
    _place(pos)
    eng.sync(BAR, last_price=1.19)
    assert [(i.order_type, i.limit, i.stop, i.triggered_entry) for i, _ in broker.entries] == [
        (OrderType.LIMIT, 1.16, 1.18, False), (OrderType.LIMIT, 1.16, None, True)]
    assert broker.quote_reads == 0


# === A triggered LIMIT the venue refuses as marketable =====================

def _refused_limit(broker: _FakeBroker) -> None:
    broker.reject_next_entries.append(ExchangeOrderRejectedError(
        "error.validation.limit.price", level_crossed=True))


def __test_a_refused_marketable_limit_goes_at_market_within_the_limit__():
    """The fresh quote is still within the limit: the LIMIT goes out at market."""
    broker = _FakeBroker(quote=Quote(bid=1.199, ask=1.2))
    _refused_limit(broker)
    eng, pos = _engine(broker)
    _place(pos, limit=1.20, stop=1.18)
    eng.sync(BAR, last_price=1.185)
    assert [(i.order_type, i.triggered_entry) for i, _ in broker.entries] == [
        (OrderType.LIMIT, True), (OrderType.MARKET, True)]
    assert broker.quote_reads == 1


def __test_a_refused_limit_whose_quote_moved_away_rests_instead__():
    """The quote moved beyond the limit: the LIMIT is sent again to rest, never at market."""
    broker = _FakeBroker(quote=Quote(bid=1.204, ask=1.205))
    _refused_limit(broker)
    eng, pos = _engine(broker)
    _place(pos, limit=1.20, stop=1.18)
    eng.sync(BAR, last_price=1.185)
    assert [(i.order_type, i.limit) for i, _ in broker.entries] == [
        (OrderType.LIMIT, 1.20), (OrderType.LIMIT, 1.20)]
    coids = [env.client_order_id(KIND_ENTRY_TRIGGERED) for _, env in broker.entries]
    assert coids[0] != coids[1]
    watch = eng._stop_limit_engine.get_watch("Long")
    assert watch is not None and watch.state == STOP_LIMIT_STATE_TRIGGERED
    assert watch.order_coid == coids[1]


def __test_a_resent_limit_refused_again_is_not_resent_a_second_time__():
    """The re-send of a refused LIMIT is judged once: a second refusal is a
    counted reject, never a loop of re-sends."""
    broker = _FakeBroker(quote=Quote(bid=1.204, ask=1.205))
    _refused_limit(broker)
    _refused_limit(broker)
    _refused_limit(broker)
    eng, pos = _engine(broker)
    _place(pos, limit=1.20, stop=1.18)
    eng.sync(BAR, last_price=1.185)
    assert [i.order_type for i, _ in broker.entries] == [OrderType.LIMIT, OrderType.LIMIT]
    assert broker.quote_reads == 1
    assert eng._stop_limit_engine.get_watch("Long") is None


def __test_a_refused_limit_without_a_quote_is_not_sent_at_market__():
    """Without a quote the refusal stays a counted reject; nothing goes at market."""
    broker = _FakeBroker(quote=None)
    _refused_limit(broker)
    eng, pos = _engine(broker)
    _place(pos, limit=1.20, stop=1.18)
    eng.sync(BAR, last_price=1.185)
    assert [i.order_type for i, _ in broker.entries] == [OrderType.LIMIT]
    assert eng._stop_limit_engine.get_watch("Long") is None
