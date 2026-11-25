"""
The binding book across a restart.

The live engine keeps a binding per entry fill, settled by the origin of each
closing fill (see :class:`~pynecore.lib.strategy._EntryBinding`). The trade
rows a restart seeds from the journal are FIFO-shaped, so the book itself is
written to the store after every change and resumed from there when it holds
the adopted position and saw every fill the journal shows; otherwise the FIFO
adoption stands and the book stays unfaithful.
"""
import dataclasses
import itertools
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.core.broker.idempotency import KIND_CLOSE, KIND_ENTRY, KIND_EXIT_SL, KIND_EXIT_TP
from pynecore.core.broker.models import (
    CapabilityLevel,
    DispatchEnvelope,
    ExchangeCapabilities,
    ExchangeOrder,
    ExchangePosition,
    ExitIntent,
    INTENT_KEY_SEP,
    LegType,
    OrderEvent,
    OrderStatus,
    OrderType,
)
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import (
    ENTRY_BOOK_CURSOR_EVENT_KIND,
    ENTRY_BOOK_EVENT_KIND,
    ENTRY_FILL_ORDER_EVENT_KIND,
    BrokerStore,
    RunContext,
)
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.lib.strategy import Order, _order_type_close, _order_type_entry

SYMBOL = "BTCUSDT"
PLUGIN = "testbroker"
BAR_TS = 1_700_000_000_000

#: The two ways the Bybit plugin journals the exposure an exit fill takes off.
#: ``keep_until_flat`` (spot): the entry rows keep their ``filled_qty`` and
#: stay live until the flat sweep closes them once the account holds nothing
#: (``events.py`` ``_close_entry_rows_when_flat``; spot fills do not net the
#: rows). ``retire_on_net`` (derivatives): every reducing fill nets the entry
#: rows FIFO, lowering ``qty`` and ``filled_qty`` together and closing a row
#: left with nothing (``events.py`` ``_reduce_entry_ownership`` /
#: ``_net_entry_row``).
JOURNAL_SHAPES = ("keep_until_flat", "retire_on_net")


@pytest.fixture(autouse=True)
def _stub_script():
    prev = lib._script
    lib._script = SimpleNamespace(initial_capital=1_000_000.0, pyramiding=3)
    try:
        yield
    finally:
        lib._script = prev


@dataclass
class _Venue:
    """A venue whose plugin journals its orders the way the Bybit plugin does.

    Entry rows carry the entry's ``filled_qty``. Each exit rests as a
    take-profit and a stop-loss leg row; a filled leg's row gets its
    ``filled_qty`` and closes, a cancelled leg's row closes. What an exit
    fill does to the entry rows depends on :attr:`shape` (see
    :data:`JOURNAL_SHAPES`).
    """
    client_order_id_max_len = 30
    on_unexpected_cancel = "stop"
    journal: RunContext | None = None
    shape: str = "keep_until_flat"
    position: ExchangePosition | None = None
    open_orders: list[ExchangeOrder] = field(default_factory=list)
    entry_calls: list[DispatchEnvelope] = field(default_factory=list)
    exit_calls: list[DispatchEnvelope] = field(default_factory=list)
    modify_exit_calls: list[tuple[DispatchEnvelope, DispatchEnvelope]] = field(
        default_factory=list)
    cancel_calls: list[DispatchEnvelope] = field(default_factory=list)
    cancel_ref_calls: list[str] = field(default_factory=list)
    close_calls: list[DispatchEnvelope] = field(default_factory=list)

    @staticmethod
    def get_capabilities() -> ExchangeCapabilities:
        return ExchangeCapabilities(short_selling=CapabilityLevel.NATIVE)

    @staticmethod
    def _order(order_id: str, coid: str, side: str, qty: float) -> ExchangeOrder:
        return ExchangeOrder(
            id=order_id, symbol=SYMBOL, side=side, order_type=OrderType.LIMIT, qty=qty,
            filled_qty=0.0, remaining_qty=qty, price=None, stop_price=None,
            average_fill_price=None, status=OrderStatus.OPEN, timestamp=0.0, fee=0.0,
            fee_currency="", client_order_id=coid,
        )

    async def execute_entry(self, envelope):
        self.entry_calls.append(envelope)
        intent = envelope.intent
        coid = envelope.client_order_id(KIND_ENTRY)
        order_id = f"ent-{intent.pine_id}"
        assert self.journal is not None
        self.journal.upsert_order(
            coid, symbol=SYMBOL, side=intent.side, qty=intent.qty, state="confirmed",
            intent_key=intent.intent_key, exchange_order_id=order_id,
            pine_entry_id=intent.pine_id, filled_qty=0.0, extras={"kind": "position"})
        return [self._order(order_id, coid, intent.side, intent.qty)]

    async def execute_exit(self, envelope):
        self.exit_calls.append(envelope)
        intent = envelope.intent
        assert self.journal is not None and isinstance(intent, ExitIntent)
        orders = []
        for kind, leg, tp, sl in ((KIND_EXIT_TP, "tp", intent.tp_price, None),
                                  (KIND_EXIT_SL, "sl", None, intent.sl_price)):
            coid = envelope.client_order_id(kind)
            order_id = f"{intent.pine_id}-{leg}"
            self.journal.upsert_order(
                coid, symbol=SYMBOL, side=intent.side, qty=intent.qty, state="confirmed",
                intent_key=intent.intent_key, exchange_order_id=order_id,
                from_entry=intent.from_entry, tp_level=tp, sl_level=sl,
                extras={"kind": "exit_leg"})
            order = self._order(order_id, coid, intent.side, intent.qty)
            self.open_orders.append(order)
            orders.append(order)
        return orders

    async def modify_exit(self, old, new):
        self.modify_exit_calls.append((old, new))
        return []

    async def execute_cancel(self, envelope):
        self.cancel_calls.append(envelope)
        intent = envelope.intent
        from_entry = getattr(intent, 'from_entry', None)
        if self.journal is not None and from_entry is not None:
            key = f"{intent.pine_id}{INTENT_KEY_SEP}{from_entry}"
            for row in list(self.journal.iter_live_orders()):
                if row.intent_key == key:
                    self.journal.close_order(row.client_order_id)
            self.open_orders = [o for o in self.open_orders
                                if not o.id.startswith(f"{intent.pine_id}-")]
        return True

    async def cancel_broker_order_ref(self, ref):
        self.cancel_ref_calls.append(ref)
        self.open_orders = [o for o in self.open_orders if o.id != ref]

    async def execute_close(self, envelope):
        self.close_calls.append(envelope)
        coid = envelope.client_order_id(KIND_CLOSE)
        return self._order(f"close-{len(self.close_calls)}", coid, "sell", 1.0)

    async def get_open_orders(self, symbol=None):
        return list(self.open_orders)

    async def get_position(self, symbol):
        return self.position


def _long(size: float) -> ExchangePosition | None:
    if size <= 1e-12:
        return None
    return ExchangePosition(
        symbol=SYMBOL, side="long", size=size, entry_price=50_000.0, unrealized_pnl=0.0,
        liquidation_price=None, leverage=1.0, margin_mode="cash",
    )


def _identity() -> RunIdentity:
    return RunIdentity(strategy_id="t030", symbol=SYMBOL, timeframe="60",
                       account_id="test-account", label=None)


def _start(store: BrokerStore, position: float,
           open_orders: list[ExchangeOrder] | None = None, *,
           shape: str = "keep_until_flat", reconcile_every: int = 0,
           ) -> tuple[RunContext, OrderSyncEngine, BrokerPosition, _Venue]:
    """A process of the run starts over the venue position ``position``.

    :param reconcile_every: The engine's ``reconcile_every_n_syncs``; a
        periodic reconcile runs the reduce-coverage guard.
    """
    ctx = store.open_run(_identity(), script_source="src", script_path="t030.py")
    venue = _Venue(journal=ctx, shape=shape, position=_long(position),
                   open_orders=list(open_orders or []))
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=venue,  # type: ignore[arg-type]
        position=pos, symbol=SYMBOL, run_tag=ctx.run_tag, mintick=1.0, store_ctx=ctx,
        reconcile_every_n_syncs=reconcile_every,
    )
    engine.reconcile()
    return ctx, engine, pos, venue


def _fill(side: str, qty: float, *, pine_id: str, order_id: str,
          leg: LegType = LegType.ENTRY, from_entry: str | None = None,
          event_type: str = 'filled', filled_qty: float | None = None,
          remaining_qty: float = 0.0, journal_qty: float | None = None) -> OrderEvent:
    """A fill of ``qty``; the journal records it as ``journal_qty`` (``qty`` by default)."""
    order = ExchangeOrder(
        id=order_id, symbol=SYMBOL, side=side, order_type=OrderType.MARKET, qty=qty,
        filled_qty=qty if filled_qty is None else filled_qty, remaining_qty=remaining_qty,
        price=None, stop_price=None, average_fill_price=50_000.0,
        status=OrderStatus.FILLED, timestamp=0.0, fee=0.0, fee_currency="",
    )
    return OrderEvent(
        order=order, event_type=event_type, fill_price=50_000.0, fill_qty=qty,
        timestamp=0.0, pine_id=pine_id, leg_type=leg, from_entry=from_entry,
        journal_fill_qty=qty if journal_qty is None else journal_qty,
    )


def _deliver(engine: OrderSyncEngine, event: OrderEvent) -> None:
    """The event stream hands the event over and the engine drains it."""
    engine.on_order_event(event)
    engine._drain_events()  # type: ignore[attr-defined]


def _journal_entry_fill(ctx: RunContext, pine_id: str, filled: float) -> None:
    """The plugin journals an entry fill before the engine sees it."""
    row = next(row for row in ctx.iter_live_orders() if row.pine_entry_id == pine_id
               and row.from_entry is None)
    ctx.set_filled(row.client_order_id, filled)


def _journal_exit_fill(ctx: RunContext, venue: _Venue, exit_id: str, qty: float) -> None:
    """The plugin journals an exit's take-profit fill before the engine sees it.

    The leg row gets its ``filled_qty`` and closes. The entry rows follow the
    venue's journal shape (:data:`JOURNAL_SHAPES`).
    """
    for row in list(ctx.iter_live_orders()):
        if (row.intent_key is not None and row.intent_key.startswith(f"{exit_id}{INTENT_KEY_SEP}")
                and row.tp_level is not None):
            ctx.set_filled(row.client_order_id, qty)
            ctx.close_order(row.client_order_id)
    venue.open_orders = [o for o in venue.open_orders if o.id != f"{exit_id}-tp"]
    entries = sorted((entry for entry in ctx.iter_live_orders()
                      if entry.from_entry is None and entry.filled_qty > 0.0),
                     key=lambda entry: entry.created_ts_ms)
    if venue.shape == "retire_on_net":
        for entry in entries:
            take = min(entry.filled_qty, qty)
            if take <= 0.0:
                break
            if entry.qty - take <= 1e-12:
                ctx.close_order(entry.client_order_id)
            else:
                ctx.upsert_order(entry.client_order_id, qty=entry.qty - take,
                                 filled_qty=entry.filled_qty - take)
            qty -= take
    elif sum(entry.filled_qty for entry in entries) - qty <= 1e-12:
        for entry in entries:
            ctx.close_order(entry.client_order_id)


def _open_pyramid(ctx: RunContext, engine: OrderSyncEngine, pos: BrokerPosition,
                  venue: _Venue, ids: tuple[str, ...]) -> None:
    """Open one unit per entry id, each under its own take-profit / stop-loss bracket."""
    for i, pine_id in enumerate(ids, start=1):
        pos.entry_orders[pine_id] = Order(pine_id, 1.0, order_type=_order_type_entry)
        pos.exit_orders[(f"{pine_id}-X", pine_id)] = Order(
            pine_id, -1.0, order_type=_order_type_close, exit_id=f"{pine_id}-X",
            limit=51_000.0, stop=49_000.0)
        engine.sync(BAR_TS + i * 60_000)
        _journal_entry_fill(ctx, pine_id, 1.0)
        venue.position = _long(abs(pos.size) + 1.0)
        _deliver(engine, _fill('buy', 1.0, pine_id=pine_id, order_id=f"ent-{pine_id}"))
        engine.sync(BAR_TS + i * 60_000)


def _take_profit(ctx: RunContext, engine: OrderSyncEngine, venue: _Venue, entry_id: str, *,
                 deliver: bool = True) -> None:
    """``<entry_id>-X``'s take-profit fills one unit; the plugin journals it first.

    Without ``deliver`` the process stops before the engine sees the fill.
    """
    exit_id = f"{entry_id}-X"
    venue.position = _long(abs(venue.position.size if venue.position else 0.0) - 1.0)
    _journal_exit_fill(ctx, venue, exit_id, 1.0)
    if deliver:
        _deliver(engine, _fill('sell', 1.0, pine_id=exit_id, order_id=f"{exit_id}-tp",
                               leg=LegType.TAKE_PROFIT, from_entry=entry_id))


def _c432(store: BrokerStore, shape: str, *,
          last_fill_routed: bool = True) -> list[ExchangeOrder]:
    """The bybit-spot c432 sequence in one process, which then stops.

    L1, L2 and L3 open under their own brackets; L3-X and then L2-X take
    profit. One unit is left, bound to L1 under L1-X. Without
    ``last_fill_routed`` the process stops right after the plugin journaled
    L2-X's fill, before the engine routed it.

    :return: The orders left resting on the venue.
    """
    ctx, engine, pos, venue = _start(store, 0.0, shape=shape)
    _open_pyramid(ctx, engine, pos, venue, ("L1", "L2", "L3"))
    _take_profit(ctx, engine, venue, "L3")
    engine.sync(BAR_TS + 10 * 60_000)
    _take_profit(ctx, engine, venue, "L2", deliver=last_fill_routed)
    if last_fill_routed:
        engine.sync(BAR_TS + 11 * 60_000)
        assert [(b.entry_id, b.bound) for b in pos.bindings] == [("L1", 1.0)]
        assert [trade.entry_id for trade in pos.open_trades] == ["L3"]
    ctx.close()
    return list(venue.open_orders)


def _stored_book(ctx: RunContext) -> dict | None:
    latest = ctx.latest_event_for_run_id((ENTRY_BOOK_EVENT_KIND,))
    return latest[1] if latest is not None else None


def _stored_cursors(ctx: RunContext) -> dict[str, float]:
    """The fill cursors the store holds next to the book, one keyed event each."""
    return {key: payload['qty']
            for key, payload in ctx.keyed_events_for_run_id(
                ENTRY_BOOK_CURSOR_EVENT_KIND).items()}


def _book(pos: BrokerPosition) -> list[tuple[str | None, float]]:
    return [(binding.entry_id, binding.bound) for binding in pos.bindings]


def _rejections(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records
            if "stored binding book does not fit" in record.getMessage()]


def _run_guarded_syncs(engine: OrderSyncEngine, first_bar: int) -> None:
    """Three bars with a periodic reconcile after each: the reduce-coverage guard runs."""
    for bar in range(first_bar, first_bar + 3):
        engine.sync(BAR_TS + bar * 60_000)


@pytest.mark.parametrize("shape", JOURNAL_SHAPES)
def __test_a_restart_resumes_the_c432_book_and_keeps_l1_x__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, shape: str) -> None:
    """After the restart the book is the book before it: L1 holds the unit under L1-X.

    The journal accounts for the unit FIFO, so a FIFO adoption would bind it
    to L3. The stored book saw every fill the journal shows and comes back
    verbatim and faithful; L1-X's two legs are the resumed position's
    protection, and through bars on which the reduce-coverage guard runs
    nothing is cancelled, sent or amended. L1-X's take-profit then settles
    L1's own binding and the book ends flat and faithful.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        resting = _c432(store, shape)
        assert sorted(o.id for o in resting) == ["L1-X-sl", "L1-X-tp"]

        with caplog.at_level(logging.INFO):
            ctx, engine, pos, venue = _start(store, 1.0, resting, shape=shape,
                                             reconcile_every=1)
            _run_guarded_syncs(engine, 20)

        assert pos.size == 1.0
        assert [trade.entry_id for trade in pos.open_trades] == ["L3"]
        assert _book(pos) == [("L1", 1.0)]
        assert pos.book_faithful
        assert not _rejections(caplog)
        key = f"L1-X{INTENT_KEY_SEP}L1"
        assert engine._journal_held_exit_legs == {  # type: ignore[attr-defined]
            "L1-X-tp": key, "L1-X-sl": key}
        assert venue.cancel_calls == [] and venue.exit_calls == []
        assert venue.modify_exit_calls == []
        assert sorted(o.id for o in venue.open_orders) == ["L1-X-sl", "L1-X-tp"]

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _take_profit(ctx, engine, venue, "L1")
            engine.sync(BAR_TS + 30 * 60_000)

        assert pos.size == 0.0
        assert pos.bindings == [] and pos.book_faithful
        assert not any("names no open binding" in r.getMessage() for r in caplog.records)
        stored = _stored_book(ctx)
        assert stored is not None
        assert stored['bindings'] == [] and stored['faithful'] is True
        assert 'fill_cursors' not in stored and _stored_cursors(ctx) == {}
        ctx.close()


@pytest.mark.parametrize("shape", JOURNAL_SHAPES)
def __test_a_book_written_a_fill_behind_is_rejected_and_the_adoption_stays_fifo__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, shape: str) -> None:
    """The process stopped between the plugin's journal write and the book's.

    The stored book still holds L1 and L2 (two units) while the venue and the
    journal hold one: it is rejected with a warning, the FIFO adoption binds
    the unit to L3 and the book is not faithful. L1-X's legs, which protect
    that unit, are not cancelled through bars on which the reduce-coverage
    guard runs; the stop L2-X left behind is.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        resting = _c432(store, shape, last_fill_routed=False)

        with caplog.at_level(logging.WARNING):
            ctx, engine, pos, venue = _start(store, 1.0, resting, shape=shape,
                                             reconcile_every=1)
            _run_guarded_syncs(engine, 20)

        rejections = _rejections(caplog)
        assert len(rejections) == 1
        assert "its bindings hold 2.0, the adopted position 1.0" in rejections[0]
        assert _book(pos) == [("L3", 1.0)]
        assert not pos.book_faithful
        assert not any(getattr(c.intent, 'from_entry', None) == "L1"
                       for c in venue.cancel_calls)
        assert venue.exit_calls == [] and venue.modify_exit_calls == []
        assert {"L1-X-sl", "L1-X-tp"} <= {o.id for o in venue.open_orders}
        # The stop of L2-X, whose take-profit filled before the stop, is the
        # spent bracket's leftover: the reduce-coverage guard cancels it.
        assert venue.cancel_ref_calls == ["L2-X-sl"]
        ctx.close()


def __test_a_book_that_missed_an_entry_and_an_exit_of_equal_size_is_rejected__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The journal holds fills the stored book never took in, though the sizes agree.

    The book holds L1 and L2. L3's entry and L1-X's take-profit fill while
    the process is down (or right before it stops): the plugin journals both,
    the engine routes neither. The venue still holds two units, so the size
    and the side agree with the stored book, but L3's entry row and L1-X's
    take-profit row are filled beyond what the book took in of those
    orders: the book is rejected and the unit is attributed FIFO, unfaithful.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, pos, venue = _start(store, 0.0, shape="retire_on_net")
        _open_pyramid(ctx, engine, pos, venue, ("L1", "L2"))
        pos.entry_orders["L3"] = Order("L3", 1.0, order_type=_order_type_entry, limit=49_500.0)
        pos.exit_orders[("L3-X", "L3")] = Order(
            "L3", -1.0, order_type=_order_type_close, exit_id="L3-X",
            limit=52_000.0, stop=48_000.0)
        engine.sync(BAR_TS + 5 * 60_000)
        stored = _stored_book(ctx)
        assert stored is not None
        assert [(b['entry_id'], b['bound']) for b in stored['bindings']] == [
            ("L1", 1.0), ("L2", 1.0)]
        _journal_entry_fill(ctx, "L3", 1.0)
        _journal_exit_fill(ctx, venue, "L1-X", 1.0)
        resting = list(venue.open_orders)
        ctx.close()

        with caplog.at_level(logging.WARNING):
            ctx, engine, pos, venue = _start(store, 2.0, resting, shape="retire_on_net",
                                             reconcile_every=1)
            _run_guarded_syncs(engine, 20)

        rejections = _rejections(caplog)
        assert len(rejections) == 1
        assert "the journal shows 1.0 filled of order" in rejections[0]
        assert not pos.book_faithful
        assert _book(pos) == [("L2", 1.0), ("L3", 1.0)]
        assert not any(getattr(c.intent, 'from_entry', None) in ("L2", "L3")
                       for c in venue.cancel_calls)
        ctx.close()


def __test_fills_drained_onto_an_empty_startup_book_never_make_it_faithful__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A fill queued before the startup reconcile is booked without the adoption.

    The previous process held L1 and L2 and stopped while L3's entry filled.
    The stream hands L3's fill over before the startup reconcile, so the
    startup books it on the empty position and skips the adoption: the book
    holds only L3 while the venue holds three units. It must not be
    faithful, or the next sync retires L1-X as an exit whose parent is
    spent.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, pos, venue = _start(store, 0.0)
        _open_pyramid(ctx, engine, pos, venue, ("L1", "L2"))
        pos.entry_orders["L3"] = Order("L3", 1.0, order_type=_order_type_entry, limit=49_500.0)
        engine.sync(BAR_TS + 5 * 60_000)
        _journal_entry_fill(ctx, "L3", 1.0)
        resting = list(venue.open_orders)
        ctx.close()

        ctx = store.open_run(_identity(), script_source="src", script_path="t030.py")
        venue = _Venue(journal=ctx, position=_long(3.0), open_orders=resting)
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=venue,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=ctx.run_tag, mintick=1.0, store_ctx=ctx,
        )
        engine.on_order_event(_fill('buy', 1.0, pine_id="L3", order_id="ent-L3"))
        with caplog.at_level(logging.WARNING):
            engine.reconcile()

        assert _book(pos) == [("L3", 1.0)]
        assert not pos.book_faithful
        stored = _stored_book(ctx)
        assert stored is not None and stored['faithful'] is False
        pos.exit_orders[("L1-X", "L1")] = Order(
            "L1", -1.0, order_type=_order_type_close, exit_id="L1-X",
            limit=51_000.0, stop=49_000.0)
        with caplog.at_level(logging.WARNING):
            engine.sync(BAR_TS + 20 * 60_000)
        assert ("L1-X", "L1") in pos.exit_orders
        assert not any("outlived its parent" in r.getMessage() for r in caplog.records)
        ctx.close()


def __test_a_book_never_judged_at_startup_is_not_faithful_while_it_holds_exposure__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """No startup adoption ran before the first completed sync: the book is not faithful.

    Its book holds only the fills this process booked, not what the previous
    process held.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = store.open_run(_identity(), script_source="src", script_path="t030.py")
        venue = _Venue(journal=ctx, position=_long(1.0))
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=venue,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=ctx.run_tag, mintick=1.0, store_ctx=ctx,
        )
        engine.on_order_event(_fill('buy', 1.0, pine_id="L1", order_id="ent-L1"))
        with caplog.at_level(logging.WARNING):
            engine.sync(BAR_TS)

        assert pos.size == 1.0
        assert not pos.book_faithful
        stored = _stored_book(ctx)
        assert stored is not None and stored['faithful'] is False
        assert any("never judged against the venue" in r.getMessage() for r in caplog.records)
        ctx.close()


def __test_a_flat_stored_book_resurrects_nothing__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A book written flat comes back flat, and never fits an open position.

    The first restart finds the venue flat and the book stays empty. When the
    venue holds a unit the flat book never saw (an entry the plugin journaled
    before the process stopped), the stored book is rejected and the
    adoption binds the unit FIFO, unfaithful.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, pos, venue = _start(store, 0.0)
        _open_pyramid(ctx, engine, pos, venue, ("L1",))
        _take_profit(ctx, engine, venue, "L1")
        engine.sync(BAR_TS + 10 * 60_000)
        stored = _stored_book(ctx)
        assert stored is not None and stored['bindings'] == []
        ctx.close()

        ctx, engine, pos, venue = _start(store, 0.0)
        assert pos.bindings == [] and pos.book_faithful
        ctx.close()

        ctx = store.open_run(_identity(), script_source="src", script_path="t030.py")
        ctx.upsert_order("row-l2", symbol=SYMBOL, side="buy", qty=1.0, filled_qty=1.0,
                         state="confirmed", pine_entry_id="L2", exchange_order_id="ent-L2",
                         extras={"kind": "position"})
        ctx.close()
        with caplog.at_level(logging.WARNING):
            ctx, engine, pos, venue = _start(store, 1.0)

        rejections = _rejections(caplog)
        assert len(rejections) == 1
        assert "its bindings hold 0.0, the adopted position 1.0" in rejections[0]
        assert _book(pos) == [("L2", 1.0)]
        assert not pos.book_faithful
        ctx.close()


def __test_a_journal_without_a_stored_book_adopts_fifo_and_starts_writing_it__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A run that never wrote a book adopts as before, says so, and writes its book from then on."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = store.open_run(_identity(), script_source="src", script_path="t030.py")
        for pine_id in ("A", "B"):
            ctx.upsert_order(f"row-{pine_id}", symbol=SYMBOL, side="buy", qty=1.0,
                             filled_qty=1.0, state="confirmed", pine_entry_id=pine_id,
                             exchange_order_id=f"ent-{pine_id}", extras={"kind": "position"})
        ctx.close()

        with caplog.at_level(logging.INFO):
            ctx, engine, pos, venue = _start(store, 2.0)

        assert _book(pos) == [("A", 1.0), ("B", 1.0)]
        assert not pos.book_faithful
        assert not _rejections(caplog)
        assert any("no stored binding book for the adopted position" in r.getMessage()
                   for r in caplog.records if r.levelno == logging.INFO)
        stored = _stored_book(ctx)
        assert stored is not None
        assert [(b['entry_id'], b['bound']) for b in stored['bindings']] == [
            ("A", 1.0), ("B", 1.0)]
        assert stored['faithful'] is False
        ctx.close()


def __test_a_malformed_stored_book_is_rejected__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A stored binding with a non-positive initial size is malformed: FIFO adoption, unfaithful."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = store.open_run(_identity(), script_source="src", script_path="t030.py")
        ctx.upsert_order("row-a", symbol=SYMBOL, side="buy", qty=1.0, filled_qty=1.0,
                         state="confirmed", pine_entry_id="A", exchange_order_id="ent-A",
                         extras={"kind": "position"})
        ctx.log_event(ENTRY_BOOK_EVENT_KIND, payload={
            'size': 1.0, 'entry_seq': 1, 'faithful': True, 'written_ms': int(time.time() * 1000),
            'fill_cursors': {"ent-A": 1.0},
            'bindings': [{'seq': 1, 'entry_id': "A", 'init_size': 0.0, 'bound': 1.0,
                          'sign': 1.0, 'entry_price': 50_000.0, 'order_ids': ["ent-A"]}],
        })
        ctx.close()

        with caplog.at_level(logging.WARNING):
            ctx, engine, pos, venue = _start(store, 1.0)

        rejections = _rejections(caplog)
        assert len(rejections) == 1 and "it is malformed" in rejections[0]
        assert _book(pos) == [("A", 1.0)]
        assert not pos.book_faithful
        ctx.close()


def __test_a_later_slice_of_an_entry_order_grows_its_restored_binding__(
        tmp_path: Path) -> None:
    """The venue order id of a binding survives the restart with it.

    L1's order fills 0.4 before the restart and 0.6 after it: the second
    slice grows the restored binding instead of opening a second one.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, pos, venue = _start(store, 0.0)
        pos.entry_orders["L1"] = Order("L1", 1.0, order_type=_order_type_entry, limit=49_000.0)
        engine.sync(BAR_TS + 60_000)
        _journal_entry_fill(ctx, "L1", 0.4)
        venue.position = _long(0.4)
        _deliver(engine, _fill('buy', 0.4, pine_id="L1", order_id="ent-L1",
                               event_type='partial', filled_qty=0.4, remaining_qty=0.6))
        assert _book(pos) == [("L1", 0.4)]
        stored = _stored_book(ctx)
        assert stored is not None and stored['bindings'][0]['order_ids'] == ["ent-L1"]
        resting = list(venue.open_orders)
        ctx.close()

        ctx, engine, pos, venue = _start(store, 0.4, resting)
        assert _book(pos) == [("L1", 0.4)] and pos.book_faithful
        _journal_entry_fill(ctx, "L1", 1.0)
        venue.position = _long(1.0)
        _deliver(engine, _fill('buy', 0.6, pine_id="L1", order_id="ent-L1", filled_qty=1.0))

        assert pos.size == 1.0
        assert _book(pos) == [("L1", 1.0)]
        assert pos.bindings[0].init_size == 1.0
        ctx.close()


@pytest.mark.parametrize("shape", JOURNAL_SHAPES)
def __test_an_unfaithful_book_stays_unfaithful_across_a_restart_until_flat__(
        tmp_path: Path, shape: str) -> None:
    """A book attributed FIFO before the restart comes back as it was, not re-attributed.

    The first process adopted A and B FIFO from the journal; B's take-profit
    then settled B's binding while the rows and the journal accounted the
    remaining unit FIFO. The restart resumes A's binding, still unfaithful (a
    FIFO adoption would bind the unit to B). Once a close flattens the book
    it is faithful again.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = store.open_run(_identity(), script_source="src", script_path="t030.py")
        for pine_id in ("A", "B"):
            ctx.upsert_order(f"row-{pine_id}", symbol=SYMBOL, side="buy", qty=1.0,
                             filled_qty=1.0, state="confirmed", pine_entry_id=pine_id,
                             exchange_order_id=f"ent-{pine_id}", extras={"kind": "position"})
        ctx.close()
        ctx, engine, pos, venue = _start(store, 2.0, shape=shape)
        assert not pos.book_faithful
        _take_profit(ctx, engine, venue, "B")
        assert _book(pos) == [("A", 1.0)]
        assert [trade.entry_id for trade in pos.open_trades] == ["B"]
        ctx.close()

        ctx, engine, pos, venue = _start(store, 1.0, shape=shape)
        assert _book(pos) == [("A", 1.0)]
        assert not pos.book_faithful

        venue.position = None
        _deliver(engine, _fill('sell', 1.0, pine_id="", order_id="close-1",
                               leg=LegType.CLOSE))
        assert pos.size == 0.0
        assert pos.bindings == [] and pos.book_faithful
        stored = _stored_book(ctx)
        assert stored is not None
        assert stored['bindings'] == [] and stored['faithful'] is True
        ctx.close()


def __test_the_retention_purge_keeps_the_newest_stored_book_of_a_run__(
        tmp_path: Path) -> None:
    """Every older event is purged past the retention window; the newest book survives."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = store.open_run(_identity(), script_source="src", script_path="t030.py")
        for size in (1.0, 2.0):
            ctx.log_event(ENTRY_BOOK_EVENT_KIND, payload={'size': size})
        ctx.log_event('some_audit', payload={})
        ctx.close()

        # A negative window puts the cutoff in the future: every event is past it.
        store.cleanup_old_data(retention_days=-1)

        ctx = store.open_run(_identity(), script_source="src", script_path="t030.py")
        assert _stored_book(ctx) == {'size': 2.0}
        assert [p for *_ids, p in ctx.iter_events_by_kind_for_run_id(ENTRY_BOOK_EVENT_KIND)] == [
            {'size': 2.0}]
        assert list(ctx.iter_events_by_kind_for_run_id('some_audit')) == []
        ctx.close()


@pytest.mark.parametrize("shape", JOURNAL_SHAPES)
def __test_the_stored_book_does_not_grow_with_the_orders_cycled_under_an_open_entry__(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str) -> None:
    """Fill cursors of orders closed past the restart check's slack leave the stored book.

    L1 stays open while C0..C5 each open and take profit, minutes apart. A
    cursor of an order closed that long before the write is never consulted
    by the restart check, so the stored book keeps only the cursors of live
    journal rows and of the orders closed within the slack: with rows retired
    as they net, a constant handful. The restart still resumes the book.
    """
    clock = [time.time()]
    monkeypatch.setattr(time, 'time', lambda: clock[0])
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, pos, venue = _start(store, 0.0, shape=shape)
        _open_pyramid(ctx, engine, pos, venue, ("L1",))
        stored_bytes: list[int] = []
        for cycle in range(6):
            pine_id = f"C{cycle}"
            bar = BAR_TS + (10 + cycle * 2) * 60_000
            clock[0] += 300.0
            pos.entry_orders[pine_id] = Order(pine_id, 1.0, order_type=_order_type_entry)
            pos.exit_orders[(f"{pine_id}-X", pine_id)] = Order(
                pine_id, -1.0, order_type=_order_type_close, exit_id=f"{pine_id}-X",
                limit=51_000.0, stop=49_000.0)
            engine.sync(bar)
            _journal_entry_fill(ctx, pine_id, 1.0)
            venue.position = _long(2.0)
            _deliver(engine, _fill('buy', 1.0, pine_id=pine_id, order_id=f"ent-{pine_id}"))
            engine.sync(bar)
            _take_profit(ctx, engine, venue, pine_id)
            engine.sync(bar + 60_000)
            books = [p for *_ids, p in ctx.iter_events_by_kind_for_run_id(ENTRY_BOOK_EVENT_KIND)]
            assert len(books) == 1
            stored_bytes.append(len(json.dumps(books[0])))

        assert max(stored_bytes[2:]) <= stored_bytes[1] + 64
        assert _book(pos) == [("L1", 1.0)]
        stored = _stored_book(ctx)
        assert stored is not None
        cursors = _stored_cursors(ctx)
        recent = itertools.chain(
            ctx.iter_live_orders(),
            ctx.iter_orders_closed_since_for_run_id(int(clock[0] * 1000) - 60_000))
        consulted = {key for row in recent
                     for key in OrderSyncEngine._journal_row_keys(row)}  # type: ignore[attr-defined]
        assert set(cursors) <= consulted | {"C5-X-tp"}
        assert not any(key.startswith(("C0-X", "C3-X")) for key in cursors)
        if shape == "retire_on_net":
            assert len(cursors) <= 3
        resting = list(venue.open_orders)
        ctx.close()

        ctx, engine, pos, venue = _start(store, 1.0, resting, shape=shape)
        assert _book(pos) == [("L1", 1.0)] and pos.book_faithful
        ctx.close()


@pytest.mark.parametrize("shape", JOURNAL_SHAPES)
def __test_each_book_write_supersedes_the_older_stored_books_of_the_run__(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture, shape: str) -> None:
    """The store holds one binding book per run, across restarts, and it is the newest.

    Only the newest book is ever read, so every landed write deletes the
    older ones, including those an earlier process instance wrote. A failed
    delete keeps the new book and is caught up by the next write; the
    restart resumes from the single stored book.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, pos, venue = _start(store, 0.0, shape=shape)
        _open_pyramid(ctx, engine, pos, venue, ("L1", "L2"))
        books = [p for *_ids, p in ctx.iter_events_by_kind_for_run_id(ENTRY_BOOK_EVENT_KIND)]
        assert [[b['entry_id'] for b in book['bindings']] for book in books] == [["L1", "L2"]]

        def _fail(_kind: str) -> int:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(ctx, 'delete_superseded_events', _fail)
        with caplog.at_level(logging.WARNING):
            _take_profit(ctx, engine, venue, "L2")
            engine.sync(BAR_TS + 10 * 60_000)
        assert any("superseded binding books could not be deleted" in rec.getMessage()
                   for rec in caplog.records)
        books = [p for *_ids, p in ctx.iter_events_by_kind_for_run_id(ENTRY_BOOK_EVENT_KIND)]
        assert len(books) > 1
        assert [b['entry_id'] for b in books[-1]['bindings']] == ["L1"]
        monkeypatch.undo()
        resting = list(venue.open_orders)
        ctx.close()

        ctx, engine, pos, venue = _start(store, 1.0, resting, shape=shape)
        assert _book(pos) == [("L1", 1.0)] and pos.book_faithful
        pos.entry_orders["L3"] = Order("L3", 1.0, order_type=_order_type_entry)
        engine.sync(BAR_TS + 11 * 60_000)
        _journal_entry_fill(ctx, "L3", 1.0)
        venue.position = _long(2.0)
        _deliver(engine, _fill('buy', 1.0, pine_id="L3", order_id="ent-L3"))
        books = [p for *_ids, p in ctx.iter_events_by_kind_for_run_id(ENTRY_BOOK_EVENT_KIND)]
        assert [[b['entry_id'] for b in book['bindings']] for book in books] == [["L1", "L3"]]
        resting = list(venue.open_orders)
        ctx.close()

        ctx, engine, pos, venue = _start(store, 2.0, resting, shape=shape)
        assert _book(pos) == [("L1", 1.0), ("L3", 1.0)] and pos.book_faithful
        ctx.close()


def __test_a_journal_read_failure_while_pruning_does_not_stop_the_fill_handling__(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture) -> None:
    """Cursor pruning is best effort: a failed journal read keeps the cursors.

    The book write runs at the end of the fill router; a pruning read that
    raised there aborted the drain. The fill is booked, the book is written
    with every cursor, and pruning resumes with the next write.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, pos, venue = _start(store, 0.0, shape="retire_on_net")
        _open_pyramid(ctx, engine, pos, venue, ("L1",))

        def _fail(_now_ms: int) -> bool:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(engine, '_prune_fill_cursors', _fail)
        pos.entry_orders["L2"] = Order("L2", 1.0, order_type=_order_type_entry)
        engine.sync(BAR_TS + 2 * 60_000)
        _journal_entry_fill(ctx, "L2", 1.0)
        venue.position = _long(2.0)
        with caplog.at_level(logging.WARNING):
            _deliver(engine, _fill('buy', 1.0, pine_id="L2", order_id="ent-L2"))

        assert _book(pos) == [("L1", 1.0), ("L2", 1.0)]
        assert any("could not be pruned" in rec.getMessage() for rec in caplog.records)
        stored = _stored_book(ctx)
        assert stored is not None
        assert [binding['entry_id'] for binding in stored['bindings']] == ["L1", "L2"]
        assert "ent-L2" in _stored_cursors(ctx)
        ctx.close()


@pytest.mark.parametrize(("fill_qty", "journal_qty"), [
    pytest.param(0.999, 1.0, id="spot_base_fee_net"),
    pytest.param(0.002, 100.0, id="inverse_contracts"),
])
def __test_a_restart_compares_the_book_with_the_journal_in_the_journal_unit__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
        fill_qty: float, journal_qty: float) -> None:
    """The fill moves the position by one figure and the journal row by another.

    Bybit spot books a buy fee-net in base units while the row keeps the gross
    executed quantity; Bybit inverse books base units while the row counts
    contracts. The fill cursor sums the journal quantities the fills carry, so
    the stored book that saw every fill is resumed, not discarded for a unit
    mismatch.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, pos, venue = _start(store, 0.0)
        pos.entry_orders["L1"] = Order("L1", 1.0, order_type=_order_type_entry)
        engine.sync(BAR_TS + 60_000)
        _journal_entry_fill(ctx, "L1", journal_qty)
        venue.position = _long(fill_qty)
        _deliver(engine, _fill('buy', fill_qty, pine_id="L1", order_id="ent-L1",
                               journal_qty=journal_qty))
        engine.sync(BAR_TS + 2 * 60_000)
        assert _book(pos) == [("L1", fill_qty)]
        assert set(_stored_cursors(ctx).values()) == {journal_qty}
        ctx.close()

        with caplog.at_level(logging.WARNING):
            ctx, engine, pos, venue = _start(store, fill_qty)

        assert not _rejections(caplog)
        assert _book(pos) == [("L1", fill_qty)] and pos.book_faithful
        ctx.close()


def __test_a_fill_without_a_journal_quantity_is_reported_and_never_vouched_for__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A plugin that leaves ``journal_fill_qty`` unset breaks the port contract.

    The fill is booked, the violation is reported once per order at ERROR, and
    its order's cursor does not move: the restart finds the journal ahead of
    the book and adopts the position FIFO instead of trusting a cursor in
    another unit.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, pos, venue = _start(store, 0.0)
        pos.entry_orders["L1"] = Order("L1", 1.0, order_type=_order_type_entry, limit=49_000.0)
        engine.sync(BAR_TS + 60_000)
        _journal_entry_fill(ctx, "L1", 0.4)
        venue.position = _long(0.4)
        unmeasured = dataclasses.replace(
            _fill('buy', 0.4, pine_id="L1", order_id="ent-L1", event_type='partial',
                  filled_qty=0.4, remaining_qty=0.6),
            journal_fill_qty=None)
        with caplog.at_level(logging.ERROR):
            _deliver(engine, unmeasured)
            _journal_entry_fill(ctx, "L1", 0.8)
            venue.position = _long(0.8)
            _deliver(engine, dataclasses.replace(unmeasured, order=dataclasses.replace(
                unmeasured.order, filled_qty=0.8, remaining_qty=0.2)))
        assert pos.size == pytest.approx(0.8)
        reports = [r for r in caplog.records if "carries no journal quantity" in r.getMessage()]
        assert len(reports) == 1 and reports[0].levelno == logging.ERROR
        resting = list(venue.open_orders)
        ctx.close()

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            ctx, engine, pos, venue = _start(store, 0.8, resting)

        rejections = _rejections(caplog)
        assert len(rejections) == 1
        assert "the journal shows 0.8 filled of order" in rejections[0]
        assert not pos.book_faithful
        ctx.close()


def __test_a_spot_book_write_stores_only_the_cursors_that_changed__(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On spot the entry rows stay live until flat, and so do their fill cursors.

    L1 stays open while C0..C7 each open and take profit; every entry row stays
    live, so the restart check needs every one of their cursors. They are kept
    as one keyed event each, and a book write replaces only the cursors its
    change moved: the book event stays the size of its open bindings and no
    write rewrites the cursors of earlier cycles. The restart resumes the book
    from the cursors the keyed events hold.
    """
    clock = [time.time()]
    monkeypatch.setattr(time, 'time', lambda: clock[0])
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, pos, venue = _start(store, 0.0, shape="keep_until_flat")
        _open_pyramid(ctx, engine, pos, venue, ("L1",))
        written: list[int] = []
        replace = ctx.replace_keyed_events

        def _count(kind, values, removed=()):
            written.append(len(values) + len(list(removed)))
            return replace(kind, values, removed)

        monkeypatch.setattr(ctx, 'replace_keyed_events', _count)
        book_bytes: list[int] = []
        for cycle in range(8):
            pine_id = f"C{cycle}"
            bar = BAR_TS + (10 + cycle * 2) * 60_000
            clock[0] += 300.0
            pos.entry_orders[pine_id] = Order(pine_id, 1.0, order_type=_order_type_entry)
            pos.exit_orders[(f"{pine_id}-X", pine_id)] = Order(
                pine_id, -1.0, order_type=_order_type_close, exit_id=f"{pine_id}-X",
                limit=51_000.0, stop=49_000.0)
            engine.sync(bar)
            _journal_entry_fill(ctx, pine_id, 1.0)
            venue.position = _long(2.0)
            _deliver(engine, _fill('buy', 1.0, pine_id=pine_id, order_id=f"ent-{pine_id}"))
            engine.sync(bar)
            _take_profit(ctx, engine, venue, pine_id)
            engine.sync(bar + 60_000)
            stored = _stored_book(ctx)
            assert stored is not None
            book_bytes.append(len(json.dumps(stored)))

        assert _book(pos) == [("L1", 1.0)]
        assert max(book_bytes) == min(book_bytes)
        assert max(written) <= 4
        live_entries = [row for row in ctx.iter_live_orders()
                        if row.from_entry is None and row.filled_qty > 0.0]
        assert len(live_entries) == 9
        cursors = _stored_cursors(ctx)
        for row in live_entries:
            assert max(cursors.get(key, 0.0)
                       for key in OrderSyncEngine._journal_row_keys(row)  # type: ignore[attr-defined]
                       ) == row.filled_qty
        monkeypatch.undo()
        resting = list(venue.open_orders)
        ctx.close()

        ctx, engine, pos, venue = _start(store, 1.0, resting, shape="keep_until_flat")
        assert _book(pos) == [("L1", 1.0)] and pos.book_faithful
        ctx.close()


def __test_the_retention_purge_keeps_the_stored_fill_cursors__(tmp_path: Path) -> None:
    """The keyed cursor events are what a restart judges the book with; age never purges them."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = store.open_run(_identity(), script_source="src", script_path="t030.py")
        ctx.replace_keyed_events(ENTRY_BOOK_CURSOR_EVENT_KIND,
                                 {"ent-A": {'qty': 1.0}, "ent-B": {'qty': 2.0}})
        ctx.replace_keyed_events(ENTRY_BOOK_CURSOR_EVENT_KIND,
                                 {"ent-A": {'qty': 1.5}}, removed=["ent-B"])
        ctx.close()

        store.cleanup_old_data(retention_days=-1)

        ctx = store.open_run(_identity(), script_source="src", script_path="t030.py")
        assert _stored_cursors(ctx) == {"ent-A": 1.5}
        assert len(list(ctx.iter_events_by_kind_for_run_id(ENTRY_BOOK_CURSOR_EVENT_KIND))) == 1
        ctx.close()


def _fill_b_before_a(store: BrokerStore, *, route_c: bool | None = None) -> None:
    """One process: entry A is sent first, entry B second, and B fills first.

    The binding book opens B's binding first, so a close that names no entry
    settles B first. With ``route_c`` set, a third entry C is sent and filled
    after both; with ``route_c=False`` the process stops right after the
    plugin journaled C's fill, before the engine routed it.
    """
    ctx, engine, pos, venue = _start(store, 0.0)
    ids = ("A", "B") if route_c is None else ("A", "B", "C")
    for i, pine_id in enumerate(ids, start=1):
        pos.entry_orders[pine_id] = Order(pine_id, 1.0, order_type=_order_type_entry)
        engine.sync(BAR_TS + i * 60_000)
    fills = ("B", "A") if route_c is None else ("B", "A", "C")
    for i, pine_id in enumerate(fills, start=1):
        _journal_entry_fill(ctx, pine_id, 1.0)
        venue.position = _long(float(i))
        if pine_id == "C" and not route_c:
            break
        _deliver(engine, _fill('buy', 1.0, pine_id=pine_id, order_id=f"ent-{pine_id}"))
        engine.sync(BAR_TS + (10 + i) * 60_000)
    assert _book(pos)[:2] == [("B", 1.0), ("A", 1.0)]
    places = {key: payload['seq'] for key, payload in ctx.keyed_events_for_run_id(
        ENTRY_FILL_ORDER_EVENT_KIND).items()}
    assert places["ent-B"] < places["ent-A"]
    ctx.close()


def __test_a_rejected_book_is_rebuilt_in_fill_order_not_creation_order__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The fallback book trims and orders the entries the way the binding book settled them.

    A was sent before B, but B filled first. While the bot is down one unit is
    closed without naming an entry; the stored book (A and B) no longer fits
    the venue's one unit and is rejected. A close naming no entry settles the
    entry that filled first, B, so the unit left is A's — the cTrader journal
    debits such a close in the same order. Rebuilt in creation order, the
    book would bind the unit to B.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        _fill_b_before_a(store)

        with caplog.at_level(logging.WARNING):
            ctx, _engine, pos, _venue = _start(store, 1.0)

        assert len(_rejections(caplog)) == 1
        assert [trade.entry_id for trade in pos.open_trades] == ["A"]
        assert _book(pos) == [("A", 1.0)] and not pos.book_faithful
        ctx.close()


def __test_an_entry_whose_fill_was_never_booked_follows_the_placed_ones__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Rows with a recorded place go by it; a row the engine never booked comes after them.

    The process stops after the plugin journaled C's fill, before the engine
    routed it: the stored book (B, A) misses C and is rejected. The rebuilt
    book holds B, A, C: the fill order the engine recorded, then C.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        _fill_b_before_a(store, route_c=False)

        with caplog.at_level(logging.WARNING):
            ctx, _engine, pos, _venue = _start(store, 3.0)

        assert len(_rejections(caplog)) == 1
        assert [trade.entry_id for trade in pos.open_trades] == ["B", "A", "C"]
        assert _book(pos) == [("B", 1.0), ("A", 1.0), ("C", 1.0)]
        ctx.close()


def __test_the_fill_order_is_kept_across_restarts_and_by_the_retention_purge__(
        tmp_path: Path) -> None:
    """A later process numbers its fills after the earlier ones; age never purges a place."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        _fill_b_before_a(store)
        store.cleanup_old_data(retention_days=-1)

        ctx, engine, pos, venue = _start(store, 2.0)
        pos.entry_orders["D"] = Order("D", 1.0, order_type=_order_type_entry)
        engine.sync(BAR_TS + 30 * 60_000)
        _journal_entry_fill(ctx, "D", 1.0)
        venue.position = _long(3.0)
        _deliver(engine, _fill('buy', 1.0, pine_id="D", order_id="ent-D"))
        engine.sync(BAR_TS + 31 * 60_000)
        places = {key: payload['seq'] for key, payload in ctx.keyed_events_for_run_id(
            ENTRY_FILL_ORDER_EVENT_KIND).items()}
        assert places["ent-B"] < places["ent-A"] < places["ent-D"]
        ctx.close()
