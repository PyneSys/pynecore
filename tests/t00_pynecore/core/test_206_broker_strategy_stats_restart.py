"""
The strategy statistics of a live run across restarts and server moves.

``strategy.netprofit``, ``strategy.closedtrades`` and the rest of the
``strategy.*`` namespace are kept by the live
:class:`~pynecore.core.broker.position.BrokerPosition`. The sync engine stores
them with every book write and the next process of the run restores them
before it books anything, so a restart, or the store copied to another host,
continues the statistics an uninterrupted run would have.
"""
import math
import shutil
import sqlite3
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pynecore import lib
from pynecore.core.broker.defaults import load_broker_defaults
from pynecore.core.broker.idempotency import KIND_ENTRY, KIND_EXIT_SL, KIND_EXIT_TP
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
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.strategy_state import BarIndexRemap, trade_to_payload
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.lib.strategy import Order, Trade, _order_type_close, _order_type_entry

SYMBOL = "BTCUSDT"
PLUGIN = "testbroker"
BAR_MS = 60_000
DAY_MS = 86_400_000
# A minute-aligned bar time of yesterday: within every retention window.
_T0 = (int(time.time() * 1000) - DAY_MS) // BAR_MS * BAR_MS


@pytest.fixture(autouse=True)
def _stub_script():
    prev = lib._script
    prev_time, prev_index = lib._time, lib.bar_index
    prev_opens = list(lib._bar_opens)
    lib._script = SimpleNamespace(initial_capital=100_000.0, pyramiding=3)
    try:
        yield
    finally:
        lib._script = prev
        lib._time, lib.bar_index = prev_time, prev_index
        lib._bar_opens[:] = prev_opens


@dataclass
class _Venue:
    """A venue whose plugin journals entries and bracket legs (derivatives: rows net)."""
    client_order_id_max_len = 30
    on_unexpected_cancel = "stop"
    journal: RunContext | None = None
    position: ExchangePosition | None = None
    open_orders: list[ExchangeOrder] = field(default_factory=list)

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

    async def execute_entry(self, envelope: DispatchEnvelope):
        intent = envelope.intent
        coid = envelope.client_order_id(KIND_ENTRY)
        order_id = f"ent-{intent.pine_id}"
        assert self.journal is not None
        self.journal.upsert_order(
            coid, symbol=SYMBOL, side=intent.side, qty=intent.qty, state="confirmed",
            intent_key=intent.intent_key, exchange_order_id=order_id,
            pine_entry_id=intent.pine_id, filled_qty=0.0, extras={"kind": "position"})
        return [self._order(order_id, coid, intent.side, intent.qty)]

    async def execute_exit(self, envelope: DispatchEnvelope):
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
        return []

    async def execute_cancel(self, envelope):
        return True

    async def cancel_broker_order_ref(self, ref):
        self.open_orders = [o for o in self.open_orders if o.id != ref]

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
    return RunIdentity(strategy_id="t206", symbol=SYMBOL, timeframe="1",
                       account_id="test-account", label=None)


@dataclass
class _Process:
    ctx: RunContext
    engine: OrderSyncEngine
    pos: BrokerPosition
    venue: _Venue


def _start(store: BrokerStore, venue_size: float,
           open_orders: list[ExchangeOrder] | None = None) -> _Process:
    """A process of the run starts, before any bar of it ran, and runs its startup reconcile."""
    lib._time = 0
    lib.bar_index = 0.0
    ctx = store.open_run(_identity(), script_source="src", script_path="t206.py")
    venue = _Venue(journal=ctx, position=_long(venue_size), open_orders=list(open_orders or []))
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=venue,  # type: ignore[arg-type]
        position=pos, symbol=SYMBOL, run_tag=ctx.run_tag, mintick=1.0, store_ctx=ctx,
    )
    engine.reconcile()
    return _Process(ctx, engine, pos, venue)


def _bar(t0: int, index: int) -> int:
    """Run bar ``index`` of the history that starts at ``t0``; returns its open time."""
    lib.bar_index = float(index)
    lib._time = t0 + index * BAR_MS
    lib._bar_opens[index % lib._BAR_OPENS_SIZE] = lib._time
    return lib._time


def _fill(side: str, qty: float, price: float, *, pine_id: str, order_id: str,
          fill_id: str | None,
          ts_ms: int, fee: float, leg: LegType = LegType.ENTRY,
          from_entry: str | None = None) -> OrderEvent:
    order = ExchangeOrder(
        id=order_id, symbol=SYMBOL, side=side, order_type=OrderType.MARKET, qty=qty,
        filled_qty=qty, remaining_qty=0.0, price=None, stop_price=None,
        average_fill_price=price, status=OrderStatus.FILLED, timestamp=ts_ms / 1000.0,
        fee=fee, fee_currency="USDT",
    )
    return OrderEvent(
        order=order, event_type='filled', fill_price=price, fill_qty=qty,
        timestamp=ts_ms / 1000.0, pine_id=pine_id, leg_type=leg, from_entry=from_entry,
        fee=fee, fill_id=fill_id, journal_fill_qty=qty,
    )


def _deliver(proc: _Process, event: OrderEvent) -> None:
    proc.engine.on_order_event(event)
    proc.engine._drain_events()  # type: ignore[attr-defined]


def _enter(proc: _Process, pine_id: str, price: float, t0: int, index: int) -> None:
    """Bar ``index``: the entry ``pine_id`` of 1 opens under a bracket and fills."""
    bar_ts = _bar(t0, index)
    proc.pos.entry_orders[pine_id] = Order(pine_id, 1.0, order_type=_order_type_entry)
    proc.pos.exit_orders[(f"{pine_id}-X", pine_id)] = Order(
        pine_id, -1.0, order_type=_order_type_close, exit_id=f"{pine_id}-X",
        limit=price + 5_000.0, stop=price - 5_000.0)
    proc.engine.sync(bar_ts)
    row = next(row for row in proc.ctx.iter_live_orders()
               if row.pine_entry_id == pine_id and row.from_entry is None)
    proc.ctx.set_filled(row.client_order_id, 1.0)
    proc.venue.position = _long(abs(proc.pos.size) + 1.0)
    _deliver(proc, _fill('buy', 1.0, price, pine_id=pine_id, order_id=f"ent-{pine_id}",
                         fill_id=f"f-{pine_id}", ts_ms=bar_ts + 1_000, fee=5.0))
    proc.engine.sync(bar_ts)


def _journal_take_profit(proc: _Process, entry_id: str) -> None:
    """The plugin journals ``<entry_id>-X``'s take-profit fill of 1; the entry row nets."""
    exit_id = f"{entry_id}-X"
    for row in list(proc.ctx.iter_live_orders()):
        if row.intent_key is not None and row.intent_key.startswith(
                f"{exit_id}{INTENT_KEY_SEP}"):
            if row.tp_level is not None:
                proc.ctx.set_filled(row.client_order_id, 1.0)
            proc.ctx.close_order(row.client_order_id)
        elif row.pine_entry_id == entry_id and row.from_entry is None:
            proc.ctx.close_order(row.client_order_id)
    proc.venue.open_orders = [o for o in proc.venue.open_orders
                              if not o.id.startswith(f"{exit_id}-")]
    size = proc.venue.position.size if proc.venue.position is not None else 0.0
    proc.venue.position = _long(size - 1.0)


def _take_profit_event(entry_id: str, price: float, ts_ms: int, *,
                       leg: LegType = LegType.TAKE_PROFIT) -> OrderEvent:
    exit_id = f"{entry_id}-X"
    return _fill('sell', 1.0, price, pine_id=exit_id, order_id=f"{exit_id}-tp",
                 fill_id=f"f-{exit_id}", ts_ms=ts_ms, fee=5.0, leg=leg, from_entry=entry_id)


def _take_profit(proc: _Process, entry_id: str, price: float, t0: int, index: int) -> None:
    """Bar ``index``: the entry's take-profit fills at ``price``."""
    bar_ts = _bar(t0, index)
    _journal_take_profit(proc, entry_id)
    _deliver(proc, _take_profit_event(entry_id, price, bar_ts + 2_000))
    proc.engine.sync(bar_ts)


def _mark(proc: _Process, price: float, t0: int, index: int) -> None:
    """Bar ``index``: the position is marked at ``price`` and the bar's sync runs."""
    bar_ts = _bar(t0, index)
    proc.pos.update_unrealized_pnl(price)
    proc.engine.sync(bar_ts)


def _warm_up(proc: _Process, t0: int, last_index: int) -> None:
    """The new process replays its warm-up history (bars 0..last_index of ``t0``)."""
    for index in range(last_index + 1):
        bar_ts = _bar(t0, index)
        if proc.pos.bar_remap_pending:
            proc.pos.observe_bar(index, bar_ts)


def _snapshot(pos: BrokerPosition, *, bars: bool = True) -> dict[str, Any]:
    """What the ``strategy.*`` namespace reads off the position, the bar anchors aside.

    :param bars: Whether the trades' bar indexes count (a run without bars has none).
    """
    payload, _key, _mark_key = pos.strategy_state_payload()
    skipped = {'bar_anchor'} if bars else {'bar_anchor', 'entry_bar_index', 'exit_bar_index'}

    def trades(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in record.items() if k not in skipped} for record in records]

    return {
        'aggregates': payload['aggregates'],
        'stats': payload['closed_trade_stats'],
        'open': trades(payload['open_trades']),
        'closed': trades([trade_to_payload(trade, None) for trade in pos.closed_trades]),
        'first_index': pos.closed_trades_count - len(pos.closed_trades),
        'size': pos.size,
    }


def _scenario(store_path: Path, *, restart_at: int | None, move_to: Path | None = None
              ) -> dict[str, Any]:
    """L1 opens, takes profit; L2 opens and is marked; then L2 takes profit.

    With ``restart_at`` the process stops after that many steps and a new one
    continues on the same store (or, with ``move_to``, on a copy of it at
    another path: a server move); the new process replays the same history in
    its warm-up first and marks its position at the current price, as the
    runner does on every live tick.
    """
    t0 = _T0
    steps = [
        lambda p: _enter(p, "L1", 50_000.0, t0, 1),
        lambda p: _take_profit(p, "L1", 51_000.0, t0, 3),
        lambda p: _enter(p, "L2", 50_500.0, t0, 4),
        lambda p: _mark(p, 49_800.0, t0, 5),
        lambda p: _take_profit(p, "L2", 52_000.0, t0, 7),
        lambda p: _mark(p, 52_000.0, t0, 8),
    ]
    store = BrokerStore(store_path, plugin_name=PLUGIN)
    proc = _start(store, 0.0)
    for i, step in enumerate(steps):
        if restart_at is not None and i == restart_at:
            resting = list(proc.venue.open_orders)
            size = proc.venue.position.size if proc.venue.position is not None else 0.0
            proc.ctx.close()
            if move_to is not None:
                store.close()
                shutil.copy(store_path, move_to)
                store = BrokerStore(move_to, plugin_name=PLUGIN)
            proc = _start(store, size, resting)
            _warm_up(proc, t0, [1, 3, 4, 5, 7, 8][i - 1])
            if i > 3:
                proc.pos.update_unrealized_pnl(49_800.0)
        step(proc)
    result = _snapshot(proc.pos)
    proc.ctx.close()
    store.close()
    return result


# --- Restart and server move -------------------------------------------------


@pytest.mark.parametrize("restart_at", [1, 2, 3, 4, 5])
def __test_a_restart_continues_the_statistics_of_an_uninterrupted_run__(
        tmp_path: Path, restart_at: int) -> None:
    """Stopped after any step and restarted, the run ends with the statistics it would have had.

    Net profit, the win / loss counters, the cumulative closed-trade
    statistics (bar lengths included), the closed trades and the open trades
    with their entry prices, times and commissions all equal those of the run
    that never stopped.
    """
    reference = _scenario(tmp_path / "ref.sqlite", restart_at=None)
    restarted = _scenario(tmp_path / "restart.sqlite", restart_at=restart_at)
    assert reference['aggregates']['closed_trades_count'] == 2
    assert reference['aggregates']['netprofit'] == pytest.approx(990.0 + 1_490.0)
    assert restarted == reference


@pytest.mark.parametrize("restart_at", [2, 4])
def __test_a_server_move_continues_the_statistics_of_an_uninterrupted_run__(
        tmp_path: Path, restart_at: int) -> None:
    """The store copied to another path and the run started there: the same statistics."""
    reference = _scenario(tmp_path / "ref.sqlite", restart_at=None)
    moved = _scenario(tmp_path / "old-host.sqlite", restart_at=restart_at,
                      move_to=tmp_path / "new-host.sqlite")
    assert moved == reference


def __test_the_restored_open_trade_keeps_its_own_entry__(tmp_path: Path) -> None:
    """The adoption takes the stored open trade, not the venue's average price."""
    t0 = _T0
    with BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN) as store:
        proc = _start(store, 0.0)
        _enter(proc, "L1", 50_500.0, t0, 2)
        resting = list(proc.venue.open_orders)
        proc.ctx.close()
        proc = _start(store, 1.0, resting)
        [trade] = proc.pos.open_trades
        assert (trade.entry_id, trade.size, trade.entry_price, trade.commission) == (
            "L1", 1.0, 50_500.0, 5.0)
        assert trade.entry_time == t0 + 2 * BAR_MS + 1_000
        # The warm-up history starts 2 bars later: the entry bar is bar 0 now.
        _warm_up(proc, t0 + 2 * BAR_MS, 3)
        assert trade.entry_bar_index == 0
        assert not proc.pos.bar_remap_pending
        proc.ctx.close()


# --- Crash between booking and persistence, journaled closes -----------------


@pytest.mark.parametrize("at_startup", [False, True], ids=["after_warmup", "at_startup"])
@pytest.mark.parametrize("leg", [LegType.TAKE_PROFIT, LegType.CLOSE])
def __test_a_journaled_close_books_its_profit_once__(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leg: LegType, at_startup: bool) -> None:
    """The close the dead process never stored is booked by its redelivery, exactly once.

    The first process books L1's take-profit but its book write fails (the
    process dies before the write landed), so the store holds L1 open and no
    receipt of the close. The restart finds the venue flat; the plugin
    delivers the close again with ``journaled_before_restart``: L1 closes at
    the fill's price with its fee, the position does not move, and a second
    delivery books nothing. A further restart keeps the one trade. Delivered
    at startup, before any bar of the new process ran, the close takes the
    bar its fill time falls in once the warm-up shows it, and its bar length
    reaches the statistics then.
    """
    t0 = _T0
    with BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN) as store:
        proc = _start(store, 0.0)
        _enter(proc, "L1", 50_000.0, t0, 1)

        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(proc.ctx, 'write_strategy_state', fail)
        _take_profit(proc, "L1", 51_000.0, t0, 2)
        assert proc.pos.closed_trades_count == 1
        assert proc.ctx.fill_receipt("f-L1-X") is None
        monkeypatch.undo()
        proc.ctx.close()

        proc = _start(store, 0.0)
        assert proc.pos.closed_trades_count == 0
        event = replace(_take_profit_event("L1", 51_000.0, t0 + 2 * BAR_MS + 2_000, leg=leg),
                        journaled_before_restart=True)
        if at_startup:
            _deliver(proc, event)
            assert proc.pos.closed_trades[0].exit_bar_index == -1
        # The warm-up history ends with the live bar 3, on which the close arrives.
        _warm_up(proc, t0, 3)
        _deliver(proc, event)
        _deliver(proc, event)
        proc.engine.sync(t0 + 3 * BAR_MS)
        pos = proc.pos
        assert pos.size == 0.0
        assert pos.closed_trades_count == 1 and pos.wintrades == 1
        [trade] = pos.closed_trades
        assert (trade.entry_price, trade.exit_price, trade.commission) == (
            50_000.0, 51_000.0, 10.0)
        assert trade.entry_bar_index == 1 and trade.exit_bar_index == 2
        assert pos.netprofit == pytest.approx(990.0)
        assert pos._closed_trade_stats.bar_length_count == 1  # type: ignore[attr-defined]
        assert proc.ctx.fill_delivered("f-L1-X")
        proc.ctx.close()

        proc = _start(store, 0.0)
        _deliver(proc, event)
        assert proc.pos.closed_trades_count == 1
        assert proc.pos.netprofit == pytest.approx(990.0)
        proc.ctx.close()


def __test_a_crash_before_the_book_write_loses_nothing_and_counts_nothing_twice__(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The statistics land with the book and the receipt, or not at all.

    The failed write leaves the stored statistics, the book and the receipts
    as they were before the fill; the retried write in the same process
    stores them together.
    """
    t0 = _T0
    with BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN) as store:
        proc = _start(store, 0.0)
        _enter(proc, "L1", 50_000.0, t0, 1)
        stored = proc.ctx.strategy_state()
        assert stored is not None and stored['aggregates']['closed_trades_count'] == 0

        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(proc.ctx, 'write_strategy_state', fail)
        _take_profit(proc, "L1", 51_000.0, t0, 2)
        stored = proc.ctx.strategy_state()
        assert stored is not None and stored['aggregates']['closed_trades_count'] == 0
        assert proc.ctx.fill_receipt("f-L1-X") is None
        monkeypatch.undo()
        proc.engine.sync(t0 + 3 * BAR_MS)
        stored = proc.ctx.strategy_state()
        assert stored is not None and stored['aggregates']['closed_trades_count'] == 1
        assert proc.ctx.fill_delivered("f-L1-X")
        assert [no for no, _payload in proc.ctx.strategy_closed_trades(
            from_trade_no=1, since_ms=0)] == [1]
        proc.ctx.close()


def _restart_over_an_unstored_take_profit(store: BrokerStore) -> tuple[_Process, int]:
    """L1 opens; its take-profit is journaled, but the process dies before the engine sees it.

    :return: The restarted process (venue flat, fill-report grace 0) and the
        base time.
    """
    t0 = _T0
    proc = _start(store, 0.0)
    _enter(proc, "L1", 50_000.0, t0, 1)
    _bar(t0, 2)
    _journal_take_profit(proc, "L1")
    proc.ctx.close()
    proc = _start(store, 0.0)
    proc.venue.fill_report_grace_s = 0.0  # type: ignore[attr-defined]
    _warm_up(proc, t0, 3)
    proc.engine.sync(t0 + 3 * BAR_MS)
    assert [trade.entry_id for trade in proc.pos.restored_held] == ["L1"]
    return proc, t0


def _settle(proc: _Process) -> None:
    """Two reconcile passes: the venue has held the same size for the (zero) grace."""
    proc.engine.reconcile()
    proc.engine.reconcile()


def __test_an_unflagged_close_the_venue_already_holds_closes_the_restored_trade_once__(
        tmp_path: Path) -> None:
    """The restart's venue is flat; the plugin delivers the take-profit as an ordinary fill.

    The book is flat, so the fill is held and settled against the venue, which
    already accounts for it: it closes the restored L1 with its own price and
    fee, once. Delivered again it books nothing, and a further restart keeps
    the one trade.
    """
    with BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN) as store:
        proc, t0 = _restart_over_an_unstored_take_profit(store)
        event = _take_profit_event("L1", 51_000.0, t0 + 2 * BAR_MS + 2_000)
        _deliver(proc, event)
        assert proc.pos.closed_trades_count == 0
        _settle(proc)
        pos = proc.pos
        assert pos.size == 0.0 and pos.closed_trades_count == 1 and pos.wintrades == 1
        [trade] = pos.closed_trades
        assert (trade.exit_price, trade.commission, trade.exit_bar_index) == (51_000.0, 10.0, 2)
        assert pos.netprofit == pytest.approx(990.0)
        proc.engine.sync(t0 + 4 * BAR_MS)
        assert proc.ctx.fill_delivered("f-L1-X")
        _deliver(proc, event)
        _settle(proc)
        assert pos.closed_trades_count == 1 and pos.netprofit == pytest.approx(990.0)
        proc.ctx.close()

        proc = _start(store, 0.0)
        assert proc.pos.closed_trades_count == 1
        assert proc.pos.netprofit == pytest.approx(990.0)
        assert list(proc.pos.restored_held) == []
        proc.ctx.close()


@pytest.mark.parametrize("restart_between", [False, True])
def __test_a_restored_trade_no_close_reports_is_closed_out_and_repriced_by_its_close__(
        tmp_path: Path, restart_between: bool) -> None:
    """No close of the restored L1 arrives within the grace: it is closed out at its entry price.

    The venue stays flat with nothing in flight: L1 closes at 50 000, its
    realized P&L its entry commission (-5), with a WARNING and a
    ``restored_trade_close_adopted`` audit. The take-profit the plugin then
    delivers (journaled before the restart), also after a further restart,
    re-prices that close to its own 51 000 and fee: one trade, a win of 990.
    """
    with BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN) as store:
        proc, t0 = _restart_over_an_unstored_take_profit(store)
        _settle(proc)
        pos = proc.pos
        assert list(pos.restored_held) == []
        assert pos.closed_trades_count == 1 and pos.losstrades == 1
        [trade] = pos.closed_trades
        assert trade.exit_price == 50_000.0 and pos.netprofit == pytest.approx(-5.0)
        audit = proc.ctx.latest_event_for_run_id(('restored_trade_close_adopted',))
        assert audit is not None and audit[1]['trades'][0]['entry_id'] == "L1"
        proc.engine.sync(t0 + 4 * BAR_MS)
        if restart_between:
            proc.ctx.close()
            proc = _start(store, 0.0)
            _warm_up(proc, t0, 4)
            pos = proc.pos
            assert pos.netprofit == pytest.approx(-5.0)
        event = replace(_take_profit_event("L1", 51_000.0, t0 + 2 * BAR_MS + 2_000),
                        journaled_before_restart=True)
        _deliver(proc, event)
        _deliver(proc, event)
        assert pos.closed_trades_count == 1
        assert (pos.wintrades, pos.losstrades) == (1, 0)
        [trade] = pos.closed_trades
        assert (trade.exit_price, trade.commission) == (51_000.0, 10.0)
        assert pos.netprofit == pytest.approx(990.0)
        assert pos._closed_trade_stats.winning_count == 1  # type: ignore[attr-defined]
        proc.engine.sync(t0 + 5 * BAR_MS)
        proc.ctx.close()

        proc = _start(store, 0.0)
        assert proc.pos.netprofit == pytest.approx(990.0)
        assert proc.pos.closed_trades_count == 1
        proc.ctx.close()


# --- The closed-trade window ---------------------------------------------------


def _close_trades(pos: BrokerPosition, count: int, *, first: int, exit_ms,
                  fill_ids: bool = True) -> None:
    """Book ``count`` round trips straight into the position (entry 100, exit 101 + i).

    :param fill_ids: Whether the fills carry ids (a fee target keeps their trades).
    """
    for i in range(first, first + count):
        ts = exit_ms(i)
        pos.record_fill(_fill('buy', 1.0, 100.0, pine_id=f"E{i}", order_id=f"e{i}",
                              fill_id=f"fe{i}" if fill_ids else None, ts_ms=ts - 1_000,
                              fee=0.0))
        pos.record_fill(_fill('sell', 1.0, 101.0 + (i % 7) - 3, pine_id=f"X{i}",
                              order_id=f"x{i}", fill_id=f"fx{i}" if fill_ids else None,
                              ts_ms=ts, fee=0.0, leg=LegType.CLOSE))


def __test_the_store_keeps_the_newest_9000_closed_trades_and_every_aggregate__(
        tmp_path: Path) -> None:
    """9005 closed trades: the oldest 5 rows go, the totals and statistics stay whole."""
    now = int(time.time() * 1000)
    with BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN) as store:
        proc = _start(store, 0.0)
        _close_trades(proc.pos, 5, first=1, exit_ms=lambda i: now - DAY_MS + i)
        proc.engine.sync(now)
        _close_trades(proc.pos, 9_000, first=6, exit_ms=lambda i: now - DAY_MS + i)
        proc.engine.sync(now + BAR_MS)
        numbers = [no for no, _payload in proc.ctx.strategy_closed_trades(
            from_trade_no=0, since_ms=0)]
        assert numbers == list(range(6, 9_006))
        before = _snapshot(proc.pos, bars=False)
        proc.ctx.close()

        proc = _start(store, 0.0)
        after = _snapshot(proc.pos, bars=False)
        assert after == before
        assert after['first_index'] == 5 and len(proc.pos.closed_trades) == 9_000
        assert proc.pos.closed_trades[0].entry_id == "E6"
        proc.ctx.close()


def __test_closed_trades_past_the_retention_go_and_the_aggregates_stay__(
        tmp_path: Path) -> None:
    """Three trades closed 200 days ago, one 40 days ago and one yesterday; retention 180 days.

    The restart lists the two recent trades as trades 4 and 5 of 5, folds the
    three old ones into the aggregates (no fee can reach them: their fills
    carry no id), and its next write drops their rows; every total still
    counts all five. A store whose retention is 30 days lists trades of 30
    days at most.
    """
    now = int(time.time() * 1000)
    path = tmp_path / "b.sqlite"

    def stored_numbers() -> list[int]:
        with sqlite3.connect(path) as conn:
            return [row[0] for row in conn.execute(
                "SELECT trade_no FROM strategy_closed_trades ORDER BY trade_no")]

    with BrokerStore(path, plugin_name=PLUGIN) as store:
        proc = _start(store, 0.0)
        _close_trades(proc.pos, 3, first=1, exit_ms=lambda i: now - 200 * DAY_MS + i,
                      fill_ids=False)
        _close_trades(proc.pos, 1, first=4, exit_ms=lambda i: now - 40 * DAY_MS,
                      fill_ids=False)
        _close_trades(proc.pos, 1, first=5, exit_ms=lambda i: now - DAY_MS, fill_ids=False)
        proc.engine.sync(now)
        before = _snapshot(proc.pos, bars=False)
        proc.ctx.close()
        store.cleanup_old_data()
        assert stored_numbers() == [1, 2, 3, 4, 5]

        proc = _start(store, 0.0)
        after = _snapshot(proc.pos, bars=False)
        assert after['aggregates'] == before['aggregates']
        assert after['stats'] == before['stats']
        assert after['closed'] == before['closed'][3:]
        assert after['first_index'] == 3
        proc.engine.sync(now + BAR_MS)
        assert stored_numbers() == [4, 5]
        proc.ctx.close()

    with BrokerStore(path, plugin_name=PLUGIN, retention_days=30) as store:
        proc = _start(store, 0.0)
        assert [trade.entry_id for trade in proc.pos.closed_trades] == ["E5"]
        assert proc.pos.closed_trades_count == 5
        assert _snapshot(proc.pos, bars=False)['aggregates'] == before['aggregates']
        proc.engine.sync(now + 2 * BAR_MS)
        assert stored_numbers() == [5]
        proc.ctx.close()


def _pyramid_run(path: Path, *, now: int, retention_days: int,
                 restart: bool) -> dict[str, Any]:
    """9005 entries, closed by two fills; then late fees of both close fills.

    The close fill ``X1`` closes trades 1..10 forty days ago, ``X2`` the other
    8995 yesterday. The newest 9000 trades are listed, so trades 1..5 are
    evicted by count, and with a 30-day retention a restart evicts 6..10 by
    age as well. Late fees of ``X1`` (its trades all evicted) and ``X2`` (its
    trades listed) then arrive, with a restart in between when ``restart``.
    """
    with BrokerStore(path, plugin_name=PLUGIN, retention_days=retention_days) as store:
        proc = _start(store, 0.0)
        pos = proc.pos
        for i in range(1, 9_006):
            pos.record_fill(_fill('buy', 1.0, 100.0 + i % 5, pine_id="L", order_id=f"e{i}",
                                  fill_id=f"fe{i}", ts_ms=now - 50 * DAY_MS + i, fee=0.01))
        pos.record_fill(_fill('sell', 10.0, 103.0, pine_id="X1", order_id="x1", fill_id="fx1",
                              ts_ms=now - 40 * DAY_MS, fee=0.5, leg=LegType.CLOSE))
        pos.record_fill(_fill('sell', 8_995.0, 101.0, pine_id="X2", order_id="x2",
                              fill_id="fx2", ts_ms=now - DAY_MS, fee=4.0, leg=LegType.CLOSE))
        proc.engine.sync(now)
        assert pos.closed_trades_count == 9_005 and len(pos.closed_trades) == 9_000
        if restart:
            proc.ctx.close()
            proc = _start(store, 0.0)
            pos = proc.pos
        assert pos.book_fill_fee("fx1", 30.0, entry_id="L", opening=False)
        assert pos.book_fill_fee("fx2", 20.0, entry_id="L", opening=False)
        proc.engine.sync(now + BAR_MS)
        result = _snapshot(pos, bars=False)
        proc.ctx.close()
    return {key: result[key] for key in ('aggregates', 'stats')}


@pytest.mark.parametrize("retention_days", [180, 30])
def __test_a_late_fee_on_an_evicted_or_listed_trade_keeps_the_statistics_exact__(
        tmp_path: Path, retention_days: int) -> None:
    """A late fee on trades evicted by count or by age, and on listed ones, after a restart.

    The aggregates and the cumulative closed-trade statistics (sums, counts,
    streaks, largest trades, bar lengths, monthly returns) end equal to those
    of the run that never stopped.
    """
    now = int(time.time() * 1000)
    reference = _pyramid_run(tmp_path / "ref.sqlite", now=now,
                             retention_days=retention_days, restart=False)
    restarted = _pyramid_run(tmp_path / "restart.sqlite", now=now,
                             retention_days=retention_days, restart=True)
    assert restarted == reference
    assert reference['aggregates']['closed_trades_count'] == 9_005


HOUR_MS = 3_600_000
_AGING_BARS = 36


def _aging_run(path: Path, monkeypatch: pytest.MonkeyPatch, *,
               restart_at: int | None) -> list[tuple[dict[str, Any], list[int]]]:
    """A one-day retention run of hourly bars, each closing one trade; the clock is the bars'.

    Every bar starts as the runner starts it (:meth:`BrokerPosition.observe_bar`),
    closes a trade and syncs. From bar 25 on, the trade of 25 bars ago is
    older than the retention. Trade 3's fills carry ids, so a late fee can
    still reach it: one arrives on bar 30, after it aged out. With
    ``restart_at`` the process stops before that bar and a new one replays
    the earlier bars as its warm-up.

    :return: Per bar, the ``strategy.*`` snapshot and the stored trade numbers.
    """
    t0 = (int(time.time() * 1000) - 2 * DAY_MS) // HOUR_MS * HOUR_MS
    clock = [float(t0) / 1000.0]
    monkeypatch.setattr(time, 'time', lambda: clock[0])

    def bar(index: int, offset_ms: int) -> int:
        bar_ts = t0 + index * HOUR_MS
        clock[0] = (bar_ts + offset_ms) / 1000.0
        lib.bar_index = float(index)
        lib._time = bar_ts
        lib._bar_opens[index % lib._BAR_OPENS_SIZE] = bar_ts
        return bar_ts

    def stored_numbers() -> list[int]:
        with sqlite3.connect(path) as conn:
            return [row[0] for row in conn.execute(
                "SELECT trade_no FROM strategy_closed_trades ORDER BY trade_no")]

    result: list[tuple[dict[str, Any], list[int]]] = []
    with BrokerStore(path, plugin_name=PLUGIN, retention_days=1) as store:
        proc = _start(store, 0.0)
        for index in range(_AGING_BARS):
            if index == restart_at:
                proc.ctx.close()
                proc = _start(store, 0.0)
                for replayed in range(index):
                    proc.pos.observe_bar(replayed, bar(replayed, 500))
            bar_ts = bar(index, 500)
            proc.pos.observe_bar(index, bar_ts)
            clock[0] = (bar_ts + 2_000) / 1000.0
            _close_trades(proc.pos, 1, first=index, exit_ms=lambda _i: bar_ts + 1_000,
                          fill_ids=index == 3)
            if index == 30:
                assert proc.pos.book_fill_fee("fx3", 7.0, entry_id="E3", opening=False)
            proc.engine.sync(bar_ts)
            result.append((_snapshot(proc.pos), stored_numbers()))
        proc.ctx.close()
    monkeypatch.undo()
    return result


@pytest.mark.parametrize("restart_at", [1, 20, 25, 26, 29, 31, 35])
def __test_closed_trades_age_out_on_the_same_bar_with_or_without_a_restart__(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restart_at: int) -> None:
    """A run living past the retention unlists and drops the aged trades on every bar.

    The uninterrupted run and one restarted before any bar show the same
    ``strategy.*`` values (listed trades, first index, totals, cumulative
    statistics, bar indexes) and store the same rows on every bar. An aged
    trade nothing can reach any more loses its row; the one a fee may still
    reach stays stored, unlisted, and its late fee moves every total exactly.
    """
    reference = _aging_run(tmp_path / "ref.sqlite", monkeypatch, restart_at=None)
    restarted = _aging_run(tmp_path / "restart.sqlite", monkeypatch, restart_at=restart_at)
    for index, (expected, actual) in enumerate(zip(reference, restarted)):
        assert actual == expected, f"bar {index}"
    # Trades 1..3 aged out on bars 25..27 and their rows went; trade 4, which
    # a fee can still reach, keeps its row and those after it (the folded
    # statistics are those of the oldest trades, in closing order).
    assert reference[24][1][0] == 1 and reference[25][1][0] == 2
    assert reference[27][1][0] == 4
    last, rows = reference[-1]
    assert last['aggregates']['closed_trades_count'] == _AGING_BARS
    assert last['first_index'] == _AGING_BARS - 25
    assert rows == list(range(4, _AGING_BARS + 1))


# --- Bar-index remap -----------------------------------------------------------


def _trade(entry_bar: int, entry_time: int, exit_bar: int = -1, exit_time: int = -1) -> Trade:
    trade = Trade(size=1.0, entry_id="L", entry_bar_index=entry_bar, entry_time=entry_time,
                  entry_price=100.0, commission=0.0)
    trade.exit_bar_index, trade.exit_time = exit_bar, exit_time
    return trade


def __test_anchored_trades_move_by_their_anchor_bar__() -> None:
    """A trade numbered against bar 500 at T moves by T's index in the new numbering."""
    t = 1_700_000_000_000
    remap = BarIndexRemap()
    closed = _trade(480, t - 20 * BAR_MS, 490, t - 10 * BAR_MS)
    opened = _trade(495, t - 5 * BAR_MS)
    remap.add_anchored(closed, (500, t))
    remap.add_anchored(opened, (500, t))
    for index in range(30):
        remap.observe(index, t - 20 * BAR_MS + index * BAR_MS)
    assert (closed.entry_bar_index, closed.exit_bar_index) == (0, 10)
    assert opened.entry_bar_index == 15
    assert remap.done and not remap.approximate


def __test_an_anchor_before_the_history_is_placed_by_the_bar_period__() -> None:
    """The anchor bar predates the new history: placed by the period, flagged approximate."""
    t = 1_700_000_000_000
    remap = BarIndexRemap()
    trade = _trade(100, t - 3 * BAR_MS)
    remap.add_anchored(trade, (103, t))
    remap.observe(0, t + 5 * BAR_MS)
    assert not remap.done
    remap.observe(1, t + 6 * BAR_MS)
    assert remap.done and remap.approximate
    assert trade.entry_bar_index == -8


def __test_a_timed_trade_takes_the_bar_its_fill_time_falls_in__() -> None:
    """Closed before the new process ran a bar: entry and exit by their own times."""
    t = 1_700_000_000_000
    remap = BarIndexRemap()
    trade = _trade(0, t + 2 * BAR_MS + 5, 0, t + 4 * BAR_MS + 59_000)
    trade.exit_bar_index = -1
    remap.add_timed(trade)
    owed: list[Trade] = []
    for index in range(7):
        owed += remap.observe(index, t + index * BAR_MS)
    assert (trade.entry_bar_index, trade.exit_bar_index) == (2, 4)
    assert owed == [trade] and remap.done


# --- Retention configuration and spot cleanup ---------------------------------


def __test_retention_days_is_read_from_brokers_toml_and_validated__(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    assert load_broker_defaults(config).retention_days == 180
    other = tmp_path / "config2"
    other.mkdir()
    (other / "brokers.toml").write_text("retention_days = 30\n")
    assert load_broker_defaults(other).retention_days == 30
    bad = tmp_path / "config3"
    bad.mkdir()
    (bad / "brokers.toml").write_text("retention_days = 0\n")
    with pytest.raises(ValueError, match="retention_days"):
        load_broker_defaults(bad)
    with pytest.raises(ValueError, match="retention_days"):
        BrokerStore(tmp_path / "x.sqlite", plugin_name=PLUGIN, retention_days=0)


def __test_the_purge_uses_the_store_retention__(tmp_path: Path) -> None:
    """A store with a 10-day retention drops a closed order of 20 days by default."""
    with BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN, retention_days=10) as store:
        ctx = store.open_run(_identity(), script_source="src", script_path="t206.py")
        ctx.upsert_order("c-1", symbol=SYMBOL, side="buy", qty=1.0, state="confirmed")
        ctx.close_order("c-1")
        old = int(time.time() * 1000) - 20 * DAY_MS
        with store.transaction() as conn:
            conn.execute("UPDATE orders SET closed_ts_ms = ?", (old,))
        assert store.cleanup_old_data() >= 1
        assert ctx.get_order("c-1") is None
        ctx.close()


def __test_spot_cleanup_drops_only_retired_rows_and_superseded_closed_epochs__(
        tmp_path: Path) -> None:
    """Retired rows and closed epochs a later epoch superseded past retention go; nothing else.

    Epoch 2 opened 200 days ago and epoch 3 10 days ago. The retired fill of
    300 days ago predates epoch 2, so it was retired at least 200 days ago:
    purged, and so is epoch 1, which epoch 2 superseded. The retired fill of
    100 days ago is within the retention, epoch 2 was superseded only 10 days
    ago, epoch 3 is open, the live fill (however old) is its inventory, and
    the asset lease stays.
    """
    now = int(time.time() * 1000)
    path = tmp_path / "b.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = store.open_run(_identity(), script_source="src", script_path="t206.py")

        def execution(fill_id: str, ts_ms: int) -> None:
            ctx.record_spot_execution(
                "acct", "BTC-USD", fill_id=fill_id, side='buy', base_delta='1',
                quote_delta='-100', price='100', fee_amount='0', fee_currency='USD',
                ts_ms=ts_ms)

        execution("old", now - 300 * DAY_MS)
        for _ in range(3):
            ctx.insert_spot_epoch(account_id="acct", base_asset="BTC", product_id="BTC-USD",
                                  foreign_baseline='0', cursor_scope='product',
                                  exec_cursor='c')
        ctx.set_spot_epoch_state("BTC-USD", 1, 'closed')
        ctx.set_spot_epoch_state("BTC-USD", 2, 'closed')
        ctx.retire_spot_executions("acct", "BTC-USD")
        execution("mid", now - 100 * DAY_MS)
        ctx.retire_spot_executions("acct", "BTC-USD")
        execution("live", now - 250 * DAY_MS)
        ctx.claim_spot_asset("acct", "BTC", "USD")
        with store.transaction() as conn:
            for seq, created in ((1, now - 400 * DAY_MS), (2, now - 200 * DAY_MS),
                                 (3, now - 10 * DAY_MS)):
                conn.execute("UPDATE spot_inventory_epoch SET created_ts_ms = ? "
                             "WHERE epoch_seq = ?", (created, seq))
        store.cleanup_old_data()
        with sqlite3.connect(path) as conn:
            assert sorted(row[0] for row in conn.execute(
                "SELECT fill_id FROM spot_executions")) == ["live", "mid"]
            assert [row[0] for row in conn.execute(
                "SELECT epoch_seq FROM spot_inventory_epoch ORDER BY epoch_seq")] == [2, 3]
            assert conn.execute("SELECT COUNT(*) FROM spot_asset_owner").fetchone()[0] == 1
        assert [row.fill_id for row in ctx.iter_spot_executions("acct", "BTC-USD")] == ["live"]
        ctx.close()


def __test_a_malformed_stored_state_starts_from_zero__(tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN) as store:
        ctx = store.open_run(_identity(), script_source="src", script_path="t206.py")
        ctx.write_strategy_state({'version': 99}, {}, keep_from_trade_no=0)
        ctx.close()
        proc = _start(store, 0.0)
        assert proc.pos.closed_trades_count == 0 and proc.pos.netprofit == 0.0
        assert not math.isnan(proc.pos.max_drawdown)
        proc.ctx.close()
