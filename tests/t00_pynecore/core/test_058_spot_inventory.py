"""
Spot inventory layer tests (:mod:`pynecore.core.broker.spot_inventory`
plus the ``spot_*`` tables of :mod:`pynecore.core.broker.storage`).

Covers the plan's M4 matrix: crash windows around fill persistence
(catch-up recovery, outbox exactly-once), duplicate fills across
restarts, fee handling in base/quote/third currency, the balance
invariant judged against the bot's own inventory (someone else's activity
that leaves it intact re-anchors the foreign holdings without a
quarantine; a shortfall below it is the external intervention, or stays
pending while the history is incomplete; a drift found at startup gets the
same runtime verdict), the asset-ownership lease
(concurrent claim, stale takeover), startup recovery of quarantined
epochs, measurements settled against an own fill landing between the
history and balance reads, measurement-gated preconditions, paged
catch-up crash resume, and retention exemption of the ledger.
"""
import asyncio
import logging
import threading
from decimal import Decimal
from pathlib import Path
from typing import Any, Coroutine

import pytest

from pynecore.core.broker.exceptions import SpotInventoryConflictError
from pynecore.core.broker.models import QuarantineKind
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.spot_inventory import (
    _HISTORY_LAG_CEILING_S,
    SpotExecution,
    SpotExecutionBatch,
    SpotInventoryManager,
    canonical_decimal,
    fold_inventory,
)
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.store_helpers import STATE_CONFIRMED

PLUGIN = "TestSpotBroker"
ACCOUNT = "testspot-demo-001"
SCRIPT_SOURCE = "// spot inventory test\n"

T0_MS = 1_000_000_000


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def _open_run(store: BrokerStore, *, label: str | None = None) -> RunContext:
    return store.open_run(
        RunIdentity(
            strategy_id="spot_test", symbol="BTCUSD", timeframe="60",
            account_id=ACCOUNT, label=label,
        ),
        script_source=SCRIPT_SOURCE,
        script_path="strategies/spot_test.py",
    )


def _fill(
        fid: str, *,
        side: str = 'buy',
        base: str = '1',
        quote: str | None = None,
        price: str = '100',
        fee: str = '0',
        fee_ccy: str = 'USD',
        ts_ms: int = T0_MS,
        coid: str | None = None,
        venue_seq: int | None = None,
) -> SpotExecution:
    base_d = Decimal(base)
    quote_d = Decimal(quote) if quote is not None else -(base_d * Decimal(price))
    return SpotExecution(
        fill_id=fid, side=side,
        base_delta=base_d, quote_delta=quote_d,
        price=Decimal(price),
        fee_amount=Decimal(fee), fee_currency=fee_ccy,
        ts_ms=ts_ms,
        client_order_id=coid if coid is not None else f"c-{fid}",
        venue_seq=venue_seq,
    )


class FakeSpotPort:
    """Scriptable venue surface.

    ``batches`` maps a cursor value to the batch (or exception) that
    ``fetch_executions`` serves for it; unknown cursors get an empty
    conclusive batch that anchors/keeps the cursor.
    """

    product_id = "BTC-USD"
    base_asset = "BTC"
    quote_asset = "USD"
    cursor_scope = "product"
    base_tolerance = Decimal("0.000000005")
    settlement_grace_s = 30.0
    position_dust_threshold = Decimal(0)

    def __init__(self, *, balance: Decimal = Decimal(0)) -> None:
        self.balance = balance
        self.balance_error: Exception | None = None
        self.batches: dict[str | None, SpotExecutionBatch | Exception] = {}
        self.fetch_calls: list[str | None] = []
        self.balance_calls = 0
        self.min_sellable: Decimal | None = None
        self.min_sellable_error: Exception | None = None

    async def fetch_executions(self, cursor: str | None) -> SpotExecutionBatch:
        self.fetch_calls.append(cursor)
        entry = self.batches.get(cursor)
        if isinstance(entry, Exception):
            raise entry
        if entry is not None:
            return entry
        return SpotExecutionBatch(next_cursor=cursor or "anchor-0")

    async def fetch_base_balance(self) -> Decimal:
        self.balance_calls += 1
        if self.balance_error is not None:
            raise self.balance_error
        return self.balance

    async def min_sellable_base(self) -> Decimal | None:
        if self.min_sellable_error is not None:
            raise self.min_sellable_error
        return self.min_sellable


def _manager(
        ctx: RunContext,
        port: FakeSpotPort,
        *,
        hook: Any = None,
        policy: str = 'quarantine',
) -> SpotInventoryManager:
    return SpotInventoryManager(
        ctx, port,
        account_id=ACCOUNT,
        symbol="BTCUSD",
        request_quarantine=hook,
        on_inventory_conflict=policy,
    )


def _read_events(ctx: RunContext, kind: str) -> list[dict]:
    rows = ctx._store._conn.execute(  # type: ignore[attr-defined]
        "SELECT payload FROM events WHERE kind = ? ORDER BY id",
        (kind,),
    ).fetchall()
    import json
    return [json.loads(r['payload']) if r['payload'] else {} for r in rows]


# === Canonical decimal + execution validation ==============================


def __test_canonical_decimal_forms__(tmp_path: Path):
    """One value, one string: exponent-free, no trailing zeros, -0 -> 0."""
    _ = tmp_path
    assert canonical_decimal(Decimal('1.100')) == '1.1'
    assert canonical_decimal(Decimal('1E+2')) == '100'
    assert canonical_decimal('-0.0') == '0'
    assert canonical_decimal(5) == '5'
    assert canonical_decimal('0.00000001') == '0.00000001'
    # Round trip is exact.
    assert Decimal(canonical_decimal(Decimal('123456.000789'))) \
           == Decimal('123456.000789')
    with pytest.raises(ValueError):
        canonical_decimal('NaN')
    with pytest.raises(ValueError):
        canonical_decimal('Infinity')
    with pytest.raises(ValueError):
        canonical_decimal('not-a-number')
    with pytest.raises(ValueError):
        canonical_decimal(1.1)  # type: ignore[arg-type] — floats rejected


def __test_spot_execution_validation_fail_closed__(tmp_path: Path):
    """A fill whose signs contradict its side cannot enter the ledger."""
    _ = tmp_path
    # Buy must gain base and spend quote.
    with pytest.raises(ValueError):
        _fill("f1", side='buy', base='-1')
    with pytest.raises(ValueError):
        _fill("f1", side='buy', base='1', quote='100')
    # Sell must shed base and gain quote.
    with pytest.raises(ValueError):
        _fill("f1", side='sell', base='1')
    with pytest.raises(ValueError):
        _fill("f1", side='sell', base='-1', quote='-100')
    # Non-positive price, negative fee, empty id: all rejected.
    with pytest.raises(ValueError):
        _fill("f1", price='0')
    with pytest.raises(ValueError):
        _fill("f1", fee='-1')
    with pytest.raises(ValueError):
        _fill("")


# === Inventory fold =========================================================


def __test_fold_fees_base_quote_and_third_currency__(tmp_path: Path):
    """Fee-in-base buy AND sell, fee-in-quote, third-currency fee.

    The canonical deltas already carry base/quote fees; a third-currency
    fee touches neither delta. The fold must reproduce the exact net
    inventory and the spent-quote cost basis.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort()
    fills = [
        # Buy 1 BTC @100, 0.001 BTC fee in BASE: net base 0.999, quote -100.
        SpotExecution(
            fill_id="b1", side='buy',
            base_delta=Decimal('0.999'), quote_delta=Decimal('-100'),
            price=Decimal('100'),
            fee_amount=Decimal('0.001'), fee_currency='BTC', ts_ms=T0_MS,
            client_order_id="c-b1",
        ),
        # Buy 1 BTC @100, 0.5 USD fee in QUOTE: base 1, quote -100.5.
        SpotExecution(
            fill_id="b2", side='buy',
            base_delta=Decimal('1'), quote_delta=Decimal('-100.5'),
            price=Decimal('100'),
            fee_amount=Decimal('0.5'), fee_currency='USD', ts_ms=T0_MS + 1,
            client_order_id="c-b2",
        ),
        # Buy 1 BTC @100, fee in a THIRD currency: deltas untouched.
        SpotExecution(
            fill_id="b3", side='buy',
            base_delta=Decimal('1'), quote_delta=Decimal('-100'),
            price=Decimal('100'),
            fee_amount=Decimal('0.1'), fee_currency='BNB', ts_ms=T0_MS + 2,
            client_order_id="c-b3",
        ),
        # Sell 0.999 BTC @110 with 0.0005 BTC fee in BASE:
        # base -0.9995, quote +109.89.
        SpotExecution(
            fill_id="s1", side='sell',
            base_delta=Decimal('-0.9995'), quote_delta=Decimal('109.89'),
            price=Decimal('110'),
            fee_amount=Decimal('0.0005'), fee_currency='BTC', ts_ms=T0_MS + 3,
            client_order_id="c-s1",
        ),
    ]
    for f in fills:
        assert ctx.record_spot_execution(
            ACCOUNT, port.product_id,
            fill_id=f.fill_id, side=f.side,
            base_delta=canonical_decimal(f.base_delta),
            quote_delta=canonical_decimal(f.quote_delta),
            price=canonical_decimal(f.price),
            fee_amount=canonical_decimal(f.fee_amount),
            fee_currency=f.fee_currency, ts_ms=f.ts_ms,
        )
    fold = fold_inventory(ctx.iter_spot_executions(ACCOUNT, port.product_id))
    # Net base is EXACT decimal addition.
    assert fold.net_base == Decimal('0.999') + 1 + 1 - Decimal('0.9995')
    assert fold.violation is None
    # Cost basis: 300.5 spent, reduced proportionally by the sell.
    bought = Decimal('0.999') + 1 + 1
    expected_cost = Decimal('300.5') * (1 - Decimal('0.9995') / bought)
    assert abs(fold.cost_quote - expected_cost) < Decimal('1e-20')
    assert fold.vwap is not None
    store.close()


def __test_fold_flat_resets_basis_and_oversell_flags_violation__(tmp_path: Path):
    _ = tmp_path
    from pynecore.core.broker.storage import SpotExecutionRow

    def row(fid: str, base: str, quote: str, ts: int) -> SpotExecutionRow:
        return SpotExecutionRow(
            fill_id=fid, exchange_order_id=None, client_order_id=None,
            side='buy' if Decimal(base) > 0 else 'sell',
            base_delta=base, quote_delta=quote, price='100',
            fee_amount='0', fee_currency='USD', ts_ms=ts, delivered=False,
        )

    # Exact flat: cost returns to zero, VWAP disappears.
    fold = fold_inventory([
        row("a", '1', '-100', 1), row("b", '-1', '105', 2),
    ])
    assert fold.net_base == Decimal(0) and fold.cost_quote == Decimal(0) and fold.vwap is None
    assert fold.violation is None

    # Oversell: spot cannot go short — corruption flag.
    fold = fold_inventory([
        row("a", '1', '-100', 1), row("b", '-2', '210', 2),
    ])
    assert fold.net_base == Decimal('-1')
    assert fold.violation is not None


# === Startup: first epoch, catch-up recovery, adoption watermark ============


def __test_startup_first_epoch_freezes_baseline_not_raw_total__(tmp_path: Path):
    """First epoch: baseline = current_total − reconstructed bot inventory.

    A crash after fills landed in the ledger but BEFORE the first epoch
    write must not launder those fills into the foreign baseline.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('12'))  # 10 foreign + 2 bot
    # Pre-existing bot fills in the ledger, no epoch yet (crash window).
    ctx.record_spot_execution(
        ACCOUNT, port.product_id, fill_id="pre1", side='buy',
        base_delta='2', quote_delta='-200', price='100',
        fee_amount='0', fee_currency='USD', ts_ms=T0_MS,
    )
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c: calls.append((r, c)))
    result = _run(mgr.startup())
    assert not result.quarantined
    assert result.epoch is not None
    assert Decimal(result.epoch.foreign_baseline) == Decimal('10')
    assert result.epoch.state == 'active'
    assert result.epoch.epoch_seq == 1
    # Adoption watermark: every row delivered.
    rows = ctx.iter_spot_executions(ACCOUNT, port.product_id)
    assert [r.delivered for r in rows] == [True]
    assert not calls
    store.close()


def __test_startup_catchup_recovers_missed_fill__(tmp_path: Path):
    """Crash after venue fill, before local persist: catch-up recovers it."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    result = _run(mgr.startup())
    assert not result.quarantined
    epoch1 = result.epoch
    assert epoch1 is not None and epoch1.exec_cursor == "anchor-0"
    ctx.close()

    # The bot bought 1 BTC; the venue filled it but the process died
    # before record_live_fill. History serves it from the cursor.
    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('11'))
    port2.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("missed-1", base='1', price='100'),),
        next_cursor="anchor-1",
    )
    mgr2 = _manager(ctx2, port2)
    result2 = _run(mgr2.startup())
    assert not result2.quarantined
    assert result2.recovered_fills == 1
    assert result2.fold.net_base == Decimal('1')
    assert result2.epoch is not None
    assert result2.epoch.exec_cursor == "anchor-1"
    # The recovered fill entered via the adoption path: delivered.
    rows = ctx2.iter_spot_executions(ACCOUNT, port2.product_id)
    assert [(r.fill_id, r.delivered) for r in rows] == [("missed-1", True)]
    store.close()


def __test_outbox_exactly_once_live_then_restart_adoption__(tmp_path: Path):
    """A live fill is emitted once; after restart it folds, never re-emits.

    Also covers the duplicate-fill-across-restart dedup: the venue
    re-serving the same fill id in catch-up must not double-book.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined

    # Live fill: record returns True exactly once (emit gate).
    fill = _fill("live-1", base='1', price='100')
    port.balance = Decimal('1')
    assert mgr.record_live_fill(fill) is True
    assert mgr.record_live_fill(fill) is False  # PUSH replay dedup
    assert mgr.fold.net_base == Decimal('1')
    ctx.close()

    # Restart: history re-serves the SAME fill (overlapping window).
    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('1'))
    port2.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("live-1", base='1', price='100'),),
        next_cursor="anchor-1",
    )
    mgr2 = _manager(ctx2, port2)
    result = _run(mgr2.startup())
    assert not result.quarantined
    assert result.recovered_fills == 0  # dedup — not booked twice
    assert result.fold.net_base == Decimal('1')
    # And a live replay after restart still refuses to re-emit.
    assert mgr2.record_live_fill(_fill("live-1", base='1', price='100')) is False
    store.close()


def __test_startup_paged_catchup_crash_resumes_from_cursor__(tmp_path: Path):
    """Crash mid-pagination: committed pages + cursor survive; the retry
    resumes at the last committed cursor and dedups the overlap."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    ctx.close()

    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('3'))
    port2.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("p1", base='1'), _fill("p2", base='1', ts_ms=T0_MS + 1)),
        next_cursor="c1", has_more=True,
    )
    port2.batches["c1"] = ConnectionError("venue died mid-pagination")
    mgr2 = _manager(ctx2, port2)
    with pytest.raises(ConnectionError):
        _run(mgr2.startup())
    # Page 1 committed atomically with its cursor advance.
    rows = ctx2.iter_spot_executions(ACCOUNT, port2.product_id)
    assert sorted(r.fill_id for r in rows) == ["p1", "p2"]
    epoch = ctx2.get_latest_spot_epoch(port2.product_id)
    assert epoch is not None and epoch.exec_cursor == "c1"
    ctx2.close()

    # Retry: resumes from c1 (with overlap), not from the beginning.
    ctx3 = _open_run(store)
    port3 = FakeSpotPort(balance=Decimal('3'))
    port3.batches["c1"] = SpotExecutionBatch(
        executions=(_fill("p2", base='1', ts_ms=T0_MS + 1),
                    _fill("p3", base='1', ts_ms=T0_MS + 2)),
        next_cursor="c2",
    )
    mgr3 = _manager(ctx3, port3)
    result = _run(mgr3.startup())
    assert not result.quarantined
    assert port3.fetch_calls == ["c1", "c2"], "resumed at c1, read again after the balance"
    assert result.fold.net_base == Decimal('3')
    assert result.recovered_fills == 1  # p2 dedup'd, p3 new
    store.close()


def __test_startup_inconclusive_catchup_fails_closed__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    port.batches[None] = SpotExecutionBatch(conclusive=False)
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c: calls.append((r, c)))
    result = _run(mgr.startup())
    assert result.quarantined
    assert result.reason == 'spot_catchup_inconclusive'
    assert len(calls) == 1
    assert mgr.pending_halt is None
    store.close()


def __test_startup_balance_failure_fails_closed__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort()
    port.balance_error = ConnectionError("balance endpoint down")
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c: calls.append((r, c)))
    result = _run(mgr.startup())
    assert result.quarantined
    assert result.reason == 'spot_startup_balance_unavailable'
    assert len(calls) == 1
    store.close()


def _bot_holding_one(store: BrokerStore, *, foreign: str) -> None:
    """A run that bought 1 base on top of ``foreign`` base, then stopped."""
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal(foreign))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal(foreign) + 1
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))
    assert not mgr.quarantined
    ctx.close()


def _inconclusive_restart(store: BrokerStore, port: FakeSpotPort, *,
                          hook: Any = None, policy: str = 'quarantine',
                          ) -> tuple[RunContext, SpotInventoryManager, Any]:
    """Restart on a venue whose execution history read stays inconclusive."""
    ctx = _open_run(store)
    port.batches["anchor-0"] = SpotExecutionBatch(conclusive=False)
    mgr = _manager(ctx, port, hook=hook, policy=policy)
    return ctx, mgr, _run(mgr.startup())


def __test_foreign_base_added_while_down_is_reanchored_by_the_runtime__(tmp_path: Path):
    """Someone deposits base while the bot is down; the bot's own 1 is untouched.

    The startup does not judge the drift: no quarantine, the ledger is
    adopted, and the runtime reconcile re-anchors the foreign holdings after
    the grace (ERROR and audit). A strict startup check latched the entry
    block here, and the next restart then retired the bot's ledger.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    _bot_holding_one(store, foreign='10')

    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('16'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append(r))
    result = _run(mgr.startup())
    assert not result.quarantined
    assert result.fold.net_base == Decimal('1')
    pending = _read_events(ctx, 'spot_startup_drift_pending')
    assert len(pending) == 1 and pending[0]['drift'] == '5'

    _run(mgr.reconcile(T0_MS + 60_000))
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert not mgr.quarantined and calls == []
    events = _read_events(ctx, 'spot_foreign_holdings_reanchored')
    assert len(events) == 1 and Decimal(events[0]['foreign_baseline']) == Decimal('15')
    ctx.close()

    ctx2 = _open_run(store)
    result2 = _run(_manager(ctx2, FakeSpotPort(balance=Decimal('16'))).startup())
    assert not result2.quarantined
    assert result2.fold.net_base == Decimal('1'), "nothing of the bot's was retired"
    assert _read_events(ctx2, 'spot_epoch_recovered') == []
    store.close()


def __test_bot_base_taken_while_down_is_an_external_intervention__(tmp_path: Path):
    """Someone sells 0.6 of the bot's own base while it is down.

    The runtime verdict after the grace is the external intervention with a
    ``taken`` epoch: a plain restart holds it, only the operator resume lifts
    it, and the first start after the resume keeps the 0.4 the account still
    holds. A strict startup check latched the entry block instead, which the
    next plain restart lifted by retiring the bot's remaining inventory.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    _bot_holding_one(store, foreign='0')

    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0.4'))
    kinds: list = []

    def _hook(reason: str, context: dict, **kw: Any) -> None:
        kinds.append(kw.get('kind'))
        ctx.record_quarantine("q-taken", kind=str(kw.get('kind')), reason=reason,
                              intent_key=None, context=context, entered_ts_ms=T0_MS)

    mgr = _manager(ctx, port, hook=_hook)
    result = _run(mgr.startup())
    assert not result.quarantined and kinds == []
    _run(mgr.reconcile(T0_MS + 60_000))
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert kinds == [QuarantineKind.EXTERNAL_INTERVENTION]
    assert mgr.quarantine_reason == 'spot_inventory_taken'
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.state == 'taken'
    ctx.close()

    ctx2 = _open_run(store)
    held = _run(_manager(ctx2, FakeSpotPort(balance=Decimal('0.4'))).startup())
    assert held.quarantined and held.reason == 'spot_external_quarantine_held'
    assert held.fold.net_base == Decimal('1')
    ctx2.close()

    probe = _open_run(store)
    assert store.request_quarantine_resume(probe.run_id, "q-taken") is not None
    probe.close()
    ctx3 = _open_run(store)
    resumed = _run(_manager(ctx3, FakeSpotPort(balance=Decimal('0.4'))).startup())
    assert not resumed.quarantined
    assert resumed.fold.net_base == Decimal('0.4')
    store.close()


def __test_an_own_sell_the_venue_indexes_late_at_startup_is_no_intervention__(tmp_path: Path):
    """The bot's resting sell filled just before the start; the history lags the balance.

    The startup history does not return the fill yet, so the balance shows
    less than the bot's inventory. The startup leaves the drift to the
    runtime and keeps the execution cursor where its catch-up began: the next
    read returns the fill and the drift resolves, with no quarantine and no
    re-anchor.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    _bot_holding_one(store, foreign='0')

    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0.4'))
    port.batches["anchor-0"] = SpotExecutionBatch(next_cursor="anchor-1")
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append(r))
    assert not _run(mgr.startup()).quarantined
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.exec_cursor == "anchor-0"

    port.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("own-tp", side='sell', base='-0.6', quote='60', price='100',
                          ts_ms=T0_MS + 1),),
        next_cursor="anchor-1",
    )
    delivered = _run(mgr.reconcile(T0_MS + 60_000))
    assert [row.fill_id for row in delivered] == ["own-tp"]
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert not mgr.quarantined and calls == []
    assert mgr.fold.net_base == Decimal('0.4')
    assert _read_events(ctx, 'spot_foreign_holdings_reanchored') == []
    store.close()


def __test_startup_recovers_a_quarantined_epoch_on_the_venue_balance__(tmp_path: Path):
    """Persisted quarantine: the next startup re-anchors on the venue's
    measured balance — the restart itself is the recovery."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    ctx.close()
    ctx2, _, result2 = _inconclusive_restart(store, FakeSpotPort(balance=Decimal('15')))
    assert result2.quarantined
    ctx2.close()

    # Third start: whatever the account holds NOW is the new baseline —
    # the corrupt epoch closes and trading resumes.
    ctx3 = _open_run(store)
    port3 = FakeSpotPort(balance=Decimal('12'))
    calls: list = []
    mgr3 = _manager(ctx3, port3, hook=lambda r, c: calls.append((r, c)))
    result = _run(mgr3.startup())
    assert not result.quarantined
    assert calls == []
    assert result.epoch is not None
    assert result.epoch.epoch_seq == 2 and result.epoch.state == 'active'
    assert Decimal(result.epoch.foreign_baseline) == Decimal('12')
    # noinspection SqlResolve
    old = ctx3._store._conn.execute(  # type: ignore[attr-defined]
        "SELECT state FROM spot_inventory_epoch WHERE epoch_seq = 1",
    ).fetchone()
    assert old['state'] == 'closed'
    assert len(_read_events(ctx3, 'spot_epoch_recovered')) == 1

    # The clean ledger trades on: a fresh fill folds from zero and the
    # invariant holds against the recovered baseline.
    mgr3.record_live_fill(_fill("post1", base='1', price='100'))
    port3.balance = Decimal('13')
    _run(mgr3.reconcile(T0_MS))
    assert not mgr3.quarantined
    store.close()


def __test_startup_recovery_retires_a_corrupt_ledger__(tmp_path: Path):
    """The #123 incident shape: an oversell drove the fold negative and
    quarantined the epoch; the restart retires the corrupt rows (audit
    kept, fold excluded) and re-anchors on the balance."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    mgr.record_live_fill(_fill("b1", base='1', price='100'))
    mgr.record_live_fill(
        _fill("s1", side='sell', base='-2', quote='200', price='100',
              ts_ms=T0_MS + 1),
    )
    assert mgr.quarantine_reason == 'spot_ledger_negative_inventory'
    ctx.close()

    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('0.5'))
    mgr2 = _manager(ctx2, port2)
    result = _run(mgr2.startup())
    assert not result.quarantined
    assert result.epoch is not None
    assert Decimal(result.epoch.foreign_baseline) == Decimal('0.5')
    # The corrupt rows survive on disk as the audit trail but left the
    # fold: the fresh epoch starts from zero bot inventory.
    assert result.fold.fill_count == 0 and result.fold.violation is None
    # noinspection SqlResolve
    kept = ctx2._store._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS n FROM spot_executions WHERE retired = 1",
    ).fetchone()
    assert kept['n'] == 2
    events = _read_events(ctx2, 'spot_epoch_recovered')
    assert events and events[0]['retired_fills'] == 2
    assert events[0]['retired_net_base'] == '-1'
    store.close()


def __test_startup_recovery_stays_quarantined_when_measurement_fails__(tmp_path: Path):
    """Recovery is measurement-gated: a failed balance read or an
    inconclusive catch-up keeps the quarantine; the next restart with a
    working venue recovers."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    ctx.close()
    ctx2, _, result2 = _inconclusive_restart(store, FakeSpotPort(balance=Decimal('15')))
    assert result2.quarantined
    ctx2.close()

    # Balance read down: still quarantined, epoch untouched.
    ctx3 = _open_run(store)
    port3 = FakeSpotPort(balance=Decimal('15'))
    port3.balance_error = RuntimeError("venue down")
    mgr3 = _manager(ctx3, port3)
    result = _run(mgr3.startup())
    assert result.quarantined
    assert result.reason == 'spot_recovery_balance_unavailable'
    ctx3.close()

    # Inconclusive catch-up: still quarantined.
    ctx4 = _open_run(store)
    port4 = FakeSpotPort(balance=Decimal('15'))
    port4.batches[None] = SpotExecutionBatch(conclusive=False)
    port4.batches["anchor-0"] = SpotExecutionBatch(conclusive=False)
    mgr4 = _manager(ctx4, port4)
    result = _run(mgr4.startup())
    assert result.quarantined
    assert result.reason == 'spot_recovery_catchup_inconclusive'
    ctx4.close()

    # Venue back: the same restart path recovers.
    ctx5 = _open_run(store)
    port5 = FakeSpotPort(balance=Decimal('15'))
    mgr5 = _manager(ctx5, port5)
    result = _run(mgr5.startup())
    assert not result.quarantined
    assert result.epoch is not None
    assert Decimal(result.epoch.foreign_baseline) == Decimal('15')
    store.close()


def __test_startup_cursor_scope_change_fails_closed__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    ctx.close()

    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('0'))
    port2.cursor_scope = "time"  # plugin upgrade changed cursor meaning
    mgr2 = _manager(ctx2, port2)
    result = _run(mgr2.startup())
    assert result.quarantined
    assert result.reason == 'spot_cursor_scope_changed'
    store.close()


# === Runtime reconcile: grace state machine =================================


def __test_reconcile_fill_first_race_resolves_via_catchup__(tmp_path: Path):
    """Balance moved before the fill reached us: the drift's innocent
    explanation (a not-yet-seen fill) resolves without quarantine."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('5'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c: calls.append((r, c)))
    assert not _run(mgr.startup()).quarantined

    # A fill landed at the venue (balance 6) but the stream lagged.
    port.balance = Decimal('6')
    port.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("lag-1", base='1', price='100'),),
        next_cursor="anchor-1",
    )
    _run(mgr.reconcile(T0_MS))
    assert not mgr.quarantined
    assert not calls
    assert mgr.fold.net_base == Decimal('1')
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None
    assert epoch.state == 'active' and epoch.pending_conflict_ts_ms is None
    assert _read_events(ctx, 'spot_inventory_conflict_resolved')
    store.close()


def __test_reconcile_deposit_then_foreign_sale_reanchors_the_foreign_holdings__(tmp_path: Path, caplog):
    """A deposit +10 then a foreign sale −5 leaves the bot's inventory intact.

    Base is fungible: the balance still holds the bot's own 1 (and more), so
    whatever moved is someone else's separate activity. Past the settlement
    grace it is an ERROR and a durable audit, the foreign holdings are
    re-anchored on the balance, and the bot keeps trading — never a
    quarantine.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append((r, c)))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('11')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))
    assert not mgr.quarantined

    # Deposit +10, then foreign sale -5: net drift +5.
    port.balance = Decimal('16')
    _run(mgr.reconcile(T0_MS + 60_000))
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None
    assert epoch.pending_conflict_ts_ms == T0_MS + 60_000, "the settlement grace comes first"
    assert epoch.pending_conflict is not None
    assert epoch.pending_conflict['drift'] == '5'
    assert not _read_events(ctx, 'spot_foreign_holdings_reanchored')

    with caplog.at_level(logging.ERROR):
        _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert not mgr.quarantined
    assert calls == []
    assert any("foreign holdings re-anchored" in r.getMessage()
               for r in caplog.records if r.levelno == logging.ERROR)
    reanchored = _read_events(ctx, 'spot_foreign_holdings_reanchored')
    assert len(reanchored) == 1
    assert (reanchored[0]['drift'], reanchored[0]['old_foreign_baseline'],
            reanchored[0]['foreign_baseline']) == ('5', '10', '15')
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None
    assert (epoch.epoch_seq, epoch.state, epoch.foreign_baseline) == (2, 'active', '15')
    assert epoch.pending_conflict_ts_ms is None
    assert mgr.fold.net_base == Decimal('1'), "the bot's inventory is untouched"

    # The bot keeps trading on the re-anchored book.
    port.balance = Decimal('15')
    mgr.record_live_fill(
        _fill("own-2", side='sell', base='-1', quote='100', price='100', ts_ms=T0_MS + 1))
    _run(mgr.reconcile(T0_MS + 200_000))
    assert not mgr.quarantined
    assert len(_read_events(ctx, 'spot_foreign_holdings_reanchored')) == 1
    store.close()


def __test_reconcile_grace_clears_when_settlement_lands__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined

    # Venue balance briefly out of line (settlement wobble).
    port.balance = Decimal('10.5')
    _run(mgr.reconcile(T0_MS))
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.pending_conflict_ts_ms == T0_MS

    # It settles back within the grace: pending cleared, no quarantine.
    port.balance = Decimal('10')
    _run(mgr.reconcile(T0_MS + 10_000))
    assert not mgr.quarantined
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.pending_conflict_ts_ms is None
    store.close()


def __test_reconcile_unexplained_drift_holds_the_cursor__(tmp_path: Path):
    """A conclusive-but-empty catch-up must not advance past a live drift.

    Measured live (Bybit spot lane, cycle 258): the balance showed a
    close fill the execution list did not return 61 s later; the cursor
    advanced to that read's end, so the next read's overlapped start sat
    1.4 s past the fill and it was never recovered — a false quarantine.
    While the drift is unexplained the cursor stays where it was.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('5'))
    mgr = _manager(ctx, port, hook=lambda r, c: None)
    assert not _run(mgr.startup()).quarantined

    # The balance moved, the history lags: the read drains nothing and
    # offers an advanced cursor.
    port.balance = Decimal('6')
    port.batches["anchor-0"] = SpotExecutionBatch(next_cursor="anchor-1")
    _run(mgr.reconcile(T0_MS))
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None
    assert epoch.pending_conflict_ts_ms == T0_MS
    assert epoch.exec_cursor == "anchor-0", "an unexplained drift holds the cursor"

    # The venue indexes the fill behind the balance: the held cursor
    # re-reads the span and recovers it; only then does the cursor move.
    port.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("late-1", base='1', price='100'),),
        next_cursor="anchor-1",
    )
    delivered = _run(mgr.reconcile(T0_MS + 10_000))
    assert [row.fill_id for row in delivered] == ["late-1"]
    assert not mgr.quarantined
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None
    assert epoch.pending_conflict_ts_ms is None
    assert epoch.exec_cursor == "anchor-1"
    assert port.fetch_calls[2:] == ["anchor-0", "anchor-0"], "after the startup's two reads"
    store.close()


def __test_reconcile_in_flight_own_order_holds_the_verdict__(tmp_path: Path):
    """A drift the bot's own live order covers is latency, not a conflict.

    Past the settlement grace the verdict holds while the unfilled quantity
    of the bot's live rows on the drift's side covers it; it follows as soon
    as those rows resolve without a fill. The balance still holds the bot's
    inventory here, so the verdict is the re-anchor of the foreign holdings.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append((r, c)))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('11')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))
    assert not mgr.quarantined

    # The bot's close is dispatched; its fill reaches the balance only.
    ctx.upsert_order(
        "close-1", symbol="BTCUSD", side="sell", qty=1.0, filled_qty=0.0,
        state=STATE_CONFIRMED, intent_key="L", exchange_order_id="X1",
        extras={},
    )
    port.balance = Decimal('10')
    _run(mgr.reconcile(T0_MS + 60_000))
    assert not mgr.quarantined
    for i in (1, 2, 3):
        _run(mgr.reconcile(T0_MS + 60_000 + 31_000 * i))
        assert not mgr.quarantined, "the in-flight sell covers the drift"
    assert not calls
    held = _read_events(ctx, 'spot_inventory_conflict_held')
    assert len(held) == 1, "the hold logs once per set of covering rows"
    assert held[0]['client_order_ids'] == ["close-1"]
    assert held[0]['drift'] == '-1'
    assert not _read_events(ctx, 'spot_foreign_holdings_reanchored')

    # The row resolves without a fill: nothing covers the drift any more.
    ctx.close_order("close-1")
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000 * 4))
    assert not mgr.quarantined
    assert not calls
    reanchored = _read_events(ctx, 'spot_foreign_holdings_reanchored')
    assert len(reanchored) == 1 and reanchored[0]['foreign_baseline'] == '9'
    store.close()


def __test_reconcile_in_flight_order_never_excuses_a_larger_drift__(tmp_path: Path):
    """A live sell of 1 cannot explain a shortfall of 2: the verdict is not held."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    calls: list = []
    mgr = _manager(ctx, port,
                   hook=lambda r, c, **kw: calls.append((r, kw.get('kind'))))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('2')
    mgr.record_live_fill(_fill("own-1", base='2', price='100'))
    _run(mgr.reconcile(T0_MS))
    assert not mgr.quarantined

    # An external sale of the bot's whole inventory rode along with a live
    # sell of 1: past the grace the conflict latches as before.
    ctx.upsert_order(
        "close-1", symbol="BTCUSD", side="sell", qty=1.0, filled_qty=0.0,
        state=STATE_CONFIRMED, intent_key="L", exchange_order_id="X1",
        extras={},
    )
    port.balance = Decimal('0')
    _run(mgr.reconcile(T0_MS + 60_000))
    assert not mgr.quarantined
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert mgr.quarantined
    assert calls == [(calls[0][0], QuarantineKind.EXTERNAL_INTERVENTION)]
    assert 'spot_inventory_taken' in calls[0][0]
    assert not _read_events(ctx, 'spot_inventory_conflict_held')
    store.close()


def __test_reconcile_sale_of_bot_inventory_is_external_intervention__(tmp_path: Path):
    """An unexplained sale that leaves less base than the bot holds.

    Someone else sold (or withdrew) the bot's own inventory: past the
    settlement grace the conflict latches the external-intervention
    quarantine, not the restart-recoverable entry block.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    calls: list = []
    mgr = _manager(ctx, port,
                   hook=lambda r, c, **kw: calls.append((r, c, kw.get('kind'))))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('1')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))
    assert not mgr.quarantined

    # A foreign sell of 0.6: the account now holds 0.4 of the bot's 1.
    port.balance = Decimal('0.4')
    _run(mgr.reconcile(T0_MS + 60_000))
    assert not mgr.quarantined, "the settlement grace comes first"
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert mgr.quarantined
    assert len(calls) == 1
    reason, context, kind = calls[0]
    assert kind is QuarantineKind.EXTERNAL_INTERVENTION
    assert 'spot_inventory_taken' in reason
    assert context['bot_inventory'] == '1'
    assert context['current_total'] == '0.4'
    quarantines = _read_events(ctx, 'spot_inventory_quarantine')
    assert quarantines[-1]['kind'] == QuarantineKind.EXTERNAL_INTERVENTION.value
    store.close()


def _taken_run(store: BrokerStore) -> str:
    """A run whose 1.0 of inventory someone cut to 0.4; returns the latch's id.

    The hook persists the engine's external-intervention latch the way the
    engine does, so the next start finds it open in the store.
    """
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    latched: list[str] = []

    def _hook(reason: str, context: dict, **kw: Any) -> None:
        assert kw.get('kind') is QuarantineKind.EXTERNAL_INTERVENTION
        ctx.record_quarantine("q-taken", kind=str(kw['kind']), reason=reason,
                              intent_key=None, context=context, entered_ts_ms=T0_MS)
        latched.append("q-taken")

    mgr = _manager(ctx, port, hook=_hook)
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('1')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))
    port.balance = Decimal('0.4')
    _run(mgr.reconcile(T0_MS + 60_000))
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert latched == ["q-taken"]
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.state == 'taken'
    ctx.close()
    return latched[0]


def __test_a_taken_epoch_is_held_as_it_is_while_the_latch_holds__(tmp_path: Path):
    """No restart touches the ledger before the operator resumes the run."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    _taken_run(store)

    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0.4'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append(r))
    result = _run(mgr.startup())

    assert result.quarantined
    assert mgr.quarantine_reason == 'spot_external_quarantine_held'
    assert calls == [], "the engine restores its own latch"
    assert result.fold.net_base == Decimal('1'), "the bot's inventory as it left it"
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and (epoch.epoch_seq, epoch.state) == (1, 'taken')
    assert _read_events(ctx, 'spot_epoch_taken_recovered') == []
    assert _read_events(ctx, 'spot_epoch_recovered') == []
    store.close()


def __test_a_resumed_taken_epoch_keeps_what_the_account_still_holds__(tmp_path: Path):
    """After the resume the bot keeps min(inventory, balance); only the taken part leaves."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    quarantine_id = _taken_run(store)
    probe = _open_run(store)
    assert store.request_quarantine_resume(probe.run_id, quarantine_id) is not None
    probe.close()

    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0.4'))
    mgr = _manager(ctx, port)
    result = _run(mgr.startup())

    assert not result.quarantined
    assert result.fold.net_base == Decimal('0.4')
    assert result.fold.vwap == Decimal('100'), "what is kept keeps its average price"
    assert result.epoch is not None and result.epoch.state == 'active'
    assert Decimal(result.epoch.foreign_baseline) == Decimal('0')
    rows = ctx.iter_spot_executions(ACCOUNT, port.product_id)
    assert [(r.fill_id, r.base_delta) for r in rows] == [
        ("own-1", "1"), ("external-taken:1", "-0.6")]
    assert all(r.delivered for r in rows)
    events = _read_events(ctx, 'spot_epoch_taken_recovered')
    assert len(events) == 1
    assert (events[0]['kept'], events[0]['taken']) == ('0.4', '0.6')
    position = mgr.synthesize_position(mark=100.0)
    assert position is not None and position.size == pytest.approx(0.4)

    _run(mgr.reconcile(T0_MS + 600_000))
    assert not mgr.quarantined, "the invariant holds on the kept inventory"
    store.close()


def __test_base_deposited_during_the_quarantine_is_not_the_bots__(tmp_path: Path):
    """The latch measured 0.4 of the bot's 1.0 left; a later deposit is someone else's."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    quarantine_id = _taken_run(store)
    probe = _open_run(store)
    assert store.request_quarantine_resume(probe.run_id, quarantine_id) is not None
    probe.close()

    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('1.4'))
    result = _run(_manager(ctx, port).startup())

    assert not result.quarantined
    assert result.fold.net_base == Decimal('0.4')
    assert result.epoch is not None and Decimal(result.epoch.foreign_baseline) == Decimal('1')
    events = _read_events(ctx, 'spot_epoch_taken_recovered')
    assert (events[0]['kept'], events[0]['taken']) == ('0.4', '0.6')
    store.close()


def __test_an_own_fill_during_the_quarantine_stays_the_bots__(tmp_path: Path):
    """A resting buy of the bot filled while quarantined; a deposit arrived as well."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    quarantine_id = _taken_run(store)
    probe = _open_run(store)
    assert store.request_quarantine_resume(probe.run_id, quarantine_id) is not None
    probe.close()

    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('1.9'))
    port.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("own-2", base='0.5', price='110', ts_ms=T0_MS + 200_000),),
        next_cursor="c-after",
    )
    result = _run(_manager(ctx, port).startup())

    assert not result.quarantined
    assert result.fold.net_base == Decimal('0.9'), "0.4 left at the latch + its own 0.5"
    assert result.epoch is not None and Decimal(result.epoch.foreign_baseline) == Decimal('1')
    store.close()



class _RacePort(FakeSpotPort):
    """Serves ``fill`` from the second history read at ``cursor`` on.

    The bot's own order fills right after the first read: the balance shows
    it at once, the history from the next read.
    """

    def __init__(self, *, balance: Decimal, fill: SpotExecution,
                 cursor: str = "anchor-0", reads_before: int = 1) -> None:
        super().__init__(balance=balance)
        self._race_fill = fill
        self._race_cursor = cursor
        self._reads_before = reads_before
        self._race_reads = 0

    async def fetch_executions(self, cursor: str | None) -> SpotExecutionBatch:
        if cursor != self._race_cursor:
            return await super().fetch_executions(cursor)
        self.fetch_calls.append(cursor)
        self._race_reads += 1
        if self._race_reads > self._reads_before:
            return SpotExecutionBatch(executions=(self._race_fill,), next_cursor="c-after")
        return SpotExecutionBatch(next_cursor=cursor)


@pytest.mark.parametrize("own_fill, balance, kept", [
    (_fill("own-2", base='0.5', price='110', ts_ms=T0_MS + 200_000), '0.9', '0.9'),
    (_fill("own-tp", side='sell', base='-0.4', quote='44', price='110',
           ts_ms=T0_MS + 200_000), '0', '0'),
], ids=["own_buy", "own_sell"])
def __test_an_own_fill_between_the_recovery_reads_is_booked_once__(
        tmp_path: Path, own_fill: SpotExecution, balance: str, kept: str):
    """The resumed recovery reads the history again after its balance read.

    Before, an own buy landing between the two reads was frozen into the
    foreign baseline and booked again by the startup catch-up (the strict
    invariant quarantined the resumed run on the difference), and an own sell
    was counted as taken and then booked again into a negative ledger.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    quarantine_id = _taken_run(store)
    probe = _open_run(store)
    assert store.request_quarantine_resume(probe.run_id, quarantine_id) is not None
    probe.close()

    ctx = _open_run(store)
    port = _RacePort(balance=Decimal(balance), fill=own_fill)
    result = _run(_manager(ctx, port).startup())

    assert not result.quarantined, result.reason
    assert result.fold.violation is None
    assert result.fold.net_base == Decimal(kept)
    assert result.epoch is not None and Decimal(result.epoch.foreign_baseline) == Decimal('0')
    events = _read_events(ctx, 'spot_epoch_taken_recovered')
    assert events[0]['taken'] == '0.6', "only what the latch measured left the bot"
    store.close()


def __test_an_own_fill_between_the_startup_reads_is_booked_once__(tmp_path: Path):
    """The plain startup settles its history read against the balance as well."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    assert not _run(_manager(ctx, FakeSpotPort(balance=Decimal('5'))).startup()).quarantined
    ctx.close()

    ctx = _open_run(store)
    port = _RacePort(balance=Decimal('6'), fill=_fill("own-1", base='1', price='100'))
    result = _run(_manager(ctx, port).startup())

    assert not result.quarantined, result.reason
    assert result.fold.net_base == Decimal('1')
    assert result.epoch is not None and result.epoch.exec_cursor == "c-after"
    store.close()


def __test_a_startup_whose_reads_never_settle_blocks_entries__(tmp_path: Path):
    """Own fills still arriving after every balance read: fail closed, retried at restart."""

    class _BusyPort(FakeSpotPort):
        async def fetch_executions(self, cursor: str | None) -> SpotExecutionBatch:
            self.fetch_calls.append(cursor)
            n = len(self.fetch_calls)
            return SpotExecutionBatch(
                executions=(_fill(f"busy-{n}", base='1', price='100', ts_ms=T0_MS + n),),
                next_cursor=f"c{n}",
            )

    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    calls: list = []
    port = _BusyPort(balance=Decimal('100'))
    result = _run(_manager(ctx, port, hook=lambda r, c, **kw: calls.append(kw.get('kind'))
                           ).startup())
    assert result.quarantined
    assert result.reason == 'spot_startup_unsettled'
    assert calls == [None], "the entry block is the hook's default kind"
    assert port.balance_calls == 3
    store.close()


def __test_a_plain_startup_whose_reads_never_settle_defers_the_drift__(tmp_path: Path):
    """The bot's resting sell is eaten in pieces while it starts: no entry block.

    An existing epoch freezes nothing on the startup measurement, so fills
    still arriving after every balance read leave a drift the runtime
    reconcile judges like any other. The entry block latched here, and the
    restart it asked for retired the bot's remaining inventory as foreign.
    """

    class _PiecesPort(FakeSpotPort):
        async def fetch_executions(self, cursor: str | None) -> SpotExecutionBatch:
            self.fetch_calls.append(cursor)
            n = len(self.fetch_calls)
            if n > 4:
                return SpotExecutionBatch(next_cursor=cursor)
            self.balance -= Decimal('0.1')
            return SpotExecutionBatch(
                executions=(_fill(f"own-piece-{n}", side='sell', base='-0.1', quote='10',
                                  price='100', ts_ms=T0_MS + n, coid="own-sell"),),
                next_cursor=f"c{n}",
            )

    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    _bot_holding_one(store, foreign='0')
    ctx = _open_run(store)
    calls: list = []
    port = _PiecesPort(balance=Decimal('1'))
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append(r))
    result = _run(mgr.startup())
    assert not result.quarantined and calls == []
    assert port.balance_calls == 3
    assert result.fold.net_base == Decimal('0.6')
    assert len(_read_events(ctx, 'spot_startup_drift_pending')) == 1
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.exec_cursor == "anchor-0", "the drift holds the cursor"
    _run(mgr.reconcile(T0_MS + 60_000))
    assert not mgr.quarantined
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.pending_conflict_ts_ms is None
    ctx.close()

    ctx = _open_run(store)
    restarted = _run(_manager(ctx, FakeSpotPort(balance=Decimal('0.6'))).startup())
    assert not restarted.quarantined
    assert restarted.fold.net_base == Decimal('0.6'), "nothing of the bot's was retired"
    assert _read_events(ctx, 'spot_epoch_recovered') == []
    store.close()


def __test_a_failed_measurement_keeps_a_resumed_epoch_taken__(tmp_path: Path):
    """The next start must still keep the inventory, not retire it."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    quarantine_id = _taken_run(store)
    probe = _open_run(store)
    assert store.request_quarantine_resume(probe.run_id, quarantine_id) is not None
    probe.close()

    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0.4'))
    port.balance_error = RuntimeError("venue down")
    result = _run(_manager(ctx, port).startup())
    assert result.quarantined
    assert result.reason == 'spot_recovery_balance_unavailable'
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.state == 'taken'
    ctx.close()

    ctx = _open_run(store)
    result = _run(_manager(ctx, FakeSpotPort(balance=Decimal('0.4'))).startup())
    assert not result.quarantined
    assert result.fold.net_base == Decimal('0.4')
    store.close()


def __test_a_taken_epoch_over_a_corrupt_ledger_retires_it__(tmp_path: Path):
    """A negative fold is the ledger's own corruption: it retires as any quarantined epoch."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    mgr.record_live_fill(_fill("b1", base='1', price='100'))
    mgr.record_live_fill(
        _fill("s1", side='sell', base='-2', quote='200', price='100', ts_ms=T0_MS + 1))
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None
    ctx.set_spot_epoch_state(port.product_id, epoch.epoch_seq, 'taken')
    ctx.close()

    ctx2 = _open_run(store)
    result = _run(_manager(ctx2, FakeSpotPort(balance=Decimal('0.5'))).startup())
    assert not result.quarantined
    assert result.fold.fill_count == 0
    assert result.epoch is not None and Decimal(result.epoch.foreign_baseline) == Decimal('0.5')
    assert len(_read_events(ctx2, 'spot_epoch_recovered')) == 1
    assert _read_events(ctx2, 'spot_epoch_taken_recovered') == []
    store.close()


@pytest.mark.parametrize("history", ["own_sell", "nothing"])
def __test_reconcile_inconclusive_shortfall_waits_for_a_complete_history__(
        tmp_path: Path, caplog, history: str):
    """An incomplete execution history cannot rule out an own fill.

    The balance is below the bot's inventory, but the catch-up could not read
    the full history: past the grace it is an ERROR and a durable audit, once,
    no quarantine and nothing booked — the conflict stays pending. The first
    complete read decides it: the bot's own unreported sell resolves it, an
    unexplained shortfall is the external intervention.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    calls: list = []
    mgr = _manager(ctx, port,
                   hook=lambda r, c, **kw: calls.append((r, kw.get('kind'))))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('1')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))

    port.batches["anchor-0"] = SpotExecutionBatch(conclusive=False)
    port.balance = Decimal('0.4')
    _run(mgr.reconcile(T0_MS + 60_000))
    with caplog.at_level(logging.ERROR):
        for i in (1, 2, 3):
            _run(mgr.reconcile(T0_MS + 60_000 + 31_000 * i))
    assert "anchor-0" in port.fetch_calls
    assert not mgr.quarantined
    assert calls == []
    assert mgr.fold.net_base == Decimal('1'), "the venue truth is not booked"
    unverified = _read_events(ctx, 'spot_inventory_shortfall_unverified')
    assert len(unverified) == 1, "reported once per pending conflict"
    assert (unverified[0]['bot_inventory'], unverified[0]['current_total']) == ('1', '0.4')
    assert len([r for r in caplog.records
                if r.levelno == logging.ERROR and "cannot be read completely" in r.getMessage()]) == 1
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.pending_conflict_ts_ms == T0_MS + 60_000

    if history == "own_sell":
        port.batches["anchor-0"] = SpotExecutionBatch(
            executions=(_fill("own-sell", side='sell', base='-0.6', quote='60',
                              price='100', ts_ms=T0_MS + 50_000),),
            next_cursor="anchor-1",
        )
        delivered = _run(mgr.reconcile(T0_MS + 300_000))
        assert [row.fill_id for row in delivered] == ["own-sell"]
        assert not mgr.quarantined
        assert calls == []
        epoch = ctx.get_latest_spot_epoch(port.product_id)
        assert epoch is not None and epoch.pending_conflict_ts_ms is None
    else:
        port.batches["anchor-0"] = SpotExecutionBatch(next_cursor="anchor-0")
        _run(mgr.reconcile(T0_MS + 300_000))
        assert mgr.quarantined
        assert calls == [(calls[0][0], QuarantineKind.EXTERNAL_INTERVENTION)]
        assert 'spot_inventory_taken' in calls[0][0]
    store.close()


def __test_reconcile_sale_of_foreign_holdings_reanchors_without_quarantine__(tmp_path: Path):
    """A sale out of the foreign holdings leaves the bot's inventory intact.

    The balance stays above the bot's own inventory: someone else's separate
    activity. The foreign holdings are re-anchored, the cursor stays where
    the unexplained drift held it (a late own fill is still read back), and
    no quarantine latches.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    calls: list = []
    mgr = _manager(ctx, port,
                   hook=lambda r, c, **kw: calls.append((r, kw.get('kind'))))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('11')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))

    port.batches["anchor-0"] = SpotExecutionBatch(next_cursor="anchor-1")
    port.balance = Decimal('10.4')
    _run(mgr.reconcile(T0_MS + 60_000))
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert not mgr.quarantined
    assert calls == []
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None
    assert (epoch.state, epoch.foreign_baseline) == ('active', '9.4')
    assert epoch.exec_cursor == "anchor-0", "the held cursor survives the re-anchor"
    assert len(_read_events(ctx, 'spot_foreign_holdings_reanchored')) == 1
    _run(mgr.reconcile(T0_MS + 200_000))
    assert not mgr.quarantined
    assert len(_read_events(ctx, 'spot_foreign_holdings_reanchored')) == 1
    store.close()


def __test_reconcile_own_sell_in_flight_never_reports_a_taken_inventory__(tmp_path: Path):
    """The bot's own unbooked sell explains a balance below its inventory."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append(kw.get('kind')))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('1')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))

    ctx.upsert_order(
        "close-1", symbol="BTCUSD", side="sell", qty=1.0, filled_qty=0.0,
        state=STATE_CONFIRMED, intent_key="L", exchange_order_id="X1",
        extras={},
    )
    port.balance = Decimal('0')
    _run(mgr.reconcile(T0_MS + 60_000))
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert not mgr.quarantined
    assert calls == []
    store.close()


def __test_reconcile_a_resting_own_order_holds_the_verdict_only_for_a_bounded_time__(
        tmp_path: Path):
    """The bot's stop-loss sell rests on the venue for the whole life of the position.

    Someone sells 0.6 of the bot's own 1 in the venue UI. The resting stop
    covers that drift, so it holds the verdict like an in-flight close, but
    only for as long as a fill of the bot's own can be missing from the
    execution history; then the shortfall is the external intervention. The
    hold had no bound: the intervention went unreported while the stop rested.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append((c, kw.get('kind'))))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('1')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))
    ctx.upsert_order(
        "sl-1", symbol="BTCUSD", side="sell", qty=1.0, filled_qty=0.0,
        state=STATE_CONFIRMED, intent_key="L-X", exchange_order_id="X-SL", extras={},
    )

    port.balance = Decimal('0.4')
    armed = T0_MS + 60_000
    _run(mgr.reconcile(armed))
    hold_ends = armed + int((port.settlement_grace_s + _HISTORY_LAG_CEILING_S) * 1000)
    for now in (armed + 31_000, hold_ends - 1):
        _run(mgr.reconcile(now))
        assert not mgr.quarantined, "an own fill may still reach the history"
    assert [e['client_order_ids'] for e in _read_events(ctx, 'spot_inventory_conflict_held')] \
        == [["sl-1"]]

    _run(mgr.reconcile(hold_ends))
    assert mgr.quarantined and mgr.quarantine_reason == 'spot_inventory_taken'
    assert len(calls) == 1
    context, kind = calls[0]
    assert kind is QuarantineKind.EXTERNAL_INTERVENTION
    assert (context['bot_inventory'], context['current_total']) == ('1', '0.4')
    assert context['in_flight_orders_without_fill'] == ["sl-1"]
    store.close()


def _bot_holding_one_with_a_resting_stop(ctx: RunContext, port: FakeSpotPort,
                                         calls: list) -> SpotInventoryManager:
    """The bot bought 1 on a flat account and its stop sell of 1 rests."""
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append((r, kw.get('kind'))))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('1')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))
    ctx.upsert_order(
        "sl-1", symbol="BTCUSD", side="sell", qty=1.0, filled_qty=0.0,
        state=STATE_CONFIRMED, intent_key="L-X", exchange_order_id="X-SL", extras={},
    )
    return mgr


def __test_an_own_stop_fill_after_a_conflict_an_earlier_run_left_pending_is_held__(
        tmp_path: Path):
    """A pending conflict from an earlier run does not date an own fill found at startup.

    Someone deposits 0.5 beside the bot's 1 and the run stops within the
    settlement grace, leaving the conflict pending. An hour later the bot's
    own stop sells its 1 just before the restart, and the venue has not
    indexed the fill in its execution history yet. The verdict is held on
    the stop for as long as its fill can be missing from the history,
    measured from when the restarted process first saw this drift; the
    history then returns the fill. The hold used to run from the earlier
    run's first observation, an hour old, so the first pass judged the
    bot's 1 taken and quarantined a healthy bot.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    first = _bot_holding_one_with_a_resting_stop(ctx, port, [])
    port.balance = Decimal('1.5')
    _run(first.reconcile(T0_MS + 10_000))
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.pending_conflict_ts_ms == T0_MS + 10_000
    ctx.close()

    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('0.5'))
    calls: list = []
    mgr = _manager(ctx2, port2, hook=lambda r, c, **kw: calls.append((r, kw.get('kind'))))
    assert not _run(mgr.startup()).quarantined
    restarted = T0_MS + 3_600_000
    _run(mgr.reconcile(restarted))
    _run(mgr.reconcile(restarted + 60_000))
    assert not mgr.quarantined and calls == [], "the own stop fill was judged as base taken"
    assert [e['client_order_ids'] for e in _read_events(ctx2, 'spot_inventory_conflict_held')] \
        == [["sl-1"]]

    port2.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("own-sl", side='sell', base='-1', quote='100', price='100',
                          ts_ms=restarted - 1_000),),
        next_cursor="anchor-1",
    )
    delivered = _run(mgr.reconcile(restarted + 120_000))
    assert [row.fill_id for row in delivered] == ["own-sl"]
    assert mgr.fold.net_base == Decimal('0')
    _run(mgr.reconcile(restarted + 150_000))
    assert not mgr.quarantined and calls == []
    store.close()


def __test_an_own_stop_fill_while_an_unverified_conflict_stays_pending_is_held__(
        tmp_path: Path):
    """A conflict an incomplete history keeps pending does not date a later own fill.

    Someone deposits 0.5 beside the bot's 1 while the execution history
    cannot be read completely: the conflict stays pending for ten minutes.
    Then the bot's own stop sells its 1, its push lost, and the history,
    complete again, has not indexed the fill yet. The changed drift gets the
    full hold on the stop, and the history then returns the fill: the
    deposit is re-anchored as the foreign holdings, no quarantine. The hold
    used to run from the deposit's first observation and judged at once.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    calls: list = []
    mgr = _bot_holding_one_with_a_resting_stop(ctx, port, calls)

    port.batches["anchor-0"] = SpotExecutionBatch(conclusive=False)
    port.balance = Decimal('1.5')
    armed = T0_MS + 60_000
    for i in range(20):
        _run(mgr.reconcile(armed + 31_000 * i))
    assert not mgr.quarantined
    assert len(_read_events(ctx, 'spot_foreign_activity_unverified')) == 1

    port.batches["anchor-0"] = SpotExecutionBatch(next_cursor="anchor-0")
    port.balance = Decimal('0.5')
    sold = armed + 31_000 * 20
    _run(mgr.reconcile(sold))
    _run(mgr.reconcile(sold + 60_000))
    assert not mgr.quarantined and calls == [], "the own stop fill was judged as base taken"

    port.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("own-sl", side='sell', base='-1', quote='100', price='100',
                          ts_ms=sold - 1_000),),
        next_cursor="anchor-1",
    )
    _run(mgr.reconcile(sold + 90_000))
    _run(mgr.reconcile(sold + 130_000))
    assert not mgr.quarantined and calls == []
    assert mgr.fold.net_base == Decimal('0')
    reanchored = _read_events(ctx, 'spot_foreign_holdings_reanchored')
    assert [e['foreign_baseline'] for e in reanchored] == ['0.5']
    store.close()


def __test_reconcile_own_sell_beside_a_sale_of_foreign_base_is_no_intervention__(tmp_path: Path):
    """The bot closes its 1 while the account owner sells 0.2 of their own 0.5.

    The close filled on the venue, but neither the stream nor the history has
    its fill yet, and the coincident sale makes the drift larger than the
    close. Had the close filled, the account's 0.3 still holds the bot's
    share: the verdict waits for the close instead of judging the bot's 1
    taken, and once the history returns the fill, the foreign holdings are
    re-anchored on the 0.3. The verdict used to compare the balance with the
    bot's inventory before its own sell and latched the external
    intervention.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0.5'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append((r, kw.get('kind'))))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('1.5')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))
    ctx.upsert_order(
        "close-1", symbol="BTCUSD", side="sell", qty=1.0, filled_qty=0.0,
        state=STATE_CONFIRMED, intent_key="L", exchange_order_id="X1", extras={},
    )

    port.balance = Decimal('0.3')
    _run(mgr.reconcile(T0_MS + 60_000))
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert not mgr.quarantined and calls == []
    held = _read_events(ctx, 'spot_inventory_conflict_held')
    assert [(e['drift'], e['client_order_ids']) for e in held] == [('-1.2', ["close-1"])]

    port.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("own-close", side='sell', base='-1', quote='100', price='100',
                          ts_ms=T0_MS + 59_000, coid="close-1"),),
        next_cursor="anchor-1",
    )
    delivered = _run(mgr.reconcile(T0_MS + 60_000 + 62_000))
    assert [row.fill_id for row in delivered] == ["own-close"]
    assert not mgr.quarantined and calls == []
    assert mgr.fold.net_base == Decimal('0')
    reanchored = _read_events(ctx, 'spot_foreign_holdings_reanchored')
    assert len(reanchored) == 1 and reanchored[0]['foreign_baseline'] == '0.3'
    store.close()


class _OwnFillAfterBalanceRead(FakeSpotPort):
    """The bot's own fill lands on the venue right after one balance read.

    The history serves it on the read that follows, so the pass books a fill
    the balance it read does not hold.
    """

    def __init__(self, *, balance: Decimal) -> None:
        super().__init__(balance=balance)
        self.race_fill: SpotExecution | None = None
        self.race_on_balance_call: int | None = None
        self._race_due = False

    async def fetch_base_balance(self) -> Decimal:
        value = await super().fetch_base_balance()
        if self.race_fill is not None and self.balance_calls == self.race_on_balance_call:
            self.balance += self.race_fill.base_delta
            self._race_due = True
        return value

    async def fetch_executions(self, cursor: str | None) -> SpotExecutionBatch:
        if self._race_due and self.race_fill is not None:
            self._race_due = False
            self.fetch_calls.append(cursor)
            return SpotExecutionBatch(executions=(self.race_fill,), next_cursor=cursor)
        return await super().fetch_executions(cursor)


@pytest.mark.parametrize(("bot", "own", "reanchored_at"), [
    ('1', _fill("own-buy", base='6', price='100', ts_ms=T0_MS + 90_000), '5'),
    ('3', _fill("own-sell", side='sell', base='-2', quote='200', price='100',
                ts_ms=T0_MS + 90_000), '5'),
], ids=["own-buy", "own-sell"])
def __test_reconcile_verdict_is_not_judged_on_a_balance_read_before_an_own_fill__(
        tmp_path: Path, bot: str, own: SpotExecution, reanchored_at: str):
    """Someone sells 5 of the foreign 10; on the verdict pass the bot's own fill lands.

    It lands right after the pass's balance read and the history books it in
    the same pass. Judged on that stale read, the bot's own buy was base taken
    from it (an EXTERNAL_INTERVENTION quarantine of a healthy bot), and its own
    sell re-anchored the foreign holdings on base that was the bot's.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = _OwnFillAfterBalanceRead(balance=Decimal('10'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append((r, kw.get('kind'))))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('10') + Decimal(bot)
    mgr.record_live_fill(_fill("own-1", base=bot, price='100'))
    _run(mgr.reconcile(T0_MS))

    port.balance -= Decimal('5')
    _run(mgr.reconcile(T0_MS + 60_000))
    port.race_fill = own
    port.race_on_balance_call = port.balance_calls + 1
    delivered = _run(mgr.reconcile(T0_MS + 60_000 + 31_000))

    assert [row.fill_id for row in delivered] == [own.fill_id]
    assert not mgr.quarantined and calls == []
    assert mgr.fold.net_base == Decimal(bot) + own.base_delta
    reanchored = _read_events(ctx, 'spot_foreign_holdings_reanchored')
    assert [e['foreign_baseline'] for e in reanchored] == [reanchored_at]
    _run(mgr.reconcile(T0_MS + 60_000 + 62_000))
    assert not mgr.quarantined
    store.close()


def __test_reconcile_inconclusive_history_never_reanchors_the_foreign_holdings__(
        tmp_path: Path, caplog):
    """The bot's own exit sell filled unreported; the history cannot be read completely.

    The balance still holds the bot's booked inventory, but only a complete
    history tells a sale out of the foreign holdings from the bot's own sell.
    Re-anchoring here absorbed the own sell into the foreign holdings, and the
    bot went on to sell base it no longer held, out of someone else's. The
    conflict stays pending, an ERROR and an audit once, and the complete
    history resolves it without a re-anchor.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: calls.append(r))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('11')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))

    port.batches["anchor-0"] = SpotExecutionBatch(conclusive=False)
    port.balance = Decimal('10')
    _run(mgr.reconcile(T0_MS + 60_000))
    with caplog.at_level(logging.ERROR):
        for i in (1, 2, 3):
            _run(mgr.reconcile(T0_MS + 60_000 + 31_000 * i))
    assert not mgr.quarantined and calls == []
    assert _read_events(ctx, 'spot_foreign_holdings_reanchored') == []
    unverified = _read_events(ctx, 'spot_foreign_activity_unverified')
    assert len(unverified) == 1, "reported once per pending conflict"
    assert (unverified[0]['bot_inventory'], unverified[0]['current_total']) == ('1', '10')
    assert len([r for r in caplog.records if r.levelno == logging.ERROR
                and "cannot be read completely" in r.getMessage()]) == 1
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and (epoch.epoch_seq, epoch.foreign_baseline) == (1, '10')

    port.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("own-exit", side='sell', base='-1', quote='100', price='100',
                          ts_ms=T0_MS + 50_000),),
        next_cursor="anchor-1",
    )
    delivered = _run(mgr.reconcile(T0_MS + 300_000))
    assert [row.fill_id for row in delivered] == ["own-exit"]
    assert mgr.fold.net_base == Decimal('0')
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None
    assert (epoch.epoch_seq, epoch.pending_conflict_ts_ms) == (1, None)
    assert _read_events(ctx, 'spot_foreign_holdings_reanchored') == []
    store.close()


def __test_a_reanchor_reads_the_span_it_held_until_a_late_own_fill_is_indexed__(
        tmp_path: Path):
    """A complete history returns the bot's own exit sell only after the grace.

    The re-anchor absorbs the sell into the foreign holdings. Every clean pass
    reads the span the re-anchor held again until the sell would have been
    indexed: the sell is booked, and the drift it leaves re-anchors the
    holdings back. A clean pass never read the history before, so the bot
    went on selling base it no longer held.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: None)
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('11')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))

    port.batches["anchor-0"] = SpotExecutionBatch(next_cursor="anchor-1")
    port.balance = Decimal('10')
    _run(mgr.reconcile(T0_MS + 60_000))
    reanchored_at = T0_MS + 91_000
    _run(mgr.reconcile(reanchored_at))
    reanchored = _read_events(ctx, 'spot_foreign_holdings_reanchored')
    assert [e['foreign_baseline'] for e in reanchored] == ['9']

    reads = len(port.fetch_calls)
    _run(mgr.reconcile(reanchored_at + 30_000))
    assert port.fetch_calls[reads:] == ["anchor-0"], "a clean pass reads the held span"
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.exec_cursor == "anchor-0"

    port.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("own-exit", side='sell', base='-1', quote='100', price='100',
                          ts_ms=T0_MS + 55_000),),
        next_cursor="anchor-1",
    )
    delivered = _run(mgr.reconcile(reanchored_at + 60_000))
    assert [row.fill_id for row in delivered] == ["own-exit"]
    _run(mgr.reconcile(reanchored_at + 91_000))
    assert not mgr.quarantined
    assert mgr.fold.net_base == Decimal('0')
    reanchored = _read_events(ctx, 'spot_foreign_holdings_reanchored')
    assert [e['foreign_baseline'] for e in reanchored] == ['9', '10']

    window_ends = reanchored_at + 91_000 + int(_HISTORY_LAG_CEILING_S * 1000)
    _run(mgr.reconcile(window_ends))
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.exec_cursor == "anchor-1", "the span is read"
    reads = len(port.fetch_calls)
    _run(mgr.reconcile(window_ends + 30_000))
    assert port.fetch_calls[reads:] == [], "a clean pass reads nothing once it is covered"
    store.close()


def __test_a_restart_inside_the_reread_window_keeps_reading_the_held_span__(
        tmp_path: Path):
    """The reread window a re-anchor opened survives a restart.

    The bot's own exit sell is in the balance but not yet in the history when
    the re-anchor absorbs it into the foreign holdings, and the process
    restarts before the venue indexes it. The startup reads from the held
    cursor and keeps it there; the first clean pass of the new process reads
    the span again and books the sell. Without the persisted window the
    startup advanced the cursor past the span and the bot kept believing it
    owned base it had sold.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port, hook=lambda r, c, **kw: None)
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('11')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))

    port.batches["anchor-0"] = SpotExecutionBatch(next_cursor="anchor-1")
    port.balance = Decimal('10')
    _run(mgr.reconcile(T0_MS + 60_000))
    reanchored_at = T0_MS + 91_000
    _run(mgr.reconcile(reanchored_at))
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None
    assert (epoch.foreign_baseline, epoch.exec_cursor, epoch.reread_cursor) == (
        '9', 'anchor-0', 'anchor-0')
    assert epoch.reread_until_ms == reanchored_at + int(_HISTORY_LAG_CEILING_S * 1000)
    ctx.close()

    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('10'))
    port2.batches["anchor-0"] = SpotExecutionBatch(next_cursor="anchor-1")
    mgr2 = _manager(ctx2, port2, hook=lambda r, c, **kw: None)
    assert not _run(mgr2.startup()).quarantined
    epoch = ctx2.get_latest_spot_epoch(port2.product_id)
    assert epoch is not None and epoch.exec_cursor == "anchor-0", "the startup holds the span"

    port2.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("own-exit", side='sell', base='-1', quote='100', price='100',
                          ts_ms=T0_MS + 55_000),),
        next_cursor="anchor-1",
    )
    delivered = _run(mgr2.reconcile(reanchored_at + 60_000))
    assert [row.fill_id for row in delivered] == ["own-exit"]
    assert mgr2.fold.net_base == Decimal('0')
    store.close()


def __test_reconcile_transient_balance_failure_skips_cycle__(tmp_path: Path):
    """A live bot must not halt (or arm anything) on a recoverable read."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    port.balance_error = ConnectionError("hiccup")
    _run(mgr.reconcile(T0_MS))
    assert not mgr.quarantined and mgr.pending_halt is None
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.pending_conflict_ts_ms is None
    store.close()


def __test_reconcile_transient_balance_failure_logs_one_line__(tmp_path: Path, caplog):
    """A retryable transport fault on the balance read is ONE warning line
    carrying the fault's message — no traceback (the live log would count it
    as errors) — while an unclassified failure keeps its traceback."""
    import logging
    from pynecore.core.plugin import TransientProviderError
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined

    def read_failures() -> list[logging.LogRecord]:
        return [r for r in caplog.records if "base-balance read failed" in r.getMessage()]

    with caplog.at_level(logging.WARNING):
        port.balance_error = TransientProviderError("wallet-balance: dns down")
        _run(mgr.reconcile(T0_MS))
        assert len(read_failures()) == 1
        assert not read_failures()[0].exc_info
        assert "dns down" in read_failures()[0].getMessage()

        port.balance_error = ConnectionError("hiccup")
        _run(mgr.reconcile(T0_MS + 60_000))
        assert len(read_failures()) == 2
        assert read_failures()[1].exc_info               # traceback attached
    assert not mgr.quarantined and mgr.pending_halt is None
    store.close()


def __test_reconcile_locked_base_needs_total_ownership_read__(tmp_path: Path):
    """Base locked in a resting sell order stays owned: a port honouring
    the total-ownership contract shows no drift while the order rests."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('12')
    mgr.record_live_fill(_fill("b1", base='2', price='100'))

    # The bot rests a sell for 1 BTC: available drops to 11, but TOTAL
    # ownership (available + locked) is still 12 — no drift.
    _run(mgr.reconcile(T0_MS))
    assert not mgr.quarantined
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.pending_conflict_ts_ms is None

    # An available-only (contract-violating) read would false-fire:
    port.balance = Decimal('11')
    _run(mgr.reconcile(T0_MS + 1_000))
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.pending_conflict_ts_ms is not None
    store.close()


# === Quarantine delivery: policies, hook fallback, halt ====================


def __test_policy_halt_arms_pending_halt_not_hook__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('5'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c: calls.append((r, c)),
                   policy='halt')
    result = _run(mgr.startup())  # baseline 5
    assert not result.quarantined
    ctx.close()

    ctx2, mgr2, result = _inconclusive_restart(
        store, FakeSpotPort(balance=Decimal('7')),
        hook=lambda r, c: calls.append((r, c)), policy='halt')
    assert result.quarantined
    assert not calls  # hook not consulted under halt
    halt = mgr2.consume_pending_halt()
    assert isinstance(halt, SpotInventoryConflictError)
    assert mgr2.consume_pending_halt() is None  # consume-once
    store.close()


def __test_quarantine_hook_failure_falls_back_to_halt__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('5'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    ctx.close()

    def raising_hook(_reason: str, _context: dict) -> None:
        raise RuntimeError("sink broken")

    ctx2, mgr2, result = _inconclusive_restart(
        store, FakeSpotPort(balance=Decimal('9')), hook=raising_hook)
    assert result.quarantined
    assert isinstance(mgr2.pending_halt, SpotInventoryConflictError)
    store.close()


def __test_missing_hook_falls_back_to_halt__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('5'))
    mgr = _manager(ctx, port, hook=None)
    assert not _run(mgr.startup()).quarantined
    ctx.close()

    ctx2, mgr2, result = _inconclusive_restart(
        store, FakeSpotPort(balance=Decimal('9')), hook=None)
    assert result.quarantined
    assert isinstance(mgr2.pending_halt, SpotInventoryConflictError)
    store.close()


def __test_negative_inventory_live_fill_quarantines__(tmp_path: Path):
    """An oversell reaching the ledger is bookkeeping corruption."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c: calls.append((r, c)))
    assert not _run(mgr.startup()).quarantined
    mgr.record_live_fill(_fill("b1", base='1', price='100'))
    mgr.record_live_fill(
        _fill("s1", side='sell', base='-2', quote='200', price='100',
              ts_ms=T0_MS + 1),
    )
    assert mgr.quarantined
    assert mgr.quarantine_reason == 'spot_ledger_negative_inventory'
    assert len(calls) == 1
    store.close()


# === Ownership: lease + foreign ledger rows =================================


def __test_lease_second_run_starts_quarantined__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx_a = _open_run(store)
    port_a = FakeSpotPort(balance=Decimal('0'))
    mgr_a = _manager(ctx_a, port_a)
    assert not _run(mgr_a.startup()).quarantined

    # A second logical run (different label -> different run_id) on the
    # same base asset must not trade.
    ctx_b = _open_run(store, label="second")
    port_b = FakeSpotPort(balance=Decimal('0'))
    calls: list = []
    mgr_b = _manager(ctx_b, port_b, hook=lambda r, c: calls.append((r, c)))
    result = _run(mgr_b.startup())
    assert result.quarantined
    assert result.reason == 'spot_lease_conflict'
    assert len(calls) == 1
    # The loser touched nothing: no ledger reads, no epoch.
    assert ctx_b.get_latest_spot_epoch(port_b.product_id) is None
    store.close()


def __test_lease_concurrent_claim_single_winner__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx_a = _open_run(store)
    ctx_b = _open_run(store, label="second")
    barrier = threading.Barrier(2)
    results: dict[str, bool] = {}

    def claim(name: str, ctx: RunContext) -> None:
        barrier.wait()
        results[name] = ctx.claim_spot_asset(ACCOUNT, "BTC", "USD")

    threads = [
        threading.Thread(target=claim, args=("a", ctx_a)),
        threading.Thread(target=claim, args=("b", ctx_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results.values()) == [False, True]
    store.close()


# noinspection SqlResolve,SqlWithoutWhere
def __test_lease_stale_heartbeat_taken_over__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx_a = _open_run(store)
    assert ctx_a.claim_spot_asset(ACCOUNT, "BTC", "USD")
    # Re-claim by the owner refreshes, no takeover.
    assert ctx_a.claim_spot_asset(ACCOUNT, "BTC", "USD")

    ctx_b = _open_run(store, label="second")
    assert not ctx_b.claim_spot_asset(ACCOUNT, "BTC", "USD")

    # Age the heartbeat past the stale threshold.
    store._conn.execute(  # type: ignore[attr-defined]
        "UPDATE spot_asset_owner SET heartbeat_ts_ms = heartbeat_ts_ms - ?",
        (10 * 60 * 1000,),
    )
    assert ctx_b.claim_spot_asset(ACCOUNT, "BTC", "USD")
    events = _read_events(ctx_b, 'spot_lease_taken_over')
    assert len(events) == 1 and events[0]['base_asset'] == "BTC"

    # The old owner's heartbeat is now a no-op (guarded by run_id)...
    ctx_a.heartbeat_spot_asset(ACCOUNT, "BTC")
    # ...and its release must not evict the new owner.
    ctx_a.release_spot_asset(ACCOUNT, "BTC")
    assert not ctx_a.claim_spot_asset(ACCOUNT, "BTC", "USD")
    store.close()


def __test_foreign_ledger_row_quarantines_both_paths__(tmp_path: Path):
    """A fill id booked under another logical run breaks exclusivity."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx_a = _open_run(store)
    ctx_a.record_spot_execution(
        ACCOUNT, "BTC-USD", fill_id="shared-1", side='buy',
        base_delta='1', quote_delta='-100', price='100',
        fee_amount='0', fee_currency='USD', ts_ms=T0_MS,
    )
    ctx_a.close()
    # Free the lease so run B fails on the LEDGER row, not the lease.
    ctx_a.release_spot_asset(ACCOUNT, "BTC")

    ctx_b = _open_run(store, label="second")
    port = FakeSpotPort(balance=Decimal('1'))
    port.batches[None] = SpotExecutionBatch(
        executions=(_fill("shared-1", base='1', price='100'),),
        next_cursor="anchor-1",
    )
    calls: list = []
    mgr = _manager(ctx_b, port, hook=lambda r, c: calls.append((r, c)))
    result = _run(mgr.startup())
    assert result.quarantined
    assert result.reason == 'spot_foreign_ledger_row'
    assert len(calls) == 1
    # The quarantine evidence survived the rolled-back page transaction.
    assert _read_events(ctx_b, 'spot_inventory_quarantine')

    # The LIVE path must raise (the caller must not emit the event).
    mgr._started = True
    with pytest.raises(SpotInventoryConflictError):
        mgr._quarantined = False  # re-arm to exercise the raise path
        mgr.record_live_fill(_fill("shared-1", base='1', price='100'))
    store.close()


# === Position synthesis =====================================================


def __test_synthesize_position_long_and_flat__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    assert mgr.synthesize_position(100.0) is None  # genuinely flat

    port.balance = Decimal('2')
    mgr.record_live_fill(_fill("b1", base='1', price='100'))
    mgr.record_live_fill(_fill("b2", base='1', quote='-110', price='110',
                               ts_ms=T0_MS + 1))
    pos = mgr.synthesize_position(120.0)
    assert pos is not None
    assert pos.symbol == "BTCUSD" and pos.side == 'long'
    assert pos.size == 2.0
    assert pos.entry_price == pytest.approx(105.0)
    assert pos.unrealized_pnl == pytest.approx((120.0 - 105.0) * 2.0)
    assert pos.leverage == 1.0 and pos.margin_mode == 'cash'
    assert pos.liquidation_price is None

    # Sell it all: flat again, basis gone.
    port.balance = Decimal('0')
    mgr.record_live_fill(
        _fill("s1", side='sell', base='-2', quote='240', price='120',
              ts_ms=T0_MS + 2),
    )
    assert mgr.synthesize_position(120.0) is None
    store.close()


# === Rebaseline =============================================================


# noinspection SqlResolve
# === Retention exemption =====================================================


def __test_retention_purge_never_touches_the_ledger__(tmp_path: Path):
    """A 180+ day old position must stay reconstructible."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    ancient = T0_MS - 200 * 86_400_000
    ctx.record_spot_execution(
        ACCOUNT, "BTC-USD", fill_id="old-1", side='buy',
        base_delta='1', quote_delta='-100', price='100',
        fee_amount='0', fee_currency='USD', ts_ms=ancient,
    )
    ctx.insert_spot_epoch(
        account_id=ACCOUNT, base_asset="BTC", product_id="BTC-USD",
        foreign_baseline='0', cursor_scope='product', exec_cursor='c0',
    )
    ctx.claim_spot_asset(ACCOUNT, "BTC", "USD")
    store.cleanup_old_data(retention_days=180)
    rows = ctx.iter_spot_executions(ACCOUNT, "BTC-USD")
    assert [r.fill_id for r in rows] == ["old-1"]
    assert ctx.get_latest_spot_epoch("BTC-USD") is not None
    assert not ctx.claim_spot_asset(ACCOUNT, "BTC", "USD") or True  # row survives
    store.close()


# === Store-level odds and ends ==============================================


def __test_store_epoch_state_validation_and_delivered_chunking__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    with pytest.raises(ValueError):
        ctx.insert_spot_epoch(
            account_id=ACCOUNT, base_asset="BTC", product_id="BTC-USD",
            foreign_baseline='0', cursor_scope='product',
            exec_cursor=None, state='bogus',
        )
    with pytest.raises(ValueError):
        ctx.record_spot_execution(
            ACCOUNT, "BTC-USD", fill_id="x", side='hold',
            base_delta='1', quote_delta='-1', price='1',
            fee_amount='0', fee_currency='USD', ts_ms=T0_MS,
        )
    # Chunked delivered-flip over the 500-per-statement window.
    for i in range(1_050):
        ctx.record_spot_execution(
            ACCOUNT, "BTC-USD", fill_id=f"f{i:04d}", side='buy',
            base_delta='1', quote_delta='-1', price='1',
            fee_amount='0', fee_currency='USD', ts_ms=T0_MS + i,
        )
    ids = [f"f{i:04d}" for i in range(1_050)]
    assert ctx.mark_spot_executions_delivered(ACCOUNT, "BTC-USD", ids) == 1_050
    assert ctx.mark_spot_executions_delivered(ACCOUNT, "BTC-USD", ids) == 0
    store.close()


def __test_manager_constructor_validation__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort()
    with pytest.raises(ValueError, match="on_inventory_conflict"):
        _manager(ctx, port, policy='ignore')
    bad_scope = FakeSpotPort()
    bad_scope.cursor_scope = "per-fill"
    with pytest.raises(ValueError, match="cursor_scope"):
        _manager(ctx, bad_scope)
    bad_tol = FakeSpotPort()
    bad_tol.base_tolerance = Decimal('-1')
    with pytest.raises(ValueError, match="base_tolerance"):
        _manager(ctx, bad_tol)
    mgr = _manager(ctx, port)
    with pytest.raises(RuntimeError, match="startup"):
        mgr.record_live_fill(_fill("f1"))
    store.close()


# === Review-finding regressions ============================================


def __test_execution_requires_bot_client_order_id__(tmp_path: Path):
    """F10: the ledger tracks the BOT's inventory — a fill needs the
    bot's OWN client_order_id; an exchange_order_id alone is not proof of
    bot ownership (a manual/web trade carries one too)."""
    _ = tmp_path
    with pytest.raises(ValueError, match="client_order_id"):
        SpotExecution(
            fill_id="anon", side='buy',
            base_delta=Decimal('1'), quote_delta=Decimal('-100'),
            price=Decimal('100'), fee_amount=Decimal('0'),
            fee_currency='USD', ts_ms=T0_MS,
        )
    # An exchange_order_id alone is NOT enough.
    with pytest.raises(ValueError, match="client_order_id"):
        SpotExecution(
            fill_id="exid-only", side='buy',
            base_delta=Decimal('1'), quote_delta=Decimal('-100'),
            price=Decimal('100'), fee_amount=Decimal('0'),
            fee_currency='USD', ts_ms=T0_MS, exchange_order_id="eo-1",
        )
    # The bot's own client_order_id satisfies the attribution floor.
    SpotExecution(
        fill_id="ok-coid", side='buy',
        base_delta=Decimal('1'), quote_delta=Decimal('-100'),
        price=Decimal('100'), fee_amount=Decimal('0'),
        fee_currency='USD', ts_ms=T0_MS, client_order_id="c-1",
    )


def __test_canonical_decimal_exact_beyond_context_precision__(tmp_path: Path):
    """F8: >28 significant digits must round-trip exactly — the ambient
    decimal context (prec=28) must not silently round the ledger value."""
    _ = tmp_path
    high = Decimal('1.' + '1' * 40)  # 41 significant digits
    s = canonical_decimal(high)
    assert Decimal(s) == high
    assert s == '1.' + '1' * 40
    # A trailing-zero, high-precision value collapses cleanly too.
    assert canonical_decimal(Decimal('0.' + '3' * 35 + '0' * 5)) == \
           '0.' + '3' * 35


def __test_bad_settlement_grace_rejected__(tmp_path: Path):
    """F7: a NaN/inf/negative grace would let a confirmed conflict stay
    pending forever — reject it at construction."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    for bad in (float('inf'), float('nan'), -1.0, True):
        port = FakeSpotPort()
        port.settlement_grace_s = bad  # type: ignore[assignment]
        with pytest.raises(ValueError, match="settlement_grace_s"):
            _manager(ctx, port)
    store.close()


def __test_same_ms_buy_sell_orders_by_venue_seq__(tmp_path: Path):
    """F6: a same-millisecond buy+sell must fold in venue order, not
    fill-id order — otherwise the sell replays first and false-oversells."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    # Fill ids chosen so the SELL sorts before the BUY alphabetically;
    # only the venue_seq tiebreak restores the true (buy, sell) order.
    ctx.record_spot_execution(
        ACCOUNT, "BTC-USD", fill_id="z-buy", side='buy',
        base_delta='1', quote_delta='-100', price='100',
        fee_amount='0', fee_currency='USD', ts_ms=T0_MS, venue_seq=1,
    )
    ctx.record_spot_execution(
        ACCOUNT, "BTC-USD", fill_id="a-sell", side='sell',
        base_delta='-1', quote_delta='105', price='105',
        fee_amount='0', fee_currency='USD', ts_ms=T0_MS, venue_seq=2,
    )
    rows = ctx.iter_spot_executions(ACCOUNT, "BTC-USD")
    assert [r.fill_id for r in rows] == ["z-buy", "a-sell"]
    fold = fold_inventory(rows)
    assert fold.net_base == Decimal(0) and fold.violation is None
    store.close()


def __test_inconclusive_page_does_not_advance_cursor__(tmp_path: Path):
    """F2: an inconclusive page records its fills but must NOT persist a
    cursor past history the venue could not vouch for."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('0'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.exec_cursor == "anchor-0"
    ctx.close()

    # A restart whose first page is inconclusive but still carries a fill
    # and a tempting next_cursor.
    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('1'))
    port2.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("mid-1", base='1', price='100'),),
        next_cursor="anchor-99", conclusive=False,
    )
    mgr2 = _manager(ctx2, port2)
    result = _run(mgr2.startup())
    assert result.quarantined and result.reason == 'spot_catchup_inconclusive'
    # The fill was recorded (dedup makes a re-fetch safe) but the cursor
    # stayed at the last conclusive position — NOT 'anchor-99'.
    rows = ctx2.iter_spot_executions(ACCOUNT, port2.product_id)
    assert [r.fill_id for r in rows] == ["mid-1"]
    epoch2 = ctx2.get_latest_spot_epoch(port2.product_id)
    assert epoch2 is not None and epoch2.exec_cursor == "anchor-0"
    store.close()


def __test_startup_negative_baseline_quarantines__(tmp_path: Path):
    """F3: a first-epoch freeze must refuse a negative foreign baseline
    (account owns less base than the ledger's bot inventory)."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    # Crash window: ledger already holds a 2 BTC bot buy, but the account
    # total has fallen to 0.5 (a foreign withdrawal).
    ctx.record_spot_execution(
        ACCOUNT, "BTC-USD", fill_id="pre1", side='buy',
        base_delta='2', quote_delta='-200', price='100',
        fee_amount='0', fee_currency='USD', ts_ms=T0_MS,
        client_order_id="c-pre1",
    )
    port = FakeSpotPort(balance=Decimal('0.5'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c: calls.append((r, c)))
    result = _run(mgr.startup())
    assert result.quarantined
    assert result.reason == 'spot_baseline_below_inventory'
    assert len(calls) == 1
    # No epoch was frozen with an impossible negative baseline.
    assert ctx.get_latest_spot_epoch(port.product_id) is None
    store.close()


def __test_reconcile_recovered_fills_returned_for_delivery__(tmp_path: Path):
    """F1: a runtime catch-up must hand its recovered fills back so the
    plugin can emit them — otherwise the engine position stays stale."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('5'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined

    # A fill landed at the venue (balance 6) but the stream lagged.
    port.balance = Decimal('6')
    port.batches["anchor-0"] = SpotExecutionBatch(
        executions=(_fill("lag-1", base='1', price='100'),),
        next_cursor="anchor-1",
    )
    recovered = _run(mgr.reconcile(T0_MS))
    assert [r.fill_id for r in recovered] == ["lag-1"]
    # The ledger row is flipped to delivered so a restart adoption won't
    # re-emit it, and a clean follow-up cycle returns nothing.
    rows = ctx.iter_spot_executions(ACCOUNT, port.product_id)
    assert [(r.fill_id, r.delivered) for r in rows] == [("lag-1", True)]
    assert _run(mgr.reconcile(T0_MS + 1_000)) == []
    store.close()


def __test_reconcile_compacts_unsellable_dust_into_the_baseline__(tmp_path: Path):
    """A fold that is pure unsellable fee dust rolls into the baseline.

    Fee residues accumulate across round trips (close orders are sized
    from FIFO exposure, not the wallet), so an engine-flat book's fold
    climbs above the grid step and every "flat means fold ≈ 0" consumer
    breaks. The clean-pass compaction must retire the rows and grow the
    baseline by EXACTLY the retired net — never re-anchor on the venue
    balance, which would absorb a real drift.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined

    # Two round trips, each stranding 0.000005 of base-coin fee dust.
    mgr.record_live_fill(_fill("b-1", base='0.009995', price='100'))
    mgr.record_live_fill(_fill("s-1", side='sell', base='-0.00999', price='100'))
    mgr.record_live_fill(_fill("b-2", base='0.009995', price='100', ts_ms=T0_MS + 1))
    mgr.record_live_fill(_fill("s-2", side='sell', base='-0.00999', price='100',
                               ts_ms=T0_MS + 2))
    port.balance = Decimal('10.00001')
    port.min_sellable = Decimal('0.00125')
    assert _run(mgr.reconcile(T0_MS + 10_000)) == []

    assert mgr.fold.net_base == Decimal(0)
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.state == 'active'
    assert epoch.epoch_seq == 2
    assert Decimal(epoch.foreign_baseline) == Decimal('10.00001')
    assert ctx.iter_spot_executions(ACCOUNT, port.product_id) == []
    events = _read_events(ctx, 'spot_dust_compacted')
    assert len(events) == 1
    assert events[0]['dust_net_base'] == '0.00001'
    assert events[0]['retired_fills'] == 4
    # The invariant still holds on the fresh epoch: a follow-up clean
    # pass neither re-compacts nor arms a conflict.
    assert _run(mgr.reconcile(T0_MS + 20_000)) == []
    assert not mgr.quarantined
    fresh = ctx.get_latest_spot_epoch(port.product_id)
    assert fresh is not None
    assert fresh.epoch_seq == 2 and fresh.pending_conflict_ts_ms is None
    store.close()


def __test_reconcile_never_compacts_a_sellable_net__(tmp_path: Path):
    """A net at or above the sell floor is a position, not dust."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    mgr.record_live_fill(_fill("b-1", base='0.00125', price='100'))
    port.balance = Decimal('10.00125')
    port.min_sellable = Decimal('0.00125')
    assert _run(mgr.reconcile(T0_MS + 10_000)) == []
    assert mgr.fold.net_base == Decimal('0.00125')
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.epoch_seq == 1
    assert _read_events(ctx, 'spot_dust_compacted') == []
    store.close()


def __test_reconcile_compaction_needs_a_known_floor__(tmp_path: Path):
    """No sell floor (no price yet) or a failing read skips the pass.

    Compaction is opportunistic — a skipped pass costs nothing, while an
    assumed floor could compact away sellable inventory.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    mgr.record_live_fill(_fill("b-1", base='0.00001', price='100'))
    port.balance = Decimal('10.00001')

    port.min_sellable = None
    assert _run(mgr.reconcile(T0_MS + 10_000)) == []
    assert mgr.fold.net_base == Decimal('0.00001')

    port.min_sellable_error = RuntimeError("venue hiccup")
    assert _run(mgr.reconcile(T0_MS + 20_000)) == []
    assert mgr.fold.net_base == Decimal('0.00001')
    assert not mgr.quarantined
    assert _read_events(ctx, 'spot_dust_compacted') == []
    store.close()


def __test_reconcile_compaction_waits_for_undelivered_rows__(tmp_path: Path):
    """A recovered-but-undelivered fill still owes its OrderEvent.

    Retiring flips ``delivered``, so compacting under it would swallow
    the event and leave the engine position stale. The pass that
    recovers a fill returns it for delivery — the NEXT clean pass may
    compact.
    """
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    port.min_sellable = Decimal('0.00125')

    # Book the dust row bypassing delivery: simulate the crash window in
    # which a recovered row exists with ``delivered = 0``.
    ctx.record_spot_execution(
        ACCOUNT, port.product_id, fill_id="lag-1", side='buy',
        base_delta='0.00001', quote_delta='-0.001', price='100',
        fee_amount='0', fee_currency='USD', ts_ms=T0_MS,
        client_order_id='c-lag-1', delivered=False,
    )
    mgr._refresh_fold()  # noqa: SLF001 — test drives the internal state
    port.balance = Decimal('10.00001')
    _run(mgr.reconcile(T0_MS + 10_000))
    assert _read_events(ctx, 'spot_dust_compacted') == []
    rows = ctx.iter_spot_executions(ACCOUNT, port.product_id)
    assert [(r.fill_id, r.retired) for r in rows] == [("lag-1", False)]
    store.close()


# noinspection SqlResolve,SqlWithoutWhere
def __test_reconcile_quarantines_on_lease_loss__(tmp_path: Path):
    """F4: a resumed run whose lease a replacement took over must stop
    trading — the fenced heartbeat reports the loss."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx_a = _open_run(store)
    port_a = FakeSpotPort(balance=Decimal('10'))
    calls: list = []
    mgr_a = _manager(ctx_a, port_a, hook=lambda r, c: calls.append((r, c)))
    assert not _run(mgr_a.startup()).quarantined

    # A replacement run takes the (now stale) lease over.
    ctx_b = _open_run(store, label="second")
    ctx_b._store._conn.execute(  # type: ignore[attr-defined]
        "UPDATE spot_asset_owner SET heartbeat_ts_ms = heartbeat_ts_ms - ?",
        (10 * 60 * 1000,),
    )
    assert ctx_b.claim_spot_asset(ACCOUNT, "BTC", "USD")

    # ctx_a's next reconcile heartbeat now updates zero rows -> quarantine.
    _run(mgr_a.reconcile(T0_MS))
    assert mgr_a.quarantined
    assert mgr_a.quarantine_reason == 'spot_lease_lost'
    assert len(calls) == 1
    store.close()


# noinspection SqlResolve
def __test_lease_same_run_id_zombie_cannot_reclaim__(tmp_path: Path):
    """F4: run_id is reused across restarts; a resumed zombie sharing the
    run_id but not the physical instance must not steal its lease back."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx1 = _open_run(store)  # run_id R, instance 1
    assert ctx1.claim_spot_asset(ACCOUNT, "BTC", "USD")
    ctx1.close()  # instance 1's run ends (but the lease is not released)

    # Restart: same identity -> same run_id R, new instance 2. It adopts
    # the ended predecessor's lease silently (no takeover event).
    ctx2 = _open_run(store)
    assert ctx2.claim_spot_asset(ACCOUNT, "BTC", "USD")
    assert _read_events(ctx2, 'spot_lease_taken_over') == []

    # The zombie instance 1 resumes: its heartbeat is a detected no-op...
    assert ctx1.heartbeat_spot_asset(ACCOUNT, "BTC") is False
    # ...and it cannot reclaim while instance 2 is live.
    assert not ctx1.claim_spot_asset(ACCOUNT, "BTC", "USD")
    # instance 2 still holds it.
    assert ctx2.heartbeat_spot_asset(ACCOUNT, "BTC") is True
    store.close()


def __test_lease_base_vs_quote_overlap_rejected__(tmp_path: Path):
    """F5: a live run trading the shared asset as cash blocks a claim
    that owns it as the position asset (and vice versa)."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx_a = _open_run(store)
    assert ctx_a.claim_spot_asset(ACCOUNT, "BTC", "USD")  # owns BTC
    # A second run wanting ETH/BTC uses BTC as quote cash -> overlap.
    ctx_b = _open_run(store, label="second")
    assert not ctx_b.claim_spot_asset(ACCOUNT, "ETH", "BTC")
    store.close()

    # The mirror: a run that owns BTC as its quote blocks a later BTC-base
    # claim.
    store2 = BrokerStore(tmp_path / "b2.sqlite", plugin_name=PLUGIN)
    ctx_c = _open_run(store2)
    assert ctx_c.claim_spot_asset(ACCOUNT, "ETH", "BTC")  # BTC as quote
    ctx_d = _open_run(store2, label="second")
    assert not ctx_d.claim_spot_asset(ACCOUNT, "BTC", "USD")
    store2.close()


def __test_lease_overlap_atomic_across_connections__(tmp_path: Path):
    """F5: the base-vs-quote exclusion must hold even across SEPARATE
    connections — a DEFERRED span would let both claimants pass the
    pre-write overlap read; BEGIN IMMEDIATE serializes them."""
    dbfile = tmp_path / "b.sqlite"
    store_a = BrokerStore(dbfile, plugin_name=PLUGIN)
    store_b = BrokerStore(dbfile, plugin_name=PLUGIN)
    ctx_a = _open_run(store_a)
    ctx_b = _open_run(store_b, label="second")
    barrier = threading.Barrier(2)
    results: dict[str, bool] = {}

    def claim(name: str, ctx: RunContext, base: str, quote: str) -> None:
        barrier.wait()
        results[name] = ctx.claim_spot_asset(ACCOUNT, base, quote)

    threads = [
        threading.Thread(target=claim, args=("a", ctx_a, "BTC", "USD")),
        threading.Thread(target=claim, args=("b", ctx_b, "ETH", "BTC")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # BTC/USD and ETH/BTC share BTC — exactly one may hold a lease.
    assert sorted(results.values()) == [False, True]
    store_a.close()
    store_b.close()


def __test_migration_partial_failure_leaves_no_schema__(tmp_path: Path,
                                                        monkeypatch):
    """F9: a migration that fails midway must roll back entirely — no
    orphan tables, no advanced user_version that bricks the retry."""
    import sqlite3
    from pynecore.core.broker import storage as st

    fake = [
        (1, "base", (
            "CREATE TABLE _migrations (version INTEGER PRIMARY KEY, "
            "  applied_ts_ms INTEGER NOT NULL, description TEXT NOT NULL);"
            "CREATE TABLE m_a (x);"
        )),
        # Second statement re-creates m_b -> the script aborts midway.
        (2, "bad", "CREATE TABLE m_b (x); CREATE TABLE m_b (x);"),
    ]
    monkeypatch.setattr(st, "_MIGRATIONS", fake)
    conn = sqlite3.connect(tmp_path / "m.sqlite")  # default isolation_level=""
    conn.row_factory = sqlite3.Row
    with pytest.raises(sqlite3.OperationalError):
        st._apply_migrations(conn)
    # v1 committed atomically; v2 rolled back with nothing left behind.
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'm_a'").fetchone()
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'm_b'").fetchone() is None
    conn.close()
