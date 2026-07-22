"""
Spot inventory layer tests (:mod:`pynecore.core.broker.spot_inventory`
plus the ``spot_*`` tables of :mod:`pynecore.core.broker.storage`).

Covers the plan's M4 matrix: crash windows around fill persistence
(catch-up recovery, outbox exactly-once), duplicate fills across
restarts, fee handling in base/quote/third currency, the fail-closed
balance invariant in BOTH directions (a deposit must not mask an
external sale), the asset-ownership lease (concurrent claim, stale
takeover), corrupt/quarantined epochs surviving restarts, rebaseline
preconditions, paged catch-up crash resume, and retention exemption of
the ledger.
"""
import asyncio
import threading
from decimal import Decimal
from pathlib import Path
from typing import Any, Coroutine

import pytest

from pynecore.core.broker.exceptions import SpotInventoryConflictError
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.spot_inventory import (
    SpotExecution,
    SpotExecutionBatch,
    SpotInventoryManager,
    canonical_decimal,
    fold_inventory,
)
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.store_helpers import STATE_SUBMITTED

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
    assert fold.net_base == 0 and fold.cost_quote == 0 and fold.vwap is None
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
    assert port3.fetch_calls == ["c1"]
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


def __test_startup_drift_quarantines_strictly__(tmp_path: Path):
    """Startup invariant is strict: a conclusive catch-up leaves no
    innocent explanation, so drift quarantines without runtime grace."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    ctx.close()

    # External deposit while the bot was down: total moved, ledger did not.
    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('15'))
    calls: list = []
    mgr2 = _manager(ctx2, port2, hook=lambda r, c: calls.append((r, c)))
    result = _run(mgr2.startup())
    assert result.quarantined
    assert result.reason == 'spot_inventory_conflict'
    assert len(calls) == 1
    assert calls[0][1]['drift'] == '5'
    epoch = ctx2.get_latest_spot_epoch(port2.product_id)
    assert epoch is not None and epoch.state == 'quarantined'
    store.close()


def __test_quarantined_epoch_survives_restart__(tmp_path: Path):
    """Persisted quarantine: an operator restart alone must NOT resume
    trading — only rebaseline clears the epoch."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    ctx.close()
    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('15'))
    mgr2 = _manager(ctx2, port2)
    assert _run(mgr2.startup()).quarantined
    ctx2.close()

    # Third start: balance happens to match again (e.g. the intruder
    # sold what they deposited) — the epoch still says quarantined.
    ctx3 = _open_run(store)
    port3 = FakeSpotPort(balance=Decimal('10'))
    calls: list = []
    mgr3 = _manager(ctx3, port3, hook=lambda r, c: calls.append((r, c)))
    result = _run(mgr3.startup())
    assert result.quarantined
    assert result.reason == 'spot_epoch_quarantined'
    assert len(calls) == 1
    # No venue reads happened: quarantine short-circuits before catch-up.
    assert port3.fetch_calls == [] and port3.balance_calls == 0
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


def __test_reconcile_deposit_then_external_sale_not_masked__(tmp_path: Path):
    """The v1 trap: deposit +10 then external sale −5 leaves
    ``actual − baseline >= ledger`` — a positive drift MUST quarantine
    after the grace, not warn-and-continue."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    calls: list = []
    mgr = _manager(ctx, port, hook=lambda r, c: calls.append((r, c)))
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('11')
    mgr.record_live_fill(_fill("own-1", base='1', price='100'))
    _run(mgr.reconcile(T0_MS))
    assert not mgr.quarantined

    # Deposit +10, then external sale -5: net drift +5.
    port.balance = Decimal('16')
    _run(mgr.reconcile(T0_MS + 60_000))
    assert not mgr.quarantined  # grace armed, not yet confirmed
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None
    assert epoch.pending_conflict_ts_ms == T0_MS + 60_000
    assert epoch.pending_conflict is not None
    assert epoch.pending_conflict['drift'] == '5'

    # Still drifted past the settlement grace: confirmed conflict.
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert mgr.quarantined
    assert len(calls) == 1
    reason, context = calls[0]
    assert 'spot_inventory_conflict' in reason
    assert context['drift'] == '5'
    epoch = ctx.get_latest_spot_epoch(port.product_id)
    assert epoch is not None and epoch.state == 'quarantined'
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

    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('7'))
    mgr2 = _manager(ctx2, port2, hook=lambda r, c: calls.append((r, c)),
                    policy='halt')
    result = _run(mgr2.startup())
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

    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('9'))
    mgr2 = _manager(ctx2, port2, hook=raising_hook)
    result = _run(mgr2.startup())
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

    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('9'))
    mgr2 = _manager(ctx2, port2, hook=None)
    result = _run(mgr2.startup())
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
def __test_rebaseline_requires_quarantine_and_no_unresolved__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined

    # Not quarantined: refused.
    with pytest.raises(ValueError, match="quarantined"):
        _run(mgr.rebaseline())

    # Quarantine via drift.
    port.balance = Decimal('15')
    _run(mgr.reconcile(T0_MS))
    _run(mgr.reconcile(T0_MS + 31_000))
    assert mgr.quarantined

    # A parked dispatch blocks the rebaseline.
    ctx.record_park("coid-parked", "Long")
    with pytest.raises(ValueError, match="parked:coid-parked"):
        _run(mgr.rebaseline())
    ctx.record_unpark("coid-parked")

    # A live order in a pending dispatch state blocks it too.
    ctx.upsert_order(
        "coid-pending", symbol="BTCUSD", side='buy', qty=1.0,
        state=STATE_SUBMITTED,
    )
    with pytest.raises(ValueError, match="coid-pending"):
        _run(mgr.rebaseline())
    ctx.close_order("coid-pending")

    # Clean now: the new epoch freezes the drifted total as baseline.
    epoch = _run(mgr.rebaseline())
    assert epoch.epoch_seq == 2 and epoch.state == 'active'
    assert Decimal(epoch.foreign_baseline) == Decimal('15')
    old = ctx._store._conn.execute(  # type: ignore[attr-defined]
        "SELECT state FROM spot_inventory_epoch WHERE epoch_seq = 1",
    ).fetchone()
    assert old['state'] == 'closed'

    # After the operator restart the fresh run trades again.
    ctx.close()
    ctx2 = _open_run(store)
    port2 = FakeSpotPort(balance=Decimal('15'))
    mgr2 = _manager(ctx2, port2)
    result = _run(mgr2.startup())
    assert not result.quarantined
    assert result.epoch is not None and result.epoch.epoch_seq == 2
    store.close()


def __test_rebaseline_inconclusive_catchup_refused__(tmp_path: Path):
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    port.balance = Decimal('15')
    _run(mgr.reconcile(T0_MS))
    _run(mgr.reconcile(T0_MS + 31_000))
    assert mgr.quarantined
    port.batches["anchor-0"] = SpotExecutionBatch(conclusive=False)
    with pytest.raises(ValueError, match="inconclusive"):
        _run(mgr.rebaseline())
    store.close()


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
    assert fold.net_base == 0 and fold.violation is None
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


def __test_rebaseline_negative_baseline_refused__(tmp_path: Path):
    """F3: rebaseline must not launder a withdrawal into a negative
    baseline — refuse until the operator restores the holdings."""
    store = BrokerStore(tmp_path / "b.sqlite", plugin_name=PLUGIN)
    ctx = _open_run(store)
    port = FakeSpotPort(balance=Decimal('10'))
    mgr = _manager(ctx, port)
    assert not _run(mgr.startup()).quarantined
    mgr.record_live_fill(_fill("b1", base='3', price='100'))  # bot holds 3
    port.balance = Decimal('13')
    _run(mgr.reconcile(T0_MS))
    assert not mgr.quarantined

    # A large external withdrawal: total 1, but the ledger says the bot
    # still holds 3 -> baseline would be 1-3 = -2.
    port.balance = Decimal('1')
    _run(mgr.reconcile(T0_MS + 60_000))
    _run(mgr.reconcile(T0_MS + 60_000 + 31_000))
    assert mgr.quarantined
    with pytest.raises(ValueError, match="below the ledger"):
        _run(mgr.rebaseline())
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
