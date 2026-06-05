"""DispatchJournal close / cancel / modify path contract tests.

Mirrors ``test_031_broker_dispatch_journal`` for the new M4 paths:

- ``run_close`` (full and partial branches)
- ``run_cancel``
- ``run_modify_entry``
- ``run_modify_exit``

Each test exercises a single dispatch lifecycle through the journal
against a real :class:`BrokerStore` and asserts the on-disk shape:
state, ``extras``, ``order_refs``, and the audit event sequence.

The hooks are fakes that return canned outcomes (or raise canned
exceptions) — no exchange is involved. The point is to pin the
journal's persist-first contract, not to test plugin code.
"""
import asyncio
import json
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pynecore.core.broker.exceptions import (
    ExchangeOrderRejectedError,
    OrderDispositionUnknownError,
)
from pynecore.core.broker.journal import (
    CancelOutcome,
    CloseOutcome,
    DispatchJournal,
    ModifyEntryOutcome,
    ModifyExitOutcome,
)
from pynecore.core.broker.models import (
    CancelIntent,
    CloseIntent,
    EntryIntent,
    ExchangeOrder,
    ExitIntent,
    OrderStatus,
    OrderType,
)
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, OrderRow, RunContext
from pynecore.core.broker.store_helpers import (
    KIND_CANCEL,
    KIND_FULL_CLOSE,
    KIND_MODIFY_ENTRY,
    KIND_MODIFY_EXIT,
    KIND_PARTIAL_CLOSE,
    STATE_CLOSING,
    STATE_CONFIRMED,
    STATE_DISPOSITION_UNKNOWN,
    STATE_REJECTED,
    STATE_SERVER_REF_SEEN,
    STATE_SUBMITTED,
)


PLUGIN = "TestBroker"
SCRIPT_SOURCE = "// dispatch journal extended path tests\n"
RUN_TAG = "test"
BAR_TS_MS = 1_700_000_000_000


# === Helpers ===============================================================

def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)


def _open_run(store: BrokerStore) -> RunContext:
    return store.open_run(
        RunIdentity(
            strategy_id="dj_ext_test", symbol="EURUSD", timeframe="60",
            account_id="testbroker-demo", label=None,
        ),
        script_source=SCRIPT_SOURCE,
        script_path="strategies/dj_ext_test.py",
    )


def _coid(ctx: RunContext, pine_id: str, kind_char: str = 'c') -> str:
    """Build a deterministic COID for these tests."""
    return f"{ctx.run_instance_id}:{kind_char}:{pine_id}"


def _make_close_intent(*, symbol: str = 'EURUSD', side: str = 'sell',
                       qty: float = 1.0, pine_id: str = 'Long') -> CloseIntent:
    return CloseIntent(
        pine_id=pine_id, symbol=symbol, side=side, qty=qty,
    )


def _make_cancel_intent(*, symbol: str = 'EURUSD', pine_id: str = 'Long',
                       from_entry: str | None = None) -> CancelIntent:
    return CancelIntent(
        pine_id=pine_id, symbol=symbol, from_entry=from_entry,
    )


def _make_entry_intent(*, pine_id: str = 'Long', symbol: str = 'EURUSD',
                       side: str = 'buy', qty: float = 1.0,
                       order_type: OrderType = OrderType.LIMIT,
                       limit: float | None = 1.10,
                       stop: float | None = None) -> EntryIntent:
    return EntryIntent(
        pine_id=pine_id, symbol=symbol, side=side, qty=qty,
        order_type=order_type, limit=limit, stop=stop,
    )


def _make_exit_intent(*, pine_id: str = 'TP_SL', from_entry: str = 'Long',
                     symbol: str = 'EURUSD', side: str = 'sell',
                     qty: float = 1.0,
                     tp: float | None = 1.15, sl: float | None = 1.05,
                     trail: float | None = None) -> ExitIntent:
    return ExitIntent(
        pine_id=pine_id, from_entry=from_entry, symbol=symbol, side=side,
        qty=qty, tp_price=tp, sl_price=sl, trail_offset=trail,
    )


def _read_events(ctx: RunContext, coid: str) -> list[tuple[str, dict | None]]:
    """Return the (event_kind, payload) list for one COID in order."""
    rows = ctx._store._conn.execute(  # type: ignore[attr-defined]
        "SELECT kind, payload FROM events "
        "WHERE run_instance_id = ? AND client_order_id = ? "
        "ORDER BY ts_ms ASC, id ASC",
        (ctx.run_instance_id, coid),
    ).fetchall()
    out: list[tuple[str, dict | None]] = []
    for r in rows:
        payload = json.loads(r['payload']) if r['payload'] else None
        out.append((r['kind'], payload))
    return out


def _seed_target_row(ctx: RunContext, *, coid: str, symbol: str = 'EURUSD',
                     side: str = 'buy', qty: float = 1.0,
                     state: str = STATE_CONFIRMED,
                     exchange_order_id: str | None = 'DEAL-1',
                     pine_entry_id: str = 'Long',
                     intent_key: str = 'Long',
                     kind_extra: str = 'position') -> OrderRow:
    ctx.upsert_order(
        coid,
        symbol=symbol, side=side, qty=qty, state=state,
        intent_key=intent_key, pine_entry_id=pine_entry_id,
        exchange_order_id=exchange_order_id,
        extras={'kind': kind_extra},
    )
    return ctx.get_order(coid)  # type: ignore[return-value]


# === Fake hooks ============================================================

@dataclass
class _FakeFullCloseHooks:
    """Fake :class:`CloseDispatchHooks` returning canned full-close outcomes."""
    full_outcome: CloseOutcome | None = None
    full_exc: BaseException | None = None
    calls: list[str] = field(default_factory=list)

    async def submit_full_close(
            self, *, coid: str, intent: CloseIntent,
            targets: list[OrderRow],
    ) -> CloseOutcome:
        self.calls.append(f"full:{coid}:{len(targets)}")
        if self.full_exc is not None:
            raise self.full_exc
        assert self.full_outcome is not None
        return self.full_outcome

    async def submit_partial_close(
            self, *, coid: str, intent: CloseIntent,
    ) -> CloseOutcome:  # pragma: no cover — defensive
        raise RuntimeError("submit_partial_close not expected in full tests")

    def exchange_order_from_state(
            self, *, row: OrderRow, intent: CloseIntent,
            outcome: CloseOutcome,
    ) -> ExchangeOrder:
        return ExchangeOrder(
            id=outcome.exchange_id or '',
            symbol=intent.symbol, side=intent.side,
            order_type=OrderType.MARKET,
            qty=intent.qty, filled_qty=outcome.filled_qty,
            remaining_qty=max(0.0, intent.qty - outcome.filled_qty),
            price=None, stop_price=None,
            average_fill_price=outcome.fill_price,
            status=OrderStatus.FILLED, timestamp=0.0,
            fee=0.0, fee_currency='',
            reduce_only=True, client_order_id=row.client_order_id,
        )


@dataclass
class _FakePartialCloseHooks:
    partial_outcome: CloseOutcome | None = None
    partial_exc: BaseException | None = None
    calls: list[str] = field(default_factory=list)

    async def submit_full_close(
            self, *, coid: str, intent: CloseIntent,
            targets: list[OrderRow],
    ) -> CloseOutcome:  # pragma: no cover — defensive
        raise RuntimeError("submit_full_close not expected in partial tests")

    async def submit_partial_close(
            self, *, coid: str, intent: CloseIntent,
    ) -> CloseOutcome:
        self.calls.append(f"partial:{coid}")
        if self.partial_exc is not None:
            raise self.partial_exc
        assert self.partial_outcome is not None
        return self.partial_outcome

    def exchange_order_from_state(
            self, *, row: OrderRow, intent: CloseIntent,
            outcome: CloseOutcome,
    ) -> ExchangeOrder:
        return ExchangeOrder(
            id=outcome.exchange_id or '',
            symbol=intent.symbol, side=intent.side,
            order_type=OrderType.MARKET,
            qty=intent.qty, filled_qty=outcome.filled_qty,
            remaining_qty=max(0.0, intent.qty - outcome.filled_qty),
            price=None, stop_price=None,
            average_fill_price=outcome.fill_price,
            status=OrderStatus.FILLED, timestamp=0.0,
            fee=0.0, fee_currency='',
            reduce_only=True, client_order_id=row.client_order_id,
        )


@dataclass
class _FakeCancelHooks:
    outcome: CancelOutcome | None = None
    exc: BaseException | None = None
    calls: list[str] = field(default_factory=list)

    async def submit_cancel(
            self, *, coid: str, intent: CancelIntent,
            targets: list[OrderRow],
    ) -> CancelOutcome:
        self.calls.append(f"cancel:{coid}:{len(targets)}")
        if self.exc is not None:
            raise self.exc
        assert self.outcome is not None
        return self.outcome

    def exchange_order_from_state(
            self, *, row: OrderRow, intent: CancelIntent,
            outcome: CancelOutcome,
    ) -> ExchangeOrder:
        return ExchangeOrder(
            id='', symbol=intent.symbol, side=row.side,
            order_type=OrderType.MARKET,
            qty=row.qty, filled_qty=0.0, remaining_qty=row.qty,
            price=None, stop_price=None,
            average_fill_price=None,
            status=OrderStatus.CANCELLED, timestamp=0.0,
            fee=0.0, fee_currency='',
            reduce_only=False, client_order_id=row.client_order_id,
        )


@dataclass
class _FakeModifyEntryHooks:
    outcome: ModifyEntryOutcome | None = None
    exc: BaseException | None = None
    calls: list[str] = field(default_factory=list)

    async def submit_amend(
            self, *, coid: str, target_coid: str,
            old_intent: EntryIntent, new_intent: EntryIntent,
    ) -> ModifyEntryOutcome:
        self.calls.append(f"amend_entry:{coid}->{target_coid}")
        if self.exc is not None:
            raise self.exc
        assert self.outcome is not None
        return self.outcome

    def exchange_order_from_state(
            self, *, row: OrderRow, new_intent: EntryIntent,
            outcome: ModifyEntryOutcome,
    ) -> list[ExchangeOrder]:
        return [
            ExchangeOrder(
                id='', symbol=new_intent.symbol, side=new_intent.side,
                order_type=new_intent.order_type,
                qty=row.qty, filled_qty=0.0, remaining_qty=row.qty,
                price=outcome.new_level, stop_price=None,
                average_fill_price=None,
                status=OrderStatus.OPEN, timestamp=0.0,
                fee=0.0, fee_currency='',
                reduce_only=False, client_order_id=row.client_order_id,
            ),
        ]


@dataclass
class _FakeModifyExitHooks:
    outcome: ModifyExitOutcome | None = None
    exc: BaseException | None = None
    calls: list[str] = field(default_factory=list)
    mirror_calls: list[str] = field(default_factory=list)

    async def submit_amend(
            self, *, coid: str, target_coid: str,
            old_intent: ExitIntent, new_intent: ExitIntent,
    ) -> ModifyExitOutcome:
        self.calls.append(f"amend_exit:{coid}->{target_coid}")
        if self.exc is not None:
            raise self.exc
        assert self.outcome is not None
        return self.outcome

    def mirror_bracket_legs(
            self, *, target_row: OrderRow, new_intent: ExitIntent,
            outcome: ModifyExitOutcome,
    ) -> None:
        self.mirror_calls.append(target_row.client_order_id)

    def exchange_order_from_state(
            self, *, row: OrderRow, new_intent: ExitIntent,
            outcome: ModifyExitOutcome,
    ) -> list[ExchangeOrder]:
        return []


# === Close path tests ======================================================

def __test_journal_full_close_happy_path__(tmp_path: Path) -> None:
    """Full close: command row → submitted → closing; targets are pinned."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            target_coid = _coid(ctx, "T1", kind_char='e')
            target_row = _seed_target_row(ctx, coid=target_coid)
            intent = _make_close_intent()
            coid = _coid(ctx, "Long")
            hooks = _FakeFullCloseHooks(
                full_outcome=CloseOutcome(
                    mode='full',
                    applied_targets=[target_row.exchange_order_id or ''],
                    deal_reference=None,
                    exchange_id=target_row.exchange_order_id,
                    filled_qty=1.0,
                    fill_price=None,
                    raw={'deleted': 1},
                ),
            )
            journal = DispatchJournal(ctx)
            result = _run(journal.run_close(
                coid=coid, intent=intent,
                kind=KIND_FULL_CLOSE, targets=[target_row],
                hooks=hooks,
            ))
            assert result.status == OrderStatus.FILLED

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_CLOSING
            assert row.extras['kind'] == KIND_FULL_CLOSE
            assert row.extras['targets'] == [target_row.exchange_order_id]
        finally:
            ctx.close()


def __test_journal_full_close_submit_timeout_marks_disposition_unknown__(
        tmp_path: Path) -> None:
    """Full close: submit timeout re-raises and marks the row disposition-unknown."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            target_coid = _coid(ctx, "T1", kind_char='e')
            target_row = _seed_target_row(ctx, coid=target_coid)
            intent = _make_close_intent()
            coid = _coid(ctx, "Long")
            hooks = _FakeFullCloseHooks(
                full_exc=OrderDispositionUnknownError(
                    "timeout during DELETE", coid,
                ),
            )
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_close(
                    coid=coid, intent=intent,
                    kind=KIND_FULL_CLOSE, targets=[target_row],
                    hooks=hooks,
                ))
            except OrderDispositionUnknownError:
                pass
            else:
                raise AssertionError("expected OrderDispositionUnknownError")

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_DISPOSITION_UNKNOWN
            kinds = [k for k, _ in _read_events(ctx, coid)]
            assert kinds == ['dispatch_submitted', 'disposition_unknown']
        finally:
            ctx.close()


def __test_journal_partial_close_happy_path__(tmp_path: Path) -> None:
    """Partial close: command row → submitted → server_ref → confirmed (closed)."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            intent = _make_close_intent(qty=0.5)
            coid = _coid(ctx, "Long")
            hooks = _FakePartialCloseHooks(
                partial_outcome=CloseOutcome(
                    mode='partial',
                    applied_targets=['DEAL-NEW'],
                    deal_reference='REF-PARTIAL',
                    exchange_id='DEAL-NEW',
                    filled_qty=0.5,
                    fill_price=1.105,
                    raw={'dealReference': 'REF-PARTIAL'},
                ),
            )
            journal = DispatchJournal(ctx)
            result = _run(journal.run_close(
                coid=coid, intent=intent,
                kind=KIND_PARTIAL_CLOSE, targets=[],
                hooks=hooks,
            ))
            assert result.id == 'DEAL-NEW'

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_CONFIRMED
            assert row.closed_ts_ms is not None
            assert row.extras['kind'] == KIND_PARTIAL_CLOSE
            assert row.extras['deal_reference'] == 'REF-PARTIAL'
            # ``mark_close_completed`` calls ``close_order(coid)`` which
            # also clears ``order_refs`` for the row; the audit event
            # is where the reference lives long-term. ``close_order``
            # also emits an ``order_closed`` event.
            events = _read_events(ctx, coid)
            kinds = [k for k, _ in events]
            assert kinds == [
                'dispatch_submitted', 'deal_reference_seen',
                'confirmed', 'order_closed',
            ]
            ref_event = [p for k, p in events if k == 'deal_reference_seen'][0]
            assert ref_event is not None
            assert ref_event['deal_reference'] == 'REF-PARTIAL'
        finally:
            ctx.close()


def __test_journal_partial_close_submit_reject_marks_rejected__(
        tmp_path: Path) -> None:
    """Partial close: submit reject re-raises and marks the row rejected."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            intent = _make_close_intent(qty=0.5)
            coid = _coid(ctx, "Long")
            hooks = _FakePartialCloseHooks(
                partial_exc=ExchangeOrderRejectedError("dealStatus=REJECTED"),
            )
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_close(
                    coid=coid, intent=intent,
                    kind=KIND_PARTIAL_CLOSE, targets=[],
                    hooks=hooks,
                ))
            except ExchangeOrderRejectedError:
                pass
            else:
                raise AssertionError("expected ExchangeOrderRejectedError")

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_REJECTED
        finally:
            ctx.close()


def __test_run_close_rejects_unknown_kind__(tmp_path: Path) -> None:
    """``run_close`` raises ``ValueError`` naming an unknown close ``kind``."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            intent = _make_close_intent()
            coid = _coid(ctx, "Long")
            hooks = _FakeFullCloseHooks()
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_close(
                    coid=coid, intent=intent,
                    kind='bogus_kind', targets=[],
                    hooks=hooks,
                ))
            except ValueError as exc:
                assert 'kind' in str(exc)
                assert 'bogus_kind' in str(exc)
            else:
                raise AssertionError("expected ValueError for bogus kind")
        finally:
            ctx.close()


# === Cancel path tests =====================================================

def __test_journal_cancel_happy_path_with_targets__(tmp_path: Path) -> None:
    """Cancel: command row → submitted → confirmed (closed) + reason_path."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            t1_coid = _coid(ctx, "T1", kind_char='w')
            t1_row = _seed_target_row(
                ctx, coid=t1_coid, kind_extra='working', state='confirmed',
            )
            intent = _make_cancel_intent()
            coid = _coid(ctx, "Long")
            hooks = _FakeCancelHooks(
                outcome=CancelOutcome(
                    succeeded=True,
                    reason_path='deleted',
                    cleared_legs=1,
                    applied_target_coids=[t1_coid],
                ),
            )
            journal = DispatchJournal(ctx)
            result = _run(journal.run_cancel(
                coid=coid, intent=intent, targets=[t1_row], hooks=hooks,
            ))
            assert result.status == OrderStatus.CANCELLED

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_CONFIRMED
            assert row.closed_ts_ms is not None
            assert row.extras['kind'] == KIND_CANCEL
            assert row.extras['target_coids'] == [t1_coid]
            assert row.extras['reason_path'] == 'deleted'
            assert row.extras['applied_target_coids'] == [t1_coid]
        finally:
            ctx.close()


def __test_journal_cancel_noop_when_no_targets__(tmp_path: Path) -> None:
    """Cancel with empty targets: command row still persisted, reason='noop'."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            intent = _make_cancel_intent()
            coid = _coid(ctx, "Long")
            hooks = _FakeCancelHooks(
                outcome=CancelOutcome(
                    succeeded=True, reason_path='noop',
                    cleared_legs=0, applied_target_coids=[],
                ),
            )
            journal = DispatchJournal(ctx)
            result = _run(journal.run_cancel(
                coid=coid, intent=intent, targets=[], hooks=hooks,
            ))
            assert result.status == OrderStatus.CANCELLED

            row = ctx.get_order(coid)
            assert row is not None
            assert row.extras['reason_path'] == 'noop'
        finally:
            ctx.close()


def __test_journal_cancel_already_gone_path__(tmp_path: Path) -> None:
    """Cancel records ``reason_path='already_gone'`` when the target was already cleared."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            t1_coid = _coid(ctx, "T1", kind_char='w')
            t1_row = _seed_target_row(
                ctx, coid=t1_coid, kind_extra='working',
            )
            intent = _make_cancel_intent()
            coid = _coid(ctx, "Long")
            hooks = _FakeCancelHooks(
                outcome=CancelOutcome(
                    succeeded=True, reason_path='already_gone',
                    cleared_legs=0, applied_target_coids=[],
                ),
            )
            journal = DispatchJournal(ctx)
            _run(journal.run_cancel(
                coid=coid, intent=intent, targets=[t1_row], hooks=hooks,
            ))

            row = ctx.get_order(coid)
            assert row is not None
            assert row.extras['reason_path'] == 'already_gone'
        finally:
            ctx.close()


def __test_journal_cancel_audit_event_sequence__(tmp_path: Path) -> None:
    """Cancel emits ``dispatch_submitted`` -> ``confirmed`` -> ``order_closed`` in order."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            t1_coid = _coid(ctx, "T1", kind_char='w')
            t1_row = _seed_target_row(
                ctx, coid=t1_coid, kind_extra='working',
            )
            intent = _make_cancel_intent()
            coid = _coid(ctx, "Long")
            hooks = _FakeCancelHooks(
                outcome=CancelOutcome(
                    succeeded=True, reason_path='deleted',
                    cleared_legs=1, applied_target_coids=[t1_coid],
                ),
            )
            journal = DispatchJournal(ctx)
            _run(journal.run_cancel(
                coid=coid, intent=intent, targets=[t1_row], hooks=hooks,
            ))
            kinds = [k for k, _ in _read_events(ctx, coid)]
            assert kinds == [
                'dispatch_submitted', 'confirmed', 'order_closed',
            ]
        finally:
            ctx.close()


# === Modify entry path tests ==============================================

def __test_journal_modify_entry_happy_path__(tmp_path: Path) -> None:
    """Modify entry: amend confirms the new level and closes the command row."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            target_coid = _coid(ctx, "T1", kind_char='e')
            _seed_target_row(
                ctx, coid=target_coid, kind_extra='working',
            )
            old_i = _make_entry_intent(limit=1.10)
            new_i = _make_entry_intent(limit=1.12)
            coid = _coid(ctx, "Long-mod")
            hooks = _FakeModifyEntryHooks(
                outcome=ModifyEntryOutcome(
                    server_ref='REF-AMEND-1',
                    new_level=1.12,
                    raw={'dealReference': 'REF-AMEND-1'},
                ),
            )
            journal = DispatchJournal(ctx)
            result = _run(journal.run_modify_entry(
                coid=coid, target_coid=target_coid,
                old_intent=old_i, new_intent=new_i,
                qty=1.0, hooks=hooks,
            ))
            assert result[0].price == 1.12

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_CONFIRMED
            assert row.closed_ts_ms is not None
            assert row.extras['kind'] == KIND_MODIFY_ENTRY
            assert row.extras['target_coid'] == target_coid
            assert row.extras['new_level'] == 1.12
            assert row.extras['echoed_level'] == 1.12

            events = _read_events(ctx, coid)
            kinds = [k for k, _ in events]
            assert kinds == [
                'dispatch_submitted', 'deal_reference_seen',
                'confirmed', 'order_closed',
            ]
            ref_event = [p for k, p in events if k == 'deal_reference_seen'][0]
            assert ref_event is not None
            assert ref_event['deal_reference'] == 'REF-AMEND-1'
        finally:
            ctx.close()


def __test_journal_modify_entry_reject_marks_rejected__(tmp_path: Path) -> None:
    """Modify entry: amend reject re-raises and marks the command row rejected."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            target_coid = _coid(ctx, "T1", kind_char='e')
            _seed_target_row(
                ctx, coid=target_coid, kind_extra='working',
            )
            old_i = _make_entry_intent(limit=1.10)
            new_i = _make_entry_intent(limit=1.12)
            coid = _coid(ctx, "Long-mod")
            hooks = _FakeModifyEntryHooks(
                exc=ExchangeOrderRejectedError("amend reject"),
            )
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_modify_entry(
                    coid=coid, target_coid=target_coid,
                    old_intent=old_i, new_intent=new_i,
                    qty=1.0, hooks=hooks,
                ))
            except ExchangeOrderRejectedError:
                pass
            else:
                raise AssertionError("expected reject")

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_REJECTED
        finally:
            ctx.close()


def __test_journal_modify_entry_timeout_marks_disposition_unknown__(
        tmp_path: Path) -> None:
    """Modify entry: amend timeout re-raises and marks the row disposition-unknown."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            target_coid = _coid(ctx, "T1", kind_char='e')
            _seed_target_row(
                ctx, coid=target_coid, kind_extra='working',
            )
            old_i = _make_entry_intent(limit=1.10)
            new_i = _make_entry_intent(limit=1.12)
            coid = _coid(ctx, "Long-mod")
            hooks = _FakeModifyEntryHooks(
                exc=OrderDispositionUnknownError("amend timeout", coid),
            )
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_modify_entry(
                    coid=coid, target_coid=target_coid,
                    old_intent=old_i, new_intent=new_i,
                    qty=1.0, hooks=hooks,
                ))
            except OrderDispositionUnknownError:
                pass
            else:
                raise AssertionError("expected disposition unknown")

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_DISPOSITION_UNKNOWN
        finally:
            ctx.close()


# === Modify exit path tests ===============================================

def __test_journal_modify_exit_happy_path_invokes_mirror__(tmp_path: Path) -> None:
    """Modify exit: amend confirms, invokes the mirror hook, and closes the command row."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            target_coid = _coid(ctx, "T1", kind_char='e')
            target_row = _seed_target_row(
                ctx, coid=target_coid, kind_extra='position',
            )
            old_i = _make_exit_intent(tp=1.15, sl=1.05)
            new_i = _make_exit_intent(tp=1.18, sl=1.04)
            coid = _coid(ctx, "TPSL-mod")
            hooks = _FakeModifyExitHooks(
                outcome=ModifyExitOutcome(
                    server_ref='REF-MEX-1',
                    deal_status='ACCEPTED',
                    rejected_reason=None,
                    post_put_state={
                        'profit_level': 1.18, 'stop_level': 1.04,
                    },
                    raw={'dealReference': 'REF-MEX-1'},
                ),
            )
            journal = DispatchJournal(ctx)
            result = _run(journal.run_modify_exit(
                coid=coid, target_coid=target_coid, target_row=target_row,
                old_intent=old_i, new_intent=new_i, qty=1.0, hooks=hooks,
            ))
            assert result == []
            assert hooks.mirror_calls == [target_coid]

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_CONFIRMED
            assert row.closed_ts_ms is not None
            assert row.extras['kind'] == KIND_MODIFY_EXIT
            assert row.extras['target_coid'] == target_coid
            assert row.extras['new_tp'] == 1.18
            assert row.extras['new_sl'] == 1.04
            assert row.extras['post_put_state'] == {
                'profit_level': 1.18, 'stop_level': 1.04,
            }

            kinds = [k for k, _ in _read_events(ctx, coid)]
            assert kinds == [
                'dispatch_submitted', 'deal_reference_seen',
                'confirmed', 'order_closed',
            ]
        finally:
            ctx.close()


def __test_journal_modify_exit_reject_skips_mirror__(tmp_path: Path) -> None:
    """Modify exit: amend reject re-raises, marks rejected, and skips the mirror hook."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            target_coid = _coid(ctx, "T1", kind_char='e')
            target_row = _seed_target_row(
                ctx, coid=target_coid, kind_extra='position',
            )
            old_i = _make_exit_intent(tp=1.15, sl=1.05)
            new_i = _make_exit_intent(tp=1.18, sl=1.04)
            coid = _coid(ctx, "TPSL-mod")
            hooks = _FakeModifyExitHooks(
                exc=ExchangeOrderRejectedError("modify_exit REJECTED"),
            )
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_modify_exit(
                    coid=coid, target_coid=target_coid, target_row=target_row,
                    old_intent=old_i, new_intent=new_i, qty=1.0, hooks=hooks,
                ))
            except ExchangeOrderRejectedError:
                pass
            else:
                raise AssertionError("expected reject")

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_REJECTED
            assert hooks.mirror_calls == []
        finally:
            ctx.close()


def __test_journal_modify_exit_timeout_skips_mirror__(tmp_path: Path) -> None:
    """Modify exit: amend timeout marks disposition-unknown and skips the mirror hook."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            target_coid = _coid(ctx, "T1", kind_char='e')
            target_row = _seed_target_row(
                ctx, coid=target_coid, kind_extra='position',
            )
            old_i = _make_exit_intent(tp=1.15, sl=1.05)
            new_i = _make_exit_intent(tp=1.18, sl=1.04)
            coid = _coid(ctx, "TPSL-mod")
            hooks = _FakeModifyExitHooks(
                exc=OrderDispositionUnknownError(
                    "modify_exit timeout", coid,
                ),
            )
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_modify_exit(
                    coid=coid, target_coid=target_coid, target_row=target_row,
                    old_intent=old_i, new_intent=new_i, qty=1.0, hooks=hooks,
                ))
            except OrderDispositionUnknownError:
                pass
            else:
                raise AssertionError("expected disposition unknown")

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_DISPOSITION_UNKNOWN
            assert hooks.mirror_calls == []
        finally:
            ctx.close()


# === State helper coverage tests (close / cancel / modify) ================

def __test_create_close_target_row_rejects_bogus_kind__(tmp_path: Path) -> None:
    """``create_close_target_row`` raises ``ValueError`` for a non-close ``kind``."""
    from pynecore.core.broker.store_helpers import create_close_target_row
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            try:
                create_close_target_row(
                    ctx, coid='C-X', symbol='EURUSD', side='sell', qty=1.0,
                    intent_key='Long', kind='not_a_close',
                )
            except ValueError as exc:
                assert 'not_a_close' in str(exc)
            else:
                raise AssertionError("expected ValueError")
        finally:
            ctx.close()


def __test_record_close_server_ref_rejects_full_close_kind__(
        tmp_path: Path) -> None:
    """``record_close_server_ref`` raises ``ValueError`` for a full-close ``kind``."""
    from pynecore.core.broker.store_helpers import (
        create_close_target_row, record_close_server_ref,
    )
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            create_close_target_row(
                ctx, coid='C-X', symbol='EURUSD', side='sell', qty=1.0,
                intent_key='Long', kind=KIND_FULL_CLOSE,
            )
            try:
                record_close_server_ref(
                    ctx, coid='C-X', deal_reference='R-1',
                    kind=KIND_FULL_CLOSE,
                )
            except ValueError as exc:
                assert KIND_FULL_CLOSE in str(exc)
            else:
                raise AssertionError("expected ValueError")
        finally:
            ctx.close()


def __test_mark_closing_rejects_partial_kind__(tmp_path: Path) -> None:
    """``mark_closing`` raises ``ValueError`` for a partial-close ``kind``."""
    from pynecore.core.broker.store_helpers import (
        create_close_target_row, mark_closing,
    )
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            create_close_target_row(
                ctx, coid='C-X', symbol='EURUSD', side='sell', qty=0.5,
                intent_key='Long', kind=KIND_PARTIAL_CLOSE,
            )
            try:
                mark_closing(
                    ctx, coid='C-X', kind=KIND_PARTIAL_CLOSE,
                    targets=['DEAL-1'],
                )
            except ValueError as exc:
                assert KIND_PARTIAL_CLOSE in str(exc)
            else:
                raise AssertionError("expected ValueError")
        finally:
            ctx.close()


def __test_mark_close_completed_rejects_bogus_kind__(tmp_path: Path) -> None:
    """``mark_close_completed`` raises ``ValueError`` for an unknown ``kind``."""
    from pynecore.core.broker.store_helpers import (
        create_close_target_row, mark_close_completed,
    )
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            create_close_target_row(
                ctx, coid='C-X', symbol='EURUSD', side='sell', qty=0.5,
                intent_key='Long', kind=KIND_PARTIAL_CLOSE,
            )
            try:
                mark_close_completed(ctx, coid='C-X', kind='nope')
            except ValueError as exc:
                assert 'nope' in str(exc)
            else:
                raise AssertionError("expected ValueError")
        finally:
            ctx.close()


def __test_pending_dispatch_states_excludes_new_states__(tmp_path: Path) -> None:
    """``STATE_CLOSING`` / ``STATE_CANCEL_PENDING`` are not pending in M4 phase 1.

    These states will join ``PENDING_DISPATCH_STATES`` only once the
    plugin-side recovery branches exist (M4 phase 3 / phase 4). The
    Core layer must keep the set narrow until then so the existing
    entry recovery hooks do not get fed close / cancel rows they
    cannot handle.
    """
    from pynecore.core.broker.store_helpers import (
        PENDING_DISPATCH_STATES, STATE_CANCEL_PENDING, STATE_CLOSING,
    )
    assert STATE_CLOSING not in PENDING_DISPATCH_STATES
    assert STATE_CANCEL_PENDING not in PENDING_DISPATCH_STATES
