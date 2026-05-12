"""
Standalone unit tests for the Core :class:`DispatchJournal` and the
typed :mod:`pynecore.core.broker.store_helpers` API.

These tests exercise only the Core layer — no Capital.com, no live
exchange, no httpx. A small in-test :class:`_FakeEntryHooks` plays
the role of a plugin and lets each path through the journal's state
machine be triggered deterministically.

Coverage:

- happy path: submit → server_ref → confirm → confirmed (filled)
- happy path: limit order — confirmed but unfilled
- submit-timeout disposition unknown
- confirm REJECTED
- confirm-timeout disposition unknown
- recovery replay → confirmed
- recovery replay → rejected
- recovery replay → still_unknown
- idempotency of repeated state advances
- audit-event sequence

End-to-end integration with the sync engine lives elsewhere; this
file is intentionally narrow.
"""
import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pynecore.core.broker.exceptions import (
    ExchangeOrderRejectedError,
    InsufficientMarginError,
    OrderDispositionUnknownError,
)
from pynecore.core.broker.journal import (
    ConfirmOutcome,
    DispatchJournal,
    ResumeOutcome,
    SubmitOutcome,
)
from pynecore.core.broker.models import (
    EntryIntent,
    ExchangeOrder,
    OrderStatus,
    OrderType,
)
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, OrderRow, RunContext
from pynecore.core.broker.store_helpers import (
    ENTRY_KIND_POSITION,
    ENTRY_KIND_WORKING,
    PENDING_DISPATCH_STATES,
    STATE_CANCEL_PENDING,
    STATE_CLOSING,
    STATE_CONFIRMED,
    STATE_DISPOSITION_UNKNOWN,
    STATE_REJECTED,
    STATE_SERVER_REF_SEEN,
    STATE_SUBMITTED,
    create_entry_order_row,
    find_pending_dispatch,
    mark_confirmed_with_fill,
    mark_disposition_unknown,
    mark_rejected,
    record_server_ref,
)


PLUGIN = "TestBroker"
SCRIPT_SOURCE = "// dispatch journal proof-of-shape test\n"
RUN_TAG = "test"
BAR_TS_MS = 1_700_000_000_000


def _make_intent(
        *,
        pine_id: str = "Long",
        symbol: str = "EURUSD",
        side: str = "buy",
        qty: float = 1.0,
        order_type: OrderType = OrderType.MARKET,
        limit: float | None = None,
        stop: float | None = None,
) -> EntryIntent:
    return EntryIntent(
        pine_id=pine_id, symbol=symbol, side=side, qty=qty,
        order_type=order_type, limit=limit, stop=stop,
    )


def _open_run(store: BrokerStore) -> RunContext:
    return store.open_run(
        RunIdentity(
            strategy_id="dj_test", symbol="EURUSD", timeframe="60",
            account_id="testbroker-demo", label=None,
        ),
        script_source=SCRIPT_SOURCE,
        script_path="strategies/dj_test.py",
    )


def _coid(ctx: RunContext, pine_id: str, kind_char: str = 'e') -> str:
    """Helper: build a plausible 30-char-bounded COID by hand.

    Bypasses the full ``build_client_order_id`` formula because the
    journal tests do not exercise idempotency-formula edge cases —
    that belongs in ``test_027_broker_idempotency.py``.
    """
    from pynecore.core.broker.idempotency import build_client_order_id
    return build_client_order_id(
        run_tag=ctx.run_tag, pine_id=pine_id,
        bar_ts_ms=BAR_TS_MS, kind=kind_char, retry_seq=0,
    )


# === Fake hooks ===========================================================

@dataclass
class _FakeEntryHooks:
    """Deterministic stand-in for a plugin's EntryDispatchHooks set.

    Each field controls one path through the state machine. The
    journal calls ``submit`` first, ``confirm_submission`` second;
    setting the corresponding action lets tests trigger every branch
    without an exchange.
    """
    # Action for ``submit``: a SubmitOutcome to return OR an exception
    # class to raise (class instantiated with a default message).
    submit_action: SubmitOutcome | type[BaseException] | BaseException = field(
        default_factory=lambda: SubmitOutcome(server_ref='ref-1'),
    )
    # Action for ``confirm_submission``: a ConfirmOutcome to return OR an
    # exception instance to raise.
    confirm_action: ConfirmOutcome | BaseException = field(
        default_factory=lambda: ConfirmOutcome(
            exchange_id='deal-1', is_filled=True,
            filled_qty=1.0, fill_price=1.10,
        ),
    )
    # Action for ``resume_pending_dispatch``.
    resume_action: ResumeOutcome = field(
        default_factory=lambda: ResumeOutcome(status='still_unknown'),
    )
    # Call counters for assertions.
    submit_calls: list[tuple[str, EntryIntent, float]] = field(default_factory=list)
    confirm_calls: list[tuple[str, EntryIntent, str]] = field(default_factory=list)
    resume_calls: list[OrderRow] = field(default_factory=list)

    async def submit(
            self, *, coid: str, intent: EntryIntent, qty: float,
    ) -> SubmitOutcome:
        self.submit_calls.append((coid, intent, qty))
        action = self.submit_action
        if isinstance(action, SubmitOutcome):
            return action
        if isinstance(action, type) and issubclass(action, BaseException):
            raise action("synthetic submit failure")
        if isinstance(action, BaseException):
            raise action
        raise AssertionError(f"unsupported submit_action: {action!r}")

    async def confirm_submission(
            self, *, coid: str, intent: EntryIntent, server_ref: str,
    ) -> ConfirmOutcome:
        self.confirm_calls.append((coid, intent, server_ref))
        if isinstance(self.confirm_action, ConfirmOutcome):
            return self.confirm_action
        raise self.confirm_action

    def exchange_order_from_state(
            self, *, row: OrderRow, intent: EntryIntent,
    ) -> ExchangeOrder:
        is_filled = row.filled_qty > 0.0
        return ExchangeOrder(
            id=row.exchange_order_id or '',
            symbol=intent.symbol, side=intent.side,
            order_type=intent.order_type,
            qty=row.qty, filled_qty=row.filled_qty,
            remaining_qty=max(0.0, row.qty - row.filled_qty),
            price=intent.limit, stop_price=intent.stop,
            average_fill_price=(
                float(row.extras.get('confirm_level'))
                if is_filled and row.extras.get('confirm_level') is not None
                else None
            ),
            status=OrderStatus.FILLED if is_filled else OrderStatus.OPEN,
            timestamp=0.0,  # deterministic in tests
            fee=0.0, fee_currency='',
            reduce_only=False,
            client_order_id=row.client_order_id,
        )

    async def resume_pending_dispatch(
            self, *, row: OrderRow, refs,
    ) -> ResumeOutcome:
        self.resume_calls.append(row)
        return self.resume_action


# === store_helpers unit tests =============================================

def __test_create_entry_order_row_writes_canonical_extras__(tmp_path: Path) -> None:
    """The helper persists the row with kind + order_type in extras."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            create_entry_order_row(
                ctx, coid=coid, symbol="EURUSD", side="buy",
                qty=1.0, intent_key="Long", pine_entry_id="Long",
                kind=ENTRY_KIND_POSITION, order_type=OrderType.MARKET.value,
            )
            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_SUBMITTED
            assert row.symbol == "EURUSD"
            assert row.side == "buy"
            assert row.qty == 1.0
            assert row.intent_key == "Long"
            assert row.pine_entry_id == "Long"
            assert row.extras == {
                'kind': ENTRY_KIND_POSITION,
                'order_type': 'market',
            }
        finally:
            ctx.close()


def __test_create_entry_order_row_rejects_bogus_kind__(tmp_path: Path) -> None:
    """An invalid kind raises before any DB write."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            try:
                create_entry_order_row(
                    ctx, coid=_coid(ctx, "Long"),
                    symbol="EURUSD", side="buy", qty=1.0,
                    intent_key="Long", pine_entry_id="Long",
                    kind="bogus", order_type="market",
                )
            except ValueError as exc:
                assert "kind must be one of" in str(exc)
            else:
                raise AssertionError("expected ValueError")
        finally:
            ctx.close()


def __test_record_server_ref_advances_state_and_adds_ref__(tmp_path: Path) -> None:
    """deal_reference goes into order_refs and extras; state advances."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            create_entry_order_row(
                ctx, coid=coid, symbol="EURUSD", side="buy", qty=1.0,
                intent_key="Long", pine_entry_id="Long",
                kind=ENTRY_KIND_POSITION, order_type='market',
            )
            record_server_ref(
                ctx, coid=coid, deal_reference='ref-abc',
                kind=ENTRY_KIND_POSITION, order_type='market',
            )
            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_SERVER_REF_SEEN
            assert row.extras.get('deal_reference') == 'ref-abc'
            assert row.extras.get('kind') == ENTRY_KIND_POSITION
            assert row.extras.get('order_type') == 'market'

            looked_up = ctx.find_by_ref('deal_reference', 'ref-abc')
            assert looked_up is not None
            assert looked_up.client_order_id == coid
        finally:
            ctx.close()


def __test_mark_confirmed_with_fill_market__(tmp_path: Path) -> None:
    """MARKET fill records deal_id ref, exchange_id, filled_qty, confirm_level."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            create_entry_order_row(
                ctx, coid=coid, symbol="EURUSD", side="buy", qty=1.0,
                intent_key="Long", pine_entry_id="Long",
                kind=ENTRY_KIND_POSITION, order_type='market',
            )
            record_server_ref(
                ctx, coid=coid, deal_reference='ref-1',
                kind=ENTRY_KIND_POSITION, order_type='market',
            )
            mark_confirmed_with_fill(
                ctx, coid=coid, exchange_id='deal-77',
                is_filled=True, filled_qty=1.0, fill_price=1.2345,
            )
            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_CONFIRMED
            assert row.exchange_order_id == 'deal-77'
            assert row.filled_qty == 1.0
            assert row.extras['confirm_level'] == 1.2345
            # deal_reference must survive the merge.
            assert row.extras['deal_reference'] == 'ref-1'

            looked_up = ctx.find_by_ref('deal_id', 'deal-77')
            assert looked_up is not None
            assert looked_up.client_order_id == coid
        finally:
            ctx.close()


def __test_mark_confirmed_with_fill_unfilled_limit__(tmp_path: Path) -> None:
    """LIMIT order: state becomes confirmed, no fill mutation."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Lim")
            create_entry_order_row(
                ctx, coid=coid, symbol="EURUSD", side="buy", qty=1.0,
                intent_key="Lim", pine_entry_id="Lim",
                kind=ENTRY_KIND_WORKING, order_type='limit',
            )
            record_server_ref(
                ctx, coid=coid, deal_reference='ref-l',
                kind=ENTRY_KIND_WORKING, order_type='limit',
            )
            mark_confirmed_with_fill(
                ctx, coid=coid, exchange_id='deal-l',
                is_filled=False, filled_qty=0.0, fill_price=None,
            )
            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_CONFIRMED
            assert row.exchange_order_id == 'deal-l'
            assert row.filled_qty == 0.0
            assert 'confirm_level' not in row.extras
        finally:
            ctx.close()


def __test_mark_confirmed_skips_zero_fill_price__(tmp_path: Path) -> None:
    """A fill_price of 0.0 must NOT be persisted to confirm_level."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            create_entry_order_row(
                ctx, coid=coid, symbol="EURUSD", side="buy", qty=1.0,
                intent_key="Long", pine_entry_id="Long",
                kind=ENTRY_KIND_POSITION, order_type='market',
            )
            mark_confirmed_with_fill(
                ctx, coid=coid, exchange_id='d1',
                is_filled=True, filled_qty=1.0, fill_price=0.0,
            )
            row = ctx.get_order(coid)
            assert row is not None
            assert row.filled_qty == 1.0
            assert 'confirm_level' not in row.extras
        finally:
            ctx.close()


def __test_find_pending_dispatch_filters_by_state__(tmp_path: Path) -> None:
    """Only rows in pending states are returned; terminal ones excluded.

    ``closing`` and ``cancel_pending`` are listed for documentation but
    not in ``PENDING_DISPATCH_STATES`` until the M4 close / cancel
    phases ship the corresponding recovery branches.
    """
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            states_to_coids: dict[str, str] = {}
            for label, state in (
                ('subm', STATE_SUBMITTED),
                ('refs', STATE_SERVER_REF_SEEN),
                ('disp', STATE_DISPOSITION_UNKNOWN),
                ('clos', STATE_CLOSING),
                ('canc', STATE_CANCEL_PENDING),
                ('conf', STATE_CONFIRMED),
                ('rej', STATE_REJECTED),
            ):
                coid = _coid(ctx, f"PID{label}")
                ctx.upsert_order(
                    coid, symbol='X', side='buy', qty=1.0,
                    state=state, intent_key=label, pine_entry_id=label,
                    extras={'kind': ENTRY_KIND_POSITION, 'order_type': 'market'},
                )
                states_to_coids[state] = coid

            pending = {r.state for r in find_pending_dispatch(ctx)}
            assert pending == PENDING_DISPATCH_STATES
            assert STATE_CONFIRMED not in pending
            assert STATE_REJECTED not in pending
            assert STATE_CLOSING not in pending
            assert STATE_CANCEL_PENDING not in pending
        finally:
            ctx.close()


# === DispatchJournal.run_entry tests ======================================

def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)


def __test_journal_happy_path_market_fill__(tmp_path: Path) -> None:
    """Happy path: submit → server_ref → confirm → confirmed + filled."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            intent = _make_intent(order_type=OrderType.MARKET)
            hooks = _FakeEntryHooks(
                submit_action=SubmitOutcome(server_ref='ref-happy'),
                confirm_action=ConfirmOutcome(
                    exchange_id='deal-happy', is_filled=True,
                    filled_qty=1.0, fill_price=1.10,
                ),
            )
            journal = DispatchJournal(ctx)
            result = _run(journal.run_entry(
                coid=coid, intent=intent, qty=1.0,
                kind=ENTRY_KIND_POSITION, hooks=hooks,
            ))
            assert len(result) == 1
            order = result[0]
            assert order.id == 'deal-happy'
            assert order.status == OrderStatus.FILLED
            assert order.filled_qty == 1.0
            assert order.average_fill_price == 1.10
            assert order.client_order_id == coid

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_CONFIRMED
            assert row.exchange_order_id == 'deal-happy'
            assert row.filled_qty == 1.0
            assert row.extras['confirm_level'] == 1.10
            assert row.extras['deal_reference'] == 'ref-happy'

            assert ctx.find_by_ref('deal_reference', 'ref-happy') is not None
            assert ctx.find_by_ref('deal_id', 'deal-happy') is not None

            assert len(hooks.submit_calls) == 1
            assert len(hooks.confirm_calls) == 1
        finally:
            ctx.close()


def __test_journal_happy_path_working_order__(tmp_path: Path) -> None:
    """Working order: state confirmed, no fill (LIMIT accepted but unfilled)."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Lim")
            intent = _make_intent(order_type=OrderType.LIMIT, limit=1.20)
            hooks = _FakeEntryHooks(
                submit_action=SubmitOutcome(server_ref='ref-lim'),
                confirm_action=ConfirmOutcome(
                    exchange_id='deal-lim', is_filled=False,
                    filled_qty=0.0, fill_price=None,
                ),
            )
            journal = DispatchJournal(ctx)
            result = _run(journal.run_entry(
                coid=coid, intent=intent, qty=1.0,
                kind=ENTRY_KIND_WORKING, hooks=hooks,
            ))
            order = result[0]
            assert order.status == OrderStatus.OPEN
            assert order.filled_qty == 0.0
            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_CONFIRMED
            assert 'confirm_level' not in row.extras
        finally:
            ctx.close()


def __test_journal_submit_timeout_persists_disposition_unknown__(
        tmp_path: Path,
) -> None:
    """OrderDispositionUnknownError from submit flips state + re-raises."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            intent = _make_intent()
            timeout = OrderDispositionUnknownError(
                "synthetic POST timeout", client_order_id=coid,
            )
            hooks = _FakeEntryHooks(submit_action=timeout)
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_entry(
                    coid=coid, intent=intent, qty=1.0,
                    kind=ENTRY_KIND_POSITION, hooks=hooks,
                ))
            except OrderDispositionUnknownError as exc:
                assert exc.client_order_id == coid
            else:
                raise AssertionError("expected OrderDispositionUnknownError")

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_DISPOSITION_UNKNOWN
            assert row.exchange_order_id is None
            assert len(hooks.confirm_calls) == 0
        finally:
            ctx.close()


def __test_journal_confirm_reject_persists_rejected__(tmp_path: Path) -> None:
    """ExchangeOrderRejectedError from confirm flips state to rejected."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            intent = _make_intent()
            hooks = _FakeEntryHooks(
                submit_action=SubmitOutcome(server_ref='ref-rej'),
                confirm_action=ExchangeOrderRejectedError("MARKET_CLOSED"),
            )
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_entry(
                    coid=coid, intent=intent, qty=1.0,
                    kind=ENTRY_KIND_POSITION, hooks=hooks,
                ))
            except ExchangeOrderRejectedError:
                pass
            else:
                raise AssertionError("expected ExchangeOrderRejectedError")

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_REJECTED
            # deal_reference must still be on the rejected row — it
            # was a valid submission, just rejected at confirm time.
            assert row.extras.get('deal_reference') == 'ref-rej'
        finally:
            ctx.close()


def __test_journal_confirm_reject_margin_subclass__(tmp_path: Path) -> None:
    """A subclass of ExchangeOrderRejectedError is also caught."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            intent = _make_intent()
            hooks = _FakeEntryHooks(
                submit_action=SubmitOutcome(server_ref='ref-margin'),
                confirm_action=InsufficientMarginError("insufficient margin"),
            )
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_entry(
                    coid=coid, intent=intent, qty=1.0,
                    kind=ENTRY_KIND_POSITION, hooks=hooks,
                ))
            except InsufficientMarginError:
                pass
            else:
                raise AssertionError("expected InsufficientMarginError")
            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_REJECTED
        finally:
            ctx.close()


def __test_journal_confirm_timeout_persists_disposition_unknown__(
        tmp_path: Path,
) -> None:
    """OrderDispositionUnknownError from confirm also flips disposition_unknown."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            intent = _make_intent()
            hooks = _FakeEntryHooks(
                submit_action=SubmitOutcome(server_ref='ref-x'),
                confirm_action=OrderDispositionUnknownError(
                    "confirm timeout", client_order_id=coid,
                ),
            )
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_entry(
                    coid=coid, intent=intent, qty=1.0,
                    kind=ENTRY_KIND_POSITION, hooks=hooks,
                ))
            except OrderDispositionUnknownError:
                pass
            else:
                raise AssertionError("expected OrderDispositionUnknownError")
            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_DISPOSITION_UNKNOWN
            # deal_reference was recorded before the confirm crashed.
            assert row.extras.get('deal_reference') == 'ref-x'
        finally:
            ctx.close()


# === Audit event sequence =================================================

def _read_events(ctx: RunContext, coid: str) -> list[tuple[str, dict | None]]:
    """Return (kind, payload) pairs for a COID in ts order."""
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


def __test_journal_audit_event_sequence_happy__(tmp_path: Path) -> None:
    """Audit kinds appear in order: submitted, ref_seen, confirmed."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            intent = _make_intent()
            hooks = _FakeEntryHooks(
                submit_action=SubmitOutcome(server_ref='ref-a'),
                confirm_action=ConfirmOutcome(
                    exchange_id='deal-a', is_filled=True,
                    filled_qty=1.0, fill_price=1.10,
                ),
            )
            _run(DispatchJournal(ctx).run_entry(
                coid=coid, intent=intent, qty=1.0,
                kind=ENTRY_KIND_POSITION, hooks=hooks,
            ))
            kinds = [k for (k, _) in _read_events(ctx, coid)]
            assert kinds == [
                'dispatch_submitted',
                'deal_reference_seen',
                'confirmed',
            ]
        finally:
            ctx.close()


def __test_journal_audit_event_sequence_submit_timeout__(
        tmp_path: Path,
) -> None:
    """On submit timeout: dispatch_submitted then disposition_unknown."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            intent = _make_intent()
            hooks = _FakeEntryHooks(
                submit_action=OrderDispositionUnknownError(
                    "POST timeout", client_order_id=coid,
                ),
            )
            try:
                _run(DispatchJournal(ctx).run_entry(
                    coid=coid, intent=intent, qty=1.0,
                    kind=ENTRY_KIND_POSITION, hooks=hooks,
                ))
            except OrderDispositionUnknownError:
                pass
            kinds = [k for (k, _) in _read_events(ctx, coid)]
            assert kinds == [
                'dispatch_submitted',
                'disposition_unknown',
            ]
        finally:
            ctx.close()


# === recover_pending tests ================================================

def __test_recover_pending_confirms_pending_row__(tmp_path: Path) -> None:
    """Recovery: resume returns confirmed → row promoted + event written."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            # Plant a pending row directly (simulating crash mid-dispatch).
            ctx.upsert_order(
                coid, symbol="EURUSD", side="buy", qty=1.0,
                state=STATE_DISPOSITION_UNKNOWN,
                intent_key="Long", pine_entry_id="Long",
                extras={
                    'kind': ENTRY_KIND_POSITION,
                    'order_type': 'market',
                    'deal_reference': 'ref-crashed',
                },
            )
            hooks = _FakeEntryHooks(
                resume_action=ResumeOutcome(
                    status='confirmed', exchange_id='deal-rec',
                    is_filled=True, filled_qty=1.0, fill_price=1.20,
                ),
            )
            resolutions = _run(DispatchJournal(ctx).recover_pending(
                lambda row: hooks,
            ))
            assert len(resolutions) == 1
            assert resolutions[0].status == 'confirmed'
            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_CONFIRMED
            assert row.exchange_order_id == 'deal-rec'
            assert row.extras['confirm_level'] == 1.20
            # The resume hook was given the pre-recovery row.
            assert len(hooks.resume_calls) == 1
            assert hooks.resume_calls[0].state == STATE_DISPOSITION_UNKNOWN

            kinds = [k for (k, _) in _read_events(ctx, coid)]
            assert 'recovered_confirmed' in kinds
        finally:
            ctx.close()


def __test_recover_pending_rejects_pending_row__(tmp_path: Path) -> None:
    """Recovery: resume returns rejected → state goes to rejected."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            ctx.upsert_order(
                coid, symbol="EURUSD", side="buy", qty=1.0,
                state=STATE_SUBMITTED,
                intent_key="Long", pine_entry_id="Long",
                extras={'kind': ENTRY_KIND_POSITION, 'order_type': 'market'},
            )
            hooks = _FakeEntryHooks(
                resume_action=ResumeOutcome(
                    status='rejected', reject_reason='post_never_landed',
                ),
            )
            resolutions = _run(DispatchJournal(ctx).recover_pending(
                lambda row: hooks,
            ))
            assert resolutions[0].status == 'rejected'
            assert resolutions[0].reason == 'post_never_landed'
            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_REJECTED
            kinds = [k for (k, _) in _read_events(ctx, coid)]
            assert 'recovered_rejected' in kinds
        finally:
            ctx.close()


def __test_recover_pending_keeps_still_unknown_row__(tmp_path: Path) -> None:
    """Recovery: still_unknown leaves the row in place, logs a marker."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            ctx.upsert_order(
                coid, symbol="EURUSD", side="buy", qty=1.0,
                state=STATE_DISPOSITION_UNKNOWN,
                intent_key="Long", pine_entry_id="Long",
                extras={'kind': ENTRY_KIND_POSITION, 'order_type': 'market'},
            )
            hooks = _FakeEntryHooks(
                resume_action=ResumeOutcome(status='still_unknown'),
            )
            resolutions = _run(DispatchJournal(ctx).recover_pending(
                lambda row: hooks,
            ))
            assert resolutions[0].status == 'still_unknown'
            row = ctx.get_order(coid)
            assert row is not None
            # Row unchanged.
            assert row.state == STATE_DISPOSITION_UNKNOWN
            kinds = [k for (k, _) in _read_events(ctx, coid)]
            assert 'recovery_pending' in kinds
        finally:
            ctx.close()


def __test_recover_pending_skips_when_hooks_provider_returns_none__(
        tmp_path: Path,
) -> None:
    """Rows whose kind the entry journal does not handle are skipped."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Br")
            # Plant a fake bracket-leg row (kind='tp') — not the entry
            # journal's responsibility.
            ctx.upsert_order(
                coid, symbol="EURUSD", side="sell", qty=1.0,
                state=STATE_SUBMITTED,
                intent_key="Br", pine_entry_id="Long",
                extras={'kind': 'tp', 'order_type': 'limit'},
            )
            resolutions = _run(DispatchJournal(ctx).recover_pending(
                lambda row: None,
            ))
            assert len(resolutions) == 1
            assert resolutions[0].status == 'skipped'
            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_SUBMITTED  # untouched
        finally:
            ctx.close()


# === Idempotency =========================================================

def __test_journal_state_advances_are_idempotent__(tmp_path: Path) -> None:
    """Running the same helpers twice produces the same final row."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            for _ in range(2):
                create_entry_order_row(
                    ctx, coid=coid, symbol="EURUSD", side="buy",
                    qty=1.0, intent_key="Long", pine_entry_id="Long",
                    kind=ENTRY_KIND_POSITION, order_type='market',
                )
                record_server_ref(
                    ctx, coid=coid, deal_reference='ref-z',
                    kind=ENTRY_KIND_POSITION, order_type='market',
                )
                mark_confirmed_with_fill(
                    ctx, coid=coid, exchange_id='deal-z',
                    is_filled=True, filled_qty=1.0, fill_price=1.10,
                )
            # One row only; second pass updated rather than duplicated.
            rows = list(ctx.iter_live_orders())
            assert len(rows) == 1
            row = rows[0]
            assert row.state == STATE_CONFIRMED
            assert row.exchange_order_id == 'deal-z'
            assert row.extras['confirm_level'] == 1.10
        finally:
            ctx.close()


# === Submit-time reject ===================================================

def __test_journal_submit_reject_persists_rejected__(tmp_path: Path) -> None:
    """ExchangeOrderRejectedError from submit flips state to rejected.

    A synchronous POST-side reject (4xx with a parseable reason and
    no server reference) must terminalise the row instead of leaving
    it in :data:`STATE_SUBMITTED`, otherwise recovery would treat the
    row as pending and could retry an already-rejected order.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            intent = _make_intent()
            hooks = _FakeEntryHooks(
                submit_action=ExchangeOrderRejectedError(
                    "synchronous POST reject: MARKET_HALTED",
                ),
            )
            journal = DispatchJournal(ctx)
            try:
                _run(journal.run_entry(
                    coid=coid, intent=intent, qty=1.0,
                    kind=ENTRY_KIND_POSITION, hooks=hooks,
                ))
            except ExchangeOrderRejectedError:
                pass
            else:
                raise AssertionError("expected ExchangeOrderRejectedError")

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_REJECTED
            # No deal_reference because submit never returned a server ref.
            assert 'deal_reference' not in row.extras
            # No confirm call should have been attempted.
            assert len(hooks.confirm_calls) == 0

            kinds = [k for (k, _) in _read_events(ctx, coid)]
            assert kinds == [
                'dispatch_submitted',
                'rejected',
            ]
            # The rejected event payload tags the submit phase.
            evt = next(
                (p for (k, p) in _read_events(ctx, coid) if k == 'rejected'),
                None,
            )
            assert evt is not None
            assert evt.get('phase') == 'submit'
        finally:
            ctx.close()


# === Refs passed to recovery hooks ========================================

def __test_recover_pending_passes_refs_from_order_refs__(tmp_path: Path) -> None:
    """resume_pending_dispatch sees ``deal_reference`` even when only
    ``order_refs`` was written.

    Between the ``add_ref('deal_reference', ...)`` commit inside
    :func:`record_server_ref` and the subsequent
    ``upsert_order(extras={...})`` commit the reference exists only in
    ``order_refs``. A resume hook that relied on ``row.extras`` alone
    would treat the row as never posted; the journal must materialise
    the refs from ``order_refs`` so the hook can recover the dispatch.
    """
    captured_refs: list[dict[str, str]] = []

    class _RefCapturingHooks(_FakeEntryHooks):
        async def resume_pending_dispatch(
                self, *, row: OrderRow, refs,
        ) -> ResumeOutcome:
            captured_refs.append(dict(refs))
            return await super().resume_pending_dispatch(row=row, refs=refs)

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            # Plant a SUBMITTED row whose deal_reference lives ONLY in
            # order_refs — simulating a crash mid-record_server_ref.
            ctx.upsert_order(
                coid, symbol="EURUSD", side="buy", qty=1.0,
                state=STATE_SUBMITTED,
                intent_key="Long", pine_entry_id="Long",
                extras={'kind': ENTRY_KIND_POSITION, 'order_type': 'market'},
            )
            ctx.add_ref(coid, 'deal_reference', 'ref-orphan')

            hooks = _RefCapturingHooks(
                resume_action=ResumeOutcome(status='still_unknown'),
            )
            _run(DispatchJournal(ctx).recover_pending(lambda row: hooks))

            assert captured_refs == [{'deal_reference': 'ref-orphan'}]
        finally:
            ctx.close()


# === Cross-instance recovery (orphan adoption) =============================

def __test_recover_pending_adopts_orphan_from_previous_instance__(
        tmp_path: Path,
) -> None:
    """A pending dispatch from a *previous* run instance is visible to
    :meth:`DispatchJournal.recover_pending` after restart.

    Without adoption, :func:`find_pending_dispatch` filters by the
    current ``run_instance_id`` and misses the crashed instance's
    pending row, so recovery would silently leave it ambiguous on
    disk.
    """
    db = tmp_path / "broker.sqlite"

    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_run(store_a)
        coid = _coid(ctx_a, "Long")
        # Plant a pending dispatch in run A and force a stale heartbeat
        # so the next open_run sees it as crashed.
        ctx_a.upsert_order(
            coid, symbol="EURUSD", side="buy", qty=1.0,
            state=STATE_DISPOSITION_UNKNOWN,
            intent_key="Long", pine_entry_id="Long",
            extras={
                'kind': ENTRY_KIND_POSITION,
                'order_type': 'market',
                'deal_reference': 'ref-crashed',
            },
        )
        ctx_a.add_ref(coid, 'deal_reference', 'ref-crashed')
        store_a._conn.execute(
            "UPDATE runs SET last_heartbeat_ts_ms = 1 "
            "WHERE run_instance_id = ?",
            (ctx_a.run_instance_id,),
        )
        store_a._conn.commit()
        prior_run_instance_id = ctx_a.run_instance_id

    # Restart — open_run cleans up run A, then adopts its orphan row.
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_run(store_b)
        try:
            assert ctx_b.run_instance_id != prior_run_instance_id

            # Pre-recovery: iter_live_orders for the new instance now
            # sees the adopted row.
            live = list(ctx_b.iter_live_orders())
            assert len(live) == 1
            assert live[0].client_order_id == coid

            # The order_adopted event is on the NEW instance.
            ev = store_b._conn.execute(
                "SELECT kind, payload FROM events "
                "WHERE run_instance_id = ? AND client_order_id = ? "
                "AND kind = 'order_adopted'",
                (ctx_b.run_instance_id, coid),
            ).fetchone()
            assert ev is not None
            payload = json.loads(ev['payload'])
            assert payload['prior_run_instance_id'] == prior_run_instance_id
            assert payload['prior_state'] == STATE_DISPOSITION_UNKNOWN

            # The order_refs row migrated too, so the recovery hook can
            # see deal_reference via iter_refs_for_coid.
            refs_on_b = dict(ctx_b.iter_refs_for_coid(coid))
            assert refs_on_b == {'deal_reference': 'ref-crashed'}

            hooks = _FakeEntryHooks(
                resume_action=ResumeOutcome(
                    status='confirmed', exchange_id='deal-recovered',
                    is_filled=True, filled_qty=1.0, fill_price=1.21,
                ),
            )
            resolutions = _run(
                DispatchJournal(ctx_b).recover_pending(lambda row: hooks),
            )
            assert len(resolutions) == 1
            assert resolutions[0].status == 'confirmed'

            row = ctx_b.get_order(coid)
            assert row is not None
            assert row.state == STATE_CONFIRMED
            assert row.exchange_order_id == 'deal-recovered'
        finally:
            ctx_b.close()


def __test_open_run_skips_adoption_when_no_orphans__(tmp_path: Path) -> None:
    """Adoption is a no-op when no orphan rows exist."""
    db = tmp_path / "broker.sqlite"
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            ev = store._conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE kind = 'order_adopted'",
            ).fetchone()
            assert ev['n'] == 0
            assert list(ctx.iter_live_orders()) == []
        finally:
            ctx.close()


def __test_open_run_does_not_adopt_closed_orders__(tmp_path: Path) -> None:
    """A previously closed order stays linked to its original instance."""
    db = tmp_path / "broker.sqlite"

    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_run(store_a)
        coid = _coid(ctx_a, "Done")
        ctx_a.upsert_order(
            coid, symbol="EURUSD", side="buy", qty=1.0,
            state=STATE_CONFIRMED,
            intent_key="Done", pine_entry_id="Done",
            extras={'kind': ENTRY_KIND_POSITION, 'order_type': 'market'},
        )
        ctx_a.close_order(coid)
        store_a._conn.execute(
            "UPDATE runs SET last_heartbeat_ts_ms = 1 "
            "WHERE run_instance_id = ?",
            (ctx_a.run_instance_id,),
        )
        store_a._conn.commit()
        prior_run_instance_id = ctx_a.run_instance_id

    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_run(store_b)
        try:
            # iter_live_orders sees nothing (the order was closed).
            assert list(ctx_b.iter_live_orders()) == []
            # No adoption event was written.
            ev = store_b._conn.execute(
                "SELECT COUNT(*) AS n FROM events "
                "WHERE run_instance_id = ? AND kind = 'order_adopted'",
                (ctx_b.run_instance_id,),
            ).fetchone()
            assert ev['n'] == 0
            # The closed row remains pinned to its original instance.
            owner = store_b._conn.execute(
                "SELECT run_instance_id FROM orders WHERE client_order_id = ?",
                (coid,),
            ).fetchone()
            assert owner['run_instance_id'] == prior_run_instance_id
        finally:
            ctx_b.close()


def __test_adopt_orphans_emits_single_summary_info__(
        tmp_path: Path, caplog,
) -> None:
    """Adoption stays quiet at WARNING and emits one summary INFO line.

    Operators do not need to see every per-row supersede/adopt entry —
    those are forensic and live in the ``events`` table. The user-
    visible signal is a single INFO line counting adopted rows.
    """
    import logging as _logging

    db = tmp_path / "broker.sqlite"

    # Plant two prior orphan rows under run A and stale the heartbeat.
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_run(store_a)
        coid_kept = _coid(ctx_a, "KeptLong")
        coid_dup = _coid(ctx_a, "DupLong")
        for coid in (coid_kept, coid_dup):
            ctx_a.upsert_order(
                coid, symbol="EURUSD", side="buy", qty=1.0,
                state=STATE_CONFIRMED,
                intent_key=coid, pine_entry_id=coid,
                extras={'kind': ENTRY_KIND_POSITION, 'order_type': 'market'},
            )
        store_a._conn.execute(
            "UPDATE runs SET last_heartbeat_ts_ms = 1 "
            "WHERE run_instance_id = ?",
            (ctx_a.run_instance_id,),
        )
        store_a._conn.commit()

    caplog.set_level(_logging.DEBUG, logger="pynecore.core.broker.storage")

    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_run(store_b)
        try:
            warning_lines = [
                rec for rec in caplog.records
                if rec.name == "pynecore.core.broker.storage"
                and rec.levelno == _logging.WARNING
                and (
                    "adopted" in rec.getMessage()
                    or "superseded" in rec.getMessage()
                )
            ]
            assert warning_lines == [], (
                "Adoption must not emit WARNING-level lines per row "
                "(forensic data lives in the events table)"
            )

            info_lines = [
                rec for rec in caplog.records
                if rec.name == "pynecore.core.broker.storage"
                and rec.levelno == _logging.INFO
                and "adopted" in rec.getMessage()
            ]
            assert len(info_lines) == 1, (
                f"Expected exactly one summary INFO line, got "
                f"{[r.getMessage() for r in info_lines]!r}"
            )
            assert "2 order(s)" in info_lines[0].getMessage()

            # Audit events still recorded — both adopted, no supersede
            # because each COID is unique across instances.
            n_adopted = store_b._conn.execute(
                "SELECT COUNT(*) AS n FROM events "
                "WHERE kind = 'order_adopted'",
            ).fetchone()['n']
            assert n_adopted == 2
        finally:
            ctx_b.close()
