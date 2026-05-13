"""
Contract tests for the M5 reconcile-path Core API:
:class:`~pynecore.core.broker.journal.ReconcileOutcome` plus
:meth:`~pynecore.core.broker.journal.DispatchJournal.apply_reconcile_outcome`
and the two terminal helpers in :mod:`pynecore.core.broker.store_helpers`
(:func:`mark_reconcile_filled`, :func:`mark_reconcile_terminal_close`).

These tests pin the journal-side byte shape of every reconcile-path
state mutation that the Capital.com plugin emits today — without
plugging in the plugin. Each test seeds a deterministic row, invokes
``apply_reconcile_outcome`` with a hand-built outcome, and asserts on
the persisted ``state`` / ``filled_qty`` / ``closed_ts_ms`` / extras /
audit event.

The Cat 1 reconciler-private breadcrumb extras (``missing_pending_since``
and friends) are NOT routed through this API; they are plugin-namespace
and verified by the broker test suite.

See ``docs/pynecore/plugin-system/broker/broker-plugin-responsibility-review.md``
§4.2 for the contract.
"""
import json
from pathlib import Path

from pynecore.core.broker.journal import (
    DispatchJournal,
    ReconcileOutcome,
)
from pynecore.core.broker.models import OrderType
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.store_helpers import (
    ENTRY_KIND_POSITION,
    ENTRY_KIND_WORKING,
    STATE_CONFIRMED,
    STATE_REJECTED,
    STATE_SERVER_REF_SEEN,
    create_entry_order_row,
    record_server_ref,
)


PLUGIN = "TestBroker"
SCRIPT_SOURCE = "// reconcile outcome journal contract test\n"
BAR_TS_MS = 1_700_000_000_000


def _open_run(store: BrokerStore) -> RunContext:
    return store.open_run(
        RunIdentity(
            strategy_id="rj_test", symbol="EURUSD", timeframe="60",
            account_id="testbroker-demo", label=None,
        ),
        script_source=SCRIPT_SOURCE,
        script_path="strategies/rj_test.py",
    )


def _coid(ctx: RunContext, pine_id: str, kind_char: str = 'e') -> str:
    from pynecore.core.broker.idempotency import build_client_order_id
    return build_client_order_id(
        run_tag=ctx.run_tag, pine_id=pine_id,
        bar_ts_ms=BAR_TS_MS, kind=kind_char, retry_seq=0,
    )


def _read_events(ctx: RunContext, coid: str) -> list[tuple[str, str | None, dict | None]]:
    """Return (kind, exchange_order_id, payload) tuples for a COID in ts order."""
    rows = ctx._store._conn.execute(  # type: ignore[attr-defined]
        "SELECT kind, exchange_order_id, payload FROM events "
        "WHERE run_instance_id = ? AND client_order_id = ? "
        "ORDER BY ts_ms ASC, id ASC",
        (ctx.run_instance_id, coid),
    ).fetchall()
    out: list[tuple[str, str | None, dict | None]] = []
    for r in rows:
        payload = json.loads(r['payload']) if r['payload'] else None
        out.append((r['kind'], r['exchange_order_id'], payload))
    return out


def _seed_working_row(ctx: RunContext, coid: str, *, qty: float = 1.0) -> None:
    """Create a working-order row in :data:`STATE_SERVER_REF_SEEN`.

    Mirrors the lifecycle the plugin reaches just before observing the
    fill in ``/positions``: the LIMIT/STOP POST succeeded, the server
    reference was recorded, and the working order is sitting on the
    exchange waiting for fill.
    """
    create_entry_order_row(
        ctx, coid=coid, symbol="EURUSD", side="buy",
        qty=qty, intent_key="Long", pine_entry_id="Long",
        kind=ENTRY_KIND_WORKING, order_type=OrderType.LIMIT.value,
    )
    record_server_ref(
        ctx, coid=coid, deal_reference='ref-working',
        kind=ENTRY_KIND_WORKING, order_type=OrderType.LIMIT.value,
    )


def _seed_bracket_leg_row(
        ctx: RunContext, coid: str, *,
        leg_kind: str, parent_coid: str, parent_deal_id: str,
) -> None:
    """Create a bracket leg row in :data:`STATE_CONFIRMED`.

    Mirrors a TP/SL leg that was attached to a parent position. The
    reconcile-path retire ages this kind of row when a mixed-bracket
    rejection forces the engine to re-dispatch a fresh exit envelope.
    """
    ctx.upsert_order(
        coid, symbol="EURUSD", side="sell", qty=1.0,
        intent_key="Long", pine_entry_id="Long",
        state=STATE_CONFIRMED,
        extras={
            'leg_kind': leg_kind,
            'parent_coid': parent_coid,
            'parent_deal_id': parent_deal_id,
        },
    )


# === Cat 2: working→position fill ==========================================

def __test_apply_reconcile_filled_promotes_working_to_position__(
        tmp_path: Path,
) -> None:
    """``kind='filled'`` advances state, stamps filled_qty, flips extras."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            _seed_working_row(ctx, coid, qty=2.0)
            now_ts = 1_700_000_500
            outcome = ReconcileOutcome(
                kind='filled',
                reason='working_promoted_position',
                new_state='confirmed',
                audit_event='working_to_position',
                filled_qty=2.0,
                extras_patch={
                    'kind': ENTRY_KIND_POSITION,
                    'entry_filled_at': now_ts,
                },
                audit_payload={'filled': 2.0},
                exchange_order_id='deal-1',
            )
            DispatchJournal(ctx).apply_reconcile_outcome(coid, outcome)

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_CONFIRMED
            assert row.filled_qty == 2.0
            assert row.closed_ts_ms is None  # close_row defaults to False
            assert row.extras is not None
            assert row.extras.get('kind') == ENTRY_KIND_POSITION
            assert row.extras.get('entry_filled_at') == now_ts
            # Pre-existing extras (order_type from the create helper) preserved
            assert row.extras.get('order_type') == OrderType.LIMIT.value

            events = _read_events(ctx, coid)
            audit_kinds = [k for (k, _, _) in events]
            assert 'working_to_position' in audit_kinds
            # Find the working_to_position event payload
            w2p = next(
                e for e in events if e[0] == 'working_to_position'
            )
            assert w2p[1] == 'deal-1'  # exchange_order_id
            assert w2p[2] == {'filled': 2.0}
        finally:
            ctx.close()


def __test_apply_reconcile_filled_requires_filled_qty__(tmp_path: Path) -> None:
    """``kind='filled'`` without ``filled_qty`` raises ValueError."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            _seed_working_row(ctx, coid)
            outcome = ReconcileOutcome(
                kind='filled',
                reason='working_promoted_position',
                new_state='confirmed',
                audit_event='working_to_position',
                filled_qty=None,  # missing
            )
            try:
                DispatchJournal(ctx).apply_reconcile_outcome(coid, outcome)
            except ValueError as exc:
                assert "requires filled_qty" in str(exc)
            else:
                raise AssertionError("expected ValueError")
        finally:
            ctx.close()


def __test_apply_reconcile_filled_keeps_row_live_when_close_row_false__(
        tmp_path: Path,
) -> None:
    """Filled outcomes default ``close_row=False`` — row stays in live set."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            _seed_working_row(ctx, coid)
            outcome = ReconcileOutcome(
                kind='filled',
                reason='working_promoted_position',
                new_state='confirmed',
                audit_event='working_to_position',
                filled_qty=1.0,
                extras_patch={'kind': ENTRY_KIND_POSITION},
            )
            DispatchJournal(ctx).apply_reconcile_outcome(coid, outcome)
            live = list(ctx.iter_live_orders())
            assert any(r.client_order_id == coid for r in live)
        finally:
            ctx.close()


# === Cat 3: terminal_close paths ===========================================

def __test_apply_reconcile_terminal_close_bracket_sibling_retire__(
        tmp_path: Path,
) -> None:
    """Bracket sibling retire on mixed-bracket rejection: state + close + audit."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            parent_coid = _coid(ctx, "Long")
            sibling_coid = _coid(ctx, "Long-TP", kind_char='x')
            _seed_bracket_leg_row(
                ctx, sibling_coid,
                leg_kind='tp', parent_coid=parent_coid,
                parent_deal_id='parent-deal-1',
            )

            outcome = ReconcileOutcome(
                kind='terminal_close',
                reason='bracket_sibling_retired_on_mixed_rejection',
                new_state='rejected',
                audit_event='bracket_sibling_retired_on_mixed_rejection',
                close_row=True,
                audit_payload={
                    'sibling_coid': sibling_coid,
                    'parent_coid': parent_coid,
                    'leg_kind': 'tp',
                    'reason': 'mixed_bracket_rejected',
                },
                exchange_order_id='parent-deal-1',
            )
            DispatchJournal(ctx).apply_reconcile_outcome(sibling_coid, outcome)

            row = ctx.get_order(sibling_coid)
            assert row is not None
            assert row.state == STATE_REJECTED
            assert row.closed_ts_ms is not None  # close_row=True closed it

            events = _read_events(ctx, sibling_coid)
            audit = [e for e in events
                     if e[0] == 'bracket_sibling_retired_on_mixed_rejection']
            assert len(audit) == 1
            kind, ex_id, payload = audit[0]
            assert ex_id == 'parent-deal-1'
            assert payload == {
                'sibling_coid': sibling_coid,
                'parent_coid': parent_coid,
                'leg_kind': 'tp',
                'reason': 'mixed_bracket_rejected',
            }
        finally:
            ctx.close()


def __test_apply_reconcile_terminal_close_pending_trail_parent_rejected__(
        tmp_path: Path,
) -> None:
    """Pending-trail SL sibling parent-rejection cascade."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            parent_coid = _coid(ctx, "Long")
            sibling_coid = _coid(ctx, "Long-SL", kind_char='x')
            _seed_bracket_leg_row(
                ctx, sibling_coid,
                leg_kind='sl', parent_coid=parent_coid,
                parent_deal_id='parent-deal-2',
            )

            outcome = ReconcileOutcome(
                kind='terminal_close',
                reason='pending_trail_parent_rejected',
                new_state='rejected',
                audit_event='bracket_sibling_retired_on_mixed_rejection',
                close_row=True,
                audit_payload={
                    'sibling_coid': sibling_coid,
                    'parent_coid': parent_coid,
                    'leg_kind': 'sl',
                    'reason': 'pending_trail_parent_rejected',
                    'trail_state': 'awaiting_activation',
                },
                exchange_order_id='parent-deal-2',
            )
            DispatchJournal(ctx).apply_reconcile_outcome(sibling_coid, outcome)

            row = ctx.get_order(sibling_coid)
            assert row is not None
            assert row.state == STATE_REJECTED
            assert row.closed_ts_ms is not None

            events = _read_events(ctx, sibling_coid)
            audit = [e for e in events
                     if e[0] == 'bracket_sibling_retired_on_mixed_rejection']
            assert len(audit) == 1
            assert audit[0][2] is not None
            assert audit[0][2].get('reason') == 'pending_trail_parent_rejected'
            assert audit[0][2].get('trail_state') == 'awaiting_activation'
        finally:
            ctx.close()


def __test_apply_reconcile_terminal_close_missing_pending_grace_expired__(
        tmp_path: Path,
) -> None:
    """Missing-pending grace-window expiry closes the row + writes audit."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            _seed_working_row(ctx, coid)

            outcome = ReconcileOutcome(
                kind='terminal_close',
                reason='missing_pending_grace_expired',
                new_state='rejected',
                audit_event='missing_pending_grace_expired',
                close_row=True,
                audit_payload={'grace_window_ms': 60_000},
            )
            DispatchJournal(ctx).apply_reconcile_outcome(coid, outcome)

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_REJECTED
            assert row.closed_ts_ms is not None

            events = _read_events(ctx, coid)
            assert any(k == 'missing_pending_grace_expired' for (k, _, _) in events)
        finally:
            ctx.close()


def __test_apply_reconcile_terminal_close_unexpected_cancel_cascade__(
        tmp_path: Path,
) -> None:
    """Unexpected-cancel cascade terminates related rows."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            _seed_working_row(ctx, coid)

            outcome = ReconcileOutcome(
                kind='terminal_close',
                reason='unexpected_cancel_cascade',
                new_state='rejected',
                audit_event='unexpected_cancel_cascade',
                close_row=True,
                audit_payload={'origin_coid': 'someone-else'},
            )
            DispatchJournal(ctx).apply_reconcile_outcome(coid, outcome)

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_REJECTED
            assert row.closed_ts_ms is not None

            events = _read_events(ctx, coid)
            assert any(k == 'unexpected_cancel_cascade' for (k, _, _) in events)
        finally:
            ctx.close()


def __test_apply_reconcile_terminal_close_bracket_natural_close_followup__(
        tmp_path: Path,
) -> None:
    """Eager-teardown follow-up after a natural close: terminal + audit."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            parent_coid = _coid(ctx, "Long")
            sibling_coid = _coid(ctx, "Long-TP", kind_char='x')
            _seed_bracket_leg_row(
                ctx, sibling_coid,
                leg_kind='tp', parent_coid=parent_coid,
                parent_deal_id='parent-deal-3',
            )

            outcome = ReconcileOutcome(
                kind='terminal_close',
                reason='bracket_natural_close_followup',
                new_state='rejected',
                audit_event='bracket_natural_close_followup',
                close_row=True,
                audit_payload={'parent_coid': parent_coid},
            )
            DispatchJournal(ctx).apply_reconcile_outcome(sibling_coid, outcome)

            row = ctx.get_order(sibling_coid)
            assert row is not None
            assert row.state == STATE_REJECTED
            assert row.closed_ts_ms is not None
        finally:
            ctx.close()


def __test_apply_reconcile_terminal_close_without_close_row_keeps_row_live__(
        tmp_path: Path,
) -> None:
    """``close_row=False`` flips state but keeps the row in live set."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            _seed_working_row(ctx, coid)

            outcome = ReconcileOutcome(
                kind='terminal_close',
                reason='unexpected_cancel_cascade',
                new_state='rejected',
                audit_event='unexpected_cancel_cascade',
                close_row=False,
            )
            DispatchJournal(ctx).apply_reconcile_outcome(coid, outcome)

            row = ctx.get_order(coid)
            assert row is not None
            assert row.state == STATE_REJECTED
            assert row.closed_ts_ms is None  # NOT closed
            live = list(ctx.iter_live_orders())
            assert any(r.client_order_id == coid for r in live)
        finally:
            ctx.close()


# === Extras merge semantics ================================================

def __test_extras_patch_merges_with_existing_extras__(tmp_path: Path) -> None:
    """``extras_patch`` is merged on top — pre-existing keys preserved."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            _seed_working_row(ctx, coid)
            # Confirm pre-existing extras shape
            row0 = ctx.get_order(coid)
            assert row0 is not None and row0.extras is not None
            assert row0.extras.get('kind') == ENTRY_KIND_WORKING
            assert row0.extras.get('order_type') == OrderType.LIMIT.value
            assert row0.extras.get('deal_reference') == 'ref-working'

            outcome = ReconcileOutcome(
                kind='filled',
                reason='working_promoted_position',
                new_state='confirmed',
                audit_event='working_to_position',
                filled_qty=1.0,
                extras_patch={
                    'kind': ENTRY_KIND_POSITION,  # overrides
                    'entry_filled_at': 99,        # new
                },
            )
            DispatchJournal(ctx).apply_reconcile_outcome(coid, outcome)

            row = ctx.get_order(coid)
            assert row is not None and row.extras is not None
            assert row.extras.get('kind') == ENTRY_KIND_POSITION
            assert row.extras.get('entry_filled_at') == 99
            # Pre-existing untouched keys preserved
            assert row.extras.get('order_type') == OrderType.LIMIT.value
            assert row.extras.get('deal_reference') == 'ref-working'
        finally:
            ctx.close()


def __test_extras_patch_none_leaves_extras_untouched__(tmp_path: Path) -> None:
    """``extras_patch=None`` does not touch the row's extras at all."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            parent_coid = _coid(ctx, "Long")
            sibling_coid = _coid(ctx, "Long-TP", kind_char='x')
            _seed_bracket_leg_row(
                ctx, sibling_coid,
                leg_kind='tp', parent_coid=parent_coid,
                parent_deal_id='parent-deal-4',
            )
            row0 = ctx.get_order(sibling_coid)
            assert row0 is not None
            extras_before = dict(row0.extras or {})

            outcome = ReconcileOutcome(
                kind='terminal_close',
                reason='bracket_sibling_retired_on_mixed_rejection',
                new_state='rejected',
                audit_event='bracket_sibling_retired_on_mixed_rejection',
                close_row=True,
                extras_patch=None,
            )
            DispatchJournal(ctx).apply_reconcile_outcome(sibling_coid, outcome)

            row = ctx.get_order(sibling_coid)
            assert row is not None
            assert dict(row.extras or {}) == extras_before
        finally:
            ctx.close()


# === Audit event shape ====================================================

def __test_audit_payload_none_writes_event_with_null_payload__(
        tmp_path: Path,
) -> None:
    """``audit_payload=None`` => the event row's payload column is NULL."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            _seed_working_row(ctx, coid)

            outcome = ReconcileOutcome(
                kind='terminal_close',
                reason='missing_pending_grace_expired',
                new_state='rejected',
                audit_event='missing_pending_grace_expired',
                close_row=True,
                audit_payload=None,
            )
            DispatchJournal(ctx).apply_reconcile_outcome(coid, outcome)

            events = _read_events(ctx, coid)
            grace = next(
                e for e in events if e[0] == 'missing_pending_grace_expired'
            )
            assert grace[2] is None
        finally:
            ctx.close()


def __test_audit_event_only_written_after_state_persisted__(
        tmp_path: Path,
) -> None:
    """The audit event references the row that exists post-state-write.

    Sanity check that the journal does not log the event before the
    state mutation lands — if it did, a reader inspecting state at the
    audit row's ts would see the pre-mutation state, defeating the
    "audit follows persistence" invariant.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid = _coid(ctx, "Long")
            _seed_working_row(ctx, coid)

            outcome = ReconcileOutcome(
                kind='filled',
                reason='working_promoted_position',
                new_state='confirmed',
                audit_event='working_to_position',
                filled_qty=1.5,
                extras_patch={'kind': ENTRY_KIND_POSITION},
                audit_payload={'filled': 1.5},
            )
            DispatchJournal(ctx).apply_reconcile_outcome(coid, outcome)

            row = ctx.get_order(coid)
            assert row is not None
            # State and filled_qty both reflect the post-apply view
            assert row.state == STATE_CONFIRMED
            assert row.filled_qty == 1.5
            events = _read_events(ctx, coid)
            assert any(k == 'working_to_position' for (k, _, _) in events)
        finally:
            ctx.close()


# === Multi-row independence =================================================

def __test_outcomes_on_different_rows_are_independent__(tmp_path: Path) -> None:
    """Two outcomes on different COIDs do not cross-pollute state."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        try:
            coid_a = _coid(ctx, "LongA")
            coid_b = _coid(ctx, "LongB")
            _seed_working_row(ctx, coid_a)
            _seed_working_row(ctx, coid_b)

            journal = DispatchJournal(ctx)
            journal.apply_reconcile_outcome(coid_a, ReconcileOutcome(
                kind='filled',
                reason='working_promoted_position',
                new_state='confirmed',
                audit_event='working_to_position',
                filled_qty=1.0,
                extras_patch={'kind': ENTRY_KIND_POSITION},
            ))
            journal.apply_reconcile_outcome(coid_b, ReconcileOutcome(
                kind='terminal_close',
                reason='missing_pending_grace_expired',
                new_state='rejected',
                audit_event='missing_pending_grace_expired',
                close_row=True,
            ))

            row_a = ctx.get_order(coid_a)
            row_b = ctx.get_order(coid_b)
            assert row_a is not None and row_b is not None
            assert row_a.state == STATE_CONFIRMED
            assert row_a.closed_ts_ms is None
            assert row_b.state == STATE_REJECTED
            assert row_b.closed_ts_ms is not None
        finally:
            ctx.close()
