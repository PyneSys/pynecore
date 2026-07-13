"""
Behavior-matrix tests for the core disappearance tracker
(:mod:`pynecore.core.broker.disappearance`).

The tracker extracts the stamp/clear/grace state machine that the
Capital.com and cTrader plugins each implemented on their own. These
tests pin the venue-agnostic core against fake hooks that emulate both
plugins' semantics: the simple grace-expiry-equals-cancel behavior
(Capital.com) and the richer deal-history classification — inconclusive
reads, fills discovered during confirmation, filled-then-closed
retirement (cTrader).

Also covered: the store-transaction nesting the tracker's atomic
confirmation apply relies on.
"""
import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine, cast

import pytest

from pynecore.core.broker.disappearance import (
    MISSING_PENDING_EXTRA,
    DisappearanceTracker,
    MissingConfirmation,
    MissingResolution,
)
from pynecore.core.broker.exceptions import UnexpectedCancelError
from pynecore.core.broker.models import OrderEvent, OrderStatus
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import (
    BrokerStore,
    OrderRow,
    RunContext,
    TransactionRollbackError,
)
from pynecore.core.broker.store_helpers import STATE_CONFIRMED

PLUGIN = "TestBroker"
SCRIPT_SOURCE = "// disappearance tracker behavior-matrix test\n"

#: Stamp / grace anchors used across the tests. The tracker takes
#: explicit ``now_ts`` values, so the clock is fully deterministic.
T0 = 1_000.0
GRACE = 10.0
T_EXPIRED = T0 + GRACE + 1.0


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def _open_run(store: BrokerStore) -> RunContext:
    return store.open_run(
        RunIdentity(
            strategy_id="dt_test", symbol="EURUSD", timeframe="60",
            account_id="testbroker-demo", label=None,
        ),
        script_source=SCRIPT_SOURCE,
        script_path="strategies/dt_test.py",
    )


def _seed_row(
        ctx: RunContext, coid: str, *,
        eoid: str | None = "D1", qty: float = 1.0, filled: float = 0.0,
        extras: dict | None = None,
) -> None:
    ctx.upsert_order(
        coid, symbol="EURUSD", side="buy", qty=qty, filled_qty=filled,
        state=STATE_CONFIRMED, intent_key="Long", pine_entry_id="Long",
        exchange_order_id=eoid, extras=extras if extras is not None else {},
    )


def _read_events(ctx: RunContext, kind: str) -> list[tuple[str | None, dict | None]]:
    """Return (client_order_id, payload) tuples for an event kind."""
    rows = ctx._store._conn.execute(  # type: ignore[attr-defined]
        "SELECT client_order_id, payload FROM events "
        "WHERE run_instance_id = ? AND kind = ? ORDER BY ts_ms, id",
        (ctx.run_instance_id, kind),
    ).fetchall()
    return [(r['client_order_id'],
             json.loads(r['payload']) if r['payload'] else None)
            for r in rows]


def _confirm_const(
        confirmation: MissingConfirmation,
) -> Callable[[OrderRow], Awaitable[MissingConfirmation]]:
    async def _confirm(_row: OrderRow) -> MissingConfirmation:
        return confirmation
    return _confirm


def _make_tracker(
        ctx: RunContext, *,
        policy: str = 'stop',
        confirm: Callable[[OrderRow], Awaitable[MissingConfirmation]] | None = None,
        **kwargs: Any,
) -> DisappearanceTracker:
    return DisappearanceTracker(
        ctx, grace_s=GRACE, policy=policy,
        tracked_refs=lambda row: (
            {('working', row.exchange_order_id)}
            if row.exchange_order_id else set()
        ),
        confirm_missing=confirm or _confirm_const(
            MissingConfirmation(MissingResolution.CANCELLED)),
        **kwargs,
    )


async def _drain(
        tracker: DisappearanceTracker,
        present: dict[str, set[str] | None],
        now_ts: float,
) -> list[OrderEvent]:
    return [ev async for ev in tracker.observe(present, now_ts)]


# === Phase 1: stamp / clear ==================================================

def __test_stamps_on_proven_absence_and_clears_on_return__(tmp_path: Path) -> None:
    """Full absence stamps ``missing_pending_since``; reappearance clears it."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1")
        tracker = _make_tracker(ctx)

        events = _run(_drain(tracker, {'working': set()}, T0))
        assert events == []
        row = ctx.get_order("c1")
        assert row is not None
        assert row.extras[MISSING_PENDING_EXTRA] == T0

        events = _run(_drain(tracker, {'working': {"D1"}}, T0 + 2.0))
        assert events == []
        row = ctx.get_order("c1")
        assert row is not None
        assert MISSING_PENDING_EXTRA not in row.extras


def __test_incomplete_namespace_snapshot_never_stamps__(tmp_path: Path) -> None:
    """A failed namespace fetch proves nothing: no stamp, no clear."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1")
        tracker = DisappearanceTracker(
            ctx, grace_s=GRACE, policy='stop',
            tracked_refs=lambda r: {('working', 'D1'), ('position', 'D1')},
            confirm_missing=_confirm_const(
                MissingConfirmation(MissingResolution.CANCELLED)),
        )

        # Explicit None marks a failed fetch.
        _run(_drain(tracker, {'working': None, 'position': None}, T0))
        row = ctx.get_order("c1")
        assert row is not None and MISSING_PENDING_EXTRA not in row.extras

        # A missing key is the same as a failed fetch.
        _run(_drain(tracker, {}, T0))
        row = ctx.get_order("c1")
        assert row is not None and MISSING_PENDING_EXTRA not in row.extras

        # One namespace absent + the other failed: absence is unproven.
        _run(_drain(tracker, {'working': set(), 'position': None}, T0))
        row = ctx.get_order("c1")
        assert row is not None and MISSING_PENDING_EXTRA not in row.extras

        # Every relevant namespace fetched and every ref absent: stamp.
        _run(_drain(tracker, {'working': set(), 'position': set()}, T0))
        row = ctx.get_order("c1")
        assert row is not None
        assert row.extras[MISSING_PENDING_EXTRA] == T0


def __test_any_visible_ref_clears_despite_failed_namespace__(tmp_path: Path) -> None:
    """Clear wins on ANY visible authoritative ref, even mid-outage."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        tracker = DisappearanceTracker(
            ctx, grace_s=GRACE, policy='stop',
            tracked_refs=lambda r: {('working', 'D1'), ('position', 'D1')},
            confirm_missing=_confirm_const(
                MissingConfirmation(MissingResolution.CANCELLED)),
        )
        _run(_drain(tracker, {'working': {'D1'}, 'position': None}, T0 + 1.0))
        row = ctx.get_order("c1")
        assert row is not None and MISSING_PENDING_EXTRA not in row.extras


def __test_exempt_and_untracked_rows_are_ignored__(tmp_path: Path) -> None:
    """``is_exempt`` and empty ``tracked_refs`` both bypass the machine."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        # Untracked: no exchange ref -> tracked_refs returns an empty set.
        _seed_row(ctx, "no-ref", eoid=None)
        # Exempt: pre-stamped and expired, but the hook exempts it.
        _seed_row(ctx, "exempt", eoid="D9",
                  extras={MISSING_PENDING_EXTRA: T0, 'natural_close_at': T0})
        confirm_calls: list[str] = []

        async def _confirm(row: OrderRow) -> MissingConfirmation:
            confirm_calls.append(row.client_order_id)
            return MissingConfirmation(MissingResolution.CANCELLED)

        tracker = _make_tracker(
            ctx, confirm=_confirm,
            is_exempt=lambda row: 'natural_close_at' in (row.extras or {}),
        )
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert events == []
        assert confirm_calls == []
        no_ref = ctx.get_order("no-ref")
        assert no_ref is not None and MISSING_PENDING_EXTRA not in no_ref.extras
        exempt = ctx.get_order("exempt")
        assert exempt is not None and exempt.closed_ts_ms is None


# === Phase 2: grace + confirmation ==========================================

def __test_grace_window_defers_confirmation__(tmp_path: Path) -> None:
    """No confirmation call before the stamp ages past the grace window."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        confirm_calls: list[str] = []

        async def _confirm(row: OrderRow) -> MissingConfirmation:
            confirm_calls.append(row.client_order_id)
            return MissingConfirmation(MissingResolution.CANCELLED)

        tracker = _make_tracker(ctx, confirm=_confirm)
        events = _run(_drain(tracker, {'working': set()}, T0 + GRACE - 1.0))
        assert events == []
        assert confirm_calls == []


def __test_confirm_still_present_clears_stamp__(tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        tracker = _make_tracker(ctx, confirm=_confirm_const(
            MissingConfirmation(MissingResolution.STILL_PRESENT)))
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert events == []
        row = ctx.get_order("c1")
        assert row is not None
        assert MISSING_PENDING_EXTRA not in row.extras
        assert row.closed_ts_ms is None


def __test_confirm_inconclusive_keeps_stamp_and_warns_once__(tmp_path: Path) -> None:
    """An inconclusive re-check defers, keeps the stamp, and warns once
    per row until the check resolves — then re-arms after a clear."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        tracker = _make_tracker(ctx, confirm=_confirm_const(
            MissingConfirmation(MissingResolution.INCONCLUSIVE)))

        for offset in (1.0, 2.0, 3.0):
            events = _run(_drain(tracker, {'working': set()},
                                 T_EXPIRED + offset))
            assert events == []
        row = ctx.get_order("c1")
        assert row is not None
        assert row.extras[MISSING_PENDING_EXTRA] == T0
        assert row.closed_ts_ms is None
        assert len(_read_events(ctx, 'missing_pending_recheck_inconclusive')) == 1

        # The row comes back -> clear resets the warn throttle; a later
        # re-stamped inconclusive cycle warns again.
        _run(_drain(tracker, {'working': {'D1'}}, T_EXPIRED + 4.0))
        _run(_drain(tracker, {'working': set()}, T_EXPIRED + 5.0))
        _run(_drain(tracker, {'working': set()},
                    T_EXPIRED + 5.0 + GRACE + 1.0))
        assert len(_read_events(ctx, 'missing_pending_recheck_inconclusive')) == 2


def __test_confirm_cancelled_dual_signal_stop_policy__(tmp_path: Path) -> None:
    """CANCELLED: persist first, yield the cancelled event, then halt."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        tracker = _make_tracker(ctx, policy='stop')

        async def scenario() -> None:
            gen = tracker.observe({'working': set()}, T_EXPIRED)
            event = await anext(gen)
            assert event.event_type == 'cancelled'
            assert event.order.client_order_id == "c1"
            assert event.order.status is OrderStatus.CANCELLED
            assert event.pine_id == "Long"
            # Persisted BEFORE the event was even yielded.
            row = ctx.get_order("c1")
            assert row is not None
            assert row.state == 'rejected'
            assert row.closed_ts_ms is not None
            with pytest.raises(UnexpectedCancelError):
                await anext(gen)

        _run(scenario())
        cancels = _read_events(ctx, 'unexpected_cancel')
        assert cancels == [("c1", {'missing_since': T0, 'grace': GRACE})]


def __test_abandoned_generator_still_halts_next_pass__(tmp_path: Path) -> None:
    """Persist + policy never depend on the consumer pulling the next
    element: a halt decided under an abandoned generator re-raises on
    the next observe call."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        tracker = _make_tracker(ctx, policy='stop')

        async def scenario() -> None:
            # ``observe`` is annotated as AsyncIterator; the concrete
            # object is an async generator, whose aclose() simulates the
            # consumer abandoning the stream mid-drain.
            gen = cast('AsyncGenerator[OrderEvent, None]',
                       tracker.observe({'working': set()}, T_EXPIRED))
            event = await anext(gen)
            assert event.event_type == 'cancelled'
            await gen.aclose()
            assert tracker.pending_halt is not None
            row = ctx.get_order("c1")
            assert row is not None and row.closed_ts_ms is not None
            with pytest.raises(UnexpectedCancelError):
                await anext(tracker.observe({'working': set()},
                                            T_EXPIRED + 1.0))

        _run(scenario())


def __test_policy_ignore_retires_without_halt__(tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        tracker = _make_tracker(ctx, policy='ignore')
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert [ev.event_type for ev in events] == ['cancelled']
        assert tracker.pending_halt is None
        row = ctx.get_order("c1")
        assert row is not None and row.state == 'rejected'
        assert len(_read_events(ctx, 'unexpected_cancel_ignored')) == 1


def __test_policy_re_place_retires_without_halt__(tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        tracker = _make_tracker(ctx, policy='re_place')
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert [ev.event_type for ev in events] == ['cancelled']
        assert tracker.pending_halt is None
        assert len(_read_events(ctx, 'unexpected_cancel_re_place')) == 1


def __test_policy_stop_and_cancel_sweeps_before_event__(tmp_path: Path) -> None:
    """The sibling sweep runs before the cancelled event is delivered."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        swept: list[str] = []

        async def _sweep(row: OrderRow) -> None:
            swept.append(row.client_order_id)

        tracker = _make_tracker(ctx, policy='stop_and_cancel',
                                cancel_siblings=_sweep)

        async def scenario() -> None:
            gen = tracker.observe({'working': set()}, T_EXPIRED)
            event = await anext(gen)
            assert event.event_type == 'cancelled'
            assert swept == ["c1"]
            with pytest.raises(UnexpectedCancelError):
                await anext(gen)

        _run(scenario())


def __test_stop_and_cancel_requires_sweep_hook__(tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        with pytest.raises(ValueError):
            _make_tracker(ctx, policy='stop_and_cancel')
        with pytest.raises(ValueError):
            _make_tracker(ctx, policy='bogus')


def __test_policy_stop_with_quarantine_hook_does_not_halt__(
        tmp_path: Path,
) -> None:
    """With the quarantine hook wired, ``stop`` latches quarantine and the
    observation loop survives — the process-exiting halt is never armed."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        latched: list[tuple[str, dict]] = []
        tracker = _make_tracker(
            ctx, policy='stop',
            request_quarantine=lambda r, c: latched.append((r, c)),
        )
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert [ev.event_type for ev in events] == ['cancelled']
        assert tracker.pending_halt is None
        assert len(latched) == 1
        reason, context = latched[0]
        assert "c1" in reason
        assert context['policy'] == 'stop'
        assert context['client_order_id'] == "c1"
        row = ctx.get_order("c1")
        assert row is not None and row.state == 'rejected'
        assert len(_read_events(ctx, 'unexpected_cancel_quarantine')) == 1
        # The stream stays usable: a later pass neither raises nor
        # re-fires the already-latched quarantine for the retired row.
        assert _run(_drain(tracker, {'working': set()}, T_EXPIRED + 1.0)) == []
        assert len(latched) == 1


def __test_policy_halt_raises_process_exit_signal__(tmp_path: Path) -> None:
    """The explicit ``halt`` policy keeps the process-exiting behavior:
    dual signal first, then the raise."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        latched: list[tuple[str, dict]] = []
        tracker = _make_tracker(
            ctx, policy='halt',
            request_quarantine=lambda reason, context: latched.append(
                (reason, context)),
        )

        async def scenario() -> None:
            gen = tracker.observe({'working': set()}, T_EXPIRED)
            event = await anext(gen)
            assert event.event_type == 'cancelled'
            with pytest.raises(UnexpectedCancelError):
                await anext(gen)

        _run(scenario())
        # ``halt`` never touches the quarantine hook, even when wired.
        assert latched == []
        assert len(_read_events(ctx, 'unexpected_cancel_quarantine')) == 0


def __test_quarantine_hook_failure_falls_back_to_halt__(
        tmp_path: Path,
) -> None:
    """A raising quarantine hook must fail-safe into the halt, never
    fail-open into continued trading."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})

        def _boom(_reason: str, _context: dict) -> None:
            raise RuntimeError("latch wiring broken")

        tracker = _make_tracker(ctx, policy='stop', request_quarantine=_boom)

        async def scenario() -> None:
            gen = tracker.observe({'working': set()}, T_EXPIRED)
            event = await anext(gen)
            assert event.event_type == 'cancelled'
            with pytest.raises(UnexpectedCancelError):
                await anext(gen)

        _run(scenario())
        assert len(_read_events(ctx, 'unexpected_cancel_quarantine')) == 0


def __test_policy_stop_and_cancel_quarantines_and_sweeps__(
        tmp_path: Path,
) -> None:
    """``stop_and_cancel`` with the hook wired: quarantine latched, sweep
    still runs, no halt."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        latched: list[str] = []
        swept: list[str] = []

        async def _sweep(row: OrderRow) -> None:
            swept.append(row.client_order_id)

        tracker = _make_tracker(
            ctx, policy='stop_and_cancel', cancel_siblings=_sweep,
            request_quarantine=lambda reason, _context: latched.append(reason),
        )
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert [ev.event_type for ev in events] == ['cancelled']
        assert tracker.pending_halt is None
        assert len(latched) == 1
        assert swept == ["c1"]


# === Confirmation with discovered fills =====================================

def __test_confirm_filled_books_slice_and_clears_stamp__(tmp_path: Path) -> None:
    """FILLED with fill data: slice booked, extras patched, stamp cleared,
    row stays live, execution ids handed to the dedup hook."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        registered: list[tuple[str, float, tuple[str, ...]]] = []
        tracker = _make_tracker(
            ctx,
            confirm=_confirm_const(MissingConfirmation(
                MissingResolution.FILLED,
                cumulative_filled_qty=0.4, fill_price=1.2345, fill_fee=0.1,
                execution_ids=('E1',), position_ref='P9',
                extras_patch={'position_id': 'P9'}, executed_ts=T0 + 5.0,
            )),
            register_executions=lambda r, ids: registered.append(
                (r.client_order_id, r.filled_qty, ids)),
        )
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert len(events) == 1
        event = events[0]
        assert event.event_type == 'partial'
        assert event.fill_qty == pytest.approx(0.4)
        assert event.fill_price == pytest.approx(1.2345)
        assert event.fill_id == 'E1'
        assert event.timestamp == T0 + 5.0
        assert event.order.filled_qty == pytest.approx(0.4)
        row = ctx.get_order("c1")
        assert row is not None
        assert row.filled_qty == pytest.approx(0.4)
        assert row.state == 'confirmed'
        assert row.closed_ts_ms is None
        assert MISSING_PENDING_EXTRA not in row.extras
        assert row.extras['position_id'] == 'P9'
        # The dedup-seed hook sees the POST-fill row (filled_qty already 0.4).
        assert len(registered) == 1
        assert registered[0][0] == "c1"
        assert registered[0][1] == pytest.approx(0.4)
        assert registered[0][2] == ('E1',)
        recovered = _read_events(ctx, 'reconcile_missing_fill_recovered')
        assert len(recovered) == 1
        payload = recovered[0][1]
        assert payload is not None
        assert payload['execution_ids'] == ['E1']


def __test_confirm_filled_monotonic_clamp__(tmp_path: Path) -> None:
    """The cumulative never regresses and never exceeds the order size."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        # Regression attempt: cumulative below the recorded fill.
        _seed_row(ctx, "c1", filled=0.5,
                  extras={MISSING_PENDING_EXTRA: T0})
        tracker = _make_tracker(ctx, confirm=_confirm_const(
            MissingConfirmation(MissingResolution.FILLED,
                                cumulative_filled_qty=0.3)))
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert events == []
        row = ctx.get_order("c1")
        assert row is not None
        assert row.filled_qty == pytest.approx(0.5)
        assert MISSING_PENDING_EXTRA not in row.extras

        # Overstatement: cumulative past the order size clamps to qty.
        _seed_row(ctx, "c2", eoid="D2", filled=0.5,
                  extras={MISSING_PENDING_EXTRA: T0})
        tracker2 = _make_tracker(ctx, confirm=_confirm_const(
            MissingConfirmation(MissingResolution.FILLED,
                                cumulative_filled_qty=5.0, fill_price=1.5)))
        events = _run(_drain(tracker2, {'working': set()}, T_EXPIRED))
        assert len(events) == 1
        assert events[0].event_type == 'filled'
        assert events[0].fill_qty == pytest.approx(0.5)
        assert events[0].order.status is OrderStatus.FILLED
        row2 = ctx.get_order("c2")
        assert row2 is not None
        assert row2.filled_qty == pytest.approx(1.0)


def __test_confirm_closed_retires_row_and_siblings_without_event__(
        tmp_path: Path,
) -> None:
    """CLOSED: natural-close retirement — no cancel event, no fill event
    for exposure that no longer exists, siblings retired atomically."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", filled=1.0,
                  extras={MISSING_PENDING_EXTRA: T0, 'position_id': 'P1'})
        _seed_row(ctx, "sib", eoid="D2", filled=1.0,
                  extras={'position_id': 'P1'})
        tracker = _make_tracker(
            ctx,
            confirm=_confirm_const(MissingConfirmation(
                MissingResolution.CLOSED, position_ref='P1')),
            sibling_coids=lambda origin, conf: [
                r.client_order_id for r in ctx.iter_live_orders()
                if r.client_order_id != origin.client_order_id
                and (r.extras or {}).get('position_id') == conf.position_ref
            ],
        )
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert events == []
        assert tracker.pending_halt is None
        row = ctx.get_order("c1")
        assert row is not None
        assert row.state == 'closed'
        assert row.closed_ts_ms is not None
        sibling = ctx.get_order("sib")
        assert sibling is not None and sibling.closed_ts_ms is not None
        retired = _read_events(ctx, 'reconcile_filled_then_closed_retired')
        assert len(retired) == 1


def __test_terminal_resolution_registers_execution_ids_without_fill__(
        tmp_path: Path,
) -> None:
    """CLOSED / CANCELLED register their backing execution ids even when
    no fresh fill slice was booked: the retirement was concluded FROM
    that evidence, so a replayed push copy must already be suppressed."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", filled=1.0,
                  extras={MISSING_PENDING_EXTRA: T0, 'position_id': 'P1'})
        _seed_row(ctx, "c2", eoid="D2",
                  extras={MISSING_PENDING_EXTRA: T0})
        registered: list[tuple[str, tuple[str, ...]]] = []

        async def _confirm(row: OrderRow) -> MissingConfirmation:
            if row.client_order_id == "c1":
                # Fully-filled position closed externally: the closing
                # deals prove the closure but book no new quantity.
                return MissingConfirmation(
                    MissingResolution.CLOSED, position_ref='P1',
                    execution_ids=('DEAL-CLOSE-1', 'DEAL-CLOSE-2'),
                )
            # Zero-fill order cancelled: evidence ids, nothing to book.
            return MissingConfirmation(
                MissingResolution.CANCELLED,
                execution_ids=('DEAL-CXL-1',),
            )

        tracker = _make_tracker(
            ctx, policy='ignore', confirm=_confirm,
            register_executions=lambda row, ids: registered.append(
                (row.client_order_id, ids)),
        )
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert [ev.event_type for ev in events] == ['cancelled']
        assert sorted(registered) == [
            ("c1", ('DEAL-CLOSE-1', 'DEAL-CLOSE-2')),
            ("c2", ('DEAL-CXL-1',)),
        ]


def __test_partial_fill_then_cancel_preserves_slice_atomically__(
        tmp_path: Path,
) -> None:
    """CANCELLED with a discovered fill slice books the fill AND the
    terminal close together: fill event first, cancelled event second."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        tracker = _make_tracker(
            ctx, policy='ignore',
            confirm=_confirm_const(MissingConfirmation(
                MissingResolution.CANCELLED,
                cumulative_filled_qty=0.4, fill_price=1.1,
                execution_ids=('E7',),
            )),
        )
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert [ev.event_type for ev in events] == ['partial', 'cancelled']
        assert events[0].fill_qty == pytest.approx(0.4)
        assert events[0].fill_id == 'E7'
        # The cancelled event carries the POST-fill quantity — the engine
        # gates the parent's native fail-safe retirement on filled_qty<=0,
        # so a stale zero here would strand a live partial position.
        assert events[1].event_type == 'cancelled'
        assert events[1].order.filled_qty == pytest.approx(0.4)
        row = ctx.get_order("c1")
        assert row is not None
        assert row.filled_qty == pytest.approx(0.4)
        assert row.state == 'rejected'
        assert row.closed_ts_ms is not None


def __test_stale_confirmation_dropped_when_row_returns_mid_confirm__(
        tmp_path: Path,
) -> None:
    """The stamp-version guard drops an outcome whose row came back (the
    stamp was cleared by another path) while the confirmation awaited."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})

        async def _confirm(confirmed_row: OrderRow) -> MissingConfirmation:
            # Simulates a PUSH-path comeback landing during the await:
            # the stamp is cleared underneath the running confirmation.
            ctx.upsert_order(confirmed_row.client_order_id, extras={})
            return MissingConfirmation(MissingResolution.CANCELLED)

        tracker = _make_tracker(ctx, policy='stop', confirm=_confirm)
        events = _run(_drain(tracker, {'working': None}, T_EXPIRED))
        assert events == []
        assert tracker.pending_halt is None
        row = ctx.get_order("c1")
        assert row is not None
        assert row.state == 'confirmed'
        assert row.closed_ts_ms is None
        assert _read_events(ctx, 'unexpected_cancel') == []


# === Review hardening: fail-closed, defensive hooks, consume-once ===========

def __test_unpriced_fill_defers_and_keeps_stamp__(tmp_path: Path) -> None:
    """A discovered fill without a usable price is never booked: the store
    is untouched, the stamp is kept, and nothing is handed to the dedup
    hook (which would suppress the real fill event later)."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        registered: list[tuple[str, tuple[str, ...]]] = []
        tracker = _make_tracker(
            ctx,
            confirm=_confirm_const(MissingConfirmation(
                MissingResolution.FILLED,
                cumulative_filled_qty=0.4, fill_price=None,
                execution_ids=('E1',),
            )),
            register_executions=lambda r, ids: registered.append(
                (r.client_order_id, ids)),
        )
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert events == []
        assert registered == []
        row = ctx.get_order("c1")
        assert row is not None
        assert row.filled_qty == pytest.approx(0.0)
        assert row.extras[MISSING_PENDING_EXTRA] == T0
        assert row.closed_ts_ms is None
        assert len(_read_events(ctx, 'missing_pending_fill_unpriced')) == 1


def __test_non_finite_fill_price_defers_like_missing__(tmp_path: Path) -> None:
    """nan / +inf / -inf prices are as unbookable as a missing price: each
    defers, keeping the stamp and touching neither store nor dedup."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        outcome: dict[str, MissingConfirmation] = {}

        async def _confirm(_row: OrderRow) -> MissingConfirmation:
            return outcome['value']

        tracker = _make_tracker(ctx, confirm=_confirm)
        for i, bad in enumerate((float('nan'), float('inf'), float('-inf'))):
            outcome['value'] = MissingConfirmation(
                MissingResolution.FILLED,
                cumulative_filled_qty=0.4, fill_price=bad)
            events = _run(_drain(tracker, {'working': set()}, T_EXPIRED + i))
            assert events == []
        row = ctx.get_order("c1")
        assert row is not None
        assert row.filled_qty == pytest.approx(0.0)
        assert row.extras[MISSING_PENDING_EXTRA] == T0
        assert row.closed_ts_ms is None


def __test_throwing_register_executions_does_not_strand_row__(
        tmp_path: Path,
) -> None:
    """A raising dedup-seed hook is logged, not propagated: the cancelled
    row still emits both its fill and cancelled events."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})

        def _boom(_row: OrderRow, _ids: tuple[str, ...]) -> None:
            raise RuntimeError("dedup channel down")

        tracker = _make_tracker(
            ctx, policy='ignore',
            confirm=_confirm_const(MissingConfirmation(
                MissingResolution.CANCELLED,
                cumulative_filled_qty=0.4, fill_price=1.1,
                execution_ids=('E7',),
            )),
            register_executions=_boom,
        )
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert [ev.event_type for ev in events] == ['partial', 'cancelled']
        row = ctx.get_order("c1")
        assert row is not None and row.closed_ts_ms is not None


def __test_throwing_cancel_siblings_still_arms_halt__(tmp_path: Path) -> None:
    """A best-effort sweep that raises must not swallow the halt: it is
    armed BEFORE the sweep, logged, and still delivered."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})

        async def _sweep(_row: OrderRow) -> None:
            raise RuntimeError("sweep transport down")

        tracker = _make_tracker(ctx, policy='stop_and_cancel',
                                cancel_siblings=_sweep)

        async def scenario() -> None:
            gen = tracker.observe({'working': set()}, T_EXPIRED)
            event = await anext(gen)
            assert event.event_type == 'cancelled'
            with pytest.raises(UnexpectedCancelError):
                await anext(gen)

        _run(scenario())
        assert len(_read_events(ctx, 'unexpected_cancel_sweep_failed')) == 1


def __test_cancel_siblings_sweep_receives_post_fill_row__(tmp_path: Path) -> None:
    """A partial-fill-then-cancel under stop_and_cancel hands the sweep the
    POST-fill row, so it can size its cleanup on the recovered quantity."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        swept: list[float] = []

        async def _sweep(r: OrderRow) -> None:
            swept.append(r.filled_qty)

        tracker = _make_tracker(
            ctx, policy='stop_and_cancel', cancel_siblings=_sweep,
            confirm=_confirm_const(MissingConfirmation(
                MissingResolution.CANCELLED,
                cumulative_filled_qty=0.4, fill_price=1.1,
                execution_ids=('E7',),
            )),
        )

        async def scenario() -> None:
            gen = tracker.observe({'working': set()}, T_EXPIRED)
            assert (await anext(gen)).event_type == 'partial'
            assert (await anext(gen)).event_type == 'cancelled'
            with pytest.raises(UnexpectedCancelError):
                await anext(gen)

        _run(scenario())
        assert swept == [pytest.approx(0.4)]


def __test_promotion_without_quantity_only_merges_metadata__(
        tmp_path: Path,
) -> None:
    """FILLED with a position_ref but NO cumulative quantity does not flip
    state via a fill outcome — a position reference proves association,
    not an authoritative filled qty. Any extras patch is still merged and
    the normal snapshot path completes the promotion."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", filled=0.0,
                  extras={MISSING_PENDING_EXTRA: T0})
        tracker = _make_tracker(ctx, confirm=_confirm_const(
            MissingConfirmation(
                MissingResolution.FILLED, position_ref='P3',
                extras_patch={'position_id': 'P3'},
            )))
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert events == []
        row = ctx.get_order("c1")
        assert row is not None
        assert row.filled_qty == pytest.approx(0.0)
        assert row.extras['position_id'] == 'P3'
        assert MISSING_PENDING_EXTRA not in row.extras
        # No manufactured fill outcome for a zero-quantity promotion.
        assert _read_events(ctx, 'reconcile_missing_fill_recovered') == []


def __test_concurrent_breadcrumb_preserved_on_clear__(tmp_path: Path) -> None:
    """Phase-1 clearing re-reads the live row: a breadcrumb written between
    the snapshot and the clear survives; only the stamp key is removed."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})

        def _tracked(r: OrderRow) -> set[tuple[str, str]]:
            if not (r.extras or {}).get('pushed'):
                ctx.upsert_order(
                    "c1", extras={MISSING_PENDING_EXTRA: T0, 'pushed': 'Y'})
            return ({('working', r.exchange_order_id)}
                    if r.exchange_order_id else set())

        tracker = DisappearanceTracker(
            ctx, grace_s=GRACE, policy='stop', tracked_refs=_tracked,
            confirm_missing=_confirm_const(
                MissingConfirmation(MissingResolution.CANCELLED)),
        )
        # The ref is visible -> phase-1 clears the stamp.
        _run(_drain(tracker, {'working': {'D1'}}, T0 + 1.0))
        row = ctx.get_order("c1")
        assert row is not None
        assert row.extras['pushed'] == 'Y'
        assert MISSING_PENDING_EXTRA not in row.extras


def __test_halt_delivered_exactly_once__(tmp_path: Path) -> None:
    """The pending halt is consumed by the raise: a later pass with no new
    disappearance does not re-raise the same halt."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        tracker = _make_tracker(ctx, policy='stop')

        with pytest.raises(UnexpectedCancelError):
            _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        assert tracker.pending_halt is None

        # Second pass: the row is retired, no new halt, no re-raise.
        events = _run(_drain(tracker, {'working': set()}, T_EXPIRED + 1.0))
        assert events == []
        assert tracker.pending_halt is None


def __test_concurrent_breadcrumb_preserved_on_stamp__(tmp_path: Path) -> None:
    """Phase-1 stamping re-reads the live row: a breadcrumb another thread
    wrote after the snapshot but before the stamp is preserved, not
    clobbered by the stale snapshot's extras."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={})

        def _tracked(r: OrderRow) -> set[tuple[str, str]]:
            # Simulate a concurrent push-path write landing between the
            # phase-1 snapshot read and the stamp write.
            if not (r.extras or {}).get('pushed'):
                ctx.upsert_order("c1", extras={'pushed': 'Y'})
            return ({('working', r.exchange_order_id)}
                    if r.exchange_order_id else set())

        tracker = DisappearanceTracker(
            ctx, grace_s=GRACE, policy='stop', tracked_refs=_tracked,
            confirm_missing=_confirm_const(
                MissingConfirmation(MissingResolution.CANCELLED)),
        )
        _run(_drain(tracker, {'working': set()}, T0))
        row = ctx.get_order("c1")
        assert row is not None
        assert row.extras['pushed'] == 'Y'
        assert row.extras[MISSING_PENDING_EXTRA] == T0


def __test_deferred_throttle_is_per_reason__(tmp_path: Path) -> None:
    """An inconclusive re-check must not mute a later unpriced-fill audit on
    the same row: the throttle keys on (coid, reason), not coid alone."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})
        outcome = {'value': MissingConfirmation(MissingResolution.INCONCLUSIVE)}

        async def _confirm(_row: OrderRow) -> MissingConfirmation:
            return outcome['value']

        tracker = _make_tracker(ctx, confirm=_confirm)
        _run(_drain(tracker, {'working': set()}, T_EXPIRED))
        # Same row, now an unpriced fill during the next grace-expired pass.
        outcome['value'] = MissingConfirmation(
            MissingResolution.FILLED,
            cumulative_filled_qty=0.4, fill_price=None)
        _run(_drain(tracker, {'working': set()}, T_EXPIRED + 1.0))
        assert len(_read_events(ctx, 'missing_pending_recheck_inconclusive')) == 1
        assert len(_read_events(ctx, 'missing_pending_fill_unpriced')) == 1


def __test_invalid_resolution_rejected_at_construction__() -> None:
    """A bogus resolution raises at construction; a plain string equal to
    an enum value is coerced (never falls through to CANCELLED)."""
    with pytest.raises(ValueError):
        MissingConfirmation('not_a_resolution')  # type: ignore[arg-type]
    coerced = MissingConfirmation('cancelled')  # type: ignore[arg-type]
    assert coerced.resolution is MissingResolution.CANCELLED


def __test_stale_inconclusive_after_stamp_cleared_is_dropped__(
        tmp_path: Path,
) -> None:
    """An INCONCLUSIVE outcome whose stamp was cleared mid-confirm is
    dropped by the guard — no stale warning is written."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", extras={MISSING_PENDING_EXTRA: T0})

        async def _confirm(confirmed_row: OrderRow) -> MissingConfirmation:
            ctx.upsert_order(confirmed_row.client_order_id, extras={})
            return MissingConfirmation(MissingResolution.INCONCLUSIVE)

        tracker = _make_tracker(ctx, confirm=_confirm)
        events = _run(_drain(tracker, {'working': None}, T_EXPIRED))
        assert events == []
        assert _read_events(ctx, 'missing_pending_recheck_inconclusive') == []


# === Store-transaction nesting (the atomic-apply substrate) =================

def __test_store_transaction_nesting_is_atomic__(tmp_path: Path) -> None:
    """Nested ``transaction()`` blocks join one span: an exception rolls
    back every write made through the nested helpers."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1")

        class _Boom(Exception):
            pass

        with pytest.raises(_Boom):
            with ctx.transaction():
                ctx.upsert_order("c1", filled_qty=0.7)
                ctx.log_event('nested_txn_probe', client_order_id="c1")
                raise _Boom()

        row = ctx.get_order("c1")
        assert row is not None
        assert row.filled_qty == pytest.approx(0.0)
        assert _read_events(ctx, 'nested_txn_probe') == []

        # The happy path commits both nested writes.
        with ctx.transaction():
            ctx.upsert_order("c1", filled_qty=0.7)
            ctx.log_event('nested_txn_probe', client_order_id="c1")
        row = ctx.get_order("c1")
        assert row is not None
        assert row.filled_qty == pytest.approx(0.7)
        assert len(_read_events(ctx, 'nested_txn_probe')) == 1


def __test_store_transaction_swallowed_inner_exception_rolls_back__(
        tmp_path: Path,
) -> None:
    """A nested-level exception caught INSIDE the outer block still aborts
    the whole span: the outer rolls back and surfaces a
    ``TransactionRollbackError`` so the discarded span cannot be mistaken
    for a commit."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", filled=0.1)

        class _Boom(Exception):
            pass

        with pytest.raises(TransactionRollbackError) as exc_info:
            with ctx.transaction():
                ctx.upsert_order("c1", filled_qty=0.2)
                try:
                    with ctx.transaction():
                        ctx.upsert_order("c1", filled_qty=0.3)
                        raise _Boom()
                except _Boom:
                    pass
                # The caller swallowed the inner failure and keeps writing —
                # the span is already rollback-only, so this is discarded too.
                ctx.upsert_order("c1", filled_qty=0.9)

        # The surfaced error chains the first nested cause.
        assert isinstance(exc_info.value.__cause__, _Boom)
        row = ctx.get_order("c1")
        assert row is not None
        assert row.filled_qty == pytest.approx(0.1)


def __test_store_transaction_unswallowed_nested_propagates_original__(
        tmp_path: Path,
) -> None:
    """An inner exception that is NOT swallowed propagates the exact
    original object (not TransactionRollbackError) and still rolls back."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        _seed_row(ctx, "c1", filled=0.1)

        class _Boom(Exception):
            pass

        boom = _Boom()
        with pytest.raises(_Boom) as exc_info:
            with ctx.transaction():
                ctx.upsert_order("c1", filled_qty=0.5)
                with ctx.transaction():
                    ctx.upsert_order("c1", filled_qty=0.7)
                    raise boom

        assert exc_info.value is boom
        row = ctx.get_order("c1")
        assert row is not None
        assert row.filled_qty == pytest.approx(0.1)
