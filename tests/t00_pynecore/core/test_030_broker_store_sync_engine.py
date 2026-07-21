"""
End-to-end integration: :class:`OrderSyncEngine` × :class:`BrokerStore`.

These tests cover the full pipeline previously provided by the old
WS1.5 recovery layer, now via the unified SQLite-based storage:

- After a restart, the same ``client_order_id`` is produced for a live
  intent (by reading back the persisted ``bar_ts_ms`` anchor).
- Parked dispatches are recovered on the next sync from the
  ``get_open_orders`` view, without re-dispatching.
- ``record_complete`` actually drops the persisted row — a cancelled
  intent does not resurrect.
- Stale-run cleanup, triggered on the next ``open_run`` call, closes
  a previously crashed run (SIGKILL simulation).

Unit tests for the ``BrokerStore`` API itself live in
``test_029_broker_store.py``; here we only exercise the cooperation of
the two layers.
"""
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.core.broker.exceptions import OrderDispositionUnknownError
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.core.broker.models import (
    DispatchEnvelope,
    ExchangeCapabilities,
    ExchangeOrder,
    ExchangePosition,
    OrderStatus,
    OrderType,
)
from pynecore.lib.strategy import Order, _order_type_entry
from pynecore.types.strategy import ADOPTED_STARTUP_EXTRA_KEY


SYMBOL = "BTCUSDT"
BAR_TS = 1_700_000_000_000
PLUGIN = "Mock"


@pytest.fixture(autouse=True)
def _stub_script():
    prev = lib._script
    lib._script = SimpleNamespace(initial_capital=1_000_000.0)
    try:
        yield
    finally:
        lib._script = prev


@dataclass
class _MockBroker:
    """Duck-typed broker for integration scenarios."""
    client_order_id_max_len = 30  # BrokerPlugin contract attribute
    on_unexpected_cancel = "stop"  # BrokerPlugin contract attribute
    entry_calls: list[DispatchEnvelope] = field(default_factory=list)
    modify_entry_calls: list[tuple[DispatchEnvelope, DispatchEnvelope]] = field(
        default_factory=list,
    )
    cancel_calls: list[DispatchEnvelope] = field(default_factory=list)
    open_orders: list[ExchangeOrder] = field(default_factory=list)
    raise_on_next_entry: Exception | None = None
    raise_on_next_modify_entry: Exception | None = None
    capabilities: ExchangeCapabilities = field(default_factory=ExchangeCapabilities)
    _next_id: int = 0

    def get_capabilities(self) -> ExchangeCapabilities:
        return self.capabilities

    def _mk_order(self, envelope: DispatchEnvelope) -> ExchangeOrder:
        self._next_id += 1
        intent = envelope.intent
        return ExchangeOrder(
            id=f"xchg-{self._next_id}",
            symbol=getattr(intent, 'symbol', SYMBOL),
            side=getattr(intent, 'side', 'buy'),
            order_type=OrderType.LIMIT,
            qty=getattr(intent, 'qty', 0.0),
            filled_qty=0.0,
            remaining_qty=getattr(intent, 'qty', 0.0),
            price=None, stop_price=None, average_fill_price=None,
            status=OrderStatus.OPEN,
            timestamp=0.0, fee=0.0, fee_currency="",
            client_order_id=envelope.client_order_id('e'),
        )

    async def execute_entry(self, envelope):
        self.entry_calls.append(envelope)
        if self.raise_on_next_entry is not None:
            err = self.raise_on_next_entry
            self.raise_on_next_entry = None
            raise err
        return [self._mk_order(envelope)]

    async def execute_exit(self, envelope):  # pragma: no cover — unused
        return [self._mk_order(envelope)]

    async def execute_close(self, envelope):  # pragma: no cover — unused
        return self._mk_order(envelope)

    async def execute_cancel(self, envelope):
        self.cancel_calls.append(envelope)
        return True

    async def modify_entry(self, old, new):
        self.modify_entry_calls.append((old, new))
        if self.raise_on_next_modify_entry is not None:
            err = self.raise_on_next_modify_entry
            self.raise_on_next_modify_entry = None
            raise err
        return [self._mk_order(new)]

    async def modify_exit(self, old, new):  # pragma: no cover — unused
        return [self._mk_order(new)]

    async def get_open_orders(self, symbol=None):
        return list(self.open_orders)

    async def get_position(self, symbol):  # pragma: no cover — unused
        return None

    def watch_orders(self):  # pragma: no cover — unused
        raise NotImplementedError


def _open_ctx(store: BrokerStore) -> RunContext:
    identity = RunIdentity(
        strategy_id="integration", symbol=SYMBOL, timeframe="60",
        account_id="test-account", label=None,
    )
    return store.open_run(identity, script_source="// integration test")


def _mk_engine(
        broker: _MockBroker, ctx: RunContext | None,
) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    run_tag = ctx.run_tag if ctx is not None else "test"
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=run_tag,
        mintick=1.0,
        store_ctx=ctx,
    )
    return engine, pos


# === Core restart round-trip ==============================================


def __test_restart_regenerates_same_client_order_id_for_live_intent__(
        tmp_path: Path,
) -> None:
    """Restart → the first dispatch produces the same CO-ID.

    Without persistence the post-restart engine would anchor on a fresh
    ``bar_ts_ms`` and emit a brand-new ``client_order_id`` — the broker
    would treat it as a new order, not a duplicate. The BrokerStore
    reads the previous anchor back, so the retry's CO-ID stays
    unchanged.
    """
    db = tmp_path / "broker.sqlite"

    broker_a = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        engine_a, pos_a = _mk_engine(broker_a, ctx_a)
        pos_a.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_a.sync(BAR_TS)
        coid_first = broker_a.entry_calls[0].client_order_id('e')
        ctx_a.close()  # happy-path run teardown

    # Process-restart simulation: new BrokerStore + new run on the same
    # identity. The previous run is already closed — the collision
    # check lets it through.
    broker_b = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        engine_b, pos_b = _mk_engine(broker_b, ctx_b)
        pos_b.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        # Delayed "new" bar — without replay the CO-ID would differ.
        engine_b.sync(BAR_TS + 60_000)
        coid_second = broker_b.entry_calls[0].client_order_id('e')

    assert coid_first == coid_second, (
        "post-restart dispatch must use the persisted bar_ts_ms anchor "
        "so the CO-ID stays stable"
    )


def __test_restart_completed_intent_does_not_replay__(tmp_path: Path) -> None:
    """An intent closed before the restart does not resurrect.

    ``record_complete`` deletes the ``envelopes`` row, so a fresh
    engine that no longer sees the Pine-side order does not anchor on
    a stale ``bar_ts_ms``.
    """
    db = tmp_path / "broker.sqlite"

    broker_a = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        engine_a, pos_a = _mk_engine(broker_a, ctx_a)
        pos_a.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_a.sync(BAR_TS)
        # Pine cancel: the diff engine emits a cancel + complete pair.
        pos_a.entry_orders.clear()
        engine_a.sync(BAR_TS + 1)
        assert len(broker_a.cancel_calls) == 1
        ctx_a.close()

    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        envelopes, pending = ctx_b.replay()
        assert envelopes == {}
        assert pending == {}


def __test_restart_recovers_parked_dispatch_via_get_open_orders__(
        tmp_path: Path,
) -> None:
    """A dispatch parked before restart is re-registered without re-dispatch.

    A dispatch parked before restart is re-registered on the next
    sync via the open-orders view, without a re-dispatch.
    """
    db = tmp_path / "broker.sqlite"

    broker_a = _MockBroker()
    # The CO-ID is deterministic — the engine produces the same one on
    # both the A and B sides as long as run_tag is identical (which it
    # is, thanks to the persisted identity).
    expected_envelope = DispatchEnvelope(
        intent=SimpleNamespace(pine_id="L"),  # type: ignore[arg-type]
        run_tag="",  # placeholder for CO-ID derivation; real tag comes from ctx
        bar_ts_ms=BAR_TS,
        retry_seq=0,
    )
    # Compute the expected CO-ID from the A-side tag after the run opens.

    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        expected_envelope = DispatchEnvelope(
            intent=SimpleNamespace(pine_id="L"),  # type: ignore[arg-type]
            run_tag=ctx_a.run_tag,
            bar_ts_ms=BAR_TS,
            retry_seq=0,
        )
        expected_coid = expected_envelope.client_order_id('e')

        broker_a.raise_on_next_entry = OrderDispositionUnknownError(
            "simulated network timeout",
            client_order_id=expected_coid,
            cause=TimeoutError("simulated network timeout"),
        )

        engine_a, pos_a = _mk_engine(broker_a, ctx_a)
        pos_a.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_a.sync(BAR_TS)
        assert expected_coid in engine_a.pending_verification, "park did not happen"
        ctx_a.close()

    # Restart. On the broker side the order is present with the pending CO-ID.
    broker_b = _MockBroker()
    broker_b.open_orders = [
        ExchangeOrder(
            id="xchg-from-restart",
            symbol=SYMBOL,
            side="buy",
            order_type=OrderType.LIMIT,
            qty=1.0,
            filled_qty=0.0,
            remaining_qty=1.0,
            price=50_000.0,
            stop_price=None,
            average_fill_price=None,
            status=OrderStatus.OPEN,
            timestamp=0.0,
            fee=0.0,
            fee_currency="",
            client_order_id=expected_coid,
        ),
    ]
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        engine_b, pos_b = _mk_engine(broker_b, ctx_b)
        pos_b.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_b.sync(BAR_TS + 60_000)

        # No re-dispatch occurred: the engine adopted the pending dispatch.
        assert len(broker_b.entry_calls) == 0, (
            "post-restart engine must not dispatch a parked entry "
            "while the broker already has it"
        )
        assert "xchg-from-restart" in engine_b.order_mapping["L"]

        # The persisted park is consumed; the envelope is still alive
        # (the entry has not been filled yet).
        envelopes, pending = ctx_b.replay()
        assert pending == {}
        assert "L" in envelopes


# === Plugin-resolved parked dispatches =====================================


def _park_entry(broker: _MockBroker, ctx: RunContext) -> tuple[
        OrderSyncEngine, BrokerPosition, str]:
    """Park a single entry and return (engine, position, parked_coid).

    Helper for the resolution tests below — re-uses the same primitives
    as :func:`__test_unknown_disposition_parks_pending__` from the unit
    suite so the parked state precisely matches what production would
    produce.
    """
    expected = DispatchEnvelope(
        intent=SimpleNamespace(pine_id="L"),  # type: ignore[arg-type]
        run_tag=ctx.run_tag, bar_ts_ms=BAR_TS, retry_seq=0,
    )
    expected_coid = expected.client_order_id('e')
    broker.raise_on_next_entry = OrderDispositionUnknownError(
        "simulated timeout", client_order_id=expected_coid,
    )
    engine, pos = _mk_engine(broker, ctx)
    pos.entry_orders["L"] = Order(
        "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
    )
    engine.sync(BAR_TS)
    assert expected_coid in engine.pending_verification
    return engine, pos, expected_coid


def __test_plugin_resolution_attached_clears_park_keeps_intent__(
        tmp_path: Path,
) -> None:
    """An ``'attached'`` resolution clears the parked envelope but keeps the intent.

    A ``'attached'`` resolution removes the in-memory parked envelope
    but keeps the active intent — the dispatch landed, no re-dispatch
    needed.

    Mirrors the Capital.com bracket recovery: the position-snapshot loop
    confirms the bracket is on the parent and writes ``'attached'``; the
    engine consumes it on the next sync and stops watching the COID via
    ``get_open_orders`` (which would never echo it).

    Contract: the persisted ``pending_verifications`` row stays alive
    after consume so a late ``'rejected'`` write (per-leg bracket
    resolver, slow plugin re-evaluation) can still flip the resolution
    via the sticky-rejected SQL in
    :meth:`RunContext.record_resolution`. The row is reaped on intent
    retirement (cancel / fill / rejected) — see
    :func:`__test_plugin_resolution_late_rejected_after_attached_flips__`.
    """
    db = tmp_path / "broker.sqlite"
    broker = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine, _pos, coid = _park_entry(broker, ctx)

        ctx.record_resolution(coid, 'attached')
        # Second sync drains the resolution before the get_open_orders poll.
        engine.sync(BAR_TS + 60_000)

        assert coid not in engine.pending_verification, (
            "attached resolution must clear the in-memory pending entry"
        )
        assert "L" in engine.active_intents, (
            "attached: the intent stays live — the dispatch landed"
        )
        # No re-dispatch: still exactly the original (timed-out) call.
        assert len(broker.entry_calls) == 1
        # Persisted side intentionally NOT cleared on attached: a late
        # 'rejected' must still be observable via the sticky SQL. The
        # row is deleted on retirement (cancel / fill / rejected flip).
        _envelopes, pending = ctx.replay()
        assert coid in pending and pending[coid].resolution == 'attached', (
            "attached row must stay alive after consume so a late "
            "'rejected' write can still flip it; the row is reaped on "
            "intent retirement"
        )
        ctx.close()


def __test_plugin_resolution_rejected_clears_intent_for_redispatch__(
        tmp_path: Path,
) -> None:
    """A ``'rejected'`` resolution clears the park and intent so the next sync re-dispatches.

    A ``'rejected'`` resolution clears both the parked envelope and the
    active intent so the next sync re-dispatches the original Pine intent
    against the broker.

    For a Capital.com bracket whose attach PUT timed out and a snapshot
    later proved no level was set, this is what restores protection.
    """
    db = tmp_path / "broker.sqlite"
    broker = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine, pos, coid = _park_entry(broker, ctx)
        # Pine still wants the entry — keep it in pos.entry_orders so the
        # next sync builds the same intent again.
        assert "L" in pos.entry_orders

        ctx.record_resolution(coid, 'rejected')
        engine.sync(BAR_TS + 60_000)

        assert coid not in engine.pending_verification
        # Pine intent re-dispatched: a fresh execute_entry call landed.
        assert len(broker.entry_calls) == 2, (
            "rejected resolution must re-dispatch on the next sync"
        )
        # And the new dispatch successfully populated _order_mapping.
        assert "L" in engine.order_mapping
        ctx.close()


def __test_plugin_resolution_attached_post_restart_does_not_redispatch__(
        tmp_path: Path,
) -> None:
    """Cross-restart attached: the resolution consumer adopts instead of re-dispatching.

    Cross-restart attached: ``_active_intents`` is empty after the
    process bounce, so without an adoption marker
    :meth:`_diff_and_dispatch` would re-issue the bracket on the first
    post-restart sync — duplicating a position-attached protective leg
    that ``get_open_orders`` will never echo back. The resolution
    consumer must populate ``_order_mapping`` so the same adoption path
    that handles ``get_open_orders`` recoveries also covers this one.
    """
    db = tmp_path / "broker.sqlite"

    # Phase A: park the entry, then close the run.
    broker_a = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        _, _, coid_a = _park_entry(broker_a, ctx_a)
        ctx_a.record_resolution(coid_a, 'attached')
        ctx_a.close()

    # Phase B: restart. Pine still wants the same entry. The persisted
    # 'attached' resolution must be consumed and the intent adopted —
    # NO re-dispatch.
    broker_b = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        engine_b, pos_b = _mk_engine(broker_b, ctx_b)
        pos_b.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_b.sync(BAR_TS + 60_000)

        assert len(broker_b.entry_calls) == 0, (
            "attached resolution must adopt across restart, not "
            "re-dispatch the live order"
        )
        assert "L" in engine_b.active_intents, (
            "intent must be marked live so subsequent diffs treat it "
            "as already-dispatched"
        )
        # Persisted park row stays alive (late-rejected stickiness);
        # reaped on intent retirement, not on attached consume.
        _envelopes, pending = ctx_b.replay()
        assert coid_a in pending, (
            "attached row must survive the restart-and-adopt path so a "
            "late 'rejected' write remains observable"
        )
        ctx_b.close()


def __test_plugin_resolution_late_rejected_after_attached_flips__(
        tmp_path: Path,
) -> None:
    """An ``'attached'`` consume keeps the persisted row so a late ``'rejected'`` can flip it.

    An ``'attached'`` consume must NOT delete the persisted row —
    otherwise a late ``'rejected'`` write (per-leg bracket resolver,
    slow plugin re-evaluation) finds zero rows and the engine never
    learns the leg is missing.

    Scenario: a TP/SL bracket times out → entry coid parked. Plugin's
    poll resolves the TP leg first as ``'attached'`` and commits;
    engine syncs, consumes attached, places adoption marker, intent
    is adopted (the bracket is still wanted by Pine). Plugin's next
    pass discovers SL is actually missing (level mismatch) and writes
    ``'rejected'`` to the SAME row. The sticky-rejected SQL must find
    the row alive and flip it. Engine's next sync consumes the
    rejected, drops the active intent, re-dispatches — the protective
    leg goes out again.

    Without the fix the engine kept the now-incomplete bracket as
    ``active`` forever (TP attached, SL silently missing), exposing
    the position.
    """
    db = tmp_path / "broker.sqlite"
    broker = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine, _pos, coid = _park_entry(broker, ctx)

        # Step 1: attached arrives first.
        ctx.record_resolution(coid, 'attached')
        engine.sync(BAR_TS + 60_000)
        assert "L" in engine.active_intents, (
            "attached must keep the intent live (Pine still wants it)"
        )
        assert len(broker.entry_calls) == 1, (
            "attached must not re-dispatch"
        )

        # Step 2: a late rejected write hits the SAME row. The sticky
        # SQL must find it alive and flip it.
        ctx.record_resolution(coid, 'rejected')
        _envelopes, pending = ctx.replay()
        assert pending[coid].resolution == 'rejected', (
            "sticky-rejected SQL must find the row alive after the "
            "earlier attached consume; if the engine had deleted it "
            "the UPDATE would have hit zero rows and we would still "
            "see 'attached' here"
        )

        # Step 3: engine consumes the rejected → drops intent, calls
        # _drop_envelope (which DELETEs the row + envelope). The next
        # _diff_and_dispatch re-dispatches the same Pine intent.
        engine.sync(BAR_TS + 120_000)
        assert len(broker.entry_calls) == 2, (
            "rejected must trigger re-dispatch — the protective leg "
            "must go out again, not stay silently missing"
        )
        retry_coid = broker.entry_calls[1].client_order_id('e')
        assert retry_coid != coid, (
            "retry must use a fresh client_order_id"
        )
        # Persisted state cleaned up by _drop_envelope.
        envelopes_after, pending_after = ctx.replay()
        # The row for the OLD parked coid is gone (record_complete
        # delete-by-key reaped it). A new envelope was just persisted
        # for the retry dispatch — that one stays until its own retire.
        assert coid not in pending_after
        ctx.close()


def __test_plugin_resolution_modify_rejected_preserves_active_intent__(
        tmp_path: Path,
) -> None:
    """A parked-modify ``'rejected'`` preserves ``_active_intents`` / ``_order_mapping``.

    A parked **modify** dispatch whose plugin resolution arrives as
    ``'rejected'`` must NOT clear ``_active_intents`` /
    ``_order_mapping`` for the same key. The ORIGINAL exchange order
    is still alive on the broker — only the amend failed to apply.
    The buggy ``rejected`` branch (which treated every parked-rejected
    case identically to a new-dispatch rejected) made
    :meth:`_diff_and_dispatch` see the Pine intent as brand new and
    re-dispatch it via ``execute_*``, placing a SECOND order alongside
    the still-live original.

    This test bypasses the normal modify-call path and parks directly
    via :meth:`OrderSyncEngine._park_pending` with ``kind='modify'``
    so we can audit the contract independent of the broker's modify
    plumbing: after the rejected resolution is consumed, no new
    ``execute_entry`` call may fire and the original intent / mapping
    must survive.
    """
    db = tmp_path / "broker.sqlite"
    broker = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        # Phase A: establish a normally-dispatched live order at "L".
        engine, pos = _mk_engine(broker, ctx)
        pos.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine.sync(BAR_TS)
        assert "L" in engine.active_intents, "phase A: intent must be live"
        assert "L" in engine.order_mapping, (
            "phase A: order_mapping must reflect the dispatched order"
        )
        original_mapping = list(engine.order_mapping["L"])
        original_intent = engine.active_intents["L"]
        assert len(broker.entry_calls) == 1

        # Phase B: simulate a modify-park (an amend that timed out).
        # Build a dispatch envelope at the same key with a different
        # retry_seq so the COID is fresh — this models what
        # ``_dispatch_modify`` would persist on its own
        # ``OrderDispositionUnknownError`` catch.
        modify_envelope = DispatchEnvelope(
            intent=SimpleNamespace(  # type: ignore[arg-type]
                pine_id="L", intent_key="L",
            ),
            run_tag=ctx.run_tag, bar_ts_ms=BAR_TS, retry_seq=1,
        )
        modify_coid = modify_envelope.client_order_id('e')
        modify_err = OrderDispositionUnknownError(
            "simulated modify timeout", client_order_id=modify_coid,
        )
        engine._park_pending(modify_envelope, modify_err, kind='modify')

        # Verify the persisted park row carries the modify kind so a
        # subsequent restart would also see it correctly.
        _envelopes, pending = ctx.replay()
        assert pending[modify_coid].dispatch_kind == 'modify'

        # Phase C: plugin polls and decides the amend never applied.
        ctx.record_resolution(modify_coid, 'rejected')
        entry_calls_before = len(broker.entry_calls)

        engine.sync(BAR_TS + 60_000)

        assert len(broker.entry_calls) == entry_calls_before, (
            f"modify-rejected must NOT trigger execute_entry — the "
            f"original order is still live on the exchange and would "
            f"be duplicated. expected {entry_calls_before} entry "
            f"calls, got {len(broker.entry_calls)}"
        )
        assert "L" in engine.active_intents, (
            "modify-rejected must preserve _active_intents — "
            "_diff_and_dispatch needs it to recognise the Pine intent "
            "as already-live and re-emit a modify (not a new dispatch)"
        )
        assert engine.active_intents["L"] is original_intent, (
            "the surviving _active_intents['L'] must be the ORIGINAL "
            "intent the live order was built from, not the rejected "
            "modify's intent"
        )
        assert engine.order_mapping.get("L") == original_mapping, (
            f"_order_mapping['L'] must still point at the original "
            f"exchange order id. expected {original_mapping}, got "
            f"{engine.order_mapping.get('L')!r}"
        )
        assert modify_coid not in engine.pending_verification
        # The persisted park row is reaped by _drop_envelope (delete-
        # by-key); the original order's envelope is rebuilt fresh on
        # the next dispatch via ``_persisted_envelope_anchors`` reset.
        _envelopes_after, pending_after = ctx.replay()
        assert modify_coid not in pending_after
        ctx.close()


def __test_plugin_resolution_modify_rejected_restores_pre_modify_active_intent__(
        tmp_path: Path,
) -> None:
    """A real-path modify-rejected restores the pre-modify ``_active_intents`` and re-emits.

    Going through the real ``_dispatch_modify`` path: a timed-out
    amend lands in ``_park_pending`` AND ``_diff_and_dispatch`` then
    promotes ``_active_intents[key]`` to the NEW intent. If the plugin
    later resolves the parked modify as ``'rejected'``, simply preserving
    the slot leaves it set to that NEW intent — the next sync sees Pine
    == active and the diff stays silent, leaving the original exchange
    order indefinitely on the OLD parameters even though Pine still
    wants the amend.

    Contract enforced here: the rejected branch restores
    ``_active_intents[key]`` from the pre-modify snapshot stashed by
    ``_park_pending(..., old_intent=old)`` so the very next diff
    re-emits ``modify_entry`` (Pine != active again) and the broker
    gets a fresh chance to apply the amendment.

    Phase A: dispatch the original entry; broker accepts.
    Phase B: Pine flips a parameter (different limit) and the broker's
             ``modify_entry`` raises ``OrderDispositionUnknownError``.
             Engine parks ``kind='modify'``; the diff promotes
             ``_active_intents['L']`` to the NEW intent.
    Phase C: plugin records ``'rejected'`` for the parked COID. Next
             sync's ``_consume_plugin_resolutions`` must restore
             ``_active_intents['L']`` to the ORIGINAL intent and keep
             ``_order_mapping['L']`` pointing at the live order id.
    Phase D: Pine still emits the NEW intent. The diff observes the
             restored OLD active vs. NEW Pine and re-emits
             ``modify_entry`` — broker.modify_entry_calls grows by 1.
    """
    db = tmp_path / "broker.sqlite"
    broker = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine, pos = _mk_engine(broker, ctx)
        # Phase A: dispatch the original entry.
        pos.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine.sync(BAR_TS)
        assert "L" in engine.active_intents, "phase A: original intent live"
        original_intent = engine.active_intents["L"]
        original_mapping = list(engine.order_mapping["L"])
        assert len(broker.entry_calls) == 1
        assert len(broker.modify_entry_calls) == 0

        # Phase B: Pine flips a parameter — same intent_key 'L', new
        # limit. The broker's modify_entry raises
        # ``OrderDispositionUnknownError`` so ``_dispatch_modify`` parks
        # the dispatch with kind='modify' AND
        # ``_diff_and_dispatch`` then promotes
        # ``_active_intents['L']`` to the new intent (line 1328).
        pos.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=51_000.0,
        )
        # Build the NEW envelope's COID the same way ``_dispatch_modify``
        # would so we can target the resolution write at the right row.
        # ``_build_envelope`` reuses the original anchor (same retry_seq)
        # — modifies do NOT bump retry_seq — so the COID derives from
        # the unchanged ``bar_ts_ms`` / ``retry_seq`` pair.
        modify_coid = next(iter(engine._envelopes.values())).client_order_id('e')
        broker.raise_on_next_modify_entry = OrderDispositionUnknownError(
            "simulated modify timeout", client_order_id=modify_coid,
        )
        engine.sync(BAR_TS + 60_000)

        assert len(broker.modify_entry_calls) == 1, (
            "phase B: broker.modify_entry must have been called once "
            "for the parameter flip"
        )
        # The promoted-new state — what makes the rejected branch
        # tricky: without restoration the next diff would see
        # Pine == active and silently drop the amend.
        promoted_new_intent = engine.active_intents["L"]
        assert promoted_new_intent is not original_intent, (
            "phase B: _diff_and_dispatch must have promoted "
            "_active_intents['L'] to the NEW intent after the parked "
            "modify returned. Without the promotion this test is moot."
        )
        assert modify_coid in engine.pending_verification, (
            f"phase B: park must have stashed the modify; got pending "
            f"keys={list(engine.pending_verification)!r}"
        )
        _envelopes, pending = ctx.replay()
        assert pending[modify_coid].dispatch_kind == 'modify'

        # Phase C: plugin polls and decides the amend never applied.
        # The very next sync runs ``_consume_plugin_resolutions`` →
        # restores ``_active_intents['L']`` to the original intent →
        # ``_diff_and_dispatch`` sees Pine (limit=51_000) != restored
        # active (limit=50_000) → fires a fresh ``modify_entry`` call
        # in the SAME sync. The contract under test: that fresh
        # ``modify_entry`` call must happen; without the restoration
        # ``_active_intents['L']`` would still be the NEW intent that
        # _diff_and_dispatch promoted, Pine == active, no diff, no
        # retry, original exchange order indefinitely on OLD params.
        #
        # We arrange the second modify_entry to ALSO timeout-park, so
        # the test can also inspect the post-restoration state on the
        # rollback snapshot (the new park stashes the just-restored
        # original as its OWN snapshot, ready for another rejection
        # cycle).
        second_modify_coid_holder: dict[str, str] = {}
        original_dispatch_modify = engine._dispatch_modify

        def _capturing_dispatch_modify(old, new):
            # Capture the COID the *next* park will use (the engine
            # rebuilds the envelope inside _dispatch_modify); this
            # lets the test target the second resolution write
            # accurately. Wrapper has to mirror the body's exception
            # handling — replicating it is brittle, so just capture
            # then delegate.
            new_env = engine._build_envelope(new)
            second_modify_coid_holder['coid'] = new_env.client_order_id('e')
            original_dispatch_modify(old, new)

        engine._dispatch_modify = _capturing_dispatch_modify  # type: ignore[method-assign]
        ctx.record_resolution(modify_coid, 'rejected')
        broker.raise_on_next_modify_entry = OrderDispositionUnknownError(
            "simulated second modify timeout",
            client_order_id="placeholder",  # overwritten right before raise
        )

        # The mock raises with whatever client_order_id is set above,
        # but the engine reads ``error.client_order_id`` to identify
        # which row to park. Patch the field at raise time via a
        # stub that derives the COID from the envelope under
        # dispatch.
        async def _raising_modify_entry(old_env, new_env):
            broker.modify_entry_calls.append((old_env, new_env))
            err = broker.raise_on_next_modify_entry
            broker.raise_on_next_modify_entry = None
            assert err is not None
            err.client_order_id = new_env.client_order_id('e')
            raise err

        # Replace just for this one second-modify call; the original
        # implementation is already what we want for the third
        # dispatch (which will not be exercised here).
        broker.modify_entry = _raising_modify_entry  # type: ignore[method-assign]

        engine.sync(BAR_TS + 120_000)

        assert len(broker.entry_calls) == 1, (
            f"modify-rejected must NOT trigger execute_entry (the "
            f"original is still alive on the exchange and would be "
            f"duplicated). got entry_calls={len(broker.entry_calls)}"
        )
        # The fix's core public-contract assertion: a fresh modify
        # dispatch happened in the same sync that processed the
        # rejection. Without the restoration, the diff would have
        # observed Pine == active (both NEW) and emitted nothing.
        assert len(broker.modify_entry_calls) == 2, (
            f"modify-rejected must let the very next "
            f"_diff_and_dispatch see a restored OLD active vs. Pine "
            f"NEW delta and re-emit modify_entry. Without the "
            f"restoration, _active_intents['L'] would still be the "
            f"NEW intent that _diff_and_dispatch promoted, Pine "
            f"would still want NEW, the diff would observe no delta, "
            f"and the original exchange order would stay on OLD "
            f"parameters indefinitely. got "
            f"modify_entry_calls={len(broker.modify_entry_calls)}"
        )
        # The second modify also parked → ``_active_intents['L']``
        # is once again the NEW intent (promoted right after the
        # parked second _dispatch_modify returned), and a fresh
        # rollback snapshot is in place for any future rejected
        # resolution cycle.
        assert engine.active_intents["L"].limit == 51_000.0, (
            f"phase C post-retry-park: _active_intents['L'] is "
            f"promoted to NEW again (the retry was dispatched). "
            f"got limit={engine.active_intents['L'].limit!r}"
        )
        assert "L" in engine._modify_old_intents, (
            "phase C post-retry-park: a fresh rollback snapshot must "
            "be stashed for the second parked modify"
        )
        assert engine._modify_old_intents["L"].limit == 50_000.0, (
            f"phase C post-retry-park: the new rollback snapshot "
            f"must mirror the just-restored OLD intent (limit=50_000), "
            f"NOT the promoted NEW (limit=51_000). got "
            f"{engine._modify_old_intents['L'].limit!r}"
        )
        assert engine.order_mapping.get("L") == original_mapping, (
            f"modify-rejected must keep _order_mapping['L'] pointing "
            f"at the original live order id; got "
            f"{engine.order_mapping.get('L')!r}"
        )
        assert modify_coid not in engine.pending_verification, (
            f"first parked modify's pending row must have been "
            f"reaped on rejection consume; got "
            f"pending={list(engine.pending_verification)!r}"
        )
        ctx.close()


def __test_plugin_resolution_modify_rejected_then_pine_cancels_does_not_resurrect_intent__(
        tmp_path: Path,
) -> None:
    """A late modify-rejected after a Pine cancel must not resurrect the cancelled key.

    Cleanup invariant: a parked modify whose resolution is still
    pending when Pine drops the intent (cancel) must NOT resurrect the
    cancelled key into ``_active_intents`` once the late ``'rejected'``
    arrives. The rollback snapshot is keyed by ``intent_key``; without
    cleanup at cancel time, the rejected branch would see the snapshot
    and re-install ``_active_intents[key]`` to a value the strategy no
    longer wants.

    Phase A: dispatch original entry.
    Phase B: parameter flip → modify timeout → park (snapshot stashed).
    Phase C: Pine drops the intent (entry_orders no longer has 'L');
             ``_diff_and_dispatch`` cancels via ``_dispatch_cancel``.
             The cancel branch must drop the rollback snapshot.
    Phase D: late ``'rejected'`` arrives. Must NOT restore
             ``_active_intents['L']`` (the strategy moved on).
    """
    db = tmp_path / "broker.sqlite"
    broker = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine, pos = _mk_engine(broker, ctx)
        # Phase A
        pos.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine.sync(BAR_TS)
        # Phase B
        pos.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=51_000.0,
        )
        modify_coid = next(iter(engine._envelopes.values())).client_order_id('e')
        broker.raise_on_next_modify_entry = OrderDispositionUnknownError(
            "simulated modify timeout", client_order_id=modify_coid,
        )
        engine.sync(BAR_TS + 60_000)
        assert "L" in engine._modify_old_intents, (
            "phase B: rollback snapshot must have been stashed on park"
        )
        # Phase C: Pine drops the intent.
        pos.entry_orders.pop("L", None)
        engine.sync(BAR_TS + 120_000)
        assert "L" not in engine.active_intents, (
            "phase C: cancel must have dropped _active_intents['L']"
        )
        assert "L" not in engine._modify_old_intents, (
            "phase C: cancel must have dropped the rollback snapshot — "
            "without this, a late rejected resolution would resurrect "
            "the cancelled key. got "
            f"_modify_old_intents={engine._modify_old_intents!r}"
        )
        # Phase D: late rejected arrives.
        ctx.record_resolution(modify_coid, 'rejected')
        engine.sync(BAR_TS + 180_000)
        assert "L" not in engine.active_intents, (
            "phase D: late rejected must NOT resurrect _active_intents "
            f"for the cancelled key. got "
            f"active_intents={engine.active_intents!r}"
        )
        ctx.close()


def __test_plugin_resolution_repark_same_coid_clears_attached_dedup__(
        tmp_path: Path,
) -> None:
    """Re-parking a COID clears the ``_consumed_attached_coids`` dedup so re-attach isn't skipped.

    If a COID was already consumed as ``'attached'`` and the same
    ``client_order_id`` is parked again in the same engine instance
    (legitimate modify/retry timeout), a later ``'attached'``
    resolution write must NOT be skipped by the in-memory
    ``_consumed_attached_coids`` dedup. Otherwise the fresh
    ``_pending_verification[coid]`` entry stays stuck indefinitely
    for brokers whose orders never appear in ``get_open_orders``
    (e.g. Capital.com position-attached brackets).

    The fix: :meth:`OrderSyncEngine._park_pending` discards the
    coid from ``_consumed_attached_coids`` whenever it parks, so
    every fresh park starts with a clean dedup slate.
    """
    db = tmp_path / "broker.sqlite"
    broker = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine, _pos, coid = _park_entry(broker, ctx)

        # Plugin records 'attached'; engine consumes it and marks the
        # coid as seen.
        ctx.record_resolution(coid, 'attached')
        engine.sync(BAR_TS + 60_000)
        assert coid in engine._consumed_attached_coids

        # The same coid is parked again (modify/retry timeout that
        # happens to produce the same client_order_id). _park_pending
        # must clear the stale dedup entry.
        repark_envelope = DispatchEnvelope(
            intent=SimpleNamespace(  # type: ignore[arg-type]
                pine_id="L", intent_key="L",
            ),
            run_tag=ctx.run_tag, bar_ts_ms=BAR_TS, retry_seq=0,
        )
        repark_coid = repark_envelope.client_order_id('e')
        assert repark_coid == coid, (
            "test setup invariant: the re-park envelope must produce "
            "the same client_order_id as the original park"
        )
        repark_err = OrderDispositionUnknownError(
            "simulated retry timeout", client_order_id=coid,
        )
        engine._park_pending(repark_envelope, repark_err, kind='modify')

        assert coid not in engine._consumed_attached_coids, (
            "re-park must clear the in-memory attached dedup; without "
            "this the next 'attached' resolution would be silently "
            "skipped"
        )
        assert coid in engine.pending_verification, (
            "re-park must establish a fresh _pending_verification entry"
        )

        # Plugin re-resolves: 'attached' on the now-fresh row.
        ctx.record_resolution(coid, 'attached')
        engine.sync(BAR_TS + 120_000)

        assert coid not in engine.pending_verification, (
            "the second 'attached' consume must clear "
            "_pending_verification — without the dedup reset the row "
            "would stay stuck indefinitely"
        )
        ctx.close()


# === Recovery-confirm self-heal (record_unpark orphan) =====================


def __test_record_unpark_self_heals_orphaned_in_memory_pending__(
        tmp_path: Path,
) -> None:
    """A reconnect handshake self-heals an in-memory park whose ``record_unpark`` dropped its row.

    A plugin recovery-confirm (``record_unpark``) of a MARKET /
    already-filled order deletes only the persisted park row; the engine
    must self-heal its in-memory ``_pending_verification`` entry against
    the replayed set instead of stranding it across reconnects.

    Models the cTrader ``_confirm_recovered_entry`` path: it calls
    ``store_ctx.record_unpark(coid)`` for a fill ``get_open_orders`` never
    re-surfaces, so neither the ``get_open_orders`` match in
    :meth:`OrderSyncEngine._verify_pending_dispatches` nor a
    ``record_resolution`` consume can ever clear the in-memory park. The
    reconnect handshake (:meth:`refresh_anchors_from_store`) re-replays
    the anchors and must drop the now-rowless entry.
    """
    db = tmp_path / "broker.sqlite"
    broker = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine, _pos, coid = _park_entry(broker, ctx)

        # Recovery-confirm out-of-band: only the persisted row is deleted.
        ctx.record_unpark(coid)
        _envelopes, pending = ctx.replay()
        assert coid not in pending, (
            "record_unpark must delete the persisted park row"
        )
        assert coid in engine.pending_verification, (
            "the in-memory park is still stranded until the self-heal runs"
        )

        # In-process reconnect handshake re-replays the anchors.
        engine.refresh_anchors_from_store()

        assert coid not in engine.pending_verification, (
            "self-heal must drop the in-memory park whose persisted row "
            "record_unpark deleted"
        )
        ctx.close()


def __test_sync_self_heals_orphaned_in_memory_pending__(
        tmp_path: Path,
) -> None:
    """The start-of-sync re-replay also self-heals an orphaned in-memory park.

    The start-of-sync wholesale re-replay self-heals an orphaned
    in-memory park too — not only the explicit reconnect handshake.

    A filled MARKET order never echoes in ``get_open_orders``
    (``broker.open_orders`` stays empty), so the
    :meth:`OrderSyncEngine._verify_pending_dispatches` match path cannot
    clear the park; the per-sync reconciliation must.
    """
    db = tmp_path / "broker.sqlite"
    broker = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine, _pos, coid = _park_entry(broker, ctx)

        ctx.record_unpark(coid)
        engine.sync(BAR_TS + 60_000)

        assert coid not in engine.pending_verification, (
            "the per-sync re-replay must drop the orphaned in-memory park"
        )
        _envelopes, pending = ctx.replay()
        assert pending == {}
        ctx.close()


def __test_legitimate_park_survives_reconnect_replay__(
        tmp_path: Path,
) -> None:
    """A still-pending park with an intact persisted row survives the self-heal.

    A still-pending park whose persisted row is intact must NOT be
    pruned by the self-heal — the reconciliation only drops entries the
    persisted set no longer backs.
    """
    db = tmp_path / "broker.sqlite"
    broker = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine, _pos, coid = _park_entry(broker, ctx)

        # No record_unpark: the dispatch is still genuinely pending.
        engine.refresh_anchors_from_store()

        assert coid in engine.pending_verification, (
            "a park with a live persisted row must survive the self-heal"
        )
        _envelopes, pending = ctx.replay()
        assert coid in pending
        ctx.close()


def __test_plugin_resolution_same_key_rejected_dominates_attached__(
        tmp_path: Path,
) -> None:
    """For two same-key rows, ``'rejected'`` dominates ``'attached'`` regardless of order.

    Two rows can share an ``intent_key`` (per-leg bracket resolver,
    or a restart that finds an older 'attached' alongside a newer
    'rejected' row from a retry storm). When a snapshot returns BOTH a
    ``'rejected'`` and an ``'attached'`` row for the same key,
    ``_consume_plugin_resolutions`` must process them as a *group* and
    let rejected dominate — otherwise the order in which the storage
    layer hands the rows back decides correctness, and a stale
    ``'attached'`` processed *after* a ``'rejected'`` for the same key
    would re-install the adoption marker the rejected branch had just
    cleared. The engine would then silently adopt an unverified
    dispatch on the next ``_diff_and_dispatch`` instead of
    re-dispatching the missing protective leg.
    """
    db = tmp_path / "broker.sqlite"
    broker = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine, _pos, coid = _park_entry(broker, ctx)

        # Park a SECOND coid under the same intent_key, simulating a
        # per-leg bracket resolver writing both legs to separate rows.
        sibling_coid = "coid-sibling"
        ctx.record_park(sibling_coid, "L")

        # Write resolutions in the **order that previously triggered
        # the bug**: rejected first, attached second. With the
        # row-by-row consumer the rejected row's _drop_envelope cleared
        # _order_mapping/_active_intents/etc., then the later attached
        # row called _order_mapping.setdefault('L', []) and re-added
        # the adoption marker — at the next _diff_and_dispatch the
        # engine adopted instead of re-dispatching.
        ctx.record_resolution(coid, 'rejected')
        ctx.record_resolution(sibling_coid, 'attached')

        # Sanity: both rows are visible to iter_pending_resolutions.
        snapshot = ctx.iter_pending_resolutions()
        assert {r.coid for r in snapshot} == {coid, sibling_coid}
        assert {r.resolution for r in snapshot} == {'attached', 'rejected'}

        engine.sync(BAR_TS + 60_000)

        # Re-dispatch on the same sync (Pine still wants the entry).
        # An adoption-path bug would have produced exactly one entry
        # call total + an empty-list adoption marker; the rejected-
        # dominates path produces a SECOND real dispatch.
        assert len(broker.entry_calls) == 2, (
            "rejected dominance must trigger re-dispatch, not adoption"
        )
        retry_coid = broker.entry_calls[1].client_order_id('e')
        assert retry_coid != coid, (
            "retry must use a fresh client_order_id (cleared envelope)"
        )
        # The post-sync ``_order_mapping['L']`` is the *new* dispatch's
        # mapping (real exchange ids), not the empty-list adoption
        # marker the buggy ordering would have produced.
        assert engine.order_mapping.get("L"), (
            "_order_mapping['L'] must reflect the re-dispatched order's "
            "exchange ids (non-empty); an empty list here would mean "
            "the stale 'attached' row's adoption marker survived"
        )
        assert "L" not in engine._attached_adoption_keys, (
            "rejected must dominate — _attached_adoption_keys must "
            "not retain the key (would re-install the marker on "
            "future cleanup runs)"
        )
        assert coid not in engine.pending_verification
        assert sibling_coid not in engine.pending_verification

        # Both persisted rows under this key reaped by _drop_envelope's
        # record_complete (DELETE by intent_key).
        _envelopes_after, pending_after = ctx.replay()
        assert coid not in pending_after
        assert sibling_coid not in pending_after
        ctx.close()


def __test_plugin_resolution_attached_no_pine_intent_clears_stale_marker__(
        tmp_path: Path,
) -> None:
    """Cross-restart attached with no Pine intent drops the stale empty adoption marker.

    Cross-restart attached + Pine intent gone: ``_order_mapping`` and
    the persisted envelope anchor must NOT keep an empty adoption marker.

    Scenario: a bracket dispatch was parked, the plugin's snapshot
    confirmed it as attached, then the bot restarted. By the time of
    the first post-restart sync, Pine no longer has an intent at the
    same key — the position has since closed, the strategy moved on,
    or the user removed the order. Without cleanup the empty marker
    placed by ``_consume_plugin_resolutions`` would silently absorb a
    *future* same-key dispatch (Pine ``intent_key``s are deterministic
    from ``pine_id`` — reuse is the norm) and skip the broker call
    entirely. The end-of-sync sweep drops markers no Pine intent
    claimed.
    """
    db = tmp_path / "broker.sqlite"

    # Phase A: park, mark attached, close (simulates a bot crash after
    # the snapshot resolver wrote its result).
    broker_a = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        _, _, coid_a = _park_entry(broker_a, ctx_a)
        ctx_a.record_resolution(coid_a, 'attached')
        ctx_a.close()

    # Phase B: restart with NO Pine intent at "L" (position closed
    # while bot was down, strategy regenerated empty order book).
    broker_b = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        engine_b, _pos_b = _mk_engine(broker_b, ctx_b)
        # No Pine intent registered.
        engine_b.sync(BAR_TS + 60_000)

        assert "L" not in engine_b.order_mapping, (
            "stale empty adoption marker would silently absorb a future "
            "same-key dispatch — cleanup must drop it when no Pine "
            "intent claimed it this sync"
        )
        assert "L" not in engine_b._persisted_envelope_anchors, (
            "envelope anchor must also be dropped — keeping it would "
            "pin a future intent at this key to the previously-attached "
            "client_order_id and the broker may dedupe the retry"
        )
        # Park row still cleared — the resolution itself was consumed.
        _envelopes, pending = ctx_b.replay()
        assert pending == {}

        # Phase C: a *later* Pine intent at the same key must reach the
        # broker as a fresh dispatch — not be silently absorbed by the
        # (now-cleared) marker.
        pos_b2 = engine_b._position
        pos_b2.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_b.sync(BAR_TS + 120_000)
        assert len(broker_b.entry_calls) == 1, (
            "after cleanup, a fresh same-key intent must dispatch to the "
            "broker — the stale marker is gone"
        )
        ctx_b.close()


def __test_attached_no_pine_intent_cleanup_deletes_persisted_envelope_row__(
        tmp_path: Path,
) -> None:
    """The cleanup-no-intent path also DELETEs the persisted ``envelopes`` row.

    The cleanup-no-intent path must also DELETE the SQLite
    ``envelopes`` row; otherwise a *future* process restart replays
    the stale anchor through :meth:`_build_envelope` and reuses the
    old ``client_order_id`` for a genuinely fresh order.

    Scenario: park, attached, restart-without-intent + cleanup
    (covered by the previous test). Then a SECOND restart with a
    fresh Pine intent at the same key — the retry must build a fresh
    envelope, not pick up the stale anchor.
    """
    db = tmp_path / "broker.sqlite"

    # Phase A: park, mark attached, close.
    broker_a = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        _, _, parked_coid = _park_entry(broker_a, ctx_a)
        ctx_a.record_resolution(parked_coid, 'attached')
        ctx_a.close()

    # Phase B: restart with no Pine intent → cleanup-no-intent fires.
    # The cleanup must DELETE the SQLite envelope row, not just pop
    # the in-memory anchor.
    broker_b = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        engine_b, _pos_b = _mk_engine(broker_b, ctx_b)
        engine_b.sync(BAR_TS + 60_000)

        envelopes_after_cleanup, pending_after_cleanup = ctx_b.replay()
        assert envelopes_after_cleanup == {}, (
            "cleanup-no-intent must DELETE the persisted envelopes "
            "row — keeping it would let a *future* restart replay the "
            "stale anchor and reuse the old client_order_id for a "
            "genuinely fresh order"
        )
        assert pending_after_cleanup == {}, (
            "cleanup must also delete the persisted pending row "
            "(record_complete deletes by intent_key)"
        )
        ctx_b.close()

    # Phase C: a second restart with a fresh Pine intent at the same
    # key. The retry must dispatch with a fresh client_order_id —
    # if the cleanup left the stale envelope row alive, this would
    # silently retry under the parked coid.
    broker_c = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_c:
        ctx_c = _open_ctx(store_c)
        engine_c, pos_c = _mk_engine(broker_c, ctx_c)
        pos_c.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_c.sync(BAR_TS + 120_000)

        assert len(broker_c.entry_calls) == 1, (
            "fresh Pine intent at the cleaned-up key must dispatch"
        )
        retry_coid = broker_c.entry_calls[0].client_order_id('e')
        assert retry_coid != parked_coid, (
            f"retry must build a FRESH client_order_id; got the same "
            f"coid as the originally parked dispatch ({parked_coid!r}) "
            f"— the stale envelopes row was not deleted in Phase B"
        )
        ctx_c.close()


def __test_plugin_resolution_rejected_post_restart_uses_fresh_envelope__(
        tmp_path: Path,
) -> None:
    """Cross-restart rejected discards the replayed anchor so the retry gets a fresh COID.

    Cross-restart rejected: the replayed envelope anchor must be
    discarded so :meth:`_build_envelope` allocates a fresh
    ``client_order_id`` for the retry. Reusing the rejected dispatch's
    COID would let any broker that retains idempotency state for failed
    submissions dedupe the retry instead of accepting it as a new order.
    """
    db = tmp_path / "broker.sqlite"

    # Phase A: park the entry. The original (timed-out) dispatch's COID
    # is the parked one we must NOT reuse on the post-restart retry.
    broker_a = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        _, _, parked_coid = _park_entry(broker_a, ctx_a)
        ctx_a.record_resolution(parked_coid, 'rejected')
        ctx_a.close()

    # Phase B: restart. Pine still wants the entry. Resolution consumer
    # must drop both the active-intents and the in-memory envelope
    # anchor; the re-dispatch then builds a fresh envelope at the new
    # bar timestamp.
    broker_b = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        engine_b, pos_b = _mk_engine(broker_b, ctx_b)
        pos_b.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_b.sync(BAR_TS + 60_000)

        assert len(broker_b.entry_calls) == 1, (
            "rejected resolution must re-dispatch on the next sync"
        )
        retry_coid = broker_b.entry_calls[0].client_order_id('e')
        assert retry_coid != parked_coid, (
            "retry must use a fresh client_order_id; reusing the parked "
            "dispatch's COID would let the broker dedupe the retry as a "
            "duplicate of the rejected submission"
        )
        ctx_b.close()


# === Stale-run cleanup ====================================================


def __test_crashed_run_is_cleaned_on_next_open_run__(tmp_path: Path) -> None:
    """A crashed run with a stale heartbeat is closed on the next ``open_run``.

    SIGKILL simulation: the first run never calls ctx.close(), its
    heartbeat ages out, and the next ``open_run`` automatically closes
    it and starts a fresh instance.
    """
    db = tmp_path / "broker.sqlite"

    broker_a = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        engine_a, pos_a = _mk_engine(broker_a, ctx_a)
        pos_a.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_a.sync(BAR_TS)

        # Crash simulation: do NOT call ctx_a.close(). Instead, force
        # last_heartbeat_ts_ms below the stale threshold.
        store_a._conn.execute(
            "UPDATE runs SET last_heartbeat_ts_ms = 1 WHERE run_instance_id = ?",
            (ctx_a.run_instance_id,),
        )
        store_a._conn.commit()

    # Restart — open_run cleanup walks the stale row.
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        assert ctx_b.run_instance_id != ctx_a.run_instance_id

        # The crashed row now has ``ended_ts_ms`` set and a
        # ``stale_run_cleaned`` event attached.
        old = store_b._conn.execute(
            "SELECT ended_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx_a.run_instance_id,),
        ).fetchone()
        assert old['ended_ts_ms'] is not None

        ev = store_b._conn.execute(
            "SELECT kind FROM events WHERE run_instance_id = ? AND kind = 'stale_run_cleaned'",
            (ctx_a.run_instance_id,),
        ).fetchone()
        assert ev is not None


# === Capital.com plugin × OrderSyncEngine integration ======================
#
# The tests below stand up a real ``CapitalCom`` plugin (with the live
# ``_call`` method replaced by an in-memory response table) wired through
# the real ``OrderSyncEngine`` and ``BrokerStore``. Three end-to-end
# scenarios are exercised:
#
# 1. Entry → confirm → restart → ``deal_id`` ref survives.
# 2. POST-confirm timeout parks the dispatch, restart recovery promotes
#    the row to ``confirmed``, next sync's ``get_open_orders`` unparks
#    the envelope without re-dispatching.
# 3. Market entry + bracket exit in the same sync cycle — the Capital
#    plugin receives both envelopes and attaches the bracket to the
#    confirmed entry row.
#
# These complement the plugin-local unit tests (which stub REST at
# ``_call``) and the sync-engine-local tests above (which stub the broker
# at the plugin interface) by exercising both layers together.


import httpx  # noqa: E402

from pynecore.lib.strategy import _order_type_close  # noqa: E402


class _FakeCapitalCom:  # Forward-declared; replaced at import time below.
    pass


# Conditional import — the capitalcom plugin is editable-installed in the
# monorepo, but tests should not hard-fail if the plugin tree is missing
# (e.g. a pynecore-only sdist). ``pytest.skip`` provides a soft gate.
try:
    from pynecore_capitalcom import CapitalCom, CapitalComConfig
except ImportError:  # pragma: no cover — defensive; plugin is installed in repo
    CapitalCom = None  # type: ignore[assignment]
    CapitalComConfig = None  # type: ignore[assignment]


if CapitalCom is not None:

    class _FakeCapitalCom(CapitalCom):  # type: ignore[no-redef]
        """Capital plugin with REST stubbed via an (endpoint, method) table.

        Mirrors the plugin-local test harness, but lives here so the
        integration tests can drive the plugin end-to-end through the real
        :class:`OrderSyncEngine` without importing the plugin's private
        test helpers.
        """

        def __init__(self, *, config, responses=None):
            super().__init__(config=config)
            self._responses: dict = responses or {}
            self._calls: list = []

        async def _call(self, endpoint, *, data=None, method='post'):
            self._calls.append((endpoint, method, data))
            err = self._responses.get(('error', endpoint, method))
            if err is not None:
                # One-shot errors — consume so restart replays can succeed.
                del self._responses[('error', endpoint, method)]
                raise err
            return self._responses.get((endpoint, method), {})


_CAP_SYMBOL = "EURUSD"
_CAP_BAR = 1_700_000_000_000
# Minimal dealingRules payload so ``_get_instrument_rules`` returns a
# usable :class:`_InstrumentRules` without each test re-declaring it.
_CAP_RULES = {
    'dealingRules': {
        'minStepDistance': {'value': 0.01},
        'minDealSize': {'value': 0.01},
        'minNormalStopOrLimitDistance': {'value': 0.0001},
    },
    'instrument': {'lotSize': 0.01},
}


def _cap_config():
    return CapitalComConfig(
        demo=True, user_email="t@example.com",
        api_key="k", api_password="p",
    )


def _open_cap_ctx(store: BrokerStore) -> RunContext:
    identity = RunIdentity(
        strategy_id="cap-integration", symbol=_CAP_SYMBOL, timeframe="60",
        account_id="cap-acct", label=None,
    )
    return store.open_run(identity, script_source="// cap integration")


def _mk_cap_engine(
        broker, ctx: RunContext,
) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=_CAP_SYMBOL,
        run_tag=ctx.run_tag,
        mintick=0.0001,
        store_ctx=ctx,
    )
    return engine, pos


pytestmark_cap = pytest.mark.skipif(
    CapitalCom is None, reason="pynecore_capitalcom plugin not installed",
)


@pytestmark_cap
def __test_integration_capitalcom_entry_confirm_persists_deal_id_and_maps_order__(
        tmp_path: Path,
) -> None:
    """A LIMIT entry commits the deal_id ref and maps the exchange id in one sync.

    LIMIT entry dispatched through the engine → plugin's PERSIST-FIRST
    chain commits the deal_id ref and the engine's order_mapping reflects
    the exchange id in a single sync cycle.

    Exercises the full wiring: ``OrderSyncEngine._dispatch_new`` →
    :meth:`CapitalCom.execute_entry` → BrokerStore upserts/refs →
    return into :attr:`OrderSyncEngine.order_mapping`.
    """
    db = tmp_path / "broker.sqlite"

    broker = _FakeCapitalCom(config=_cap_config(), responses={
        (f'markets/{_CAP_SYMBOL}', 'get'): _CAP_RULES,
        ('workingorders', 'post'): {'dealReference': 'ref-A'},
        ('confirms/ref-A', 'get'): {
            'dealStatus': 'ACCEPTED', 'dealId': 'deal-A',
            'level': 1.0800, 'size': 1.0, 'status': 'WORKING',
        },
    })

    with BrokerStore(db, plugin_name="capitalcom") as store:
        ctx = _open_cap_ctx(store)
        broker.store_ctx = ctx
        engine, pos = _mk_cap_engine(broker, ctx)
        pos.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=1.0800,
        )
        engine.sync(_CAP_BAR)

        # engine.order_mapping is keyed by intent_key ('L') and values are
        # the exchange ids — the plugin must return deal-A for the LIMIT
        # entry.
        assert engine.order_mapping["L"] == ["deal-A"]
        # deal_id ref committed before execute_entry returned.
        row = ctx.find_by_ref('deal_id', 'deal-A')
        assert row is not None
        assert row.exchange_order_id == 'deal-A'
        assert row.state == 'confirmed'
        # A deal_reference ref is also attached from the PERSIST-FIRST
        # intermediate state — tests the whole §5.1 ordering.
        assert ctx.find_by_ref('deal_reference', 'ref-A') is not None
        ctx.close()


@pytestmark_cap
def __test_integration_capitalcom_parked_dispatch_recovers_and_unparks__(
        tmp_path: Path,
) -> None:
    """A POST-timeout park is recovered and unparked via ``get_open_orders`` without re-POSTing.

    POST timeout parks the dispatch; a subsequent recovery pass + a new
    engine sync unpark it via ``get_open_orders`` without re-POSTing.

    Same run_instance throughout — the current architecture scopes
    ``orders`` / ``order_refs`` per run_instance, so this is the
    widest recovery cycle the plugin + engine actually support. A
    cross-restart variant would need run_instance-spanning reads, which
    is out of scope here.
    """
    db = tmp_path / "broker.sqlite"

    broker = _FakeCapitalCom(config=_cap_config(), responses={
        (f'markets/{_CAP_SYMBOL}', 'get'): _CAP_RULES,
        # First POST times out — engine parks the envelope.
        ('error', 'workingorders', 'post'): httpx.TimeoutException(
            "simulated POST timeout",
        ),
    })

    with BrokerStore(db, plugin_name="capitalcom") as store:
        ctx = _open_cap_ctx(store)
        broker.store_ctx = ctx
        engine, pos = _mk_cap_engine(broker, ctx)
        pos.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=1.0800,
        )
        engine.sync(_CAP_BAR)

        # The POST timeout path parks the envelope and marks the row
        # ``disposition_unknown`` — both persisted invariants of §5.1.
        assert len(engine.pending_verification) == 1
        parked_coid = next(iter(engine.pending_verification))
        row = ctx.get_order(parked_coid)
        assert row is not None
        assert row.state == 'disposition_unknown'

        # Reset the one-shot error and seed the recovery endpoints so the
        # next call graph succeeds. The plugin's recovery uses activity
        # with a ±3 s createdDateUTC window against the row's
        # created_ts_ms — align it to the row timestamp.
        assert row.created_ts_ms is not None
        import datetime as _dt
        act_iso = _dt.datetime.fromtimestamp(
            row.created_ts_ms / 1000.0, tz=_dt.UTC,
        ).strftime('%Y-%m-%dT%H:%M:%S.000')
        broker._responses.update({
            ('positions', 'get'): {'positions': []},
            ('workingorders', 'get'): {
                'workingOrders': [
                    {'market': {'epic': _CAP_SYMBOL},
                     'workingOrderData': {
                         'dealId': 'deal-P', 'direction': 'BUY',
                         'orderType': 'LIMIT', 'orderLevel': 1.0800,
                         'orderSize': 1.0,
                         'createdDateUTC': act_iso,
                     }},
                ],
            },
            ('history/activity', 'get'): {'activities': [
                {'epic': _CAP_SYMBOL, 'direction': 'BUY', 'size': 1.0,
                 'dateUTC': act_iso, 'dealId': 'deal-P',
                 'type': 'WORKING_ORDER', 'status': 'ACCEPTED'},
            ]},
        })

        # Run the plugin-side recovery directly — connect() would do the
        # same before the WS subscribes, minus the network I/O.
        asyncio.run(broker._recover_in_flight_submissions())

        # ``disposition_unknown`` → single activity match → confirmed.
        row = ctx.get_order(parked_coid)
        assert row.state == 'confirmed'
        assert row.exchange_order_id == 'deal-P'

        # Drop the POST-time error so a re-dispatch would succeed (for
        # the asserting-that-it-does-NOT-happen test below).
        broker._responses.pop(('error', 'workingorders', 'post'), None)
        pre_sync_calls = list(broker._calls)

        # Second sync: ``_verify_pending_dispatches`` hits get_open_orders
        # (GET /workingorders), looks the coid up via the ``deal_id`` ref
        # the recovery just committed, and unparks.
        engine.sync(_CAP_BAR + 60_000)
        new_calls = broker._calls[len(pre_sync_calls):]
        assert not any(c[0] == 'workingorders' and c[1] == 'post'
                       for c in new_calls), (
            "the parked envelope must unpark via get_open_orders echo, "
            "not via a fresh POST — that would duplicate the order"
        )
        assert 'deal-P' in engine.order_mapping.get("L", [])
        assert parked_coid not in engine.pending_verification

        ctx.close()


@pytestmark_cap
def __test_integration_capitalcom_full_roundtrip_entry_bracket_tp_fill__(
        tmp_path: Path,
) -> None:
    """A MARKET entry + bracket exit attaches the bracket to the confirmed entry in one sync.

    MARKET entry + bracket exit through the engine → plugin attaches
    the bracket to the just-confirmed entry row inside a single sync cycle.

    The engine dispatches the entry intent first, which flips the row to
    ``state='confirmed'`` before the exit intent's dispatch reads it —
    the ordering invariant the plugin's exit handler depends on.
    """
    db = tmp_path / "broker.sqlite"

    broker = _FakeCapitalCom(config=_cap_config(), responses={
        (f'markets/{_CAP_SYMBOL}', 'get'): _CAP_RULES,
        # Entry (MARKET) — POST /positions + confirm
        ('positions', 'post'): {'dealReference': 'ref-E'},
        ('confirms/ref-E', 'get'): {
            'dealStatus': 'ACCEPTED', 'dealId': 'deal-E',
            'level': 1.0800, 'size': 1.0, 'status': 'OPEN',
        },
        # Exit bracket PUT
        ('positions/deal-E', 'put'): {'dealReference': 'ref-X'},
        ('confirms/ref-X', 'get'): {
            'dealStatus': 'ACCEPTED', 'dealId': 'deal-E',
        },
    })

    with BrokerStore(db, plugin_name="capitalcom") as store:
        ctx = _open_cap_ctx(store)
        broker.store_ctx = ctx
        engine, pos = _mk_cap_engine(broker, ctx)

        pos.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry,
        )
        pos.exit_orders[("TP", "L")] = Order(
            "L", -1.0, order_type=_order_type_close, exit_id="TP",
            limit=1.0900, stop=1.0750,
        )
        engine.sync(_CAP_BAR)

        # Entry confirmed, bracket legs persisted.
        entry_row = ctx.find_by_ref('deal_id', 'deal-E')
        assert entry_row is not None
        assert entry_row.state == 'confirmed'

        # Bracket legs share the parent deal id under a composite
        # client_order_id convention managed by the plugin.
        leg_coids = [r.client_order_id for r in ctx.iter_live_orders()
                     if (r.extras or {}).get('leg_kind') in ('tp', 'sl')]
        assert len(leg_coids) == 2, (
            "both TP and SL bracket legs must be persisted after the "
            "exit dispatch completes"
        )

        # Both bracket legs are mapped under intent_key 'TP\0L' (the
        # engine's composite key for strategy.exit("TP", from_entry="L")).
        tp_sl_ids = engine.order_mapping.get("TP\0L", [])
        assert any(':tp' in oid for oid in tp_sl_ids), (
            "TP leg composite id missing from engine.order_mapping"
        )
        assert any(':sl' in oid for oid in tp_sl_ids), (
            "SL leg composite id missing from engine.order_mapping"
        )

        ctx.close()


def __test_modify_rejected_after_restart_does_not_duplicate_order__(
        tmp_path: Path,
) -> None:
    """A modify-rejected after restart adopts the live order instead of duplicating it.

    After a process restart, a modify-rejected resolution must not
    cause the engine to re-dispatch the intent via ``execute_*``,
    which would create a SECOND order alongside the still-live original.

    The in-memory ``_modify_old_intents`` snapshot does not survive a
    restart — ``restored`` is ``None`` in the rejected branch.
    Without recovery, both ``_active_intents`` and ``_order_mapping``
    stay empty for the key, and ``_diff_and_dispatch`` fires
    ``_dispatch_new``.

    Fix contract: the engine recovers the original exchange order IDs
    from persisted order rows (``iter_live_orders`` + ``intent_key``
    filter) and populates ``_order_mapping`` so the cross-restart
    adoption path prevents the duplicate dispatch.
    """
    db = tmp_path / "broker.sqlite"

    # Phase A: dispatch entry, seed the order row, modify-park, resolve
    # as rejected, close the run.
    broker_a = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        engine_a, pos_a = _mk_engine(broker_a, ctx_a)
        pos_a.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=50_000.0,
        )
        engine_a.sync(BAR_TS)
        assert len(broker_a.entry_calls) == 1
        entry_coid = broker_a.entry_calls[0].client_order_id('e')
        original_xchg_id = engine_a.order_mapping['L'][0]

        # In production the plugin's execute_entry persists the order
        # row; the mock broker doesn't touch the store, so seed manually.
        ctx_a.upsert_order(
            entry_coid, intent_key='L', symbol=SYMBOL, side='buy',
            qty=1.0, state='confirmed',
            exchange_order_id=original_xchg_id,
        )

        # Arm modify-entry to timeout. Derive the coid at raise time
        # from the new envelope so it matches what _park_pending stores.
        async def _raising_modify(old, new):
            broker_a.modify_entry_calls.append((old, new))
            raise OrderDispositionUnknownError(
                "timeout",
                client_order_id=new.client_order_id('e'),
                cause=TimeoutError("timeout"),
            )

        broker_a.modify_entry = _raising_modify  # type: ignore[assignment]

        pos_a.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=51_000.0,
        )
        engine_a.sync(BAR_TS + 60_000)
        assert len(broker_a.modify_entry_calls) == 1, "modify must have been attempted"
        modify_coid = broker_a.modify_entry_calls[0][1].client_order_id('e')
        assert modify_coid in engine_a.pending_verification, "modify must be parked"

        ctx_a.record_resolution(modify_coid, 'rejected')
        ctx_a.close()

    # Phase B: restart. Pine still wants the modified intent.
    broker_b = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        engine_b, pos_b = _mk_engine(broker_b, ctx_b)
        pos_b.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=51_000.0,
        )
        engine_b.sync(BAR_TS + 120_000)

        assert len(broker_b.entry_calls) == 0, (
            "modify-rejected after restart must NOT re-dispatch the "
            "intent as a new order — the original is still live on "
            "the exchange"
        )
        assert "L" in engine_b.active_intents, (
            "intent must be adopted via cross-restart adoption "
            "(recovered _order_mapping)"
        )
        assert "L" in engine_b.order_mapping, (
            "_order_mapping must be recovered from persisted order rows"
        )
        assert original_xchg_id in engine_b.order_mapping["L"], (
            "recovered _order_mapping must reference the original "
            f"exchange order id ({original_xchg_id!r})"
        )
        ctx_b.close()


def __test_modify_rejected_after_restart_recovers_ids_from_modify_row__(
        tmp_path: Path,
) -> None:
    """Post-restart recovery reads order IDs from the ``modify`` row, not the empty attached one.

    Multi-record group: when ``pending_verifications`` holds both an
    older ``attached`` row from the original ``new`` dispatch (parked
    before ``_order_mapping`` existed → ``order_ids=[]``) AND a
    ``modify`` row resolved as ``'rejected'`` carrying the snapshot of
    the live exchange IDs, the post-restart recovery must read the
    IDs from the *modify* row.

    SQL returns the group unordered, so a naive ``records[0].order_ids``
    can land on the empty attached row, leaving ``_order_mapping``
    unrestored — and ``_diff_and_dispatch`` then re-dispatches the
    Pine intent as a brand-new order, duplicating the still-live
    original on the exchange.

    The fix iterates the group preferring ``dispatch_kind='modify'``
    rows with non-empty ``order_ids``.
    """
    db = tmp_path / "broker.sqlite"
    new_coid = "coid-new-1"
    modify_coid = "coid-mod-1"
    original_xchg_id = "xchg-1"

    # Phase A: hand-craft two pending_verifications rows for the same
    # key ('L'): the older 'new' row resolved as 'attached' with
    # order_ids=[] (parked before mapping existed), and the newer
    # 'modify' row resolved as 'rejected' with the order_ids snapshot.
    # Insertion order biases SQLite ROWID, so the older 'attached' row
    # surfaces first in iter_pending_resolutions — exactly the trap
    # the fix must survive.
    with BrokerStore(db, plugin_name=PLUGIN) as store_a:
        ctx_a = _open_ctx(store_a)
        ctx_a.upsert_order(
            new_coid, intent_key='L', symbol=SYMBOL, side='buy',
            qty=1.0, state='confirmed',
            exchange_order_id=original_xchg_id,
        )
        ctx_a.record_park(new_coid, 'L', kind='new', order_ids=[])
        ctx_a.record_resolution(new_coid, 'attached')
        ctx_a.record_park(
            modify_coid, 'L', kind='modify',
            order_ids=[original_xchg_id],
        )
        ctx_a.record_resolution(modify_coid, 'rejected')
        ctx_a.close()

    # Phase B: restart with the same Pine intent live. The engine must
    # adopt via the recovered _order_mapping rather than re-dispatch.
    # Force the iteration order so the empty 'attached' row surfaces
    # first — SQL has no ORDER BY, so without this clamp the test
    # would silently rely on insertion-order luck.
    broker_b = _MockBroker()
    with BrokerStore(db, plugin_name=PLUGIN) as store_b:
        ctx_b = _open_ctx(store_b)
        original_iter = ctx_b.iter_pending_resolutions

        def _attached_first() -> list:
            return sorted(
                original_iter(),
                key=lambda r: 0 if r.resolution == 'attached' else 1,
            )

        ctx_b.iter_pending_resolutions = _attached_first  # type: ignore[assignment]
        engine_b, pos_b = _mk_engine(broker_b, ctx_b)
        pos_b.entry_orders["L"] = Order(
            "L", 1.0, order_type=_order_type_entry, limit=51_000.0,
        )
        engine_b.sync(BAR_TS + 60_000)

        assert len(broker_b.entry_calls) == 0, (
            "modify-rejected after restart must NOT re-dispatch the "
            "intent as a new order even when the resolution group "
            "also contains an older 'attached' row with empty order_ids"
        )
        assert "L" in engine_b.active_intents
        assert "L" in engine_b.order_mapping, (
            "_order_mapping must be recovered from the modify row's "
            "order_ids snapshot, not the older attached row"
        )
        assert original_xchg_id in engine_b.order_mapping["L"], (
            "the modify row's order_ids snapshot must seed "
            "_order_mapping['L']; an empty attached row must not mask it"
        )
        ctx_b.close()


# === Run-ownership isolation on a one-way (netting) venue ==================


@dataclass
class _PositionMockBroker(_MockBroker):
    """``_MockBroker`` variant that serves a venue net position snapshot."""
    position: ExchangePosition | None = None

    async def get_position(self, symbol):
        return self.position


def __test_startup_does_not_adopt_a_concurrent_runs_venue_net__(
        tmp_path: Path,
) -> None:
    """A fresh run must not adopt another run's account+symbol exposure.

    Two independent runs share one account+symbol on a one-way venue: run A
    holds a 0.01 long, then run B starts. B's durable order journal has no
    rows for the symbol, so B owns nothing of the shared venue net — startup
    adoption must leave B flat rather than copying A's 0.01 into
    ``_position.size`` (which later mis-reports A's close as B's external
    close).
    """
    db = tmp_path / "broker.sqlite"
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx_b = _open_ctx(store)
        broker_b = _PositionMockBroker(
            position=ExchangePosition(
                symbol=SYMBOL, side="long", size=0.01, entry_price=1_900.0,
                unrealized_pnl=0.0, liquidation_price=None,
                leverage=1.0, margin_mode="isolated",
            ),
        )
        engine_b, pos_b = _mk_engine(broker_b, ctx_b)

        engine_b.reconcile()  # startup adoption

        assert pos_b.size == 0.0, (
            "run B owns none of the shared venue net — it must stay flat"
        )
        assert pos_b.sign == 0.0
        assert pos_b.open_trades == []
        ctx_b.close()


def __test_startup_adopts_own_open_position_across_restart__(
        tmp_path: Path,
) -> None:
    """A genuine single-run restart still adopts its own venue net in full.

    The run left a 0.01 long open in a prior process; its durable entry row
    survives with a ``filled_qty`` cursor covering the position. On restart
    the run owns the whole venue net, so the ownership clamp is a no-op and
    adoption proceeds exactly as before.
    """
    db = tmp_path / "broker.sqlite"
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        # Durable evidence of this run's own open long: a filled entry row.
        ctx.upsert_order(
            "own-entry", symbol=SYMBOL, side="buy", qty=0.01,
            state="confirmed", pine_entry_id="L", filled_qty=0.01,
        )
        broker = _PositionMockBroker(
            position=ExchangePosition(
                symbol=SYMBOL, side="long", size=0.01, entry_price=1_900.0,
                unrealized_pnl=0.0, liquidation_price=None,
                leverage=1.0, margin_mode="isolated",
            ),
        )
        engine, pos = _mk_engine(broker, ctx)

        engine.reconcile()  # startup adoption

        assert pos.size == 0.01, "the run owns its whole venue net — adopt it"
        assert pos.sign == 1.0
        assert len(pos.open_trades) == 1
        assert pos.open_trades[0].size == 0.01
        ctx.close()


def __test_startup_ownership_excludes_adopted_startup_rows__(
        tmp_path: Path,
) -> None:
    """A leg adopted for bookkeeping must not count as run-owned exposure.

    Regression for the Capital.com concurrent-run double count: a plugin
    seeds a confirmed ``position`` row (flagged ``adopted_startup``) for
    every *untracked* live venue leg so the local close paths have a row to
    route against. On a one-way (netting) account those adopted rows are
    another run's slice folded into the one shared venue net. If startup
    ownership reconstruction counted their ``filled_qty`` it would re-inflate
    the clamp and copy the foreign exposure into ``_position`` — exactly the
    aggregate double count (2x100 seen as 400) the clamp exists to stop.

    Run B owns a 0.01 long via its own filled entry row and has *also*
    adopted a foreign 0.02 leg. The venue net is 0.03, but B must adopt only
    its own 0.01.
    """
    db = tmp_path / "broker.sqlite"
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        # This run's own open long — a genuine filled entry row.
        ctx.upsert_order(
            "own-entry", symbol=SYMBOL, side="buy", qty=0.01,
            state="confirmed", pine_entry_id="L", filled_qty=0.01,
        )
        # A foreign leg this run merely adopted for close-routing bookkeeping.
        ctx.upsert_order(
            "__pyne_adopted__foreign", symbol=SYMBOL, side="buy", qty=0.02,
            state="confirmed", filled_qty=0.02,
            extras={"kind": "position", ADOPTED_STARTUP_EXTRA_KEY: True},
        )
        broker = _PositionMockBroker(
            position=ExchangePosition(
                symbol=SYMBOL, side="long", size=0.03, entry_price=1_900.0,
                unrealized_pnl=0.0, liquidation_price=None,
                leverage=1.0, margin_mode="isolated",
            ),
        )
        engine, pos = _mk_engine(broker, ctx)

        engine.reconcile()  # startup adoption

        assert pos.size == 0.01, (
            "only the run's own filled slice is owned — the adopted foreign "
            "leg must not inflate the ownership clamp"
        )
        assert pos.sign == 1.0
        ctx.close()


def __test_owned_size_ignores_provider_vs_venue_symbol_domain__(
        tmp_path: Path,
) -> None:
    """Ownership reconstruction must not filter the journal by symbol.

    The engine's ``symbol`` is the provider ticker (``BTCUSDT.P`` for a
    perp), while plugins journal order rows under the venue wire symbol
    (``BTCUSDT``). A symbol-filtered journal query would match zero rows
    and wrongly report the restarting run as owning nothing.
    """
    db = tmp_path / "broker.sqlite"
    with BrokerStore(db, plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        ctx.upsert_order(
            "own-entry", symbol=SYMBOL, side="buy", qty=0.01,
            state="confirmed", pine_entry_id="L", filled_qty=0.01,
        )
        broker = _PositionMockBroker(
            position=ExchangePosition(
                symbol=SYMBOL, side="long", size=0.01, entry_price=1_900.0,
                unrealized_pnl=0.0, liquidation_price=None,
                leverage=1.0, margin_mode="isolated",
            ),
        )
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=broker,  # type: ignore[arg-type]
            position=pos,
            symbol=SYMBOL + ".P",
            run_tag=ctx.run_tag,
            mintick=1.0,
            store_ctx=ctx,
        )

        engine.reconcile()  # startup adoption

        assert pos.size == 0.01, (
            "the provider '.P' ticker must not hide the run's own "
            "venue-symbol journal rows from ownership reconstruction"
        )
        ctx.close()
