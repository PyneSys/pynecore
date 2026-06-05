"""
Tests for :class:`OrderSyncEngine` — the diff/dispatch/event-routing core.

A :class:`MockBroker` implements just the async surface the engine uses,
recording every call so assertions can check which intent ended up where.
A stubbed :attr:`lib._script.initial_capital` keeps
:class:`BrokerPosition.equity` well-defined.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from pynecore import lib
from pynecore.core.broker.exceptions import (
    BracketAttachAfterFillRejectedError,
    BrokerManualInterventionError,
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    InsufficientMarginError,
    OrderDispositionUnknownError,
    OrderSkippedByPlugin,
)
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.core.broker.models import (
    BrokerEvent,
    CapabilityLevel,
    CloseIntent,
    DispatchEnvelope,
    ExchangeOrder,
    ExchangePosition,
    ExchangeCapabilities,
    ExitIntent,
    LegPartialRepairedEvent,
    LegRepairFailedEvent,
    OcaPartialFillPolicy,
    OrderEvent,
    OrderStatus,
    OrderType,
    LegType,
    InterceptorResult,
    PositionLeg,
)
from pynecore.core.broker.native_failsafe_manager import FailsafeHealth, FailsafeOwner
from pynecore.lib.strategy import (
    Order,
    _order_type_entry,
    _order_type_close,
    oca as _oca,
)


SYMBOL = "BTCUSDT"
RUN_TAG = "test"
BAR_TS = 1_700_000_000_000


@pytest.fixture(autouse=True)
def _stub_script():
    prev = lib._script
    lib._script = SimpleNamespace(initial_capital=1_000_000.0)
    try:
        yield
    finally:
        lib._script = prev


# === Mock broker ===


@dataclass
class MockBroker:
    """Duck-typed stand-in for :class:`BrokerPlugin`. Records all calls.

    Each call captures the full :class:`DispatchEnvelope` the sync engine
    sends so tests can inspect both the wrapped intent and the allocated
    ``client_order_id``.
    """
    entry_calls: list[DispatchEnvelope] = field(default_factory=list)
    exit_calls: list[DispatchEnvelope] = field(default_factory=list)
    close_calls: list[DispatchEnvelope] = field(default_factory=list)
    cancel_calls: list[DispatchEnvelope] = field(default_factory=list)
    modify_entry_calls: list[tuple[DispatchEnvelope, DispatchEnvelope]] = field(
        default_factory=list,
    )
    modify_exit_calls: list[tuple[DispatchEnvelope, DispatchEnvelope]] = field(
        default_factory=list,
    )
    open_orders: list[ExchangeOrder] = field(default_factory=list)
    position: ExchangePosition | None = None
    streamed_events: list[OrderEvent] = field(default_factory=list)
    watch_orders_impl: str = "generator"  # "generator" | "not_implemented"
    raise_on_next_entry: Exception | None = None
    raise_on_next_exit: Exception | None = None
    raise_on_next_cancel: Exception | None = None
    raise_on_next_get_open_orders: Exception | None = None
    raise_on_next_get_position: Exception | None = None
    capabilities: ExchangeCapabilities = field(default_factory=ExchangeCapabilities)
    _next_id: int = 0
    # One-way emulation (hedging): set ``position_port = self`` + canned
    # ``raw_legs`` to drive the engine through the core OneWayEmulator.
    position_port: Any = None
    raw_legs: list[PositionLeg] = field(default_factory=list)
    close_leg_calls: list[tuple[str, int]] = field(default_factory=list)
    place_leg_calls: list[float] = field(default_factory=list)
    amend_calls: list[tuple[str, float | None, float | None]] = field(
        default_factory=list,
    )
    # Number of upcoming ``amend_bracket`` calls that raise
    # ``OrderDispositionUnknownError`` (ambiguous timeout) before succeeding.
    fail_amend_unknown_count: int = 0
    # ``leg_id`` whose ``amend_bracket`` raises ``ExchangeConnectionError`` (a
    # dropped link mid round-trip); ``None`` disables the hook.
    fail_amend_conn_leg: str | None = None

    def get_capabilities(self) -> ExchangeCapabilities:
        return self.capabilities

    def _mk_order(self, envelope: DispatchEnvelope, kind: str) -> ExchangeOrder:
        self._next_id += 1
        intent = envelope.intent
        return ExchangeOrder(
            id=f"xchg-{self._next_id}",
            symbol=getattr(intent, 'symbol', SYMBOL),
            side=getattr(intent, 'side', 'buy'),
            order_type=OrderType.MARKET,
            qty=getattr(intent, 'qty', 0.0),
            filled_qty=0.0,
            remaining_qty=getattr(intent, 'qty', 0.0),
            price=None,
            stop_price=None,
            average_fill_price=None,
            status=OrderStatus.OPEN,
            timestamp=0.0,
            fee=0.0,
            fee_currency="",
            client_order_id=envelope.client_order_id(kind),
        )

    async def execute_entry(self, envelope):
        self.entry_calls.append(envelope)
        if self.raise_on_next_entry is not None:
            err = self.raise_on_next_entry
            self.raise_on_next_entry = None
            raise err
        return [self._mk_order(envelope, 'e')]

    async def execute_exit(self, envelope):
        self.exit_calls.append(envelope)
        if self.raise_on_next_exit is not None:
            err = self.raise_on_next_exit
            self.raise_on_next_exit = None
            raise err
        return [self._mk_order(envelope, 't')]

    async def execute_close(self, envelope):
        self.close_calls.append(envelope)
        return self._mk_order(envelope, 'c')

    async def execute_cancel(self, envelope):
        self.cancel_calls.append(envelope)
        if self.raise_on_next_cancel is not None:
            err = self.raise_on_next_cancel
            self.raise_on_next_cancel = None
            raise err
        return True

    async def modify_entry(self, old, new):
        self.modify_entry_calls.append((old, new))
        return [self._mk_order(new, 'e')]

    async def modify_exit(self, old, new):
        self.modify_exit_calls.append((old, new))
        return [self._mk_order(new, 't')]

    # Defensive-close residual contract — defaults mirror BrokerPlugin
    # base. Tests that exercise residual cancellation override these on
    # the instance.
    residual_refs_for_reject: list[str] = field(default_factory=list)
    cancel_broker_order_calls: list[str] = field(default_factory=list)
    raise_on_next_cancel_broker_ref: Exception | None = None

    def get_residual_orders_after_bracket_attach_reject(self, context):
        return list(self.residual_refs_for_reject)

    async def cancel_broker_order_ref(self, ref):
        self.cancel_broker_order_calls.append(ref)
        if self.raise_on_next_cancel_broker_ref is not None:
            err = self.raise_on_next_cancel_broker_ref
            self.raise_on_next_cancel_broker_ref = None
            raise err

    async def get_open_orders(self, symbol=None):
        if self.raise_on_next_get_open_orders is not None:
            err = self.raise_on_next_get_open_orders
            self.raise_on_next_get_open_orders = None
            raise err
        return list(self.open_orders)

    async def get_position(self, symbol):
        if self.raise_on_next_get_position is not None:
            err = self.raise_on_next_get_position
            self.raise_on_next_get_position = None
            raise err
        return self.position

    def watch_orders(self):
        if self.watch_orders_impl == "not_implemented":
            raise NotImplementedError

        async def _gen():
            for event in self.streamed_events:
                yield event

        return _gen()

    # --- PositionPort surface (one-way emulation; active only when
    # ``position_port = self`` is set on the instance) ---
    async def fetch_raw_positions(self, symbol):
        return [leg for leg in self.raw_legs if leg.symbol == symbol]

    async def get_volume_quantizer(self, symbol):
        return lambda u: int(u)

    async def close_leg(self, symbol, leg_id, volume, coid):
        self.close_leg_calls.append((leg_id, volume))

    async def reject_out_of_range(self, envelope, qty):
        return None

    async def place_leg(self, envelope, qty):
        self.place_leg_calls.append(qty)
        return [self._mk_order(envelope, 'e')]

    async def amend_bracket(self, symbol, leg_id, *, side, tp_price, sl_price,
                            trail_offset, coid):
        self.amend_calls.append((leg_id, tp_price, sl_price))
        if self.fail_amend_conn_leg is not None and leg_id == self.fail_amend_conn_leg:
            raise ExchangeConnectionError("amend link dropped")
        if self.fail_amend_unknown_count > 0:
            self.fail_amend_unknown_count -= 1
            raise OrderDispositionUnknownError(
                "amend timed out", client_order_id=coid,
            )


# === Helpers ===


def _entry_order(order_id, size, **kw) -> Order:
    return Order(order_id, size, order_type=_order_type_entry, **kw)


def _exit_order(from_entry, size, exit_id, **kw) -> Order:
    return Order(from_entry, size, order_type=_order_type_close, exit_id=exit_id, **kw)


def _mk_engine(broker, mintick: float = 1.0) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        mintick=mintick,
    )
    return engine, pos


def _sync(engine: OrderSyncEngine, *, bar_ts: int = BAR_TS) -> None:
    engine.sync(bar_ts)


def _fill_event(side: str, qty: float, price: float, *,
                pine_id: str, leg: LegType = LegType.ENTRY,
                xchg_id: str = "xchg-1") -> OrderEvent:
    exch = ExchangeOrder(
        id=xchg_id, symbol=SYMBOL, side=side,
        order_type=OrderType.MARKET, qty=qty, filled_qty=qty,
        remaining_qty=0.0, price=None, stop_price=None,
        average_fill_price=price, status=OrderStatus.FILLED,
        timestamp=0.0, fee=0.0, fee_currency="",
    )
    return OrderEvent(
        order=exch, event_type='filled', fill_price=price,
        fill_qty=qty, timestamp=0.0, pine_id=pine_id, leg_type=leg,
    )


# === Diff / dispatch ===


def __test_new_entry_dispatches_execute_entry__():
    """A fresh entry intent dispatches ``execute_entry`` and registers active tracking."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)

    assert len(b.entry_calls) == 1
    assert b.entry_calls[0].intent.pine_id == "L"
    assert b.entry_calls[0].intent.limit == 50_000.0
    assert engine.active_intents.keys() == {"L"}
    assert engine.order_mapping["L"] == ["xchg-1"]


def __test_entry_exchange_reject_does_not_halt_and_retries__():
    """An entry exchange reject does not halt the bot; the next sync re-attempts."""
    # An exchange reject on an ENTRY (e.g. a risk-engine veto / insufficient
    # funds that the plugin cannot pre-empt pre-flight) must NOT kill the bot.
    # The engine drops the signal for this sync and re-evaluates next bar.
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    b.raise_on_next_entry = ExchangeOrderRejectedError("Capital confirm REJECTED: RISK_CHECK")

    # Does not propagate — the bot stays alive.
    engine.sync(BAR_TS)

    assert len(b.entry_calls) == 1
    # Skipped: not registered as active, no order mapping retained.
    assert "L" not in engine.active_intents
    assert "L" not in engine.order_mapping

    # Next sync re-attempts (broker no longer rejects) and the entry lands.
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 2
    assert engine.active_intents.keys() == {"L"}


def __test_entry_reject_same_bar_retry_bumps_retry_seq__():
    """A same-bar retry after a reject keeps ``bar_ts_ms`` but bumps ``retry_seq`` to 1."""
    # A same-bar retry after an exchange reject must mint a FRESH COID: same
    # bar_ts_ms (the bar has not advanced) but a bumped retry_seq so it does
    # not collide with the spent COID in the exchange idempotency cache.
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    b.raise_on_next_entry = ExchangeOrderRejectedError("Capital confirm REJECTED: RISK_CHECK")

    engine.sync(BAR_TS)
    engine.sync(BAR_TS)

    assert len(b.entry_calls) == 2
    rejected, retried = b.entry_calls
    assert rejected.bar_ts_ms == retried.bar_ts_ms == BAR_TS
    assert rejected.retry_seq == 0
    assert retried.retry_seq == 1


def __test_entry_reject_later_bar_reemit_mints_fresh_anchor__():
    """A later-bar re-emit after a reject is stamped with the current bar and ``retry_seq=0``."""
    # When the entry is rejected on one bar but the strategy only re-emits it
    # on a LATER bar, the bumped reject anchor must NOT carry over: the new
    # bar's order is a fresh evaluation and must be stamped with the current
    # bar's bar_ts_ms and retry_seq=0, not the rejected bar's stale identity.
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    b.raise_on_next_entry = ExchangeOrderRejectedError("Capital confirm REJECTED: RISK_CHECK")

    engine.sync(BAR_TS)
    assert "L" not in engine.active_intents

    next_bar = BAR_TS + 60_000
    engine.sync(next_bar)

    assert len(b.entry_calls) == 2
    rejected, reemit = b.entry_calls
    assert rejected.bar_ts_ms == BAR_TS
    assert reemit.bar_ts_ms == next_bar
    assert reemit.retry_seq == 0
    assert engine.active_intents.keys() == {"L"}


def __test_entry_reject_same_bar_retry_bumps_retry_seq_with_store__(tmp_path):
    """Store-backed same-bar reject retry still mints ``retry_seq=1`` despite replay re-seed."""
    # Same as ``__test_entry_reject_same_bar_retry_bumps_retry_seq__`` but with
    # a persisted ``store_ctx`` configured — the normal live mode. The bumped
    # reject anchor is intentionally never journaled (a restart must
    # re-evaluate fresh), so the start-of-cycle ``store_ctx.replay()`` cannot
    # reconstruct it. Without an explicit re-seed step the second same-bar
    # ``sync`` would wipe the in-memory bump and rebuild ``retry_seq=0`` with
    # the rejected bar's ``bar_ts_ms``, colliding with the spent COID. This
    # regression guards that the same-bar retry still mints a FRESH COID
    # (``retry_seq=1``) on the store-backed path.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025",
                symbol=SYMBOL,
                timeframe="60",
                account_id="testbroker-demo",
                label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        b = MockBroker()
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos,
            symbol=SYMBOL,
            run_tag=RUN_TAG,
            mintick=1.0,
            store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
        b.raise_on_next_entry = ExchangeOrderRejectedError(
            "Capital confirm REJECTED: RISK_CHECK"
        )

        engine.sync(BAR_TS)
        engine.sync(BAR_TS)

        assert len(b.entry_calls) == 2
        rejected, retried = b.entry_calls
        assert rejected.bar_ts_ms == retried.bar_ts_ms == BAR_TS
        assert rejected.retry_seq == 0
        assert retried.retry_seq == 1


def __test_entry_reject_later_bar_reemit_mints_fresh_anchor_with_store__(tmp_path):
    """Store-backed later-bar re-emit prunes the stale bump and stamps the current bar."""
    # Store-backed twin of
    # ``__test_entry_reject_later_bar_reemit_mints_fresh_anchor__``: once the
    # bar advances, the re-seed step must prune the stale bump (memory +
    # journal) so the later-bar re-emit is stamped with the CURRENT bar and
    # ``retry_seq=0``, not the rejected bar's identity.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025",
                symbol=SYMBOL,
                timeframe="60",
                account_id="testbroker-demo",
                label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        b = MockBroker()
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos,
            symbol=SYMBOL,
            run_tag=RUN_TAG,
            mintick=1.0,
            store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
        b.raise_on_next_entry = ExchangeOrderRejectedError(
            "Capital confirm REJECTED: RISK_CHECK"
        )

        engine.sync(BAR_TS)
        assert "L" not in engine.active_intents

        next_bar = BAR_TS + 60_000
        engine.sync(next_bar)

        assert len(b.entry_calls) == 2
        rejected, reemit = b.entry_calls
        assert rejected.bar_ts_ms == BAR_TS
        assert reemit.bar_ts_ms == next_bar
        assert reemit.retry_seq == 0
        assert engine.active_intents.keys() == {"L"}


def __test_entry_reject_then_materialized_retry_survives_restart__(tmp_path):
    """A materialised ``retry_seq=1`` entry persists its anchor so a restart rebuilds its COID."""
    # A same-bar retry after an exchange reject mints retry_seq=1. The bumped
    # anchor is deliberately NOT journaled at build time (a non-materialised
    # retry must re-evaluate fresh after a restart). But once the retry
    # MATERIALISES — execute_entry succeeds and the order is live under the
    # retry_seq=1 COID — that identity MUST survive a restart: after replay,
    # _resolve_parent_opening_ref and every modify/cancel rebuild the parent
    # COID from _persisted_envelope_anchors. Without the materialisation-time
    # persistence, a restart reconstructs retry_seq=0 and targets the wrong COID
    # for the live order (native fail-safe retire / amend / cancel all miss).
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    identity = RunIdentity(
        strategy_id="t025", symbol=SYMBOL, timeframe="60",
        account_id="testbroker-demo", label=None,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(identity, script_source="src", script_path="t025.py")
        b = MockBroker()
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
        b.raise_on_next_entry = ExchangeOrderRejectedError(
            "Capital confirm REJECTED: RISK_CHECK"
        )

        engine.sync(BAR_TS)   # reject -> bump to retry_seq=1
        engine.sync(BAR_TS)   # same-bar retry -> retry_seq=1 materialises (lands)

        assert len(b.entry_calls) == 2
        assert b.entry_calls[1].retry_seq == 1
        assert engine.active_intents.keys() == {"L"}

        # Simulate a process restart: end the live run instance, re-open the
        # same logical run_id, and build a fresh engine that replays the store.
        ctx.close()
        ctx2 = store.open_run(identity, script_source="src", script_path="t025.py")
        engine2 = OrderSyncEngine(
            broker=MockBroker(),  # type: ignore[arg-type]
            position=BrokerPosition(), symbol=SYMBOL,
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx2,
        )

        anchor = engine2._persisted_envelope_anchors.get("L")  # type: ignore[attr-defined]
        assert anchor is not None
        assert anchor.retry_seq == 1
        assert anchor.bar_ts_ms == BAR_TS

        # The real consumer: the parent COID rebuilt from the replayed anchor
        # carries the bumped retry_seq, matching the live order's identity.
        expected = build_client_order_id(
            run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
            kind=KIND_ENTRY, retry_seq=1,
        )
        assert engine2._resolve_parent_opening_ref("L") == expected  # type: ignore[attr-defined]


def __test_entry_reject_then_parked_retry_survives_restart__(tmp_path):
    """A parked ``retry_seq=1`` entry journals its anchor so a restart rebuilds the COID."""
    # Twin of the clean-success case for the unknown-disposition (park) path:
    # the same-bar retry's execute_entry ends with OrderDispositionUnknownError,
    # so the order MAY be live at the broker under the retry_seq=1 COID.
    # record_park persists only the literal COID into pending_verifications, not
    # the anchor that _resolve_parent_opening_ref rebuilds from. The bumped
    # anchor must therefore also be journaled at park time so an attached
    # resolution after a restart reconstructs the correct COID.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    identity = RunIdentity(
        strategy_id="t025", symbol=SYMBOL, timeframe="60",
        account_id="testbroker-demo", label=None,
    )
    bumped_coid = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=1,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(identity, script_source="src", script_path="t025.py")
        b = MockBroker()
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
        b.raise_on_next_entry = ExchangeOrderRejectedError(
            "Capital confirm REJECTED: RISK_CHECK"
        )

        engine.sync(BAR_TS)   # reject -> bump to retry_seq=1
        b.raise_on_next_entry = OrderDispositionUnknownError(
            "simulated timeout", client_order_id=bumped_coid,
        )
        engine.sync(BAR_TS)   # same-bar retry -> retry_seq=1 parks (unknown)

        assert len(b.entry_calls) == 2
        assert b.entry_calls[1].retry_seq == 1
        assert bumped_coid in engine.pending_verification

        ctx.close()
        ctx2 = store.open_run(identity, script_source="src", script_path="t025.py")
        engine2 = OrderSyncEngine(
            broker=MockBroker(),  # type: ignore[arg-type]
            position=BrokerPosition(), symbol=SYMBOL,
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx2,
        )

        anchor = engine2._persisted_envelope_anchors.get("L")  # type: ignore[attr-defined]
        assert anchor is not None
        assert anchor.retry_seq == 1
        assert anchor.bar_ts_ms == BAR_TS
        assert engine2._resolve_parent_opening_ref("L") == bumped_coid  # type: ignore[attr-defined]


def _live_working_order(coid: str, *, order_id: str = "live-1") -> ExchangeOrder:
    """A broker-side OPEN entry working order carrying ``coid`` — the shape
    Capital.com's ``get_open_orders`` surfaces (COID restored from its own
    ref index)."""
    return ExchangeOrder(
        id=order_id, symbol=SYMBOL, side="buy",
        order_type=OrderType.LIMIT, qty=1.0, filled_qty=0.0,
        remaining_qty=1.0, price=50_000.0, stop_price=None,
        average_fill_price=None, status=OrderStatus.OPEN,
        timestamp=0.0, fee=0.0, fee_currency="",
        client_order_id=coid,
    )


def _restart_identity() -> "RunIdentity":  # noqa: F821 - local import in callers
    from pynecore.core.broker.run_identity import RunIdentity
    return RunIdentity(
        strategy_id="t025", symbol=SYMBOL, timeframe="60",
        account_id="testbroker-demo", label=None,
    )


def __test_restart_adopts_live_entry_coid_when_anchor_missing__(tmp_path):
    """On restart with no journal anchor, the scan adopts the live entry order's COID."""
    # The crash window Option C closes: a same-bar reject DELETEs the
    # retry_seq=0 journal row, the bumped retry_seq=1 materialises at the
    # broker, then the process crashes BEFORE the bump is journaled. On
    # restart the journal holds NO anchor for "L", but the working order is
    # live under the retry_seq=1 COID. Without adoption the engine would mint
    # a fresh retry_seq=0 id (a different COID the plugin dedup cannot match)
    # and double-open. The startup scan + build-time adoption must instead
    # rebuild the exact live COID — even when the script re-emits the entry on
    # a LATER bar than the one the order was placed on.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    bumped_coid = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=1,
    )
    later_bar = BAR_TS + 60_000
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        b = MockBroker()
        b.open_orders = [_live_working_order(bumped_coid)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

        engine.sync(later_bar)  # script re-emits "L" on a later bar

        assert len(b.entry_calls) == 1
        assert b.entry_calls[0].bar_ts_ms == BAR_TS
        assert b.entry_calls[0].retry_seq == 1
        assert b.entry_calls[0].client_order_id(KIND_ENTRY) == bumped_coid

        # The adoption journaled the recovered anchor — a second restart keeps it.
        ctx.close()
        ctx2 = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        engine2 = OrderSyncEngine(
            broker=MockBroker(),  # type: ignore[arg-type]
            position=BrokerPosition(), symbol=SYMBOL,
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx2,
        )
        anchor = engine2._persisted_envelope_anchors.get("L")  # type: ignore[attr-defined]
        assert anchor is not None
        assert anchor.bar_ts_ms == BAR_TS
        assert anchor.retry_seq == 1


def __test_restart_does_not_adopt_foreign_run_or_brand_new_entry__(tmp_path):
    """The restart scan ignores a foreign run_tag order and mints fresh for a brand-new entry."""
    # A live order from a DIFFERENT run_tag must be ignored, and a brand-new
    # entry with no live order must mint a fresh current-bar retry_seq=0 — the
    # scan only adopts this run's own entry orders.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    foreign = build_client_order_id(
        run_tag="zzzz", pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=3,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        b = MockBroker()
        b.open_orders = [_live_working_order(foreign)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

        engine.sync(BAR_TS)

        assert len(b.entry_calls) == 1
        assert b.entry_calls[0].bar_ts_ms == BAR_TS
        assert b.entry_calls[0].retry_seq == 0


def __test_restart_skips_ambiguous_live_entry_orders__(tmp_path):
    """Two live entry orders with distinct ``(bar, retry_seq)`` are ambiguous, so no adoption."""
    # Two live orders for the same entry pid_hash with DISTINCT
    # (bar_ts_ms, retry_seq) tuples are ambiguous — the engine never produces
    # two live working orders for one entry, so adoption is skipped and the
    # entry mints fresh rather than guessing.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    coid_a = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=1,
    )
    coid_b = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS + 60_000,
        kind=KIND_ENTRY, retry_seq=2,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        b = MockBroker()
        b.open_orders = [
            _live_working_order(coid_a, order_id="live-a"),
            _live_working_order(coid_b, order_id="live-b"),
        ]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

        engine.sync(BAR_TS)

        assert len(b.entry_calls) == 1
        assert b.entry_calls[0].retry_seq == 0
        assert b.entry_calls[0].bar_ts_ms == BAR_TS


def __test_restart_collapses_both_set_entry_legs_to_one_anchor__(tmp_path):
    """Both-set entry legs share one ``(bar, retry_seq)`` tuple, so the anchor is adopted."""
    # A both-set entry's KIND_ENTRY (LIMIT) and KIND_ENTRY_STOP (MARKET) legs
    # share the SAME pinned (bar_ts_ms, retry_seq), so two live legs collapse
    # to one distinct tuple and are NOT treated as ambiguous — the anchor is
    # adopted.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import (
        build_client_order_id, KIND_ENTRY, KIND_ENTRY_STOP,
    )

    coid_limit = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=1,
    )
    coid_stop = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY_STOP, retry_seq=1,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        b = MockBroker()
        b.open_orders = [
            _live_working_order(coid_limit, order_id="live-e"),
            _live_working_order(coid_stop, order_id="live-b"),
        ]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

        engine.sync(BAR_TS + 60_000)

        assert len(b.entry_calls) == 1
        assert b.entry_calls[0].bar_ts_ms == BAR_TS
        assert b.entry_calls[0].retry_seq == 1


def __test_restart_scan_connection_error_skips_sync_and_retries__(tmp_path):
    """A scan ``get_open_orders`` failure skips the sync and retries adoption next sync."""
    # If get_open_orders fails transiently during the scan, the sync is
    # skipped (no fresh dispatch can mint a colliding COID) and the scan flag
    # stays unset so the next sync retries. Once the broker recovers and the
    # live order is visible, adoption proceeds.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    bumped_coid = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=1,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        b = MockBroker()
        b.raise_on_next_get_open_orders = ExchangeConnectionError("broker down")
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

        engine.sync(BAR_TS)  # scan hits the connection error -> sync skipped

        assert len(b.entry_calls) == 0
        assert engine._restart_entry_scan_done is False  # type: ignore[attr-defined]

        b.open_orders = [_live_working_order(bumped_coid)]
        engine.sync(BAR_TS)  # broker recovered -> scan + adoption

        assert engine._restart_entry_scan_done is True  # type: ignore[attr-defined]
        assert len(b.entry_calls) == 1
        assert b.entry_calls[0].retry_seq == 1


def __test_restart_adopts_higher_same_bar_retry_over_journal_anchor__(tmp_path):
    """A higher same-bar live retry is adopted over a lower journaled anchor."""
    # Double-bump crash: the journal holds retry_seq=1, but a SECOND same-bar
    # reject bumped to retry_seq=2 and that order ACKed at the broker before
    # the crash. The same-bar higher live retry is authoritative over the
    # journal anchor.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    coid2 = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=2,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        ctx.record_envelope(key="L", bar_ts_ms=BAR_TS, retry_seq=1)
        b = MockBroker()
        b.open_orders = [_live_working_order(coid2)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

        engine.sync(BAR_TS)

        assert len(b.entry_calls) == 1
        assert b.entry_calls[0].bar_ts_ms == BAR_TS
        assert b.entry_calls[0].retry_seq == 2


def __test_restart_keeps_journal_anchor_on_cross_bar_live_retry__(tmp_path):
    """A cross-bar higher live retry is an orphan; the journal anchor stays authoritative."""
    # A higher live retry on a DIFFERENT bar than the journal anchor is a shape
    # the engine never produces (a bar advance resets retry to 0). It is treated
    # as an orphan: the journal anchor stays authoritative, no adoption.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    cross_bar = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS + 60_000,
        kind=KIND_ENTRY, retry_seq=2,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        ctx.record_envelope(key="L", bar_ts_ms=BAR_TS, retry_seq=1)
        b = MockBroker()
        b.open_orders = [_live_working_order(cross_bar)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

        engine.sync(BAR_TS)

        assert len(b.entry_calls) == 1
        assert b.entry_calls[0].bar_ts_ms == BAR_TS
        assert b.entry_calls[0].retry_seq == 1


def __test_entry_insufficient_margin_does_not_halt__():
    """An ``InsufficientMarginError`` on an entry is non-fatal, same as a plain reject."""
    # InsufficientMarginError is a typed, non-terminal ExchangeOrderRejectedError
    # subclass — same survive-and-retry handling as a plain reject.
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    b.raise_on_next_entry = InsufficientMarginError("Capital reject: INSUFFICIENT_FUNDS")

    engine.sync(BAR_TS)

    assert "L" not in engine.active_intents


def __test_exit_exchange_reject_still_halts__():
    """An exchange reject on a protective exit surfaces and halts, unlike an entry reject."""
    # The non-fatal handling is ENTRY-only. A plain exchange reject on a
    # protective EXIT is a real exposure (the position is open, the bracket
    # the broker refused leaves it unprotected) and must surface, not be
    # silently dropped.
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    b.raise_on_next_exit = ExchangeOrderRejectedError("Capital confirm REJECTED: X")

    with pytest.raises(ExchangeOrderRejectedError):
        engine.sync(BAR_TS)


def __test_unchanged_entry_is_not_redispatched__():
    """An unchanged entry across two syncs is dispatched only once."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)
    engine.sync(BAR_TS)

    assert len(b.entry_calls) == 1  # only once


def __test_modified_entry_dispatches_modify_entry__():
    """A changed entry limit dispatches ``modify_entry`` with the envelope identity preserved."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)

    # Replace with a different limit price
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=49_500.0)
    engine.sync(BAR_TS)

    assert len(b.modify_entry_calls) == 1
    old, new = b.modify_entry_calls[0]
    assert old.intent.limit == 50_000.0 and new.intent.limit == 49_500.0
    # Envelope identity is pinned on first dispatch and preserved on modify —
    # that is what makes the exchange treat the amend as idempotent.
    assert old.bar_ts_ms == new.bar_ts_ms == BAR_TS
    assert old.run_tag == new.run_tag == RUN_TAG


def __test_removed_entry_dispatches_cancel__():
    """An entry removed from the position dict dispatches a cancel and clears tracking."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)

    del pos.entry_orders["L"]
    engine.sync(BAR_TS)

    assert len(b.cancel_calls) == 1
    assert b.cancel_calls[0].intent.pine_id == "L"
    assert b.cancel_calls[0].intent.from_entry is None
    assert "L" not in engine.active_intents


def __test_cancel_all_orders_dispatches_cancel_for_every_active_intent__():
    """``cancel_all()`` clears the position dicts and dispatches a cancel per tracked intent.

    ``Pine strategy.cancel_all()`` clears the position dicts; the engine
    must then dispatch one cancel per previously tracked intent. Regression
    for the broker-mode crash where ``cancel_all()`` touched a non-existent
    ``orderbook`` attribute and bailed before any cancel went out."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L1"] = _entry_order("L1", 1.0, limit=50_000.0)
    pos.entry_orders["L2"] = _entry_order("L2", 1.0, limit=49_000.0)
    pos.exit_orders[("TP", "L1")] = _exit_order(
        "L1", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)

    pos._cancel_all_orders()
    engine.sync(BAR_TS)

    cancelled_ids = {(c.intent.pine_id, c.intent.from_entry) for c in b.cancel_calls}
    assert cancelled_ids == {("L1", None), ("L2", None), ("TP", "L1")}
    assert engine.active_intents == {}
    assert engine.order_mapping == {}


def __test_close_intent_dispatches_execute_close__():
    """A Pine close order dispatches ``execute_close`` with the opposite side."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -1.0, order_type=_order_type_close,
        exit_id="Close entry(s) order L",
    )

    engine.sync(BAR_TS)

    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.pine_id == "L"
    assert b.close_calls[0].intent.side == "sell"


def __test_exit_with_prices_dispatches_execute_exit__():
    """An exit with explicit TP/SL prices dispatches ``execute_exit`` carrying those levels."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )

    engine.sync(BAR_TS)

    assert len(b.exit_calls) == 1
    assert b.exit_calls[0].intent.tp_price == 60_000.0
    assert b.exit_calls[0].intent.sl_price == 45_000.0


# === Tick deferral + resolution ===


def __test_exit_with_ticks_without_entry_is_deferred__():
    """A tick-based exit with no entry fill yet is deferred, never reaching the plugin."""
    b = MockBroker()
    engine, pos = _mk_engine(b, mintick=1.0)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", profit_ticks=100.0, loss_ticks=50.0,
    )

    engine.sync(BAR_TS)

    # Exit never reaches the plugin while ticks are unresolved.
    assert b.exit_calls == []
    assert "TP\0L" in engine.deferred_exits
    assert "TP\0L" not in engine.active_intents


def __test_entry_fill_resolves_deferred_exit__():
    """A long entry fill resolves the deferred tick exit to absolute TP-above/SL-below prices."""
    b = MockBroker()
    engine, pos = _mk_engine(b, mintick=1.0)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", profit_ticks=100.0, loss_ticks=50.0,
    )
    engine.sync(BAR_TS)  # defers it

    engine.on_order_event(_fill_event(
        "buy", qty=1.0, price=50_000.0, pine_id="L", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS)  # drains the event, resolves ticks, dispatches

    assert len(b.exit_calls) == 1
    resolved = b.exit_calls[0].intent
    # Long entry (sign=+1): TP above, SL below.
    assert resolved.tp_price == 50_100.0
    assert resolved.sl_price == 49_950.0
    assert resolved.profit_ticks is None
    assert resolved.loss_ticks is None
    assert "TP\0L" not in engine.deferred_exits


def __test_short_entry_fill_reverses_tick_direction__():
    """A short entry fill resolves tick exits with TP below and SL above the entry price."""
    b = MockBroker()
    engine, pos = _mk_engine(b, mintick=1.0)
    pos.exit_orders[("TP", "S")] = _exit_order(
        "S", 1.0, "TP", profit_ticks=100.0, loss_ticks=50.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "sell", qty=1.0, price=50_000.0, pine_id="S", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS)

    resolved = b.exit_calls[0].intent
    # Short (sign=-1): TP below entry, SL above entry.
    assert resolved.tp_price == 49_900.0
    assert resolved.sl_price == 50_050.0


def __test_pyramiding_two_tick_exits_same_from_entry_no_collision__():
    """Pyramiding attaches multiple tick-deferred exits to one entry.

    Each exit lives under its own ``intent_key`` slot in ``_deferred_exits``;
    a single entry fill resolves every exit pointing at that entry in one
    pass. Fixture mirrors the Pine-side ``exit_orders`` composite keying.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b, mintick=1.0)
    pos.exit_orders[("TP1", "L")] = _exit_order(
        "L", -1.0, "TP1", profit_ticks=100.0, loss_ticks=50.0,
    )
    pos.exit_orders[("TP2", "L")] = _exit_order(
        "L", -1.0, "TP2", profit_ticks=200.0, loss_ticks=80.0,
    )

    engine.sync(BAR_TS)  # both should defer, neither dispatch

    assert b.exit_calls == []
    assert "TP1\0L" in engine.deferred_exits
    assert "TP2\0L" in engine.deferred_exits

    engine.on_order_event(_fill_event(
        "buy", qty=1.0, price=50_000.0, pine_id="L", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS)

    # Both exits must reach the plugin with their own resolved prices.
    assert len(b.exit_calls) == 2
    by_id = {env.intent.pine_id: env.intent for env in b.exit_calls}
    assert by_id["TP1"].tp_price == 50_100.0
    assert by_id["TP1"].sl_price == 49_950.0
    assert by_id["TP2"].tp_price == 50_200.0
    assert by_id["TP2"].sl_price == 49_920.0
    assert "TP1\0L" not in engine.deferred_exits
    assert "TP2\0L" not in engine.deferred_exits


# === Interceptor ===


def __test_interceptor_rejects_intent__():
    """A registered interceptor that rejects an intent blocks dispatch and tracking."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    def veto(_intent) -> InterceptorResult:
        return InterceptorResult(intent=_intent, rejected=True, reject_reason="no")

    engine.register_interceptor(veto)
    engine.sync(BAR_TS)

    assert b.entry_calls == []
    assert engine.active_intents == {}


def __test_interceptor_modifies_qty__():
    """A registered interceptor that halves the qty changes the dispatched intent's quantity."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    def half(_intent):
        return InterceptorResult(intent=_intent, modified_qty=_intent.qty * 0.5)

    engine.register_interceptor(half)
    engine.sync(BAR_TS)

    assert b.entry_calls[0].intent.qty == 0.5


# === Reconciliation ===


# === run_event_stream (async bridge) ===


def __test_run_event_stream_queues_all_events__():
    """``run_event_stream`` queues every watched event so the next sync drains them end-to-end."""
    b = MockBroker()
    b.streamed_events = [
        _fill_event("buy", qty=1.0, price=50_000.0,
                    pine_id="L", leg=LegType.ENTRY, xchg_id="x1"),
        _fill_event("sell", qty=1.0, price=50_500.0,
                    pine_id="L", leg=LegType.CLOSE, xchg_id="x2"),
    ]
    engine, pos = _mk_engine(b)

    asyncio.run(engine.run_event_stream())

    # Drain via the public path (sync) — verifies integration with record_fill.
    pos.avg_price = 50_000.0  # make equity finite for Trade bookkeeping
    engine.sync(BAR_TS)

    assert len(pos.closed_trades) == 0 or len(pos.closed_trades) == 1
    # We at least confirm the events flowed end-to-end by checking records
    assert len(pos.open_trades) + len(pos.closed_trades) >= 1


def __test_run_event_stream_handles_not_implemented__():
    """``run_event_stream`` returns cleanly when ``watch_orders`` raises NotImplementedError."""
    b = MockBroker()
    b.watch_orders_impl = "not_implemented"
    engine, pos = _mk_engine(b)

    # Should return cleanly, not raise.
    asyncio.run(engine.run_event_stream())


def __test_run_event_stream_handles_async_gen_not_implemented__():
    """NotImplementedError from the async-gen body is handled like one raised from the outer call.

    A plugin's ``watch_orders`` may raise NotImplementedError from the
    generator body rather than from the outer call — the engine must treat
    both the same way."""
    b = MockBroker()

    def _raise_in_body():
        async def _gen():
            raise NotImplementedError
            yield  # pragma: no cover — unreachable

        return _gen()

    b.watch_orders = _raise_in_body  # type: ignore[method-assign]
    engine, pos = _mk_engine(b)

    asyncio.run(engine.run_event_stream())


# === Reconciliation ===


# === Idempotency: client_order_id allocation + unknown-disposition recovery ===


def __test_dispatch_passes_deterministic_client_order_id__():
    """Plugins receive a canonical ``client_order_id`` via the envelope."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)

    env = b.entry_calls[0]
    coid = env.client_order_id('e')
    # Deterministic prefix built from RUN_TAG + hash(pine_id="L") + BAR_TS.
    assert coid.startswith(RUN_TAG + "-")
    assert coid.endswith("-e0")
    assert len(coid) <= 30


def __test_retry_within_same_bar_reuses_client_order_id__():
    """A second dispatch attempt in the same bar yields the same CO-ID so the exchange can dedup it.

    A second dispatch attempt in the same bar yields the same CO-ID so the
    exchange can dedup the duplicate."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)
    coid_first = b.entry_calls[0].client_order_id('e')

    # Simulate a second engine building the same envelope for the same logical
    # intent on the same bar — same inputs must produce the same CO-ID.
    engine2, pos2 = _mk_engine(MockBroker())
    pos2.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine2.sync(BAR_TS)
    coid_second = engine2._envelopes["L"].client_order_id('e')  # type: ignore[attr-defined]

    assert coid_first == coid_second


def _preview_entry_coid(pine_id: str, *, limit: float, bar_ts: int = BAR_TS) -> str:
    """Learn the ``client_order_id`` the engine will allocate for a given entry."""
    noop = MockBroker()
    engine, pos = _mk_engine(noop)
    pos.entry_orders[pine_id] = _entry_order(pine_id, 1.0, limit=limit)
    engine.sync(bar_ts)
    return noop.entry_calls[0].client_order_id('e')


def __test_unknown_disposition_parks_pending__():
    """A timed-out dispatch is parked on ``pending_verification``, not retried."""
    expected_coid = _preview_entry_coid("L", limit=50_000.0)

    b = MockBroker()
    b.raise_on_next_entry = OrderDispositionUnknownError(
        "simulated timeout", client_order_id=expected_coid,
    )
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)

    assert expected_coid in engine.pending_verification
    # The engine did call execute_entry exactly once — no auto-retry.
    assert len(b.entry_calls) == 1


def __test_verify_pending_promotes_matched_order__():
    """``_verify_pending_dispatches`` matches a pending CO-ID against open orders."""
    expected_coid = _preview_entry_coid("L", limit=50_000.0)

    b = MockBroker()
    b.raise_on_next_entry = OrderDispositionUnknownError(
        "simulated timeout", client_order_id=expected_coid,
    )
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)
    assert expected_coid in engine.pending_verification

    # The order actually did land; surface it on get_open_orders.
    b.open_orders = [
        ExchangeOrder(
            id="xchg-42", symbol=SYMBOL, side="buy",
            order_type=OrderType.LIMIT, qty=1.0, filled_qty=0.0,
            remaining_qty=1.0, price=50_000.0, stop_price=None,
            average_fill_price=None, status=OrderStatus.OPEN,
            timestamp=0.0, fee=0.0, fee_currency="",
            client_order_id=expected_coid,
        ),
    ]

    engine.sync(BAR_TS)

    assert expected_coid not in engine.pending_verification
    assert engine.order_mapping["L"] == ["xchg-42"]


def __test_verify_pending_keeps_pending_when_not_found__():
    """If ``get_open_orders`` does not surface the CO-ID, the pending stays."""
    expected_coid = _preview_entry_coid("L", limit=50_000.0)

    b = MockBroker()
    b.raise_on_next_entry = OrderDispositionUnknownError(
        "simulated timeout", client_order_id=expected_coid,
    )
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)
    # Second sync: exchange has no matching order; pending stays parked.
    engine.sync(BAR_TS)

    assert expected_coid in engine.pending_verification


def __test_verify_pending_connection_error_keeps_pending__():
    """A transient read failure must leave parked dispatches for the next sync."""
    expected_coid = _preview_entry_coid("L", limit=50_000.0)

    b = MockBroker()
    b.raise_on_next_entry = OrderDispositionUnknownError(
        "simulated timeout", client_order_id=expected_coid,
    )
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)
    b.raise_on_next_get_open_orders = ExchangeConnectionError("dns failed")
    pos.entry_orders["M"] = _entry_order("M", 1.0, limit=51_000.0)

    engine.sync(BAR_TS + 60_000)

    assert expected_coid in engine.pending_verification
    assert len(b.entry_calls) == 1
    assert "M" not in engine.active_intents

    engine.sync(BAR_TS + 120_000)

    assert expected_coid in engine.pending_verification
    assert len(b.entry_calls) == 2
    assert b.entry_calls[-1].intent.pine_id == "M"


def __test_reconcile_adopts_exchange_position_size__():
    """``reconcile`` adopts the exchange position's size, price, and PnL over the local view."""
    b = MockBroker()
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=2.0, entry_price=50_000.0,
        unrealized_pnl=12.5, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    pos.size = 1.0  # local tracking disagrees

    engine.reconcile()

    assert pos.size == 2.0
    assert pos.avg_price == 50_000.0
    assert pos.openprofit == 12.5
    assert engine.exchange_position is b.position


def __test_reconcile_clears_position_when_exchange_flat__():
    """User manually closes via web UI: ``get_position`` returns ``None``.

    The exchange is the source of truth — when it shows no position, the
    engine must drop ``position.size`` to 0 even if the local view still
    thinks there is an open position. Without this, a phantom adoption
    (or a real position closed externally during operation) leaves Pine
    forever convinced the bot is in a trade and blocks new entries.
    """
    b = MockBroker()
    b.position = None  # exchange flat — no row at all
    engine, pos = _mk_engine(b)
    pos.size = 100.0  # adopted earlier; user closed manually since
    pos.sign = 1.0
    pos.avg_price = 1.17

    engine.reconcile()

    assert pos.size == 0.0
    assert pos.sign == 0.0
    from pynecore.types.na import na_float
    assert pos.avg_price is na_float


def __test_reconcile_clears_open_trades_when_exchange_flat__():
    """When the exchange goes flat externally, open_trades MUST be wiped.

    Otherwise a re-entry on the next bar would mix new fills with stale
    trade rows and corrupt P&L bookkeeping.
    """
    from pynecore.lib.strategy import Trade
    b = MockBroker()
    b.position = None
    engine, pos = _mk_engine(b)
    pos.size = 100.0
    pos.sign = 1.0
    pos.avg_price = 1.17
    pos.open_trades.append(Trade(
        size=100.0, entry_id="L", entry_bar_index=0, entry_time=0,
        entry_price=1.17, commission=0.0, entry_comment=None,
        entry_equity=1_000_000.0,
    ))
    pos.openprofit = 5.0
    pos.open_commission = 0.5

    engine.reconcile()

    assert pos.size == 0.0
    assert pos.open_trades == []
    assert pos.openprofit == 0.0
    assert pos.open_commission == 0.0


def __test_reconcile_pending_defensive_close_within_grace_does_not_halt__():
    """A fresh pending marker within the grace window does NOT halt — the close FILL is in flight.

    A fresh pending marker (pending_since within the grace window)
    must NOT halt — the close FILL is legitimately in flight."""
    b = MockBroker()
    b.position = None
    engine, pos = _mk_engine(b)
    pos.size = 1.0
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )
    import time as _time
    engine._pending_defensive_close['Long'] = PendingDefensiveClose(
        entry_id='Long',
        close_intent_key='__pyne_defensive_close__coid-1',
        close_order_ref='xchg-2',
        pending_since=_time.time(),  # fresh
        reject_context=BracketAttachRejectContext(
            intent_key='Bracket\0Long', position_coid='coid-1',
            position_side='buy', qty=1.0, symbol=SYMBOL,
        ),
    )
    engine.reconcile()
    assert engine.halted is False


def __test_reconcile_pending_defensive_close_past_grace_halts__():
    """A past-grace pending marker halts the run when the broker still reports the position open.

    A pending marker older than the grace window halts the run when
    the broker still reports the position open — the FILL we are
    waiting on is not coming and the close was not silently completed
    server-side."""
    from pynecore.core.broker.sync_engine import (
        DEFENSIVE_CLOSE_RESOLUTION_GRACE_S,
    )
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )
    import time as _time

    b = MockBroker()
    # Broker still shows the position open — the close did NOT happen
    # silently on the server, this is a genuine stuck-pending halt.
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=1.0, entry_price=1.0,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    pos.size = 1.0
    engine._pending_defensive_close['Long'] = PendingDefensiveClose(
        entry_id='Long',
        close_intent_key='__pyne_defensive_close__coid-1',
        close_order_ref='xchg-2',
        pending_since=_time.time() - (DEFENSIVE_CLOSE_RESOLUTION_GRACE_S + 60.0),
        reject_context=BracketAttachRejectContext(
            intent_key='Bracket\0Long', position_coid='coid-1',
            position_side='buy', qty=1.0, symbol=SYMBOL,
        ),
    )
    with pytest.raises(BrokerManualInterventionError) as exc:
        engine.reconcile()
    assert exc.value.intent_key == '__pyne_defensive_close__coid-1'
    assert engine.halted is True


def __test_reconcile_pending_defensive_close_past_grace_settles_when_flat__():
    """A past-grace pending marker does NOT halt when the broker snapshot already shows flat.

    A pending marker past the grace window does NOT halt when the
    broker snapshot already shows the position flat — the close did
    settle, only the FILL event has not yet been queued (long restart
    gap, poll-based broker)."""
    from pynecore.core.broker.sync_engine import (
        DEFENSIVE_CLOSE_RESOLUTION_GRACE_S,
    )
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )
    import time as _time

    b = MockBroker()
    b.position = None  # broker is flat — close already happened
    engine, pos = _mk_engine(b)
    pos.size = 0.0  # reconcile-startup will adopt flat snapshot
    engine._pending_defensive_close['Long'] = PendingDefensiveClose(
        entry_id='Long',
        close_intent_key='__pyne_defensive_close__coid-1',
        close_order_ref='xchg-2',
        close_client_order_id='coid-close-1',
        pending_since=_time.time() - (DEFENSIVE_CLOSE_RESOLUTION_GRACE_S + 60.0),
        reject_context=BracketAttachRejectContext(
            intent_key='Bracket\0Long', position_coid='coid-1',
            position_side='buy', qty=1.0, symbol=SYMBOL,
        ),
    )
    engine.reconcile()
    assert engine.halted is False
    # Marker is settled — duplicate caches seeded, marker dropped.
    assert 'Long' not in engine._pending_defensive_close
    assert (
        '__pyne_defensive_close__coid-1'
        in engine._settled_defensive_close_pine_ids
    )
    assert 'xchg-2' in engine._settled_defensive_close_order_refs
    assert (
        'coid-close-1'
        in engine._settled_defensive_close_client_order_ids
    )


def __test_reconcile_pending_defensive_close_oldest_drives_halt__():
    """When multiple stale markers exist, the OLDEST one drives the halt message.

    When multiple markers exist, the OLDEST one drives the halt
    message — so operator triage starts from the longest-stuck close."""
    from pynecore.core.broker.sync_engine import (
        DEFENSIVE_CLOSE_RESOLUTION_GRACE_S,
    )
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )
    import time as _time

    b = MockBroker()
    # Broker still shows the position open — both stale markers are
    # genuinely stuck waiting for a FILL that did not arrive.
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=2.0, entry_price=1.0,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    pos.size = 1.0
    older = _time.time() - (DEFENSIVE_CLOSE_RESOLUTION_GRACE_S + 120.0)
    newer = _time.time() - (DEFENSIVE_CLOSE_RESOLUTION_GRACE_S + 30.0)
    engine._pending_defensive_close['LongA'] = PendingDefensiveClose(
        entry_id='LongA',
        close_intent_key='__pyne_defensive_close__coid-A',
        close_order_ref='xchg-A',
        pending_since=newer,
        reject_context=BracketAttachRejectContext(
            intent_key='B\0LongA', position_coid='coid-A',
            position_side='buy', qty=1.0, symbol=SYMBOL,
        ),
    )
    engine._pending_defensive_close['LongB'] = PendingDefensiveClose(
        entry_id='LongB',
        close_intent_key='__pyne_defensive_close__coid-B',
        close_order_ref='xchg-B',
        pending_since=older,
        reject_context=BracketAttachRejectContext(
            intent_key='B\0LongB', position_coid='coid-B',
            position_side='buy', qty=1.0, symbol=SYMBOL,
        ),
    )
    with pytest.raises(BrokerManualInterventionError) as exc:
        engine.reconcile()
    assert exc.value.intent_key == '__pyne_defensive_close__coid-B'  # older one


def __test_reconcile_pending_defensive_close_past_grace_settles_when_pyramiding_reduced__():
    """Past-grace marker settles when pyramiding-reduced broker size matches pre-close minus qty.

    With pyramiding/multi-entry, a successful defensive close for one
    entry reduces — but does not flatten — the netted aggregate position.
    The stale-grace path must accept "broker matches engine's pre-close
    view minus the closed entry's qty" as proof the close filled, instead
    of false-halting because the aggregate is not zero."""
    from pynecore.core.broker.sync_engine import (
        DEFENSIVE_CLOSE_RESOLUTION_GRACE_S,
    )
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )
    import time as _time

    b = MockBroker()
    # Two long entries totalled 2.0; defensive close for one (qty=1.0)
    # has filled silently, broker now reports the remaining 1.0 long.
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=1.0, entry_price=1.0,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    # Engine still has the pre-close view (FILL not yet routed).
    pos.size = 2.0
    engine._pending_defensive_close['LongB'] = PendingDefensiveClose(
        entry_id='LongB',
        close_intent_key='__pyne_defensive_close__coid-B',
        close_order_ref='xchg-B',
        close_client_order_id='coid-close-B',
        pending_since=_time.time() - (DEFENSIVE_CLOSE_RESOLUTION_GRACE_S + 60.0),
        reject_context=BracketAttachRejectContext(
            intent_key='B\0LongB', position_coid='coid-B',
            position_side='buy', qty=1.0, symbol=SYMBOL,
        ),
    )
    # Should NOT halt: broker_signed=+1.0 matches expected
    # (pos.size 2.0 minus marker qty 1.0 on the buy side).
    engine.reconcile()
    assert engine.halted is False
    assert 'LongB' not in engine._pending_defensive_close
    assert (
        '__pyne_defensive_close__coid-B'
        in engine._settled_defensive_close_pine_ids
    )
    # Engine view must track the broker snapshot we just used to prove
    # settlement. Without this catch-up the engine would stay at the
    # pre-close aggregate (2.0) while the broker is at 1.0 — periodic
    # reconcile's adopt-mismatch branch only acts on startup, so the
    # drift would survive until restart.
    assert pos.size == 1.0
    assert pos.sign == 1.0


def __test_reconcile_pending_defensive_close_pyramiding_mismatch_still_halts__():
    """Pyramiding still halts when the broker's leftover qty mismatches the aggregate marker qty.

    Pyramiding extension must NOT accept arbitrary leftover qty. If
    the broker's reduction does not match the stale markers' aggregate
    qty, the run must still halt — the deviation could mean the close
    did not fill or an unrelated fill arrived."""
    from pynecore.core.broker.sync_engine import (
        DEFENSIVE_CLOSE_RESOLUTION_GRACE_S,
    )
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )
    import time as _time

    b = MockBroker()
    # Engine view: 2.0 long. Marker says close qty 1.0. Broker reports
    # 1.5 long — no clean match (off by 0.5) — must halt.
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=1.5, entry_price=1.0,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    pos.size = 2.0
    engine._pending_defensive_close['LongB'] = PendingDefensiveClose(
        entry_id='LongB',
        close_intent_key='__pyne_defensive_close__coid-B',
        close_order_ref='xchg-B',
        pending_since=_time.time() - (DEFENSIVE_CLOSE_RESOLUTION_GRACE_S + 60.0),
        reject_context=BracketAttachRejectContext(
            intent_key='B\0LongB', position_coid='coid-B',
            position_side='buy', qty=1.0, symbol=SYMBOL,
        ),
    )
    with pytest.raises(BrokerManualInterventionError) as exc:
        engine.reconcile()
    assert exc.value.intent_key == '__pyne_defensive_close__coid-B'
    assert engine.halted is True


def __test_no_fifo_defensive_close_fill_preserves_pyramiding_size__():
    """No-FIFO defensive-close FILL on an adopted aggregate shrinks the position by qty, not flat.

    When the engine has an adopted aggregate position (size != 0, no
    open_trades) — for example pyramiding after a restart — and a
    defensive close FILL for one entry arrives via the no-FIFO routing
    branch, the in-memory position must shrink by the close's qty rather
    than fully flatten. Otherwise the engine would think the position is
    closed while the broker still has the other entries open.
    """
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )

    b = MockBroker()
    engine, pos = _mk_engine(b)
    # Adopted-position state: two long entries netted to 2.0, no FIFO
    # rows (typical of post-restart reconcile that adopted size but did
    # not reconstruct ``open_trades``).
    pos.size = 2.0
    pos.sign = 1.0
    pos.open_trades.clear()
    engine._pending_defensive_close['LongB'] = PendingDefensiveClose(
        entry_id='LongB',
        close_intent_key='__pyne_defensive_close__coid-B',
        close_order_ref='xchg-B',
        close_client_order_id='coid-close-B',
        pending_since=0.0,
        reject_context=BracketAttachRejectContext(
            intent_key='B\0LongB', position_coid='coid-B',
            position_side='buy', qty=1.0, symbol=SYMBOL,
        ),
    )
    # Defensive close FILL for LongB, qty 1.0, sell side.
    engine.on_order_event(_fill_event(
        'sell', qty=1.0, price=50_000.0,
        pine_id="__pyne_defensive_close__coid-B",
        leg=LegType.CLOSE,
        xchg_id='xchg-B',
    ))
    engine.apply_async_events()
    # Engine must track broker reality: 2.0 - 1.0 = 1.0 remaining long.
    assert pos.size == 1.0
    assert pos.sign == 1.0
    # Marker was settled.
    assert 'LongB' not in engine._pending_defensive_close


def __test_no_fifo_defensive_close_fill_flattens_single_entry__():
    """Single-entry no-FIFO defensive close FILL still flattens the position to zero.

    The original single-entry no-FIFO defensive close path must still
    flatten the position. With ``pos.size == 1.0`` and a 1.0 close FILL,
    the signed-delta logic naturally lands at zero (regression guard for
    the historic flatten behaviour)."""
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )

    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 1.0
    pos.sign = 1.0
    pos.open_trades.clear()
    engine._pending_defensive_close['Long'] = PendingDefensiveClose(
        entry_id='Long',
        close_intent_key='__pyne_defensive_close__coid-1',
        close_order_ref='xchg-2',
        close_client_order_id='coid-close-1',
        pending_since=0.0,
        reject_context=BracketAttachRejectContext(
            intent_key='B\0Long', position_coid='coid-1',
            position_side='buy', qty=1.0, symbol=SYMBOL,
        ),
    )
    engine.on_order_event(_fill_event(
        'sell', qty=1.0, price=50_000.0,
        pine_id="__pyne_defensive_close__coid-1",
        leg=LegType.CLOSE,
        xchg_id='xchg-2',
    ))
    engine.apply_async_events()
    assert pos.size == 0.0
    assert pos.sign == 0.0
    assert 'Long' not in engine._pending_defensive_close


def __test_reconcile_plugin_override_grace_window__():
    """A plugin can extend the grace window via ``defensive_close_resolution_grace_s``.

    A plugin can extend the grace window via the
    ``defensive_close_resolution_grace_s`` class attribute — useful for
    slow venues with multi-minute post-trade reporting latency."""
    from pynecore.core.broker.sync_engine import (
        DEFENSIVE_CLOSE_RESOLUTION_GRACE_S,
    )
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )
    import time as _time

    b = MockBroker()
    # Wider than default — the marker we install would halt under the
    # default 30 s grace but stays under 5 minutes.
    b.defensive_close_resolution_grace_s = 600.0  # type: ignore[attr-defined]
    b.position = None
    engine, pos = _mk_engine(b)
    pos.size = 1.0
    engine._pending_defensive_close['Long'] = PendingDefensiveClose(
        entry_id='Long',
        close_intent_key='__pyne_defensive_close__coid-1',
        close_order_ref='xchg-2',
        pending_since=_time.time() - (DEFENSIVE_CLOSE_RESOLUTION_GRACE_S + 60.0),
        reject_context=BracketAttachRejectContext(
            intent_key='Bracket\0Long', position_coid='coid-1',
            position_side='buy', qty=1.0, symbol=SYMBOL,
        ),
    )
    engine.reconcile()
    assert engine.halted is False


def __test_reconcile_no_change_when_sizes_match__():
    """No mutation when exchange and internal already agree."""
    b = MockBroker()
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=100.0, entry_price=1.17,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    pos.size = 100.0
    pos.sign = 1.0
    pos.avg_price = 1.17

    engine.reconcile()

    assert pos.size == 100.0
    assert pos.avg_price == 1.17


def __test_periodic_reconcile_clears_state_on_external_flatten__():
    """Mid-operation reconcile after the user flattens via web UI.

    Pre-condition: bot has dispatched + filled an entry, internal mirrors
    the exchange. Then the user closes manually (exchange returns ``None``)
    and the next sync's reconcile must wipe internal state so Pine sees
    ``position_size == 0`` and can re-enter on a future bar.
    """
    from pynecore.lib.strategy import Trade
    b = MockBroker()
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=100.0, entry_price=1.17,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    pos.size = 100.0
    pos.sign = 1.0
    pos.avg_price = 1.17
    pos.open_trades.append(Trade(
        size=100.0, entry_id="L", entry_bar_index=0, entry_time=0,
        entry_price=1.17, commission=0.0, entry_comment=None,
        entry_equity=1_000_000.0,
    ))

    engine.reconcile()  # startup — agreement, no change
    assert pos.size == 100.0

    # User flattens externally; next periodic reconcile sees /positions empty.
    b.position = None
    engine._sync_count = 1  # simulate post-startup periodic call

    engine.reconcile()

    assert pos.size == 0.0
    assert pos.open_trades == []


def __test_periodic_reconcile_does_not_adopt_size_increase__():
    """Periodic reconcile MUST NOT adopt a size increase not yet seen via ``record_fill``.

    Mid-operation reconcile MUST NOT adopt a size increase the engine
    has not yet seen via ``record_fill``.

    Race scenario: a market entry the engine just dispatched fills
    *between* the activity poll and the engine's own /positions read, so
    /positions briefly shows a position the matching ``OrderEvent`` has
    not yet drained into ``BrokerPosition``. Adopting that here would
    double-count the size when the event eventually arrives.
    """
    b = MockBroker()
    # Exchange "ahead" of internal — fill in flight.
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=100.0, entry_price=1.17,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)
    pos.size = 0.0  # event has not yet drained
    engine._sync_count = 1  # post-startup

    engine.reconcile()

    assert pos.size == 0.0  # untouched — record_fill will own this update


def __test_sync_skips_periodic_reconcile_connection_error__():
    """Periodic read-side reconcile retries later instead of stopping live sync."""
    from pynecore.lib.strategy import Trade

    b = MockBroker()
    b.raise_on_next_get_position = ExchangeConnectionError("dns failed")
    engine, pos = _mk_engine(b)
    engine._reconcile_every = 1
    pos.size = 100.0
    pos.sign = 1.0
    pos.avg_price = 1.17
    pos.open_trades.append(Trade(
        size=100.0, entry_id="L", entry_bar_index=0, entry_time=0,
        entry_price=1.17, commission=0.0, entry_comment=None,
        entry_equity=1_000_000.0,
    ))

    engine.sync(BAR_TS)

    assert pos.size == 100.0
    assert pos.open_trades

    b.position = None
    engine.sync(BAR_TS + 60_000)

    assert pos.size == 0.0
    assert pos.open_trades == []


def __test_periodic_reconcile_skips_clear_while_close_in_flight__():
    """Periodic reconcile must not clear the position while a bot-dispatched close is in flight.

    Reconcile must not clear the position while a bot-dispatched close
    is in flight.

    Race scenario: bar N dispatches ``execute_close``; the broker flattens
    /positions seconds before the matching ``OrderEvent`` reaches the
    queue. If reconcile zeros the position now, the closing fill (when it
    finally drains) would arrive with ``size == 0`` and enter
    :meth:`BrokerPosition.record_fill`'s "Opening" branch — counted as a
    fresh entry in the opposite direction.
    """
    from pynecore.core.broker.models import CloseIntent
    from pynecore.lib.strategy import Trade
    b = MockBroker()
    b.position = None  # exchange has flattened — close hit
    engine, pos = _mk_engine(b)
    pos.size = 1.0
    pos.sign = 1.0
    pos.avg_price = 50_000.0
    pos.open_trades.append(Trade(
        size=1.0, entry_id="L", entry_bar_index=0, entry_time=0,
        entry_price=50_000.0, commission=0.0, entry_comment=None,
        entry_equity=1_000_000.0,
    ))
    engine._sync_count = 1
    # Simulate a CloseIntent we dispatched but whose fill event has not
    # yet drained.
    engine._active_intents["L"] = CloseIntent(
        pine_id="L", symbol=SYMBOL, side="sell", qty=1.0,
    )

    engine.reconcile()

    assert pos.size == 1.0  # left alone for record_fill to own
    assert len(pos.open_trades) == 1


def __test_reconcile_does_not_warn_on_tracked_orders_missing_from_exchange__(caplog):
    """Regression: ``reconcile()`` must not diff ``_order_mapping`` against ``get_open_orders``.

    On brokers like Capital.com a Pine entry becomes an exchange-side
    *position* (not a working order) and the bracket lives as
    ``profitLevel`` / ``stopLevel`` *attributes* on that position — neither
    is visible to ``get_open_orders``, which only enumerates the
    working-orders namespace. Diffing tracked IDs against that namespace
    produced a permanent false-positive ``tracked orders missing from
    exchange`` warning every bar.

    Detection of bot-owned-order disappearance is now plugin-owned (signal
    via ``watch_orders`` ``cancelled`` event or ``UnexpectedCancelError``);
    the engine reconcile only checks position size mismatch.
    """
    import logging
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)
    # Engine tracks both intents — the IDs were assigned by the mock broker
    # at dispatch time.
    assert engine.order_mapping  # sanity: tracking IS populated

    # Simulate the post-fill steady state: the bracket lives on a position
    # that ``get_open_orders`` cannot see (Capital.com semantics).
    b.open_orders = []
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=1.0, entry_price=50_000.0,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    pos.size = 1.0
    pos.sign = 1.0
    pos.avg_price = 50_000.0
    engine._sync_count = 1  # post-startup periodic call

    with caplog.at_level(logging.WARNING, logger="pyne_core_logger"):
        engine.reconcile()

    assert not any(
        "tracked orders missing from exchange" in rec.getMessage()
        for rec in caplog.records
    ), "engine must not diff _order_mapping against get_open_orders"


# === OCA cascade cancel ===
#
# The engine must cancel OCA-cancel siblings the moment a fill event arrives,
# not wait for the next bar's diff pass. These tests exercise the full event
# → sync → cascade path with both entry-side and exit-side fills.


def _mk_engine_with_policy(
        broker: MockBroker,
        *,
        policy: OcaPartialFillPolicy = OcaPartialFillPolicy.FILL_CANCELS,
) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        oca_partial_fill_policy=policy,
    )
    return engine, pos


def _oca_entry(order_id: str, size: float, *, oca_name: str,
               oca_type, limit: float | None = None) -> Order:
    return Order(
        order_id, size, order_type=_order_type_entry,
        limit=limit, oca_name=oca_name, oca_type=oca_type,
    )


def __test_fill_cascades_cancel_to_oca_siblings__():
    """Full fill on A triggers an immediate cancel dispatch for sibling B."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.cancel, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.cancel, limit=49_000.0,
    )
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 2
    assert set(engine.active_intents) == {"A", "B"}

    # A fills — must emit a cancel for B on the next sync's drain.
    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 60_000)

    assert len(b.cancel_calls) == 1
    assert b.cancel_calls[0].intent.pine_id == "B"
    assert "B" not in engine.active_intents
    # Pine-side cleanup mirrors SimPosition._cancel_oca_group.
    assert "B" not in pos.entry_orders


def __test_partial_fill_cascades_under_fill_cancels_policy__():
    """Default policy treats a partial fill as a committed win for the leg."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.cancel, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.cancel, limit=49_000.0,
    )
    engine.sync(BAR_TS)

    partial = _fill_event("buy", 0.4, 50_000.0, pine_id="A", leg=LegType.ENTRY)
    partial.event_type = 'partial'
    engine.on_order_event(partial)
    engine.sync(BAR_TS + 60_000)

    assert len(b.cancel_calls) == 1
    assert b.cancel_calls[0].intent.pine_id == "B"


def __test_partial_fill_does_not_cascade_under_full_fill_only_policy__():
    """FULL_FILL_ONLY keeps siblings live until the leg is fully filled."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(
        b, policy=OcaPartialFillPolicy.FULL_FILL_ONLY,
    )
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.cancel, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.cancel, limit=49_000.0,
    )
    engine.sync(BAR_TS)

    partial = _fill_event("buy", 0.4, 50_000.0, pine_id="A", leg=LegType.ENTRY)
    partial.event_type = 'partial'
    engine.on_order_event(partial)
    engine.sync(BAR_TS + 60_000)

    assert b.cancel_calls == []
    assert "B" in engine.active_intents

    # Full fill then arrives — cascade must trigger now.
    engine.on_order_event(_fill_event(
        "buy", 0.6, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 120_000)

    assert len(b.cancel_calls) == 1
    assert b.cancel_calls[0].intent.pine_id == "B"


def __test_native_oca_cancel_suppresses_cascade__():
    """When the exchange owns the OCA group, the sync engine stays hands-off."""
    b = MockBroker(
        capabilities=ExchangeCapabilities(oca_cancel=CapabilityLevel.NATIVE),
    )
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.cancel, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.cancel, limit=49_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 60_000)

    assert b.cancel_calls == []
    # Exchange takes care of B; engine's active_intents still reflect both
    # until the plugin surfaces a separate cancelled event for B.
    assert "B" in engine.active_intents


def __test_two_fills_same_group_same_sync_emit_one_cancel__():
    """Per-group dedup inside a single sync pass prevents duplicate cancels."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.cancel, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.cancel, limit=49_000.0,
    )
    pos.entry_orders["C"] = _oca_entry(
        "C", 1.0, oca_name="G", oca_type=_oca.cancel, limit=48_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    # A second spurious fill on the same group (e.g. a partial followed by a
    # full fill reported separately) must not re-trigger the cascade.
    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 60_000)

    # Two siblings cancelled, but only on the first fill — the second is no-op.
    assert len(b.cancel_calls) == 2
    assert {c.intent.pine_id for c in b.cancel_calls} == {"B", "C"}


def __test_non_cancel_oca_does_not_cascade__():
    """OCA-reduce groups stay alive on fill (partial-fill qty-amend is WS5)."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _oca_entry(
        "A", 1.0, oca_name="G", oca_type=_oca.reduce, limit=50_000.0,
    )
    pos.entry_orders["B"] = _oca_entry(
        "B", 1.0, oca_name="G", oca_type=_oca.reduce, limit=49_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 60_000)

    assert b.cancel_calls == []
    assert "B" in engine.active_intents


def __test_standalone_fill_without_oca_group_is_quiet__():
    """Fills on non-OCA intents never touch the cascade path."""
    b = MockBroker()
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["A"] = _entry_order("A", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="A", leg=LegType.ENTRY,
    ))
    engine.sync(BAR_TS + 60_000)

    assert b.cancel_calls == []


# === Partial entry fill → bracket qty amend (WS5, Option A) ===


def _partial_entry_event(*, pine_id: str, fill_delta: float,
                         cumulative_filled: float, order_qty: float,
                         price: float, xchg_id: str = "xchg-1") -> OrderEvent:
    """Build an ``event_type='partial'`` entry fill with cumulative tracking.

    ``fill_delta`` is what the plugin reports this tick; ``cumulative_filled``
    is the running total on the exchange-side order (what the sync engine
    reads via ``event.order.filled_qty``).
    """
    exch = ExchangeOrder(
        id=xchg_id, symbol=SYMBOL, side="buy",
        order_type=OrderType.LIMIT, qty=order_qty,
        filled_qty=cumulative_filled,
        remaining_qty=order_qty - cumulative_filled,
        price=price, stop_price=None, average_fill_price=price,
        status=OrderStatus.PARTIALLY_FILLED,
        timestamp=0.0, fee=0.0, fee_currency="",
    )
    return OrderEvent(
        order=exch, event_type='partial', fill_price=price,
        fill_qty=fill_delta, timestamp=0.0,
        pine_id=pine_id, leg_type=LegType.ENTRY,
    )


def _mk_engine_with_sink(
        broker: MockBroker, sink: list[BrokerEvent],
) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        broker_event_sink=sink.append,
    )
    return engine, pos


def __test_partial_entry_fill_amends_bracket_qty__():
    """A 40% partial entry fill scales the bracket down to 0.4."""
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, pos = _mk_engine_with_sink(b, events)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)
    assert len(b.exit_calls) == 1
    assert b.exit_calls[0].intent.qty == 1.0

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.4, cumulative_filled=0.4,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)

    assert len(b.modify_exit_calls) == 1
    old, new = b.modify_exit_calls[0]
    assert old.intent.qty == 1.0
    assert new.intent.qty == 0.4
    assert engine.active_intents["TP\0L"].qty == 0.4

    repair_events = [e for e in events if isinstance(e, LegPartialRepairedEvent)]
    assert len(repair_events) == 1
    assert repair_events[0].old_qty == 1.0
    assert repair_events[0].new_qty == 0.4


def __test_subsequent_partial_fill_emits_another_amend__():
    """Each partial fill with a new cumulative qty triggers a fresh amend."""
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, pos = _mk_engine_with_sink(b, events)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.3, cumulative_filled=0.3,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)
    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.4, cumulative_filled=0.7,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 120_000)

    assert len(b.modify_exit_calls) == 2
    assert b.modify_exit_calls[0][1].intent.qty == 0.3
    assert b.modify_exit_calls[1][1].intent.qty == 0.7


def __test_native_bracket_skips_partial_amend__():
    """tp_sl_bracket=NATIVE — the plugin/exchange tracks partial fills."""
    b = MockBroker(
        capabilities=ExchangeCapabilities(tp_sl_bracket=CapabilityLevel.NATIVE),
    )
    engine, pos = _mk_engine_with_policy(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.4, cumulative_filled=0.4,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)

    assert b.modify_exit_calls == []
    # Bracket intent untouched: still the original 1.0 qty.
    assert engine.active_intents["TP\0L"].qty == 1.0


def __test_partial_fill_without_bracket_is_quiet__():
    """Entry without a paired exit → no amend, no event."""
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, pos = _mk_engine_with_sink(b, events)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.5, cumulative_filled=0.5,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)

    assert b.modify_exit_calls == []
    assert events == []


def __test_overfill_is_capped_and_emits_leg_repair_failed__():
    """filled_qty > entry_intent.qty → cap at entry qty + LegRepairFailedEvent.

    The bracket was originally dispatched at 1.0; the cap lands it at 1.0
    again, so no second modify_exit is needed — the critical outcome is the
    :class:`LegRepairFailedEvent` surfacing the exchange anomaly.
    """
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, pos = _mk_engine_with_sink(b, events)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=1.2, cumulative_filled=1.2,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)

    # Bracket qty stays at entry qty (cap), no redundant modify_exit.
    assert engine.active_intents["TP\0L"].qty == 1.0
    assert b.modify_exit_calls == []

    overfill = [e for e in events if isinstance(e, LegRepairFailedEvent)]
    assert len(overfill) == 1
    assert "overfill" in overfill[0].reason.lower()
    assert overfill[0].action_taken == 'capped'


def __test_overfill_after_partial_caps_at_entry_qty__():
    """0.4 partial amends to 0.4; follow-up 1.2 cumulative caps back at 1.0."""
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, pos = _mk_engine_with_sink(b, events)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.4, cumulative_filled=0.4,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 60_000)

    # Second event over-reports 1.2 cumulative — cap at 1.0.
    engine.on_order_event(_partial_entry_event(
        pine_id="L", fill_delta=0.8, cumulative_filled=1.2,
        order_qty=1.0, price=50_000.0,
    ))
    engine.sync(BAR_TS + 120_000)

    # Two amends: 1.0 → 0.4 (first partial), 0.4 → 1.0 (second, capped).
    assert len(b.modify_exit_calls) == 2
    assert b.modify_exit_calls[0][1].intent.qty == 0.4
    assert b.modify_exit_calls[1][1].intent.qty == 1.0
    assert engine.active_intents["TP\0L"].qty == 1.0
    # The second amend carries the overfill flag.
    overfill = [e for e in events if isinstance(e, LegRepairFailedEvent)]
    assert len(overfill) == 1
    assert overfill[0].action_taken == 'capped'


# === Natural close cleanup ===========================================
# When a TP/SL/TRAILING_STOP/CLOSE leg fully closes the position
# (BrokerPosition.size hits 0), the engine must drop the entry +
# matching exit intents from ``_active_intents`` AND clear Pine's
# ``entry_orders`` / ``exit_orders`` dicts. Pine's ``strategy.exit``
# is unconditional in most scripts; only the simulator gates it via
# open trades. Without this cleanup the next bar's ``sync()`` rebuilds
# the same exit intent from the still-present dict entry and dispatches
# a pointless ``modify_exit`` against a position that no longer exists
# on the broker — which on Capital.com fails because the entry row is
# gone.


def _closing_fill_event(side: str, qty: float, price: float, *,
                        pine_id: str, from_entry: str,
                        leg: LegType = LegType.STOP_LOSS,
                        xchg_id: str = "xchg-close") -> OrderEvent:
    exch = ExchangeOrder(
        id=xchg_id, symbol=SYMBOL, side=side,
        order_type=OrderType.MARKET, qty=qty, filled_qty=qty,
        remaining_qty=0.0, price=None, stop_price=None,
        average_fill_price=price, status=OrderStatus.FILLED,
        timestamp=0.0, fee=0.0, fee_currency="",
    )
    return OrderEvent(
        order=exch, event_type='filled', fill_price=price,
        fill_qty=qty, timestamp=0.0,
        pine_id=pine_id, from_entry=from_entry, leg_type=leg,
    )


def __test_natural_close_cleans_entry_and_exit_intents__():
    """SL fill to flat wipes the entry intent, exit intent, and matching Pine-side dict entries.

    SL fill that brings position size to 0 must wipe the entry
    intent, the exit intent, and the matching Pine-side dict entries —
    otherwise Pine re-emits a stale exit on the next bar and the engine
    fires a pointless ``modify_exit`` against a closed position.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders[("Bracket", "L")] = _exit_order(
        "L", -1.0, "Bracket", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)
    assert "L" in engine.active_intents
    assert "Bracket\0L" in engine.active_intents

    # Entry fills — position opens, intents stay in tracking.
    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="L", leg=LegType.ENTRY,
    ))
    engine._drain_events()
    assert pos.size == 1.0

    # SL fires — position closes; cleanup must run.
    engine.on_order_event(_closing_fill_event(
        "sell", 1.0, 45_000.0,
        pine_id="L", from_entry="L", leg=LegType.STOP_LOSS,
    ))
    engine._drain_events()

    assert pos.size == 0.0, "SL fill must reduce position to flat"
    assert "L" not in engine.active_intents, (
        "entry intent must be dropped after natural close"
    )
    assert "Bracket\0L" not in engine.active_intents, (
        "exit intent must be dropped after natural close"
    )
    assert "L" not in pos.entry_orders, (
        "Pine entry_orders[L] must be cleared so next bar does not "
        "re-emit a modify against the closed position"
    )
    assert ("Bracket", "L") not in pos.exit_orders, (
        "Pine exit_orders[(Bracket, L)] must be cleared so next bar "
        "does not re-emit a stale Bracket exit"
    )


def __test_natural_close_partial_fill_does_not_cleanup__():
    """A partial closing fill that does not reach flat keeps the entry/exit intents intact.

    A partial closing fill that does NOT bring size to 0 must keep
    the entry/exit intents intact so the remainder can still close.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 2.0, limit=50_000.0)
    pos.exit_orders[("Bracket", "L")] = _exit_order(
        "L", -2.0, "Bracket", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)

    engine.on_order_event(_fill_event(
        "buy", 2.0, 50_000.0, pine_id="L", leg=LegType.ENTRY,
    ))
    engine._drain_events()
    assert pos.size == 2.0

    # Partial SL fill — size goes 2 → 1, not 0.
    partial = _closing_fill_event(
        "sell", 1.0, 45_000.0,
        pine_id="L", from_entry="L", leg=LegType.STOP_LOSS,
    )
    partial.event_type = 'partial'
    engine.on_order_event(partial)
    engine._drain_events()

    assert pos.size == 1.0
    assert "L" in engine.active_intents, "entry intent must survive partial close"
    assert "Bracket\0L" in engine.active_intents, "exit intent must survive partial close"
    assert "L" in pos.entry_orders
    assert ("Bracket", "L") in pos.exit_orders


# === BracketAttachAfterFillRejectedError → defensive close ===


def _bracket_reject_exit_intent() -> ExitIntent:
    return ExitIntent(
        pine_id='Bracket',
        from_entry='Long',
        symbol=SYMBOL,
        side='sell',
        qty=1.0,
        tp_price=51_000.0,
        sl_price=49_000.0,
    )


def _bracket_reject_error(
        original_cause: Exception | None = None,
) -> BracketAttachAfterFillRejectedError:
    err = BracketAttachAfterFillRejectedError(
        "bracket attach reject",
        position_deal_id='deal-L',
        position_coid='coid-entry',
        symbol=SYMBOL,
        position_side='buy',
        qty=1.0,
        from_entry='Long',
    )
    if original_cause is not None:
        err.__cause__ = original_cause
    return err


def __test_bracket_reject_dispatches_defensive_close_and_skips_intent__():
    """Bracket attach reject dispatches a defensive close, skips the exit intent, and does not halt.

    The plugin raises :class:`BracketAttachAfterFillRejectedError`
    after a parent fill committed but the protective bracket attach was
    rejected. The sync engine must:

    1. Dispatch a market :class:`CloseIntent` with the OPPOSITE side
       (long parent → 'sell' close) for the same qty/symbol — defensive
       close to flatten the unprotected position.
    2. Surface the original exit intent as :class:`OrderSkippedByPlugin`
       so the caller drops it from ``_active_intents`` and lets the next
       bar re-evaluate from real state.
    3. NOT halt — no :class:`BrokerManualInterventionError`, no
       ``_record_halt`` write.
    """
    b = MockBroker()
    b.raise_on_next_exit = _bracket_reject_error()
    engine, _pos = _mk_engine(b)

    intent = _bracket_reject_exit_intent()
    with pytest.raises(OrderSkippedByPlugin) as exc:
        engine._dispatch_new(intent)

    assert exc.value.reason == "bracket_reject_defensive_close"
    assert exc.value.intent_key == intent.intent_key

    # Defensive close was dispatched: opposite side, same qty/symbol.
    assert len(b.close_calls) == 1
    close_env = b.close_calls[0]
    assert isinstance(close_env.intent, CloseIntent)
    assert close_env.intent.side == 'sell'  # long parent → close sells
    assert close_env.intent.qty == 1.0
    assert close_env.intent.symbol == SYMBOL
    assert close_env.intent.immediately is True

    # Did not halt — no manual-intervention record on the engine.
    assert engine.halted is False


def __test_bracket_reject_short_position_close_side_is_buy__():
    """Bracket-reject defensive close of a short parent uses a 'buy' market order.

    Symmetry guard: a short parent must be closed with a 'buy'
    market order. Easy to flip accidentally because the *exit* intent's
    side ('buy' for short SL/TP) and the *position* side ('sell') are
    inverses."""
    b = MockBroker()
    err = BracketAttachAfterFillRejectedError(
        "bracket attach reject",
        position_deal_id='deal-S',
        position_coid='coid-short-entry',
        symbol=SYMBOL,
        position_side='sell',  # short parent
        qty=2.5,
        from_entry='Short',
    )
    b.raise_on_next_exit = err
    engine, _pos = _mk_engine(b)

    intent = ExitIntent(
        pine_id='Bracket', from_entry='Short', symbol=SYMBOL,
        side='buy', qty=2.5, tp_price=49_000.0, sl_price=51_000.0,
    )
    with pytest.raises(OrderSkippedByPlugin):
        engine._dispatch_new(intent)

    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.side == 'buy'
    assert b.close_calls[0].intent.qty == 2.5


def __test_bracket_reject_defensive_close_timeout_does_not_halt__():
    """A parked (timed-out) defensive close does not escalate to a halt.

    Defensive close itself parks (timeout) — at worst the position
    stays open until the next reconcile. Don't escalate to halt."""
    b = MockBroker()
    b.raise_on_next_exit = _bracket_reject_error()

    # Wire up execute_close to time out (parked disposition).
    original_close = b.execute_close

    async def _timeout_close(envelope):
        raise OrderDispositionUnknownError(
            "close timeout", client_order_id='c-coid',
        )

    b.execute_close = _timeout_close  # type: ignore[method-assign]
    engine, _pos = _mk_engine(b)

    intent = _bracket_reject_exit_intent()
    with pytest.raises(OrderSkippedByPlugin):
        engine._dispatch_new(intent)

    assert engine.halted is False

    # Cleanup so other tests (if MockBroker was shared, which it isn't here)
    # don't trip — defensive belt-and-suspenders.
    b.execute_close = original_close  # type: ignore[method-assign]


def __test_bracket_reject_defensive_close_unexpected_failure_halts__():
    """An unexpected defensive-close failure escalates to manual intervention and records the halt.

    Defensive close fails with an unexpected exception (not park, not
    skip, not already a manual-intervention) — escalate to manual
    intervention and record the halt so the runner stops gracefully."""
    b = MockBroker()
    b.raise_on_next_exit = _bracket_reject_error()

    async def _broken_close(envelope):
        raise RuntimeError("close path is wedged")

    b.execute_close = _broken_close  # type: ignore[method-assign]
    engine, _pos = _mk_engine(b)

    intent = _bracket_reject_exit_intent()
    with pytest.raises(BrokerManualInterventionError) as exc:
        engine._dispatch_new(intent)

    assert "Defensive close after bracket attach reject failed" in str(exc.value)
    assert exc.value.intent_key == intent.intent_key
    assert exc.value.context['position_deal_id'] == 'deal-L'
    # Halt recorded so subsequent syncs return early via the halt flag.
    assert engine.halted is True


def __test_bracket_reject_defensive_close_stamps_natural_close_on_entry__(
        tmp_path,
):
    """A successful defensive close stamps ``extras['natural_close_at']`` on the parent entry row.

    After a successful defensive close, the parent entry row must
    be stamped with ``extras['natural_close_at']`` so the plugin-side
    reconciler skips missing-pending accounting.

    Without this stamp, the parent ``dealId`` disappears from the
    broker snapshot (we deliberately closed the position) BEFORE the
    close activity record arrives — the plugin's missing-pending
    grace tracker then raises :class:`UnexpectedCancelError` and
    halts the bot for a position we ourselves flattened.

    The row is NOT physically closed (``close_order`` would break
    ``find_by_ref`` lookups when the broker's close activity finally
    arrives) — only the breadcrumb extras field is set.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025",
                symbol=SYMBOL,
                timeframe="60",
                account_id="testbroker-demo",
                label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        # Seed the parent entry row in 'confirmed' state — what
        # ``execute_entry`` would have persisted after Capital's fill
        # confirm, before the bracket attach attempt.
        ctx.upsert_order(
            'coid-entry',
            symbol=SYMBOL, side='buy', qty=1.0, state='confirmed',
            exchange_order_id='deal-L',
            pine_entry_id='Long',
            filled_qty=1.0,
            extras={'kind': 'position'},
        )

        b = MockBroker()
        b.raise_on_next_exit = _bracket_reject_error()
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos,
            symbol=SYMBOL,
            run_tag=RUN_TAG,
            mintick=1.0,
            store_ctx=ctx,
        )

        intent = _bracket_reject_exit_intent()
        with pytest.raises(OrderSkippedByPlugin):
            engine._dispatch_new(intent)

        # Defensive close was dispatched (sanity guard).
        assert len(b.close_calls) == 1

        # Parent entry row now carries the natural-close breadcrumb.
        row = ctx.get_order('coid-entry')
        assert row is not None
        assert (row.extras or {}).get('natural_close_at') is not None

        # Row is NOT physically closed — find_by_ref lookups for the
        # eventual close activity must still locate it.
        assert row.closed_ts_ms is None


def __test_bracket_reject_defensive_close_park_does_not_stamp_natural_close__(
        tmp_path,
):
    """A parked (timed-out) defensive close does NOT stamp ``natural_close_at`` on the parent.

    When the defensive close itself parks (timeout), the position
    may still be open — DO NOT stamp ``natural_close_at`` because
    that would mask a legitimately stuck position from the reconciler.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025",
                symbol=SYMBOL,
                timeframe="60",
                account_id="testbroker-demo",
                label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        ctx.upsert_order(
            'coid-entry',
            symbol=SYMBOL, side='buy', qty=1.0, state='confirmed',
            exchange_order_id='deal-L',
            pine_entry_id='Long',
            filled_qty=1.0,
            extras={'kind': 'position'},
        )

        b = MockBroker()
        b.raise_on_next_exit = _bracket_reject_error()

        async def _timeout_close(envelope):
            raise OrderDispositionUnknownError(
                "close timeout", client_order_id='c-coid',
            )

        b.execute_close = _timeout_close  # type: ignore[method-assign]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos,
            symbol=SYMBOL,
            run_tag=RUN_TAG,
            mintick=1.0,
            store_ctx=ctx,
        )

        intent = _bracket_reject_exit_intent()
        with pytest.raises(OrderSkippedByPlugin):
            engine._dispatch_new(intent)

        row = ctx.get_order('coid-entry')
        assert row is not None
        assert (row.extras or {}).get('natural_close_at') is None


def __test_bracket_reject_defensive_close_pending_state_set_before_dispatch__(
        tmp_path,
):
    """:class:`PendingDefensiveClose` is armed on the parent entry id before the close dispatch.

    The engine arms a :class:`PendingDefensiveClose` marker on the
    parent entry id BEFORE the synthetic close dispatches.

    This is the load-bearing invariant of the defensive-close pending
    lifecycle: the close FILL may race in synchronously with the
    dispatch return, so the marker has to exist by the time the route
    layer asks "is this FILL ours?". The marker survives in
    ``engine.pending_defensive_close`` and is mirrored to the parent
    entry row's ``extras['defensive_close_pending']``.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025",
                symbol=SYMBOL,
                timeframe="60",
                account_id="testbroker-demo",
                label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        ctx.upsert_order(
            'coid-entry',
            symbol=SYMBOL, side='buy', qty=1.0, state='confirmed',
            exchange_order_id='deal-L',
            pine_entry_id='Long',
            filled_qty=1.0,
            extras={'kind': 'position'},
        )

        b = MockBroker()
        b.raise_on_next_exit = _bracket_reject_error()
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos,
            symbol=SYMBOL,
            run_tag=RUN_TAG,
            mintick=1.0,
            store_ctx=ctx,
        )

        pos.entry_orders["Long"] = _entry_order("Long", 1.0, limit=50_000.0)
        pos.exit_orders[("Bracket", "Long")] = _exit_order(
            "Long", -1.0, "Bracket", limit=51_000.0, stop=49_000.0,
        )

        engine.sync(BAR_TS)

        # Marker exists in-memory under the parent entry id.
        marker = engine.pending_defensive_close.get("Long")
        assert marker is not None
        assert marker.entry_id == "Long"
        assert marker.close_intent_key == "__pyne_defensive_close__coid-entry"
        # close_order_ref captured from the successful dispatch (mock returns xchg-N).
        assert marker.close_order_ref == "xchg-2"
        assert marker.reject_context.position_coid == "coid-entry"
        assert marker.reject_context.symbol == SYMBOL

        # Mirrored to the parent entry row's extras for cross-restart replay.
        row = ctx.get_order('coid-entry')
        assert row is not None
        assert 'defensive_close_pending' in row.extras
        persisted = row.extras['defensive_close_pending']
        assert persisted['entry_id'] == 'Long'
        assert persisted['close_intent_key'] == "__pyne_defensive_close__coid-entry"


def __test_bracket_reject_defensive_close_cleanup_deferred_to_fill__(
        tmp_path,
):
    """Parent intent and Pine order state survive the defensive close; cleanup defers to fill.

    ``_active_intents`` + Pine ``entry_orders`` / ``exit_orders``
    state for the parent stay PUT immediately after the defensive
    close dispatches — cleanup is deferred to the FILL handler so
    :meth:`reconcile` cannot misclassify the flat broker snapshot as
    an external flatten while the close is in flight.

    A future change that re-introduces dispatch-time cleanup would
    silently re-open the same-bar duplicate-entry race the lifecycle
    redesign closed.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025",
                symbol=SYMBOL,
                timeframe="60",
                account_id="testbroker-demo",
                label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        ctx.upsert_order(
            'coid-entry',
            symbol=SYMBOL, side='buy', qty=1.0, state='confirmed',
            exchange_order_id='deal-L',
            pine_entry_id='Long',
            filled_qty=1.0,
            extras={'kind': 'position'},
        )

        b = MockBroker()
        b.raise_on_next_exit = _bracket_reject_error()
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos,
            symbol=SYMBOL,
            run_tag=RUN_TAG,
            mintick=1.0,
            store_ctx=ctx,
        )

        pos.entry_orders["Long"] = _entry_order("Long", 1.0, limit=50_000.0)
        pos.exit_orders[("Bracket", "Long")] = _exit_order(
            "Long", -1.0, "Bracket", limit=51_000.0, stop=49_000.0,
        )

        engine.sync(BAR_TS)

        # Defensive close dispatched (sanity guard).
        assert len(b.close_calls) == 1

        # Cleanup deferred — state intact until the close FILL arrives.
        assert "Long" in pos.entry_orders
        assert ("Bracket", "Long") in pos.exit_orders
        assert "Long" in engine.active_intents
        # The sibling exit intent did get dropped (its dispatch raised
        # the reject — the engine surfaces OrderSkippedByPlugin which
        # the diff loop translates into "do not register").
        assert "Bracket\0Long" not in engine.active_intents


def __test_bracket_reject_skips_sibling_exit_for_same_from_entry_in_diff_loop__(
        tmp_path,
):
    """Bracket-reject recovery skips sibling exits for the same ``from_entry`` in the diff loop.

    When :meth:`_diff_and_dispatch` iterates a precomputed ``new_map``
    and the first bracket exit for an entry triggers the
    :class:`BracketAttachAfterFillRejectedError` recovery, sibling exits
    that reference the same ``from_entry`` later in the same loop MUST
    NOT be dispatched.

    Without the guard, ``_cleanup_position_tracking`` removes the sibling
    from ``_active_intents`` mid-loop and the diff loop then treats it as
    brand-new — dispatching another bracket against a position that was
    just defensively closed. The new
    ``_defensively_closed_entries_this_sync`` set short-circuits the
    sibling so only the first (failing) exit reaches the plugin and the
    runner converges next bar.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025",
                symbol=SYMBOL,
                timeframe="60",
                account_id="testbroker-demo",
                label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        ctx.upsert_order(
            'coid-entry',
            symbol=SYMBOL, side='buy', qty=1.0, state='confirmed',
            exchange_order_id='deal-L',
            pine_entry_id='Long',
            filled_qty=1.0,
            extras={'kind': 'position'},
        )

        b = MockBroker()
        # Every execute_exit hits the bracket-reject path (sibling exits
        # would otherwise look like a fresh attach attempt against the
        # just-flattened position and re-trigger the recovery).
        async def _always_bracket_reject(envelope):
            b.exit_calls.append(envelope)
            raise _bracket_reject_error()

        b.execute_exit = _always_bracket_reject  # type: ignore[method-assign]

        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos,
            symbol=SYMBOL,
            run_tag=RUN_TAG,
            mintick=1.0,
            store_ctx=ctx,
        )

        # Two bracket exits for the same Pine entry — Pine allows multiple
        # ``strategy.exit`` calls per entry (e.g. partial TP at different
        # levels). The diff loop iterates them in insertion order.
        pos.entry_orders['Long'] = _entry_order('Long', 1.0, limit=50_000.0)
        pos.exit_orders[('BracketA', 'Long')] = _exit_order(
            'Long', -1.0, 'BracketA', limit=51_000.0, stop=49_000.0,
        )
        pos.exit_orders[('BracketB', 'Long')] = _exit_order(
            'Long', -1.0, 'BracketB', limit=52_000.0, stop=48_000.0,
        )

        engine.sync(BAR_TS)

        # Exactly ONE exit dispatch reached the plugin: the second sibling
        # was short-circuited by the defensive-close-this-sync guard.
        assert len(b.exit_calls) == 1
        # And exactly ONE defensive close was emitted — without the guard
        # the second exit dispatch would re-enter the recovery path and
        # emit a duplicate (or escalate to halt if the plugin path raised
        # a plain ``ExchangeOrderRejectedError`` instead).
        assert len(b.close_calls) == 1
        # The parent entry intent intentionally STAYS in ``_active_intents``
        # until the close FILL — the defensive-close pending lifecycle
        # uses it as the guard that keeps :meth:`reconcile` from flipping
        # state out from under us while the close is in flight.
        assert 'Long' in engine.active_intents
        # Neither sibling bracket made it into ``_active_intents``:
        # ``BracketA`` was raised away by its own dispatch, ``BracketB``
        # was short-circuited by the defensively-closed-this-sync guard.
        assert 'BracketA\0Long' not in engine.active_intents
        assert 'BracketB\0Long' not in engine.active_intents
        # No halt — defensive recovery completed and absorbed both siblings.
        assert engine.halted is False


def __test_bracket_reject_marker_survives_apply_async_events_to_sync__(tmp_path):
    """The defensively-closed-entries guard survives the apply_async_events -> script -> sync cycle.

    The ``_defensively_closed_entries_this_sync`` guard must remain
    valid across the apply_async_events -> script -> sync cycle.

    Scenario: a tick-deferred bracket exit waits for the parent entry
    fill. Between bars an async entry-fill event arrives. The runner
    calls :meth:`apply_async_events` BEFORE running the user script;
    that drain resolves the deferred exit, dispatches it, and the
    plugin raises :class:`BracketAttachAfterFillRejectedError` —
    populating ``_defensively_closed_entries_this_sync`` with the
    parent ``from_entry``. The user script then unconditionally
    re-emits ``strategy.exit('TP', from_entry='Long')``, re-populating
    ``position.exit_orders``. Finally :meth:`sync` runs and must
    short-circuit the recreated exit so it is NOT dispatched against
    the just defensively-closed position.

    Without the fix the marker is cleared at the top of :meth:`sync`,
    the diff loop treats the re-emitted exit as brand-new, and
    ``execute_exit`` is called against a flattened position (live
    behaviour: ``no confirmed entry row`` / duplicate defensive close).
    """
    b = MockBroker()
    # First exit dispatch (from the apply_async_events drain) hits the
    # bracket-reject path. Any subsequent execute_exit must NOT be
    # called — the guard must short-circuit it.
    async def _reject_first_exit_only(envelope):
        b.exit_calls.append(envelope)
        if len(b.exit_calls) == 1:
            raise _bracket_reject_error()

    b.execute_exit = _reject_first_exit_only  # type: ignore[method-assign]

    engine, pos = _mk_engine(b, mintick=1.0)
    # Deferred bracket exit pending parent fill.
    pos.exit_orders[('TP', 'Long')] = _exit_order(
        'Long', -1.0, 'TP', profit_ticks=100.0, loss_ticks=50.0,
    )
    engine.sync(BAR_TS)
    assert 'TP\0Long' in engine.deferred_exits

    # Async entry fill arrives between bars. Runner drains it via
    # apply_async_events BEFORE running the script. The drain resolves
    # the deferred exit, dispatches it, hits the bracket-reject path,
    # and populates _defensively_closed_entries_this_sync['Long'].
    engine.on_order_event(_fill_event(
        'buy', qty=1.0, price=50_000.0, pine_id='Long', leg=LegType.ENTRY,
    ))
    engine.apply_async_events()
    assert 'Long' in engine._defensively_closed_entries_this_sync
    assert len(b.exit_calls) == 1
    assert len(b.close_calls) == 1

    # Simulate the user script re-emitting strategy.exit() in the same
    # bar (Pine's strategy.exit is unconditional in most scripts), which
    # repopulates position.exit_orders after the cleanup wiped it.
    pos.exit_orders[('TP', 'Long')] = _exit_order(
        'Long', -1.0, 'TP', limit=50_100.0, stop=49_950.0,
    )

    engine.sync(BAR_TS + 1)

    # Guard held across the apply_async_events -> sync boundary: the
    # recreated exit was short-circuited, no second execute_exit, no
    # second defensive close.
    assert len(b.exit_calls) == 1
    assert len(b.close_calls) == 1
    # And cleared at end of sync — fresh bar starts clean.
    assert 'Long' not in engine._defensively_closed_entries_this_sync
    assert engine.halted is False


def _bracket_reject_scenario(tmp_path, mock_broker=None):
    """Set up an engine that has just dispatched a defensive close —
    pending marker is armed, parent state intact, and a defensive close
    FILL event has not yet arrived. Used as a fixture by the FILL-handler
    regression tests below.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity

    store = BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker")
    ctx = store.open_run(
        RunIdentity(
            strategy_id="t025",
            symbol=SYMBOL,
            timeframe="60",
            account_id="testbroker-demo",
            label=None,
        ),
        script_source="src",
        script_path="t025.py",
    )
    ctx.upsert_order(
        'coid-entry',
        symbol=SYMBOL, side='buy', qty=1.0, state='confirmed',
        exchange_order_id='deal-L',
        pine_entry_id='Long',
        filled_qty=1.0,
        extras={'kind': 'position'},
    )

    b = mock_broker if mock_broker is not None else MockBroker()
    b.raise_on_next_exit = _bracket_reject_error()
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=b,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        mintick=1.0,
        store_ctx=ctx,
    )

    pos.entry_orders["Long"] = _entry_order("Long", 1.0, limit=50_000.0)
    pos.exit_orders[("Bracket", "Long")] = _exit_order(
        "Long", -1.0, "Bracket", limit=51_000.0, stop=49_000.0,
    )

    engine.sync(BAR_TS)
    return store, ctx, engine, pos, b


def __test_defensive_close_fill_runs_deferred_cleanup__(tmp_path):
    """A defensive close FILL via the WS path runs the deferred parent-entry cleanup.

    When the defensive close FILL arrives via the WS path (pine_id
    matches the synthetic close_intent_key), the engine runs the
    deferred parent-entry cleanup that the dispatch-time path now
    skips."""
    store, ctx, engine, pos, b = _bracket_reject_scenario(tmp_path)
    try:
        # Defensive close FILL arrives — synthetic pine_id carries the
        # close_intent_key.
        engine.on_order_event(_fill_event(
            'sell', qty=1.0, price=50_000.0,
            pine_id="__pyne_defensive_close__coid-entry",
            leg=LegType.CLOSE,
            xchg_id='xchg-2',
        ))
        engine.apply_async_events()

        # Parent entry + bracket exit + Pine order book all cleared
        # NOW (FILL-time), not at dispatch time.
        assert "Long" not in engine.active_intents
        assert "Long" not in pos.entry_orders
        assert ("Bracket", "Long") not in pos.exit_orders
        # Marker dropped both in-memory and from extras.
        assert "Long" not in engine.pending_defensive_close
        row = ctx.get_order('coid-entry')
        assert row is not None
        assert 'defensive_close_pending' not in row.extras
    finally:
        store.close()


def __test_defensive_close_fill_matched_by_order_ref__(tmp_path):
    """Polled FILL without ``pine_id`` routes to defensive-close cleanup via ``close_order_ref``.

    A polled-orders FILL event without ``pine_id`` still routes to
    the defensive-close cleanup via ``close_order_ref`` match."""
    store, ctx, engine, pos, b = _bracket_reject_scenario(tmp_path)
    try:
        # FILL with pine_id=None, but order.id matches the captured
        # close_order_ref (xchg-2 from the mock's defensive close
        # dispatch).
        exch = ExchangeOrder(
            id='xchg-2', symbol=SYMBOL, side='sell',
            order_type=OrderType.MARKET, qty=1.0, filled_qty=1.0,
            remaining_qty=0.0, price=None, stop_price=None,
            average_fill_price=50_000.0, status=OrderStatus.FILLED,
            timestamp=0.0, fee=0.0, fee_currency="",
        )
        engine.on_order_event(OrderEvent(
            order=exch, event_type='filled', fill_price=50_000.0,
            fill_qty=1.0, timestamp=0.0, pine_id=None, leg_type=LegType.CLOSE,
        ))
        engine.apply_async_events()

        assert "Long" not in engine.active_intents
        assert "Long" not in engine.pending_defensive_close
    finally:
        store.close()


def __test_defensive_close_fill_writes_audit_event__(tmp_path):
    """A defensive close FILL writes a ``'defensive_close_filled'`` audit event to the events table.

    A ``'defensive_close_filled'`` audit event lands in the events
    table on FILL — startup replay uses it to detect that a marker has
    already settled after a process restart."""
    store, ctx, engine, pos, b = _bracket_reject_scenario(tmp_path)
    try:
        engine.on_order_event(_fill_event(
            'sell', qty=1.0, price=50_000.0,
            pine_id="__pyne_defensive_close__coid-entry",
            leg=LegType.CLOSE,
            xchg_id='xchg-2',
        ))
        engine.apply_async_events()

        rows = list(store._conn.execute(
            "SELECT kind, intent_key, client_order_id FROM events "
            "WHERE kind = 'defensive_close_filled'"
        ))
        assert len(rows) == 1
        kind, intent_key, client_order_id = rows[0]
        assert kind == 'defensive_close_filled'
        assert intent_key == "__pyne_defensive_close__coid-entry"
        assert client_order_id == 'coid-entry'
    finally:
        store.close()


def __test_defensive_close_fill_is_idempotent__(tmp_path):
    """A second defensive close FILL finds no marker and is a no-op (idempotent re-delivery).

    A second FILL event for the same close finds no marker and is a
    no-op — covers re-delivery scenarios (WS replay, manual FILL
    injection in tests, polled-orders cycle racing the WS path)."""
    store, ctx, engine, pos, b = _bracket_reject_scenario(tmp_path)
    try:
        fill = _fill_event(
            'sell', qty=1.0, price=50_000.0,
            pine_id="__pyne_defensive_close__coid-entry",
            leg=LegType.CLOSE,
            xchg_id='xchg-2',
        )
        engine.on_order_event(fill)
        engine.apply_async_events()
        assert "Long" not in engine.pending_defensive_close

        # Replay the same FILL — marker is already gone, helper is a no-op.
        engine.on_order_event(fill)
        engine.apply_async_events()

        rows = list(store._conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind = 'defensive_close_filled'"
        ))
        # Exactly one audit event (the second FILL did not write a duplicate).
        assert rows[0][0] == 1
    finally:
        store.close()


def _seed_pending_marker_in_store(
        ctx, *, position_coid: str, entry_id: str,
        close_intent_key: str, close_order_ref: str | None,
        pending_since: float,
        residual_refs: list[str] | None = None,
) -> None:
    """Write a fully-formed defensive_close_pending payload onto the
    parent entry row's extras column — used to simulate a marker that
    survived from a prior process instance."""
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )
    marker = PendingDefensiveClose(
        entry_id=entry_id,
        close_intent_key=close_intent_key,
        close_order_ref=close_order_ref,
        pending_since=pending_since,
        reject_context=BracketAttachRejectContext(
            intent_key='Bracket\0' + entry_id,
            position_coid=position_coid,
            position_side='buy',
            qty=1.0,
            symbol=SYMBOL,
        ),
    )
    row = ctx.get_order(position_coid)
    extras = dict(row.extras or {}) if row is not None else {}
    extras['defensive_close_pending'] = marker.to_extras_dict()
    ctx.upsert_order(position_coid, extras=extras)


def __test_startup_replay_settled_drops_marker__(tmp_path):
    """Startup replay drops a marker without re-arming when a matching settled audit event exists.

    When a 'defensive_close_filled' audit event exists for the
    marker's close_intent_key, startup replay drops the marker without
    re-arming — the FILL settled in the prior instance, current
    instance has nothing to wait on."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity
    import time as _time

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025", symbol=SYMBOL, timeframe="60",
                account_id="testbroker-demo", label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        ctx.upsert_order(
            'coid-entry', symbol=SYMBOL, side='buy', qty=1.0,
            state='confirmed', pine_entry_id='Long', filled_qty=1.0,
            extras={'kind': 'position'},
        )
        _seed_pending_marker_in_store(
            ctx, position_coid='coid-entry', entry_id='Long',
            close_intent_key='__pyne_defensive_close__coid-entry',
            close_order_ref='xchg-2',
            pending_since=_time.time(),
        )
        # Prior-instance audit event proving the FILL already settled.
        ctx.log_event(
            kind='defensive_close_filled',
            intent_key='__pyne_defensive_close__coid-entry',
            client_order_id='coid-entry',
            payload={'entry_id': 'Long'},
        )

        b = MockBroker()
        engine = OrderSyncEngine(
            broker=b, position=BrokerPosition(),  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0,
            store_ctx=ctx,
        )
        engine._replay_pending_defensive_closes()

        # Marker dropped both from memory and from extras.
        assert 'Long' not in engine.pending_defensive_close
        row = ctx.get_order('coid-entry')
        assert 'defensive_close_pending' not in row.extras


def __test_startup_replay_unsettled_rearms_marker_and_runs_residual_cancel__(
        tmp_path,
):
    """Startup replay re-arms an unsettled marker in-memory and re-runs the residual cancel loop.

    A marker without a matching audit event is re-armed in-memory
    AND the residual cancel loop is re-run via the plugin idempotency
    contract — covers crashes between dispatch and FILL."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity
    import time as _time

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025", symbol=SYMBOL, timeframe="60",
                account_id="testbroker-demo", label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        ctx.upsert_order(
            'coid-entry', symbol=SYMBOL, side='buy', qty=1.0,
            state='confirmed', pine_entry_id='Long', filled_qty=1.0,
            extras={'kind': 'position'},
        )
        pending_since = _time.time() - 5.0  # fresh enough to skip the timeout halt
        _seed_pending_marker_in_store(
            ctx, position_coid='coid-entry', entry_id='Long',
            close_intent_key='__pyne_defensive_close__coid-entry',
            close_order_ref='xchg-2',
            pending_since=pending_since,
        )

        b = MockBroker()
        b.residual_refs_for_reject = ['residual-tp', 'residual-sl']
        engine = OrderSyncEngine(
            broker=b, position=BrokerPosition(),  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0,
            store_ctx=ctx,
        )
        engine._replay_pending_defensive_closes()

        # Marker re-armed in-memory with the same fields.
        marker = engine.pending_defensive_close.get('Long')
        assert marker is not None
        assert marker.close_intent_key == '__pyne_defensive_close__coid-entry'
        assert marker.pending_since == pending_since

        # Residual cancel loop replayed — both refs cancelled.
        assert b.cancel_broker_order_calls == ['residual-tp', 'residual-sl']

        # Extras marker still present (replay does not clear unsettled markers).
        row = ctx.get_order('coid-entry')
        assert 'defensive_close_pending' in row.extras


def __test_startup_replay_idempotent_second_invocation__(tmp_path):
    """A second replay re-runs the residual cancel but does not double-register the marker.

    A second invocation on the same state runs the residual cancel
    again (idempotent by plugin contract) but does not double-register
    the marker — supports the runner calling replay twice during
    startup quirks (e.g. a manual mid-startup pause)."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity
    import time as _time

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025", symbol=SYMBOL, timeframe="60",
                account_id="testbroker-demo", label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        ctx.upsert_order(
            'coid-entry', symbol=SYMBOL, side='buy', qty=1.0,
            state='confirmed', pine_entry_id='Long', filled_qty=1.0,
            extras={'kind': 'position'},
        )
        _seed_pending_marker_in_store(
            ctx, position_coid='coid-entry', entry_id='Long',
            close_intent_key='__pyne_defensive_close__coid-entry',
            close_order_ref='xchg-2',
            pending_since=_time.time() - 5.0,
        )

        b = MockBroker()
        b.residual_refs_for_reject = ['residual-tp']
        engine = OrderSyncEngine(
            broker=b, position=BrokerPosition(),  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0,
            store_ctx=ctx,
        )
        engine._replay_pending_defensive_closes()
        engine._replay_pending_defensive_closes()

        assert len(engine.pending_defensive_close) == 1
        # Residual cancelled twice — plugin contract guarantees this is safe.
        assert b.cancel_broker_order_calls == ['residual-tp', 'residual-tp']


def __test_startup_replay_drops_malformed_payload__(tmp_path):
    """Startup replay logs and drops a malformed extras payload instead of crashing.

    A malformed extras payload (manual DB tampering, schema-skew
    after a bad migration) is logged + removed; the engine keeps going
    instead of crashing on a deserialize error."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025", symbol=SYMBOL, timeframe="60",
                account_id="testbroker-demo", label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        ctx.upsert_order(
            'coid-entry', symbol=SYMBOL, side='buy', qty=1.0,
            state='confirmed', pine_entry_id='Long', filled_qty=1.0,
            extras={
                'kind': 'position',
                'defensive_close_pending': {'garbage': True},  # malformed
            },
        )

        b = MockBroker()
        engine = OrderSyncEngine(
            broker=b, position=BrokerPosition(),  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0,
            store_ctx=ctx,
        )
        engine._replay_pending_defensive_closes()

        assert engine.pending_defensive_close == {}
        row = ctx.get_order('coid-entry')
        assert 'defensive_close_pending' not in row.extras


def __test_startup_replay_parked_unresolved_defers_residual_cancel__(tmp_path):
    """A parked-unresolved marker defers the residual cancel to the runtime parked-recovery path.

    Parked-unresolved markers (close_order_ref=None, no fill, no audit)
    must DEFER the residual cancel to the runtime parked-recovery path.

    Cancelling residual TP/SL/partial-remainder orders during replay —
    BEFORE :meth:`_verify_pending_dispatches` confirms the parked
    defensive close actually landed on the exchange — would create an
    unprotected-position window across restart. The dispatch-time path
    explicitly gates residual cancel on ``dispatch_succeeded == True``;
    replay must mirror that gate."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity
    import time as _time

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025", symbol=SYMBOL, timeframe="60",
                account_id="testbroker-demo", label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        ctx.upsert_order(
            'coid-entry', symbol=SYMBOL, side='buy', qty=1.0,
            state='confirmed', pine_entry_id='Long', filled_qty=1.0,
            extras={'kind': 'position'},
        )
        _seed_pending_marker_in_store(
            ctx, position_coid='coid-entry', entry_id='Long',
            close_intent_key='__pyne_defensive_close__coid-entry',
            close_order_ref=None,  # parked-unresolved
            pending_since=_time.time() - 5.0,
        )

        b = MockBroker()
        b.residual_refs_for_reject = ['residual-tp', 'residual-sl']
        engine = OrderSyncEngine(
            broker=b, position=BrokerPosition(),  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0,
            store_ctx=ctx,
        )
        engine._replay_pending_defensive_closes()

        # Marker re-armed in memory.
        marker = engine.pending_defensive_close.get('Long')
        assert marker is not None
        assert marker.close_order_ref is None
        # Residual cancel DEFERRED — no cancel calls during replay.
        assert b.cancel_broker_order_calls == []
        # Persisted marker untouched (no stamp of residual_cleanup_pending).
        row = ctx.get_order('coid-entry')
        payload = row.extras['defensive_close_pending']
        assert payload.get('residual_cleanup_pending') in (False, None)


def __test_startup_replay_parked_with_cleanup_pending_runs_residual_cancel__(
        tmp_path,
):
    """A parked marker with ``residual_cleanup_pending=True`` runs the residual cancel on replay.

    A parked marker stamped ``residual_cleanup_pending=True`` by a
    prior instance still runs the residual cancel on replay — the prior
    instance already confirmed cleanup was due (the flag is only stamped
    AFTER a known dispatch / recovery), so the replay must finish the
    retry instead of stalling until the FILL lands."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )
    import time as _time

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025", symbol=SYMBOL, timeframe="60",
                account_id="testbroker-demo", label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        ctx.upsert_order(
            'coid-entry', symbol=SYMBOL, side='buy', qty=1.0,
            state='confirmed', pine_entry_id='Long', filled_qty=1.0,
            extras={'kind': 'position'},
        )
        # Manually construct a marker with residual_cleanup_pending=True —
        # the helper does not expose the field.
        marker = PendingDefensiveClose(
            entry_id='Long',
            close_intent_key='__pyne_defensive_close__coid-entry',
            close_order_ref=None,
            pending_since=_time.time() - 5.0,
            reject_context=BracketAttachRejectContext(
                intent_key='Bracket\0Long',
                position_coid='coid-entry',
                position_side='buy',
                qty=1.0,
                symbol=SYMBOL,
            ),
            residual_cleanup_pending=True,
        )
        row = ctx.get_order('coid-entry')
        extras = dict(row.extras or {})
        extras['defensive_close_pending'] = marker.to_extras_dict()
        ctx.upsert_order('coid-entry', extras=extras)

        b = MockBroker()
        b.residual_refs_for_reject = ['residual-tp']
        engine = OrderSyncEngine(
            broker=b, position=BrokerPosition(),  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0,
            store_ctx=ctx,
        )
        engine._replay_pending_defensive_closes()

        # Residual cancel executed — the prior-instance flag overrides
        # the parked-unresolved deferral.
        assert b.cancel_broker_order_calls == ['residual-tp']


def __test_refresh_anchors_after_orphan_retire_drops_stale_envelope__(tmp_path):
    """Engine in-memory anchor cache must be refreshable after retire.

    Reproduces the live-trade crash where ``_retire_startup_orphans``
    deletes a stale ``envelopes`` row via ``record_complete`` AFTER the
    engine has already loaded the anchor into
    ``_persisted_envelope_anchors`` in ``__init__``. Without
    ``refresh_anchors_from_store`` the next ``_build_envelope`` would
    pop the stale ``bar_ts_ms`` and emit a ``client_order_id`` that
    collides with the just-retired (and closed_ts_ms-stamped) order
    row — the row stays invisible to ``iter_live_orders`` and the
    next ``execute_exit`` raises ``no confirmed entry row``.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.run_identity import RunIdentity
    from pynecore.core.broker.models import EntryIntent

    stale_bar_ts = 1_700_000_000_000  # represents an earlier-run anchor
    fresh_bar_ts = 1_700_000_060_000  # the bar the new sync is processing

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            RunIdentity(
                strategy_id="t025",
                symbol=SYMBOL,
                timeframe="60",
                account_id="testbroker-demo",
                label=None,
            ),
            script_source="src",
            script_path="t025.py",
        )
        ctx.record_envelope(key='L', bar_ts_ms=stale_bar_ts, retry_seq=0)

        b = MockBroker()
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos,
            symbol=SYMBOL,
            run_tag=RUN_TAG,
            mintick=1.0,
            store_ctx=ctx,
        )
        assert 'L' in engine._persisted_envelope_anchors
        assert engine._persisted_envelope_anchors['L'].bar_ts_ms == stale_bar_ts

        # Simulate the plugin's startup orphan retire: SQLite envelope row
        # gone, but the engine's in-memory cache is still stale.
        ctx.record_complete('L')
        assert 'L' in engine._persisted_envelope_anchors

        # The fix: refresh re-reads from the store.
        engine.refresh_anchors_from_store()
        assert 'L' not in engine._persisted_envelope_anchors

        # Sanity: a subsequent dispatch builds the envelope from
        # ``_current_bar_ts_ms`` (set by ``sync``) — not the stale anchor.
        pos.entry_orders['L'] = _entry_order('L', 1.0, limit=50_000.0)
        engine.sync(fresh_bar_ts)
        assert len(b.entry_calls) == 1
        assert b.entry_calls[0].bar_ts_ms == fresh_bar_ts


# === §2.6.7 native fail-safe dispatcher drive ===
#
# These pin the contract that ``drive_native_failsafe`` is the SINGLE owner
# of the PUT outcome: it records a put-success on the dispatcher's normal
# return and a put-failure on any exception, so the plugin dispatcher stays
# a pure PUT-or-raise actuator and the retry budget cannot be double-counted.

def __test_drive_native_failsafe_dispatches_and_records_success__():
    """``drive_native_failsafe`` dispatches the worst-SL once and records the PUT success."""
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0)
    snap = mgr.recompute_worst_sl(
        parent_entry_dispatch_ref=ref, active_sl_levels=[95.0, 90.0],
        now_ms=1000.0,
    )
    assert snap is not None and snap.stop_level == 90.0
    received = []
    engine.set_native_bracket_dispatcher(received.append)

    engine.drive_native_failsafe(now_ms=1000.0)

    # Dispatched exactly once with the worst-SL snapshot.
    assert len(received) == 1
    assert received[0].stop_level == 90.0
    assert received[0].generation == snap.generation
    # The else-branch recorded success: snapshot dropped + pending_put cleared
    # (NOT left in-flight, which is what a missing success-record would leave).
    assert mgr.pending_dispatch() == []
    assert mgr.get_state(ref).pending_put is False
    # A second drive does not re-dispatch — nothing is pending.
    engine.drive_native_failsafe(now_ms=1000.0)
    assert len(received) == 1


def __test_drive_native_failsafe_records_single_failure_per_dispatch__():
    """A raising dispatcher records exactly one PUT failure per drive, degrading after 3."""
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0)
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[90.0], now_ms=1000.0)
    calls = []

    def _raising_dispatcher(snapshot):
        calls.append(snapshot)
        raise RuntimeError("PUT failed")

    engine.set_native_bracket_dispatcher(_raising_dispatcher)

    # Default retry budget is 3 and exactly ONE failure is recorded per drive
    # (the engine wrapper is the sole failure owner; the dispatcher never
    # records), so it takes 3 drives to exhaust the budget and degrade — a
    # double-record would degrade after 2.
    engine.drive_native_failsafe(now_ms=1000.0)
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY
    engine.drive_native_failsafe(now_ms=1000.0)
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY
    engine.drive_native_failsafe(now_ms=1000.0)
    assert mgr.get_state(ref).health is FailsafeHealth.DEGRADING
    # Re-dispatched on each drive (the failure path re-queues the snapshot).
    assert len(calls) == 3


def __test_drive_native_failsafe_dispatcher_manual_intervention_halts__():
    """A dispatcher ``BrokerManualInterventionError`` records the halt and re-raises, not degrades.

    A dispatcher raising ``BrokerManualInterventionError`` is a terminal
    halt, not a retryable PUT failure: the drive must record the halt and
    re-raise instead of degrading the budget and letting the strategy
    continue on an unsafe broker state."""
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0)
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[90.0], now_ms=1000.0)

    def _halting_dispatcher(_snapshot):
        raise BrokerManualInterventionError(
            "cannot resolve parent dealId", intent_key=ref,
        )

    engine.set_native_bracket_dispatcher(_halting_dispatcher)

    with pytest.raises(BrokerManualInterventionError):
        engine.drive_native_failsafe(now_ms=1000.0)

    # Halt latched (so the engine stops dispatching) and the fail-safe was
    # NOT degraded as if a retryable PUT failure had occurred.
    assert engine.halted is True
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY


# === §2.6.7 native fail-safe observed recovery feed ===
#
# STEP 4: the reconcile-driven feed that ``record_native_bracket_observed``
# routes into. A successful PUT clears ``pending_put`` but leaves the state
# DEGRADING; only an observed snapshot matching the desired worst-SL flips it
# back to HEALTHY. Without this feed the stale-window timer would escalate
# DEGRADING -> DEGRADED and block new entries / brackets until a manual reset.

def __test_record_native_bracket_observed_recovers_degrading_to_healthy__():
    """An observed broker stop matching the desired worst-SL flips DEGRADING back to HEALTHY."""
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    # Restart-replay registers the parent DEGRADING (health/owner were not
    # persisted, so the broker-native stop cannot be assumed in place).
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0,
                        pending_confirmation=True, now_ms=1000.0)
    assert mgr.get_state(ref).health is FailsafeHealth.DEGRADING
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[90.0], now_ms=1000.0)
    # PUT succeeds (dispatcher returns) — clears pending_put, but the broker
    # side is not yet *confirmed* to carry the desired stop, so health holds.
    engine.set_native_bracket_dispatcher(lambda _snap: None)
    engine.drive_native_failsafe(now_ms=1000.0)
    assert mgr.get_state(ref).pending_put is False
    assert mgr.get_state(ref).health is FailsafeHealth.DEGRADING

    # Reconcile observes the broker carrying the desired worst-SL -> HEALTHY.
    engine.record_native_bracket_observed(
        ref, stop_level=90.0, profit_level=None, trailing_stop=None,
        now_ms=2000.0,
    )
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY


def __test_record_native_bracket_observed_external_edit_flips_owner_unknown__():
    """A mismatching observed stop (operator edit) flips fail-safe ownership to UNKNOWN."""
    # A mismatching observation (operator edited the stop at the broker) must
    # flip ownership to UNKNOWN — the engine must NOT silently resend its now
    # stale desired level over a manual edit. UNKNOWN also blocks new brackets
    # until a user reset, same as DEGRADED.
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0)
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[90.0], now_ms=1000.0)
    engine.set_native_bracket_dispatcher(lambda _snap: None)
    engine.drive_native_failsafe(now_ms=1000.0)
    assert mgr.get_state(ref).owner is FailsafeOwner.ENGINE_FAILSAFE

    engine.record_native_bracket_observed(
        ref, stop_level=80.0, profit_level=None, trailing_stop=None,
        now_ms=2000.0,
    )
    assert mgr.get_state(ref).owner is FailsafeOwner.UNKNOWN


def __test_enqueue_native_bracket_observed_recovers_on_drive__():
    """An enqueued observed confirm is applied only when the next drive drains the queue."""
    # Thread-safe production path: the reconcile (broker-loop) thread enqueues;
    # the MAIN thread applies it inside drive_native_failsafe, so the manager
    # state is mutated from one thread only. Nothing is applied until the next
    # drive drains the queue.
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0,
                        pending_confirmation=True, now_ms=1000.0)
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[90.0], now_ms=1000.0)
    engine.set_native_bracket_dispatcher(lambda _snap: None)
    engine.drive_native_failsafe(now_ms=1000.0)  # PUT lands, still DEGRADING
    assert mgr.get_state(ref).health is FailsafeHealth.DEGRADING

    # Enqueue the observed confirm — queued, not yet applied.
    engine.enqueue_native_bracket_observed(
        ref, stop_level=90.0, profit_level=None, trailing_stop=None)
    assert mgr.get_state(ref).health is FailsafeHealth.DEGRADING

    # The next drive drains the queue first -> HEALTHY.
    engine.drive_native_failsafe(now_ms=2000.0)
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY


def __test_drive_native_failsafe_drains_observed_before_stale_window__():
    """A queued confirm drained before the stale window still recovers DEGRADING to HEALTHY."""
    # The queued confirm must be applied BEFORE tick_stale_window: a confirm
    # that lands after the stale window has elapsed must still recover the
    # parent (DEGRADING -> HEALTHY), not lose the race to a DEGRADED escalation
    # (on_native_bracket_observed recovers only from DEGRADING).
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0,
                        pending_confirmation=True, now_ms=1000.0,
                        stale_window_ms=100.0)
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[90.0], now_ms=1000.0)
    engine.set_native_bracket_dispatcher(lambda _snap: None)
    engine.drive_native_failsafe(now_ms=1000.0)
    assert mgr.get_state(ref).health is FailsafeHealth.DEGRADING

    engine.enqueue_native_bracket_observed(
        ref, stop_level=90.0, profit_level=None, trailing_stop=None)
    # now_ms is 1000ms past degrading_since with a 100ms stale window: drained
    # confirm wins because it runs first.
    engine.drive_native_failsafe(now_ms=2000.0)
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY


def __test_drive_native_failsafe_coalesces_observed_keeps_latest__():
    """The drain coalesces observations per ref and applies only the latest snapshot."""
    # The reconcile (broker-loop) thread can enqueue several observations for
    # one parent between two main-thread drives (the bar interval spans many
    # polls). The drain must coalesce per ref and apply only the LATEST: a
    # stale pre-PUT mismatch enqueued ahead of the fresh matching snapshot must
    # NOT flip ENGINE_FAILSAFE -> UNKNOWN (which on_native_bracket_observed
    # cannot undo from a later match), or the parent would strand until reset.
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0,
                        pending_confirmation=True, now_ms=1000.0)
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[90.0], now_ms=1000.0)
    engine.set_native_bracket_dispatcher(lambda _snap: None)
    engine.drive_native_failsafe(now_ms=1000.0)  # PUT lands, still DEGRADING
    assert mgr.get_state(ref).owner is FailsafeOwner.ENGINE_FAILSAFE
    assert mgr.get_state(ref).health is FailsafeHealth.DEGRADING

    # Poll 1 still saw the pre-PUT broker level (stale mismatch); poll 2 saw
    # the desired worst-SL. Both queued before the next drive.
    engine.enqueue_native_bracket_observed(
        ref, stop_level=80.0, profit_level=None, trailing_stop=None)
    engine.enqueue_native_bracket_observed(
        ref, stop_level=90.0, profit_level=None, trailing_stop=None)
    engine.drive_native_failsafe(now_ms=2000.0)
    # Latest (matching) snapshot wins: ownership stays engine, state recovers.
    assert mgr.get_state(ref).owner is FailsafeOwner.ENGINE_FAILSAFE
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY


def __test_drive_native_failsafe_manual_edit_before_recompute_flips_unknown__():
    """A manual edit matching no outstanding level flips UNKNOWN and drops the queued PUT."""
    # The outstanding-levels exemption covers an observation that diverges from
    # the new desired level ONLY when it equals a level the broker still
    # legitimately carries (the baseline or a dispatched level). An operator's
    # manual broker-side edit observed BEFORE a same-sync recompute queued the
    # next PUT diverges from BOTH the new desired and every outstanding entry —
    # it must flip ownership to UNKNOWN, not be silently overwritten by the
    # queued PUT.
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0)
    # First worst-SL armed at 90.0, dispatched + confirmed HEALTHY/engine-owned.
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[90.0], now_ms=1000.0)
    engine.set_native_bracket_dispatcher(lambda _snap: None)
    engine.drive_native_failsafe(now_ms=1000.0)
    engine.record_native_bracket_observed(
        ref, stop_level=90.0, profit_level=None, trailing_stop=None,
        now_ms=1000.0,
    )
    assert mgr.get_state(ref).owner is FailsafeOwner.ENGINE_FAILSAFE
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY

    # Operator manually moves the broker stop to 70.0; the reconcile poll
    # observes it and enqueues it (broker-loop thread).
    engine.enqueue_native_bracket_observed(
        ref, stop_level=70.0, profit_level=None, trailing_stop=None)
    # Before the drain, a leg-driven recompute on this same sync moves the
    # worst-SL to 85.0 and queues a fresh PUT (generation bumped, _pending set,
    # PUT not yet dispatched). The outstanding baseline captures the broker's
    # old 90.0 below the freshly dispatched 85.0.
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[85.0], now_ms=2000.0)
    assert [e.sl for e in mgr.get_state(ref).outstanding] == [90.0, 85.0]

    # drive drains the queued 70.0 observation first: it matches neither the
    # new desired (85.0) nor any outstanding level (90.0) -> external edit. The
    # owner flips to UNKNOWN AND the queued 85.0 PUT must be dropped, never
    # dispatched — dispatching it here would overwrite the operator's manual
    # 70.0 edit the guard exists to preserve.
    dispatched: list[float | None] = []
    engine.set_native_bracket_dispatcher(
        lambda snap: dispatched.append(snap.stop_level))
    engine.drive_native_failsafe(now_ms=2000.0)
    assert mgr.get_state(ref).owner is FailsafeOwner.UNKNOWN
    assert dispatched == []


def __test_drive_native_failsafe_stale_baseline_after_dispatch_keeps_engine__():
    """A stale post-dispatch baseline observation is exempt, keeping the parent engine-owned."""
    # The outstanding baseline must survive the queued snapshot being popped on
    # dispatch. The reconcile thread can sample the broker AFTER the fresh PUT
    # dispatched (``mark_dispatch_in_flight`` / ``record_put_success`` already
    # cleared ``_pending`` and ``pending_put``) but BEFORE the confirming poll
    # arrives, so the lone observation still reports the old baseline SL. Gating
    # the exemption on the queued snapshot's presence would misread that stale
    # sample as an external edit and flip ENGINE_FAILSAFE -> UNKNOWN, stranding
    # the parent until manual reset. The outstanding list (populated at
    # recompute, cleared on confirm) keeps the parent engine-owned across this
    # window.
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0)
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[90.0], now_ms=1000.0)
    engine.set_native_bracket_dispatcher(lambda _snap: None)
    engine.drive_native_failsafe(now_ms=1000.0)
    engine.record_native_bracket_observed(
        ref, stop_level=90.0, profit_level=None, trailing_stop=None,
        now_ms=1000.0,
    )
    assert mgr.get_state(ref).owner is FailsafeOwner.ENGINE_FAILSAFE
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY

    # A leg-driven recompute moves the worst-SL to 85.0; THIS drive dispatches
    # it (so ``_pending`` is popped and ``pending_put`` is cleared by the
    # synchronous success record). The outstanding baseline captures the
    # broker's 90.0 below the freshly dispatched 85.0.
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[85.0], now_ms=2000.0)
    assert [e.sl for e in mgr.get_state(ref).outstanding] == [90.0, 85.0]
    dispatched: list[float | None] = []
    engine.set_native_bracket_dispatcher(
        lambda snap: dispatched.append(snap.stop_level))
    engine.drive_native_failsafe(now_ms=2000.0)
    assert dispatched == [85.0]
    assert ref not in mgr._pending
    assert mgr.get_state(ref).pending_put is False

    # A reconcile poll that ran before the 85.0 PUT landed at the broker now
    # enqueues the stale 90.0; the confirming 85.0 poll has not arrived yet.
    engine.enqueue_native_bracket_observed(
        ref, stop_level=90.0, profit_level=None, trailing_stop=None)
    dispatched.clear()
    engine.drive_native_failsafe(now_ms=3000.0)
    # Stale baseline sample is exempt: ownership stays engine, no spurious PUT.
    assert mgr.get_state(ref).owner is FailsafeOwner.ENGINE_FAILSAFE
    assert dispatched == []

    # The confirming poll lands -> HEALTHY and the outstanding list is consumed.
    engine.enqueue_native_bracket_observed(
        ref, stop_level=85.0, profit_level=None, trailing_stop=None)
    engine.drive_native_failsafe(now_ms=4000.0)
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY
    assert mgr.get_state(ref).outstanding == []

    # With the baseline cleared, a genuine edit back to 90.0 is no longer
    # exempt and correctly flips ownership to UNKNOWN.
    engine.enqueue_native_bracket_observed(
        ref, stop_level=90.0, profit_level=None, trailing_stop=None)
    engine.drive_native_failsafe(now_ms=5000.0)
    assert mgr.get_state(ref).owner is FailsafeOwner.UNKNOWN


def __test_drive_native_failsafe_first_arm_none_baseline_keeps_engine__():
    """A first-arm ``None`` baseline observation is exempt, keeping the parent engine-owned."""
    # The outstanding exemption must survive the FIRST arm, where the broker
    # legitimately carries no stop at all. ``recompute_worst_sl`` records a
    # baseline entry with ``sl=None`` (the old desired) on the first PUT, so a
    # stale reconcile poll that still sees ``stop_level=None`` after the PUT
    # dispatched but before the confirming poll lands must NOT be misread as an
    # external edit. A model keyed on "is there a non-None baseline" would skip
    # it here (the value is a legitimate ``None``) and flip ENGINE_FAILSAFE ->
    # UNKNOWN, stranding the freshly armed parent until manual reset. The
    # explicit baseline entry keeps the parent engine-owned.
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0)
    # First worst-SL armed at 90.0; the broker carried no stop before, so the
    # baseline entry carries sl=None below the freshly dispatched 90.0.
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[90.0], now_ms=1000.0)
    assert [e.sl for e in mgr.get_state(ref).outstanding] == [None, 90.0]
    dispatched: list[float | None] = []
    engine.set_native_bracket_dispatcher(
        lambda snap: dispatched.append(snap.stop_level))
    engine.drive_native_failsafe(now_ms=1000.0)
    assert dispatched == [90.0]
    assert ref not in mgr._pending
    assert mgr.get_state(ref).pending_put is False

    # A reconcile poll that ran before the 90.0 PUT landed still reports no
    # broker stop (None); the confirming poll has not arrived yet.
    engine.enqueue_native_bracket_observed(
        ref, stop_level=None, profit_level=None, trailing_stop=None)
    dispatched.clear()
    engine.drive_native_failsafe(now_ms=2000.0)
    # Stale baseline None sample is exempt: ownership stays engine, no spurious
    # PUT, the outstanding list still live.
    assert mgr.get_state(ref).owner is FailsafeOwner.ENGINE_FAILSAFE
    assert dispatched == []
    assert [e.sl for e in mgr.get_state(ref).outstanding] == [None, 90.0]

    # The confirming 90.0 poll lands -> HEALTHY and the outstanding list is
    # consumed.
    engine.enqueue_native_bracket_observed(
        ref, stop_level=90.0, profit_level=None, trailing_stop=None)
    engine.drive_native_failsafe(now_ms=3000.0)
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY
    assert mgr.get_state(ref).outstanding == []


def __test_drive_native_failsafe_coalesced_flush_records_baseline__():
    """A coalesced trail flush records the broker baseline, exempting a later stale poll."""
    # The trail-coalesce flush path (``flush_coalesced_trails``) dispatches a
    # throttled trail PUT WITHOUT going through ``recompute_worst_sl``'s
    # baseline capture. It must still record the broker baseline (the previously
    # dispatched trail level) as an outstanding entry — otherwise a stale
    # reconcile poll that still sees the old broker level after the flushed PUT
    # returns has no exemption (the trail-coalesce exemption was cleared with
    # ``pending_trail_change_ts_ms``) and wrongly flips ENGINE_FAILSAFE ->
    # UNKNOWN.
    engine, _ = _mk_engine(MockBroker())
    mgr = engine._native_failsafe_manager
    ref = "run-pi-bar-e0"
    mgr.register_parent(parent_entry_dispatch_ref=ref, symbol=SYMBOL,
                        parent_side='long', mintick=1.0)
    dispatched: list[float | None] = []
    engine.set_native_bracket_dispatcher(
        lambda snap: dispatched.append(snap.stop_level))

    # Lifecycle arm at 90.0, confirmed HEALTHY/engine-owned.
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[90.0], now_ms=1000.0)
    engine.drive_native_failsafe(now_ms=1000.0)
    engine.record_native_bracket_observed(
        ref, stop_level=90.0, profit_level=None, trailing_stop=None,
        now_ms=1000.0)
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY

    # First trail move to 88.0 dispatches immediately (no prior trail dispatch
    # timestamp) and is confirmed; this seeds last_trail_dispatched_level=88.0.
    dispatched.clear()
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[88.0], now_ms=2000.0,
                           trigger_kind='trail')
    engine.drive_native_failsafe(now_ms=2000.0)
    assert dispatched == [88.0]
    engine.record_native_bracket_observed(
        ref, stop_level=88.0, profit_level=None, trailing_stop=None,
        now_ms=2000.0)
    assert mgr.get_state(ref).last_trail_dispatched_level == 88.0

    # Second trail move to 86.0 within the coalesce window is throttled (no PUT).
    mgr.recompute_worst_sl(parent_entry_dispatch_ref=ref,
                           active_sl_levels=[86.0], now_ms=2100.0,
                           trigger_kind='trail')
    assert mgr.get_state(ref).pending_trail_change_ts_ms == 2100.0
    assert ref not in mgr._pending

    # After the coalesce window elapses, drive flushes the throttled 86.0 PUT.
    # The flush must capture the broker baseline (88.0) and arm the flag.
    dispatched.clear()
    engine.drive_native_failsafe(now_ms=2400.0)
    assert dispatched == [86.0]
    assert [e.sl for e in mgr.get_state(ref).outstanding] == [88.0, 86.0]
    assert mgr.get_state(ref).pending_trail_change_ts_ms is None

    # A stale poll that still saw 88.0 (before the 86.0 PUT landed) is exempt:
    # it equals the recorded baseline level, so ownership stays engine.
    engine.enqueue_native_bracket_observed(
        ref, stop_level=88.0, profit_level=None, trailing_stop=None)
    dispatched.clear()
    engine.drive_native_failsafe(now_ms=3000.0)
    assert mgr.get_state(ref).owner is FailsafeOwner.ENGINE_FAILSAFE
    assert dispatched == []

    # The confirming 86.0 poll lands -> HEALTHY and the outstanding list is
    # consumed.
    engine.enqueue_native_bracket_observed(
        ref, stop_level=86.0, profit_level=None, trailing_stop=None)
    engine.drive_native_failsafe(now_ms=4000.0)
    assert mgr.get_state(ref).health is FailsafeHealth.HEALTHY
    assert mgr.get_state(ref).outstanding == []


# === §2.6.7 native fail-safe state retirement on parent cancel/close ===
#
# A parent whose position vanishes (external close, cancel, reject) must have
# its NativeStopState retired — else a DEGRADING/DEGRADED state strands and
# block_new_entry blocks the symbol indefinitely under non-halting
# on_unexpected_cancel policies. The WATCH-phase flat-snapshot cascade only
# retires from_entries that still have legs in the ledger (and early-returns on
# an empty ledger), so a state that outlived its legs needs the cancel/reject
# event handlers to retire it via _retire_native_failsafe_for_entry.

def __test_retire_native_failsafe_for_entry_drops_parked_state__():
    """``_retire_native_failsafe_for_entry`` resolves the parent COID and retires the state."""
    # The helper resolves the parent COID via the live entry envelope — the
    # leg-less case the WATCH cascade misses — and retires the state.
    engine, pos = _mk_engine(MockBroker())
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    coid = engine._envelopes["L"].client_order_id('e')
    mgr = engine._native_failsafe_manager
    mgr.register_parent(parent_entry_dispatch_ref=coid, symbol=SYMBOL,
                        parent_side='long', mintick=1.0,
                        pending_confirmation=True, now_ms=float(BAR_TS))
    assert mgr.get_state(coid).health is FailsafeHealth.DEGRADING

    engine._retire_native_failsafe_for_entry("L")
    assert mgr.get_state(coid).health is FailsafeHealth.RETIRED


def __test_unexpected_cancel_event_retires_native_failsafe_state__():
    """An unexpected cancel event for an entry retires that parent's native fail-safe state."""
    engine, pos = _mk_engine(MockBroker())
    pos.entry_orders["L"] = _entry_order("L", 1.0)
    engine.sync(BAR_TS)
    coid = engine._envelopes["L"].client_order_id('e')
    deal_id = engine._order_mapping["L"][0]
    mgr = engine._native_failsafe_manager
    mgr.register_parent(parent_entry_dispatch_ref=coid, symbol=SYMBOL,
                        parent_side='long', mintick=1.0,
                        pending_confirmation=True, now_ms=float(BAR_TS))
    assert mgr.get_state(coid).health is FailsafeHealth.DEGRADING

    cancelled = OrderEvent(
        order=ExchangeOrder(
            id=deal_id, symbol=SYMBOL, side='buy',
            order_type=OrderType.MARKET, qty=1.0, filled_qty=0.0,
            remaining_qty=1.0, price=None, stop_price=None,
            average_fill_price=None, status=OrderStatus.CANCELLED,
            timestamp=0.0, fee=0.0, fee_currency="",
        ),
        event_type='cancelled', fill_price=None, fill_qty=None,
        timestamp=0.0, pine_id="L", from_entry=None,
    )
    engine._route_event(cancelled)
    assert mgr.get_state(coid).health is FailsafeHealth.RETIRED


def __test_unexpected_cancel_without_pine_id_retires_native_failsafe_state__():
    """A pine_id-less cancel matched by order id still retires the entry's fail-safe state."""
    # A broker-synthesized cancel status event may carry only the exchange
    # order id (``pine_id`` and ``from_entry`` both None). The cancel is still
    # matched to the entry intent via ``_find_key_for_order_id``, so the parent's
    # native fail-safe state must be retired using the matched ``key`` — deriving
    # the id from the event would pass ``''`` and leave a DEGRADING / DEGRADED
    # state parked under the COID, blocking the symbol indefinitely.
    engine, pos = _mk_engine(MockBroker())
    pos.entry_orders["L"] = _entry_order("L", 1.0)
    engine.sync(BAR_TS)
    coid = engine._envelopes["L"].client_order_id('e')
    deal_id = engine._order_mapping["L"][0]
    mgr = engine._native_failsafe_manager
    mgr.register_parent(parent_entry_dispatch_ref=coid, symbol=SYMBOL,
                        parent_side='long', mintick=1.0,
                        pending_confirmation=True, now_ms=float(BAR_TS))
    assert mgr.get_state(coid).health is FailsafeHealth.DEGRADING

    cancelled = OrderEvent(
        order=ExchangeOrder(
            id=deal_id, symbol=SYMBOL, side='buy',
            order_type=OrderType.MARKET, qty=1.0, filled_qty=0.0,
            remaining_qty=1.0, price=None, stop_price=None,
            average_fill_price=None, status=OrderStatus.CANCELLED,
            timestamp=0.0, fee=0.0, fee_currency="",
        ),
        event_type='cancelled', fill_price=None, fill_qty=None,
        timestamp=0.0, pine_id=None, from_entry=None,
    )
    engine._route_event(cancelled)
    assert mgr.get_state(coid).health is FailsafeHealth.RETIRED


def __test_unexpected_reject_without_pine_id_retires_native_failsafe_state__():
    """A pine_id-less reject matched by order id still retires the entry's fail-safe state."""
    # Mirror of the cancel case for the 'rejected' branch: a broker-synthesized
    # reject carrying only the exchange order id must still retire the matched
    # entry's native fail-safe state via the matched ``key``.
    engine, pos = _mk_engine(MockBroker())
    pos.entry_orders["L"] = _entry_order("L", 1.0)
    engine.sync(BAR_TS)
    coid = engine._envelopes["L"].client_order_id('e')
    deal_id = engine._order_mapping["L"][0]
    mgr = engine._native_failsafe_manager
    mgr.register_parent(parent_entry_dispatch_ref=coid, symbol=SYMBOL,
                        parent_side='long', mintick=1.0,
                        pending_confirmation=True, now_ms=float(BAR_TS))
    assert mgr.get_state(coid).health is FailsafeHealth.DEGRADING

    rejected = OrderEvent(
        order=ExchangeOrder(
            id=deal_id, symbol=SYMBOL, side='buy',
            order_type=OrderType.MARKET, qty=1.0, filled_qty=0.0,
            remaining_qty=1.0, price=None, stop_price=None,
            average_fill_price=None, status=OrderStatus.REJECTED,
            timestamp=0.0, fee=0.0, fee_currency="",
        ),
        event_type='rejected', fill_price=None, fill_qty=None,
        timestamp=0.0, pine_id=None, from_entry=None,
    )
    engine._route_event(rejected)
    assert mgr.get_state(coid).health is FailsafeHealth.RETIRED


# === One-way emulation routing (hedging, position_port set) ===============
#
# When ``broker.position_port`` is set the dispatch hub routes reducing /
# closing / reversing / bracket intents through the core OneWayEmulator (per-leg
# PositionPort primitives) instead of the single-position ``execute_*`` path.
# These drive ``_dispatch_new`` directly to assert the routing + the synthetic
# ``_order_mapping`` markers; the fan-out LOGIC itself is covered by test_039.


def _pleg(leg_id, side, qty, *, open_time=0.0) -> PositionLeg:
    return PositionLeg(
        leg_id=leg_id, symbol=SYMBOL, side=side, qty=qty,
        entry_price=100.0, open_time=open_time, unrealized_pnl=0.0,
    )


def __test_emulated_close_routes_through_position_port__():
    """With a position_port set, a close fans through ``close_leg`` not ``execute_close``."""
    b = MockBroker()
    b.position_port = b
    b.raw_legs = [_pleg("1", "buy", 2.0)]
    engine, _pos = _mk_engine(b)
    close = CloseIntent(pine_id="x", symbol=SYMBOL, side="sell", qty=2.0)
    engine._dispatch_new(close)
    # Fanned through the port, NOT execute_close; synthetic close-leg marker.
    assert b.close_leg_calls == [("1", 2)]
    assert b.close_calls == []
    assert engine.order_mapping[close.intent_key] == ["close-leg:1"]


def __test_emulated_exit_routes_through_position_port__():
    """With a position_port set, an exit replicates its bracket onto each leg, not execute_exit."""
    b = MockBroker()
    b.position_port = b
    b.raw_legs = [_pleg("1", "buy", 1.0, open_time=1.0),
                  _pleg("2", "buy", 1.0, open_time=2.0)]
    engine, _pos = _mk_engine(b)
    ex = ExitIntent(pine_id="X", from_entry="L", symbol=SYMBOL, side="sell",
                    qty=2.0, tp_price=120.0, sl_price=90.0)
    engine._dispatch_new(ex)
    # Bracket replicated onto BOTH legs via the port, NOT execute_exit.
    assert {leg_id for leg_id, _tp, _sl in b.amend_calls} == {"1", "2"}
    assert b.exit_calls == []
    assert set(engine.order_mapping[ex.intent_key]) == {"bracket:1", "bracket:2"}


def __test_emulated_entry_reversal_routes_through_position_port__():
    """An emulated reversal closes the opposing leg via the port and opens the residual qty."""
    from pynecore.core.broker.models import EntryIntent
    b = MockBroker()
    b.position_port = b
    # Short 2 leg; a combined buy 3 reverses -> close the short, open residual 1.
    b.raw_legs = [_pleg("9", "sell", 2.0, open_time=1.0)]
    engine, _pos = _mk_engine(b)
    entry = EntryIntent(pine_id="L", symbol=SYMBOL, side="buy", qty=3.0,
                        order_type=OrderType.MARKET)
    engine._dispatch_new(entry)
    assert b.close_leg_calls == [("9", 2)]  # opposing leg FIFO-closed
    assert b.place_leg_calls == [1.0]       # residual opened via the port
    assert b.entry_calls == []              # execute_entry NOT called


def __test_emulated_close_below_grid_skips_non_halting__():
    """An emulated close quantized below the grid raises ``OrderSkippedByPlugin``, no dispatch."""
    b = MockBroker()
    b.position_port = b
    b.raw_legs = [_pleg("1", "buy", 0.4)]  # int() quantizer floors 0.4 -> 0
    engine, _pos = _mk_engine(b)
    close = CloseIntent(pine_id="x", symbol=SYMBOL, side="sell", qty=0.4)
    with pytest.raises(OrderSkippedByPlugin):
        engine._dispatch_new(close)
    assert b.close_leg_calls == []  # nothing dispatched on a below-grid skip


def __test_emulated_exit_flat_skips_non_halting__():
    """An emulated exit with no legs raises ``OrderSkippedByPlugin`` and amends nothing."""
    b = MockBroker()
    b.position_port = b
    b.raw_legs = []  # flat: no legs to protect
    engine, _pos = _mk_engine(b)
    ex = ExitIntent(pine_id="X", from_entry="L", symbol=SYMBOL, side="sell",
                    qty=2.0, sl_price=90.0)
    with pytest.raises(OrderSkippedByPlugin):
        engine._dispatch_new(ex)
    assert b.amend_calls == []  # nothing amended on a flat skip


def __test_first_sync_drives_one_way_replay_when_emulating__():
    """The first sync drives the one-way restart replay once when a position_port is set."""
    b = MockBroker()
    b.position_port = b
    engine, _pos = _mk_engine(b)
    called: list = []
    orig = engine._one_way_emulator.restart_replay

    async def _spy(port):
        called.append(port)
        return await orig(port)

    engine._one_way_emulator.restart_replay = _spy
    engine.sync(BAR_TS)
    assert called == [b]  # driven once, with the port
    assert engine._one_way_replay_done is True
    engine.sync(BAR_TS + 60_000)
    assert len(called) == 1  # one-time only — not re-driven each sync


def __test_first_sync_skips_one_way_replay_when_not_emulating__():
    """A netting broker (no position_port) never drives the one-way restart replay."""
    b = MockBroker()  # position_port stays None
    engine, _pos = _mk_engine(b)
    called: list = []

    async def _spy(port):
        called.append(port)

    engine._one_way_emulator.restart_replay = _spy
    engine.sync(BAR_TS)
    assert called == []  # netting broker -> replay never driven
    assert engine._one_way_replay_done is False


def __test_one_way_replay_connection_error_retries_next_sync__():
    """A one-way replay connection error bails the sync and retries on the next sync."""
    b = MockBroker()
    b.position_port = b
    engine, _pos = _mk_engine(b)
    calls: list = []

    async def _spy(port):
        calls.append(port)
        if len(calls) == 1:
            raise ExchangeConnectionError("transient broker read failure")

    engine._one_way_emulator.restart_replay = _spy
    engine.sync(BAR_TS)  # first sync: replay errors -> sync bails, flag unset
    assert engine._one_way_replay_done is False
    assert len(calls) == 1
    engine.sync(BAR_TS + 60_000)  # retried on the next sync
    assert len(calls) == 2
    assert engine._one_way_replay_done is True


def _persist_bracket_ownership(ctx, *, leg_id="1", intent_key="X\0L",
                               pine_id="X", from_entry="L"):
    from pynecore.core.broker.store_helpers import create_bracket_ownership_row
    create_bracket_ownership_row(
        ctx, coid=f"bo-test:{leg_id}", symbol=SYMBOL, side="sell", qty=2.0,
        intent_key=intent_key, pine_entry_id=pine_id, from_entry=from_entry,
        leg_id=leg_id, attach_coid="t-attach",
        tp_price=120.0, sl_price=90.0, trail_offset=None,
    )


def __test_orphan_one_way_bracket_cleared_after_restart__(tmp_path):
    """After restart, the orphan sweep clears a bracket Pine no longer emits and frees its row."""
    # F1: after a restart the persist-first bracket-ownership ledger is
    # re-asserted, but ``_active_intents`` starts empty. An exit the script no
    # longer emits is never seen by the cancellation diff (it iterates
    # ``_active_intents``), so without an orphan sweep its bracket would stay
    # armed on the leg forever. The diff's one-way orphan cleanup clears any
    # ownership row whose intent is in neither ``_active_intents`` nor new_map.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import iter_active_bracket_ownerships
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_bracket_ownership(ctx)
        b = MockBroker()
        b.position_port = b
        b.raw_legs = [_pleg("1", "buy", 2.0)]
        engine = OrderSyncEngine(
            broker=b, position=BrokerPosition(), symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        engine._diff_and_dispatch([])  # Pine emits nothing -> the exit is orphan
        assert ("1", None, None) in b.amend_calls  # bracket cleared to None
        assert list(iter_active_bracket_ownerships(ctx)) == []  # row released


def __test_active_one_way_bracket_not_orphan_swept__(tmp_path):
    """The orphan sweep spares a bracket whose intent is still emitted, keeping the row live."""
    # The orphan sweep must spare an exit that is still live: a row whose
    # intent_key is in ``_active_intents`` (or new_map) is never cleared.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import iter_active_bracket_ownerships
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_bracket_ownership(ctx)
        b = MockBroker()
        b.position_port = b
        b.raw_legs = [_pleg("1", "buy", 2.0)]
        engine = OrderSyncEngine(
            broker=b, position=BrokerPosition(), symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        ex = ExitIntent(pine_id="X", from_entry="L", symbol=SYMBOL, side="sell",
                        qty=2.0, tp_price=120.0, sl_price=90.0)
        engine._active_intents["X\0L"] = ex
        engine._diff_and_dispatch([ex])  # still emitted -> in _active_intents + new_map
        assert b.amend_calls == []  # neither cleared nor re-dispatched
        assert any(r.intent_key == "X\0L"
                   for r in iter_active_bracket_ownerships(ctx))  # still protected


def __test_orphan_clear_timeout_drained_drops_envelope__(tmp_path):
    """An orphan clear that times out then re-clears via drain retires the stale envelope."""
    # The orphan sweep's DIRECT clear hits an ambiguous timeout: it leaves the
    # row "clearing" and `continue`s BEFORE the success path's `_drop_envelope`,
    # so the in-memory envelope/mapping for the orphan key survive. The per-sync
    # drain then re-clears + releases that same row. Without dropping the
    # envelope here, a later re-emission of the exit would have `_build_envelope`
    # reuse the stale anchor and rebuild the SAME attach coid, which an
    # idempotent plugin dedups -> the re-attach never arms. The drain must
    # therefore retire the engine state for every key it released that Pine no
    # longer emits, mirroring the direct-success path.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import iter_active_bracket_ownerships
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_bracket_ownership(ctx)
        b = MockBroker()
        b.position_port = b
        b.raw_legs = [_pleg("1", "buy", 2.0)]
        # The direct orphan clear amend times out (1), the drain's re-clear lands.
        b.fail_amend_unknown_count = 1
        engine = OrderSyncEngine(
            broker=b, position=BrokerPosition(), symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        # Seed the engine state a prior in-session bracket dispatch would leave
        # behind for this exit key (envelope + per-leg ownership mapping).
        ex = ExitIntent(pine_id="X", from_entry="L", symbol=SYMBOL, side="sell",
                        qty=2.0, tp_price=120.0, sl_price=90.0)
        engine._build_envelope(ex)
        engine._order_mapping["X\0L"] = ["bracket:1"]
        assert "X\0L" in engine._envelopes
        engine._diff_and_dispatch([])  # Pine emits nothing -> the exit is orphan
        # Direct clear timed out then the drain re-cleared + released the row.
        assert b.fail_amend_unknown_count == 0  # one failure consumed
        assert list(iter_active_bracket_ownerships(ctx)) == []  # row released
        # The stale engine state is retired so a re-emission mints a fresh coid.
        assert "X\0L" not in engine._envelopes
        assert "X\0L" not in engine._order_mapping


def __test_drain_connection_failure_preserves_already_released_keys__(tmp_path):
    """A drain hitting a dropped link reports the keys already released, leaving rest clearing."""
    # A multi-row drain releases the first clearing row (durably closed in the
    # store, its key collected) and then the second leg's re-clear amend drops the
    # link with ``ExchangeConnectionError``. If that exception escaped, the partial
    # ``released`` set would be lost: the caller logs + retries but never retires
    # the first key's envelope/mapping, so its row is gone from the store yet the
    # stale anchor survives -> a later re-emission rebuilds the same attach coid an
    # idempotent plugin dedups, leaving the leg unprotected. The drain must instead
    # stop on the dropped link but still REPORT the keys it already released, and
    # leave the unprocessed row ``clearing`` for the next sync to retry.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import (
        BRACKET_OWN_STATE_CLEARING,
        iter_active_bracket_ownerships,
        update_bracket_ownership_state,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        # Two orphan exits, each owning one leg, both pre-marked ``clearing``.
        _persist_bracket_ownership(ctx, leg_id="1", intent_key="A\0L",
                                   pine_id="A", from_entry="L")
        _persist_bracket_ownership(ctx, leg_id="2", intent_key="B\0M",
                                   pine_id="B", from_entry="M")
        for coid in ("bo-test:1", "bo-test:2"):
            update_bracket_ownership_state(
                ctx, coid=coid, new_state=BRACKET_OWN_STATE_CLEARING,
            )
        b = MockBroker()
        b.position_port = b
        b.raw_legs = [_pleg("1", "buy", 1.0), _pleg("2", "buy", 1.0)]
        # Leg "2"'s re-clear amend drops the link; leg "1" is processed first.
        b.fail_amend_conn_leg = "2"
        engine = OrderSyncEngine(
            broker=b, position=BrokerPosition(), symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        # Seed the engine state a prior in-session dispatch would leave for both.
        for pine_id, from_entry, key, coid in (
            ("A", "L", "A\0L", "bracket:1"),
            ("B", "M", "B\0M", "bracket:2"),
        ):
            ex = ExitIntent(pine_id=pine_id, from_entry=from_entry, symbol=SYMBOL,
                            side="sell", qty=1.0, tp_price=120.0, sl_price=90.0)
            engine._build_envelope(ex)
            engine._order_mapping[key] = [coid]
        # The drain must NOT propagate the connection error: it returns the
        # already-released key so the engine can retire it.
        drained = engine._run_async(
            engine._one_way_emulator.drain_clearing_rows(SYMBOL, b),
        )
        assert drained == {"A\0L"}  # leg "1" released; leg "2" lost the link
        live = list(iter_active_bracket_ownerships(ctx))
        assert [r.intent_key for r in live] == ["B\0M"]  # leg "2" still clearing
        # Engine retires the released key's stale anchor; the unfinished one stays.
        for key in drained:
            engine._order_mapping.pop(key, None)
            engine._drop_envelope(key)
        assert "A\0L" not in engine._envelopes
        assert "A\0L" not in engine._order_mapping
        assert "B\0M" in engine._envelopes  # still owns a live clearing row
