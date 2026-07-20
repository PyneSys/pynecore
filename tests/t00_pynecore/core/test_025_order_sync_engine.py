"""
Tests for :class:`OrderSyncEngine` — the diff/dispatch/event-routing core.

A :class:`MockBroker` implements just the async surface the engine uses,
recording every call so assertions can check which intent ended up where.
A stubbed :attr:`lib._script.initial_capital` keeps
:class:`BrokerPosition.equity` well-defined.
"""
from __future__ import annotations

import asyncio
import time
import threading
from concurrent import futures
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import pytest

from pynecore import lib
from pynecore.core.broker.exceptions import (
    BracketAttachAfterFillRejectedError,
    BrokerManualInterventionError,
    ClientOrderIdSpentError,
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    ExchangeRateLimitError,
    InsufficientMarginError,
    OrderDispositionUnknownError,
    OrderSkippedByPlugin,
    UnexpectedCancelError,
)
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.sync_engine import (
    OrderSyncEngine,
    READ_STUCK_GRACE_S,
    _SEEN_FILL_IDS_CAP,
    _SETTLED_DEFENSIVE_CLOSE_IDS_CAP,
    _BoundedIdSet,
)
from pynecore.core.plugin import ProviderError, TransientProviderError
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
    QuarantineEnteredEvent,
)
from pynecore.core.broker.native_failsafe_manager import FailsafeHealth, FailsafeOwner
from pynecore.lib.strategy import (
    Order,
    Trade,
    _order_type_entry,
    _order_type_close,
    _order_type_normal,
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
    client_order_id_max_len = 30  # BrokerPlugin contract attribute
    on_unexpected_cancel = "stop"  # BrokerPlugin contract attribute
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
    raise_on_next_modify_entry: Exception | None = None
    raise_on_next_modify_exit: Exception | None = None
    raise_on_next_cancel: Exception | None = None
    false_on_next_cancel: bool = False
    raise_on_next_get_open_orders: Exception | None = None
    raise_on_next_get_position: Exception | None = None
    #: Number of ``get_position`` reads that actually reached the broker. Used to
    #: prove a rate-limit backoff parks the read locally instead of issuing it.
    get_position_calls: int = 0
    # The mock emulates a margin-style venue (shorts and reversals are the
    # bread and butter of the engine tests) — declare short_selling so the
    # projected-position gate stays out of the way; the dedicated short-gate
    # tests override capabilities with the spot default (UNSUPPORTED).
    capabilities: ExchangeCapabilities = field(
        default_factory=lambda: ExchangeCapabilities(
            short_selling=CapabilityLevel.NATIVE,
        ),
    )
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
        if self.false_on_next_cancel:
            self.false_on_next_cancel = False
            return False
        return True

    async def modify_entry(self, old, new):
        self.modify_entry_calls.append((old, new))
        if self.raise_on_next_modify_entry is not None:
            err = self.raise_on_next_modify_entry
            self.raise_on_next_modify_entry = None
            raise err
        return [self._mk_order(new, 'e')]

    async def modify_exit(self, old, new):
        self.modify_exit_calls.append((old, new))
        if self.raise_on_next_modify_exit is not None:
            err = self.raise_on_next_modify_exit
            self.raise_on_next_modify_exit = None
            raise err
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
        self.get_position_calls += 1
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
                xchg_id: str = "xchg-1", fill_id: str | None = None,
                event_type: str = 'filled', filled_qty: float | None = None,
                remaining_qty: float = 0.0) -> OrderEvent:
    exch = ExchangeOrder(
        id=xchg_id, symbol=SYMBOL, side=side,
        order_type=OrderType.MARKET, qty=qty,
        filled_qty=qty if filled_qty is None else filled_qty,
        remaining_qty=remaining_qty, price=None, stop_price=None,
        average_fill_price=price, status=OrderStatus.FILLED,
        timestamp=0.0, fee=0.0, fee_currency="",
    )
    return OrderEvent(
        order=exch, event_type=event_type, fill_price=price,
        fill_qty=qty, timestamp=0.0, pine_id=pine_id, leg_type=leg,
        fill_id=fill_id,
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
    # a fresh retry_seq=0 id and double-open. The startup scan must instead
    # BIND the live working order to the re-declared entry intent so the diff
    # adopts it and dispatches NO fresh order — even when the script re-emits
    # the entry on a LATER bar than the one the order was placed on — while the
    # recovered anchor is still journaled so a modify/cancel rebuilds the exact
    # live COID.
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

        # The live order is adopted, not re-dispatched: no duplicate is sent.
        assert len(b.entry_calls) == 0
        assert engine._order_mapping["L"] == ["live-1"]  # type: ignore[attr-defined]

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


def __test_restart_adopted_entry_is_cancelled_not_duplicated__(tmp_path):
    """Restart binds the live entry order and a later cancel retires it.

    Reproduces the Capital.com / cTrader restart-adoption incident: phase A
    leaves a live entry working order; phase B re-declares the SAME entry. The
    first post-restart diff must ADOPT the live order (no second dispatch — the
    duplicate the venues created because they do not dedup working orders by
    client id), and when the script later drops the entry (``strategy.cancel``)
    the adopted order is retired instead of stranded.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    live_coid = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=0,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        b = MockBroker()
        b.open_orders = [_live_working_order(live_coid, order_id="wo-1")]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

        # First post-restart sync: adopt the live order, dispatch NOTHING.
        engine.sync(BAR_TS + 60_000)

        assert len(b.entry_calls) == 0
        assert engine._order_mapping["L"] == ["wo-1"]  # type: ignore[attr-defined]
        assert "L" in engine._active_intents  # type: ignore[attr-defined]

        # strategy.cancel(): the script stops declaring the entry.
        del pos.entry_orders["L"]
        engine.sync(BAR_TS + 120_000)

        # The adopted order is cancelled at the venue; still no duplicate entry.
        assert len(b.entry_calls) == 0
        assert len(b.cancel_calls) == 1
        assert "L" not in engine._order_mapping  # type: ignore[attr-defined]
        assert "L" not in engine._active_intents  # type: ignore[attr-defined]


def __test_clean_restart_equal_journal_anchor_adopts_not_duplicates__(tmp_path):
    """A CLEAN restart with journal and live order in agreement adopts, never duplicates.

    The exact reported incident shape (capitalcom.md / ctrader.md): phase A
    dispatches the entry THROUGH the engine, so the journal holds the
    retry_seq=0 envelope anchor; the process stops cleanly; phase B restarts
    with the working order still live under that same COID and the journal
    intact. Journal anchor == live anchor (same bar, retry 0 vs 0) — the two
    stores agree it is the SAME dispatch, so the first post-restart diff must
    bind the live order to the re-declared intent and dispatch NOTHING, and a
    later ``strategy.cancel`` must retire the adopted order.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import KIND_ENTRY

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        # --- Phase A: normal run dispatches the entry, journaling its envelope.
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        b = MockBroker()
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
        engine.sync(BAR_TS)
        assert len(b.entry_calls) == 1
        live_coid = b.entry_calls[0].client_order_id(KIND_ENTRY)
        ctx.close()  # clean stop

        # --- Phase B: restart — journal intact, working order live at the venue.
        ctx2 = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        b2 = MockBroker()
        b2.open_orders = [_live_working_order(live_coid, order_id="wo-1")]
        pos2 = BrokerPosition()
        engine2 = OrderSyncEngine(
            broker=b2,  # type: ignore[arg-type]
            position=pos2, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx2,
        )
        pos2.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

        engine2.sync(BAR_TS + 60_000)

        # No duplicate entry is dispatched; the live order is bound as the
        # active intent.
        assert len(b2.entry_calls) == 0
        assert engine2._order_mapping["L"] == ["wo-1"]  # type: ignore[attr-defined]
        assert "L" in engine2._active_intents  # type: ignore[attr-defined]

        # strategy.cancel(): the adopted order is retired, still no duplicate.
        del pos2.entry_orders["L"]
        engine2.sync(BAR_TS + 120_000)

        assert len(b2.entry_calls) == 0
        assert len(b2.cancel_calls) == 1
        assert "L" not in engine2._order_mapping  # type: ignore[attr-defined]
        assert "L" not in engine2._active_intents  # type: ignore[attr-defined]


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

        # Both live legs are bound to the entry intent; nothing is re-dispatched.
        assert len(b.entry_calls) == 0
        assert sorted(engine._order_mapping["L"]) == ["live-b", "live-e"]  # type: ignore[attr-defined]


def __test_restart_adopts_wire_form_live_entry_coid__(tmp_path):
    """A short-budget venue echoes wire ids; adoption forward-hash matches them."""
    # Same crash window as the canonical adoption test, but the venue's
    # client-id budget (20) forces the wire form: the echoed id carries
    # run/bar/kind raw and hides pid/retry in the hash tail, so the scan
    # snapshots the order whole and the builder recovers (bar, retry) by
    # rebuilding candidate canonical ids and comparing the re-encoded wire
    # form. The re-dispatch must be byte-identical to the live order's id.
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import (
        build_client_order_id, encode_wire_client_order_id, KIND_ENTRY,
    )

    wire_coid = encode_wire_client_order_id(
        build_client_order_id(
            run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
            kind=KIND_ENTRY, retry_seq=1,
        ),
        20,
    )
    assert len(wire_coid) == 20
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        b = MockBroker()
        b.client_order_id_max_len = 20
        b.open_orders = [_live_working_order(wire_coid)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

        engine.sync(BAR_TS + 60_000)  # script re-emits "L" on a later bar

        # The wire-form live order is bound to the intent and adopted, not
        # re-dispatched; the recovered anchor is still journaled below.
        assert len(b.entry_calls) == 0
        assert engine._order_mapping["L"] == ["live-1"]  # type: ignore[attr-defined]

        # The adoption journaled the recovered anchor — a restart keeps it.
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


def __test_restart_ignores_foreign_wire_form_order__(tmp_path):
    """A wire id from a different run_tag is ignored; the entry mints fresh."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import (
        build_client_order_id, encode_wire_client_order_id, KIND_ENTRY,
    )

    foreign_wire = encode_wire_client_order_id(
        build_client_order_id(
            run_tag="zzzz", pine_id="L", bar_ts_ms=BAR_TS,
            kind=KIND_ENTRY, retry_seq=3,
        ),
        20,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        b = MockBroker()
        b.client_order_id_max_len = 20
        b.open_orders = [_live_working_order(foreign_wire)]
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
        # Once the live order is visible it is adopted, not re-dispatched.
        assert len(b.entry_calls) == 0
        assert engine._order_mapping["L"] == ["live-1"]  # type: ignore[attr-defined]


# === Read-path backstop: untranslated transient faults park, never crash ===
#
# A plugin is contractually expected to translate a transient connectivity fault
# on a state-query READ into ``ExchangeConnectionError``. When a less-carefully-
# written plugin lets a raw provider/socket error escape instead, the engine's
# read-only bridge ``_run_async_read`` is the central safety net: it maps the
# untranslated transient to ``ExchangeConnectionError`` so the existing park-and-
# retry sites handle it rather than the raw error tearing down the live run.
# Reads are idempotent, so retry-next-bar is always safe. Mirrors the real
# cTrader net-drop crash on a per-bar reconcile ``get_position``.


def __test_reconcile_read_raw_retryable_provider_error_maps_to_exchange_connection__():
    """A raw retryable ``ProviderError`` on the reconcile ``get_position`` read
    is mapped to ``ExchangeConnectionError`` (the engine then parks + retries)."""
    b = MockBroker()
    b.raise_on_next_get_position = TransientProviderError("net dropped mid-read")
    engine, _ = _mk_engine(b)
    with pytest.raises(ExchangeConnectionError):
        engine.reconcile()


def __test_reconcile_read_stdlib_connection_and_timeout_map_to_exchange_connection__():
    """Raw stdlib ``ConnectionError`` / ``TimeoutError`` on a read map to
    ``ExchangeConnectionError`` too — covers a plugin that lets a socket/timeout
    fault propagate (or a wedged dispatch-bridge ``result(timeout=...)``)."""
    for raw in (ConnectionError("socket gone"), TimeoutError("read timed out")):
        b = MockBroker()
        b.raise_on_next_get_position = raw
        engine, _ = _mk_engine(b)
        with pytest.raises(ExchangeConnectionError):
            engine.reconcile()


def __test_reconcile_read_rate_limit_maps_to_exchange_connection__():
    """A venue rate limit (``error.too-many.requests`` -> ``ExchangeRateLimitError``)
    on the reconcile ``get_position`` read is a transient throttle: it maps to
    ``ExchangeConnectionError`` so the engine parks + retries instead of the run
    dying. Mirrors the real Capital.com ``GET /positions`` 429 crash."""
    b = MockBroker()
    b.raise_on_next_get_position = ExchangeRateLimitError(
        "API error occured: error.too-many.requests", retry_after=1.0,
    )
    engine, _ = _mk_engine(b)
    with pytest.raises(ExchangeConnectionError):
        engine.reconcile()


def __test_sync_skips_periodic_reconcile_rate_limit_then_recovers__():
    """End-to-end: a venue rate limit during periodic ``get_position`` polling
    parks the reconcile and keeps the live run going, then recovers on the next
    poll once the throttle clears. Sibling of the connection-error test — proves a
    recoverable 429 never terminates the run.

    Also pins the pacing contract: while the venue's ``retry_after`` interval is
    unexpired the engine must park the read *locally* — a further sync issues no
    broker request at all. Without that, ``calc_on_every_tick`` would keep
    hammering a venue that just asked to be left alone."""
    from pynecore.lib.strategy import Trade

    b = MockBroker()
    b.raise_on_next_get_position = ExchangeRateLimitError(
        "API error occured: error.too-many.requests", retry_after=1.0,
    )
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

    engine.sync(BAR_TS)  # rate limit -> reconcile parked, run survives

    assert pos.size == 100.0
    assert pos.open_trades
    reads_after_throttle = b.get_position_calls
    assert engine._read_backoff_until > 0.0, "retry_after was not recorded"

    # Throttle still unexpired: the next sync must not touch the venue.
    b.position = None  # broker recovered, but we are not allowed to ask yet
    engine.sync(BAR_TS + 60_000)

    assert b.get_position_calls == reads_after_throttle, "read issued while throttled"
    assert pos.size == 100.0

    # Throttle expires -> the very next poll reconciles against the flat broker.
    engine._read_backoff_until = 0.0
    engine.sync(BAR_TS + 120_000)

    assert b.get_position_calls > reads_after_throttle
    assert pos.size == 0.0
    assert pos.open_trades == []


def __test_timed_out_read_is_not_resubmitted_while_in_flight__():
    """
    ``result(timeout=...)`` abandons only the *wait* — the coroutine keeps
    running on the broker loop. The read bridge must therefore keep ownership of
    the timed-out future and park the next read, instead of stacking a second
    concurrent request on the same connection (three consecutive timeouts would
    otherwise leave three live reads piling into the plugin's shared executor).
    """
    import threading

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True, name="test-loop")
    thread.start()
    started = threading.Event()
    loop.call_soon_threadsafe(started.set)
    assert started.wait(timeout=5.0), "loop failed to start"

    release = asyncio.Event()
    entered = threading.Event()
    concurrent_reads = 0

    class _SlowReadBroker(MockBroker):
        async def get_position(self, symbol):
            nonlocal concurrent_reads
            concurrent_reads += 1
            entered.set()
            # Blocks past ``execute_timeout``, exactly like a venue read whose
            # own request timeout is longer than the engine's bridge timeout.
            await release.wait()
            return self.position

    b = _SlowReadBroker()
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=b,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        event_loop=loop,
        execute_timeout=0.2,
    )

    try:
        # First read: wedged on the loop, the bridge gives up waiting.
        with pytest.raises(ExchangeConnectionError):
            engine._run_async_read(b.get_position(SYMBOL))
        assert entered.wait(timeout=5.0), "read never reached the broker"
        assert concurrent_reads == 1

        # Second read while the first is still unresolved: parked locally, and
        # crucially NOT handed to the broker.
        with pytest.raises(ExchangeConnectionError):
            engine._run_async_read(b.get_position(SYMBOL))
        assert concurrent_reads == 1, "a second read was stacked on the first"

        # Once the wedged read genuinely resolves, ownership is released and
        # reads resume. Ownership is dropped by the first cycle that observes
        # the future finished, so wait on the retained future itself.
        wedged = engine._inflight_read
        assert wedged is not None, "timed-out read was not retained"
        loop.call_soon_threadsafe(release.set)
        wedged.result(timeout=5.0)

        assert engine._run_async_read(b.get_position(SYMBOL)) is b.position
        assert concurrent_reads > 1
        assert engine._inflight_read is None
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        loop.close()


def __test_late_rate_limit_from_timed_out_read_still_paces__():
    """
    A read that outruns ``execute_timeout`` and only *then* fails with a venue
    429 must still pace the venue. The bridge stopped waiting, so the fault
    surfaces on the next cycle when the retained future is collected — dropping
    it there would discard ``retry_after`` and let the next tick hammer a venue
    that just asked to be left alone.
    """
    import threading

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True, name="test-loop")
    thread.start()
    started = threading.Event()
    loop.call_soon_threadsafe(started.set)
    assert started.wait(timeout=5.0), "loop failed to start"

    release = asyncio.Event()

    class _LateRateLimitBroker(MockBroker):
        async def get_position(self, symbol):
            await release.wait()
            raise ExchangeRateLimitError(
                "API error occured: error.too-many.requests", retry_after=30.0,
            )

    b = _LateRateLimitBroker()
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=b,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        event_loop=loop,
        execute_timeout=0.2,
    )

    try:
        with pytest.raises(ExchangeConnectionError):
            engine._run_async_read(b.get_position(SYMBOL))

        wedged = engine._inflight_read
        assert wedged is not None, "timed-out read was not retained"
        loop.call_soon_threadsafe(release.set)
        with pytest.raises(ExchangeRateLimitError):
            wedged.result(timeout=5.0)

        assert engine._read_backoff_until == 0.0, "backoff armed too early"

        # Collecting the late failure must translate it and arm the backoff.
        with pytest.raises(ExchangeConnectionError):
            engine._run_async_read(b.get_position(SYMBOL))
        assert engine._read_backoff_until > time.monotonic(), \
            "late retry_after was discarded"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        loop.close()


def __test_permanently_wedged_read_halts_before_dispatching_blind__():
    """
    A read that never resolves parks every later read, which silently disables
    reconciliation — and the periodic reconcile runs AFTER ``_diff_and_dispatch``
    and swallows its connection error, so nothing else would stop the engine
    from ordering against a position view it can no longer refresh. Past
    ``READ_STUCK_GRACE_S`` the engine must fail closed instead.
    """
    import threading

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True, name="test-loop")
    thread.start()
    started = threading.Event()
    loop.call_soon_threadsafe(started.set)
    assert started.wait(timeout=5.0), "loop failed to start"

    release = asyncio.Event()

    class _WedgedReadBroker(MockBroker):
        async def get_position(self, symbol):
            await release.wait()
            return self.position

    b = _WedgedReadBroker()
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=b,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        event_loop=loop,
        execute_timeout=0.2,
    )

    try:
        with pytest.raises(ExchangeConnectionError):
            engine._run_async_read(b.get_position(SYMBOL))
        assert engine._inflight_read is not None

        # Still inside the grace: dispatch is allowed to continue.
        engine.sync(BAR_TS)
        assert not engine.halted

        # Age the wedged read past the grace.
        engine._reads_unconfirmed_since = time.monotonic() - (READ_STUCK_GRACE_S + 1.0)

        with pytest.raises(BrokerManualInterventionError):
            engine.sync(BAR_TS + 60_000)
        assert engine.halted
    finally:
        # Let the wedged read finish before stopping the loop — scheduling
        # ``release.set`` and ``loop.stop`` in the same iteration would stop
        # ``run_forever`` before the woken coroutine gets its turn, and
        # ``loop.close()`` would then destroy it pending.
        wedged = engine._inflight_read
        loop.call_soon_threadsafe(release.set)
        if wedged is not None:
            wedged.result(timeout=5.0)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        loop.close()


def __test_late_read_failure_blocks_dispatch_in_the_same_sync__():
    """
    A retained read that has already completed with a fault must be collected
    BEFORE ``_diff_and_dispatch``. The periodic reconcile that would otherwise
    collect it runs at the end of ``sync``, so an order would go out first —
    dispatched against a position view that demonstrably could not be refreshed.
    """
    import threading

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True, name="test-loop")
    thread.start()
    started = threading.Event()
    loop.call_soon_threadsafe(started.set)
    assert started.wait(timeout=5.0), "loop failed to start"

    release = asyncio.Event()

    class _LateFailingReadBroker(MockBroker):
        reads_healthy = False

        async def get_position(self, symbol):
            if self.reads_healthy:
                return await MockBroker.get_position(self, symbol)
            await release.wait()
            raise ExchangeRateLimitError(
                "API error occured: error.too-many.requests", retry_after=30.0,
            )

    b = _LateFailingReadBroker()
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=b,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        event_loop=loop,
        execute_timeout=0.2,
    )

    try:
        with pytest.raises(ExchangeConnectionError):
            engine._run_async_read(b.get_position(SYMBOL))

        wedged = engine._inflight_read
        assert wedged is not None
        loop.call_soon_threadsafe(release.set)
        with pytest.raises(ExchangeRateLimitError):
            wedged.result(timeout=5.0)

        # The read has now failed, but nothing has collected it yet.
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
        engine.sync(BAR_TS)

        assert b.entry_calls == [], "order dispatched before the late read failure was collected"
        assert engine._read_backoff_until > time.monotonic(), "late retry_after was discarded"
        assert not engine.halted, "a recoverable late fault must not kill the run"

        # Next cycle, with reads healthy again, dispatch resumes — but only
        # because the guard's re-read actually lands: clearing the backoff is
        # not enough on its own, the broker view has to be confirmed afresh.
        engine._read_backoff_until = 0.0
        engine.sync(BAR_TS + 60_000)
        assert b.entry_calls == [], "dispatch resumed without a confirmed position re-read"

        # That re-read hit the throttle again and re-armed the backoff.
        assert engine._read_backoff_until > time.monotonic()

        b.reads_healthy = True
        engine._read_backoff_until = 0.0
        engine.sync(BAR_TS + 120_000)
        assert len(b.entry_calls) == 1
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        loop.close()


def __test_wedged_reads_halt_even_when_pending_verification_bails_out_first__():
    """
    The stuck-read grace halt must not sit behind another read.
    ``_verify_pending_dispatches`` runs early in ``sync``, reads through the
    same bridge, and returns out of ``sync`` on its own
    ``ExchangeConnectionError`` — so a guard placed after it would never be
    reached on exactly the wedged runs it exists to catch, and the halt could
    never latch.
    """
    expected_coid = _preview_entry_coid("L", limit=50_000.0)

    class _AllReadsDownBroker(MockBroker):
        reads_down: bool = False

        async def get_open_orders(self, symbol=None):
            if self.reads_down:
                raise ExchangeConnectionError("socket closed")
            return await MockBroker.get_open_orders(self, symbol)

        async def get_position(self, symbol):
            if self.reads_down:
                raise ExchangeConnectionError("socket closed")
            return await MockBroker.get_position(self, symbol)

    b = _AllReadsDownBroker()
    b.raise_on_next_entry = OrderDispositionUnknownError(
        "simulated timeout", client_order_id=expected_coid,
    )
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)
    assert expected_coid in engine.pending_verification, "no pending dispatch parked"

    # Every read is down from here: the pending verification bails out of sync
    # before anything else runs.
    b.reads_down = True
    engine.sync(BAR_TS)
    assert not engine.halted, "a transient read outage must not halt inside the grace"

    engine._reads_unconfirmed_since = time.monotonic() - (READ_STUCK_GRACE_S + 1.0)
    with pytest.raises(BrokerManualInterventionError):
        engine.sync(BAR_TS)
    assert engine.halted


def __test_rate_limit_backoff_blocks_new_exposure__():
    """
    While a venue ``retry_after`` backoff is unexpired the engine cannot read at
    all — nothing is retained in flight to inspect, so the guard must fall back
    on the confirmation flag rather than reading "no stuck read" as "healthy"
    and dispatching against a position view it has no way to refresh.
    """
    b = MockBroker()
    b.raise_on_next_get_position = ExchangeRateLimitError(
        "error.too-many.requests", retry_after=30.0,
    )
    engine, pos = _mk_engine(b)

    with pytest.raises(ExchangeConnectionError):
        engine.reconcile()
    assert engine._read_backoff_until > time.monotonic()
    assert engine._inflight_read is None, "a synchronous 429 retains nothing"

    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert b.entry_calls == [], "new exposure opened during the venue backoff"

    # Backoff expired: the guard's re-read lands and dispatch resumes.
    engine._read_backoff_until = 0.0
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1


def __test_unconfirmed_view_still_cancels_and_closes__():
    """
    The unconfirmed-view gate must only block *new* exposure. Cancelling an
    obsolete resting entry and dispatching ``strategy.close`` reduce risk — if
    they were skipped for the whole grace window the cancelled entry could
    still fill and an unwanted position would stay open while the strategy
    believes it asked to flatten.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1

    # Reads are throttled from here: the view can no longer be confirmed.
    b.raise_on_next_get_position = ExchangeRateLimitError(
        "error.too-many.requests", retry_after=30.0,
    )
    with pytest.raises(ExchangeConnectionError):
        engine.reconcile()
    assert engine._read_backoff_until > time.monotonic()

    # The script drops the resting entry, asks to flatten, and signals a new
    # entry in the same bar.
    del pos.entry_orders["L"]
    pos.entry_orders["N"] = _entry_order("N", 1.0, limit=49_000.0)
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -1.0, order_type=_order_type_close,
        exit_id="Close entry(s) order L",
    )
    engine.sync(BAR_TS + 60_000)

    assert [c.intent.pine_id for c in b.cancel_calls] == ["L"], \
        "obsolete resting entry was not cancelled on an unconfirmed view"
    assert len(b.close_calls) == 1, \
        "strategy.close was withheld on an unconfirmed view"
    assert len(b.entry_calls) == 1, "new exposure opened during the venue backoff"


def __test_read_resolving_at_the_timeout_boundary_is_not_lost__():
    """
    ``result(timeout=...)`` can raise while the coroutine completes in the same
    instant. Dropping the future then would reduce a terminal fault — or a
    venue ``retry_after`` — to a plain bridge timeout, reopening the read gate
    on the very next cycle and hammering the venue it asked us to leave alone.
    """
    class _RacingFuture(futures.Future):
        """Times out on the bounded wait, yet is already resolved underneath."""

        def result(self, timeout=None):
            if timeout is not None:
                raise TimeoutError("bridge wait expired")
            return super().result()

    raced = _RacingFuture()
    raced.set_exception(
        ExchangeRateLimitError("error.too-many.requests", retry_after=30.0),
    )

    b = MockBroker()
    engine, _ = _mk_engine(b)
    # A loop only has to *exist* for the bridge to take its threadsafe path;
    # the submission itself is stubbed out below, so it never runs.
    engine._loop = asyncio.new_event_loop()

    def _fake_submit(coro, _loop):
        coro.close()
        return raced

    original = asyncio.run_coroutine_threadsafe
    asyncio.run_coroutine_threadsafe = _fake_submit
    try:
        with pytest.raises(ExchangeConnectionError):
            engine._run_async_read(b.get_position(SYMBOL))
    finally:
        asyncio.run_coroutine_threadsafe = original
        engine._loop.close()

    assert engine._read_backoff_until > time.monotonic(), \
        "the raced read's retry_after was discarded as a bridge timeout"
    assert engine._inflight_read is None, "a resolved future must not stay retained"


def __test_reconcile_read_non_retryable_provider_error_fails_loud__():
    """A non-retryable ``ProviderError`` (permanent misconfig) is NOT mapped — it
    propagates so the run fails loud instead of looping forever."""
    b = MockBroker()
    b.raise_on_next_get_position = ProviderError("unknown symbol")  # retryable=False
    engine, _ = _mk_engine(b)
    with pytest.raises(ProviderError) as excinfo:
        engine.reconcile()
    # Stays the original provider error, NOT silently reclassified to a reconnect.
    assert not isinstance(excinfo.value, ExchangeConnectionError)


def __test_restart_scan_raw_retryable_provider_error_parks_and_retries__(tmp_path):
    """End-to-end: a raw retryable ``ProviderError`` from the restart-scan
    ``get_open_orders`` parks the sync and retries, identical to an explicit
    ``ExchangeConnectionError`` — proving the backstop prevents the crash.

    Sibling of ``__test_restart_scan_connection_error_skips_sync_and_retries__``
    with an *untranslated* transient instead of a broker-taxonomy one.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    bumped_coid = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=1,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        b = MockBroker()
        b.raise_on_next_get_open_orders = TransientProviderError("broker link dropped")
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

        engine.sync(BAR_TS)  # scan hits the raw transient -> mapped + parked, no crash

        assert len(b.entry_calls) == 0
        assert engine._restart_entry_scan_done is False  # type: ignore[attr-defined]

        b.open_orders = [_live_working_order(bumped_coid)]
        engine.sync(BAR_TS)  # broker recovered -> scan + adoption

        assert engine._restart_entry_scan_done is True  # type: ignore[attr-defined]
        # Once the live order is visible it is adopted, not re-dispatched.
        assert len(b.entry_calls) == 0
        assert engine._order_mapping["L"] == ["live-1"]  # type: ignore[attr-defined]


# === Write-path backstop: untranslated dispatch transients halt, never dup ===
#
# The complement of the read backstop. A WRITE drop is disposition-ambiguous: a
# retry could duplicate a landed order, a blind park could strand a never-sent
# one. Only the plugin can tell pre-send from post-send, so an *untranslated*
# transient escaping a direct order write (``execute_*`` / ``modify_*``) is
# routed by ``_run_async_write`` to a controlled ``BrokerManualInterventionError``
# halt — strictly better than a raw crash, and only reachable for a contract-
# violating plugin. A plugin that DOES translate (ExchangeConnectionError /
# OrderDispositionUnknownError) is unaffected.


def __test_write_untranslated_retryable_transient_halts_for_manual_intervention__():
    """A raw retryable ``ProviderError`` escaping ``execute_entry`` latches a
    manual-intervention halt and does NOT re-dispatch (no duplicate order)."""
    b = MockBroker()
    b.raise_on_next_entry = TransientProviderError("link dropped after entry send")
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    # Surfaces now or latches for the next ``raise_if_halted`` — never a silent
    # re-dispatch.
    try:
        engine.sync(BAR_TS)
    except BrokerManualInterventionError:
        pass
    assert engine.halted is True
    with pytest.raises(BrokerManualInterventionError):
        engine.raise_if_halted()
    # Attempted exactly once — the ambiguous write is not retried.
    assert len(b.entry_calls) == 1


def __test_write_stdlib_connection_error_halts_for_manual_intervention__():
    """A raw stdlib ``ConnectionError`` on a write halts too (same ambiguity)."""
    b = MockBroker()
    b.raise_on_next_entry = ConnectionError("socket reset mid-dispatch")
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    try:
        engine.sync(BAR_TS)
    except BrokerManualInterventionError:
        pass
    assert engine.halted is True


def __test_write_disposition_unknown_parks_without_halt__():
    """A plugin that DOES translate a post-send drop
    (``OrderDispositionUnknownError``) is parked for verification, NOT halted —
    the contract path is unaffected by the backstop."""
    b = MockBroker()
    b.raise_on_next_entry = OrderDispositionUnknownError(
        "entry ack timed out", client_order_id="coid-x",
    )
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    engine.sync(BAR_TS)

    assert engine.halted is False


def __test_write_non_retryable_provider_error_fails_loud_without_halt__():
    """A non-retryable ``ProviderError`` on a write is NOT swallowed into a halt —
    it propagates so a permanent misconfiguration fails loud."""
    b = MockBroker()
    b.raise_on_next_entry = ProviderError("unsupported order type")  # retryable=False
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)

    with pytest.raises(ProviderError) as excinfo:
        engine.sync(BAR_TS)
    assert not isinstance(excinfo.value, BrokerManualInterventionError)
    assert engine.halted is False


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

        # The higher same-bar live order is bound and adopted, not re-dispatched.
        assert len(b.entry_calls) == 0
        assert engine._order_mapping["L"] == ["live-1"]  # type: ignore[attr-defined]


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


def __test_entry_spent_coid_redispatches_same_sync__():
    """A spent client order id re-anchors and re-dispatches within the same sync."""
    # A venue that never allows client-id reuse refuses a create whose
    # deterministic id was consumed by a now-dead order. Unlike a plain
    # reject (signal dropped, next bar re-evaluates), nothing is wrong with
    # the intent itself — the engine bumps ``retry_seq`` and re-sends
    # immediately so the entry lands in the SAME sync.
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    b.raise_on_next_entry = ClientOrderIdSpentError("orderLinkId spent")

    engine.sync(BAR_TS)

    assert len(b.entry_calls) == 2
    spent, redispatched = b.entry_calls
    assert spent.bar_ts_ms == redispatched.bar_ts_ms == BAR_TS
    assert spent.retry_seq == 0
    assert redispatched.retry_seq == 1
    assert engine.active_intents.keys() == {"L"}
    assert engine.order_mapping["L"] == ["xchg-1"]


def __test_entry_modify_spent_coid_dispatches_replacement_fresh__():
    """A spent id from the modify fallback re-anchors and dispatches the NEW intent fresh."""
    # The default cancel+recreate modify re-sends the pinned id the cancel
    # just spent; a no-reuse venue refuses it with nothing left live. The
    # engine must not halt and must not leave the key without a working
    # order: it re-anchors and dispatches the replacement as a fresh entry.
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1

    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=49_500.0)
    b.raise_on_next_modify_entry = ClientOrderIdSpentError("orderLinkId spent")
    engine.sync(BAR_TS)

    assert len(b.modify_entry_calls) == 1
    assert len(b.entry_calls) == 2
    redispatched = b.entry_calls[1]
    assert redispatched.intent.limit == 49_500.0
    # Same-bar re-anchor: identical bar_ts_ms, bumped retry_seq -> fresh id.
    assert redispatched.bar_ts_ms == BAR_TS
    assert redispatched.retry_seq == 1
    assert engine.active_intents["L"].limit == 49_500.0
    assert engine.order_mapping["L"] == ["xchg-2"]


def __test_exit_modify_spent_coid_dispatches_replacement_fresh__():
    """A spent id from an exit-bracket modify re-dispatches the bracket fresh."""
    # Same recovery on the bracket path: the position must not be left
    # silently without its TP/SL protection when the recreate collides
    # with the ids the cancel just spent.
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)
    assert len(b.exit_calls) == 1

    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=61_000.0, stop=45_000.0,
    )
    b.raise_on_next_modify_exit = ClientOrderIdSpentError("orderLinkId spent")
    engine.sync(BAR_TS)

    assert len(b.modify_exit_calls) == 1
    assert len(b.exit_calls) == 2
    redispatched = b.exit_calls[1]
    assert redispatched.bar_ts_ms == BAR_TS
    assert redispatched.retry_seq == 1


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


def _long_trade(entry_id: str, size: float) -> Trade:
    return Trade(size=size, entry_id=entry_id, entry_bar_index=0,
                 entry_time=0, entry_price=50_000.0, commission=0.0)


def __test_netted_over_close_clamps_to_flat__():
    """A netted close exceeding the position (70%+70%) clamps to flat, never reverses."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 10.0)]
    # BrokerPosition already netted the two 70% slices into one 14-unit close.
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -14.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )

    engine.sync(BAR_TS)

    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.qty == 10.0  # clamped to the 10-unit long, not 14


def __test_close_plus_close_all_clamps_total__():
    """close(id) is honoured first; close_all flattens only the residual exposure."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 10.0)]
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -3.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )
    pos.exit_orders[("Close position order", None)] = Order(
        None, -10.0, order_type=_order_type_close, exit_id="Close position order",
    )

    engine.sync(BAR_TS)

    qty_by_id = {c.intent.pine_id: c.intent.qty for c in b.close_calls}
    assert qty_by_id == {"L": 3.0, "": 7.0}  # 3 keyed + 7 residual = 10, never 13


def __test_single_close_within_exposure_unchanged__():
    """A close within the position exposure is dispatched untouched by the clamp."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 10.0)]
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -4.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )

    engine.sync(BAR_TS)

    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.qty == 4.0


def __test_close_clamped_by_per_id_exposure__():
    """A keyed close is capped by its entry id's open qty, not the whole position."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    # Pyramid: L1=6 + L2=4 = 10 net long.
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L1", 6.0), _long_trade("L2", 4.0)]
    # An oversized close of L1 (8) must clamp to L1's 6-unit exposure.
    pos.exit_orders[("Close entry(s) order L1", "L1")] = Order(
        "L1", -8.0, order_type=_order_type_close, exit_id="Close entry(s) order L1",
    )

    engine.sync(BAR_TS)

    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.qty == 6.0


def __test_clamped_away_close_all_is_removed_from_order_book__():
    """A close_all that clamps to zero is dropped from ``exit_orders``, never re-emitted.

    ``close(id)`` consumes the whole position and a same-evaluation ``close_all``
    has nothing left to flatten (clamps to zero). The backing close_all order
    must not survive in ``exit_orders``: otherwise the next sync (position now
    flat) re-derives it, hits the flat-position passthrough and dispatches a
    reduce-only close onto an already-flat account.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 10.0)]
    # close("L") takes the full 10; close_all has nothing left -> clamps to 0.
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -10.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )
    pos.exit_orders[("Close position order", None)] = Order(
        None, -10.0, order_type=_order_type_close, exit_id="Close position order",
    )

    engine.sync(BAR_TS)

    # Only the keyed close is dispatched; close_all flattens nothing.
    qty_by_id = {c.intent.pine_id: c.intent.qty for c in b.close_calls}
    assert qty_by_id == {"L": 10.0}
    # The clamped-away close_all order must be gone from the Pine order book.
    assert ("Close position order", None) not in pos.exit_orders
    # The keyed close that actually dispatched stays until its fill cleanup.
    assert ("Close entry(s) order L", "L") in pos.exit_orders


def __test_stale_keyed_close_for_flattened_entry_drops_not_redispatches__():
    """A keyed ``close(id)`` whose entry is gone clamps to zero, never to the residual.

    L1's ``close`` already flattened L1 while L2 remains, so the whole position
    is NOT flat and the close-fill cleanup never popped the stale close Order.
    On this sync ``qty_by_entry`` has no ``L1`` key. Capping to the residual
    (L2's exposure) would re-dispatch the close against an UNRELATED entry — the
    missing per-id exposure must clamp to zero and drop the backing order.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    # L1 was fully closed; only L2=4 remains open.
    pos.size = 4.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L2", 4.0)]
    # The stale close("L1") order still sits in exit_orders (no flat cleanup ran).
    pos.exit_orders[("Close entry(s) order L1", "L1")] = Order(
        "L1", -6.0, order_type=_order_type_close, exit_id="Close entry(s) order L1",
    )

    engine.sync(BAR_TS)

    # No close dispatched against L2's residual exposure.
    assert b.close_calls == []
    # The stale close order is dropped from the Pine order book.
    assert ("Close entry(s) order L1", "L1") not in pos.exit_orders


def __test_keyed_close_flattens_startup_adopted_position__():
    """A keyed ``close(id)`` flattens a startup-adopted position, never dropped.

    After a restart the real Pine entry id could not be recovered, so adoption
    seeded the open FIFO under the synthetic ``__adopted_startup__`` parent. The
    script then signals ``strategy.close("L")``: ``qty_by_entry`` has no ``L``
    key, but the FIFO is NOT faithful (it carries the synthetic id), so the close
    must dispatch against the adopted exposure rather than clamp to zero — else
    the live position can never be flattened by the script (reduce-only backstop
    guards over-close).
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("__adopted_startup__", 5.0)]
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -5.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )

    engine.sync(BAR_TS)

    # The close dispatches and flattens the adopted position (capped to 5).
    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.pine_id == "L"
    assert b.close_calls[0].intent.qty == 5.0
    # The backing close order survives until its fill cleanup (it dispatched).
    assert ("Close entry(s) order L", "L") in pos.exit_orders


def __test_in_flight_close_not_redispatched_after_partial_fill__():
    """An in-flight market close is NOT re-dispatched when a partial fill shrinks the residual.

    ``strategy.close("L")`` dispatches a 10-unit market close. The broker
    partially fills 6 (residual 4 still working); ``record_fill`` reduces
    ``position.size`` to 4 but leaves the backing close ``Order`` in
    ``exit_orders`` (the natural-close cleanup only runs at ``size == 0``), and
    the original full-qty ``CloseIntent`` stays in ``_active_intents``. The next
    sync re-derives the same close at full qty; the clamp caps it to the 4-unit
    residual, so it differs from the active intent and the diff routes it to the
    modify branch — where the ``CloseIntent`` -> ``CloseIntent`` guard recognises
    the irreversible in-flight market close and skips re-dispatch. Routing it
    through ``_dispatch_modify`` (cancel + re-execute) would otherwise issue a
    second ``execute_close`` (a market close cannot be cancelled) and double it.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 10.0)]
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -10.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )

    engine.sync(BAR_TS)
    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.qty == 10.0

    # Broker partially fills 6; residual 4 still working on the same close.
    pos.size = 4.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 4.0)]

    engine.sync(BAR_TS + 60_000)

    # No second close: the original is left to settle its residual.
    assert len(b.close_calls) == 1
    assert b.cancel_calls == []
    assert b.modify_exit_calls == []


def __test_in_flight_clamped_close_not_redispatched_after_partial_fill__():
    """A close CLAMPED on its first sync is not re-dispatched after a partial fill.

    ``strategy.close("L", qty=14)`` against a 10-unit long is clamped to 10 on
    the first sync; the 10-unit close is the intent stored in
    ``_active_intents``. The backing close ``Order`` still carries the
    script-declared full ``-14`` (the clamp never rewrites the Pine order book).
    A partial fill of 6 reduces ``position.size`` to 4, but the natural-close
    cleanup only runs at ``size == 0`` so the close Order survives. On the next
    sync ``build_intents`` re-derives the close at the full ``order.size`` (14):
    emitting that rebuilt intent unchanged would differ from the clamped active
    slot (qty 10) and route through ``_dispatch_modify`` (cancel + re-execute),
    doubling the market close. The diff-loop's ``CloseIntent`` -> ``CloseIntent``
    guard skips re-dispatching the irreversible in-flight close regardless of the
    first-sync clamp.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 10.0)]
    # Pine order book carries the full script-declared 14 (clamp never rewrites it).
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -14.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )

    engine.sync(BAR_TS)
    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.qty == 10.0  # clamped to the 10-unit long

    # Broker partially fills 6; residual 4 still working on the same close.
    pos.size = 4.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 4.0)]

    engine.sync(BAR_TS + 60_000)

    # No second close, no modify/cancel: the clamped original settles its residual.
    assert len(b.close_calls) == 1
    assert b.cancel_calls == []
    assert b.modify_exit_calls == []


def __test_inflight_smaller_close_reserves_only_working_qty_for_close_all__():
    """A same-evaluation ``close_all`` still flattens the residual past an in-flight close.

    ``strategy.close("L", qty=5)`` dispatches a 5-unit market close against a
    10-unit long; it is on the wire (``_active_intents['L']`` holds the 5-unit
    ``CloseIntent``) but not yet filled. The next evaluation grows the backing
    keyed close to the full 10 AND adds ``strategy.close_all()``. The diff-loop
    guard skips re-dispatching the in-flight keyed close (a market close cannot
    be cancelled / re-dispatched), so the clamp must reserve only the 5 actually
    working — not the rebuilt 10 — leaving the
    other 5 as residual for ``close_all`` to flatten. If it reserved the rebuilt
    10, ``close_all`` would clamp to zero and be dropped, stranding 5 units open.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 10.0)]
    # Sync 1: close("L", qty=5) — a 5-unit slice goes on the wire.
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -5.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )
    engine.sync(BAR_TS)
    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.pine_id == "L"
    assert b.close_calls[0].intent.qty == 5.0

    # Sync 2: no fill yet; script grows close("L") to the full 10 and adds close_all().
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -10.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )
    pos.exit_orders[("Close position order", None)] = Order(
        None, -10.0, order_type=_order_type_close, exit_id="Close position order",
    )
    engine.sync(BAR_TS + 60_000)

    # The keyed close is NOT re-dispatched (still the in-flight 5), and close_all
    # flattens the remaining 5 — total close coverage equals the 10-unit position.
    new_close = [c for c in b.close_calls[1:]]
    qty_by_id = {c.intent.pine_id: c.intent.qty for c in new_close}
    assert qty_by_id == {"": 5.0}  # only the close_all residual is newly dispatched
    assert b.cancel_calls == []
    assert b.modify_exit_calls == []
    total_active = sum(
        v.qty for v in engine.active_intents.values() if isinstance(v, CloseIntent)
    )
    assert total_active == 10.0  # 5 in-flight keyed + 5 close_all = full position


def __test_fully_filled_inflight_keyed_close_drops_stale_order_no_redispatch__():
    """A fully-filled keyed close on a still-open multi-entry book drops its stale
    backing order and is never re-dispatched.

    Two entries L1=6 + L2=4 (10 long). ``strategy.close("L1")`` dispatches a
    6-unit market close; the broker fills all 6, so ``record_fill`` flattens L1
    (position 10 -> 4) but the whole-position cleanup never runs (L2 keeps the
    book non-flat), leaving the now-fully-filled ``CloseIntent`` in
    ``_active_intents`` and its backing ``Order`` in ``exit_orders``. On the next
    sync the close's working remainder is 0 (``active.qty 6 - filled 6``): the
    clamp drops the stale backing ``Order``, so the rebuilt close vanishes from
    the diff and the cancellation pass retires the active slot via a local-only
    market-close cancel — never a second ``execute_close``.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L1", 6.0), _long_trade("L2", 4.0)]
    pos.exit_orders[("Close entry(s) order L1", "L1")] = Order(
        "L1", -6.0, order_type=_order_type_close, exit_id="Close entry(s) order L1",
    )
    engine.sync(BAR_TS)
    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.pine_id == "L1"
    assert b.close_calls[0].intent.qty == 6.0
    assert "L1" in engine.active_intents

    # Broker FULLY fills the 6-unit keyed close (position 10 -> 4); L2 stays open.
    full = OrderEvent(
        order=ExchangeOrder(
            id="xchg-1", symbol=SYMBOL, side="sell",
            order_type=OrderType.MARKET, qty=6.0, filled_qty=6.0,
            remaining_qty=0.0, price=None, stop_price=None,
            average_fill_price=50_000.0, status=OrderStatus.FILLED,
            timestamp=0.0, fee=0.0, fee_currency="",
        ),
        event_type='filled', fill_price=50_000.0,
        fill_qty=6.0, timestamp=0.0, pine_id="L1", leg_type=LegType.CLOSE,
    )
    engine._route_event(full)
    assert pos.size == 4.0  # L1 flattened, L2 (4) remains

    # Sync 2: the script still re-derives close("L1") from the stale backing Order.
    engine.sync(BAR_TS + 60_000)

    # No second close; the stale backing order is gone and the active slot retired.
    assert len(b.close_calls) == 1
    assert b.cancel_calls == []
    assert b.modify_exit_calls == []
    assert ("Close entry(s) order L1", "L1") not in pos.exit_orders
    assert "L1" not in engine.active_intents


def __test_completed_partial_close_retires_state_final_close_dispatches__():
    """A later ``strategy.close(id)`` for the residual dispatches after a filled partial close.

    ``strategy.close("L", qty=6)`` against a 10-unit long fills fully, leaving
    a 4-unit residual under the SAME id. The filled ``CloseIntent`` used to
    stay in ``_active_intents`` until the whole-position flat teardown, so the
    later ``strategy.close("L")`` for the residual was blocked (unchanged-skip
    against the identical slot / in-flight guard / working==0 clamp) and never
    reached the broker. The working==0 retirement at the fill site must pop
    the per-id close state (and drop the stale backing order) so the second
    close dispatches as a fresh intent. The CLOSE fill carries the entry id in
    ``from_entry`` (the Bybit / cTrader convention) to cover the
    ``from_entry or pine_id`` key derivation.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 10.0)]
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -6.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )

    engine.sync(BAR_TS)
    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.qty == 6.0

    # The 6-unit keyed close fills FULLY; 4 units of "L" stay open.
    fill = replace(
        _fill_event('sell', 6.0, 50_000.0, pine_id="", leg=LegType.CLOSE),
        pine_id=None, from_entry="L",
    )
    engine._route_event(fill)
    assert pos.size == 4.0

    # Retirement: slot + stale backing order gone, so the id is closable again.
    assert "L" not in engine.active_intents
    assert ("Close entry(s) order L", "L") not in pos.exit_orders

    # The script closes the residual — a genuinely new close on the same id.
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -4.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )
    engine.sync(BAR_TS + 60_000)

    assert len(b.close_calls) == 2
    assert b.close_calls[1].intent.pine_id == "L"
    assert b.close_calls[1].intent.qty == 4.0
    assert b.cancel_calls == []


def __test_inverse_undershoot_partial_close_retires_on_filled_event__():
    """An inverse partial close retires on the order's ``filled`` flag, not exact base sum.

    On an inverse contract the plugin converts each CLOSE fill from whole
    contracts back to base at the dispatch anchor, so the summed base
    typically UNDERSHOOTS the requested ``CloseIntent.qty`` by up to one
    contract's worth (~1e-5 BTC on BTCUSD) — orders of magnitude beyond the
    ``1e-9`` base tolerance the accumulation gate used. Retirement therefore
    never fired for inverse, the filled ``CloseIntent`` slot lingered, and the
    later keyed ``strategy.close(id)`` for the residual was suppressed (the
    live-inverse bug). The venue-authoritative ``event_type == "filled"`` flag
    must drive retirement regardless of the base-conversion residue, so the
    residual's fresh close still dispatches.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 10.0)]
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -6.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )

    engine.sync(BAR_TS)
    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.qty == 6.0

    # The keyed close fills FULLY at the venue, but the contracts->base
    # conversion lands the accumulated base 6e-5 SHORT of the 6.0 request —
    # far beyond ``1e-9``. Only the ``filled`` flag proves terminality.
    fill = replace(
        _fill_event('sell', 5.99994, 50_000.0, pine_id="", leg=LegType.CLOSE),
        pine_id=None, from_entry="L", event_type='filled',
    )
    engine._route_event(fill)
    assert pos.size == pytest.approx(4.00006)

    # Retirement fired despite the base undershoot: slot + backing order gone.
    assert "L" not in engine.active_intents
    assert ("Close entry(s) order L", "L") not in pos.exit_orders

    # The residual's fresh keyed close now dispatches instead of being blocked.
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -pos.size, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )
    engine.sync(BAR_TS + 60_000)

    assert len(b.close_calls) == 2
    assert b.close_calls[1].intent.pine_id == "L"
    assert b.cancel_calls == []


def __test_same_bar_residual_close_mints_fresh_client_order_id__():
    """A same-bar close after a filled partial close must NOT reuse its COID.

    ``strategy.close("L", qty=6)`` fills fully; the retirement re-dispatch of
    ``strategy.close("L")`` for the 4-unit residual can land on the SAME bar
    (live ``calc_on_every_tick`` syncing). The COID formula is
    ``run-pid-bar-kind+retry``, so a bare envelope drop would rebuild the
    identical ``retry_seq=0`` id the filled close already spent — an
    idempotency-caching venue (Bybit ``orderLinkId``) then returns the
    already-filled order instead of creating the fresh close, stranding the
    residual. Retirement must bump ``retry_seq`` for the same-bar window.
    """
    from pynecore.core.broker.idempotency import KIND_CLOSE
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 10.0)]
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -6.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )

    engine.sync(BAR_TS)
    assert len(b.close_calls) == 1
    first_coid = b.close_calls[0].client_order_id(KIND_CLOSE)

    # The 6-unit keyed close fills FULLY; 4 units of "L" stay open.
    fill = replace(
        _fill_event('sell', 6.0, 50_000.0, pine_id="", leg=LegType.CLOSE),
        pine_id=None, from_entry="L",
    )
    engine._route_event(fill)
    assert pos.size == 4.0

    # The script closes the residual on the SAME bar.
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -4.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )
    engine.sync(BAR_TS)

    assert len(b.close_calls) == 2
    assert b.close_calls[1].intent.qty == 4.0
    second_coid = b.close_calls[1].client_order_id(KIND_CLOSE)
    assert second_coid != first_coid, \
        "same-bar residual close reused the spent client_order_id"


def __test_partial_close_does_not_redispatch_retained_entry__():
    """A filled partial close must never let the retained market entry re-open exposure.

    ``strategy.entry("L", 200)`` fills; the consumed market ``Order`` stays in
    ``entry_orders`` as the sticky diff sentinel. ``strategy.close("L",
    qty=100)`` collides on the shared ``intent_key``, promotes the slot to the
    ``CloseIntent`` and fills. On the next sync only the retained
    ``EntryIntent`` is re-derived — before the fix the diff routed an
    active-CLOSE-vs-new-ENTRY modify through cancel + re-execute, re-dispatching
    the ORIGINAL 200-unit market entry (exposure 100 -> 300). The retirement
    plus the consumed-entry re-anchor guard must keep the entry off the wire,
    then let the final ``strategy.close("L")`` flatten the residual.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 200.0)

    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1
    engine._route_event(_fill_event('buy', 200.0, 1.0, pine_id="L"))
    assert pos.size == 200.0

    # Partial close: shares the intent key with the retained entry.
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -100.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )
    engine.sync(BAR_TS + 60_000)
    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.qty == 100.0
    assert isinstance(engine.active_intents["L"], CloseIntent)

    # The partial close fills fully; 100 units remain open.
    fill = replace(
        _fill_event('sell', 100.0, 1.0, pine_id="", leg=LegType.CLOSE,
                    xchg_id="xchg-close-1"),
        pine_id="L", from_entry=None,
    )
    engine._route_event(fill)
    assert pos.size == 100.0

    # Next sync re-derives ONLY the retained 200-unit entry. It must re-anchor
    # as the diff sentinel — never re-dispatch (the historic 200-unit re-entry).
    engine.sync(BAR_TS + 120_000)
    assert len(b.entry_calls) == 1
    assert len(b.close_calls) == 1
    from pynecore.core.broker.models import EntryIntent
    assert isinstance(engine.active_intents["L"], EntryIntent)

    # The final close for the residual dispatches fresh and flattens.
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -100.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )
    engine.sync(BAR_TS + 180_000)
    assert len(b.entry_calls) == 1
    assert len(b.close_calls) == 2
    assert b.close_calls[1].intent.qty == 100.0
    final_fill = replace(
        _fill_event('sell', 100.0, 1.0, pine_id="", leg=LegType.CLOSE,
                    xchg_id="xchg-close-2"),
        pine_id="L", from_entry=None,
    )
    engine._route_event(final_fill)
    assert pos.size == 0.0


def __test_consumed_entry_reanchors_over_stale_close_slot_without_dispatch__():
    """The consumed-entry guard alone re-anchors over a stale CloseIntent slot.

    Covers the path the fill-site retirement cannot reach: the active slot
    still holds a ``CloseIntent`` (e.g. one adopted without a captured backing
    order) while the only re-derived intent is the retained, fully-consumed
    market entry. The modify branch must re-anchor the sentinel without any
    broker round-trip — ``_dispatch_modify``'s mismatched-kinds branch would
    cancel + re-execute the entry and re-open closed exposure.
    """
    from pynecore.core.broker.models import EntryIntent

    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 100.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 100.0)]
    retained = _entry_order("L", 200.0)
    retained.filled_qty = 200.0
    pos.entry_orders["L"] = retained
    engine._active_intents["L"] = CloseIntent(
        pine_id="L", symbol=SYMBOL, side="sell", qty=100.0,
    )

    engine.sync(BAR_TS)

    assert b.entry_calls == []
    assert b.close_calls == []
    assert b.cancel_calls == []
    assert isinstance(engine.active_intents["L"], EntryIntent)


# === Duplicate-fill idempotency gate ===


def __test_duplicate_fill_id_applied_once__():
    """A redelivered fill carrying an already-applied ``fill_id`` is dropped.

    A broker can deliver the same execution twice (poll+stream race,
    reconnect replay, a cTrader correlated dispatch-response colliding with
    its uncorrelated push copy). The engine drops the second delivery on its
    broker-native ``fill_id`` BEFORE ``record_fill`` runs, so the position is
    not over-applied and the intraday risk counter is not double-counted.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)

    first = _fill_event("buy", qty=1.0, price=50_000.0, pine_id="L",
                        leg=LegType.ENTRY, fill_id="deal-1")
    engine._route_event(first)
    assert pos.size == 1.0
    assert len(pos.open_trades) == 1
    assert pos.risk_intraday_filled_orders == 1

    # Exact redelivery (same fill_id) — dropped, no mutation.
    dup = _fill_event("buy", qty=1.0, price=50_000.0, pine_id="L",
                      leg=LegType.ENTRY, fill_id="deal-1")
    engine._route_event(dup)
    assert pos.size == 1.0
    assert len(pos.open_trades) == 1
    assert pos.risk_intraday_filled_orders == 1


def __test_distinct_fill_ids_same_order_id_both_apply__():
    """Two genuine partials of ONE order (shared ``order.id``) both apply.

    ``order.id`` is per-ORDER and shared across an order's partial fills, so
    it must NOT be the dedupe key. Two slices with distinct broker ``fill_id``
    values but the same ``order.id`` are both legitimate and must both apply —
    a naive order-id seen-set would wrongly drop the second partial.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)

    p1 = _fill_event("buy", qty=1.0, price=50_000.0, pine_id="L",
                     leg=LegType.ENTRY, xchg_id="ord-1", fill_id="deal-1",
                     event_type='partial', filled_qty=1.0, remaining_qty=1.0)
    p2 = _fill_event("buy", qty=1.0, price=50_000.0, pine_id="L",
                     leg=LegType.ENTRY, xchg_id="ord-1", fill_id="deal-2",
                     event_type='partial', filled_qty=2.0, remaining_qty=0.0)
    engine._route_event(p1)
    engine._route_event(p2)
    assert pos.size == 2.0
    assert pos.risk_intraday_filled_orders == 2


def __test_fill_id_none_applies_every_time__():
    """``fill_id=None`` is a gate no-op — fills apply exactly as before.

    Cumulative-only reconcile emissions (no broker-native execution id) and
    the paper-trading simulator leave ``fill_id`` unset; the gate must not
    silently swallow such fills — those paths guarantee single delivery via
    their own persisted ``filled_qty`` cursor, not via this gate.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)

    e1 = _fill_event("buy", qty=1.0, price=50_000.0, pine_id="L",
                     leg=LegType.ENTRY, fill_id=None)
    e2 = _fill_event("buy", qty=1.0, price=50_000.0, pine_id="L",
                     leg=LegType.ENTRY, fill_id=None)
    engine._route_event(e1)
    engine._route_event(e2)
    assert pos.size == 2.0
    assert pos.risk_intraday_filled_orders == 2


def __test_malformed_fill_does_not_burn_id_for_corrected_redelivery__():
    """A malformed first delivery (qty/price <= 0, which record_fill ignores)
    must not burn its fill_id and block a later corrected redelivery.

    The gate only remembers a fill it will actually apply (mirrors record_fill's
    qty>0/price>0 gate), so a corrected event carrying the same id still applies.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)

    # Malformed: zero qty -> record_fill ignores it AND the gate must not
    # remember the id.
    malformed = _fill_event("buy", qty=0.0, price=50_000.0, pine_id="L",
                            leg=LegType.ENTRY, fill_id="deal-1")
    engine._route_event(malformed)
    assert pos.size == 0.0

    # Corrected redelivery with the SAME id -> applied (not dropped).
    corrected = _fill_event("buy", qty=1.0, price=50_000.0, pine_id="L",
                            leg=LegType.ENTRY, fill_id="deal-1")
    engine._route_event(corrected)
    assert pos.size == 1.0


def __test_seen_fill_ids_ring_evicts_oldest_at_cap__():
    """The seen-set is a bounded FIFO ring capped at ``_SEEN_FILL_IDS_CAP``.

    Within the cap a known id is a duplicate; once the cap is exceeded the
    oldest id is evicted and treated as new again — the documented bound.
    Duplicates always arrive close behind the original, so an id thousands of
    fills old can never reappear in practice.
    """
    b = MockBroker()
    engine, _ = _mk_engine(b)

    def _ev(fid: str) -> OrderEvent:
        return _fill_event("buy", qty=1.0, price=1.0, pine_id="L",
                           leg=LegType.ENTRY, fill_id=fid)

    for i in range(_SEEN_FILL_IDS_CAP):
        assert engine._is_duplicate_fill(_ev(f"f{i}")) is False
    # f1 is still in the ring -> duplicate (a True check does not mutate).
    assert engine._is_duplicate_fill(_ev("f1")) is True
    # One past the cap -> evicts the oldest (f0).
    assert engine._is_duplicate_fill(_ev(f"f{_SEEN_FILL_IDS_CAP}")) is False
    # f0 was evicted -> treated as new again.
    assert engine._is_duplicate_fill(_ev("f0")) is False
    # Bounded.
    assert len(engine._seen_fill_ids) == _SEEN_FILL_IDS_CAP


def __test_settled_defensive_close_caches_are_bounded__():
    """The settled-defensive-close identity caches are bounded FIFO rings.

    Re-adding a known id is a no-op (keeps its ring position); exceeding
    the cap evicts the oldest id. Guards the leak fix: a long-lived live
    session must not grow these caches without bound.
    """
    b = MockBroker()
    engine, _ = _mk_engine(b)

    for cache in (engine._settled_defensive_close_pine_ids,
                  engine._settled_defensive_close_order_refs,
                  engine._settled_defensive_close_client_order_ids):
        assert isinstance(cache, _BoundedIdSet)

    ring = _BoundedIdSet(3)
    ring.add("a")
    ring.add("b")
    ring.add("a")  # no-op re-add: "a" keeps its slot as the oldest entry
    ring.add("c")
    assert len(ring) == 3 and "a" in ring
    ring.add("d")  # evicts "a" (oldest), not "b"
    assert len(ring) == 3
    assert "a" not in ring
    assert "b" in ring and "c" in ring and "d" in ring
    assert _SETTLED_DEFENSIVE_CLOSE_IDS_CAP > 0


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


def __test_startup_adoption_seeds_open_trades_so_close_nets_flat__():
    """A non-zero startup adoption seeds ``open_trades`` so a later exit fill nets to flat.

    Regression: adoption used to restore only ``size``/``avg_price`` and leave
    ``open_trades`` empty. When the reconstructed bracket's reduce-only CLOSE
    leg then filled, ``record_fill`` walked an empty FIFO and minted a phantom
    opposite-side position instead of going flat.
    """
    b = MockBroker()
    b.position = ExchangePosition(
        symbol=SYMBOL, side="long", size=1000.0, entry_price=1.15200,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)

    engine.reconcile()  # startup adoption

    assert pos.size == 1000.0
    assert pos.sign == 1.0
    assert len(pos.open_trades) == 1
    assert pos.open_trades[0].size == 1000.0
    assert pos.open_trades[0].entry_id == "__adopted_startup__"

    # The bracket's TP/SL fires: a reduce-only CLOSE sell of the full size.
    pos.record_fill(
        _fill_event("sell", 1000.0, 1.15263, pine_id="Bracket", leg=LegType.CLOSE)
    )

    assert pos.size == 0.0  # flat — NOT a phantom -1000 short
    assert pos.sign == 0.0
    assert pos.open_trades == []


def __test_startup_adoption_decodes_short_side_to_negative_size__():
    """``ExchangePosition.size`` is an unsigned magnitude — a short must adopt as a negative size.

    Regression: the plain adoption branch used ``exch_pos.size`` raw, so a
    short (``side="short"``, ``size=1000``) was adopted as a +1000 long.
    """
    b = MockBroker()
    b.position = ExchangePosition(
        symbol=SYMBOL, side="short", size=1000.0, entry_price=1.15200,
        unrealized_pnl=0.0, liquidation_price=None,
        leverage=1.0, margin_mode="isolated",
    )
    engine, pos = _mk_engine(b)

    engine.reconcile()

    assert pos.size == -1000.0  # short, not +1000
    assert pos.sign == -1.0
    assert len(pos.open_trades) == 1
    assert pos.open_trades[0].size == -1000.0

    # Closing a short is a BUY; nets cleanly to flat.
    pos.record_fill(
        _fill_event("buy", 1000.0, 1.15100, pine_id="Bracket", leg=LegType.CLOSE)
    )

    assert pos.size == 0.0
    assert pos.open_trades == []


def __test_record_fill_exit_leg_clamps_to_flat_instead_of_flipping__():
    """A reduce-only/exit leg with insufficient FIFO clamps to flat, never opening an opposite side."""
    pos = BrokerPosition()
    pos.size = 1000.0
    pos.sign = 1.0
    pos.avg_price = 1.15200
    # open_trades intentionally empty — simulates a FIFO desync.

    flipped = pos.record_fill(
        _fill_event("sell", 1000.0, 1.15263, pine_id="Bracket", leg=LegType.CLOSE)
    )

    assert pos.size == 0.0  # clamped flat — NOT -1000
    assert pos.sign == 0.0
    assert pos.open_trades == []
    assert flipped is True  # side did change (long -> flat)


def __test_record_fill_partial_exit_leg_keeps_residual_size__():
    """A PARTIAL reduce-only fill against an under-counted FIFO keeps the residual size.

    Regression: the exit-leg guard used to clamp to flat on ANY leftover qty,
    so an adopted long 1000 with no FIFO rows receiving a sell TP of 400 went
    to size 0 instead of 600 — losing live broker exposure and letting the
    script re-fire the entry. The authoritative net (``self.size += signed_delta``)
    is correct; the clamp must only fire when the exit actually over-closes.
    """
    pos = BrokerPosition()
    pos.size = 1000.0
    pos.sign = 1.0
    pos.avg_price = 1.15200
    # open_trades intentionally empty — simulates a FIFO desync.

    flipped = pos.record_fill(
        _fill_event("sell", 400.0, 1.15263, pine_id="Bracket", leg=LegType.TAKE_PROFIT)
    )

    assert pos.size == 600.0  # residual kept — NOT clamped to 0
    assert pos.sign == 1.0  # still long
    assert pos.avg_price == 1.15200  # untouched on a partial reduce
    assert flipped is False  # side did NOT change (long -> long)


def __test_record_fill_entry_leg_still_flips_for_reversal__():
    """An ENTRY leg may still flip the side (stop-and-reverse) — the clamp only blocks exit legs."""
    from pynecore.lib.strategy import Trade
    pos = BrokerPosition()
    pos.size = 1000.0
    pos.sign = 1.0
    pos.avg_price = 1.15200
    pos.open_trades.append(Trade(
        size=1000.0, entry_id="L", entry_bar_index=0, entry_time=0,
        entry_price=1.15200, commission=0.0, entry_comment=None,
        entry_equity=1_000_000.0,
    ))

    pos.record_fill(
        _fill_event("sell", 1500.0, 1.15300, pine_id="Short", leg=LegType.ENTRY)
    )

    assert pos.size == -500.0  # closed +1000, reversed into -500
    assert pos.sign == -1.0


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


def __test_bracket_reject_zero_residual_skips_close_and_does_not_halt__():
    """A proven-flat (qty=0) bracket reject dispatches NO defensive close and does not halt.

    When the plugin measured that a racing sibling fill consumed the
    ENTIRE bracket quantity, the reject arrives with ``qty=0`` — the
    position is already flat. The engine must not synthesize a
    zero-quantity :class:`CloseIntent`: a real plugin would skip it
    below the venue grid and the skip branch would escalate to a
    manual-intervention halt for a position that needs no intervention.
    The exit intent is surfaced as skipped, the re-dispatch guard is
    armed, and the run continues.
    """
    b = MockBroker()
    err = BracketAttachAfterFillRejectedError(
        "bracket attach reject",
        position_deal_id='deal-L',
        position_coid='coid-entry',
        symbol=SYMBOL,
        position_side='buy',
        qty=0.0,
        from_entry='Long',
    )
    b.raise_on_next_exit = err
    engine, _pos = _mk_engine(b)

    intent = _bracket_reject_exit_intent()
    with pytest.raises(OrderSkippedByPlugin) as exc:
        engine._dispatch_new(intent)

    assert exc.value.reason == "bracket_reject_defensive_close"
    assert exc.value.intent_key == intent.intent_key
    # No close dispatched at all — the position is proven flat.
    assert b.close_calls == []
    # The sync-loop guard against re-dispatching brackets for this
    # entry is still armed, same as on the dispatched-close path.
    assert 'Long' in engine._defensively_closed_entries_this_sync
    assert engine.halted is False


def __test_bracket_reject_from_exit_modify_dispatches_defensive_close__():
    """A bracket reject raised by ``modify_exit`` runs the recovery instead of crashing the sync.

    The plugin's ``modify_exit`` can fall back to cancel+recreate (bracket
    shape change, missing inverse anchor, vanished amended leg), whose
    ``execute_exit`` recreate can reject AFTER the parent fill.
    ``_dispatch_modify`` must route the exception into
    :meth:`_handle_bracket_attach_after_fill_reject` exactly like
    ``_dispatch_new`` does — the defensive close dispatches, the exit slot
    is dropped, and ``sync()`` returns instead of propagating.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders[("Bracket", "L")] = _exit_order(
        "L", -1.0, "Bracket", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)
    assert len(b.exit_calls) == 1

    pos.exit_orders[("Bracket", "L")] = _exit_order(
        "L", -1.0, "Bracket", limit=61_000.0, stop=45_000.0,
    )
    b.raise_on_next_modify_exit = BracketAttachAfterFillRejectedError(
        "bracket recreate rejected after parent fill",
        position_deal_id='deal-L',
        position_coid='coid-entry',
        symbol=SYMBOL,
        position_side='buy',
        qty=1.0,
        from_entry='L',
    )
    engine.sync(BAR_TS)  # must not raise

    assert len(b.modify_exit_calls) == 1
    # Defensive close dispatched: opposite side, same qty/symbol.
    assert len(b.close_calls) == 1
    close_env = b.close_calls[0]
    assert isinstance(close_env.intent, CloseIntent)
    assert close_env.intent.side == 'sell'
    assert close_env.intent.qty == 1.0
    assert close_env.intent.immediately is True
    # The exit slot was dropped so the next bar re-evaluates from scratch.
    assert "Bracket\0L" not in engine.active_intents
    assert engine.halted is False


def __test_bracket_reject_zero_residual_from_exit_modify_does_not_halt__():
    """A proven-flat (qty=0) bracket reject from ``modify_exit`` skips the close and keeps running.

    Same routing as the previous test with the proven-flat contract: the
    sibling fill already flattened the position, so no defensive close
    dispatches, the re-dispatch guard is armed, and the run continues.
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.exit_orders[("Bracket", "L")] = _exit_order(
        "L", -1.0, "Bracket", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)
    assert len(b.exit_calls) == 1

    pos.exit_orders[("Bracket", "L")] = _exit_order(
        "L", -1.0, "Bracket", limit=61_000.0, stop=45_000.0,
    )
    b.raise_on_next_modify_exit = BracketAttachAfterFillRejectedError(
        "bracket recreate rejected, sibling fill flattened the position",
        position_deal_id='deal-L',
        position_coid='coid-entry',
        symbol=SYMBOL,
        position_side='buy',
        qty=0.0,
        from_entry='L',
    )
    engine.sync(BAR_TS)  # must not raise

    assert len(b.modify_exit_calls) == 1
    assert b.close_calls == []
    # Proven flat: no pending defensive-close marker was armed (no close
    # FILL will ever arrive to settle one).
    assert 'L' not in engine._pending_defensive_close
    assert "Bracket\0L" not in engine.active_intents
    assert engine.halted is False


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


def __test_strategy_cancel_echo_logged_as_own_cancel__(caplog):
    """A venue CANCELLED echo of a strategy-requested cancel is not external.

    ``_dispatch_cancel`` tears down ``_order_mapping`` synchronously on a
    confirmed cancel, so the venue's follow-up ``CANCELLED`` push no longer
    matches an intent. The engine must recognise it as its OWN cancel via the
    expected-id ring and log ``strategy cancel confirmed``, not the misleading
    ``external cancel observed`` fallback.
    """
    import logging

    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    deal_id = engine._order_mapping["L"][0]

    # Strategy drops the entry -> synchronous confirmed cancel.
    del pos.entry_orders["L"]
    engine.sync(BAR_TS)
    assert "L" not in engine._order_mapping
    assert deal_id in engine._strategy_cancel_expected_ids

    cancelled = OrderEvent(
        order=ExchangeOrder(
            id=deal_id, symbol=SYMBOL, side='buy',
            order_type=OrderType.LIMIT, qty=1.0, filled_qty=0.0,
            remaining_qty=1.0, price=50_000.0, stop_price=None,
            average_fill_price=None, status=OrderStatus.CANCELLED,
            timestamp=0.0, fee=0.0, fee_currency="",
        ),
        event_type='cancelled', fill_price=None, fill_qty=None,
        timestamp=0.0, pine_id=None, from_entry=None,
    )
    with caplog.at_level(logging.INFO, logger="pyne_core_logger"):
        engine._route_event(cancelled)
    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "strategy cancel confirmed" in messages
    assert "external cancel observed" not in messages
    # One-shot: the id is consumed so a genuinely external later cancel of a
    # reused id would still be flagged.
    assert deal_id not in engine._strategy_cancel_expected_ids


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


def _mk_bracket_with_inflight_close(b: MockBroker):
    """Entry L filled, whole-row bracket TP\\0L live, CloseIntent for L in flight.

    Returns ``(engine, pos, tp_id)`` where ``tp_id`` is the mapped venue id of
    the bracket exit leg. Mirrors the live-lab
    ``bybit_linear_entry_bracket_close`` sequence at the moment the venue
    reduce-only-cancels the TP leg during the explicit close.
    """
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0)
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)
    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="L", leg=LegType.ENTRY,
    ))
    tp_id = engine._order_mapping["TP\0L"][0]
    # ``strategy.close(id="L")`` dispatched, its fill not yet drained — the
    # CloseIntent shares the entry's ``pine_id`` key.
    engine._active_intents["L"] = CloseIntent(
        pine_id="L", symbol=SYMBOL, side="sell", qty=1.0,
    )
    return engine, pos, tp_id


def _reduce_only_cancel_event(order_id: str) -> OrderEvent:
    """A venue ``cancelled`` push for a reduce-only bracket TP leg."""
    return OrderEvent(
        order=ExchangeOrder(
            id=order_id, symbol=SYMBOL, side='sell',
            order_type=OrderType.LIMIT, qty=1.0, filled_qty=0.0,
            remaining_qty=1.0, price=60_000.0, stop_price=None,
            average_fill_price=None, status=OrderStatus.CANCELLED,
            timestamp=0.0, fee=0.0, fee_currency="", reduce_only=True,
        ),
        event_type='cancelled', fill_price=None, fill_qty=None,
        timestamp=0.0, pine_id="TP", from_entry="L",
        leg_type=LegType.TAKE_PROFIT,
    )


def __test_reduce_only_bracket_cancel_during_bot_close_is_expected__():
    """A reduce-only bracket leg the venue auto-cancels during a bot close
    must not quarantine, and its live SL sibling must be swept.

    When ``strategy.close`` flattens the position, Bybit auto-cancels the
    reduce-only TP limit (``cancelType=CancelByReduceOnly``) but leaves the
    conditional SL stop resting. The cancel is our own close's deterministic
    fallout, not an external operator cancel — the engine must recognise it,
    keep trading (no quarantine) and cancel the remaining bracket legs.
    """
    b = MockBroker()
    engine, pos, tp_id = _mk_bracket_with_inflight_close(b)

    engine._route_event(_reduce_only_cancel_event(tp_id))

    assert not engine._quarantined
    # The bracket intent's still-live SL leg is swept via a real cancel.
    swept = {(c.intent.pine_id, c.intent.from_entry) for c in b.cancel_calls}
    assert ("TP", "L") in swept
    # Tracking for the retired bracket is dropped so the next sync does not
    # re-diff it; the in-flight CloseIntent is left to settle.
    assert "TP\0L" not in engine.active_intents
    assert isinstance(engine.active_intents.get("L"), CloseIntent)
    assert ("TP", "L") not in pos.exit_orders


def __test_reduce_only_exit_cancel_without_bot_close_still_quarantines__():
    """Without a bot close in flight, a reduce-only exit cancel is external.

    The discriminator for the expected-cancel suppression is an active
    bot-initiated close for the same parent. An operator cancelling a
    reduce-only exit while no close is in flight is a genuine unexpected
    cancel and must still trip the ``on_unexpected_cancel`` quarantine.
    """
    b = MockBroker()
    engine, pos, tp_id = _mk_bracket_with_inflight_close(b)
    # Drop the in-flight close: no bot close is flattening the parent.
    engine._active_intents.pop("L", None)

    engine._route_event(_reduce_only_cancel_event(tp_id))

    assert engine._quarantined


def _mk_multi_bracket_with_cross_entry_close(b: MockBroker):
    """Two entries L1/L2, each its own whole-row bracket; close(L1) in flight.

    Mirrors the live-lab ``bybit_linear_multi_entry_brackets`` sequence at the
    instant the venue reduce-only-cancels a leg of L2's bracket while the bot
    is flattening L1. On a netting venue both entries share one net position,
    so ``strategy.close(L1)`` shrinks it and Bybit drops an excess reduce-only
    leg — here L2's TP — even though L2 stays open.

    Returns ``(engine, pos, tp2_id, sl2_id)``. L2's bracket is mapped to TWO
    venue legs (the single-leg mock exit is augmented with a seeded SL id) so
    the leg-trim keeps a live sibling.
    """
    engine, pos = _mk_engine(b)
    pos.entry_orders["L1"] = _entry_order("L1", 1.0)
    pos.entry_orders["L2"] = _entry_order("L2", 1.0)
    pos.exit_orders[("TP1", "L1")] = _exit_order(
        "L1", -1.0, "TP1", limit=60_000.0, stop=45_000.0,
    )
    pos.exit_orders[("TP2", "L2")] = _exit_order(
        "L2", -1.0, "TP2", limit=61_000.0, stop=46_000.0,
    )
    engine.sync(BAR_TS)
    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="L1", leg=LegType.ENTRY,
        xchg_id=engine._order_mapping["L1"][0],
    ))
    engine.on_order_event(_fill_event(
        "buy", 1.0, 50_000.0, pine_id="L2", leg=LegType.ENTRY,
        xchg_id=engine._order_mapping["L2"][0],
    ))
    tp2_id = engine._order_mapping["TP2\0L2"][0]
    # Model the second (conditional SL) leg the real plugin maps alongside the
    # TP: the single-leg mock exit only produced the TP id.
    sl2_id = "TP2-sl-xchg"
    engine._order_mapping["TP2\0L2"].append(sl2_id)
    # ``strategy.close(id="L1")`` dispatched, its fill not yet drained.
    engine._active_intents["L1"] = CloseIntent(
        pine_id="L1", symbol=SYMBOL, side="sell", qty=1.0,
    )
    return engine, pos, tp2_id, sl2_id


def _cross_entry_reduce_only_cancel_event(order_id: str) -> OrderEvent:
    """A venue ``cancelled`` push for L2's reduce-only TP leg."""
    return OrderEvent(
        order=ExchangeOrder(
            id=order_id, symbol=SYMBOL, side='sell',
            order_type=OrderType.LIMIT, qty=1.0, filled_qty=0.0,
            remaining_qty=1.0, price=61_000.0, stop_price=None,
            average_fill_price=None, status=OrderStatus.CANCELLED,
            timestamp=0.0, fee=0.0, fee_currency="", reduce_only=True,
        ),
        event_type='cancelled', fill_price=None, fill_qty=None,
        timestamp=0.0, pine_id="TP2", from_entry="L2",
        leg_type=LegType.TAKE_PROFIT,
    )


def __test_cross_entry_reduce_only_cancel_during_bot_close_is_expected__():
    """Closing one entry that reduce-only-cancels ANOTHER entry's leg must not
    quarantine, and must preserve the still-open sibling's bracket.

    On a netting / one-way venue the two entries share one net position, so
    ``strategy.close(L1)`` shrinks it and the venue drops an excess reduce-only
    leg belonging to L2 — a still-open entry NOT being closed. This is
    deterministic collateral of the bot's own close, not an external operator
    cancel: the engine must keep trading (no quarantine), trim only the dead
    leg, and leave L2's intent + its live SL leg intact (no defensive sweep).
    """
    b = MockBroker()
    engine, pos, tp2_id, sl2_id = _mk_multi_bracket_with_cross_entry_close(b)

    engine._route_event(_cross_entry_reduce_only_cancel_event(tp2_id))

    assert not engine._quarantined
    # L2's bracket intent survives with only its live SL leg mapped.
    assert "TP2\0L2" in engine.active_intents
    assert engine.order_mapping["TP2\0L2"] == [sl2_id]
    # The still-open sibling is NOT force-swept — no cancel dispatched for it.
    swept = {(c.intent.pine_id, c.intent.from_entry) for c in b.cancel_calls}
    assert ("TP2", "L2") not in swept
    # L2's Pine exit stays desired; the in-flight close of L1 is left to settle.
    assert ("TP2", "L2") in pos.exit_orders
    assert isinstance(engine.active_intents.get("L1"), CloseIntent)


def _cross_entry_reduce_only_sl_cancel_event(order_id: str) -> OrderEvent:
    """A venue ``cancelled`` push for L2's reduce-only conditional SL leg."""
    return OrderEvent(
        order=ExchangeOrder(
            id=order_id, symbol=SYMBOL, side='sell',
            order_type=OrderType.STOP, qty=1.0, filled_qty=0.0,
            remaining_qty=1.0, price=None, stop_price=46_000.0,
            average_fill_price=None, status=OrderStatus.CANCELLED,
            timestamp=0.0, fee=0.0, fee_currency="", reduce_only=True,
        ),
        event_type='cancelled', fill_price=None, fill_qty=None,
        timestamp=0.0, pine_id="TP2", from_entry="L2",
        leg_type=LegType.STOP_LOSS,
    )


def _settle_keyed_close_of_l1(engine, pos) -> None:
    """Model close(L1) settling: intent retired, L1's Pine orders leave the book.

    Mirrors the engine's keyed-close retirement ("close state retired") plus the
    next bar's Pine book after the trade closed — the script no longer holds
    L1's entry or its ``strategy.exit`` row, while L2 (still open, persistent
    Pine semantics) keeps both.
    """
    engine._active_intents.pop("L1", None)
    engine._order_mapping.pop("L1", None)
    del pos.entry_orders["L1"]
    del pos.exit_orders[("TP1", "L1")]


def __test_multi_entry_bracket_close_lifecycle_no_quarantine_no_halt__():
    """Full live-lab ``bybit_linear_multi_entry_brackets`` sequence: close(L1)
    with collateral cancel of L2's TP leg, then close(L2) sweeping its own
    bracket — no quarantine, no defensive close, clean convergence.

    Chains the two halves of the original failure (quarantine on the collateral
    cancel; halt via defensive close after the moot bracket re-emission) into
    the whole lifecycle from the evidence log. With the classification fix the
    surviving bracket intent is never swept, so the diff never re-emits the
    exit against a flat position and the 110017 → defensive-close → halt tail
    cannot start.
    """
    b = MockBroker()
    engine, pos, tp2_id, sl2_id = _mk_multi_bracket_with_cross_entry_close(b)
    exits_dispatched = len(b.exit_calls)

    # Venue reduce-only-cancels L2's TP as collateral of close(L1).
    engine._route_event(_cross_entry_reduce_only_cancel_event(tp2_id))
    assert not engine._quarantined
    assert engine.order_mapping["TP2\0L2"] == [sl2_id]

    # close(L1) fills and settles; next bar's book no longer holds L1.
    _settle_keyed_close_of_l1(engine, pos)
    engine.sync(BAR_TS)
    assert not engine._quarantined
    # L1's now-parentless bracket is retired via a real venue cancel; L2's
    # surviving bracket intent is adopted as-is — NOT re-dispatched.
    cancelled = {(c.intent.pine_id, c.intent.from_entry) for c in b.cancel_calls}
    assert ("TP1", "L1") in cancelled
    assert len(b.exit_calls) == exits_dispatched
    assert "TP2\0L2" in engine.active_intents

    # close(L2): the venue reduce-only-cancels L2's remaining SL leg — this
    # names the leg's OWN from_entry, so the whole bracket sweep applies.
    engine._active_intents["L2"] = CloseIntent(
        pine_id="L2", symbol=SYMBOL, side="sell", qty=1.0,
    )
    engine._route_event(_cross_entry_reduce_only_sl_cancel_event(sl2_id))
    assert not engine._quarantined
    assert "TP2\0L2" not in engine.active_intents
    assert ("TP2", "L2") not in pos.exit_orders

    # No defensive close was ever synthesised anywhere in the lifecycle.
    assert b.close_calls == []


def __test_cross_entry_cancel_of_last_leg_retires_and_redispatches_fresh__():
    """Both legs of the sibling's bracket collaterally cancelled: the intent is
    retired, the Pine exit stays desired, and the next sync re-dispatches it.

    When the venue drops BOTH the TP and the SL of a still-open sibling entry
    while another entry's close is in flight, the bracket has no venue presence
    left — the trim retires the intent but must NOT remove the Pine exit order,
    so the still-open entry regains protection through a fresh dispatch on the
    next sync instead of resting unprotected.
    """
    b = MockBroker()
    engine, pos, tp2_id, sl2_id = _mk_multi_bracket_with_cross_entry_close(b)
    exits_dispatched = len(b.exit_calls)

    engine._route_event(_cross_entry_reduce_only_cancel_event(tp2_id))
    engine._route_event(_cross_entry_reduce_only_sl_cancel_event(sl2_id))

    assert not engine._quarantined
    # Last mapped leg gone -> intent retired, but the Pine exit stays desired.
    assert "TP2\0L2" not in engine.active_intents
    assert "TP2\0L2" not in engine.order_mapping
    assert ("TP2", "L2") in pos.exit_orders

    # close(L1) settles; the next sync re-establishes L2's protection fresh.
    _settle_keyed_close_of_l1(engine, pos)
    engine.sync(BAR_TS)
    assert not engine._quarantined
    fresh = [c for c in b.exit_calls[exits_dispatched:]
             if c.intent.pine_id == "TP2"]
    assert len(fresh) == 1
    assert "TP2\0L2" in engine.active_intents
    assert b.close_calls == []


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


def __test_emulated_filled_entry_side_flip_dispatches_reversal_not_modify__():
    """A same-Pine-ID opposite-side re-entry after the market entry FILLED is a
    fresh entry cycle, never a broker amend of the consumed order.

    ``strategy.entry("pos", long)`` fills; the consumed market entry stays in
    ``_active_intents`` as the sticky diff sentinel. ``strategy.entry("pos",
    short)`` then collides on the same ``intent_key`` with a different side —
    before the fix the diff routed it through ``_dispatch_modify`` →
    ``modify_entry``, amending an already-FILLED market order (cTrader rejects
    that with ORDER_NOT_FOUND). The dispatch must instead flow through
    ``_dispatch_new`` — the one-way reversal planner on a hedging plugin —
    FIFO-closing the long leg and opening the residual short.
    """
    from pynecore.core.broker.models import EntryIntent
    b = MockBroker()
    b.position_port = b
    engine, pos = _mk_engine(b)
    pos.entry_orders["pos"] = _entry_order("pos", 1000.0)

    engine.sync(BAR_TS)
    assert b.place_leg_calls == [1000.0]  # pure add: long opened via the port
    engine._route_event(_fill_event('buy', 1000.0, 1.0, pine_id="pos"))
    assert pos.size == 1000.0

    # Pine reversal: same ID, opposite side; the broker now holds the long leg.
    pos.entry_orders["pos"] = _entry_order("pos", -1000.0)
    b.raw_legs = [_pleg("7", "buy", 1000.0, open_time=1.0)]
    engine.sync(BAR_TS + 60_000)

    assert b.modify_entry_calls == []  # a filled market order is not amendable
    assert b.close_leg_calls == [("7", 1000)]  # long leg FIFO-closed
    # Stop-and-reverse fold: 1000 raw + 1000 long = 2000 combined; the
    # reversal closes 1000 and opens the residual 1000 short.
    assert b.place_leg_calls == [1000.0, 1000.0]
    active = engine.active_intents["pos"]
    assert isinstance(active, EntryIntent)
    assert active.side == 'sell'


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
                               pine_id="X", from_entry="L",
                               oca_name=None, oca_type=None):
    from pynecore.core.broker.store_helpers import create_bracket_ownership_row
    create_bracket_ownership_row(
        ctx, coid=f"bo-test:{leg_id}", symbol=SYMBOL, side="sell", qty=2.0,
        intent_key=intent_key, pine_entry_id=pine_id, from_entry=from_entry,
        leg_id=leg_id, attach_coid="t-attach",
        tp_price=120.0, sl_price=90.0, trail_price=None, trail_offset=None,
        oca_name=oca_name, oca_type=oca_type,
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


def __test_restart_reconstructs_one_way_bracket_and_adopts__(tmp_path):
    """A restart rebuilds a persisted one-way bracket the script no longer re-emits, and adopts it.

    Pine ``strategy.exit`` orders persist across bars without re-emission, so a
    common script attaches the bracket only on the entry bar. After a restart
    with the position adopted, the entry bar never re-fires — but the bracket
    must NOT be torn down. The first sync reconstructs the Pine-side exit from
    the persisted ownership ledger, the diff adopts it, and the live broker
    protection is preserved (re-asserted, never cleared).
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import iter_active_bracket_ownerships
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_bracket_ownership(ctx)
        b = MockBroker()
        b.position_port = b
        b.raw_legs = [_pleg("1", "buy", 2.0)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b, position=pos, symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        engine.sync(BAR_TS)  # Pine emits nothing; reconstruction + adoption run
        assert ("1", None, None) not in b.amend_calls  # never cleared to None
        assert ("1", 120.0, 90.0) in b.amend_calls  # re-asserted by restart_replay
        assert ("X", "L") in pos.exit_orders  # Pine-side exit rebuilt
        assert engine._order_mapping.get("X\0L") == ["bracket:1"]
        assert "X\0L" in engine.active_intents  # adopted, not re-dispatched
        assert any(r.intent_key == "X\0L"
                   for r in iter_active_bracket_ownerships(ctx))  # still protected


def __test_first_bar_cancel_survives_restart_settle__(tmp_path):
    """settle_restart_state rebuilds the bracket BEFORE the first-bar script, so a
    first-bar strategy.cancel takes effect instead of being overwritten.

    The bar-close branch runs the script before sync. After a restart the Pine
    order dicts start empty, so a first-bar cancel would no-op against an empty
    ``exit_orders`` and the post-script sync would then reconstruct the bracket,
    silently resurrecting the exit the user just cancelled. Running
    reconstruction in :meth:`settle_restart_state` (before the script) lets the
    cancel see — and remove — the rebuilt exit; the post-script sync no longer
    reconstructs (the one-time flag latched in settle), so the cancel survives
    and the diff clears the live broker bracket.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import iter_active_bracket_ownerships
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_bracket_ownership(ctx)
        b = MockBroker()
        b.position_port = b
        b.raw_legs = [_pleg("1", "buy", 2.0)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b, position=pos, symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        # Fresh process: the Pine order book is empty until reconstruction.
        assert ("X", "L") not in pos.exit_orders
        # settle runs BEFORE the first-bar script -> rebuilds the exit so the
        # script's cancel can act on a populated book.
        engine.settle_restart_state(BAR_TS)
        assert ("X", "L") in pos.exit_orders  # reconstructed pre-script
        assert engine._pine_bracket_reconstruct_done is True  # type: ignore[attr-defined]
        # The first-bar script issues strategy.cancel -> the exit is removed.
        del pos.exit_orders[("X", "L")]
        # Post-script sync: reconstruction is one-time (already done in settle),
        # so it does NOT resurrect the cancelled exit; the diff clears the leg.
        engine.sync(BAR_TS)
        assert ("X", "L") not in pos.exit_orders  # cancel NOT overwritten
        assert ("1", None, None) in b.amend_calls  # live bracket cleared
        assert list(iter_active_bracket_ownerships(ctx)) == []  # row released


def _seed_live_entry_order_row(ctx, *, coid: str, order_id: str, pine_id: str,
                               side: str = "buy", qty: float = 1.0) -> None:
    """Journal a live entry working-order row the way a plugin's ``_persist_entry``
    would — keyed by ``pine_entry_id`` with the broker order id, no ``from_entry``
    (a bare entry, not a bracket leg). The restart entry reconstruction reads this
    to reverse-map the live order back to its Pine id."""
    ctx.upsert_order(
        coid, symbol=SYMBOL, side=side, qty=qty, state='confirmed',
        intent_key=pine_id, pine_entry_id=pine_id,
        exchange_order_id=order_id, extras={'order_id': order_id},
    )


def _live_stop_entry_order(coid: str, *, order_id: str, stop: float,
                           side: str = "buy") -> ExchangeOrder:
    """A broker-side OPEN STOP entry working order carrying ``coid`` — the shape
    ``get_open_orders`` surfaces for a resting ``strategy.entry(..., stop=X)``."""
    return ExchangeOrder(
        id=order_id, symbol=SYMBOL, side=side,
        order_type=OrderType.STOP, qty=1.0, filled_qty=0.0,
        remaining_qty=1.0, price=None, stop_price=stop,
        average_fill_price=None, status=OrderStatus.OPEN,
        timestamp=0.0, fee=0.0, fee_currency="",
        client_order_id=coid,
    )


def __test_cancel_only_restart_retires_reconstructed_entry__(tmp_path):
    """A restart whose script goes straight to ``strategy.cancel`` (no re-declare)
    retires the live entry working order instead of stranding it.

    The reported cTrader stall: phase A placed a distant STOP entry and stopped
    cleanly; phase B (same run identity) reloads its persisted phase and issues
    ``strategy.cancel`` WITHOUT re-declaring the entry. Because a fresh process
    starts with an empty Pine order book, the cancel used to no-op and the live
    order was neither adopted nor cancelled — :meth:`_hydrate_restart_entry_adoptions`
    only binds an entry the script RE-declares. Reconstructing the working order
    pre-script (in :meth:`settle_restart_state`) gives the cancel a target: the
    diff's entry-orphan sweep retires the live broker order.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    live_coid = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=0,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        _seed_live_entry_order_row(ctx, coid=live_coid, order_id="wo-1", pine_id="L")
        b = MockBroker()
        b.open_orders = [_live_stop_entry_order(live_coid, order_id="wo-1", stop=1.30000)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        # Fresh process: the Pine order book is empty until reconstruction.
        assert "L" not in pos.entry_orders
        # settle runs BEFORE the first-bar script -> rebuilds the entry so the
        # script's cancel can act on a populated book.
        engine.settle_restart_state(BAR_TS)
        assert "L" in pos.entry_orders  # reconstructed pre-script
        assert pos.entry_orders["L"].stop == 1.30000  # STOP level restored
        assert engine._order_mapping["L"] == ["wo-1"]  # type: ignore[attr-defined]

        # The first-bar script issues strategy.cancel -> the entry is removed
        # and NOT re-declared.
        del pos.entry_orders["L"]
        engine.sync(BAR_TS)

        # The reconstructed entry's live order is cancelled at the venue; no
        # duplicate entry was ever dispatched.
        assert len(b.entry_calls) == 0
        assert len(b.cancel_calls) == 1
        assert "L" not in engine._order_mapping  # type: ignore[attr-defined]
        assert engine._restart_reconstructed_entry_keys == {}  # type: ignore[attr-defined]


def __test_cancel_only_restart_kept_entry_is_adopted_not_cancelled__(tmp_path):
    """A restart that reconstructs a working order the script LEAVES STANDING adopts
    it — no duplicate dispatch, no spurious cancel.

    The counterpart to the cancel-only case: a persistent Pine entry the script
    does not re-emit (nor cancel) must survive the restart. Reconstruction seeds
    the order book and mapping; the first post-restart diff routes it through the
    cross-restart adoption branch, which pins the live order as the active intent
    and dispatches nothing. The entry-orphan sweep must NOT fire.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    live_coid = build_client_order_id(
        run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=0,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t025.py")
        _seed_live_entry_order_row(ctx, coid=live_coid, order_id="wo-1", pine_id="L")
        b = MockBroker()
        b.open_orders = [_live_stop_entry_order(live_coid, order_id="wo-1", stop=1.30000)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        engine.settle_restart_state(BAR_TS)
        assert "L" in pos.entry_orders  # reconstructed pre-script

        # The first-bar script leaves the entry standing (persistent Pine order).
        engine.sync(BAR_TS)

        # The live order is adopted, not re-dispatched, and not cancelled.
        assert len(b.entry_calls) == 0
        assert len(b.cancel_calls) == 0
        assert engine._order_mapping["L"] == ["wo-1"]  # type: ignore[attr-defined]
        assert "L" in engine._active_intents  # type: ignore[attr-defined]
        assert engine._restart_reconstructed_entry_keys == {}  # type: ignore[attr-defined]


def __test_settle_restart_state_skips_when_already_reconstructed__(tmp_path):
    """Once reconstruction has latched, settle_restart_state is an immediate no-op.

    Guards the steady-state contract: the script runner calls
    :meth:`settle_restart_state` every bar before the script, and once the
    one-time restart reconstruction is done it must do zero work — no replay, no
    reconstruction, no ``get_open_orders`` — so steady-state bars pay nothing and
    the per-bar ``verify`` runs only in :meth:`sync`.
    """
    from pynecore.core.broker.storage import BrokerStore
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_bracket_ownership(ctx)  # a reconstructable bracket exists in the ledger
        b = MockBroker()
        b.position_port = b
        b.raw_legs = [_pleg("1", "buy", 2.0)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b, position=pos, symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        engine._pine_bracket_reconstruct_done = True  # type: ignore[attr-defined]
        engine.settle_restart_state(BAR_TS)
        # Early-returned: the ledger bracket was NOT reconstructed and the live
        # leg was NOT re-asserted.
        assert ("X", "L") not in pos.exit_orders
        assert b.amend_calls == []


def __test_restart_reconstruction_restores_oca_cancel_group__(tmp_path):
    """A reconstructed one-way bracket carries the OCA group it was emitted under.

    The persist-first ownership ledger records the exit's ``oca_name`` /
    ``oca_type``; without it the rebuilt Pine ``Order`` would have no OCA group,
    so an explicit ``oca_type='cancel'`` cross-bracket cascade (the engine's job
    when ``oca_cancel`` is SOFTWARE) would silently stop firing after a restart.
    Reconstruction restores both so ``build_intents`` re-derives the same group.
    """
    from pynecore.core.broker.storage import BrokerStore
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_bracket_ownership(ctx, oca_name="G", oca_type="cancel")
        b = MockBroker()
        b.position_port = b
        b.raw_legs = [_pleg("1", "buy", 2.0)]
        b.capabilities = ExchangeCapabilities(oca_cancel=CapabilityLevel.SOFTWARE)
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b, position=pos, symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        engine.sync(BAR_TS)
        order = pos.exit_orders[("X", "L")]
        assert order.oca_name == "G"  # group name restored on the Pine Order
        assert str(order.oca_type) == "cancel"  # cancel type restored
        # build_intents re-derives the same group on the adopted intent.
        intent = engine.active_intents["X\0L"]
        assert intent.oca_name == "G"
        assert intent.oca_type == "cancel"


def __test_restart_reconstruction_without_oca_metadata_is_groupless__(tmp_path):
    """A row persisted before the OCA keys existed reconstructs as a groupless exit.

    Graceful-degradation guard: an old ownership row carries no ``oca_name`` /
    ``oca_type``, so reconstruction leaves the rebuilt exit without an OCA group
    (a single-member synthetic reduce group is a cascade no-op, so the only thing
    ever lost is an explicit group) — and never raises over the missing keys.
    """
    from pynecore.core.broker.storage import BrokerStore
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_bracket_ownership(ctx)  # no oca_name / oca_type
        b = MockBroker()
        b.position_port = b
        b.raw_legs = [_pleg("1", "buy", 2.0)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b, position=pos, symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        engine.sync(BAR_TS)
        order = pos.exit_orders[("X", "L")]
        assert order.oca_name is None  # no group on a pre-OCA-keys row
        assert engine.active_intents["X\0L"].oca_name is None


def __test_restart_one_way_multi_leg_grouped_into_one_order__(tmp_path):
    """A bracket replicated onto several hedged legs rebuilds as ONE exit, mapping every leg."""
    # The emulator writes one ownership row per position-side leg, all sharing
    # the exit's intent_key with identical levels. Reconstruction must collapse
    # them into a single Pine exit Order and seed _order_mapping with every leg.
    from pynecore.core.broker.storage import BrokerStore
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_bracket_ownership(ctx, leg_id="1")
        _persist_bracket_ownership(ctx, leg_id="2")
        _persist_bracket_ownership(ctx, leg_id="3")
        b = MockBroker()
        b.position_port = b
        b.raw_legs = [_pleg("1", "buy", 2.0), _pleg("2", "buy", 2.0),
                      _pleg("3", "buy", 2.0)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b, position=pos, symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        engine.sync(BAR_TS)
        assert len([k for k in pos.exit_orders if k == ("X", "L")]) == 1  # one Order
        assert sorted(engine._order_mapping["X\0L"]) == [
            "bracket:1", "bracket:2", "bracket:3"]
        assert all(("1", None, None) != c and ("2", None, None) != c
                   and ("3", None, None) != c for c in b.amend_calls)  # no clears


def __test_restart_reconstructed_bracket_modify_not_duplicate__(tmp_path):
    """After a restart adopts a bracket, a changed TP on the next bar amends — not re-attaches."""
    # Reconstruction + adoption pins the intent in _active_intents, so the normal
    # diff handles a subsequent level change as a modify of the live bracket
    # rather than a fresh duplicate attach.
    from pynecore.core.broker.storage import BrokerStore
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_bracket_ownership(ctx)
        b = MockBroker()
        b.position_port = b
        b.raw_legs = [_pleg("1", "buy", 2.0)]
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b, position=pos, symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        engine.sync(BAR_TS)  # adopt
        b.amend_calls.clear()
        # The script now emits the same exit with a raised TP on the next bar.
        pos.exit_orders[("X", "L")] = _exit_order("L", -2.0, "X", limit=130.0, stop=90.0)
        engine.sync(BAR_TS + 60_000)
        # The live leg's bracket is amended to the new TP; never cleared to None.
        assert ("1", 130.0, 90.0) in b.amend_calls
        assert ("1", None, None) not in b.amend_calls


def _persist_partial_leg(ctx, *, leg_kind, leg_state="armed",
                         intent_key="X\0L", pine_id="X", from_entry="L",
                         qty=0.4, intent_partial_qty=0.4, trigger_level=None,
                         trigger_offset=None, trail_activation_level=None,
                         oca_group=None, oca_type=None,
                         parent_entry_dispatch_ref="parent-ref"):
    from pynecore.core.broker.store_helpers import (
        create_engine_trigger_partial_leg_row,
    )
    create_engine_trigger_partial_leg_row(
        ctx, coid=f"pl-test:{pine_id}:{from_entry}:{leg_kind}", symbol=SYMBOL,
        side="sell", qty=qty, intent_key=intent_key, pine_entry_id=pine_id,
        from_entry=from_entry, leg_kind=leg_kind, leg_state=leg_state,
        parent_pine_entry_id=from_entry,
        parent_entry_dispatch_ref=parent_entry_dispatch_ref,
        intent_partial_qty=intent_partial_qty, trigger_level=trigger_level,
        trigger_offset=trigger_offset,
        trail_activation_level=trail_activation_level,
        oca_group=oca_group, oca_type=oca_type,
    )


def _software_partial_broker():
    b = MockBroker()
    b.position_port = b
    b.capabilities = ExchangeCapabilities(
        partial_qty_bracket_exit=CapabilityLevel.SOFTWARE,
    )
    return b


def __test_recover_adopted_parent_entry_id_from_partial_legs__(tmp_path):
    """Startup parent-id recovery folds in the partial-leg ledger, not only one-way rows.

    A partial-only restart has no one-way ownership rows, so without reading the
    partial-leg ledger the seeded parent trade keeps the synthetic id and the
    exit's ``from_entry`` finds no match — ``build_intents`` then reads a zero
    parent total and misclassifies the bracket as whole-row.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import (
        LEG_KIND_TP_PARTIAL, LEG_KIND_SL_PARTIAL,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_partial_leg(ctx, leg_kind=LEG_KIND_TP_PARTIAL, trigger_level=120.0)
        _persist_partial_leg(ctx, leg_kind=LEG_KIND_SL_PARTIAL, trigger_level=90.0)
        engine = OrderSyncEngine(
            broker=_software_partial_broker(), position=BrokerPosition(),  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        assert engine._recover_adopted_parent_entry_id() == "L"  # type: ignore[attr-defined]


def __test_restart_reconstructs_partial_bracket_exit__(tmp_path):
    """The replayed partial legs rebuild a single Pine exit Order in exit_orders.

    Mirrors the one-way reconstruction: the in-memory leg ledger that
    ``restart_replay`` rebuilds is invisible to ``build_intents`` until the
    Pine-side exit is re-installed. The TP/SL trigger levels, the partial qty,
    and the OCA group are restored onto one combined exit.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import (
        LEG_KIND_TP_PARTIAL, LEG_KIND_SL_PARTIAL,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_partial_leg(ctx, leg_kind=LEG_KIND_TP_PARTIAL, trigger_level=120.0,
                             oca_group="__partial_exit_X_L__", oca_type="cancel")
        _persist_partial_leg(ctx, leg_kind=LEG_KIND_SL_PARTIAL, trigger_level=90.0,
                             oca_group="__partial_exit_X_L__", oca_type="cancel")
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=_software_partial_broker(), position=pos,  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        engine._reconstruct_partial_bracket_exits()  # type: ignore[attr-defined]
        order = pos.exit_orders[("X", "L")]
        assert abs(order.size) == 0.4  # partial qty, not the parent total
        assert order.limit == 120.0  # tp from the TP leg's trigger_level
        assert order.stop == 90.0  # sl from the SL leg's trigger_level
        assert order.oca_name == "__partial_exit_X_L__"
        assert str(order.oca_type) == "cancel"


def __test_restart_reconstructs_partial_active_trail_leg__(tmp_path):
    """An active trail leg re-derives as a TRAIL leg (trail_price falls back to the moving stop).

    A pre-activation trail carries ``trail_activation_level``; an already-active
    trail has cleared it and tracks the live moving stop in ``trigger_level``.
    Either way the rebuilt Order must keep a non-None ``trail_price`` so
    ``_enumerate_engine_trigger_legs`` emits a trail leg and the adoption
    branch's leg-kind completeness check matches (the value is inert once the
    live leg is adopted).
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import LEG_KIND_TRAIL_PARTIAL
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_partial_leg(ctx, leg_kind=LEG_KIND_TRAIL_PARTIAL,
                             trigger_level=95.0, trigger_offset=5.0,
                             trail_activation_level=None)
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=_software_partial_broker(), position=pos,  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        engine._reconstruct_partial_bracket_exits()  # type: ignore[attr-defined]
        order = pos.exit_orders[("X", "L")]
        assert order.trail_price == 95.0  # active trail -> live moving stop
        assert order.trail_offset == 5.0


def __test_restart_partial_reconstruction_skips_pending_entry_group__(tmp_path):
    """A group still pending its parent entry is not reconstructed (no open position yet)."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import (
        LEG_KIND_TP_PARTIAL, LEG_KIND_SL_PARTIAL,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_partial_leg(ctx, leg_kind=LEG_KIND_TP_PARTIAL,
                             leg_state="pending_entry", trigger_offset=20.0)
        _persist_partial_leg(ctx, leg_kind=LEG_KIND_SL_PARTIAL,
                             leg_state="pending_entry", trigger_offset=10.0)
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=_software_partial_broker(), position=pos,  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        engine._reconstruct_partial_bracket_exits()  # type: ignore[attr-defined]
        assert ("X", "L") not in pos.exit_orders


def __test_restart_partial_reconstruction_noop_when_not_software__(tmp_path):
    """Reconstruction only runs in the SOFTWARE partial mode; default UNSUPPORTED is a no-op."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.store_helpers import (
        LEG_KIND_TP_PARTIAL, LEG_KIND_SL_PARTIAL,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        _persist_partial_leg(ctx, leg_kind=LEG_KIND_TP_PARTIAL, trigger_level=120.0)
        _persist_partial_leg(ctx, leg_kind=LEG_KIND_SL_PARTIAL, trigger_level=90.0)
        b = MockBroker()  # default ExchangeCapabilities -> partial_qty_bracket_exit UNSUPPORTED
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b, position=pos, symbol=SYMBOL,  # type: ignore[arg-type]
            run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        engine._reconstruct_partial_bracket_exits()  # type: ignore[attr-defined]
        assert ("X", "L") not in pos.exit_orders


def __test_restart_partial_bracket_adopted_not_swept__(tmp_path):
    """End-to-end: the first post-restart sync reconstructs, classifies partial, and ADOPTS the legs.

    Without reconstruction the orphan-leg sweep in :meth:`_diff_and_dispatch`
    would cancel the replayed legs (their intent_key is in neither
    ``_active_intents`` nor new_map). Reconstruction makes ``build_intents``
    re-derive the partial bracket; the parent ref matches the replayed legs and
    the leg set is complete, so the adoption branch pins the intent and the live
    legs survive instead of being torn down.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY
    from pynecore.core.broker.store_helpers import (
        LEG_KIND_TP_PARTIAL, LEG_KIND_SL_PARTIAL,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        parent_ref = build_client_order_id(
            run_tag=RUN_TAG, pine_id="L", bar_ts_ms=BAR_TS,
            kind=KIND_ENTRY, retry_seq=0,
        )
        # Persist the parent-entry envelope anchor so the engine's replay rebuilds
        # the SAME coid the legs were stamped with -> _resolve_parent_opening_ref
        # matches and the adoption branch does not treat the legs as stale.
        ctx.record_envelope("L", BAR_TS, 0)
        _persist_partial_leg(ctx, leg_kind=LEG_KIND_TP_PARTIAL, trigger_level=120.0,
                             parent_entry_dispatch_ref=parent_ref,
                             oca_group="__partial_exit_X_L__", oca_type="cancel")
        _persist_partial_leg(ctx, leg_kind=LEG_KIND_SL_PARTIAL, trigger_level=90.0,
                             parent_entry_dispatch_ref=parent_ref,
                             oca_group="__partial_exit_X_L__", oca_type="cancel")
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=_software_partial_broker(), position=pos,  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        # Adopted parent: open total 1.0 > partial 0.4 -> is_partial_qty_bracket.
        pos.size = 1.0
        pos.reconstruct_parent_trade(entry_id="L", size=1.0, entry_price=100.0)
        engine.sync(BAR_TS)
        assert ("X", "L") in pos.exit_orders  # Pine-side exit rebuilt
        assert "X\0L" in engine.active_intents  # adopted, not re-dispatched
        # The replayed legs survive the orphan sweep.
        assert engine._partial_bracket_engine.has_active_legs_for_intent("X\0L")  # type: ignore[attr-defined]


def __test_restart_multi_parent_partial_brackets_adopted_not_converted__(tmp_path):
    """A pyramided (multi-parent) restart adopts every partial bracket, never converting one.

    Startup adoption of a position opened under several distinct ``from_entry``
    parents seeds a single synthetic ``__adopted_startup__`` trade (the parent id
    cannot be collapsed into one), so each real ``from_entry`` carries no
    ``open_trades`` row and ``build_intents`` reads ``parent_total_qty == 0`` —
    misclassifying every partial bracket as a whole-row exit. Left uncorrected the
    dispatch else-branch fires the ``partial_to_whole_row_conversion`` cleanup and
    cancels the replayed legs, rewriting the live software protection. The leg
    ledger is authoritative: ``_restore_adopted_partial_bracket_classification``
    re-flags the exits as partial so the adoption branch pins each one and the
    live legs of BOTH parents survive.
    """
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY
    from pynecore.core.broker.store_helpers import (
        LEG_KIND_TP_PARTIAL, LEG_KIND_SL_PARTIAL,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src", script_path="t.py")
        for parent in ("L1", "L2"):
            ref = build_client_order_id(
                run_tag=RUN_TAG, pine_id=parent, bar_ts_ms=BAR_TS,
                kind=KIND_ENTRY, retry_seq=0,
            )
            ctx.record_envelope(parent, BAR_TS, 0)
            for leg_kind, level in (
                (LEG_KIND_TP_PARTIAL, 120.0), (LEG_KIND_SL_PARTIAL, 90.0),
            ):
                _persist_partial_leg(
                    ctx, leg_kind=leg_kind, trigger_level=level,
                    intent_key=f"TP\0{parent}", pine_id="TP", from_entry=parent,
                    parent_entry_dispatch_ref=ref,
                    oca_group=f"__partial_exit_TP_{parent}__", oca_type="cancel",
                )
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=_software_partial_broker(), position=pos,  # type: ignore[arg-type]
            symbol=SYMBOL, run_tag=RUN_TAG, mintick=1.0, store_ctx=ctx,
        )
        # Multi-parent adoption result: the net size lives under one synthetic
        # trade, so neither real from_entry has an open_trades row and
        # build_intents classifies both partial brackets as whole-row.
        pos.size = 2.0
        pos.reconstruct_parent_trade(
            entry_id="__adopted_startup__", size=2.0, entry_price=100.0,
        )
        engine.sync(BAR_TS)
        for parent in ("L1", "L2"):
            key = f"TP\0{parent}"
            assert ("TP", parent) in pos.exit_orders  # Pine-side exit rebuilt
            assert key in engine.active_intents  # adopted, not converted
            # The replayed legs survive — no partial_to_whole_row_conversion.
            assert engine._partial_bracket_engine.has_active_legs_for_intent(key)  # type: ignore[attr-defined]


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


def __test_partially_filled_inflight_close_reserves_only_working_qty_for_close_all__():
    """A same-evaluation ``close_all`` covers the full residual past a PARTIALLY filled close.

    ``strategy.close("L", qty=5)`` dispatches a 5-unit market close against a
    10-unit long; it is on the wire (``_active_intents['L']``). The broker then
    PARTIALLY fills 3 of those 5: ``record_fill`` drops ``position.size`` to 7 and
    shrinks ``open_trades`` but never shrinks the active 5-unit ``CloseIntent``.
    The next evaluation keeps ``close("L")`` and adds ``strategy.close_all()``.

    The diff-loop guard skips re-dispatching the in-flight close (a market close
    cannot be cancelled / re-dispatched), so the clamp must reserve only the
    still-WORKING 2 units (``active.qty 5 - filled 3``) — NOT the full active 5.
    Reserving 5 would double-debit the 3 already filled (``position.size`` is
    already net of them), leaving ``close_all`` only 2 instead of the 5 it must
    flatten and stranding 3 units of live exposure. Coverage must equal the
    7-unit residual: 2 (in-flight working) + 5 (close_all).
    """
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 10.0
    pos.sign = 1.0
    pos.open_trades = [_long_trade("L", 10.0)]
    # Sync 1: close("L", qty=5) — a 5-unit slice goes on the wire.
    pos.exit_orders[("Close entry(s) order L", "L")] = Order(
        "L", -5.0, order_type=_order_type_close, exit_id="Close entry(s) order L",
    )
    engine.sync(BAR_TS)
    assert len(b.close_calls) == 1
    assert b.close_calls[0].intent.pine_id == "L"
    assert b.close_calls[0].intent.qty == 5.0

    # Broker PARTIALLY fills 3 of the 5-unit keyed close (position 10 -> 7),
    # routed through the real fill path so the engine accumulates the close-fill
    # ledger and ``record_fill`` shrinks the FIFO.
    partial = OrderEvent(
        order=ExchangeOrder(
            id="xchg-1", symbol=SYMBOL, side="sell",
            order_type=OrderType.MARKET, qty=5.0, filled_qty=3.0,
            remaining_qty=2.0, price=None, stop_price=None,
            average_fill_price=50_000.0, status=OrderStatus.PARTIALLY_FILLED,
            timestamp=0.0, fee=0.0, fee_currency="",
        ),
        event_type='partial', fill_price=50_000.0,
        fill_qty=3.0, timestamp=0.0, pine_id="L", leg_type=LegType.CLOSE,
    )
    engine._route_event(partial)
    assert pos.size == 7.0  # record_fill reduced the position by the 3 filled

    # Sync 2: script keeps close("L") and adds close_all() against the 7 residual.
    pos.exit_orders[("Close position order", None)] = Order(
        None, -7.0, order_type=_order_type_close, exit_id="Close position order",
    )
    engine.sync(BAR_TS + 60_000)

    # The keyed close is NOT re-dispatched (still the in-flight 5); close_all
    # flattens the full residual minus the 2 still working = 5.
    new_close = [c for c in b.close_calls[1:]]
    qty_by_id = {c.intent.pine_id: c.intent.qty for c in new_close}
    assert qty_by_id == {"": 5.0}  # only the close_all residual is newly dispatched
    assert b.cancel_calls == []
    assert b.modify_exit_calls == []


# === MARKET stop-and-reverse fold at dispatch ===

def __test_market_reversal_dispatch_folds_opposite_position__():
    """A fresh MARKET entry against an opposite net position dispatches the
    combined stop-and-reverse quantity (TV parity), while the active-intent
    slot keeps the RAW qty so the next bar's re-emitted Pine order still
    matches and never re-triggers the diff."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 0.0004
    pos.sign = 1.0
    pos.entry_orders["S"] = _entry_order("S", -0.0002)

    engine.sync(BAR_TS)

    assert len(b.entry_calls) == 1
    dispatched = b.entry_calls[0].intent
    assert dispatched.side == 'sell'
    assert dispatched.qty == 0.0006  # 0.0002 + |0.0004|, artifact-free
    # The registered active intent stays RAW — a re-emission of the same
    # pending order next bar must diff as unchanged.
    assert engine.active_intents["S"].qty == 0.0002
    # Second sync with the same pending order: no new dispatch.
    engine.sync(BAR_TS + 60_000)
    assert len(b.entry_calls) == 1


def __test_market_add_dispatch_keeps_raw_qty__():
    """A same-direction MARKET add dispatches the raw script quantity."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 0.0004
    pos.sign = 1.0
    pos.entry_orders["L2"] = _entry_order("L2", 0.0002)

    engine.sync(BAR_TS)

    assert len(b.entry_calls) == 1
    assert b.entry_calls[0].intent.qty == 0.0002


def __test_limit_reversal_dispatch_is_not_folded_again__():
    """LIMIT/STOP entries fold at creation (``strategy.entry`` subtracts the
    position size) — the dispatch fold must not double-count them."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.size = 2.0
    pos.sign = 1.0
    pos.entry_orders["S"] = _entry_order("S", -3.0, limit=50_000.0)

    engine.sync(BAR_TS)

    assert len(b.entry_calls) == 1
    assert b.entry_calls[0].intent.qty == 3.0


# === Quarantine ===


def __test_quarantine_blocks_new_entry_but_sync_keeps_running__():
    """Quarantine drops new entry dispatch while ``sync`` itself keeps
    running — the process stays alive, unlike the halt latch."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    engine.record_quarantine("external cancel detected")
    assert engine.quarantined is True
    assert engine.halted is False

    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)  # must not raise

    assert b.entry_calls == []
    assert "L" not in engine.active_intents
    assert "L" not in engine.order_mapping
    # The signal is dropped, not queued: a later sync of the same signal
    # is blocked again without any broker call.
    engine.sync(BAR_TS)
    assert b.entry_calls == []


def __test_quarantine_allows_exit_close_and_cancel__():
    """Risk-reducing dispatch flows under quarantine: protective exits,
    closes and cancels all reach the broker."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    # A resting entry placed BEFORE the quarantine.
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1

    engine.record_quarantine("external cancel detected")

    # Protective exit for an open trade dispatches.
    pos.size = 1.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 1.0))
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -1.0, "TP", limit=60_000.0, stop=45_000.0,
    )
    engine.sync(BAR_TS)
    assert len(b.exit_calls) == 1

    # Cancelling the resting entry dispatches too (risk-reducing).
    del pos.entry_orders["L"]
    engine.sync(BAR_TS)
    assert len(b.cancel_calls) == 1


def __test_quarantine_blocks_entry_modify_and_keeps_old_intent__():
    """An entry amend under quarantine makes no broker call and keeps the
    OLD intent active, staying in sync with the still-resting order."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1

    engine.record_quarantine("external cancel detected")
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=49_500.0)
    engine.sync(BAR_TS)

    assert b.modify_entry_calls == []
    assert engine.active_intents["L"].limit == 50_000.0


def __test_record_quarantine_is_idempotent_and_emits_once__():
    """The latch emits exactly one ``QuarantineEnteredEvent``."""
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, _pos = _mk_engine_with_sink(b, events)
    engine.record_quarantine("first reason", {'origin': 'test'})
    engine.record_quarantine("second reason")

    entered = [e for e in events if isinstance(e, QuarantineEnteredEvent)]
    assert len(entered) == 1
    assert entered[0].reason == "first reason"
    assert entered[0].context == {'origin': 'test'}
    assert engine.quarantined is True


def __test_record_quarantine_concurrent_callers_latch_once__():
    """Concurrent latch attempts (the sink is called from the broker
    event-loop thread while the main thread reads the gates) latch and emit
    exactly once, and the winner's reason/context are internally consistent."""
    b = MockBroker()
    events: list[BrokerEvent] = []
    engine, _pos = _mk_engine_with_sink(b, events)
    n = 8
    barrier = threading.Barrier(n)

    def _caller(i: int) -> None:
        barrier.wait()
        engine.record_quarantine(f"reason-{i}", {'origin': i})

    threads = [threading.Thread(target=_caller, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    entered = [e for e in events if isinstance(e, QuarantineEnteredEvent)]
    assert len(entered) == 1
    assert engine.quarantined is True
    # The emitted event carries the SAME winner's reason/context the latch
    # holds — never a mix of two callers.
    winner = entered[0].reason.removeprefix("reason-")
    assert entered[0].context == {'origin': int(winner)}


# === Push-detected unexpected cancel policy ===


def _cancelled_event(
        deal_id: str, *, pine_id: str | None = "L",
        from_entry: str | None = None, filled_qty: float = 0.0,
        from_disappearance_tracker: bool = False,
) -> OrderEvent:
    """A venue ``cancelled`` status event for a mapped bot order.

    ``from_disappearance_tracker`` mimics the marker the core
    :class:`DisappearanceTracker` stamps on its own synthesised events.
    """
    return OrderEvent(
        order=ExchangeOrder(
            id=deal_id, symbol=SYMBOL, side='buy',
            order_type=OrderType.MARKET, qty=1.0, filled_qty=filled_qty,
            remaining_qty=max(0.0, 1.0 - filled_qty), price=None,
            stop_price=None, average_fill_price=None,
            status=OrderStatus.CANCELLED, timestamp=0.0, fee=0.0,
            fee_currency="", client_order_id="coid-L",
        ),
        event_type='cancelled', fill_price=None, fill_qty=None,
        timestamp=0.0, pine_id=pine_id, from_entry=from_entry,
        from_disappearance_tracker=from_disappearance_tracker,
    )


def __test_push_external_cancel_stop_quarantines_and_blocks_replace__():
    """A venue-pushed cancel of a still-mapped bot order latches the
    quarantine under 'stop' and suppresses the strategy's next-bar
    re-dispatch — the fix for the operator-vs-bot re-place duel."""
    b = MockBroker()  # on_unexpected_cancel == "stop"
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1
    deal_id = engine.order_mapping["L"][0]

    engine._route_event(_cancelled_event(deal_id))

    assert engine.quarantined is True
    assert engine.halted is False
    assert "L" not in engine.order_mapping
    assert "L" not in engine.active_intents

    # The Pine book still carries L, but the next sync must NOT re-place it:
    # the quarantine gate blocks the re-dispatch, so no duel.
    engine.sync(BAR_TS + 60_000)
    assert len(b.entry_calls) == 1


def __test_native_cancel_all_expected_no_quarantine__():
    """A plugin native bulk cancel (``execute_cancel_all``) arms the engine's
    expected-cancel set via ``enqueue_native_cancel_all_expected`` BEFORE the
    venue call, so the follow-up ``CANCELLED`` pushes retire the mapped orders
    cleanly instead of tripping the ``on_unexpected_cancel`` quarantine."""
    b = MockBroker()  # on_unexpected_cancel == "stop"
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.entry_orders["L2"] = _entry_order("L2", 1.0, limit=49_000.0)
    engine.sync(BAR_TS)
    deal_l = engine.order_mapping["L"][0]
    deal_l2 = engine.order_mapping["L2"][0]

    # The plugin arms the expected-cancel set (marker rides the event queue),
    # then the venue pushes CANCELLED for both bulk-cancelled orders.
    engine.enqueue_native_cancel_all_expected(SYMBOL)
    engine._event_queue.put(_cancelled_event(deal_l, pine_id="L"))
    engine._event_queue.put(_cancelled_event(deal_l2, pine_id="L2"))
    engine._drain_events()

    assert engine.quarantined is False
    assert engine.halted is False
    assert "L" not in engine.order_mapping
    assert "L2" not in engine.order_mapping
    assert "L" not in engine.active_intents
    assert "L2" not in engine.active_intents


def __test_native_cancel_all_expected_is_precise_one_shot__():
    """The expected-cancel arm snapshots only the orders mapped at marker time.
    A DIFFERENT order cancelled out from under the bot right after the bulk
    cancel is a genuine external cancel and must still quarantine — the arm is
    a precise per-id, one-shot latch, not a blanket symbol-wide suppression."""
    b = MockBroker()  # on_unexpected_cancel == "stop"
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    deal_l = engine.order_mapping["L"][0]

    # Bulk cancel arms + confirms L cleanly.
    engine.enqueue_native_cancel_all_expected(SYMBOL)
    engine._event_queue.put(_cancelled_event(deal_l, pine_id="L"))
    engine._drain_events()
    assert engine.quarantined is False
    assert "L" not in engine.order_mapping

    # A NEW resting entry placed after the bulk cancel is not in the arm set;
    # an external cancel of it must still fire the quarantine.
    pos.entry_orders["L3"] = _entry_order("L3", 1.0, limit=48_000.0)
    engine.sync(BAR_TS + 60_000)
    deal_l3 = engine.order_mapping["L3"][0]
    engine._route_event(_cancelled_event(deal_l3, pine_id="L3"))

    assert engine.quarantined is True
    assert "L3" not in engine.order_mapping


def __test_push_external_cancel_halt_raises_gracefully__():
    """Under 'halt' a push-detected external cancel records the halt and
    raises :class:`UnexpectedCancelError` out of the event-application
    path so the engine performs its graceful stop."""
    b = MockBroker()
    b.on_unexpected_cancel = "halt"
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    deal_id = engine.order_mapping["L"][0]

    with pytest.raises(UnexpectedCancelError):
        engine._route_event(_cancelled_event(deal_id))

    assert engine.halted is True
    assert engine.quarantined is False
    assert "L" not in engine.order_mapping


def __test_engine_initiated_cancel_does_not_trigger_policy__():
    """A cancel the engine itself dispatched (the strategy dropped the
    order) pops the mapping first, so the venue's later cancelled push
    lands in the 'external cancel observed' branch and never triggers the
    policy."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    deal_id = engine.order_mapping["L"][0]

    # Strategy drops L -> the engine dispatches its OWN cancel, popping the
    # mapping before any stream event for it arrives.
    del pos.entry_orders["L"]
    engine.sync(BAR_TS + 60_000)
    assert len(b.cancel_calls) == 1
    assert "L" not in engine.order_mapping

    # The venue now confirms that engine-initiated cancel on the stream.
    engine._route_event(_cancelled_event(deal_id))

    assert engine.quarantined is False
    assert engine.halted is False


def __test_tracker_synthesized_cancel_does_not_reapply_policy__():
    """A cancelled event carrying ``from_disappearance_tracker`` tears down
    the mapping but does NOT re-run the policy — the tracker already
    applied it before emitting the event (no double quarantine / halt)."""
    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    deal_id = engine.order_mapping["L"][0]

    engine._route_event(
        _cancelled_event(deal_id, from_disappearance_tracker=True),
    )

    # Teardown still ran (the tracker never touches ``_order_mapping``)...
    assert "L" not in engine.order_mapping
    # ...but the engine did NOT re-apply the policy.
    assert engine.quarantined is False
    assert engine.halted is False


def __test_push_external_cancel_stop_and_cancel_sweeps_siblings__():
    """Under 'stop_and_cancel' a push-detected external cancel latches the
    quarantine AND best-effort cancels the remaining bot-owned working
    orders — the engine-side analogue of the tracker's sibling sweep, which
    never runs on a reliable-push venue."""
    b = MockBroker()
    b.on_unexpected_cancel = "stop_and_cancel"
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.entry_orders["M"] = _entry_order("M", 1.0, limit=49_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 2
    deal_id = engine.order_mapping["L"][0]

    engine._route_event(_cancelled_event(deal_id))

    assert engine.quarantined is True
    assert engine.halted is False
    assert "L" not in engine.order_mapping
    # The sibling's working order received a best-effort broker cancel...
    assert len(b.cancel_calls) == 1
    assert b.cancel_calls[0].intent.pine_id == "M"
    # ...and its slot stays active (tracker-path parity), so the unchanged
    # Pine book diffs to no-op instead of re-placing the swept order.
    assert "M" in engine.active_intents
    engine.sync(BAR_TS + 60_000)
    assert len(b.entry_calls) == 2


def __test_push_external_cancel_stop_and_cancel_spares_filled_entry__():
    """The stop_and_cancel sweep leaves an entry with filled exposure for
    the operator — cancelling its residual working order would strand real
    broker exposure without tracking."""
    b = MockBroker()
    b.on_unexpected_cancel = "stop_and_cancel"
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    pos.entry_orders["M"] = _entry_order("M", 2.0, limit=49_000.0)
    engine.sync(BAR_TS)
    deal_id = engine.order_mapping["L"][0]
    sibling_id = engine.order_mapping["M"][0]
    # M partially fills — real exposure with a residual still working.
    engine._route_event(_fill_event(
        "buy", 1.0, 49_000.0, pine_id="M", xchg_id=sibling_id,
        event_type='partial', filled_qty=1.0, remaining_qty=1.0,
    ))

    engine._route_event(_cancelled_event(deal_id))

    assert engine.quarantined is True
    # The partially filled sibling was spared by the sweep.
    assert len(b.cancel_calls) == 0
    assert "M" in engine.order_mapping


def __test_parked_modify_predecessor_cancel_is_not_unexpected__():
    """The default plugin modify is cancel + re-execute. When the
    REPLACEMENT submission parks (unknown disposition), the predecessor's
    ids stay in the mapping — its venue CANCELLED push confirms the
    engine's OWN cancel and must NOT fire the unexpected-cancel policy or
    tear down the parked verification state."""
    b = MockBroker()  # on_unexpected_cancel == "stop"
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    deal_id = engine.order_mapping["L"][0]

    # The strategy moves the level; the plugin's cancel+re-execute modify
    # cancels the predecessor, then the replacement submission times out
    # ambiguously — the dispatch parks for verification.
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=51_000.0)
    b.raise_on_next_modify_entry = OrderDispositionUnknownError(
        "replacement submit timed out", client_order_id="coid-L-replacement",
    )
    engine.sync(BAR_TS + 60_000)
    assert len(engine.pending_verification) == 1

    # The venue now pushes the CANCELLED for the predecessor the plugin
    # cancelled inside modify_entry — an engine-initiated cancel.
    engine._route_event(_cancelled_event(deal_id))

    assert engine.quarantined is False
    assert engine.halted is False
    # The stale predecessor id was trimmed from the mapping, while the
    # parked verification state survives to resolve the replacement.
    assert "L" not in engine.order_mapping
    assert "L" in engine.active_intents
    assert len(engine.pending_verification) == 1


def __test_parked_modify_promoted_live_id_cancel_fires_policy__():
    """An atomic in-place amend that parks registers its OWN live order id
    as a possibly engine-cancelled predecessor (the engine cannot tell the
    modify shapes apart at park time). Once verification promotes that
    order LIVE from ``get_open_orders`` the marker must be dropped — a
    later operator cancel of the amended order is a GENUINE external
    cancel and must fire the unexpected-cancel policy, not be consumed
    silently as the predecessor confirmation."""
    b = MockBroker()  # on_unexpected_cancel == "stop"
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    deal_id = engine.order_mapping["L"][0]

    # The in-place amend times out ambiguously — the dispatch parks and
    # the still-live order id lands in the parked-cancel ring.
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=51_000.0)
    b.raise_on_next_modify_entry = OrderDispositionUnknownError(
        "amend timed out", client_order_id="coid-L-amend",
    )
    engine.sync(BAR_TS + 60_000)
    assert len(engine.pending_verification) == 1

    # The venue's open-orders view proves the amended order LIVE under
    # the parked COID — verification promotes it and retires the marker.
    b.open_orders = [_live_working_order("coid-L-amend", order_id=deal_id)]
    engine.sync(BAR_TS + 120_000)
    assert len(engine.pending_verification) == 0
    assert deal_id in engine.order_mapping["L"]

    # An operator now cancels the amended order out from under the bot.
    engine._route_event(_cancelled_event(deal_id))

    assert engine.quarantined is True
    assert "L" not in engine.order_mapping


def __test_parked_modify_declared_atomic_amend_cancel_fires_policy__():
    """A plugin that declares the parked modify an atomic in-place amend
    (``predecessor_cancel_ids=()``) registers NOTHING in the parked-cancel
    ring — the engine issued no predecessor cancel, so a venue CANCELLED
    push during the still-unresolved park is a GENUINE external cancel and
    must fire the unexpected-cancel policy immediately, in-window."""
    b = MockBroker()  # on_unexpected_cancel == "stop"
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    deal_id = engine.order_mapping["L"][0]

    # The in-place amend times out ambiguously — the plugin declares the
    # shape: no predecessor cancel was issued.
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=51_000.0)
    b.raise_on_next_modify_entry = OrderDispositionUnknownError(
        "amend timed out", client_order_id="coid-L-amend",
        predecessor_cancel_ids=(),
    )
    engine.sync(BAR_TS + 60_000)
    assert len(engine.pending_verification) == 1

    # An operator cancels the order while the park is still unresolved.
    engine._route_event(_cancelled_event(deal_id))

    assert engine.quarantined is True
    assert engine.halted is False
    assert "L" not in engine.order_mapping
    assert "L" not in engine.active_intents


def __test_parked_modify_declared_predecessor_ids_consumed__():
    """A plugin that declares the exact predecessor ids it cancel-issued
    before the ambiguous replacement submission gets exactly those pushes
    consumed as engine-initiated — no policy, parked verification state
    survives (the declared-shape mirror of the undeclared default)."""
    b = MockBroker()  # on_unexpected_cancel == "stop"
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    deal_id = engine.order_mapping["L"][0]

    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=51_000.0)
    b.raise_on_next_modify_entry = OrderDispositionUnknownError(
        "replacement submit timed out", client_order_id="coid-L-replacement",
        predecessor_cancel_ids=(deal_id,),
    )
    engine.sync(BAR_TS + 60_000)
    assert len(engine.pending_verification) == 1

    # The venue confirms the declared engine-initiated predecessor cancel.
    engine._route_event(_cancelled_event(deal_id))

    assert engine.quarantined is False
    assert engine.halted is False
    assert "L" not in engine.order_mapping
    assert "L" in engine.active_intents
    assert len(engine.pending_verification) == 1


def __test_parked_modify_teardown_retires_park_against_stale_snapshot__():
    """The external-cancel teardown retires the parked verification state
    IN-MEMORY too, in lockstep with the persisted rows — an eventually
    consistent ``get_open_orders`` snapshot that still lists the cancelled
    order under the parked COID must NOT re-promote the torn-down mapping
    on the next sync, and the modify rollback snapshot must not outlive
    the park either."""
    b = MockBroker()  # on_unexpected_cancel == "stop"
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=50_000.0)
    engine.sync(BAR_TS)
    deal_id = engine.order_mapping["L"][0]

    # Declared atomic amend parks with an empty ring — an external cancel
    # during the park fires the policy and tears the key down.
    pos.entry_orders["L"] = _entry_order("L", 1.0, limit=51_000.0)
    b.raise_on_next_modify_entry = OrderDispositionUnknownError(
        "amend timed out", client_order_id="coid-L-amend",
        predecessor_cancel_ids=(),
    )
    engine.sync(BAR_TS + 60_000)
    assert len(engine.pending_verification) == 1

    engine._route_event(_cancelled_event(deal_id))

    assert engine.quarantined is True
    # The in-memory park died with the key at teardown time...
    assert len(engine.pending_verification) == 0
    # ...and the rollback snapshot did not outlive it.
    assert engine._modify_old_intents == {}

    # A stale open-orders snapshot still lists the cancelled order under
    # the parked COID — nothing is left to match it, so the cancelled
    # mapping must not resurrect.
    b.open_orders = [_live_working_order("coid-L-amend", order_id=deal_id)]
    engine.sync(BAR_TS + 120_000)
    assert "L" not in engine.order_mapping


# === Short-selling runtime gate (spot venues) ===


def _spot_broker() -> MockBroker:
    """Mock with the spot default: ``short_selling`` UNSUPPORTED."""
    return MockBroker(capabilities=ExchangeCapabilities())


def _order_order(order_id, size, **kw) -> Order:
    """A ``strategy.order`` style Pine order (normal type — never auto-reverses)."""
    return Order(order_id, size, order_type=_order_type_normal, **kw)


def __test_short_gate_halts_entry_short_from_flat__():
    """A ``strategy.entry`` short on a flat book targets a negative position
    — graceful halt, no broker call."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["S"] = _entry_order("S", -1.0)

    with pytest.raises(BrokerManualInterventionError):
        engine.sync(BAR_TS)

    assert engine.halted is True
    assert b.entry_calls == []


def __test_short_gate_halts_entry_reversal_on_spot__():
    """A reversing MARKET ``strategy.entry`` is judged on its FOLDED
    quantity — the combined stop-and-reverse always projects negative."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 0.0004
    pos.sign = 1.0
    pos.entry_orders["S"] = _entry_order("S", -0.0002)

    with pytest.raises(BrokerManualInterventionError):
        engine.sync(BAR_TS)

    assert engine.halted is True
    assert b.entry_calls == []


def __test_short_gate_strategy_order_reduce_passes__():
    """A ``strategy.order`` sell that only reduces the long is NOT a short:
    it dispatches with its RAW quantity (no stop-and-reverse fold)."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.entry_orders["Sell"] = _order_order("Sell", -3.0)

    engine.sync(BAR_TS)

    assert engine.halted is False
    assert len(b.entry_calls) == 1
    assert b.entry_calls[0].intent.qty == 3.0


def __test_short_gate_halts_strategy_order_flip__():
    """A ``strategy.order`` sell larger than the position projects negative."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.entry_orders["Sell"] = _order_order("Sell", -8.0)

    with pytest.raises(BrokerManualInterventionError):
        engine.sync(BAR_TS)

    assert engine.halted is True
    assert b.entry_calls == []


def __test_short_gate_aggregates_active_sell_intents__():
    """Two individually-reducing sells can flip together — the projection
    aggregates every active sell-side entry intent."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.entry_orders["S1"] = _order_order("S1", -3.0, limit=50_000.0)
    engine.sync(BAR_TS)  # 5 - 3 = 2: passes
    assert engine.halted is False
    assert len(b.entry_calls) == 1

    pos.entry_orders["S2"] = _order_order("S2", -3.0, limit=51_000.0)
    with pytest.raises(BrokerManualInterventionError):
        engine.sync(BAR_TS + 60_000)  # 5 - 3 - 3 = -1: halt

    assert engine.halted is True
    assert len(b.entry_calls) == 1


def __test_short_gate_allows_exits_and_closes__():
    """Exits and closes are reduce-only by engine contract — they flow
    untouched on a spot venue."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 5.0))
    pos.exit_orders[("TP", "L")] = _exit_order(
        "L", -5.0, "TP", limit=60_000.0, stop=45_000.0,
    )

    engine.sync(BAR_TS)

    assert engine.halted is False
    assert len(b.exit_calls) == 1


def __test_short_gate_ignores_buy_entries__():
    """Buy-side entries never trip the gate on a spot venue."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["L"] = _entry_order("L", 1.0)

    engine.sync(BAR_TS)

    assert engine.halted is False
    assert len(b.entry_calls) == 1


def __test_short_gate_modify_qty_raise_halts__():
    """An entry amend that raises a resting sell's qty past the inventory
    projects negative — same gate, OLD qty excluded from the aggregation."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.entry_orders["S"] = _order_order("S", -3.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1

    # A raise still within the inventory modifies fine (old 3 is replaced,
    # not stacked: 5 - 4 = 1).
    pos.entry_orders["S"] = _order_order("S", -4.0, limit=50_000.0)
    engine.sync(BAR_TS + 60_000)
    assert engine.halted is False
    assert len(b.modify_entry_calls) == 1

    pos.entry_orders["S"] = _order_order("S", -6.0, limit=50_000.0)
    with pytest.raises(BrokerManualInterventionError):
        engine.sync(BAR_TS + 120_000)  # 5 - 6 = -1: halt

    assert engine.halted is True
    assert len(b.modify_entry_calls) == 1


def __test_short_gate_modify_replace_resets_fill_ledger__():
    """F1 regression: ``modify_entry`` defaults to cancel + re-execute, so a
    partial fill credited to the pre-amend order is stale once the amend
    lands a FRESH working order. The short gate must reserve the replacement
    in full again — a stale ledger value would under-reserve and pass an
    oversell on a spot venue."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 5.0))
    pos.entry_orders["S"] = _order_order("S", -3.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1

    # 1 of the resting 3 fills; the filled slice leaves ``_position.size``
    # and the filled entry stays active as the sticky-order sentinel.
    engine.on_order_event(_fill_event(
        'sell', 1.0, 50_000.0, pine_id="S", event_type='partial',
        filled_qty=1.0, remaining_qty=2.0,
    ))
    engine.apply_async_events()
    assert pos.size == 4.0

    # Script amends "S" to a fresh resting sell of 4 (qty + price change ->
    # a replace-style modify). 4 - 4 = 0: the amend itself passes.
    pos.entry_orders["S"] = _order_order("S", -4.0, limit=51_000.0)
    engine.sync(BAR_TS + 60_000)
    assert engine.halted is False
    assert len(b.modify_entry_calls) == 1

    # An added sell-1 must halt: the replacement "S" now works 4 units in
    # full (ledger reseeded to 0), so 4 - 4 - 1 = -1. A stale filled=1 leak
    # would reserve only 3 and let 5 working units oversell the inventory.
    pos.entry_orders["G"] = _order_order("G", -1.0)
    with pytest.raises(BrokerManualInterventionError):
        engine.sync(BAR_TS + 120_000)
    assert engine.halted is True
    assert len(b.entry_calls) == 1


def __test_short_gate_modify_late_old_order_fill_not_credited__():
    """F3 regression: ``modify_entry`` defaults to cancel + re-execute. A fill
    that was in flight on the cancelled order can arrive AFTER the replacement's
    fill ledger was reset to zero. It is keyed only by ``pine_id``, so a naive
    ledger bump would credit it to the replacement and shrink the short-gate
    reservation below the replacement's true still-working qty — passing an
    oversell. ``record_fill`` still applies the fill to the position (real
    inventory moved); only the spurious ledger credit must be suppressed."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 5.0))
    pos.entry_orders["S"] = _order_order("S", -3.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1  # order id xchg-1

    # 1 of the resting 3 fills on the ORIGINAL order (xchg-1).
    engine.on_order_event(_fill_event(
        'sell', 1.0, 50_000.0, pine_id="S", xchg_id="xchg-1",
        event_type='partial', filled_qty=1.0, remaining_qty=2.0,
    ))
    engine.apply_async_events()
    assert pos.size == 4.0

    # Script amends "S" (price change) -> replace-style modify lands a fresh
    # resting sell of 3 under a NEW order id (xchg-2); the old xchg-1 is retired.
    pos.entry_orders["S"] = _order_order("S", -3.0, limit=51_000.0)
    engine.sync(BAR_TS + 60_000)
    assert engine.halted is False
    assert len(b.modify_entry_calls) == 1

    # A straggling fill from the CANCELLED old order (xchg-1) arrives after the
    # amend. It reduces inventory (5 -> 3 total sold) but must NOT be credited
    # to the replacement's reset ledger.
    engine.on_order_event(_fill_event(
        'sell', 1.0, 50_000.0, pine_id="S", xchg_id="xchg-1",
        event_type='filled', filled_qty=1.0, remaining_qty=0.0,
    ))
    engine.apply_async_events()
    assert pos.size == 3.0

    # An added sell-1 must halt: the replacement "S" still works all 3 units
    # (ledger correctly stayed 0), so 3 - 3 - 1 = -1. A leaked filled=1 credit
    # would reserve only 2 and pass 3 - 2 - 1 = 0 — a 4-unit oversell of 3.
    pos.entry_orders["G"] = _order_order("G", -1.0)
    with pytest.raises(BrokerManualInterventionError):
        engine.sync(BAR_TS + 120_000)
    assert engine.halted is True
    assert len(b.entry_calls) == 1


def __test_short_gate_fill_time_reconcile_cancels_unbacked_sell__():
    """F3-deep regression: the dispatch gate proves ``position >= working
    sells`` only at DISPATCH time. A late fill from a retired (cancel +
    re-execute) order erodes the position below the replacement sell's
    still-working qty AFTER dispatch. Nothing re-checks the resting sell, so
    it could fill into an oversell through the async window. The fill-time
    reconcile must cancel the now-unbacked resting sell — without halting."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 5.0))
    pos.entry_orders["S"] = _order_order("S", -5.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1  # order id xchg-1, 5 - 5 = 0 backs it exactly

    # Amend "S" (price change) -> replace-style modify: xchg-1 is retired, a
    # fresh resting sell of 5 lands under xchg-2, the fill ledger resets to 0.
    pos.entry_orders["S"] = _order_order("S", -5.0, limit=51_000.0)
    engine.sync(BAR_TS + 60_000)
    assert engine.halted is False
    assert len(b.modify_entry_calls) == 1

    # A straggling fill of 2 from the CANCELLED old order (xchg-1) arrives.
    # ``record_fill`` sells 2 against the long (5 -> 3 real inventory), the
    # retired-order guard keeps it OUT of the replacement's ledger. Now the
    # replacement works all 5 units against a position of only 3: 3 - 5 = -2.
    engine.on_order_event(_fill_event(
        'sell', 2.0, 50_000.0, pine_id="S", xchg_id="xchg-1",
        event_type='filled', filled_qty=2.0, remaining_qty=0.0,
    ))
    engine.apply_async_events()
    assert pos.size == 3.0

    # The fill-time reconcile cancelled the unbacked resting sell "S" at the
    # broker (xchg-2) so it can never fill into the oversell — and kept the
    # bot running (no halt). Without the fix "S" stays live and cancel_calls
    # is empty.
    assert engine.halted is False
    assert len(b.cancel_calls) == 1
    assert b.cancel_calls[0].intent.pine_id == "S"
    assert "S" not in engine.active_intents


def __test_short_gate_fill_time_reconcile_noop_when_still_backed__():
    """The reconcile must NOT over-cancel: a fill that leaves the position
    still covering the aggregate working sell qty leaves the resting sell
    untouched (mirrors the F3 fill-not-credited scenario, which stays flat)."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 5.0))
    pos.entry_orders["S"] = _order_order("S", -3.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1  # 5 - 3 = 2 backs it

    pos.entry_orders["S"] = _order_order("S", -3.0, limit=51_000.0)
    engine.sync(BAR_TS + 60_000)
    assert len(b.modify_entry_calls) == 1

    # A late fill of 1 from the retired old order: 5 -> 4, ledger stays 0.
    # 4 - 3 = 1 >= 0, so the resting sell is still fully backed: no cancel.
    engine.on_order_event(_fill_event(
        'sell', 1.0, 50_000.0, pine_id="S", xchg_id="xchg-1",
        event_type='filled', filled_qty=1.0, remaining_qty=0.0,
    ))
    engine.apply_async_events()
    assert pos.size == 4.0

    assert engine.halted is False
    assert b.cancel_calls == []
    assert "S" in engine.active_intents


def __test_short_gate_fill_time_reconcile_noop_on_short_capable_venue__():
    """The fill-time reconcile is a spot-only guard. On a short-capable
    (margin) venue the position may legitimately go negative, so an eroding
    fill must never cancel a resting sell."""
    b = MockBroker()  # default capabilities: short_selling NATIVE
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 5.0))
    pos.entry_orders["S"] = _order_order("S", -5.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1

    pos.entry_orders["S"] = _order_order("S", -5.0, limit=51_000.0)
    engine.sync(BAR_TS + 60_000)
    assert len(b.modify_entry_calls) == 1

    engine.on_order_event(_fill_event(
        'sell', 2.0, 50_000.0, pine_id="S", xchg_id="xchg-1",
        event_type='filled', filled_qty=2.0, remaining_qty=0.0,
    ))
    engine.apply_async_events()
    assert pos.size == 3.0

    # Short selling is supported: no reconcile, the resting sell stays live.
    assert engine.halted is False
    assert b.cancel_calls == []
    assert "S" in engine.active_intents


def _reconcile_deficit_setup() -> tuple[MockBroker, OrderSyncEngine, BrokerPosition]:
    """Spot engine with a replacement sell "S" (xchg-2, works 5) and a
    long of 5, ready for a late retired-order fill of 2 to erode the position
    to 3 and leave the resting sell unbacked (deficit 2)."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 5.0))
    pos.entry_orders["S"] = _order_order("S", -5.0, limit=50_000.0)
    engine.sync(BAR_TS)  # xchg-1
    pos.entry_orders["S"] = _order_order("S", -5.0, limit=51_000.0)
    engine.sync(BAR_TS + 60_000)  # replace -> xchg-2, fill ledger reset to 0
    assert len(b.modify_entry_calls) == 1
    return b, engine, pos


def __test_short_gate_reconcile_failed_cancel_parks_and_retries__():
    """F1 regression: the corrective cancel can fail with a dropped link
    (``ExchangeConnectionError``). The engine must NOT leave the half-cancelled
    order for the cross-restart adoption branch to silently reclaim as healthy
    — that resurrects the unbacked sell. The cancel is parked and re-attempted
    every sync (never halting); the diff refuses to adopt / re-dispatch the key
    until it lands."""
    b, engine, pos = _reconcile_deficit_setup()

    # The broker link drops exactly during the corrective cancel.
    b.raise_on_next_cancel = ExchangeConnectionError("cancel link dropped")
    # Late fill of 2 from the retired old order (xchg-1): 5 -> 3, working sell
    # stays 5, so 3 - 5 = -2 deficit. The reconcile tries to cancel "S".
    engine.on_order_event(_fill_event(
        'sell', 2.0, 50_000.0, pine_id="S", xchg_id="xchg-1",
        event_type='filled', filled_qty=2.0, remaining_qty=0.0,
    ))
    engine.apply_async_events()
    assert pos.size == 3.0

    # Cancel attempted once and failed -> parked, bot still running, and the
    # order is NOT silently kept active.
    assert engine.halted is False
    assert "S" in engine._forced_cancel_pending
    assert "S" not in engine.active_intents
    assert len(b.cancel_calls) == 1

    # Next sync, "S" still emitted by Pine, link STILL down: the adoption
    # branch must NOT reclaim the surviving mapping as a healthy order (the
    # pre-fix bug), and the still-unresolved cancel must not re-dispatch into a
    # halt. It defers; the retry re-attempts the cancel.
    b.raise_on_next_cancel = ExchangeConnectionError("still down")
    engine.sync(BAR_TS + 120_000)
    assert engine.halted is False
    assert "S" not in engine.active_intents          # adoption guard held
    assert "S" in engine._forced_cancel_pending
    assert len(b.cancel_calls) == 2                   # retry attempted

    # Link recovers and the strategy drops "S": the parked cancel lands and the
    # key is released — no halt, no lingering unbacked order.
    pos.entry_orders.pop("S")
    engine.sync(BAR_TS + 180_000)
    assert engine.halted is False
    assert "S" not in engine._forced_cancel_pending
    assert "S" not in engine.active_intents
    assert len(b.cancel_calls) == 3                   # cancel landed


def __test_short_gate_reconcile_unknown_disposition_parks_no_premature_halt__():
    """F1 regression: an ambiguous cancel timeout
    (``OrderDispositionUnknownError``) means the resting sell MAY still be live.
    Pre-fix, ``_dispatch_cancel`` swallowed it and dropped the mapping, so the
    next sync re-dispatched into a dispatch-time halt while the order might
    still rest. The durable path parks it and keeps retrying without halting
    until the disposition provably resolves."""
    b, engine, pos = _reconcile_deficit_setup()

    b.raise_on_next_cancel = OrderDispositionUnknownError(
        "cancel timed out", client_order_id="xchg-2",
    )
    engine.on_order_event(_fill_event(
        'sell', 2.0, 50_000.0, pine_id="S", xchg_id="xchg-1",
        event_type='filled', filled_qty=2.0, remaining_qty=0.0,
    ))
    engine.apply_async_events()
    assert pos.size == 3.0
    assert engine.halted is False
    assert "S" in engine._forced_cancel_pending
    assert "S" not in engine.active_intents
    assert len(b.cancel_calls) == 1

    # Disposition STILL ambiguous on retry, "S" still emitted: the engine must
    # keep deferring, NOT halt (the pre-fix code would already have halted on
    # the re-dispatch of the first post-swallow sync).
    b.raise_on_next_cancel = OrderDispositionUnknownError(
        "cancel still ambiguous", client_order_id="xchg-2",
    )
    engine.sync(BAR_TS + 120_000)
    assert engine.halted is False
    assert "S" in engine._forced_cancel_pending
    assert "S" not in engine.active_intents
    assert len(b.cancel_calls) == 2

    # The cancel is finally confirmed and the strategy drops "S": released.
    pos.entry_orders.pop("S")
    engine.sync(BAR_TS + 180_000)
    assert engine.halted is False
    assert "S" not in engine._forced_cancel_pending
    assert len(b.cancel_calls) == 3


def __test_short_gate_reconcile_false_cancel_parks_and_retries__():
    """The corrective cancel can return ``False`` WITHOUT raising —
    ``execute_cancel``'s documented "cancel did not land, still pending"
    signal. Treating a clean return as proof the resting sell is gone would let
    the diff re-adopt / re-dispatch and resurrect the unbacked exposure. The
    engine must park the key (never halting) and keep retrying until a truthy
    return proves the cancel landed."""
    b, engine, pos = _reconcile_deficit_setup()

    # The corrective cancel returns False (still pending), no exception.
    b.false_on_next_cancel = True
    engine.on_order_event(_fill_event(
        'sell', 2.0, 50_000.0, pine_id="S", xchg_id="xchg-1",
        event_type='filled', filled_qty=2.0, remaining_qty=0.0,
    ))
    engine.apply_async_events()
    assert pos.size == 3.0

    # Cancel attempted once, returned False -> parked, bot still running, the
    # order is NOT treated as cancelled.
    assert engine.halted is False
    assert "S" in engine._forced_cancel_pending
    assert "S" not in engine.active_intents
    assert len(b.cancel_calls) == 1

    # Next sync, "S" still emitted by Pine, cancel STILL returns False: the
    # adoption branch must NOT reclaim the surviving mapping as a healthy order,
    # and the still-unlanded cancel must not re-dispatch into a halt. It defers;
    # the retry re-attempts the cancel (once per sync).
    b.false_on_next_cancel = True
    engine.sync(BAR_TS + 120_000)
    assert engine.halted is False
    assert "S" not in engine.active_intents          # adoption guard held
    assert "S" in engine._forced_cancel_pending
    assert len(b.cancel_calls) == 2                   # single retry this sync

    # The cancel finally lands (truthy) and the strategy drops "S": released —
    # no halt, no lingering unbacked order.
    pos.entry_orders.pop("S")
    engine.sync(BAR_TS + 180_000)
    assert engine.halted is False
    assert "S" not in engine._forced_cancel_pending
    assert "S" not in engine.active_intents
    assert len(b.cancel_calls) == 3                   # cancel landed


def __test_short_gate_reconcile_failed_first_cancel_continues_to_next__():
    """A failed cancel must NOT count towards the deficit. With TWO resting
    sells, if the newest cancel does not land (``execute_cancel`` returns
    ``False`` — still parked, still fillable), the loop must keep the deficit
    intact and cancel the next candidate so the CONFIRMED-remaining exposure is
    backed. Subtracting the parked order's working qty would stop the loop
    early, leaving a second sell live and adopted as healthy — both could then
    fill and drive the venue short."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 9.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 9.0))
    # Two resting sells of 4 each: 4 + 4 = 8 <= 9 backs both at dispatch.
    pos.entry_orders["S1"] = _order_order("S1", -4.0, limit=50_000.0)
    pos.entry_orders["S2"] = _order_order("S2", -4.0, limit=52_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 2

    # Replace-style modify of the NEWEST sell "S2" (price change): its old
    # order is retired, a fresh resting sell of 4 lands, the ledger resets.
    pos.entry_orders["S2"] = _order_order("S2", -4.0, limit=53_000.0)
    engine.sync(BAR_TS + 60_000)
    assert len(b.modify_entry_calls) == 1

    # A straggling fill of 4 from the CANCELLED old "S2" order (xchg-2) erodes
    # the long 9 -> 5; the retired-order guard keeps it out of "S2"'s ledger,
    # so both sells still work 4 each: reserved 8 vs position 5 => deficit 3.
    # The newest-first cancel of "S2" returns False (parked, still live).
    b.false_on_next_cancel = True
    engine.on_order_event(_fill_event(
        'sell', 4.0, 50_000.0, pine_id="S2", xchg_id="xchg-2",
        event_type='filled', filled_qty=4.0, remaining_qty=0.0,
    ))
    engine.apply_async_events()
    assert pos.size == 5.0

    # The failed "S2" cancel did NOT satisfy the deficit, so the loop went on
    # to cancel "S1" too (which landed). Confirmed-remaining exposure is the
    # parked "S2" (4) alone against a position of 5 — backed. Both orders were
    # cancelled at the broker; neither is left adopted as a healthy sell.
    assert engine.halted is False
    assert [c.intent.pine_id for c in b.cancel_calls] == ["S2", "S1"]
    assert "S2" in engine._forced_cancel_pending      # parked, retried
    assert "S1" not in engine.active_intents          # cancel landed
    assert "S2" not in engine.active_intents          # popped + parked


def __test_short_gate_dispatch_counts_parked_forced_cancel_sell__():
    """A forced-cancel-pending sell has left ``_active_intents`` but its working
    order may still rest live at the broker until the cancel lands. The dispatch
    gate must fold it into the reservation, otherwise a DIFFERENT sell can be
    admitted against inventory the parked order still reserves — both then fill
    and take a short-incapable venue negative."""
    b, engine, pos = _reconcile_deficit_setup()

    # Park "S" (works 5, still live) via a corrective cancel that returns False.
    b.false_on_next_cancel = True
    engine.on_order_event(_fill_event(
        'sell', 2.0, 50_000.0, pine_id="S", xchg_id="xchg-1",
        event_type='filled', filled_qty=2.0, remaining_qty=0.0,
    ))
    engine.apply_async_events()
    assert pos.size == 3.0
    assert "S" in engine._forced_cancel_pending
    assert "S" not in engine.active_intents
    assert engine._parked_working_sell_qty() == 5.0

    # A DIFFERENT sell "S2" of 3 is emitted while "S" stays parked-but-live.
    # A bare active-only aggregation would see reserved=0 and pass (3 - 3 = 0);
    # counting the still-live parked "S" (works 5) projects 3 - 5 - 3 = -5, so
    # the gate halts and never dispatches "S2" — preventing the S + S2 oversell.
    # Keep "S"'s retry from landing this sync so it stays counted.
    b.false_on_next_cancel = True
    pos.entry_orders["S2"] = _order_order("S2", -3.0, limit=52_000.0)
    with pytest.raises(BrokerManualInterventionError):
        engine.sync(BAR_TS + 120_000)

    assert engine.halted is True
    assert "S2" not in engine.active_intents  # blocked before the broker call
    assert len(b.entry_calls) == 1            # only the original "S" dispatch


def __test_short_gate_filled_sell_entry_not_double_reserved__():
    """F2 regression: a FILLED sell entry stays in ``_active_intents`` (it is
    the diff sentinel for the sticky Pine order) while ``record_fill`` already
    moved its qty into ``_position.size`` — the gate must reserve only the
    working residual, otherwise a legitimate later flatten is double-counted
    into a false halt."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 5.0))
    pos.entry_orders["S"] = _order_order("S", -3.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1

    engine.on_order_event(_fill_event('sell', 3.0, 50_000.0, pine_id="S"))
    engine.apply_async_events()
    assert pos.size == 2.0
    # The filled intent deliberately stays active (sticky-order sentinel).
    assert "S" in engine.active_intents

    # Flatten the remainder: 2 - (3 - 3 filled) - 2 = 0 — must dispatch.
    pos.entry_orders["F"] = _order_order("F", -2.0)
    engine.sync(BAR_TS + 60_000)

    assert engine.halted is False
    assert len(b.entry_calls) == 2


def __test_short_gate_partial_fill_reserves_working_residual__():
    """A partially filled sell entry reserves only its still-working slice;
    the filled slice already left ``_position.size`` via ``record_fill``."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 5.0))
    pos.entry_orders["S"] = _order_order("S", -3.0, limit=50_000.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1

    engine.on_order_event(_fill_event(
        'sell', 1.0, 50_000.0, pine_id="S", event_type='partial',
        filled_qty=1.0, remaining_qty=2.0,
    ))
    engine.apply_async_events()
    assert pos.size == 4.0

    # 4 - (3 - 1 filled) - 2 = 0: passes.
    pos.entry_orders["F"] = _order_order("F", -2.0)
    engine.sync(BAR_TS + 60_000)
    assert engine.halted is False
    assert len(b.entry_calls) == 2

    # Both working residuals aggregate: 4 - 2 - 2 - 1 = -1 halts.
    pos.entry_orders["G"] = _order_order("G", -1.0)
    with pytest.raises(BrokerManualInterventionError):
        engine.sync(BAR_TS + 120_000)
    assert engine.halted is True
    assert len(b.entry_calls) == 2


def __test_short_gate_reused_pine_id_reseeds_fill_ledger__():
    """A retired entry's fill ledger must not leak onto a NEW order reusing
    the same ``pine_id`` — the fresh dispatch reseeds to zero, so the new
    resting qty is reserved in full again."""
    b = _spot_broker()
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.open_trades.append(_long_trade("L", 5.0))
    pos.entry_orders["S"] = _order_order("S", -3.0, limit=50_000.0)
    engine.sync(BAR_TS)
    engine.on_order_event(_fill_event('sell', 3.0, 50_000.0, pine_id="S"))
    engine.apply_async_events()
    assert pos.size == 2.0

    # Script drops the settled order -> the diff retires the slot.
    del pos.entry_orders["S"]
    engine.sync(BAR_TS + 60_000)
    assert "S" not in engine.active_intents

    # Same pine_id re-armed with a fresh resting sell: 2 - 2 = 0 passes and
    # the new order is reserved IN FULL (a stale filled=3 leak would zero
    # the reservation instead).
    pos.entry_orders["S"] = _order_order("S", -2.0, limit=51_000.0)
    engine.sync(BAR_TS + 120_000)
    assert len(b.entry_calls) == 2

    # 2 - 2 (fresh working "S") - 1 = -1: halts. With a leaked ledger the
    # projection would be +1 and the oversell would dispatch.
    pos.entry_orders["G"] = _order_order("G", -1.0)
    with pytest.raises(BrokerManualInterventionError):
        engine.sync(BAR_TS + 180_000)
    assert engine.halted is True
    assert len(b.entry_calls) == 2


def __test_short_gate_restart_recovered_entry_seeds_fill_ledger__(tmp_path):
    """A cross-restart recovered parked sell entry seeds the fill ledger from
    the broker's cumulative ``filled_qty`` — the adopted position already
    contains the filled slice, and the pre-crash fill events never replay,
    so a full-qty reservation would double-count and falsely halt."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    coid = build_client_order_id(
        run_tag=RUN_TAG, pine_id="S", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=0,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src",
                             script_path="t025.py")
        b = MockBroker(capabilities=ExchangeCapabilities())
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.size = 5.0
        pos.sign = 1.0
        pos.open_trades.append(_long_trade("L", 5.0))
        pos.entry_orders["S"] = _order_order("S", -3.0, limit=50_000.0)
        b.raise_on_next_entry = OrderDispositionUnknownError(
            "simulated timeout", client_order_id=coid,
        )
        engine.sync(BAR_TS)  # parks under unknown disposition
        assert coid in engine.pending_verification

        # Crash / restart. The order landed and 2 of 3 filled while the bot
        # was down; the broker's open-orders view carries the cumulative
        # counter. The new engine adopts the reduced position (3.0).
        ctx.close()
        ctx2 = store.open_run(_restart_identity(), script_source="src",
                              script_path="t025.py")
        b2 = MockBroker(capabilities=ExchangeCapabilities())
        b2.open_orders = [ExchangeOrder(
            id="live-1", symbol=SYMBOL, side="sell",
            order_type=OrderType.LIMIT, qty=3.0, filled_qty=2.0,
            remaining_qty=1.0, price=50_000.0, stop_price=None,
            average_fill_price=50_000.0, status=OrderStatus.OPEN,
            timestamp=0.0, fee=0.0, fee_currency="",
            client_order_id=coid,
        )]
        pos2 = BrokerPosition()
        engine2 = OrderSyncEngine(
            broker=b2,  # type: ignore[arg-type]
            position=pos2, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx2,
        )
        pos2.size = 3.0
        pos2.sign = 1.0
        pos2.open_trades.append(_long_trade("L", 3.0))
        # Script re-emits the sticky sell entry plus a legit flatten of the
        # rest: 3 - (3 - 2 filled) - 2 = 0 — must dispatch, not halt.
        pos2.entry_orders["S"] = _order_order("S", -3.0, limit=50_000.0)
        pos2.entry_orders["F"] = _order_order("F", -2.0)

        engine2.sync(BAR_TS + 60_000)

        assert engine2.halted is False
        # Only the flatten dispatches fresh; "S" adopts the recovered order.
        assert len(b2.entry_calls) == 1
        assert b2.entry_calls[0].intent.pine_id == "F"


def __test_short_gate_restart_recovered_unbacked_sell_cancelled_same_sync__(tmp_path):
    """A cross-restart recovered resting sell the prior run left UNBACKED must
    be cancelled on the FIRST post-restart sync, not the next one.

    :meth:`_verify_pending_dispatches` seeds only ``_order_mapping`` for the
    recovered order; the script's re-emission adopts it into ``_active_intents``
    inside :meth:`_diff_and_dispatch`, so the pre-diff short-gate pass cannot
    see it yet. Without a post-diff re-scan the unbacked sell would rest live at
    the broker for a full extra bar (oversell window). The post-diff pass
    catches the freshly-adopted sell and cancels it in the same sync."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.idempotency import build_client_order_id, KIND_ENTRY

    coid = build_client_order_id(
        run_tag=RUN_TAG, pine_id="S", bar_ts_ms=BAR_TS,
        kind=KIND_ENTRY, retry_seq=0,
    )
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src",
                             script_path="t025.py")
        b = MockBroker(capabilities=ExchangeCapabilities())
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.size = 5.0
        pos.sign = 1.0
        pos.open_trades.append(_long_trade("L", 5.0))
        pos.entry_orders["S"] = _order_order("S", -5.0, limit=50_000.0)
        b.raise_on_next_entry = OrderDispositionUnknownError(
            "simulated timeout", client_order_id=coid,
        )
        engine.sync(BAR_TS)  # parks the sell under unknown disposition
        assert coid in engine.pending_verification

        # Crash / restart. The sell landed live (0 filled) but the long was
        # reduced to 3 while the bot was down (an exit the prior run's forced
        # cancel of "S" never completed against). The recovered book therefore
        # holds a resting sell of 5 against a long of only 3 — unbacked by 2.
        ctx.close()
        ctx2 = store.open_run(_restart_identity(), script_source="src",
                              script_path="t025.py")
        b2 = MockBroker(capabilities=ExchangeCapabilities())
        b2.open_orders = [ExchangeOrder(
            id="live-1", symbol=SYMBOL, side="sell",
            order_type=OrderType.LIMIT, qty=5.0, filled_qty=0.0,
            remaining_qty=5.0, price=50_000.0, stop_price=None,
            average_fill_price=None, status=OrderStatus.OPEN,
            timestamp=0.0, fee=0.0, fee_currency="",
            client_order_id=coid,
        )]
        pos2 = BrokerPosition()
        engine2 = OrderSyncEngine(
            broker=b2,  # type: ignore[arg-type]
            position=pos2, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx2,
        )
        pos2.size = 3.0
        pos2.sign = 1.0
        pos2.open_trades.append(_long_trade("L", 3.0))
        # Script re-emits the sticky sell entry: 3 - 5 = -2 unbacked.
        pos2.entry_orders["S"] = _order_order("S", -5.0, limit=50_000.0)

        engine2.sync(BAR_TS + 60_000)

        # The recovered unbacked sell is cancelled in THIS sync (post-diff
        # pass), never halting, and not left as a healthy adopted order.
        assert engine2.halted is False
        assert len(b2.cancel_calls) == 1
        assert b2.cancel_calls[0].intent.pine_id == "S"
        assert "S" not in engine2.active_intents
        assert "S" not in engine2._forced_cancel_pending


def __test_forced_cancel_survives_restart_reissued_from_journal__(tmp_path):
    """A parked forced cancel is journaled, so a crash while parked cannot
    orphan the still-live unbacked sell. Pre-fix the pending map was
    memory-only: if the script no longer re-emitted the intent after the
    restart, NOTHING re-detected the un-landed cancel and the resting sell
    stayed live at the broker indefinitely (the oversell the short gate
    exists to prevent). The ``dispatch_kind='forced_cancel'`` journal row
    re-arms the retry on the first post-restart sync — with no script
    re-emission and no order-recovery needed — and the landed cancel deletes
    the row along with the envelope."""
    from pynecore.core.broker.storage import BrokerStore

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src",
                             script_path="t025.py")
        b = _spot_broker()
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.size = 5.0
        pos.sign = 1.0
        pos.open_trades.append(_long_trade("L", 5.0))
        pos.entry_orders["S"] = _order_order("S", -5.0, limit=50_000.0)
        engine.sync(BAR_TS)  # xchg-1
        pos.entry_orders["S"] = _order_order("S", -5.0, limit=51_000.0)
        engine.sync(BAR_TS + 60_000)  # replace -> xchg-2, fill ledger reset
        assert len(b.modify_entry_calls) == 1

        # A late fill from the retired xchg-1 erodes the long to 3: the
        # fill-time reconcile force-cancels the resting sell, but the cancel
        # does not land — parked AND journaled.
        b.false_on_next_cancel = True
        engine.on_order_event(_fill_event(
            'sell', 2.0, 50_000.0, pine_id="S", xchg_id="xchg-1",
            event_type='filled', filled_qty=2.0, remaining_qty=0.0,
        ))
        engine.apply_async_events()
        assert "S" in engine._forced_cancel_pending
        assert len(b.cancel_calls) == 1

        # Crash while parked. The fresh run's script does NOT re-emit "S"
        # (the signal is gone), so nothing but the journal row knows the
        # resting sell still needs cancelling.
        ctx.close()
        ctx2 = store.open_run(_restart_identity(), script_source="src",
                              script_path="t025.py")
        b2 = _spot_broker()
        pos2 = BrokerPosition()
        engine2 = OrderSyncEngine(
            broker=b2,  # type: ignore[arg-type]
            position=pos2, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx2,
        )
        pos2.size = 3.0
        pos2.sign = 1.0
        pos2.open_trades.append(_long_trade("L", 3.0))

        # The journal re-armed the forced cancel at construction already.
        assert "S" in engine2._forced_cancel_pending

        engine2.sync(BAR_TS + 120_000)

        # The first post-restart sync re-drives and lands the cancel —
        # never halting — and the journal row dies with the envelope.
        assert engine2.halted is False
        assert len(b2.cancel_calls) == 1
        assert b2.cancel_calls[0].intent.pine_id == "S"
        assert "S" not in engine2._forced_cancel_pending
        envelopes, pending = ctx2.replay()
        assert "S" not in envelopes
        assert not pending


def __test_forced_cancel_restart_recovers_working_sell_qty_from_orders__(tmp_path):
    """F2 regression: a forced cancel rehydrated from the journal synthesizes a
    ``qty=0.0`` placeholder intent (only identity drives the cancel). If the
    cancel stays un-landed after the restart the still-live resting SELL order
    keeps claiming inventory, yet ``_parked_working_sell_qty`` read 0.0 from the
    placeholder — so a DIFFERENT sell passed the short gate and both could fill,
    an oversell on a short-incapable venue. The working residual is now
    recovered from the authoritative ``orders`` table (side + qty + filled_qty),
    so the parked exposure is reserved and the second sell halts."""
    from pynecore.core.broker.storage import BrokerStore
    from pynecore.core.broker.models import EntryIntent

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_restart_identity(), script_source="src",
                             script_path="t025.py")
        b = _spot_broker()
        pos = BrokerPosition()
        engine = OrderSyncEngine(
            broker=b,  # type: ignore[arg-type]
            position=pos, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx,
        )
        pos.size = 5.0
        pos.sign = 1.0
        pos.open_trades.append(_long_trade("L", 5.0))
        pos.entry_orders["S"] = _order_order("S", -5.0, limit=50_000.0)
        engine.sync(BAR_TS)  # xchg-1
        pos.entry_orders["S"] = _order_order("S", -5.0, limit=51_000.0)
        engine.sync(BAR_TS + 60_000)  # replace -> xchg-2, fill ledger reset

        # The real plugin persists the resting sell order row; MockBroker does
        # not, so mirror that persistence — the authoritative residual the
        # recovery reads after the restart lives in the ``orders`` table.
        ctx.upsert_order(
            "sell-S", symbol=SYMBOL, side="sell", qty=5.0, state="confirmed",
            intent_key="S", filled_qty=0.0,
        )

        # A late fill from the retired xchg-1 erodes the long to 3; the
        # fill-time reconcile force-cancels the resting sell but the cancel does
        # not land -> parked AND journaled.
        b.false_on_next_cancel = True
        engine.on_order_event(_fill_event(
            'sell', 2.0, 50_000.0, pine_id="S", xchg_id="xchg-1",
            event_type='filled', filled_qty=2.0, remaining_qty=0.0,
        ))
        engine.apply_async_events()
        assert "S" in engine._forced_cancel_pending
        ctx.close()

        # Crash while parked. The fresh run does NOT re-emit "S".
        ctx2 = store.open_run(_restart_identity(), script_source="src",
                              script_path="t025.py")
        b2 = _spot_broker()
        pos2 = BrokerPosition()
        engine2 = OrderSyncEngine(
            broker=b2,  # type: ignore[arg-type]
            position=pos2, symbol=SYMBOL, run_tag=RUN_TAG,
            mintick=1.0, store_ctx=ctx2,
        )
        pos2.size = 3.0
        pos2.sign = 1.0
        pos2.open_trades.append(_long_trade("L", 3.0))

        # The journal re-armed "S" with a synthesized qty=0.0 placeholder.
        assert "S" in engine2._forced_cancel_pending
        assert engine2._forced_cancel_pending["S"].qty == 0.0
        # Pre-fix the placeholder reserved 0.0; the orders-table recovery now
        # supplies the still-live working residual of 5.0.
        assert engine2._parked_working_sell_qty() == 5.0

        # A DIFFERENT sell "S2" of 3 against the eroded long of 3 would pass a
        # bare active-only gate (3 - 3 = 0) and oversell alongside the parked
        # "S"; counting the recovered parked 5 projects 3 - 5 - 3 = -5, so the
        # dispatch gate halts before the broker call.
        s2 = EntryIntent(
            pine_id="S2", symbol=SYMBOL, side='sell', qty=3.0,
            order_type=OrderType.MARKET,
        )
        with pytest.raises(BrokerManualInterventionError):
            engine2._enforce_short_gate(s2)
        assert engine2.halted is True


def __test_general_cancel_false_parks_defers_reemit_no_double_open__():
    """General-path (non-short-gate) regression for the un-landed cancel: a
    working entry the script dropped is cancelled through the default diff
    path, but ``execute_cancel`` returns ``False`` — the order is still live.
    Pre-fix the bool was discarded and the strict path tore the mapping /
    envelope down anyway, so a same-key re-emit dispatched a SECOND working
    order next to the still-resting one (2x exposure), and the retired
    partial-bracket state would have left an eventual fill unprotected. Now
    the tracking state survives, the key parks, the diff defers the re-emit,
    and the retry re-drives the cancel until it lands. Runs on a
    short-capable venue to prove the mechanism is venue-independent."""
    b = MockBroker()  # short_selling NATIVE — short gate inactive
    engine, pos = _mk_engine(b)
    pos.entry_orders["E"] = _order_order("E", 3.0, limit=100.0)
    engine.sync(BAR_TS)
    assert len(b.entry_calls) == 1
    assert "E" in engine.order_mapping

    # Script drops the entry; the broker reports the cancel did not land.
    pos.entry_orders.pop("E")
    b.false_on_next_cancel = True
    engine.sync(BAR_TS + 60_000)
    assert engine.halted is False
    assert "E" in engine._forced_cancel_pending
    assert len(b.cancel_calls) == 1
    # The strict path kept the tracking state — the order is still live and
    # its fills must keep routing.
    assert "E" in engine.order_mapping

    # Script re-emits the same entry while the cancel is still un-landed:
    # the diff must defer (no second working order next to the resting one)
    # and the per-sync retry re-drives the cancel exactly once.
    pos.entry_orders["E"] = _order_order("E", 3.0, limit=100.0)
    b.false_on_next_cancel = True
    engine.sync(BAR_TS + 120_000)
    assert engine.halted is False
    assert len(b.cancel_calls) == 2                   # retry, still False
    assert len(b.entry_calls) == 1                    # NO double-open
    assert "E" in engine._forced_cancel_pending
    assert "E" not in engine.active_intents           # deferred, not adopted

    # The cancel finally lands: the key is released, the teardown runs, and
    # the still-wanted entry re-dispatches fresh on the same sync's diff.
    engine.sync(BAR_TS + 180_000)
    assert engine.halted is False
    assert "E" not in engine._forced_cancel_pending
    assert len(b.cancel_calls) == 3                   # landed
    assert len(b.entry_calls) == 2                    # re-dispatched after


def __test_modify_cancel_reexecute_deferred_while_cancel_unlanded__():
    """Mismatched-kind modify (cancel + re-execute): when the cancel of the
    old working order does not land (``execute_cancel`` returns ``False``),
    the replacement must NOT be dispatched in the same sync — the old order
    is still live and a fresh dispatch would double-live the key. The modify
    defers (old intent stays active for the next re-diff) until the parked
    cancel lands; a repeat attempt while parked defers WITHOUT another
    broker cancel round-trip (the per-sync retry owns the cancel)."""
    from pynecore.core.broker.sync_engine import _PartialBracketModifyDeferred

    b = MockBroker()
    engine, pos = _mk_engine(b)
    pos.entry_orders["E"] = _order_order("E", 3.0, limit=100.0)
    engine.sync(BAR_TS)
    old = engine.active_intents["E"]
    new = CloseIntent(pine_id="E", symbol=SYMBOL, side='sell', qty=3.0)

    b.false_on_next_cancel = True
    with pytest.raises(_PartialBracketModifyDeferred):
        engine._dispatch_modify(old, new)
    assert "E" in engine._forced_cancel_pending
    assert b.close_calls == []            # replacement NOT dispatched
    assert len(b.cancel_calls) == 1

    with pytest.raises(_PartialBracketModifyDeferred):
        engine._dispatch_modify(old, new)
    assert len(b.cancel_calls) == 1       # no duplicate cancel round-trip
    assert b.close_calls == []


def __test_strategy_order_market_reduce_is_not_folded_on_margin__():
    """The stop-and-reverse fold is ``strategy.entry`` semantics only:
    ``strategy.order`` never auto-reverses, so an opposite-side market
    order dispatches its RAW quantity on a margin venue too."""
    b = MockBroker()  # short_selling NATIVE — gate inactive
    engine, pos = _mk_engine(b)
    pos.size = 5.0
    pos.sign = 1.0
    pos.entry_orders["Sell"] = _order_order("Sell", -3.0)

    engine.sync(BAR_TS)

    assert len(b.entry_calls) == 1
    assert b.entry_calls[0].intent.qty == 3.0
