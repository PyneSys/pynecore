"""
The bot's own protection across a restart and an operator resume.

A restart must never judge a level the bot itself left on the venue as
someone else's edit (a false quarantine stops a healthy unattended bot), and
an operator resume accepts the venue as the operator left it — after which
the bot re-asserts its own exits once. Covers the broker-native fail-safe
stop (cTrader / Capital.com partial brackets) and the re-assert of the
strategy's persistent exits.
"""
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.core.broker.idempotency import KIND_CLOSE, KIND_EXIT_SL
from pynecore.core.broker.models import (
    DispatchEnvelope,
    ExchangeCapabilities,
    ExchangeOrder,
    ExchangePosition,
    ExitIntent,
    INTENT_KEY_SEP,
    OrderStatus,
    OrderType,
    QuarantineKind,
)
from pynecore.core.broker.native_failsafe_manager import (
    NATIVE_FAILSAFE_LEVELS_EXTRA,
    FailsafeHealth,
    FailsafeOwner,
    NativeBracketSnapshot,
)
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.protection_guard import ProtectionLevelGuard, ProtectionLevels
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.software_partial_bracket_engine import PartialBracketLeg
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.store_helpers import LEG_KIND_SL_PARTIAL, LEG_STATE_ARMED
from pynecore.core.broker.sync_engine import OrderSyncEngine

SYMBOL = "EURUSD"
PLUGIN = "Mock"
BAR_TS = 1_700_000_000_000
PARENT = "parent-L"


@pytest.fixture(autouse=True)
def _stub_script():
    prev = lib._script
    lib._script = SimpleNamespace(initial_capital=1_000_000.0)
    try:
        yield
    finally:
        lib._script = prev


@dataclass
class _Broker:
    client_order_id_max_len = 30
    on_unexpected_cancel = "stop"
    position: ExchangePosition | None = None
    open_orders: list[ExchangeOrder] = field(default_factory=list)
    modify_exit_calls: list[tuple[DispatchEnvelope, DispatchEnvelope]] = field(
        default_factory=list)
    cancel_calls: list[DispatchEnvelope] = field(default_factory=list)
    exit_calls: list[DispatchEnvelope] = field(default_factory=list)
    modify_exit_errors: list[Exception] = field(default_factory=list)
    exit_errors: list[Exception | None] = field(default_factory=list)
    """Outcome of each next ``execute_exit``, in order: raised, or ``None`` to land."""
    cancel_settles: bool = True

    @staticmethod
    def get_capabilities() -> ExchangeCapabilities:
        return ExchangeCapabilities()

    async def get_position(self, symbol):
        return self.position

    async def get_open_orders(self, symbol=None) -> list[ExchangeOrder]:
        return list(self.open_orders)

    async def modify_exit(self, old, new):
        self.modify_exit_calls.append((old, new))
        if self.modify_exit_errors:
            raise self.modify_exit_errors.pop(0)
        return []

    async def execute_exit(self, envelope):
        self.exit_calls.append(envelope)
        if self.exit_errors and (error := self.exit_errors.pop(0)) is not None:
            raise error
        return []

    async def execute_cancel(self, envelope):
        self.cancel_calls.append(envelope)
        return self.cancel_settles


def _open_ctx(store: BrokerStore) -> RunContext:
    return store.open_run(
        RunIdentity(strategy_id="t164", symbol=SYMBOL, timeframe="60",
                    account_id="test-account", label=None),
        script_source="// t164",
    )


def _engine(ctx: RunContext, broker: _Broker | None = None) -> OrderSyncEngine:
    return OrderSyncEngine(
        broker=broker or _Broker(),  # type: ignore[arg-type]
        position=BrokerPosition(), symbol=SYMBOL, run_tag=ctx.run_tag, mintick=0.0001,
        store_ctx=ctx,
    )


def _resume_previous_run(store: BrokerStore) -> None:
    """A previous process latched an intervention; the operator resumed it."""
    ctx = _open_ctx(store)
    engine = _engine(ctx)
    engine.record_quarantine("operator moved the fail-safe stop",
                             kind=QuarantineKind.EXTERNAL_INTERVENTION)
    assert engine.quarantine_id is not None
    store.request_quarantine_resume(ctx.run_id, engine.quarantine_id)
    ctx.close()


def _seed_parent(ctx: RunContext, saved: dict | None) -> None:
    extras: dict = {"kind": "position"}
    if saved is not None:
        extras[NATIVE_FAILSAFE_LEVELS_EXTRA] = saved
    ctx.upsert_order(PARENT, symbol=SYMBOL, side="buy", qty=1.0, filled_qty=1.0,
                     state="confirmed", pine_entry_id="L", extras=extras)


def _rehydrate(engine: OrderSyncEngine, puts: list[NativeBracketSnapshot] | None) -> None:
    """What the first post-restart sync does for a replayed partial-bracket SL leg.

    :param puts: Collects the native fail-safe PUTs; ``None`` runs without an
        actuator (a plugin that places no native fail-safe).
    """
    leg = PartialBracketLeg(
        coid="leg-sl", symbol=SYMBOL, pine_id="L-X1", from_entry="L",
        leg_kind=LEG_KIND_SL_PARTIAL, leg_state=LEG_STATE_ARMED, side="sell", qty=0.5,
        intent_key=f"L-X1{INTENT_KEY_SEP}L", parent_pine_entry_id="L",
        parent_entry_dispatch_ref=PARENT, intent_partial_qty=0.5, trigger_level=1.0970,
    )
    pbe = engine._partial_bracket_engine  # noqa: SLF001
    pbe._legs[leg.key] = leg  # noqa: SLF001
    pbe._legs_by_parent.setdefault((SYMBOL, "L"), set()).add(leg.key)  # noqa: SLF001
    if puts is not None:
        engine.set_native_bracket_dispatcher(puts.append)
    engine._current_bar_ts_ms = BAR_TS  # noqa: SLF001
    engine._rehydrate_native_failsafe_from_replayed_legs()  # noqa: SLF001


def _observe(engine: OrderSyncEngine, stop: float | None) -> None:
    engine.enqueue_native_bracket_observed(PARENT, stop_level=stop, profit_level=None,
                                           trailing_stop=None)


def _events(ctx: RunContext, kind: str) -> list[dict]:
    return list(ctx.iter_events_by_kind_since(kind, 0))


# --- native fail-safe -------------------------------------------------------------


def __test_a_restart_without_a_native_actuator_never_blocks_entries__(tmp_path: Path) -> None:
    # Measured on Bybit (no native fail-safe actuator) after a rotation with
    # an open partial bracket, 2026-10-17/18: the replayed parent started
    # DEGRADING, nothing could confirm it, and 30 s later DEGRADED dropped
    # every entry signal until the position was gone.
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _seed_parent(ctx, None)
        engine = _engine(ctx)
        _rehydrate(engine, None)
        manager = engine._native_failsafe_manager  # noqa: SLF001
        state = manager.get_state(PARENT)
        assert state is not None and state.health is FailsafeHealth.HEALTHY
        engine.drive_native_failsafe(now_ms=BAR_TS + 3_600_000)
        assert state.health is FailsafeHealth.HEALTHY
        assert not manager.block_new_entry(symbol=SYMBOL, pine_id="L", bar_ts_ms=BAR_TS)

    with BrokerStore(tmp_path / "actuated.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _seed_parent(ctx, None)
        engine = _engine(ctx)
        _rehydrate(engine, [])
        state = engine._native_failsafe_manager.get_state(PARENT)  # noqa: SLF001
        assert state is not None and state.health is FailsafeHealth.DEGRADING


@pytest.mark.parametrize("observed_first", [True, False],
                         ids=["observed-before-the-put", "observed-after-the-put"])
def __test_a_resume_accepts_the_operators_fail_safe_level_and_re_asserts_the_bots__(
        tmp_path: Path, observed_first: bool,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        _resume_previous_run(store)
        ctx = _open_ctx(store)
        _seed_parent(ctx, {'levels': [[None, None, None], [1.0970, None, None]],
                           'confirmed': [1.0970, None, None]})
        engine = _engine(ctx)
        assert engine.resumed_from_quarantine
        puts: list[NativeBracketSnapshot] = []
        _rehydrate(engine, puts)

        if observed_first:
            _observe(engine, 1.0900)  # polled before the first live sync
        engine.drive_native_failsafe(now_ms=float(BAR_TS))
        if not observed_first:
            _observe(engine, 1.0900)  # a snapshot older than the PUT
            engine.drive_native_failsafe(now_ms=float(BAR_TS + 1_000))

        assert [p.stop_level for p in puts] == [pytest.approx(1.0970)]
        assert not engine.quarantined
        state = engine._native_failsafe_manager.get_state(PARENT)  # noqa: SLF001
        assert state is not None and state.owner is FailsafeOwner.ENGINE_FAILSAFE
        assert len(_events(ctx, 'native_failsafe_resume_level_accepted')) == 1

        _observe(engine, 1.0970)
        engine.drive_native_failsafe(now_ms=float(BAR_TS + 2_000))
        assert state.health is FailsafeHealth.HEALTHY
        ctx.close()


def __test_a_resume_whose_process_stopped_before_its_first_sync_is_not_lost__(
        tmp_path: Path,
) -> None:
    """The restarted process's connect failed: the next start still resumes."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        _resume_previous_run(store)
        died = _open_ctx(store)
        _seed_parent(died, {'levels': [[None, None, None], [1.0970, None, None]],
                            'confirmed': [1.0970, None, None]})
        assert _engine(died).resumed_from_quarantine
        died.close()

        ctx = _open_ctx(store)
        engine = _engine(ctx)
        assert engine.resumed_from_quarantine
        puts: list[NativeBracketSnapshot] = []
        _rehydrate(engine, puts)
        _observe(engine, 1.0900)
        engine.drive_native_failsafe(now_ms=float(BAR_TS))

        assert not engine.quarantined, "the operator's level was accepted by the resume"
        assert len(_events(ctx, 'native_failsafe_resume_level_accepted')) == 1
        ctx.close()


def __test_a_plain_restart_accepts_an_older_own_level_but_quarantines_a_foreign_one__(
        tmp_path: Path,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        # The previous process sent 1.0950, then ratcheted to 1.0960 and died
        # before that PUT was answered.
        _seed_parent(ctx, {'levels': [[1.0950, None, None], [1.0960, None, None]],
                           'confirmed': [1.0950, None, None]})
        engine = _engine(ctx)
        puts: list[NativeBracketSnapshot] = []
        _rehydrate(engine, puts)

        _observe(engine, 1.0960)
        engine.drive_native_failsafe(now_ms=float(BAR_TS))
        assert not engine.quarantined, "the bot's own older level is never an edit"
        assert [p.stop_level for p in puts] == [pytest.approx(1.0970)]

        _observe(engine, 1.0800)
        engine.drive_native_failsafe(now_ms=float(BAR_TS + 1_000))
        assert engine.quarantined
        assert engine.quarantine_kind is QuarantineKind.EXTERNAL_INTERVENTION
        ctx.close()


def __test_a_plain_restart_without_a_record_reports_but_never_quarantines__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _seed_parent(ctx, None)
        engine = _engine(ctx)
        puts: list[NativeBracketSnapshot] = []
        _rehydrate(engine, puts)

        _observe(engine, 1.0900)
        engine.drive_native_failsafe(now_ms=float(BAR_TS))

        assert not engine.quarantined
        assert len(_events(ctx, 'native_failsafe_restart_unverified_level')) == 1
        assert any(r.levelno == logging.ERROR and "no own record" in r.getMessage()
                   for r in caplog.records)
        ctx.close()


def __test_every_fail_safe_put_is_recorded_before_it_is_sent__(tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _seed_parent(ctx, None)
        engine = _engine(ctx)
        seen_on_record: list[list] = []

        def _dispatcher(snapshot: NativeBracketSnapshot) -> None:
            row = ctx.get_order(PARENT)
            assert row is not None
            seen_on_record.append(row.extras[NATIVE_FAILSAFE_LEVELS_EXTRA]['levels'])
            raise ConnectionError("the process dies before the answer")

        _rehydrate(engine, [])
        engine.set_native_bracket_dispatcher(_dispatcher)
        engine.drive_native_failsafe(now_ms=float(BAR_TS))

        assert seen_on_record and [1.097, None, None] in [
            [round(v, 4) if v is not None else None for v in triple]
            for triple in seen_on_record[0]]
        ctx.close()


# --- re-assert of the strategy's persistent exits after a resume --------------------


def _long(size: float) -> ExchangePosition:
    return ExchangePosition(
        symbol=SYMBOL, side="long", size=size, entry_price=1.1000,
        unrealized_pnl=0.0, liquidation_price=None, leverage=1.0, margin_mode="cross",
    )


def _adopted_exit_run(store: BrokerStore, *, resumed: bool, mapping: list[str] | None,
                      ) -> tuple[RunContext, OrderSyncEngine, _Broker]:
    """A restarted run whose venue carries the exit the diff will adopt (``mapping``)."""
    from pynecore.lib.strategy import Order, _order_type_close

    if resumed:
        _resume_previous_run(store)
    ctx = _open_ctx(store)
    ctx.upsert_order("row-l", symbol=SYMBOL, side="buy", qty=1.0, filled_qty=1.0,
                     state="confirmed", pine_entry_id="L", extras={"kind": "position"})
    broker = _Broker(position=_long(1.0))
    engine = _engine(ctx, broker)
    engine.reconcile()
    engine._position.exit_orders[("L-X", "L")] = Order(  # noqa: SLF001
        "L", -1.0, order_type=_order_type_close, exit_id="L-X",
        limit=1.1200, stop=1.0900,
    )
    if mapping is not None:
        engine._order_mapping[f"L-X{INTENT_KEY_SEP}L"] = list(mapping)  # noqa: SLF001
    return ctx, engine, broker


def __test_a_resume_re_arms_an_adopted_order_based_exit_on_fresh_ids__(tmp_path: Path) -> None:
    """The operator may have moved or cancelled a leg: the bot restores its own bracket.

    The fresh bracket goes out in the same sync as the cancel, so the position
    is never left without the bot's levels until the next bar.
    """
    key = f"L-X{INTENT_KEY_SEP}L"
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=True, mapping=["xchg-sl"])

        engine.sync(BAR_TS, last_price=1.1000)
        assert [c.intent.pine_id for c in broker.cancel_calls] == ["L-X"]
        assert len(broker.exit_calls) == 1, "re-armed in the same sync"
        rearmed = broker.exit_calls[0]
        assert isinstance(rearmed.intent, ExitIntent)
        assert (rearmed.intent.tp_price, rearmed.intent.sl_price) == (1.1200, 1.0900)
        cancelled = broker.cancel_calls[0]
        assert (rearmed.client_order_id(KIND_EXIT_SL)
                != cancelled.client_order_id(KIND_EXIT_SL)), \
            "fresh client order ids for the new bracket"
        assert key in engine.active_intents
        assert len(list(ctx.iter_events_by_kind_since('resume_exit_reasserted', 0))) == 1

        engine.sync(BAR_TS + 60_000, last_price=1.1000)
        engine.sync(BAR_TS + 120_000, last_price=1.1000)
        assert len(broker.exit_calls) == 1 and len(broker.cancel_calls) == 1, "once only"
        ctx.close()


def __test_a_re_arm_whose_cancel_did_not_settle_waits_for_it__(tmp_path: Path) -> None:
    """The old and the new legs must never rest together."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=True, mapping=["xchg-sl"])
        broker.cancel_settles = False

        engine.sync(BAR_TS, last_price=1.1000)
        assert len(broker.cancel_calls) == 1
        assert broker.exit_calls == []

        broker.cancel_settles = True
        engine.sync(BAR_TS + 60_000, last_price=1.1000)
        assert len(broker.cancel_calls) == 2, "the parked cancel is retried first"
        assert len(broker.exit_calls) == 1, "then the bracket is re-armed"
        ctx.close()


def __test_a_resume_amends_an_adopted_position_attached_bracket_in_place__(
        tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=True,
                                                mapping=["bracket:pos-1"])
        engine.sync(BAR_TS, last_price=1.1000)

        assert broker.cancel_calls == []
        assert len(broker.modify_exit_calls) == 1
        old, new = broker.modify_exit_calls[0]
        assert old.intent == new.intent
        assert (new.intent.tp_price, new.intent.sl_price) == (1.1200, 1.0900)
        ctx.close()


@pytest.mark.parametrize(("resumed", "mapping"), [
    (False, ["xchg-sl"]),
    (True, None),
], ids=["plain-restart", "fresh-dispatch"])
def __test_no_re_assert_without_a_resume_or_for_a_freshly_dispatched_exit__(
        tmp_path: Path, resumed: bool, mapping: list[str] | None) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=resumed, mapping=mapping)
        engine.sync(BAR_TS, last_price=1.1000)
        engine.sync(BAR_TS + 60_000, last_price=1.1000)

        assert broker.cancel_calls == [] and broker.modify_exit_calls == []
        assert len(broker.exit_calls) == (0 if mapping else 1)
        ctx.close()


def __test_an_interrupted_re_assert_is_retried_on_the_next_sync__(tmp_path: Path) -> None:
    from pynecore.core.broker.exceptions import ExchangeConnectionError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=True,
                                                mapping=["bracket:pos-1"])
        broker.modify_exit_errors.append(ExchangeConnectionError("socket reset"))
        engine.sync(BAR_TS, last_price=1.1000)
        assert len(broker.modify_exit_calls) == 1
        assert list(ctx.iter_events_by_kind_since('resume_exit_reasserted', 0)) == []

        engine.sync(BAR_TS + 60_000, last_price=1.1000)
        assert len(broker.modify_exit_calls) == 2
        events = list(ctx.iter_events_by_kind_since('resume_exit_reasserted', 0))
        assert len(events) == 1

        engine.sync(BAR_TS + 120_000, last_price=1.1000)
        assert len(broker.modify_exit_calls) == 2
        ctx.close()


def __test_a_software_partial_bracket_is_not_re_asserted__(tmp_path: Path) -> None:
    """Its venue stop is the native fail-safe, which the restart handles on its own."""
    key = f"L-X{INTENT_KEY_SEP}L"
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=True, mapping=["leg-tp"])
        leg = PartialBracketLeg(
            coid="leg-tp", symbol=SYMBOL, pine_id="L-X", from_entry="L",
            leg_kind=LEG_KIND_SL_PARTIAL, leg_state=LEG_STATE_ARMED, side="sell", qty=1.0,
            intent_key=key, parent_pine_entry_id="L", parent_entry_dispatch_ref="row-l",
            intent_partial_qty=1.0, trigger_level=1.0900,
        )
        pbe = engine._partial_bracket_engine  # noqa: SLF001
        pbe._legs[leg.key] = leg  # noqa: SLF001
        pbe._legs_by_parent.setdefault((SYMBOL, "L"), set()).add(leg.key)  # noqa: SLF001
        engine._resume_reassert = {key: ("leg-tp",)}  # noqa: SLF001
        engine._active_intents[key] = ExitIntent(  # noqa: SLF001
            pine_id="L-X", from_entry="L", symbol=SYMBOL, side="sell", qty=1.0,
            tp_price=1.1200, sl_price=1.0900,
        )

        engine._reassert_exits_after_resume()  # noqa: SLF001

        assert broker.cancel_calls == [] and broker.modify_exit_calls == []
        assert engine._resume_reassert == {}  # noqa: SLF001
        assert list(ctx.iter_events_by_kind_since('resume_exit_reasserted', 0)) == []
        ctx.close()


def __test_a_re_arm_waiting_for_its_cancel_is_not_reported_as_re_asserted__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The venue carries no bot level until the parked cancel lands."""
    import logging

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=True, mapping=["xchg-sl"])
        broker.cancel_settles = False
        caplog.clear()

        engine.sync(BAR_TS, last_price=1.1000)

        assert broker.exit_calls == []
        assert not [r for r in caplog.records if "re-asserted after" in r.getMessage()]
        assert any(r.levelno == logging.WARNING and "not applied (rearm_deferred)"
                   in r.getMessage() for r in caplog.records)
        events = list(ctx.iter_events_by_kind_since('resume_exit_reasserted', 0))
        assert [e['mode'] for e in events] == ['rearm_deferred']
        ctx.close()


def __test_an_amend_the_plugin_skips_is_not_reported_as_re_asserted__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    from pynecore.core.broker.exceptions import OrderSkippedByPlugin

    key = f"L-X{INTENT_KEY_SEP}L"
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=True,
                                                mapping=["bracket:pos-1"])
        broker.modify_exit_errors.append(
            OrderSkippedByPlugin("below min size", intent_key=key, reason="below_min_size"))
        caplog.clear()

        engine.sync(BAR_TS, last_price=1.1000)

        assert len(broker.modify_exit_calls) == 1
        assert not [r for r in caplog.records if "re-asserted after" in r.getMessage()]
        events = list(ctx.iter_events_by_kind_since('resume_exit_reasserted', 0))
        assert [e['mode'] for e in events] == ['amend_skipped']
        assert key not in engine.active_intents, "the next sync dispatches it afresh"
        ctx.close()


def __test_the_resumed_latch_ends_once_the_re_assert_went_out__(tmp_path: Path) -> None:
    from pynecore.core.broker.exceptions import ExchangeConnectionError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=True,
                                                mapping=["bracket:pos-1"])
        open_ids = [row.quarantine_id for row in store.open_quarantines()]
        assert len(open_ids) == 1
        broker.modify_exit_errors.append(ExchangeConnectionError("socket reset"))

        engine.sync(BAR_TS, last_price=1.1000)
        assert [row.quarantine_id for row in store.open_quarantines()] == open_ids, \
            "the re-assert is still owed"

        engine.sync(BAR_TS + 60_000, last_price=1.1000)
        assert store.open_quarantines() == []
        row = ctx.get_quarantine(open_ids[0])
        assert row is not None and row.end_reason == 'operator_resume'
        ctx.close()


# --- re-assert of a journaled exit the restarted diff dispatches again ---------------


@dataclass
class _JournalingBroker(_Broker):
    """A venue that answers an exit dispatch with ``refs`` and records closes."""

    refs: tuple[str, ...] = ("xchg-sl",)
    close_calls: list[DispatchEnvelope] = field(default_factory=list)

    async def execute_exit(self, envelope):
        self.exit_calls.append(envelope)
        if self.exit_errors and (error := self.exit_errors.pop(0)) is not None:
            raise error
        return [ExchangeOrder(
            id=ref, symbol=SYMBOL, side="sell", order_type=OrderType.STOP, qty=1.0,
            filled_qty=0.0, remaining_qty=1.0, price=None, stop_price=1.0900,
            average_fill_price=None, status=OrderStatus.OPEN, timestamp=0.0, fee=0.0,
            fee_currency="", client_order_id=envelope.client_order_id(KIND_EXIT_SL),
        ) for ref in self.refs]

    async def execute_close(self, envelope):
        self.close_calls.append(envelope)
        return ExchangeOrder(
            id=f"close-{len(self.close_calls)}", symbol=SYMBOL, side="sell",
            order_type=OrderType.MARKET, qty=1.0, filled_qty=0.0, remaining_qty=1.0,
            price=None, stop_price=None, average_fill_price=None, status=OrderStatus.OPEN,
            timestamp=0.0, fee=0.0, fee_currency="",
            client_order_id=envelope.client_order_id(KIND_CLOSE),
        )

    @staticmethod
    def get_residual_orders_after_bracket_attach_reject(_context) -> list[str]:
        return []

    async def cancel_broker_order_ref(self, ref) -> None:
        raise AssertionError(f"unexpected residual cancel of {ref}")


def _strategy_exit(engine: OrderSyncEngine) -> None:
    """The strategy's persistent exit, as the script declares it on every start."""
    from pynecore.lib.strategy import Order, _order_type_close

    engine._position.exit_orders[("L-X", "L")] = Order(  # noqa: SLF001
        "L", -1.0, order_type=_order_type_close, exit_id="L-X",
        limit=1.1200, stop=1.0900,
    )


def _restart_with_a_journaled_exit(
        store: BrokerStore, *, refs: tuple[str, ...], resumed: bool,
) -> tuple[RunContext, OrderSyncEngine, _JournalingBroker, DispatchEnvelope]:
    """The previous process dispatched the exit; the next one starts over it.

    Nothing seeds the restarted engine's order mapping: what it does with the
    exit is what a real restart does.
    """
    first = _open_ctx(store)
    first.upsert_order("row-l", symbol=SYMBOL, side="buy", qty=1.0, filled_qty=1.0,
                       state="confirmed", pine_entry_id="L", extras={"kind": "position"})
    previous_broker = _JournalingBroker(position=_long(1.0), refs=refs)
    previous = _engine(first, previous_broker)
    previous.reconcile()
    _strategy_exit(previous)
    previous.sync(BAR_TS, last_price=1.1000)
    assert len(previous_broker.exit_calls) == 1
    journaled = previous_broker.exit_calls[0]
    if resumed:
        previous.record_quarantine("operator moved the stop-loss",
                                   kind=QuarantineKind.EXTERNAL_INTERVENTION)
        assert previous.quarantine_id is not None
        store.request_quarantine_resume(first.run_id, previous.quarantine_id)
    first.close()

    ctx = _open_ctx(store)
    broker = _JournalingBroker(position=_long(1.0), refs=refs)
    engine = _engine(ctx, broker)
    engine.reconcile()
    _strategy_exit(engine)
    return ctx, engine, broker, journaled


def __test_a_resume_re_arms_a_journaled_order_based_exit_the_restart_dispatched_again__(
        tmp_path: Path) -> None:
    """The same client order id may come back as the operator left it: re-arm on fresh ids."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker, journaled = _restart_with_a_journaled_exit(
            store, refs=("xchg-sl",), resumed=True)

        engine.sync(BAR_TS + 60_000, last_price=1.1000)

        redispatched, rearmed = broker.exit_calls
        assert (redispatched.client_order_id(KIND_EXIT_SL)
                == journaled.client_order_id(KIND_EXIT_SL))
        assert [c.intent.pine_id for c in broker.cancel_calls] == ["L-X"]
        assert (rearmed.client_order_id(KIND_EXIT_SL)
                != journaled.client_order_id(KIND_EXIT_SL))
        assert isinstance(rearmed.intent, ExitIntent)
        assert (rearmed.intent.tp_price, rearmed.intent.sl_price) == (1.1200, 1.0900)
        assert [e['mode'] for e in _events(ctx, 'resume_exit_reasserted')] == ['rearm']
        assert store.open_quarantines() == []
        assert not engine.quarantined

        engine.sync(BAR_TS + 120_000, last_price=1.1000)
        assert len(broker.exit_calls) == 2 and len(broker.cancel_calls) == 1, "once only"
        ctx.close()


def __test_a_resume_amends_a_journaled_position_attached_exit_the_restart_dispatched_again__(
        tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker, _journaled = _restart_with_a_journaled_exit(
            store, refs=("bracket:pos-1",), resumed=True)

        engine.sync(BAR_TS + 60_000, last_price=1.1000)

        assert len(broker.exit_calls) == 1
        assert broker.cancel_calls == []
        assert len(broker.modify_exit_calls) == 1
        old, new = broker.modify_exit_calls[0]
        assert old.intent == new.intent
        assert (new.intent.tp_price, new.intent.sl_price) == (1.1200, 1.0900)
        assert [e['mode'] for e in _events(ctx, 'resume_exit_reasserted')] == ['amend']
        assert store.open_quarantines() == []
        ctx.close()


@pytest.mark.parametrize("refs", [("xchg-sl",), ("bracket:pos-1",)],
                         ids=["order-based", "position-attached"])
def __test_a_plain_restart_never_re_asserts_a_journaled_exit__(
        tmp_path: Path, refs: tuple[str, ...]) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker, _journaled = _restart_with_a_journaled_exit(
            store, refs=refs, resumed=False)

        engine.sync(BAR_TS + 60_000, last_price=1.1000)
        engine.sync(BAR_TS + 120_000, last_price=1.1000)

        assert len(broker.exit_calls) == 1
        assert broker.cancel_calls == [] and broker.modify_exit_calls == []
        assert _events(ctx, 'resume_exit_reasserted') == []
        ctx.close()


def __test_a_re_assert_over_a_crossed_level_closes_the_parent_at_market__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The bot's own stop is already crossed after the resume: Pine's exit, not 'not applied'."""
    import logging

    from pynecore.core.broker.exceptions import ExchangeOrderRejectedError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker, _journaled = _restart_with_a_journaled_exit(
            store, refs=("bracket:pos-1",), resumed=True)
        broker.modify_exit_errors.append(
            ExchangeOrderRejectedError("stop above the market", level_crossed=True))
        caplog.clear()

        engine.sync(BAR_TS + 60_000, last_price=1.0850)

        assert len(broker.modify_exit_calls) == 1
        assert len(broker.close_calls) == 1
        close = broker.close_calls[0].intent
        assert getattr(close, 'synthetic_kind', None) == 'defensive_close'
        assert [e['mode'] for e in _events(ctx, 'resume_exit_reasserted')] == [
            'crossed_close']
        assert not [r for r in caplog.records if "not applied" in r.getMessage()]
        assert any(r.levelno == logging.WARNING and "closed at market" in r.getMessage()
                   for r in caplog.records)
        assert not engine.quarantined
        ctx.close()


@pytest.mark.parametrize(("crossed", "level_name"), [(True, "WARNING"), (False, "ERROR")],
                         ids=["crossed", "refused"])
def __test_a_replay_refusal_over_a_crossed_level_is_explained_on_a_plain_restart__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, crossed: bool, level_name: str,
) -> None:
    """The market crossing the journaled level explains the refusal; nothing else does."""
    import logging

    from pynecore.core.broker.exceptions import ExchangeOrderRejectedError
    from pynecore.core.broker.one_way_emulator import BracketReplayRefusal

    key = f"L-X{INTENT_KEY_SEP}L"
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        engine = _engine(ctx)
        engine._one_way_emulator._replay_refusals.append(BracketReplayRefusal(  # noqa: SLF001
            intent_key=key, leg_id="10", tp_price=1.1200, sl_price=1.0900,
            error=ExchangeOrderRejectedError("stop above the market", level_crossed=crossed),
        ))
        caplog.clear()

        engine._note_bracket_replay_refusals()  # noqa: SLF001

        records = [r for r in caplog.records
                   if "refused to re-assert bracket" in r.getMessage()]
        assert [r.levelno for r in records] == [getattr(logging, level_name)]
        events = _events(ctx, 'bracket_replay_refused')
        assert [(e['level_crossed'], e['resumed']) for e in events] == [(crossed, False)]
        ctx.close()


# --- a refused, declined or unconfirmed re-assert ------------------------------------


def __test_a_re_assert_declined_by_a_closed_market_is_retried_before_the_resume_ends__(
        tmp_path: Path) -> None:
    from pynecore.core.broker.exceptions import MarketClosedError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=True,
                                                mapping=["bracket:pos-1"])
        open_ids = [row.quarantine_id for row in store.open_quarantines()]
        broker.modify_exit_errors.append(MarketClosedError("market closed"))

        engine.sync(BAR_TS, last_price=1.1000)
        assert len(broker.modify_exit_calls) == 1
        assert _events(ctx, 'resume_exit_reasserted') == []
        assert [row.quarantine_id for row in store.open_quarantines()] == open_ids, \
            "the re-assert is still owed"

        engine.sync(BAR_TS + 60_000, last_price=1.1000)
        assert len(broker.modify_exit_calls) == 2, "sent again once the market reopens"
        assert [e['mode'] for e in _events(ctx, 'resume_exit_reasserted')] == ['amend']
        assert store.open_quarantines() == []
        ctx.close()


def __test_a_re_assert_amend_the_venue_refuses_is_dispatched_fresh_at_once__(
        tmp_path: Path) -> None:
    """The levels the operator left are not the bot's: a refusal never keeps them silently."""
    from pynecore.core.broker.exceptions import ExchangeOrderRejectedError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker, _journaled = _restart_with_a_journaled_exit(
            store, refs=("bracket:pos-1",), resumed=True)
        broker.modify_exit_errors.append(ExchangeOrderRejectedError("venue rule"))

        engine.sync(BAR_TS + 60_000, last_price=1.1000)

        assert len(broker.modify_exit_calls) == 1
        assert len(broker.exit_calls) == 2, "the restart's dispatch, then the fresh re-assert"
        fresh = broker.exit_calls[1].intent
        assert isinstance(fresh, ExitIntent)
        assert (fresh.tp_price, fresh.sl_price) == (1.1200, 1.0900)
        assert [e['mode'] for e in _events(ctx, 'resume_exit_reasserted')] == ['amend']
        assert store.open_quarantines() == []
        assert not engine.quarantined
        ctx.close()


def __test_a_re_assert_the_venue_refuses_twice_closes_the_parent_defensively__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from pynecore.core.broker.exceptions import ExchangeOrderRejectedError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker, _journaled = _restart_with_a_journaled_exit(
            store, refs=("bracket:pos-1",), resumed=True)
        broker.modify_exit_errors.append(ExchangeOrderRejectedError("venue rule"))
        # The restart dispatches the journaled exit again, then the re-assert's fresh one.
        broker.exit_errors.extend([None, ExchangeOrderRejectedError("venue rule")])
        caplog.clear()

        engine.sync(BAR_TS + 60_000, last_price=1.1000)

        assert len(broker.close_calls) == 1
        assert getattr(broker.close_calls[0].intent, 'synthetic_kind', None) == \
            'defensive_close'
        assert [e['mode'] for e in _events(ctx, 'resume_exit_reasserted')] == [
            'defensive_close']
        assert not [r for r in caplog.records if "re-asserted after" in r.getMessage()]
        assert engine.quarantine_kind is QuarantineKind.ENTRY_BLOCK
        assert not engine.halted
        ctx.close()


def __test_a_re_arm_the_venue_refuses_twice_is_audited_as_a_defensive_close__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from pynecore.core.broker.exceptions import ExchangeOrderRejectedError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker, _journaled = _restart_with_a_journaled_exit(
            store, refs=("xchg-sl",), resumed=True)
        # The restart dispatches the journaled exit again; the re-arm is refused twice.
        broker.exit_errors.extend([None, ExchangeOrderRejectedError("venue rule"),
                                   ExchangeOrderRejectedError("venue rule")])
        caplog.clear()

        engine.sync(BAR_TS + 60_000, last_price=1.1000)

        assert len(broker.close_calls) == 1
        assert getattr(broker.close_calls[0].intent, 'synthetic_kind', None) == \
            'defensive_close'
        assert [e['mode'] for e in _events(ctx, 'resume_exit_reasserted')] == [
            'defensive_close']
        assert any("closed defensively" in r.getMessage() for r in caplog.records)
        assert not [r for r in caplog.records
                    if "re-asserted after" in r.getMessage()
                    or "the next sync arms the exit" in r.getMessage()]
        assert engine.quarantine_kind is QuarantineKind.ENTRY_BLOCK
        ctx.close()


def __test_a_re_arm_whose_outcome_is_unknown_is_not_reported_as_re_asserted__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from pynecore.core.broker.exceptions import OrderDispositionUnknownError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker, _journaled = _restart_with_a_journaled_exit(
            store, refs=("xchg-sl",), resumed=True)
        broker.exit_errors.extend([
            None, OrderDispositionUnknownError("timed out", client_order_id="rearm-coid")])
        caplog.clear()

        engine.sync(BAR_TS + 60_000, last_price=1.1000)

        assert len(broker.exit_calls) == 2, "the re-arm went out"
        assert not [r for r in caplog.records if "re-asserted after" in r.getMessage()]
        assert any(r.levelno == logging.WARNING and "(rearm_unconfirmed)" in r.getMessage()
                   for r in caplog.records)
        assert [e['mode'] for e in _events(ctx, 'resume_exit_reasserted')] == [
            'rearm_unconfirmed']
        ctx.close()


def __test_a_re_assert_whose_outcome_is_unknown_is_not_reported_as_re_asserted__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from pynecore.core.broker.exceptions import OrderDispositionUnknownError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=True,
                                                mapping=["bracket:pos-1"])
        broker.modify_exit_errors.append(
            OrderDispositionUnknownError("timed out", client_order_id="amend-coid"))
        caplog.clear()

        engine.sync(BAR_TS, last_price=1.1000)

        assert not [r for r in caplog.records if "re-asserted after" in r.getMessage()]
        assert any(r.levelno == logging.WARNING and "(amend_unconfirmed)" in r.getMessage()
                   for r in caplog.records)
        assert [e['mode'] for e in _events(ctx, 'resume_exit_reasserted')] == [
            'amend_unconfirmed']
        ctx.close()


def __test_an_amend_the_plugin_skips_is_dispatched_afresh_on_the_next_sync__(
        tmp_path: Path) -> None:
    """Not adopted back from the venue: the operator's bracket is not the bot's exit."""
    from pynecore.core.broker.exceptions import OrderSkippedByPlugin

    key = f"L-X{INTENT_KEY_SEP}L"
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx, engine, broker = _adopted_exit_run(store, resumed=True,
                                                mapping=["bracket:pos-1"])
        broker.modify_exit_errors.append(
            OrderSkippedByPlugin("below min size", intent_key=key, reason="below_min_size"))

        engine.sync(BAR_TS, last_price=1.1000)
        assert broker.exit_calls == []
        engine.sync(BAR_TS + 60_000, last_price=1.1000)

        assert len(broker.exit_calls) == 1
        assert (broker.exit_calls[0].intent.tp_price,
                broker.exit_calls[0].intent.sl_price) == (1.1200, 1.0900)
        ctx.close()


# --- a resume applied again, a resumed entry block, a failed completion -------------


def _resume_after_a_bracket(store: BrokerStore) -> None:
    """The bot held a guarded bracket, an intervention latched, the operator resumed it."""
    ctx = _open_ctx(store)
    ctx.upsert_order("leg-1", symbol=SYMBOL, side="sell", qty=1.0, state="confirmed",
                     extras={})
    guard = ProtectionLevelGuard(lambda reason, context: None, store_ctx=ctx)
    guard.expect("p1", _BOT_LEVELS, row_coid="leg-1")
    guard.settle("p1", _BOT_LEVELS)
    ctx.close()
    _resume_previous_run(store)


_BOT_LEVELS = ProtectionLevels(stop=1.0900, limit=1.1200)
_OPERATOR_LEVELS = ProtectionLevels(stop=1.0800, limit=None)
_DOWNTIME_LEVELS = ProtectionLevels(stop=1.0700, limit=None)


def _guard_reports(ctx: RunContext) -> tuple[ProtectionLevelGuard, list[str]]:
    reports: list[str] = []
    return ProtectionLevelGuard(lambda reason, context: reports.append(reason),
                                store_ctx=ctx), reports


@pytest.fixture
def _wall_clock(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A wall clock the test moves, so two processes never share a millisecond."""
    import time

    clock = SimpleNamespace(now=1_000.0)
    monkeypatch.setattr(time, "time", lambda: clock.now)
    return clock


def __test_a_resume_applied_again_judges_what_changed_after_the_first_process_judged__(
        tmp_path: Path, _wall_clock: SimpleNamespace) -> None:
    """The first post-resume process adopted the operator's stop, then stopped early.

    Someone moved the stop while no process ran: the next process applies the
    resume again but keeps the baseline the first one adopted, so the move is
    an intervention, and the resume boundary stays the first application's.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        _resume_after_a_bracket(store)
        first = _open_ctx(store)
        first_engine = _engine(first)
        assert first_engine.resumed_from_quarantine
        guard, reports = _guard_reports(first)
        assert not guard.observe("p1", _OPERATOR_LEVELS, tick=0.0001, immediate=True)
        assert reports == []
        first.close()  # stopped before its first sync: the resume is not completed

        _wall_clock.now += 3_600.0
        again = _open_ctx(store)
        engine = _engine(again)
        assert engine.resumed_from_quarantine
        assert engine.operator_resume_ts_ms == first_engine.operator_resume_ts_ms
        payloads = [p for _k, _c, _x, p in again.iter_events_by_kind_for_run_id(
            'quarantine_resumed')]
        assert [p['again'] for p in payloads] == [False, True]
        guard, reports = _guard_reports(again)
        assert guard.observe("p1", _DOWNTIME_LEVELS, tick=0.0001, immediate=True)
        assert len(reports) == 1
        again.close()


def __test_a_resume_applied_again_still_accepts_what_no_process_judged__(
        tmp_path: Path) -> None:
    """The first post-resume process stopped before its guard saw the venue."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        _resume_after_a_bracket(store)
        first = _open_ctx(store)
        assert _engine(first).resumed_from_quarantine
        first.close()

        again = _open_ctx(store)
        assert _engine(again).resumed_from_quarantine
        guard, reports = _guard_reports(again)
        assert not guard.observe("p1", _OPERATOR_LEVELS, tick=0.0001, immediate=True)
        assert reports == []
        again.close()


def __test_a_resume_applied_again_keeps_the_order_acceptance_the_first_one_settled__(
        tmp_path: Path, _wall_clock: SimpleNamespace) -> None:
    from pynecore.core.broker.disappearance import MISSING_ACCEPTED_BY_RESUME_EXTRA

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        seed = _open_ctx(store)
        for coid in ("seen", "unseen", "stamped"):
            seed.upsert_order(coid, symbol=SYMBOL, side="sell", qty=1.0,
                              state="confirmed", extras={})
        seed.close()
        _resume_previous_run(store)

        first = _open_ctx(store)
        _engine(first)
        # The first process saw "seen" on the venue (its acceptance is gone) and
        # stamped "stamped" missing after its window (judged as usual).
        seen = first.get_order("seen")
        assert seen is not None and seen.extras[MISSING_ACCEPTED_BY_RESUME_EXTRA]
        first.upsert_order("seen", extras={})
        stamped = first.get_order("stamped")
        assert stamped is not None
        first.upsert_order("stamped", extras={**stamped.extras,
                                              'missing_pending_since': 1_500.0})
        first.close()

        _wall_clock.now = 9_000.0
        again = _open_ctx(store)
        _engine(again)
        rows = {coid: (again.get_order(coid).extras or {})  # type: ignore[union-attr]
                for coid in ("seen", "unseen", "stamped")}
        assert MISSING_ACCEPTED_BY_RESUME_EXTRA not in rows["seen"]
        assert rows["unseen"]['missing_resumed_at'] == 9_000.0, \
            "a row no process judged gets a fresh window"
        assert rows["stamped"]['missing_resumed_at'] == 1_000.0, \
            "a stamp of the first process keeps its window"
        again.close()


@pytest.mark.parametrize(("observed", "intervention"), [(1.0900, False), (1.0800, True)],
                         ids=["the-accepted-level", "moved-while-no-process-ran"])
def __test_a_resume_applied_again_judges_a_fail_safe_level_the_first_process_accepted__(
        tmp_path: Path, _wall_clock: SimpleNamespace, observed: float, intervention: bool,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        _resume_previous_run(store)
        first = _open_ctx(store)
        _seed_parent(first, {'levels': [[None, None, None], [1.0970, None, None]],
                             'confirmed': [1.0970, None, None]})
        engine = _engine(first)
        _rehydrate(engine, [])
        _observe(engine, 1.0900)  # the operator's level, accepted
        engine.drive_native_failsafe(now_ms=float(BAR_TS))
        assert not engine.quarantined
        first.close()  # stopped before the resume completed

        _wall_clock.now += 3_600.0
        again = _open_ctx(store)
        engine = _engine(again)
        assert engine.resumed_from_quarantine
        _rehydrate(engine, [])
        _observe(engine, observed)
        engine.drive_native_failsafe(now_ms=float(BAR_TS + 1_000))
        assert engine.quarantined is intervention
        again.close()


def __test_a_fail_safe_stop_removed_after_one_confirmed_put_is_an_intervention__(
        tmp_path: Path) -> None:
    """A confirmation leaves only the confirmed level on record as the bot's own."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _seed_parent(ctx, None)
        engine = _engine(ctx)
        puts: list[NativeBracketSnapshot] = []
        _rehydrate(engine, puts)
        engine.drive_native_failsafe(now_ms=float(BAR_TS))
        assert [p.stop_level for p in puts] == [pytest.approx(1.0970)]
        _observe(engine, 1.0970)
        engine.drive_native_failsafe(now_ms=float(BAR_TS + 1_000))
        row = ctx.get_order(PARENT)
        assert row is not None
        saved = row.extras[NATIVE_FAILSAFE_LEVELS_EXTRA]
        assert saved['levels'] == [] and saved['confirmed'][0] == pytest.approx(1.0970)
        ctx.close()

        restarted = _open_ctx(store)
        engine = _engine(restarted)
        puts = []
        _rehydrate(engine, puts)
        _observe(engine, None)  # removed while the bot was down
        engine.drive_native_failsafe(now_ms=float(BAR_TS + 60_000))
        assert engine.quarantine_kind is QuarantineKind.EXTERNAL_INTERVENTION
        assert puts == [], "the bot never puts back what someone else removed"
        restarted.close()


def __test_a_resumed_entry_block_is_a_plain_restart__(tmp_path: Path) -> None:
    """Lifting an entry block accepts no venue change: a stop moved meanwhile is judged."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        ctx.upsert_order("leg-1", symbol=SYMBOL, side="sell", qty=1.0, state="confirmed",
                         extras={})
        guard = ProtectionLevelGuard(lambda reason, context: None, store_ctx=ctx)
        guard.expect("p1", _BOT_LEVELS, row_coid="leg-1")
        guard.settle("p1", _BOT_LEVELS)
        engine = _engine(ctx)
        engine.record_quarantine("protective_exit_rejected")
        assert engine.quarantine_kind is QuarantineKind.ENTRY_BLOCK
        quarantine_id = engine.quarantine_id
        assert quarantine_id is not None
        assert store.request_quarantine_resume(ctx.run_id, quarantine_id) is not None
        ctx.close()

        restarted = _open_ctx(store)
        engine = _engine(restarted)
        assert not engine.resumed_from_quarantine
        assert engine.operator_resume_ts_ms is None
        assert store.open_quarantines() == []
        row = restarted.get_quarantine(quarantine_id)
        assert row is not None and row.end_reason == 'operator_resume'
        guard, reports = _guard_reports(restarted)
        assert guard.observe("p1", _OPERATOR_LEVELS, tick=0.0001, immediate=True)
        assert len(reports) == 1
        restarted.close()


def __test_a_failed_resume_completion_never_fails_a_new_intervention_latch__(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture) -> None:
    import logging
    import sqlite3

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        _resume_previous_run(store)
        ctx = _open_ctx(store)
        engine = _engine(ctx)
        assert engine.resumed_from_quarantine
        resumed_id = store.open_quarantines()[0].quarantine_id
        real_end = ctx.end_quarantine

        def _locked(*_args, **_kwargs) -> None:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(ctx, "end_quarantine", _locked)
        caplog.clear()

        assert engine.record_quarantine("operator moved the stop again",
                                        kind=QuarantineKind.EXTERNAL_INTERVENTION)

        assert engine.quarantine_kind is QuarantineKind.EXTERNAL_INTERVENTION
        assert any(r.levelno == logging.ERROR
                   and "kind=external_intervention" in r.getMessage()
                   for r in caplog.records), "the quarantine record is logged"
        monkeypatch.setattr(ctx, "end_quarantine", real_end)
        new_id = engine.quarantine_id
        ctx.close()

        restarted = _open_ctx(store)
        engine = _engine(restarted)
        assert engine.quarantined and engine.quarantine_id == new_id
        row = restarted.get_quarantine(resumed_id)
        assert row is not None and row.end_reason == 'operator_resume'
        restarted.close()
