"""
A position someone else changed: the book takes the venue truth, the journal stays the plugin's.

Covers the sync engine's side of an external reduction or flatten of the bot's
own position, with a broker store behind the engine:

- the book adopts the venue; the journal is left to the plugin, which
  attributes the reduction to its rows (a second writer would subtract it
  twice), and a restart does not report the same change again;
- at startup, an unexplained reduction is an ERROR, an operator-resumed one a
  WARNING, both audited once; a spot run (gross journal, fee-net inventory)
  is not compared;
- on an account-net view the opposite side is the book's side gone, unless a
  resting reversal of ours may have filled; on a hedged view it is someone
  else's separate position;
- a late fill of an order that was live when a reduction was adopted is held
  until the venue shows whether it was that reduction (not booked twice) or a
  new one (booked), and is never reported as someone else's change: not when it
  takes the venue flat, not on a venue net another run trades, not beside a
  further foreign reduction (only that one is reported);
- an intervention latched in the middle of the booking never stops the
  booking, and nothing is written to the venue.
"""
import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.core.broker import sync_engine as sync_engine_module
from pynecore.core.broker.models import (
    DispatchEnvelope,
    EntryIntent,
    ExchangeCapabilities,
    ExchangeOrder,
    ExchangePosition,
    ExitIntent,
    INTENT_KEY_SEP,
    LegType,
    OrderEvent,
    OrderStatus,
    OrderType,
    PositionLeg,
    QuarantineKind,
)
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.sync_engine import (
    EXTERNAL_FLATTEN_CONFIRM_GRACE_S,
    OrderSyncEngine,
)
from pynecore.types.strategy import (
    ADOPTED_STARTUP_ENTRY_ID,
    JOURNAL_EXPOSURE_RETIRED_EXTRA_KEY,
    JOURNAL_WIRE_QTY_ANCHOR_EXTRA_KEY,
)

SYMBOL = "BTCUSDT"
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
class _Broker:
    client_order_id_max_len = 30
    on_unexpected_cancel = "stop"
    position: ExchangePosition | None = None
    position_port: object | None = None
    cancel_calls: list[DispatchEnvelope] = field(default_factory=list)
    on_get_position: object | None = None
    open_orders: list[ExchangeOrder] = field(default_factory=list)
    open_orders_error: Exception | None = None
    spot_inventory_port: object | None = None

    @staticmethod
    def get_capabilities() -> ExchangeCapabilities:
        return ExchangeCapabilities()

    async def get_position(self, symbol):
        if callable(self.on_get_position):
            self.on_get_position()
        return self.position

    async def get_open_orders(self, symbol=None) -> list[ExchangeOrder]:
        if self.open_orders_error is not None:
            raise self.open_orders_error
        return list(self.open_orders)

    async def execute_cancel(self, envelope):
        self.cancel_calls.append(envelope)
        return True

    async def cancel_broker_order_ref(self, ref):  # pragma: no cover — unused
        raise AssertionError("no venue write expected")


@dataclass
class _Port:
    legs: list[PositionLeg]

    async def fetch_raw_positions(self, symbol: str) -> list[PositionLeg]:
        return list(self.legs)


def _long(size: float) -> ExchangePosition:
    return ExchangePosition(
        symbol=SYMBOL, side="long", size=size, entry_price=50_000.0,
        unrealized_pnl=0.0, liquidation_price=None, leverage=1.0, margin_mode="cross",
    )


def _short(size: float) -> ExchangePosition:
    return ExchangePosition(
        symbol=SYMBOL, side="short", size=size, entry_price=50_000.0,
        unrealized_pnl=0.0, liquidation_price=None, leverage=1.0, margin_mode="cross",
    )


def _leg(leg_id: str, side: str, qty: float) -> PositionLeg:
    return PositionLeg(leg_id=leg_id, symbol=SYMBOL, side=side, qty=qty,
                       entry_price=50_000.0, open_time=0.0)


def _open_ctx(store: BrokerStore) -> RunContext:
    return store.open_run(
        RunIdentity(strategy_id="t163", symbol=SYMBOL, timeframe="60",
                    account_id="test-account", label=None),
        script_source="// t163",
    )


def _engine(broker: _Broker, ctx: RunContext) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos, symbol=SYMBOL, run_tag=ctx.run_tag, mintick=0.01,
        store_ctx=ctx,
    )
    return engine, pos


def _entry_row(ctx: RunContext, coid: str, pine_id: str, filled: float, *,
               side: str = "buy", extras: dict | None = None) -> None:
    ctx.upsert_order(coid, symbol=SYMBOL, side=side, qty=filled, state="confirmed",
                     pine_entry_id=pine_id, filled_qty=filled,
                     extras={"kind": "position", **(extras or {})})


def _owned(ctx: RunContext) -> float:
    owned = 0.0
    for row in ctx.iter_live_orders():
        retired = float((row.extras or {}).get(JOURNAL_EXPOSURE_RETIRED_EXTRA_KEY) or 0.0)
        remaining = max(0.0, row.filled_qty - min(row.filled_qty, retired))
        owned += remaining if row.side == "buy" else -remaining
    return round(owned, 12)


def _events(ctx: RunContext, kind: str) -> list[dict]:
    return list(ctx.iter_events_by_kind_since(kind, 0))


def _go_periodic(engine: OrderSyncEngine) -> None:
    engine._sync_count = 1  # noqa: SLF001 — past the startup reconcile


def _confirm_shrink(engine: OrderSyncEngine) -> None:
    """Observe the reduced venue, then age the observation past the grace."""
    engine.reconcile()
    observed = engine._shrink_observed  # noqa: SLF001
    assert observed is not None
    seconds = EXTERNAL_FLATTEN_CONFIRM_GRACE_S + 1.0
    engine._shrink_observed = (observed[0], observed[1] - seconds)  # noqa: SLF001
    engine._last_position_fill_monotonic -= seconds  # noqa: SLF001
    engine.reconcile()


def _confirm_flatten(engine: OrderSyncEngine) -> None:
    engine.reconcile()
    aged = time.monotonic() - EXTERNAL_FLATTEN_CONFIRM_GRACE_S - 1.0
    engine._flat_observed_since = aged  # noqa: SLF001
    engine._last_position_fill_monotonic = aged  # noqa: SLF001
    engine.reconcile()


def _errors(caplog: pytest.LogCaptureFixture, text: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records
            if r.levelno >= logging.ERROR and text in r.getMessage()]


# --- periodic reduction / flatten ------------------------------------------------


def __test_an_external_reduction_leaves_the_journal_and_a_restart_stays_quiet__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l1", "L1", 0.5)
        _entry_row(ctx, "row-l2", "L2", 0.5)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        assert pos.size == pytest.approx(1.0)
        _go_periodic(engine)

        broker.position = _long(0.6)
        _confirm_shrink(engine)

        assert pos.size == pytest.approx(0.6)
        assert _owned(ctx) == pytest.approx(1.0), "the journal is the plugin's to attribute"
        for coid in ("row-l1", "row-l2"):
            row = ctx.get_order(coid)
            assert row is not None and JOURNAL_EXPOSURE_RETIRED_EXTRA_KEY not in row.extras
        adopted = _events(ctx, 'external_reduce_adopted')
        assert len(adopted) == 1
        assert adopted[0]['journal_owned'] == pytest.approx(1.0)
        assert adopted[0]['venue_signed'] == pytest.approx(0.6)
        ctx.close()

        caplog.clear()
        restarted = _open_ctx(store)
        engine, pos = _engine(_Broker(position=_long(0.6)), restarted)
        engine.reconcile()
        assert pos.size == pytest.approx(0.6)
        assert not _errors(caplog, "startup:"), "the same change is not reported twice"
        assert not _events(restarted, 'startup_unattributed_reduce')
        restarted.close()


def _net_filled(ctx: RunContext, coid: str, qty: float) -> None:
    """A plugin attributing a reduction by lowering the row's owned ``filled_qty``."""
    row = ctx.get_order(coid)
    assert row is not None
    ctx.upsert_order(coid, filled_qty=row.filled_qty - qty)


def __test_a_plugin_attributing_the_reduction_later_is_not_subtracted_twice__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The core books the shrink first, the plugin nets its journal after."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)

        broker.position = _long(0.6)
        _confirm_shrink(engine)
        assert pos.size == pytest.approx(0.6)
        _net_filled(ctx, "row-l", 0.4)
        assert _owned(ctx) == pytest.approx(0.6), "owned 0.6, not 1.0 - 0.4 - 0.4"
        ctx.close()

        caplog.clear()
        restarted = _open_ctx(store)
        engine, pos = _engine(_Broker(position=_long(0.6)), restarted)
        engine.reconcile()
        assert pos.size == pytest.approx(0.6), "the whole venue position is the bot's own"
        assert not [r for r in caplog.records
                    if "startup adoption limited" in r.getMessage()]
        assert not _errors(caplog, "startup:")
        restarted.close()


def __test_an_external_flatten_is_audited_and_leaves_the_journal__(tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)

        broker.position = None
        _confirm_flatten(engine)

        assert pos.size == 0.0
        assert _owned(ctx) == pytest.approx(1.0)
        audit = _events(ctx, 'external_flatten_adopted')
        assert len(audit) == 1
        assert audit[0]['book_size'] == pytest.approx(1.0)
        assert audit[0]['cleared'] == ['L']
        assert (audit[0]['journal_owned'], audit[0]['venue_signed']) == (1.0, 0.0)
        assert not engine.quarantined
        ctx.close()


# --- account-net opposite side ----------------------------------------------------


def __test_the_opposite_side_on_an_account_net_view_is_the_books_side_gone__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    from pynecore.core.broker.models import UnattributedPositionChangeEvent

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        events: list = []
        engine._broker_event_sink = events.append  # noqa: SLF001
        engine.reconcile()
        _go_periodic(engine)

        broker.position = _short(0.4)
        engine.reconcile()
        assert pos.size == pytest.approx(1.0), "the first observation only starts the clock"
        _confirm_flatten(engine)

        assert pos.size == 0.0, "the opposite exposure is never adopted"
        assert _errors(caplog, "exchange shows the opposite side")
        assert events == [UnattributedPositionChangeEvent(
            symbol=SYMBOL, change='flatten', book_size=1.0, venue_size=0.0)]
        audit = _events(ctx, 'external_flatten_adopted')[0]
        assert audit['venue_side'] == 'short'
        assert audit['venue_signed'] == pytest.approx(-0.4)
        assert _owned(ctx) == pytest.approx(1.0)
        # The next passes leave the foreign short alone.
        engine.reconcile()
        assert pos.size == 0.0
        ctx.close()


def __test_an_own_market_reversal_awaiting_its_fill_is_not_a_lost_side__(
        tmp_path: Path,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        engine._active_intents["S"] = EntryIntent(  # noqa: SLF001
            pine_id="S", symbol=SYMBOL, side="sell", qty=2.0, order_type=OrderType.MARKET)

        broker.position = _short(1.0)
        _confirm_flatten(engine)
        engine._flat_observed_since = time.monotonic() - 1_000.0  # noqa: SLF001
        engine.reconcile()

        assert pos.size == pytest.approx(1.0)
        assert not _events(ctx, 'external_flatten_adopted')
        ctx.close()


def _resting_sell(filled: float) -> ExchangeOrder:
    return ExchangeOrder(
        id="xchg-s", symbol=SYMBOL, side="sell", order_type=OrderType.LIMIT, qty=2.0,
        filled_qty=filled, remaining_qty=2.0 - filled, price=49_000.0, stop_price=None,
        average_fill_price=None, status=OrderStatus.OPEN, timestamp=0.0, fee=0.0,
        fee_currency="",
    )


@pytest.mark.parametrize(("open_orders", "error", "explained"), [
    ([], None, True),
    ([_resting_sell(1.5)], None, True),
    (None, "read", True),
    ([_resting_sell(0.0)], None, False),
], ids=["filled-and-gone", "filled-beyond-the-book", "unreadable", "still-resting-unfilled"])
def __test_a_resting_reversal_that_may_have_filled_is_not_a_lost_side__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
        open_orders: list[ExchangeOrder] | None, error: str | None, explained: bool,
) -> None:
    """Its fill report may arrive after the grace; only an untouched one proves nothing."""
    from pynecore.core.broker.exceptions import ExchangeConnectionError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        engine._active_intents["S"] = EntryIntent(  # noqa: SLF001
            pine_id="S", symbol=SYMBOL, side="sell", qty=2.0, order_type=OrderType.LIMIT,
            limit=49_000.0)
        engine._order_mapping["S"] = ["xchg-s"]  # noqa: SLF001

        broker.position = _short(1.0)
        broker.open_orders = list(open_orders or [])
        broker.open_orders_error = (ExchangeConnectionError("socket reset")
                                    if error is not None else None)
        caplog.clear()
        _confirm_flatten(engine)
        engine._flat_observed_since = time.monotonic() - 1_000.0  # noqa: SLF001
        engine.reconcile()

        if explained:
            assert pos.size == pytest.approx(1.0), "the late fill books the reversal"
            assert not _errors(caplog, "opposite side")
            assert not _events(ctx, 'external_flatten_adopted')
        else:
            assert pos.size == 0.0
            assert _errors(caplog, "exchange shows the opposite side")
            assert len(_events(ctx, 'external_flatten_adopted')) == 1
        ctx.close()


def __test_the_opposite_side_of_a_hedged_view_is_someone_elses_position__(
        tmp_path: Path,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        broker = _Broker(position=_long(1.0), position_port=_Port([]))
        engine, pos = _engine(broker, ctx)
        engine._one_way_replay_done = True  # noqa: SLF001
        engine.reconcile()
        _go_periodic(engine)

        broker.position = _short(0.4)
        _confirm_flatten(engine)

        assert pos.size == pytest.approx(1.0)
        assert not _events(ctx, 'external_flatten_adopted')
        ctx.close()


# --- startup ---------------------------------------------------------------------


def __test_a_reduction_found_at_a_plain_startup_is_an_error_once__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        engine, pos = _engine(_Broker(position=_long(0.7)), ctx)
        engine.reconcile()

        assert pos.size == pytest.approx(0.7)
        assert len(_errors(caplog, "startup: the venue holds 0.7")) == 1
        audit = _events(ctx, 'startup_unattributed_reduce')
        assert len(audit) == 1
        assert (audit[0]['journal_owned'], audit[0]['venue_signed']) == (1.0, 0.7)
        assert _owned(ctx) == pytest.approx(1.0), "the journal is the plugin's to attribute"
        ctx.close()

        caplog.clear()
        again = _open_ctx(store)
        engine, pos = _engine(_Broker(position=_long(0.7)), again)
        engine.reconcile()
        assert pos.size == pytest.approx(0.7)
        assert not _errors(caplog, "startup:")
        assert _events(again, 'startup_unattributed_reduce') == []
        again.close()

        caplog.clear()
        further = _open_ctx(store)
        engine, pos = _engine(_Broker(position=_long(0.5)), further)
        engine.reconcile()
        assert pos.size == pytest.approx(0.5)
        assert len(_errors(caplog, "startup: the venue holds 0.5")) == 1, "a new change is"
        further.close()


@pytest.mark.parametrize("resumed", [False, True], ids=["plain", "resumed"])
def __test_a_later_reduction_of_the_same_size_is_reported_again__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, resumed: bool,
) -> None:
    """A fixed-size bot reduced to the same size twice: two changes, two reports."""
    kind = 'resume_venue_position_accepted' if resumed else 'startup_unattributed_reduce'
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        for cycle, coid in enumerate(("row-1", "row-2")):
            ctx = _open_ctx(store)
            _entry_row(ctx, coid, f"L{cycle}", 1.0)
            if resumed:
                engine, _pos = _engine(_Broker(position=_long(1.0)), ctx)
                engine.record_quarantine("operator closed part of the position",
                                         kind=QuarantineKind.EXTERNAL_INTERVENTION)
                assert engine.quarantine_id is not None
                store.request_quarantine_resume(ctx.run_id, engine.quarantine_id)
                ctx.close()
                ctx = _open_ctx(store)
            caplog.clear()
            engine, pos = _engine(_Broker(position=_long(0.7)), ctx)
            engine.reconcile()

            assert pos.size == pytest.approx(0.7)
            reports = [r for r in caplog.records
                       if r.levelno >= logging.WARNING and "venue holds 0.7" in r.getMessage()]
            assert len(reports) == 1, f"cycle {cycle} is a new change"
            assert len(list(ctx.iter_events_by_kind_for_run_id(kind))) == cycle + 1
            ctx.close_order(coid)
            ctx.close()


def __test_a_reduction_accepted_by_an_operator_resume_is_a_warning_once__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        engine, _pos = _engine(_Broker(position=_long(1.0)), ctx)
        engine.record_quarantine("operator closed part of the position",
                                 kind=QuarantineKind.EXTERNAL_INTERVENTION)
        quarantine_id = engine.quarantine_id
        assert quarantine_id is not None
        ctx.close()
        assert store.request_quarantine_resume(ctx.run_id, quarantine_id) is not None

        caplog.clear()
        resumed = _open_ctx(store)
        engine, pos = _engine(_Broker(position=_long(0.4)), resumed)
        assert engine.resumed_from_quarantine
        engine.reconcile()

        assert pos.size == pytest.approx(0.4)
        assert not _errors(caplog, "startup")
        assert any("the operator's change is accepted" in r.getMessage()
                   for r in caplog.records if r.levelno == logging.WARNING)
        audit = _events(resumed, 'resume_venue_position_accepted')
        assert (audit[0]['journal_owned'], audit[0]['venue_signed']) == (1.0, 0.4)
        engine.sync(1_700_000_000_000)  # the first sync completes the resume
        resumed.close()

        caplog.clear()
        plain = _open_ctx(store)
        engine, pos = _engine(_Broker(position=_long(0.4)), plain)
        assert not engine.resumed_from_quarantine
        engine.reconcile()
        assert pos.size == pytest.approx(0.4)
        assert not _errors(caplog, "startup"), "the accepted change is not an ERROR later"
        plain.close()


@pytest.mark.parametrize(
    ("first_view", "again_view", "verdict"),
    [(1.0, 0.4, "error"), (0.6, 0.2, "error"), (0.6, 0.6, "quiet"), (None, 0.4, "accepted")],
    ids=["reduced-after-it-saw-the-venue", "accepted-gap-grew", "accepted-gap-unchanged",
         "never-saw-the-venue"],
)
def __test_a_resume_applied_again_accepts_only_what_the_first_process_saw__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
        first_view: float | None, again_view: float, verdict: str,
) -> None:
    """A reduction made while no bot ran, after a process took the venue over, is an ERROR.

    The first process applied the resume and saw the venue (``first_view``),
    then stopped before its first sync, so the next start applies the resume
    again. What the first process accepted stands; anything beyond it
    happened after the resume and nobody's fill explains it. A first process
    that never saw the venue leaves the acceptance to the next one.
    """
    accepted_kind, error_kind = 'resume_venue_position_accepted', 'startup_unattributed_reduce'
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        engine, _pos = _engine(_Broker(position=_long(1.0)), ctx)
        engine.record_quarantine("operator moved the stop",
                                 kind=QuarantineKind.EXTERNAL_INTERVENTION)
        assert engine.quarantine_id is not None
        store.request_quarantine_resume(ctx.run_id, engine.quarantine_id)
        ctx.close()

        first = _open_ctx(store)
        engine, _pos = _engine(_Broker(position=None if first_view is None
                                       else _long(first_view)), first)
        assert engine.resumed_from_quarantine
        if first_view is not None:
            engine.reconcile()  # it saw the venue, then stopped before its first sync
        first.close()

        caplog.clear()
        again = _open_ctx(store)
        engine, pos = _engine(_Broker(position=_long(again_view)), again)
        assert engine.resumed_from_quarantine
        engine.reconcile()

        assert pos.size == pytest.approx(again_view)
        errors = _errors(caplog, f"startup: the venue holds {again_view}")
        accepted = [r for r in caplog.records
                    if "the operator's change is accepted" in r.getMessage()]
        if verdict == "error":
            assert len(errors) == 1 and not accepted
            assert len(_events(again, error_kind)) == 1
            assert _events(again, accepted_kind) == []
        elif verdict == "quiet":
            assert not errors and not accepted
            assert _events(again, error_kind) == [] and _events(again, accepted_kind) == []
        else:
            assert not errors and len(accepted) == 1
            assert len(_events(again, accepted_kind)) == 1
        engine.sync(1_700_000_000_000)  # the first sync completes the resume
        again.close()

        caplog.clear()
        plain = _open_ctx(store)
        engine, _pos = _engine(_Broker(position=_long(again_view)), plain)
        assert not engine.resumed_from_quarantine
        engine.reconcile()
        assert not _errors(caplog, "startup"), "a reported change is not reported again"
        plain.close()


@pytest.mark.parametrize("resumed", [False, True], ids=["plain", "resumed"])
def __test_a_reported_reduction_stays_reported_after_the_bots_own_later_fills__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, resumed: bool,
) -> None:
    """An own fill adds a journal row; the old difference is still the one reported."""
    kinds = ('startup_unattributed_reduce', 'resume_venue_position_accepted')

    def _reports(ctx: RunContext) -> int:
        return sum(len(list(ctx.iter_events_by_kind_for_run_id(kind))) for kind in kinds)

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l1", "L1", 1.0)
        if resumed:
            engine, _pos = _engine(_Broker(position=_long(1.0)), ctx)
            engine.record_quarantine("operator closed part of the position",
                                     kind=QuarantineKind.EXTERNAL_INTERVENTION)
            assert engine.quarantine_id is not None
            store.request_quarantine_resume(ctx.run_id, engine.quarantine_id)
            ctx.close()
            ctx = _open_ctx(store)
        engine, _pos = _engine(_Broker(position=_long(0.6)), ctx)
        engine.reconcile()
        engine.sync(1_700_000_000_000)
        assert _reports(ctx) == 1
        _entry_row(ctx, "row-l2", "L2", 0.5)  # the bot's own later entry fill
        ctx.close()

        caplog.clear()
        later = _open_ctx(store)
        engine, pos = _engine(_Broker(position=_long(1.1)), later)
        engine.reconcile()
        assert pos.size == pytest.approx(1.1)
        assert not _errors(caplog, "startup")
        assert _reports(later) == 1
        _entry_row(later, "row-x", "X", 1.1, side="sell")  # the bot's own exit fill
        later.close()

        caplog.clear()
        flat = _open_ctx(store)
        engine, pos = _engine(_Broker(position=None), flat)
        engine.reconcile()
        assert not _errors(caplog, "startup")
        assert _reports(flat) == 1
        flat.close()

        caplog.clear()
        reduced = _open_ctx(store)
        _entry_row(reduced, "row-l3", "L3", 1.0)
        engine, pos = _engine(_Broker(position=_long(0.8)), reduced)
        engine.reconcile()
        assert len(_errors(caplog, "startup: the venue holds 0.8")) == 1, \
            "a further reduction is a new change"
        reduced.close()


def __test_a_spot_run_is_not_compared_at_startup__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A spot journal counts the gross buy; the venue holds the fee-net inventory."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 0.01)
        broker = _Broker(position=_long(0.00999), spot_inventory_port=SimpleNamespace())
        engine, pos = _engine(broker, ctx)
        engine.reconcile()

        assert pos.size == pytest.approx(0.00999)
        assert not _errors(caplog, "startup")
        assert not [r for r in caplog.records if "operator's change" in r.getMessage()]
        assert _events(ctx, 'startup_unattributed_reduce') == []
        row = ctx.get_order("row-l")
        assert row is not None and JOURNAL_EXPOSURE_RETIRED_EXTRA_KEY not in row.extras
        ctx.close()


def __test_a_journal_in_a_wire_unit_is_not_compared_at_startup__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Inverse rows count contracts; the venue position counts base units."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 100.0,
                   extras={JOURNAL_WIRE_QTY_ANCHOR_EXTRA_KEY: "50000"})
        engine, _pos = _engine(_Broker(position=_long(0.002)), ctx)
        engine.reconcile()

        assert not _errors(caplog, "startup")
        assert _events(ctx, 'startup_unattributed_reduce') == []
        ctx.close()


def __test_a_pyramid_journaled_in_a_wire_unit_is_not_split_per_entry_at_startup__(
        tmp_path: Path,
) -> None:
    """Contracts are not base units: no per-entry split, one parent carries the venue size.

    Comparing 246 contracts with a 0.003 BTC position would trim the older
    entry to nothing and hand the whole position to the younger one.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l1", "L1", 164.0,
                   extras={JOURNAL_WIRE_QTY_ANCHOR_EXTRA_KEY: "82000"})
        _entry_row(ctx, "row-l2", "L2", 82.0,
                   extras={JOURNAL_WIRE_QTY_ANCHOR_EXTRA_KEY: "82000"})
        engine, pos = _engine(_Broker(position=_long(0.003)), ctx)
        engine.reconcile()

        assert pos.size == pytest.approx(0.003)
        assert [(t.entry_id, t.size) for t in pos.open_trades] == [
            (ADOPTED_STARTUP_ENTRY_ID, 0.003)]
        ctx.close()


def _long_at(size: float, price: float) -> ExchangePosition:
    return ExchangePosition(
        symbol=SYMBOL, side="long", size=size, entry_price=price,
        unrealized_pnl=0.0, liquidation_price=None, leverage=1.0, margin_mode="cross",
    )


@pytest.mark.parametrize(("venue_contracts", "book_contracts"), [(164.0, 164.0), (328.0, 164.0)],
                         ids=["whole-venue-is-the-runs", "shared-with-another-party"])
def __test_a_wire_unit_journal_adopts_only_the_runs_contracts_at_startup__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
        venue_contracts: float, book_contracts: float,
) -> None:
    """Contracts convert at the venue's entry price, the price the venue reports base at.

    A same-direction position of another party on the one-way inverse account
    is not this run's: a later close of it would read as a reduction of the
    bot's own exposure, and the bot's exits would close it.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 164.0,
                   extras={JOURNAL_WIRE_QTY_ANCHOR_EXTRA_KEY: "82100"})
        broker = _Broker(position=_long_at(venue_contracts / 82_000.0, 82_000.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()

        assert pos.size == pytest.approx(book_contracts / 82_000.0, rel=1e-12)
        limited = [r for r in caplog.records if "startup adoption limited" in r.getMessage()]
        assert len(limited) == (0 if venue_contracts == book_contracts else 1)

        # The other party closes its own exposure: the bot's share is intact.
        _go_periodic(engine)
        broker.position = _long_at(164.0 / 82_000.0, 82_000.0)
        engine.reconcile()
        engine._last_position_fill_monotonic -= EXTERNAL_FLATTEN_CONFIRM_GRACE_S + 1.0  # noqa: SLF001
        engine.reconcile()
        assert pos.size == pytest.approx(164.0 / 82_000.0, rel=1e-12)
        assert _events(ctx, 'external_reduce_adopted') == []
        ctx.close()


@pytest.mark.parametrize("side", ["long", "short"])
def __test_exposure_on_top_of_a_reported_reduction_is_not_adopted_at_restart__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, side: str,
) -> None:
    """No plugin attributed the reduction: the journal still counts what is gone.

    Someone closes 0.2 of the bot's 0.5, then opens a separate 0.2 of their
    own on the same side of the account-net view. A restart adopts the bot's
    0.3, not the stale journal's 0.5.
    """
    venue_at = _long if side == "long" else _short
    sign = 1.0 if side == "long" else -1.0
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 0.5, side="buy" if side == "long" else "sell")
        broker = _Broker(position=venue_at(0.5))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        broker.position = venue_at(0.3)
        _confirm_shrink(engine)
        assert pos.size == pytest.approx(sign * 0.3)
        broker.position = venue_at(0.5)  # the other party's own position
        engine.reconcile()
        assert pos.size == pytest.approx(sign * 0.3)
        ctx.close()

        caplog.clear()
        restarted = _open_ctx(store)
        broker = _Broker(position=venue_at(0.5))
        engine, pos = _engine(broker, restarted)
        engine.reconcile()
        assert pos.size == pytest.approx(sign * 0.3)
        assert any("startup adoption limited" in r.getMessage() for r in caplog.records)
        assert not _errors(caplog, "startup:")

        # The other party closes its own position: nothing of the bot's changed.
        _go_periodic(engine)
        broker.position = venue_at(0.3)
        engine.reconcile()
        engine._last_position_fill_monotonic -= EXTERNAL_FLATTEN_CONFIRM_GRACE_S + 1.0  # noqa: SLF001
        engine.reconcile()
        assert pos.size == pytest.approx(sign * 0.3)
        assert _events(restarted, 'external_reduce_adopted') == []
        restarted.close()


def __test_a_reported_reduction_attributed_by_the_plugin_no_longer_caps_adoption__(
        tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 0.5)
        engine, pos = _engine(_Broker(position=_long(0.3)), ctx)
        engine.reconcile()
        assert len(_events(ctx, 'startup_unattributed_reduce')) == 1
        _net_filled(ctx, "row-l", 0.2)
        _entry_row(ctx, "row-l2", "L2", 0.2)  # the bot's own later entry
        ctx.close()

        restarted = _open_ctx(store)
        engine, pos = _engine(_Broker(position=_long(0.5)), restarted)
        engine.reconcile()
        assert pos.size == pytest.approx(0.5)
        restarted.close()


def _exit_fill(qty: float, *, leg_type: LegType = LegType.TAKE_PROFIT,
               fill_id: str = "tp-fill-1") -> OrderEvent:
    return OrderEvent(
        order=ExchangeOrder(
            id="xchg-tp", symbol=SYMBOL, side="sell", order_type=OrderType.LIMIT,
            qty=qty, filled_qty=qty, remaining_qty=0.0, price=51_000.0, stop_price=None,
            average_fill_price=51_000.0, status=OrderStatus.FILLED, timestamp=0.0,
            fee=0.0, fee_currency="",
        ),
        event_type="filled", fill_price=51_000.0, fill_qty=qty, timestamp=0.0,
        pine_id="L-X", from_entry="L", leg_type=leg_type, fill_id=fill_id,
    )


def _settle_held_exit_fills(engine: OrderSyncEngine) -> None:
    """Age the held exit fills past the grace on an unchanged venue, then settle them."""
    seconds = EXTERNAL_FLATTEN_CONFIRM_GRACE_S + 1.0
    engine._last_position_fill_monotonic -= seconds  # noqa: SLF001
    engine.reconcile()
    observed = engine._unbooked_exit_observed  # noqa: SLF001
    assert observed is not None
    engine._unbooked_exit_observed = (observed[0], observed[1] - seconds)  # noqa: SLF001
    engine.reconcile()
    assert engine._unbooked_exit_ids == []  # noqa: SLF001


def __test_a_tp_fill_replayed_at_startup_is_not_an_unexplained_reduction__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The TP filled while the bot was down; the plugin nets the journal after the replay."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 0.1)
        engine, pos = _engine(_Broker(position=None), ctx)
        engine._event_queue.put(_exit_fill(0.1))  # noqa: SLF001
        engine.reconcile()

        assert pos.size == 0.0
        assert not _errors(caplog, "startup:")
        assert _events(ctx, 'startup_unattributed_reduce') == []
        assert any("compared with this run's journal once they settle" in r.getMessage()
                   for r in caplog.records)

        _net_filled(ctx, "row-l", 0.1)
        _go_periodic(engine)
        _settle_held_exit_fills(engine)
        assert pos.size == 0.0
        assert not _errors(caplog, "startup:")
        assert _events(ctx, 'startup_unattributed_reduce') == []
        ctx.close()


def __test_a_reduction_beside_a_replayed_tp_fill_is_reported_once_it_settles__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The plugin netted the TP fill already; someone else closed the rest while down."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 0.1)  # 0.2 entered, the TP's 0.1 netted
        engine, pos = _engine(_Broker(position=None), ctx)
        engine._event_queue.put(_exit_fill(0.1))  # noqa: SLF001
        engine.reconcile()
        assert not _errors(caplog, "startup:")

        _go_periodic(engine)
        _settle_held_exit_fills(engine)
        assert len(_errors(caplog, "startup: the venue holds 0.0")) == 1
        assert len(_events(ctx, 'startup_unattributed_reduce')) == 1
        engine.reconcile()
        assert len(_events(ctx, 'startup_unattributed_reduce')) == 1, "judged once"
        ctx.close()


def __test_a_reported_reduction_stays_covered_after_the_bots_own_close__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The plugin books the bot's own close as retired exposure on the reported row."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        broker = _Broker(position=_long(0.7))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        assert len(_errors(caplog, "startup: the venue holds 0.7")) == 1
        _go_periodic(engine)
        engine.reconcile()

        close = _exit_fill(0.4, leg_type=LegType.CLOSE, fill_id="close-1")
        pos.record_fill(close)
        ctx.upsert_order("row-l", extras={"kind": "position",
                                          JOURNAL_EXPOSURE_RETIRED_EXTRA_KEY: 0.4})
        broker.position = _long(0.3)
        engine.reconcile()
        assert pos.size == pytest.approx(0.3)
        assert len(_events(ctx, 'unattributed_view_carried_over')) == 1
        engine.reconcile()
        assert len(_events(ctx, 'unattributed_view_carried_over')) == 1
        ctx.close()

        caplog.clear()
        restarted = _open_ctx(store)
        engine, pos = _engine(_Broker(position=_long(0.3)), restarted)
        engine.reconcile()
        assert pos.size == pytest.approx(0.3)
        assert not _errors(caplog, "startup:")
        assert len(list(restarted.iter_events_by_kind_for_run_id(
            'startup_unattributed_reduce'))) == 1
        restarted.close()

        caplog.clear()
        further = _open_ctx(store)
        engine, pos = _engine(_Broker(position=_long(0.1)), further)
        engine.reconcile()
        assert len(_errors(caplog, "startup: the venue holds 0.1")) == 1, "a new change is"
        further.close()


def __test_a_partially_closed_pyramid_is_adopted_per_entry__(tmp_path: Path) -> None:
    """Retired exposure keeps the per-entry trades instead of one synthetic parent."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l1", "L1", 0.5)
        _entry_row(ctx, "row-l2", "L2", 0.5,
                   extras={JOURNAL_EXPOSURE_RETIRED_EXTRA_KEY: 0.2})
        engine, pos = _engine(_Broker(position=_long(0.8)), ctx)
        engine.reconcile()

        assert [(t.entry_id, round(t.size, 12)) for t in pos.open_trades] == [
            ("L1", 0.5), ("L2", 0.3)]
        ctx.close()


def __test_a_reduction_found_at_startup_trims_the_oldest_entries_first__(
        tmp_path: Path) -> None:
    """No plugin attributed the reduction yet: the per-entry trades survive, FIFO-trimmed.

    The venue closes a merged position oldest exposure first, as Pine's FIFO
    does for a close that names no entry; the journal is left as it is.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l1", "L1", 0.5)
        _entry_row(ctx, "row-l2", "L2", 0.5)
        engine, pos = _engine(_Broker(position=_long(0.6)), ctx)
        engine.reconcile()

        assert pos.size == pytest.approx(0.6)
        assert [(t.entry_id, round(t.size, 12)) for t in pos.open_trades] == [
            ("L1", 0.1), ("L2", 0.5)]
        assert len(_events(ctx, 'startup_unattributed_reduce')) == 1
        assert _owned(ctx) == pytest.approx(1.0), "the journal is the plugin's to attribute"
        ctx.close()


def __test_a_reduction_that_consumed_the_oldest_entry_leaves_only_the_younger_one__(
        tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l1", "L1", 0.5)
        _entry_row(ctx, "row-l2", "L2", 0.5)
        engine, pos = _engine(_Broker(position=_long(0.3)), ctx)
        engine.reconcile()

        assert [(t.entry_id, round(t.size, 12)) for t in pos.open_trades] == [("L2", 0.3)]
        ctx.close()


@pytest.mark.parametrize("noise", [5.8e-11, 2.67e-07], ids=["ulps", "sub-micro"])
def __test_float_noise_below_the_book_is_never_an_external_reduction__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, noise: float,
) -> None:
    """A venue size converted from another unit misses the book by noise, not by a trade."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 0.002)
        broker = _Broker(position=_long(0.002))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)

        broker.position = _long(0.002 - noise)
        engine.reconcile()
        assert engine._shrink_observed is None  # noqa: SLF001
        engine._last_position_fill_monotonic -= EXTERNAL_FLATTEN_CONFIRM_GRACE_S + 1.0  # noqa: SLF001
        engine.reconcile()

        assert pos.size == pytest.approx(0.002, abs=0.0)
        assert _events(ctx, 'external_reduce_adopted') == []
        assert not _errors(caplog, "exchange holds")

        broker.position = _long(0.0015)
        _confirm_shrink(engine)
        assert pos.size == pytest.approx(0.0015)
        assert len(_events(ctx, 'external_reduce_adopted')) == 1
        ctx.close()


def _other_run_ctx(store: BrokerStore) -> RunContext:
    return store.open_run(
        RunIdentity(strategy_id="t163-other", symbol=SYMBOL, timeframe="60",
                    account_id="test-account", label=None),
        script_source="// t163 other",
    )


def __test_a_merged_position_changes_the_bots_book_only_below_its_own_share__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Two runs on one account-net position: only a reduction below this run's share is its own.

    Run A owns 1.0 and run B 0.5 of a merged 1.5 long. B closing its own
    part and a third party adding to the position leave A's share intact;
    a reduction below A's share is booked with an ERROR and an audit, and
    never quarantines (the evidence for a quarantine is the plugin's).
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        other = _other_run_ctx(store)
        other.upsert_order("row-b", symbol=SYMBOL, side="buy", qty=0.5, state="confirmed",
                           pine_entry_id="B", filled_qty=0.5,
                           exchange_order_id="xchg-b", extras={"kind": "position"})
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-a", "A", 1.0)
        broker = _Broker(position=_long(1.5))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        assert pos.size == pytest.approx(1.0), "the other run's share is not adopted"
        _go_periodic(engine)

        for venue in (1.0, 1.8):  # B closes its own share; a third party adds
            broker.position = _long(venue)
            engine.reconcile()
            engine._last_position_fill_monotonic -= (  # noqa: SLF001
                EXTERNAL_FLATTEN_CONFIRM_GRACE_S + 1.0)
            engine.reconcile()
            assert pos.size == pytest.approx(1.0)
            assert engine._shrink_observed is None  # noqa: SLF001
        assert _events(ctx, 'external_reduce_adopted') == []

        broker.position = _long(0.7)
        _confirm_shrink(engine)
        assert pos.size == pytest.approx(0.7)
        assert len(_errors(caplog, "exchange holds 0.7")) == 1
        assert len(_events(ctx, 'external_reduce_adopted')) == 1
        assert not engine.quarantined
        assert broker.cancel_calls == []
        other.close()
        ctx.close()


# --- a late fill of the bot's own order after an adopted reduction -----------------


def _tp_row(ctx: RunContext) -> None:
    """The bot's resting TP leg of entry ``L``, live in the journal as ``xchg-tp``."""
    ctx.upsert_order("row-l-tp", symbol=SYMBOL, side="sell", qty=1.0, state="confirmed",
                     pine_entry_id="L", filled_qty=0.0, exchange_order_id="xchg-tp",
                     extras={"kind": "exit"})


def _settle_held_reduction_fills(engine: OrderSyncEngine) -> None:
    """Age the held fills past the grace on an unchanged venue, then settle them."""
    seconds = EXTERNAL_FLATTEN_CONFIRM_GRACE_S + 1.0
    engine._last_position_fill_monotonic -= seconds  # noqa: SLF001
    engine.reconcile()
    observed = engine._held_reduction_observed  # noqa: SLF001
    assert observed is not None
    engine._held_reduction_observed = (observed[0], observed[1] - seconds)  # noqa: SLF001
    engine.reconcile()
    assert engine._held_reduction_fills == []  # noqa: SLF001


def _settled(ctx: RunContext) -> list[str]:
    return [e['outcome'] for e in _events(ctx, 'adopted_reduction_fill_settled')]


def __test_a_late_fill_of_the_adopted_reduction_is_not_booked_twice__(tmp_path: Path) -> None:
    """The TP filled during a stream gap; the shrink was adopted before its fill arrived.

    Booking the late fill would take the book to 0.2 while the venue holds 0.6,
    and a venue above the book is never adopted back.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        _tp_row(ctx)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        broker.position = _long(0.6)
        _confirm_shrink(engine)
        assert pos.size == pytest.approx(0.6)

        engine._event_queue.put(_exit_fill(0.4))  # noqa: SLF001
        engine.reconcile()
        assert pos.size == pytest.approx(0.6), "held, not booked"
        _settle_held_reduction_fills(engine)

        assert pos.size == pytest.approx(0.6)
        assert _settled(ctx) == ['already_booked']
        assert engine._adopted_reduction_qty == 0.0  # noqa: SLF001
        engine.reconcile()
        assert pos.size == pytest.approx(0.6)
        ctx.close()


def __test_a_fill_the_venue_shows_after_the_adopted_reduction_is_booked__(
        tmp_path: Path) -> None:
    """Someone else closed 0.4; later the bot's TP fills 0.4 for real."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        _tp_row(ctx)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        broker.position = _long(0.6)
        _confirm_shrink(engine)

        broker.position = _long(0.2)
        engine._event_queue.put(_exit_fill(0.4))  # noqa: SLF001
        engine.reconcile()
        assert pos.size == pytest.approx(0.6)
        _settle_held_reduction_fills(engine)

        assert pos.size == pytest.approx(0.2)
        assert _settled(ctx) == ['booked']
        assert engine._adopted_reduction_qty == pytest.approx(0.4), (  # noqa: SLF001
            "the adopted reduction is still unexplained")
        ctx.close()


def __test_a_held_fill_is_split_into_the_adopted_reduction_and_a_new_one__(
        tmp_path: Path) -> None:
    """The TP filled 0.4 during the gap and 0.2 more after the adoption, reported as one."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        _tp_row(ctx)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        broker.position = _long(0.6)
        _confirm_shrink(engine)

        broker.position = _long(0.4)
        engine._event_queue.put(_exit_fill(0.6))  # noqa: SLF001
        engine.reconcile()
        _settle_held_reduction_fills(engine)

        assert pos.size == pytest.approx(0.4)
        assert _settled(ctx) == ['split']
        assert engine._adopted_reduction_qty == 0.0  # noqa: SLF001
        ctx.close()


def _slots(obj: object) -> dict:
    """Every slot of ``obj``, nested slotted objects as dicts, floats rounded."""
    out = {}
    for name in type(obj).__slots__:
        value = getattr(obj, name)
        if hasattr(type(value), '__slots__'):
            value = _slots(value)
        elif isinstance(value, float):
            value = round(value, 6)
        out[name] = value
    return out


def _book_view(pos: BrokerPosition) -> dict:
    """What the script and the trade statistics read off the book."""
    return {
        'closed': [(t.entry_id, round(t.size, 12), t.exit_price, t.exit_id, t.exit_time,
                    round(t.profit, 6), round(t.profit_percent, 9), round(t.commission, 6),
                    round(t.exit_equity, 6)) for t in pos.closed_trades],
        'open': [(t.entry_id, round(t.size, 12), round(t.commission, 6))
                 for t in pos.open_trades],
        'size': round(pos.size, 12), 'netprofit': round(pos.netprofit, 6),
        'grossprofit': round(pos.grossprofit, 6), 'grossloss': round(pos.grossloss, 6),
        'counts': (pos.wintrades, pos.losstrades, pos.eventrades, pos.closed_trades_count),
        'ratios': (round(pos.sum_profit_ratio, 12), round(pos.sum_win_profit_ratio, 12),
                   round(pos.sum_loss_profit_ratio, 12)),
        'stats': _slots(pos._closed_trade_stats),  # noqa: SLF001
    }


def _pyramid_exit_fill(qty: float, *, fee: float = 0.0) -> OrderEvent:
    """``L2``'s TP filling ``qty`` at 51000, under the journaled ``xchg-tp`` order."""
    event = _exit_fill(qty)
    return dataclasses.replace(event, pine_id="L2-X", from_entry="L2", fee=fee)


def _late_tp_engine(tmp_path: Path, name: str, *, pyramid: bool,
                    ) -> tuple[RunContext, _Broker, OrderSyncEngine, BrokerPosition]:
    """A periodic engine on a 1.0 long (``L``, or ``L1`` + ``L2``) with a resting TP."""
    store = BrokerStore(tmp_path / name / "broker.sqlite", plugin_name=PLUGIN)
    ctx = _open_ctx(store)
    if pyramid:
        _entry_row(ctx, "row-l1", "L1", 0.5)
        _entry_row(ctx, "row-l2", "L2", 0.5)
    else:
        _entry_row(ctx, "row-l", "L", 1.0)
    _tp_row(ctx)
    broker = _Broker(position=_long(1.0))
    engine, pos = _engine(broker, ctx)
    engine.reconcile()
    _go_periodic(engine)
    return ctx, broker, engine, pos


@pytest.mark.parametrize('pyramid', [False, True], ids=['single_entry', 'pyramid'])
def __test_a_late_fill_of_the_adopted_reduction_books_what_an_on_time_fill_books__(
        tmp_path: Path, pyramid: bool) -> None:
    """The adoption estimated the reduction at the entry price; the late fill knows its own.

    Settled as the adopted reduction, the fill re-prices the synthetic close:
    the book ends exactly where the same fill reported on time leaves it (the
    FIFO attribution, the fill's price, exit id and fee, the P&L, the counters
    and the closed-trade statistics), so equity-based sizing is not off by
    the exit's gain.
    """
    qty = 0.5 if pyramid else 0.4
    fill = _pyramid_exit_fill(qty, fee=3.0) if pyramid else dataclasses.replace(
        _exit_fill(qty), fee=3.0)

    on_time_ctx, on_time_broker, on_time, on_time_pos = _late_tp_engine(
        tmp_path, "on_time", pyramid=pyramid)
    on_time_broker.position = _long(1.0 - qty)
    on_time._event_queue.put(fill)  # noqa: SLF001
    on_time.reconcile()
    assert on_time_pos.size == pytest.approx(1.0 - qty)

    ctx, broker, engine, pos = _late_tp_engine(tmp_path, "late", pyramid=pyramid)
    broker.position = _long(1.0 - qty)
    _confirm_shrink(engine)
    engine._event_queue.put(fill)  # noqa: SLF001
    engine.reconcile()
    _settle_held_reduction_fills(engine)

    assert _settled(ctx) == ['already_booked']
    assert _book_view(pos) == _book_view(on_time_pos)
    expected_profit = qty * 1_000.0 - 3.0
    assert pos.netprofit == pytest.approx(expected_profit)
    first_closed = 'L1' if pyramid else 'L'
    assert [(t.entry_id, t.exit_id, t.exit_price) for t in pos.closed_trades] == [
        (first_closed, fill.pine_id, 51_000.0)], "Pine FIFO, as an on-time fill books it"
    ctx.close()
    on_time_ctx.close()


def __test_a_held_fill_split_into_the_adopted_reduction_charges_its_fee_once__(
        tmp_path: Path) -> None:
    """0.4 of a 0.6 fill was the adopted reduction, 0.2 is new: one fee, one price."""
    ctx, broker, engine, pos = _late_tp_engine(tmp_path, "split", pyramid=False)
    broker.position = _long(0.6)
    _confirm_shrink(engine)
    broker.position = _long(0.4)
    engine._event_queue.put(dataclasses.replace(_exit_fill(0.6), fee=6.0))  # noqa: SLF001
    engine.reconcile()
    _settle_held_reduction_fills(engine)

    assert _settled(ctx) == ['split']
    assert pos.size == pytest.approx(0.4)
    assert sorted((round(abs(t.size), 12), t.exit_price, round(t.commission, 6))
                  for t in pos.closed_trades) == [(0.2, 51_000.0, 2.0), (0.4, 51_000.0, 4.0)]
    assert pos.netprofit == pytest.approx(0.6 * 1_000.0 - 6.0)
    ctx.close()


def __test_a_late_fill_smaller_than_the_adopted_reduction_reprices_only_its_part__(
        tmp_path: Path) -> None:
    """A 0.3 fill explains 0.3 of a 0.4 reduction; 0.1 stays an estimate, still expected.

    The FIFO trade stays one closed record, its exit price the quantity-weighted
    blend (0.3 at 51000, 0.1 at the 50000 estimate): no record is inserted into
    the closed-trade history a script indexes.
    """
    ctx, broker, engine, pos = _late_tp_engine(tmp_path, "part", pyramid=False)
    broker.position = _long(0.6)
    _confirm_shrink(engine)
    engine._event_queue.put(dataclasses.replace(_exit_fill(0.3), fee=3.0))  # noqa: SLF001
    engine.reconcile()
    _settle_held_reduction_fills(engine)

    assert _settled(ctx) == ['already_booked']
    assert pos.size == pytest.approx(0.6)
    assert [(round(t.size, 12), round(t.exit_price, 6), t.exit_id, round(t.commission, 6))
            for t in pos.closed_trades] == [(0.4, 50_750.0, "L-X", 3.0)]
    assert pos.closed_trades_count == 1
    assert pos.netprofit == pytest.approx(0.3 * 1_000.0 - 3.0)
    assert pos.wintrades == 1 and pos.eventrades == 0
    assert pos._closed_trade_stats.winning_count == 1  # noqa: SLF001
    assert engine._adopted_reduction_qty == pytest.approx(0.1)  # noqa: SLF001
    ctx.close()


def __test_a_late_fill_of_the_adopted_reduction_reprices_it_after_the_book_went_flat__(
        tmp_path: Path) -> None:
    """The rest of the position closed before the late fill of the adopted reduction arrived.

    The book and the venue are flat: the late fill reduces nothing, but it is
    the adopted reduction, so its synthetic close takes the fill's price and
    fee, and the P&L is what the fills reported on time book.
    """
    ctx, broker, engine, pos = _late_tp_engine(tmp_path, "flat", pyramid=False)
    broker.position = _long(0.6)
    _confirm_shrink(engine)
    close = OrderEvent(
        order=ExchangeOrder(
            id="xchg-close", symbol=SYMBOL, side="sell", order_type=OrderType.MARKET,
            qty=0.6, filled_qty=0.6, remaining_qty=0.0, price=None, stop_price=None,
            average_fill_price=52_000.0, status=OrderStatus.FILLED, timestamp=0.0,
            fee=0.0, fee_currency="",
        ),
        event_type="filled", fill_price=52_000.0, fill_qty=0.6, timestamp=0.0,
        pine_id="Close", from_entry="L", leg_type=LegType.CLOSE, fill_id="close-fill-1",
    )
    broker.position = None
    engine._event_queue.put(close)  # noqa: SLF001
    engine.reconcile()
    assert pos.size == 0.0
    engine._event_queue.put(dataclasses.replace(_exit_fill(0.4), fee=3.0))  # noqa: SLF001
    engine.reconcile()
    _settle_held_reduction_fills(engine)

    assert _settled(ctx) == ['already_booked']
    assert pos.size == 0.0
    assert pos.netprofit == pytest.approx(0.4 * 1_000.0 - 3.0 + 0.6 * 2_000.0)
    assert sorted((round(abs(t.size), 12), t.exit_price, round(t.commission, 6))
                  for t in pos.closed_trades) == [(0.4, 51_000.0, 3.0), (0.6, 52_000.0, 0.0)]
    assert engine._adopted_reduction_qty == 0.0  # noqa: SLF001
    assert _events(ctx, 'unbooked_exit_fill') == []
    ctx.close()


def __test_a_late_fill_of_the_adopted_reduction_reprices_it_beside_another_runs_exposure__(
        tmp_path: Path) -> None:
    """As above, but another run's 0.5 keeps the shared venue net long once the bot is flat.

    The venue being non-flat says nothing about the bot's book there: the late
    fill is still the adopted reduction and re-prices its close, it is not left
    to the unbooked-exit settlement, which would drop it with the estimate.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        other = _other_run_ctx(store)
        other.upsert_order("row-b", symbol=SYMBOL, side="buy", qty=0.5, state="confirmed",
                           pine_entry_id="B", filled_qty=0.0,
                           exchange_order_id="xchg-b", extras={"kind": "position"})
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        _tp_row(ctx)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        broker.position = _long(0.6)
        _confirm_shrink(engine)
        assert pos.size == pytest.approx(0.6)
        close = OrderEvent(
            order=ExchangeOrder(
                id="xchg-close", symbol=SYMBOL, side="sell", order_type=OrderType.MARKET,
                qty=0.6, filled_qty=0.6, remaining_qty=0.0, price=None, stop_price=None,
                average_fill_price=52_000.0, status=OrderStatus.FILLED, timestamp=0.0,
                fee=0.0, fee_currency="",
            ),
            event_type="filled", fill_price=52_000.0, fill_qty=0.6, timestamp=0.0,
            pine_id="Close", from_entry="L", leg_type=LegType.CLOSE, fill_id="close-fill-1",
        )
        broker.position = None
        engine._event_queue.put(close)  # noqa: SLF001
        engine.reconcile()
        assert pos.size == 0.0
        other.upsert_order("row-b", filled_qty=0.5)
        broker.position = _long(0.5)
        engine._event_queue.put(dataclasses.replace(_exit_fill(0.4), fee=3.0))  # noqa: SLF001
        engine.reconcile()
        _settle_held_reduction_fills(engine)

        assert _settled(ctx) == ['already_booked']
        assert pos.size == 0.0
        assert pos.netprofit == pytest.approx(0.4 * 1_000.0 - 3.0 + 0.6 * 2_000.0)
        assert sorted((round(abs(t.size), 12), t.exit_price, round(t.commission, 6))
                      for t in pos.closed_trades) == [(0.4, 51_000.0, 3.0), (0.6, 52_000.0, 0.0)]
        assert engine._adopted_reduction_qty == 0.0  # noqa: SLF001
        assert _events(ctx, 'unbooked_exit_fill') == []
        other.close()
        ctx.close()


class _FakeClock:
    """``time`` stand-in for the sync engine whose monotonic clock the test advances."""

    def __init__(self) -> None:
        self.now = time.monotonic()

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name: str):
        return getattr(time, name)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(sync_engine_module, 'time', fake)
    return fake


def _run_for(engine: OrderSyncEngine, clock: _FakeClock, seconds: float) -> None:
    """Reconcile every 30 s like a live run, for ``seconds``."""
    for _ in range(int(seconds // 30.0)):
        clock.now += 30.0
        engine.reconcile()


def _adopt_reduction_to(engine: OrderSyncEngine, broker: _Broker, clock: _FakeClock,
                        venue: float) -> None:
    """Someone else reduces the bot's 1.0 to ``venue``; the shrink is confirmed and adopted."""
    broker.position = _long(venue)
    engine.reconcile()
    clock.now += EXTERNAL_FLATTEN_CONFIRM_GRACE_S + 1.0
    engine.reconcile()


@pytest.mark.parametrize('venue_after', [None, _short(0.3)], ids=['flat', 'opposite_side'])
def __test_an_own_exit_fill_closing_the_rest_is_booked_not_a_reported_flatten__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, clock: _FakeClock,
        venue_after: ExchangePosition | None,
) -> None:
    """The bot's TP, live at the adoption, closes what is left; the venue goes flat or across.

    The fill is held until the venue settles it, so the external-flatten branch
    must not clear the book first: that reported the bot's own TP as a manual
    close and lost its realized P&L.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        _tp_row(ctx)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        _adopt_reduction_to(engine, broker, clock, 0.6)
        assert pos.size == pytest.approx(0.6)
        caplog.clear()

        clock.now += 30.0
        broker.position = venue_after
        engine._event_queue.put(_exit_fill(0.6))  # noqa: SLF001
        engine.reconcile()
        _run_for(engine, clock, 8 * 60.0)

        assert pos.size == 0.0
        assert _settled(ctx) == ['booked']
        assert _events(ctx, 'external_flatten_adopted') == []
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert (0.6, 600.0) in [(round(t.size, 12), round(t.profit, 6))
                                for t in pos.closed_trades], "the TP's own P&L is booked"
        ctx.close()


def __test_a_held_fill_on_a_venue_net_another_run_trades_is_booked_not_reported__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, clock: _FakeClock) -> None:
    """Another run trades the same venue net; the bot's own TP fill is still the bot's.

    Dropping the fill there left the shrink reconcile to adopt the same quantity
    as an unexplained reduction: an ERROR and an audit for the bot's own TP.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        other = _other_run_ctx(store)
        other.upsert_order("row-b", symbol=SYMBOL, side="buy", qty=0.5, state="confirmed",
                           pine_entry_id="B", filled_qty=0.0,
                           exchange_order_id="xchg-b", extras={"kind": "position"})
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        _tp_row(ctx)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        _adopt_reduction_to(engine, broker, clock, 0.6)
        reported = len(_events(ctx, 'external_reduce_adopted'))
        caplog.clear()

        clock.now += 30.0
        broker.position = _long(0.2)
        engine._event_queue.put(_exit_fill(0.4))  # noqa: SLF001
        engine.reconcile()
        _run_for(engine, clock, 8 * 60.0)

        assert pos.size == pytest.approx(0.2)
        assert _settled(ctx) == ['booked']
        assert len(_events(ctx, 'external_reduce_adopted')) == reported
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        other.close()
        ctx.close()


def _resting_tp(filled: float) -> ExchangeOrder:
    """The bot's TP ``xchg-tp`` as the venue's open orders show it."""
    return ExchangeOrder(
        id="xchg-tp", symbol=SYMBOL, side="sell", order_type=OrderType.LIMIT, qty=1.0,
        filled_qty=filled, remaining_qty=1.0 - filled, price=51_000.0, stop_price=None,
        average_fill_price=None, status=OrderStatus.OPEN, timestamp=0.0, fee=0.0,
        fee_currency="",
    )


def __test_a_fill_of_an_order_resting_unfilled_at_the_adoption_is_booked_on_a_shared_net__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, clock: _FakeClock) -> None:
    """Someone closes 0.4 of the bot's 1.0 while its TP rests unfilled; the TP fills later.

    Another run's entry fills in between, so the venue net holds that run's
    exposure too and no longer tells a late report of the adopted reduction
    from a new fill. The venue's open orders do: the TP rested with nothing
    filled when the reduction was adopted, so its fill came after it and is
    booked at once. Held, the settlement measured the shared net against the
    book and dropped the bot's real exit as already booked: the book stayed
    0.4 too large and the bot's next exit took the other run's exposure.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        other = _other_run_ctx(store)
        other.upsert_order("row-b", symbol=SYMBOL, side="buy", qty=0.5, state="confirmed",
                           pine_entry_id="B", filled_qty=0.0,
                           exchange_order_id="xchg-b", extras={"kind": "position"})
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        _tp_row(ctx)
        broker = _Broker(position=_long(1.0), open_orders=[_resting_tp(0.0)])
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        _adopt_reduction_to(engine, broker, clock, 0.6)
        assert pos.size == pytest.approx(0.6)
        reported = len(_events(ctx, 'external_reduce_adopted'))
        assert reported == 1
        caplog.clear()

        other.upsert_order("row-b", filled_qty=0.5)
        broker.position = _long(1.1)
        clock.now += 30.0
        engine.reconcile()
        clock.now += 30.0
        broker.position = _long(0.7)
        broker.open_orders = []
        engine._event_queue.put(_exit_fill(0.4))  # noqa: SLF001
        engine.reconcile()
        assert pos.size == pytest.approx(0.2), "booked at once, never held"
        _run_for(engine, clock, 8 * 60.0)

        assert pos.size == pytest.approx(0.2)
        assert engine._held_reduction_fills == []  # noqa: SLF001
        assert _settled(ctx) == []
        assert len(_events(ctx, 'external_reduce_adopted')) == reported
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        other.close()
        ctx.close()


@pytest.mark.parametrize(("open_orders", "error"), [
    ([], None),
    ([_resting_tp(0.4)], None),
    (None, "read"),
], ids=["gone", "filled-beyond-the-journal", "unreadable"])
def __test_an_order_that_may_have_filled_before_the_adoption_is_held__(
        tmp_path: Path, clock: _FakeClock,
        open_orders: list[ExchangeOrder] | None, error: str | None) -> None:
    """Gone from the open orders, filled beyond what the journal knows, or unreadable."""
    from pynecore.core.broker.exceptions import ExchangeConnectionError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        _tp_row(ctx)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        broker.open_orders = list(open_orders or [])
        broker.open_orders_error = (ExchangeConnectionError("socket reset")
                                    if error is not None else None)
        _adopt_reduction_to(engine, broker, clock, 0.6)
        assert pos.size == pytest.approx(0.6)

        broker.open_orders_error = None
        engine._event_queue.put(_exit_fill(0.4))  # noqa: SLF001
        engine.reconcile()
        assert pos.size == pytest.approx(0.6), "held until the venue settles it"
        assert len(engine._held_reduction_fills) == 1  # noqa: SLF001
        _run_for(engine, clock, 8 * 60.0)
        assert pos.size == pytest.approx(0.6)
        assert _settled(ctx) == ['already_booked']
        ctx.close()


def __test_a_held_fill_beside_a_further_reduction_reports_only_the_rest__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, clock: _FakeClock) -> None:
    """The bot's TP fills 0.4 while someone else closes 0.1 more: only the 0.1 is reported."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        _tp_row(ctx)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        _adopt_reduction_to(engine, broker, clock, 0.6)
        reported = len(_events(ctx, 'external_reduce_adopted'))
        caplog.clear()

        clock.now += 30.0
        broker.position = _long(0.1)
        engine._event_queue.put(_exit_fill(0.4))  # noqa: SLF001
        engine.reconcile()
        _run_for(engine, clock, 8 * 60.0)

        assert pos.size == pytest.approx(0.1)
        assert _settled(ctx) == ['booked']
        adopted = _events(ctx, 'external_reduce_adopted')[reported:]
        assert [round(e['delta'], 12) for e in adopted] == [0.1]
        assert len(_errors(caplog, "exchange holds 0.1")) == 1
        ctx.close()


def __test_a_fill_of_an_order_placed_after_the_adopted_reduction_is_booked_at_once__(
        tmp_path: Path) -> None:
    """Only an order live at the adoption can be in it; the rest is booked as usual."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        broker.position = _long(0.6)
        _confirm_shrink(engine)

        engine._event_queue.put(_exit_fill(0.2))  # noqa: SLF001 — xchg-tp was never live
        engine.reconcile()
        assert pos.size == pytest.approx(0.4)
        assert engine._held_reduction_fills == []  # noqa: SLF001
        ctx.close()


def _attached_exit(engine: OrderSyncEngine, refs: list[str]) -> None:
    """The bot's exit ``L-X`` of ``L`` whose bracket is a venue position attribute."""
    key = f"L-X{INTENT_KEY_SEP}L"
    engine._active_intents[key] = ExitIntent(  # noqa: SLF001
        pine_id="L-X", from_entry="L", symbol=SYMBOL, side="sell", qty=1.0,
        tp_price=51_000.0, sl_price=40_000.0)
    engine._order_mapping[key] = list(refs)  # noqa: SLF001


def _fired_protection_fill(qty: float) -> OrderEvent:
    """A position-attached TP firing: its venue order id exists only from now on."""
    event = _exit_fill(qty)
    return dataclasses.replace(
        event, order=dataclasses.replace(event.order, id="venue-materialised-tp"))


_ATTACHED_REFS = pytest.mark.parametrize(
    "refs", [["bracket:L-X-tp", "bracket:L-X-sl"], ["4711:tp", "4711:sl"]],
    ids=["netting-trading-stop", "hedged-position-sltp"])


@_ATTACHED_REFS
def __test_a_late_fill_of_a_position_attached_bracket_is_not_booked_twice__(
        tmp_path: Path, clock: _FakeClock, refs: list[str]) -> None:
    """The position's own TP fired during a stream gap longer than the grace.

    Its fill carries the venue order id the TP got when it fired, never one the
    bot knew at the adoption: matched by the order id alone it was booked a
    second time, the book fell to 0.2 while the venue held 0.6, and a venue
    above the book is never adopted back.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        _attached_exit(engine, refs)
        _adopt_reduction_to(engine, broker, clock, 0.6)
        assert pos.size == pytest.approx(0.6)

        engine._event_queue.put(_fired_protection_fill(0.4))  # noqa: SLF001
        engine.reconcile()
        assert pos.size == pytest.approx(0.6), "held until the venue settles it"
        _run_for(engine, clock, 8 * 60.0)

        assert pos.size == pytest.approx(0.6)
        assert _settled(ctx) == ['already_booked']
        ctx.close()


@_ATTACHED_REFS
def __test_a_bracket_attached_after_the_adopted_reduction_is_booked_at_once__(
        tmp_path: Path, clock: _FakeClock, refs: list[str]) -> None:
    """Only a bracket live at the adoption can be in it."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        _adopt_reduction_to(engine, broker, clock, 0.6)
        _attached_exit(engine, refs)

        broker.position = _long(0.2)
        engine._event_queue.put(_fired_protection_fill(0.4))  # noqa: SLF001
        engine.reconcile()
        assert pos.size == pytest.approx(0.2)
        assert engine._held_reduction_fills == []  # noqa: SLF001
        ctx.close()


def __test_open_orders_read_again_after_an_unreadable_adoption_prove_a_resting_tp__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, clock: _FakeClock) -> None:
    """The open orders were unreadable at the adoption; a later pass reads them.

    The TP still rests unfilled after the reduction, so it was never part of it
    and its later fill is booked at once. Held on the unreadable view, the
    settlement measured the venue net another run trades and dropped the bot's
    real exit as already booked: the book stayed 0.4 too large and the bot's
    next exit took the other run's exposure.
    """
    from pynecore.core.broker.exceptions import ExchangeConnectionError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        other = _other_run_ctx(store)
        other.upsert_order("row-b", symbol=SYMBOL, side="buy", qty=0.5, state="confirmed",
                           pine_entry_id="B", filled_qty=0.0,
                           exchange_order_id="xchg-b", extras={"kind": "position"})
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        _tp_row(ctx)
        broker = _Broker(position=_long(1.0), open_orders=[_resting_tp(0.0)])
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        broker.open_orders_error = ExchangeConnectionError("socket reset")
        _adopt_reduction_to(engine, broker, clock, 0.6)
        assert pos.size == pytest.approx(0.6)
        broker.open_orders_error = None
        caplog.clear()

        other.upsert_order("row-b", filled_qty=0.5)
        broker.position = _long(1.1)
        clock.now += 30.0
        engine.reconcile()
        clock.now += 30.0
        broker.position = _long(0.7)
        broker.open_orders = []
        engine._event_queue.put(_exit_fill(0.4))  # noqa: SLF001
        engine.reconcile()
        assert pos.size == pytest.approx(0.2), "booked at once, never held"
        _run_for(engine, clock, 8 * 60.0)

        assert pos.size == pytest.approx(0.2)
        assert _settled(ctx) == []
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        other.close()
        ctx.close()


def __test_a_held_fill_settled_on_a_shared_net_never_verified_is_reported__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, clock: _FakeClock) -> None:
    """The open orders stayed unreadable from the adoption to the fill on a shared net.

    The split the venue net gives may be another run's exposure moving, so the
    settlement is an ERROR and its audit says so.
    """
    from pynecore.core.broker.exceptions import ExchangeConnectionError

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        other = _other_run_ctx(store)
        other.upsert_order("row-b", symbol=SYMBOL, side="buy", qty=0.5, state="confirmed",
                           pine_entry_id="B", filled_qty=0.0,
                           exchange_order_id="xchg-b", extras={"kind": "position"})
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        _tp_row(ctx)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        broker.open_orders_error = ExchangeConnectionError("socket reset")
        _adopt_reduction_to(engine, broker, clock, 0.6)
        clock.now += 30.0
        engine.reconcile()
        caplog.clear()

        engine._event_queue.put(_exit_fill(0.4))  # noqa: SLF001
        engine.reconcile()
        _run_for(engine, clock, 8 * 60.0)

        settled = _events(ctx, 'adopted_reduction_fill_settled')
        assert [e['unverified_shared_net'] for e in settled] == [True]
        assert len(_errors(caplog, "held exit fill(s) of ['L'] settled")) == 1
        other.close()
        ctx.close()


# --- an intervention latched while the booking runs --------------------------------


def _armed_exit(engine: OrderSyncEngine, entry_id: str) -> str:
    key = f"{entry_id}-X{INTENT_KEY_SEP}{entry_id}"
    engine._active_intents[key] = ExitIntent(  # noqa: SLF001
        pine_id=f"{entry_id}-X", from_entry=entry_id, symbol=SYMBOL, side="sell",
        qty=0.5, sl_price=40_000.0)
    engine._order_mapping[key] = [f"xchg-{entry_id}-sl"]  # noqa: SLF001
    return key


@pytest.mark.parametrize("venue", [_long(0.5), None], ids=["reduce", "flatten"])
def __test_an_intervention_latched_mid_booking_never_stops_the_booking__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, venue: ExchangePosition | None,
) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l1", "L1", 0.5)
        _entry_row(ctx, "row-l2", "L2", 0.5)
        broker = _Broker(position=_long(1.0))
        engine, pos = _engine(broker, ctx)
        engine.reconcile()
        _go_periodic(engine)
        key = _armed_exit(engine, "L1")
        broker.position = venue
        engine.reconcile()  # starts the confirmation clock
        seconds = EXTERNAL_FLATTEN_CONFIRM_GRACE_S + 1.0
        if engine._shrink_observed is not None:  # noqa: SLF001
            observed = engine._shrink_observed  # noqa: SLF001
            engine._shrink_observed = (observed[0], observed[1] - seconds)  # noqa: SLF001
        engine._flat_observed_since = time.monotonic() - seconds  # noqa: SLF001
        engine._last_position_fill_monotonic -= seconds  # noqa: SLF001

        def _plugin_latches() -> None:
            engine.record_quarantine("manual close reported by the plugin",
                                     kind=QuarantineKind.EXTERNAL_INTERVENTION)

        broker.on_get_position = _plugin_latches
        caplog.clear()
        engine.reconcile()

        assert engine.venue_writes_blocked
        assert broker.cancel_calls == [], "no venue write once the latch holds"
        assert not [r for r in caplog.records if r.exc_info], "no traceback"
        assert key not in engine.active_intents
        if venue is None:
            assert pos.size == 0.0
            assert len(_events(ctx, 'external_flatten_adopted')) == 1
        else:
            assert pos.size == pytest.approx(0.5)
            assert len(_events(ctx, 'external_reduce_adopted')) == 1
        ctx.close()


# --- a bracket the venue refuses to re-assert at a restart ------------------------


@dataclass
class _BracketPort:
    """A hedged account's port: one leg; bracket amends raise ``refusals`` first."""
    legs: list[PositionLeg]
    refusals: list[Exception] = field(default_factory=list)
    amended: list[tuple[str, float | None, float | None]] = field(default_factory=list)

    async def fetch_raw_positions(self, symbol: str) -> list[PositionLeg]:
        return list(self.legs)

    async def get_volume_quantizer(self, symbol: str):
        return lambda units: int(units * 100)

    async def amend_bracket(self, symbol, leg_id, *, side, tp_price, sl_price,
                            trail_offset, coid) -> None:
        if self.refusals:
            raise self.refusals.pop(0)
        self.amended.append((leg_id, tp_price, sl_price))


@pytest.mark.parametrize("resumed", [False, True], ids=["plain-restart", "after-resume"])
def __test_a_bracket_refused_at_restart_never_kills_the_run_and_is_re_armed__(
        tmp_path: Path, caplog: pytest.LogCaptureFixture, resumed: bool,
) -> None:
    import asyncio

    from pynecore.core.broker.exceptions import ExchangeOrderRejectedError
    from pynecore.core.broker.one_way_emulator import OneWayEmulator

    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_ctx(store)
        _entry_row(ctx, "row-l", "L", 1.0)
        port = _BracketPort([_leg("10", "buy", 1.0)])
        exit_intent = ExitIntent(pine_id="L-X", from_entry="L", symbol=SYMBOL, side="sell",
                                 qty=1.0, tp_price=60_000.0, sl_price=45_000.0)
        asyncio.run(OneWayEmulator(store_ctx=ctx).run_exit_bracket(
            DispatchEnvelope(intent=exit_intent, run_tag=ctx.run_tag, bar_ts_ms=1),
            port))
        if resumed:
            engine, _pos = _engine(_Broker(), ctx)
            engine.record_quarantine("operator moved the stop",
                                     kind=QuarantineKind.EXTERNAL_INTERVENTION)
            quarantine_id = engine.quarantine_id
            assert quarantine_id is not None
            store.request_quarantine_resume(ctx.run_id, quarantine_id)
        ctx.close()

        restarted = _open_ctx(store)
        port.amended.clear()
        port.refusals = [ExchangeOrderRejectedError("stop above the market")]
        broker = _Broker(position=_long(1.0), position_port=port)
        engine, pos = _engine(broker, restarted)
        assert engine.resumed_from_quarantine is resumed
        engine.reconcile()
        caplog.clear()
        engine.sync(1_700_000_000_000)  # must not raise

        key = f"L-X{INTENT_KEY_SEP}L"
        refused = [r for r in caplog.records if "refused to re-assert bracket" in r.getMessage()]
        assert len(refused) == 1
        assert refused[0].levelno == (logging.WARNING if resumed else logging.ERROR)
        assert len(_events(restarted, 'bracket_replay_refused')) == 1
        # The diff re-armed the exit through the regular dispatch.
        assert port.amended == [("10", 60_000.0, 45_000.0)]
        assert key in engine.active_intents
        assert engine.order_mapping[key] == ["bracket:10"]
        assert pos.size == pytest.approx(1.0)
        restarted.close()
