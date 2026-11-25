"""
Unit + persistence tests for :class:`OneWayEmulator` — the engine that fans a
Pine ``CloseIntent`` out across a hedging account's legs and owns the
persist-first close-leg ledger.

The first group drives ``run_close`` against a fake :class:`PositionPort` with
``store_ctx=None`` (pure orchestration: FIFO fan-out, partial reduce, flat
no-op, below-grid skip, shortfall). The second uses a real on-disk
:class:`BrokerStore` to verify the close-leg rows persist and that
``restart_replay`` resumes an interrupted fan-out — re-dispatching a still-open
leg's residual, finalising a vanished leg without re-closing it, and staying
idempotent across repeated replays.
"""
import asyncio
import sqlite3
from dataclasses import replace
from collections.abc import Iterable, Iterator

import pytest

from pynecore.core.broker.exceptions import (
    BracketAttachAfterFillRejectedError,
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    ExchangeRateLimitError,
    JournalWriteError,
    OrderDispositionUnknownError,
    OrderSkippedByPlugin,
)
from pynecore.core.broker.idempotency import (
    KIND_CANCEL, KIND_CLOSE, KIND_ENTRY, KIND_ENTRY_TRIGGERED,
)
from pynecore.core.broker.models import (
    CancelIntent,
    CapabilityLevel,
    CloseIntent,
    DispatchEnvelope,
    EntryIntent,
    ExitIntent,
    OrderType,
    PositionLeg,
)
import pynecore.core.broker.one_way_emulator as one_way_emulator_module
from pynecore.core.broker.one_way_emulator import (
    BOOK_INCOMPLETE_GRACE_S,
    UNSETTLED_WRITE_WINDOW_S,
    BracketFanResult,
    ClearFanResult,
    CloseFanResult,
    LegSettleResult,
    OneWayEmulator,
    close_leg_attempt_client_order_id,
    close_leg_client_order_id,
    close_leg_row_client_order_id,
)
from pynecore.core.broker.storage import BrokerStore, OrderRow, RunContext, RunIdentity
from pynecore.core.broker.store_helpers import (
    BRACKET_OWN_STATE_CLEARING,
    ENTRY_KIND_POSITION,
    ENTRY_KIND_WORKING,
    EXTRAS_KEY_BRACKET_OWN_LEG_ID,
    EXTRAS_KEY_BRACKET_OWN_SOFTWARE,
    EXTRAS_KEY_BRACKET_OWN_STATE,
    EXTRAS_KEY_BRACKET_OWN_TP,
    EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID,
    STATE_CONFIRMED,
    create_close_leg_row,
    create_entry_order_row,
    create_residual_open_row,
    iter_active_bracket_ownerships,
    iter_active_close_legs,
    iter_active_residual_opens,
)
from pynecore.core.plugin import ProviderError


class _FakePort:
    """Minimal :class:`PositionPort`: canned legs, ``int`` quantizer, records
    every ``close_leg`` call as ``(symbol, leg_id, volume, coid)``."""

    def __init__(
            self, legs: list[PositionLeg], *,
            min_qty: float = 0.0, max_qty: float | None = None,
            fail_amend_leg: str | None = None,
            fail_amend_unknown_leg: str | None = None,
            fail_place_leg: Exception | None = None,
            fail_close_leg: Exception | None = None,
            fail_close_leg_id: str | None = None,
    ) -> None:
        self._legs = list(legs)
        self._min_qty = min_qty
        self._max_qty = max_qty
        self._fail_amend_leg = fail_amend_leg
        # When set, place_leg raises this exception instead of recording the
        # placement — models a residual entry the exchange rejects (definitive)
        # or whose disposition is ambiguous.
        self._fail_place_leg = fail_place_leg
        # When set, close_leg raises this exception instead of recording the
        # close — models a reversal's FIFO close leg the exchange rejects
        # (definitive) or whose disposition is ambiguous.
        self._fail_close_leg = fail_close_leg
        # When set, only THIS leg's close_leg raises ``fail_close_leg``; the
        # other legs dispatch normally — models a fan the venue cuts short.
        self._fail_close_leg_id = fail_close_leg_id
        # Per-leg ambiguous-timeout: amend_bracket for THIS leg raises
        # OrderDispositionUnknownError, modelling an independent broker round-trip
        # that times out while the other legs amend cleanly.
        self.fail_amend_unknown_leg = fail_amend_unknown_leg
        # Toggle: when True, every amend_bracket raises the ambiguous-timeout
        # error, modelling a clear whose disposition the broker never confirmed.
        self.fail_amend_unknown = False
        # Broker volume grid: 1 keeps the identity quantizer most tests
        # rely on, 100 models a centi-unit venue (cTrader-like).
        self.volume_scale = 1
        self.closed: list[tuple[str, str, int, str]] = []
        self.placed: list[tuple[str, float]] = []
        self.amended: list[tuple[str, str, float | None, float | None, float | None, str]] = []
        # The protected quantity of every confirmed amend, in call order.
        self.amended_qty: list[tuple[str, float]] = []
        # What each leg holds at the venue: (tp, sl, trail, qty) of the last
        # write that took effect. An amend of unknown outcome takes effect
        # only when ``unknown_amends_land`` is set.
        self.venue: dict[str, tuple[float | None, float | None, float | None, float]] = {}
        self.unknown_amends_land = False
        # When True, bracket_matches answers "cannot tell" (None), modelling a
        # venue that does not report what tells the write from the leg's
        # earlier protection (a trailing distance it does not return).
        self.cannot_tell = False
        self.match_calls = 0
        self.fetch_calls = 0
        # What the venue records each close coid executed (broker grid), and
        # whether it can answer at all: ``None`` models records that cannot
        # settle a close yet (still working, or a read that came back short).
        self.executed_by_coid: dict[str, int] = {}
        self.executed_unknown = False
        self.executed_queries: list[tuple[str, tuple[str, ...]]] = []
        # When True, a close executes in full at once: the next read no longer
        # lists the closed volume (a venue whose position view is current as
        # soon as the close call returns). Off by default, so the legs stay
        # listed exactly as a test set them.
        self.closes_execute = False

    def set_legs(self, legs: list[PositionLeg]) -> None:
        self._legs = list(legs)

    async def fetch_raw_positions(self, symbol: str) -> list[PositionLeg]:
        self.fetch_calls += 1
        return [leg for leg in self._legs if leg.symbol == symbol]

    async def get_volume_quantizer(self, symbol: str):
        scale = self.volume_scale
        return lambda u: int(u * scale)

    async def close_leg(self, symbol: str, leg_id: str, volume: int, coid: str) -> None:
        if self._fail_close_leg is not None and (
                self._fail_close_leg_id is None or leg_id == self._fail_close_leg_id):
            raise self._fail_close_leg
        self.closed.append((symbol, leg_id, volume, coid))
        if self.closes_execute:
            left = []
            for leg in self._legs:
                if leg.leg_id == leg_id:
                    remaining = leg.qty - volume / self.volume_scale
                    if remaining <= 1e-12:
                        continue
                    leg = replace(leg, qty=remaining)
                left.append(leg)
            self._legs = left

    async def close_leg_executed(
            self, symbol: str, leg_id: str, coids, since_ms: int,
    ) -> int | None:
        self.executed_queries.append((leg_id, tuple(coids)))
        if self.executed_unknown:
            return None
        return sum(self.executed_by_coid.get(coid, 0) for coid in coids)

    async def reject_out_of_range(self, envelope: DispatchEnvelope, qty: float) -> None:
        if qty < self._min_qty or (self._max_qty is not None and qty > self._max_qty):
            raise OrderSkippedByPlugin(
                f"size {qty} out of range",
                intent_key=envelope.intent.intent_key,
                reason="out_of_range",
            )

    async def place_leg(self, envelope: DispatchEnvelope, qty: float):
        if self._fail_place_leg is not None:
            raise self._fail_place_leg
        self.placed.append((envelope.intent.symbol, qty))
        return ["order"]

    async def amend_bracket(
            self, symbol: str, leg_id: str, *,
            side: str, tp_price: float | None, sl_price: float | None,
            trail_offset: float | None, qty: float, coid: str,
    ) -> None:
        if leg_id == self._fail_amend_leg:
            raise ExchangeOrderRejectedError(f"amend rejected for leg {leg_id}")
        if self.fail_amend_unknown or leg_id == self.fail_amend_unknown_leg:
            if self.unknown_amends_land:
                self.venue[leg_id] = (tp_price, sl_price, trail_offset, qty)
            raise OrderDispositionUnknownError(
                f"amend disposition unknown for leg {leg_id}", client_order_id=coid,
            )
        self.amended.append((leg_id, side, tp_price, sl_price, trail_offset, coid))
        self.amended_qty.append((leg_id, qty))
        self.venue[leg_id] = (tp_price, sl_price, trail_offset, qty)

    async def bracket_matches(
            self, symbol: str, leg_id: str, *,
            side: str, tp_price: float | None, sl_price: float | None,
            trail_offset: float | None, qty: float,
    ) -> bool | None:
        self.match_calls += 1
        if self.cannot_tell:
            return None
        return self.venue.get(leg_id, (None, None, None, qty)) == (
            tp_price, sl_price, trail_offset, qty)


def _leg(leg_id, side, qty, *, symbol="EURUSD", open_time=0.0, price=1.10,
         entry_id=None, venue_qty=None) -> PositionLeg:
    return PositionLeg(
        leg_id=leg_id, symbol=symbol, side=side, qty=qty,
        entry_price=price, open_time=open_time, unrealized_pnl=0.0,
        entry_id=entry_id, venue_qty=venue_qty,
    )


def _close_env(side, qty, *, symbol="EURUSD", pine_id="Long") -> DispatchEnvelope:
    intent = CloseIntent(pine_id=pine_id, symbol=symbol, side=side, qty=qty)
    return DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000)


def _dispatched(port: _FakePort) -> list[tuple[str, int]]:
    return [(leg_id, volume) for _sym, leg_id, volume, _coid in port.closed]


def _run(coro):
    return asyncio.run(coro)


# === run_close — pure orchestration (store_ctx=None) ====================

def __test_run_close_fans_out_fifo__():
    """``run_close`` fans the close volume across long legs in FIFO order."""
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 3.0), port))
    assert res.legs == (("10", 2), ("11", 1))
    assert _dispatched(port) == [("10", 2), ("11", 1)]
    assert res.shortfall == 0.0 and res.skipped is False


def __test_run_close_partial_targets_oldest_leg__():
    """A partial close reduces only the oldest leg, leaving the rest open."""
    port = _FakePort([_leg("10", "buy", 3.0, open_time=0.0),
                      _leg("11", "buy", 2.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 2.0), port))
    assert _dispatched(port) == [("10", 2)]
    assert res.legs == (("10", 2),)


def __test_run_close_short_reduces_sell_legs__():
    """A ``side='buy'`` close reduces the sell legs, closing a short position."""
    # CloseIntent.side='buy' closes a short -> the legs to reduce are the sells.
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    _run(eng.run_close(_close_env("buy", 2.0), port))
    assert _dispatched(port) == [("20", 2)]


def __test_run_close_flat_is_noop__():
    """Closing a flat book dispatches no orders and returns an empty result."""
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 1.0), port))
    assert res == CloseFanResult(
        legs=(), dispatched_qty=0.0, shortfall=0.0, skipped=False,
    )
    assert port.closed == []


def __test_run_close_below_grid_skips_without_order__():
    """A close below the broker minimum is skipped with no order dispatched."""
    port = _FakePort([_leg("10", "buy", 0.4, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 0.4), port))
    assert res.skipped is True
    assert port.closed == []


def __test_run_close_reports_dispatched_qty_in_pine_units__():
    """``dispatched_qty`` stays in Pine units on a non-identity volume grid.

    ``legs`` carries the broker's integer grid volumes (centi-units here),
    but the caller compares the fan's size against Pine-unit fill
    quantities — so the result must expose the plan's Pine total, not the
    scaled leg volumes.
    """
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    port.volume_scale = 100
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 3.0), port))
    assert res.legs == (("10", 200), ("11", 100))
    assert res.dispatched_qty == 3.0
    # A broker-holds-less plan reports only what the legs could cover.
    port2 = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    port2.volume_scale = 100
    res2 = _run(eng.run_close(_close_env("sell", 3.0), port2))
    assert res2.dispatched_qty == 1.0 and res2.shortfall == 2.0


def __test_run_close_shortfall_surfaced_not_halted__():
    """When the broker holds less than owed, close what is open and report the shortfall."""
    # Broker holds less than Pine believes: close what is open, report the rest.
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 3.0), port))
    assert _dispatched(port) == [("10", 1)]
    assert res.shortfall == 2.0 and res.skipped is False


def __test_run_close_partial_leg_unsupported_closes_the_whole_leg__(monkeypatch):
    """A slice of a leg on a no-partial port closes that leg whole, with a warning."""
    warnings: list[str] = []
    monkeypatch.setattr(
        "pynecore.core.broker.one_way_emulator._blog_warning",
        lambda msg, *args, **_kw: warnings.append(msg % args),
    )
    port = _FakePort([_leg("10", "buy", 3.0, open_time=0.0),
                      _leg("11", "buy", 2.0, open_time=1.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 2.0), port))
    assert _dispatched(port) == [("10", 3)]
    assert res.dispatched_qty == 3.0 and res.skipped is False
    assert len([w for w in warnings if "whole leg is closed" in w]) == 1


def __test_run_close_past_a_whole_leg_widens_only_the_sliced_leg__():
    """Whole legs of the plan stay as planned; only the sliced leg is widened."""
    port = _FakePort([_leg("10", "buy", 3.0, open_time=0.0),
                      _leg("11", "buy", 2.0, open_time=1.0),
                      _leg("12", "buy", 1.0, open_time=2.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 4.0), port))
    assert _dispatched(port) == [("10", 3), ("11", 2)]
    assert res.dispatched_qty == 5.0


def __test_run_close_whole_legs_pass_without_partial_support__():
    """A whole-leg-only plan dispatches normally on a no-partial port."""
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 3.0), port))
    assert _dispatched(port) == [("10", 2), ("11", 1)]
    assert res.skipped is False


def __test_run_close_of_one_entry_closes_that_entrys_leg__():
    """``strategy.close("B")`` closes B's attributed leg, not the older leg of A.

    The fill settles B's binding, so a FIFO pick of A's leg would leave B's
    leg open while B's exits are retired, and A's exits never cover a leg
    attributed to B: the surviving leg would stay unprotected.
    """
    legs = [_leg("1", "buy", 1.0, open_time=1.0, entry_id="A"),
            _leg("2", "buy", 1.0, open_time=2.0, entry_id="B")]
    port = _FakePort(legs)
    eng = OneWayEmulator(store_ctx=None)
    _run(eng.run_close(_close_env("sell", 1.0, pine_id="B"), port))
    assert _dispatched(port) == [("2", 1)]
    # A larger close of B takes B's leg first, then the rest FIFO.
    port = _FakePort(legs)
    _run(eng.run_close(_close_env("sell", 2.0, pine_id="B"), port))
    assert _dispatched(port) == [("2", 1), ("1", 1)]
    # An engine close aimed at one entry follows its target entry.
    port = _FakePort(legs)
    intent = CloseIntent(pine_id="B-X", symbol="EURUSD", side="sell", qty=1.0,
                         synthetic_kind='marketable_exit', target_entry_id="B")
    _run(eng.run_close(DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000), port))
    assert _dispatched(port) == [("2", 1)]
    # ``strategy.close_all`` (empty pine_id) stays FIFO.
    port = _FakePort(legs)
    _run(eng.run_close(_close_env("sell", 1.0, pine_id=""), port))
    assert _dispatched(port) == [("1", 1)]


def __test_defensive_close_closes_the_leg_it_targets__():
    """A defensive close of B's unprotected leg closes that leg, not the older leg of A.

    The defensive-close marker settles B's binding, so a FIFO pick of A's
    leg would leave B's leg open and unprotected while the book counts it
    closed. A defensive close larger than its target (the leg shrank) closes
    only the target and reports the rest as shortfall.
    """
    legs = [_leg("1", "buy", 1.0, open_time=1.0, entry_id="A"),
            _leg("2", "buy", 1.0, open_time=2.0, entry_id="B"),
            _leg("3", "buy", 1.0, open_time=3.0, entry_id="B")]
    eng = OneWayEmulator(store_ctx=None)

    def defensive(qty: float, leg_id: str) -> DispatchEnvelope:
        intent = CloseIntent(
            pine_id=f"__pyne_defensive_close__{leg_id}", symbol="EURUSD", side="sell",
            qty=qty, immediately=True, synthetic_kind='defensive_close',
            target_position_coid="__pyne_orphan__EURUSD__B", target_exchange_id=leg_id,
            target_leg_id=leg_id,
        )
        return DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000)

    port = _FakePort(legs)
    _run(eng.run_close(defensive(1.0, "3"), port))
    assert _dispatched(port) == [("3", 1)]
    port = _FakePort(legs)
    fan = _run(eng.run_close(defensive(3.0, "3"), port))
    assert _dispatched(port) == [("3", 1)]
    assert fan.shortfall == 2.0


def __test_defensive_close_of_a_vanished_leg_closes_no_sibling__():
    """A defensive close whose target leg is gone dispatches nothing.

    The target closed between the bracket rejection and the close's fresh
    leg read; the remaining leg of A is a separate holding and must not be
    liquidated in its place. The whole quantity is reported as shortfall.
    """
    eng = OneWayEmulator(store_ctx=None)
    port = _FakePort([_leg("1", "buy", 1.0, open_time=1.0, entry_id="A")])
    intent = CloseIntent(
        pine_id="__pyne_defensive_close__2", symbol="EURUSD", side="sell",
        qty=1.0, immediately=True, synthetic_kind='defensive_close',
        target_position_coid="__pyne_orphan__EURUSD__B", target_exchange_id="2",
        target_leg_id="2",
    )
    fan = _run(eng.run_close(
        DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000), port,
    ))
    assert _dispatched(port) == []
    assert fan.legs == ()
    assert fan.shortfall == 1.0


# === Persistence + restart replay =======================================

def _identity() -> RunIdentity:
    return RunIdentity(
        strategy_id="oneway-test",
        symbol="EURUSD",
        timeframe="60",
        account_id="acc-1",
    )


def _make_store(tmp_path) -> tuple[BrokerStore, RunContext]:
    store = BrokerStore(tmp_path / "broker.db", plugin_name="ctrader")
    ctx = store.open_run(_identity(), script_source="// test")
    return store, ctx


class _Journal:
    """Where an emulator keeps its bracket ownership: an on-disk store, or no store.

    Without a store the emulator keeps the ownership rows in memory; the
    in-session scenarios run in both modes and must behave the same.
    """

    def __init__(self, mode: str, tmp_path) -> None:
        self.mode = mode
        self._store: BrokerStore | None = None
        self._ctx: RunContext | None = None
        if mode == "store":
            self._store, self._ctx = _make_store(tmp_path)
        self._emulator: OneWayEmulator | None = None

    def emulator(self, **kwargs) -> OneWayEmulator:
        self._emulator = OneWayEmulator(store_ctx=self._ctx, **kwargs)
        return self._emulator

    def rows(self) -> list[OrderRow]:
        """The live ownership rows: read from the store itself, or from the emulator."""
        if self._ctx is not None:
            return list(iter_active_bracket_ownerships(self._ctx))
        assert self._emulator is not None
        return list(self._emulator.iter_bracket_ownerships())

    def close(self) -> None:
        if self._store is not None:
            self._store.close()


@pytest.fixture(params=["store", "memory"])
def journal(request, tmp_path) -> Iterator[_Journal]:
    j = _Journal(request.param, tmp_path)
    yield j
    j.close()


def __test_run_close_persists_and_finalises_leg_rows__(tmp_path):
    """``run_close`` persists each close-leg row and finalises them all on success."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_close(_close_env("sell", 3.0), port))
    assert _dispatched(port) == [("10", 2), ("11", 1)]
    # Every dispatched leg row was finalised — nothing left for replay.
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def _pending_leg_row(ctx, *, leg_id, volume, qty, pre_volume=None):
    create_close_leg_row(
        ctx,
        coid=f"t000-x-y-c:{leg_id}",
        symbol="EURUSD",
        side="sell",
        qty=qty,
        intent_key="Long",
        pine_entry_id="Long",
        parent_close_coid="t000-x-y-c",
        leg_id=leg_id,
        leg_volume=volume,
        leg_pre_volume=volume if pre_volume is None else pre_volume,
    )


def __test_restart_replay_redispatches_still_open_leg__(tmp_path):
    """Restart replay re-dispatches a pending close leg that is still open on the broker."""
    # Crash sim: a pending close-leg row whose dispatch never acked; the leg is
    # still open on the broker -> replay re-dispatches and finalises it.
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=2, qty=2.0)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))
    assert _dispatched(port) == [("10", 2)]
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_restart_replay_finalises_vanished_leg_without_redispatch__(tmp_path):
    """Restart replay finalises a vanished close leg without re-dispatching it."""
    # The leg is GONE (the close landed before the crash) -> no re-dispatch.
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=2, qty=2.0)
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))
    assert port.closed == []
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_restart_replay_redispatches_only_residual_for_partly_closed_leg__(tmp_path):
    """Restart replay re-dispatches only the still-open residual of a partly-closed leg."""
    # The leg was partly closed before the crash: only 1.0 of the owed 3 remains
    # open -> re-dispatch min(persisted 3, live 1) = 1, never the full 3.
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=3, qty=3.0)
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))
    assert _dispatched(port) == [("10", 1)]
    store.close()


def __test_restart_replay_is_idempotent__(tmp_path):
    """Running restart replay twice dispatches the close leg exactly once."""
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=2, qty=2.0)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))
    _run(eng.restart_replay(port))  # second pass: no pending rows remain
    assert _dispatched(port) == [("10", 2)]  # dispatched exactly once
    store.close()


def __test_restart_replay_keeps_refused_close_leg_pending__(tmp_path):
    """A replayed close the exchange refuses stays pending and is retried by the next replay.

    A bot restarted into a closed market (symbol holiday) gets its owed close
    refused; the refusal must not settle the row (the leg is still open) and
    must not halt the startup — the next replay, once trading resumes, closes
    the leg exactly once.
    """
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=2, qty=2.0)
    port = _FakePort(
        [_leg("10", "buy", 2.0, open_time=0.0)],
        fail_close_leg=ExchangeOrderRejectedError("SYMBOL_HAS_HOLIDAY"),
    )
    eng = OneWayEmulator(store_ctx=ctx)
    assert _run(eng.restart_replay(port)) is False
    assert port.closed == []
    assert [row.client_order_id for row in iter_active_close_legs(ctx)] == ["t000-x-y-c:10"]
    # Trading resumed: the same row is replayed and settled.
    port._fail_close_leg = None
    assert _run(eng.restart_replay(port)) is True
    assert _dispatched(port) == [("10", 2)]
    assert list(iter_active_close_legs(ctx)) == []
    # A settled replay reports completion on every later pass.
    assert _run(eng.restart_replay(port)) is True
    assert _dispatched(port) == [("10", 2)]
    store.close()


def __test_restart_replay_never_repeats_an_executed_partial_close__(tmp_path):
    """A partial close that executed before the crash is not sent again.

    Four of a ten-unit leg were closed, the venue executed them, and the
    process died before the ack: the leg is six now. Capping the replay at the
    live size would send four again and leave two; the replay reads the
    executed part from the leg's recorded size and sends nothing.
    """
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=4, qty=4.0, pre_volume=10)
    port = _FakePort([_leg("10", "buy", 6.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    assert _run(eng.restart_replay(port)) is True
    assert port.closed == []
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_restart_replay_trusts_the_venue_record_over_a_grown_leg__(tmp_path):
    """A leg that grew after the close still counts the close the venue executed.

    The four-unit close executed, and an add filled on the same venue position
    (hedge-mode side positions grow): the leg went 10 -> 12 -> 8, so its size
    shows only two units gone. The venue's record of the close's own id says
    four executed, and nothing is re-sent.
    """
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=4, qty=4.0, pre_volume=10)
    port = _FakePort([_leg("10", "buy", 8.0, open_time=0.0)])
    port.executed_by_coid["t000-x-y-c:10"] = 4
    eng = OneWayEmulator(store_ctx=ctx)
    assert _run(eng.restart_replay(port)) is True
    assert port.closed == []
    assert port.executed_queries == [("10", ("t000-x-y-c:10",))]
    store.close()


def __test_restart_replay_sends_only_the_unexecuted_part_under_a_new_id__(tmp_path):
    """A close the venue executed in part is completed under a fresh attempt id.

    The id is recorded on the row before the send, so a second crash leaves a
    replay that asks the venue about both sends before it sends anything.
    """
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=4, qty=4.0, pre_volume=12)
    # One of the four executed, and an add of one filled since: 12 -> 13 -> 12.
    port = _FakePort([_leg("10", "buy", 12.0, open_time=0.0)])
    port.executed_by_coid["t000-x-y-c:10"] = 1
    port._fail_close_leg = OrderDispositionUnknownError("timeout", client_order_id="x")
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.restart_replay(port))
    [row] = list(iter_active_close_legs(ctx))
    assert row.extras["close_leg_attempts"] == ["t000-x-y-c:10", "t000-x-y-c:10#r1"]
    # The re-send of three executed after all; the next replay sends nothing.
    port._fail_close_leg = None
    port.executed_by_coid["t000-x-y-c:10#r1"] = 3
    port.set_legs([_leg("10", "buy", 9.0, open_time=0.0)])
    assert _run(eng.restart_replay(port)) is True
    assert port.closed == []
    assert port.executed_queries[-1] == ("10", ("t000-x-y-c:10", "t000-x-y-c:10#r1"))
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_restart_replay_completes_a_partly_executed_close__(tmp_path):
    """The part of a close the venue provably did not execute is sent once."""
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=4, qty=4.0, pre_volume=10)
    port = _FakePort([_leg("10", "buy", 9.0, open_time=0.0)])
    port.executed_by_coid["t000-x-y-c:10"] = 1
    eng = OneWayEmulator(store_ctx=ctx)
    assert _run(eng.restart_replay(port)) is True
    assert port.closed == [("EURUSD", "10", 3, "t000-x-y-c:10#r1")]
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_restart_replay_waits_when_the_close_outcome_is_unknown__(tmp_path):
    """No evidence of what the close executed: nothing is sent, the row waits.

    The venue cannot settle the close yet and the leg's size does not account
    for it: a re-send could close the leg twice, so the replay reports itself
    incomplete (the bot is not halted) and asks again on the next sync.
    """
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=4, qty=4.0, pre_volume=10)
    port = _FakePort([_leg("10", "buy", 10.0, open_time=0.0)])
    port.executed_unknown = True
    eng = OneWayEmulator(store_ctx=ctx)
    assert _run(eng.restart_replay(port)) is False
    assert _run(eng.restart_replay(port)) is False
    assert port.closed == []
    assert [row.client_order_id for row in iter_active_close_legs(ctx)] == ["t000-x-y-c:10"]
    # The venue now shows the close never executed: the owed close goes out once.
    port.executed_unknown = False
    assert _run(eng.restart_replay(port)) is True
    assert port.closed == [("EURUSD", "10", 4, "t000-x-y-c:10#r1")]
    store.close()


def __test_restart_replay_settles_an_unknown_outcome_the_leg_size_proves__(tmp_path):
    """The venue cannot answer, but the leg already lost the whole close: settled, nothing sent."""
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=4, qty=4.0, pre_volume=10)
    port = _FakePort([_leg("10", "buy", 6.0, open_time=0.0)])
    port.executed_unknown = True
    eng = OneWayEmulator(store_ctx=ctx)
    assert _run(eng.restart_replay(port)) is True
    assert port.closed == []
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_overlapping_replays_never_send_the_same_close_twice__(tmp_path):
    """A replay still running when the next starts owns its row; the second sends nothing.

    The engine's bridge can stop waiting for a slow replay while it keeps
    running, and the next sync starts another. Both would read the same
    attempts and send the same remainder under the same new id.
    """
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=4, qty=4.0, pre_volume=10)

    class _SlowPort(_FakePort):
        def __init__(self, legs) -> None:
            super().__init__(legs)
            self.gate = asyncio.Event()

        async def close_leg_executed(self, symbol, leg_id, coids, since_ms):
            await self.gate.wait()
            return 0

    async def scenario(port: _SlowPort) -> tuple[bool, bool]:
        eng = OneWayEmulator(store_ctx=ctx)
        first = asyncio.create_task(eng.restart_replay(port))
        await asyncio.sleep(0)
        second = await eng.restart_replay(port)
        port.gate.set()
        return await first, second

    port = _SlowPort([_leg("10", "buy", 10.0, open_time=0.0)])
    assert _run(scenario(port)) == (True, False)
    assert port.closed == [("EURUSD", "10", 4, "t000-x-y-c:10#r1")]
    store.close()


def __test_a_refired_leg_guard_never_resends_a_close_of_unknown_outcome__(tmp_path):
    """A software guard firing again after an unknown close sends only what is provably owed.

    The first close ended without a verdict and the position read does not
    show it yet. Re-sending the guard's quantity under the same id could close
    the leg twice on a venue that does not dedup it: the repeat waits for the
    venue's record, and sends nothing once that shows the close executed.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 10.0, open_time=0.0),
                      _leg("20", "sell", 4.0, open_time=1.0)],
                     fail_close_leg=OrderDispositionUnknownError("timeout", client_order_id="x"))
    intent = CloseIntent(pine_id="__pyne_leg_guard__10", symbol="EURUSD", side="sell",
                         qty=6.0, immediately=True, synthetic_kind='leg_guard',
                         target_leg_id="10")
    envelope = DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000)
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_close(envelope, port))
    [row] = list(iter_active_close_legs(ctx))
    port._fail_close_leg = None
    port.executed_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_close(envelope, port))
    assert port.closed == []
    port.executed_unknown = False
    port.executed_by_coid[row.client_order_id] = 6
    res = _run(eng.run_close(envelope, port))
    assert port.closed == [] and res.legs == (("10", 0),) and res.skipped is False
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_the_residual_drain_settles_an_unknown_reversal_close_first__(tmp_path):
    """The in-session residual drain never re-sends a reversal close blindly.

    The reversal's close ended without a verdict. The drain neither re-closes
    the leg under the same id nor opens the residual while that close cannot
    be settled; once the venue shows it never executed, the owed close goes out
    once under a new id. That close is only dispatched, so the residual opens
    on a later pass, once the book shows the opposing leg gone.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("20", "sell", 1.0, open_time=0.0)],
                     fail_close_leg=OrderDispositionUnknownError("timeout", client_order_id="x"))
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    [row] = list(iter_active_close_legs(ctx))
    port._fail_close_leg = None
    port.executed_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.drain_residual_opens(port))
    assert port.closed == [] and port.placed == []
    assert len(list(iter_active_residual_opens(ctx))) == 1
    port.executed_unknown = False
    _run(eng.drain_residual_opens(port))
    assert port.closed == [("EURUSD", "20", 1, f"{row.client_order_id}#r1")]
    assert port.placed == []
    assert len(list(iter_active_residual_opens(ctx))) == 1
    assert list(iter_active_close_legs(ctx)) == []
    # The re-sent close filled: the next pass opens the residual, once.
    port.set_legs([])
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 2.0)]
    assert port.closed == [("EURUSD", "20", 1, f"{row.client_order_id}#r1")]
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def _guard_env(qty: float) -> DispatchEnvelope:
    intent = CloseIntent(pine_id="__pyne_leg_guard__10", symbol="EURUSD", side="sell",
                         qty=qty, immediately=True, synthetic_kind='leg_guard',
                         target_leg_id="10")
    return DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000)


def __test_a_refired_guard_without_a_store_never_resends_a_close_of_unknown_outcome__():
    """A run without a store keeps the close's sends in memory: the repeat waits for evidence.

    Without a row the guard re-firing on the next price would send the same
    six units again under the same id while the first close's outcome is
    still unknown.
    """
    port = _FakePort([_leg("10", "buy", 10.0, open_time=0.0),
                      _leg("20", "sell", 4.0, open_time=1.0)],
                     fail_close_leg=OrderDispositionUnknownError("timeout", client_order_id="x"))
    eng = OneWayEmulator(store_ctx=None)
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_close(_guard_env(6.0), port))
    port._fail_close_leg = None
    port.executed_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_close(_guard_env(6.0), port))
    assert port.closed == []
    [(_leg_id, (coid,))] = port.executed_queries
    port.executed_unknown = False
    port.executed_by_coid[coid] = 6
    res = _run(eng.run_close(_guard_env(6.0), port))
    assert port.closed == [] and res.legs == (("10", 0),) and res.dispatched_qty == 0.0


def __test_a_settled_close_retry_never_sends_more_than_is_safe_now__(tmp_path):
    """The owed part of an earlier close is capped by what the fresh plan may close.

    The guard's six-unit close provably never executed, but an opposing leg
    grew meanwhile and only one unit of the long survives the netting: the
    retry sends one, never six (which would flip the position five short),
    and the fan reports the one it sent.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 10.0, open_time=0.0),
                      _leg("20", "sell", 4.0, open_time=1.0)],
                     fail_close_leg=OrderDispositionUnknownError("timeout", client_order_id="x"))
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_close(_guard_env(6.0), port))
    [row] = list(iter_active_close_legs(ctx))
    port._fail_close_leg = None
    port.set_legs([_leg("10", "buy", 10.0, open_time=0.0),
                   _leg("20", "sell", 9.0, open_time=1.0)])
    res = _run(eng.run_close(_guard_env(6.0), port))
    assert port.closed == [("EURUSD", "10", 1, f"{row.client_order_id}#r1")]
    assert res.legs == (("10", 1),) and res.dispatched_qty == 1.0
    store.close()


def __test_a_reopened_close_row_is_judged_from_its_first_send__(tmp_path):
    """A repeat of a settled close asks the venue from the first send, not from the reopen.

    Reopening the row restarts its creation time; a venue asked from then
    would miss the earlier send's executions and the repeat would close the
    leg again.
    """
    class _SincePort(_FakePort):
        def __init__(self, legs) -> None:
            super().__init__(legs)
            self.since: list[int] = []

        async def close_leg_executed(self, symbol, leg_id, coids, since_ms):
            self.since.append(since_ms)
            return await super().close_leg_executed(symbol, leg_id, coids, since_ms)

    store, ctx = _make_store(tmp_path)
    port = _SincePort([_leg("10", "buy", 10.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_close(_guard_env(6.0), port))
    [(_sym, _leg_id, _volume, coid)] = port.closed
    row = ctx.get_order(coid)
    # The row stays live while its send is reserved (the read does not show it).
    assert row is not None and row.extras["close_leg_state"] == "dispatched"
    first_send = row.extras["close_leg_since_ms"]
    # The guard fires again before the fill is visible in the book.
    port.executed_by_coid[coid] = 6
    res = _run(eng.run_close(_guard_env(6.0), port))
    assert port.since == [first_send]
    assert res.legs == (("10", 0),) and len(port.closed) == 1
    store.close()


def __test_restart_replay_never_flips_the_net_position__(tmp_path):
    """A replayed close is capped by the net exposure of its side in the fresh read.

    The four-unit close provably never executed, but an opposing leg of eight
    now offsets most of the ten-unit long: only two may close without turning
    the position short.
    """
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=4, qty=4.0, pre_volume=10)
    port = _FakePort([_leg("10", "buy", 10.0, open_time=0.0),
                      _leg("20", "sell", 8.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    assert _run(eng.restart_replay(port)) is True
    assert port.closed == [("EURUSD", "10", 2, "t000-x-y-c:10#r1")]
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_restart_replay_never_flips_the_net_position_on_a_whole_leg_port__(tmp_path):
    """A port that only closes whole legs gets no re-send whose whole leg would flip the net.

    Two units of the ten-unit long may close against the eight-unit short,
    but the port would close all ten and leave the account eight short.
    """
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=4, qty=4.0, pre_volume=10)
    port = _FakePort([_leg("10", "buy", 10.0, open_time=0.0),
                      _leg("20", "sell", 8.0, open_time=1.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=ctx)
    assert _run(eng.restart_replay(port)) is True
    assert port.closed == []
    store.close()


def __test_a_whole_leg_port_re_send_closes_the_whole_leg_when_that_is_safe__(tmp_path):
    """The owed part of a close is sent as the whole leg where that keeps the net side."""
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=4, qty=4.0, pre_volume=10)
    port = _FakePort([_leg("10", "buy", 10.0, open_time=0.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=ctx)
    assert _run(eng.restart_replay(port)) is True
    assert port.closed == [("EURUSD", "10", 10, "t000-x-y-c:10#r1")]
    store.close()


def __test_a_whole_leg_port_re_send_never_closes_exposure_the_run_does_not_own__(tmp_path):
    """No re-send on a whole-position port whose venue position holds foreign exposure."""
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=4, qty=4.0, pre_volume=10)
    port = _FakePort([_leg("10", "buy", 10.0, open_time=0.0, venue_qty=15.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=ctx)
    assert _run(eng.restart_replay(port)) is True
    assert port.closed == []
    store.close()


def __test_overlapping_restart_replays_never_spend_one_close_budget_twice__(tmp_path):
    """A replay still running when the next starts cannot spend the symbol's budget again.

    Two one-unit closes are owed against a book that is net long one unit:
    only one may go out. The second replay overlaps the first while that one
    waits for the venue's evidence of its first row.
    """
    class _SlowEvidencePort(_FakePort):
        def __init__(self, legs):
            super().__init__(legs)
            self.gate: asyncio.Event | None = None

        async def close_leg_executed(self, symbol, leg_id, coids, since_ms):
            if self.gate is not None and leg_id == "10":
                await self.gate.wait()
            return await super().close_leg_executed(symbol, leg_id, coids, since_ms)

    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=1, qty=1.0)
    _pending_leg_row(ctx, leg_id="11", volume=1, qty=1.0)
    port = _SlowEvidencePort([_leg("10", "buy", 1.0, open_time=0.0),
                              _leg("11", "buy", 1.0, open_time=1.0),
                              _leg("20", "sell", 1.0, open_time=2.0)])
    eng = OneWayEmulator(store_ctx=ctx)

    async def overlap():
        port.gate = asyncio.Event()
        first = asyncio.ensure_future(eng.restart_replay(port))
        for _ in range(5):
            await asyncio.sleep(0)
        second = await eng.restart_replay(port)
        port.gate.set()
        return await first, second

    first, second = _run(overlap())
    assert first is True and second is False
    assert sum(volume for _leg_id, volume in _dispatched(port)) == 1
    store.close()


class _GatedClosePort(_FakePort):
    """A port whose first ``close_leg`` after arming signals ``entered`` and waits on ``gate``."""

    def __init__(self, legs):
        super().__init__(legs)
        self.gate: asyncio.Event | None = None
        self.entered: asyncio.Event | None = None

    async def close_leg(self, symbol, leg_id, volume, coid):
        if (self.gate is not None and self.entered is not None
                and not self.entered.is_set()):
            self.entered.set()
            await self.gate.wait()
        await super().close_leg(symbol, leg_id, volume, coid)


def __test_overlapping_closes_never_spend_one_net_budget_twice__(journal):
    """A close still sending when the next one starts cannot spend the symbol's net again.

    The book is long 3 against a short 1, net long 2. Each close asks for 2,
    the net its own read shows. If the second planned from a read that does
    not show the first one's send, the two would close 4 of the long side
    and flip the net position short.
    """
    port = _GatedClosePort([_leg("10", "buy", 3.0, open_time=0.0),
                            _leg("20", "sell", 1.0, open_time=1.0)])
    eng = journal.emulator()

    async def overlap():
        port.gate = asyncio.Event()
        port.entered = asyncio.Event()
        first = asyncio.ensure_future(eng.run_close(_close_env("sell", 2.0), port))
        await port.entered.wait()
        fetches = port.fetch_calls
        with pytest.raises(ExchangeConnectionError):
            await eng.run_close(_close_env("sell", 2.0, pine_id="Other"), port)
        assert port.fetch_calls == fetches
        port.gate.set()
        return await first

    result = _run(overlap())
    assert result.dispatched_qty == 2.0
    assert _dispatched(port) == [("10", 2)]
    # Released once the first close returned: the next one is planned again.
    port.gate = None
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0), _leg("20", "sell", 1.0, open_time=1.0)])
    again = _run(eng.run_close(_close_env("sell", 2.0, pine_id="Other"), port))
    assert again.dispatched_qty == 0.0 and _dispatched(port) == [("10", 2)]


def __test_a_reversal_never_plans_while_a_close_holds_the_symbol__(journal):
    """A market reversal arriving while a close sends neither reads, closes nor opens."""
    port = _GatedClosePort([_leg("10", "buy", 3.0, open_time=0.0),
                            _leg("20", "sell", 1.0, open_time=1.0)])
    eng = journal.emulator()

    async def overlap():
        port.gate = asyncio.Event()
        port.entered = asyncio.Event()
        first = asyncio.ensure_future(eng.run_close(_close_env("sell", 2.0), port))
        await port.entered.wait()
        with pytest.raises(ExchangeConnectionError):
            await eng.run_reversal(_entry_env("sell", 4.0), port)
        port.gate.set()
        await first

    _run(overlap())
    assert _dispatched(port) == [("10", 2)]
    assert port.placed == []
    assert _live_residuals(eng) == []


def __test_a_residual_drain_waits_while_a_close_holds_the_symbol__(journal):
    """A drain never closes the opposing legs from a read another close pass is acting on."""
    port = _GatedClosePort([_leg("20", "sell", 2.0, open_time=0.0),
                            _leg("21", "sell", 1.0, open_time=1.0)])
    eng = journal.emulator()
    # The reversal's close of leg 20 and 21 was dispatched; the read still lists them.
    _run(eng.run_reversal(_entry_env("buy", 4.0), port))
    assert len(_live_residuals(eng)) == 1
    sent = list(port.closed)

    async def overlap():
        port.gate = asyncio.Event()
        port.entered = asyncio.Event()
        close = CloseIntent(pine_id="Short", symbol="EURUSD", side="buy", qty=1.0)
        first = asyncio.ensure_future(eng.run_close(
            DispatchEnvelope(intent=close, run_tag="t000", bar_ts_ms=2000), port))
        await port.entered.wait()
        fetches = port.fetch_calls
        await eng.drain_residual_opens(port)
        assert port.fetch_calls == fetches
        port.gate.set()
        await first

    _run(overlap())
    assert port.closed[:len(sent)] == sent
    # Only the close's own send went out; the drain sent nothing.
    assert len(port.closed) == len(sent) + 1
    assert port.placed == []
    assert len(_live_residuals(eng)) == 1


def __test_close_leg_attempt_ids_map_back_to_the_leg_close__():
    """A replay re-send's id resolves to the leg close's own id; other ids pass through."""
    leg_coid = close_leg_client_order_id("t000-x-y-c", "10")
    attempt = close_leg_attempt_client_order_id(leg_coid, 2)
    assert attempt != leg_coid
    assert close_leg_row_client_order_id(attempt) == leg_coid
    assert close_leg_row_client_order_id(leg_coid) == leg_coid


# === run_reversal — decomposition =======================================

def _entry_env(side, qty, *, symbol="EURUSD", pine_id="Long",
               order_type=OrderType.MARKET, limit=None, stop=None,
               stop_fired=False) -> DispatchEnvelope:
    intent = EntryIntent(
        pine_id=pine_id, symbol=symbol, side=side, qty=qty,
        order_type=order_type, limit=limit, stop=stop,
        triggered_entry=stop_fired,
    )
    return DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000)


def __test_run_reversal_pure_add_opens_only__():
    """A reversal with no opposing legs closes nothing and opens the whole size."""
    # No opposing legs -> nothing to close, open the whole size.
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_reversal(_entry_env("buy", 1.0), port))
    assert port.closed == []
    assert port.placed == [("EURUSD", 1.0)]
    assert res.open_qty == 1.0 and len(res.opened_orders) == 1


def __test_run_reversal_flip_closes_opposing_then_opens_residual__():
    """A flipping reversal closes the opposing legs, then opens the residual size."""
    # Short 2.0, buy 3.0 -> close the 2.0 sell leg, open the 1.0 long residual.
    # The venue's read right after the close no longer lists the leg, so the
    # residual opens within the same call.
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    port.closes_execute = True
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert _dispatched(port) == [("20", 2)]
    assert port.placed == [("EURUSD", 1.0)]
    assert res.closes == (("20", 2),) and res.open_qty == 1.0


def __test_run_reversal_exact_flatten_opens_nothing__():
    """An exact-flatten reversal closes the opposing legs and opens nothing."""
    # Short 2.0, buy 2.0 -> close the sell leg, open nothing.
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_reversal(_entry_env("buy", 2.0), port))
    assert _dispatched(port) == [("20", 2)]
    assert port.placed == []
    assert res.open_qty == 0.0 and res.opened_orders == ()


def __test_run_reversal_below_min_residual_skips_whole_reversal__():
    """A below-minimum residual makes the pre-flight skip the reversal before any close lands."""
    # Residual 1.0 is below the broker minimum -> the pre-flight raises BEFORE
    # any close lands, so the book is never left half-reduced.
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)], min_qty=2.0)
    eng = OneWayEmulator(store_ctx=None)
    with pytest.raises(OrderSkippedByPlugin):
        _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert port.closed == [] and port.placed == []


def __test_run_reversal_partial_leg_unsupported_closes_the_whole_leg__():
    """An order smaller than the opposing leg closes that leg whole on a no-partial port."""
    # Broker holds a 3.0 sell leg while the combined order is only 2.0 —
    # the FIFO plan would reduce the leg partially, which the port cannot
    # express, so the leg is closed whole and nothing is opened.
    port = _FakePort([_leg("20", "sell", 3.0, open_time=0.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_reversal(_entry_env("buy", 2.0), port))
    assert _dispatched(port) == [("20", 3)]
    assert res.open_qty == 0.0 and port.placed == []


def __test_run_reversal_whole_leg_flip_passes_without_partial_support__():
    """A whole-leg flip works unchanged on a no-partial port."""
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert _dispatched(port) == [("20", 2)]
    assert res.open_qty == 1.0


def __test_run_reversal_persists_close_leg_rows__(tmp_path):
    """A reversal persists its FIFO close-leg rows and finalises them on success."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    port.closes_execute = True
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert _dispatched(port) == [("20", 2)]
    assert port.placed == [("EURUSD", 1.0)]
    assert list(iter_active_close_legs(ctx)) == []  # close-leg row finalised
    store.close()


def __test_run_reversal_resting_limit_rests_without_closing_opposing__():
    """A LIMIT reversal rests at full size without closing the opposing legs at placement."""
    # A LIMIT reversal opposing an open short must NOT close the short at
    # placement — it rests full-size; decomposition happens only on fill.
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_reversal(
        _entry_env("buy", 3.0, order_type=OrderType.LIMIT, limit=1.05), port,
    ))
    assert port.closed == []  # no opposing-leg close at placement
    assert port.placed == [("EURUSD", 3.0)]  # full combined size rests
    assert res.closes == () and res.open_qty == 3.0


def __test_run_reversal_resting_stop_rests_without_closing_opposing__():
    """A STOP reversal rests at full size without closing the opposing legs at placement."""
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_reversal(
        _entry_env("buy", 3.0, order_type=OrderType.STOP, stop=1.15), port,
    ))
    assert port.closed == []
    assert port.placed == [("EURUSD", 3.0)]
    assert res.open_qty == 3.0 and len(res.opened_orders) == 1


# === run_exit_bracket / clear — per-leg ownership =======================

def _exit_env(pine_id, from_entry, *, side="sell", qty=3.0, symbol="EURUSD",
              tp=None, sl=None, trail=None, trail_price=None) -> DispatchEnvelope:
    intent = ExitIntent(
        pine_id=pine_id, from_entry=from_entry, symbol=symbol, side=side, qty=qty,
        tp_price=tp, sl_price=sl, trail_offset=trail, trail_price=trail_price,
    )
    return DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000)


def _cancel_env(pine_id, *, from_entry=None, symbol="EURUSD") -> DispatchEnvelope:
    intent = CancelIntent(pine_id=pine_id, symbol=symbol, from_entry=from_entry)
    return DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000)


def _amended_levels(port: _FakePort) -> list[tuple[str, float | None, float | None]]:
    return [(leg_id, tp, sl) for leg_id, _side, tp, sl, _trail, _coid in port.amended]


def __test_run_exit_bracket_replicates_across_legs__():
    """``run_exit_bracket`` replicates the same bracket onto every leg in FIFO order."""
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))
    assert res.legs == ("10", "11") and res.skipped is False
    # Same bracket on every position-side leg, FIFO order.
    assert _amended_levels(port) == [("10", 1.20, 1.00), ("11", 1.20, 1.00)]


def __test_run_exit_bracket_converts_trailing_ticks_to_price_distance__():
    """The strategy's tick offset crosses PositionPort in absolute price units."""
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None, mintick=0.01)

    _run(eng.run_exit_bracket(_exit_env("TR", "Long", trail=10.0), port))

    assert len(port.amended) == 1
    leg_id, side, tp, sl, trail, _coid = port.amended[0]
    assert (leg_id, side, tp, sl, trail) == ("10", "sell", None, None, 0.1)


def __test_run_exit_bracket_trail_activation_limit_reported_once_per_exit__(monkeypatch):
    """The per-leg ``trail_price`` limitation is reported once per exit, not per dispatch.

    A trailing exit is re-dispatched on every bar it changes; repeating the
    same limitation notice on each one floods an unattended live log
    (ctrader cycle 335: 20 identical warnings in one 6h cycle).
    """
    warnings: list[tuple] = []
    monkeypatch.setattr(
        "pynecore.core.broker.one_way_emulator._blog_warning",
        lambda msg, *args, **_kw: warnings.append(args),
    )
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None, mintick=0.01)

    for _ in range(3):
        _run(eng.run_exit_bracket(
            _exit_env("TR", "Long", trail=10.0, trail_price=1.15), port))
    assert len(warnings) == 1
    # The identical re-dispatches change nothing on the leg, so they are not
    # re-sent (a re-send would reset the venue's trailing stop).
    assert len(port.amended) == 1

    # A different exit is its own limitation instance.
    _run(eng.run_exit_bracket(
        _exit_env("TR2", "Long", trail=10.0, trail_price=1.15), port))
    assert [w[1] for w in warnings] == ["TR", "TR2"]


def __test_run_exit_bracket_flat_skips__():
    """``run_exit_bracket`` on a flat book skips, amending no legs."""
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_exit_bracket(_exit_env("TP", "Long", sl=1.00), port))
    assert res == BracketFanResult(legs=(), skipped=True)
    assert port.amended == []


def __test_run_exit_bracket_opposite_net_side_skips__():
    """An exit whose parent was reversed away skips instead of bracketing the new side.

    ``intent.side`` is the closing direction, so a ``sell`` exit belongs to a
    long. When the book has since flipped short, the exit's levels are upside
    down for that side (a long's take-profit sits above the price, a short's
    must sit below) and the venue rejects the attach — which escalates into a
    defensive close for a parent that no longer exists. The protected position
    is simply gone, so this is the same non-halting skip as a flat book.
    """
    port = _FakePort([_leg("2", "sell", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))
    assert res == BracketFanResult(legs=(), skipped=True)
    assert port.amended == []
    # The mirror case still attaches: a buy exit protects the short book.
    res = _run(eng.run_exit_bracket(
        _exit_env("TP", "Short", side="buy", tp=1.00, sl=1.20), port,
    ))
    assert res.legs == ("2",) and res.skipped is False
    assert _amended_levels(port) == [("2", 1.00, 1.20)]


def __test_run_exit_bracket_protects_only_net_survivor_legs__():
    """The bracket is amended onto only the net-survivor legs, not the gross majority side."""
    # Mixed book net long 2 (3 buys, 1 sell): the opposing sell virtually
    # FIFO-closes the oldest buy, so the bracket is amended onto ONLY the two
    # net-survivor legs — never the gross majority side, which on an SL hit
    # would close 3 and flip the book to the minority short.
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0),
                      _leg("12", "buy", 1.0, open_time=2.0),
                      _leg("20", "sell", 1.0, open_time=3.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))
    assert res.legs == ("11", "12")  # oldest buy "10" consumed by the sell, dropped
    assert [leg_id for leg_id, *_rest in port.amended] == ["11", "12"]


def _entry_row(ctx, coid, *, side, qty, pine_id, kind=ENTRY_KIND_POSITION,
               leg_id=None, filled=0.0, symbol="EURUSD"):
    create_entry_order_row(
        ctx, coid=coid, symbol=symbol, side=side, qty=qty,
        intent_key=pine_id, pine_entry_id=pine_id,
        kind=kind, order_type="market" if kind == ENTRY_KIND_POSITION else "limit",
    )
    if leg_id is not None or filled:
        ctx.upsert_order(coid, state=STATE_CONFIRMED,
                         exchange_order_id=leg_id, filled_qty=filled)


def __test_run_exit_bracket_defers_while_a_sibling_entry_fill_is_pending__(tmp_path):
    """A protected-side position entry with an unprocessed fill defers the fan.

    The live pyramiding incident: L1's leg is open and fetchable, L2's market
    entry was dispatched in the same sync but its FILL has not been processed
    yet, so neither the ownership index nor the venue snapshot can be tied to
    L2's leg. Fanning now would attach L2-X onto L1's leg only and the intent
    would go active without ever re-fanning — L2's own leg would stay
    permanently unprotected. The fan must instead skip (the same non-halting
    defer as a flat book) and cover BOTH legs on the re-run after the fill.
    """
    store, ctx = _make_store(tmp_path)
    _entry_row(ctx, "e-l1", side="buy", qty=1.0, pine_id="L1", leg_id="10", filled=1.0)
    _entry_row(ctx, "e-l2", side="buy", qty=1.0, pine_id="L2")  # fill pending
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    res = _run(eng.run_exit_bracket(
        _exit_env("L2-X", "L2", qty=1.0, tp=1.20, sl=1.00), port,
    ))
    assert res.skipped is True and res.legs == ()
    assert res.skip_reason is not None
    assert port.amended == []
    assert list(iter_active_bracket_ownerships(ctx)) == []
    # The fill lands (leg "11" journaled + fetchable): the re-run covers both.
    ctx.upsert_order("e-l2", state=STATE_CONFIRMED,
                     exchange_order_id="11", filled_qty=1.0)
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0),
                   _leg("11", "buy", 1.0, open_time=1.0)])
    res = _run(eng.run_exit_bracket(
        _exit_env("L2-X", "L2", qty=1.0, tp=1.20, sl=1.00), port,
    ))
    assert res.legs == ("10", "11") and res.skipped is False
    assert _amended_levels(port) == [("10", 1.20, 1.00), ("11", 1.20, 1.00)]
    store.close()


def __test_run_exit_bracket_incomplete_book_stalls_after_the_grace__(tmp_path):
    """The routine fill wait defers quietly; one that outgrows the grace is flagged once.

    Measured live (ctrader cycle 360, pyramid): every pyramid round deferred its
    new leg's exit for the one or two syncs its fill took to drain, each logged
    as a WARNING. Bybit cycle 438 took four syncs within the same second, and a
    sync count flagged that routine wait as a stall. A wait past
    :data:`BOOK_INCOMPLETE_GRACE_S` leaves the leg unprotected for real, so
    exactly that skip is ``stalled``, however many syncs came before it; the
    wait restarts once the fan passes the gate.
    """
    store, ctx = _make_store(tmp_path)
    _entry_row(ctx, "e-l1", side="buy", qty=1.0, pine_id="L1", leg_id="10", filled=1.0)
    _entry_row(ctx, "e-l2", side="buy", qty=1.0, pine_id="L2")  # fill pending
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    env = _exit_env("L2-X", "L2", qty=1.0, tp=1.20, sl=1.00)

    # Many syncs within the grace, then one past it, then another.
    times = [0.1 * i for i in range(10)] + [BOOK_INCOMPLETE_GRACE_S, BOOK_INCOMPLETE_GRACE_S + 1.0]
    results = [_run(eng.run_exit_bracket(env, port, now=t)) for t in times]
    assert all(r.skipped for r in results)
    assert [r.stalled for r in results] == [False] * 10 + [True, False]
    assert "unprotected" in (results[10].skip_reason or "")

    ctx.upsert_order("e-l2", state=STATE_CONFIRMED, exchange_order_id="11", filled_qty=1.0)
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0),
                   _leg("11", "buy", 1.0, open_time=1.0)])
    assert _run(eng.run_exit_bracket(env, port)).skipped is False
    assert not eng._book_incomplete_deferrals  # noqa: SLF001
    _entry_row(ctx, "e-l3", side="buy", qty=1.0, pine_id="L3")  # next round
    later = BOOK_INCOMPLETE_GRACE_S + 100.0
    again = [_run(eng.run_exit_bracket(env, port, now=later + t)) for t in (0.0, 1.0, 2.0)]
    assert [r.stalled for r in again] == [False] * 3
    store.close()


def __test_run_exit_bracket_ignores_resting_and_opposite_side_entries__(tmp_path):
    """Unfilled rows that cannot hide a protected-side leg do not defer the fan.

    A resting (working) entry has no venue leg until it fills, and an
    opposite-side entry's future leg is irrelevant to this exit's protected
    side — neither may hold the bracket back from the legs that ARE open.
    """
    store, ctx = _make_store(tmp_path)
    _entry_row(ctx, "e-l1", side="buy", qty=1.0, pine_id="L1", leg_id="10", filled=1.0)
    _entry_row(ctx, "e-rest", side="buy", qty=1.0, pine_id="L9",
               kind=ENTRY_KIND_WORKING, leg_id="order-77")
    _entry_row(ctx, "e-s1", side="sell", qty=1.0, pine_id="S1")  # fill pending
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    res = _run(eng.run_exit_bracket(
        _exit_env("L1-X", "L1", qty=1.0, tp=1.20, sl=1.00), port,
    ))
    assert res.legs == ("10",) and res.skipped is False
    assert _amended_levels(port) == [("10", 1.20, 1.00)]
    store.close()


def __test_clear_only_owns__(journal):
    """Clearing one exit wipes only its own legs; a co-located exit's bracket survives."""
    # THE regression test: a TP-exit and an SL-exit on the SAME legs. Cancelling
    # TP must clear ONLY TP's legs; SL's bracket must survive (the plugin's
    # broadcast clear strips both).
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "Long", sl=1.00), port))
    assert len(list(journal.rows())) == 4  # 2 exits x 2 legs
    port.amended.clear()
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="Long"), port))
    assert set(res.legs) == {"10", "11"}
    # Each owned leg is re-amended to what its remaining owner (SL) asks for:
    # the TP is dropped, the stop stays.
    assert sorted(_amended_levels(port)) == [("10", None, 1.00), ("11", None, 1.00)]
    surviving = list(journal.rows())
    assert len(surviving) == 2
    assert all(row.intent_key == "SL\0Long" for row in surviving)


def __test_two_exits_distinct_ownership__(journal):
    """Two exits with different pine_ids get distinct intent_keys and attach coids."""
    # Different pine_ids -> different attach coids -> disjoint coid namespaces +
    # distinct intent_keys, which is what makes the scoped clear work.
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "Long", sl=1.00), port))
    rows = list(journal.rows())
    assert {row.intent_key for row in rows} == {"TP\0Long", "SL\0Long"}
    assert len({row.client_order_id for row in rows}) == 2


def __test_clear_all_by_pine_id_when_from_entry_none__(journal):
    """A cancel with ``from_entry=None`` clears every exit sharing the pine_id, leaving others."""
    # A cancel-all (from_entry None) drops every exit sharing the pine_id, across
    # from_entry values, and leaves a different pine_id untouched.
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "L1", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("TP", "L2", tp=1.30), port))
    _run(eng.run_exit_bracket(_exit_env("RUN", "L1", sl=1.00), port))
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry=None), port))
    assert len(res.legs) == 2  # both TP rows (L1 + L2), not RUN
    surviving = list(journal.rows())
    assert all(row.intent_key == "RUN\0L1" for row in surviving)


def __test_clear_pure_path_is_noop__():
    """Clearing an exit that owns no leg is a no-op, without a store too (``store_ctx=None``)."""
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="Long"), port))
    assert res == ClearFanResult(legs=())
    assert port.amended == []


def __test_without_a_store_co_owners_are_merged_and_retired_like_with_one__():
    """A library caller without a store keeps the owner merge of a shared leg.

    The live defect (review F4): without a store no ownership was recorded, so
    the SL exit's attach replaced the TP exit's take-profit on the leg, and
    cancelling the TP exit retired nothing, leaving its level armed.
    """
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = OneWayEmulator(store_ctx=None)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", qty=1.0, tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "Long", qty=1.0, sl=1.00), port))
    assert _amended_levels(port) == [("10", 1.20, None), ("10", 1.20, 1.00)]
    port.amended.clear()
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="Long"), port))
    assert res.legs == ("10",)
    assert _amended_levels(port) == [("10", None, 1.00)]
    port.amended.clear()
    res = _run(eng.run_exit_bracket_clear(_cancel_env("SL", from_entry="Long"), port))
    assert res.legs == ("10",)
    assert _amended_levels(port) == [("10", None, None)]


def __test_an_ambiguous_retirement_is_finished_by_the_drain_in_session__(journal):
    """A retirement left ``clearing`` is finished by the per-sync drain, with or without a store.

    The leg is re-amended to the remaining owner's levels under the persisted
    clear coid, and only the retired exit's key is reported.
    """
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "Long", sl=1.00), port))
    cancel = _cancel_env("TP", from_entry="Long")
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket_clear(cancel, port))
    port.fail_amend_unknown = False
    assert _own_states(journal.rows()) == {
        "TP\0Long": BRACKET_OWN_STATE_CLEARING, "SL\0Long": "active"}
    # The leg takes no other write until the unsettled one is settled: once
    # its window has passed.
    _run(eng.settle_leg_writes("EURUSD", port, 0.0))
    _run(eng.settle_leg_writes("EURUSD", port, UNSETTLED_WRITE_WINDOW_S))
    port.amended.clear()
    assert _run(eng.drain_clearing_rows("EURUSD", port)) == {"TP\0Long"}
    assert port.amended == [("10", "sell", None, 1.00, None, cancel.client_order_id(KIND_CANCEL))]
    assert _own_states(journal.rows()) == {"SL\0Long": "active"}


def __test_bracket_ownership_persist_and_replay__(tmp_path):
    """Restart replay releases the orphan ownership row and re-asserts the surviving leg."""
    # Crash sim: a bracket replicated onto two legs, then one leg vanishes. The
    # restart pass releases the orphan row and re-asserts the surviving leg.
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))
    assert len(list(iter_active_bracket_ownerships(ctx))) == 2
    port.set_legs([_leg("10", "buy", 2.0, open_time=0.0)])  # leg "11" gone
    port.amended.clear()
    _run(eng.restart_replay(port))
    surviving = list(iter_active_bracket_ownerships(ctx))
    assert [row.extras[EXTRAS_KEY_BRACKET_OWN_LEG_ID] for row in surviving] == ["10"]
    assert [leg_id for leg_id, *_rest in port.amended] == ["10"]  # only "10" re-asserted
    store.close()


def __test_clear_timeout_leaves_clearing_row_then_replay_reclears__(tmp_path):
    """A clear whose amend times out marks the row clearing; restart re-clears and releases it."""
    # F2: a bracket-clear whose amend times out (OrderDispositionUnknownError)
    # must NOT leave the row "active" — that would make restart_replay re-assert
    # the very bracket the script asked to cancel. Persist-first marks it
    # "clearing"; the next restart re-CLEARS (amend to None) and releases it.
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))
    # The clear's amend times out: the row stays live in the "clearing" phase.
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="Long"), port))
    rows = list(iter_active_bracket_ownerships(ctx))
    assert len(rows) == 1
    assert rows[0].extras[EXTRAS_KEY_BRACKET_OWN_STATE] == BRACKET_OWN_STATE_CLEARING
    # Restart (a fresh emulator: the process that sent the timed-out clear is
    # gone) with a healthy broker: the clearing row re-clears (amend to None,
    # NOT the original 1.20/1.00) and is released — the cancel is honoured.
    port.fail_amend_unknown = False
    port.amended.clear()
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))
    assert _amended_levels(port) == [("10", None, None)]  # re-clear, not re-assert
    assert list(iter_active_bracket_ownerships(ctx)) == []  # released
    store.close()


def __test_run_exit_bracket_midfan_reject_releases_only_unamended_row__(tmp_path):
    """A mid-fan amend reject releases only the unamended leg's row, keeping the attached one."""
    # Leg "10" amends OK, leg "11" rejects mid-fan. The reject surfaces as a
    # BracketAttachAfterFillRejectedError (open + unprotected position -> the
    # engine flattens defensively, not halts). Leg "11"'s persist-first
    # ownership row is released (its bracket never attached); leg "10"'s row
    # stays active — that bracket DOES exist and the defensive close flattens it.
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)],
                     fail_amend_leg="11")
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(BracketAttachAfterFillRejectedError) as excinfo:
        _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))
    assert excinfo.value.from_entry == "Long" and excinfo.value.exit_id == "TP"
    assert excinfo.value.position_side == "buy"  # flatten a long (exit side sell)
    assert [leg_id for leg_id, *_rest in port.amended] == ["10"]  # "11" never amended
    surviving = list(iter_active_bracket_ownerships(ctx))
    assert [row.extras[EXTRAS_KEY_BRACKET_OWN_LEG_ID] for row in surviving] == ["10"]


def __test_dropped_leg_clear_timeout_does_not_skip_survivor_amend__(tmp_path):
    """A dropped leg's clear timeout does not abort the fan; survivors still get new levels."""
    # F: run_exit_bracket pre-clears dropped survivor legs BEFORE amending the
    # survivors. If that pre-clear times out (OrderDispositionUnknownError) the
    # error MUST NOT abort the fan — the dispatch path would park it while
    # promoting the new intent into _active_intents, so the next diff sees
    # Pine == active, never re-runs the fan, and the survivors stay on STALE
    # protection forever. The dropped row stays "clearing" for drain/replay to
    # retry; the survivors get their NEW levels now.
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0),
                      _leg("12", "buy", 1.0, open_time=2.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    # First attach protects all three buys.
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))
    assert {row.extras[EXTRAS_KEY_BRACKET_OWN_LEG_ID]
            for row in iter_active_bracket_ownerships(ctx)} == {"10", "11", "12"}
    # An opposing sell appears: virtual FIFO consumes the oldest buy "10", which
    # now drops out of the survivor set on re-attach. The dropped leg's clear
    # times out; the survivor amend on "11"/"12" must still land the new levels.
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0),
                   _leg("11", "buy", 1.0, open_time=1.0),
                   _leg("12", "buy", 1.0, open_time=2.0),
                   _leg("20", "sell", 1.0, open_time=3.0)])
    port.fail_amend_unknown_leg = "10"
    port.amended.clear()
    res = _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.30, sl=0.90), port))
    # Survivors re-amended with the NEW levels despite the dropped-leg timeout.
    assert res.legs == ("11", "12")
    assert _amended_levels(port) == [("11", 1.30, 0.90), ("12", 1.30, 0.90)]
    rows = {row.extras[EXTRAS_KEY_BRACKET_OWN_LEG_ID]: row
            for row in iter_active_bracket_ownerships(ctx)}
    # Dropped leg "10" stranded in "clearing" for drain_clearing_rows to retry.
    assert rows["10"].extras[EXTRAS_KEY_BRACKET_OWN_STATE] == BRACKET_OWN_STATE_CLEARING
    # A healthy drain re-clears + releases the dropped row.
    port.fail_amend_unknown_leg = None
    _run(eng.drain_clearing_rows("EURUSD", port))
    assert {row.extras[EXTRAS_KEY_BRACKET_OWN_LEG_ID]
            for row in iter_active_bracket_ownerships(ctx)} == {"11", "12"}
    store.close()


def __test_restart_replay_clear_timeout_does_not_abort_startup__(tmp_path):
    """A re-clear timeout during restart replay does not abort startup; the row stays clearing."""
    # F: _replay_bracket_leg re-clears a "clearing" row at startup. If that
    # re-clear times out (OrderDispositionUnknownError) it must NOT propagate —
    # restart_replay runs inside a sync wrapper that only catches
    # ExchangeConnectionError, so an ambiguous round-trip would abort startup.
    # The row stays "clearing" for the next drain/replay to retry.
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))
    # Drive the row into "clearing" via a clear whose amend times out.
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="Long"), port))
    assert (list(iter_active_bracket_ownerships(ctx))[0]
            .extras[EXTRAS_KEY_BRACKET_OWN_STATE] == BRACKET_OWN_STATE_CLEARING)
    # Restart: the re-clear for leg "10" times out again — must NOT raise.
    port.fail_amend_unknown = False
    port.fail_amend_unknown_leg = "10"
    _run(eng.restart_replay(port))  # no exception escapes
    rows = list(iter_active_bracket_ownerships(ctx))
    assert len(rows) == 1
    assert rows[0].extras[EXTRAS_KEY_BRACKET_OWN_STATE] == BRACKET_OWN_STATE_CLEARING
    # A later healthy replay re-clears (amend to None) and releases the row.
    port.fail_amend_unknown_leg = None
    port.amended.clear()
    _run(eng.restart_replay(port))
    assert _amended_levels(port) == [("10", None, None)]
    assert list(iter_active_bracket_ownerships(ctx)) == []
    store.close()


class _RefusingPort(_FakePort):
    """A port whose next bracket amends raise the given errors, in order."""

    def __init__(self, legs, refusals: list[Exception]) -> None:
        super().__init__(legs)
        self.refusals = list(refusals)

    async def amend_bracket(self, symbol, leg_id, **kwargs) -> None:
        if self.refusals:
            raise self.refusals.pop(0)
        await super().amend_bracket(symbol, leg_id, **kwargs)


def __test_a_refused_bracket_re_assert_never_aborts_the_replay__(tmp_path):
    """The venue refusing a journaled bracket (a crossed level) is recorded, not raised."""
    from pynecore.core.broker.exceptions import ExchangeOrderRejectedError as Rejected

    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))

    refusing = _RefusingPort([_leg("10", "buy", 2.0)],
                             [Rejected("stop above the market", level_crossed=True)])
    assert _run(eng.restart_replay(refusing)) is True
    refusals = eng.take_replay_refusals()
    assert [(r.intent_key, r.leg_id, r.sl_price) for r in refusals] == [
        ("TP\x00Long", "10", 1.00)]
    assert refusals[0].error.level_crossed
    assert eng.take_replay_refusals() == []
    assert [row.extras[EXTRAS_KEY_BRACKET_OWN_STATE]
            for row in iter_active_bracket_ownerships(ctx)] == ["active"]
    store.close()


def __test_the_re_arm_after_a_refused_replay_rewrites_the_same_ownership_row__(tmp_path):
    """A refused row is never left behind next to a second row for the same leg.

    The engine re-arms the refused exit through the regular dispatch, under
    a later bar's attach coid; the ownership row is keyed by exit and leg
    only, so the re-arm rewrites the refused row instead of adding one, and
    the next restart replays exactly the re-armed levels.
    """
    from pynecore.core.broker.exceptions import ExchangeOrderRejectedError as Rejected

    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))

    refusing = _RefusingPort([_leg("10", "buy", 2.0)],
                             [Rejected("stop above the market", level_crossed=True)])
    assert _run(eng.restart_replay(refusing)) is True
    assert len(eng.take_replay_refusals()) == 1

    rearm = _exit_env("TP", "Long", tp=1.20, sl=0.98)
    rearm = DispatchEnvelope(intent=rearm.intent, run_tag="t000", bar_ts_ms=2000)
    _run(eng.run_exit_bracket(rearm, refusing))
    rows = list(iter_active_bracket_ownerships(ctx))
    assert len(rows) == 1, "one leg, one ownership row"
    assert rows[0].extras[EXTRAS_KEY_BRACKET_OWN_STATE] == "active"

    replay_port = _FakePort([_leg("10", "buy", 2.0)])
    assert _run(eng.restart_replay(replay_port)) is True
    assert _amended_levels(replay_port) == [("10", 1.20, 0.98)]
    store.close()


def __test_a_crossed_refusal_of_a_bracket_attach_keeps_the_crossed_verdict__(tmp_path):
    """The engine tells Pine's crossed-level exit from a venue refusing protection."""
    from pynecore.core.broker.exceptions import (
        BracketAttachAfterFillRejectedError,
        ExchangeOrderRejectedError as Rejected,
    )

    store, ctx = _make_store(tmp_path)
    port = _RefusingPort([_leg("10", "buy", 2.0)],
                         [Rejected("stop above the market", level_crossed=True)])
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(BracketAttachAfterFillRejectedError) as raised:
        _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))
    assert raised.value.level_crossed
    assert list(iter_active_bracket_ownerships(ctx)) == [], "the refused row is released"
    store.close()


def __test_a_bracket_re_assert_into_a_closed_market_retries_the_replay__(tmp_path):
    from pynecore.core.broker.exceptions import MarketClosedError

    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))

    closed = _RefusingPort([_leg("10", "buy", 2.0)], [MarketClosedError("market closed")])
    assert _run(eng.restart_replay(closed)) is False
    assert eng.take_replay_refusals() == []
    assert _run(eng.restart_replay(closed)) is True
    assert _amended_levels(closed) == [("10", 1.20, 1.00)]
    store.close()


def __test_a_refused_re_clear_on_restart_keeps_the_row_clearing__(tmp_path):
    from pynecore.core.broker.exceptions import MarketClosedError

    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="Long"), port))

    refusing = _RefusingPort([_leg("10", "buy", 2.0)], [MarketClosedError("closed")])
    assert _run(eng.restart_replay(refusing)) is True  # must not raise
    rows = list(iter_active_bracket_ownerships(ctx))
    assert [row.extras[EXTRAS_KEY_BRACKET_OWN_STATE] for row in rows] == [
        BRACKET_OWN_STATE_CLEARING]
    store.close()


# === Per-entry targeting and merged levels on shared legs ===============

_PYRAMID_LEVELS = {"L1": (1.30, 1.00), "L2": (1.25, 1.05), "L3": (1.20, 1.08)}


def _venue_levels(port: _FakePort) -> dict[str, tuple[float | None, float | None]]:
    """The protection each leg carries after every amend so far (the last one wins)."""
    levels: dict[str, tuple[float | None, float | None]] = {}
    for leg_id, _side, tp, sl, _trail, _coid in port.amended:
        levels[leg_id] = (tp, sl)
    return levels


def _pyramid_legs() -> list[PositionLeg]:
    return [_leg(str(n), "buy", 1.0, open_time=float(n), entry_id=f"L{n}") for n in (1, 2, 3)]


def _attach_pyramid_exits(eng: OneWayEmulator, port: _FakePort) -> None:
    for entry, (tp, sl) in _PYRAMID_LEVELS.items():
        res = _run(eng.run_exit_bracket(
            _exit_env(f"{entry}-X", entry, qty=1.0, tp=tp, sl=sl), port,
        ))
        assert res.legs == (entry.removeprefix("L"),)


def _capture_warnings(monkeypatch) -> list[str]:
    warnings: list[str] = []
    monkeypatch.setattr(
        "pynecore.core.broker.one_way_emulator._blog_warning",
        lambda msg, *args, **_kw: warnings.append(msg % args),
    )
    return warnings


def __test_hedged_pyramid_each_leg_carries_its_own_entrys_bracket__(journal):
    """Three pyramided entries, one leg each: every leg carries its own exit's levels.

    The live defect: every exit was amended onto every leg, so the last
    writer (L3-X) won on all three legs and L1's and L2's own levels existed
    nowhere on the venue.
    """
    port = _FakePort(_pyramid_legs())
    eng = journal.emulator()
    _attach_pyramid_exits(eng, port)
    assert _venue_levels(port) == {"1": (1.30, 1.00), "2": (1.25, 1.05), "3": (1.20, 1.08)}
    assert [leg_id for leg_id, *_rest in port.amended] == ["1", "2", "3"]
    assert len(list(journal.rows())) == 3  # one row per exit


def __test_hedged_pyramid_retiring_one_exit_touches_only_its_own_leg__(tmp_path):
    """Retiring L3-X amends only L3's leg; L1's and L2's protection stays untouched."""
    store, ctx = _make_store(tmp_path)
    legs = _pyramid_legs()
    port = _FakePort(legs)
    eng = OneWayEmulator(store_ctx=ctx)
    _attach_pyramid_exits(eng, port)
    port.set_legs(legs[:2])  # the venue closed leg 3 at L3-X's take-profit
    port.amended.clear()
    res = _run(eng.run_exit_bracket_clear(_cancel_env("L3-X", from_entry="L3"), port))
    assert res.legs == ("3",)
    # Leg 3 is closed: it carries no bracket, so its row is released without a
    # venue call, and no other leg is written.
    assert _amended_levels(port) == []
    assert {row.intent_key for row in iter_active_bracket_ownerships(ctx)} == {
        "L1-X\0L1", "L2-X\0L2"}
    # A restart re-asserts each remaining leg's own levels.
    port.amended.clear()
    assert _run(eng.restart_replay(port)) is True
    assert _venue_levels(port) == {"1": (1.30, 1.00), "2": (1.25, 1.05)}
    store.close()


def __test_tp_only_and_sl_only_exits_on_one_leg_union__(journal):
    """A TP-only and an SL-only exit of one entry both stay on the leg; retiring one keeps the other."""
    port = _FakePort([_leg("1", "buy", 1.0, entry_id="L1")])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "L1", qty=1.0, tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "L1", qty=1.0, sl=1.00), port))
    assert _amended_levels(port) == [("1", 1.20, None), ("1", 1.20, 1.00)]
    # The SL exit is modified: the TP stays.
    _run(eng.run_exit_bracket(_exit_env("SL", "L1", qty=1.0, sl=1.02), port))
    assert _venue_levels(port) == {"1": (1.20, 1.02)}
    port.amended.clear()
    _run(eng.run_exit_bracket_clear(_cancel_env("SL", from_entry="L1"), port))
    assert _amended_levels(port) == [("1", 1.20, None)]
    port.amended.clear()
    _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="L1"), port))
    assert _amended_levels(port) == [("1", None, None)]
    assert list(journal.rows()) == []


def __test_unattributed_legs_keep_the_single_exit_behaviour__(journal):
    """A plugin that never attributes legs gets every exit replicated onto every leg."""
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = journal.emulator()
    res = _run(eng.run_exit_bracket(_exit_env("L1-X", "L1", qty=1.0, tp=1.20, sl=1.00), port))
    assert res.legs == ("10", "11")
    assert _amended_levels(port) == [("10", 1.20, 1.00), ("11", 1.20, 1.00)]


def __test_two_exits_on_unattributed_legs_merge_to_the_more_protective_levels__(
        journal, monkeypatch):
    """Conflicting levels on one leg give the level the price reaches first, reported once.

    For a long the highest stop-loss and the lowest take-profit, independent
    of the order the exits attached in; retiring the tighter exit hands the
    leg back to the remaining one.
    """
    warnings = _capture_warnings(monkeypatch)
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("L1-X", "L1", qty=1.0, tp=1.30, sl=1.00), port))
    _run(eng.run_exit_bracket(_exit_env("L2-X", "L2", qty=1.0, tp=1.25, sl=1.05), port))
    assert _venue_levels(port) == {"10": (1.25, 1.05)}
    # A re-dispatch of the looser exit does not loosen the leg.
    _run(eng.run_exit_bracket(_exit_env("L1-X", "L1", qty=1.0, tp=1.30, sl=1.00), port))
    assert _venue_levels(port) == {"10": (1.25, 1.05)}
    assert len([w for w in warnings if "different levels" in w]) == 1
    _run(eng.run_exit_bracket_clear(_cancel_env("L2-X", from_entry="L2"), port))
    assert _venue_levels(port) == {"10": (1.30, 1.00)}


def __test_merge_mirrors_for_a_short_and_keeps_the_tightest_trail__(journal):
    """A short takes the lowest stop-loss and the highest take-profit; trailing takes the tightest."""
    port = _FakePort([_leg("20", "sell", 1.0)])
    eng = journal.emulator(mintick=0.01)
    _run(eng.run_exit_bracket(
        _exit_env("S1-X", "S1", side="buy", qty=1.0, tp=0.90, sl=1.20, trail=20.0), port))
    _run(eng.run_exit_bracket(
        _exit_env("S2-X", "S2", side="buy", qty=1.0, tp=0.95, sl=1.15, trail=10.0), port))
    leg_id, side, tp, sl, trail, _coid = port.amended[-1]
    assert (leg_id, side, tp, sl, trail) == ("20", "buy", 0.95, 1.15, 0.1)


def __test_an_entry_without_a_leg_never_covers_other_entries_legs__(journal, monkeypatch):
    """An exit whose entry holds no leg skips instead of covering other entries' legs.

    Attribution comes only from the bot's own records and lost attribution
    is ``None``, so every open leg attributed to another entry means the
    exit's entry holds no leg — e.g. its leg was just closed at the venue by
    its own take-profit and Pine re-emits the exit before the fill is booked.
    Putting that exit's (touched) levels on the other entry's leg would close
    or flatten a position it does not protect. The fan is a non-halting skip,
    reported once per exit, and the exit's rows from an earlier attach on a
    leg it no longer covers are retired.
    """
    warnings = _capture_warnings(monkeypatch)
    legs = [_leg("1", "buy", 1.0, open_time=1.0, entry_id="L1"),
            _leg("2", "buy", 1.0, open_time=2.0, entry_id="L2"),
            _leg("3", "buy", 1.0, open_time=3.0, entry_id="L3")]
    port = _FakePort(legs)
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("L1-X", "L1", qty=1.0, tp=1.30, sl=1.00), port))
    _run(eng.run_exit_bracket(_exit_env("L3-X", "L3", qty=1.0, tp=1.20, sl=1.08), port))
    assert _venue_levels(port) == {"1": (1.30, 1.00), "3": (1.20, 1.08)}

    # L3's leg closes at its take-profit; the fill is not booked yet and Pine
    # re-emits L3-X.
    port.set_legs(legs[:2])
    port.amended.clear()
    for _ in range(2):
        res = _run(eng.run_exit_bracket(
            _exit_env("L3-X", "L3", qty=1.0, tp=1.20, sl=1.08), port))
        assert res.skipped and res.entry_without_leg and res.legs == ()
        assert "'L3' holds no open leg" in (res.skip_reason or "")
    # L1's and L2's legs are never touched; only L3-X's own row on the closed
    # leg "3" is retired, without a venue call (a closed leg carries no bracket).
    assert _amended_levels(port) == []
    assert len([w for w in warnings if "attributed to an entry other than" in w]) == 1
    assert {row.intent_key for row in journal.rows()} == {"L1-X\0L1"}


def __test_an_entry_without_a_leg_retires_its_rows_from_legs_it_covered__(journal):
    """The skip still retires the exit from a leg it covered before, keeping co-owners' levels.

    L3-X protected the unattributed leg "9" next to L1-X; once the plugin
    attributes "9" to L1, L3 holds no leg: L3-X's row comes off "9" and the
    leg is re-amended to L1-X's levels alone.
    """
    port = _FakePort([_leg("9", "buy", 1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("L1-X", "L1", qty=1.0, tp=1.30, sl=1.00), port))
    _run(eng.run_exit_bracket(_exit_env("L3-X", "L3", qty=1.0, tp=1.20, sl=1.08), port))
    assert _venue_levels(port) == {"9": (1.20, 1.08)}
    port.set_legs([_leg("9", "buy", 1.0, entry_id="L1")])
    port.amended.clear()
    res = _run(eng.run_exit_bracket(_exit_env("L3-X", "L3", qty=1.0, tp=1.20, sl=1.08), port))
    assert res.skipped and res.entry_without_leg
    assert _amended_levels(port) == [("9", 1.30, 1.00)]
    assert {row.intent_key for row in journal.rows()} == {"L1-X\0L1"}


def __test_an_entry_without_an_attributed_leg_covers_the_unattributed_legs__(journal):
    """With unattributed legs open, an exit without a leg of its own covers just those."""
    port = _FakePort([_leg("1", "buy", 1.0, open_time=1.0, entry_id="L1"),
                      _leg("9", "buy", 1.0, open_time=2.0)])
    eng = journal.emulator()
    res = _run(eng.run_exit_bracket(_exit_env("L3-X", "L3", qty=1.0, tp=1.20, sl=1.08), port))
    assert res.legs == ("9",)
    # An exit without a from_entry covers every survivor leg.
    res = _run(eng.run_exit_bracket(_exit_env("ALL", "", qty=2.0, sl=1.00), port))
    assert res.legs == ("1", "9")


def __test_restart_replay_re_asserts_the_merged_levels_once_per_leg__(tmp_path):
    """The replay amends each leg once, with the merge of all of its owners."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20), port))
    sl_env = _exit_env("SL", "Long", sl=1.00)
    sl_env = DispatchEnvelope(intent=sl_env.intent, run_tag="t000", bar_ts_ms=2000)
    _run(eng.run_exit_bracket(sl_env, port))
    replay_port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                             _leg("11", "buy", 1.0, open_time=1.0)])
    assert _run(OneWayEmulator(store_ctx=ctx).restart_replay(replay_port)) is True
    assert sorted(_amended_levels(replay_port)) == [("10", 1.20, 1.00), ("11", 1.20, 1.00)]
    store.close()


def __test_a_retirement_left_clearing_re_asserts_the_remaining_owner__(tmp_path):
    """The drain and the replay finish an ambiguous retirement with the remaining owner's levels.

    Never all-``None`` while another exit still owns the leg, and under the
    persisted clear coid, never an attach coid.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "Long", sl=1.00), port))
    cancel = _cancel_env("TP", from_entry="Long")
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket_clear(cancel, port))
    port.fail_amend_unknown = False

    # The restart finishes it.
    port.amended.clear()
    assert _run(eng.restart_replay(port)) is True
    assert port.amended == [("10", "sell", None, 1.00, None, cancel.client_order_id(KIND_CANCEL))]
    assert [row.intent_key for row in iter_active_bracket_ownerships(ctx)] == ["SL\0Long"]

    # The first retirement's write never got a verdict, so the leg takes no
    # different write until it is settled: once its window has passed.
    _run(eng.settle_leg_writes("EURUSD", port, 0.0))
    _run(eng.settle_leg_writes("EURUSD", port, UNSETTLED_WRITE_WINDOW_S))

    # The per-sync drain does the same for a retirement left clearing in-session.
    port.amended.clear()
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20), port))
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket_clear(cancel, port))
    port.fail_amend_unknown = False
    port.amended.clear()
    assert _run(eng.drain_clearing_rows("EURUSD", port)) == {"TP\0Long"}
    assert port.amended == [("10", "sell", None, 1.00, None, cancel.client_order_id(KIND_CANCEL))]
    assert [row.intent_key for row in iter_active_bracket_ownerships(ctx)] == ["SL\0Long"]
    store.close()


def __test_a_dropped_leg_keeps_its_other_owners_levels__(journal):
    """A leg leaving one exit's covered set is handed back to the exits still owning it."""
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "Long", sl=1.00), port))
    # An opposing leg nets "10" away; the TP exit re-attaches on "11" only.
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0),
                   _leg("11", "buy", 1.0, open_time=1.0),
                   _leg("20", "sell", 1.0, open_time=2.0)])
    port.amended.clear()
    res = _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.25), port))
    assert res.legs == ("11",)
    # Leg "10" is offset whole by the opposing leg: nothing of it is net
    # exposure, so it carries no bracket at all — the SL exit's stop left on it
    # would close an offset leg and flip the net position. Leg "11" holds the
    # merge of both owners.
    assert _amended_levels(port) == [("10", None, None), ("11", 1.25, 1.00)]


# === Venue rules on shared legs: unchanged amends, refused retirements ===

class _VenueRulesPort(_FakePort):
    """A port with two real venue rules.

    A stop-loss closer than ``min_dist`` to ``mid`` is refused (Capital.com's
    minimum stop distance), and with ``reject_identical`` an amend that
    changes nothing on a protected leg is refused ("nothing to amend").
    ``rate_limited`` makes every amend raise a venue throttle.
    """

    def __init__(self, legs, *, mid=1.10, min_dist=0.0, reject_identical=False) -> None:
        super().__init__(legs)
        self.mid = mid
        self.min_dist = min_dist
        self.reject_identical = reject_identical
        self.rate_limited = False
        self.current: dict[str, tuple[float | None, float | None, float | None]] = {}
        self.calls = 0

    async def amend_bracket(self, symbol, leg_id, *, side, tp_price, sl_price,
                            trail_offset, qty, coid) -> None:
        self.calls += 1
        if self.rate_limited:
            raise ExchangeRateLimitError("too many requests", retry_after=1.0)
        levels = (tp_price, sl_price, trail_offset)
        if sl_price is not None and abs(self.mid - sl_price) < self.min_dist:
            raise ExchangeOrderRejectedError(
                f"stop distance too small (sl={sl_price} mid={self.mid})")
        if (self.reject_identical and levels != (None, None, None)
                and self.current.get(leg_id) == levels):
            raise ExchangeOrderRejectedError("INVALID_REQUEST: Nothing to amend.")
        self.current[leg_id] = levels
        await super().amend_bracket(
            symbol, leg_id, side=side, tp_price=tp_price, sl_price=sl_price,
            trail_offset=trail_offset, qty=qty, coid=coid,
        )


def _own_states(rows: Iterable[OrderRow]) -> dict[str, str]:
    return {row.intent_key: row.extras[EXTRAS_KEY_BRACKET_OWN_STATE] for row in rows}


def __test_an_attach_dominated_by_a_co_owner_sends_no_amend__(journal):
    """Attaching an exit whose levels a co-owner dominates changes nothing on the venue.

    The merge equals what the leg already holds, so no amend goes out (a
    venue refuses "nothing to amend", and a re-send resets a moved trailing
    stop); the attaching exit still owns the leg.
    """
    port = _VenueRulesPort([_leg("10", "buy", 1.0)], reject_identical=True)
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("A", "L", tp=1.15), port))
    res = _run(eng.run_exit_bracket(_exit_env("B", "L", tp=1.20), port))
    assert res.legs == ("10",) and not res.skipped
    assert _amended_levels(port) == [("10", 1.15, None)]
    assert _own_states(journal.rows()) == {"A\0L": "active", "B\0L": "active"}
    # Once A is retired, B's own level is what the leg must carry: sent.
    _run(eng.run_exit_bracket_clear(_cancel_env("A", from_entry="L"), port))
    assert _amended_levels(port) == [("10", 1.15, None), ("10", 1.20, None)]


def __test_retiring_a_dominated_owner_sends_no_amend__(journal):
    """Retiring an exit whose levels were all dominated leaves the leg as it is, row released."""
    port = _VenueRulesPort([_leg("10", "buy", 1.0)], reject_identical=True)
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("B", "L", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("A", "L", tp=1.15), port))
    port.amended.clear()
    res = _run(eng.run_exit_bracket_clear(_cancel_env("B", from_entry="L"), port))
    assert res.legs == ("10",)
    assert port.amended == []
    assert _own_states(journal.rows()) == {"A\0L": "active"}


def __test_an_unconfirmed_amend_is_never_skipped_as_unchanged__(tmp_path):
    """Only an amend the venue confirmed makes a leg's levels known; anything else re-sends.

    An ambiguous attach, then the identical re-dispatch, goes out again; the
    restart replay re-asserts even on the instance that just confirmed the
    levels.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket(_exit_env("A", "L", tp=1.20, sl=1.00), port))
    port.fail_amend_unknown = False
    _run(eng.run_exit_bracket(_exit_env("A", "L", tp=1.20, sl=1.00), port))
    assert _amended_levels(port) == [("10", 1.20, 1.00)]
    _run(eng.run_exit_bracket(_exit_env("A", "L", tp=1.20, sl=1.00), port))
    assert len(port.amended) == 1  # confirmed and unchanged: skipped
    assert _run(eng.restart_replay(port)) is True
    assert _amended_levels(port) == [("10", 1.20, 1.00), ("10", 1.20, 1.00)]
    store.close()


def __test_a_refused_retirement_stays_clearing_and_never_raises__(tmp_path, monkeypatch):
    """The venue refusing the remaining owners' levels never ends the run.

    TP-only exit at 1.20 and SL-only exit at 1.09 on one leg; the price drifts
    to 1.093, inside the venue's minimum stop distance. Cancelling the TP
    exit re-sends sl=1.09 alone, which is refused: the TP row stays
    ``clearing``, one WARNING says the retired levels may still be armed,
    and neither the clear, the per-sync drain nor a restart raises. Once
    the price moves away the drain lands the retirement.
    """
    warnings = _capture_warnings(monkeypatch)
    store, ctx = _make_store(tmp_path)
    port = _VenueRulesPort([_leg("10", "buy", 1.0)], mid=1.10, min_dist=0.005)
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "L", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "L", sl=1.09), port))
    port.mid = 1.093
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="L"), port))
    assert res.legs == ()
    assert _own_states(iter_active_bracket_ownerships(ctx)) == {"TP\0L": BRACKET_OWN_STATE_CLEARING, "SL\0L": "active"}
    refusals = [w for w in warnings if "may still be armed" in w]
    assert len(refusals) == 1
    assert "TP" in refusals[0] and "stop distance too small" in refusals[0]

    assert _run(eng.drain_clearing_rows("EURUSD", port)) == set()
    restarted = OneWayEmulator(store_ctx=ctx)
    assert _run(restarted.restart_replay(port)) is True
    assert _run(restarted.drain_clearing_rows("EURUSD", port)) == set()
    assert _own_states(iter_active_bracket_ownerships(ctx)) == {"TP\0L": BRACKET_OWN_STATE_CLEARING, "SL\0L": "active"}
    assert len([w for w in warnings if "may still be armed" in w]) == 2  # once per instance

    port.mid = 1.12
    port.amended.clear()
    assert _run(restarted.drain_clearing_rows("EURUSD", port)) == {"TP\0L"}
    assert _amended_levels(port) == [("10", None, 1.09)]
    assert _own_states(iter_active_bracket_ownerships(ctx)) == {"SL\0L": "active"}
    store.close()


def __test_a_refused_retirement_does_not_stop_the_other_legs__(journal):
    """A refusal on one leg keeps that leg clearing; the exit's other legs are still retired."""
    port = _VenueRulesPort([_leg("10", "buy", 1.0, entry_id="L1"),
                            _leg("11", "buy", 1.0, entry_id="L2")],
                           mid=1.10, min_dist=0.005)
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "", qty=2.0, tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL1", "L1", qty=1.0, sl=1.09), port))
    port.mid = 1.093
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry=""), port))
    assert res.legs == ("11",)
    assert _amended_levels(port)[-1] == ("11", None, None)
    states = {(row.intent_key, row.extras[EXTRAS_KEY_BRACKET_OWN_LEG_ID]):
              row.extras[EXTRAS_KEY_BRACKET_OWN_STATE]
              for row in journal.rows()}
    assert states == {("TP\0", "10"): BRACKET_OWN_STATE_CLEARING,
                      ("SL1\0L1", "10"): "active"}


def __test_a_throttled_retirement_stays_clearing_and_never_raises__(tmp_path):
    """A venue throttle on a retirement stops the loop, keeps the rows clearing, and retries."""
    store, ctx = _make_store(tmp_path)
    port = _VenueRulesPort([_leg("10", "buy", 1.0), _leg("11", "buy", 1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "L", tp=1.20), port))
    port.rate_limited = True
    port.calls = 0
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="L"), port))
    assert res.legs == ()
    assert port.calls == 1  # the second leg is not hammered
    assert set(_own_states(iter_active_bracket_ownerships(ctx)).values()) == {BRACKET_OWN_STATE_CLEARING}
    assert _run(eng.drain_clearing_rows("EURUSD", port)) == set()
    assert _run(OneWayEmulator(store_ctx=ctx).restart_replay(port)) is True
    port.rate_limited = False
    assert _run(eng.drain_clearing_rows("EURUSD", port)) == {"TP\0L"}
    assert list(iter_active_bracket_ownerships(ctx)) == []
    store.close()


def __test_a_refused_retirement_of_a_dropped_leg_keeps_the_survivor_amend__(journal):
    """A refused retirement on a dropped leg stays clearing and never fails the exit's own fan.

    The retirement concerns a leg the exit no longer covers; raising it as
    the exit's refusal would send the engine into the refused-exit recovery
    of a bracket the venue accepted on every leg it does cover.
    """
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20), port))
    # An opposing leg nets "10" away; the venue refuses any amend on "10".
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0),
                   _leg("11", "buy", 1.0, open_time=1.0),
                   _leg("20", "sell", 1.0, open_time=2.0)])
    port._fail_amend_leg = "10"
    port.amended.clear()
    res = _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.25), port))
    assert res.legs == ("11",) and not res.skipped
    assert _amended_levels(port) == [("11", 1.25, None)]
    states = {row.extras[EXTRAS_KEY_BRACKET_OWN_LEG_ID]: row.extras[EXTRAS_KEY_BRACKET_OWN_STATE]
              for row in journal.rows()}
    assert states == {"10": BRACKET_OWN_STATE_CLEARING, "11": "active"}


def __test_a_co_owned_attach_refusal_is_not_an_after_fill_flatten__(journal, monkeypatch):
    """A refused attach on a leg another exit protects is the exit's refusal, not a flatten.

    The SL exit's 1.09 sits on the leg; the price drifts to 1.093 and the TP
    exit attaches: the venue refuses the merge (its stop rule). The leg still
    holds SL's level, so the engine must get the venue's plain reject (the
    refused-exit recovery: retry, then the defensive close) instead of
    :class:`BracketAttachAfterFillRejectedError` (an immediate flatten). The
    TP exit's row is released; SL's stays.
    """
    warnings = _capture_warnings(monkeypatch)
    port = _VenueRulesPort([_leg("10", "buy", 1.0)], mid=1.10, min_dist=0.005)
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("SL", "L", sl=1.09), port))
    port.mid = 1.093
    with pytest.raises(ExchangeOrderRejectedError) as raised:
        _run(eng.run_exit_bracket(_exit_env("TP", "L", tp=1.20), port))
    assert not isinstance(raised.value, BracketAttachAfterFillRejectedError)
    assert "stop distance too small" in str(raised.value)
    assert _own_states(journal.rows()) == {"SL\0L": "active"}
    assert len([w for w in warnings if "other exits also protect" in w]) == 1


def __test_a_co_owned_modify_refusal_restores_the_previous_row__(journal):
    """A refused modify on a co-owned leg keeps the exit's previous levels as its row.

    The venue still holds the merge with the exit's previous levels, so the
    row must keep them (a later retirement must take them off the leg).
    """
    port = _VenueRulesPort([_leg("10", "buy", 1.0)], mid=1.10, min_dist=0.005)
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "L", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "L", sl=1.09), port))
    port.mid = 1.093
    with pytest.raises(ExchangeOrderRejectedError) as raised:
        _run(eng.run_exit_bracket(_exit_env("TP", "L", tp=1.25), port))
    assert not isinstance(raised.value, BracketAttachAfterFillRejectedError)
    rows = {row.intent_key: row for row in journal.rows()}
    assert rows["TP\0L"].extras[EXTRAS_KEY_BRACKET_OWN_STATE] == "active"
    assert rows["TP\0L"].extras[EXTRAS_KEY_BRACKET_OWN_TP] == 1.20


def __test_a_sole_owner_attach_refusal_keeps_the_after_fill_flatten__(journal):
    """With no co-owner the refused attach still means an unprotected position."""
    port = _VenueRulesPort([_leg("10", "buy", 1.0)], mid=1.093, min_dist=0.005)
    eng = journal.emulator()
    with pytest.raises(BracketAttachAfterFillRejectedError):
        _run(eng.run_exit_bracket(_exit_env("SL", "L", sl=1.09), port))
    assert list(journal.rows()) == []


def __test_a_stop_and_a_trail_of_different_exits_are_reported_once__(journal, monkeypatch):
    """A fixed stop of one exit and a trail of another cannot be carried as asked; WARN once."""
    warnings = _capture_warnings(monkeypatch)
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = journal.emulator(mintick=0.01)
    _run(eng.run_exit_bracket(_exit_env("ST", "L", sl=1.00), port))
    _run(eng.run_exit_bracket(_exit_env("TR", "L", trail=10.0), port))
    _run(eng.run_exit_bracket(_exit_env("TR", "L", trail=10.0), port))
    leg_id, _side, tp, sl, trail, _coid = port.amended[-1]
    assert (leg_id, tp, sl, trail) == ("10", None, 1.00, 0.1)
    assert len([w for w in warnings if "fixed stop-loss and a trailing stop" in w]) == 1


def __test_one_exit_asking_for_a_stop_and_a_trail_is_not_reported__(journal, monkeypatch):
    warnings = _capture_warnings(monkeypatch)
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = journal.emulator(mintick=0.01)
    _run(eng.run_exit_bracket(_exit_env("ST", "L", sl=1.00, trail=10.0), port))
    _run(eng.run_exit_bracket(_exit_env("TP", "L", tp=1.20), port))
    assert [w for w in warnings if "fixed stop-loss and a trailing stop" in w] == []


# === Reversal residual-open replay (2.7) ================================

def _seed_residual(ctx, *, entry_coid, qty=2.0, side="buy"):
    create_residual_open_row(
        ctx, coid="rev:residual", symbol="EURUSD", side=side, qty=qty,
        intent_key="Long", pine_entry_id="Long", entry_coid=entry_coid,
        run_tag="t000", bar_ts_ms=1000, retry_seq=0,
    )


def __test_reversal_creates_then_clears_residual_breadcrumb__(tmp_path):
    """A real reversal writes the residual breadcrumb before closing and discharges it on open."""
    # A real reversal (close the opposing short, open the residual long) writes
    # the persist-first breadcrumb before the closes and discharges it once
    # place_leg has persisted the residual entry row.
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("20", "sell", 1.0, open_time=0.0)])
    port.closes_execute = True
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert _dispatched(port) == [("20", 1)]
    assert port.placed == [("EURUSD", 2.0)]
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_reversal_pure_reduce_writes_no_breadcrumb__(tmp_path):
    """A pure reduce (smaller than opposing exposure) opens nothing and writes no breadcrumb."""
    # An order smaller than the opposing exposure is a pure reduce: no residual
    # opens, so no breadcrumb is ever persisted.
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("20", "sell", 3.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_reversal(_entry_env("buy", 2.0), port))
    assert port.placed == []
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_pure_add_writes_no_breadcrumb__(tmp_path):
    """A pure add opens the whole size but persists no residual breadcrumb row."""
    # A pure add (no opposing legs, no close owed) opens the whole size but must
    # NOT persist a residual breadcrumb: re-opening it from a crash would bypass
    # the next sync's Pine re-evaluation and resurrect an unwanted entry. Only a
    # genuine reversal (closes actually dispatched) earns a breadcrumb. Asserting
    # on the breadcrumb ROW (not just the live iterator) catches the prior bug
    # where a pure add created-then-cleared a residual row in the happy path.
    store, ctx = _make_store(tmp_path)
    env = _entry_env("buy", 2.0)
    residual_coid = f"{env.client_order_id(KIND_CLOSE)}:residual"
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_reversal(env, port))
    assert port.placed == [("EURUSD", 2.0)]
    assert list(iter_active_residual_opens(ctx)) == []
    assert ctx.get_order(residual_coid) is None
    store.close()


def __test_restart_replay_redispatches_residual_when_entry_absent__(tmp_path):
    """Restart replay re-opens the residual when its entry row never persisted."""
    # Crash after the closes landed but before place_leg persisted the residual
    # entry row: the breadcrumb survives and replay re-opens the residual.
    store, ctx = _make_store(tmp_path)
    _seed_residual(ctx, entry_coid="entry-missing", qty=2.0)
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))
    assert port.placed == [("EURUSD", 2.0)]
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_replay_withholds_residual_reopen_under_quarantine__(tmp_path):
    """Under quarantine the residual reopen is withheld (new exposure) while
    the owed FIFO closes still run (risk-reducing), and the breadcrumb stays
    live so the committed reversal finishes once the operator restart lifts
    the quarantine."""
    store, ctx = _make_store(tmp_path)
    # The breadcrumb carries the combined order: close the short 1, open 2.
    _seed_residual(ctx, entry_coid="entry-missing", qty=3.0)
    # A still-live opposing leg: the crash hit before the FIFO closes landed.
    port = _FakePort([_leg("20", "sell", 1.0, open_time=0.0)])
    quarantined = True
    eng = OneWayEmulator(store_ctx=ctx,
                         block_exposure_reopens=lambda: quarantined)
    _run(eng.restart_replay(port))
    assert _dispatched(port) == [("20", 1)]  # owed close still ran
    assert port.placed == []                 # reopen withheld
    assert len(list(iter_active_residual_opens(ctx))) == 1  # breadcrumb live
    # The quarantine lifted (operator restart): the drain finishes the reopen.
    quarantined = False
    port2 = _FakePort([])
    _run(eng.drain_residual_opens(port2))
    assert port2.placed == [("EURUSD", 2.0)]
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_restart_replay_skips_residual_when_entry_row_exists__(tmp_path):
    """Restart replay only clears the breadcrumb when the residual entry row already exists."""
    # The residual entry row already exists (place_leg's persist-first write
    # landed): the entry path owns it, so replay must NOT re-open — only clear.
    store, ctx = _make_store(tmp_path)
    ctx.upsert_order("entry-present", symbol="EURUSD", side="buy", qty=2.0,
                     state="confirmed", pine_entry_id="Long")
    _seed_residual(ctx, entry_coid="entry-present", qty=2.0)
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))
    assert port.placed == []
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_stop_fired_reversal_breadcrumb_anchors_on_entry_stop_coid__(tmp_path):
    """A stop-fired reversal's breadcrumb entry coid uses KIND_ENTRY_TRIGGERED, so replay finds the row."""
    # A stop-fired market reversal dispatches its residual under KIND_ENTRY_TRIGGERED
    # (the plugin picks the kind from triggered_entry). The breadcrumb's
    # entry_coid must anchor on that SAME coid: anchoring on KIND_ENTRY would
    # make the replay's row-existence check miss the landed entry row and
    # double-open the residual.
    store, ctx = _make_store(tmp_path)
    env = _entry_env("buy", 3.0, stop_fired=True)
    port = _FakePort(
        [_leg("20", "sell", 2.0, open_time=0.0)],
        fail_place_leg=OrderDispositionUnknownError(
            "residual timed out", client_order_id="residual-coid",
        ),
    )
    port.closes_execute = True
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_reversal(env, port))
    rows = list(iter_active_residual_opens(ctx))
    assert len(rows) == 1
    stop_coid = env.client_order_id(KIND_ENTRY_TRIGGERED)
    assert (rows[0].extras or {})[EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID] == stop_coid
    # The ambiguous send actually landed: the plugin's persist-first entry row
    # exists under the KIND_ENTRY_TRIGGERED coid. Replay must find it and discharge
    # the breadcrumb WITHOUT re-opening the residual.
    ctx.upsert_order(stop_coid, symbol="EURUSD", side="buy", qty=1.0,
                     state="confirmed", pine_entry_id="Long")
    replay_port = _FakePort([])
    _run(eng.restart_replay(replay_port))
    assert replay_port.placed == []
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_replay_restores_stop_fired_flag_on_residual_redispatch__(tmp_path):
    """Residual replay re-dispatches a stop-fired reversal with triggered_entry restored."""
    # The re-dispatched place_leg must persist and dedup under the SAME
    # KIND_ENTRY_TRIGGERED coid as the original attempt, so the rebuilt intent needs
    # the triggered_entry flag back — recovered from the persisted coid's
    # kind code.
    class _CapturingPort(_FakePort):
        def __init__(self, legs):
            super().__init__(legs)
            self.place_envelopes: list[DispatchEnvelope] = []

        async def place_leg(self, envelope, qty):
            self.place_envelopes.append(envelope)
            return await super().place_leg(envelope, qty)

    store, ctx = _make_store(tmp_path)
    entry_coid = _entry_env("buy", 2.0).client_order_id(KIND_ENTRY_TRIGGERED)
    _seed_residual(ctx, entry_coid=entry_coid, qty=2.0)
    port = _CapturingPort([])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))
    assert port.placed == [("EURUSD", 2.0)]
    intent = port.place_envelopes[0].intent
    assert isinstance(intent, EntryIntent) and intent.triggered_entry is True
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_reversal_clears_breadcrumb_on_definitive_entry_reject__(tmp_path):
    """A definitively-rejected residual entry discharges the breadcrumb after the closes land."""
    # The residual place_leg is definitively rejected by the exchange after the
    # closes already landed: the dispatch path turns this into a non-halting skip,
    # so the breadcrumb must be discharged here — leaving it live would let a
    # later restart re-open a residual the exchange rejected and Pine never
    # re-signalled.
    store, ctx = _make_store(tmp_path)
    env = _entry_env("buy", 3.0)
    residual_coid = f"{env.client_order_id(KIND_CLOSE)}:residual"
    port = _FakePort(
        [_leg("20", "sell", 2.0, open_time=0.0)],
        fail_place_leg=ExchangeOrderRejectedError("insufficient margin"),
    )
    port.closes_execute = True
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(ExchangeOrderRejectedError):
        _run(eng.run_reversal(env, port))
    # Closes still landed; the breadcrumb is discharged (closed) so no stale
    # residual replays after restart.
    assert _dispatched(port) == [("20", 2)]
    assert list(iter_active_residual_opens(ctx)) == []
    breadcrumb = ctx.get_order(residual_coid)
    assert breadcrumb is not None and breadcrumb.closed_ts_ms is not None
    store.close()


def __test_reversal_keeps_breadcrumb_on_ambiguous_entry_failure__(tmp_path):
    """An ambiguous residual entry failure keeps the breadcrumb live for replay to reconcile."""
    # An ambiguous place_leg failure (disposition unknown) leaves the residual's
    # fate unknown: the breadcrumb MUST survive so restart_replay can reconcile
    # and re-dispatch / clear it. A definitive-reject clear must NOT swallow it.
    store, ctx = _make_store(tmp_path)
    env = _entry_env("buy", 3.0)
    port = _FakePort(
        [_leg("20", "sell", 2.0, open_time=0.0)],
        fail_place_leg=OrderDispositionUnknownError(
            "residual timed out", client_order_id="residual-coid",
        ),
    )
    port.closes_execute = True
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_reversal(env, port))
    assert len(list(iter_active_residual_opens(ctx))) == 1
    store.close()


def _live_residuals(eng: OneWayEmulator) -> list[OrderRow]:
    """The live residual-open breadcrumbs, from the store or the in-memory journal."""
    return list(iter_active_residual_opens(eng._close_journal))  # type: ignore[attr-defined]


def __test_a_reversal_opens_its_residual_only_once_the_opposing_leg_is_gone__(journal):
    """A dispatched close that the read still lists never has the residual opened beside it.

    ``close_leg`` only dispatches: a reduce-only IOC close can fill in part
    and be cancelled. The residual opens only on a pass whose fresh read shows
    no opposing leg; meanwhile the unexecuted part of the close is re-sent from
    the venue's evidence, in-session, with or without a store.
    """
    env = _entry_env("buy", 3.0)
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = journal.emulator()
    res = _run(eng.run_reversal(env, port))
    assert _dispatched(port) == [("20", 2)]
    # The read right after the close still lists the short leg: no open.
    assert port.placed == []
    assert res.open_qty == 1.0 and res.opened_orders == ()
    assert len(_live_residuals(eng)) == 1
    # The close was an IOC that filled 1 of 2 and was cancelled.
    leg_coid = close_leg_client_order_id(env.client_order_id(KIND_CLOSE), "20")
    port.executed_by_coid[leg_coid] = 1
    port.set_legs([_leg("20", "sell", 1.0, open_time=0.0)])
    _run(eng.drain_residual_opens(port))
    assert eng.take_opened_residuals() == []
    # Only the unexecuted part is re-sent, under a new attempt coid; still
    # no residual beside the remaining short.
    assert _dispatched(port) == [("20", 2), ("20", 1)]
    assert port.closed[-1][3] == close_leg_attempt_client_order_id(leg_coid, 1)
    assert port.placed == []
    assert len(_live_residuals(eng)) == 1
    # The book shows the short gone: the residual opens exactly once and the
    # drain hands its orders back for the reversal's entry intent.
    port.set_legs([])
    _run(eng.drain_residual_opens(port))
    assert eng.take_opened_residuals() == [(env.intent.intent_key, ("order",))]
    assert port.placed == [("EURUSD", 1.0)]
    assert _live_residuals(eng) == []
    _run(eng.drain_residual_opens(port))
    assert eng.take_opened_residuals() == []
    assert port.placed == [("EURUSD", 1.0)]


def __test_a_deferred_residual_waits_while_the_close_outcome_is_unknown__(journal):
    """While the venue cannot say what the close executed, nothing is re-sent or opened."""
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = journal.emulator()
    _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    port.executed_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.drain_residual_opens(port))
    assert _dispatched(port) == [("20", 2)]
    assert port.placed == []
    assert len(_live_residuals(eng)) == 1


def __test_a_drained_residual_opens_under_the_reversals_own_envelope__():
    """In the same process the drain opens the residual with the reversal's own envelope."""
    class _CapturingPort(_FakePort):
        def __init__(self, legs):
            super().__init__(legs)
            self.place_envelopes: list[DispatchEnvelope] = []

        async def place_leg(self, envelope, qty):
            self.place_envelopes.append(envelope)
            return await super().place_leg(envelope, qty)

    env = _entry_env("buy", 3.0, stop_fired=True)
    port = _CapturingPort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    _run(eng.run_reversal(env, port))
    port.set_legs([])
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 1.0)]
    assert port.place_envelopes == [env]


def __test_an_ambiguous_residual_open_without_a_store_is_never_reopened__():
    """Without a store, a residual open of unknown outcome is discharged, not sent again.

    The store's persist-first entry row proves an open went out; on a run
    without one, the emulator records the entry coid before ``place_leg``, so
    the drain discharges the breadcrumb instead of opening a second residual.
    """
    port = _FakePort(
        [_leg("20", "sell", 2.0, open_time=0.0)],
        fail_place_leg=OrderDispositionUnknownError(
            "residual timed out", client_order_id="residual-coid",
        ),
    )
    port.closes_execute = True
    eng = OneWayEmulator(store_ctx=None)
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert len(_live_residuals(eng)) == 1
    port._fail_place_leg = None
    _run(eng.drain_residual_opens(port))
    assert eng.take_opened_residuals() == []
    assert port.placed == []
    assert _live_residuals(eng) == []


def __test_overlapping_drains_never_open_one_residual_twice__():
    """A drain still running when the next starts cannot open the same residual again."""
    class _SlowReadPort(_FakePort):
        def __init__(self, legs):
            super().__init__(legs)
            self.gate: asyncio.Event | None = None

        async def fetch_raw_positions(self, symbol):
            if self.gate is not None:
                await self.gate.wait()
            return await super().fetch_raw_positions(symbol)

    port = _SlowReadPort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    port.set_legs([])

    async def overlap():
        port.gate = asyncio.Event()
        first = asyncio.ensure_future(eng.drain_residual_opens(port))
        await asyncio.sleep(0)
        second = asyncio.ensure_future(eng.drain_residual_opens(port))
        await asyncio.sleep(0)
        port.gate.set()
        await asyncio.gather(first, second)

    _run(overlap())
    assert port.placed == [("EURUSD", 1.0)]
    assert _live_residuals(eng) == []


def __test_an_unsent_residual_open_without_a_store_is_retried__():
    """A residual open whose request never left the client stays owed and opens next sync."""
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    port.set_legs([])
    port._fail_place_leg = ExchangeConnectionError("link down")
    _run(eng.drain_residual_opens(port))
    assert port.placed == []
    assert len(_live_residuals(eng)) == 1
    port._fail_place_leg = None
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 1.0)]
    assert _live_residuals(eng) == []


class _RowLeavingPort(_FakePort):
    """A port whose ``place_leg`` persists its entry row before it fails pre-send.

    Models a plugin's persist-first ``submitted`` row left behind by an
    :class:`ExchangeConnectionError` (the request never left the client).
    """

    def __init__(self, legs, ctx: RunContext | None) -> None:
        super().__init__(legs)
        self._ctx = ctx

    async def place_leg(self, envelope, qty):
        if self._fail_place_leg is not None and self._ctx is not None:
            intent = envelope.intent
            create_entry_order_row(
                self._ctx, coid=envelope.client_order_id(KIND_ENTRY), symbol=intent.symbol,
                side=intent.side, qty=qty, intent_key=intent.intent_key,
                pine_entry_id=intent.pine_id, kind=ENTRY_KIND_POSITION, order_type="market",
            )
        return await super().place_leg(envelope, qty)


def __test_an_unsent_in_session_residual_open_is_owned_by_the_breadcrumb__(journal):
    """An in-session residual open that never left the client opens its residual next sync.

    The closes already reduced the book: raising would let the dispatch path
    re-plan the combined entry from the reduced book and open its whole size.
    The breadcrumb keeps the residual's size, and the drain opens exactly it,
    also where the plugin's ``submitted`` entry row was left behind.
    """
    port = _RowLeavingPort([_leg("20", "sell", 2.0, open_time=0.0)], journal._ctx)
    port._fail_place_leg = ExchangeConnectionError("link down")
    port.closes_execute = True
    eng = journal.emulator()
    env = _entry_env("buy", 3.0)
    res = _run(eng.run_reversal(env, port))
    assert res.opened_orders == () and res.open_qty == 1.0
    assert port.placed == []
    assert len(_live_residuals(eng)) == 1
    port._fail_place_leg = None
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 1.0)]
    assert eng.take_opened_residuals() == [(env.intent.intent_key, ("order",))]
    assert _live_residuals(eng) == []


def __test_an_unsent_read_after_the_closes_leaves_the_residual_to_the_drain__(journal):
    """A read fault after the closes went out is absorbed; the drain opens the residual."""
    class _ReadFailsAfterClosePort(_FakePort):
        def __init__(self, legs):
            super().__init__(legs)
            self.read_ok = False

        async def fetch_raw_positions(self, symbol):
            if self.closed and not self.read_ok:
                raise ConnectionError("read dropped")
            return await super().fetch_raw_positions(symbol)

    port = _ReadFailsAfterClosePort([_leg("20", "sell", 2.0, open_time=0.0)])
    port.closes_execute = True
    eng = journal.emulator()
    res = _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert res.opened_orders == () and port.placed == []
    assert len(_live_residuals(eng)) == 1
    port.read_ok = True
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 1.0)]


def __test_a_reversal_cut_short_after_a_close_went_out_is_finished_by_the_drain__(journal):
    """A close that never left the client after another went out leaves the reversal committed.

    The dispatch path would re-plan the combined three-unit buy from the book
    the first close reduced (one short left) and open two; the drain closes
    the remaining short and opens the one-unit residual instead.
    """
    port = _FakePort([_leg("20", "sell", 1.0, open_time=0.0),
                      _leg("21", "sell", 1.0, open_time=1.0)],
                     fail_close_leg=ExchangeConnectionError("link down"),
                     fail_close_leg_id="21")
    port.closes_execute = True
    eng = journal.emulator()
    res = _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert res.opened_orders == () and res.closes == (("20", 1),)
    assert len(_live_residuals(eng)) == 1
    # The close that never went out leaves no pending row behind.
    assert list(iter_active_close_legs(eng._close_journal)) == []  # type: ignore[attr-defined]
    port._fail_close_leg = None
    _run(eng.drain_residual_opens(port))
    assert _dispatched(port) == [("20", 1), ("21", 1)]
    assert port.placed == []
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 1.0)]
    assert _live_residuals(eng) == []


def __test_a_flatten_cut_short_after_a_close_went_out_never_becomes_a_reversal__(journal):
    """An exact flatten whose second close never left the client only finishes its closes."""
    port = _FakePort([_leg("20", "sell", 1.0, open_time=0.0),
                      _leg("21", "sell", 1.0, open_time=1.0)],
                     fail_close_leg=ExchangeConnectionError("link down"),
                     fail_close_leg_id="21")
    port.closes_execute = True
    eng = journal.emulator()
    res = _run(eng.run_reversal(_entry_env("buy", 2.0), port))
    assert res.opened_orders == () and res.open_qty == 0.0
    port._fail_close_leg = None
    _run(eng.drain_residual_opens(port))
    _run(eng.drain_residual_opens(port))
    assert _dispatched(port) == [("20", 1), ("21", 1)]
    assert port.placed == []
    assert _live_residuals(eng) == []


def __test_a_restart_still_opens_a_residual_whose_open_never_left_the_client__(tmp_path):
    """The unsent outcome of a residual open survives a restart.

    The plugin's ``place_leg`` left its persist-first ``submitted`` entry row
    behind before the request failed to leave the client. A restarted
    emulator must not read that row as an open that went out.
    """
    store, ctx = _make_store(tmp_path)
    port = _RowLeavingPort([_leg("20", "sell", 2.0, open_time=0.0)], ctx)
    port._fail_place_leg = ExchangeConnectionError("link down")
    port.closes_execute = True
    res = _run(OneWayEmulator(store_ctx=ctx).run_reversal(_entry_env("buy", 3.0), port))
    assert res.opened_orders == () and port.placed == []
    port._fail_place_leg = None
    restarted = OneWayEmulator(store_ctx=ctx)
    assert _run(restarted.restart_replay(port)) is True
    assert port.placed == [("EURUSD", 1.0)]
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_a_restart_during_a_residual_retry_still_opens_the_residual__(tmp_path):
    """A process stop right before the retry's ``place_leg`` never retires the owed open.

    The first attempt's ``place_leg`` left its ``submitted`` entry row
    behind and never left the client. The process then stops inside the
    retry, before the retry's own ``place_leg`` persisted anything: the
    stale row of the first attempt is no evidence of the retry, so the
    restarted emulator opens the residual, under the next attempt's coid.
    """
    class _ProcessStop(BaseException):
        pass

    class _StopBeforePlacePort(_RowLeavingPort):
        def __init__(self, legs, ctx):
            super().__init__(legs, ctx)
            self.stop = False
            self.place_envelopes: list[DispatchEnvelope] = []

        async def place_leg(self, envelope, qty):
            if self.stop:
                raise _ProcessStop("process stopped")
            self.place_envelopes.append(envelope)
            return await super().place_leg(envelope, qty)

    store, ctx = _make_store(tmp_path)
    port = _StopBeforePlacePort([_leg("20", "sell", 2.0, open_time=0.0)], ctx)
    port._fail_place_leg = ExchangeConnectionError("link down")
    port.closes_execute = True
    env = _entry_env("buy", 3.0)
    eng = OneWayEmulator(store_ctx=ctx)
    res = _run(eng.run_reversal(env, port))
    assert res.opened_orders == () and port.placed == []
    port._fail_place_leg = None
    port.stop = True
    with pytest.raises(_ProcessStop):
        _run(eng.drain_residual_opens(port))
    port.stop = False
    restarted = OneWayEmulator(store_ctx=ctx)
    assert _run(restarted.restart_replay(port)) is True
    assert port.placed == [("EURUSD", 1.0)]
    retry = port.place_envelopes[-1]
    # The re-attempt takes the first value of the dispatch's own retry block.
    assert retry.retry_seq == (env.retry_seq + 1) * 36 ** 3
    assert retry.client_order_id(KIND_ENTRY) != env.client_order_id(KIND_ENTRY)
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_an_engine_bump_of_the_entry_never_reads_as_the_residual_open__(tmp_path):
    """An engine dispatch under the entry's bumped id is no evidence of the residual's re-attempt.

    The residual open's first attempt never left the client and left its
    entry row behind. On the same bar the engine spends the entry's id and
    dispatches it again under ``retry_seq + 1``, leaving an entry row under
    that id. The breadcrumb's next attempt must not share that id: the drain
    still opens the residual.
    """
    store, ctx = _make_store(tmp_path)
    port = _RowLeavingPort([_leg("20", "sell", 2.0, open_time=0.0)], ctx)
    port._fail_place_leg = ExchangeConnectionError("link down")
    port.closes_execute = True
    env = _entry_env("buy", 3.0)
    eng = OneWayEmulator(store_ctx=ctx)
    res = _run(eng.run_reversal(env, port))
    assert res.opened_orders == () and port.placed == []
    bumped = replace(env, retry_seq=env.retry_seq + 1)
    intent = env.intent
    create_entry_order_row(
        ctx, coid=bumped.client_order_id(KIND_ENTRY), symbol=intent.symbol,
        side=intent.side, qty=intent.qty, intent_key=intent.intent_key,
        pine_entry_id=intent.pine_id, kind=ENTRY_KIND_POSITION, order_type="market",
    )
    port._fail_place_leg = None
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 1.0)]
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_a_reduction_cut_short_after_a_close_went_out_closes_only_its_plan__(journal):
    """A reduction whose second close never left the client never flattens the opposing side.

    A buy of 2 against shorts of 1, 1 and 3 reduces the two oldest. The drain
    finishes that plan and leaves the third short open.
    """
    port = _FakePort([_leg("20", "sell", 1.0, open_time=0.0),
                      _leg("21", "sell", 1.0, open_time=1.0),
                      _leg("22", "sell", 3.0, open_time=2.0)],
                     fail_close_leg=ExchangeConnectionError("link down"),
                     fail_close_leg_id="21")
    port.closes_execute = True
    eng = journal.emulator()
    res = _run(eng.run_reversal(_entry_env("buy", 2.0), port))
    assert res.opened_orders == () and res.open_qty == 0.0
    port._fail_close_leg = None
    _run(eng.drain_residual_opens(port))
    _run(eng.drain_residual_opens(port))
    assert _dispatched(port) == [("20", 1), ("21", 1)]
    assert port.placed == []
    assert _live_residuals(eng) == []


def __test_a_partial_slice_cut_short_is_discharged_once_the_venue_settles_it__(journal):
    """A reduction slicing a leg leaves the rest of that leg open and its breadcrumb discharged."""
    port = _FakePort([_leg("20", "sell", 1.0, open_time=0.0),
                      _leg("21", "sell", 3.0, open_time=1.0)],
                     fail_close_leg=ExchangeConnectionError("link down"),
                     fail_close_leg_id="21")
    port.closes_execute = True
    eng = journal.emulator()
    _run(eng.run_reversal(_entry_env("buy", 2.0), port))
    port._fail_close_leg = None
    _run(eng.drain_residual_opens(port))
    assert _dispatched(port) == [("20", 1), ("21", 1)]
    _run(eng.drain_residual_opens(port))
    _run(eng.drain_residual_opens(port))
    assert _dispatched(port) == [("20", 1), ("21", 1)]
    assert _live_residuals(eng) == []


def __test_a_close_the_read_does_not_show_yet_is_not_spent_again__(journal):
    """Consecutive closes never reuse net exposure an earlier close was sent against.

    Long 3 against short 1 is net long 2. The first close of 2 is
    dispatched, but the read still lists the long in full and the venue
    cannot settle the send yet: the second close of 2 must not close more
    of the long, or both executing would flip the book short.
    """
    port = _FakePort([_leg("10", "buy", 3.0, open_time=0.0),
                      _leg("20", "sell", 1.0, open_time=1.0)])
    eng = journal.emulator()
    _run(eng.run_close(_close_env("sell", 2.0), port))
    assert _dispatched(port) == [("10", 2)]
    port.executed_unknown = True
    second = _run(eng.run_close(_close_env("sell", 2.0, pine_id="Other"), port))
    assert _dispatched(port) == [("10", 2)]
    assert second.dispatched_qty == 0.0 and second.legs == ()
    # Once the venue's record settles the first send as executed, the read
    # lagging behind it still does not free the exposure.
    port.executed_unknown = False
    port.executed_by_coid[port.closed[0][3]] = 2
    _run(eng.run_close(_close_env("sell", 2.0, pine_id="Third"), port))
    assert _dispatched(port) == [("10", 2)]
    # The read catches up: the reservation is gone and the book is flat net.
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0), _leg("20", "sell", 1.0, open_time=1.0)])
    _run(eng.run_close(_close_env("sell", 1.0, pine_id="Fourth"), port))
    assert _dispatched(port) == [("10", 2)]


def __test_one_shrink_never_frees_two_closes_of_the_same_leg__(journal):
    """A leg losing one unit settles one of two one-unit closes sent against it, not both.

    Long 3 against short 1 is net long 2. Two one-unit closes go out on the
    long, both recording its size as 3, and the venue cannot settle either.
    The read then shows the long at 2: that unit is the first close's, the
    second is still outstanding, so the book is already flat net and a
    third close must send nothing.
    """
    port = _FakePort([_leg("10", "buy", 3.0, open_time=0.0),
                      _leg("20", "sell", 1.0, open_time=1.0)])
    port.executed_unknown = True
    eng = journal.emulator()
    _run(eng.run_close(_close_env("sell", 1.0), port))
    _run(eng.run_close(_close_env("sell", 1.0, pine_id="Other"), port))
    assert _dispatched(port) == [("10", 1), ("10", 1)]
    port.set_legs([_leg("10", "buy", 2.0, open_time=0.0), _leg("20", "sell", 1.0, open_time=1.0)])
    _run(eng.run_close(_close_env("sell", 1.0, pine_id="Third"), port))
    assert _dispatched(port) == [("10", 1), ("10", 1)]
    # The same shrink is not credited again on the next view either.
    _run(eng.run_close(_close_env("sell", 1.0, pine_id="Fourth"), port))
    assert _dispatched(port) == [("10", 1), ("10", 1)]
    # The second close shows up as well: the book is flat net and stays so.
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0), _leg("20", "sell", 1.0, open_time=1.0)])
    _run(eng.run_close(_close_env("sell", 1.0, pine_id="Fifth"), port))
    assert _dispatched(port) == [("10", 1), ("10", 1)]


def __test_a_restart_still_holds_back_a_close_the_read_does_not_show_yet__(tmp_path):
    """A close sent before a restart still holds its exposure back after it.

    Long 3 against short 1 is net long 2. The close of 2 is dispatched and
    its row finished, the process restarts, and the read still lists the
    long in full while the venue cannot settle the send: a second close of 2
    must send nothing, or both executing would flip the book short.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 3.0, open_time=0.0),
                      _leg("20", "sell", 1.0, open_time=1.0)])
    _run(OneWayEmulator(store_ctx=ctx).run_close(_close_env("sell", 2.0), port))
    assert _dispatched(port) == [("10", 2)]
    port.executed_unknown = True
    restarted = OneWayEmulator(store_ctx=ctx)
    assert _run(restarted.restart_replay(port)) is True
    second = _run(restarted.run_close(_close_env("sell", 2.0, pine_id="Other"), port))
    assert _dispatched(port) == [("10", 2)]
    assert second.dispatched_qty == 0.0 and second.legs == ()
    # The read catches up: the reservation ends and its row is closed.
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0), _leg("20", "sell", 1.0, open_time=1.0)])
    _run(restarted.run_close(_close_env("sell", 1.0, pine_id="Third"), port))
    assert _dispatched(port) == [("10", 2)]
    row = ctx.get_order(port.closed[0][3])
    assert row is not None and row.closed_ts_ms is not None
    store.close()


def __test_a_handed_down_shrink_is_not_credited_again_after_a_restart__(tmp_path):
    """The shrink one close was credited with stays its own across a restart.

    Two one-unit closes go out on a long of 3 against a short of 1. The read
    shows the long at 2: the first close is settled and the second keeps
    holding its unit. A restarted emulator must not credit that same unit
    to the second close, so a further close sends nothing.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 3.0, open_time=0.0),
                      _leg("20", "sell", 1.0, open_time=1.0)])
    port.executed_unknown = True
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_close(_close_env("sell", 1.0), port))
    _run(eng.run_close(_close_env("sell", 1.0, pine_id="Other"), port))
    port.set_legs([_leg("10", "buy", 2.0, open_time=0.0), _leg("20", "sell", 1.0, open_time=1.0)])
    _run(eng.run_close(_close_env("sell", 1.0, pine_id="Third"), port))
    assert _dispatched(port) == [("10", 1), ("10", 1)]
    restarted = OneWayEmulator(store_ctx=ctx)
    _run(restarted.run_close(_close_env("sell", 1.0, pine_id="Fourth"), port))
    assert _dispatched(port) == [("10", 1), ("10", 1)]
    store.close()


def __test_a_stop_inside_the_credit_hand_down_never_frees_one_shrink_twice__(tmp_path, monkeypatch):
    """A process stop between a reservation's release and its hand-down leaves both undone.

    Two one-unit closes go out on a long of 3 against a short of 1, and the
    read then shows the long at 2. The view settles the first close and hands
    its unit down to the second; the process stops in between. A restarted
    emulator must still hold the second close, so a further close sends
    nothing.
    """
    class _ProcessStop(BaseException):
        pass

    def _stop(*_args, **_kwargs):
        raise _ProcessStop("process stopped")

    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 3.0, open_time=0.0),
                      _leg("20", "sell", 1.0, open_time=1.0)])
    port.executed_unknown = True
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_close(_close_env("sell", 1.0), port))
    _run(eng.run_close(_close_env("sell", 1.0, pine_id="Other"), port))
    port.set_legs([_leg("10", "buy", 2.0, open_time=0.0), _leg("20", "sell", 1.0, open_time=1.0)])
    monkeypatch.setattr(one_way_emulator_module, "lower_close_leg_send_pre_volume", _stop)
    with pytest.raises(_ProcessStop):
        _run(eng.run_close(_close_env("sell", 1.0, pine_id="Third"), port))
    monkeypatch.undo()
    restarted = OneWayEmulator(store_ctx=ctx)
    _run(restarted.run_close(_close_env("sell", 1.0, pine_id="Fourth"), port))
    assert _dispatched(port) == [("10", 1), ("10", 1)]
    store.close()


def __test_an_entry_never_opens_next_to_an_opposing_leg_a_close_still_holds__(journal):
    """A buy whose opposing short is held by an unsettled close opens only once the short is gone.

    A close of the whole short of 2 is dispatched; the read still lists the
    short and the venue cannot settle the send. A buy of 3 must neither open
    beside the short (a failed close would leave a hedge) nor close the short
    again: it opens once the read shows the short gone.
    """
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = journal.emulator()
    close = CloseIntent(pine_id="Short", symbol="EURUSD", side="buy", qty=2.0)
    _run(eng.run_close(DispatchEnvelope(intent=close, run_tag="t000", bar_ts_ms=500), port))
    port.executed_unknown = True
    res = _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert res.opened_orders == () and port.placed == []
    _run(eng.drain_residual_opens(port))
    assert _dispatched(port) == [("20", 2)] and port.placed == []
    port.set_legs([])
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 3.0)]
    assert _dispatched(port) == [("20", 2)]
    assert _live_residuals(eng) == []


def __test_a_close_the_drain_sends_itself_comes_off_the_reversal_open__(journal):
    """A deferred reversal opens only what its combined order leaves after the closes it sends.

    A close of the whole short of 2 is dispatched for another intent and a
    buy of 3 waits behind it. The venue then proves that close executed
    nothing: the drain closes the short itself under the reversal, so the
    combined order of 3 is the close of 2 plus an open of 1, never an open of
    3 on top of it.
    """
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = journal.emulator()
    close = CloseIntent(pine_id="Short", symbol="EURUSD", side="buy", qty=2.0)
    _run(eng.run_close(DispatchEnvelope(intent=close, run_tag="t000", bar_ts_ms=500), port))
    port.executed_unknown = True
    res = _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert res.opened_orders == () and port.placed == []
    # The earlier close executed nothing: the short is the reversal's to close.
    port.executed_unknown = False
    _run(eng.drain_residual_opens(port))
    assert _dispatched(port) == [("20", 2), ("20", 2)] and port.placed == []
    port.set_legs([])
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 1.0)]
    assert _live_residuals(eng) == []


def __test_consecutive_replay_passes_never_spend_one_net_budget_twice__(tmp_path):
    """A replay pass leaves the net exposure an earlier pass already sent a close against.

    Two longs of 1 against a short of 1 are net long 1, with a pending close
    row on each long. The first pass re-sends the first row (the venue says
    it executed nothing) and cannot decide the second. In the next pass the
    first re-send is still outstanding and the read still lists its leg, so
    the second row must not send another close: both executing would turn
    the book short.
    """
    class _PerLegEvidencePort(_FakePort):
        def __init__(self, legs):
            super().__init__(legs)
            self.unknown_legs: set[str] = set()

        async def close_leg_executed(self, symbol, leg_id, coids, since_ms):
            if leg_id in self.unknown_legs:
                return None
            return await super().close_leg_executed(symbol, leg_id, coids, since_ms)

    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=1, qty=1.0)
    _pending_leg_row(ctx, leg_id="11", volume=1, qty=1.0)
    port = _PerLegEvidencePort([_leg("10", "buy", 1.0, open_time=0.0),
                                _leg("11", "buy", 1.0, open_time=1.0),
                                _leg("20", "sell", 1.0, open_time=2.0)])
    port.unknown_legs = {"11"}
    eng = OneWayEmulator(store_ctx=ctx)
    assert _run(eng.restart_replay(port)) is False
    assert _dispatched(port) == [("10", 1)]
    port.unknown_legs = {"10"}
    _run(eng.restart_replay(port))
    assert _dispatched(port) == [("10", 1)]
    store.close()


def __test_a_close_the_venue_settles_as_unexecuted_frees_its_reservation__(journal):
    """A send the venue records as not executed no longer holds exposure back."""
    port = _FakePort([_leg("10", "buy", 3.0, open_time=0.0)])
    eng = journal.emulator()
    _run(eng.run_close(_close_env("sell", 2.0), port))
    _run(eng.run_close(_close_env("sell", 2.0, pine_id="Other"), port))
    assert _dispatched(port) == [("10", 2), ("10", 2)]


def __test_a_reversal_closes_only_what_earlier_closes_left__(journal):
    """A market reversal is planned on the book less what an earlier close was sent against."""
    port = _FakePort([_leg("20", "sell", 3.0, open_time=0.0)])
    eng = journal.emulator()
    close = CloseIntent(pine_id="Short", symbol="EURUSD", side="buy", qty=2.0)
    _run(eng.run_close(DispatchEnvelope(intent=close, run_tag="t000", bar_ts_ms=500), port))
    port.executed_unknown = True
    res = _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert _dispatched(port) == [("20", 2), ("20", 1)]
    assert res.open_qty == 2.0


def __test_a_reversal_whose_first_close_never_went_out_is_retried_whole__(journal):
    """With no close on the wire the fault is raised and no breadcrumb or close row is left."""
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)],
                     fail_close_leg=ExchangeConnectionError("link down"))
    eng = journal.emulator()
    with pytest.raises(ExchangeConnectionError):
        _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert port.closed == [] and port.placed == []
    assert _live_residuals(eng) == []
    assert list(iter_active_close_legs(eng._close_journal)) == []  # type: ignore[attr-defined]


def __test_residual_recovery_never_closes_another_owners_share_on_a_whole_leg_port__(journal):
    """A new opposing position the run shares is not closed whole by the residual drain."""
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    port.supports_partial_leg_close = False
    eng = journal.emulator()
    _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert _dispatched(port) == [("20", 2)]
    # The short is gone, but a five-unit venue position shows up on the short
    # side of which the run owns one unit.
    port.set_legs([_leg("30", "sell", 1.0, open_time=2.0, venue_qty=5.0)])
    _run(eng.drain_residual_opens(port))
    assert _dispatched(port) == [("20", 2)]
    assert port.placed == []
    assert len(_live_residuals(eng)) == 1


def __test_an_opened_residual_is_handed_over_even_when_a_later_row_fails__():
    """The orders of a residual the drain opened survive a later row's failure."""
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0),
                      _leg("30", "buy", 2.0, symbol="GBPUSD", open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    first = _entry_env("buy", 3.0)
    second = _entry_env("sell", 3.0, symbol="GBPUSD", pine_id="Short")
    _run(eng.run_reversal(first, port))
    _run(eng.run_reversal(second, port))
    assert len(_live_residuals(eng)) == 2
    # EURUSD is flat now; GBPUSD still lists its long and the venue cannot
    # tell what its close executed, so that row raises.
    port.set_legs([_leg("30", "buy", 2.0, symbol="GBPUSD", open_time=0.0)])
    port.executed_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 1.0)]
    assert eng.take_opened_residuals() == [(first.intent.intent_key, ("order",))]
    assert eng.take_opened_residuals() == []


def __test_reversal_clears_breadcrumb_on_definitive_close_reject__(tmp_path):
    """A definitively-rejected FIFO close leg discharges the breadcrumb before any residual."""
    # A FIFO close leg is DEFINITIVELY rejected: the fan-out raises before the
    # residual place_leg ever runs. _dispatch_new turns this reject into a
    # non-halting skip for the entry, so the breadcrumb must be discharged here —
    # leaving it live would let a later restart re-close + re-open a reversal the
    # exchange already rejected and Pine never re-signalled.
    store, ctx = _make_store(tmp_path)
    env = _entry_env("buy", 3.0)
    residual_coid = f"{env.client_order_id(KIND_CLOSE)}:residual"
    port = _FakePort(
        [_leg("20", "sell", 2.0, open_time=0.0)],
        fail_close_leg=ExchangeOrderRejectedError("close rejected"),
    )
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(ExchangeOrderRejectedError):
        _run(eng.run_reversal(env, port))
    assert port.placed == []
    assert list(iter_active_residual_opens(ctx)) == []
    breadcrumb = ctx.get_order(residual_coid)
    assert breadcrumb is not None and breadcrumb.closed_ts_ms is not None
    store.close()


def _skipped_close_env(side, qty, *, synthetic_kind, pine_id="Long") -> DispatchEnvelope:
    if synthetic_kind == 'defensive_close':
        intent = CloseIntent(pine_id=pine_id, symbol="EURUSD", side=side, qty=qty,
                             synthetic_kind=synthetic_kind,
                             target_position_coid="t000-x-y-e")
    else:
        intent = CloseIntent(pine_id=pine_id, symbol="EURUSD", side=side, qty=qty,
                             synthetic_kind=synthetic_kind, target_entry_id=pine_id)
    return DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000)


def _leg_skip() -> OrderSkippedByPlugin:
    return OrderSkippedByPlugin(
        "leg busy", intent_key="Long", reason="position_locked",
    )


def __test_run_close_partial_trigger_leg_skip_is_non_halting_and_discharges_the_row__(tmp_path):
    """A plugin-skipped leg on an engine-trigger partial close propagates as a skip."""
    # Measured live (ctrader cycle 121): the native fail-safe fill locked the
    # leg the engine-trigger partial close targeted, and the plugin's per-leg
    # POSITION_LOCKED skip surfaced as a raw reject through the fan. The skip
    # must reach the engine as-is (re-evaluated next tick, no halt), and the
    # persist-first pending row must not survive for a restart replay to
    # re-close a leg the venue settled itself.
    store, ctx = _make_store(tmp_path)
    env = _skipped_close_env("sell", 1.0, synthetic_kind='partial_trigger')
    port = _FakePort([_leg("20", "buy", 2.0, open_time=0.0)],
                     fail_close_leg=_leg_skip())
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(OrderSkippedByPlugin) as info:
        _run(eng.run_close(env, port))
    assert info.value.reason == "position_locked"
    assert port.closed == []
    assert list(iter_active_close_legs(ctx)) == []
    leg_row = ctx.get_order(f"{env.client_order_id(KIND_CLOSE)}:20")
    assert leg_row is not None and leg_row.closed_ts_ms is not None
    store.close()


def __test_run_close_leg_skip_after_a_dispatched_leg_is_a_partial_success__(tmp_path):
    """A skip on a later leg ends the fan as a partial success, keeping the sent legs."""
    # Leg 10 went on the wire before leg 11 was declined: its fill is coming
    # back, so the caller must own it (order mapping, fill ledger) — the fan
    # returns what it sent instead of raising the skip, and the Pine-unit
    # dispatched quantity covers only the sent leg.
    store, ctx = _make_store(tmp_path)
    env = _skipped_close_env("sell", 3.0, synthetic_kind='partial_trigger')
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)],
                     fail_close_leg=_leg_skip(), fail_close_leg_id="11")
    eng = OneWayEmulator(store_ctx=ctx)
    res = _run(eng.run_close(env, port))
    assert res.legs == (("10", 2),)
    assert res.dispatched_qty == 2.0
    assert res.skipped is False
    assert _dispatched(port) == [("10", 2)]
    assert list(iter_active_close_legs(ctx)) == []
    parent = env.client_order_id(KIND_CLOSE)
    skipped_row = ctx.get_order(f"{parent}:11")
    assert skipped_row is not None and skipped_row.closed_ts_ms is not None
    store.close()


def __test_run_close_defensive_leg_skip_after_a_dispatched_leg_still_rejects__(tmp_path):
    """A non-skippable close keeps the definitive reject even mid-fan."""
    store, ctx = _make_store(tmp_path)
    env = _skipped_close_env("sell", 3.0, synthetic_kind='defensive_close')
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)],
                     fail_close_leg=_leg_skip(), fail_close_leg_id="11")
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(ExchangeOrderRejectedError):
        _run(eng.run_close(env, port))
    assert _dispatched(port) == [("10", 2)]
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_run_close_script_close_leg_skip_is_non_halting__(tmp_path):
    """A plain script close is skippable per leg the same way."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("20", "buy", 2.0, open_time=0.0)],
                     fail_close_leg=_leg_skip())
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(OrderSkippedByPlugin):
        _run(eng.run_close(_close_env("sell", 2.0), port))
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_run_close_defensive_leg_skip_stays_a_definitive_reject__(tmp_path):
    """A defensive close keeps the loud contract: the leg skip becomes a reject."""
    store, ctx = _make_store(tmp_path)
    env = _skipped_close_env("sell", 2.0, synthetic_kind='defensive_close')
    port = _FakePort([_leg("20", "buy", 2.0, open_time=0.0)],
                     fail_close_leg=_leg_skip())
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(ExchangeOrderRejectedError) as info:
        _run(eng.run_close(env, port))
    assert isinstance(info.value.__cause__, OrderSkippedByPlugin)
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_reversal_leg_skip_is_a_definitive_reject_and_clears_breadcrumb__(tmp_path):
    """A reversal's FIFO close leg skipped by the plugin is a definitive reject."""
    # The residual never opens and the breadcrumb is discharged exactly as for
    # a venue reject — a skipped close leg can never leave a reversal half-run.
    store, ctx = _make_store(tmp_path)
    env = _entry_env("buy", 3.0)
    residual_coid = f"{env.client_order_id(KIND_CLOSE)}:residual"
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)],
                     fail_close_leg=_leg_skip())
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(ExchangeOrderRejectedError):
        _run(eng.run_reversal(env, port))
    assert port.placed == []
    assert list(iter_active_residual_opens(ctx)) == []
    assert list(iter_active_close_legs(ctx)) == []
    breadcrumb = ctx.get_order(residual_coid)
    assert breadcrumb is not None and breadcrumb.closed_ts_ms is not None
    store.close()


def __test_reversal_keeps_breadcrumb_on_ambiguous_close_failure__(tmp_path):
    """An ambiguous FIFO close leg keeps the breadcrumb live for replay to reconcile."""
    # An ambiguous FIFO close leg (disposition unknown) leaves the close — and
    # thus a possibly-owed residual — unresolved: the breadcrumb MUST survive so
    # restart_replay reconciles it against the then-known leg state. A
    # definitive-reject clear must NOT swallow the ambiguous case.
    store, ctx = _make_store(tmp_path)
    env = _entry_env("buy", 3.0)
    port = _FakePort(
        [_leg("20", "sell", 2.0, open_time=0.0)],
        fail_close_leg=OrderDispositionUnknownError(
            "close timed out", client_order_id="close-coid",
        ),
    )
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_reversal(env, port))
    assert port.placed == []
    assert len(list(iter_active_residual_opens(ctx))) == 1
    store.close()


def __test_restart_replay_closes_opposing_legs_before_residual__(tmp_path):
    """Residual replay FIFO-closes every still-live opposing leg before opening the residual."""
    # Crash in the window AFTER the breadcrumb but BEFORE the FIFO closes landed:
    # the opposing legs are still live and no close-leg rows were persisted, so
    # _replay_close_legs has nothing to re-dispatch. Replaying the residual on top
    # of the still-open opposing legs would over-expose the book — the residual
    # replay must FIFO-close every still-live opposing leg before the open. The
    # close is only dispatched, so the open waits for a read that shows it.
    store, ctx = _make_store(tmp_path)
    # The breadcrumb carries the combined order: close the short 2, open 1.
    _seed_residual(ctx, entry_coid="entry-missing", qty=3.0, side="buy")
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))
    assert _dispatched(port) == [("20", 2)]
    assert port.placed == []
    assert len(list(iter_active_residual_opens(ctx))) == 1
    port.set_legs([])
    _run(eng.drain_residual_opens(port))
    assert _dispatched(port) == [("20", 2)]
    assert port.placed == [("EURUSD", 1.0)]
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_restart_replay_clears_breadcrumb_on_definitive_residual_reject__(tmp_path):
    """A definitive residual reject in replay discharges the breadcrumb, not aborting startup."""
    # The residual re-open is definitively rejected during replay. restart_replay
    # runs inside the sync startup wrapper, which only catches
    # ExchangeConnectionError, so the reject must NOT propagate (that would abort
    # startup and re-arm the same rejected residual every sync). The breadcrumb is
    # discharged so the residual the exchange refused never replays again.
    store, ctx = _make_store(tmp_path)
    _seed_residual(ctx, entry_coid="entry-missing", qty=2.0)
    port = _FakePort(
        [], fail_place_leg=ExchangeOrderRejectedError("insufficient margin"),
    )
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))  # no exception escapes
    assert port.placed == []
    assert list(iter_active_residual_opens(ctx)) == []
    breadcrumb = ctx.get_order("rev:residual")
    assert breadcrumb is not None and breadcrumb.closed_ts_ms is not None
    store.close()


def __test_restart_replay_keeps_breadcrumb_on_ambiguous_residual_failure__(tmp_path):
    """An ambiguous residual re-open in replay keeps the breadcrumb live, not aborting startup."""
    # An ambiguous residual re-open (disposition unknown) during replay leaves the
    # open's fate unknown. The breadcrumb MUST survive so the next sync's replay
    # reconciles it, and the error must NOT propagate — propagating would abort
    # startup on a recoverable round-trip.
    store, ctx = _make_store(tmp_path)
    _seed_residual(ctx, entry_coid="entry-missing", qty=2.0)
    port = _FakePort(
        [], fail_place_leg=OrderDispositionUnknownError(
            "residual timed out", client_order_id="rev:residual",
        ),
    )
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))  # no exception escapes
    assert port.placed == []
    assert len(list(iter_active_residual_opens(ctx))) == 1
    store.close()


# === Reversal residual-open per-sync drain (2.10) =======================

def __test_drain_residual_opens_reconciles_breadcrumb_restart_left_stranded__(tmp_path):
    """The per-sync drain re-opens a residual that one-shot restart replay left stranded."""
    # End-to-end: an ambiguous residual re-open during restart_replay leaves the
    # breadcrumb live (the 2.7 contract). restart_replay is one-shot — the engine
    # latches _one_way_replay_done after the first sync — so without a per-sync
    # drain that stranded breadcrumb would wait for the NEXT process restart.
    # drain_residual_opens reconciles it in-session: once the round-trip is healthy
    # the residual is re-opened and the breadcrumb discharged, no restart required.
    store, ctx = _make_store(tmp_path)
    _seed_residual(ctx, entry_coid="entry-missing", qty=2.0)
    eng = OneWayEmulator(store_ctx=ctx)
    # First sync: restart_replay's residual re-open times out ambiguously; the
    # breadcrumb survives and restart_replay will not retry it again.
    ambiguous = _FakePort(
        [], fail_place_leg=OrderDispositionUnknownError(
            "residual timed out", client_order_id="rev:residual",
        ),
    )
    _run(eng.restart_replay(ambiguous))
    assert ambiguous.placed == []
    assert len(list(iter_active_residual_opens(ctx))) == 1
    # Later sync, healthy broker: the per-sync drain re-opens the residual and
    # discharges the breadcrumb.
    healthy = _FakePort([])
    _run(eng.drain_residual_opens(healthy))
    assert healthy.placed == [("EURUSD", 2.0)]
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_drain_residual_opens_keeps_breadcrumb_on_ambiguous_retry__(tmp_path):
    """An ambiguous drain re-open keeps the breadcrumb live and never escapes the drain."""
    # The per-sync drain's own re-open is ALSO ambiguous: the breadcrumb must stay
    # live for the next sync to retry, and the error must NOT escape the drain —
    # halting on a recoverable round-trip would strand the bot. Mirrors
    # drain_clearing_rows' leave-live-and-retry contract.
    store, ctx = _make_store(tmp_path)
    _seed_residual(ctx, entry_coid="entry-missing", qty=2.0)
    port = _FakePort(
        [], fail_place_leg=OrderDispositionUnknownError(
            "residual still timing out", client_order_id="rev:residual",
        ),
    )
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.drain_residual_opens(port))  # no exception escapes
    assert port.placed == []
    assert len(list(iter_active_residual_opens(ctx))) == 1
    store.close()


def __test_drain_residual_opens_discharges_when_entry_row_exists__(tmp_path):
    """The drain discharges the breadcrumb without re-opening when the entry row already exists."""
    # The residual's persist-first entry row already landed (the live dispatch's
    # place_leg wrote it before the ambiguous send). The drain must DISCHARGE the
    # breadcrumb WITHOUT re-opening — the entry / recovery path owns the order, so a
    # re-open would double-open.
    store, ctx = _make_store(tmp_path)
    ctx.upsert_order("entry-present", symbol="EURUSD", side="buy", qty=2.0,
                     state="confirmed", pine_entry_id="Long")
    _seed_residual(ctx, entry_coid="entry-present", qty=2.0)
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.drain_residual_opens(port))
    assert port.placed == []  # entry path owns the open; no re-dispatch
    assert list(iter_active_residual_opens(ctx)) == []  # breadcrumb discharged
    store.close()


def __test_drain_residual_opens_drains_a_breadcrumb_in_the_venue_spelling__(tmp_path):
    """A run trades one instrument: a breadcrumb under the venue's spelling is the run's own."""
    store, ctx = _make_store(tmp_path)
    create_residual_open_row(
        ctx, coid="rev:residual", symbol="EUR/USD", side="buy", qty=2.0,
        intent_key="Long", pine_entry_id="Long", entry_coid="entry-missing",
        run_tag="t000", bar_ts_ms=1000, retry_seq=0,
    )
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EUR/USD", 2.0)]
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


# === transient port-read faults inside a dispatch =======================

class _TransientProviderError(ProviderError):
    """A plugin's own transient wire fault (a retryable ``ProviderError``)."""

    retryable = True


class _PermanentProviderError(ProviderError):
    """A plugin's own permanent misconfiguration (a non-retryable ``ProviderError``)."""


class _FailingReadPort(_FakePort):
    """A port whose position read (or quantizer read) raises ``fault``."""

    def __init__(self, legs: list[PositionLeg], fault: Exception, *, quantizer: bool) -> None:
        super().__init__(legs)
        self._fault = fault
        self._quantizer = quantizer

    async def fetch_raw_positions(self, symbol: str) -> list[PositionLeg]:
        if not self._quantizer:
            raise self._fault
        return await super().fetch_raw_positions(symbol)

    async def get_volume_quantizer(self, symbol: str):
        if self._quantizer:
            raise self._fault
        return await super().get_volume_quantizer(symbol)


@pytest.mark.parametrize("fault", [
    _TransientProviderError("HTTP transport error on /v5/position/list"),
    ConnectionError("Server disconnected without sending a response"),
    TimeoutError("read timed out"),
    ExchangeRateLimitError("too many requests", retry_after=1.0),
], ids=["retryable-provider-error", "connection-error", "timeout-error", "rate-limit"])
@pytest.mark.parametrize("quantizer", [False, True], ids=["legs", "quantizer"])
def __test_transient_port_read_fault_parks_the_dispatch__(fault, quantizer):
    """A transient fault on a fan-out's venue READ surfaces as ``ExchangeConnectionError``.

    The fan-outs run inside the engine's dispatch bridge, which halts the bot
    on a raw transient fault; a read sends nothing, so it must park instead.
    """
    legs = [_leg("10", "buy", 2.0)]
    dispatches = [
        lambda eng, legs_port: eng.run_exit_bracket(
            _exit_env("TP", "Long", tp=1.20, sl=1.00), legs_port),
        lambda eng, legs_port: eng.run_close(_close_env("sell", 2.0), legs_port),
        lambda eng, legs_port: eng.run_reversal(
            _entry_env("sell", 3.0, pine_id="Short"), legs_port),
    ]
    for dispatch in dispatches:
        if quantizer and dispatch is dispatches[0]:
            continue  # the bracket fan-out never quantizes
        port = _FailingReadPort(legs, fault, quantizer=quantizer)
        with pytest.raises(ExchangeConnectionError) as info:
            _run(dispatch(OneWayEmulator(store_ctx=None), port))
        assert info.value.__cause__ is fault
        assert port.closed == [] and port.placed == [] and port.amended == []


def __test_permanent_port_read_fault_propagates_unchanged__():
    """A non-retryable ``ProviderError`` on a fan-out read fails loud, untranslated."""
    fault = _PermanentProviderError("unknown symbol")
    port = _FailingReadPort([_leg("10", "buy", 2.0)], fault, quantizer=False)
    with pytest.raises(_PermanentProviderError):
        _run(OneWayEmulator(store_ctx=None).run_exit_bracket(
            _exit_env("TP", "Long", tp=1.20, sl=1.00), port))


# === One write per leg: an unsettled write holds every different write ===

class _GatedPort(_FakePort):
    """A port whose amend of ``gated_leg`` blocks until ``release`` is set.

    Models a venue call still in flight after the engine's bridge stopped
    waiting for it: the write takes effect only when the call is released.
    """

    def __init__(self, legs) -> None:
        super().__init__(legs)
        self.gated_leg: str | None = None
        self.release: asyncio.Event | None = None
        self.entered: asyncio.Event | None = None

    async def amend_bracket(self, symbol, leg_id, **kwargs) -> None:
        if leg_id == self.gated_leg:
            assert self.release is not None and self.entered is not None
            self.entered.set()
            await self.release.wait()
        await super().amend_bracket(symbol, leg_id, **kwargs)


def __test_a_write_still_in_flight_holds_a_newer_one_and_cannot_overwrite_it__(journal):
    """A stop modify still in flight is not overtaken by a newer one that it would later undo.

    The live defect: the bridge gave up on the first modify, the next sync sent
    a newer stop, and the first one then reached the venue and put the older
    stop back.
    """
    port = _GatedPort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()

    async def scenario() -> None:
        await eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port)
        port.release = asyncio.Event()
        port.entered = asyncio.Event()
        port.gated_leg = "10"
        older = asyncio.ensure_future(
            eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.02), port))
        await port.entered.wait()
        # The newer modify meets the leg's unsettled write and is not sent.
        with pytest.raises(OrderDispositionUnknownError):
            await eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.05), port)
        assert _amended_levels(port) == [("10", None, 1.00)]
        # Nothing to settle while the call is running: no venue read at all.
        reads = port.fetch_calls
        await eng.settle_leg_writes("EURUSD", port, 0.0)
        assert port.fetch_calls == reads
        # The older write completes; the newest levels the exit asks for go out
        # after it, never before.
        port.release.set()
        await older
        port.gated_leg = None
        await eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.05), port)

    _run(scenario())
    assert _amended_levels(port) == [("10", None, 1.00), ("10", None, 1.02), ("10", None, 1.05)]
    assert port.venue["10"][1] == 1.05


def __test_overlapping_settle_passes_judge_a_write_once__(journal):
    """A settle pass still reading the venue when the next starts is the only one that judges.

    The bridge can stop waiting for a slow pass while it keeps running. Two
    passes would both judge the same write from their reads, and the later
    one would remove a record the first already removed (or a newer one).
    """
    class _SlowMatchPort(_FakePort):
        def __init__(self, legs) -> None:
            super().__init__(legs)
            self.gate: asyncio.Event | None = None

        async def bracket_matches(self, symbol, leg_id, **kwargs):
            if self.gate is not None:
                await self.gate.wait()
            return await super().bracket_matches(symbol, leg_id, **kwargs)

    port = _SlowMatchPort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()

    async def scenario() -> None:
        port.fail_amend_unknown = True
        port.unknown_amends_land = True
        with pytest.raises(OrderDispositionUnknownError):
            await eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port)
        port.fail_amend_unknown = False
        port.gate = asyncio.Event()
        first = asyncio.ensure_future(eng.settle_leg_writes("EURUSD", port, 0.0))
        await asyncio.sleep(0)
        second = await eng.settle_leg_writes("EURUSD", port, 0.0)
        assert second == LegSettleResult()
        port.gate.set()
        await first

    _run(scenario())
    assert port.match_calls == 1
    assert not eng.needs_settling()


def __test_a_settle_read_raced_by_a_repeat_of_the_write_judges_nothing__(journal):
    """A repeat sent while the settle pass reads the venue voids that read.

    The read judges the earlier send; the repeat is another send of the same
    write, which ended without a verdict and restarted its settle window. The
    pass must neither confirm the write from the stale read nor fail on the
    reset window; the write stays unsettled for the next pass.
    """
    class _SlowMatchPort(_FakePort):
        def __init__(self, legs) -> None:
            super().__init__(legs)
            self.gate: asyncio.Event | None = None
            self.reading: asyncio.Event | None = None

        async def bracket_matches(self, symbol, leg_id, **kwargs):
            if self.gate is not None and self.reading is not None:
                self.reading.set()
                await self.gate.wait()
            return await super().bracket_matches(symbol, leg_id, **kwargs)

    port = _SlowMatchPort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()

    async def scenario() -> LegSettleResult:
        port.fail_amend_unknown = True
        port.unknown_amends_land = True
        with pytest.raises(OrderDispositionUnknownError):
            await eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port)
        port.gate = asyncio.Event()
        port.reading = asyncio.Event()
        pending = asyncio.ensure_future(eng.settle_leg_writes("EURUSD", port, 0.0))
        await port.reading.wait()
        # The identical repeat goes out and ends without a verdict too.
        port.unknown_amends_land = False
        with pytest.raises(OrderDispositionUnknownError):
            await eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port)
        port.gate.set()
        return await pending

    result = _run(scenario())
    assert result.confirmed == frozenset()
    assert eng.needs_settling()


def __test_a_timed_out_modify_that_landed_is_settled_by_reading_the_leg__(journal):
    """After a modify without a verdict the leg is written again only once its fate is read."""
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    port.fail_amend_unknown = True
    port.unknown_amends_land = True  # the venue applied it, the answer was lost
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.02), port))
    port.fail_amend_unknown = False
    port.amended.clear()
    # A different stop is held back: nothing reaches the venue.
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.05), port))
    assert port.amended == []
    assert eng.needs_settling()
    # The leg shows the timed-out write, so it landed; the leg is then brought
    # to what the exit asks for now.
    result = _run(eng.settle_leg_writes("EURUSD", port, 100.0))
    assert _amended_levels(port) == [("10", None, 1.05)]
    assert result.confirmed == frozenset({"X\0Long"}) and result.refused == ()
    assert not eng.needs_settling()


def __test_a_timed_out_modify_the_leg_never_shows_is_lost_only_after_its_window__(journal):
    """A write the leg does not show may still land: the leg stays held for the window."""
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.02), port))
    port.fail_amend_unknown = False
    port.amended.clear()
    # The leg does not show the write. The settle pass repeats the identical
    # stop at once (it cannot be undone by its own late arrival)...
    _run(eng.settle_leg_writes("EURUSD", port, 100.0))
    assert _amended_levels(port) == [("10", None, 1.02)]
    _run(eng.settle_leg_writes("EURUSD", port, 100.0 + UNSETTLED_WRITE_WINDOW_S - 0.5))
    assert _amended_levels(port) == [("10", None, 1.02)] and eng.needs_settling()
    # ...but a different one stays held, even though the leg now shows 1.02:
    # which of the two identical writes landed is not known.
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.05), port))
    _run(eng.settle_leg_writes("EURUSD", port, 100.0 + UNSETTLED_WRITE_WINDOW_S - 0.1))
    assert _amended_levels(port) == [("10", None, 1.02)]
    # Once the window has passed the leg takes the levels the exit asks for.
    _run(eng.settle_leg_writes("EURUSD", port, 100.0 + UNSETTLED_WRITE_WINDOW_S))
    assert _amended_levels(port) == [("10", None, 1.02), ("10", None, 1.05)]
    assert not eng.needs_settling()


def __test_a_lost_first_attach_is_sent_again_by_the_settle_pass__(journal, monkeypatch):
    """An attach that never reached the venue is repeated on the next sync, not at a restart."""
    warnings = _capture_warnings(monkeypatch)
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    port.fail_amend_unknown = False
    result = _run(eng.settle_leg_writes("EURUSD", port, 0.0))
    assert _amended_levels(port) == [("10", None, 1.00)]
    assert result.confirmed == frozenset()  # the first write may still land
    result = _run(eng.settle_leg_writes("EURUSD", port, UNSETTLED_WRITE_WINDOW_S))
    assert _amended_levels(port) == [("10", None, 1.00)]
    assert result.confirmed == frozenset({"X\0Long"}) and not eng.needs_settling()
    assert warnings == []


def __test_a_write_the_venue_cannot_read_back_is_re_sent_once_after_its_window__(journal, monkeypatch):
    """A "cannot tell" answer neither confirms the write nor drops it: one re-send settles it.

    The cTrader case: a trailing-to-trailing amend with the take-profit kept
    whose answer was lost. The position shows a trailing stop either way and
    does not report its distance, so the read cannot tell the new write from
    the old protection. A confirmation would leave a lost amend in force
    forever; a "lost" verdict inside the window would rewrite a moved trail
    on every sync.
    """
    warnings = _capture_warnings(monkeypatch)
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, tp=1.20, sl=1.00), port))
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, tp=1.20, sl=1.02), port))
    port.fail_amend_unknown = False
    port.cannot_tell = True
    port.amended.clear()
    # Inside the window: held, not confirmed, and nothing is repeated.
    result = _run(eng.settle_leg_writes("EURUSD", port, 100.0))
    assert port.amended == [] and result.confirmed == frozenset() and eng.needs_settling()
    _run(eng.settle_leg_writes("EURUSD", port, 100.0 + UNSETTLED_WRITE_WINDOW_S - 0.1))
    assert port.amended == [] and eng.needs_settling()
    assert warnings == []
    # After the window the same write is sent once, and its confirmed return
    # settles the leg.
    result = _run(eng.settle_leg_writes("EURUSD", port, 100.0 + UNSETTLED_WRITE_WINDOW_S))
    assert _amended_levels(port) == [("10", 1.20, 1.02)]
    assert result.confirmed == frozenset({"X\0Long"}) and not eng.needs_settling()
    assert len(warnings) == 1 and "does not show whether the leg holds it" in warnings[0]
    # Settled: the confirmed levels are not written again.
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, tp=1.20, sl=1.02), port))
    assert _amended_levels(port) == [("10", 1.20, 1.02)]


def __test_a_cannot_tell_re_send_without_a_verdict_holds_the_leg_again__(journal):
    """The re-send after the window is a write of its own: an unknown outcome holds the leg again."""
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    port.cannot_tell = True
    _run(eng.settle_leg_writes("EURUSD", port, 0.0))
    result = _run(eng.settle_leg_writes("EURUSD", port, UNSETTLED_WRITE_WINDOW_S))
    assert result.confirmed == frozenset() and eng.needs_settling()
    port.fail_amend_unknown = False
    # The re-send's own window: held again, then re-sent once more.
    _run(eng.settle_leg_writes("EURUSD", port, UNSETTLED_WRITE_WINDOW_S + 1.0))
    assert port.amended == []
    result = _run(eng.settle_leg_writes("EURUSD", port, 2 * UNSETTLED_WRITE_WINDOW_S + 1.0))
    assert _amended_levels(port) == [("10", None, 1.00)]
    assert result.confirmed == frozenset({"X\0Long"}) and not eng.needs_settling()


def __test_a_cannot_tell_write_whose_repeat_was_confirmed_is_not_sent_again__(journal):
    """An identical repeat the venue confirmed inside the window is the one re-assertion."""
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.02), port))
    port.fail_amend_unknown = False
    port.cannot_tell = True
    port.amended.clear()
    # The exit's own retry repeats the identical write; the venue confirms it.
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.02), port))
    assert _amended_levels(port) == [("10", None, 1.02)]
    _run(eng.settle_leg_writes("EURUSD", port, 0.0))
    result = _run(eng.settle_leg_writes("EURUSD", port, UNSETTLED_WRITE_WINDOW_S))
    assert _amended_levels(port) == [("10", None, 1.02)]
    assert result.confirmed == frozenset({"X\0Long"}) and not eng.needs_settling()


def __test_a_refused_re_drive_is_handed_to_the_engine__(journal):
    """The venue refusing the levels after an unsettled write is reported, never raised."""
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()
    port.fail_amend_unknown = True
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    refusing = _FakePort([_leg("10", "buy", 1.0)], fail_amend_leg="10")
    _run(eng.settle_leg_writes("EURUSD", refusing, 0.0))
    result = _run(eng.settle_leg_writes("EURUSD", refusing, UNSETTLED_WRITE_WINDOW_S))
    assert [key for key, _error in result.refused] == ["X\0Long"]
    assert result.confirmed == frozenset() and not eng.needs_settling()


# === Protection covers the surviving quantity only ===

def __test_a_partly_offset_leg_gets_no_whole_leg_bracket_but_a_software_guard__(
        journal, monkeypatch):
    """A whole-leg bracket on a partly offset leg would flip the net position.

    Long 1.0 on leg "10", an opposing 0.4 on leg "20": the net position is
    long 0.6. A venue bracket closes the whole 1.0 when it fires and leaves
    the book short 0.4, so the leg is guarded in software for 0.6 instead.
    """
    warnings = _capture_warnings(monkeypatch)
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("20", "sell", 0.4, open_time=1.0)])
    eng = journal.emulator()
    res = _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=0.6, tp=1.20, sl=1.00), port))
    assert res.legs == ("10",) and not res.skipped
    assert port.amended == []  # no venue bracket anywhere
    assert eng.has_software_guards()
    assert len([w for w in warnings if "guarded in software" in w]) == 1
    # Inside the levels nothing fires.
    assert eng.software_guard_triggers("EURUSD", 1.10) == []
    (trigger,) = eng.software_guard_triggers("EURUSD", 0.99)
    assert (trigger.leg_id, trigger.side, trigger.reason, trigger.level) == (
        "10", "sell", "stop_loss", 1.00)
    assert trigger.qty == pytest.approx(0.6)
    assert trigger.from_entry == "Long" and trigger.exit_ids == ("X",)
    # A firing guard is not evaluated again until the engine settles its close.
    assert eng.software_guard_triggers("EURUSD", 0.98) == []
    eng.software_guard_settled("EURUSD", "10", outcome="retry")
    assert len(eng.software_guard_triggers("EURUSD", 0.98)) == 1
    eng.software_guard_settled("EURUSD", "10", outcome="closed")
    assert not eng.has_software_guards()


def __test_a_venue_bracket_is_cleared_before_the_leg_goes_under_the_software_guard__(tmp_path):
    """A leg bracketed whole and then partly offset loses the venue bracket first."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, tp=1.20, sl=1.00), port))
    assert _amended_levels(port) == [("10", 1.20, 1.00)]
    assert port.amended_qty == [("10", 1.0)]  # the whole venue position
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0),
                   _leg("20", "sell", 0.4, open_time=1.0)])
    port.amended.clear()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=0.6, tp=1.25, sl=1.00), port))
    assert _amended_levels(port) == [("10", None, None)]
    (trigger,) = eng.software_guard_triggers("EURUSD", 1.26)
    assert (trigger.reason, trigger.level) == ("take_profit", 1.25)
    assert trigger.qty == pytest.approx(0.6)
    eng.software_guard_settled("EURUSD", "10", outcome="retry")

    # A restart keeps the leg under the guard and writes no venue bracket: its
    # rows record that the venue bracket is gone.
    replay_port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                             _leg("20", "sell", 0.4, open_time=1.0)])
    restarted = OneWayEmulator(store_ctx=ctx)
    assert _run(restarted.restart_replay(replay_port)) is True
    assert replay_port.amended == []
    (trigger,) = restarted.software_guard_triggers("EURUSD", 0.99)
    assert trigger.reason == "stop_loss" and trigger.qty == pytest.approx(0.6)

    # The opposing leg closes: the leg is whole net exposure again and gets
    # its venue bracket back, the guard is gone.
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0)])
    port.amended.clear()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, tp=1.30, sl=1.00), port))
    assert _amended_levels(port) == [("10", 1.30, 1.00)]
    assert not eng.has_software_guards()
    store.close()


def __test_a_leg_sharing_its_venue_position_is_never_bracketed_whole__(journal):
    """A venue position holding exposure the run does not own gets no position bracket.

    The run owns 1.0 of a venue position of 3.0: a position-attribute stop
    would close all 3.0. The leg is guarded for the run's 1.0, and a leg this
    run never wrote is not sent a clear either (it would wipe a bracket of the
    exposure's other owner).
    """
    port = _FakePort([_leg("1", "buy", 1.0, venue_qty=3.0)])
    eng = journal.emulator()
    res = _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    assert res.legs == ("1",)
    assert port.amended == []
    (trigger,) = eng.software_guard_triggers("EURUSD", 1.00)
    assert trigger.qty == pytest.approx(1.0) and trigger.reason == "stop_loss"
    eng.software_guard_settled("EURUSD", "1", outcome="retry")
    # Retiring the exit drops the guard, again without a venue call.
    _run(eng.run_exit_bracket_clear(_cancel_env("X", from_entry="Long"), port))
    assert port.amended == [] and not eng.has_software_guards()
    assert list(journal.rows()) == []


def __test_a_leg_that_starts_sharing_its_venue_position_loses_its_whole_stop__(journal):
    """Foreign exposure joining a whole-bracketed leg moves its stop to the software guard.

    The run owned the whole venue position when its stop went on. Once the
    venue position also holds exposure the run does not own, the stop would
    close that too; the next settle pass clears it and guards the run's own
    quantity instead, with no bracket operation of the strategy. When the
    foreign exposure leaves, the venue stop comes back. Neither switch
    confirms a parked exit.
    """
    port = _FakePort([_leg("1", "buy", 1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    assert _amended_levels(port) == [("1", None, 1.00)]
    assert eng.watches_leg_shares() and not eng.needs_settling()
    port.set_legs([_leg("1", "buy", 1.0, venue_qty=3.0)])
    port.amended.clear()
    result = _run(eng.settle_leg_writes("EURUSD", port, 0.0))
    assert _amended_levels(port) == [("1", None, None)]
    assert result.confirmed == frozenset() and result.refused == ()
    (trigger,) = eng.software_guard_triggers("EURUSD", 1.00)
    assert trigger.qty == pytest.approx(1.0) and trigger.reason == "stop_loss"
    eng.software_guard_settled("EURUSD", "1", outcome="retry")
    # An unchanged share is not re-driven: no venue write.
    port.amended.clear()
    _run(eng.settle_leg_writes("EURUSD", port, 1.0))
    assert port.amended == []
    # The foreign exposure leaves: the whole-position stop is written again.
    port.set_legs([_leg("1", "buy", 1.0)])
    result = _run(eng.settle_leg_writes("EURUSD", port, 2.0))
    assert _amended_levels(port) == [("1", None, 1.00)]
    assert result.confirmed == frozenset()
    assert not eng.has_software_guards()
    # Retired: nothing is watched any more.
    _run(eng.run_exit_bracket_clear(_cancel_env("X", from_entry="Long"), port))
    assert not eng.watches_leg_shares()


def __test_a_share_switch_never_leaves_the_leg_without_either_protection__(journal):
    """The guard is armed before the venue stop is cleared, and kept until it is replaced.

    Both switch writes end without a verdict here: the venue may still hold
    the old stop or may already hold the new one, so the guard stays armed
    throughout.
    """
    port = _FakePort([_leg("1", "buy", 1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    port.set_legs([_leg("1", "buy", 1.0, venue_qty=3.0)])
    port.fail_amend_unknown = True
    _run(eng.settle_leg_writes("EURUSD", port, 0.0))
    assert eng.has_software_guards() and eng.needs_settling()
    # The clear's outcome becomes known (it landed), and the share goes back
    # to whole while the venue stop is re-written without a verdict.
    port.fail_amend_unknown = False
    port.venue["1"] = (None, None, None, 3.0)
    _run(eng.settle_leg_writes("EURUSD", port, 1.0))
    assert eng.has_software_guards()
    port.set_legs([_leg("1", "buy", 1.0)])
    port.fail_amend_unknown = True
    _run(eng.settle_leg_writes("EURUSD", port, 2.0))
    assert eng.has_software_guards()
    (trigger,) = eng.software_guard_triggers("EURUSD", 1.00)
    assert trigger.qty == pytest.approx(1.0)


def __test_a_share_switch_whose_clear_never_left_is_retried_on_the_next_sync__(journal):
    """A dropped connection on the clear keeps the switch pending until the clear goes out.

    The clear raised :class:`ExchangeConnectionError` (never sent): the
    venue still holds the whole-position stop. The settle pass of the next
    sync must clear it again, the guard protecting the run's share meanwhile.
    """
    port = _FakePort([_leg("1", "buy", 1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    port.set_legs([_leg("1", "buy", 1.0, venue_qty=3.0)])
    amend = port.amend_bracket
    drops = [ExchangeConnectionError("connection dropped")]

    async def dropping_amend(*args, **kwargs) -> None:
        if drops:
            raise drops.pop()
        await amend(*args, **kwargs)

    port.amend_bracket = dropping_amend
    port.amended.clear()
    with pytest.raises(ExchangeConnectionError):
        _run(eng.settle_leg_writes("EURUSD", port, 0.0))
    assert port.amended == [] and eng.has_software_guards()
    assert port.venue["1"][1] == 1.00
    _run(eng.settle_leg_writes("EURUSD", port, 1.0))
    assert _amended_levels(port) == [("1", None, None)]
    assert eng.has_software_guards()
    # Done: the next pass sends nothing more.
    _run(eng.settle_leg_writes("EURUSD", port, 2.0))
    assert _amended_levels(port) == [("1", None, None)]


def __test_a_failed_share_switch_reverted_by_the_venue_drops_the_guard__(journal):
    """A switch whose clear never left is reconciled even when the share goes back.

    The clear raised :class:`ExchangeConnectionError` with the guard already
    armed, and the foreign exposure left before the next sync: the leg is
    whole again, its earlier mode. The settle pass must still reconcile it,
    so the venue stop stays the leg's only protection and no software guard
    races it with a second close.
    """
    port = _FakePort([_leg("1", "buy", 1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    port.set_legs([_leg("1", "buy", 1.0, venue_qty=3.0)])
    amend = port.amend_bracket
    drops = [ExchangeConnectionError("connection dropped")]

    async def dropping_amend(*args, **kwargs) -> None:
        if drops:
            raise drops.pop()
        await amend(*args, **kwargs)

    port.amend_bracket = dropping_amend
    port.amended.clear()
    with pytest.raises(ExchangeConnectionError):
        _run(eng.settle_leg_writes("EURUSD", port, 0.0))
    assert eng.has_software_guards() and port.venue["1"][1] == 1.00
    port.set_legs([_leg("1", "buy", 1.0)])
    _run(eng.settle_leg_writes("EURUSD", port, 1.0))
    assert not eng.has_software_guards()
    assert eng.software_guard_triggers("EURUSD", 1.00) == []
    assert port.venue["1"][1] == 1.00
    assert all(sl == 1.00 for _leg_id, _tp, sl in _amended_levels(port))
    # Reconciled: the next pass sends nothing more.
    sent = list(port.amended)
    _run(eng.settle_leg_writes("EURUSD", port, 2.0))
    assert port.amended == sent and eng.watches_leg_shares()


def __test_a_port_with_a_native_sized_bracket_receives_the_protected_quantity__(journal):
    """Where the venue can size a leg bracket, the amend carries the surviving quantity."""
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("20", "sell", 0.4, open_time=1.0)])
    eng = journal.emulator(sized_leg_bracket=CapabilityLevel.NATIVE)
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=0.6, sl=1.00), port))
    assert _amended_levels(port) == [("10", None, 1.00)]
    assert [(leg_id, pytest.approx(qty)) for leg_id, qty in port.amended_qty] == [("10", 0.6)]
    assert not eng.has_software_guards()


def __test_a_software_guard_trails_from_the_first_price_and_only_in_favour__():
    """The guard's trailing stop arms at the first price and never moves back."""
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("20", "sell", 0.4, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=None, mintick=0.01)
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=0.6, trail=5.0), port))
    assert eng.software_guard_triggers("EURUSD", 1.10) == []  # arms at 1.05
    assert eng.software_guard_triggers("EURUSD", 1.20) == []  # trails to 1.15
    assert eng.software_guard_triggers("EURUSD", 1.16) == []  # does not move back
    (trigger,) = eng.software_guard_triggers("EURUSD", 1.15)
    assert trigger.reason == "trailing_stop" and trigger.level == pytest.approx(1.15)


def __test_a_partly_offset_leg_on_a_whole_leg_only_venue_is_reported_unprotectable__(
        journal, monkeypatch):
    """Neither a sized bracket nor a partial close: the leg gets no bracket, loudly."""
    errors: list[str] = []
    monkeypatch.setattr(
        "pynecore.core.broker.one_way_emulator._blog_error",
        lambda msg, *args, **_kw: errors.append(msg % args),
    )
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("20", "sell", 0.4, open_time=1.0)])
    port.supports_partial_leg_close = False
    eng = journal.emulator()
    for _ in range(2):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=0.6, sl=1.00), port))
    assert port.amended == [] and not eng.has_software_guards()
    assert len([e for e in errors if "cannot be protected" in e]) == 1
    # With the opposing leg gone the leg is whole net exposure and is bracketed.
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0)])
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.00), port))
    assert _amended_levels(port) == [("10", None, 1.00)]


# === Review round: fan, guard, whole-leg and resend hardening ===

def __test_an_unknown_first_leg_still_writes_and_records_the_other_legs__(journal):
    """An attach of unknown outcome on one leg does not leave the later legs unowned.

    The fan used to stop at the first unknown outcome: the second leg had no
    ownership row, so the settle pass confirmed the exit while that leg carried
    no stop.
    """
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)],
                     fail_amend_unknown_leg="10")
    eng = journal.emulator()
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=2.0, sl=1.00), port))
    assert _amended_levels(port) == [("11", None, 1.00)]
    assert sorted((row.extras or {})[EXTRAS_KEY_BRACKET_OWN_LEG_ID]
                  for row in journal.rows()) == ["10", "11"]
    port.fail_amend_unknown_leg = None
    _run(eng.settle_leg_writes("EURUSD", port, 0.0))
    result = _run(eng.settle_leg_writes("EURUSD", port, UNSETTLED_WRITE_WINDOW_S))
    assert result.confirmed == frozenset({"X\0Long"})
    assert port.venue["10"][1] == 1.00 and port.venue["11"][1] == 1.00


def __test_a_software_guard_close_never_exceeds_what_survives_now__(journal):
    """The guard closes at most the leg's current net share, never its armed quantity.

    Long 10 against an opposing 4 arms a guard for 6. The opposing leg then
    grows to 9: closing 6 would turn a net long of 1 into a net short of 5.
    """
    port = _FakePort([_leg("10", "buy", 10.0, open_time=0.0),
                      _leg("20", "sell", 4.0, open_time=1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=6.0, sl=1.00), port))
    (trigger,) = eng.software_guard_triggers("EURUSD", 0.99)
    assert trigger.qty == pytest.approx(6.0)
    port.set_legs([_leg("10", "buy", 10.0, open_time=0.0),
                   _leg("20", "sell", 9.0, open_time=1.0)])

    def guard_close() -> DispatchEnvelope:
        intent = CloseIntent(
            pine_id="__pyne_leg_guard__10", symbol="EURUSD", side="sell",
            qty=trigger.qty, immediately=True, synthetic_kind="leg_guard",
            target_leg_id="10", target_entry_id=trigger.from_entry,
        )
        return DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000)

    res = _run(eng.run_close(guard_close(), port))
    assert _dispatched(port) == [("10", 1)] and res.dispatched_qty == pytest.approx(1.0)
    # Nothing of the leg survives any more: the guard close sends nothing.
    port.closed.clear()
    port.set_legs([_leg("10", "buy", 10.0, open_time=0.0),
                   _leg("20", "sell", 10.0, open_time=1.0)])
    res = _run(eng.run_close(guard_close(), port))
    assert port.closed == [] and res.legs == () and not res.skipped


def __test_a_whole_leg_close_that_would_flip_the_net_is_not_sent__(monkeypatch):
    """On a whole-leg-only port a close is never widened past the net exposure."""
    warnings: list[str] = []
    monkeypatch.setattr(
        "pynecore.core.broker.one_way_emulator._blog_warning",
        lambda msg, *args, **_kw: warnings.append(msg % args),
    )
    port = _FakePort([_leg("10", "buy", 10.0, open_time=0.0),
                      _leg("20", "sell", 9.0, open_time=1.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=None)
    with pytest.raises(OrderSkippedByPlugin):
        _run(eng.run_close(_close_env("sell", 1.0), port))
    assert port.closed == []
    assert len([w for w in warnings if "would flip the net position" in w]) == 1
    # The mirror through a reversal-shaped reduce: net short 1, a buy of 1.
    port = _FakePort([_leg("20", "sell", 10.0, open_time=0.0),
                      _leg("10", "buy", 9.0, open_time=1.0)])
    port.supports_partial_leg_close = False
    with pytest.raises(OrderSkippedByPlugin):
        _run(eng.run_reversal(_entry_env("buy", 1.0), port))
    assert port.closed == [] and port.placed == []


def __test_a_close_larger_than_the_net_never_flips_it__():
    """A close is capped at its side's net exposure, whole legs in the plan included.

    Long legs of four and six against a short leg of nine net to one long. A
    close of four would take the whole first leg and leave three short; it
    closes the one unit of net exposure, the rest reported as a shortfall.
    On a whole-leg-only port no leg fits inside that one unit, so nothing goes
    out.
    """
    legs = [_leg("10", "buy", 4.0, open_time=0.0), _leg("11", "buy", 6.0, open_time=1.0),
            _leg("20", "sell", 9.0, open_time=2.0)]
    port = _FakePort(legs)
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 4.0), port))
    assert _dispatched(port) == [("10", 1)]
    assert res.shortfall == pytest.approx(3.0)
    port = _FakePort(legs)
    port.supports_partial_leg_close = False
    with pytest.raises(OrderSkippedByPlugin):
        _run(eng.run_close(_close_env("sell", 4.0), port))
    assert port.closed == []


def __test_a_whole_position_close_of_a_shared_venue_position_is_not_sent__():
    """A whole-position-only port never closes a venue position the run only partly owns."""
    port = _FakePort([_leg("1", "buy", 2.0, venue_qty=10.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=None)
    with pytest.raises(OrderSkippedByPlugin):
        _run(eng.run_close(_close_env("sell", 2.0), port))
    assert port.closed == []


def __test_a_shared_venue_position_on_a_whole_position_venue_gets_no_guard__(
        journal, monkeypatch):
    """A guard there would close the whole venue position, the other owner's share too."""
    errors: list[str] = []
    monkeypatch.setattr(
        "pynecore.core.broker.one_way_emulator._blog_error",
        lambda msg, *args, **_kw: errors.append(msg % args),
    )
    port = _FakePort([_leg("1", "buy", 2.0, venue_qty=10.0)])
    port.supports_partial_leg_close = False
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=2.0, sl=1.00), port))
    assert port.amended == [] and not eng.has_software_guards()
    assert len([e for e in errors if "cannot be protected" in e]) == 1


def __test_a_repeat_still_in_flight_cannot_undo_a_newer_stop__(journal):
    """The identical repeat of an unsettled write holds the leg until the repeat ends.

    The repeat used to run under the original write's settle window: once that
    expired a newer stop went out, and the repeat arriving late put the older
    stop back.
    """
    port = _GatedPort([_leg("10", "buy", 1.0)])
    eng = journal.emulator()

    async def scenario() -> None:
        port.fail_amend_unknown = True
        with pytest.raises(OrderDispositionUnknownError):
            await eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.01), port)
        port.fail_amend_unknown = False
        port.release = asyncio.Event()
        port.entered = asyncio.Event()
        port.gated_leg = "10"
        # The settle pass repeats the identical stop; the repeat hangs.
        repeat = asyncio.ensure_future(eng.settle_leg_writes("EURUSD", port, 0.0))
        await port.entered.wait()
        # The script moves the stop; the leg is held.
        with pytest.raises(OrderDispositionUnknownError):
            await eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, sl=1.05), port)
        # The original write's window passes while the repeat is in flight:
        # nothing else may go out.
        await eng.settle_leg_writes("EURUSD", port, UNSETTLED_WRITE_WINDOW_S + 1.0)
        assert port.amended == []
        port.release.set()
        await repeat
        port.gated_leg = None
        assert _amended_levels(port) == [("10", None, 1.01)]
        # Only now, with the repeat ended, does the newer stop go out.
        await eng.settle_leg_writes("EURUSD", port, UNSETTLED_WRITE_WINDOW_S + 2.0)

    _run(scenario())
    assert _amended_levels(port) == [("10", None, 1.01), ("10", None, 1.05)]
    assert port.venue["10"][1] == 1.05
    assert not eng.needs_settling()


# === One write per close-leg / breadcrumb transition ======================

class _WriteFault(Exception):
    """The injected one-shot failure of a journal write."""


def _fail_write_once(journal, method: str, when=lambda *args, **fields: True) -> None:
    """Make the next ``method`` call of ``journal`` matching ``when`` raise once."""
    real = getattr(journal, method)
    state = {'armed': True}

    def wrapper(*args, **fields):
        if state['armed'] and when(*args, **fields):
            state['armed'] = False
            raise _WriteFault(method)
        return real(*args, **fields)

    setattr(journal, method, wrapper)


def __test_a_failed_reservation_write_leaves_no_close_row_to_re_send__(tmp_path):
    """A close row is never persisted without the reservation of its send."""
    from pynecore.core.broker.store_helpers import EXTRAS_KEY_CLOSE_LEG_RESERVED
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _fail_write_once(
        ctx, 'upsert_order',
        when=lambda coid, **fields: ('symbol' not in fields
                                     and EXTRAS_KEY_CLOSE_LEG_RESERVED
                                     in (fields.get('extras') or {})),
    )
    with pytest.raises(_WriteFault):
        _run(eng.run_close(_close_env("sell", 1.0), port))
    assert port.closed == []
    # A pending row without its reservation would be re-sent by the next
    # settlement although nothing of it went out.
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


@pytest.mark.parametrize("mode", ["store", "memory"])
def __test_a_failed_close_of_a_finished_close_row_never_leaks_it_live__(tmp_path, mode):
    """Finishing a close row is one write: no live ``dispatched`` row without a send."""
    from pynecore.core.broker.ownership_journal import MemoryOwnershipJournal
    from pynecore.core.broker.store_helpers import finish_close_leg_row
    store = None
    if mode == "store":
        store, journal = _make_store(tmp_path)
    else:
        journal = MemoryOwnershipJournal()
    _pending_leg_row(journal, leg_id="10", volume=1, qty=1.0)
    coid = "t000-x-y-c:10"
    _fail_write_once(journal, 'close_order')
    with pytest.raises(_WriteFault):
        finish_close_leg_row(journal, coid)
    row = journal.get_order(coid)
    assert row is not None and row.closed_ts_ms is None
    # Still pending, so a settlement pass reads and finishes it.
    assert [r.client_order_id for r in iter_active_close_legs(journal)] == [coid]
    finish_close_leg_row(journal, coid)
    row = journal.get_order(coid)
    assert row is not None and row.closed_ts_ms is not None
    if store is not None:
        store.close()


def __test_a_failed_breadcrumb_rewrite_never_revives_the_earlier_one__(tmp_path):
    """Re-persisting a breadcrumb under the same coid is one write with its reopen."""
    from pynecore.core.broker.store_helpers import clear_residual_open_row
    store, ctx = _make_store(tmp_path)
    fields = dict(
        coid="t000-x-y-c:residual", symbol="EURUSD", side="buy", qty=1.0,
        intent_key="Long", pine_entry_id="Long", entry_coid="t000-x-y-e",
        run_tag="t000", bar_ts_ms=1000, retry_seq=0,
    )
    create_residual_open_row(ctx, **fields, close_plan=[("20", 2.0)])
    clear_residual_open_row(ctx, fields['coid'])
    _fail_write_once(ctx, 'upsert_order')
    with pytest.raises(_WriteFault):
        create_residual_open_row(ctx, **{**fields, 'retry_seq': 1})
    # A reopened breadcrumb still carrying the earlier close plan would be
    # replayed: the earlier reversal closed and opened again.
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


# === Storage failures on either side of a leg close ========================

def _fail_storage_once(journal, method: str, when=lambda *args, **fields: True, *,
                       times: int = 1) -> None:
    """Make the next ``times`` matching ``method`` calls of ``journal`` raise a storage error."""
    import sqlite3
    real = getattr(journal, method)
    state = {'left': times}

    def wrapper(*args, **fields):
        if state['left'] and when(*args, **fields):
            state['left'] -= 1
            raise sqlite3.OperationalError("disk I/O error")
        return real(*args, **fields)

    setattr(journal, method, wrapper)


def __test_a_storage_failure_before_a_leg_close_sends_nothing__(tmp_path):
    """A failed persist-first close-row write is a fault that never left the client."""
    from pynecore.core.broker.exceptions import JournalWriteError
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _fail_storage_once(ctx, 'upsert_order')
    with pytest.raises(JournalWriteError) as raised:
        _run(eng.run_close(_close_env("sell", 1.0), port))
    assert isinstance(raised.value, ExchangeConnectionError)
    assert port.closed == []
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_a_storage_failure_after_a_leg_close_parks_it_for_settlement__(tmp_path):
    """A failed write after the sent close is an unknown outcome its row still owns.

    The store stays down for the ``dispatched`` mark and for its owed retry:
    the row keeps its reservation. Once the store is back the owed mark lands,
    so the running process, not only a restart, finishes the row.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    # The first writes once the close is on the wire: the row's ``dispatched`` mark.
    _fail_storage_once(ctx, 'upsert_order', when=lambda *a, **f: bool(port.closed), times=2)
    with pytest.raises(OrderDispositionUnknownError) as raised:
        _run(eng.run_close(_close_env("sell", 1.0), port))
    assert raised.value.client_order_id.endswith(":10")
    assert len(port.closed) == 1
    # The row keeps its reservation, so the settlement reads what it executed.
    assert ([row.client_order_id for row in iter_active_close_legs(ctx)]
            == [raised.value.client_order_id])

    ctx.claim_owed_writes(lambda _label, write: write())
    assert list(iter_active_close_legs(ctx)) == []
    assert len(port.closed) == 1
    store.close()


def __test_a_store_still_down_after_a_sent_leg_close_parks_it_and_owes_the_finish__(tmp_path):
    """A store that also fails the follow-up read still parks the close and owes its finish.

    Nothing of the owed finish runs before it is queued, so the unknown
    disposition is raised and the finish lands once the store is back.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    # The finish and its owed retry both fail on their first read.
    _fail_storage_once(ctx, 'get_order', when=lambda *a, **f: bool(port.closed), times=2)
    with pytest.raises(OrderDispositionUnknownError) as raised:
        _run(eng.run_close(_close_env("sell", 1.0), port))
    assert len(port.closed) == 1
    assert ([row.client_order_id for row in iter_active_close_legs(ctx)]
            == [raised.value.client_order_id])

    ctx.claim_owed_writes(lambda _label, write: write())
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


# === Storage failures: a failed write leaves memory and journal as before ===


def _fail_call(monkeypatch, target, name: str, *, when=None) -> list[int]:
    """Make ``target.name`` raise a storage error once, on the first call ``when`` accepts."""
    original = getattr(target, name)
    failed = [0]

    def failing(*args, **kwargs):
        if not failed[0] and (when is None or when(*args, **kwargs)):
            failed[0] += 1
            raise sqlite3.OperationalError("disk I/O error")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, failing)
    return failed


def __test_a_failed_release_in_the_drain_still_reports_the_keys_released__(journal, monkeypatch):
    """Keys released before a failed release reach the engine; the rest is redone next sync."""
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0, entry_id="A"),
                      _leg("11", "buy", 1.0, open_time=1.0, entry_id="B")])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "A", qty=1.0, tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("TP", "B", qty=1.0, tp=1.30), port))
    port.fail_amend_unknown = True
    for entry in ("A", "B"):
        with pytest.raises(OrderDispositionUnknownError):
            _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry=entry), port))
    port.fail_amend_unknown = False
    _run(eng.settle_leg_writes("EURUSD", port, 0.0))
    _run(eng.settle_leg_writes("EURUSD", port, UNSETTLED_WRITE_WINDOW_S))
    assert _own_states(journal.rows()) == {
        "TP\0A": BRACKET_OWN_STATE_CLEARING, "TP\0B": BRACKET_OWN_STATE_CLEARING}
    calls = [0]

    def second_release(*_args, **_kwargs) -> bool:
        calls[0] += 1
        return calls[0] == 2

    _fail_call(monkeypatch, one_way_emulator_module, "update_bracket_ownership_state",
               when=second_release)
    assert _run(eng.drain_clearing_rows("EURUSD", port)) == {"TP\0A"}
    assert _own_states(journal.rows()) == {"TP\0B": BRACKET_OWN_STATE_CLEARING}
    assert _run(eng.drain_clearing_rows("EURUSD", port)) == {"TP\0B"}
    assert journal.rows() == []
    assert _run(eng.drain_clearing_rows("EURUSD", port)) == set()


def __test_a_failed_clearing_mark_leaves_every_row_active__(journal, monkeypatch):
    """A clear whose persist-first marking fails marks no row and sends nothing; a retry clears."""
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", qty=2.0, tp=1.20), port))
    port.amended.clear()
    calls = [0]

    def second_mark(*_args, **_kwargs) -> bool:
        calls[0] += 1
        return calls[0] == 2

    _fail_call(monkeypatch, one_way_emulator_module, "update_bracket_ownership_state",
               when=second_mark)
    cancel = _cancel_env("TP", from_entry="Long")
    with pytest.raises(sqlite3.OperationalError):
        _run(eng.run_exit_bracket_clear(cancel, port))
    assert sorted(row.extras[EXTRAS_KEY_BRACKET_OWN_STATE] for row in journal.rows()) == [
        "active", "active"]
    assert port.amended == []
    res = _run(eng.run_exit_bracket_clear(cancel, port))
    assert sorted(res.legs) == ["10", "11"]
    assert journal.rows() == []


def __test_a_failed_software_flag_write_flags_no_row__(tmp_path, monkeypatch):
    """The software flag of a leg's rows flips in one write; a failed one is redone by the re-run."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=1.0, tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("Y", "Long", qty=1.0, sl=1.00), port))
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0),
                   _leg("20", "sell", 0.4, open_time=1.0)])

    def flags() -> dict[str, bool]:
        return {row.intent_key: bool(row.extras.get(EXTRAS_KEY_BRACKET_OWN_SOFTWARE))
                for row in iter_active_bracket_ownerships(ctx)}

    def flagging_y(coid, **fields) -> bool:
        return coid.startswith("bo:Y") and bool(
            (fields.get('extras') or {}).get(EXTRAS_KEY_BRACKET_OWN_SOFTWARE))

    _fail_call(monkeypatch, ctx, "upsert_order", when=flagging_y)
    with pytest.raises(sqlite3.OperationalError):
        _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=0.6, tp=1.25), port))
    assert flags() == {"X\0Long": False, "Y\0Long": False}
    _run(eng.run_exit_bracket(_exit_env("X", "Long", qty=0.6, tp=1.25), port))
    assert flags() == {"X\0Long": True, "Y\0Long": True}
    store.close()


def __test_a_failed_attempt_move_is_written_by_the_next_pass__(tmp_path, monkeypatch):
    """A residual open that never left the client opens once, even when its attempt move fails."""
    store, ctx = _make_store(tmp_path)
    port = _RowLeavingPort([_leg("20", "sell", 2.0, open_time=0.0)], ctx)
    port._fail_place_leg = ExchangeConnectionError("link down")
    port.closes_execute = True
    eng = OneWayEmulator(store_ctx=ctx)
    env = _entry_env("buy", 3.0)
    _fail_call(monkeypatch, one_way_emulator_module, "advance_residual_open_attempt")
    res = _run(eng.run_reversal(env, port))
    assert res.opened_orders == () and port.placed == []
    assert len(_live_residuals(eng)) == 1
    port._fail_place_leg = None
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 1.0)]
    assert eng.take_opened_residuals() == [(env.intent.intent_key, ("order",))]
    assert _live_residuals(eng) == []
    _run(eng.drain_residual_opens(port))
    assert port.placed == [("EURUSD", 1.0)]
    store.close()


def __test_a_failed_record_of_a_cut_short_flatten_is_an_unknown_disposition__(
        tmp_path, monkeypatch):
    """A close went out, so a failed breadcrumb write never reads as "nothing sent"."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("20", "sell", 1.0, open_time=0.0),
                      _leg("21", "sell", 1.0, open_time=1.0)],
                     fail_close_leg=ExchangeConnectionError("link down"),
                     fail_close_leg_id="21")
    eng = OneWayEmulator(store_ctx=ctx)
    _fail_call(monkeypatch, one_way_emulator_module, "create_residual_open_row")
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_reversal(_entry_env("buy", 2.0), port))
    assert _dispatched(port) == [("20", 1)]
    store.close()


def __test_a_failed_discharge_of_an_unsent_close_is_settled_from_the_venue__(
        tmp_path, monkeypatch):
    """The row of a close that never left the client stays reserved when its discharge fails."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)],
                     fail_close_leg=ExchangeConnectionError("link down"))
    eng = OneWayEmulator(store_ctx=ctx)
    _fail_call(monkeypatch, one_way_emulator_module, "release_close_leg_send")
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_close(_close_env("sell", 2.0), port))
    assert [row.client_order_id for row in iter_active_close_legs(ctx)] == [
        close_leg_client_order_id(_close_env("sell", 2.0).client_order_id(KIND_CLOSE), "10")]
    port._fail_close_leg = None
    assert _run(eng.restart_replay(port)) is True
    assert len(port.closed) == 1 and port.closed[0][2] == 2
    assert list(iter_active_close_legs(ctx)) == []
    assert _run(eng.restart_replay(port)) is True
    assert len(port.closed) == 1
    store.close()


def __test_a_failed_finish_after_a_re_send_never_sends_it_again__(tmp_path, monkeypatch):
    """A re-send whose row write failed is settled from the venue's record, not sent twice."""
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=2, qty=2.0)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _fail_call(monkeypatch, one_way_emulator_module, "finish_close_leg_row")
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.restart_replay(port))
    assert len(port.closed) == 1
    _sym, _leg_id, volume, attempt = port.closed[0]
    port.executed_by_coid[attempt] = volume
    assert _run(eng.restart_replay(port)) is True
    assert len(port.closed) == 1
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_a_failed_dropped_leg_mark_leaves_every_row_active__(journal, monkeypatch):
    """A re-attach whose retirement marking fails marks no dropped row; the re-run retires them."""
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0, entry_id="A"),
                      _leg("11", "buy", 1.0, open_time=1.0, entry_id="A")])
    eng = journal.emulator()
    _run(eng.run_exit_bracket(_exit_env("TP", "A", qty=2.0, tp=1.20), port))
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0, entry_id="B"),
                   _leg("11", "buy", 1.0, open_time=1.0, entry_id="B"),
                   _leg("12", "buy", 1.0, open_time=2.0, entry_id="A")])
    calls = [0]

    def second_mark(*_args, **kwargs) -> bool:
        if kwargs.get('new_state') != BRACKET_OWN_STATE_CLEARING:
            return False
        calls[0] += 1
        return calls[0] == 2

    _fail_call(monkeypatch, one_way_emulator_module, "update_bracket_ownership_state",
               when=second_mark)
    with pytest.raises(sqlite3.OperationalError):
        _run(eng.run_exit_bracket(_exit_env("TP", "A", qty=1.0, tp=1.20), port))
    assert sorted((row.extras[EXTRAS_KEY_BRACKET_OWN_LEG_ID],
                   row.extras[EXTRAS_KEY_BRACKET_OWN_STATE]) for row in journal.rows()) == [
        ("10", "active"), ("11", "active")]
    _run(eng.run_exit_bracket(_exit_env("TP", "A", qty=1.0, tp=1.20), port))
    assert sorted((row.extras[EXTRAS_KEY_BRACKET_OWN_LEG_ID],
                   row.extras[EXTRAS_KEY_BRACKET_OWN_STATE]) for row in journal.rows()) == [
        ("12", "active")]


def __test_a_failed_re_send_record_sends_nothing__(tmp_path, monkeypatch):
    """A re-send whose attempt record fails is a journal fault before the send; a retry sends once."""
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=2, qty=2.0)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _fail_call(monkeypatch, one_way_emulator_module, "update_close_leg_state")
    with pytest.raises(JournalWriteError):
        _run(eng.restart_replay(port))
    assert port.closed == []
    assert _run(eng.restart_replay(port)) is True
    assert _dispatched(port) == [("10", 2)]
    assert list(iter_active_close_legs(ctx)) == []
    store.close()


def __test_a_failed_release_of_an_unsent_re_send_is_settled_from_the_venue__(
        tmp_path, monkeypatch):
    """An unsent re-send whose release fails stays reserved and is settled, never sent twice."""
    store, ctx = _make_store(tmp_path)
    _pending_leg_row(ctx, leg_id="10", volume=2, qty=2.0)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)],
                     fail_close_leg=ExchangeConnectionError("link down"))
    eng = OneWayEmulator(store_ctx=ctx)
    _fail_call(monkeypatch, one_way_emulator_module, "release_close_leg_send")
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.restart_replay(port))
    assert port.closed == []
    port._fail_close_leg = None
    assert _run(eng.restart_replay(port)) is True
    assert _dispatched(port) == [("10", 2)]
    assert _run(eng.restart_replay(port)) is True
    assert _dispatched(port) == [("10", 2)]
    store.close()
