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

import pytest

from pynecore.core.broker.exceptions import (
    BracketAttachAfterFillRejectedError,
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    ExchangeRateLimitError,
    OrderDispositionUnknownError,
    OrderSkippedByPlugin,
)
from pynecore.core.broker.idempotency import KIND_CANCEL, KIND_CLOSE, KIND_ENTRY_TRIGGERED
from pynecore.core.broker.models import (
    CancelIntent,
    CloseIntent,
    DispatchEnvelope,
    EntryIntent,
    ExitIntent,
    OrderType,
    PositionLeg,
)
from pynecore.core.broker.one_way_emulator import (
    BOOK_INCOMPLETE_GRACE_S,
    BracketFanResult,
    ClearFanResult,
    CloseFanResult,
    OneWayEmulator,
)
from pynecore.core.broker.storage import BrokerStore, RunContext, RunIdentity
from pynecore.core.broker.store_helpers import (
    BRACKET_OWN_STATE_CLEARING,
    ENTRY_KIND_POSITION,
    ENTRY_KIND_WORKING,
    EXTRAS_KEY_BRACKET_OWN_LEG_ID,
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
        self.fetch_calls = 0

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
            trail_offset: float | None, coid: str,
    ) -> None:
        if leg_id == self._fail_amend_leg:
            raise ExchangeOrderRejectedError(f"amend rejected for leg {leg_id}")
        if self.fail_amend_unknown or leg_id == self.fail_amend_unknown_leg:
            raise OrderDispositionUnknownError(
                f"amend disposition unknown for leg {leg_id}", client_order_id=coid,
            )
        self.amended.append((leg_id, side, tp_price, sl_price, trail_offset, coid))


def _leg(leg_id, side, qty, *, symbol="EURUSD", open_time=0.0, price=1.10,
         entry_id=None) -> PositionLeg:
    return PositionLeg(
        leg_id=leg_id, symbol=symbol, side=side, qty=qty,
        entry_price=price, open_time=open_time, unrealized_pnl=0.0,
        entry_id=entry_id,
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


def __test_run_close_partial_leg_unsupported_skips_atomically__():
    """A plan with a partial slice on a no-partial port skips BEFORE any dispatch."""
    port = _FakePort([_leg("10", "buy", 3.0, open_time=0.0),
                      _leg("11", "buy", 2.0, open_time=1.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=None)
    with pytest.raises(OrderSkippedByPlugin) as exc:
        _run(eng.run_close(_close_env("sell", 2.0), port))
    assert exc.value.reason == "partial_leg_close_unsupported"
    assert port.closed == []


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


def _pending_leg_row(ctx, *, leg_id, volume, qty):
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
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
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


def __test_run_reversal_partial_leg_unsupported_skips_atomically__():
    """A reversal plan with a partial slice skips atomically on a no-partial port."""
    # Broker holds a 3.0 sell leg while the combined order is only 2.0 —
    # the FIFO plan would reduce the leg partially, which the port cannot
    # express. The pre-flight raises BEFORE any close or breadcrumb.
    port = _FakePort([_leg("20", "sell", 3.0, open_time=0.0)])
    port.supports_partial_leg_close = False
    eng = OneWayEmulator(store_ctx=None)
    with pytest.raises(OrderSkippedByPlugin) as exc:
        _run(eng.run_reversal(_entry_env("buy", 2.0), port))
    assert exc.value.reason == "partial_leg_close_unsupported"
    assert port.closed == [] and port.placed == []


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


def __test_clear_only_owns__(tmp_path):
    """Clearing one exit wipes only its own legs; a co-located exit's bracket survives."""
    # THE regression test: a TP-exit and an SL-exit on the SAME legs. Cancelling
    # TP must clear ONLY TP's legs; SL's bracket must survive (the plugin's
    # broadcast clear strips both).
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "Long", sl=1.00), port))
    assert len(list(iter_active_bracket_ownerships(ctx))) == 4  # 2 exits x 2 legs
    port.amended.clear()
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="Long"), port))
    assert set(res.legs) == {"10", "11"}
    # Each owned leg is re-amended to what its remaining owner (SL) asks for:
    # the TP is dropped, the stop stays.
    assert sorted(_amended_levels(port)) == [("10", None, 1.00), ("11", None, 1.00)]
    surviving = list(iter_active_bracket_ownerships(ctx))
    assert len(surviving) == 2
    assert all(row.intent_key == "SL\0Long" for row in surviving)
    store.close()


def __test_two_exits_distinct_ownership__(tmp_path):
    """Two exits with different pine_ids get distinct intent_keys and attach coids."""
    # Different pine_ids -> different attach coids -> disjoint coid namespaces +
    # distinct intent_keys, which is what makes the scoped clear work.
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "Long", sl=1.00), port))
    rows = list(iter_active_bracket_ownerships(ctx))
    assert {row.intent_key for row in rows} == {"TP\0Long", "SL\0Long"}
    assert len({row.client_order_id for row in rows}) == 2
    store.close()


def __test_clear_all_by_pine_id_when_from_entry_none__(tmp_path):
    """A cancel with ``from_entry=None`` clears every exit sharing the pine_id, leaving others."""
    # A cancel-all (from_entry None) drops every exit sharing the pine_id, across
    # from_entry values, and leaves a different pine_id untouched.
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "L1", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("TP", "L2", tp=1.30), port))
    _run(eng.run_exit_bracket(_exit_env("RUN", "L1", sl=1.00), port))
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry=None), port))
    assert len(res.legs) == 2  # both TP rows (L1 + L2), not RUN
    surviving = list(iter_active_bracket_ownerships(ctx))
    assert all(row.intent_key == "RUN\0L1" for row in surviving)
    store.close()


def __test_clear_pure_path_is_noop__():
    """The ownership-aware clear is a no-op without a persisted store (``store_ctx=None``)."""
    # Ownership-aware clear needs the persisted index: store_ctx=None -> no-op.
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="Long"), port))
    assert res == ClearFanResult(legs=())
    assert port.amended == []


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
    # Restart with a healthy broker: the clearing row re-clears (amend to None,
    # NOT the original 1.20/1.00) and is released — the cancel is honoured.
    port.fail_amend_unknown = False
    port.amended.clear()
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


def __test_hedged_pyramid_each_leg_carries_its_own_entrys_bracket__(tmp_path):
    """Three pyramided entries, one leg each: every leg carries its own exit's levels.

    The live defect: every exit was amended onto every leg, so the last
    writer (L3-X) won on all three legs and L1's and L2's own levels existed
    nowhere on the venue.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort(_pyramid_legs())
    eng = OneWayEmulator(store_ctx=ctx)
    _attach_pyramid_exits(eng, port)
    assert _venue_levels(port) == {"1": (1.30, 1.00), "2": (1.25, 1.05), "3": (1.20, 1.08)}
    assert [leg_id for leg_id, *_rest in port.amended] == ["1", "2", "3"]
    assert len(list(iter_active_bracket_ownerships(ctx))) == 3  # one row per exit
    store.close()


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
    assert _amended_levels(port) == [("3", None, None)]  # no owner left on leg 3
    assert {row.intent_key for row in iter_active_bracket_ownerships(ctx)} == {
        "L1-X\0L1", "L2-X\0L2"}
    # A restart re-asserts each remaining leg's own levels.
    port.amended.clear()
    assert _run(eng.restart_replay(port)) is True
    assert _venue_levels(port) == {"1": (1.30, 1.00), "2": (1.25, 1.05)}
    store.close()


def __test_tp_only_and_sl_only_exits_on_one_leg_union__(tmp_path):
    """A TP-only and an SL-only exit of one entry both stay on the leg; retiring one keeps the other."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("1", "buy", 1.0, entry_id="L1")])
    eng = OneWayEmulator(store_ctx=ctx)
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
    assert list(iter_active_bracket_ownerships(ctx)) == []
    store.close()


def __test_unattributed_legs_keep_the_single_exit_behaviour__(tmp_path):
    """A plugin that never attributes legs gets every exit replicated onto every leg."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    res = _run(eng.run_exit_bracket(_exit_env("L1-X", "L1", qty=1.0, tp=1.20, sl=1.00), port))
    assert res.legs == ("10", "11")
    assert _amended_levels(port) == [("10", 1.20, 1.00), ("11", 1.20, 1.00)]
    store.close()


def __test_two_exits_on_unattributed_legs_merge_to_the_more_protective_levels__(
        tmp_path, monkeypatch):
    """Conflicting levels on one leg give the level the price reaches first, reported once.

    For a long the highest stop-loss and the lowest take-profit, independent
    of the order the exits attached in; retiring the tighter exit hands the
    leg back to the remaining one.
    """
    warnings = _capture_warnings(monkeypatch)
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("L1-X", "L1", qty=1.0, tp=1.30, sl=1.00), port))
    _run(eng.run_exit_bracket(_exit_env("L2-X", "L2", qty=1.0, tp=1.25, sl=1.05), port))
    assert _venue_levels(port) == {"10": (1.25, 1.05)}
    # A re-dispatch of the looser exit does not loosen the leg.
    _run(eng.run_exit_bracket(_exit_env("L1-X", "L1", qty=1.0, tp=1.30, sl=1.00), port))
    assert _venue_levels(port) == {"10": (1.25, 1.05)}
    assert len([w for w in warnings if "different levels" in w]) == 1
    _run(eng.run_exit_bracket_clear(_cancel_env("L2-X", from_entry="L2"), port))
    assert _venue_levels(port) == {"10": (1.30, 1.00)}
    store.close()


def __test_merge_mirrors_for_a_short_and_keeps_the_tightest_trail__(tmp_path):
    """A short takes the lowest stop-loss and the highest take-profit; trailing takes the tightest."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("20", "sell", 1.0)])
    eng = OneWayEmulator(store_ctx=ctx, mintick=0.01)
    _run(eng.run_exit_bracket(
        _exit_env("S1-X", "S1", side="buy", qty=1.0, tp=0.90, sl=1.20, trail=20.0), port))
    _run(eng.run_exit_bracket(
        _exit_env("S2-X", "S2", side="buy", qty=1.0, tp=0.95, sl=1.15, trail=10.0), port))
    leg_id, side, tp, sl, trail, _coid = port.amended[-1]
    assert (leg_id, side, tp, sl, trail) == ("20", "buy", 0.95, 1.15, 0.1)
    store.close()


def __test_an_entry_without_a_leg_never_covers_other_entries_legs__(tmp_path, monkeypatch):
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
    store, ctx = _make_store(tmp_path)
    legs = [_leg("1", "buy", 1.0, open_time=1.0, entry_id="L1"),
            _leg("2", "buy", 1.0, open_time=2.0, entry_id="L2"),
            _leg("3", "buy", 1.0, open_time=3.0, entry_id="L3")]
    port = _FakePort(legs)
    eng = OneWayEmulator(store_ctx=ctx)
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
    # leg "3" is retired (once — the port treats the vanished leg as a no-op).
    assert _amended_levels(port) == [("3", None, None)]
    assert len([w for w in warnings if "attributed to an entry other than" in w]) == 1
    assert {row.intent_key for row in iter_active_bracket_ownerships(ctx)} == {"L1-X\0L1"}
    store.close()


def __test_an_entry_without_a_leg_retires_its_rows_from_legs_it_covered__(tmp_path):
    """The skip still retires the exit from a leg it covered before, keeping co-owners' levels.

    L3-X protected the unattributed leg "9" next to L1-X; once the plugin
    attributes "9" to L1, L3 holds no leg: L3-X's row comes off "9" and the
    leg is re-amended to L1-X's levels alone.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("9", "buy", 1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("L1-X", "L1", qty=1.0, tp=1.30, sl=1.00), port))
    _run(eng.run_exit_bracket(_exit_env("L3-X", "L3", qty=1.0, tp=1.20, sl=1.08), port))
    assert _venue_levels(port) == {"9": (1.20, 1.08)}
    port.set_legs([_leg("9", "buy", 1.0, entry_id="L1")])
    port.amended.clear()
    res = _run(eng.run_exit_bracket(_exit_env("L3-X", "L3", qty=1.0, tp=1.20, sl=1.08), port))
    assert res.skipped and res.entry_without_leg
    assert _amended_levels(port) == [("9", 1.30, 1.00)]
    assert {row.intent_key for row in iter_active_bracket_ownerships(ctx)} == {"L1-X\0L1"}
    store.close()


def __test_an_entry_without_an_attributed_leg_covers_the_unattributed_legs__(tmp_path):
    """With unattributed legs open, an exit without a leg of its own covers just those."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("1", "buy", 1.0, open_time=1.0, entry_id="L1"),
                      _leg("9", "buy", 1.0, open_time=2.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    res = _run(eng.run_exit_bracket(_exit_env("L3-X", "L3", qty=1.0, tp=1.20, sl=1.08), port))
    assert res.legs == ("9",)
    # An exit without a from_entry covers every survivor leg.
    res = _run(eng.run_exit_bracket(_exit_env("ALL", "", qty=2.0, sl=1.00), port))
    assert res.legs == ("1", "9")
    store.close()


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

    # The per-sync drain does the same for a retirement left clearing in-session.
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


def __test_a_dropped_leg_keeps_its_other_owners_levels__(tmp_path):
    """A leg leaving one exit's covered set is handed back to the exits still owning it."""
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "Long", sl=1.00), port))
    # An opposing leg nets "10" away; the TP exit re-attaches on "11" only.
    port.set_legs([_leg("10", "buy", 1.0, open_time=0.0),
                   _leg("11", "buy", 1.0, open_time=1.0),
                   _leg("20", "sell", 1.0, open_time=2.0)])
    port.amended.clear()
    res = _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.25), port))
    assert res.legs == ("11",)
    assert _amended_levels(port) == [("10", None, 1.00), ("11", 1.25, 1.00)]
    store.close()


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
                            trail_offset, coid) -> None:
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
            trail_offset=trail_offset, coid=coid,
        )


def _own_states(ctx) -> dict[str, str]:
    return {row.intent_key: row.extras[EXTRAS_KEY_BRACKET_OWN_STATE]
            for row in iter_active_bracket_ownerships(ctx)}


def __test_an_attach_dominated_by_a_co_owner_sends_no_amend__(tmp_path):
    """Attaching an exit whose levels a co-owner dominates changes nothing on the venue.

    The merge equals what the leg already holds, so no amend goes out (a
    venue refuses "nothing to amend", and a re-send resets a moved trailing
    stop); the attaching exit still owns the leg.
    """
    store, ctx = _make_store(tmp_path)
    port = _VenueRulesPort([_leg("10", "buy", 1.0)], reject_identical=True)
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("A", "L", tp=1.15), port))
    res = _run(eng.run_exit_bracket(_exit_env("B", "L", tp=1.20), port))
    assert res.legs == ("10",) and not res.skipped
    assert _amended_levels(port) == [("10", 1.15, None)]
    assert _own_states(ctx) == {"A\0L": "active", "B\0L": "active"}
    # Once A is retired, B's own level is what the leg must carry: sent.
    _run(eng.run_exit_bracket_clear(_cancel_env("A", from_entry="L"), port))
    assert _amended_levels(port) == [("10", 1.15, None), ("10", 1.20, None)]
    store.close()


def __test_retiring_a_dominated_owner_sends_no_amend__(tmp_path):
    """Retiring an exit whose levels were all dominated leaves the leg as it is, row released."""
    store, ctx = _make_store(tmp_path)
    port = _VenueRulesPort([_leg("10", "buy", 1.0)], reject_identical=True)
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("B", "L", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("A", "L", tp=1.15), port))
    port.amended.clear()
    res = _run(eng.run_exit_bracket_clear(_cancel_env("B", from_entry="L"), port))
    assert res.legs == ("10",)
    assert port.amended == []
    assert _own_states(ctx) == {"A\0L": "active"}
    store.close()


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
    assert _own_states(ctx) == {"TP\0L": BRACKET_OWN_STATE_CLEARING, "SL\0L": "active"}
    refusals = [w for w in warnings if "may still be armed" in w]
    assert len(refusals) == 1
    assert "TP" in refusals[0] and "stop distance too small" in refusals[0]

    assert _run(eng.drain_clearing_rows("EURUSD", port)) == set()
    restarted = OneWayEmulator(store_ctx=ctx)
    assert _run(restarted.restart_replay(port)) is True
    assert _run(restarted.drain_clearing_rows("EURUSD", port)) == set()
    assert _own_states(ctx) == {"TP\0L": BRACKET_OWN_STATE_CLEARING, "SL\0L": "active"}
    assert len([w for w in warnings if "may still be armed" in w]) == 2  # once per instance

    port.mid = 1.12
    port.amended.clear()
    assert _run(restarted.drain_clearing_rows("EURUSD", port)) == {"TP\0L"}
    assert _amended_levels(port) == [("10", None, 1.09)]
    assert _own_states(ctx) == {"SL\0L": "active"}
    store.close()


def __test_a_refused_retirement_does_not_stop_the_other_legs__(tmp_path):
    """A refusal on one leg keeps that leg clearing; the exit's other legs are still retired."""
    store, ctx = _make_store(tmp_path)
    port = _VenueRulesPort([_leg("10", "buy", 1.0, entry_id="L1"),
                            _leg("11", "buy", 1.0, entry_id="L2")],
                           mid=1.10, min_dist=0.005)
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "", qty=2.0, tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL1", "L1", qty=1.0, sl=1.09), port))
    port.mid = 1.093
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry=""), port))
    assert res.legs == ("11",)
    assert _amended_levels(port)[-1] == ("11", None, None)
    states = {(row.intent_key, row.extras[EXTRAS_KEY_BRACKET_OWN_LEG_ID]):
              row.extras[EXTRAS_KEY_BRACKET_OWN_STATE]
              for row in iter_active_bracket_ownerships(ctx)}
    assert states == {("TP\0", "10"): BRACKET_OWN_STATE_CLEARING,
                      ("SL1\0L1", "10"): "active"}
    store.close()


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
    assert set(_own_states(ctx).values()) == {BRACKET_OWN_STATE_CLEARING}
    assert _run(eng.drain_clearing_rows("EURUSD", port)) == set()
    assert _run(OneWayEmulator(store_ctx=ctx).restart_replay(port)) is True
    port.rate_limited = False
    assert _run(eng.drain_clearing_rows("EURUSD", port)) == {"TP\0L"}
    assert list(iter_active_bracket_ownerships(ctx)) == []
    store.close()


def __test_a_refused_retirement_of_a_dropped_leg_keeps_the_survivor_amend__(tmp_path):
    """A refused retirement on a dropped leg stays clearing and never fails the exit's own fan.

    The retirement concerns a leg the exit no longer covers; raising it as
    the exit's refusal would send the engine into the refused-exit recovery
    of a bracket the venue accepted on every leg it does cover.
    """
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=ctx)
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
              for row in iter_active_bracket_ownerships(ctx)}
    assert states == {"10": BRACKET_OWN_STATE_CLEARING, "11": "active"}
    store.close()


def __test_a_co_owned_attach_refusal_is_not_an_after_fill_flatten__(tmp_path, monkeypatch):
    """A refused attach on a leg another exit protects is the exit's refusal, not a flatten.

    The SL exit's 1.09 sits on the leg; the price drifts to 1.093 and the TP
    exit attaches: the venue refuses the merge (its stop rule). The leg still
    holds SL's level, so the engine must get the venue's plain reject (the
    refused-exit recovery: retry, then the defensive close) instead of
    :class:`BracketAttachAfterFillRejectedError` (an immediate flatten). The
    TP exit's row is released; SL's stays.
    """
    warnings = _capture_warnings(monkeypatch)
    store, ctx = _make_store(tmp_path)
    port = _VenueRulesPort([_leg("10", "buy", 1.0)], mid=1.10, min_dist=0.005)
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("SL", "L", sl=1.09), port))
    port.mid = 1.093
    with pytest.raises(ExchangeOrderRejectedError) as raised:
        _run(eng.run_exit_bracket(_exit_env("TP", "L", tp=1.20), port))
    assert not isinstance(raised.value, BracketAttachAfterFillRejectedError)
    assert "stop distance too small" in str(raised.value)
    assert _own_states(ctx) == {"SL\0L": "active"}
    assert len([w for w in warnings if "other exits also protect" in w]) == 1
    store.close()


def __test_a_co_owned_modify_refusal_restores_the_previous_row__(tmp_path):
    """A refused modify on a co-owned leg keeps the exit's previous levels as its row.

    The venue still holds the merge with the exit's previous levels, so the
    row must keep them (a later retirement must take them off the leg).
    """
    store, ctx = _make_store(tmp_path)
    port = _VenueRulesPort([_leg("10", "buy", 1.0)], mid=1.10, min_dist=0.005)
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_exit_bracket(_exit_env("TP", "L", tp=1.20), port))
    _run(eng.run_exit_bracket(_exit_env("SL", "L", sl=1.09), port))
    port.mid = 1.093
    with pytest.raises(ExchangeOrderRejectedError) as raised:
        _run(eng.run_exit_bracket(_exit_env("TP", "L", tp=1.25), port))
    assert not isinstance(raised.value, BracketAttachAfterFillRejectedError)
    rows = {row.intent_key: row for row in iter_active_bracket_ownerships(ctx)}
    assert rows["TP\0L"].extras[EXTRAS_KEY_BRACKET_OWN_STATE] == "active"
    assert rows["TP\0L"].extras[EXTRAS_KEY_BRACKET_OWN_TP] == 1.20
    store.close()


def __test_a_sole_owner_attach_refusal_keeps_the_after_fill_flatten__(tmp_path):
    """With no co-owner the refused attach still means an unprotected position."""
    store, ctx = _make_store(tmp_path)
    port = _VenueRulesPort([_leg("10", "buy", 1.0)], mid=1.093, min_dist=0.005)
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(BracketAttachAfterFillRejectedError):
        _run(eng.run_exit_bracket(_exit_env("SL", "L", sl=1.09), port))
    assert list(iter_active_bracket_ownerships(ctx)) == []
    store.close()


def __test_a_stop_and_a_trail_of_different_exits_are_reported_once__(tmp_path, monkeypatch):
    """A fixed stop of one exit and a trail of another cannot be carried as asked; WARN once."""
    warnings = _capture_warnings(monkeypatch)
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = OneWayEmulator(store_ctx=ctx, mintick=0.01)
    _run(eng.run_exit_bracket(_exit_env("ST", "L", sl=1.00), port))
    _run(eng.run_exit_bracket(_exit_env("TR", "L", trail=10.0), port))
    _run(eng.run_exit_bracket(_exit_env("TR", "L", trail=10.0), port))
    leg_id, _side, tp, sl, trail, _coid = port.amended[-1]
    assert (leg_id, tp, sl, trail) == ("10", None, 1.00, 0.1)
    assert len([w for w in warnings if "fixed stop-loss and a trailing stop" in w]) == 1
    store.close()


def __test_one_exit_asking_for_a_stop_and_a_trail_is_not_reported__(tmp_path, monkeypatch):
    warnings = _capture_warnings(monkeypatch)
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("10", "buy", 1.0)])
    eng = OneWayEmulator(store_ctx=ctx, mintick=0.01)
    _run(eng.run_exit_bracket(_exit_env("ST", "L", sl=1.00, trail=10.0), port))
    _run(eng.run_exit_bracket(_exit_env("TP", "L", tp=1.20), port))
    assert [w for w in warnings if "fixed stop-loss and a trailing stop" in w] == []
    store.close()


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
    _seed_residual(ctx, entry_coid="entry-missing", qty=2.0)
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
    eng = OneWayEmulator(store_ctx=ctx)
    with pytest.raises(OrderDispositionUnknownError):
        _run(eng.run_reversal(env, port))
    assert len(list(iter_active_residual_opens(ctx))) == 1
    store.close()


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
    # replay must FIFO-close every still-live opposing leg before the open.
    store, ctx = _make_store(tmp_path)
    _seed_residual(ctx, entry_coid="entry-missing", qty=1.0, side="buy")
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))
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
