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
    ExchangeOrderRejectedError,
    OrderDispositionUnknownError,
    OrderSkippedByPlugin,
)
from pynecore.core.broker.idempotency import KIND_CLOSE, KIND_ENTRY_STOP
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
    BracketFanResult,
    ClearFanResult,
    CloseFanResult,
    OneWayEmulator,
)
from pynecore.core.broker.storage import BrokerStore, RunContext, RunIdentity
from pynecore.core.broker.store_helpers import (
    BRACKET_OWN_STATE_CLEARING,
    EXTRAS_KEY_BRACKET_OWN_LEG_ID,
    EXTRAS_KEY_BRACKET_OWN_STATE,
    EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID,
    create_close_leg_row,
    create_residual_open_row,
    iter_active_bracket_ownerships,
    iter_active_close_legs,
    iter_active_residual_opens,
)


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
        # Per-leg ambiguous-timeout: amend_bracket for THIS leg raises
        # OrderDispositionUnknownError, modelling an independent broker round-trip
        # that times out while the other legs amend cleanly.
        self.fail_amend_unknown_leg = fail_amend_unknown_leg
        # Toggle: when True, every amend_bracket raises the ambiguous-timeout
        # error, modelling a clear whose disposition the broker never confirmed.
        self.fail_amend_unknown = False
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
        return lambda u: int(u)

    async def close_leg(self, symbol: str, leg_id: str, volume: int, coid: str) -> None:
        if self._fail_close_leg is not None:
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


def _leg(leg_id, side, qty, *, symbol="EURUSD", open_time=0.0, price=1.10) -> PositionLeg:
    return PositionLeg(
        leg_id=leg_id, symbol=symbol, side=side, qty=qty,
        entry_price=price, open_time=open_time, unrealized_pnl=0.0,
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
    assert res == CloseFanResult(legs=(), shortfall=0.0, skipped=False)
    assert port.closed == []


def __test_run_close_below_grid_skips_without_order__():
    """A close below the broker minimum is skipped with no order dispatched."""
    port = _FakePort([_leg("10", "buy", 0.4, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 0.4), port))
    assert res.skipped is True
    assert port.closed == []


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


# === run_reversal — decomposition =======================================

def _entry_env(side, qty, *, symbol="EURUSD", pine_id="Long",
               order_type=OrderType.MARKET, limit=None, stop=None,
               stop_fired=False) -> DispatchEnvelope:
    intent = EntryIntent(
        pine_id=pine_id, symbol=symbol, side=side, qty=qty,
        order_type=order_type, limit=limit, stop=stop,
        stop_fired_market=stop_fired,
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
              tp=None, sl=None, trail=None) -> DispatchEnvelope:
    intent = ExitIntent(
        pine_id=pine_id, from_entry=from_entry, symbol=symbol, side=side, qty=qty,
        tp_price=tp, sl_price=sl, trail_offset=trail,
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


def __test_run_exit_bracket_flat_skips__():
    """``run_exit_bracket`` on a flat book skips, amending no legs."""
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_exit_bracket(_exit_env("TP", "Long", sl=1.00), port))
    assert res == BracketFanResult(legs=(), skipped=True)
    assert port.amended == []


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
    # The clearing amends wipe the whole protection set on each owned leg.
    assert all(tp is None and sl is None and trail is None
               for _leg, _side, tp, sl, trail, _coid in port.amended)
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
    # F: _replay_bracket_one re-clears a "clearing" row at startup. If that
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
    _run(eng.drain_residual_opens("EURUSD", port2))
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
    """A stop-fired reversal's breadcrumb entry coid uses KIND_ENTRY_STOP, so replay finds the row."""
    # A stop-fired market reversal dispatches its residual under KIND_ENTRY_STOP
    # (the plugin picks the kind from stop_fired_market). The breadcrumb's
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
    stop_coid = env.client_order_id(KIND_ENTRY_STOP)
    assert (rows[0].extras or {})[EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID] == stop_coid
    # The ambiguous send actually landed: the plugin's persist-first entry row
    # exists under the KIND_ENTRY_STOP coid. Replay must find it and discharge
    # the breadcrumb WITHOUT re-opening the residual.
    ctx.upsert_order(stop_coid, symbol="EURUSD", side="buy", qty=1.0,
                     state="confirmed", pine_entry_id="Long")
    replay_port = _FakePort([])
    _run(eng.restart_replay(replay_port))
    assert replay_port.placed == []
    assert list(iter_active_residual_opens(ctx)) == []
    store.close()


def __test_replay_restores_stop_fired_flag_on_residual_redispatch__(tmp_path):
    """Residual replay re-dispatches a stop-fired reversal with stop_fired_market restored."""
    # The re-dispatched place_leg must persist and dedup under the SAME
    # KIND_ENTRY_STOP coid as the original attempt, so the rebuilt intent needs
    # the stop_fired_market flag back — recovered from the persisted coid's
    # kind code.
    class _CapturingPort(_FakePort):
        def __init__(self, legs):
            super().__init__(legs)
            self.place_envelopes: list[DispatchEnvelope] = []

        async def place_leg(self, envelope, qty):
            self.place_envelopes.append(envelope)
            return await super().place_leg(envelope, qty)

    store, ctx = _make_store(tmp_path)
    entry_coid = _entry_env("buy", 2.0).client_order_id(KIND_ENTRY_STOP)
    _seed_residual(ctx, entry_coid=entry_coid, qty=2.0)
    port = _CapturingPort([])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.restart_replay(port))
    assert port.placed == [("EURUSD", 2.0)]
    intent = port.place_envelopes[0].intent
    assert isinstance(intent, EntryIntent) and intent.stop_fired_market is True
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
    _run(eng.drain_residual_opens("EURUSD", healthy))
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
    _run(eng.drain_residual_opens("EURUSD", port))  # no exception escapes
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
    _run(eng.drain_residual_opens("EURUSD", port))
    assert port.placed == []  # entry path owns the open; no re-dispatch
    assert list(iter_active_residual_opens(ctx)) == []  # breadcrumb discharged
    store.close()


def __test_drain_residual_opens_is_symbol_scoped__(tmp_path):
    """The per-sync drain is symbol-scoped, leaving a different symbol's breadcrumb untouched."""
    # The per-sync drain is per-symbol (the engine owns one symbol). A live
    # breadcrumb for a DIFFERENT symbol must be left untouched — that symbol's own
    # engine drains it. (restart_replay is global; the per-sync drain is scoped.)
    store, ctx = _make_store(tmp_path)
    create_residual_open_row(
        ctx, coid="rev:residual", symbol="GBPUSD", side="buy", qty=2.0,
        intent_key="Long", pine_entry_id="Long", entry_coid="entry-missing",
        run_tag="t000", bar_ts_ms=1000, retry_seq=0,
    )
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.drain_residual_opens("EURUSD", port))  # different symbol
    assert port.placed == []  # the GBPUSD breadcrumb is not this symbol's concern
    assert len(list(iter_active_residual_opens(ctx))) == 1  # still live
    store.close()
