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
    OrderSkippedByPlugin,
)
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
    EXTRAS_KEY_BRACKET_OWN_LEG_ID,
    create_close_leg_row,
    iter_active_bracket_ownerships,
    iter_active_close_legs,
)


class _FakePort:
    """Minimal :class:`PositionPort`: canned legs, ``int`` quantizer, records
    every ``close_leg`` call as ``(symbol, leg_id, volume, coid)``."""

    def __init__(
            self, legs: list[PositionLeg], *,
            min_qty: float = 0.0, max_qty: float | None = None,
            fail_amend_leg: str | None = None,
    ) -> None:
        self._legs = list(legs)
        self._min_qty = min_qty
        self._max_qty = max_qty
        self._fail_amend_leg = fail_amend_leg
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
        self.closed.append((symbol, leg_id, volume, coid))

    async def reject_out_of_range(self, envelope: DispatchEnvelope, qty: float) -> None:
        if qty < self._min_qty or (self._max_qty is not None and qty > self._max_qty):
            raise OrderSkippedByPlugin(
                f"size {qty} out of range",
                intent_key=envelope.intent.intent_key,
                reason="out_of_range",
            )

    async def place_leg(self, envelope: DispatchEnvelope, qty: float):
        self.placed.append((envelope.intent.symbol, qty))
        return ["order"]

    async def amend_bracket(
            self, symbol: str, leg_id: str, *,
            side: str, tp_price: float | None, sl_price: float | None,
            trail_offset: float | None, coid: str,
    ) -> None:
        if leg_id == self._fail_amend_leg:
            raise ExchangeOrderRejectedError(f"amend rejected for leg {leg_id}")
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
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 3.0), port))
    assert res.legs == (("10", 2), ("11", 1))
    assert _dispatched(port) == [("10", 2), ("11", 1)]
    assert res.shortfall == 0.0 and res.skipped is False


def __test_run_close_partial_targets_oldest_leg__():
    port = _FakePort([_leg("10", "buy", 3.0, open_time=0.0),
                      _leg("11", "buy", 2.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 2.0), port))
    assert _dispatched(port) == [("10", 2)]
    assert res.legs == (("10", 2),)


def __test_run_close_short_reduces_sell_legs__():
    # CloseIntent.side='buy' closes a short -> the legs to reduce are the sells.
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    _run(eng.run_close(_close_env("buy", 2.0), port))
    assert _dispatched(port) == [("20", 2)]


def __test_run_close_flat_is_noop__():
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 1.0), port))
    assert res == CloseFanResult(legs=(), shortfall=0.0, skipped=False)
    assert port.closed == []


def __test_run_close_below_grid_skips_without_order__():
    port = _FakePort([_leg("10", "buy", 0.4, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 0.4), port))
    assert res.skipped is True
    assert port.closed == []


def __test_run_close_shortfall_surfaced_not_halted__():
    # Broker holds less than Pine believes: close what is open, report the rest.
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_close(_close_env("sell", 3.0), port))
    assert _dispatched(port) == [("10", 1)]
    assert res.shortfall == 2.0 and res.skipped is False


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
               order_type=OrderType.MARKET, limit=None, stop=None) -> DispatchEnvelope:
    intent = EntryIntent(
        pine_id=pine_id, symbol=symbol, side=side, qty=qty,
        order_type=order_type, limit=limit, stop=stop,
    )
    return DispatchEnvelope(intent=intent, run_tag="t000", bar_ts_ms=1000)


def __test_run_reversal_pure_add_opens_only__():
    # No opposing legs -> nothing to close, open the whole size.
    port = _FakePort([_leg("10", "buy", 1.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_reversal(_entry_env("buy", 1.0), port))
    assert port.closed == []
    assert port.placed == [("EURUSD", 1.0)]
    assert res.open_qty == 1.0 and len(res.opened_orders) == 1


def __test_run_reversal_flip_closes_opposing_then_opens_residual__():
    # Short 2.0, buy 3.0 -> close the 2.0 sell leg, open the 1.0 long residual.
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert _dispatched(port) == [("20", 2)]
    assert port.placed == [("EURUSD", 1.0)]
    assert res.closes == (("20", 2),) and res.open_qty == 1.0


def __test_run_reversal_exact_flatten_opens_nothing__():
    # Short 2.0, buy 2.0 -> close the sell leg, open nothing.
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_reversal(_entry_env("buy", 2.0), port))
    assert _dispatched(port) == [("20", 2)]
    assert port.placed == []
    assert res.open_qty == 0.0 and res.opened_orders == ()


def __test_run_reversal_below_min_residual_skips_whole_reversal__():
    # Residual 1.0 is below the broker minimum -> the pre-flight raises BEFORE
    # any close lands, so the book is never left half-reduced.
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)], min_qty=2.0)
    eng = OneWayEmulator(store_ctx=None)
    with pytest.raises(OrderSkippedByPlugin):
        _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert port.closed == [] and port.placed == []


def __test_run_reversal_persists_close_leg_rows__(tmp_path):
    store, ctx = _make_store(tmp_path)
    port = _FakePort([_leg("20", "sell", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=ctx)
    _run(eng.run_reversal(_entry_env("buy", 3.0), port))
    assert _dispatched(port) == [("20", 2)]
    assert port.placed == [("EURUSD", 1.0)]
    assert list(iter_active_close_legs(ctx)) == []  # close-leg row finalised
    store.close()


def __test_run_reversal_resting_limit_rests_without_closing_opposing__():
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
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0),
                      _leg("11", "buy", 1.0, open_time=1.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_exit_bracket(_exit_env("TP", "Long", tp=1.20, sl=1.00), port))
    assert res.legs == ("10", "11") and res.skipped is False
    # Same bracket on every position-side leg, FIFO order.
    assert _amended_levels(port) == [("10", 1.20, 1.00), ("11", 1.20, 1.00)]


def __test_run_exit_bracket_flat_skips__():
    port = _FakePort([])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_exit_bracket(_exit_env("TP", "Long", sl=1.00), port))
    assert res == BracketFanResult(legs=(), skipped=True)
    assert port.amended == []


def __test_clear_only_owns__(tmp_path):
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
    # Ownership-aware clear needs the persisted index: store_ctx=None -> no-op.
    port = _FakePort([_leg("10", "buy", 2.0, open_time=0.0)])
    eng = OneWayEmulator(store_ctx=None)
    res = _run(eng.run_exit_bracket_clear(_cancel_env("TP", from_entry="Long"), port))
    assert res == ClearFanResult(legs=())
    assert port.amended == []


def __test_bracket_ownership_persist_and_replay__(tmp_path):
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


def __test_run_exit_bracket_midfan_reject_releases_only_unamended_row__(tmp_path):
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
