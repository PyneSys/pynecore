"""
Unit + persistence tests for :class:`SoftwareStopLimitEngine`, the state
machine of a software stop-limit order (a Pine order with both ``stop`` and
``limit`` on a venue without a native stop-limit order).

The persistence tests use a real on-disk :class:`BrokerStore` and reopen the
run to prove every phase survives a restart.
"""
from pathlib import Path

import pytest

from pynecore.core.broker.software_stop_limit_engine import SoftwareStopLimitEngine
from pynecore.core.broker.storage import BrokerStore, RunContext, RunIdentity
from pynecore.core.broker.store_helpers import (
    EXTRAS_KEY_STOP_LIMIT_CHECKED_BAR,
    LEGACY_EXTRAS_KEY_OCO_LIMIT_COID,
    LEGACY_EXTRAS_KEY_OCO_MARKET_COID,
    LEGACY_OCO_STATE_MARKET_PENDING,
    STOP_LIMIT_STATE_ABORTED,
    STOP_LIMIT_STATE_ARMED,
    STOP_LIMIT_STATE_FILLED,
    STOP_LIMIT_STATE_LIMIT_PENDING,
    STOP_LIMIT_STATE_TRIGGERED,
    create_stop_limit_watch_row,
    update_stop_limit_watch,
)

BAR = 1_700_000_000_000
MINUTE = 60_000


def _arm(eng: SoftwareStopLimitEngine, *, side: str = "buy", stop: float = 1.18,
         limit: float = 1.16, bar: int = BAR):
    return eng.arm(coid="watch-Long", symbol="EURUSD", pine_id="Long", side=side,
                   qty=1.0, stop=stop, limit=limit, bar_ts_ms=bar)


def _identity() -> RunIdentity:
    return RunIdentity(strategy_id='stop-limit', symbol='EURUSD', timeframe='1',
                       account_id='test-account', label=None)


def _reopen(store: BrokerStore, ctx: RunContext) -> tuple[RunContext, SoftwareStopLimitEngine]:
    ctx.close()
    fresh = store.open_run(_identity(), script_source='// stop-limit')
    eng = SoftwareStopLimitEngine(store_ctx=fresh)
    eng.restart_replay()
    return fresh, eng


# === In-memory state machine ============================================

def __test_arm_registers_an_armed_watch__():
    """A fresh watch is ``armed`` and judged from its arming bar on."""
    eng = SoftwareStopLimitEngine(store_ctx=None)
    watch = _arm(eng)
    assert eng.get_watch("Long") is watch
    assert watch.state == STOP_LIMIT_STATE_ARMED
    assert (watch.stop, watch.limit) == (1.18, 1.16)
    assert watch.armed_bar_ts_ms == watch.checked_bar_ts_ms == BAR
    with pytest.raises(ValueError):
        _arm(eng)


@pytest.mark.parametrize("side,price,crossed", [
    ("buy", 1.1799, False), ("buy", 1.18, True), ("buy", 1.19, True),
    ("sell", 1.1801, False), ("sell", 1.18, True), ("sell", 1.17, True),
])
def __test_stop_crossed_follows_the_side__(side, price, crossed):
    """A buy stop triggers on a rise to it, a sell stop on a fall to it."""
    eng = SoftwareStopLimitEngine(store_ctx=None)
    watch = _arm(eng, side=side)
    assert SoftwareStopLimitEngine.stop_crossed(watch, price) is crossed


def __test_history_bar_after_the_judged_bar_triggers__():
    """A bar the engine did not see live whose range reached the stop triggers the order."""
    eng = SoftwareStopLimitEngine(store_ctx=None)
    watch = _arm(eng)
    assert eng.observe_history_bar(BAR + MINUTE, high=1.179, low=1.15) == []
    assert not watch.crossed_offline
    assert eng.observe_history_bar(BAR + 2 * MINUTE, high=1.181, low=1.17) == [watch]
    assert watch.crossed_offline


def __test_sell_history_bar_uses_the_low__():
    """A sell stop is judged against the bar's low."""
    eng = SoftwareStopLimitEngine(store_ctx=None)
    watch = _arm(eng, side="sell", stop=1.15, limit=1.17)
    assert eng.observe_history_bar(BAR + MINUTE, high=1.20, low=1.151) == []
    assert eng.observe_history_bar(BAR + 2 * MINUTE, high=1.20, low=1.15) == [watch]


def __test_the_arming_bar_is_not_judged_by_its_range__():
    """The arming bar's range holds ticks from before the order existed."""
    eng = SoftwareStopLimitEngine(store_ctx=None)
    watch = _arm(eng)
    assert eng.observe_history_bar(BAR, high=1.30, low=1.10) == []
    assert eng.observe_history_bar(BAR - MINUTE, high=1.30, low=1.10) == []
    assert not watch.crossed_offline


def __test_the_last_judged_bar_is_judged_again_when_armed_earlier__():
    """The process stopped inside the last judged bar: its later ticks were never seen."""
    eng = SoftwareStopLimitEngine(store_ctx=None)
    watch = _arm(eng)
    eng.note_judged("Long", BAR + MINUTE)
    assert eng.observe_history_bar(BAR + MINUTE, high=1.19, low=1.15) == [watch]


def __test_triggered_and_pending_watches_ignore_history_bars__():
    """Only an ``armed`` watch is judged; a triggered order already left the stop behind."""
    eng = SoftwareStopLimitEngine(store_ctx=None)
    _arm(eng)
    eng.begin_trigger("Long", order_coid="limit-1")
    assert eng.observe_history_bar(BAR + MINUTE, high=1.30, low=1.10) == []


def __test_limit_pending_is_never_aborted_blindly__():
    """A LIMIT in flight may have landed: only the send verification settles it."""
    eng = SoftwareStopLimitEngine(store_ctx=None)
    _arm(eng)
    eng.begin_trigger("Long", order_coid="limit-1")
    assert eng.mark_aborted("Long", reason="strategy_cancel") is None
    assert eng.get_watch("Long") is not None
    assert eng.abort_unsent("Long", reason="strategy_cancel") is not None
    assert eng.get_watch("Long") is None


def __test_transitions_follow_the_phases__():
    """armed -> limit_pending -> triggered -> filled, and nothing out of order."""
    eng = SoftwareStopLimitEngine(store_ctx=None)
    _arm(eng)
    assert eng.mark_triggered("Long") is None
    assert eng.mark_filled("Long") is None
    watch = eng.begin_trigger("Long", order_coid="limit-1")
    assert watch is not None and watch.state == STOP_LIMIT_STATE_LIMIT_PENDING
    assert eng.begin_trigger("Long", order_coid="limit-2") is None
    assert watch.order_coid == "limit-1"
    assert eng.rebind_order_coid("Long", order_coid="limit-2") is watch
    assert watch.order_coid == "limit-2"
    assert eng.mark_triggered("Long") is watch
    assert watch.state == STOP_LIMIT_STATE_TRIGGERED
    assert eng.mark_filled("Long") is watch
    assert eng.get_watch("Long") is None


def __test_amend_follows_the_phase__():
    """Armed: every term follows. Triggered: the stop has done its job."""
    eng = SoftwareStopLimitEngine(store_ctx=None)
    watch = _arm(eng)
    eng.amend("Long", stop=1.19, limit=1.17, qty=2.0, side="buy")
    assert (watch.stop, watch.limit, watch.qty) == (1.19, 1.17, 2.0)
    eng.begin_trigger("Long", order_coid="limit-1")
    eng.amend("Long", stop=1.25, limit=1.18, qty=3.0, side="buy")
    assert (watch.stop, watch.limit, watch.qty) == (1.19, 1.17, 2.0)
    eng.mark_triggered("Long")
    eng.amend("Long", stop=1.25, limit=1.18, qty=3.0, side="buy")
    assert (watch.stop, watch.limit, watch.qty) == (1.19, 1.18, 3.0)


def __test_fill_while_armed_adopts_the_order__():
    """The invariant breach takes the filling order as the watch's LIMIT."""
    eng = SoftwareStopLimitEngine(store_ctx=None)
    watch = _arm(eng)
    assert eng.adopt_fill_while_armed("Long") is watch
    assert watch.state == STOP_LIMIT_STATE_TRIGGERED
    assert eng.begin_trigger("Long", order_coid="limit-1") is None


# === Persistence ==========================================================

def __test_every_phase_survives_a_restart__(tmp_path: Path):
    """The watch row carries the phase, the prices, the judged bar and the LIMIT id."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = store.open_run(_identity(), script_source='// stop-limit')
        eng = SoftwareStopLimitEngine(store_ctx=ctx)
        _arm(eng)
        eng.note_judged("Long", BAR + 3 * MINUTE)

        ctx, eng = _reopen(store, ctx)
        watch = eng.get_watch("Long")
        assert watch is not None and not watch.legacy
        assert watch.state == STOP_LIMIT_STATE_ARMED
        assert (watch.stop, watch.limit, watch.side, watch.qty) == (1.18, 1.16, "buy", 1.0)
        assert watch.armed_bar_ts_ms == BAR
        assert watch.checked_bar_ts_ms == BAR + 3 * MINUTE

        eng.begin_trigger("Long", order_coid="limit-1")
        ctx, eng = _reopen(store, ctx)
        watch = eng.get_watch("Long")
        assert watch is not None
        assert (watch.state, watch.order_coid) == (STOP_LIMIT_STATE_LIMIT_PENDING, "limit-1")

        eng.mark_triggered("Long")
        eng.mark_filled("Long")
        row = ctx.get_order("watch-Long")
        assert row is not None and row.closed_ts_ms is not None
        assert (row.extras or {})["entry_stop_state"] == STOP_LIMIT_STATE_FILLED
        ctx, eng = _reopen(store, ctx)
        assert eng.get_watch("Long") is None
        ctx.close()


def __test_note_judged_persists_only_newer_bars__(tmp_path: Path):
    """The judged bar only moves forward."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = store.open_run(_identity(), script_source='// stop-limit')
        eng = SoftwareStopLimitEngine(store_ctx=ctx)
        _arm(eng)
        eng.note_judged("Long", BAR + 2 * MINUTE)
        eng.note_judged("Long", BAR + MINUTE)
        row = ctx.get_order("watch-Long")
        assert row is not None
        assert (row.extras or {})[EXTRAS_KEY_STOP_LIMIT_CHECKED_BAR] == BAR + 2 * MINUTE
        ctx.close()


def __test_a_legacy_oco_row_replays_as_legacy__(tmp_path: Path):
    """A row of the OCO model (a native LIMIT id on it) is only retired, never driven."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = store.open_run(_identity(), script_source='// stop-limit')
        create_stop_limit_watch_row(
            ctx, coid="watch-Long", symbol="EURUSD", side="buy", qty=1.0,
            intent_key="Long", pine_entry_id="Long", stop=1.18, limit=1.16,
            armed_bar_ts_ms=BAR)
        update_stop_limit_watch(ctx, coid="watch-Long",
                                extras_patch={LEGACY_EXTRAS_KEY_OCO_LIMIT_COID: "limit-coid"})
        ctx, eng = _reopen(store, ctx)
        watch = eng.get_watch("Long")
        assert watch is not None and watch.legacy
        assert eng.begin_trigger("Long", order_coid="x") is None
        assert eng.mark_aborted("Long", reason="x") is None
        assert eng.observe_history_bar(BAR + MINUTE, high=2.0, low=1.0) == []
        assert eng.retire_legacy("Long", reason="legacy_oco_limit_cancelled") is watch
        row = ctx.get_order("watch-Long")
        assert row is not None and row.closed_ts_ms is not None
        assert (row.extras or {})["entry_stop_state"] == STOP_LIMIT_STATE_ABORTED
        ctx.close()


def __test_a_legacy_market_pending_row_carries_its_market_id__(tmp_path: Path):
    """The OCO model's stop-fired MARKET phase replays with the MARKET's id."""
    with BrokerStore(tmp_path / 'broker.sqlite', plugin_name='Test') as store:
        ctx = store.open_run(_identity(), script_source='// stop-limit')
        create_stop_limit_watch_row(
            ctx, coid="watch-Long", symbol="EURUSD", side="buy", qty=1.0,
            intent_key="Long", pine_entry_id="Long", stop=1.18, limit=1.16,
            armed_bar_ts_ms=BAR)
        ctx.upsert_order("watch-Long", extras={
            **(ctx.get_order("watch-Long").extras or {}),  # type: ignore[union-attr]
            "entry_stop_state": LEGACY_OCO_STATE_MARKET_PENDING,
            LEGACY_EXTRAS_KEY_OCO_MARKET_COID: "market-coid",
        })
        ctx, eng = _reopen(store, ctx)
        watch = eng.get_watch("Long")
        assert watch is not None and watch.legacy
        assert watch.legacy_market_coid == "market-coid"
        ctx.close()
