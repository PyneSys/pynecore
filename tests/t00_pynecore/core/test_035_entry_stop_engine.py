"""
Unit + persistence tests for :class:`SoftwareEntryStopEngine` — the software
state machine for the STOP leg of a both-set Pine entry
(``strategy.entry(limit=, stop=)``).

The first group exercises the in-memory state machine in isolation
(``store_ctx=None``); the second uses a real on-disk :class:`BrokerStore` to
verify the persisted watch rows survive a simulated restart, including the
latched ``cancel_pending`` / ``stop_market_pending`` intermediates that the
sync engine re-drives deterministically rather than re-evaluating by price.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pynecore.core.broker.software_entry_stop_engine import (
    EntryStopWatch,
    SoftwareEntryStopEngine,
)
from pynecore.core.broker.storage import BrokerStore, RunIdentity, RunContext
from pynecore.core.broker.store_helpers import (
    ENTRY_STOP_STATE_ARMED,
    ENTRY_STOP_STATE_CANCEL_PENDING,
    ENTRY_STOP_STATE_LIMIT_WON,
    ENTRY_STOP_STATE_MARKET_PENDING,
    ENTRY_STOP_STATE_STOP_WON,
    EXTRAS_KEY_ENTRY_STOP_MARKET_COID,
    EXTRAS_KEY_ENTRY_STOP_STATE,
    create_entry_stop_watch_row,
    iter_active_entry_stop_watches,
)


def _watch(
        *,
        pine_id="Long",
        side="buy",
        qty=1.0,
        stop_level=1.18,
        state=ENTRY_STOP_STATE_ARMED,
) -> EntryStopWatch:
    return EntryStopWatch(
        coid=f"watch-{pine_id}",
        symbol="EURUSD",
        pine_id=pine_id,
        side=side,
        qty=qty,
        stop_level=stop_level,
        limit_coid=f"limit-{pine_id}",
        state=state,
    )


# === In-memory state machine ============================================

def __test_register_and_query__():
    """A registered watch is found by ``has_watch``, ``get_watch`` and ``iter_watches``."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    w = _watch()
    eng.register_watch(w)
    assert eng.has_watch("Long")
    assert eng.get_watch("Long") is w
    assert list(eng.iter_watches()) == [w]


def __test_register_rejects_terminal_state__():
    """Registering a watch already in a terminal state raises ``ValueError``."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    with pytest.raises(ValueError):
        eng.register_watch(_watch(state=ENTRY_STOP_STATE_STOP_WON))


def __test_register_rejects_duplicate__():
    """Registering a second watch with the same pine id raises ``ValueError``."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    eng.register_watch(_watch())
    with pytest.raises(ValueError):
        eng.register_watch(_watch())


def __test_stop_crossed_long_fires_on_rise__():
    """A long watch's stop fires at or above its level, not below it."""
    # A long both-set entry's STOP sits ABOVE the open and fires when price
    # rises to it.
    w = _watch(side="buy", stop_level=1.18)
    assert SoftwareEntryStopEngine.stop_crossed(w, 1.18) is True
    assert SoftwareEntryStopEngine.stop_crossed(w, 1.19) is True
    assert SoftwareEntryStopEngine.stop_crossed(w, 1.17) is False


def __test_stop_crossed_short_fires_on_fall__():
    """A short watch's stop fires at or below its level, not above it."""
    # A short both-set entry's STOP sits BELOW the open and fires when price
    # falls to it.
    w = _watch(side="sell", stop_level=1.10)
    assert SoftwareEntryStopEngine.stop_crossed(w, 1.10) is True
    assert SoftwareEntryStopEngine.stop_crossed(w, 1.09) is True
    assert SoftwareEntryStopEngine.stop_crossed(w, 1.11) is False


def __test_begin_cancel_armed_to_cancel_pending__():
    """``begin_cancel`` moves an armed watch into ``cancel_pending``."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    eng.register_watch(_watch())
    w = eng.begin_cancel("Long")
    assert w is not None
    assert w.state == ENTRY_STOP_STATE_CANCEL_PENDING


def __test_begin_cancel_noop_when_not_armed__():
    """``begin_cancel`` returns ``None`` when the watch is not armed."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    eng.register_watch(_watch(state=ENTRY_STOP_STATE_CANCEL_PENDING))
    assert eng.begin_cancel("Long") is None


def __test_confirm_fire_persists_market_coid_and_advances__():
    """Confirming the cancel advances to ``market_pending`` and stores the market coid."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    eng.register_watch(_watch(state=ENTRY_STOP_STATE_CANCEL_PENDING))
    w = eng.confirm_limit_cancelled_fire_market("Long", market_coid="mkt-1")
    assert w is not None
    assert w.state == ENTRY_STOP_STATE_MARKET_PENDING
    assert w.market_coid == "mkt-1"


def __test_confirm_fire_noop_when_not_cancel_pending__():
    """Confirming the cancel returns ``None`` when the watch is not ``cancel_pending``."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    eng.register_watch(_watch())  # armed
    assert eng.confirm_limit_cancelled_fire_market(
        "Long", market_coid="mkt-1") is None


def __test_mark_stop_won_terminal_and_evicts__():
    """``mark_stop_won`` settles a ``market_pending`` watch and evicts it from the ledger."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    eng.register_watch(_watch(state=ENTRY_STOP_STATE_MARKET_PENDING))
    w = eng.mark_stop_won("Long")
    assert w is not None and w.state == ENTRY_STOP_STATE_STOP_WON
    assert eng.get_watch("Long") is None  # evicted from the ledger


def __test_mark_stop_won_noop_when_not_market_pending__():
    """``mark_stop_won`` is a no-op on an armed watch and leaves it tracked."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    eng.register_watch(_watch())  # armed
    assert eng.mark_stop_won("Long") is None
    assert eng.has_watch("Long")  # still tracked


def __test_mark_limit_won_from_armed_terminal__():
    """``mark_limit_won`` settles an armed watch to ``limit_won`` and evicts it."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    eng.register_watch(_watch())
    w = eng.mark_limit_won("Long", reason="native_limit_filled")
    assert w is not None and w.state == ENTRY_STOP_STATE_LIMIT_WON
    assert eng.get_watch("Long") is None


def __test_mark_limit_won_from_cancel_pending_terminal__():
    """A limit filled during ``cancel_pending`` wins the OCO: watch settles to ``limit_won``."""
    # The hard gate: a LIMIT that filled while we were cancelling it wins the
    # OCO; no market may fire.
    eng = SoftwareEntryStopEngine(store_ctx=None)
    eng.register_watch(_watch(state=ENTRY_STOP_STATE_CANCEL_PENDING))
    w = eng.mark_limit_won("Long", reason="limit_filled_during_stop_cancel")
    assert w is not None and w.state == ENTRY_STOP_STATE_LIMIT_WON
    assert eng.get_watch("Long") is None


def __test_mark_aborted_terminal__():
    """``mark_aborted`` settles an armed watch and evicts it from the ledger."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    eng.register_watch(_watch())
    w = eng.mark_aborted("Long", reason="strategy_cancel")
    assert w is not None
    assert eng.get_watch("Long") is None


def __test_mark_aborted_noop_once_committed_to_stop__():
    """Once in ``market_pending``, ``mark_aborted`` is a no-op and the watch still stop-wins."""
    # Once the watch has committed to the stop side (stop_market_pending), a
    # delayed broker cancelled/rejected ack for the OCO's own now-cancelled
    # native LIMIT must NOT abort it — that echo would evict a watch whose
    # market is already in flight and drop the verify-before-resend net on
    # restart. mark_aborted is an idempotent no-op here, and mark_stop_won
    # still settles the watch afterwards.
    eng = SoftwareEntryStopEngine(store_ctx=None)
    eng.register_watch(_watch())
    eng.begin_cancel("Long")
    eng.confirm_limit_cancelled_fire_market("Long", market_coid="mkt-1")
    assert eng.get_watch("Long").state == ENTRY_STOP_STATE_MARKET_PENDING

    assert eng.mark_aborted("Long", reason="parent_cancelled") is None
    w = eng.get_watch("Long")
    assert w is not None and w.state == ENTRY_STOP_STATE_MARKET_PENDING

    settled = eng.mark_stop_won("Long")
    assert settled is not None and settled.state == ENTRY_STOP_STATE_STOP_WON
    assert eng.get_watch("Long") is None


def __test_mark_helpers_noop_when_absent__():
    """All ``mark_*`` helpers return ``None`` for an unknown pine id."""
    eng = SoftwareEntryStopEngine(store_ctx=None)
    assert eng.mark_limit_won("nope", reason="x") is None
    assert eng.mark_aborted("nope", reason="x") is None
    assert eng.mark_stop_won("nope") is None


# === Persistence + restart replay =======================================

def _identity() -> RunIdentity:
    return RunIdentity(
        strategy_id="entry-stop-test",
        symbol="EURUSD",
        timeframe="60",
        account_id="acc-1",
    )


def _make_store(tmp_path: Path) -> tuple[BrokerStore, RunContext]:
    store = BrokerStore(tmp_path / "broker.db", plugin_name="capitalcom")
    ctx = store.open_run(_identity(), script_source="// test")
    return store, ctx


def __test_persist_and_replay_armed_watch__(tmp_path):
    """Replay restores a persisted armed watch with its stop level and limit coid intact."""
    store, ctx = _make_store(tmp_path)
    create_entry_stop_watch_row(
        ctx,
        coid="watch-Long",
        symbol="EURUSD",
        side="buy",
        qty=1.0,
        intent_key="Long",
        pine_entry_id="Long",
        stop_level=1.18,
        limit_coid="limit-Long",
    )
    eng = SoftwareEntryStopEngine(store_ctx=ctx)
    eng.restart_replay()
    watches = list(eng.iter_watches())
    assert len(watches) == 1
    w = watches[0]
    assert w.pine_id == "Long"
    assert w.state == ENTRY_STOP_STATE_ARMED
    assert w.stop_level == 1.18
    assert w.limit_coid == "limit-Long"
    store.close()


def __test_latched_cancel_pending_survives_replay__(tmp_path):
    """A latched ``cancel_pending`` watch survives replay rather than re-arming."""
    # A crash after the stop crossed but before the cancel resolved must
    # resume in cancel_pending (NOT re-armed) so the sync engine re-drives the
    # idempotent cancel-then-fire gate regardless of the current price.
    store, ctx = _make_store(tmp_path)
    eng = SoftwareEntryStopEngine(store_ctx=ctx)
    create_entry_stop_watch_row(
        ctx, coid="watch-Long", symbol="EURUSD", side="buy", qty=1.0,
        intent_key="Long", pine_entry_id="Long", stop_level=1.18,
        limit_coid="limit-Long",
    )
    eng.register_watch(_watch())
    eng.begin_cancel("Long")
    # Fresh engine replays from disk.
    eng2 = SoftwareEntryStopEngine(store_ctx=ctx)
    eng2.restart_replay()
    w = eng2.get_watch("Long")
    assert w is not None
    assert w.state == ENTRY_STOP_STATE_CANCEL_PENDING
    store.close()


def __test_latched_market_pending_survives_replay_with_coid__(tmp_path):
    """A latched ``market_pending`` watch survives replay carrying its market coid."""
    # A crash after the cancel was confirmed but before the market settled
    # must resume in stop_market_pending carrying the deterministic market
    # coid, so the re-fire reuses the same id and the broker dedups it.
    store, ctx = _make_store(tmp_path)
    eng = SoftwareEntryStopEngine(store_ctx=ctx)
    create_entry_stop_watch_row(
        ctx, coid="watch-Long", symbol="EURUSD", side="buy", qty=1.0,
        intent_key="Long", pine_entry_id="Long", stop_level=1.18,
        limit_coid="limit-Long",
    )
    eng.register_watch(_watch())
    eng.begin_cancel("Long")
    eng.confirm_limit_cancelled_fire_market("Long", market_coid="mkt-1")
    eng2 = SoftwareEntryStopEngine(store_ctx=ctx)
    eng2.restart_replay()
    w = eng2.get_watch("Long")
    assert w is not None
    assert w.state == ENTRY_STOP_STATE_MARKET_PENDING
    assert w.market_coid == "mkt-1"
    store.close()


def __test_terminal_watch_filtered_on_replay__(tmp_path):
    """A settled watch closes its row and is not resurrected by replay."""
    # A won/aborted watch closes its row; replay must not resurrect it.
    store, ctx = _make_store(tmp_path)
    eng = SoftwareEntryStopEngine(store_ctx=ctx)
    create_entry_stop_watch_row(
        ctx, coid="watch-Long", symbol="EURUSD", side="buy", qty=1.0,
        intent_key="Long", pine_entry_id="Long", stop_level=1.18,
        limit_coid="limit-Long",
    )
    eng.register_watch(_watch())
    eng.mark_limit_won("Long", reason="native_limit_filled")
    # The row is closed; the active iterator yields nothing.
    assert list(iter_active_entry_stop_watches(ctx)) == []
    eng2 = SoftwareEntryStopEngine(store_ctx=ctx)
    eng2.restart_replay()
    assert list(eng2.iter_watches()) == []
    store.close()


def __test_persisted_state_string_matches__(tmp_path):
    """The persisted extras carry the armed state string and a null market coid."""
    # Guard the persisted extras key/value so a rename can't silently break
    # restart replay.
    store, ctx = _make_store(tmp_path)
    create_entry_stop_watch_row(
        ctx, coid="watch-Long", symbol="EURUSD", side="buy", qty=1.0,
        intent_key="Long", pine_entry_id="Long", stop_level=1.18,
        limit_coid="limit-Long",
    )
    rows = list(iter_active_entry_stop_watches(ctx))
    assert len(rows) == 1
    extras = rows[0].extras or {}
    assert extras[EXTRAS_KEY_ENTRY_STOP_STATE] == ENTRY_STOP_STATE_ARMED
    assert extras[EXTRAS_KEY_ENTRY_STOP_MARKET_COID] is None
    store.close()
