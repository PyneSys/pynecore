"""
Unit tests for :class:`NativeFailsafeManager` — the §2.6 / §2.6.7
broker-native fail-safe worst-SL state machine.

The manager is exercised in isolation: no sync engine, no broker, no
storage. Time is driven by explicit ``now_ms`` arguments; the manager
itself has no clock. The dispatch hook is a list-appender so each test
sees the snapshots the manager would have asked the plugin to PUT.
"""
from typing import Any

import pytest

from pynecore.core.broker.models import (
    BrokerEvent,
    BrokerNativeFailsafeExternalEditEvent,
    BrokerNativeFailsafeUnavailableEvent,
    EntryBlockedDegradedFailsafeEvent,
    EntrySkippedDueToDegradedFailsafeEvent,
    NativeFailsafeStateTransitionEvent,
    PartialBracketBlockedDegradedFailsafeEvent,
)
from pynecore.core.broker.native_failsafe_manager import (
    FailsafeHealth,
    FailsafeOwner,
    NativeBracketSnapshot,
    NativeFailsafeManager,
)


SYMBOL = "BTCUSDT"
PARENT_REF = "run0-pi00cafe-bar000abcd-e0"
MINTICK = 0.5


def _make_manager(**kwargs: Any) -> tuple[NativeFailsafeManager, list[BrokerEvent]]:
    events: list[BrokerEvent] = []
    manager = NativeFailsafeManager(event_sink=events.append, **kwargs)
    return manager, events


def _register_long(
        manager: NativeFailsafeManager,
        *,
        ref: str = PARENT_REF,
        symbol: str = SYMBOL,
        mintick: float = MINTICK,
) -> None:
    manager.register_parent(
        parent_entry_dispatch_ref=ref,
        symbol=symbol,
        parent_side='long',
        mintick=mintick,
    )


# ---------------------------------------------------------------------
# Registration & ownership
# ---------------------------------------------------------------------

def __test_register_parent_starts_healthy_with_engine_failsafe_ownership__():
    manager, _ = _make_manager()
    _register_long(manager)
    state = manager.get_state(PARENT_REF)
    assert state is not None
    assert state.health is FailsafeHealth.HEALTHY
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE


def __test_register_parent_is_idempotent_for_same_ref__():
    manager, _ = _make_manager()
    _register_long(manager)
    first = manager.get_state(PARENT_REF)
    manager.register_parent(
        parent_entry_dispatch_ref=PARENT_REF,
        symbol=SYMBOL,
        parent_side='long',
        mintick=MINTICK,
        initial_profit_level=110.0,
    )
    second = manager.get_state(PARENT_REF)
    assert first is second  # same object — state preserved
    assert second.desired_profit_level == 110.0


def __test_invalid_parent_side_rejected__():
    manager, _ = _make_manager()
    with pytest.raises(ValueError):
        manager.register_parent(
            parent_entry_dispatch_ref=PARENT_REF,
            symbol=SYMBOL,
            parent_side='wrong',
            mintick=MINTICK,
        )


# ---------------------------------------------------------------------
# Worst-SL computation
# ---------------------------------------------------------------------

def __test_long_worst_sl_is_minimum_of_active_levels__():
    """§2.6.3: long parent worst-SL = min(level) across active SL legs."""
    manager, _ = _make_manager()
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[95.0, 90.0, 92.0],
        now_ms=1000.0,
    )
    assert snap is not None
    assert snap.stop_level == 90.0
    assert manager.get_state(PARENT_REF).generation == 1


def __test_short_worst_sl_is_maximum_of_active_levels__():
    manager, _ = _make_manager()
    manager.register_parent(
        parent_entry_dispatch_ref=PARENT_REF,
        symbol=SYMBOL,
        parent_side='short',
        mintick=MINTICK,
    )
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[110.0, 120.0, 115.0],
        now_ms=1000.0,
    )
    assert snap is not None
    assert snap.stop_level == 120.0


def __test_empty_sl_set_queues_clear_snapshot__():
    """§2.6.3 #4: last SL leg drops out → PUT stopLevel=None."""
    manager, _ = _make_manager()
    _register_long(manager)
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[],
        now_ms=2000.0,
    )
    assert snap is not None
    assert snap.stop_level is None
    assert manager.get_state(PARENT_REF).desired_level is None


def __test_no_change_in_worst_sl_is_no_op__():
    """Same worst-SL value across recomputes → no new dispatch queued."""
    manager, _ = _make_manager()
    _register_long(manager)
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0, 95.0],
        now_ms=1000.0,
    )
    assert len(manager.pending_dispatch()) == 1
    # Re-dispatch the same level set: generation must NOT bump, no new pending.
    state_gen = manager.get_state(PARENT_REF).generation
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0, 95.0],
        now_ms=2000.0,
    )
    assert snap is None
    assert manager.get_state(PARENT_REF).generation == state_gen


def __test_first_sl_leg_arm_queues_dispatch__():
    """§2.6.3 #1: a new armed SL leg seeds the worst-SL dispatch."""
    manager, _ = _make_manager()
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    assert snap is not None
    pending = manager.pending_dispatch()
    assert len(pending) == 1
    assert pending[0].stop_level == 90.0
    assert pending[0].parent_entry_dispatch_ref == PARENT_REF


def __test_worst_sl_jumps_on_intermediate_leg_triggered__():
    """§2.6.3 #4: triggered leg drops out → worst-SL jumps to next leg."""
    manager, _ = _make_manager()
    _register_long(manager)
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[95.0, 90.0],
        now_ms=1000.0,
    )
    # Worst is 90. Now 90 triggers; only 95 remains.
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[95.0],
        now_ms=2000.0,
    )
    assert snap is not None
    assert snap.stop_level == 95.0


# ---------------------------------------------------------------------
# Full bracket snapshot (§2.6.6 / §9 #19 Exp D2)
# ---------------------------------------------------------------------

def __test_snapshot_carries_coexisting_tp_and_trailing__():
    """Full-replacement PUT semantics: every snapshot includes the
    desired TP and trailing-stop fields so they survive the worst-SL
    update."""
    manager, _ = _make_manager()
    manager.register_parent(
        parent_entry_dispatch_ref=PARENT_REF,
        symbol=SYMBOL,
        parent_side='long',
        mintick=MINTICK,
        initial_profit_level=110.0,
        initial_trailing_stop=2.5,
    )
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    assert snap is not None
    assert snap.profit_level == 110.0
    assert snap.trailing_stop == 2.5
    assert snap.stop_level == 90.0


# ---------------------------------------------------------------------
# PUT result handling — happy path
# ---------------------------------------------------------------------

def __test_record_put_success_clears_pending_for_matching_generation__():
    manager, _ = _make_manager()
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    assert snap is not None
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=1020.0)
    assert manager.pending_dispatch() == []
    state = manager.get_state(PARENT_REF)
    assert state.pending_put is False


def __test_record_put_success_stale_generation_keeps_newer_pending__():
    """A newer desired snapshot queued while the older PUT was in flight
    must survive when the older PUT acks."""
    manager, _ = _make_manager()
    _register_long(manager)
    first = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    # Worst-SL moves while the first PUT is in flight.
    second = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[85.0],
        now_ms=1015.0,
    )
    assert second is not None
    assert second.generation > first.generation
    # First PUT acks — the newer pending must NOT be dropped.
    manager.record_put_success(PARENT_REF, generation=first.generation, now_ms=1020.0)
    pending = manager.pending_dispatch()
    assert len(pending) == 1
    assert pending[0].generation == second.generation


# ---------------------------------------------------------------------
# PUT failure → degrading → degraded
# ---------------------------------------------------------------------

def __test_put_failure_retry_budget_keeps_state_healthy__():
    """First N-1 failures stay in `healthy` (immediate retry budget)."""
    manager, _ = _make_manager(put_max_attempts=3)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.record_put_failure(
        PARENT_REF, generation=snap.generation, reason='rate-limit', now_ms=1010.0,
    )
    assert manager.get_state(PARENT_REF).health is FailsafeHealth.HEALTHY
    manager.record_put_failure(
        PARENT_REF, generation=snap.generation, reason='rate-limit', now_ms=1020.0,
    )
    assert manager.get_state(PARENT_REF).health is FailsafeHealth.HEALTHY


def __test_put_failure_exhausts_budget_promotes_to_degrading__():
    """N consecutive failures → state moves to `degrading`."""
    manager, events = _make_manager(put_max_attempts=3)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    for i in range(3):
        manager.record_put_failure(
            PARENT_REF, generation=snap.generation,
            reason='rate-limit', now_ms=1010.0 + i * 10,
        )
    assert manager.get_state(PARENT_REF).health is FailsafeHealth.DEGRADING
    transitions = [
        e for e in events if isinstance(e, NativeFailsafeStateTransitionEvent)
    ]
    assert any(t.to_state == 'degrading' for t in transitions)


def __test_degrading_blocks_new_partial_bracket__():
    """§2.6.7: degrading state rejects new partial brackets on the same parent."""
    manager, events = _make_manager(put_max_attempts=1)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.record_put_failure(
        PARENT_REF, generation=snap.generation, reason='rate-limit', now_ms=1010.0,
    )
    assert manager.get_state(PARENT_REF).health is FailsafeHealth.DEGRADING
    blocked = manager.block_new_partial_bracket(
        parent_entry_dispatch_ref=PARENT_REF,
        symbol=SYMBOL,
        pine_id='Long',
        from_entry='Long',
    )
    assert blocked is True
    blocks = [
        e for e in events
        if isinstance(e, PartialBracketBlockedDegradedFailsafeEvent)
    ]
    assert len(blocks) == 1
    assert blocks[0].health == 'degrading'


def __test_healthy_state_does_not_block__():
    manager, _ = _make_manager()
    _register_long(manager)
    blocked = manager.block_new_partial_bracket(
        parent_entry_dispatch_ref=PARENT_REF,
        symbol=SYMBOL,
        pine_id='Long',
        from_entry='Long',
    )
    assert blocked is False


def __test_stale_window_promotes_degrading_to_degraded__():
    """§2.6.7: degrading without confirm within stale window → degraded."""
    manager, events = _make_manager(put_max_attempts=1, stale_window_ms=5_000.0)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.record_put_failure(
        PARENT_REF, generation=snap.generation, reason='rate-limit', now_ms=1010.0,
    )
    assert manager.get_state(PARENT_REF).health is FailsafeHealth.DEGRADING

    # Within the window: no promotion.
    manager.tick_stale_window(now_ms=2000.0)
    assert manager.get_state(PARENT_REF).health is FailsafeHealth.DEGRADING

    # Past the window: degraded.
    manager.tick_stale_window(now_ms=10_000.0)
    assert manager.get_state(PARENT_REF).health is FailsafeHealth.DEGRADED
    unavail = [
        e for e in events
        if isinstance(e, BrokerNativeFailsafeUnavailableEvent)
    ]
    assert len(unavail) == 1


def __test_degrading_recovers_to_healthy_on_snapshot_confirm__():
    """§2.6.7: degrading + observed snapshot matching desired → healthy automatically."""
    manager, events = _make_manager(put_max_attempts=1)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.record_put_failure(
        PARENT_REF, generation=snap.generation, reason='rate-limit', now_ms=1010.0,
    )
    assert manager.get_state(PARENT_REF).health is FailsafeHealth.DEGRADING
    # PUT actually landed on the broker (eventual consistency); next
    # reconcile snapshot confirms.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=90.0, profit_level=None,
        trailing_stop=None, now_ms=2000.0,
    )
    assert manager.get_state(PARENT_REF).health is FailsafeHealth.HEALTHY


# ---------------------------------------------------------------------
# External edit detection (§2.6.7)
# ---------------------------------------------------------------------

def __test_external_stop_edit_flips_owner_to_unknown__():
    """Snapshot diff with no PUT in flight + mismatch → owner=unknown."""
    manager, events = _make_manager()
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=1020.0)
    # Now the user manually moves the broker stop to 85.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=85.0, profit_level=None,
        trailing_stop=None, now_ms=3000.0,
    )
    state = manager.get_state(PARENT_REF)
    assert state.owner is FailsafeOwner.UNKNOWN
    external = [
        e for e in events
        if isinstance(e, BrokerNativeFailsafeExternalEditEvent)
    ]
    assert len(external) == 1
    assert external[0].desired_level == 90.0
    assert external[0].actual_level == 85.0


def __test_owner_unknown_blocks_engine_recompute__():
    """Once owner is `unknown`, recompute_worst_sl is a no-op."""
    manager, _ = _make_manager()
    _register_long(manager)
    state = manager.get_state(PARENT_REF)
    state.owner = FailsafeOwner.UNKNOWN
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    assert snap is None
    assert state.desired_level is None


def __test_user_reset_restores_engine_ownership_and_requeues__():
    """User reset (set_risk(disable) + reset_to_engine) clears the owner
    back to engine-failsafe and re-queues the current desired level."""
    manager, _ = _make_manager()
    _register_long(manager)
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    state = manager.get_state(PARENT_REF)
    state.owner = FailsafeOwner.UNKNOWN
    manager.reset_to_engine(PARENT_REF, now_ms=5000.0)
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE
    assert state.health is FailsafeHealth.HEALTHY
    # Desired survives — a fresh snapshot is queued for re-dispatch.
    pending = manager.pending_dispatch()
    assert len(pending) == 1
    assert pending[0].stop_level == 90.0


# ---------------------------------------------------------------------
# Retire path (§2.6.7 lifecycle cleanup)
# ---------------------------------------------------------------------

def __test_deal_id_disappear_retires_state__():
    manager, events = _make_manager()
    _register_long(manager)
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.on_deal_id_disappeared(PARENT_REF, now_ms=5000.0)
    state = manager.get_state(PARENT_REF)
    assert state.health is FailsafeHealth.RETIRED
    transitions = [
        e for e in events if isinstance(e, NativeFailsafeStateTransitionEvent)
    ]
    assert transitions[-1].to_state == 'retired'
    # Idempotent: a second call is a no-op.
    manager.on_deal_id_disappeared(PARENT_REF, now_ms=6000.0)
    assert state.health is FailsafeHealth.RETIRED


def __test_retired_state_does_not_block_new_partial_bracket__():
    manager, _ = _make_manager()
    _register_long(manager)
    manager.on_deal_id_disappeared(PARENT_REF, now_ms=5000.0)
    blocked = manager.block_new_partial_bracket(
        parent_entry_dispatch_ref=PARENT_REF,
        symbol=SYMBOL,
        pine_id='Long',
        from_entry='Long',
    )
    assert blocked is False


# ---------------------------------------------------------------------
# Entry gate (§2.6.7) — drop-semantics
# ---------------------------------------------------------------------

def __test_new_entry_blocked_when_any_symbol_state_is_degrading__():
    manager, events = _make_manager(put_max_attempts=1)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.record_put_failure(
        PARENT_REF, generation=snap.generation, reason='rate-limit', now_ms=1010.0,
    )
    blocked = manager.block_new_entry(
        symbol=SYMBOL, pine_id='LongB', bar_ts_ms=1_700_000_000_000,
    )
    assert blocked is True
    blocks = [e for e in events if isinstance(e, EntryBlockedDegradedFailsafeEvent)]
    skipped = [e for e in events if isinstance(e, EntrySkippedDueToDegradedFailsafeEvent)]
    assert len(blocks) == 1
    assert len(skipped) == 1
    assert skipped[0].symbol == SYMBOL


def __test_new_entry_on_different_symbol_not_blocked__():
    manager, _ = _make_manager(put_max_attempts=1)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.record_put_failure(
        PARENT_REF, generation=snap.generation, reason='rate-limit', now_ms=1010.0,
    )
    blocked = manager.block_new_entry(
        symbol='ETHUSDT', pine_id='Long', bar_ts_ms=1_700_000_000_000,
    )
    assert blocked is False


def __test_entry_gate_picks_worst_health_per_symbol__():
    """When several parents on the same symbol are degraded, the block
    event surfaces `degraded`, not `degrading`."""
    manager, events = _make_manager(put_max_attempts=1, stale_window_ms=1_000.0)
    _register_long(manager, ref='ref-1')
    _register_long(manager, ref='ref-2')
    snap1 = manager.recompute_worst_sl(
        parent_entry_dispatch_ref='ref-1',
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.record_put_failure(
        'ref-1', generation=snap1.generation, reason='r', now_ms=1010.0,
    )
    manager.tick_stale_window(now_ms=10_000.0)
    assert manager.get_state('ref-1').health is FailsafeHealth.DEGRADED
    snap2 = manager.recompute_worst_sl(
        parent_entry_dispatch_ref='ref-2',
        active_sl_levels=[85.0],
        now_ms=20_000.0,
    )
    manager.record_put_failure(
        'ref-2', generation=snap2.generation, reason='r', now_ms=20_010.0,
    )
    assert manager.get_state('ref-2').health is FailsafeHealth.DEGRADING
    manager.block_new_entry(
        symbol=SYMBOL, pine_id='LongC', bar_ts_ms=1_700_000_000_000,
    )
    block = next(e for e in events if isinstance(e, EntryBlockedDegradedFailsafeEvent))
    assert block.health == 'degraded'


# ---------------------------------------------------------------------
# Trail coalesce (§2.6.5)
# ---------------------------------------------------------------------

def __test_trail_move_within_coalesce_window_not_dispatched__():
    """Trail updates within the coalesce window do not produce a new PUT."""
    manager, _ = _make_manager(trail_coalesce_window_ms=250.0)
    _register_long(manager)
    # Initial trail dispatch.
    snap1 = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[100.0],
        now_ms=1000.0,
        trigger_kind='trail',
    )
    assert snap1 is not None
    assert len(manager.pending_dispatch()) == 1
    # Quick follow-up tick within the coalesce window — desired updates
    # but no new pending PUT.
    snap2 = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[101.0],
        now_ms=1100.0,
        trigger_kind='trail',
    )
    assert snap2 is None
    # The desired level reflects the latest tick, even though the
    # dispatch is throttled.
    assert manager.get_state(PARENT_REF).desired_level == 101.0


def __test_trail_move_outside_window_dispatched_on_flush__():
    """After the coalesce window, the manager releases the latest level."""
    manager, _ = _make_manager(
        trail_coalesce_window_ms=250.0, trail_step_threshold_ticks=1.0,
    )
    _register_long(manager)
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[100.0],
        now_ms=1000.0,
        trigger_kind='trail',
    )
    # Drain the initial dispatch.
    snap = manager.pending_dispatch()[0]
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=1020.0)
    # Subsequent trail moves WITHIN the window are throttled.
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[101.0],
        now_ms=1100.0,
        trigger_kind='trail',
    )
    assert manager.pending_dispatch() == []
    # Window expires → flush_coalesced_trails releases the latest level.
    released = manager.flush_coalesced_trails(now_ms=2000.0)
    assert len(released) == 1
    assert released[0].stop_level == 101.0
