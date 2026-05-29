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


def __test_second_recompute_in_flight_preserves_outstanding_baseline__():
    """A recompute arriving after ``mark_dispatch_in_flight`` popped the queued
    snapshot but before the confirming poll must not lose the captured broker
    baseline. The first arm captured ``None`` (the broker carried no stop yet);
    a stale poll still reporting ``None`` must stay exempt via the outstanding
    baseline, not flip ownership to UNKNOWN."""
    manager, events = _make_manager()
    _register_long(manager)
    # First arm: broker carries no stop, so the baseline entry is None.
    first = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    assert first is not None
    state = manager.get_state(PARENT_REF)
    # Outstanding = [baseline(None), dispatched(90.0)].
    assert [e.sl for e in state.outstanding] == [None, 90.0]
    # The dispatcher takes the PUT in flight and the success is recorded, but
    # the confirming snapshot has not arrived yet (``pending_put`` cleared,
    # ``outstanding`` still carrying both the baseline and 90.0).
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=first.generation, now_ms=1015.0)
    # A tighter SL recompute queues behind, still unconfirmed. The batch is
    # already in flight (``outstanding`` non-empty), so the baseline is
    # preserved and 85.0 is appended below it.
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[85.0],
        now_ms=1020.0,
    )
    assert [e.sl for e in state.outstanding] == [None, 90.0, 85.0]
    # A stale reconcile poll still reporting the original (absent) broker stop
    # equals the baseline entry → exempt, no external-edit flip.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=None, profit_level=None,
        trailing_stop=None, now_ms=1030.0,
    )
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE
    assert not [
        e for e in events
        if isinstance(e, BrokerNativeFailsafeExternalEditEvent)
    ]


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


def __test_user_reset_seeds_a_fresh_confirmation_cycle__():
    """``reset_to_engine`` must re-arm the outstanding/confirmation tracking for
    the PUT it re-queues, exactly like the recompute / flush dispatch paths.
    Without it the reset PUT is invisible to ``tick_stale_window`` (an acked-
    but-unconfirmed reset never times out) and a lagging pre-reset broker
    observation would match no outstanding entry and falsely flip back to
    UNKNOWN."""
    manager, _ = _make_manager(stale_window_ms=5_000.0)
    _register_long(manager)
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[90.0], now_ms=1000.0,
    )
    state = manager.get_state(PARENT_REF)
    state.owner = FailsafeOwner.UNKNOWN

    manager.reset_to_engine(PARENT_REF, now_ms=5000.0)
    # A fresh confirmation cycle is armed: one outstanding entry at the re-queued
    # level, anchored at the reset timestamp.
    assert [e.sl for e in state.outstanding] == [90.0]
    assert state.outstanding_since_ts_ms == 5000.0
    queued = manager.pending_dispatch()[0]
    assert state.outstanding[0].generation == queued.generation

    # The reset PUT is acked but the broker never confirms -> confirmation
    # timeout fires (it would never fire with an empty outstanding list).
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=5010.0)
    manager.record_put_success(PARENT_REF, generation=queued.generation, now_ms=5020.0)
    manager.tick_stale_window(now_ms=11_000.0)
    assert state.health is FailsafeHealth.DEGRADED
    assert state.degraded_reason == 'confirmation-timeout'
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE


def __test_user_reset_exempts_lagging_reset_level_poll__():
    """After a reset re-queues the desired level, a lagging reconcile poll that
    reports that very level must be a confirming observation, not an external
    edit — the seeded outstanding entry exempts it."""
    manager, events = _make_manager()
    _register_long(manager)
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[90.0], now_ms=1000.0,
    )
    state = manager.get_state(PARENT_REF)
    state.owner = FailsafeOwner.UNKNOWN

    manager.reset_to_engine(PARENT_REF, now_ms=5000.0)
    queued = manager.pending_dispatch()[0]
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=5010.0)
    manager.record_put_success(PARENT_REF, generation=queued.generation, now_ms=5020.0)

    # A poll of the re-queued level lands: confirming, ownership stays ENGINE.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=90.0, profit_level=None, trailing_stop=None, now_ms=5030.0,
    )
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE
    assert not [e for e in events if isinstance(e, BrokerNativeFailsafeExternalEditEvent)]
    assert state.outstanding == []  # latest desired confirmed -> list cleared


# ---------------------------------------------------------------------
# Outstanding-levels confirmation tracking (§2.6.7)
# ---------------------------------------------------------------------

def __test_intermediate_in_flight_level_is_exempt_not_external_edit__():
    """Facet A: when the desired worst-SL moves more than once inside one
    reconcile round-trip (95 -> 90 -> 85, both PUTs acked but neither observed
    back), a lagging poll of the INTERMEDIATE 90 matches neither the latest
    desired (85) nor the broker baseline (95). A single scalar slot would
    misread it as an external edit and strand the parent; the outstanding list
    keeps every dispatched level exempt."""
    manager, events = _make_manager()
    _register_long(manager)
    # Confirm a baseline at 95.0 so the pre-burst broker level is X = 95.0.
    first = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[95.0], now_ms=1000.0,
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1005.0)
    manager.record_put_success(PARENT_REF, generation=first.generation, now_ms=1006.0)
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=95.0, profit_level=None, trailing_stop=None, now_ms=1010.0,
    )
    state = manager.get_state(PARENT_REF)
    assert state.health is FailsafeHealth.HEALTHY
    assert state.outstanding == []  # confirmed -> list cleared

    # Burst: 95 -> 90 -> 85, both PUTs dispatched + acked, neither confirmed.
    g90 = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[90.0], now_ms=1020.0,
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1021.0)
    manager.record_put_success(PARENT_REF, generation=g90.generation, now_ms=1022.0)
    g85 = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[85.0], now_ms=1023.0,
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1024.0)
    manager.record_put_success(PARENT_REF, generation=g85.generation, now_ms=1025.0)
    # Outstanding = [baseline(95), dispatched(90), dispatched(85)].
    assert [e.sl for e in state.outstanding] == [95.0, 90.0, 85.0]

    # A lagging poll reports the intermediate 90.0: exempt, ownership intact.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=90.0, profit_level=None, trailing_stop=None, now_ms=1030.0,
    )
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE
    assert not [e for e in events if isinstance(e, BrokerNativeFailsafeExternalEditEvent)]
    # The match prunes the baseline (95) the broker has provably moved past.
    assert [e.sl for e in state.outstanding] == [90.0, 85.0]

    # The confirming poll of the latest target (85) lands -> full confirm.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=85.0, profit_level=None, trailing_stop=None, now_ms=1040.0,
    )
    assert state.outstanding == []
    assert state.last_confirm_ts_ms == 1040.0

    # A genuine external edit after the list is cleared still flips to UNKNOWN.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=70.0, profit_level=None, trailing_stop=None, now_ms=1050.0,
    )
    assert state.owner is FailsafeOwner.UNKNOWN


def __test_acked_but_unconfirmed_put_times_out_to_degraded__():
    """Facet B: a PUT that is acked but whose level a reconcile snapshot never
    reflects back must not keep the state HEALTHY forever. After the
    confirmation-timeout window the state escalates to DEGRADED (reason
    'confirmation-timeout') — WITHOUT flipping ownership to UNKNOWN and WITHOUT
    re-dispatching — and a later confirming snapshot auto-recovers it."""
    manager, events = _make_manager(stale_window_ms=5_000.0)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[90.0], now_ms=1000.0,
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=1020.0)
    state = manager.get_state(PARENT_REF)

    # PUT acked, broker never confirms. Within the window: stays HEALTHY.
    manager.tick_stale_window(now_ms=2000.0)
    assert state.health is FailsafeHealth.HEALTHY

    # Past the window (anchored at the batch start 1000.0): DEGRADED.
    manager.tick_stale_window(now_ms=7000.0)
    assert state.health is FailsafeHealth.DEGRADED
    assert state.degraded_reason == 'confirmation-timeout'
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE  # never UNKNOWN
    assert manager.pending_dispatch() == []  # no blind re-dispatch
    unavail = [e for e in events if isinstance(e, BrokerNativeFailsafeUnavailableEvent)]
    assert len(unavail) == 1
    assert unavail[0].reason == 'confirmation-timeout'
    transitions = [e for e in events if isinstance(e, NativeFailsafeStateTransitionEvent)]
    assert transitions[-1].to_state == 'degraded'

    # A confirmation that finally lands auto-recovers DEGRADED -> HEALTHY.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=90.0, profit_level=None, trailing_stop=None, now_ms=8000.0,
    )
    assert state.health is FailsafeHealth.HEALTHY
    assert state.degraded_reason is None


def __test_put_failure_degraded_does_not_auto_recover_on_confirm__():
    """A PUT-failure DEGRADED (``degraded_reason`` None) requires an explicit
    user reset (§2.6.7); a confirming snapshot must NOT silently recover it,
    unlike a confirmation-timeout DEGRADED."""
    manager, _ = _make_manager(put_max_attempts=1, stale_window_ms=1_000.0)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[90.0], now_ms=1000.0,
    )
    manager.record_put_failure(
        PARENT_REF, generation=snap.generation, reason='rate-limit', now_ms=1010.0,
    )
    manager.tick_stale_window(now_ms=5000.0)
    state = manager.get_state(PARENT_REF)
    assert state.health is FailsafeHealth.DEGRADED
    assert state.degraded_reason is None
    # A confirming observation does NOT auto-recover a PUT-failure DEGRADED.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=90.0, profit_level=None, trailing_stop=None, now_ms=6000.0,
    )
    assert state.health is FailsafeHealth.DEGRADED


def __test_confirmation_timeout_degraded_tracks_leg_cancel_in_memory__():
    """A leg cancelled WHILE a confirmation-timeout DEGRADED is in force must
    not be silently dropped. ``recompute_worst_sl`` still no-ops on dispatch
    (the DEGRADED freeze forbids re-PUT), but for a confirmation-timeout
    DEGRADED — where ownership never left the engine — it keeps the in-memory
    ``desired_level`` tracking the live leg set, so a later reconcile of the
    still-armed obsolete broker stop matches the CURRENT intent (``None``)
    rather than a stale level and does NOT falsely auto-recover to HEALTHY."""
    manager, _ = _make_manager(stale_window_ms=5_000.0)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[90.0], now_ms=1000.0,
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=1020.0)
    state = manager.get_state(PARENT_REF)

    # Broker never confirms -> confirmation-timeout DEGRADED, broker carries 90.
    manager.tick_stale_window(now_ms=7000.0)
    assert state.health is FailsafeHealth.DEGRADED
    assert state.degraded_reason == 'confirmation-timeout'
    assert state.desired_level == 90.0

    # The SL leg is cancelled during the timeout: no dispatch (freeze holds),
    # but the in-memory desired now tracks "no stop wanted".
    assert manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[], now_ms=7500.0,
    ) is None
    assert state.desired_level is None
    assert manager.pending_dispatch() == []  # still frozen, no re-PUT

    # The feed recovers and observes the still-armed obsolete 90: it now
    # MISMATCHES the live desired (None), so the state holds DEGRADED instead
    # of recovering to HEALTHY against a stop the strategy no longer wants.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=90.0, profit_level=None, trailing_stop=None, now_ms=8000.0,
    )
    assert state.health is FailsafeHealth.DEGRADED
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE  # never flipped UNKNOWN

    # A user reset re-seeds the (None) desired and queues the clear PUT that
    # finally removes the stale broker stop.
    manager.reset_to_engine(PARENT_REF, now_ms=9000.0)
    assert state.health is FailsafeHealth.HEALTHY
    queued = [
        s for s in manager.pending_dispatch()
        if s.parent_entry_dispatch_ref == PARENT_REF
    ]
    assert len(queued) == 1 and queued[0].stop_level is None


def __test_confirmation_timeout_degraded_still_auto_recovers_unchanged_legs__():
    """The in-memory tracking must NOT regress the legitimate auto-recovery:
    when the leg set is UNCHANGED through a confirmation-timeout DEGRADED, a
    later confirming snapshot of the (still-current) desired triple recovers
    the state to HEALTHY exactly as before."""
    manager, _ = _make_manager(stale_window_ms=5_000.0)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[80.0], now_ms=1000.0,
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=1020.0)
    state = manager.get_state(PARENT_REF)
    manager.tick_stale_window(now_ms=7000.0)
    assert state.health is FailsafeHealth.DEGRADED

    # A redundant recompute with the same leg set keeps desired at 80.
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[80.0], now_ms=7500.0,
    )
    assert state.desired_level == 80.0

    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=80.0, profit_level=None, trailing_stop=None, now_ms=8000.0,
    )
    assert state.health is FailsafeHealth.HEALTHY
    assert state.degraded_reason is None


def __test_confirmation_timeout_degraded_with_manual_edit_does_not_auto_recover__():
    """A confirmation-timeout DEGRADED whose broker stop is then manually edited
    flips ownership to UNKNOWN (external-edit path) but keeps ``degraded_reason``.
    A later observation that happens to equal the (still-current) desired triple
    must NOT auto-recover to HEALTHY: ownership left the engine, so §2.6.7 requires
    an explicit ``reset_to_engine``. Without the ownership guard on the recovery
    branch the state would silently go HEALTHY while ``block_new_entry`` stops
    blocking and ``recompute_worst_sl`` stays a no-op for non-engine ownership."""
    manager, _ = _make_manager(stale_window_ms=5_000.0)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[80.0], now_ms=1000.0,
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=1020.0)
    state = manager.get_state(PARENT_REF)
    manager.tick_stale_window(now_ms=7000.0)
    assert state.health is FailsafeHealth.DEGRADED
    assert state.degraded_reason == 'confirmation-timeout'
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE

    # Operator manually moves the broker stop -> external-edit path: owner UNKNOWN,
    # reason intentionally untouched (it is a confirmation-timeout DEGRADED still).
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=75.0, profit_level=None, trailing_stop=None, now_ms=7500.0,
    )
    assert state.owner is FailsafeOwner.UNKNOWN
    assert state.degraded_reason == 'confirmation-timeout'

    # A later observation equal to the still-current desired (80) must NOT recover:
    # ownership is UNKNOWN, recovery needs an explicit reset.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=80.0, profit_level=None, trailing_stop=None, now_ms=8000.0,
    )
    assert state.health is FailsafeHealth.DEGRADED  # not auto-recovered
    assert state.owner is FailsafeOwner.UNKNOWN
    assert manager.block_new_entry(
        symbol=SYMBOL, pine_id='pi00cafe', bar_ts_ms=8000,
    ) is True

    # Explicit reset restores engine ownership; recovery is now possible again.
    manager.reset_to_engine(PARENT_REF, now_ms=8500.0)
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE


def __test_confirmation_timeout_never_fires_without_an_acked_put__():
    """State-only run (no dispatcher ever acks the queued PUT): the confirmation
    timeout must NOT fire. ``recompute_worst_sl`` arms ``outstanding`` (and the
    anchor) at queue time, and a mere ``mark_dispatch_in_flight`` hand-off is not
    acknowledgement — only a broker-acked PUT (``record_put_success``) means an
    acked-but-unconfirmed broker stop exists. Escalating to DEGRADED before any
    ack would wrongly block entries on a run that never confirmed broker traffic."""
    manager, events = _make_manager(stale_window_ms=5_000.0)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[90.0], now_ms=1000.0,
    )
    state = manager.get_state(PARENT_REF)
    assert state.outstanding  # armed at queue time
    assert state.batch_put_acked is False  # but nothing was ever acked

    # Far past the window: the snapshot was never acked, so HEALTHY holds — even a
    # bare hand-off (no ack) does not arm the window.
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=2000.0)
    assert state.batch_put_acked is False
    manager.tick_stale_window(now_ms=100_000.0)
    assert state.health is FailsafeHealth.HEALTHY
    assert state.degraded_reason is None
    assert not [e for e in events if isinstance(e, BrokerNativeFailsafeUnavailableEvent)]

    # Once a PUT is actually acked the window legitimately applies again.
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=100_010.0)
    assert state.batch_put_acked is True
    manager.tick_stale_window(now_ms=200_000.0)
    assert state.health is FailsafeHealth.DEGRADED
    assert state.degraded_reason == 'confirmation-timeout'


def __test_failed_put_with_retries_left_does_not_confirmation_timeout__():
    """A failed PUT (rate-limit / reject) with retry budget remaining must keep
    the state HEALTHY and the retry queued. Because the broker never acked the
    PUT, the confirmation-timeout path must NOT arm — even a ``tick_stale_window``
    past ``stale_window_ms`` must leave the retry intact rather than promoting
    straight to DEGRADED and bypassing the retry budget."""
    manager, events = _make_manager(stale_window_ms=5_000.0)
    _register_long(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF, active_sl_levels=[90.0], now_ms=1000.0,
    )
    state = manager.get_state(PARENT_REF)
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    # The PUT fails but retries remain (default budget is 3): state stays HEALTHY
    # with the retry re-queued, and no PUT was ever acknowledged.
    manager.record_put_failure(
        PARENT_REF, generation=snap.generation, reason='rate-limit', now_ms=1020.0,
    )
    assert state.health is FailsafeHealth.HEALTHY
    assert state.pending_retry is True
    assert state.batch_put_acked is False
    assert manager.pending_dispatch()  # the retry is still queued

    # Far past the stale window: the failed PUT is owned by the retry budget, not
    # the confirmation-timeout. The state must stay HEALTHY and the retry survive.
    manager.tick_stale_window(now_ms=100_000.0)
    assert state.health is FailsafeHealth.HEALTHY
    assert state.degraded_reason is None
    assert state.pending_retry is True
    assert manager.pending_dispatch()
    assert not [e for e in events if isinstance(e, BrokerNativeFailsafeUnavailableEvent)]


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


def __test_second_flush_in_flight_preserves_outstanding_baseline__():
    """A coalesced trail flush that fires while an earlier flushed PUT is still
    unconfirmed must not lose the captured broker baseline.
    ``mark_dispatch_in_flight`` pops the queued snapshot the moment the
    dispatcher takes it, so a follow-up trail move can coalesce and reach a
    second flush before the confirming poll arrives; the outstanding baseline
    must keep the ORIGINAL broker level so a stale observation of it stays
    exempt instead of flipping ownership."""
    manager, events = _make_manager(
        trail_coalesce_window_ms=250.0, trail_step_threshold_ticks=1.0,
    )
    _register_long(manager)
    # Initial trail dispatch lands and is confirmed: broker carries 100.0.
    init = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[100.0],
        now_ms=1000.0,
        trigger_kind='trail',
    )
    assert init is not None
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=init.generation, now_ms=1020.0)
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=100.0, profit_level=None,
        trailing_stop=None, now_ms=1030.0,
    )
    # First throttled trail move; flush captures the broker baseline (100.0)
    # and dispatches 101.0. The dispatcher takes it in flight and acks it, but
    # the confirming snapshot has not arrived yet.
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[101.0],
        now_ms=1100.0,
        trigger_kind='trail',
    )
    released = manager.flush_coalesced_trails(now_ms=1400.0)
    assert len(released) == 1
    state = manager.get_state(PARENT_REF)
    # Outstanding = [baseline(100.0), dispatched(101.0)].
    assert [e.sl for e in state.outstanding] == [100.0, 101.0]
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1410.0)
    manager.record_put_success(
        PARENT_REF, generation=released[0].generation, now_ms=1410.0,
    )
    # Second throttled trail move arrives before the 101.0 PUT confirms.
    manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[102.0],
        now_ms=1500.0,
        trigger_kind='trail',
    )
    # Second flush while still in flight: the batch is non-empty so the
    # original baseline (100.0) is preserved and 102.0 is appended below it.
    manager.flush_coalesced_trails(now_ms=1800.0)
    assert [e.sl for e in state.outstanding] == [100.0, 101.0, 102.0]
    # A stale reconcile poll still reporting the ORIGINAL broker stop (100.0)
    # equals the baseline entry → exempt, no external-edit flip.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=100.0, profit_level=None,
        trailing_stop=None, now_ms=1850.0,
    )
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE
    assert not [
        e for e in events
        if isinstance(e, BrokerNativeFailsafeExternalEditEvent)
    ]


# ---------------------------------------------------------------------
# Mintick rounding (§2.6.3 / round-at-source)
# ---------------------------------------------------------------------

# 0.5 tick grid expressed as Pine ``mintick == minmove / pricescale``
# (the provider's loop yields these for a 0.5 step).
GRID_05 = dict(mintick=0.5, minmove=5.0, pricescale=10)
# Awkward 0.025 tick (QM1!-style): minmove is fractional, pricescale a
# power of ten — the integer reconstruction must stay drift-free.
GRID_0025 = dict(mintick=0.025, minmove=2.5, pricescale=100)


def _register_long_grid(
        manager: NativeFailsafeManager,
        *,
        ref: str = PARENT_REF,
        grid: dict[str, Any] = GRID_05,
        **kwargs: Any,
) -> None:
    manager.register_parent(
        parent_entry_dispatch_ref=ref,
        symbol=SYMBOL,
        parent_side='long',
        **grid,
        **kwargs,
    )


def __test_recompute_snaps_worst_sl_to_mintick_grid__():
    """Off-grid worst-SL is snapped to the symbol's tick grid before it
    becomes the desired level (round-at-source, ties up)."""
    manager, _ = _make_manager()
    _register_long_grid(manager)
    # 90.24 -> 90.0 (closer to 90.0); 90.25 is a tie that rounds up to 90.5.
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.24],
        now_ms=1000.0,
    )
    assert snap is not None
    assert snap.stop_level == 90.0
    assert manager.get_state(PARENT_REF).desired_level == 90.0


def __test_recompute_tie_rounds_up__():
    """§ Pine round_to_mintick semantics: an exact half-tick rounds up."""
    manager, _ = _make_manager()
    _register_long_grid(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.25],
        now_ms=1000.0,
    )
    assert snap is not None
    assert snap.stop_level == 90.5


def __test_awkward_tick_rounds_without_float_drift__():
    """A fractional ``minmove`` (0.025 tick) must land exactly on the grid,
    not a sub-tick approximation — the integer reconstruction guarantees it."""
    manager, _ = _make_manager()
    _register_long_grid(manager, grid=GRID_0025)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[10.037],
        now_ms=1000.0,
    )
    assert snap is not None
    assert snap.stop_level == 10.025
    # Exactly on grid: dividing by the tick yields a whole number of ticks.
    ticks = snap.stop_level / 0.025
    assert abs(ticks - round(ticks)) < 1e-9


def __test_observed_subtick_perturbation_confirms_as_match__():
    """The core live-risk fix: a broker observation perturbed by a sub-tick
    float fraction snaps to the desired grid level and confirms — instead of
    diverging past the 1e-9 epsilon and falsely flipping ownership to UNKNOWN."""
    manager, _ = _make_manager()
    _register_long_grid(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=1020.0)
    # Broker echoes the stored stop, but float round-tripping perturbs it by
    # ~3e-7 — well past the exact-compare epsilon.
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=90.0000003, profit_level=None,
        trailing_stop=None, now_ms=2000.0,
    )
    state = manager.get_state(PARENT_REF)
    assert state.owner is FailsafeOwner.ENGINE_FAILSAFE
    assert state.last_confirm_ts_ms == 2000.0


def __test_genuine_external_edit_still_flips_with_grid_active__():
    """Rounding must not mask a real manual edit: a stop moved to a different
    grid point still flips ownership to UNKNOWN."""
    manager, _ = _make_manager()
    _register_long_grid(manager)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.0],
        now_ms=1000.0,
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=1020.0)
    manager.on_native_bracket_observed(
        PARENT_REF, stop_level=85.0, profit_level=None,
        trailing_stop=None, now_ms=3000.0,
    )
    assert manager.get_state(PARENT_REF).owner is FailsafeOwner.UNKNOWN


def __test_no_grid_leaves_levels_unrounded__():
    """Backward compatibility: with the grid factors at their ``0`` sentinel
    (no symbol info), levels pass through unrounded."""
    manager, _ = _make_manager()
    _register_long(manager)  # mintick only, no minmove / pricescale
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[90.123456],
        now_ms=1000.0,
    )
    assert snap is not None
    assert snap.stop_level == 90.123456


def __test_one_tick_trail_move_dispatches_on_awkward_grid__():
    """Regression: round-at-source snaps trail levels onto the grid, so a
    genuine 1-tick move is a float difference (0.075 - 0.05 ==
    0.024999999999999994) that lands a sub-ULP below the 1-tick threshold.
    The immediate step-threshold compare must tolerate that, else a real trail
    tightening is silently dropped and the broker stop stays a tick too loose."""
    manager, _ = _make_manager()
    _register_long_grid(manager, grid=GRID_0025)
    # Seed and confirm the first trail level at 0.05.
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[0.05],
        now_ms=1000.0,
        trigger_kind='trail',
    )
    assert snap is not None
    assert snap.stop_level == 0.05
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=1020.0)
    # A genuine 1-tick-higher move, past the coalesce window, must dispatch.
    snap2 = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[0.075],
        now_ms=2000.0,
        trigger_kind='trail',
    )
    assert snap2 is not None
    assert snap2.stop_level == 0.075


def __test_one_tick_trail_move_flushed_on_awkward_grid__():
    """Regression (coalesce-flush mirror): a 1-tick trail move throttled into
    the coalesce window must still be released by ``flush_coalesced_trails`` —
    its step-threshold re-check must tolerate the same sub-ULP float error."""
    manager, _ = _make_manager()
    _register_long_grid(manager, grid=GRID_0025)
    snap = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[0.05],
        now_ms=1000.0,
        trigger_kind='trail',
    )
    manager.mark_dispatch_in_flight(PARENT_REF, now_ms=1010.0)
    manager.record_put_success(PARENT_REF, generation=snap.generation, now_ms=1020.0)
    # 1-tick move INSIDE the coalesce window -> throttled (no immediate dispatch).
    throttled = manager.recompute_worst_sl(
        parent_entry_dispatch_ref=PARENT_REF,
        active_sl_levels=[0.075],
        now_ms=1100.0,
        trigger_kind='trail',
    )
    assert throttled is None
    assert manager.pending_dispatch() == []
    # Window expires -> flush must release the 1-tick move, not swallow it.
    released = manager.flush_coalesced_trails(now_ms=2000.0)
    assert len(released) == 1
    assert released[0].stop_level == 0.075
