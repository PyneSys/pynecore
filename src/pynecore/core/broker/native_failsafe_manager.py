"""
Broker-native fail-safe worst-SL manager (§2.6 + §2.6.7).

The SOFTWARE partial-qty-bracket path keeps the engine in charge of
every intermediate TP / SL / trailing leg, which means the parent
position is *unprotected* whenever the engine itself is offline (process
crash, network split, WS staleness). The §2.6 fail-safe complements the
engine layer with a single broker-native ``stopLevel`` on the parent
``dealId`` — the *worst* SL across the active partial legs — so the
downtime exposure is bounded.

This module owns the engine-side bookkeeping for that safety net. It does
**not** issue the actual ``PUT /positions/{dealId}`` itself; the plugin
keeps the wire format and the broker-side mapping from
``parent_entry_dispatch_ref`` (a stable client-side COID) to ``dealId``.
The manager produces:

- a per-parent ``NativeStopState`` machine that tracks
  ``healthy → degrading → degraded → retired`` transitions (§2.6.7);
- the *desired* :class:`NativeBracketSnapshot` to send on every PUT (full
  replacement, never patch — Capital.com PUT semantics per §9 #19 Exp D2);
- block / allow decisions for new partial brackets and new
  ``strategy.entry`` signals while the failsafe is not healthy.

Time enters via an explicit ``now_ms`` argument; the manager itself does
not call ``time.time()`` so unit tests and replay paths get deterministic
clocks. Retry pacing is **event-driven**: the manager records the failed
PUT and exposes :meth:`legs_wanting_retry` for the engine's poll tick to
re-dispatch — never a ``sleep`` (``feedback_no_sleep``).
"""
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Iterable

from pynecore.core.broker.models import (
    BrokerEvent,
    BrokerNativeFailsafeExternalEditEvent,
    BrokerNativeFailsafeUnavailableEvent,
    EntryBlockedDegradedFailsafeEvent,
    EntrySkippedDueToDegradedFailsafeEvent,
    NativeFailsafeStateTransitionEvent,
    PartialBracketBlockedDegradedFailsafeEvent,
)


__all__ = [
    'FailsafeHealth',
    'FailsafeOwner',
    'NativeBracketSnapshot',
    'NativeStopState',
    'NativeFailsafeManager',
    'TRAIL_COALESCE_WINDOW_MS_DEFAULT',
    'PUT_MAX_ATTEMPTS_DEFAULT',
    'STALE_WINDOW_MS_DEFAULT',
    'TRAIL_STEP_THRESHOLD_TICKS_DEFAULT',
]


# === Defaults (config-tunable later) =====================================

TRAIL_COALESCE_WINDOW_MS_DEFAULT: float = 250.0
"""§2.6.5 trailing PUT coalesce window. Trail moves within this slice are
collapsed into the latest desired level; the PUT fires once at the end."""

TRAIL_STEP_THRESHOLD_TICKS_DEFAULT: float = 1.0
"""§2.6.5 step-distance threshold expressed in ticks. Trail moves smaller
than this are not dispatched (sub-tick noise / broker ``minStepDistance``)."""

PUT_MAX_ATTEMPTS_DEFAULT: int = 3
"""§2.6.7 immediate retry budget before the state transitions to
``degrading``. Each ``record_put_failure`` call counts as one attempt."""

STALE_WINDOW_MS_DEFAULT: float = 30_000.0
"""§2.6.7 stale window. ``degrading`` states that do not confirm
``actual_level == desired_level`` within this window flip to ``degraded``."""


# === Public types ========================================================

class FailsafeHealth(StrEnum):
    """``NativeStopState`` health (§2.6.7)."""
    HEALTHY = 'healthy'
    DEGRADING = 'degrading'
    DEGRADED = 'degraded'
    RETIRED = 'retired'


class FailsafeOwner(StrEnum):
    """Who currently owns the broker-native bracket on this parent."""
    ENGINE_FAILSAFE = 'engine-failsafe'
    USER_NATIVE = 'user-native'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class NativeBracketSnapshot:
    """Full desired bracket state to send on a single PUT (§2.6.6).

    The Capital.com ``PUT /positions/{dealId}`` is *full replacement* — any
    bracket field omitted from the body is *deleted*. The engine therefore
    carries the desired ``profit_level`` and ``trailing_stop`` alongside
    ``stop_level`` so that worst-SL recompute updates do not accidentally
    erase coexisting TP or trailing legs that the plugin previously
    attached.
    """
    parent_entry_dispatch_ref: str
    symbol: str
    parent_side: str  # 'long' | 'short'
    stop_level: float | None
    profit_level: float | None
    trailing_stop: float | None
    generation: int


@dataclass
class NativeStopState:
    """Per-parent fail-safe stop state machine (§2.6.7).

    The ``generation`` counter increments every time the *desired* snapshot
    changes; correlation back from a Capital.com PUT response is not
    possible (no request token in the success body), so generation is
    purely local anti-stale bookkeeping for the audit journal.
    """
    parent_entry_dispatch_ref: str
    symbol: str
    parent_side: str  # 'long' | 'short'
    desired_level: float | None = None
    desired_profit_level: float | None = None
    desired_trailing_stop: float | None = None
    actual_level: float | None = None
    actual_profit_level: float | None = None
    actual_trailing_stop: float | None = None
    owner: FailsafeOwner = FailsafeOwner.ENGINE_FAILSAFE
    health: FailsafeHealth = FailsafeHealth.HEALTHY
    generation: int = 0
    immediate_attempts_remaining: int = PUT_MAX_ATTEMPTS_DEFAULT
    last_put_ts_ms: float | None = None
    last_failure_ts_ms: float | None = None
    last_failure_reason: str | None = None
    last_confirm_ts_ms: float | None = None
    last_desired_change_ts_ms: float | None = None
    # Anchor for the §2.6.7 stale-window timer. Set to the failure
    # timestamp on the HEALTHY → DEGRADING transition; cleared on
    # DEGRADING → HEALTHY recovery. Stays frozen across subsequent
    # retry failures so the stale window measures "time since the
    # failsafe started failing", not "time since the most recent
    # failure".
    degrading_since_ts_ms: float | None = None
    last_trail_dispatch_ts_ms: float | None = None
    last_trail_dispatched_level: float | None = None
    # §2.6.5 trail-flush gating. ``last_desired_change_ts_ms`` is set on
    # every desired-level change including immediate lifecycle PUTs, so it
    # cannot be used by :meth:`flush_coalesced_trails` to tell "a trail
    # update was suppressed and is waiting" from "the lifecycle path just
    # dispatched". This dedicated marker is set ONLY when
    # :meth:`recompute_worst_sl` actually throttles a trail move into the
    # coalesce window; the flush clears it once the queued snapshot ships
    # (or when an immediate lifecycle / trail dispatch overtakes the
    # suppressed level). Without it, a confirmed-lifecycle PUT would be
    # followed by a phantom duplicate flush once the coalesce window
    # elapsed, bumping the generation and consuming retry budget on a
    # transient failure.
    pending_trail_change_ts_ms: float | None = None
    pending_put: bool = False
    pending_retry: bool = False
    # SL level the broker is still expected to legitimately carry while a
    # freshly recomputed PUT is in flight and unconfirmed. ``recompute_worst_sl``
    # (and the coalesced :meth:`flush_coalesced_trails` path) runs on the main
    # thread and captures the OLD broker level here when a new un-dispatched
    # batch begins (``_pending`` empty). A reconcile observation captured between
    # syncs can still report this old level whether it was sampled BEFORE the
    # queued PUT dispatches (snapshot still in ``_pending``) or AFTER dispatch but
    # before the confirming poll (``mark_dispatch_in_flight`` /
    # ``record_put_success`` already popped ``_pending``).
    # :meth:`on_native_bracket_observed` exempts only that exact stale level — an
    # observation diverging from BOTH the new desired and this pre-PUT level is an
    # operator's manual broker-side edit and must flip ownership to UNKNOWN, not
    # be silently overwritten by the queued PUT.
    #
    # ``None`` is a LEGITIMATE pre-PUT value (the very first PUT arms a stop where
    # the broker carried no stop at all), so the value alone cannot signal whether
    # an exemption is live. ``pre_put_active`` is the dedicated lifetime flag: set
    # ``True`` the moment a baseline is captured (in either dispatch path) and
    # cleared the moment a matching confirmation lands. The exemption gates on the
    # flag, then matches the stored level — so a stale ``stop_level=None`` pre-PUT
    # observation during a first-arm in-flight window is exempt instead of being
    # misread as an external edit that strands the parent.
    pre_put_sl_level: float | None = None
    pre_put_active: bool = False
    # Empirical mintick for the symbol — needed by the trail step gate.
    mintick: float = 0.0
    # ``minmove`` / ``pricescale`` reconstruct the mintick grid with the same
    # integer math as Pine ``math.round_to_mintick`` (mintick == minmove /
    # pricescale), so engine-computed worst-SL floats are snapped to the exact
    # tick grid the broker stores its stop on before they are compared in
    # :meth:`on_native_bracket_observed`. ``minmove`` is not always integral
    # (e.g. mintick 0.025 yields minmove 2.5, pricescale 100). Both default to
    # the "grid unknown" sentinel (``0``); :meth:`_round_to_tick` is a no-op
    # until the engine passes real symbol values via :meth:`register_parent`.
    minmove: float = 0.0
    pricescale: int = 0
    # Stale-window override (None = use manager default).
    stale_window_ms: float | None = None
    # SL leg-kind set tracked for `degrading → healthy` recovery telemetry.
    last_active_sl_count: int = 0


# === Manager =============================================================

class NativeFailsafeManager:
    """Engine-side coordinator for the §2.6 broker-native fail-safe stop.

    The manager owns one :class:`NativeStopState` per
    ``parent_entry_dispatch_ref``. The sync engine drives the lifecycle:

    - on partial-bracket dispatch: :meth:`register_parent`,
      then :meth:`recompute_worst_sl` to seed the desired snapshot;
    - on each leg lifecycle event (arm / trigger / cancel / trail move):
      :meth:`recompute_worst_sl` again — the manager diff-detects whether
      a new PUT is needed and returns it via :meth:`pending_dispatch`;
    - on PUT result: :meth:`record_put_success` / :meth:`record_put_failure`;
    - on every reconcile snapshot: :meth:`on_native_bracket_observed` and
      :meth:`on_deal_id_disappeared`;
    - on Pine-side new partial bracket / new entry: :meth:`block_new_partial_bracket`
      / :meth:`block_new_entry`.

    The manager itself emits no PUT and uses no clock; the engine controls
    timing through the ``now_ms`` arguments. The Slice B plugin wires its
    own dispatcher into :attr:`dispatch_hook` to actually issue the PUT.
    """

    def __init__(
            self,
            *,
            event_sink: Callable[[BrokerEvent], None] | None = None,
            trail_coalesce_window_ms: float = TRAIL_COALESCE_WINDOW_MS_DEFAULT,
            trail_step_threshold_ticks: float = TRAIL_STEP_THRESHOLD_TICKS_DEFAULT,
            put_max_attempts: int = PUT_MAX_ATTEMPTS_DEFAULT,
            stale_window_ms: float = STALE_WINDOW_MS_DEFAULT,
            level_epsilon: float = 1e-9,
    ) -> None:
        self._event_sink = event_sink
        self._trail_coalesce_window_ms = trail_coalesce_window_ms
        self._trail_step_threshold_ticks = trail_step_threshold_ticks
        self._put_max_attempts = put_max_attempts
        self._stale_window_ms = stale_window_ms
        self._eps = level_epsilon
        self._states: dict[str, NativeStopState] = {}
        # Pending desired snapshots ready to dispatch. The engine drains
        # this on every sync tick and forwards to the plugin. Entries
        # disappear on :meth:`record_put_success` / :meth:`record_put_failure`.
        self._pending: dict[str, NativeBracketSnapshot] = {}

    # --- Lifecycle ------------------------------------------------------

    def register_parent(
            self,
            *,
            parent_entry_dispatch_ref: str,
            symbol: str,
            parent_side: str,
            mintick: float,
            minmove: float = 0.0,
            pricescale: int = 0,
            initial_profit_level: float | None = None,
            initial_trailing_stop: float | None = None,
            stale_window_ms: float | None = None,
            pending_confirmation: bool = False,
            now_ms: float | None = None,
    ) -> NativeStopState:
        """Register a parent the engine is about to attach a partial bracket
        to (§2.6.7 ownership flip).

        Idempotent: subsequent calls for the same ref keep the existing
        state but refresh the coexisting ``profit_level`` / ``trailing_stop``
        desired snapshot fields (those can legitimately change when the
        plugin re-establishes a different TP at attach time).

        ``pending_confirmation=True`` is the restart-replay entry point:
        the previous process owned a NativeStopState whose health/owner
        were not persisted, so we cannot tell whether the broker-native
        stop is still in place. Start the freshly created state in
        ``DEGRADING`` with ``degrading_since_ts_ms=now_ms``; this blocks
        ``block_new_entry`` / ``block_new_partial_bracket`` until a
        snapshot from :meth:`on_native_bracket_observed` confirms the
        stop is healthy, and lets the stale-window timer expire to
        ``DEGRADED`` if no confirmation arrives — both prevent adding
        exposure on an unknown protection state. ``recompute_worst_sl``
        still runs (the worst-SL machinery is allowed under
        ``DEGRADING``), so the first post-restart PUT re-attaches the
        broker stop normally. The flag is ignored on idempotent re-calls
        for the same ref (caller's first registration sets the policy).
        """
        if parent_side not in ('long', 'short'):
            raise ValueError(f"parent_side must be 'long' or 'short', got {parent_side!r}")
        state = self._states.get(parent_entry_dispatch_ref)
        if state is None:
            state = NativeStopState(
                parent_entry_dispatch_ref=parent_entry_dispatch_ref,
                symbol=symbol,
                parent_side=parent_side,
                mintick=mintick,
                minmove=minmove,
                pricescale=pricescale,
                stale_window_ms=stale_window_ms,
            )
            if pending_confirmation:
                state.health = FailsafeHealth.DEGRADING
                state.degrading_since_ts_ms = now_ms
            self._states[parent_entry_dispatch_ref] = state
        # Always refresh the coexisting desired fields so the next PUT
        # carries the full picture even if the plugin re-attached the TP.
        # Snap to the tick grid so a later broker observation (also snapped in
        # :meth:`on_native_bracket_observed`) confirms by exact compare rather
        # than drifting by a sub-tick fraction.
        state.desired_profit_level = self._round_to_tick(initial_profit_level, state)
        state.desired_trailing_stop = self._round_to_tick(initial_trailing_stop, state)
        return state

    def unregister_parent(self, parent_entry_dispatch_ref: str) -> None:
        """Drop the per-parent state outright (used by tests / explicit
        teardown). Production code goes through :meth:`on_deal_id_disappeared`
        which keeps a one-shot retired marker."""
        self._states.pop(parent_entry_dispatch_ref, None)
        self._pending.pop(parent_entry_dispatch_ref, None)

    def get_state(self, parent_entry_dispatch_ref: str) -> NativeStopState | None:
        return self._states.get(parent_entry_dispatch_ref)

    def iter_states(self) -> Iterable[NativeStopState]:
        return self._states.values()

    # --- Worst-SL computation ------------------------------------------

    def recompute_worst_sl(
            self,
            *,
            parent_entry_dispatch_ref: str,
            active_sl_levels: Iterable[float],
            now_ms: float,
            trigger_kind: str = 'lifecycle',
    ) -> NativeBracketSnapshot | None:
        """Recompute the worst-SL for one parent and queue a PUT if it
        differs from the current desired snapshot.

        ``trigger_kind`` is one of ``lifecycle`` (leg armed/triggered/cancelled)
        and ``trail`` (trailing leg level moved). Trail moves are subject to
        the §2.6.5 coalesce + step threshold; lifecycle moves dispatch
        immediately.

        Returns the queued snapshot when one was generated, else ``None``.
        """
        state = self._states.get(parent_entry_dispatch_ref)
        if state is None:
            return None
        if state.health is FailsafeHealth.RETIRED:
            return None
        if state.owner is not FailsafeOwner.ENGINE_FAILSAFE:
            # `user-native` and `unknown` ownership: the engine does NOT
            # overwrite. Worst-SL recompute is a no-op for telemetry
            # accuracy. Recovery requires explicit user action.
            return None
        if state.health is FailsafeHealth.DEGRADED:
            # §2.6.7: once the stale window has expired and the state is
            # ``DEGRADED``, the engine must not write the broker-native stop
            # again until the user calls ``set_risk`` / ``reset_to_engine``.
            # A leg cancellation or trailing-level move arriving in this
            # window would otherwise re-queue a ``NativeBracketSnapshot`` and
            # overwrite the stop the operator may already have edited
            # manually. Skip the recompute entirely; recovery re-seeds the
            # desired snapshot from ``reset_to_engine``.
            return None

        levels = list(active_sl_levels)
        state.last_active_sl_count = len(levels)
        if not levels:
            new_desired: float | None = None
        elif state.parent_side == 'long':
            new_desired = min(levels)
        else:
            new_desired = max(levels)

        # Snap the engine-computed worst-SL to the symbol's tick grid before it
        # becomes the desired level. The broker stores its stop on that grid,
        # so an unrounded float would make every confirming observation
        # mismatch by a sub-tick and falsely flip ownership / health. Snapping
        # here also keeps the derived ``pre_put_sl_level`` /
        # ``last_trail_dispatched_level`` (both sourced from ``desired_level``)
        # on the grid for free.
        new_desired = self._round_to_tick(new_desired, state)

        if self._levels_equal(new_desired, state.desired_level):
            return None

        if trigger_kind == 'trail' and not self._trail_should_dispatch(
                state, new_desired, now_ms,
        ):
            # Coalesce window or step threshold blocks dispatch for now;
            # the recompute still updates the in-memory desired so the
            # next tick can compare against the latest level. The engine
            # is expected to call :meth:`flush_coalesced_trails` after
            # the coalesce window so the throttled PUT eventually fires.
            state.desired_level = new_desired
            state.last_desired_change_ts_ms = now_ms
            # Mark a *trail* update as pending the coalesce flush. The
            # generic ``last_desired_change_ts_ms`` cannot be used as the
            # flush gate because the immediate-lifecycle path below also
            # updates it; flushing on that signal alone would queue a
            # duplicate PUT after every lifecycle dispatch.
            state.pending_trail_change_ts_ms = now_ms
            return None

        # Capture the level the broker still carries before this PUT lands.
        # Only at the start of a fresh batch — when no snapshot is queued for
        # this ref AND no pre-PUT exemption is already live. ``_pending`` alone
        # is not enough: :meth:`mark_dispatch_in_flight` pops the queued
        # snapshot the moment the dispatcher takes it, so a follow-up recompute
        # arriving after dispatch but before the confirming poll would re-enter
        # this branch with the ref absent and overwrite ``pre_put_sl_level``
        # with the previous (still-unconfirmed) ``desired_level`` — discarding
        # the genuine broker baseline the first capture recorded. ``pre_put_active``
        # is ``True`` for exactly as long as the broker may still carry that
        # original level, so gating on ``not state.pre_put_active`` preserves
        # the captured baseline across the whole in-flight window. The
        # generation guard in :meth:`on_native_bracket_observed` uses this to
        # tell a legitimately stale pre-PUT observation from an external edit.
        if parent_entry_dispatch_ref not in self._pending and not state.pre_put_active:
            state.pre_put_sl_level = state.desired_level
            state.pre_put_active = True
        state.desired_level = new_desired
        state.generation += 1
        state.last_desired_change_ts_ms = now_ms
        state.immediate_attempts_remaining = self._put_max_attempts
        snapshot = self._build_snapshot(state)
        self._pending[parent_entry_dispatch_ref] = snapshot
        if trigger_kind == 'trail':
            state.last_trail_dispatched_level = new_desired
            state.last_trail_dispatch_ts_ms = now_ms
        # An immediate dispatch (lifecycle OR a trail that passed the
        # throttle gate) supersedes any earlier coalesced trail snapshot
        # — clear the marker so :meth:`flush_coalesced_trails` does not
        # re-queue an obsolete duplicate after the coalesce window.
        state.pending_trail_change_ts_ms = None
        return snapshot

    def flush_coalesced_trails(self, now_ms: float) -> list[NativeBracketSnapshot]:
        """Release any trail-coalesced desired snapshots whose window has
        expired (§2.6.5). Each released snapshot is queued for dispatch
        and returned in the result list.
        """
        released: list[NativeBracketSnapshot] = []
        for state in self._states.values():
            if state.health is FailsafeHealth.RETIRED:
                continue
            if state.owner is not FailsafeOwner.ENGINE_FAILSAFE:
                continue
            if state.health is FailsafeHealth.DEGRADED:
                # §2.6.7: DEGRADED requires user reset before the engine
                # writes the broker-native stop again. Mirror the guard in
                # :meth:`recompute_worst_sl` so a coalesced trail flush
                # cannot bypass it.
                continue
            if state.parent_entry_dispatch_ref in self._pending:
                continue  # already queued / in flight
            # Only flush when a trail update was actually throttled into
            # the coalesce window. ``last_desired_change_ts_ms`` cannot
            # be used here because the lifecycle path also updates it on
            # every immediate dispatch — the flush would then queue a
            # duplicate PUT carrying the just-dispatched lifecycle level.
            if state.pending_trail_change_ts_ms is None:
                continue
            if self._levels_equal(state.desired_level, state.last_trail_dispatched_level):
                state.pending_trail_change_ts_ms = None
                continue
            if (now_ms - state.pending_trail_change_ts_ms) < self._trail_coalesce_window_ms:
                continue
            # §2.6.5 step-threshold guard: when ``recompute_worst_sl`` was
            # suppressed only because the new desired level was inside the
            # ``trail_step_threshold_ticks`` band of the last dispatched
            # level, ``state.desired_level`` was still updated. Without
            # re-checking the threshold here the flush would queue that
            # sub-threshold level once the coalesce window elapses,
            # defeating the rate-limit / min-step guard the threshold is
            # there to enforce. Mirror the dispatch check used at recompute
            # time before queueing.
            if (state.last_trail_dispatched_level is not None
                    and state.desired_level is not None):
                step = abs(state.desired_level - state.last_trail_dispatched_level)
                threshold = state.mintick * self._trail_step_threshold_ticks
                # Round-at-source snaps every level onto the tick grid, so a
                # genuine N-tick step is a difference of grid points that can
                # land a sub-ULP below ``N * mintick`` (a 0.025-grid 1-tick move
                # computes as 0.024999999999999994). Tolerate that with the same
                # ``self._eps`` the manager uses for level equality, else a
                # legitimate 1-tick trail tightening is dropped and the broker
                # stop stays a tick too loose on a safety-critical stop.
                if step < threshold - self._eps:
                    continue
            # Capture the level the broker still carries before this coalesced
            # PUT lands — the snapshot is built fresh from an empty ``_pending``
            # (guard above), so this is a batch start and the broker baseline is
            # the previously dispatched trail level. ``flush_coalesced_trails``
            # clears ``pending_trail_change_ts_ms`` below, so the trail-coalesce
            # exemption no longer covers a stale post-dispatch poll;
            # ``pre_put_sl_level`` / ``pre_put_active`` carry that exemption
            # instead, exactly as the immediate-recompute path does. Gate on
            # ``not state.pre_put_active`` for the same reason
            # :meth:`recompute_worst_sl` does: ``mark_dispatch_in_flight`` pops
            # ``_pending`` the moment the dispatcher takes the earlier PUT, so a
            # follow-up trail change can coalesce and reach this flush while the
            # first PUT is still unconfirmed. Overwriting the baseline then would
            # discard the genuine broker level the first capture recorded, and a
            # stale observation of that original stop would match neither the new
            # desired nor the clobbered baseline — flipping ownership to UNKNOWN
            # despite no manual edit.
            if not state.pre_put_active:
                state.pre_put_sl_level = state.last_trail_dispatched_level
                state.pre_put_active = True
            state.generation += 1
            state.immediate_attempts_remaining = self._put_max_attempts
            snapshot = self._build_snapshot(state)
            self._pending[state.parent_entry_dispatch_ref] = snapshot
            state.last_trail_dispatched_level = state.desired_level
            state.last_trail_dispatch_ts_ms = now_ms
            # Suppressed trail change has now shipped — clear the marker
            # so a follow-up flush tick does not re-enter for the same
            # desired level.
            state.pending_trail_change_ts_ms = None
            released.append(snapshot)
        return released

    # --- Dispatch hand-off ---------------------------------------------

    def pending_dispatch(self) -> list[NativeBracketSnapshot]:
        """Snapshots queued for the engine to forward to the plugin.

        The engine is expected to call :meth:`mark_dispatch_in_flight`
        before actually issuing the PUT and then :meth:`record_put_success`
        / :meth:`record_put_failure` once the result is known.
        """
        return list(self._pending.values())

    def mark_dispatch_in_flight(
            self, parent_entry_dispatch_ref: str, *, now_ms: float,
    ) -> None:
        state = self._states.get(parent_entry_dispatch_ref)
        if state is None:
            return
        state.pending_put = True
        state.last_put_ts_ms = now_ms
        # Drop the queued snapshot now that the dispatcher has it in
        # flight. ``pending_dispatch()`` would otherwise keep returning
        # the same generation on every ``drive_native_failsafe`` tick
        # until the PUT result arrives, defeating the coalescing /
        # rate-limit logic and inviting duplicate or out-of-order PUT
        # callbacks. A newer ``recompute_worst_sl`` (or the retry path
        # in ``record_put_failure``) re-queues a fresh snapshot when
        # one is needed.
        self._pending.pop(parent_entry_dispatch_ref, None)

    def record_put_success(
            self,
            parent_entry_dispatch_ref: str,
            *,
            generation: int,
            now_ms: float,
    ) -> None:
        """Engine reports a successful PUT round-trip.

        Only the structured PUT is recorded here. Snapshot confirmation
        (``actual_level == desired_level``) lands via
        :meth:`on_native_bracket_observed` from the next reconcile.
        """
        state = self._states.get(parent_entry_dispatch_ref)
        if state is None:
            return
        # Ignore late successes from a generation that has already been
        # superseded by a newer recompute, mirroring the failure path
        # (:meth:`record_put_failure`). Clearing ``pending_put`` /
        # ``pending_retry`` / failure metadata on a stale callback would
        # erase the in-flight / retry state of the current snapshot —
        # leaving the latest desired level un-dispatched (the queued
        # snapshot is left alone here, but the next reconcile would
        # treat the matched-but-superseded actual as an external edit
        # in :meth:`on_native_bracket_observed`).
        if generation < state.generation:
            return
        state.pending_put = False
        state.pending_retry = False
        state.last_failure_reason = None
        state.last_failure_ts_ms = None
        # Only drop the queued snapshot when the generation matches —
        # a newer desired might have queued behind it while this PUT was
        # in flight; that one must still be dispatched.
        queued = self._pending.get(parent_entry_dispatch_ref)
        if queued is not None and queued.generation == generation:
            self._pending.pop(parent_entry_dispatch_ref, None)

    def record_put_failure(
            self,
            parent_entry_dispatch_ref: str,
            *,
            generation: int,
            reason: str,
            now_ms: float,
    ) -> None:
        """Engine reports a failed PUT (rate limit, network, broker reject).

        Three immediate retries are budgeted (§2.6.7); after exhaustion the
        state moves to ``degrading``. If a confirmation does not land within
        the stale window, ``degrading`` flips to ``degraded`` via
        :meth:`tick_stale_window`.
        """
        state = self._states.get(parent_entry_dispatch_ref)
        if state is None:
            return
        # Ignore late failures from a generation that has already been
        # superseded by a newer recompute. ``state.generation`` only ever
        # grows (see :meth:`recompute_worst_sl` / :meth:`flush_coalesced_trails`),
        # so ``generation < state.generation`` means a fresher desired
        # snapshot is already queued and possibly in flight. Decrementing
        # the retry budget or flipping to ``degrading`` on a stale failure
        # would penalise the latest snapshot for an outcome that does not
        # describe it; the stale-generation handling already protects the
        # success path at :meth:`record_put_success` and the failure path
        # needs the same guard.
        if generation < state.generation:
            return
        state.pending_put = False
        state.last_failure_reason = reason
        state.last_failure_ts_ms = now_ms
        # Drop only the queued snapshot for this exact generation; newer
        # ones queued meanwhile must survive.
        queued = self._pending.get(parent_entry_dispatch_ref)
        if queued is not None and queued.generation == generation:
            self._pending.pop(parent_entry_dispatch_ref, None)

        state.immediate_attempts_remaining = max(
            0, state.immediate_attempts_remaining - 1,
        )
        if state.immediate_attempts_remaining > 0:
            # Same-tick retry: re-queue the current desired so the engine
            # picks it up again on the next pending_dispatch() drain —
            # *unless* the stale window already promoted the state to
            # DEGRADED while this PUT was in flight (the DEGRADING-then-
            # DEGRADED transition is driven by :meth:`tick_stale_window`,
            # not by retry budget exhaustion, so a state with budget
            # remaining can still flip when a fresh ``recompute_worst_sl``
            # restocked ``immediate_attempts_remaining`` mid-window).
            # Once DEGRADED, §2.6.7 forbids further engine-driven PUTs
            # until ``set_risk`` / ``reset_to_engine``; re-queueing would
            # have ``drive_native_failsafe`` dispatch another snapshot on
            # the next tick and potentially overwrite a manual broker-side
            # edit. Mirror the post-exhaustion DEGRADED guard below.
            if state.health is FailsafeHealth.DEGRADED:
                return
            state.pending_retry = True
            self._pending[parent_entry_dispatch_ref] = self._build_snapshot(state)
            return

        # Budget exhausted — start the degrading window.
        if state.health is FailsafeHealth.HEALTHY:
            # Anchor the stale-window timer on the first failure that drove
            # the state to DEGRADING. ``tick_stale_window`` reads
            # ``degrading_since_ts_ms``; ``last_failure_ts_ms`` keeps
            # refreshing on every retry failure (the audit journal needs
            # the latest), so using it as the anchor would let an API that
            # keeps failing every poll slide the window forever.
            state.degrading_since_ts_ms = now_ms
            self._transition(state, FailsafeHealth.DEGRADING, reason)
        if state.health is FailsafeHealth.DEGRADED:
            # A retry PUT that was already in flight when
            # :meth:`tick_stale_window` promoted the state to DEGRADED
            # can land here with the current generation. Once DEGRADED,
            # §2.6.7 forbids further engine-driven PUTs until the user
            # explicitly calls ``set_risk`` / ``reset_to_engine``;
            # re-queueing a snapshot now would have ``drive_native_failsafe``
            # dispatch it on the next tick and potentially overwrite any
            # manual broker-side edit the operator made after the
            # failsafe was retired. Leave ``pending_retry`` cleared and
            # the pending queue empty — :meth:`tick_stale_window`
            # already wiped both at the DEGRADED transition.
            return
        state.pending_retry = True
        # Keep the desired queued so the next poll tick can retry; the
        # engine's poll loop is what drives the event-driven retry cadence
        # (no sleep — `feedback_no_sleep`).
        self._pending[parent_entry_dispatch_ref] = self._build_snapshot(state)

    def tick_stale_window(self, *, now_ms: float) -> None:
        """Poll-tick driven ``degrading → degraded`` transition.

        Called periodically (once per sync, or once per reconcile poll).
        Promotes any ``degrading`` state whose last desired change was
        more than ``stale_window_ms`` ago without a snapshot confirmation.
        """
        for state in list(self._states.values()):
            if state.health is not FailsafeHealth.DEGRADING:
                continue
            window = state.stale_window_ms or self._stale_window_ms
            # ``degrading_since_ts_ms`` is set when the state enters
            # DEGRADING and stays frozen across subsequent retry failures
            # so the timer measures "time since the failsafe started
            # failing". The legacy ``last_failure_ts_ms`` fall-back covers
            # states that were already DEGRADING before this field was
            # introduced (persisted/restored state). Use ``is not None``
            # rather than truthiness so a deterministic test/replay clock
            # starting at ``0.0`` still anchors at the first failure
            # instead of sliding forward on every retry.
            if state.degrading_since_ts_ms is not None:
                anchor = state.degrading_since_ts_ms
            elif state.last_failure_ts_ms is not None:
                anchor = state.last_failure_ts_ms
            elif state.last_desired_change_ts_ms is not None:
                anchor = state.last_desired_change_ts_ms
            else:
                continue
            if (now_ms - anchor) < window:
                continue
            self._transition(state, FailsafeHealth.DEGRADED, 'stale-window expired')
            # Once the stale window has expired, manual ``set_risk`` /
            # ``reset_to_engine`` is required before the engine writes the
            # broker-native stop again (§2.6.7). Leaving ``pending_retry``
            # and the queued snapshot intact would let the very next
            # ``drive_native_failsafe`` tick re-dispatch the same failed
            # PUT — ``pending_dispatch()`` runs immediately after
            # ``tick_stale_window`` — and ``record_put_failure`` would
            # then re-queue it on every failure. Drop both so DEGRADED
            # really blocks dispatch until the user resets.
            state.pending_retry = False
            self._pending.pop(state.parent_entry_dispatch_ref, None)
            self._emit(BrokerNativeFailsafeUnavailableEvent(
                parent_entry_dispatch_ref=state.parent_entry_dispatch_ref,
                symbol=state.symbol,
                reason=state.last_failure_reason or 'stale-window expired',
            ))

    # --- Snapshot reconcile --------------------------------------------

    def on_native_bracket_observed(
            self,
            parent_entry_dispatch_ref: str,
            *,
            stop_level: float | None,
            profit_level: float | None,
            trailing_stop: float | None,
            now_ms: float,
    ) -> None:
        """Reconcile callback for an observed broker-side bracket snapshot.

        Drives three transitions:

        - ``degrading → healthy`` when ``actual_level == desired_level``
          and no PUT is in flight (§2.6.7 automatic recovery).
        - ``engine-failsafe → unknown`` when ``actual_level != desired_level``
          with no PUT in flight (external manual edit).
        - ``healthy`` confirmation timestamp refresh on every match.
        """
        state = self._states.get(parent_entry_dispatch_ref)
        if state is None:
            return
        # Snap observed broker levels onto the tick grid the desired snapshot
        # already lives on, so confirmation / external-edit detection compares
        # like-for-like. The broker reports its stored, tick-aligned levels,
        # but float round-tripping (JSON, unit conversion) can still perturb
        # them by a sub-tick — without this an exact match would never land.
        stop_level = self._round_to_tick(stop_level, state)
        profit_level = self._round_to_tick(profit_level, state)
        trailing_stop = self._round_to_tick(trailing_stop, state)
        state.actual_level = stop_level
        state.actual_profit_level = profit_level
        state.actual_trailing_stop = trailing_stop

        # The full-replacement PUT carries SL, TP and trailing-stop
        # together (see :meth:`_build_snapshot`), so confirmation /
        # ownership must compare all three fields against the desired
        # snapshot. Comparing the SL alone would treat an external edit
        # to the TP or trailing distance (with SL untouched) as a match;
        # ownership would stay with the engine and the next PUT would
        # resend stale ``desired_profit_level`` / ``desired_trailing_stop``,
        # overwriting that external edit.
        match = (
            self._levels_equal(stop_level, state.desired_level)
            and self._levels_equal(profit_level, state.desired_profit_level)
            and self._levels_equal(trailing_stop, state.desired_trailing_stop)
        )
        if state.pending_put:
            # Confirmation must wait for the in-flight PUT result.
            return

        if match:
            state.last_confirm_ts_ms = now_ms
            # The broker now carries the desired level — the pre-PUT baseline
            # is consumed and must not exempt a later mismatch.
            state.pre_put_sl_level = None
            state.pre_put_active = False
            if state.health is FailsafeHealth.DEGRADING:
                # Clear the stale-window anchor: the next HEALTHY →
                # DEGRADING cycle must re-anchor on the next first-failure.
                state.degrading_since_ts_ms = None
                self._transition(state, FailsafeHealth.HEALTHY, 'snapshot confirm')
            # A queued retry whose desired level matches what the broker
            # is already carrying is a confirmed PUT we just couldn't
            # observe directly (the failure report was wrong, or the
            # success report was lost). Clear the retry flag and drop
            # the queued snapshot so the next ``drive_native_failsafe``
            # does not re-dispatch an already-landed PUT, and so the
            # ``state.pending_retry`` guard below does not swallow a
            # subsequent genuine external-edit mismatch.
            if state.pending_retry:
                state.pending_retry = False
                queued = self._pending.get(parent_entry_dispatch_ref)
                if queued is not None and self._levels_equal(
                        queued.stop_level, stop_level,
                ):
                    self._pending.pop(parent_entry_dispatch_ref, None)
            return

        # Mismatch with no PUT in flight → either retry-pending (covered
        # below), engine-throttled trail (covered next), or external edit
        # (owner flip).
        if state.pending_retry:
            return
        # A lifecycle / trail recompute may have already moved ``desired_level``
        # and pushed a fresh PUT whose result the broker has not reflected back
        # yet. While that PUT is unconfirmed the broker still legitimately
        # carries the pre-PUT level, so an observation diverging from the new
        # ``desired_level`` but equal to that pre-PUT level is stale, not an
        # external edit — flipping ownership to UNKNOWN here would block future
        # brackets until a manual reset. The reconcile thread can sample the
        # broker either BEFORE the queued PUT dispatches (snapshot still in
        # ``_pending``) OR after it dispatches but before the confirming poll
        # arrives (``mark_dispatch_in_flight`` / ``record_put_success`` already
        # popped ``_pending`` and cleared ``pending_put``). Both windows feed
        # the same stale pre-PUT level, so gating on the queued snapshot's
        # presence would leave the post-dispatch sample unprotected and strand
        # the parent. The correct lifetime signal is ``pre_put_active``: it is
        # set when either dispatch path (immediate recompute or coalesced trail
        # flush) captures the broker baseline and cleared the moment a matching
        # confirmation lands above, so it is ``True`` for exactly as long as the
        # broker may still carry the old level. The flag — not
        # ``pre_put_sl_level is not None`` — is the lifetime gate, because
        # ``None`` is a legitimate pre-PUT level when the first PUT arms a stop
        # the broker did not carry.
        #
        # The exemption must match the SPECIFIC level the broker still carries
        # (``pre_put_sl_level``) and the unchanged TP / trailing (the recompute
        # only moves the SL field). An observation diverging from BOTH the new
        # desired and that pre-PUT level is an operator's manual broker-side
        # edit; suppressing it here would let ``drive_native_failsafe`` dispatch
        # the queued PUT and overwrite the manual edit instead of flipping
        # ownership to UNKNOWN. Only the genuinely stale pre-PUT level is exempt.
        if (state.pre_put_active
                and self._levels_equal(stop_level, state.pre_put_sl_level)
                and self._levels_equal(profit_level, state.desired_profit_level)
                and self._levels_equal(trailing_stop, state.desired_trailing_stop)):
            return
        # §2.6.5 trail coalesce / step-threshold: when `recompute_worst_sl`
        # was called with ``trigger_kind='trail'`` and the dispatch was
        # suppressed by the coalesce window or the step threshold, the
        # engine deliberately updated ``state.desired_level`` without
        # queueing a PUT. The broker therefore still carries
        # ``last_trail_dispatched_level``. Treating that legitimate delay
        # as an external edit would flip ownership to UNKNOWN and then
        # ``flush_coalesced_trails`` would skip the parent forever (the
        # owner filter at line 327 rejects non-ENGINE_FAILSAFE states).
        # Only flag external edits when the actual diverges from BOTH
        # the desired level and the last level we dispatched.
        #
        # ``pending_trail_change_ts_ms`` is the only signal that a trail
        # update was just throttled — :meth:`recompute_worst_sl` sets it
        # on the throttle branch and clears it on every immediate dispatch
        # (lifecycle or trail that passed the gate). Without this gate
        # the exemption would silently absorb genuine mismatches whenever
        # a lifecycle recompute has moved ``desired_level`` away from
        # ``last_trail_dispatched_level``: the broker would be carrying
        # the OLD trail level while the engine thinks the level matches
        # the throttled trail and the manager would stay HEALTHY /
        # engine-owned despite the missing desired stop.
        #
        # TP / trailing-stop coalesce is *not* part of §2.6.5 — only the
        # SL field is throttled. If the actual TP or trailing diverges
        # from the desired snapshot, this is a genuine external edit
        # regardless of the SL-side coalesce match, so the trail-coalesce
        # exemption must also require those fields to agree.
        if (state.pending_trail_change_ts_ms is not None
                and state.last_trail_dispatched_level is not None
                and self._levels_equal(stop_level, state.last_trail_dispatched_level)
                and self._levels_equal(profit_level, state.desired_profit_level)
                and self._levels_equal(trailing_stop, state.desired_trailing_stop)):
            return
        if state.owner is FailsafeOwner.ENGINE_FAILSAFE:
            state.owner = FailsafeOwner.UNKNOWN
            # Ownership left the engine because the broker carries an operator's
            # manual edit — the pre-PUT baseline is no longer meaningful. Clear
            # the lifetime flag so a later ``reset_to_engine`` re-queue followed
            # by an observation that happens to equal the now-stale
            # ``pre_put_sl_level`` cannot be wrongly exempted.
            state.pre_put_active = False
            state.pre_put_sl_level = None
            # Drop any snapshot a same-sync recompute queued but did not yet
            # dispatch. The owner just flipped to UNKNOWN because the broker
            # carries an operator's manual edit; the engine no longer owns the
            # bracket, so it must not push the queued PUT. Leaving it in
            # ``_pending`` would let the very next ``pending_dispatch()`` in
            # this same ``drive_native_failsafe`` call (which does not filter
            # by owner) realise the snapshot and overwrite the manual edit
            # this guard exists to preserve.
            self._pending.pop(parent_entry_dispatch_ref, None)
            self._emit(BrokerNativeFailsafeExternalEditEvent(
                parent_entry_dispatch_ref=parent_entry_dispatch_ref,
                symbol=state.symbol,
                desired_level=state.desired_level,
                actual_level=stop_level,
            ))

    def on_deal_id_disappeared(
            self, parent_entry_dispatch_ref: str, *, now_ms: float,
    ) -> None:
        """The reconcile snapshot no longer lists this parent. The state
        retires immediately (§2.6.7 lifecycle cleanup) so symbol-level
        blocks unwind automatically.
        """
        state = self._states.get(parent_entry_dispatch_ref)
        if state is None:
            return
        if state.health is FailsafeHealth.RETIRED:
            return
        self._transition(state, FailsafeHealth.RETIRED, 'dealId disappeared')
        self._pending.pop(parent_entry_dispatch_ref, None)

    # --- Ownership transitions -----------------------------------------

    def claim_user_native(self, parent_entry_dispatch_ref: str) -> None:
        """Mark a parent's bracket as user-managed (native full-row path
        outside the SOFTWARE partial-qty-bracket capability). §2.6 does
        not run for these parents."""
        state = self._states.get(parent_entry_dispatch_ref)
        if state is None:
            return
        state.owner = FailsafeOwner.USER_NATIVE

    def reset_to_engine(self, parent_entry_dispatch_ref: str, *, now_ms: float) -> None:
        """User explicitly reset ownership back to engine-failsafe after
        a manual intervention (§2.6.7 recovery from ``unknown`` /
        ``degraded``). Clears pending failure markers and re-queues the
        current desired snapshot.

        The re-queue runs even when ``desired_level is None`` (clear
        snapshot). That case arises when the engine drove the broker-
        native stop *away* — the last SL leg was cancelled and the
        clear PUT exhausted retries, leaving the state DEGRADED with a
        stale stop still armed at the broker. A user reset must resend
        that clear so the stale broker stop is finally removed; gating
        on ``desired_level is not None`` would leave the parent
        protected by a stop the strategy no longer wants and risk an
        unexpected full-position close.
        """
        state = self._states.get(parent_entry_dispatch_ref)
        if state is None:
            return
        state.owner = FailsafeOwner.ENGINE_FAILSAFE
        state.immediate_attempts_remaining = self._put_max_attempts
        state.pending_retry = False
        state.last_failure_reason = None
        state.last_failure_ts_ms = None
        state.degrading_since_ts_ms = None
        state.generation += 1
        state.last_desired_change_ts_ms = now_ms
        self._pending[parent_entry_dispatch_ref] = self._build_snapshot(state)
        if state.health is not FailsafeHealth.HEALTHY:
            self._transition(state, FailsafeHealth.HEALTHY, 'user reset')

    # --- Gates ---------------------------------------------------------

    def is_new_partial_bracket_blocked(
            self,
            *,
            parent_entry_dispatch_ref: str,
    ) -> bool:
        """Side-effect-free probe of the §2.6.7 new-partial-bracket gate.

        Returns the same boolean as :meth:`block_new_partial_bracket`
        without emitting :class:`PartialBracketBlockedDegradedFailsafeEvent`.
        Callers that need to *preflight* the gate (e.g. the sync engine
        deciding whether to evict the currently armed legs of an
        existing partial bracket before re-dispatching a modify) should
        use this probe so a "would-block" check does not pollute the
        audit log with spurious blocked-bracket events.
        """
        state = self._states.get(parent_entry_dispatch_ref)
        if state is None:
            return False
        return (
            state.health in (FailsafeHealth.DEGRADING, FailsafeHealth.DEGRADED)
            or state.owner is FailsafeOwner.UNKNOWN
        )

    def block_new_partial_bracket(
            self,
            *,
            parent_entry_dispatch_ref: str,
            symbol: str,
            pine_id: str,
            from_entry: str,
    ) -> bool:
        """Return ``True`` when a new SOFTWARE partial-qty bracket
        dispatch for this parent must be rejected (§2.6.7).
        Emits the structured block event when blocking.

        The gate blocks for two distinct conditions:

        - ``health`` is ``degrading`` or ``degraded`` — PUT retries are
          exhausted or the stale window has expired.
        - ``owner`` is ``UNKNOWN`` (external manual edit observed via
          :meth:`on_native_bracket_observed`) — health may still read
          ``healthy``, but :meth:`recompute_worst_sl` is a no-op for
          non-``ENGINE_FAILSAFE`` ownership, so a freshly armed leg would
          add exposure without driving an updated worst-SL to the broker.
          Recovery requires :meth:`reset_to_engine` (user reset).
        """
        state = self._states.get(parent_entry_dispatch_ref)
        if state is None:
            return False
        if (
            state.health not in (FailsafeHealth.DEGRADING, FailsafeHealth.DEGRADED)
            and state.owner is not FailsafeOwner.UNKNOWN
        ):
            return False
        self._emit(PartialBracketBlockedDegradedFailsafeEvent(
            parent_entry_dispatch_ref=parent_entry_dispatch_ref,
            symbol=symbol,
            pine_id=pine_id,
            from_entry=from_entry,
            health=state.health.value,
        ))
        return True

    def block_new_entry(
            self,
            *,
            symbol: str,
            pine_id: str,
            bar_ts_ms: int,
    ) -> bool:
        """Return ``True`` when ``strategy.entry`` on ``symbol`` must be
        dropped because at least one parent on this symbol holds a
        ``degrading`` / ``degraded`` failsafe (§2.6.7).

        Drop-semantics: the engine does NOT queue the signal — see the
        :class:`EntrySkippedDueToDegradedFailsafeEvent` rationale.
        """
        worst: FailsafeHealth | None = None
        for state in self._states.values():
            if state.symbol != symbol:
                continue
            if state.health in (FailsafeHealth.DEGRADING, FailsafeHealth.DEGRADED):
                if worst is None or state.health is FailsafeHealth.DEGRADED:
                    worst = state.health
        if worst is None:
            return False
        self._emit(EntryBlockedDegradedFailsafeEvent(
            symbol=symbol,
            pine_id=pine_id,
            health=worst.value,
        ))
        self._emit(EntrySkippedDueToDegradedFailsafeEvent(
            symbol=symbol,
            pine_id=pine_id,
            bar_ts_ms=bar_ts_ms,
        ))
        return True

    # --- Internals -----------------------------------------------------

    def _build_snapshot(self, state: NativeStopState) -> NativeBracketSnapshot:
        return NativeBracketSnapshot(
            parent_entry_dispatch_ref=state.parent_entry_dispatch_ref,
            symbol=state.symbol,
            parent_side=state.parent_side,
            stop_level=state.desired_level,
            profit_level=state.desired_profit_level,
            trailing_stop=state.desired_trailing_stop,
            generation=state.generation,
        )

    def _round_to_tick(
            self, level: float | None, state: NativeStopState,
    ) -> float | None:
        """Round a price level to the symbol's mintick grid.

        Mirrors :func:`pynecore.lib.math.round_to_mintick` exactly — ties round
        up, and the grid is reconstructed as ``int(level / mintick + 0.5) *
        minmove / pricescale`` so awkward ticks (e.g. ``0.025`` →
        ``minmove=2.5``, ``pricescale=100``) do not accumulate float drift. The
        manager is symbol-agnostic — one instance serves many parents across
        symbols — so it cannot read the process-global ``syminfo``; the grid
        travels per state via :meth:`register_parent`.

        :param level: Price level to snap, or ``None`` (passed through).
        :param state: Per-parent state carrying the symbol's tick grid.
        :returns: ``level`` snapped to the grid, or unchanged when the grid is
            unknown (any factor still at the ``0`` sentinel) — which keeps
            default-constructed / test states on their original exact levels.
        """
        if level is None:
            return None
        if state.mintick <= 0.0 or state.minmove <= 0.0 or state.pricescale <= 0:
            return level
        return int(level / state.mintick + 0.5) * state.minmove / state.pricescale

    def _levels_equal(self, a: float | None, b: float | None) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return abs(a - b) <= self._eps

    def _trail_should_dispatch(
            self,
            state: NativeStopState,
            new_desired: float | None,
            now_ms: float,
    ) -> bool:
        last_ts = state.last_trail_dispatch_ts_ms
        if last_ts is not None and (now_ms - last_ts) < self._trail_coalesce_window_ms:
            return False
        if state.last_trail_dispatched_level is None or new_desired is None:
            return True
        step = abs(new_desired - state.last_trail_dispatched_level)
        threshold = state.mintick * self._trail_step_threshold_ticks
        # See :meth:`flush_coalesced_trails`: tolerate sub-ULP float error so a
        # genuine grid-snapped 1-tick move is not swallowed by the threshold.
        return step >= threshold - self._eps

    def _transition(
            self, state: NativeStopState, new_health: FailsafeHealth, reason: str,
    ) -> None:
        if state.health is new_health:
            return
        old = state.health
        state.health = new_health
        self._emit(NativeFailsafeStateTransitionEvent(
            parent_entry_dispatch_ref=state.parent_entry_dispatch_ref,
            symbol=state.symbol,
            from_state=old.value,
            to_state=new_health.value,
            reason=reason,
        ))

    def _emit(self, event: BrokerEvent) -> None:
        if self._event_sink is not None:
            self._event_sink(event)
