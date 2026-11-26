"""
Typed helpers over :class:`~pynecore.core.broker.storage.RunContext`.

The :class:`~pynecore.core.broker.storage.BrokerStore` exposes
:meth:`~BrokerStore.upsert_order` with a free-form
``extras`` dict and individual setters per column. Each broker plugin
currently re-implements the same micro-patterns on top of that surface:
seed a row with ``{'kind': ..., 'order_type': ...}`` extras, record the
server reference, lift the state machine to ``'confirmed'`` with the
fill price persisted under ``extras['confirm_level']``. The keys are
implicit, the ordering of writes is implicit, and a typo causes silent
divergence between Pine intent and broker truth.

These helpers turn the most common ``execute_entry`` lifecycle steps
into typed function calls. They are intentionally a *thin* layer:
nothing in this module talks to the exchange, retries, sleeps, or makes
state-machine decisions beyond what the orchestrating
:class:`~pynecore.core.broker.journal.DispatchJournal` instructs. The
module is the single place where the canonical ``extras`` schema for
non-bracket entries lives.

Scope is intentionally limited to the M1 proof-of-shape — entry orders
only, no bracket legs, no trail state, no natural-close flags. Those
land in later milestones once a second broker plugin confirms the
shape.

See ``docs/pynecore/plugin-system/broker/broker-plugin-responsibility-review.md``
section §4 for the rationale.
"""
from collections.abc import Iterable, Iterator, Mapping, Sequence
from time import time as epoch_time
from typing import Any, TYPE_CHECKING

from pynecore.types.strategy import ADOPTED_STARTUP_EXTRA_KEY

if TYPE_CHECKING:
    from pynecore.core.broker.models import PositionLeg
    from pynecore.core.broker.ownership_journal import OwnershipJournal
    from pynecore.core.broker.storage import OrderRow, RunContext

__all__ = [
    'ENTRY_KIND_POSITION',
    'ENTRY_KIND_WORKING',
    'KIND_FULL_CLOSE',
    'KIND_PARTIAL_CLOSE',
    'KIND_CANCEL',
    'KIND_MODIFY_ENTRY',
    'KIND_MODIFY_EXIT',
    'CLOSE_KINDS',
    'STATE_SUBMITTED',
    'STATE_SERVER_REF_SEEN',
    'STATE_CONFIRMED',
    'STATE_REJECTED',
    'EXTRAS_KEY_REJECTED_ON_ABSENCE',
    'CLOSE_ENDED_UNFILLED_EVENT',
    'CLOSE_ENDED_PARTIALLY_EVENT',
    'CLOSE_EXECUTED_QTY_KEY',
    'is_proven_refusal',
    'STATE_DISPOSITION_UNKNOWN',
    'STATE_CLOSING',
    'STATE_CANCEL_PENDING',
    'STATE_PARTIAL_BRACKET_LEG',
    'PENDING_DISPATCH_STATES',
    'LEG_KIND_TP_PARTIAL',
    'LEG_KIND_SL_PARTIAL',
    'LEG_KIND_TRAIL_PARTIAL',
    'ENGINE_TRIGGER_LEG_KINDS',
    'LEG_STATE_ARMED',
    'LEG_STATE_PENDING_ENTRY',
    'LEG_STATE_TRIGGERING',
    'LEG_STATE_TRIGGERED',
    'LEG_STATE_TRIGGERED_FAILED',
    'LEG_STATE_TRIGGERED_UNKNOWN',
    'LEG_STATE_ABORTED_PARENT_GONE',
    'LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED',
    'LEG_STATE_CASCADED_CANCEL',
    'LEG_STATE_CASCADED_CANCEL_BY_PARENT_CLOSE',
    'LEG_STATE_CASCADED_CANCEL_BY_NATIVE_SL',
    'LEG_STATE_CANCEL_TENTATIVE',
    'LEG_STATE_TERMINAL',
    'LEG_STATE_ACTIVE',
    'LEG_STATE_LIVE',
    'EXTRAS_KEY_LEG_KIND',
    'EXTRAS_KEY_LEG_STATE',
    'EXTRAS_KEY_TRIGGER_LEVEL',
    'EXTRAS_KEY_TRIGGER_OFFSET',
    'EXTRAS_KEY_TRAIL_ACTIVATION_LEVEL',
    'EXTRAS_KEY_TRAIL_ACTIVATION_OFFSET',
    'EXTRAS_KEY_PARENT_ENTRY_DISPATCH_REF',
    'EXTRAS_KEY_INTENT_PARTIAL_QTY',
    'EXTRAS_KEY_PARENT_PINE_ENTRY_ID',
    'EXTRAS_KEY_OCA_GROUP',
    'EXTRAS_KEY_OCA_TYPE',
    'EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS',
    'EXTRAS_KEY_TRIGGER_CLOSE_COID',
    'create_entry_order_row',
    'record_server_ref',
    'mark_confirmed_with_fill',
    'mark_disposition_unknown',
    'mark_rejected',
    'find_pending_dispatch',
    'adopt_untracked_position_legs',
    'create_close_target_row',
    'record_close_server_ref',
    'mark_closing',
    'mark_close_completed',
    'create_cancel_command_row',
    'mark_cancel_completed',
    'create_modify_entry_row',
    'create_modify_exit_row',
    'mark_modify_completed',
    'mark_reconcile_filled',
    'mark_reconcile_terminal_close',
    'create_engine_trigger_partial_leg_row',
    'update_engine_trigger_partial_leg_state',
    'iter_active_engine_trigger_partial_legs',
    'STATE_CLOSE_LEG',
    'CLOSE_LEG_STATE_PENDING',
    'CLOSE_LEG_STATE_DISPATCHED',
    'CLOSE_LEG_STATE_LIVE',
    'EXTRAS_KEY_CLOSE_LEG_STATE',
    'EXTRAS_KEY_CLOSE_LEG_ID',
    'EXTRAS_KEY_CLOSE_PARENT_COID',
    'EXTRAS_KEY_CLOSE_LEG_VOLUME',
    'EXTRAS_KEY_CLOSE_FAN_VOLUME',
    'EXTRAS_KEY_CLOSE_REQUESTED_VOLUME',
    'EXTRAS_KEY_CLOSE_LEG_PRE_VOLUME',
    'EXTRAS_KEY_CLOSE_LEG_ATTEMPTS',
    'EXTRAS_KEY_CLOSE_LEG_SINCE_MS',
    'EXTRAS_KEY_CLOSE_LEG_RESERVED',
    'create_close_leg_row',
    'update_close_leg_state',
    'iter_active_close_legs',
    'reserve_close_leg_send',
    'release_close_leg_send',
    'lower_close_leg_send_pre_volume',
    'finish_close_leg_row',
    'iter_close_leg_sends',
    'STATE_RESIDUAL_OPEN',
    'EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID',
    'EXTRAS_KEY_RESIDUAL_OPEN_RUN_TAG',
    'EXTRAS_KEY_RESIDUAL_OPEN_BAR_TS_MS',
    'EXTRAS_KEY_RESIDUAL_OPEN_RETRY_SEQ',
    'EXTRAS_KEY_RESIDUAL_OPEN_CLOSE_PLAN',
    'EXTRAS_KEY_RESIDUAL_OPEN_COUNTED_CLOSES',
    'count_residual_open_close',
    'create_residual_open_row',
    'advance_residual_open_attempt',
    'iter_active_residual_opens',
    'clear_residual_open_row',
    'EXTRAS_KEY_RESIDUAL_OPEN_NETTING',
    'STATE_NETTING',
    'STATE_NETTING_ABSORBED',
    'EXTRAS_KEY_NETTING_FILL_ID',
    'EXTRAS_KEY_NETTING_ENTRY_ID',
    'EXTRAS_KEY_NETTING_SINCE_MS',
    'EXTRAS_KEY_NETTING_PASSES',
    'EXTRAS_KEY_NETTING_EXPECTED',
    'EXTRAS_KEY_NETTING_SETTLED_MS',
    'EXTRAS_KEY_NETTING_CONTENDED',
    'create_netting_row',
    'iter_active_nettings',
    'netting_absorbed_row_coid',
    'netting_expected',
    'absorb_netting_fills',
    'STATE_BRACKET_OWN',
    'BRACKET_OWN_STATE_ACTIVE',
    'BRACKET_OWN_STATE_CLEARING',
    'BRACKET_OWN_STATE_RELEASED',
    'BRACKET_OWN_STATE_LIVE',
    'LEDGER_ROW_STATES',
    'EXTRAS_KEY_BRACKET_OWN_STATE',
    'EXTRAS_KEY_BRACKET_OWN_LEG_ID',
    'EXTRAS_KEY_BRACKET_OWN_ATTACH_COID',
    'EXTRAS_KEY_BRACKET_OWN_CLEAR_COID',
    'EXTRAS_KEY_BRACKET_OWN_TP',
    'EXTRAS_KEY_BRACKET_OWN_SL',
    'EXTRAS_KEY_BRACKET_OWN_TRAIL_PRICE',
    'EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET',
    'EXTRAS_KEY_BRACKET_OWN_OCA_NAME',
    'EXTRAS_KEY_BRACKET_OWN_OCA_TYPE',
    'EXTRAS_KEY_BRACKET_OWN_SOFTWARE',
    'EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_EXTREME',
    'EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_OFFSET',
    'create_bracket_ownership_row',
    'update_bracket_ownership_state',
    'iter_active_bracket_ownerships',
    'STATE_STOP_LIMIT_WATCH',
    'STOP_LIMIT_STATE_ARMED',
    'STOP_LIMIT_STATE_LIMIT_PENDING',
    'STOP_LIMIT_STATE_TRIGGERED',
    'STOP_LIMIT_STATE_FILLED',
    'STOP_LIMIT_STATE_ABORTED',
    'STOP_LIMIT_STATE_TERMINAL',
    'STOP_LIMIT_STATE_LIVE',
    'LEGACY_OCO_STATE_CANCEL_PENDING',
    'LEGACY_OCO_STATE_MARKET_PENDING',
    'LEGACY_OCO_STATE_LIVE',
    'EXTRAS_KEY_STOP_LIMIT_STATE',
    'EXTRAS_KEY_STOP_LIMIT_STOP',
    'EXTRAS_KEY_STOP_LIMIT_LIMIT',
    'EXTRAS_KEY_STOP_LIMIT_ORDER_COID',
    'EXTRAS_KEY_STOP_LIMIT_ARMED_BAR',
    'EXTRAS_KEY_STOP_LIMIT_CHECKED_BAR',
    'LEGACY_EXTRAS_KEY_OCO_LIMIT_COID',
    'LEGACY_EXTRAS_KEY_OCO_MARKET_COID',
    'create_stop_limit_watch_row',
    'update_stop_limit_watch',
    'iter_live_stop_limit_watches',
]


# === Canonical extras values ===============================================

# ``extras['kind']`` for an entry that opens a position immediately
# (MARKET order). The plugin's reconcile / activity paths key off this
# to know the row maps to ``/positions`` rather than ``/workingorders``.
ENTRY_KIND_POSITION = 'position'

# ``extras['kind']`` for an entry placed as a working order (LIMIT or
# STOP). The reconcile / activity paths key off this to know the row
# maps to ``/workingorders`` rather than ``/positions``.
ENTRY_KIND_WORKING = 'working'

# ``extras['kind']`` for a close that targets every unit of one or more
# live positions. The plugin issues a DELETE per target ``dealId``; no
# server reference is allocated (DELETE is fire-and-forget). Recovery
# reconciles the persisted ``extras['targets']`` against the live
# positions snapshot.
KIND_FULL_CLOSE = 'full_close'

# ``extras['kind']`` for a close that reduces an existing position by
# a strict subset of its units. Emulated via an opposite-direction
# POST to ``/positions`` (Capital.com has no native partial-close
# endpoint), so the row owns a server reference like an entry would.
# Recovery uses the ``deal_reference`` confirm GET plus a unit-count
# delta against the pre/post position snapshot.
KIND_PARTIAL_CLOSE = 'partial_close_emulated'

# Convenience set used by close-aware recovery branches that route by
# ``extras['kind']`` and need to handle both full and partial close
# rows identically.
CLOSE_KINDS: frozenset[str] = frozenset({KIND_FULL_CLOSE, KIND_PARTIAL_CLOSE})

# ``extras['kind']`` for a cancel command row. Carries the per-dispatch
# audit trail for a ``CancelIntent``; the row does not itself land on
# the exchange. The targets it sweeps are referenced via
# ``extras['target_coids']`` so a mid-loop crash can re-evaluate which
# targets still need cancelling on recovery.
KIND_CANCEL = 'cancel'

# ``extras['kind']`` for a working-order amend (PUT level). The row
# carries the audit trail for the amend dispatch and stores the new
# level under ``extras['new_level']`` so recovery can verify whether
# the broker landed the change.
KIND_MODIFY_ENTRY = 'modify_entry'

# ``extras['kind']`` for a position bracket amend (PUT TP / SL /
# trailing). The row is the entry-side audit trail; the synthetic
# bracket leg rows that mirror the resulting TP / SL state remain
# under the plugin's leg state machine for the duration of M4.
KIND_MODIFY_EXIT = 'modify_exit'


# === Canonical states ======================================================

# Order row persisted, REST POST not yet attempted.
STATE_SUBMITTED = 'submitted'

# REST POST returned a server reference (``dealReference`` for
# Capital.com). Confirm / readback is the next step.
STATE_SERVER_REF_SEEN = 'server_ref_seen'

# Confirm succeeded; ``exchange_order_id`` populated. For MARKET orders
# this also implies a fill, with the fill price under
# ``extras['confirm_level']`` and ``filled_qty`` non-zero.
STATE_CONFIRMED = 'confirmed'

# Exchange rejected the dispatch synchronously (confirm REJECTED, or a
# 4xx response with a known reject reason). Terminal.
STATE_REJECTED = 'rejected'

# ``extras`` flag of a :data:`STATE_REJECTED` row whose rejection is an
# inference, not the venue's answer: the parked-dispatch resolver did not find
# the order on the venue on enough consecutive lookups
# (:class:`~pynecore.core.broker.parked_dispatch.ParkedDispatchResolver`). An
# order that did land can still show up later, so such a row does not prove
# that nothing of the dispatch executed (:func:`is_proven_refusal`).
EXTRAS_KEY_REJECTED_ON_ABSENCE = 'rejected_on_absence'

# Event kind a plugin journals under a close's client order id once the venue
# order of that close ended (cancelled, expired, deactivated, or rejected after
# the venue accepted it) and the venue's final cumulative executed quantity of
# it is zero: nothing of the close reduced the position, and nothing of it can
# any more. It is the uniform "ended with zero fill" mark, distinct from a
# venue refusal (a :data:`STATE_REJECTED` row): the row keeps its own state.
# The sync engine settles a held partial-bracket close carrying it as refused
# (the leg re-arms). Write it only from the venue's own terminal figure, never
# from the absence of a fill.
CLOSE_ENDED_UNFILLED_EVENT = 'close_ended_unfilled'

# Event kind a plugin journals under a close's client order id once the venue
# order of that close ended with a final cumulative executed quantity above
# zero and below the close row's ``qty``. The payload carries that quantity
# under :data:`CLOSE_EXECUTED_QTY_KEY`, in the unit of the close row's ``qty``
# (the plugin's wire unit). The sync engine settles a held partial-bracket close
# carrying it as partially executed: the leg re-arms for what is left.
CLOSE_ENDED_PARTIALLY_EVENT = 'close_ended_partially'

# Payload key of :data:`CLOSE_ENDED_PARTIALLY_EVENT`: the close's final executed
# quantity, in the unit of the close row's ``qty``.
CLOSE_EXECUTED_QTY_KEY = 'executed_qty'

# Submission outcome unknown — typically a network timeout between the
# POST and the response, or a successful POST without a server
# reference. The recovery path uses these rows on startup to replay or
# reconcile against the exchange.
STATE_DISPOSITION_UNKNOWN = 'disposition_unknown'

# A full-close DELETE landed at the exchange but the final fill /
# settlement has not yet been observed in the activity stream. The
# recovery path treats this state like a pending dispatch: the targets
# are re-queried against the live positions snapshot to confirm they
# have actually vanished.
STATE_CLOSING = 'closing'

# A cancel command row is mid-flight: the per-target loop has started
# but not finished. Recovery re-evaluates the targets against the
# snapshot and retries the DELETEs that did not land.
STATE_CANCEL_PENDING = 'cancel_pending'

# ``orders.state`` marker for a synthetic engine-trigger partial bracket
# leg row. The order lifecycle column does NOT track the engine-trigger
# state machine — the actual leg phase (``armed`` → ``triggering`` →
# ``triggered`` / ``cascaded_cancel`` / ...) lives under
# :data:`EXTRAS_KEY_LEG_STATE`. This marker exists so the legacy
# recovery / reconcile paths can short-circuit on these rows (they own
# no exchange-side order; the engine state machine in
# :mod:`software_partial_bracket_engine` owns them).
STATE_PARTIAL_BRACKET_LEG = 'partial_bracket_leg'

# States the recovery path must examine after a restart. ``submitted``
# is here because a crash between the initial helper and the REST call
# leaves the row in this state — same recovery semantics as
# ``disposition_unknown``. The non-entry pending states (``closing``,
# ``cancel_pending``) are added to this set by the corresponding M4
# phases once the matching ``resume_pending_dispatch`` branches exist
# on the plugin side; until then the journal does not surface them to
# the recovery loop.
#
# :data:`STATE_PARTIAL_BRACKET_LEG` is deliberately NOT a member —
# engine-trigger leg rows are replayed by
# :meth:`SoftwarePartialBracketEngine.restart_replay`, not by the
# journal's resume hook.
PENDING_DISPATCH_STATES: frozenset[str] = frozenset({
    STATE_SUBMITTED,
    STATE_SERVER_REF_SEEN,
    STATE_DISPOSITION_UNKNOWN,
})


# === Engine-trigger partial bracket leg constants ==========================

# ``extras[EXTRAS_KEY_LEG_KIND]`` for the take-profit leg of an
# engine-trigger partial bracket. The motor closes a strict subset of
# the parent position at this price level.
LEG_KIND_TP_PARTIAL = 'tp_partial'

# ``extras[EXTRAS_KEY_LEG_KIND]`` for the stop-loss leg of an
# engine-trigger partial bracket.
LEG_KIND_SL_PARTIAL = 'sl_partial'

# ``extras[EXTRAS_KEY_LEG_KIND]`` for the trailing-stop leg of an
# engine-trigger partial bracket. The ``trigger_level`` for a trail
# leg is recomputed by the engine on every favourable price move
# from ``trigger_offset``.
LEG_KIND_TRAIL_PARTIAL = 'trail_partial'

# Set of leg kinds owned by the engine-trigger state machine —
# reconciliation / replay paths key off membership here to identify
# rows whose disposition is NOT the journal's responsibility.
ENGINE_TRIGGER_LEG_KINDS: frozenset[str] = frozenset({
    LEG_KIND_TP_PARTIAL,
    LEG_KIND_SL_PARTIAL,
    LEG_KIND_TRAIL_PARTIAL,
})

# ``extras[EXTRAS_KEY_LEG_STATE]`` values — the engine-trigger state
# machine's current phase. See the partial-qty bracket exit design
# dossier §3 for the full lifecycle diagram. ``armed`` is the steady
# state where the motor watches WS price ticks; ``triggering`` is the
# transient between trigger detection and the close dispatch landing;
# ``triggered`` / ``triggered_failed`` / ``triggered_unknown`` are the
# three close-dispatch outcomes. The ``aborted_*`` and ``cascaded_*``
# branches are the terminal states for legs that the engine does not
# fire (parent gone, OCA cancellation, broker-native SL hit).
LEG_STATE_ARMED = 'armed'
LEG_STATE_PENDING_ENTRY = 'pending_entry'
LEG_STATE_TRIGGERING = 'triggering'
LEG_STATE_TRIGGERED = 'triggered'
LEG_STATE_TRIGGERED_FAILED = 'triggered_failed'
LEG_STATE_TRIGGERED_UNKNOWN = 'triggered_unknown'
LEG_STATE_ABORTED_PARENT_GONE = 'aborted_parent_gone'
LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED = 'aborted_parent_never_arrived'
LEG_STATE_CASCADED_CANCEL = 'cascaded_cancel'
LEG_STATE_CASCADED_CANCEL_BY_PARENT_CLOSE = 'cascaded_cancel_by_parent_close'
LEG_STATE_CASCADED_CANCEL_BY_NATIVE_SL = 'cascaded_cancel_by_native_sl'
# Intermediate verification state for ``pending_entry`` legs whose
# parent ``EntryIntent`` was cancelled but the broker's disposition is
# **unknown** (network timeout, ambiguous response). The leg is
# neither terminal nor active: it does not contribute to the
# ``NativeFailsafeManager`` worst-SL set, does not arm on price ticks,
# and is excluded from the diff-loop adoption path. The sync engine's
# ``reconcile()`` cancel-retry-loop resolves it within the stale-grace
# window (default 10s) by re-invoking ``execute_cancel_with_outcome``;
# the outcome flips the leg to ``aborted_parent_never_arrived`` (on
# ``CANCEL_CONFIRMED``), to ``aborted_parent_gone`` (on
# ``FILLED_AND_CLOSED``) or back to ``pending_entry`` / ``armed`` (on
# ``ALREADY_FILLED``). Stale-grace expiry promotes the parent to
# ``DEGRADED_HALT``. See the cancel-tentative state design dossier.
LEG_STATE_CANCEL_TENTATIVE = 'cancel_tentative'

# Terminal states — the row is closed (``closed_ts_ms`` is set) and
# the engine no longer owns it. Used by both
# :func:`iter_active_engine_trigger_partial_legs` (to filter them out)
# and the engine's audit emitters.
LEG_STATE_TERMINAL: frozenset[str] = frozenset({
    LEG_STATE_TRIGGERED,
    LEG_STATE_ABORTED_PARENT_GONE,
    LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED,
    LEG_STATE_CASCADED_CANCEL,
    LEG_STATE_CASCADED_CANCEL_BY_PARENT_CLOSE,
    LEG_STATE_CASCADED_CANCEL_BY_NATIVE_SL,
})

# Active states — the row is alive and the engine state machine owns
# the next transition. :data:`LEG_STATE_TRIGGERED_FAILED` and
# :data:`LEG_STATE_TRIGGERED_UNKNOWN` are intermediate (the engine
# retries / reconciles to either ``triggered`` or back to ``armed``).
LEG_STATE_ACTIVE: frozenset[str] = frozenset({
    LEG_STATE_ARMED,
    LEG_STATE_PENDING_ENTRY,
    LEG_STATE_TRIGGERING,
    LEG_STATE_TRIGGERED_FAILED,
    LEG_STATE_TRIGGERED_UNKNOWN,
})

# Live states — every non-terminal leg state. Superset of
# :data:`LEG_STATE_ACTIVE` that additionally includes
# :data:`LEG_STATE_CANCEL_TENTATIVE`: a tentative leg's row is still
# open (``closed_ts_ms IS NULL``) and must survive restart replay,
# but the engine does NOT own its next transition — the sync engine's
# cancel-retry-loop drives it via ``execute_cancel_with_outcome``.
# Used by :func:`iter_active_engine_trigger_partial_legs` to include
# tentative rows in restart rehydration, and by the leg-state update
# validator to permit transitions into ``cancel_tentative``.
LEG_STATE_LIVE: frozenset[str] = LEG_STATE_ACTIVE | frozenset({
    LEG_STATE_CANCEL_TENTATIVE,
})

# Canonical ``extras`` keys for an engine-trigger partial bracket leg
# row. Defined as string constants so a typo at a call site fails the
# import rather than producing silent dictionary divergence between
# the persist helper and the recovery / iteration helpers.
EXTRAS_KEY_LEG_KIND = 'leg_kind'
EXTRAS_KEY_LEG_STATE = 'leg_state'
EXTRAS_KEY_TRIGGER_LEVEL = 'trigger_level'
EXTRAS_KEY_TRIGGER_OFFSET = 'trigger_offset'
# Trailing-stop activation level (price units). While set, the trail
# leg is in its pre-activation phase: the engine only watches whether
# the current price has crossed this level. Cleared (set to ``None``)
# the moment activation fires, at which point :data:`EXTRAS_KEY_TRIGGER_LEVEL`
# starts carrying the moving stop level computed from
# :data:`EXTRAS_KEY_TRIGGER_OFFSET`.
EXTRAS_KEY_TRAIL_ACTIVATION_LEVEL = 'trail_activation_level'
# Pre-activation activation OFFSET in price units (i.e.
# ``trail_points_ticks * mintick``). Present only on a trail leg
# whose parent entry is still pending — at parent fill time the engine
# resolves it to an absolute :data:`EXTRAS_KEY_TRAIL_ACTIVATION_LEVEL`.
EXTRAS_KEY_TRAIL_ACTIVATION_OFFSET = 'trail_activation_offset'
EXTRAS_KEY_PARENT_ENTRY_DISPATCH_REF = 'parent_entry_dispatch_ref'
EXTRAS_KEY_INTENT_PARTIAL_QTY = 'intent_partial_qty'
EXTRAS_KEY_PARENT_PINE_ENTRY_ID = 'parent_pine_entry_id'
EXTRAS_KEY_OCA_GROUP = 'oca_group'
EXTRAS_KEY_OCA_TYPE = 'oca_type'
# Wall-clock timestamp (ms) marking when this leg entered the
# ``cancel_tentative`` state. Persisted on the leg row so that a
# restart can rehydrate the sync engine's cancel-disposition shadow
# map without losing the stale-grace deadline. Cleared the moment
# the leg leaves ``cancel_tentative`` (either back to
# ``pending_entry`` / ``armed`` on restore, or forward to
# ``aborted_parent_never_arrived`` on confirmed cancel).
EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS = 'cancel_tentative_since_ts_ms'
# Client order id of the close a ``triggering`` leg is about to send, written
# before the venue call. While set on a ``triggering`` (or, after a crash
# between the settling writes, ``triggered_unknown``) row, the close may have
# reached the venue: a restart holds the leg until the venue evidence of that
# client order id decides it, and never sends a second close for it. Cleared
# when the leg re-arms.
EXTRAS_KEY_TRIGGER_CLOSE_COID = 'trigger_close_coid'


# === Stop-limit watch constants ============================================

# ``orders.state`` marker for a software stop-limit WATCH row. A Pine entry or
# ``strategy.order`` placed with both ``stop`` and ``limit`` is ONE stop-limit
# order (measured on TradingView, BINANCE:BTCUSDT 30m): it rests as a stop
# order, and when the stop triggers it becomes a plain LIMIT at the limit
# price. On a venue without a native stop-limit order the engine emulates it:
# nothing rests on the venue before the trigger, and this row is the engine's
# own record of the order. It carries no exchange-side order (the LIMIT sent
# on the trigger has its own row) — the
# :class:`~pynecore.core.broker.software_stop_limit_engine.SoftwareStopLimitEngine`
# state machine owns it, and the journal / reconcile paths skip it
# (:data:`LEDGER_ROW_STATES`). The persisted value predates the stop-limit
# model; it stays so a row an earlier version wrote is still found.
STATE_STOP_LIMIT_WATCH = 'entry_stop_watch'

# ``extras[EXTRAS_KEY_STOP_LIMIT_STATE]`` values — the watch's phase.
# ``armed`` watches the price against the stop while nothing rests on the
# venue. ``limit_pending`` is latched on the trigger: the LIMIT's
# client-order-id is persisted BEFORE its POST, so a restart verifies the
# earlier send before sending again and never opens twice. ``triggered`` holds
# while the LIMIT lives on the venue, so a restart and the modify path still
# know the order already triggered. ``filled`` and ``aborted`` are terminal.
STOP_LIMIT_STATE_ARMED = 'armed'
STOP_LIMIT_STATE_LIMIT_PENDING = 'limit_pending'
STOP_LIMIT_STATE_TRIGGERED = 'triggered'
STOP_LIMIT_STATE_FILLED = 'filled'
STOP_LIMIT_STATE_ABORTED = 'aborted'

# States of a watch row written by a version that modelled a both-set entry as
# two OCO legs (a native LIMIT resting from the start plus a software STOP
# that fired a MARKET). Such a row is never driven: the restart retires it
# (:meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._retire_legacy_oco_watch`).
LEGACY_OCO_STATE_CANCEL_PENDING = 'cancel_pending'
LEGACY_OCO_STATE_MARKET_PENDING = 'stop_market_pending'

# Terminal states — the row is closed and the engine no longer owns it. The
# legacy OCO terminals count too, so a closed legacy row is never replayed.
STOP_LIMIT_STATE_TERMINAL: frozenset[str] = frozenset({
    STOP_LIMIT_STATE_FILLED,
    STOP_LIMIT_STATE_ABORTED,
    'limit_won',
    'stop_won',
})

# Live states — the row is alive and the engine drives its next transition.
STOP_LIMIT_STATE_LIVE: frozenset[str] = frozenset({
    STOP_LIMIT_STATE_ARMED,
    STOP_LIMIT_STATE_LIMIT_PENDING,
    STOP_LIMIT_STATE_TRIGGERED,
})

# Live states of a legacy OCO row: replayed only to be retired.
LEGACY_OCO_STATE_LIVE: frozenset[str] = frozenset({
    STOP_LIMIT_STATE_ARMED,
    LEGACY_OCO_STATE_CANCEL_PENDING,
    LEGACY_OCO_STATE_MARKET_PENDING,
})

# Canonical ``extras`` keys for a stop-limit watch row. The persisted names of
# the state and the stop level predate the stop-limit model; they stay so a
# legacy row is still recognised.
EXTRAS_KEY_STOP_LIMIT_STATE = 'entry_stop_state'
# The stop price whose cross triggers the order.
EXTRAS_KEY_STOP_LIMIT_STOP = 'entry_stop_level'
# The limit price the triggered LIMIT rests at.
EXTRAS_KEY_STOP_LIMIT_LIMIT = 'stop_limit_limit'
# The triggered LIMIT's client-order-id, persisted on the transition into
# ``limit_pending`` BEFORE the POST.
EXTRAS_KEY_STOP_LIMIT_ORDER_COID = 'stop_limit_order_coid'
# The bar the watch was armed on, and the last bar whose price it judged: a
# restart judges the bars after it, which the engine never saw.
EXTRAS_KEY_STOP_LIMIT_ARMED_BAR = 'stop_limit_armed_bar_ts'
EXTRAS_KEY_STOP_LIMIT_CHECKED_BAR = 'stop_limit_checked_bar_ts'
# Only a legacy OCO row carries these: the native LIMIT leg's client-order-id
# (the marker of a legacy row) and the stop-fired MARKET's client-order-id.
LEGACY_EXTRAS_KEY_OCO_LIMIT_COID = 'entry_stop_limit_coid'
LEGACY_EXTRAS_KEY_OCO_MARKET_COID = 'entry_stop_market_coid'


# === One-way emulation close-leg constants =================================

# ``orders.state`` marker for a one-way emulation close-leg row. The row owns no
# exchange-side order of its own — it records one leg of a ``CloseIntent`` fanned
# FIFO across a hedging account's legs, so the
# :class:`~pynecore.core.broker.one_way_emulator.OneWayEmulator` can re-derive an
# interrupted fan-out on restart. Like :data:`STATE_PARTIAL_BRACKET_LEG`, this
# marker tells the journal / reconcile paths to short-circuit: the leg's close
# FILL itself flows through the normal natural-close path, applied FIFO by
# ``record_fill`` as a ``LegType.CLOSE``.
STATE_CLOSE_LEG = 'close_leg'

# ``extras[EXTRAS_KEY_CLOSE_LEG_STATE]`` phases of one fanned close leg.
# ``pending`` — row persisted, the per-leg close has NOT been acked yet.
# ``dispatched`` — the transport's ``close_leg`` returned without raising
#   (terminal; the row is closed once no send of it is reserved any more, see
#   :data:`EXTRAS_KEY_CLOSE_LEG_RESERVED`). ``restart_replay`` resumes only ``pending``
#   legs (a crash before the ack) and, before it re-sends anything, establishes
#   what the close already executed (the venue's record of every attempt, and
#   how much the leg lost since the dispatch), so an executed close is never
#   repeated.
CLOSE_LEG_STATE_PENDING = 'pending'
CLOSE_LEG_STATE_DISPATCHED = 'dispatched'

# Live (non-terminal) close-leg phases ``restart_replay`` must resume.
CLOSE_LEG_STATE_LIVE: frozenset[str] = frozenset({CLOSE_LEG_STATE_PENDING})

# Canonical ``extras`` keys for a close-leg row.
EXTRAS_KEY_CLOSE_LEG_STATE = 'close_leg_state'
# Broker leg id (cTrader ``positionId``) this row closes.
EXTRAS_KEY_CLOSE_LEG_ID = 'close_leg_id'
# The owning CloseIntent dispatch's ``KIND_CLOSE`` client-order-id; the per-leg
# coid is derived deterministically as ``f"{parent}:{leg_id}"`` so a
# restart-mid-fan never produces two rows for one leg.
EXTRAS_KEY_CLOSE_PARENT_COID = 'close_parent_coid'
# Broker-grid integer volume dispatched for this leg (audit + replay residual).
EXTRAS_KEY_CLOSE_LEG_VOLUME = 'close_leg_volume'
# Broker-grid integer volume of the WHOLE fan the leg's close belongs to (every
# leg the plan meant to send), written on each leg row before its send. A leg
# the fan never reached (a crash between two sends, a plugin skip that ended the
# fan) has no row, so what the rows hold does not show the close's full size;
# this does.
EXTRAS_KEY_CLOSE_FAN_VOLUME = 'close_fan_volume'
# Broker-grid integer volume the close was ASKED for (a strategy or trigger
# close), written on each leg row next to the fan volume. The plan can differ
# from it both ways: clipped to the net exposure or short of a leg it cannot
# close, or widened to whole legs on a venue that closes a leg only in full.
# What the close executed is measured against this, capped at it, so neither a
# short plan reads as complete nor a widened one under-credits the close.
EXTRAS_KEY_CLOSE_REQUESTED_VOLUME = 'close_requested_volume'
# Broker-grid size of the leg when its close was dispatched: the replay reads
# how much the leg lost since then as evidence of what the close executed.
EXTRAS_KEY_CLOSE_LEG_PRE_VOLUME = 'close_leg_pre_volume'
# Every client-order-id the leg's close went out under, oldest first: the row's
# own coid, then each replay re-send of the part still open. The replay asks the
# venue what all of them executed before it sends anything more.
EXTRAS_KEY_CLOSE_LEG_ATTEMPTS = 'close_leg_attempts'
# Epoch ms (local clock) when the first of those sends was about to go out: the
# ``since_ms`` the venue is asked from. Kept in extras because reopening the row
# for a later settlement restarts its ``created_ts_ms``.
EXTRAS_KEY_CLOSE_LEG_SINCE_MS = 'close_leg_since_ms'
# The sends of the leg's close the position read may not show yet, oldest
# first: ``[[attempt_coid, volume, pre_volume, since_ms, sent_ns], ...]`` with
# the send's broker-grid volume, the leg's broker-grid size the send is
# credited from, the epoch ms the venue is asked about it from, and the epoch
# ns it went out at (the send order across rows). A send is recorded BEFORE its
# wire call and dropped once the read or the venue's record shows it; the row
# stays live while any send is listed, so a restart rebuilds the reserved view
# from it.
EXTRAS_KEY_CLOSE_LEG_RESERVED = 'close_leg_reserved'


# === One-way emulation residual-open constants =============================

# ``orders.state`` marker for a reversal's residual-OPEN intent. Like
# :data:`STATE_CLOSE_LEG`, the row owns no exchange order and the journal /
# reconcile / startup-recovery paths short-circuit on it (they only act on
# ``submitted`` / ``confirmed`` / ``closing`` / ``rejected`` rows). It is a
# persist-first breadcrumb written by
# :meth:`~pynecore.core.broker.one_way_emulator.OneWayEmulator.run_reversal`
# BEFORE the FIFO closes, so a crash in the window between the closes landing
# and the residual ``place_leg`` persisting its own entry row does not lose the
# residual open. :meth:`OneWayEmulator.restart_replay` reconciles it against the
# residual entry row (re-dispatching the open only when that row never landed)
# and then clears it.
STATE_RESIDUAL_OPEN = 'residual_open'

# Canonical ``extras`` keys for a residual-open row. The deterministic
# ``KIND_ENTRY`` coid of the residual ``place_leg`` (existence proves the open
# already entered the persist-first entry path / 2.x recovery), plus the
# envelope fields needed to rebuild the dispatch on replay (the residual is
# always a MARKET entry; the early resting-order branch never persists one).
EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID = 'residual_open_entry_coid'
EXTRAS_KEY_RESIDUAL_OPEN_RUN_TAG = 'residual_open_run_tag'
EXTRAS_KEY_RESIDUAL_OPEN_BAR_TS_MS = 'residual_open_bar_ts_ms'
EXTRAS_KEY_RESIDUAL_OPEN_RETRY_SEQ = 'residual_open_retry_seq'
# The entry coid and ``retry_seq`` name the open's CURRENT attempt. A
# ``place_leg`` that raised
# :class:`~pynecore.core.broker.exceptions.ExchangeConnectionError` (the request
# never left the client) may leave its persist-first ``submitted`` entry row
# behind; the breadcrumb then moves on to the next attempt
# (:func:`advance_residual_open_attempt`), so only a row of the current attempt's
# coid reads as an open that went out.
# The close plan of a reversal-path reduction or flatten that was cut short after
# a close went out (``qty`` 0, nothing to open): ``[[leg_id, pine_qty], ...]``.
# Recovery closes only these legs, at most the planned quantity of each, never
# every opposing leg the book shows.
EXTRAS_KEY_RESIDUAL_OPEN_CLOSE_PLAN = 'residual_open_close_plan'
# The close-leg row coids a reversal's breadcrumb has counted against its
# combined order: ``[coid, ...]``. A reversal's breadcrumb is written with the
# combined order size Pine dispatched as its ``qty``; every close-leg row the
# reversal's closes go out under is counted once, lowering ``qty`` by the Pine
# quantity it closes, so ``qty`` is always what is left to open. A close a
# drain sends because an earlier close of another intent executed less than it
# held is counted just like the reversal's own planned ones.
EXTRAS_KEY_RESIDUAL_OPEN_COUNTED_CLOSES = 'residual_open_counted_closes'
# The netting record (:data:`STATE_NETTING`) a close-plan breadcrumb belongs to:
# one of the two halves of a fill-time netting pass. Its closes are paired, not
# a reduction of the net position, so a restart's close-leg budget does not cap
# them.
EXTRAS_KEY_RESIDUAL_OPEN_NETTING = 'residual_open_netting'


# === One-way emulation fill-time netting constants =========================

# ``orders.state`` marker of a fill-time netting record. A resting LIMIT / STOP
# entry on a hedging account rests as a leg of its own; once it fills against
# opposing legs, the venue holds both sides where a one-way account would hold
# only the net. The sync engine writes one record per such fill slice
# (PERSIST-FIRST, before any venue write, keyed by the fill id), and the
# :class:`~pynecore.core.broker.one_way_emulator.OneWayEmulator` nets it: it
# closes the opposing legs FIFO and the new leg by the same quantity, through
# two close-plan breadcrumbs (:data:`STATE_RESIDUAL_OPEN`, flagged
# :data:`EXTRAS_KEY_RESIDUAL_OPEN_NETTING`). Like the other ledger rows it owns
# no exchange order. ``qty`` is the quantity still to net; ``side`` the fill's.
STATE_NETTING = 'netting'
# The fill id of the slice the record nets (its idempotency key).
EXTRAS_KEY_NETTING_FILL_ID = 'netting_fill_id'
# The Pine entry id of the filled entry; its legs are reduced first.
EXTRAS_KEY_NETTING_ENTRY_ID = 'netting_entry_id'
# Epoch ms the record was written at.
EXTRAS_KEY_NETTING_SINCE_MS = 'netting_since_ms'
# How many netting passes planned closes for the record so far; the breadcrumbs
# of pass ``n`` are keyed by it.
EXTRAS_KEY_NETTING_PASSES = 'netting_passes'
# The close fills the record's planned closes are expected to bring that move no
# book: ``{'buy': qty, 'sell': qty}`` (Pine units, by fill side), the quantity
# each side nets. The fills of the closes are book-neutral up to it.
EXTRAS_KEY_NETTING_EXPECTED = 'netting_expected'
# Epoch ms since the record has nothing left to plan or close; the expectation
# then waits at most the fill-report grace for its fills.
EXTRAS_KEY_NETTING_SETTLED_MS = 'netting_settled_ms'
# Close fill sides (``['buy', 'sell']``) on which a close that is not the
# record's own could still fill when the record was written (its close-leg row
# live, or closed less than the fill-report grace before), or was fanned out
# while the record was live. A close fill of such a
# side that names no close the run sent cannot be told from the record's own
# fills (:meth:`~pynecore.core.broker.one_way_emulator.OneWayEmulator.netting_close_owner`).
EXTRAS_KEY_NETTING_CONTENDED = 'netting_contended'

# ``orders.state`` marker of the sync engine's count of a netting record's
# expected close fills it booked as netting (``{'buy': qty, 'sell': qty}`` under
# :data:`EXTRAS_KEY_NETTING_EXPECTED`). Keyed ``{record}:absorbed`` and written
# by the engine only, in the transaction of the fills' receipts, so a fill
# counts toward it exactly when it is in the books.
STATE_NETTING_ABSORBED = 'netting_absorbed'


# === One-way emulation bracket-ownership constants =========================

# ``orders.state`` marker for a per-leg bracket-ownership row. Like
# :data:`STATE_CLOSE_LEG`, the row owns no exchange order: it records that one
# exit's native TP/SL/trailing bracket was replicated onto one broker leg of a
# hedging position, so the
# :class:`~pynecore.core.broker.one_way_emulator.OneWayEmulator` can clear ONLY
# the legs a given exit owns (the plugin's broadcast clear strips brackets a
# different exit set) and re-assert / release them on restart. The marker tells
# the journal / reconcile paths to short-circuit, exactly as the close-leg and
# partial-bracket-leg markers do.
STATE_BRACKET_OWN = 'bracket_own'

# ``extras[EXTRAS_KEY_BRACKET_OWN_STATE]`` phases of one owned leg's bracket.
# ``active`` — the bracket is replicated onto this leg (steady state). ``clearing``
#   — a clear is in flight: the row is marked PERSIST-FIRST before the amend-to-
#   clear, mirroring the close-leg ``pending`` -> ``dispatched`` two-phase, so an
#   ambiguous (timed-out) clear re-runs the CLEAR on restart instead of
#   re-asserting the original levels (which would resurrect a bracket the script
#   cancelled). ``released`` — the bracket was cleared from this leg (terminal; the
#   row is closed). ``restart_replay`` re-asserts ``active`` rows whose leg is
#   still open, finishes ``clearing`` rows by re-clearing them, and releases any
#   whose leg vanished.
BRACKET_OWN_STATE_ACTIVE = 'active'
BRACKET_OWN_STATE_CLEARING = 'clearing'
BRACKET_OWN_STATE_RELEASED = 'released'

# Live (non-terminal) bracket-ownership phases ``restart_replay`` re-asserts /
# finishes. ``clearing`` stays live so a clear interrupted by a crash or an
# ambiguous timeout is resumed (re-cleared, then released), never left dangling.
BRACKET_OWN_STATE_LIVE: frozenset[str] = frozenset(
    {BRACKET_OWN_STATE_ACTIVE, BRACKET_OWN_STATE_CLEARING}
)

# Every ``orders.state`` marker of a ledger row that owns no exchange order of
# its own. A path walking the journal for venue orders (an ``orderLinkId``
# lookup, a fill walk, a terminality check) skips these rows: the venue never
# heard of their client order id.
LEDGER_ROW_STATES: frozenset[str] = frozenset({
    STATE_PARTIAL_BRACKET_LEG,
    STATE_STOP_LIMIT_WATCH,
    STATE_CLOSE_LEG,
    STATE_RESIDUAL_OPEN,
    STATE_NETTING,
    STATE_NETTING_ABSORBED,
    STATE_BRACKET_OWN,
})

# Canonical ``extras`` keys for a bracket-ownership row.
EXTRAS_KEY_BRACKET_OWN_STATE = 'bracket_own_state'
# Broker leg id (cTrader ``positionId``) this exit's bracket is replicated onto.
EXTRAS_KEY_BRACKET_OWN_LEG_ID = 'bracket_own_leg_id'
# The owning exit's bracket-attach dispatch coid (``KIND_EXIT_SL``); the per-leg
# row coid is derived as ``f"{attach_coid}:{leg_id}"`` so a re-attach upserts the
# SAME row and two different exits get disjoint coid namespaces.
EXTRAS_KEY_BRACKET_OWN_ATTACH_COID = 'bracket_own_attach_coid'
# The amend-to-clear dispatch coid (``KIND_CANCEL``) recorded PERSIST-FIRST when a
# row enters ``clearing``, so a restart re-CLEARS under the SAME clear coid instead
# of the attach coid, so the venue call correlates with the retirement. The coid is
# a correlation id only: a retry may carry other levels than the first call under
# it, and ``PositionPort.amend_bracket`` must never dedup an amend by coid.
EXTRAS_KEY_BRACKET_OWN_CLEAR_COID = 'bracket_own_clear_coid'
# Pine-unit protective levels the owning exit asks for on the leg (audit + modify
# diff). Several exits can own one leg; the leg itself carries the merge of every
# active owner's levels, computed by the emulator, never one row's alone.
EXTRAS_KEY_BRACKET_OWN_TP = 'bracket_own_tp'
EXTRAS_KEY_BRACKET_OWN_SL = 'bracket_own_sl'
# Absolute trailing-stop activation price (``ExitIntent.trail_price``). Persisted
# so a restart can faithfully rebuild a pending-trailing exit's Pine ``Order``;
# ``trail_offset`` alone cannot distinguish a pre-activation trail from an active
# one. Absent on rows written before this key existed — reconstruct trail_price
# as ``None`` in that case (never tear the bracket down over the gap).
EXTRAS_KEY_BRACKET_OWN_TRAIL_PRICE = 'bracket_own_trail_price'
EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET = 'bracket_own_trail_offset'
# The owning exit's OCA group identity (``ExitIntent.oca_name`` /
# ``oca_type``). Persisted so a restart rebuilds the Pine ``Order`` with the same
# group it was emitted under — otherwise the reconstructed exit carries no OCA
# group and an explicit ``oca_type='cancel'`` cross-bracket cascade silently
# stops firing across the restart. Absent on rows written before these keys
# existed — reconstruct as ``None`` in that case (a single-member synthetic
# reduce group is a cascade no-op, so the gap only ever loses an explicit group).
EXTRAS_KEY_BRACKET_OWN_OCA_NAME = 'bracket_own_oca_name'
EXTRAS_KEY_BRACKET_OWN_OCA_TYPE = 'bracket_own_oca_type'
# ``True`` while the leg is protected by the emulator's software leg guard and
# carries NO venue bracket of this run (the venue bracket could not be limited
# to the protected quantity). Absent or ``False`` means the leg may hold a venue
# bracket this run wrote, which a switch to the software guard must clear first.
# The flag is set only after that clear is confirmed and dropped before a venue
# bracket is written again, so a crash in between leaves the conservative state.
EXTRAS_KEY_BRACKET_OWN_SOFTWARE = 'bracket_own_software'
# The trailing stop of the leg's software guard: the most favourable price the
# guard has trailed from (a long's highest, a short's lowest; its stop lies the
# trailing distance behind it), and the merged trailing offset in Pine ticks it
# trails at. Written on every live ownership row of the leg before the guard
# takes a new extreme, so a restart resumes the trail from it instead of from
# the current price and the restored stop is never looser than the last
# persisted one. Absent while the guard has seen no price (not armed), and
# dropped once the leg gets a venue bracket again; an offset that differs from
# the guard's current one is not resumed.
EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_EXTREME = 'bracket_own_guard_trail_extreme'
EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_OFFSET = 'bracket_own_guard_trail_offset'


# === Entry order row lifecycle =============================================

def create_entry_order_row(
        store: 'RunContext',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        pine_entry_id: str,
        kind: str,
        order_type: str,
) -> None:
    """Insert the initial ``submitted`` row for an entry dispatch.

    Replaces the manual
    ``store.upsert_order(coid, ..., extras={'kind': ..., 'order_type': ...})``
    that every plugin re-implements. The function does **not** write an
    audit event — the orchestrator does that *after* the upsert so a
    crash between the two does not leave an event without its row.

    :param store: The active run context (`plugin.store_ctx`).
    :param coid: The dispatch's canonical client-order-id.
    :param symbol: Exchange-side symbol (epic for Capital.com, etc.).
    :param side: ``'buy'`` or ``'sell'`` — Pine intent's side.
    :param qty: Already quantized to the broker's lot step.
    :param intent_key: The Pine intent's diff key (e.g. ``pine_id``).
    :param pine_entry_id: The Pine ``strategy.entry(id=...)`` value.
    :param kind: One of :data:`ENTRY_KIND_POSITION`,
        :data:`ENTRY_KIND_WORKING`. Decides how downstream recovery and
        activity reconcile interpret the row.
    :param order_type: The ``OrderType`` enum's ``.value`` (``'market'``,
        ``'limit'``, ``'stop'``). Persisted into extras so recovery can
        replay an entry without re-deriving it from the intent.
    """
    if kind not in (ENTRY_KIND_POSITION, ENTRY_KIND_WORKING):
        raise ValueError(
            f"create_entry_order_row: kind must be one of "
            f"{{ENTRY_KIND_POSITION, ENTRY_KIND_WORKING}}, got {kind!r}"
        )
    # Reopen a previously-closed row sharing this COID before writing the
    # fresh submitted fields. ``DispatchEnvelope.client_order_id`` is a pure
    # function of (run_tag, pine_id, bar_ts_ms, kind, retry_seq), so a Pine
    # strategy that cancels an order and re-creates it under the same id
    # within the same bar produces an *identical* COID. The first lifecycle
    # closed the row (``closed_ts_ms`` set); ``upsert_order`` then takes its
    # UPDATE path, which does not touch ``closed_ts_ms`` — leaving the
    # re-created order invisible to ``iter_live_orders`` and therefore to the
    # cancel sweep, reconcile, and fill-fallback. A later ``strategy.cancel``
    # finds no live target and degrades to a no-op while the order is still
    # live on the exchange. Clearing ``closed_ts_ms`` and resetting the
    # per-lifecycle fields (exchange id, fill) restores true insert semantics
    # for the fresh submission. Mirrors the plugin-side TP/SL reopen guard.
    existing = store.get_order(coid)
    if existing is not None and existing.closed_ts_ms is not None:
        store.reopen_order(coid)
        store.upsert_order(coid, exchange_order_id=None, filled_qty=0.0)
    # A row an earlier process left pending now tracks this process's send.
    store.release_inherited_dispatch(coid)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_SUBMITTED,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        extras={'kind': kind, 'order_type': order_type},
    )


def record_server_ref(
        store: 'RunContext',
        *,
        coid: str,
        deal_reference: str,
        kind: str,
        order_type: str,
) -> None:
    """Record the exchange's submission reference + advance state.

    Two writes in a defined order:

    1. ``order_refs`` row keyed by ``'deal_reference'`` so activity /
       reconcile can resolve the reference back to the COID with a
       single indexed lookup.
    2. ``orders.extras`` updated to mirror the reference *together
       with* ``orders.state`` advanced to
       :data:`STATE_SERVER_REF_SEEN` — single
       :meth:`RunContext.upsert_order` transaction, so state + extras
       can never disagree.

    Crash-safety: if the process crashes between (1) and (2), the row
    is still in :data:`STATE_SUBMITTED` (pending), and the resume hook
    sees the ``deal_reference`` via the ``order_refs`` table —
    :func:`~pynecore.core.broker.journal._collect_refs_for` materialises
    it from there. That is the contract :func:`find_pending_dispatch`
    relies on.

    :param store: The active run context.
    :param coid: The dispatch's COID.
    :param deal_reference: Server-allocated reference for the POST.
    :param kind: Same value originally passed to
        :func:`create_entry_order_row`. Repeated here because
        ``upsert_order(extras=...)`` overwrites the whole dict.
    :param order_type: Same value originally passed to
        :func:`create_entry_order_row`. Repeated for the same reason.
    """
    store.add_ref(coid, 'deal_reference', deal_reference)
    store.upsert_order(
        coid,
        state=STATE_SERVER_REF_SEEN,
        extras={
            'kind': kind,
            'order_type': order_type,
            'deal_reference': deal_reference,
        },
    )


def mark_confirmed_with_fill(
        store: 'RunContext',
        *,
        coid: str,
        exchange_id: str | None,
        is_filled: bool,
        filled_qty: float,
        fill_price: float | None,
) -> None:
    """Promote the row to ``confirmed``, optionally recording a fill.

    A MARKET entry that fills immediately needs four facts persisted in
    a single logical step:

    - ``order_refs`` row keyed by ``'deal_id'`` (so activity tail rows
      that carry only the deal id can map back).
    - ``orders.exchange_order_id`` populated.
    - ``orders.state`` = :data:`STATE_CONFIRMED`.
    - ``orders.filled_qty`` set and ``extras['confirm_level']`` set
      from the confirm response, as a recovery fallback when the
      ``/history/activity`` row arrives with ``level=0`` and the
      ``/positions`` snapshot is also empty for this deal id.

    LIMIT / STOP entries call this with ``is_filled=False``; only the
    first three facts apply.

    :param store: The active run context.
    :param coid: The dispatch's COID.
    :param exchange_id: Exchange-allocated id (``dealId`` for
        Capital.com). May be ``None`` if confirm returns no id — the
        function still advances state but skips the id-related writes.
    :param is_filled: ``True`` only for MARKET-side fills that confirm
        as OPEN. LIMIT / STOP submissions land as live working orders
        and pass ``False`` here; the fill arrives later via the
        activity stream.
    :param filled_qty: Confirmed fill quantity. Ignored when
        ``is_filled`` is ``False``.
    :param fill_price: Confirm-side fill price. Persisted under
        ``extras['confirm_level']`` only when ``is_filled`` is ``True``
        and the value is strictly positive — a zero/negative level is
        a no-quote artefact and would corrupt the recovery fallback.

    Crash-safety: ``exchange_order_id``, ``state``, ``filled_qty``, and
    the merged ``extras`` (with ``confirm_level``) are written in a
    *single* :meth:`RunContext.upsert_order` transaction, so a crash
    cannot leave the row in :data:`STATE_CONFIRMED` (terminal) without
    its fill details. The ``deal_id`` ref is written first as a separate
    transaction — if a crash occurs between the ref write and the
    consolidated update, the row's state is still
    :data:`STATE_SERVER_REF_SEEN` (pending), so
    :func:`find_pending_dispatch` re-yields it and the resume hook can
    rebuild the confirmation from the ``deal_id`` already in
    ``order_refs``.
    """
    if exchange_id:
        store.add_ref(coid, 'deal_id', exchange_id)

    fields: dict[str, Any] = {'state': STATE_CONFIRMED}
    if exchange_id:
        fields['exchange_order_id'] = exchange_id
    if is_filled:
        fields['filled_qty'] = filled_qty
        if fill_price is not None and fill_price > 0.0:
            existing = store.get_order(coid)
            merged = dict(existing.extras or {}) if existing is not None else {}
            merged['confirm_level'] = fill_price
            fields['extras'] = merged
    store.upsert_order(coid, **fields)


def mark_disposition_unknown(
        store: 'RunContext',
        *,
        coid: str,
) -> None:
    """Flip the row to :data:`STATE_DISPOSITION_UNKNOWN`.

    Used when a POST times out or the response is missing the server
    reference. The row is *not* deleted — recovery on the next restart
    re-evaluates it against the exchange's authoritative view.

    From this state on, a resolver may settle the dispatch from venue data
    (:func:`~pynecore.core.broker.parked_dispatch.awaits_venue_answer`), which
    can happen before the engine parks it. The verdict slot is opened first
    (:meth:`~RunContext.await_park_row`), so a verdict recorded in that gap
    lands with the park row instead of finding neither.
    """
    store.await_park_row(coid)
    store.set_order_state(coid, STATE_DISPOSITION_UNKNOWN)


def is_proven_refusal(row: 'OrderRow') -> bool:
    """Whether a journal row proves that its dispatch executed nothing, ever.

    A :data:`STATE_REJECTED` row with nothing filled does, unless its rejection
    is the parked-dispatch resolver's inference from the order's absence on the
    venue (:data:`EXTRAS_KEY_REJECTED_ON_ABSENCE`): an order that did land can
    show up executed after that verdict.

    :param row: The dispatch's journal row.
    :return: ``True`` for a venue refusal with nothing filled.
    """
    return (row.state == STATE_REJECTED and row.filled_qty <= 0.0
            and not (row.extras or {}).get(EXTRAS_KEY_REJECTED_ON_ABSENCE))


def mark_rejected(
        store: 'RunContext',
        *,
        coid: str,
) -> None:
    """Flip the row to :data:`STATE_REJECTED` (terminal) and close it.

    Used when the exchange returns a definitive reject — confirm
    ``REJECTED``, a 4xx response, or a synchronous reason string the
    plugin maps to ``ExchangeOrderRejectedError``. The row stays in
    the table for audit, but leaves the live set: nothing of a rejected
    dispatch can land on the venue, and a live row reads as an order
    the run still owns (an entry awaiting its fill defers every exit on
    its side). The engine clears its intent slot on the next sync.

    Both writes share one transaction: a ``rejected`` row left live is
    not pending, so no recovery pass would ever close it.
    """
    with store.transaction():
        store.set_order_state(coid, STATE_REJECTED)
        store.close_order(coid)


# === Recovery query ========================================================

def find_pending_dispatch(store: 'RunContext') -> Iterator['OrderRow']:
    """Yield live rows whose dispatch the journal still owns.

    A row is *pending* when its state is in
    :data:`PENDING_DISPATCH_STATES`. The journal's
    :meth:`~pynecore.core.broker.journal.DispatchJournal.recover_pending`
    iterates these on startup and asks the plugin's
    ``resume_pending_dispatch`` hook for a verdict.

    The query relies on the partial ``idx_orders_live`` index, so the
    cost is O(log n) per call.
    """
    for row in store.iter_live_orders():
        if row.state in PENDING_DISPATCH_STATES:
            yield row


# === Startup position adoption =============================================

def adopt_untracked_position_legs(
        store: 'RunContext',
        legs: 'Iterable[PositionLeg]',
        *,
        symbol: str,
        ref_kind: str = 'deal_id',
) -> int:
    """Seed confirmed ``position`` rows for untracked live venue legs.

    For venues that close positions by addressing a broker-native
    position handle (Capital.com / IG-style ``DELETE
    /positions/{dealId}``), ``execute_close`` / ``execute_exit`` derive
    their targets from confirmed ``position`` rows. A fresh process (or
    a run under a new ``run_id``) that finds existing venue exposure has
    no such row — the engine's startup reconcile adopts the net size
    into the in-memory position, but a subsequent
    ``strategy.close_all()`` would find no confirmed row to route the
    close through and strand the adopted exposure. Seeding one confirmed
    row per untracked leg restores the normal close/exit paths.

    Hedging accounts must NOT call this: there the core one-way emulator
    closes per leg via ``close_leg`` reading ``fetch_raw_positions``
    directly, and a seeded row would double-count against the emulator's
    virtual FIFO.

    ``legs`` must be symbol-scoped — the output of the plugin's
    ``fetch_raw_positions(symbol)`` transport primitive or an equivalent
    parse of an already-fetched snapshot, NEVER a raw account-wide
    payload. The scoping is enforced structurally: a leg whose
    ``symbol`` differs raises :class:`ValueError`, because adopting a
    foreign instrument plants a row that same-``run_id`` adoption then
    copies into every later restart (observed live on Capital.com when a
    lane switched instruments over a weekend).

    Idempotent: a leg already tracked by a live store row (matched on
    ``exchange_order_id`` or a ``ref_kind`` ref — e.g. an entry row
    adopted across a same-``run_id`` restart, or a row just promoted by
    in-flight recovery), or owned by another run on the same account
    (:meth:`~pynecore.core.broker.storage.RunContext.foreign_live_exchange_order_ids`),
    is skipped, so no duplicate row is created. Rows are seeded
    ``filled_qty == qty`` with ``kind=ENTRY_KIND_POSITION`` +
    ``entry_filled_at`` so fill/activity classification treats the
    eventual close leg as a reduction, and a ``ref_kind`` ref mirrors
    the normal entry path so venue events can route back to the row. A
    ``startup_position_adopted`` audit event is written per seeded row.

    :param store: The active run context (``plugin.store_ctx``).
    :param legs: Symbol-scoped candidate legs (untracked ones are seeded).
    :param symbol: The single symbol this run trades.
    :param ref_kind: Ref namespace for the broker-native leg id
        (``'deal_id'`` for Capital.com-style venues).
    :return: Number of legs actually adopted after dedup.
    """
    tracked_ids: set[str] = set()
    for row in store.iter_live_orders():
        if row.exchange_order_id:
            tracked_ids.add(str(row.exchange_order_id))
    foreign_ids = store.foreign_live_exchange_order_ids()

    adopted_count = 0
    for leg in legs:
        if leg.symbol != symbol:
            raise ValueError(
                f"adopt_untracked_position_legs: leg {leg.leg_id!r} is on "
                f"{leg.symbol!r} but this run trades {symbol!r} — pass only "
                f"symbol-scoped legs (fetch_raw_positions output), never an "
                f"account-wide snapshot"
            )
        if leg.qty <= 0.0 or leg.side not in ('buy', 'sell'):
            continue
        leg_id = str(leg.leg_id)
        if not leg_id or leg_id in tracked_ids or leg_id in foreign_ids:
            continue
        if store.find_by_ref(ref_kind, leg_id) is not None:
            continue
        synthetic_coid = f"__pyne_adopted__{symbol}__{leg_id}"
        store.upsert_order(
            synthetic_coid,
            symbol=symbol,
            side=leg.side,
            qty=leg.qty,
            filled_qty=leg.qty,
            state=STATE_CONFIRMED,
            exchange_order_id=leg_id,
            extras={
                'kind': ENTRY_KIND_POSITION,
                'entry_filled_at': epoch_time(),
                ADOPTED_STARTUP_EXTRA_KEY: True,
            },
        )
        store.add_ref(synthetic_coid, ref_kind, leg_id)
        store.log_event(
            'startup_position_adopted',
            client_order_id=synthetic_coid,
            exchange_order_id=leg_id,
            payload={
                'symbol': symbol,
                'side': leg.side,
                'qty': leg.qty,
            },
        )
        tracked_ids.add(leg_id)
        adopted_count += 1

    return adopted_count


# === Close lifecycle =======================================================

def create_close_target_row(
        store: 'RunContext',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        kind: str,
        pine_entry_id: str | None = None,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Insert the initial ``submitted`` row for a close dispatch.

    Both full-close and partial-close dispatches start with the same
    persist-first row; the ``kind`` discriminator (:data:`KIND_FULL_CLOSE`
    vs :data:`KIND_PARTIAL_CLOSE`) decides the downstream lifecycle and
    the recovery contract.

    :param store: The active run context.
    :param coid: The dispatch's COID
        (``envelope.client_order_id(KIND_CLOSE)``).
    :param symbol: Exchange-side symbol (epic).
    :param side: Side of the closing leg — for a full close this is
        the side of the position being closed; for a partial close
        this is the opposite-direction emulation side.
    :param qty: Quantity to close, already quantized.
    :param intent_key: The Pine intent's diff key.
    :param kind: :data:`KIND_FULL_CLOSE` or :data:`KIND_PARTIAL_CLOSE`.
    :param pine_entry_id: Optional Pine ``strategy.entry(id=...)`` value
        when the close is tied to a single named entry. ``None`` when
        the close sweeps multiple positions (e.g. full close across
        ids).
    :param extra_payload: Optional extras to merge in alongside the
        canonical ``kind`` key. Use for plugin-specific bookkeeping
        the recovery contract needs (e.g. ``pre_total_units``,
        ``intent_units`` for the partial-close emulation).
    """
    if kind not in CLOSE_KINDS:
        raise ValueError(
            f"create_close_target_row: kind must be one of {{KIND_FULL_CLOSE, "
            f"KIND_PARTIAL_CLOSE}}, got {kind!r}"
        )
    extras: dict[str, Any] = {'kind': kind}
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_SUBMITTED,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        extras=extras,
    )


def record_close_server_ref(
        store: 'RunContext',
        *,
        coid: str,
        deal_reference: str,
        kind: str,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Record the partial-close POST's server reference + advance state.

    Only the partial-close path issues a POST that allocates a
    ``dealReference``; the full-close DELETE returns no such id, so
    this helper is not called there. The two-write order matches
    :func:`record_server_ref`: ``order_refs`` first, then a consolidated
    ``upsert_order`` that mirrors the reference into ``extras`` and
    advances state to :data:`STATE_SERVER_REF_SEEN`.

    :param store: The active run context.
    :param coid: The dispatch's COID.
    :param deal_reference: Server-allocated reference for the POST.
    :param kind: Must be :data:`KIND_PARTIAL_CLOSE`. Re-stated because
        ``upsert_order(extras=...)`` overwrites the whole dict.
    :param extra_payload: Optional extras to preserve through the
        state advance (e.g. ``pre_total_units``, ``intent_units``).
    """
    if kind != KIND_PARTIAL_CLOSE:
        raise ValueError(
            f"record_close_server_ref: kind must be {KIND_PARTIAL_CLOSE!r}, "
            f"got {kind!r}"
        )
    store.add_ref(coid, 'deal_reference', deal_reference)
    # ``upsert_order(extras=...)`` overwrites the whole dict, so read the
    # current extras first and merge: plugin-side context written between
    # row creation and this helper (e.g. ``pre_total_units`` /
    # ``intent_units`` from the partial-close emulation's pre-snapshot)
    # must survive the state advance so recovery has them.
    existing = store.get_order(coid)
    extras: dict[str, Any] = dict(existing.extras or {}) if existing else {}
    extras['kind'] = kind
    extras['deal_reference'] = deal_reference
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        state=STATE_SERVER_REF_SEEN,
        extras=extras,
    )


def mark_closing(
        store: 'RunContext',
        *,
        coid: str,
        kind: str,
        targets: list[str],
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Flip a full-close row to :data:`STATE_CLOSING` and pin its targets.

    Called after every DELETE has been issued but before the activity
    stream has confirmed the resulting fills. The ``targets`` list is
    persisted under ``extras['targets']`` so a restart-mid-stream can
    reconcile each ``dealId`` against the live positions snapshot.

    :param store: The active run context.
    :param coid: The dispatch's COID.
    :param kind: Must be :data:`KIND_FULL_CLOSE`. Partial-close uses
        :func:`mark_close_completed` directly because its emulation
        finishes synchronously inside the POST.
    :param targets: Exchange ``dealId`` strings the DELETE chain
        targeted. Empty list is a no-op close (the engine should have
        elided the dispatch, but this is harmless).
    :param extra_payload: Optional extras to preserve.
    """
    if kind != KIND_FULL_CLOSE:
        raise ValueError(
            f"mark_closing: kind must be {KIND_FULL_CLOSE!r}, got {kind!r}"
        )
    extras: dict[str, Any] = {
        'kind': kind,
        'targets': list(targets),
    }
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        state=STATE_CLOSING,
        extras=extras,
    )


def mark_close_completed(
        store: 'RunContext',
        *,
        coid: str,
        kind: str,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Finalise a close dispatch row's state to ``confirmed``.

    For full close, called once every target's vanished position has
    been observed (or the recovery path has decided the targets are
    gone). For partial close, called when the POST + post-snapshot
    reconcile confirm the unit-count delta matches.

    Does **not** call :meth:`RunContext.close_order` — the journal
    issues that as a separate step *after* it has emitted the
    ``confirmed`` audit event, so the event order is
    ``dispatch_submitted → deal_reference_seen → confirmed →
    order_closed``.

    :param store: The active run context.
    :param coid: The dispatch's COID.
    :param kind: :data:`KIND_FULL_CLOSE` or :data:`KIND_PARTIAL_CLOSE`.
    :param extra_payload: Optional extras to preserve.
    """
    if kind not in CLOSE_KINDS:
        raise ValueError(
            f"mark_close_completed: kind must be one of {{KIND_FULL_CLOSE, "
            f"KIND_PARTIAL_CLOSE}}, got {kind!r}"
        )
    fields: dict[str, Any] = {'state': STATE_CONFIRMED}
    if extra_payload:
        existing = store.get_order(coid)
        merged = dict(existing.extras or {}) if existing is not None else {}
        merged.update(extra_payload)
        fields['extras'] = merged
    store.upsert_order(coid, **fields)


# === Cancel lifecycle ======================================================

def create_cancel_command_row(
        store: 'RunContext',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        pine_entry_id: str | None = None,
        from_entry: str | None = None,
        target_coids: list[str] | None = None,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Insert the initial ``submitted`` row for a cancel dispatch.

    A cancel row is the per-dispatch audit trail for a ``CancelIntent``:
    it carries the list of target COIDs the per-target loop will sweep,
    so a mid-loop crash can resume cleanly. The row itself does not
    land at the exchange — only its targets do (via DELETE / PUT-null).

    :param store: The active run context.
    :param coid: The cancel dispatch's COID
        (``envelope.client_order_id(KIND_CANCEL)``).
    :param symbol: Exchange-side symbol of the targets.
    :param side: Side of the targets (uniform across a single dispatch).
    :param qty: Aggregate quantity of the targets — for diagnostics
        only; the recovery contract uses ``target_coids`` directly.
    :param intent_key: The Pine intent's diff key.
    :param pine_entry_id: ``strategy.entry(id=...)`` value the cancel
        addresses, or ``None`` for cross-entry cancels.
    :param from_entry: ``ExitIntent`` ``from_entry`` echo when the
        cancel sweeps a bracket leg, ``None`` otherwise.
    :param target_coids: COIDs the per-target loop intends to sweep.
        Persisted under ``extras['target_coids']`` so recovery can
        re-evaluate after a mid-loop crash. Empty list is permitted —
        the journal will still emit the audit trail and finalise the
        row with :func:`mark_cancel_completed`.
    :param extra_payload: Optional extras to merge in.
    """
    extras: dict[str, Any] = {
        'kind': KIND_CANCEL,
        'target_coids': list(target_coids or ()),
    }
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_SUBMITTED,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        from_entry=from_entry,
        extras=extras,
    )


def mark_cancel_completed(
        store: 'RunContext',
        *,
        coid: str,
        reason_path: str,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Finalise a cancel command row's state to ``confirmed``.

    Called after the per-target loop has finished. ``reason_path``
    records why the cancel resolved this way (``'deleted'`` when the
    targets were swept, ``'already_gone'`` when every target had
    vanished benignly, ``'noop'`` when nothing matched). The value is
    preserved in ``extras['reason_path']`` for forensics. Does **not**
    call :meth:`RunContext.close_order` — see
    :func:`mark_close_completed` for the rationale.

    :param store: The active run context.
    :param coid: The cancel dispatch's COID.
    :param reason_path: One of ``'deleted'``, ``'already_gone'``,
        ``'noop'`` (the canonical :class:`CancelOutcome.reason_path`
        values).
    :param extra_payload: Optional extras to preserve.
    """
    existing = store.get_order(coid)
    merged: dict[str, Any] = dict(existing.extras or {}) if existing is not None else {}
    merged['reason_path'] = reason_path
    if extra_payload:
        merged.update(extra_payload)
    store.upsert_order(
        coid,
        state=STATE_CONFIRMED,
        extras=merged,
    )


# === Modify lifecycle ======================================================

def create_modify_entry_row(
        store: 'RunContext',
        *,
        coid: str,
        target_coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        new_level: float,
        pine_entry_id: str | None = None,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Insert the initial ``submitted`` row for a working-order amend.

    The amend dispatch produces its own COID + audit row separate from
    the working order it mutates. ``target_coid`` and ``new_level``
    travel through ``extras`` so a restart-mid-amend can verify the
    broker landed the change.

    :param store: The active run context.
    :param coid: The amend dispatch's COID
        (``envelope.client_order_id(KIND_MODIFY)`` for the new envelope).
    :param target_coid: COID of the working order being amended.
    :param symbol: Exchange-side symbol.
    :param side: Side of the target order.
    :param qty: Order quantity (unchanged by the amend).
    :param intent_key: The Pine intent's diff key.
    :param new_level: Requested new ``level`` for the working order.
    :param pine_entry_id: ``strategy.entry(id=...)`` echo.
    :param extra_payload: Optional extras to merge in.
    """
    extras: dict[str, Any] = {
        'kind': KIND_MODIFY_ENTRY,
        'target_coid': target_coid,
        'new_level': new_level,
    }
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_SUBMITTED,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        extras=extras,
    )


def create_modify_exit_row(
        store: 'RunContext',
        *,
        coid: str,
        target_coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        new_tp: float | None,
        new_sl: float | None,
        new_trail: float | None,
        new_trail_price: float | None = None,
        pine_entry_id: str | None = None,
        from_entry: str | None = None,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Insert the initial ``submitted`` row for a position bracket amend.

    Mirrors :func:`create_modify_entry_row` but for ``ExitIntent``
    amends: the target is the entry-row representing the position and
    the requested change covers TP / SL / trailing levels. Recovery
    reads ``new_tp`` / ``new_sl`` / ``new_trail`` / ``new_trail_price``
    from ``extras`` and compares them against the post-amend snapshot.

    ``new_trail_price`` is Pine's trailing-stop *activation* price
    (``strategy.exit(trail_price=...)``). When both ``new_trail`` and
    ``new_trail_price`` are set the bracket is **pending trailing** —
    the broker carries no native trailing stop yet (the local
    activation monitor will PUT one once price crosses the threshold),
    so the post-amend snapshot legitimately shows ``trailingStop=False``
    and recovery must NOT require ``trailingStop=True`` to declare the
    amend landed. Persisting the activation price is what lets the
    snapshot verdict distinguish the two trailing shapes.

    Synthetic bracket leg rows (``leg_kind`` ``'tp'`` / ``'sl'``)
    remain under the plugin's leg state machine for M4; this helper
    persists only the entry-side audit trail.
    """
    extras: dict[str, Any] = {
        'kind': KIND_MODIFY_EXIT,
        'target_coid': target_coid,
        'new_tp': new_tp,
        'new_sl': new_sl,
        'new_trail': new_trail,
        'new_trail_price': new_trail_price,
    }
    if extra_payload:
        extras.update(extra_payload)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_SUBMITTED,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        from_entry=from_entry,
        extras=extras,
    )


def mark_modify_completed(
        store: 'RunContext',
        *,
        coid: str,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Finalise a modify dispatch row's state to ``confirmed``.

    Both :data:`KIND_MODIFY_ENTRY` and :data:`KIND_MODIFY_EXIT` use
    this terminal helper. The target order it amended is untouched;
    only the modify command row transitions. Does **not** call
    :meth:`RunContext.close_order` — see :func:`mark_close_completed`
    for the rationale.

    :param store: The active run context.
    :param coid: The modify dispatch's COID.
    :param extra_payload: Optional extras to preserve (e.g. the
        amended values echoed back from the confirm response).
    """
    fields: dict[str, Any] = {'state': STATE_CONFIRMED}
    if extra_payload:
        existing = store.get_order(coid)
        merged = dict(existing.extras or {}) if existing is not None else {}
        merged.update(extra_payload)
        fields['extras'] = merged
    store.upsert_order(coid, **fields)


# === Reconcile-path terminal helpers =======================================

def mark_reconcile_filled(
        store: 'RunContext',
        *,
        coid: str,
        filled_qty: float,
        new_state: str,
        extras_patch: Mapping[str, Any] | None,
) -> None:
    """Promote a row to ``confirmed`` from a reconcile-path observation.

    Used by :meth:`~pynecore.core.broker.journal.DispatchJournal.apply_reconcile_outcome`
    for the working→position case: a row sitting in
    :data:`STATE_SERVER_REF_SEEN` is observed in the live positions
    snapshot, so the journal records the fill and flips
    ``extras['kind']`` from :data:`ENTRY_KIND_WORKING` to
    :data:`ENTRY_KIND_POSITION` in a single transaction.

    :param store: The active run context.
    :param coid: The row's client-order-id.
    :param filled_qty: Confirmed fill quantity from the snapshot.
    :param new_state: The state to land in (always :data:`STATE_CONFIRMED`
        for the current call sites; passed through for clarity).
    :param extras_patch: Plugin-supplied extras to merge into the row
        (e.g. ``{'kind': 'position', 'entry_filled_at': now_ts}``).
        Merged on top of the existing ``extras`` — the journal does not
        validate the keys.
    """
    fields: dict[str, Any] = {
        'state': new_state,
        'filled_qty': filled_qty,
    }
    if extras_patch:
        existing = store.get_order(coid)
        merged = dict(existing.extras or {}) if existing is not None else {}
        merged.update(extras_patch)
        fields['extras'] = merged
    store.upsert_order(coid, **fields)


def mark_reconcile_terminal_close(
        store: 'RunContext',
        *,
        coid: str,
        new_state: str,
        extras_patch: Mapping[str, Any] | None,
        close_row: bool,
) -> None:
    """Terminate a row from a reconcile-path observation.

    Used by :meth:`~pynecore.core.broker.journal.DispatchJournal.apply_reconcile_outcome`
    for every terminal close that originates in the reconciler:
    bracket sibling retire on mixed-bracket rejection, pending-trail
    sibling parent-rejection cascade, missing-pending grace expiry,
    unexpected-cancel cascade, and eager-teardown follow-up after a
    natural close.

    The state mutation, the optional extras merge, and the
    :meth:`RunContext.close_order` (when ``close_row`` is ``True``)
    happen in this single helper; the journal then writes a separate
    audit event. The split mirrors :func:`mark_confirmed_with_fill`
    style — terminal facts persisted first, audit logging by the
    caller.

    :param store: The active run context.
    :param coid: The row's client-order-id.
    :param new_state: Terminal state to land in — typically
        :data:`STATE_REJECTED` for bracket retire / cascade paths.
        ``close_row=True`` is what drops the row from the live set.
    :param extras_patch: Plugin-supplied extras to merge before the
        terminal transition (rarely used on this path; usually
        ``None``).
    :param close_row: When ``True``, also calls
        :meth:`RunContext.close_order` after the state update — this
        is the steady-state default for reconcile terminal closures.
    """
    fields: dict[str, Any] = {'state': new_state}
    if extras_patch:
        existing = store.get_order(coid)
        merged = dict(existing.extras or {}) if existing is not None else {}
        merged.update(extras_patch)
        fields['extras'] = merged
    store.upsert_order(coid, **fields)
    if close_row:
        store.close_order(coid)


# === Engine-trigger partial bracket leg helpers ============================

def create_engine_trigger_partial_leg_row(
        store: 'RunContext',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        pine_entry_id: str,
        from_entry: str,
        leg_kind: str,
        leg_state: str,
        parent_pine_entry_id: str,
        parent_entry_dispatch_ref: str,
        intent_partial_qty: float,
        trigger_level: float | None = None,
        trigger_offset: float | None = None,
        trail_activation_level: float | None = None,
        trail_activation_offset: float | None = None,
        oca_group: str | None = None,
        oca_type: str | None = None,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Persist one engine-trigger partial bracket leg row.

    The row carries no exchange-side order — the engine state machine
    in :class:`~pynecore.core.broker.software_partial_bracket_engine.SoftwarePartialBracketEngine`
    owns it. ``orders.state`` is set to :data:`STATE_PARTIAL_BRACKET_LEG`
    so the journal / reconcile paths short-circuit on these rows; the
    actual engine-trigger phase lives under
    :data:`EXTRAS_KEY_LEG_STATE` and is updated via
    :func:`update_engine_trigger_partial_leg_state`.

    The function deliberately mirrors :func:`create_entry_order_row`'s
    "insert-only" shape — the row is freshly written by the dispatch
    path and never re-inserted by the same call site.

    :param store: The active run context.
    :param coid: The leg's canonical client-order-id. The engine
        constructs this deterministically from ``(parent_dispatch_ref,
        leg_kind)`` so a restart-mid-dispatch never produces two
        rows for the same leg.
    :param symbol: Exchange-side symbol.
    :param side: ``'buy'`` / ``'sell'`` — the CLOSE side, opposite the
        parent position direction.
    :param qty: Partial-close quantity, already quantized to the
        broker's lot step.
    :param intent_key: The owning :class:`~pynecore.core.broker.models.ExitIntent`'s diff key.
    :param pine_entry_id: Pine ``strategy.exit(id=...)`` value, kept
        for audit symmetry with non-leg rows.
    :param from_entry: Pine ``strategy.exit(from_entry=...)`` — the
        parent entry id the leg attaches to.
    :param leg_kind: One of :data:`LEG_KIND_TP_PARTIAL`,
        :data:`LEG_KIND_SL_PARTIAL`, :data:`LEG_KIND_TRAIL_PARTIAL`.
    :param leg_state: Initial state — either :data:`LEG_STATE_ARMED`
        (entry already filled, absolute trigger level known) or
        :data:`LEG_STATE_PENDING_ENTRY` (entry pending, tick offsets
        not yet resolved).
    :param parent_pine_entry_id: Same as ``from_entry`` — kept
        separately for clarity at recovery time.
    :param parent_entry_dispatch_ref: The parent entry row's
        ``client_order_id``. Used by reconciliation to guard against
        a stale leg attaching to a new parent that happens to share
        the same Pine entry id (the unique COID prevents that
        confusion).
    :param intent_partial_qty: Original :class:`ExitIntent.qty` —
        kept separately from ``qty`` (which is the dispatched close
        quantity) so the safety check can compare intent vs. live
        parent qty at trigger time.
    :param trigger_level: Absolute price the engine watches for to
        fire the leg. ``None`` while the leg is ``pending_entry``.
    :param trigger_offset: Trail offset (price units, not ticks) for
        :data:`LEG_KIND_TRAIL_PARTIAL`. ``None`` for tp / sl legs.
    :param trail_activation_level: Absolute activation price for a
        :data:`LEG_KIND_TRAIL_PARTIAL` leg. ``None`` for tp / sl legs
        or while the leg is ``pending_entry``.
    :param trail_activation_offset: Activation offset (price units) for
        a trail leg, resolved to an absolute level once the parent fill
        price is known. ``None`` for tp / sl legs.
    :param oca_group: OCA group name the cascade-cancel keys on.
    :param oca_type: OCA group type (``'cancel'`` for partial
        brackets — TP filling cancels the SL and vice versa).
    :param extra_payload: Additional plugin-specific extras to merge
        into the row's ``extras``. Useful for journal anchors.
    :raises ValueError: When ``leg_kind`` is not one of the three
        canonical values, or ``leg_state`` is not ``armed`` /
        ``pending_entry``.
    """
    if leg_kind not in ENGINE_TRIGGER_LEG_KINDS:
        raise ValueError(
            f"create_engine_trigger_partial_leg_row: leg_kind must be "
            f"one of {sorted(ENGINE_TRIGGER_LEG_KINDS)}, got {leg_kind!r}"
        )
    if leg_state not in (LEG_STATE_ARMED, LEG_STATE_PENDING_ENTRY):
        raise ValueError(
            f"create_engine_trigger_partial_leg_row: leg_state must be "
            f"{LEG_STATE_ARMED!r} or {LEG_STATE_PENDING_ENTRY!r}, "
            f"got {leg_state!r}"
        )
    extras: dict[str, Any] = {
        EXTRAS_KEY_LEG_KIND: leg_kind,
        EXTRAS_KEY_LEG_STATE: leg_state,
        EXTRAS_KEY_TRIGGER_LEVEL: trigger_level,
        EXTRAS_KEY_TRIGGER_OFFSET: trigger_offset,
        EXTRAS_KEY_TRAIL_ACTIVATION_LEVEL: trail_activation_level,
        EXTRAS_KEY_TRAIL_ACTIVATION_OFFSET: trail_activation_offset,
        EXTRAS_KEY_PARENT_PINE_ENTRY_ID: parent_pine_entry_id,
        EXTRAS_KEY_PARENT_ENTRY_DISPATCH_REF: parent_entry_dispatch_ref,
        EXTRAS_KEY_INTENT_PARTIAL_QTY: intent_partial_qty,
        EXTRAS_KEY_OCA_GROUP: oca_group,
        EXTRAS_KEY_OCA_TYPE: oca_type,
    }
    if extra_payload:
        extras.update(extra_payload)
    # Same-bar modify/cancel→recreate path: the prior leg row was
    # closed by ``_dispatch_modify`` (which calls
    # :meth:`RunContext.close_order` on the legs via
    # :func:`update_engine_trigger_partial_leg_state` →
    # ``cancel_legs_for_intent``). The fresh dispatch re-uses the
    # same bar timestamp and retry sequence and therefore reaches
    # this helper with the SAME ``coid``. ``upsert_order`` does not
    # clear ``closed_ts_ms``, so without an explicit reopen the row
    # stays out of ``iter_live_orders`` / ``iter_active_engine_trigger_partial_legs``
    # after a restart and the bracket protection disappears. The
    # call is a no-op when the row does not yet exist or is already
    # live.
    store.reopen_order(coid)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_PARTIAL_BRACKET_LEG,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        from_entry=from_entry,
        extras=extras,
    )


def update_engine_trigger_partial_leg_state(
        store: 'RunContext',
        *,
        coid: str,
        new_leg_state: str,
        trigger_level: float | None = None,
        trigger_offset: float | None = None,
        qty: float | None = None,
        extras_patch: Mapping[str, Any] | None = None,
        close_row: bool | None = None,
) -> None:
    """Transition an engine-trigger partial bracket leg row.

    Three categories of state move call this helper:

    - ``pending_entry`` → ``armed`` after the parent entry fills (the
      caller resolves the tick offsets into absolute price levels and
      passes them via ``trigger_level``).
    - ``armed`` → ``triggering`` → ``triggered`` / ``triggered_failed``
      / ``triggered_unknown`` along the close-dispatch path.
    - ``armed`` → ``cascaded_cancel`` / ``aborted_*`` along the OCA
      cascade, parent-gone abort, and broker-native fail-safe paths.

    The function MUST be called with ``close_row=True`` for any
    transition into :data:`LEG_STATE_TERMINAL` — the helper does NOT
    infer terminality on its own because some intermediate states
    (``triggered_failed``, ``triggered_unknown``) look terminal but
    are retried by the engine in place.

    :param store: The active run context.
    :param coid: The leg row's client-order-id.
    :param new_leg_state: Target value for
        :data:`EXTRAS_KEY_LEG_STATE`. Must be one of the
        :data:`LEG_STATE_*` constants.
    :param trigger_level: When non-``None``, overwrites the leg's
        absolute trigger price (used both for the pending→armed
        promotion and for trail recompute moves).
    :param trigger_offset: When non-``None``, overwrites the leg's
        trail offset (rare; the engine's pricepath is to bump
        ``trigger_level`` instead).
    :param qty: When non-``None``, overwrites the leg row's ``qty``
        column. The WATCH-phase safety check may cap the close
        quantity below the originally-recorded ``intent_partial_qty``
        when the parent has since been partially reduced; persisting
        the capped value here keeps ``restart_replay`` from rebuilding
        the leg with the stale, larger size.
    :param extras_patch: Additional ``extras`` keys to merge before
        the upsert — typically empty, used by audit-rich call sites
        (e.g. failure-reason capture on ``triggered_failed``).
    :param close_row: When ``True``, also calls
        :meth:`RunContext.close_order` after the state update, in the
        same transaction. The engine sets this on every transition into
        :data:`LEG_STATE_TERMINAL`.
    :raises ValueError: When ``new_leg_state`` is not one of the
        canonical :data:`LEG_STATE_*` values.
    """
    if new_leg_state not in (LEG_STATE_LIVE | LEG_STATE_TERMINAL):
        raise ValueError(
            f"update_engine_trigger_partial_leg_state: new_leg_state must be "
            f"one of {sorted(LEG_STATE_LIVE | LEG_STATE_TERMINAL)}, "
            f"got {new_leg_state!r}"
        )
    existing = store.get_order(coid)
    merged: dict[str, Any] = dict(existing.extras or {}) if existing is not None else {}
    merged[EXTRAS_KEY_LEG_STATE] = new_leg_state
    if trigger_level is not None:
        merged[EXTRAS_KEY_TRIGGER_LEVEL] = trigger_level
    if trigger_offset is not None:
        merged[EXTRAS_KEY_TRIGGER_OFFSET] = trigger_offset
    if extras_patch:
        merged.update(extras_patch)
    upsert_fields: dict[str, Any] = {'extras': merged}
    if qty is not None:
        upsert_fields['qty'] = qty
    # One transaction: a terminal state never lands on a row left live.
    with store.transaction():
        store.upsert_order(coid, **upsert_fields)
        if close_row:
            store.close_order(coid)


def iter_active_engine_trigger_partial_legs(
        store: 'RunContext',
        *,
        from_entry: str | None = None,
) -> Iterator['OrderRow']:
    """Iterate live engine-trigger partial bracket leg rows.

    Used by :meth:`SoftwarePartialBracketEngine.restart_replay` to
    rebuild the in-memory leg ledger after a runner restart, and by
    the cascade-cancel paths that need every leg attached to one
    parent. Returns rows whose ``orders.state`` is
    :data:`STATE_PARTIAL_BRACKET_LEG`, are not closed
    (``closed_ts_ms IS NULL``), and whose
    :data:`EXTRAS_KEY_LEG_STATE` is one of :data:`LEG_STATE_LIVE`
    (terminal rows are filtered out even when ``close_order`` has
    not yet flipped ``closed_ts_ms``). :data:`LEG_STATE_LIVE` includes
    both engine-owned active states and the cancel-tentative
    verification state.

    :param store: The active run context.
    :param from_entry: Optional Pine parent entry id filter — same
        passthrough; the cascade-cancel callers use this to target
        all legs of one parent.
    """
    for row in store.iter_live_orders(from_entry=from_entry):
        if row.state != STATE_PARTIAL_BRACKET_LEG:
            continue
        leg_state = (row.extras or {}).get(EXTRAS_KEY_LEG_STATE)
        if leg_state not in LEG_STATE_LIVE:
            continue
        yield row


# === One-way emulation close-leg helpers ===================================

def create_close_leg_row(
        store: 'OwnershipJournal',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        pine_entry_id: str,
        parent_close_coid: str,
        leg_id: str,
        leg_volume: int,
        leg_pre_volume: int,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Persist one one-way emulation close-leg row (PERSIST-FIRST).

    Written by :class:`~pynecore.core.broker.one_way_emulator.OneWayEmulator`
    BEFORE it dispatches the leg's close, so a crash mid-fan-out leaves a durable
    record of which legs were owed. The row carries no exchange order — its
    ``orders.state`` is :data:`STATE_CLOSE_LEG` so the journal / reconcile paths
    short-circuit on it; the leg's close FILL arrives on the normal order-event
    stream and is applied FIFO by ``record_fill`` as a ``LegType.CLOSE``.

    The ``coid`` is the parent CloseIntent's coid suffixed with the broker leg id
    (``f"{parent_close_coid}:{leg_id}"``), so a restart that re-runs the same
    dispatch upserts the SAME row rather than duplicating it.

    :param store: The active run context, or the emulator's in-memory journal
        on a run without a store.
    :param coid: The leg row's deterministic client-order-id.
    :param symbol: Exchange-side symbol.
    :param side: The close side (opposite the position direction being reduced).
    :param qty: The Pine-unit quantity this leg closes (its FIFO slice).
    :param intent_key: The owning :class:`~pynecore.core.broker.models.CloseIntent`'s diff key.
    :param pine_entry_id: The owning ``strategy.close(id=...)`` value, for audit.
    :param parent_close_coid: The CloseIntent dispatch's ``KIND_CLOSE`` coid.
    :param leg_id: Broker leg id (cTrader ``positionId``) this row closes.
    :param leg_volume: Broker-grid integer volume dispatched for the leg.
    :param leg_pre_volume: Broker-grid size of the leg as the close is dispatched.
    :param extra_payload: Additional plugin-specific extras to merge.
    """
    extras: dict[str, Any] = {
        EXTRAS_KEY_CLOSE_LEG_STATE: CLOSE_LEG_STATE_PENDING,
        EXTRAS_KEY_CLOSE_LEG_ID: leg_id,
        EXTRAS_KEY_CLOSE_PARENT_COID: parent_close_coid,
        EXTRAS_KEY_CLOSE_LEG_VOLUME: leg_volume,
        EXTRAS_KEY_CLOSE_LEG_PRE_VOLUME: leg_pre_volume,
        EXTRAS_KEY_CLOSE_LEG_ATTEMPTS: [coid],
        EXTRAS_KEY_CLOSE_LEG_SINCE_MS: int(epoch_time() * 1000),
    }
    if extra_payload:
        extras.update(extra_payload)
    # A restart-mid-fan re-dispatch reaches this helper with the SAME derived
    # coid; reopen so the re-persisted row rejoins the live set (upsert_order
    # does not clear ``closed_ts_ms`` on its own). No-op when the row is absent
    # or already live.
    # One write: a row reopened but not re-persisted would carry the stale
    # state of its earlier lifecycle.
    with store.transaction():
        store.reopen_order(coid)
        store.upsert_order(
            coid,
            symbol=symbol,
            side=side,
            qty=qty,
            state=STATE_CLOSE_LEG,
            intent_key=intent_key,
            pine_entry_id=pine_entry_id,
            extras=extras,
        )


def update_close_leg_state(
        store: 'OwnershipJournal',
        *,
        coid: str,
        new_state: str,
        close_row: bool = False,
        extras_patch: Mapping[str, Any] | None = None,
) -> None:
    """Transition a close-leg row's :data:`EXTRAS_KEY_CLOSE_LEG_STATE`.

    Set ``close_row=True`` for the terminal :data:`CLOSE_LEG_STATE_DISPATCHED`
    transition (the per-leg close acked), which also calls
    :meth:`~OwnershipJournal.close_order` so the row leaves
    :func:`iter_active_close_legs`'s live set.

    :param store: The active run context.
    :param coid: The leg row's client-order-id.
    :param new_state: Target :data:`CLOSE_LEG_STATE_PENDING` /
        :data:`CLOSE_LEG_STATE_DISPATCHED`.
    :param close_row: When ``True``, also close the row (terminal).
    :param extras_patch: Additional extras to merge before the upsert.
    :raises ValueError: When ``new_state`` is not a canonical close-leg state.
    """
    if new_state not in (CLOSE_LEG_STATE_PENDING, CLOSE_LEG_STATE_DISPATCHED):
        raise ValueError(
            f"update_close_leg_state: new_state must be one of "
            f"{{CLOSE_LEG_STATE_PENDING, CLOSE_LEG_STATE_DISPATCHED}}, "
            f"got {new_state!r}"
        )
    # One write: a terminal state left on a live row is read by no later pass.
    with store.transaction():
        existing = store.get_order(coid)
        merged: dict[str, Any] = dict(existing.extras or {}) if existing is not None else {}
        merged[EXTRAS_KEY_CLOSE_LEG_STATE] = new_state
        if extras_patch:
            merged.update(extras_patch)
        store.upsert_order(coid, extras=merged)
        if close_row:
            store.close_order(coid)


def iter_active_close_legs(
        store: 'OwnershipJournal',
) -> Iterator['OrderRow']:
    """Iterate live (pending) close-leg rows for restart replay.

    Returns rows whose ``orders.state`` is :data:`STATE_CLOSE_LEG`, are not
    closed, and whose :data:`EXTRAS_KEY_CLOSE_LEG_STATE` is in
    :data:`CLOSE_LEG_STATE_LIVE` — a fan-out leg whose close was persisted but
    never acked. :meth:`OneWayEmulator.restart_replay` reconciles each against
    the live broker legs before re-dispatching.

    :param store: The active run context.
    """
    for row in store.iter_live_orders():
        if row.state != STATE_CLOSE_LEG:
            continue
        leg_state = (row.extras or {}).get(EXTRAS_KEY_CLOSE_LEG_STATE)
        if leg_state not in CLOSE_LEG_STATE_LIVE:
            continue
        yield row


def reserve_close_leg_send(
        store: 'OwnershipJournal',
        coid: str,
        *,
        attempt_coid: str,
        volume: int,
        pre_volume: int,
        since_ms: int,
        sent_ns: int,
) -> None:
    """Record one send of a leg close on its row before the send goes out.

    :param store: The active run context, or the emulator's in-memory journal.
    :param coid: The close-leg row's client-order-id.
    :param attempt_coid: The client-order-id the send goes out under.
    :param volume: The broker-grid volume of the send.
    :param pre_volume: The leg's broker-grid size the send is credited from.
    :param since_ms: Epoch ms the venue is asked about the send from.
    :param sent_ns: Epoch ns of the send, ordering it among every send.
    """
    existing = store.get_order(coid)
    if existing is None:
        return
    extras: dict[str, Any] = dict(existing.extras or {})
    sends = [list(sent) for sent in extras.get(EXTRAS_KEY_CLOSE_LEG_RESERVED) or ()
             if str(sent[0]) != attempt_coid]
    sends.append([attempt_coid, int(volume), int(pre_volume), int(since_ms), int(sent_ns)])
    extras[EXTRAS_KEY_CLOSE_LEG_RESERVED] = sends
    store.upsert_order(coid, extras=extras)


def release_close_leg_send(store: 'OwnershipJournal', coid: str, attempt_coid: str) -> None:
    """Drop one reserved send of a leg close from its row.

    A :data:`CLOSE_LEG_STATE_DISPATCHED` row with no reserved send left is
    closed: its close is fully accounted for.

    :param store: The active run context, or the emulator's in-memory journal.
    :param coid: The close-leg row's client-order-id.
    :param attempt_coid: The client-order-id of the send to drop.
    """
    # One span: a dispatched row left live without a reserved send is read
    # by no later pass, so nothing would ever close it.
    with store.transaction():
        existing = store.get_order(coid)
        if existing is None:
            return
        extras: dict[str, Any] = dict(existing.extras or {})
        sends = [list(sent) for sent in extras.get(EXTRAS_KEY_CLOSE_LEG_RESERVED) or ()
                 if str(sent[0]) != attempt_coid]
        extras[EXTRAS_KEY_CLOSE_LEG_RESERVED] = sends
        store.upsert_order(coid, extras=extras)
        if (not sends and existing.closed_ts_ms is None
                and extras.get(EXTRAS_KEY_CLOSE_LEG_STATE) == CLOSE_LEG_STATE_DISPATCHED):
            store.close_order(coid)


def lower_close_leg_send_pre_volume(
        store: 'OwnershipJournal', coid: str, attempt_coid: str, pre_volume: int,
) -> None:
    """Lower the leg size a reserved send is credited from.

    :param store: The active run context, or the emulator's in-memory journal.
    :param coid: The close-leg row's client-order-id.
    :param attempt_coid: The client-order-id of the send.
    :param pre_volume: The new broker-grid leg size.
    """
    existing = store.get_order(coid)
    if existing is None:
        return
    extras: dict[str, Any] = dict(existing.extras or {})
    sends = [list(sent) for sent in extras.get(EXTRAS_KEY_CLOSE_LEG_RESERVED) or ()]
    for sent in sends:
        if str(sent[0]) == attempt_coid:
            sent[2] = int(pre_volume)
    extras[EXTRAS_KEY_CLOSE_LEG_RESERVED] = sends
    store.upsert_order(coid, extras=extras)


def finish_close_leg_row(
        store: 'OwnershipJournal', coid: str, *, release_sends: bool = False,
) -> None:
    """Mark a close-leg row :data:`CLOSE_LEG_STATE_DISPATCHED` and close it once accounted for.

    The row is closed when none of its sends is reserved any more; otherwise
    it stays live until the last one is dropped (:func:`release_close_leg_send`).

    :param store: The active run context, or the emulator's in-memory journal.
    :param coid: The close-leg row's client-order-id.
    :param release_sends: ``True`` when the leg is gone, so no send of it can
        still take anything from it: every reserved send is dropped and the
        row closed.
    """
    # One span: a dispatched row left live without a reserved send is read
    # by no later pass, so nothing would ever close it.
    with store.transaction():
        existing = store.get_order(coid)
        if existing is None:
            return
        patch: dict[str, Any] = {}
        if release_sends:
            patch[EXTRAS_KEY_CLOSE_LEG_RESERVED] = []
        update_close_leg_state(store, coid=coid, new_state=CLOSE_LEG_STATE_DISPATCHED,
                               extras_patch=patch)
        current = store.get_order(coid)
        sends = (current.extras or {}).get(EXTRAS_KEY_CLOSE_LEG_RESERVED) if current else None
        if not sends:
            store.close_order(coid)


def iter_close_leg_sends(
        store: 'OwnershipJournal', symbol: str,
) -> Iterator[tuple['OrderRow', list[Any]]]:
    """Iterate the reserved sends of the live close-leg rows of ``symbol``.

    :param store: The active run context, or the emulator's in-memory journal.
    :param symbol: The traded symbol.
    :return: ``(row, [attempt_coid, volume, pre_volume, since_ms, sent_ns])``
        pairs, in no particular order.
    """
    for row in store.iter_live_orders():
        if row.state != STATE_CLOSE_LEG or row.symbol != symbol:
            continue
        for sent in (row.extras or {}).get(EXTRAS_KEY_CLOSE_LEG_RESERVED) or ():
            yield row, list(sent)


# === One-way emulation residual-open helpers ===============================

def create_residual_open_row(
        store: 'OwnershipJournal',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        pine_entry_id: str,
        entry_coid: str,
        run_tag: str,
        bar_ts_ms: int,
        retry_seq: int,
        close_plan: Sequence[tuple[str, float]] | None = None,
        netting: str | None = None,
) -> None:
    """Persist a reversal's residual-OPEN breadcrumb (PERSIST-FIRST).

    Written by :meth:`~pynecore.core.broker.one_way_emulator.OneWayEmulator.run_reversal`
    BEFORE the FIFO closes, so a crash in the window between the closes landing
    and the residual :meth:`PositionPort.place_leg` persisting its own entry row
    leaves a durable record that an open is owed. The row owns no exchange order —
    its ``orders.state`` is :data:`STATE_RESIDUAL_OPEN`, so the journal /
    reconcile / startup-recovery paths short-circuit on it.
    :meth:`OneWayEmulator.restart_replay` reconciles it against the residual entry
    row (keyed on ``entry_coid``) and re-dispatches the open only when that row
    never landed, then clears this breadcrumb.

    The ``coid`` is the reversal's ``KIND_CLOSE`` parent coid suffixed
    ``":residual"``, so a restart re-run upserts the SAME row.

    :param store: The active run context, or the emulator's in-memory journal
        on a run without a store.
    :param coid: The breadcrumb row's deterministic client-order-id.
    :param symbol: Exchange-side symbol.
    :param side: The residual open's side (the entry's own direction).
    :param qty: The combined order size of a reversal (Pine units), lowered by
        each close it counts (:func:`count_residual_open_close`); ``0`` for a
        cut-short reduction or flatten.
    :param intent_key: The owning :class:`~pynecore.core.broker.models.EntryIntent`'s diff key.
    :param pine_entry_id: The owning ``strategy.entry(id=...)`` value.
    :param entry_coid: The residual ``place_leg``'s deterministic entry coid
        (``KIND_ENTRY``, or ``KIND_ENTRY_TRIGGERED`` for a triggered reversal) —
        its presence on replay proves the open already persisted.
    :param run_tag: Dispatch envelope run tag (to rebuild the envelope on replay).
    :param bar_ts_ms: Dispatch envelope bar timestamp.
    :param retry_seq: Dispatch envelope retry sequence.
    :param close_plan: ``(leg_id, pine_qty)`` of every close of a cut-short
        reduction or flatten (:data:`EXTRAS_KEY_RESIDUAL_OPEN_CLOSE_PLAN`);
        ``None`` for a reversal, which closes every opposing leg.
    :param netting: The coid of the fill-time netting record the breadcrumb is
        a half of (:data:`EXTRAS_KEY_RESIDUAL_OPEN_NETTING`), or ``None``.
    """
    extras: dict[str, Any] = {
        EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID: entry_coid,
        EXTRAS_KEY_RESIDUAL_OPEN_RUN_TAG: run_tag,
        EXTRAS_KEY_RESIDUAL_OPEN_BAR_TS_MS: bar_ts_ms,
        EXTRAS_KEY_RESIDUAL_OPEN_RETRY_SEQ: retry_seq,
    }
    if close_plan is not None:
        extras[EXTRAS_KEY_RESIDUAL_OPEN_CLOSE_PLAN] = [
            [leg_id, qty] for leg_id, qty in close_plan]
    if netting is not None:
        extras[EXTRAS_KEY_RESIDUAL_OPEN_NETTING] = netting
    # One write: a breadcrumb reopened but not re-persisted would replay the
    # close plan and the open of its earlier lifecycle.
    with store.transaction():
        store.reopen_order(coid)
        store.upsert_order(
            coid,
            symbol=symbol,
            side=side,
            qty=qty,
            state=STATE_RESIDUAL_OPEN,
            intent_key=intent_key,
            pine_entry_id=pine_entry_id,
            extras=extras,
        )


def advance_residual_open_attempt(
        store: 'OwnershipJournal', coid: str, *, entry_coid: str, retry_seq: int,
) -> None:
    """Move a live breadcrumb's residual open on to its next attempt.

    Called after a ``place_leg`` that provably never left the client: the
    entry row that ``place_leg`` may have left under the old coid is no
    evidence of the next attempt, which goes out under ``entry_coid``. A no-op
    for an absent row.

    :param store: The active run context, or the emulator's in-memory journal.
    :param coid: The breadcrumb's client-order-id.
    :param entry_coid: The next attempt's deterministic entry coid.
    :param retry_seq: The next attempt's ``retry_seq``.
    """
    existing = store.get_order(coid)
    if existing is None:
        return
    extras: dict[str, Any] = dict(existing.extras or {})
    extras[EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID] = entry_coid
    extras[EXTRAS_KEY_RESIDUAL_OPEN_RETRY_SEQ] = retry_seq
    store.upsert_order(coid, extras=extras)


def count_residual_open_close(
        store: 'OwnershipJournal', coid: str, *, close_coid: str, qty: float,
) -> None:
    """Count one close-leg row of a reversal against its combined order.

    Lowers the breadcrumb's ``qty`` (what is left to open) by ``qty`` the first
    time ``close_coid`` is counted (:data:`EXTRAS_KEY_RESIDUAL_OPEN_COUNTED_CLOSES`);
    a row already counted is a re-send of the same close and lowers nothing. A
    no-op for an absent or closed breadcrumb.

    :param store: The active run context, or the emulator's in-memory journal.
    :param coid: The breadcrumb's client-order-id.
    :param close_coid: The close-leg row's client-order-id.
    :param qty: The Pine quantity the close takes off the opposing exposure.
    """
    with store.transaction():
        existing = store.get_order(coid)
        if existing is None or existing.closed_ts_ms is not None:
            return
        extras: dict[str, Any] = dict(existing.extras or {})
        counted = [str(c) for c in extras.get(EXTRAS_KEY_RESIDUAL_OPEN_COUNTED_CLOSES) or ()]
        if close_coid in counted:
            return
        extras[EXTRAS_KEY_RESIDUAL_OPEN_COUNTED_CLOSES] = [*counted, close_coid]
        remaining = round(max(0.0, existing.qty - max(0.0, float(qty))), 12)
        store.upsert_order(coid, qty=remaining, extras=extras)


def clear_residual_open_row(store: 'OwnershipJournal', coid: str) -> None:
    """Finalise a residual-open breadcrumb once the open is durable / replayed."""
    store.close_order(coid)


def create_netting_row(
        store: 'OwnershipJournal',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str | None,
        entry_id: str | None,
        fill_id: str,
        contended: Iterable[str] = (),
) -> None:
    """Persist a fill-time netting record (:data:`STATE_NETTING`, PERSIST-FIRST).

    :param store: The active run context, or the emulator's in-memory journal.
    :param coid: The record's deterministic client-order-id (the filled entry's
        coid and the fill's sequence).
    :param symbol: Exchange-side symbol.
    :param side: The fill's side (the new leg's side).
    :param qty: The Pine quantity the fill closed of the opposite position:
        what the venue must net.
    :param intent_key: The filled entry's diff key, when known.
    :param entry_id: The filled entry's Pine id.
    :param fill_id: The fill slice's id.
    :param contended: The close fill sides on which another close of the run is
        still in flight as the record is written
        (:data:`EXTRAS_KEY_NETTING_CONTENDED`).
    """
    extras: dict[str, Any] = {
        EXTRAS_KEY_NETTING_FILL_ID: fill_id,
        EXTRAS_KEY_NETTING_ENTRY_ID: entry_id,
        EXTRAS_KEY_NETTING_SINCE_MS: int(epoch_time() * 1000),
        EXTRAS_KEY_NETTING_PASSES: 0,
        EXTRAS_KEY_NETTING_EXPECTED: {'buy': 0.0, 'sell': 0.0},
    }
    if contended:
        extras[EXTRAS_KEY_NETTING_CONTENDED] = sorted(set(contended))
    with store.transaction():
        store.reopen_order(coid)
        store.upsert_order(
            coid,
            symbol=symbol,
            side=side,
            qty=qty,
            state=STATE_NETTING,
            intent_key=intent_key,
            pine_entry_id=entry_id,
            extras=extras,
        )


def iter_active_nettings(store: 'OwnershipJournal') -> Iterator['OrderRow']:
    """Iterate the live fill-time netting records (:data:`STATE_NETTING`), oldest first.

    :param store: The active run context, or the emulator's in-memory journal.
    """
    rows = [row for row in store.iter_live_orders() if row.state == STATE_NETTING]
    rows.sort(key=lambda row: (row.created_ts_ms, row.client_order_id))
    yield from rows


def netting_absorbed_row_coid(coid: str) -> str:
    """The client-order-id of the engine's absorbed-fill count of netting record ``coid``."""
    return f"{coid}:absorbed"


def netting_expected(store: 'OwnershipJournal', row: 'OrderRow') -> dict[str, float]:
    """What of a netting record's expected close fills has not been booked as netting yet.

    :param store: The journal the record lives in.
    :param row: The netting record.
    :return: ``{'buy': qty, 'sell': qty}``, never negative.
    """
    expected = (row.extras or {}).get(EXTRAS_KEY_NETTING_EXPECTED) or {}
    # The count of a record an earlier process closed stays with that process.
    absorbed_row = store.get_order_for_run_id(netting_absorbed_row_coid(row.client_order_id))
    absorbed = ((absorbed_row.extras or {}).get(EXTRAS_KEY_NETTING_EXPECTED) or {}
                if absorbed_row is not None else {})
    return {side: max(0.0, round(float(expected.get(side) or 0.0)
                                 - float(absorbed.get(side) or 0.0), 12))
            for side in ('buy', 'sell')}


def absorb_netting_fills(
        store: 'OwnershipJournal', absorbed: Mapping[str, Mapping[str, float]],
) -> None:
    """Count close fills the engine booked as netting toward their records.

    Engine-only, in the transaction of the fills' receipts. A record an
    earlier process of the run closed (its late fill identified by its close
    id) is counted too: the count continues from the one that process left,
    in a row of this process, closed like the record.

    :param store: The journal the records live in.
    :param absorbed: Per netting record coid, the Pine quantity booked as
        netting by fill side.
    """
    with store.transaction():
        for coid, by_side in absorbed.items():
            key = netting_absorbed_row_coid(coid)
            existing = store.get_order_for_run_id(key)
            counted = dict((existing.extras or {}).get(EXTRAS_KEY_NETTING_EXPECTED) or {}
                           if existing is not None else {})
            for side, qty in by_side.items():
                counted[side] = round(float(counted.get(side) or 0.0) + float(qty), 12)
            if store.get_order(key) is None:
                record = store.get_order_for_run_id(coid)
                store.upsert_order(
                    key, symbol=record.symbol if record is not None else '', side='buy',
                    qty=0.0, state=STATE_NETTING_ABSORBED,
                    extras={EXTRAS_KEY_NETTING_EXPECTED: counted},
                )
                if record is not None and record.closed_ts_ms is not None:
                    store.close_order(key)
            else:
                store.upsert_order(key, extras={EXTRAS_KEY_NETTING_EXPECTED: counted})


def iter_active_residual_opens(
        store: 'OwnershipJournal',
) -> Iterator['OrderRow']:
    """Iterate live residual-open breadcrumbs for restart replay.

    Returns rows whose ``orders.state`` is :data:`STATE_RESIDUAL_OPEN` and which
    are not yet closed — a reversal whose residual open was recorded but whose
    durability (its own entry row) the crash may have pre-empted.

    :param store: The active run context, or the emulator's in-memory journal
        on a run without a store.
    """
    for row in store.iter_live_orders():
        if row.state == STATE_RESIDUAL_OPEN:
            yield row


# === One-way emulation bracket-ownership helpers ===========================

def create_bracket_ownership_row(
        store: 'OwnershipJournal',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        pine_entry_id: str,
        from_entry: str,
        leg_id: str,
        attach_coid: str,
        tp_price: float | None,
        sl_price: float | None,
        trail_price: float | None,
        trail_offset: float | None,
        oca_name: str | None = None,
        oca_type: str | None = None,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Persist one per-leg bracket-ownership row (PERSIST-FIRST).

    Written by :class:`~pynecore.core.broker.one_way_emulator.OneWayEmulator`
    BEFORE it amends an exit's TP/SL/trailing bracket onto one hedging leg, so a
    crash mid-replication leaves a durable record of exactly which legs this exit
    protects. The row carries no exchange order — its ``orders.state`` is
    :data:`STATE_BRACKET_OWN` so the journal / reconcile paths short-circuit; the
    leg's protective FILL (if it fires) arrives on the normal order-event stream.

    The ``coid`` is the exit's bracket-attach coid suffixed with the broker leg
    id (``f"{attach_coid}:{leg_id}"``), so a re-attach upserts the SAME row and
    two different exits (distinct attach coids) own disjoint coid namespaces. The
    ``from_entry`` is written to its own column so the clear path can pre-filter
    the live set cheaply via :meth:`~OwnershipJournal.iter_live_orders`.

    :param store: The ownership journal: the run's store, or the emulator's in-memory one.
    :param coid: The ownership row's deterministic client-order-id.
    :param symbol: Exchange-side symbol.
    :param side: The exit (bracket) side.
    :param qty: The owning :class:`~pynecore.core.broker.models.ExitIntent`'s quantity, for audit.
    :param intent_key: The owning :class:`~pynecore.core.broker.models.ExitIntent`'s diff key
        (``f"{pine_id}\\0{from_entry}"``), matched by the clear path.
    :param pine_entry_id: The owning ``strategy.exit(id=...)`` value.
    :param from_entry: The exit's parent entry id (its own column for filtering).
    :param leg_id: Broker leg id (cTrader ``positionId``) the bracket is on.
    :param attach_coid: The exit's ``KIND_EXIT_SL`` bracket-attach coid.
    :param tp_price: Pine-unit take-profit level the exit asks for (or ``None``).
    :param sl_price: Pine-unit stop-loss level the exit asks for (or ``None``).
    :param trail_price: Pine-unit trailing-stop activation price (or ``None``).
    :param trail_offset: Pine-unit trailing offset the exit asks for (or ``None``).
    :param oca_name: The owning exit's OCA group name (or ``None``).
    :param oca_type: The owning exit's OCA type string (``"reduce"`` /
        ``"cancel"`` / ``"none"``, or ``None``).
    :param extra_payload: Additional plugin-specific extras to merge.
    """
    extras: dict[str, Any] = {
        EXTRAS_KEY_BRACKET_OWN_STATE: BRACKET_OWN_STATE_ACTIVE,
        EXTRAS_KEY_BRACKET_OWN_LEG_ID: leg_id,
        EXTRAS_KEY_BRACKET_OWN_ATTACH_COID: attach_coid,
        EXTRAS_KEY_BRACKET_OWN_TP: tp_price,
        EXTRAS_KEY_BRACKET_OWN_SL: sl_price,
        EXTRAS_KEY_BRACKET_OWN_TRAIL_PRICE: trail_price,
        EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET: trail_offset,
        EXTRAS_KEY_BRACKET_OWN_OCA_NAME: oca_name,
        EXTRAS_KEY_BRACKET_OWN_OCA_TYPE: oca_type,
    }
    if extra_payload:
        extras.update(extra_payload)
    # A re-attach (modify, or restart re-assert) reaches this helper with the
    # SAME derived coid; reopen so the re-persisted row rejoins the live set
    # (upsert_order does not clear ``closed_ts_ms`` on its own). No-op when the
    # row is absent or already live.
    # One write: a row reopened but not re-persisted would carry the stale
    # state of its earlier lifecycle.
    with store.transaction():
        store.reopen_order(coid)
        store.upsert_order(
            coid,
            symbol=symbol,
            side=side,
            qty=qty,
            state=STATE_BRACKET_OWN,
            intent_key=intent_key,
            pine_entry_id=pine_entry_id,
            from_entry=from_entry,
            extras=extras,
        )


def update_bracket_ownership_state(
        store: 'OwnershipJournal',
        *,
        coid: str,
        new_state: str,
        close_row: bool = False,
        extras_patch: Mapping[str, Any] | None = None,
) -> None:
    """Transition a bracket-ownership row's :data:`EXTRAS_KEY_BRACKET_OWN_STATE`.

    Set ``close_row=True`` for the terminal :data:`BRACKET_OWN_STATE_RELEASED`
    transition (the bracket cleared from this leg), which also calls
    :meth:`~OwnershipJournal.close_order` so the row leaves
    :func:`iter_active_bracket_ownerships`'s live set.

    :param store: The ownership journal: the run's store, or the emulator's in-memory one.
    :param coid: The ownership row's client-order-id.
    :param new_state: Target :data:`BRACKET_OWN_STATE_ACTIVE` /
        :data:`BRACKET_OWN_STATE_CLEARING` / :data:`BRACKET_OWN_STATE_RELEASED`.
    :param close_row: When ``True``, also close the row (terminal).
    :param extras_patch: Additional extras to merge before the upsert.
    :raises ValueError: When ``new_state`` is not a canonical ownership state.
    """
    if new_state not in (
            BRACKET_OWN_STATE_ACTIVE,
            BRACKET_OWN_STATE_CLEARING,
            BRACKET_OWN_STATE_RELEASED,
    ):
        raise ValueError(
            f"update_bracket_ownership_state: new_state must be one of "
            f"{{BRACKET_OWN_STATE_ACTIVE, BRACKET_OWN_STATE_CLEARING, "
            f"BRACKET_OWN_STATE_RELEASED}}, got {new_state!r}"
        )
    # One write: a terminal state left on a live row is read by no later pass.
    with store.transaction():
        existing = store.get_order(coid)
        merged: dict[str, Any] = dict(existing.extras or {}) if existing is not None else {}
        merged[EXTRAS_KEY_BRACKET_OWN_STATE] = new_state
        if extras_patch:
            merged.update(extras_patch)
        store.upsert_order(coid, extras=merged)
        if close_row:
            store.close_order(coid)


def iter_active_bracket_ownerships(
        store: 'OwnershipJournal',
        *,
        from_entry: str | None = None,
) -> Iterator['OrderRow']:
    """Iterate live (active) bracket-ownership rows.

    Returns rows whose ``orders.state`` is :data:`STATE_BRACKET_OWN`, are not
    closed, and whose :data:`EXTRAS_KEY_BRACKET_OWN_STATE` is in
    :data:`BRACKET_OWN_STATE_LIVE`. The clear path filters further by
    ``intent_key`` to touch only the legs a specific exit owns; the restart pass
    re-asserts every active row against the live broker legs.

    :param store: The ownership journal: the run's store, or the emulator's in-memory one.
    :param from_entry: Optional parent-entry filter, passed to
        :meth:`~OwnershipJournal.iter_live_orders` (the indexed ``from_entry`` column).
    """
    for row in store.iter_live_orders(from_entry=from_entry):
        if row.state != STATE_BRACKET_OWN:
            continue
        own_state = (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_STATE)
        if own_state not in BRACKET_OWN_STATE_LIVE:
            continue
        yield row


# === Entry-stop watch helpers ==============================================

def create_stop_limit_watch_row(
        store: 'RunContext',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        pine_entry_id: str,
        stop: float,
        limit: float,
        armed_bar_ts_ms: int,
) -> None:
    """Persist one software stop-limit watch row in ``armed`` (PERSIST-FIRST).

    The row carries no exchange-side order — the
    :class:`~pynecore.core.broker.software_stop_limit_engine.SoftwareStopLimitEngine`
    state machine owns it. ``orders.state`` is :data:`STATE_STOP_LIMIT_WATCH`,
    so the journal / reconcile paths skip it; the watch phase lives under
    :data:`EXTRAS_KEY_STOP_LIMIT_STATE` and is advanced via
    :func:`update_stop_limit_watch`.

    :param store: The active run context.
    :param coid: The watch row's deterministic client-order-id — built from
        the entry's envelope with
        :data:`~pynecore.core.broker.idempotency.KIND_STOP_LIMIT_WATCH`, so a
        restart mid-dispatch never produces two rows for one watch.
    :param symbol: Exchange-side symbol.
    :param side: The order side (``'buy'`` / ``'sell'``).
    :param qty: Order quantity, already quantized to the broker's lot step.
    :param intent_key: The owning :class:`~pynecore.core.broker.models.EntryIntent`'s
        diff key.
    :param pine_entry_id: The Pine order id.
    :param stop: The stop price whose cross triggers the order.
    :param limit: The limit price the triggered LIMIT rests at.
    :param armed_bar_ts_ms: The bar the watch is armed on; also its first
        judged bar.
    """
    store.reopen_order(coid)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_STOP_LIMIT_WATCH,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        extras={
            EXTRAS_KEY_STOP_LIMIT_STATE: STOP_LIMIT_STATE_ARMED,
            EXTRAS_KEY_STOP_LIMIT_STOP: stop,
            EXTRAS_KEY_STOP_LIMIT_LIMIT: limit,
            EXTRAS_KEY_STOP_LIMIT_ORDER_COID: None,
            EXTRAS_KEY_STOP_LIMIT_ARMED_BAR: armed_bar_ts_ms,
            EXTRAS_KEY_STOP_LIMIT_CHECKED_BAR: armed_bar_ts_ms,
        },
    )


def update_stop_limit_watch(
        store: 'RunContext',
        *,
        coid: str,
        state: str | None = None,
        extras_patch: Mapping[str, Any] | None = None,
        qty: float | None = None,
        side: str | None = None,
        close_row: bool = False,
) -> None:
    """Update a stop-limit watch row (a legacy OCO row included).

    :param store: The active run context.
    :param coid: The watch row's client-order-id.
    :param state: When non-``None``, the new :data:`EXTRAS_KEY_STOP_LIMIT_STATE`
        value.
    :param extras_patch: Extras keys to merge into the row.
    :param qty: When non-``None``, overwrites the row's ``qty`` column.
    :param side: When non-``None``, overwrites the row's ``side`` column.
    :param close_row: Also close the row (:meth:`RunContext.close_order`) — set
        on every transition into :data:`STOP_LIMIT_STATE_TERMINAL`.
    :raises ValueError: When ``state`` is not a canonical value.
    """
    if state is not None and state not in (STOP_LIMIT_STATE_LIVE | STOP_LIMIT_STATE_TERMINAL):
        raise ValueError(
            f"update_stop_limit_watch: state must be one of "
            f"{sorted(STOP_LIMIT_STATE_LIVE | STOP_LIMIT_STATE_TERMINAL)}, got {state!r}"
        )
    existing = store.get_order(coid)
    merged: dict[str, Any] = dict(existing.extras or {}) if existing is not None else {}
    if state is not None:
        merged[EXTRAS_KEY_STOP_LIMIT_STATE] = state
    if extras_patch:
        merged.update(extras_patch)
    upsert_fields: dict[str, Any] = {'extras': merged}
    if qty is not None:
        upsert_fields['qty'] = qty
    if side is not None:
        upsert_fields['side'] = side
    store.upsert_order(coid, **upsert_fields)
    if close_row:
        store.close_order(coid)


def iter_live_stop_limit_watches(store: 'RunContext') -> Iterator['OrderRow']:
    """Iterate the live stop-limit watch rows, legacy OCO rows included.

    Used by :meth:`SoftwareStopLimitEngine.restart_replay` to rebuild the
    in-memory ledger after a restart. Yields the open rows whose
    ``orders.state`` is :data:`STATE_STOP_LIMIT_WATCH` and whose phase is live:
    :data:`STOP_LIMIT_STATE_LIVE` or :data:`LEGACY_OCO_STATE_LIVE` (terminal
    rows are skipped even when ``close_order`` has not flipped
    ``closed_ts_ms`` yet).

    :param store: The active run context.
    """
    for row in store.iter_live_orders():
        if row.state != STATE_STOP_LIMIT_WATCH:
            continue
        phase = (row.extras or {}).get(EXTRAS_KEY_STOP_LIMIT_STATE)
        if phase in STOP_LIMIT_STATE_LIVE or phase in LEGACY_OCO_STATE_LIVE:
            yield row
