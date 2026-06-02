"""
Typed helpers over :class:`~pynecore.core.broker.storage.RunContext`.

The :class:`BrokerStore` exposes :meth:`upsert_order` with a free-form
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
from collections.abc import Iterator, Mapping
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
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
    'create_entry_order_row',
    'record_server_ref',
    'mark_confirmed_with_fill',
    'mark_disposition_unknown',
    'mark_rejected',
    'find_pending_dispatch',
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
    'create_close_leg_row',
    'update_close_leg_state',
    'iter_active_close_legs',
    'STATE_BRACKET_OWN',
    'BRACKET_OWN_STATE_ACTIVE',
    'BRACKET_OWN_STATE_RELEASED',
    'BRACKET_OWN_STATE_LIVE',
    'EXTRAS_KEY_BRACKET_OWN_STATE',
    'EXTRAS_KEY_BRACKET_OWN_LEG_ID',
    'EXTRAS_KEY_BRACKET_OWN_ATTACH_COID',
    'EXTRAS_KEY_BRACKET_OWN_TP',
    'EXTRAS_KEY_BRACKET_OWN_SL',
    'EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET',
    'create_bracket_ownership_row',
    'update_bracket_ownership_state',
    'iter_active_bracket_ownerships',
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
# ``CANCEL_CONFIRMED``) or back to ``pending_entry`` / ``armed`` (on
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


# === Entry-stop watch constants ============================================

# ``orders.state`` marker for a synthetic entry-stop WATCH row. A both-set
# Pine entry (``strategy.entry(limit=, stop=)``) is two OCO legs: the LIMIT
# leg rests natively as a working order, while the STOP leg is a software
# price-watch that fires a MARKET order on the stop side. This row carries
# NO exchange-side order (the native LIMIT and the eventual MARKET have their
# own rows) — the :class:`~pynecore.core.broker.software_entry_stop_engine.SoftwareEntryStopEngine`
# state machine owns it. Mirrors :data:`STATE_PARTIAL_BRACKET_LEG`: the
# journal / reconcile paths short-circuit on these rows and the actual watch
# phase lives under :data:`EXTRAS_KEY_ENTRY_STOP_STATE`.
STATE_ENTRY_STOP_WATCH = 'entry_stop_watch'

# ``extras[EXTRAS_KEY_ENTRY_STOP_STATE]`` values — the entry-stop watch state
# machine's current phase. ``armed`` watches the price against the stop level
# while the native LIMIT rests. ``cancel_pending`` is latched once the stop is
# crossed: the engine cancels the native LIMIT and gates the market on the
# cancel disposition. ``stop_market_pending`` is reached only after the LIMIT
# cancel is CONFIRMED — the deterministic MARKET client-order-id is persisted
# before the POST so a restart verifies-before-resends and never double-opens.
# The three terminal states record which leg of the OCO won.
ENTRY_STOP_STATE_ARMED = 'armed'
ENTRY_STOP_STATE_CANCEL_PENDING = 'cancel_pending'
ENTRY_STOP_STATE_MARKET_PENDING = 'stop_market_pending'
ENTRY_STOP_STATE_LIMIT_WON = 'limit_won'
ENTRY_STOP_STATE_STOP_WON = 'stop_won'
ENTRY_STOP_STATE_ABORTED = 'aborted'

# Terminal states — the row is closed and the engine no longer owns it.
ENTRY_STOP_STATE_TERMINAL: frozenset[str] = frozenset({
    ENTRY_STOP_STATE_LIMIT_WON,
    ENTRY_STOP_STATE_STOP_WON,
    ENTRY_STOP_STATE_ABORTED,
})

# Live states — the row is alive and the engine state machine drives the next
# transition. ``cancel_pending`` and ``stop_market_pending`` are latched
# intermediates: on restart the engine re-drives them deterministically
# (re-issue the idempotent cancel / re-dispatch the idempotent market) rather
# than re-evaluating the price.
ENTRY_STOP_STATE_LIVE: frozenset[str] = frozenset({
    ENTRY_STOP_STATE_ARMED,
    ENTRY_STOP_STATE_CANCEL_PENDING,
    ENTRY_STOP_STATE_MARKET_PENDING,
})

# Abortable states — an external cancel / reject of the native LIMIT leg may
# retire the watch only here. Once the watch has committed to the stop side
# (``stop_market_pending`` — the deterministic KIND_ENTRY_STOP market id is
# already persisted and the MARKET is in flight), a delayed broker
# cancelled/rejected ack for the now-cancelled LIMIT must NOT abort it; that
# echo would corrupt the persist-first ledger and drop the verify-before-resend
# watch on restart. ``stop_market_pending`` is therefore deliberately excluded.
ENTRY_STOP_STATE_ABORTABLE: frozenset[str] = frozenset({
    ENTRY_STOP_STATE_ARMED,
    ENTRY_STOP_STATE_CANCEL_PENDING,
})

# Canonical ``extras`` keys for an entry-stop watch row.
EXTRAS_KEY_ENTRY_STOP_STATE = 'entry_stop_state'
# Absolute price the watch fires the market at (the Pine entry's ``stop``).
EXTRAS_KEY_ENTRY_STOP_LEVEL = 'entry_stop_level'
# The native LIMIT leg's client-order-id — the leg-scoped cancel target.
EXTRAS_KEY_ENTRY_STOP_LIMIT_COID = 'entry_stop_limit_coid'
# The stop-fired MARKET order's deterministic client-order-id. Persisted on
# the transition into ``stop_market_pending`` (BEFORE the POST) so a restart
# can verify-before-resend.
EXTRAS_KEY_ENTRY_STOP_MARKET_COID = 'entry_stop_market_coid'


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
#   (terminal; the row is closed). ``restart_replay`` resumes only ``pending``
#   legs (a crash before the ack) and reconciles each against the live legs
#   before re-dispatching, so an already-executed close is never repeated.
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
# ``active`` — the bracket is replicated onto this leg (steady state, unlike the
#   close-leg row there is no separate post-ack transition: the amend either
#   landed or a clear/release follows). ``released`` — the bracket was cleared
#   from this leg (terminal; the row is closed). ``restart_replay`` re-asserts
#   ``active`` rows whose leg is still open and releases those whose leg vanished.
BRACKET_OWN_STATE_ACTIVE = 'active'
BRACKET_OWN_STATE_RELEASED = 'released'

# Live (non-terminal) bracket-ownership phases ``restart_replay`` re-asserts.
BRACKET_OWN_STATE_LIVE: frozenset[str] = frozenset({BRACKET_OWN_STATE_ACTIVE})

# Canonical ``extras`` keys for a bracket-ownership row.
EXTRAS_KEY_BRACKET_OWN_STATE = 'bracket_own_state'
# Broker leg id (cTrader ``positionId``) this exit's bracket is replicated onto.
EXTRAS_KEY_BRACKET_OWN_LEG_ID = 'bracket_own_leg_id'
# The owning exit's bracket-attach dispatch coid (``KIND_EXIT_SL``); the per-leg
# row coid is derived as ``f"{attach_coid}:{leg_id}"`` so a re-attach upserts the
# SAME row and two different exits get disjoint coid namespaces.
EXTRAS_KEY_BRACKET_OWN_ATTACH_COID = 'bracket_own_attach_coid'
# Pine-unit protective levels last replicated onto the leg (audit + modify diff).
EXTRAS_KEY_BRACKET_OWN_TP = 'bracket_own_tp'
EXTRAS_KEY_BRACKET_OWN_SL = 'bracket_own_sl'
EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET = 'bracket_own_trail_offset'


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
    """
    store.set_order_state(coid, STATE_DISPOSITION_UNKNOWN)


def mark_rejected(
        store: 'RunContext',
        *,
        coid: str,
) -> None:
    """Flip the row to :data:`STATE_REJECTED` (terminal).

    Used when the exchange returns a definitive reject — confirm
    ``REJECTED``, a 4xx response, or a synchronous reason string the
    plugin maps to ``ExchangeOrderRejectedError``. The row stays in
    place for audit; the engine clears its intent slot on the next
    sync.
    """
    store.set_order_state(coid, STATE_REJECTED)


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
    :param intent_key: The owning :class:`ExitIntent`'s diff key.
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
        :meth:`RunContext.close_order` after the state update. The
        engine sets this on every transition into
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
    store.upsert_order(coid, **upsert_fields)
    if close_row:
        store.close_order(coid)


def iter_active_engine_trigger_partial_legs(
        store: 'RunContext',
        *,
        symbol: str | None = None,
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
    :param symbol: Optional symbol filter — passed straight through
        to :meth:`RunContext.iter_live_orders`.
    :param from_entry: Optional Pine parent entry id filter — same
        passthrough; the cascade-cancel callers use this to target
        all legs of one parent.
    """
    for row in store.iter_live_orders(symbol=symbol, from_entry=from_entry):
        if row.state != STATE_PARTIAL_BRACKET_LEG:
            continue
        leg_state = (row.extras or {}).get(EXTRAS_KEY_LEG_STATE)
        if leg_state not in LEG_STATE_LIVE:
            continue
        yield row


# === One-way emulation close-leg helpers ===================================

def create_close_leg_row(
        store: 'RunContext',
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

    :param store: The active run context.
    :param coid: The leg row's deterministic client-order-id.
    :param symbol: Exchange-side symbol.
    :param side: The close side (opposite the position direction being reduced).
    :param qty: The Pine-unit quantity this leg closes (its FIFO slice).
    :param intent_key: The owning :class:`CloseIntent`'s diff key.
    :param pine_entry_id: The owning ``strategy.close(id=...)`` value, for audit.
    :param parent_close_coid: The CloseIntent dispatch's ``KIND_CLOSE`` coid.
    :param leg_id: Broker leg id (cTrader ``positionId``) this row closes.
    :param leg_volume: Broker-grid integer volume dispatched for the leg.
    :param extra_payload: Additional plugin-specific extras to merge.
    """
    extras: dict[str, Any] = {
        EXTRAS_KEY_CLOSE_LEG_STATE: CLOSE_LEG_STATE_PENDING,
        EXTRAS_KEY_CLOSE_LEG_ID: leg_id,
        EXTRAS_KEY_CLOSE_PARENT_COID: parent_close_coid,
        EXTRAS_KEY_CLOSE_LEG_VOLUME: leg_volume,
    }
    if extra_payload:
        extras.update(extra_payload)
    # A restart-mid-fan re-dispatch reaches this helper with the SAME derived
    # coid; reopen so the re-persisted row rejoins the live set (upsert_order
    # does not clear ``closed_ts_ms`` on its own). No-op when the row is absent
    # or already live.
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
        store: 'RunContext',
        *,
        coid: str,
        new_state: str,
        close_row: bool = False,
        extras_patch: Mapping[str, Any] | None = None,
) -> None:
    """Transition a close-leg row's :data:`EXTRAS_KEY_CLOSE_LEG_STATE`.

    Set ``close_row=True`` for the terminal :data:`CLOSE_LEG_STATE_DISPATCHED`
    transition (the per-leg close acked), which also calls
    :meth:`RunContext.close_order` so the row leaves
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
    existing = store.get_order(coid)
    merged: dict[str, Any] = dict(existing.extras or {}) if existing is not None else {}
    merged[EXTRAS_KEY_CLOSE_LEG_STATE] = new_state
    if extras_patch:
        merged.update(extras_patch)
    store.upsert_order(coid, extras=merged)
    if close_row:
        store.close_order(coid)


def iter_active_close_legs(
        store: 'RunContext',
        *,
        symbol: str | None = None,
) -> Iterator['OrderRow']:
    """Iterate live (pending) close-leg rows for restart replay.

    Returns rows whose ``orders.state`` is :data:`STATE_CLOSE_LEG`, are not
    closed, and whose :data:`EXTRAS_KEY_CLOSE_LEG_STATE` is in
    :data:`CLOSE_LEG_STATE_LIVE` — a fan-out leg whose close was persisted but
    never acked. :meth:`OneWayEmulator.restart_replay` reconciles each against
    the live broker legs before re-dispatching.

    :param store: The active run context.
    :param symbol: Optional symbol filter, passed to
        :meth:`RunContext.iter_live_orders`.
    """
    for row in store.iter_live_orders(symbol=symbol):
        if row.state != STATE_CLOSE_LEG:
            continue
        leg_state = (row.extras or {}).get(EXTRAS_KEY_CLOSE_LEG_STATE)
        if leg_state not in CLOSE_LEG_STATE_LIVE:
            continue
        yield row


# === One-way emulation bracket-ownership helpers ===========================

def create_bracket_ownership_row(
        store: 'RunContext',
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
        trail_offset: float | None,
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
    the live set cheaply via :meth:`RunContext.iter_live_orders`.

    :param store: The active run context.
    :param coid: The ownership row's deterministic client-order-id.
    :param symbol: Exchange-side symbol.
    :param side: The exit (bracket) side.
    :param qty: The owning :class:`ExitIntent`'s quantity, for audit.
    :param intent_key: The owning :class:`ExitIntent`'s diff key
        (``f"{pine_id}\\0{from_entry}"``), matched by the clear path.
    :param pine_entry_id: The owning ``strategy.exit(id=...)`` value.
    :param from_entry: The exit's parent entry id (its own column for filtering).
    :param leg_id: Broker leg id (cTrader ``positionId``) the bracket is on.
    :param attach_coid: The exit's ``KIND_EXIT_SL`` bracket-attach coid.
    :param tp_price: Pine-unit take-profit level last replicated (or ``None``).
    :param sl_price: Pine-unit stop-loss level last replicated (or ``None``).
    :param trail_offset: Pine-unit trailing offset last replicated (or ``None``).
    :param extra_payload: Additional plugin-specific extras to merge.
    """
    extras: dict[str, Any] = {
        EXTRAS_KEY_BRACKET_OWN_STATE: BRACKET_OWN_STATE_ACTIVE,
        EXTRAS_KEY_BRACKET_OWN_LEG_ID: leg_id,
        EXTRAS_KEY_BRACKET_OWN_ATTACH_COID: attach_coid,
        EXTRAS_KEY_BRACKET_OWN_TP: tp_price,
        EXTRAS_KEY_BRACKET_OWN_SL: sl_price,
        EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET: trail_offset,
    }
    if extra_payload:
        extras.update(extra_payload)
    # A re-attach (modify, or restart re-assert) reaches this helper with the
    # SAME derived coid; reopen so the re-persisted row rejoins the live set
    # (upsert_order does not clear ``closed_ts_ms`` on its own). No-op when the
    # row is absent or already live.
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
        store: 'RunContext',
        *,
        coid: str,
        new_state: str,
        close_row: bool = False,
        extras_patch: Mapping[str, Any] | None = None,
) -> None:
    """Transition a bracket-ownership row's :data:`EXTRAS_KEY_BRACKET_OWN_STATE`.

    Set ``close_row=True`` for the terminal :data:`BRACKET_OWN_STATE_RELEASED`
    transition (the bracket cleared from this leg), which also calls
    :meth:`RunContext.close_order` so the row leaves
    :func:`iter_active_bracket_ownerships`'s live set.

    :param store: The active run context.
    :param coid: The ownership row's client-order-id.
    :param new_state: Target :data:`BRACKET_OWN_STATE_ACTIVE` /
        :data:`BRACKET_OWN_STATE_RELEASED`.
    :param close_row: When ``True``, also close the row (terminal).
    :param extras_patch: Additional extras to merge before the upsert.
    :raises ValueError: When ``new_state`` is not a canonical ownership state.
    """
    if new_state not in (BRACKET_OWN_STATE_ACTIVE, BRACKET_OWN_STATE_RELEASED):
        raise ValueError(
            f"update_bracket_ownership_state: new_state must be one of "
            f"{{BRACKET_OWN_STATE_ACTIVE, BRACKET_OWN_STATE_RELEASED}}, "
            f"got {new_state!r}"
        )
    existing = store.get_order(coid)
    merged: dict[str, Any] = dict(existing.extras or {}) if existing is not None else {}
    merged[EXTRAS_KEY_BRACKET_OWN_STATE] = new_state
    if extras_patch:
        merged.update(extras_patch)
    store.upsert_order(coid, extras=merged)
    if close_row:
        store.close_order(coid)


def iter_active_bracket_ownerships(
        store: 'RunContext',
        *,
        symbol: str | None = None,
        from_entry: str | None = None,
) -> Iterator['OrderRow']:
    """Iterate live (active) bracket-ownership rows.

    Returns rows whose ``orders.state`` is :data:`STATE_BRACKET_OWN`, are not
    closed, and whose :data:`EXTRAS_KEY_BRACKET_OWN_STATE` is in
    :data:`BRACKET_OWN_STATE_LIVE`. The clear path filters further by
    ``intent_key`` to touch only the legs a specific exit owns; the restart pass
    re-asserts every active row against the live broker legs.

    :param store: The active run context.
    :param symbol: Optional symbol filter, passed to
        :meth:`RunContext.iter_live_orders`.
    :param from_entry: Optional parent-entry filter, passed to
        :meth:`RunContext.iter_live_orders` (the indexed ``from_entry`` column).
    """
    for row in store.iter_live_orders(symbol=symbol, from_entry=from_entry):
        if row.state != STATE_BRACKET_OWN:
            continue
        own_state = (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_STATE)
        if own_state not in BRACKET_OWN_STATE_LIVE:
            continue
        yield row


# === Entry-stop watch helpers ==============================================

def create_entry_stop_watch_row(
        store: 'RunContext',
        *,
        coid: str,
        symbol: str,
        side: str,
        qty: float,
        intent_key: str,
        pine_entry_id: str,
        stop_level: float,
        limit_coid: str,
        entry_stop_state: str = ENTRY_STOP_STATE_ARMED,
        extra_payload: Mapping[str, Any] | None = None,
) -> None:
    """Persist one entry-stop watch row (PERSIST-FIRST).

    The row carries no exchange-side order — the
    :class:`~pynecore.core.broker.software_entry_stop_engine.SoftwareEntryStopEngine`
    state machine owns it. ``orders.state`` is set to
    :data:`STATE_ENTRY_STOP_WATCH` so the journal / reconcile paths
    short-circuit on these rows; the watch phase lives under
    :data:`EXTRAS_KEY_ENTRY_STOP_STATE` and is advanced via
    :func:`update_entry_stop_watch_state`.

    :param store: The active run context.
    :param coid: The watch row's deterministic client-order-id — built from
        the parent entry envelope with
        :data:`~pynecore.core.broker.idempotency.KIND_ENTRY_STOP_WATCH`, so a
        restart-mid-dispatch never produces two rows for one watch. Distinct
        from the stop-fired MARKET order's id
        (:data:`~pynecore.core.broker.idempotency.KIND_ENTRY_STOP`).
    :param symbol: Exchange-side symbol.
    :param side: The ENTRY side (``'buy'`` / ``'sell'``) — the direction the
        stop-fired MARKET opens in, same as the native LIMIT leg.
    :param qty: Entry quantity, already quantized to the broker's lot step.
    :param intent_key: The owning :class:`EntryIntent`'s diff key (≡ pine_id).
    :param pine_entry_id: The Pine ``strategy.entry(id=...)`` value.
    :param stop_level: Absolute price the watch fires the MARKET at.
    :param limit_coid: The native LIMIT leg's client-order-id — the
        leg-scoped cancel target when the stop crosses.
    :param entry_stop_state: Initial state, normally
        :data:`ENTRY_STOP_STATE_ARMED`.
    :param extra_payload: Additional extras to merge into the row.
    :raises ValueError: When ``entry_stop_state`` is not a live state.
    """
    if entry_stop_state not in ENTRY_STOP_STATE_LIVE:
        raise ValueError(
            f"create_entry_stop_watch_row: entry_stop_state must be one of "
            f"{sorted(ENTRY_STOP_STATE_LIVE)}, got {entry_stop_state!r}"
        )
    extras: dict[str, Any] = {
        EXTRAS_KEY_ENTRY_STOP_STATE: entry_stop_state,
        EXTRAS_KEY_ENTRY_STOP_LEVEL: stop_level,
        EXTRAS_KEY_ENTRY_STOP_LIMIT_COID: limit_coid,
        EXTRAS_KEY_ENTRY_STOP_MARKET_COID: None,
    }
    if extra_payload:
        extras.update(extra_payload)
    store.reopen_order(coid)
    store.upsert_order(
        coid,
        symbol=symbol,
        side=side,
        qty=qty,
        state=STATE_ENTRY_STOP_WATCH,
        intent_key=intent_key,
        pine_entry_id=pine_entry_id,
        extras=extras,
    )


def update_entry_stop_watch_state(
        store: 'RunContext',
        *,
        coid: str,
        new_state: str,
        market_coid: str | None = None,
        extras_patch: Mapping[str, Any] | None = None,
        close_row: bool | None = None,
        stop_level: float | None = None,
        qty: float | None = None,
        side: str | None = None,
) -> None:
    """Transition an entry-stop watch row.

    :param store: The active run context.
    :param coid: The watch row's client-order-id.
    :param new_state: Target :data:`EXTRAS_KEY_ENTRY_STOP_STATE` value.
    :param market_coid: When non-``None``, persists the stop-fired MARKET
        order's deterministic client-order-id under
        :data:`EXTRAS_KEY_ENTRY_STOP_MARKET_COID`. Written on the transition
        into :data:`ENTRY_STOP_STATE_MARKET_PENDING`, BEFORE the POST, so a
        restart can verify-before-resend.
    :param extras_patch: Additional extras keys to merge before the upsert.
    :param close_row: When ``True``, also calls
        :meth:`RunContext.close_order` — set on every transition into
        :data:`ENTRY_STOP_STATE_TERMINAL`.
    :param stop_level: When non-``None``, overwrites the watch's fire level
        under :data:`EXTRAS_KEY_ENTRY_STOP_LEVEL`. Used by the modify path to
        re-sync the software STOP leg to an amended both-set entry.
    :param qty: When non-``None``, overwrites the watch row's ``qty`` column
        (the stop-fired MARKET size). Same modify-path use as ``stop_level``.
    :param side: When non-``None``, overwrites the watch row's ``side`` column
        (the entry direction). Same modify-path use as ``stop_level``.
    :raises ValueError: When ``new_state`` is not a canonical value.
    """
    if new_state not in (ENTRY_STOP_STATE_LIVE | ENTRY_STOP_STATE_TERMINAL):
        raise ValueError(
            f"update_entry_stop_watch_state: new_state must be one of "
            f"{sorted(ENTRY_STOP_STATE_LIVE | ENTRY_STOP_STATE_TERMINAL)}, "
            f"got {new_state!r}"
        )
    existing = store.get_order(coid)
    merged: dict[str, Any] = dict(existing.extras or {}) if existing is not None else {}
    merged[EXTRAS_KEY_ENTRY_STOP_STATE] = new_state
    if market_coid is not None:
        merged[EXTRAS_KEY_ENTRY_STOP_MARKET_COID] = market_coid
    if stop_level is not None:
        merged[EXTRAS_KEY_ENTRY_STOP_LEVEL] = stop_level
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


def iter_active_entry_stop_watches(
        store: 'RunContext',
        *,
        symbol: str | None = None,
) -> Iterator['OrderRow']:
    """Iterate live entry-stop watch rows.

    Used by :meth:`SoftwareEntryStopEngine.restart_replay` to rebuild the
    in-memory watch ledger after a runner restart. Returns rows whose
    ``orders.state`` is :data:`STATE_ENTRY_STOP_WATCH`, are not closed, and
    whose :data:`EXTRAS_KEY_ENTRY_STOP_STATE` is one of
    :data:`ENTRY_STOP_STATE_LIVE` (terminal rows are filtered out even when
    ``close_order`` has not yet flipped ``closed_ts_ms``).

    :param store: The active run context.
    :param symbol: Optional symbol filter — passed straight through to
        :meth:`RunContext.iter_live_orders`.
    """
    for row in store.iter_live_orders(symbol=symbol):
        if row.state != STATE_ENTRY_STOP_WATCH:
            continue
        watch_state = (row.extras or {}).get(EXTRAS_KEY_ENTRY_STOP_STATE)
        if watch_state not in ENTRY_STOP_STATE_LIVE:
            continue
        yield row
