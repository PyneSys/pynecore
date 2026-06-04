"""
Engine-side trigger state machine for partial-quantity bracket exits.

This module owns the lifecycle of a partial-quantity TP / SL / trailing
bracket that the *engine* watches and fires — as opposed to a native
broker bracket attached to the parent ``dealId`` (covered by
``execute_exit``).

The state machine and the persisted leg rows in :mod:`store_helpers`
form one pair: the rows are the durable representation (PERSIST-FIRST),
the in-memory :class:`PartialBracketLeg` ledger is the working set the
WATCH phase reads on every WS price tick. On a clean restart the
ledger is rebuilt from the rows by :meth:`SoftwarePartialBracketEngine.restart_replay`.

The dispatch-side counterpart lives in
:meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._dispatch_engine_trigger_partial_bracket`:
it writes the leg rows, then hands them to this state machine via
:meth:`SoftwarePartialBracketEngine.register_legs`. After that the
machine owns the legs until a terminal state.

Lifecycle ownership: the state machine itself, the in-memory ledger,
the PERSIST-FIRST → ledger handoff, the cascade-cancel paths (OCA,
parent close, broker-native SL), the restart replay, and the
close-dispatch settling step
(:meth:`SoftwarePartialBracketEngine.confirm_trigger_dispatched`,
:meth:`SoftwarePartialBracketEngine.mark_trigger_dispatch_failed`,
:meth:`SoftwarePartialBracketEngine.mark_trigger_dispatch_unknown`).
The price-tick wiring runs from
:meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._drive_partial_bracket_triggers`
at the tail of every sync cycle for plugins that advertise
``partial_qty_bracket_exit = CapabilityLevel.SOFTWARE``: the sync
engine refreshes the parent snapshot, calls
:meth:`SoftwarePartialBracketEngine.on_price_tick`, synthesises a
:class:`CloseIntent` for each triggering leg, and dispatches it
through the regular ``_dispatch_new`` path before calling the
matching settling method on the state machine. See the partial-qty
bracket exit design dossier §3 for the full lifecycle.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable

from pynecore.core.broker.models import OcaType
from pynecore.core.broker.store_helpers import (
    EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS,
    EXTRAS_KEY_INTENT_PARTIAL_QTY,
    EXTRAS_KEY_LEG_KIND,
    EXTRAS_KEY_LEG_STATE,
    EXTRAS_KEY_OCA_GROUP,
    EXTRAS_KEY_OCA_TYPE,
    EXTRAS_KEY_PARENT_ENTRY_DISPATCH_REF,
    EXTRAS_KEY_PARENT_PINE_ENTRY_ID,
    EXTRAS_KEY_TRAIL_ACTIVATION_LEVEL,
    EXTRAS_KEY_TRAIL_ACTIVATION_OFFSET,
    EXTRAS_KEY_TRIGGER_LEVEL,
    EXTRAS_KEY_TRIGGER_OFFSET,
    LEG_KIND_SL_PARTIAL,
    LEG_KIND_TP_PARTIAL,
    LEG_KIND_TRAIL_PARTIAL,
    LEG_STATE_ABORTED_PARENT_GONE,
    LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED,
    LEG_STATE_ACTIVE,
    LEG_STATE_ARMED,
    LEG_STATE_CANCEL_TENTATIVE,
    LEG_STATE_CASCADED_CANCEL,
    LEG_STATE_CASCADED_CANCEL_BY_NATIVE_SL,
    LEG_STATE_CASCADED_CANCEL_BY_PARENT_CLOSE,
    LEG_STATE_LIVE,
    LEG_STATE_PENDING_ENTRY,
    LEG_STATE_TRIGGERED,
    LEG_STATE_TRIGGERED_FAILED,
    LEG_STATE_TRIGGERED_UNKNOWN,
    LEG_STATE_TRIGGERING,
    iter_active_engine_trigger_partial_legs,
    update_engine_trigger_partial_leg_state,
)

if TYPE_CHECKING:
    from pynecore.core.broker.models import ExchangePosition
    from pynecore.core.broker.storage import OrderRow, RunContext


__all__ = [
    'PartialBracketLeg',
    'PartialBracketSafetyVerdict',
    'SoftwarePartialBracketEngine',
    'TickOffsetResolver',
]


# Composite key uniquely identifying a leg in the in-memory ledger.
# The third element discriminates between concurrent legs sharing the
# same (pine_id, from_entry) — a tp_partial / sl_partial / trail_partial
# trio attached to one ``strategy.exit(...)``.
LegKey = tuple[str, str, str]


@dataclass
class PartialBracketLeg:
    """In-memory mirror of one engine-trigger partial bracket leg row.

    Mirrors the canonical extras keys in :mod:`store_helpers`; the
    persistent representation is the authoritative one (the engine
    reloads this struct from there on restart). Field names match
    the ``EXTRAS_KEY_*`` constants so a future refactor that flips
    the storage layout (e.g. dedicated SQL columns) is a one-place
    rename.

    ``side`` is the CLOSE side — opposite of the parent direction —
    because the leg's purpose is to dispatch a partial close, not a
    fresh open.
    """
    coid: str
    symbol: str
    pine_id: str
    from_entry: str
    leg_kind: str
    leg_state: str
    side: str
    qty: float
    intent_key: str
    parent_pine_entry_id: str
    parent_entry_dispatch_ref: str
    intent_partial_qty: float
    trigger_level: float | None = None
    trigger_offset: float | None = None
    # Trail-only fields. ``trail_activation_level`` is the absolute price
    # the engine waits for before the trailing stop activates: while it
    # is non-``None`` the leg is in its pre-activation phase and
    # :meth:`SoftwarePartialBracketEngine._is_triggered` always returns
    # ``False``. The moment the WATCH-phase price crosses this level,
    # the engine clears the field and seeds ``trigger_level`` with the
    # initial moving stop (``activation - offset`` for a long parent,
    # ``activation + offset`` for a short). ``trail_activation_offset``
    # carries the *pre*-fill distance (price units) for trail legs
    # whose parent entry is still pending — the parent-fill handler
    # resolves it to an absolute activation level.
    trail_activation_level: float | None = None
    trail_activation_offset: float | None = None
    oca_group: str | None = None
    oca_type: str | None = None
    extras: dict = field(default_factory=dict)

    @property
    def key(self) -> LegKey:
        return self.pine_id, self.from_entry, self.leg_kind


@dataclass(frozen=True)
class PartialBracketSafetyVerdict:
    """Outcome of the §2.4 safety check at trigger time.

    ``ok=True`` means the live parent snapshot still authorises the
    close: the position is on the expected side and has enough open
    quantity to reduce by ``effective_close_qty`` without flipping
    or opening. ``ok=False`` means the engine must NOT dispatch a
    close — the parent vanished or reversed under us.
    """
    ok: bool
    reason: str
    effective_close_qty: float


# Sign of the parent side, normalised to +1 (long) / -1 (short) — used
# by trigger comparisons. ``unknown`` is a guard for pending legs whose
# parent direction has not been observed yet.
_SIDE_LONG = 'long'
_SIDE_SHORT = 'short'


def _parent_sign(side: str) -> int | None:
    if side == _SIDE_LONG:
        return 1
    if side == _SIDE_SHORT:
        return -1
    return None


def _close_side_for_parent(parent_side: str) -> str:
    """The CLOSE order side for a given parent direction."""
    if parent_side == _SIDE_LONG:
        return 'sell'
    if parent_side == _SIDE_SHORT:
        return 'buy'
    raise ValueError(f"unexpected parent side: {parent_side!r}")


class SoftwarePartialBracketEngine:
    """In-memory state machine for engine-trigger partial brackets.

    One instance per :class:`OrderSyncEngine`. The engine does not
    talk to the broker directly; the sync engine performs every
    broker-side action (close dispatch, audit emission, safety
    snapshot) on the state machine's behalf via dedicated hooks
    that will be wired in Slice B. This separation keeps the state
    machine deterministic and unit-testable in isolation.

    The legacy bracket leg row format (the ``tp`` / ``sl`` legs the
    journal already manages for native brackets) is NOT touched by
    this state machine — those rows remain owned by the plugin's
    bracket lifecycle. The engine-trigger leg rows use the
    dedicated :data:`~pynecore.core.broker.store_helpers.STATE_PARTIAL_BRACKET_LEG`
    marker so the two never collide.
    """

    def __init__(
            self,
            store_ctx: 'RunContext | None',
            *,
            state_change_listener: 'Callable[[PartialBracketLeg, str | None, str], None] | None' = None,
    ) -> None:
        self._store_ctx = store_ctx
        # Listener fired after every leg-state mutation (register / transition).
        # The sync engine wires this to drive the §2.6 worst-SL recompute
        # so the broker-native fail-safe state machine sees every armed /
        # triggered / cancelled SL leg without coupling the partial-bracket
        # engine to the failsafe manager directly.
        self._state_change_listener = state_change_listener
        self._legs: dict[LegKey, PartialBracketLeg] = {}
        # Membership index for the OCA cascade. ``oca_group`` is the
        # private partial-exit group name the dispatch helper mints
        # (``__partial_exit_{pine_id}_{from_entry}__``); the cascade
        # walks all members of one group when any of them triggers.
        self._legs_by_oca_group: dict[str, set[LegKey]] = {}
        # Membership index keyed by Pine parent entry id, for the
        # parent-gone / native-SL cascade paths that target every leg
        # attached to one parent regardless of OCA grouping.
        self._legs_by_parent: dict[tuple[str, str], set[LegKey]] = {}

    # === Registration =====================================================

    def register_leg(self, leg: PartialBracketLeg) -> None:
        """Add one freshly-persisted leg to the in-memory ledger.

        Called by the sync engine after a successful
        :func:`~pynecore.core.broker.store_helpers.create_engine_trigger_partial_leg_row`.
        The leg must already be in :data:`LEG_STATE_ACTIVE` — terminal
        rows are rejected because they have no business in the ledger.
        """
        if leg.leg_state not in LEG_STATE_ACTIVE:
            raise ValueError(
                f"register_leg: refuses to track leg in non-active state "
                f"{leg.leg_state!r} (key={leg.key!r})"
            )
        if leg.key in self._legs:
            raise ValueError(
                f"register_leg: leg already tracked (key={leg.key!r}, "
                f"existing coid={self._legs[leg.key].coid!r}, "
                f"new coid={leg.coid!r})"
            )
        self._legs[leg.key] = leg
        if leg.oca_group is not None:
            self._legs_by_oca_group.setdefault(leg.oca_group, set()).add(leg.key)
        self._legs_by_parent.setdefault(
            (leg.symbol, leg.from_entry), set(),
        ).add(leg.key)
        if self._state_change_listener is not None:
            self._state_change_listener(leg, None, leg.leg_state)

    def register_legs(self, legs: Iterable[PartialBracketLeg]) -> None:
        for leg in legs:
            self.register_leg(leg)

    # === Queries ==========================================================

    def get_leg(self, key: LegKey) -> PartialBracketLeg | None:
        return self._legs.get(key)

    def iter_legs(self) -> Iterable[PartialBracketLeg]:
        return list(self._legs.values())

    def iter_legs_for_parent(
            self, symbol: str, from_entry: str,
    ) -> list[PartialBracketLeg]:
        keys = self._legs_by_parent.get((symbol, from_entry), set())
        return [self._legs[k] for k in keys if k in self._legs]

    def has_active_partial_bracket(
            self, symbol: str, from_entry: str,
    ) -> bool:
        """Whether any active engine-trigger partial leg exists for a parent.

        Used by the sync engine's invariant guard at
        ``_dispatch_engine_trigger_partial_bracket`` entry: a freshly
        dispatched native full-row bracket on the same parent must
        not coexist with engine-trigger partial legs (§12 #4).
        """
        for leg in self.iter_legs_for_parent(symbol, from_entry):
            if leg.leg_state in LEG_STATE_ACTIVE:
                return True
        return False

    def has_active_legs_for_intent(self, intent_key: str) -> bool:
        """Whether any active leg with the given ``intent_key`` is tracked.

        Used by the dispatch-side guard to reject a *re-dispatch* of the
        same :class:`~pynecore.core.broker.models.ExitIntent`. The check
        is intent-scoped (not parent-scoped) because Pine permits
        multiple scale-out exits under one parent — e.g. TP1 / TP2 with
        different ``strategy.exit(id=...)`` values share the same
        ``from_entry`` but have distinct ``intent_key`` values
        (``(pine_id, from_entry)``). A parent-wide guard would block the
        second scale-out as a "duplicate".
        """
        for leg in self._legs.values():
            if leg.intent_key == intent_key \
                    and leg.leg_state in LEG_STATE_ACTIVE:
                return True
        return False

    def has_sibling_active_legs(
            self, symbol: str, from_entry: str, exclude_intent_key: str,
    ) -> bool:
        """Whether any active leg exists on a parent under a DIFFERENT intent_key.

        Used by the partial-bracket modify preflight to detect the
        scale-out sibling case: parent ``from_entry`` already carries
        another scale-out exit (TP1 / TP2 under distinct ``pine_id`` →
        distinct ``intent_key``) whose legs would survive a cancel of
        ``exclude_intent_key``. The whole-row exit replacement issued
        after that cancel would then hit the §12 #4 coexistence guard
        in :meth:`OrderSyncEngine._dispatch_new`, so the caller must
        refuse the modify before evicting the legs that would leave
        the parent unprotected.
        """
        for leg in self.iter_legs_for_parent(symbol, from_entry):
            if leg.intent_key != exclude_intent_key \
                    and leg.leg_state in LEG_STATE_ACTIVE:
                return True
        return False

    def cancel_legs_for_intent(
            self, intent_key: str, *, reason: str,
    ) -> list[PartialBracketLeg]:
        """Cancel every active leg under one ``intent_key``.

        Used by the sync engine when the strategy drops or replaces a
        partial-bracket ``ExitIntent``: the leg row dispatch is
        engine-internal, so the standard ``execute_cancel`` /
        ``modify_exit`` broker calls do not apply. The cascade
        transitions each leg to :data:`LEG_STATE_CASCADED_CANCEL`
        with the supplied ``reason`` recorded in the row's audit
        extras.
        """
        cancelled: list[PartialBracketLeg] = []
        # Snapshot first; the transition mutates the ledger.
        targets = [
            leg for leg in self._legs.values()
            if leg.intent_key == intent_key
            and leg.leg_state in LEG_STATE_ACTIVE
        ]
        for leg in targets:
            self._transition(
                leg, LEG_STATE_CASCADED_CANCEL,
                close_row=True,
                extras_patch={'cascade_reason': reason},
            )
            cancelled.append(leg)
        return cancelled

    # === Pending → armed promotion =======================================

    def on_parent_entry_filled(
            self,
            *,
            symbol: str,
            from_entry: str,
            fill_price: float,
            parent_side: str,
            parent_qty: float,
            resolver: 'TickOffsetResolver | None' = None,
    ) -> list[PartialBracketLeg]:
        """Promote ``pending_entry`` legs to ``armed`` on the parent fill.

        Resolves ``trigger_offset`` (tick distance) into an absolute
        ``trigger_level`` and updates the persisted row. The caller
        is the sync engine's parent-fill handler; the ``resolver``
        encodes the ``profit_ticks`` / ``loss_ticks`` / ``trail_points_ticks``
        → price conversion using the symbol's ``mintick``.

        Returns the legs that were promoted. The empty list is fine
        (no pending legs for this parent).
        """
        promoted: list[PartialBracketLeg] = []
        parent_sign = _parent_sign(parent_side)
        if parent_sign is None:
            return promoted
        for leg in self.iter_legs_for_parent(symbol, from_entry):
            if leg.leg_state != LEG_STATE_PENDING_ENTRY:
                continue
            extras_patch: dict | None = None
            level = leg.trigger_level
            # For trail legs we resolve the *activation* level — the
            # leg stays in pre-activation phase until WATCH observes
            # the price crossing it. TP / SL legs have no activation
            # concept: ``trigger_level`` is the final price the engine
            # watches and the resolver fills it directly.
            if leg.leg_kind == LEG_KIND_TRAIL_PARTIAL:
                activation = leg.trail_activation_level
                if activation is None and leg.trail_activation_offset is not None:
                    activation = (
                        fill_price + parent_sign * leg.trail_activation_offset
                    )
                if activation is None:
                    continue
                leg.trail_activation_level = activation
                extras_patch = {EXTRAS_KEY_TRAIL_ACTIVATION_LEVEL: activation}
            else:
                if level is None and resolver is not None:
                    level = resolver.resolve(leg, fill_price=fill_price,
                                             parent_sign=parent_sign)
                if level is None:
                    continue
                leg.trigger_level = level
            old_state = leg.leg_state
            leg.leg_state = LEG_STATE_ARMED
            if self._store_ctx is not None:
                update_engine_trigger_partial_leg_state(
                    self._store_ctx,
                    coid=leg.coid,
                    new_leg_state=LEG_STATE_ARMED,
                    trigger_level=level,
                    extras_patch=extras_patch,
                )
            # Fire the state-change listener so the §2.6 worst-SL
            # recompute picks up the just-armed leg's absolute SL.
            # Without this notification, ``_recompute_native_failsafe_for_parent``
            # only re-walks the legs when a *later* unrelated leg event
            # mutates state — until then the leg is invisible to the
            # broker-native failsafe even though it is fully armed.
            if self._state_change_listener is not None:
                self._state_change_listener(leg, old_state, LEG_STATE_ARMED)
            promoted.append(leg)
            _ = parent_qty  # reserved for §2.4 future qty-bookkeeping
        return promoted

    # === WATCH phase ======================================================

    def on_price_tick(
            self,
            *,
            symbol: str,
            last_price: float | None,
            bid: float | None,
            ask: float | None,
            parent_snapshot: 'ExchangePosition | None',
    ) -> list[PartialBracketLeg]:
        """Run the WATCH-phase trigger check for all armed legs.

        Returns the legs that crossed their trigger this tick and
        passed the safety check — i.e. legs in :data:`LEG_STATE_TRIGGERING`
        that the sync engine should now dispatch a partial
        :class:`CloseIntent` for. The close dispatch itself is the
        sync engine's responsibility (Slice B wiring); this method
        owns only the state-machine bookkeeping.

        :param symbol: Exchange symbol whose armed legs are checked this
            tick; legs belonging to other symbols are skipped.
        :param last_price: Last traded price. Used for trail recompute
            and as the default trigger comparison source.
        :param bid: Best bid; long-side TP and short-side SL compare
            against this once the §12 #2 quote source is resolved.
        :param ask: Best ask; short-side TP and long-side SL compare
            against this.
        :param parent_snapshot: Live position snapshot for the safety
            check. ``None`` means the engine could not refresh the
            snapshot — every trigger is suppressed this tick (the
            machine waits for a snapshot rather than firing blind).
        """
        triggering: list[PartialBracketLeg] = []
        if parent_snapshot is None:
            return triggering
        # Same-tick reservation: when more than one armed leg under the
        # same (symbol, from_entry) crosses on the same tick, each call
        # to :meth:`_safety_check` would otherwise cap against the
        # unchanged ``parent.size``. The loop decrements the remaining
        # parent size as legs are accepted so two 0.75 legs on a 1.0
        # position cannot collectively dispatch 1.5 and flip the parent.
        remaining_by_parent: dict[tuple[str, str], float] = {}
        # OCA groups that already produced a triggering leg in *this*
        # tick. Used to skip incompatible siblings (e.g. TP + SL crossing
        # together on a gap) without cancelling them: cancelling SL/trail
        # legs synchronously here would fire the §2.6 worst-SL recompute
        # listener and drop the broker-native stop *before* the
        # triggered TP's CLOSE has been accepted/filled. If the close
        # fails, times out, or the process crashes between detection and
        # dispatch, the parent would be left with no software *and* no
        # native protection. Keep the siblings ``armed`` instead; the
        # cascade cancel runs later, on the ``triggering → triggered``
        # transition once the close is confirmed (Slice B dispatch
        # wiring) or on the parent-flat observation.
        reserved_oca_groups: set[str] = set()
        # Cross-tick reservation: a sibling already in ``triggering`` /
        # ``triggered_failed`` / ``triggered_unknown`` from an earlier
        # tick still has an in-flight close — the cascade cancel runs
        # only on the ``triggering → triggered`` confirmation. Until
        # then, another armed sibling in the same OCA group must not be
        # allowed to cross and dispatch a second close (e.g. TP close
        # pending, then price reverses into the SL/trail level on the
        # next tick): that would over-reduce the parent before the
        # first close resolves. Seed the reservation set so the
        # per-iteration skip below covers in-flight siblings too.
        for leg in self._legs.values():
            if leg.symbol != symbol:
                continue
            if leg.oca_group is None:
                continue
            if leg.leg_state in (
                    LEG_STATE_TRIGGERING,
                    LEG_STATE_TRIGGERED_FAILED,
                    LEG_STATE_TRIGGERED_UNKNOWN,
            ):
                reserved_oca_groups.add(leg.oca_group)
        for leg in list(self._legs.values()):
            if leg.symbol != symbol:
                continue
            if leg.leg_state != LEG_STATE_ARMED:
                continue
            if (leg.oca_group is not None
                    and leg.oca_group in reserved_oca_groups):
                # Sibling in the same OCA group already crossed this
                # tick; skip without cancelling so SL protection stays
                # in place until the winning leg's close is confirmed.
                continue
            # Trail legs are armed *before* activation and intentionally
            # carry ``trigger_level=None`` until the price crosses
            # ``trail_activation_level``. Run :meth:`_maybe_advance_trail`
            # first so a pre-activation tick can seed the initial stop;
            # only after that does a missing ``trigger_level`` mean the
            # leg has nothing to compare against this tick (still
            # pre-activation, or non-trail leg without a resolved level).
            if leg.leg_kind == LEG_KIND_TRAIL_PARTIAL:
                self._maybe_advance_trail(leg, last_price)
            if leg.trigger_level is None:
                continue
            if not self._is_triggered(leg, last_price=last_price,
                                      bid=bid, ask=ask):
                continue
            parent_key = (leg.symbol, leg.from_entry)
            remaining = remaining_by_parent.get(parent_key, parent_snapshot.size)
            verdict = self._safety_check(leg, parent_snapshot,
                                         remaining_size=remaining)
            if not verdict.ok:
                self._transition(
                    leg, LEG_STATE_ABORTED_PARENT_GONE,
                    close_row=True,
                    extras_patch={'safety_abort_reason': verdict.reason},
                )
                continue
            remaining_by_parent[parent_key] = max(
                0.0, remaining - verdict.effective_close_qty,
            )
            self._transition(leg, LEG_STATE_TRIGGERING, close_row=False,
                             qty=verdict.effective_close_qty)
            triggering.append(leg)
            if leg.oca_group is not None:
                reserved_oca_groups.add(leg.oca_group)
        return triggering

    # noinspection PyMethodMayBeStatic
    def _is_triggered(
            self,
            leg: PartialBracketLeg,
            *,
            last_price: float | None,
            bid: float | None,
            ask: float | None,
    ) -> bool:
        """Whether the current quote crosses the leg's trigger level.

        The bid/ask vs. last-price selection is intentionally simple
        in Slice A and defers to ``last_price`` whenever it is
        available; §12 #2 will refine the choice once the WS quote
        feed semantics are validated in production. The comparison
        directions:

        - Long parent, TP leg: trigger when price >= level.
        - Long parent, SL / trail leg: trigger when price <= level.
        - Short parent, TP leg: trigger when price <= level.
        - Short parent, SL / trail leg: trigger when price >= level.

        Trail legs whose ``trail_activation_level`` is still set
        return ``False`` here — the leg is armed but the trailing
        stop has not yet "activated" so the trigger comparison is
        meaningless. Activation itself is handled by
        :meth:`_maybe_advance_trail`, which clears the activation
        field and seeds ``trigger_level`` once the price crosses the
        activation threshold.
        """
        if leg.leg_kind == LEG_KIND_TRAIL_PARTIAL \
                and leg.trail_activation_level is not None:
            return False
        if leg.trigger_level is None:
            return False
        price = last_price if last_price is not None else (
            bid if bid is not None else ask
        )
        if price is None:
            return False
        parent_long = leg.side == 'sell'
        is_take_profit = leg.leg_kind == LEG_KIND_TP_PARTIAL
        if parent_long:
            return (price >= leg.trigger_level) if is_take_profit \
                else (price <= leg.trigger_level)
        return (price <= leg.trigger_level) if is_take_profit \
            else (price >= leg.trigger_level)

    def _maybe_advance_trail(
            self,
            leg: PartialBracketLeg,
            last_price: float | None,
    ) -> None:
        """Drive a trailing-stop leg through activation and the trail itself.

        Two-phase contract:

        1. **Pre-activation** (``trail_activation_level`` set): the
           method waits for ``last_price`` to cross the activation
           threshold in the favourable direction (>= for a long parent,
           <= for a short). When it does, it clears
           ``trail_activation_level`` and seeds ``trigger_level`` with
           the initial moving stop (``activation - offset`` long /
           ``activation + offset`` short). The leg now enters the
           normal trail phase and the next price tick may already
           trigger via :meth:`_is_triggered`.
        2. **Active trail** (``trail_activation_level`` is ``None``):
           the trigger level only moves in the favourable direction —
           a long parent trail can only rise, a short can only fall.

        Both transitions persist the new ``trigger_level`` /
        ``trail_activation_level`` so a restart mid-trail does not lose
        the high-water mark or the activation state.
        """
        if leg.leg_kind != LEG_KIND_TRAIL_PARTIAL:
            return
        if last_price is None or leg.trigger_offset is None:
            return
        parent_long = leg.side == 'sell'
        if leg.trail_activation_level is not None:
            # Pre-activation: wait for the activation threshold.
            activation = leg.trail_activation_level
            activated = (last_price >= activation) if parent_long \
                else (last_price <= activation)
            if not activated:
                return
            # Seed the initial stop from the current favourable price,
            # not from the activation threshold. When the tick jumps
            # past activation (e.g. activation=100, last_price=110,
            # offset=5 on a long), the initial stop must be
            # ``last_price - offset = 105`` so the trail starts as
            # tight as the offset prescribes. Anchoring on activation
            # would leave the stop at 95 — too loose by the entire
            # over-shoot — until another favourable tick arrived.
            initial_stop = (last_price - leg.trigger_offset) if parent_long \
                else (last_price + leg.trigger_offset)
            leg.trail_activation_level = None
            leg.trigger_level = initial_stop
            if self._store_ctx is not None:
                update_engine_trigger_partial_leg_state(
                    self._store_ctx,
                    coid=leg.coid,
                    new_leg_state=leg.leg_state,
                    trigger_level=initial_stop,
                    extras_patch={EXTRAS_KEY_TRAIL_ACTIVATION_LEVEL: None},
                )
            if self._state_change_listener is not None:
                # No state transition — but the trail leg's worst-SL
                # contribution just changed (pre-activation watch level
                # → post-activation trailing stop). Fire so the §2.6
                # failsafe manager recomputes with the new level.
                self._state_change_listener(leg, leg.leg_state, leg.leg_state)
            return
        candidate = (last_price - leg.trigger_offset) if parent_long \
            else (last_price + leg.trigger_offset)
        current = leg.trigger_level
        if current is None:
            new_level = candidate
        elif parent_long and candidate > current:
            new_level = candidate
        elif (not parent_long) and candidate < current:
            new_level = candidate
        else:
            return
        leg.trigger_level = new_level
        if self._store_ctx is not None:
            update_engine_trigger_partial_leg_state(
                self._store_ctx,
                coid=leg.coid,
                new_leg_state=leg.leg_state,
                trigger_level=new_level,
            )
        if self._state_change_listener is not None:
            # Trail leg moved its trigger_level while staying armed —
            # the §2.6 failsafe manager needs the new contribution.
            self._state_change_listener(leg, leg.leg_state, leg.leg_state)

    # noinspection PyMethodMayBeStatic
    def _safety_check(
            self,
            leg: PartialBracketLeg,
            parent: 'ExchangePosition',
            *,
            remaining_size: float | None = None,
    ) -> PartialBracketSafetyVerdict:
        """§2.4 invariant: parent on the expected side and large enough.

        The leg's recorded ``side`` is the CLOSE side: ``'sell'`` means
        the leg closes a long parent. If the live snapshot disagrees
        (parent flat, parent on the other side), the leg must abort —
        firing the close now could reopen a position in the wrong
        direction. The qty cap prevents over-reduction when the parent
        was partially closed by a different actor.

        :param remaining_size: When supplied, caps ``effective_close_qty``
            against this value instead of the raw ``parent.size``. The
            same-tick caller decrements the remaining parent size as
            sibling legs are accepted so two crossings on one tick cannot
            collectively close more than the live parent holds.
        """
        if parent.side == 'flat' or parent.size <= 0.0:
            return PartialBracketSafetyVerdict(
                ok=False,
                reason='parent_flat',
                effective_close_qty=0.0,
            )
        expected_parent_side = _SIDE_LONG if leg.side == 'sell' else _SIDE_SHORT
        if parent.side != expected_parent_side:
            return PartialBracketSafetyVerdict(
                ok=False,
                reason='parent_reversed',
                effective_close_qty=0.0,
            )
        size_cap = parent.size if remaining_size is None \
            else min(parent.size, remaining_size)
        # ``leg.qty`` is the per-leg cap that survives the row's lifecycle:
        # at first arming it equals ``intent_partial_qty``, but the
        # same-tick sibling cascade in :meth:`on_price_tick` rewrites it
        # via :meth:`_transition` with the smaller ``effective_close_qty``
        # before the close dispatches. If the process crashed between the
        # cap write and the close fill, :meth:`restart_replay` demotes the
        # leg back to ``armed`` but keeps the persisted capped value;
        # falling back to ``intent_partial_qty`` here would let the retry
        # close the full original size and over-reduce a parent that was
        # already partially closed by the prior cap's settled fill.
        desired_qty = min(leg.qty, leg.intent_partial_qty) \
            if leg.qty > 0.0 else leg.intent_partial_qty
        effective_qty = min(desired_qty, size_cap)
        if effective_qty <= 0.0:
            return PartialBracketSafetyVerdict(
                ok=False,
                reason='zero_effective_qty',
                effective_close_qty=0.0,
            )
        return PartialBracketSafetyVerdict(
            ok=True,
            reason='ok',
            effective_close_qty=effective_qty,
        )

    # === Close-dispatch settling ==========================================

    def confirm_trigger_dispatched(
            self,
            leg_key: LegKey,
            *,
            close_pine_id: str,
    ) -> list[PartialBracketLeg]:
        """Settle a :data:`LEG_STATE_TRIGGERING` leg as fired.

        Called by the sync engine immediately after the synthetic
        :class:`CloseIntent` for ``leg`` has been dispatched to the
        broker plugin. The leg moves to
        :data:`LEG_STATE_TRIGGERED` (terminal, row closed) and the
        OCA cascade fires for sibling legs sharing the same
        :attr:`PartialBracketLeg.oca_group` with
        :attr:`OcaType.CANCEL` semantics. ``close_pine_id`` is
        the synthesised exit id stamped on the close envelope so the
        cascade audit row can correlate the two.

        :return: List of OCA siblings that were cascaded by this
            settlement; empty if the leg has no OCA group or no
            cancellable sibling. The triggered leg itself is not
            included.
        """
        leg = self._legs.get(leg_key)
        if leg is None or leg.leg_state != LEG_STATE_TRIGGERING:
            return []
        # Cascade FIRST while the triggered leg is still in the ledger —
        # :meth:`cascade_cancel_oca` resolves the OCA group via
        # ``self._legs[triggered_key]``, so a prior eviction would silently
        # drop the cascade. The triggered leg's own state stays
        # ``triggering`` for this call (siblings already in a terminal
        # state are skipped by the cascade), and the final transition
        # below flips it to :data:`LEG_STATE_TRIGGERED` and closes the row.
        cascaded = self.cascade_cancel_oca(leg_key)
        self._transition(
            leg, LEG_STATE_TRIGGERED,
            close_row=True,
            extras_patch={'close_pine_id': close_pine_id},
        )
        return cascaded

    def mark_trigger_dispatch_failed(
            self,
            leg_key: LegKey,
            *,
            reason: str,
    ) -> None:
        """Settle a :data:`LEG_STATE_TRIGGERING` leg as a hard failure.

        Used when the close dispatch was rejected with an error the
        engine cannot reasonably retry on its own (e.g.
        :class:`~pynecore.core.broker.exceptions.OrderSkippedByPlugin`
        or a non-recoverable broker error that is NOT a network
        ambiguity). The leg lands briefly in
        :data:`LEG_STATE_TRIGGERED_FAILED` for the audit row, then
        demotes back to :data:`LEG_STATE_ARMED` so the next price
        tick re-evaluates against a fresh parent snapshot. Idempotent
        no-op when the leg is not in :data:`LEG_STATE_TRIGGERING`.
        """
        leg = self._legs.get(leg_key)
        if leg is None or leg.leg_state != LEG_STATE_TRIGGERING:
            return
        self._transition(
            leg, LEG_STATE_TRIGGERED_FAILED,
            close_row=False,
            extras_patch={'trigger_failed_reason': reason},
        )
        self._transition(
            leg, LEG_STATE_ARMED,
            close_row=False,
        )

    def mark_trigger_dispatch_unknown(
            self,
            leg_key: LegKey,
            *,
            reason: str,
    ) -> None:
        """Settle a :data:`LEG_STATE_TRIGGERING` leg as a parked dispatch.

        Used when the close dispatch raised
        :class:`~pynecore.core.broker.exceptions.OrderDispositionUnknownError`
        — the broker may or may not have accepted the order. The leg
        lands briefly in :data:`LEG_STATE_TRIGGERED_UNKNOWN` for the
        audit row, then demotes back to :data:`LEG_STATE_ARMED` so the
        next price tick re-evaluates the trigger. The sync engine's
        regular parked-dispatch resolution path (``_verify_pending_dispatches``)
        will reconcile any in-flight close that did in fact land — the
        re-armed leg's :meth:`_safety_check` caps against the live
        parent size and avoids over-closing if the prior attempt
        already reduced the position. Idempotent no-op when the leg
        is not in :data:`LEG_STATE_TRIGGERING`.
        """
        leg = self._legs.get(leg_key)
        if leg is None or leg.leg_state != LEG_STATE_TRIGGERING:
            return
        self._transition(
            leg, LEG_STATE_TRIGGERED_UNKNOWN,
            close_row=False,
            extras_patch={'trigger_unknown_reason': reason},
        )
        self._transition(
            leg, LEG_STATE_ARMED,
            close_row=False,
        )

    # === OCA cascade ======================================================

    def cascade_cancel_oca(
            self,
            triggered_key: LegKey,
            *,
            reason: str = 'oca_sibling_triggered',
    ) -> list[PartialBracketLeg]:
        """Cancel every armed sibling sharing the triggered leg's OCA group.

        Only :data:`OcaType.CANCEL` groups cascade — ``reduce`` and
        ``none`` groups leave their siblings alone, matching the
        sync-engine OCA path (:meth:`OrderSyncEngine._cascade_oca_cancel`)
        which gates on ``oca_type == OcaType.CANCEL.value``. Keying solely
        on ``oca_group`` here would otherwise tear down unrelated
        ``reduce``/``none`` siblings as soon as one leg triggers, dropping
        partial-exit protection the script intentionally kept independent.

        The triggered leg itself is left alone (the caller flips it to
        :data:`~pynecore.core.broker.store_helpers.LEG_STATE_TRIGGERED`
        after the close fill). Siblings already in a terminal state
        are skipped.
        """
        triggered = self._legs.get(triggered_key)
        if triggered is None or triggered.oca_group is None:
            return []
        if triggered.oca_type != OcaType.CANCEL.value:
            return []
        cancelled: list[PartialBracketLeg] = []
        for key in list(self._legs_by_oca_group.get(triggered.oca_group, ())):
            if key == triggered_key:
                continue
            sibling = self._legs.get(key)
            if sibling is None or sibling.leg_state not in LEG_STATE_ACTIVE:
                continue
            if sibling.oca_type != OcaType.CANCEL.value:
                # Mixed-type groupings are degenerate (a single oca_group
                # name should carry one consistent oca_type), but skip
                # defensively rather than cancel a ``reduce``/``none``
                # sibling that the script wanted to keep alive.
                continue
            self._transition(
                sibling, LEG_STATE_CASCADED_CANCEL,
                close_row=True,
                extras_patch={'cascade_reason': reason},
            )
            cancelled.append(sibling)
        return cancelled

    def cascade_cancel_by_parent_close(
            self,
            *,
            symbol: str,
            from_entry: str,
            reason: str = 'parent_closed',
    ) -> list[PartialBracketLeg]:
        """Cancel every active leg under a parent that just flattened."""
        cancelled: list[PartialBracketLeg] = []
        for leg in self.iter_legs_for_parent(symbol, from_entry):
            if leg.leg_state not in LEG_STATE_ACTIVE:
                continue
            self._transition(
                leg, LEG_STATE_CASCADED_CANCEL_BY_PARENT_CLOSE,
                close_row=True,
                extras_patch={'cascade_reason': reason},
            )
            cancelled.append(leg)
        return cancelled

    def cascade_cancel_by_native_sl(
            self,
            *,
            symbol: str,
            from_entry: str,
            sl_level: float,
    ) -> list[PartialBracketLeg]:
        """§3.4: a broker-native fail-safe SL hit replaces every active leg.

        The snapshot-driven reconcile in the sync engine observes
        the position vanishing; every remaining engine-trigger leg
        (intermediate SL, TP, trail) goes to
        :data:`~pynecore.core.broker.store_helpers.LEG_STATE_CASCADED_CANCEL_BY_NATIVE_SL`
        with the SL level recorded for audit.
        """
        cancelled: list[PartialBracketLeg] = []
        for leg in self.iter_legs_for_parent(symbol, from_entry):
            if leg.leg_state not in LEG_STATE_ACTIVE:
                continue
            self._transition(
                leg, LEG_STATE_CASCADED_CANCEL_BY_NATIVE_SL,
                close_row=True,
                extras_patch={
                    'cascade_reason': 'native_sl_hit',
                    'native_sl_level': sl_level,
                },
            )
            cancelled.append(leg)
        return cancelled

    def abort_pending_legs_for_parent_never_arrived(
            self,
            *,
            symbol: str,
            from_entry: str,
            reason: str,
    ) -> list[PartialBracketLeg]:
        """§3.2 cleanup: pending entry never filled (rejected / cancelled / expired).

        Only ``pending_entry`` legs are affected — already-armed legs
        belong to a different lifecycle path and would have aborted
        via the standard parent-gone cascade.
        """
        cleaned: list[PartialBracketLeg] = []
        for leg in self.iter_legs_for_parent(symbol, from_entry):
            if leg.leg_state != LEG_STATE_PENDING_ENTRY:
                continue
            self._transition(
                leg, LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED,
                close_row=True,
                extras_patch={'abort_reason': reason},
            )
            cleaned.append(leg)
        return cleaned

    # === Cancel-tentative state machine ===================================

    def mark_legs_cancel_tentative(
            self,
            parent_from_entry: str,
            *,
            reason: str,
            now_ms: int,
    ) -> list[PartialBracketLeg]:
        """Flip every ``pending_entry`` leg of one parent to
        ``cancel_tentative``.

        Called by the sync engine's swallowed-unknown branch of
        ``_dispatch_cancel`` when ``execute_cancel`` raises
        :class:`OrderDispositionUnknownError` on a parent entry. The
        legs stay live (the row's ``closed_ts_ms`` is not set) and are
        excluded from worst-SL contribution / price-tick arming until
        either the ``reconcile()`` cancel-retry-loop resolves the
        disposition (→ :meth:`confirm_cancel_tentative` or
        :meth:`restore_legs_from_cancel_tentative`) or the stale-grace
        deadline expires (→ ``DEGRADED_HALT`` by the caller).

        Already ``armed`` legs are NOT affected: per the dossier's leg-
        trio atomicity invariant (:meth:`on_parent_entry_filled` flips
        the whole trio atomically), ``ARMED + PENDING_ENTRY`` coexistence
        under a single parent is structurally impossible while the parent
        is still pending.

        :param parent_from_entry: The parent ``EntryIntent.intent_key``
            (≡ Pine ``from_entry`` on every child :class:`ExitIntent`).
        :param reason: Audit reason recorded on the leg row's extras.
        :param now_ms: Wall-clock timestamp (ms) marking entry into
            the cancel-tentative state. Persisted under
            :data:`EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS`; survives
            restart so the stale-grace deadline can be rehydrated.
        :return: List of legs that were transitioned.
        """
        flipped: list[PartialBracketLeg] = []
        targets = [
            leg for leg in self._legs.values()
            if leg.from_entry == parent_from_entry
            and leg.leg_state == LEG_STATE_PENDING_ENTRY
        ]
        for leg in targets:
            self._transition(
                leg, LEG_STATE_CANCEL_TENTATIVE,
                close_row=False,
                extras_patch={
                    EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS: now_ms,
                    'cancel_tentative_reason': reason,
                },
            )
            flipped.append(leg)
        return flipped

    def restore_legs_from_cancel_tentative(
            self,
            parent_from_entry: str,
            *,
            parent_filled: bool,
            reason: str,
    ) -> list[PartialBracketLeg]:
        """Reverse :meth:`mark_legs_cancel_tentative` for one parent.

        Called when the cancel-retry-loop or a late parent FILL event
        proves the parent is in fact alive (the cancel attempt that
        timed out never actually landed). Legs return to
        :data:`LEG_STATE_PENDING_ENTRY` when the parent is still pending
        (no fill event observed yet) or directly to
        :data:`LEG_STATE_ARMED` when the parent has filled
        (``parent_filled=True``). In the armed case the caller is
        responsible for re-registering the parent with the
        ``NativeFailsafeManager``.

        Idempotent: when no leg is in cancel-tentative for the parent
        (e.g. because both the event-driven path and the reconcile-retry
        path delivered the resolution on the same tick), the second
        call is a no-op.

        :param parent_from_entry: The parent ``EntryIntent.intent_key``
            whose tentative legs are being restored.
        :param parent_filled: When ``True``, restore to ``armed``
            (the parent is filled, the leg should resume worst-SL
            contribution and tick processing). When ``False``,
            restore to ``pending_entry`` (parent still pending; the
            next :meth:`on_parent_entry_filled` will promote).
        :param reason: Audit reason recorded on the leg row's extras.
        :return: List of legs that were transitioned. Empty if no
            tentative leg matched the parent (idempotent no-op).
        """
        target_state = (
            LEG_STATE_ARMED if parent_filled else LEG_STATE_PENDING_ENTRY
        )
        restored: list[PartialBracketLeg] = []
        targets = [
            leg for leg in self._legs.values()
            if leg.from_entry == parent_from_entry
            and leg.leg_state == LEG_STATE_CANCEL_TENTATIVE
        ]
        for leg in targets:
            self._transition(
                leg, target_state,
                close_row=False,
                extras_patch={
                    EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS: None,
                    'cancel_tentative_resolved_reason': reason,
                },
            )
            restored.append(leg)
        return restored

    def confirm_cancel_tentative(
            self,
            parent_from_entry: str,
            *,
            reason: str,
    ) -> list[PartialBracketLeg]:
        """Resolve :meth:`mark_legs_cancel_tentative` forward for one parent.

        Called when the cancel-retry-loop receives a
        :attr:`CancelDispositionOutcome.CANCEL_CONFIRMED`,
        :attr:`CancelDispositionOutcome.STILL_OPEN`, or
        :attr:`CancelDispositionOutcome.TOO_LATE_TO_CANCEL` outcome,
        or when a parent CANCELLED order event arrives for a parent
        currently in cancel-tentative. The tentative legs are flipped
        to :data:`LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED` (terminal)
        and their rows are closed.

        Idempotent: if no tentative leg matches the parent (because
        a previous call already terminated them), this is a no-op.

        :param parent_from_entry: The parent ``EntryIntent.intent_key``
            whose tentative legs are being confirmed-cancelled.
        :param reason: Audit reason recorded on the leg row's extras.
        :return: List of legs that were transitioned. Empty if no
            tentative leg matched (idempotent no-op).
        """
        confirmed: list[PartialBracketLeg] = []
        targets = [
            leg for leg in self._legs.values()
            if leg.from_entry == parent_from_entry
            and leg.leg_state == LEG_STATE_CANCEL_TENTATIVE
        ]
        for leg in targets:
            self._transition(
                leg, LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED,
                close_row=True,
                extras_patch={
                    'abort_reason': reason,
                    EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS: None,
                },
            )
            confirmed.append(leg)
        return confirmed

    def iter_cancel_tentative_parents(self) -> set[str]:
        """Snapshot of every distinct parent ``from_entry`` that currently
        has at least one leg in :data:`LEG_STATE_CANCEL_TENTATIVE`.

        Used by the sync engine's ``reconcile()`` cancel-retry-loop to
        decide which parents still need disposition resolution. The
        result is a set (not a list) so the caller can intersect /
        difference against its shadow map without worrying about leg-
        trio multiplicity.
        """
        return {
            leg.from_entry for leg in self._legs.values()
            if leg.leg_state == LEG_STATE_CANCEL_TENTATIVE
        }

    def has_cancel_tentative_legs(self, parent_from_entry: str) -> bool:
        """Whether any leg under ``parent_from_entry`` is in
        :data:`LEG_STATE_CANCEL_TENTATIVE`.

        Quick check used by the sync engine's diff-loop refuse-and-defer
        guard before the adoption branch — symmetric to
        :meth:`has_active_legs_for_intent` but for the verification state
        and parent-scoped (not intent-scoped).
        """
        for leg in self._legs.values():
            if leg.from_entry == parent_from_entry \
                    and leg.leg_state == LEG_STATE_CANCEL_TENTATIVE:
                return True
        return False

    # === Restart replay ===================================================

    def restart_replay(self) -> None:
        """Rebuild the in-memory ledger from persisted leg rows.

        Called once by the sync engine during startup, after the
        journal's regular replay but before the first :meth:`sync`.
        Terminal rows are filtered out by
        :func:`iter_active_engine_trigger_partial_legs`; active rows
        in ``triggering`` / ``triggered_failed`` / ``triggered_unknown``
        are reloaded as-is and the next :meth:`on_price_tick` re-
        evaluates them (the sync engine's reconciliation path
        decides whether the prior trigger landed or needs retry).

        Stale-price recovery (§3.5): when this method observes an
        ``armed`` leg whose ``trigger_level`` has already been crossed
        by the current spot quote, it does NOT fire on the spot —
        firing belongs to :meth:`on_price_tick` once the sync engine
        supplies a fresh parent snapshot. The stale-trigger policy
        choice (conservative / strict, see §12 #3) lives in the sync
        engine, not here.

        In-flight state recovery: legs persisted in
        :data:`LEG_STATE_TRIGGERING`,
        :data:`LEG_STATE_TRIGGERED_FAILED` or
        :data:`LEG_STATE_TRIGGERED_UNKNOWN` are intermediate — the
        process crashed between trigger detection and the close
        dispatch settling. :meth:`on_price_tick` only advances
        ``armed`` legs, and no other code path acts on those
        intermediate states, so they would remain stuck across the
        restart. Demote them back to ``armed`` here so the next tick
        re-evaluates the trigger against a fresh parent snapshot;
        if the prior close already landed at the broker, the parent
        size will have shrunk and the safety check will cap or abort
        the re-fire accordingly.
        """
        if self._store_ctx is None:
            return
        self._legs.clear()
        self._legs_by_oca_group.clear()
        self._legs_by_parent.clear()
        for row in iter_active_engine_trigger_partial_legs(self._store_ctx):
            leg = _leg_from_row(row)
            if leg is None:
                continue
            self._legs[leg.key] = leg
            if leg.oca_group is not None:
                self._legs_by_oca_group.setdefault(
                    leg.oca_group, set(),
                ).add(leg.key)
            self._legs_by_parent.setdefault(
                (leg.symbol, leg.from_entry), set(),
            ).add(leg.key)
            if leg.leg_state in (
                    LEG_STATE_TRIGGERING,
                    LEG_STATE_TRIGGERED_FAILED,
                    LEG_STATE_TRIGGERED_UNKNOWN,
            ):
                self._transition(
                    leg, LEG_STATE_ARMED,
                    close_row=False,
                    extras_patch={'rearmed_after_restart_from': leg.leg_state},
                )

    # === State machine plumbing ==========================================

    def _transition(
            self,
            leg: PartialBracketLeg,
            new_state: str,
            *,
            close_row: bool,
            extras_patch: dict | None = None,
            qty: float | None = None,
    ) -> None:
        old_state = leg.leg_state
        leg.leg_state = new_state
        if qty is not None:
            leg.qty = qty
        if extras_patch:
            leg.extras.update(extras_patch)
        if self._store_ctx is not None:
            update_engine_trigger_partial_leg_state(
                self._store_ctx,
                coid=leg.coid,
                new_leg_state=new_state,
                qty=qty,
                extras_patch=extras_patch,
                close_row=close_row,
            )
        if self._state_change_listener is not None:
            self._state_change_listener(leg, old_state, new_state)
        if close_row:
            self._evict(leg.key)

    def _evict(self, key: LegKey) -> None:
        leg = self._legs.pop(key, None)
        if leg is None:
            return
        if leg.oca_group is not None:
            group = self._legs_by_oca_group.get(leg.oca_group)
            if group is not None:
                group.discard(key)
                if not group:
                    self._legs_by_oca_group.pop(leg.oca_group, None)
        parent_key = (leg.symbol, leg.from_entry)
        parent_group = self._legs_by_parent.get(parent_key)
        if parent_group is not None:
            parent_group.discard(key)
            if not parent_group:
                self._legs_by_parent.pop(parent_key, None)


# === Helpers ==============================================================

class TickOffsetResolver:
    """Convert a leg's ``trigger_offset`` (price units) into an absolute
    price level at parent-fill time.

    Slice A keeps the resolver protocol thin — the sync engine
    converts raw tick fields (``profit_ticks`` / ``loss_ticks`` /
    ``trail_points_ticks``) to price units at dispatch time before
    storing them in ``trigger_offset``, so the resolver itself just
    applies the offset to the fill price with the correct sign for
    the leg kind. The ``mintick`` parameter is kept on the resolver
    for forward-compat with a future call site that does the
    conversion here.
    """

    def __init__(self, mintick: float) -> None:
        self._mintick = mintick

    # noinspection PyMethodMayBeStatic
    def resolve(
            self,
            leg: PartialBracketLeg,
            *,
            fill_price: float,
            parent_sign: int,
    ) -> float | None:
        if leg.trigger_offset is None:
            return None
        if leg.leg_kind == LEG_KIND_TP_PARTIAL:
            return fill_price + parent_sign * leg.trigger_offset
        if leg.leg_kind == LEG_KIND_SL_PARTIAL:
            return fill_price - parent_sign * leg.trigger_offset
        if leg.leg_kind == LEG_KIND_TRAIL_PARTIAL:
            return fill_price - parent_sign * leg.trigger_offset
        return None


def _leg_from_row(row: 'OrderRow') -> PartialBracketLeg | None:
    extras = row.extras or {}
    leg_kind = extras.get(EXTRAS_KEY_LEG_KIND, '')
    leg_state = extras.get(EXTRAS_KEY_LEG_STATE, '')
    if leg_kind not in (
            LEG_KIND_TP_PARTIAL, LEG_KIND_SL_PARTIAL, LEG_KIND_TRAIL_PARTIAL,
    ):
        return None
    # Accept every live state, including ``cancel_tentative``. The
    # tentative row is still open (``closed_ts_ms IS NULL``) and the
    # sync engine's cancel-retry loop drives its next transition — the
    # in-memory ledger must therefore carry it so
    # :meth:`SoftwarePartialBracketEngine.iter_cancel_tentative_parents`
    # and the post-restart
    # :meth:`OrderSyncEngine._rehydrate_cancel_tentative_from_replayed_legs`
    # rehydrate can see it. The ``restart_replay`` only re-arms the three
    # in-flight engine-owned states (``triggering`` / ``triggered_failed``
    # / ``triggered_unknown``), so a tentative row loaded here stays
    # tentative until the cancel-retry loop resolves it.
    if leg_state not in LEG_STATE_LIVE:
        return None
    parent_pine_entry_id = extras.get(EXTRAS_KEY_PARENT_PINE_ENTRY_ID, '')
    parent_entry_dispatch_ref = extras.get(
        EXTRAS_KEY_PARENT_ENTRY_DISPATCH_REF, '',
    )
    intent_partial_qty = float(extras.get(EXTRAS_KEY_INTENT_PARTIAL_QTY, 0.0))
    trigger_level = extras.get(EXTRAS_KEY_TRIGGER_LEVEL)
    trigger_offset = extras.get(EXTRAS_KEY_TRIGGER_OFFSET)
    trail_activation_level = extras.get(EXTRAS_KEY_TRAIL_ACTIVATION_LEVEL)
    trail_activation_offset = extras.get(EXTRAS_KEY_TRAIL_ACTIVATION_OFFSET)
    pine_id = row.pine_entry_id or ''
    from_entry = row.from_entry or ''
    return PartialBracketLeg(
        coid=row.client_order_id,
        symbol=row.symbol,
        pine_id=pine_id,
        from_entry=from_entry,
        leg_kind=leg_kind,
        leg_state=leg_state,
        side=row.side,
        qty=row.qty,
        intent_key=row.intent_key or '',
        parent_pine_entry_id=parent_pine_entry_id,
        parent_entry_dispatch_ref=parent_entry_dispatch_ref,
        intent_partial_qty=intent_partial_qty,
        trigger_level=trigger_level,
        trigger_offset=trigger_offset,
        trail_activation_level=trail_activation_level,
        trail_activation_offset=trail_activation_offset,
        oca_group=extras.get(EXTRAS_KEY_OCA_GROUP),
        oca_type=extras.get(EXTRAS_KEY_OCA_TYPE),
        extras=dict(extras),
    )
