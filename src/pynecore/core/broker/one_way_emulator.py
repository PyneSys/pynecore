"""One-way position emulation engine over a hedging-mode broker account.

A hedging account holds several open positions ("legs") per symbol; a Pine
strategy sees a single *one-way* position. The pure helpers in
:mod:`~pynecore.core.broker.emulator` decide *what* to do (net the legs, pick a
FIFO close plan, snap per-leg volumes); this engine *drives* it: it fans a
:class:`~pynecore.core.broker.models.CloseIntent` out across the legs through a
thin :class:`~pynecore.core.plugin.broker.PositionPort`, and owns the
persist-first ledger that lets an interrupted fan-out resume on restart without
re-closing a leg that already settled.

It is the close/reversal counterpart of
:class:`~pynecore.core.broker.software_entry_stop_engine.SoftwareEntryStopEngine`
and :class:`~pynecore.core.broker.software_partial_bracket_engine.SoftwarePartialBracketEngine`:
one instance per :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine`, the
persisted close-leg rows in :mod:`store_helpers` are the durable representation
(PERSIST-FIRST), and :meth:`restart_replay` re-derives an unfinished fan-out
from them. The legs' close FILLs themselves flow through the engine's ordinary
natural-close path (mapped to ``LegType.CLOSE`` by the plugin), so this engine
never touches the fill router — only the dispatch + its crash/replay.

The engine carries no broker, clock, or exchange-model state. Each broker action
goes through the :class:`PositionPort`; aggregation / selection / quantization
are the pure functions in :mod:`emulator`. That keeps it deterministic and
unit-testable in isolation, with ``store_ctx=None`` for the no-persistence path.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pynecore.core.broker.emulator import (
    LegClose,
    aggregate_positions,
    net_survivor_legs,
    plan_leg_close_volumes,
    plan_reversal,
    select_legs_for_close,
)
from pynecore.core.broker.exceptions import (
    BracketAttachAfterFillRejectedError,
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    OrderDispositionUnknownError,
)
from pynecore.core.broker.idempotency import (
    KIND_CANCEL,
    KIND_CLOSE,
    KIND_ENTRY,
    KIND_EXIT_SL,
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
from pynecore.core.broker.store_helpers import (
    BRACKET_OWN_STATE_CLEARING,
    BRACKET_OWN_STATE_RELEASED,
    CLOSE_LEG_STATE_DISPATCHED,
    EXTRAS_KEY_BRACKET_OWN_ATTACH_COID,
    EXTRAS_KEY_BRACKET_OWN_CLEAR_COID,
    EXTRAS_KEY_BRACKET_OWN_LEG_ID,
    EXTRAS_KEY_BRACKET_OWN_SL,
    EXTRAS_KEY_BRACKET_OWN_STATE,
    EXTRAS_KEY_BRACKET_OWN_TP,
    EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET,
    EXTRAS_KEY_CLOSE_LEG_ID,
    EXTRAS_KEY_CLOSE_LEG_VOLUME,
    EXTRAS_KEY_RESIDUAL_OPEN_BAR_TS_MS,
    EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID,
    EXTRAS_KEY_RESIDUAL_OPEN_RETRY_SEQ,
    EXTRAS_KEY_RESIDUAL_OPEN_RUN_TAG,
    clear_residual_open_row,
    create_bracket_ownership_row,
    create_close_leg_row,
    create_residual_open_row,
    iter_active_bracket_ownerships,
    iter_active_close_legs,
    iter_active_residual_opens,
    update_bracket_ownership_state,
    update_close_leg_state,
)

if TYPE_CHECKING:
    from pynecore.core.broker.storage import RunContext
    from pynecore.core.plugin.broker import PositionPort


__all__ = [
    'CloseFanResult',
    'ReversalFanResult',
    'BracketFanResult',
    'ClearFanResult',
    'OneWayEmulator',
]


@dataclass(frozen=True)
class CloseFanResult:
    """Outcome of one fanned-out close, for the caller's return shaping / audit.

    :ivar legs: ``(leg_id, broker-grid volume)`` pairs actually dispatched, in
        FIFO order. Empty when nothing was open or the close was skipped.
    :ivar shortfall: Pine-unit quantity the open legs could not cover (the
        broker holds less than Pine believes). ``0.0`` in the normal case; a
        positive value is the caller's signal to log, not halt.
    :ivar skipped: ``True`` when the whole close quantized below the broker's
        volume grid, so no order was sent (a non-halting skip — the engine
        re-evaluates next bar). Distinct from an empty ``legs`` with
        ``skipped=False``, which means the symbol was already flat.
    """
    legs: tuple[tuple[str, int], ...]
    shortfall: float
    skipped: bool


@dataclass(frozen=True)
class ReversalFanResult:
    """Outcome of one decomposed reversal / add, for the caller's return shaping.

    :ivar closes: ``(leg_id, broker-grid volume)`` pairs FIFO-closed to retire
        the opposing exposure. Empty for a pure add (no opposing legs).
    :ivar open_qty: Pine-unit size opened in the order's own direction after the
        closes. ``0.0`` for an exact flatten (close all, open nothing).
    :ivar opened_orders: The ``ExchangeOrder``(s) the residual open produced.
        Empty when ``open_qty`` was zero.
    """
    closes: tuple[tuple[str, int], ...]
    open_qty: float
    opened_orders: tuple


@dataclass(frozen=True)
class BracketFanResult:
    """Outcome of one exit-bracket replication across a position's legs.

    :ivar legs: Broker leg ids the bracket was replicated onto, in FIFO order.
    :ivar skipped: ``True`` when the symbol was flat (no position-side legs to
        protect) — a non-halting no-op; the caller re-evaluates next bar.
    """
    legs: tuple[str, ...]
    skipped: bool


@dataclass(frozen=True)
class ClearFanResult:
    """Outcome of one ownership-scoped exit-bracket clear.

    :ivar legs: Broker leg ids actually cleared — the subset the cancelled exit
        owns, never the whole position side. Empty when the exit owned nothing
        (already released, or no persisted ownership index).
    """
    legs: tuple[str, ...]


class OneWayEmulator:
    """Drives Pine one-way close/reversal over a hedging account's legs.

    One instance per :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine`.
    The engine owns the persisted close-leg ledger and its restart replay; every
    broker-side action is performed through the :class:`PositionPort` the
    emulating plugin exposes.
    """

    def __init__(self, store_ctx: 'RunContext | None') -> None:
        self._store_ctx = store_ctx

    # === Close fan-out ====================================================

    async def run_close(
            self, envelope: 'DispatchEnvelope', port: 'PositionPort',
    ) -> CloseFanResult:
        """Fan a :class:`CloseIntent` out as per-leg FIFO closes.

        Reads the live legs, nets them, selects the oldest legs on the position
        side to cover ``intent.qty``, snaps the plan to the broker grid, and
        dispatches one :meth:`PositionPort.close_leg` per leg — persisting each
        leg PERSIST-FIRST so a crash mid-fan resumes via :meth:`restart_replay`.

        Never raises for an undersized close: a whole-close-below-grid returns
        ``skipped=True`` and a broker-holds-less returns a positive
        ``shortfall`` — the bot keeps running and the caller decides how to
        surface it.
        """
        intent = envelope.intent
        assert isinstance(intent, CloseIntent)
        legs = await port.fetch_raw_positions(intent.symbol)
        pos = aggregate_positions(intent.symbol, legs)
        if pos is None or pos.side == 'flat':
            # Nothing open, or offsetting legs net to flat — benign no-op.
            return CloseFanResult(legs=(), shortfall=0.0, skipped=False)
        # CloseIntent.side is the close direction ("sell" closes a long); the
        # legs to reduce sit on the opposite side of the book.
        leg_side = 'buy' if intent.side == 'sell' else 'sell'
        closes, shortfall = select_legs_for_close(intent.qty, legs, leg_side)
        if not closes:
            return CloseFanResult(legs=(), shortfall=shortfall, skipped=False)
        parent_coid = envelope.client_order_id(KIND_CLOSE)
        dispatched = await self._fan_out_closes(
            closes, symbol=intent.symbol, side=intent.side,
            intent_key=intent.intent_key, pine_id=intent.pine_id,
            parent_coid=parent_coid, port=port,
        )
        if not dispatched:
            # Every slice rounded below the broker grid — skip, do not halt.
            return CloseFanResult(legs=(), shortfall=shortfall, skipped=True)
        return CloseFanResult(
            legs=tuple(dispatched), shortfall=shortfall, skipped=False,
        )

    async def run_reversal(
            self, envelope: 'DispatchEnvelope', port: 'PositionPort',
    ) -> ReversalFanResult:
        """Decompose a Pine reversal/add entry over the hedging legs.

        Pine folds a reversal into one combined-size :class:`EntryIntent`
        assuming a netting auto-flip. On a hedging account that single order
        would open a separate opposing leg, so it is split: FIFO-close the
        opposing legs (up to the order size) and open only the residual in the
        order's own direction. Pure add (no opposing legs) opens the whole size;
        exact flatten (opposing exposure == order size) closes all and opens
        nothing.
        """
        intent = envelope.intent
        assert isinstance(intent, EntryIntent)
        if intent.order_type is not OrderType.MARKET:
            # A resting LIMIT / STOP entry is not Pine's combined-size market
            # auto-flip: it rests as a fresh working order and is decomposed
            # (if it is still a reversal when it triggers) only at FILL time.
            # Place it full-size now with NO opposing-leg close — retiring the
            # opposing exposure here would flatten the position before the entry
            # ever fires. (Fill-time decomposition of a resting reversal is a
            # follow-up.) ``place_leg`` runs the broker's own volume-bounds
            # pre-flight, so an out-of-range resting order still skips cleanly.
            opened = tuple(await port.place_leg(envelope, intent.qty))
            return ReversalFanResult(
                closes=(), open_qty=intent.qty, opened_orders=opened,
            )
        legs = await port.fetch_raw_positions(intent.symbol)
        plan = plan_reversal(intent.side, intent.qty, legs)
        # Pre-flight the broker volume bounds BEFORE any close lands: an
        # out-of-range order must raise the non-halting skip while it is still
        # true, otherwise the closes reduce the book yet the whole reversal is
        # reported skipped, desyncing the engine. A pure reduce (open_qty == 0)
        # validates the combined order size; a real reversal validates its
        # residual leg.
        preflight_qty = plan.open_qty if plan.open_qty > 0.0 else intent.qty
        await port.reject_out_of_range(envelope, preflight_qty)
        parent_coid = envelope.client_order_id(KIND_CLOSE)
        # Persist-first residual-open breadcrumb BEFORE the closes: a crash
        # after the closes land but before ``place_leg`` persists the residual
        # entry row would otherwise leave the book reduced with the residual
        # open lost — neither the close-leg replay nor the entry journal owns
        # it. ``restart_replay`` reconciles this against the residual entry row
        # and re-dispatches only if that row never landed. Only a genuine
        # reversal needs it: a pure add (no opposing legs, ``plan.closes``
        # empty) leaves the book intact, so re-opening it on restart would
        # bypass the next sync's Pine re-evaluation and resurrect an entry the
        # strategy may no longer want. The breadcrumb is gated on closes being
        # actually owed, never on a positive ``open_qty`` alone.
        residual_coid = f"{parent_coid}:residual"
        wrote_breadcrumb = bool(plan.closes) and plan.open_qty > 0.0
        if wrote_breadcrumb and self._store_ctx is not None:
            create_residual_open_row(
                self._store_ctx,
                coid=residual_coid,
                symbol=intent.symbol,
                side=intent.side,
                qty=plan.open_qty,
                intent_key=intent.intent_key,
                pine_entry_id=intent.pine_id,
                entry_coid=envelope.client_order_id(KIND_ENTRY),
                run_tag=envelope.run_tag,
                bar_ts_ms=envelope.bar_ts_ms,
                retry_seq=envelope.retry_seq,
            )
        dispatched: list[tuple[str, int]] = []
        if plan.closes:
            try:
                dispatched = await self._fan_out_closes(
                    plan.closes, symbol=intent.symbol, side=intent.side,
                    intent_key=intent.intent_key, pine_id=intent.pine_id,
                    parent_coid=parent_coid, port=port,
                )
            except ExchangeOrderRejectedError:
                # A close leg was DEFINITIVELY refused — the fan-out raised
                # before the residual ``place_leg`` ran, so the residual never
                # opened. ``_dispatch_new`` turns this reject into a non-halting
                # skip for the entry (signal dropped, bot continues), so the
                # breadcrumb must be discharged here exactly as the residual
                # reject path below does: leaving it live would let a later
                # restart replay re-close + re-open a reversal the exchange
                # already rejected and Pine never re-signalled. An ambiguous
                # ``OrderDispositionUnknownError`` (caught by the broader
                # ``BrokerError`` contract, not here) leaves the close's fate —
                # and thus a possibly-owed residual — unknown, so its breadcrumb
                # must survive for restart reconciliation and is left intact.
                if wrote_breadcrumb and self._store_ctx is not None:
                    clear_residual_open_row(self._store_ctx, residual_coid)
                raise
        opened_orders: tuple = ()
        if plan.open_qty > 0.0:
            try:
                opened_orders = tuple(await port.place_leg(envelope, plan.open_qty))
            except ExchangeOrderRejectedError:
                # The exchange definitively refused the residual entry — nothing
                # opened. The dispatch path turns this into a non-halting skip
                # (signal dropped, bot continues), so the breadcrumb must be
                # discharged here: leaving it live would let a later restart
                # replay re-open a residual the exchange already rejected and Pine
                # never re-signalled. Only a DEFINITIVE reject clears it — an
                # ambiguous ``OrderDispositionUnknownError`` / connection error
                # leaves the open's fate unknown, so its breadcrumb must survive
                # for restart reconciliation and is left intact (re-raised).
                if wrote_breadcrumb and self._store_ctx is not None:
                    clear_residual_open_row(self._store_ctx, residual_coid)
                raise
            if wrote_breadcrumb and self._store_ctx is not None:
                # The residual entry row is now persisted (persist-first inside
                # ``place_leg``) and dispatched — the breadcrumb is discharged.
                clear_residual_open_row(self._store_ctx, residual_coid)
        return ReversalFanResult(
            closes=tuple(dispatched), open_qty=plan.open_qty,
            opened_orders=opened_orders,
        )

    async def _fan_out_closes(
            self, closes: tuple[LegClose, ...], *,
            symbol: str, side: str, intent_key: str, pine_id: str,
            parent_coid: str, port: 'PositionPort',
    ) -> list[tuple[str, int]]:
        """Snap a FIFO close plan to the broker grid and dispatch it per leg.

        Shared by :meth:`run_close` and :meth:`run_reversal`. Each leg is
        PERSIST-FIRST: its row (coid ``f"{parent_coid}:{leg_id}"``, so a restart
        re-dispatch upserts the SAME row) is written ``pending`` BEFORE the wire
        call and finalised ``dispatched`` only after
        :meth:`PositionPort.close_leg` returns. Returns the
        ``(leg_id, broker-grid volume)`` pairs actually sent (empty when the
        whole plan rounded below the grid).
        """
        quantize = await port.get_volume_quantizer(symbol)
        dispatched = plan_leg_close_volumes(closes, quantize)
        for leg_id, volume in dispatched:
            leg_coid = f"{parent_coid}:{leg_id}"
            if self._store_ctx is not None:
                create_close_leg_row(
                    self._store_ctx,
                    coid=leg_coid,
                    symbol=symbol,
                    side=side,
                    qty=float(volume),
                    intent_key=intent_key,
                    pine_entry_id=pine_id,
                    parent_close_coid=parent_coid,
                    leg_id=leg_id,
                    leg_volume=volume,
                )
            await port.close_leg(symbol, leg_id, volume, leg_coid)
            if self._store_ctx is not None:
                update_close_leg_state(
                    self._store_ctx, coid=leg_coid,
                    new_state=CLOSE_LEG_STATE_DISPATCHED, close_row=True,
                )
        return dispatched

    # === Exit-bracket replication =========================================

    async def run_exit_bracket(
            self, envelope: 'DispatchEnvelope', port: 'PositionPort',
    ) -> BracketFanResult:
        """Replicate a Pine exit's bracket onto the net-survivor position legs.

        Hedging venues carry protective levels per position, so a one-way bracket
        over a multi-leg position is delivered by amending the SAME
        TP/SL/trailing onto each leg that makes up the net one-way position. On a
        mixed book (a manual hedge or a crash-interrupted reversal left opposing
        legs open) only the legs that survive virtual-FIFO netting are protected
        (:func:`net_survivor_legs`), never the gross majority side — amending the
        whole majority side would close more than the net position when the stop
        fires and flip it to the minority side. Each leg is recorded PERSIST-FIRST
        in the ownership index (keyed by the exit's ``intent_key``) BEFORE its
        amend, so :meth:`run_exit_bracket_clear` later clears ONLY the legs this
        exit owns and :meth:`restart_replay` can re-assert or release them. A flat
        symbol is a non-halting skip.

        Re-running it (a modify, or a re-attach after pyramiding) upserts the
        same per-leg rows idempotently: the row key is STABLE per (exit
        identity, leg) — derived from the exit's pine_id + from_entry + leg_id,
        NOT the bar-varying dispatch coid — so a modify on a later bar updates
        the same rows instead of accreting stale ones. Two exits sharing a
        pine_id but differing in from_entry get disjoint rows even on a shared
        leg (the dispatch coid alone would collide — it encodes pine_id but not
        from_entry). Legs that have since closed are released by the restart /
        reconcile pass, not here.
        """
        intent = envelope.intent
        assert isinstance(intent, ExitIntent)
        legs = await port.fetch_raw_positions(intent.symbol)
        side, on_side = net_survivor_legs(legs)
        if side == 'flat' or not on_side:
            return BracketFanResult(legs=(), skipped=True)
        survivors = {leg.leg_id for leg in on_side}
        # A re-attach can narrow the survivor set: a manual hedge or an
        # interrupted reversal opens an opposing leg that virtually FIFO-consumes
        # the oldest majority leg, so :func:`net_survivor_legs` now drops a leg
        # this exit owned on a prior attach. That leg is still PHYSICALLY open
        # (only virtually netted) and still carries the broker bracket from the
        # earlier attach; its ownership row would otherwise stay ``active`` and the
        # restart pass would re-assert the stale stop, which on a hit closes more
        # than the net exposure and flips the book. Clear + release those dropped
        # legs FIRST: a survivor amend below can raise
        # :class:`OrderDispositionUnknownError`, which the dispatch path parks
        # while promoting the NEW intent into ``_active_intents`` — the next diff
        # then sees Pine == active and never re-runs this fan, so a dropped-leg
        # clear left AFTER the survivor loop would never get retried. The dropped
        # set is disjoint from ``survivors`` (the clear skips survivor rows), so
        # ordering it before the amend loop is otherwise behaviour-neutral.
        await self._clear_dropped_survivor_legs(
            envelope, intent, survivors=survivors, port=port,
        )
        # Bar-varying dispatch coid (matches the plugin's per-leg amend anchor);
        # one per bracket attach, reused across its legs.
        attach_coid = envelope.client_order_id(KIND_EXIT_SL)
        replicated: list[str] = []
        for leg in on_side:
            own_coid = self._ownership_coid(intent.intent_key, leg.leg_id)
            if self._store_ctx is not None:
                create_bracket_ownership_row(
                    self._store_ctx,
                    coid=own_coid,
                    symbol=intent.symbol,
                    side=intent.side,
                    qty=intent.qty,
                    intent_key=intent.intent_key,
                    pine_entry_id=intent.pine_id,
                    from_entry=intent.from_entry,
                    leg_id=leg.leg_id,
                    attach_coid=attach_coid,
                    tp_price=intent.tp_price,
                    sl_price=intent.sl_price,
                    trail_price=intent.trail_price,
                    trail_offset=intent.trail_offset,
                )
            try:
                await port.amend_bracket(
                    intent.symbol, leg.leg_id, side=intent.side,
                    tp_price=intent.tp_price, sl_price=intent.sl_price,
                    trail_offset=intent.trail_offset, coid=attach_coid,
                )
            except ExchangeOrderRejectedError as exc:
                # The amend was DEFINITIVELY rejected on a leg of an OPEN
                # position (we hold ``on_side`` legs), so the position is now
                # open and UNPROTECTED. PERSIST-FIRST wrote this leg's ownership
                # row before the call, but a reject means we are still alive (not
                # a crash), so release that one never-attached row synchronously
                # — leaving it active would make a later ownership-scoped clear
                # or restart replay believe this exit protects a leg it does not.
                # The legs already amended earlier in this fan keep their active
                # rows: those brackets DO exist, and the defensive close the
                # engine issues for the wrapped error flattens the whole
                # position. (A leg that vanished mid-fan surfaces NOT as a reject
                # but as the port's benign no-op, so it never reaches here; an
                # ambiguous timeout is intentionally not caught — the amend may
                # have landed, so its row stays active for replay / reconcile.)
                if self._store_ctx is not None:
                    update_bracket_ownership_state(
                        self._store_ctx, coid=own_coid,
                        new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
                    )
                raise BracketAttachAfterFillRejectedError(
                    f"one-way bracket attach rejected after fill "
                    f"(leg={leg.leg_id}, from_entry={intent.from_entry!r}): {exc}",
                    position_coid=f"__pyne_orphan__{intent.symbol}__{intent.from_entry}",
                    symbol=intent.symbol,
                    position_side=('buy' if intent.side == 'sell' else 'sell'),
                    qty=intent.qty,
                    position_deal_id=leg.leg_id,
                    from_entry=intent.from_entry,
                    exit_id=intent.pine_id,
                ) from exc
            replicated.append(leg.leg_id)
        return BracketFanResult(legs=tuple(replicated), skipped=False)

    async def _clear_dropped_survivor_legs(
            self, envelope: 'DispatchEnvelope', intent: ExitIntent, *,
            survivors: set[str], port: 'PositionPort',
    ) -> None:
        """Clear + release this exit's owned legs that fell out of the survivor set.

        On a re-attach the net survivor set can shrink (an opposing leg now
        virtually FIFO-consumes a leg this exit protected before). Each such leg is
        still physically open and still carries the prior broker bracket, so its
        active ownership row is cleared on the broker (amend-to-None) under a clear
        coid distinct from the attach coid, then released — mirroring
        :meth:`run_exit_bracket_clear` for the implicit drop. A ``*_NOT_FOUND``
        race on a vanished leg is the port's benign no-op; the restart pass would
        otherwise re-assert the stale stop.

        An ambiguous clear (:class:`OrderDispositionUnknownError`) on a dropped
        leg must NOT abort this call: :meth:`run_exit_bracket` runs the drop FIRST,
        before the survivor amend loop, so propagating here would skip the NEW
        protection on the surviving legs entirely — the dispatch path then parks
        the error while promoting the new intent into ``_active_intents``, so the
        next diff sees Pine == active, never re-runs the fan, and the survivors
        stay on STALE TP/SL indefinitely. The drop is persist-first ``clearing``,
        so leaving its row in that phase lets :meth:`drain_clearing_rows` (per
        sync) and :meth:`restart_replay` retry the idempotent re-clear; the
        survivor amend below proceeds with the new levels. This mirrors the
        unknown-disposition handling in :meth:`drain_clearing_rows`.
        """
        if self._store_ctx is None:
            return
        clear_coid = envelope.client_order_id(KIND_CANCEL)
        rows = list(iter_active_bracket_ownerships(
            self._store_ctx, symbol=intent.symbol, from_entry=intent.from_entry,
        ))
        for row in rows:
            if (row.intent_key or '') != intent.intent_key:
                continue
            leg_id: str | None = (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_LEG_ID)
            if leg_id is None or leg_id in survivors:
                continue
            update_bracket_ownership_state(
                self._store_ctx, coid=row.client_order_id,
                new_state=BRACKET_OWN_STATE_CLEARING,
                extras_patch={EXTRAS_KEY_BRACKET_OWN_CLEAR_COID: clear_coid},
            )
            try:
                await port.amend_bracket(
                    intent.symbol, leg_id, side=row.side,
                    tp_price=None, sl_price=None, trail_offset=None,
                    coid=clear_coid,
                )
            except OrderDispositionUnknownError:
                # The clear did not confirm; leave the row ``clearing`` so
                # ``drain_clearing_rows`` / ``restart_replay`` retry the
                # idempotent re-clear, and DO NOT abort the survivor amend
                # below — those legs need their new protection now.
                continue
            update_bracket_ownership_state(
                self._store_ctx, coid=row.client_order_id,
                new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
            )

    async def drain_clearing_rows(self, symbol: str, port: 'PositionPort') -> set[str]:
        """Re-clear + release any ``clearing`` bracket-ownership rows in-session.

        :meth:`_clear_dropped_survivor_legs` (and :meth:`run_exit_bracket_clear`)
        mark a row ``clearing`` PERSIST-FIRST, then amend-to-None on the broker.
        An ambiguous :class:`OrderDispositionUnknownError` on that amend leaves the
        row stranded in ``clearing``: the dispatch path parks the exit and promotes
        it into ``_active_intents``, so the next diff sees Pine == active and never
        re-runs the owning fan, and the one-way orphan sweep skips active keys —
        the stale leg stays armed until the next restart's :meth:`restart_replay`.

        This drain closes that window. The engine calls it once per sync (after the
        orphan sweep): every live ``clearing`` row is re-cleared under its persisted
        clear coid and released, exactly like :meth:`_replay_bracket_one`'s clearing
        branch but per-sync rather than restart-only. A ``clearing`` row is by
        definition marked-for-removal, NEVER wanted protection, so re-clearing it is
        always safe even while the owning exit is still active (its survivor legs
        carry ``active`` rows, untouched here). A vanished leg's bracket is moot —
        release the row. Amending to None is idempotent (an already-cleared leg
        no-ops), so a re-clear that already landed is a broker no-op. A timeout on
        the re-clear leaves the row ``clearing`` for the next sync to retry rather
        than halting the bot.

        Returns the set of ``intent_key`` values whose LAST surviving ownership row
        the drain released this call. The engine drops the in-memory
        envelope/mapping for any such key whose owning exit Pine no longer emits —
        the orphan sweep's direct-clear timeout (``continue`` before
        ``_drop_envelope``) leaves that engine state behind, and the drain
        releasing the row here is the second half of the same retirement. Keys that
        still carry an ``active`` row (a live exit's survivor legs) are excluded, so
        a live bracket never loses its anchor.
        """
        if self._store_ctx is None:
            return set()
        rows = [
            row
            for row in iter_active_bracket_ownerships(self._store_ctx, symbol=symbol)
            if (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_STATE)
            == BRACKET_OWN_STATE_CLEARING
        ]
        if not rows:
            return set()
        legs = await port.fetch_raw_positions(symbol)
        live_ids = {leg.leg_id for leg in legs}
        released: set[str] = set()
        for row in rows:
            extras = row.extras or {}
            leg_id: str | None = extras.get(EXTRAS_KEY_BRACKET_OWN_LEG_ID)
            if leg_id is not None and leg_id in live_ids:
                try:
                    await port.amend_bracket(
                        symbol, leg_id, side=row.side,
                        tp_price=None, sl_price=None, trail_offset=None,
                        coid=extras.get(EXTRAS_KEY_BRACKET_OWN_CLEAR_COID)
                        or row.client_order_id,
                    )
                except OrderDispositionUnknownError:
                    # Re-clear did not confirm; leave the row ``clearing`` so the
                    # next sync retries it (idempotent amend-to-None). Halting here
                    # would strand the bot on a recoverable broker round-trip.
                    continue
                except ExchangeConnectionError:
                    # The connection dropped mid-drain. Earlier rows in this loop
                    # may already be RELEASED (durably closed in the store) and
                    # their keys collected in ``released``; if we let the exception
                    # escape, that partial progress is lost — the caller logs +
                    # retries but never runs the envelope/mapping retirement, so a
                    # released row's stale anchor survives and a later re-emission
                    # rebuilds the same attach coid an idempotent plugin dedups,
                    # leaving the leg unprotected. Stop draining (the link is down)
                    # but RETURN what was already released so the engine retires it;
                    # this row stays ``clearing`` for the next sync to retry.
                    break
            update_bracket_ownership_state(
                self._store_ctx, coid=row.client_order_id,
                new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
            )
            if row.intent_key is not None:
                released.add(row.intent_key)
        # Only report a key whose every ownership row is now gone: a key still
        # carrying an ``active`` row (a live exit's survivor legs) must keep its
        # envelope, so the engine must not retire it.
        if not released:
            return released
        still_owned = {
            row.intent_key
            for row in iter_active_bracket_ownerships(self._store_ctx, symbol=symbol)
            if row.intent_key is not None
        }
        return released - still_owned

    async def drain_residual_opens(self, symbol: str, port: 'PositionPort') -> None:
        """Reconcile any live reversal residual-open breadcrumb per sync (in-session).

        The restart counterpart :meth:`_replay_residual_opens` reconciles these
        breadcrumbs once at startup, but the engine latches that replay after the
        first sync (``_one_way_replay_done``). Two paths leave a breadcrumb live
        mid-session: :meth:`run_reversal`'s residual ``place_leg`` raising
        :class:`OrderDispositionUnknownError` (the open's fate unknown, so the
        breadcrumb is intentionally NOT discharged), and :meth:`_replay_residual_one`
        hitting the same on its own re-dispatch. Without this drain such a breadcrumb
        would wait for the NEXT process restart before being reconciled.

        This closes that window. The engine calls it once per sync (alongside
        :meth:`drain_clearing_rows`), reconciling every live breadcrumb for
        ``symbol`` through the SAME :meth:`_replay_residual_one` path restart replay
        uses: if the residual's persist-first entry row already landed the breadcrumb
        is simply discharged (the entry / recovery path owns the open), otherwise the
        residual is re-opened under its deterministic ``KIND_ENTRY`` coid — a
        duplicate is impossible at the exchange dedup, so a per-sync re-dispatch
        cannot double-open. A still-ambiguous re-dispatch leaves the row live for the
        next sync to retry rather than halting the bot, mirroring
        :meth:`drain_clearing_rows`. A successful in-session reversal discharges its
        own breadcrumb synchronously, so this only ever sees genuinely unresolved
        rows. Scoped to ``symbol`` (the engine owns one symbol); a breadcrumb for
        another symbol is that engine's drain to reconcile.
        """
        if self._store_ctx is None:
            return
        for row in list(iter_active_residual_opens(self._store_ctx, symbol=symbol)):
            await self._replay_residual_one(row, port)

    @staticmethod
    def _ownership_coid(intent_key: str, leg_id: str) -> str:
        """Stable per-(exit, leg) bracket-ownership row key.

        Keyed on the exit's ``intent_key`` (pine_id + from_entry) and the broker
        leg id, with no bar timestamp, so a modify re-attach upserts the SAME row
        and two exits sharing a pine_id but differing in from_entry never collide
        on a shared leg. Distinct from the bar-varying dispatch coid.
        """
        return f"bo:{intent_key}:{leg_id}"

    async def run_exit_bracket_clear(
            self, envelope: 'DispatchEnvelope', port: 'PositionPort',
    ) -> ClearFanResult:
        """Clear ONLY the legs the cancelled exit owns (the per-leg ownership fix).

        A broadcast clear amends every position-side leg, stripping a bracket a
        DIFFERENT exit set. This consults the persisted ownership index instead:
        it amends-to-clear only the legs whose row matches the cancel's
        ``intent_key`` (or, for a cancel-all-by-pine_id with ``from_entry`` None,
        every row whose exit shares that pine_id), then releases each cleared
        row. Without a persisted index (``store_ctx`` None) there is nothing to
        clear — a benign empty result.
        """
        intent = envelope.intent
        assert isinstance(intent, CancelIntent)
        if self._store_ctx is None:
            return ClearFanResult(legs=())
        cancel_coid = envelope.client_order_id(KIND_CANCEL)
        rows = list(iter_active_bracket_ownerships(
            self._store_ctx, symbol=intent.symbol, from_entry=intent.from_entry,
        ))
        cleared: list[str] = []
        for row in rows:
            if not self._owns(row, intent):
                continue
            leg_id: str | None = (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_LEG_ID)
            if leg_id is None:
                continue
            # PERSIST-FIRST: mark the row ``clearing`` BEFORE the amend, so an
            # ambiguous (timed-out) clear leaves a row that :meth:`restart_replay`
            # re-CLEARS rather than re-asserting the original bracket — the
            # close-leg ``pending`` -> ``dispatched`` two-phase applied to the
            # clear. Without it a timed-out clear stays ``active`` and the next
            # restart resurrects the bracket the script asked to cancel. Record the
            # clear coid too, so the restart re-clears under the SAME clear coid —
            # NOT the attach coid, which a coid-idempotent plugin could dedup as a
            # repeat of the attach and swallow, leaving the bracket armed.
            update_bracket_ownership_state(
                self._store_ctx, coid=row.client_order_id,
                new_state=BRACKET_OWN_STATE_CLEARING,
                extras_patch={EXTRAS_KEY_BRACKET_OWN_CLEAR_COID: cancel_coid},
            )
            await port.amend_bracket(
                intent.symbol, leg_id, side=row.side,
                tp_price=None, sl_price=None, trail_offset=None,
                coid=cancel_coid,
            )
            update_bracket_ownership_state(
                self._store_ctx, coid=row.client_order_id,
                new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
            )
            cleared.append(leg_id)
        return ClearFanResult(legs=tuple(cleared))

    @staticmethod
    def _owns(row, intent: 'CancelIntent') -> bool:
        """Does ``row`` belong to the exit ``intent`` cancels?

        Exact ``intent_key`` match when the cancel targets one (pine_id,
        from_entry); an ``f"{pine_id}\\0"`` prefix match for a cancel-all
        (``from_entry`` None) that drops every exit sharing the pine_id.
        """
        row_key = row.intent_key or ''
        if intent.from_entry is not None:
            return row_key == intent.intent_key
        return row_key.startswith(f"{intent.pine_id}\0")

    # === Restart replay ===================================================

    async def restart_replay(self, port: 'PositionPort') -> None:
        """Resume any close-leg fan-out or bracket replication a crash interrupted.

        Called once at startup. Three ordered passes: pending close-leg rows
        are reconciled against the live legs and the residual re-dispatched (an
        already-settled leg is never re-closed); THEN reversal residual-open
        breadcrumbs are reconciled (the closes are now resolved, so a still-owed
        residual is safely re-opened) and the open re-dispatched only when its
        own entry row never landed; THEN active bracket-ownership rows are
        re-asserted on still-open legs (idempotent on the per-leg coid) and
        released when their leg has vanished.
        """
        if self._store_ctx is None:
            return
        await self._replay_close_legs(port)
        await self._replay_residual_opens(port)
        await self._replay_bracket_ownership(port)

    async def _replay_close_legs(self, port: 'PositionPort') -> None:
        """Reconcile + re-dispatch any pending close-leg rows (first replay pass).

        For every persisted ``pending`` close-leg row, reconcile against the live
        broker legs before re-sending: if the leg is gone, the close already
        landed — finalise the row without dispatching; if it is still open,
        re-dispatch only the residual (capped at the live leg size, re-snapped to
        the grid) so an already-partly-closed leg is never over-reduced. The
        natural-close fill dedup guards the fill side.
        """
        if self._store_ctx is None:
            return
        pending = list(iter_active_close_legs(self._store_ctx))
        if not pending:
            return
        by_symbol: dict[str, list] = {}
        for row in pending:
            by_symbol.setdefault(row.symbol, []).append(row)
        for symbol, rows in by_symbol.items():
            legs = await port.fetch_raw_positions(symbol)
            live_by_id = {leg.leg_id: leg for leg in legs}
            quantize = await port.get_volume_quantizer(symbol)
            for row in rows:
                await self._replay_one(row, symbol, live_by_id, quantize, port)

    async def _replay_one(
            self, row, symbol: str, live_by_id: dict[str, PositionLeg], quantize, port: 'PositionPort',
    ) -> None:
        """Reconcile + (if needed) re-dispatch one pending close-leg row."""
        if self._store_ctx is None:
            return
        extras = row.extras or {}
        leg_id: str | None = extras.get(EXTRAS_KEY_CLOSE_LEG_ID)
        live_leg = live_by_id.get(leg_id) if leg_id is not None else None
        if live_leg is None or leg_id is None:
            # Leg vanished — the close landed before the crash. Finalise only.
            update_close_leg_state(
                self._store_ctx, coid=row.client_order_id,
                new_state=CLOSE_LEG_STATE_DISPATCHED, close_row=True,
            )
            return
        persisted = extras.get(EXTRAS_KEY_CLOSE_LEG_VOLUME) or 0
        residual = min(int(persisted), quantize(live_leg.qty))
        if residual > 0:
            await port.close_leg(symbol, leg_id, residual, row.client_order_id)
        update_close_leg_state(
            self._store_ctx, coid=row.client_order_id,
            new_state=CLOSE_LEG_STATE_DISPATCHED, close_row=True,
        )

    async def _replay_residual_opens(self, port: 'PositionPort') -> None:
        """Re-dispatch any reversal residual-open a crash left un-persisted.

        Runs AFTER :meth:`_replay_close_legs`, so the reversal's FIFO closes are
        already reconciled and re-opening the residual cannot race them. For each
        live breadcrumb: if the residual's own entry row already exists, the
        ``place_leg`` persist-first write landed and the entry journal / startup
        recovery own it — just clear the breadcrumb; otherwise rebuild the
        dispatch envelope and re-open the residual under the SAME deterministic
        ``KIND_ENTRY`` coid (a duplicate is therefore impossible at the exchange
        dedup), then clear it.
        """
        if self._store_ctx is None:
            return
        for row in list(iter_active_residual_opens(self._store_ctx)):
            await self._replay_residual_one(row, port)

    async def _replay_residual_one(self, row, port: 'PositionPort') -> None:
        """Reconcile + (if needed) re-dispatch one residual-open breadcrumb."""
        if self._store_ctx is None:
            return
        extras = row.extras or {}
        entry_coid: str | None = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID)
        if (entry_coid is not None
                and self._store_ctx.get_order(entry_coid) is not None):
            # ``place_leg`` reached its persist-first entry-row write before the
            # crash — the open is durable and owned by the entry path. Discharge.
            clear_residual_open_row(self._store_ctx, row.client_order_id)
            return
        # The residual open never persisted (persist-first guarantees the entry
        # row precedes any wire send, so its absence proves no dispatch happened —
        # re-opening cannot double-open). The breadcrumb is written BEFORE the
        # FIFO closes, so a crash in that window can also leave the opposing legs
        # un-closed AND their close-leg rows un-persisted — ``_replay_close_legs``
        # then has nothing to re-dispatch. Re-opening the residual on top of those
        # still-live opposing legs would over-expose the book. Reconcile against
        # the live legs first: a genuine reversal (positive residual) fully
        # consumes the opposing exposure, so every still-live opposing leg is an
        # owed close — FIFO-close them under the SAME deterministic parent coid
        # (idempotent at the exchange dedup) before the residual open.
        run_tag = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_RUN_TAG)
        bar_ts_ms = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_BAR_TS_MS)
        retry_seq = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_RETRY_SEQ)
        assert isinstance(run_tag, str)
        assert isinstance(bar_ts_ms, int)
        assert isinstance(retry_seq, int)
        opposing_side = 'sell' if row.side == 'buy' else 'buy'
        live_legs = await port.fetch_raw_positions(row.symbol)
        owed_closes = tuple(
            LegClose(leg_id=leg.leg_id, qty=leg.qty)
            for leg in live_legs
            if leg.side == opposing_side and leg.qty > 0.0
        )
        if owed_closes:
            parent_coid = row.client_order_id.removesuffix(':residual')
            await self._fan_out_closes(
                owed_closes, symbol=row.symbol, side=row.side,
                intent_key=row.intent_key or row.pine_entry_id or '',
                pine_id=row.pine_entry_id or '', parent_coid=parent_coid, port=port,
            )
        envelope = DispatchEnvelope(
            intent=EntryIntent(
                pine_id=row.pine_entry_id, symbol=row.symbol, side=row.side,
                qty=row.qty, order_type=OrderType.MARKET,
            ),
            run_tag=run_tag, bar_ts_ms=bar_ts_ms, retry_seq=retry_seq,
        )
        try:
            await port.place_leg(envelope, row.qty)
        except ExchangeOrderRejectedError:
            # The exchange definitively refused the residual entry — nothing
            # opened. ``restart_replay`` runs inside the sync startup wrapper,
            # which only catches ``ExchangeConnectionError`` (sync_engine), so
            # propagating here would abort startup and the breadcrumb would
            # survive un-cleared, retrying the same rejected residual on every
            # later sync. Discharge the breadcrumb and stop: the residual the
            # exchange rejected and Pine never re-signalled must not re-open.
            clear_residual_open_row(self._store_ctx, row.client_order_id)
            return
        except OrderDispositionUnknownError:
            # The residual entry's fate is unknown (ambiguous round-trip). Leave
            # the breadcrumb live so the next sync's replay / drain reconciles it
            # against the then-known leg state, but do NOT propagate — that would
            # abort startup on a recoverable round-trip, mirroring the bracket
            # re-clear path above.
            return
        clear_residual_open_row(self._store_ctx, row.client_order_id)

    async def _replay_bracket_ownership(self, port: 'PositionPort') -> None:
        """Re-assert / finish live bracket-ownership rows (second replay pass).

        For every live ownership row: if its leg is still open, an ``active`` row
        is re-asserted idempotently on the same per-leg coid (the broker no-ops an
        unchanged amend) while a ``clearing`` row (a clear a crash interrupted) is
        finished by re-clearing then releasing it; if the leg has vanished, the
        bracket it carried is moot — release the orphan row so it leaves the live
        index.
        """
        if self._store_ctx is None:
            return
        rows = list(iter_active_bracket_ownerships(self._store_ctx))
        if not rows:
            return
        by_symbol: dict[str, list] = {}
        for row in rows:
            by_symbol.setdefault(row.symbol, []).append(row)
        for symbol, srows in by_symbol.items():
            legs = await port.fetch_raw_positions(symbol)
            live_ids = {leg.leg_id for leg in legs}
            for row in srows:
                await self._replay_bracket_one(row, symbol, live_ids, port)

    async def _replay_bracket_one(
            self, row, symbol: str, live_ids: set, port: 'PositionPort',
    ) -> None:
        """Re-assert / finish one bracket-ownership row, or release a vanished leg."""
        if self._store_ctx is None:
            return
        extras = row.extras or {}
        leg_id: str | None = extras.get(EXTRAS_KEY_BRACKET_OWN_LEG_ID)
        if leg_id is None or leg_id not in live_ids:
            update_bracket_ownership_state(
                self._store_ctx, coid=row.client_order_id,
                new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
            )
            return
        if extras.get(EXTRAS_KEY_BRACKET_OWN_STATE) == BRACKET_OWN_STATE_CLEARING:
            # A clear interrupted before it confirmed (crash or ambiguous
            # timeout): finish it. Amending to None is idempotent — an already
            # cleared leg no-ops — so re-clear then release, NEVER re-assert the
            # original levels (that would resurrect a cancelled bracket). Re-clear
            # under the persisted clear coid (NOT the attach coid): the contract
            # lets a plugin dedup a repeated coid, so reusing the attach coid here
            # could be swallowed as a duplicate attach, leaving the bracket armed.
            try:
                await port.amend_bracket(
                    symbol, leg_id, side=row.side,
                    tp_price=None, sl_price=None, trail_offset=None,
                    coid=extras.get(EXTRAS_KEY_BRACKET_OWN_CLEAR_COID)
                    or row.client_order_id,
                )
            except OrderDispositionUnknownError:
                # The re-clear did not confirm. ``restart_replay`` runs inside the
                # sync startup wrapper, which only catches ``ExchangeConnectionError``;
                # propagating this sibling error would abort startup on a recoverable
                # ambiguous round-trip. Leave the row ``clearing`` (NOT released) so
                # the next ``drain_clearing_rows`` / replay retries the idempotent
                # re-clear, mirroring ``drain_clearing_rows``.
                return
            update_bracket_ownership_state(
                self._store_ctx, coid=row.client_order_id,
                new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
            )
            return
        await port.amend_bracket(
            symbol, leg_id, side=row.side,
            tp_price=extras.get(EXTRAS_KEY_BRACKET_OWN_TP),
            sl_price=extras.get(EXTRAS_KEY_BRACKET_OWN_SL),
            trail_offset=extras.get(EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET),
            coid=extras.get(EXTRAS_KEY_BRACKET_OWN_ATTACH_COID) or row.client_order_id,
        )
