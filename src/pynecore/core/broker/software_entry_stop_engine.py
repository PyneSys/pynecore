"""
Engine-side state machine for the STOP leg of a both-set Pine entry.

Pine has no "stop-limit" entry. ``strategy.entry(limit=L, stop=S)`` is two OCO
legs: a LIMIT below the open (a guaranteed-price pullback) and a STOP above
(market-on-rise) for a long — mirrored for a short. The first leg to fill
cancels the other.

The broker layer realises this asymmetrically so a double-fill race is
impossible by design:

- The LIMIT leg is a **native** working order (``execute_entry`` with
  ``order_type=LIMIT``). It rests on the exchange.
- The STOP leg is a **software** price-watch owned by this engine. When the
  price crosses the stop level the engine cancels the native LIMIT and — only
  once that cancel is *confirmed* — fires a MARKET order.

Because only one leg is ever native, there are never two simultaneous native
triggers: no race, and no need for a halt (a halt is forbidden for a live bot).

This module is the dispatch-time + watch-time counterpart of
:mod:`software_partial_bracket_engine`. The state machine and the persisted
watch rows in :mod:`store_helpers` form one pair: the rows are the durable
representation (PERSIST-FIRST), the in-memory :class:`EntryStopWatch` ledger is
the working set the WATCH phase reads on every price tick. On a clean restart
the ledger is rebuilt from the rows by :meth:`SoftwareEntryStopEngine.restart_replay`.

Ownership split (mirrors the partial-bracket engine): this state machine owns
only the in-memory ledger, the persisted transitions, and the restart replay.
Every broker-side action — the leg-scoped LIMIT cancel, the cancel-disposition
gate, and the MARKET dispatch — is performed by
:meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._drive_entry_stop_triggers`,
which calls the transition methods here at each step. That keeps the state
machine deterministic and unit-testable in isolation.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from pynecore.core.broker.store_helpers import (
    ENTRY_STOP_STATE_ABORTABLE,
    ENTRY_STOP_STATE_ABORTED,
    ENTRY_STOP_STATE_ARMED,
    ENTRY_STOP_STATE_CANCEL_PENDING,
    ENTRY_STOP_STATE_LIMIT_WON,
    ENTRY_STOP_STATE_LIVE,
    ENTRY_STOP_STATE_MARKET_PENDING,
    ENTRY_STOP_STATE_STOP_WON,
    EXTRAS_KEY_ENTRY_STOP_LEVEL,
    EXTRAS_KEY_ENTRY_STOP_LIMIT_COID,
    EXTRAS_KEY_ENTRY_STOP_MARKET_COID,
    EXTRAS_KEY_ENTRY_STOP_STATE,
    iter_active_entry_stop_watches,
    update_entry_stop_watch_state,
)

if TYPE_CHECKING:
    from pynecore.core.broker.storage import OrderRow, RunContext


__all__ = [
    'EntryStopWatch',
    'SoftwareEntryStopEngine',
]


@dataclass
class EntryStopWatch:
    """In-memory mirror of one entry-stop watch row.

    Mirrors the canonical extras keys in :mod:`store_helpers`; the persistent
    representation is authoritative (the engine reloads this struct from there
    on restart). Keyed by ``pine_id`` — a both-set entry is exactly one Pine
    entry id, so there is at most one live watch per id.

    ``side`` is the ENTRY side (``'buy'`` for a long, ``'sell'`` for a short):
    the direction the stop-fired MARKET opens in, identical to the native
    LIMIT leg's direction.
    """
    coid: str
    symbol: str
    pine_id: str
    side: str
    qty: float
    stop_level: float
    limit_coid: str
    state: str
    market_coid: str | None = None
    extras: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.pine_id


class SoftwareEntryStopEngine:
    """In-memory state machine for both-set entry STOP legs.

    One instance per :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine`.
    The engine does not talk to the broker directly; the sync engine performs
    every broker-side action (leg-scoped LIMIT cancel, cancel-disposition gate,
    MARKET dispatch) on the state machine's behalf, calling the transition
    methods here at each step.
    """

    def __init__(self, store_ctx: 'RunContext | None') -> None:
        self._store_ctx = store_ctx
        self._watches: dict[str, EntryStopWatch] = {}

    # === Registration =====================================================

    def register_watch(self, watch: EntryStopWatch) -> None:
        """Add one freshly-persisted watch to the in-memory ledger.

        Called by the sync engine after a successful
        :func:`~pynecore.core.broker.store_helpers.create_entry_stop_watch_row`.
        """
        if watch.state not in ENTRY_STOP_STATE_LIVE:
            raise ValueError(
                f"register_watch: refuses to track watch in non-live state "
                f"{watch.state!r} (pine_id={watch.pine_id!r})"
            )
        if watch.key in self._watches:
            raise ValueError(
                f"register_watch: watch already tracked (pine_id="
                f"{watch.pine_id!r}, existing coid={self._watches[watch.key].coid!r}, "
                f"new coid={watch.coid!r})"
            )
        self._watches[watch.key] = watch

    # === Queries ==========================================================

    def get_watch(self, pine_id: str) -> EntryStopWatch | None:
        return self._watches.get(pine_id)

    def has_watch(self, pine_id: str) -> bool:
        return pine_id in self._watches

    def iter_watches(self) -> Iterable[EntryStopWatch]:
        return list(self._watches.values())

    @staticmethod
    def stop_crossed(watch: EntryStopWatch, last_price: float) -> bool:
        """Whether the current price has crossed the watch's stop level.

        A long both-set entry (``side='buy'``) places its STOP *above* the
        open and fires when price rises to it; a short (``side='sell'``)
        places its STOP *below* and fires when price falls to it.
        """
        if watch.side == 'buy':
            return last_price >= watch.stop_level
        return last_price <= watch.stop_level

    # === Transitions ======================================================

    def begin_cancel(self, pine_id: str) -> EntryStopWatch | None:
        """``armed`` → ``cancel_pending``: the stop crossed, the engine is
        about to cancel the native LIMIT leg.

        Latched: once here, the decision to take the stop side is committed.
        On restart the watch resumes in ``cancel_pending`` and the sync engine
        re-drives the (idempotent) cancel gate regardless of the live price.
        Idempotent no-op when the watch is not ``armed``.
        """
        watch = self._watches.get(pine_id)
        if watch is None or watch.state != ENTRY_STOP_STATE_ARMED:
            return None
        self._transition(watch, ENTRY_STOP_STATE_CANCEL_PENDING, close_row=False)
        return watch

    def confirm_limit_cancelled_fire_market(
            self, pine_id: str, *, market_coid: str,
    ) -> EntryStopWatch | None:
        """``cancel_pending`` → ``stop_market_pending``: the LIMIT cancel is
        CONFIRMED, so the MARKET may go.

        Persists the deterministic MARKET client-order-id BEFORE the sync
        engine POSTs it, so a crash-restart can verify-before-resend and never
        double-open. Idempotent no-op when the watch is not ``cancel_pending``.
        """
        watch = self._watches.get(pine_id)
        if watch is None or watch.state != ENTRY_STOP_STATE_CANCEL_PENDING:
            return None
        watch.market_coid = market_coid
        self._transition(
            watch, ENTRY_STOP_STATE_MARKET_PENDING,
            close_row=False, market_coid=market_coid,
        )
        return watch

    def mark_stop_won(self, pine_id: str) -> EntryStopWatch | None:
        """``stop_market_pending`` → ``stop_won`` (terminal): the stop-fired
        MARKET dispatch landed. Idempotent no-op otherwise.
        """
        watch = self._watches.get(pine_id)
        if watch is None or watch.state != ENTRY_STOP_STATE_MARKET_PENDING:
            return None
        self._transition(watch, ENTRY_STOP_STATE_STOP_WON, close_row=True)
        return watch

    def mark_limit_won(self, pine_id: str, *, reason: str) -> EntryStopWatch | None:
        """Any live state → ``limit_won`` (terminal): the native LIMIT leg
        filled (or the cancel attempt reported ALREADY_FILLED), so the watch's
        job is done and no MARKET must ever fire. Idempotent no-op when no live
        watch matches.
        """
        watch = self._watches.get(pine_id)
        if watch is None:
            return None
        self._transition(
            watch, ENTRY_STOP_STATE_LIMIT_WON,
            close_row=True, extras_patch={'limit_won_reason': reason},
        )
        return watch

    def mark_aborted(self, pine_id: str, *, reason: str) -> EntryStopWatch | None:
        """Abortable state (``armed`` / ``cancel_pending``) → ``aborted``
        (terminal): the strategy cancelled the parent entry or its native LIMIT
        leg went away before the stop side committed, so the watch stops
        watching and never fires.

        Idempotent no-op once the watch has committed to the stop side
        (``stop_market_pending`` — the deterministic KIND_ENTRY_STOP market id
        is already persisted and the MARKET is in flight) or terminalised, so a
        delayed broker cancelled/rejected ack for the OCO's own now-cancelled
        LIMIT cannot retire a watch that has already fired the market. Also a
        no-op when no watch matches.
        """
        watch = self._watches.get(pine_id)
        if watch is None or watch.state not in ENTRY_STOP_STATE_ABORTABLE:
            return None
        self._transition(
            watch, ENTRY_STOP_STATE_ABORTED,
            close_row=True, extras_patch={'abort_reason': reason},
        )
        return watch

    def amend_watch(
            self, pine_id: str, *, stop_level: float, qty: float, side: str,
    ) -> EntryStopWatch | None:
        """Re-sync an ``armed`` watch to an amended both-set entry.

        When the strategy re-emits the same ``pine_id`` both-set entry with a
        changed ``stop`` / ``qty`` / ``side``, the sync engine amends the native
        LIMIT leg (``modify_entry``) and calls this to keep the software STOP
        leg in step. ``modify_entry`` preserves the native LIMIT's
        :data:`~pynecore.core.broker.idempotency.KIND_ENTRY` client-order-id
        (the dispatch envelope anchor is pinned per ``intent_key`` across amend
        cycles), so the watch's leg-scoped cancel target
        (:attr:`EntryStopWatch.limit_coid`) stays valid — only the fire level,
        size, and side change.

        PERSIST-FIRST: the watch row is updated before the in-memory mirror.
        Idempotent guards:

        - no watch for ``pine_id`` → ``None`` (the entry has no STOP leg, e.g. a
          plain limit-only entry); the caller treats this as a no-op.
        - the watch has left ``armed`` (``cancel_pending`` /
          ``stop_market_pending`` — the OCO is already resolving toward the stop
          side) → returned unchanged WITHOUT amending: rewinding the fire level /
          size mid-cancel would race the in-flight LIMIT cancel and the
          deterministic MARKET identity. The next terminalisation settles it.
        """
        watch = self._watches.get(pine_id)
        if watch is None:
            return None
        if watch.state != ENTRY_STOP_STATE_ARMED:
            return watch
        if self._store_ctx is not None:
            update_entry_stop_watch_state(
                self._store_ctx,
                coid=watch.coid,
                new_state=watch.state,
                stop_level=stop_level,
                qty=qty,
                side=side,
            )
        watch.stop_level = stop_level
        watch.qty = qty
        watch.side = side
        return watch

    # === Restart replay ===================================================

    def restart_replay(self) -> None:
        """Rebuild the in-memory ledger from persisted watch rows.

        Called once by the sync engine during startup, after the journal's
        regular replay but before the first :meth:`sync`. Terminal rows are
        filtered out by :func:`iter_active_entry_stop_watches`. The latched
        intermediate states (``cancel_pending`` / ``stop_market_pending``) are
        reloaded as-is: the next :meth:`_drive_entry_stop_triggers` re-drives
        them deterministically (re-issue the idempotent cancel / re-dispatch
        the idempotent MARKET), so no demotion is needed.
        """
        if self._store_ctx is None:
            return
        self._watches.clear()
        for row in iter_active_entry_stop_watches(self._store_ctx):
            watch = _watch_from_row(row)
            if watch is None:
                continue
            self._watches[watch.key] = watch

    # === State machine plumbing ==========================================

    def _transition(
            self,
            watch: EntryStopWatch,
            new_state: str,
            *,
            close_row: bool,
            market_coid: str | None = None,
            extras_patch: dict | None = None,
    ) -> None:
        watch.state = new_state
        if extras_patch:
            watch.extras.update(extras_patch)
        if self._store_ctx is not None:
            update_entry_stop_watch_state(
                self._store_ctx,
                coid=watch.coid,
                new_state=new_state,
                market_coid=market_coid,
                extras_patch=extras_patch,
                close_row=close_row,
            )
        if close_row:
            self._watches.pop(watch.key, None)


def _watch_from_row(row: 'OrderRow') -> EntryStopWatch | None:
    extras = row.extras or {}
    state = extras.get(EXTRAS_KEY_ENTRY_STOP_STATE, '')
    if state not in ENTRY_STOP_STATE_LIVE:
        return None
    stop_level: float | None = extras.get(EXTRAS_KEY_ENTRY_STOP_LEVEL)
    limit_coid: str | None = extras.get(EXTRAS_KEY_ENTRY_STOP_LIMIT_COID)
    if stop_level is None or limit_coid is None:
        return None
    return EntryStopWatch(
        coid=row.client_order_id,
        symbol=row.symbol,
        pine_id=row.pine_entry_id or '',
        side=row.side,
        qty=row.qty,
        stop_level=float(stop_level),
        limit_coid=limit_coid,
        state=state,
        market_coid=extras.get(EXTRAS_KEY_ENTRY_STOP_MARKET_COID),
        extras=dict(extras),
    )
