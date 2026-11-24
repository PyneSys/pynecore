"""
Engine-side state machine of a software stop-limit order.

A Pine ``strategy.entry`` or ``strategy.order`` placed with both ``stop`` and
``limit`` is ONE stop-limit order (measured on TradingView, BINANCE:BTCUSDT
30m; the backtest follows it in
:meth:`~pynecore.lib.strategy.SimPosition._trigger_stop_limit`):

- before the stop triggers nothing can fill — a pullback to the limit price
  does not open the position;
- when the price reaches the stop, the order becomes a plain LIMIT at the
  limit price: a marketable limit fills at once, any other rests at the limit;
- a stop that is already passed when the order is placed triggers at once.

A venue with a native stop-limit order
(:attr:`~ExchangeCapabilities.stop_limit_order`
``NATIVE``) gets the order as it is. On any other venue the engine emulates it
with this state machine: nothing rests on the venue before the trigger, and
the trigger sends a plain LIMIT entry.

Watch phases (persisted on the watch row, PERSIST-FIRST):

- ``armed`` — the engine judges the price against the stop. The live price is
  the chart's, the one every software trigger of the engine reads (the bar
  close at each sync); after a restart the bars the engine did not see are
  judged by their high / low (:meth:`SoftwareStopLimitEngine.observe_history_bar`).
- ``limit_pending`` — the stop triggered. The LIMIT's client-order-id is
  persisted BEFORE its POST, so a retry, in this process or after a restart,
  sends it again only once the earlier send provably did not land.
- ``triggered`` — the LIMIT lives on the venue as a regular entry order. The
  row stays live so the modify path and a restart know the order already
  triggered (a later change of the stop price has no effect, as on
  TradingView).
- ``filled`` / ``aborted`` — terminal.

A row written by a version that modelled a both-set entry as two OCO legs (a
native LIMIT resting from the start) is replayed as a legacy row
(:attr:`StopLimitWatch.legacy`) and only retired, never driven.

Ownership split (mirrors the partial-bracket engine): this state machine owns
the in-memory ledger, the persisted transitions and the restart replay. Every
broker-side action is performed by the sync engine
(:meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._drive_stop_limit_watches`),
which calls the transition methods here at each step.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from pynecore.core.broker.store_helpers import (
    EXTRAS_KEY_STOP_LIMIT_ARMED_BAR,
    EXTRAS_KEY_STOP_LIMIT_CHECKED_BAR,
    EXTRAS_KEY_STOP_LIMIT_LIMIT,
    EXTRAS_KEY_STOP_LIMIT_ORDER_COID,
    EXTRAS_KEY_STOP_LIMIT_STATE,
    EXTRAS_KEY_STOP_LIMIT_STOP,
    LEGACY_EXTRAS_KEY_OCO_LIMIT_COID,
    LEGACY_EXTRAS_KEY_OCO_MARKET_COID,
    LEGACY_OCO_STATE_CANCEL_PENDING,
    LEGACY_OCO_STATE_MARKET_PENDING,
    STOP_LIMIT_STATE_ABORTED,
    STOP_LIMIT_STATE_ARMED,
    STOP_LIMIT_STATE_FILLED,
    STOP_LIMIT_STATE_LIMIT_PENDING,
    STOP_LIMIT_STATE_LIVE,
    STOP_LIMIT_STATE_TRIGGERED,
    create_stop_limit_watch_row,
    iter_live_stop_limit_watches,
    update_stop_limit_watch,
)

if TYPE_CHECKING:
    from pynecore.core.broker.storage import OrderRow, RunContext


__all__ = [
    'StopLimitWatch',
    'SoftwareStopLimitEngine',
]


@dataclass
class StopLimitWatch:
    """In-memory mirror of one stop-limit watch row.

    The persisted row is authoritative; a restart rebuilds this struct from it.
    Keyed by ``pine_id``: one Pine order id carries at most one live watch.

    :ivar coid: The watch row's client-order-id (no venue order carries it).
    :ivar symbol: Exchange-side symbol.
    :ivar pine_id: The Pine order id.
    :ivar side: The order side, ``'buy'`` or ``'sell'``.
    :ivar qty: Order quantity.
    :ivar stop: The stop price whose cross triggers the order.
    :ivar limit: The limit price the triggered LIMIT rests at.
    :ivar state: The watch phase.
    :ivar armed_bar_ts_ms: The bar the watch was armed on.
    :ivar checked_bar_ts_ms: The last bar whose price the engine judged.
    :ivar order_coid: The triggered LIMIT's client-order-id, once persisted.
    :ivar crossed_offline: The stop was crossed on a bar the engine did not
        see live (judged after a restart); the next drive triggers it.
    :ivar legacy: A row of the legacy OCO model: only retired, never driven.
    :ivar legacy_market_coid: A legacy row's stop-fired MARKET id, if any.
    """
    coid: str
    symbol: str
    pine_id: str
    side: str
    qty: float
    stop: float
    limit: float
    state: str
    armed_bar_ts_ms: int = 0
    checked_bar_ts_ms: int = 0
    order_coid: str | None = None
    crossed_offline: bool = False
    legacy: bool = False
    legacy_market_coid: str | None = None
    extras: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.pine_id


class SoftwareStopLimitEngine:
    """In-memory state machine of the software stop-limit watches.

    One instance per :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine`.
    It never talks to the broker; the sync engine performs every broker-side
    action and calls the transition methods here.
    """

    def __init__(self, store_ctx: 'RunContext | None') -> None:
        self._store_ctx = store_ctx
        self._watches: dict[str, StopLimitWatch] = {}

    # === Registration =====================================================

    def arm(
            self, *, coid: str, symbol: str, pine_id: str, side: str, qty: float,
            stop: float, limit: float, bar_ts_ms: int,
    ) -> StopLimitWatch:
        """Persist and register a fresh ``armed`` watch (PERSIST-FIRST).

        :param coid: The watch row's deterministic client-order-id.
        :param symbol: Exchange-side symbol.
        :param pine_id: The Pine order id.
        :param side: The order side.
        :param qty: Order quantity.
        :param stop: The stop price.
        :param limit: The limit price.
        :param bar_ts_ms: The bar the watch is armed on.
        :raises ValueError: When a watch is already tracked for ``pine_id``.
        """
        if pine_id in self._watches:
            raise ValueError(
                f"arm: a stop-limit watch is already tracked for {pine_id!r} "
                f"(coid={self._watches[pine_id].coid!r})"
            )
        if self._store_ctx is not None:
            create_stop_limit_watch_row(
                self._store_ctx, coid=coid, symbol=symbol, side=side, qty=qty,
                intent_key=pine_id, pine_entry_id=pine_id, stop=stop, limit=limit,
                armed_bar_ts_ms=bar_ts_ms,
            )
        watch = StopLimitWatch(
            coid=coid, symbol=symbol, pine_id=pine_id, side=side, qty=qty,
            stop=stop, limit=limit, state=STOP_LIMIT_STATE_ARMED,
            armed_bar_ts_ms=bar_ts_ms, checked_bar_ts_ms=bar_ts_ms,
        )
        self._watches[pine_id] = watch
        return watch

    # === Queries ==========================================================

    def get_watch(self, pine_id: str) -> StopLimitWatch | None:
        return self._watches.get(pine_id)

    def has_watch(self, pine_id: str) -> bool:
        return pine_id in self._watches

    def iter_watches(self) -> Iterable[StopLimitWatch]:
        return list(self._watches.values())

    @staticmethod
    def stop_crossed(watch: StopLimitWatch, price: float) -> bool:
        """Whether ``price`` reached the watch's stop.

        A buy stop triggers when the price rises to it, a sell stop when the
        price falls to it.
        """
        if watch.side == 'buy':
            return price >= watch.stop
        return price <= watch.stop

    def observe_history_bar(self, bar_ts_ms: int, high: float, low: float) -> list[StopLimitWatch]:
        """Judge an ``armed`` watch against a bar the engine did not see live.

        Fed with the bars a restart replays. A bar counts when it lies after
        the watch's last judged bar, or IS that bar while the watch was armed
        on an earlier one (the process stopped inside it, so its later ticks
        were never judged; a bar the watch was armed on also holds ticks from
        before the arming and is not judged by its range). A buy stop at or
        below the bar's high, or a sell stop at or above its low, crossed while
        the engine was away: as on TradingView the order triggered, so the
        next drive turns it into its LIMIT.

        :param bar_ts_ms: The bar's open time.
        :param high: The bar's high.
        :param low: The bar's low.
        :return: The watches this bar crossed.
        """
        crossed: list[StopLimitWatch] = []
        for watch in self._watches.values():
            if (watch.legacy or watch.state != STOP_LIMIT_STATE_ARMED
                    or watch.crossed_offline):
                continue
            unseen = (bar_ts_ms > watch.checked_bar_ts_ms
                      or (bar_ts_ms == watch.checked_bar_ts_ms
                          and watch.armed_bar_ts_ms < watch.checked_bar_ts_ms))
            if not unseen:
                continue
            if self.stop_crossed(watch, high if watch.side == 'buy' else low):
                watch.crossed_offline = True
                crossed.append(watch)
        return crossed

    # === Transitions ======================================================

    def note_judged(self, pine_id: str, bar_ts_ms: int) -> None:
        """Record that the live price of ``bar_ts_ms`` was judged for an armed watch.

        Persisted once per new bar, so a restart judges only the bars after it.
        """
        watch = self._watches.get(pine_id)
        if (watch is None or watch.state != STOP_LIMIT_STATE_ARMED
                or bar_ts_ms <= watch.checked_bar_ts_ms):
            return
        watch.checked_bar_ts_ms = bar_ts_ms
        self._persist(watch, extras_patch={EXTRAS_KEY_STOP_LIMIT_CHECKED_BAR: bar_ts_ms})

    def begin_trigger(self, pine_id: str, *, order_coid: str) -> StopLimitWatch | None:
        """``armed`` → ``limit_pending``: the stop triggered.

        Persists the LIMIT's client-order-id BEFORE the sync engine POSTs it.
        A no-op unless the watch is ``armed``.
        """
        watch = self._watches.get(pine_id)
        if watch is None or watch.legacy or watch.state != STOP_LIMIT_STATE_ARMED:
            return None
        watch.order_coid = order_coid
        self._transition(watch, STOP_LIMIT_STATE_LIMIT_PENDING,
                         extras_patch={EXTRAS_KEY_STOP_LIMIT_ORDER_COID: order_coid})
        return watch

    def rebind_order_coid(self, pine_id: str, *, order_coid: str) -> StopLimitWatch | None:
        """Record a fresh client-order-id for a ``limit_pending`` LIMIT before its POST.

        The LIMIT keeps its persisted id across retries unless its earlier
        send was refused and the id spent: the retry then goes out under a new
        id, persisted first so the next retry verifies the send in flight.
        """
        watch = self._watches.get(pine_id)
        if (watch is None or watch.state != STOP_LIMIT_STATE_LIMIT_PENDING
                or watch.order_coid == order_coid):
            return None
        watch.order_coid = order_coid
        self._persist(watch, extras_patch={EXTRAS_KEY_STOP_LIMIT_ORDER_COID: order_coid})
        return watch

    def mark_triggered(self, pine_id: str) -> StopLimitWatch | None:
        """``limit_pending`` → ``triggered``: the venue took the LIMIT."""
        watch = self._watches.get(pine_id)
        if watch is None or watch.state != STOP_LIMIT_STATE_LIMIT_PENDING:
            return None
        self._transition(watch, STOP_LIMIT_STATE_TRIGGERED)
        return watch

    def adopt_fill_while_armed(self, pine_id: str) -> StopLimitWatch | None:
        """``armed`` → ``triggered`` on an entry fill the watch never sent.

        Nothing rests on the venue for an armed watch, so a fill of its Pine
        id breaks the model's invariant (a leftover order of a legacy run, an
        order someone placed under the bot's id). The fill is real exposure
        and is booked as the order's fill; the watch must never send its own
        LIMIT on top of it, so it takes the filling order as its LIMIT.
        """
        watch = self._watches.get(pine_id)
        if watch is None or watch.legacy or watch.state != STOP_LIMIT_STATE_ARMED:
            return None
        self._transition(watch, STOP_LIMIT_STATE_TRIGGERED,
                         extras_patch={'adopted_reason': 'fill_while_armed'})
        return watch

    def mark_filled(self, pine_id: str) -> StopLimitWatch | None:
        """``triggered`` → ``filled`` (terminal): the LIMIT filled completely."""
        watch = self._watches.get(pine_id)
        if watch is None or watch.state != STOP_LIMIT_STATE_TRIGGERED:
            return None
        self._transition(watch, STOP_LIMIT_STATE_FILLED, close_row=True)
        return watch

    def mark_aborted(self, pine_id: str, *, reason: str) -> StopLimitWatch | None:
        """``armed`` / ``triggered`` → ``aborted`` (terminal).

        The order was cancelled before its trigger, or its triggered LIMIT
        left the venue unfilled. A ``limit_pending`` watch is not aborted
        here: its LIMIT may be in flight, and only the send verification
        (:meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._drive_stop_limit_watches`)
        may settle it. A legacy row is retired by :meth:`retire_legacy`.
        """
        watch = self._watches.get(pine_id)
        if (watch is None or watch.legacy
                or watch.state not in (STOP_LIMIT_STATE_ARMED, STOP_LIMIT_STATE_TRIGGERED)):
            return None
        self._transition(watch, STOP_LIMIT_STATE_ABORTED, close_row=True,
                         extras_patch={'abort_reason': reason})
        return watch

    def abort_unsent(self, pine_id: str, *, reason: str) -> StopLimitWatch | None:
        """``limit_pending`` → ``aborted``: the LIMIT provably never landed and
        is not sent any more (its Pine order is gone)."""
        watch = self._watches.get(pine_id)
        if watch is None or watch.state != STOP_LIMIT_STATE_LIMIT_PENDING:
            return None
        self._transition(watch, STOP_LIMIT_STATE_ABORTED, close_row=True,
                         extras_patch={'abort_reason': reason})
        return watch

    def retire_legacy(self, pine_id: str, *, reason: str) -> StopLimitWatch | None:
        """Close a legacy OCO row (terminal ``aborted``)."""
        watch = self._watches.get(pine_id)
        if watch is None or not watch.legacy:
            return None
        self._transition(watch, STOP_LIMIT_STATE_ABORTED, close_row=True,
                         extras_patch={'abort_reason': reason})
        return watch

    def amend(
            self, pine_id: str, *, stop: float, limit: float, qty: float, side: str,
    ) -> StopLimitWatch | None:
        """Keep a watch in step with its amended Pine order.

        ``armed``: every term follows. ``triggered``: the stop already did its
        job, so only the limit and the quantity follow (the venue LIMIT is
        amended by the caller). ``limit_pending``: unchanged — the LIMIT is in
        flight under the persisted terms.

        :return: The watch, or ``None`` when none is tracked for ``pine_id``.
        """
        watch = self._watches.get(pine_id)
        if watch is None or watch.legacy:
            return None
        if watch.state == STOP_LIMIT_STATE_ARMED:
            watch.stop, watch.limit, watch.qty, watch.side = stop, limit, qty, side
        elif watch.state == STOP_LIMIT_STATE_TRIGGERED:
            watch.limit, watch.qty = limit, qty
        else:
            return watch
        self._persist(
            watch,
            extras_patch={EXTRAS_KEY_STOP_LIMIT_STOP: watch.stop,
                          EXTRAS_KEY_STOP_LIMIT_LIMIT: watch.limit},
            qty=watch.qty, side=watch.side,
        )
        return watch

    # === Restart replay ===================================================

    def restart_replay(self) -> None:
        """Rebuild the in-memory ledger from the live watch rows.

        Called once at startup, after the journal replay and before the first
        sync. A legacy OCO row comes back with :attr:`StopLimitWatch.legacy`
        set; the sync engine retires it.
        """
        if self._store_ctx is None:
            return
        self._watches.clear()
        for row in iter_live_stop_limit_watches(self._store_ctx):
            watch = _watch_from_row(row)
            if watch is not None:
                self._watches[watch.key] = watch

    # === State machine plumbing ==========================================

    def _transition(
            self,
            watch: StopLimitWatch,
            new_state: str,
            *,
            close_row: bool = False,
            extras_patch: dict | None = None,
    ) -> None:
        watch.state = new_state
        if extras_patch:
            watch.extras.update(extras_patch)
        if self._store_ctx is not None:
            update_stop_limit_watch(
                self._store_ctx, coid=watch.coid, state=new_state,
                extras_patch=extras_patch, close_row=close_row,
            )
        if close_row:
            self._watches.pop(watch.key, None)

    def _persist(
            self,
            watch: StopLimitWatch,
            *,
            extras_patch: dict,
            qty: float | None = None,
            side: str | None = None,
    ) -> None:
        watch.extras.update(extras_patch)
        if self._store_ctx is not None:
            update_stop_limit_watch(
                self._store_ctx, coid=watch.coid, extras_patch=extras_patch,
                qty=qty, side=side,
            )


def _watch_from_row(row: 'OrderRow') -> StopLimitWatch | None:
    extras = row.extras or {}
    state = str(extras.get(EXTRAS_KEY_STOP_LIMIT_STATE, ''))
    stop = extras.get(EXTRAS_KEY_STOP_LIMIT_STOP)
    pine_id = row.pine_entry_id or ''
    if stop is None or not pine_id:
        return None
    legacy = (LEGACY_EXTRAS_KEY_OCO_LIMIT_COID in extras
              or state in (LEGACY_OCO_STATE_CANCEL_PENDING, LEGACY_OCO_STATE_MARKET_PENDING))
    if not legacy and state not in STOP_LIMIT_STATE_LIVE:
        return None
    limit = extras.get(EXTRAS_KEY_STOP_LIMIT_LIMIT)
    armed_bar = int(extras.get(EXTRAS_KEY_STOP_LIMIT_ARMED_BAR) or 0)
    return StopLimitWatch(
        coid=row.client_order_id,
        symbol=row.symbol,
        pine_id=pine_id,
        side=row.side,
        qty=row.qty,
        stop=float(stop),
        limit=float(limit) if limit is not None else 0.0,
        state=state,
        armed_bar_ts_ms=armed_bar,
        checked_bar_ts_ms=int(extras.get(EXTRAS_KEY_STOP_LIMIT_CHECKED_BAR) or armed_bar),
        order_coid=extras.get(EXTRAS_KEY_STOP_LIMIT_ORDER_COID),
        legacy=legacy,
        legacy_market_coid=extras.get(LEGACY_EXTRAS_KEY_OCO_MARKET_COID) if legacy else None,
        extras=dict(extras),
    )
