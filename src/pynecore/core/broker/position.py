"""
Position tracking for live broker trading.

:class:`BrokerPosition` extends :class:`~pynecore.lib.strategy.PositionBase`
with no simulation logic — the exchange is the source of truth for fills,
prices, fees, and margin state.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from pynecore import lib
from pynecore.core.broker.intent_builder import CLOSE_ALL_EXIT_ID, CLOSE_EXIT_ID_PREFIX
from pynecore.core.broker.models import LegType
from pynecore.core.closed_trade_stats import ClosedTradeStats
from pynecore.lib.log import broker_error as _blog_error
from pynecore.lib.log import broker_info as _blog_info
from pynecore.lib.log import broker_warning as _blog_warning
from pynecore.lib.strategy import PositionBase, Trade
# noinspection PyProtectedMember
from pynecore.lib.strategy import _EntryBinding
from pynecore.types.na import na_float

if TYPE_CHECKING:
    from pynecore.lib.strategy import direction
    # noinspection PyProtectedMember
    from pynecore.lib.strategy import Order, _ExitOrderKey
    from pynecore.types.strategy import QtyType
    from pynecore.core.broker.models import OrderEvent

__all__ = ['BrokerPosition', 'EntryBookSnapshot', 'ProvisionalClose']

# Legs that only ever reduce exposure: the plugin contract requires reduce-only
# delivery for them (``ExchangeCapabilities.reduce_only``)
_EXIT_LEG_TYPES = frozenset((
    LegType.CLOSE, LegType.TAKE_PROFIT, LegType.STOP_LOSS, LegType.TRAILING_STOP,
))
# The legs of a ``strategy.exit`` bracket: a fill settles the bindings of the
# leg's ``from_entry``
_BOUND_LEG_TYPES = frozenset((
    LegType.TAKE_PROFIT, LegType.STOP_LOSS, LegType.TRAILING_STOP,
))


@dataclass(frozen=True, slots=True)
class EntryBookSnapshot:
    """A binding book read back from the store (:meth:`BrokerPosition.parse_entry_book_snapshot`).

    :ivar size: The position size the book was written for.
    :ivar entry_seq: The binding sequence counter at the time.
    :ivar faithful: Whether every binding was opened by a fill
        (:attr:`~BrokerPosition.book_faithful`).
    :ivar written_ms: Wall-clock time of the write, epoch milliseconds.
    :ivar bindings: The open bindings, oldest first.
    :ivar order_bindings: Venue order id -> index into :attr:`bindings` of the
        binding a further slice of that order grows.
    :ivar fill_cursors: Client / venue order id -> the most the book had
        taken in of that order (:meth:`BrokerPosition.fill_cursor`).
    """
    size: float
    entry_seq: int
    faithful: bool
    written_ms: int
    bindings: tuple[_EntryBinding, ...]
    order_bindings: dict[str, int]
    fill_cursors: dict[str, float]


@dataclass(slots=True)
class ProvisionalClose:
    """A close booked at an estimated price, with no fill of its own.

    The sync engine books a venue reduction nobody reported this way
    (:meth:`BrokerPosition.record_provisional_close`). A late fill that turns
    out to be that reduction moves the booked trades to its own price
    (:meth:`BrokerPosition.reprice_provisional_closes`).

    :ivar price: The estimated exit price the close was booked at.
    :ivar trades: The closed trades it booked, FIFO order.
    :ivar unpriced: Per trade, the quantity still booked at :attr:`price`.
    :ivar entry_commissions: Per trade, its entry commission (the close itself
        carries no fee).
    :ivar profit: The realized profit the close is counted with in the win /
        loss counters, re-priced parts included.
    :ivar stats_base: The closed-trade statistics before the close.
    :ivar count_base: :attr:`BrokerPosition.closed_trades_count` before the close.
    """
    price: float
    trades: list[Trade]
    unpriced: list[float]
    entry_commissions: list[float]
    profit: float
    stats_base: ClosedTradeStats
    count_base: int


class BrokerPosition(PositionBase):
    """
    Position state tracker for live broker trading.

    The exchange determines fills, prices, fees, and margin state;
    :meth:`record_fill` consumes :class:`OrderEvent` objects emitted by a
    :class:`~pynecore.core.plugin.broker.BrokerPlugin` and updates the
    local view of the position accordingly.

    Trades are tracked FIFO: the first entry filled is the first closed
    when the position is reduced, matching TradingView default semantics.

    Beside the trade rows the position keeps the simulator's binding book
    (:attr:`~BrokerPosition._entry_book`, see
    :class:`~pynecore.lib.strategy._EntryBinding`): one binding per entry fill,
    settled by the origin of each closing fill rather than FIFO. Its bound
    quantities add up to the position size.

    Note: margin, liquidation price, and fee currency conversion are all
    handled by the exchange. This class only records what the exchange
    tells it.
    """

    __slots__ = (
        'size', 'sign', 'avg_price',
        'netprofit', 'openprofit', 'grossprofit', 'grossloss',
        'open_commission',
        'eventrades', 'wintrades', 'losstrades',
        'closed_trades_count', '_closed_trade_stats',
        'max_drawdown', 'max_drawdown_percent', 'max_runup', 'max_runup_percent',
        'max_equity', 'min_equity',
        'max_contracts_held_long', 'max_contracts_held_short',
        'sum_profit_ratio', 'sum_win_profit_ratio', 'sum_loss_profit_ratio',
        'open_trades', 'closed_trades', 'new_closed_trades',
        'entry_orders', 'exit_orders',
        # The binding book (see :class:`~pynecore.lib.strategy._EntryBinding`):
        '_entry_book', '_entry_seq', '_order_bindings', 'last_spent_entry_ids',
        'book_faithful', '_fill_cursors', 'book_moves',
        # Per-evaluation set of close keys already seen this script run, so a
        # second same-key ``strategy.close()`` THIS evaluation nets onto the
        # first while a next-tick re-issue (calc_on_every_tick) replaces it.
        '_closes_this_eval',
        # === Risk management state (mirrors SimPosition) ===
        # Configuration set by ``strategy.risk.*`` setters:
        'risk_allowed_direction',
        'risk_max_drawdown_value', 'risk_max_drawdown_type', 'risk_max_drawdown_alert',
        'risk_max_intraday_loss_value', 'risk_max_intraday_loss_type',
        'risk_max_intraday_loss_alert',
        'risk_max_cons_loss_days', 'risk_max_cons_loss_days_alert',
        'risk_max_intraday_filled_orders', 'risk_max_intraday_filled_orders_alert',
        'risk_max_position_size',
        # Runtime counters / day-rollover tracking:
        'risk_cons_loss_days', 'risk_last_trading_day', 'risk_last_day_equity',
        'risk_intraday_filled_orders', 'risk_intraday_start_equity',
        '_current_price',
        'least_tradable_qty',
    )

    def __init__(self) -> None:
        self.size: float = 0.0
        self.sign: float = 0.0
        self.avg_price = na_float

        self.netprofit: float = 0.0
        self.openprofit: float = 0.0
        self.grossprofit: float = 0.0
        self.grossloss: float = 0.0
        self.open_commission: float = 0.0

        self.eventrades: int = 0
        self.wintrades: int = 0
        self.losstrades: int = 0
        self.closed_trades_count: int = 0
        self._closed_trade_stats = ClosedTradeStats()
        self.max_drawdown: float = 0.0
        self.max_drawdown_percent: float = 0.0
        self.max_runup: float = 0.0
        self.max_runup_percent: float = 0.0
        self.max_contracts_held_long: float = 0.0
        self.max_contracts_held_short: float = 0.0
        self.sum_profit_ratio: float = 0.0
        self.sum_win_profit_ratio: float = 0.0
        self.sum_loss_profit_ratio: float = 0.0
        # Mark-to-market peak equity, used by ``_peak_equity`` for the
        # ``max_drawdown(percent_of_equity)`` threshold. Updated on every
        # :meth:`update_unrealized_pnl` and :meth:`record_fill` call.
        self.max_equity: float = -float("inf")
        # Mark-to-market trough, the run-up's low-water mark.
        self.min_equity: float = float("inf")

        self.open_trades: list[Trade] = []
        self.closed_trades: deque[Trade] = deque(maxlen=9000)
        self.new_closed_trades: list[Trade] = []
        # The binding book, one binding per entry fill, oldest first. The trade
        # rows above settle FIFO; a binding settles by the origin of the closing
        # fill (see :meth:`~BrokerPosition._settle_bindings`).
        self._entry_book: list[_EntryBinding] = []
        self._entry_seq: int = 0
        # Venue order id -> the binding its fills opened, so a further slice of
        # the same entry order grows that binding instead of adding one.
        self._order_bindings: dict[str, _EntryBinding] = {}
        # Entry ids of the bindings the last booked fill spent, oldest first.
        self.last_spent_entry_ids: list[str] = []
        # Whether every open binding was opened by a fill. A binding seeded
        # without one (an adopted parent, a rebuild from the trade rows) is
        # attributed FIFO, not by origin, so a missing binding proves nothing
        # about an entry until the book is flat again.
        self.book_faithful: bool = True
        # Client order id / venue order id -> the most a fill booked into
        # the book reported filled of that order (see
        # :meth:`~BrokerPosition.fill_cursor`): what a restart compares the
        # journal's fill cursors with to prove the stored book saw every fill.
        self._fill_cursors: dict[str, float] = {}
        # How many times a booked fill or an unbooked reduction moved the
        # book in this process: a reading of the venue taken at one count is
        # comparable with the book only while the count stays the same.
        self.book_moves: int = 0
        # The venue's least tradable quantity at a price, in the book's units
        # (``BrokerPlugin.least_tradable_qty``); ``None`` keeps every remainder.
        self.least_tradable_qty: Callable[[float], float] | None = None

        self.entry_orders: dict[str | None, 'Order'] = {}
        # Composite key ``(exit_id, from_entry)`` mirrors
        # :class:`~pynecore.lib.strategy.SimPosition.exit_orders`. Single-field
        # keys collide on partial-TP fan-out (multiple exits for one entry)
        # and on ``from_entry=na`` fan-out (one exit_id, many per-entry rows).
        self.exit_orders: dict['_ExitOrderKey', 'Order'] = {}

        # Close keys (``(exit_id, order_id)``) already issued in the current
        # script evaluation; reset by :meth:`begin_evaluation`. See
        # :meth:`_add_order` for the same-eval netting it drives.
        self._closes_this_eval: set['_ExitOrderKey'] = set()

        # === Risk management state ===
        # Configuration (filled by the ``strategy.risk.*`` setters via
        # ``__init__.py``'s shared lib-property bridge):
        self.risk_allowed_direction: 'direction.Direction | None' = None
        self.risk_max_drawdown_value: float | None = None
        self.risk_max_drawdown_type: 'QtyType | None' = None
        self.risk_max_drawdown_alert: str | None = None
        self.risk_max_intraday_loss_value: float | None = None
        self.risk_max_intraday_loss_type: 'QtyType | None' = None
        self.risk_max_intraday_loss_alert: str | None = None
        self.risk_max_cons_loss_days: int | None = None
        self.risk_max_cons_loss_days_alert: str | None = None
        self.risk_max_intraday_filled_orders: int | None = None
        self.risk_max_intraday_filled_orders_alert: str | None = None
        self.risk_max_position_size: float | None = None
        # Runtime counters / day-rollover tracking:
        self.risk_cons_loss_days: int = 0
        self.risk_last_trading_day: int = -1
        self.risk_last_day_equity: float = 0.0
        self.risk_intraday_filled_orders: int = 0
        self.risk_intraday_start_equity: float = 0.0
        self.risk_halt_trading: bool = False
        self.risk_halt_day: int = -1

        self._current_price: float = 0.0

    # === Pine API compatibility shims ======================================
    # Pine strategy.* functions read ``position.c`` / ``.o`` / ``.h`` / ``.l``
    # for the simulator's creation-time margin check. In broker mode those
    # attributes are served from the live OHLCV module; the exchange enforces
    # margin for real, so the Pine-level check still acts as a safety net
    # on script-side state without a separate simulator update path.

    @property
    def c(self) -> float:
        try:
            v = lib.close
        except AttributeError:
            return self._current_price or 0.0
        try:
            return float(v) if v is not None else self._current_price or 0.0
        except (TypeError, ValueError):
            return self._current_price or 0.0

    @property
    def o(self) -> float:
        try:
            return float(lib.open)
        except (AttributeError, TypeError, ValueError):
            return self.c

    @property
    def h(self) -> float:
        try:
            return float(lib.high)
        except (AttributeError, TypeError, ValueError):
            return self.c

    @property
    def l(self) -> float:  # noqa: E743 — mirrors the Pine attribute name
        try:
            return float(lib.low)
        except (AttributeError, TypeError, ValueError):
            return self.c

    # === Pine-side order book ===

    def begin_evaluation(self) -> None:
        """Mark the start of a fresh script evaluation (one ``main()`` run).

        Called by the runner before the libraries / ``main`` execute, in broker
        mode only. It clears the per-evaluation close-key set so that two
        ``strategy.close()`` calls issued in the SAME evaluation net onto one
        order, while the SAME close re-issued on the next ``calc_on_every_tick``
        evaluation replaces the pending order instead of doubling it. There is
        no other per-evaluation order-book reset in live mode (the Pine order
        book is purely event-driven), so this is the netting's idempotency
        anchor.
        """
        self._closes_this_eval.clear()

    def _add_order(self, order: 'Order') -> None:
        """Register an order locally (the sync engine forwards it to the exchange).

        Pre-submit risk gates run before the order is enqueued — same policy
        as :meth:`SimPosition.fill_order` enforces at fill time, but applied
        at the submit boundary because the broker fill is asynchronous.
        Rejected entry/normal orders are silently dropped (matching the sim
        ``_remove_order`` behavior on cap/direction violation); the
        :attr:`risk_halt_trading` flag is set out-of-band by
        :meth:`_enforce_post_bar_risk`.
        """
        order.bar_index = int(lib.bar_index)
        # noinspection PyProtectedMember
        from pynecore.lib.strategy import (
            _exit_order_key, _order_type_close, _order_type_entry, _order_type_normal,
        )
        if order.order_type in (_order_type_entry, _order_type_normal):
            if self._is_intraday_filled_cap_reached():
                return
            adjusted = self._adjust_for_max_position_size(float(order.size), order.sign)
            if adjusted is None:
                return
            order.size = adjusted
            if self.size == 0.0 and not self._is_direction_allowed(order.sign):
                return
        if order.order_type == _order_type_close:
            key = _exit_order_key(order)
            existing = self.exit_orders.get(key)
            # Netting is for market closes only (``strategy.close(id)`` /
            # ``strategy.close_all()``), identified by their reserved exit-id
            # patterns. A sticky ``strategy.exit`` bracket re-issued in the SAME
            # evaluation must still REPLACE: summing its size would dispatch an
            # oversized protective order and the first leg's stale limit/stop/
            # trailing levels would survive (netting only carries metadata).
            exit_id = order.exit_id
            is_market_close = exit_id == CLOSE_ALL_EXIT_ID or (
                exit_id is not None and exit_id.startswith(CLOSE_EXIT_ID_PREFIX)
            )
            if is_market_close and key in self._closes_this_eval and existing is not None:
                # Second+ same-key close THIS evaluation: net the slices into
                # one reduce-only market close. Both ``strategy.close`` qty
                # expressions are evaluated against the same ``position.size``
                # (no fill lands mid-evaluation), so the slices simply sum; the
                # over-close cap is applied later by the sync engine. Metadata
                # is last-wins, matching the prior overwrite behaviour for this
                # collision class. A NEXT-evaluation re-issue takes the ``else``
                # branch (``begin_evaluation`` cleared the key) and replaces the
                # pending order, keeping calc_on_every_tick idempotent.
                existing.size += order.size
                existing.reserved_size = abs(existing.size)
                existing.comment = order.comment
                existing.alert_message = order.alert_message
            else:
                self._closes_this_eval.add(key)
                self.exit_orders[key] = order
        else:
            self.entry_orders[order.order_id] = order

    def _remove_order(self, order: 'Order') -> None:
        """Cancel an order locally."""
        order.cancelled = True
        # noinspection PyProtectedMember
        from pynecore.lib.strategy import _exit_order_key, _order_type_close
        if order.order_type == _order_type_close:
            self.exit_orders.pop(_exit_order_key(order), None)
        else:
            self.entry_orders.pop(order.order_id, None)

    def _remove_order_by_id(self, order_id: str) -> None:
        # TV-verified semantics: ``strategy.cancel(id)`` matches an exit by
        # its ``exit_id`` and an entry by its entry id; no cross-matching.
        for exit_order in list(self.exit_orders.values()):
            if exit_order.exit_id == order_id:
                self._remove_order(exit_order)
        entry = self.entry_orders.get(order_id)
        if entry is not None:
            self._remove_order(entry)

    def _cancel_all_orders(self) -> None:
        # No ``orderbook`` attribute — that lives on ``SimPosition`` and drives
        # the simulator's price-keyed fill loop, which has no analog in live
        # trading. Clearing the two Pine-side dicts is enough; the next
        # ``OrderSyncEngine.sync()`` diffs against ``_active_intents`` and
        # dispatches a per-id cancel for every previously tracked intent.
        self.entry_orders.clear()
        self.exit_orders.clear()

    # === Restart-time Pine-side reconstruction ===

    def reconstruct_exit_order(
            self,
            *,
            pine_id: str,
            from_entry: str,
            side: str,
            qty: float,
            tp_price: float | None,
            sl_price: float | None,
            trail_price: float | None,
            trail_offset: float | None,
            oca_name: str | None = None,
            oca_type: str | None = None,
    ) -> None:
        """Re-install a persistent ``strategy.exit`` bracket order after a restart.

        Pine ``strategy.exit`` orders are persistent: once placed they live in
        :attr:`exit_orders` across bars (the script need not re-emit them) until
        they fill or are cancelled. A fresh process starts with empty order
        dicts, so the bracket the previous run armed is invisible to
        :func:`~pynecore.core.broker.intent_builder.build_intents` until it is
        rebuilt here from the durable broker-side ledger (the one-way
        bracket-ownership rows or the engine-trigger partial-leg ledger). The
        :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine` calls this
        once at startup so the diff sees the same exit it saw before the crash
        and adopts (rather than tears down) the live broker protection.

        ``side`` is the CLOSE side (``"buy"``/``"sell"``); the stored
        :class:`~pynecore.lib.strategy.Order` carries the opposite-of-position
        signed size so ``build_intents`` re-derives the same side. Tick fields
        are left ``None`` — the ledger persists resolved absolute prices, never
        the original tick distances.

        :param pine_id: The ``strategy.exit(id=...)`` value (the exit id).
        :param from_entry: The parent entry id the exit protects.
        :param side: The exit (close) side, ``"buy"`` or ``"sell"``.
        :param qty: Exit quantity magnitude.
        :param tp_price: Absolute take-profit price, or ``None``.
        :param sl_price: Absolute stop-loss price, or ``None``.
        :param trail_price: Absolute trailing-stop activation price, or ``None``.
        :param trail_offset: Trailing-stop offset (price units), or ``None``.
        :param oca_name: OCA group name, or ``None``.
        :param oca_type: OCA type string (``"reduce"`` / ``"cancel"`` /
            ``"none"``), or ``None`` for rows persisted before the OCA fields
            existed. Restored so ``build_intents`` re-derives the same group the
            exit was emitted under and the cross-bracket OCA-cancel cascade keeps
            firing across the restart.
        """
        # noinspection PyProtectedMember
        from pynecore.lib.strategy import Order, _exit_order_key, _order_type_close, oca as _oca
        signed_size = qty if side == "buy" else -qty
        order = Order(
            from_entry,
            signed_size,
            order_type=_order_type_close,
            exit_id=pine_id,
            limit=tp_price,
            stop=sl_price,
            trail_price=trail_price,
            trail_offset=trail_offset,
            oca_name=oca_name,
            oca_type=_oca.Oca(oca_type) if oca_type is not None else None,
        )
        self.exit_orders[_exit_order_key(order)] = order

    def reconstruct_entry_order(
            self,
            *,
            pine_id: str,
            side: str,
            qty: float,
            limit: float | None,
            stop: float | None,
    ) -> None:
        """Re-install a persistent ``strategy.entry`` working order after a restart.

        Pine ``strategy.entry`` / ``strategy.order`` working orders are
        persistent: once placed a pending LIMIT / STOP entry lives in
        :attr:`entry_orders` across bars (the script need not re-emit it) until
        it fills or is cancelled. A fresh process starts with empty order dicts,
        so a first-bar ``strategy.cancel`` after a restart would operate on an
        empty book (a silent no-op) and the live broker working order would be
        stranded — neither adopted nor cancelled. The
        :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine` rebuilds the
        entry here, once, from the live broker snapshot so the first post-restart
        script sees the same order it placed: a ``strategy.cancel`` removes it and
        the diff retires the live order, while leaving it standing (or re-emitting
        it) adopts the live order without a duplicate dispatch.

        ``side`` is the entry side (``"buy"``/``"sell"``); the stored
        :class:`~pynecore.lib.strategy.Order` carries the matching signed size so
        :func:`~pynecore.core.broker.intent_builder.build_intents` re-derives the
        same intent. Tick fields are left ``None`` — the venue reports resolved
        absolute prices, never the original tick distances.

        :param pine_id: The ``strategy.entry(id=...)`` value (the entry id).
        :param side: The entry side, ``"buy"`` or ``"sell"``.
        :param qty: Entry quantity magnitude.
        :param limit: Absolute limit price, or ``None`` (STOP entry).
        :param stop: Absolute stop trigger price, or ``None`` (LIMIT entry).
        """
        # noinspection PyProtectedMember
        from pynecore.lib.strategy import Order, _order_type_entry
        signed_size = qty if side == "buy" else -qty
        order = Order(
            pine_id,
            signed_size,
            order_type=_order_type_entry,
            limit=limit,
            stop=stop,
        )
        self.entry_orders[pine_id] = order

    def reconstruct_parent_trade(
            self,
            *,
            entry_id: str,
            size: float,
            entry_price: float,
    ) -> None:
        """Seed one open :class:`~pynecore.lib.strategy.Trade` for an adopted parent.

        Startup adoption restores :attr:`size`/:attr:`avg_price` but leaves
        :attr:`open_trades` empty. A partial-quantity bracket needs the parent's
        open qty to classify
        :attr:`~ExitIntent.is_partial_qty_bracket`
        consistently across bars (``build_intents`` derives the parent total
        from :attr:`open_trades`); without it the exit would be misclassified as
        a whole-row bracket and adopted on the wrong dispatch path. The trade is
        inert until a real broker fill routes through :meth:`record_fill` (the
        sole :attr:`open_trades` mutator), so seeding it cannot emit an intent or
        re-open anything. The parent also gets its binding, so the binding book
        keeps adding up to the adopted size; the book is then no longer
        faithful (:attr:`~BrokerPosition.book_faithful`): the adoption
        attributes the exposure FIFO, not by the origin of its fills.

        :param entry_id: The parent entry id (``from_entry``).
        :param size: Signed parent open size (positive long, negative short).
        :param entry_price: Parent average entry price.
        """
        if self.size > self.max_contracts_held_long:
            self.max_contracts_held_long = self.size
        elif -self.size > self.max_contracts_held_short:
            self.max_contracts_held_short = -self.size
        if any(t.entry_id == entry_id for t in self.open_trades):
            return
        # ``entry_equity`` stays 0.0: the parent opened in a prior process so
        # its true entry equity is unknowable, and startup adoption runs in
        # ``start_broker`` before the script is attached (``self.equity`` reads
        # ``lib._script``, not yet set). The broker close path never divides by
        # a trade's entry equity, so the placeholder is inert.
        self.open_trades.append(Trade(
            size=size,
            entry_id=entry_id,
            entry_bar_index=int(getattr(lib, 'bar_index', 0)),
            entry_time=0,
            entry_price=entry_price,
            commission=0.0,
            entry_equity=0.0,
        ))
        if not any(binding.entry_id == entry_id for binding in self._entry_book):
            self._bind_fill(entry_id, None, size, entry_price)
            self.book_faithful = False

    # === Exchange-side state updates ===

    def is_non_reducing_exit(self, event: 'OrderEvent') -> bool:
        """
        Whether an exit-leg fill would not reduce the booked position.

        The plugin contract delivers exit legs reduce-only, so on the venue such a
        fill reduced exposure the book no longer holds on that side: the book is
        flat or holds the fill's own side. :meth:`record_fill` does not book it.

        :param event: A fill event with ``fill_qty`` populated.
        :return: ``True`` for an exit leg (close, TP, SL, trailing) whose fill
            would open or grow a position in the book.
        """
        if event.leg_type not in _EXIT_LEG_TYPES:
            return False
        fill_qty = event.fill_qty or 0.0
        signed_delta = fill_qty if event.order.side == "buy" else -fill_qty
        return not self.size * signed_delta < 0.0

    def record_fill(self, event: 'OrderEvent', *, close_entry_id: str | None = None) -> bool:
        """
        Record an exchange fill.

        An entry fill opens a binding (a further slice of the same venue order
        grows it). A closing fill settles the trade rows FIFO and, apart from
        them, the binding book by its origin (see :meth:`_settle_bindings`): a
        take-profit, stop-loss or trailing leg settles the bindings of its
        ``from_entry`` (of its ``pine_id`` when the venue labels the leg with
        the parent entry's own id and no ``from_entry``); a close the caller
        attributes through ``close_entry_id`` settles that id's bindings; every
        other reduction walks the whole book. The entry ids of the bindings it
        spent are left in :attr:`~BrokerPosition.last_spent_entry_ids`.

        :param event: An :class:`OrderEvent` with ``fill_qty`` and
            ``fill_price`` populated, plus Pine identity fields
            (``pine_id``, ``from_entry``, ``leg_type``) filled by the plugin.
        :param close_entry_id: The entry id a closing fill other than a
            take-profit, stop-loss or trailing leg is bound to: the id of a
            ``strategy.close(id)``, of an engine-made close aimed at one entry,
            or the owner of a venue's per-position protective close. ``None``
            books the fill as an unbound reduction.
        :return: ``True`` if the position side changed as a result of this fill.
        """
        self.last_spent_entry_ids = []
        self._keep_book_in_step()
        fill_qty = event.fill_qty or 0.0
        fill_price = event.fill_price or 0.0
        if fill_qty <= 0.0 or fill_price <= 0.0:
            # A zero-or-missing qty/price means the broker plugin emitted a
            # fill event without resolving the actual fill quantity / price.
            # Without a warning the silent skip leaves ``position.size``
            # stuck at zero — the script keeps thinking it is flat and
            # re-fires the entry next bar.  Surface the offending event so
            # the operator can spot the broker-side data hole instead of
            # debugging by guesswork.
            _blog_warning(
                "ignoring fill with zero qty/price (qty=%s price=%s pine=%r leg=%s) "
                "— position.size NOT updated",
                fill_qty, fill_price,
                event.pine_id, event.leg_type,
            )
            return False

        signed_delta = fill_qty if event.order.side == "buy" else -fill_qty
        old_sign = self.sign
        new_size = self.size + signed_delta

        # A late or out-of-order exit fill must not open or grow a position the
        # venue may not hold: the diff engine could then "close" it with a real
        # order. The sync engine settles such a fill against the venue position.
        if self.is_non_reducing_exit(event):
            return False
        self._advance_fill_cursor(event, fill_qty)
        self.book_moves += 1

        # Commission bookkeeping — realized fee becomes part of net P&L at close
        fee = event.fee

        # Risk management: count every filled order toward the intraday cap,
        # matching ``SimPosition._fill_order``. The counter is read by the
        # pre-submit gate in :meth:`_add_order` and the post-bar halt in
        # :meth:`_enforce_post_bar_risk`.
        self.risk_intraday_filled_orders += 1

        if self.size == 0.0 or (old_sign * signed_delta) > 0.0:
            # Opening or adding to an existing position (same direction)
            new_abs = abs(new_size)
            old_abs = abs(self.size)
            if old_abs == 0.0 or not (self.avg_price == self.avg_price):
                self.avg_price = fill_price
            else:
                self.avg_price = (self.avg_price * old_abs + fill_price * fill_qty) / new_abs
            self.size = new_size
            self.sign = 1.0 if new_size > 0.0 else (-1.0 if new_size < 0.0 else 0.0)
            if self.size > self.max_contracts_held_long:
                self.max_contracts_held_long = self.size
            elif -self.size > self.max_contracts_held_short:
                self.max_contracts_held_short = -self.size

            trade = Trade(
                size=signed_delta,
                entry_id=event.pine_id,
                entry_bar_index=int(getattr(lib, 'bar_index', 0)),
                entry_time=int(event.timestamp * 1000.0),
                entry_price=fill_price,
                commission=fee,
                entry_comment=None,
                entry_equity=self.equity,
            )
            self.open_trades.append(trade)
            self._bind_fill(event.pine_id, event.order.id, signed_delta, fill_price)
            self.open_commission += fee
            # The entry Order stays in ``entry_orders`` for intent stability, but
            # its filled slice now lives in ``open_trades``. Record how much of it
            # has filled so ``strategy.exit``'s bound-size reservation does not
            # count the same quantity twice (issue BYBIT-001).
            entry_order = self.entry_orders.get(event.pine_id)
            if entry_order is not None:
                entry_order.filled_qty += fill_qty
            return False

        # Reducing or flipping — FIFO close of existing trades
        reduced = min(fill_qty, abs(self.size))
        remaining = fill_qty
        closed_profit = 0.0
        closed_fee = 0.0
        while remaining > 0.0 and self.open_trades:
            trade = self.open_trades[0]
            trade_abs = abs(trade.size)
            if trade_abs <= remaining + 1e-12:
                # Close this trade fully
                self._close_trade(trade, fill_price, event, fee_share=fee * (trade_abs / fill_qty))
                closed_profit += trade.profit
                closed_fee += trade.commission
                remaining -= trade_abs
            else:
                # Partial close: split the trade
                residue = trade_abs - remaining
                floor = self._untradable_floor(residue, fill_price, len(self.open_trades))
                absorb = floor is not None
                closed_piece = Trade(
                    size=trade.sign * remaining,
                    entry_id=trade.entry_id,
                    entry_bar_index=trade.entry_bar_index,
                    entry_time=trade.entry_time,
                    entry_price=trade.entry_price,
                    commission=(trade.commission if absorb
                                else trade.commission * (remaining / trade_abs)),
                    entry_comment=trade.entry_comment,
                    entry_equity=trade.entry_equity,
                )
                closed_entry_commission = closed_piece.commission
                self._close_trade(
                    closed_piece, fill_price, event,
                    fee_share=fee * (remaining / fill_qty),
                )
                closed_profit += closed_piece.profit
                if absorb:
                    self.open_trades.remove(trade)
                    self._spread_residue(trade, residue, floor)
                else:
                    # Shrink the remaining open trade
                    trade.size -= closed_piece.size
                    trade.commission -= closed_entry_commission
                remaining = 0.0
        # A trade within 1e-12 of the remainder closes in full, so the
        # subtraction can leave float residue behind: no leftover exposure
        # (measured live: 8.7e-19 after an inverse close, bybit-inverse cycle 249).
        if remaining < 1e-12:
            remaining = 0.0

        self.size += signed_delta
        # Clamp tiny residuals to zero
        if abs(self.size) < 1e-12:
            self.size = 0.0
            self.sign = 0.0
            self.avg_price = na_float
        else:
            self.sign = 1.0 if self.size > 0.0 else -1.0

        if event.leg_type in _BOUND_LEG_TYPES and (event.from_entry or event.pine_id):
            # A venue with position-attached brackets (Capital.com) labels the
            # leg's fill with the parent entry's own ``pine_id`` and no
            # ``from_entry``.
            bound_entry_id: str | None = event.from_entry or event.pine_id
            if not any(binding.entry_id == bound_entry_id for binding in self._entry_book):
                _blog_warning(
                    "%s fill (pine=%r from_entry=%r qty=%s) names no open binding "
                    "of its entry — settled against the oldest bindings instead",
                    event.leg_type, event.pine_id, event.from_entry, reduced,
                )
        else:
            bound_entry_id = close_entry_id
        self._settle_bindings(bound_entry_id, reduced, fill_price)

        # If there is leftover qty after closing all open_trades → side flip.
        # On a one-way account only an ENTRY leg may legitimately open the
        # opposite side (stop-and-reverse). A reduce-only/exit leg (close, TP,
        # SL, trailing) must NEVER flip: leftover qty means the local FIFO
        # under-counted exposure (e.g. a restart adopted the net size without
        # seeding open_trades). The authoritative net is already in ``self.size``
        # (``self.size += signed_delta`` above), so two cases split apart:
        #   * the net kept its original sign → the exit only PARTIALLY reduced;
        #     the leftover is a stale FIFO row shortfall, NOT an over-close.
        #     Keep the (already-correct) residual size — clamping to flat would
        #     drop live broker exposure and let the script re-fire the entry.
        #   * the net reached or crossed zero → the exit closed at least the
        #     whole position; reconcile toward flat instead of fabricating an
        #     opposite position the diff engine could then "close" with a real
        #     reversing order.
        if remaining > 0.0:
            if event.leg_type in _EXIT_LEG_TYPES:
                if old_sign * self.size > 0.0:
                    # Partial reduce-only fill against an under-counted FIFO:
                    # the residual already in ``self.size`` is authoritative.
                    _blog_warning(
                        "exit fill (leg=%s qty=%s) under-counted FIFO exposure "
                        "by %s — keeping residual size %s (partial reduce, "
                        "not an over-close)",
                        event.leg_type, fill_qty, remaining, self.size,
                    )
                else:
                    _blog_warning(
                        "exit fill (leg=%s qty=%s) exceeded known FIFO exposure "
                        "by %s — clamping to flat instead of opening an opposite "
                        "position",
                        event.leg_type, fill_qty, remaining,
                    )
                    self.size = 0.0
                    self.sign = 0.0
                    self.avg_price = na_float
            else:
                new_size = self.sign * remaining if self.sign != 0.0 else signed_delta
                self.size = new_size
                self.sign = 1.0 if new_size > 0.0 else (-1.0 if new_size < 0.0 else 0.0)
                if self.size > self.max_contracts_held_long:
                    self.max_contracts_held_long = self.size
                elif -self.size > self.max_contracts_held_short:
                    self.max_contracts_held_short = -self.size
                self.avg_price = fill_price
                flipped = Trade(
                    size=new_size,
                    entry_id=event.pine_id,
                    entry_bar_index=int(getattr(lib, 'bar_index', 0)),
                    entry_time=int(event.timestamp * 1000.0),
                    entry_price=fill_price,
                    commission=0.0,
                    entry_comment=None,
                    entry_equity=self.equity,
                )
                self.open_trades.append(flipped)
                # Every trade row is closed, so the flipped trade is the whole
                # position and its binding the whole book.
                self.clear_entry_book()
                self._bind_fill(event.pine_id, event.order.id, new_size, fill_price)

        if self.size == 0.0:
            self.clear_entry_book()

        self._count_close_profit(closed_profit, 1)

        self.open_commission = float(sum(t.commission for t in self.open_trades))

        return self.sign != old_sign

    def _untradable_floor(self, residue: float, price: float, holders: int) -> float | None:
        """The venue's least tradable quantity, when a remainder falls below it.

        A partially closed trade's remainder below the venue's least tradable
        quantity can never be closed by an order of its own: its exit could
        only be resized to a quantity the venue rounds to nothing. Pyramid legs
        that open with the same venue quantity book slightly different base
        quantities on an inverse contract (whole USD contracts converted at
        each fill's price), so a FIFO close of one leg leaves such a remainder
        on another. The remainder moves only when another trade stays open to
        take it; the last trade keeps it for the flat-book dust handling. A
        partially settled binding's remainder follows the same rule.

        :param residue: What the trade would keep, unsigned.
        :param price: The closing fill's price; ``0.0`` when the reduction has
            no price, which keeps every remainder.
        :param holders: The open trades (or bindings) the remainder is one of.
        :return: The least tradable quantity when ``residue`` is below it and
            another trade can take it, otherwise ``None``.
        """
        if (residue <= 0.0 or holders < 2 or price <= 0.0
                or self.least_tradable_qty is None):
            return None
        floor = self.least_tradable_qty(price)
        return floor if residue < floor else None

    def _spread_residue(self, source: Trade, residue: float, floor: float) -> None:
        """Spread a trade's untradable remainder over the other open trades.

        Each open trade takes a share in proportion to its size, so the trades
        still add up to the venue position. A trade's entry price becomes the
        volume-weighted price of its own quantity and its share, so the share
        keeps the source trade's entry cost and the later closes book the P&L
        of the fills.

        :param source: The trade the remainder came from, already removed
            from :attr:`open_trades`.
        :param residue: The remainder, unsigned.
        :param floor: The least tradable quantity it fell below.
        """
        total = sum(abs(trade.size) for trade in self.open_trades)
        given = 0.0
        last = len(self.open_trades) - 1
        for i, trade in enumerate(self.open_trades):
            share = residue - given if i == last else residue * abs(trade.size) / total
            given += share
            held = abs(trade.size)
            trade.entry_price = ((trade.entry_price * held + source.entry_price * share)
                                 / (held + share))
            trade.size += source.sign * share
        _blog_info(
            "closing fill left %s of entry %r below the least tradable %s — "
            "spread over the open trades",
            residue, source.entry_id, floor,
        )

    # === Binding book ===

    @property
    def bindings(self) -> Sequence[_EntryBinding]:
        """The open bindings, oldest first; read-only for callers."""
        return self._entry_book

    def bound_exposure(self, entry_id: str | None) -> tuple[float, float]:
        """Open bound quantity and side of an entry id's bindings.

        What an exit or close of the id may still take off the position: the
        id's own bound quantity, not its trade rows, which a fill of another
        entry may already have consumed FIFO.

        :param entry_id: The entry id.
        :return: ``(qty, sign)``: the summed bound quantity of the id's
            bindings (:class:`~pynecore.lib.strategy._EntryBinding`), unsigned, and
            ``+1.0`` long / ``-1.0`` short; ``(0.0, 0.0)`` once every binding
            of the id is spent.
        """
        qty = 0.0
        sign = 0.0
        if entry_id is None:
            return qty, sign
        for binding in self._entry_book:
            if binding.entry_id == entry_id:
                qty += binding.bound
                sign = binding.sign
        return qty, sign

    def open_entry_count(self) -> int:
        """Entries counted against ``pyramiding``: one per open binding.

        An entry keeps its slot until its own bound quantity is spent, even
        after a fill of another entry consumed its trade rows FIFO.
        """
        return self._pyramid_count()

    def _keep_book_in_step(self) -> None:
        """Rebuild the binding book from the trade rows when it no longer adds up.

        Every fill and every reduction the position takes moves the bindings by
        the same quantity as the trade rows and the size, so the bound
        quantities add up to the position size up to float noise. The venue's
        least tradable quantity governs single remainders, which move onto
        the other bindings without changing the sum (:meth:`_reduce_binding`).
        A larger difference is an invariant breach: a change of the rows and
        the size reached the position without its bindings. It is reported,
        and while the rows still add up to the size the book is rebuilt from
        them, one binding per row, so every held quantity keeps an entry its
        exits and closes are bound to (FIFO, like the rows) instead of being
        judged unprotected exposure. The rebuilt book is FIFO-shaped, so it is
        no longer faithful (:attr:`~BrokerPosition.book_faithful`). Rows that
        do not add up to the size either (an adopted size the rows were not
        seeded for) are left alone.
        """
        size = abs(self.size)
        tolerance = 1e-9 * max(size, 1.0)
        bound = sum(binding.bound for binding in self._entry_book)
        if abs(bound - size) <= tolerance:
            return
        held = sum(abs(trade.size) for trade in self.open_trades)
        if abs(held - size) > tolerance:
            return
        _blog_warning(
            "binding book invariant breached: the bindings hold %s, the "
            "position %s and its %d open trade rows %s — rebuilt from the rows",
            bound, self.size, len(self.open_trades), held,
        )
        self._entry_book.clear()
        self._order_bindings.clear()
        self.book_faithful = False
        for trade in self.open_trades:
            self._entry_seq += 1
            self._entry_book.append(_EntryBinding(
                self._entry_seq, trade.entry_id, trade.size, trade.entry_price,
            ))

    def _bind_fill(self, entry_id: str | None, order_id: str | None,
                   size: float, price: float) -> None:
        """Book an entry fill in the binding book.

        A further slice of an entry order whose binding is still open grows
        that binding: one entry order is one binding however many fills the
        venue reports for it. Every other fill opens a new binding.

        :param entry_id: The entry id the fill opened.
        :param order_id: The venue order id of the fill, or ``None`` when it
            has none (an adopted parent).
        :param size: The signed quantity the fill opened.
        :param price: The fill price.
        """
        binding = self._order_bindings.get(order_id) if order_id is not None else None
        if (binding is not None and binding in self._entry_book
                and binding.entry_id == entry_id and binding.sign * size > 0.0):
            qty = abs(size)
            binding.entry_price = ((binding.entry_price * binding.init_size + price * qty)
                                   / (binding.init_size + qty))
            binding.init_size += qty
            binding.bound += qty
            return
        self._entry_seq += 1
        binding = _EntryBinding(self._entry_seq, entry_id, size, price)
        self._entry_book.append(binding)
        if order_id is not None:
            self._order_bindings[order_id] = binding

    def _settle_bindings(self, entry_id: str | None, qty: float, price: float) -> None:
        """Settle a closing fill's quantity against the binding book.

        The fill settles the bindings of ``entry_id`` oldest first; an unbound
        fill (``entry_id`` ``None``) walks the whole book oldest first. What a
        bound fill finds no binding of its id for still reduces the position
        on the venue, so it walks the rest of the book oldest first and the
        bound quantities keep adding up to the position size.

        :param entry_id: The entry id the fill is bound to, or ``None``.
        :param qty: The quantity the fill took off the position, unsigned.
        :param price: The fill price.
        """
        if entry_id is not None:
            for binding in list(self._entry_book):
                if qty <= 1e-12:
                    return
                if binding.entry_id == entry_id:
                    qty = self._reduce_binding(binding, qty, price)
        for binding in list(self._entry_book):
            if qty <= 1e-12:
                return
            qty = self._reduce_binding(binding, qty, price)

    def _reduce_binding(self, binding: _EntryBinding, qty: float, price: float) -> float:
        """Take up to ``qty`` off one binding; return what it could not absorb.

        A binding whose remainder falls below the venue's least tradable
        quantity is spent as well: the remainder moves onto the other bindings
        in proportion to their bound quantity, the way
        :meth:`_spread_residue` moves a trade row's.

        :param binding: The binding to reduce.
        :param qty: The quantity to take, unsigned.
        :param price: The closing fill's price.
        :return: The part of ``qty`` the binding could not absorb.
        """
        take = min(binding.bound, qty)
        binding.bound -= take
        if binding.bound <= 1e-12:
            self._drop_binding(binding)
            return qty - take
        floor = self._untradable_floor(binding.bound, price, len(self._entry_book))
        if floor is not None:
            residue = binding.bound
            self._drop_binding(binding)
            total = sum(b.bound for b in self._entry_book)
            given = 0.0
            last = len(self._entry_book) - 1
            for i, other in enumerate(self._entry_book):
                share = residue - given if i == last else residue * other.bound / total
                given += share
                other.bound += share
        return qty - take

    def _drop_binding(self, binding: _EntryBinding) -> None:
        """Retire a spent binding and note its entry id in :attr:`~BrokerPosition.last_spent_entry_ids`."""
        self._entry_book.remove(binding)
        for order_id in [k for k, v in self._order_bindings.items() if v is binding]:
            del self._order_bindings[order_id]
        entry_id = binding.entry_id
        if entry_id is not None and entry_id not in self.last_spent_entry_ids:
            self.last_spent_entry_ids.append(entry_id)

    def clear_entry_book(self) -> list[str]:
        """Spend every binding: the position holds nothing they could be bound to.

        Their entry ids are added to
        :attr:`~BrokerPosition.last_spent_entry_ids` as well. The empty book is
        faithful again (:attr:`~BrokerPosition.book_faithful`).

        :return: The distinct entry ids of the bindings spent, oldest first.
        """
        spent: list[str] = []
        for binding in list(self._entry_book):
            self._drop_binding(binding)
            if binding.entry_id is not None and binding.entry_id not in spent:
                spent.append(binding.entry_id)
        self._order_bindings.clear()
        self.book_faithful = True
        return spent

    def settle_unbooked_reduction(self, qty: float, *, entry_id: str | None = None,
                                  price: float = 0.0) -> list[str]:
        """Settle a reduction the trade rows took without a booked fill.

        A reduction the caller applies to the rows or the size by itself (a
        close the venue settled while no trade rows were held, a correction a
        snapshot proved) settles the binding book the way
        :meth:`record_fill` settles a closing fill, so the bound quantities
        keep adding up to the position size.

        :param qty: The quantity taken off the position, unsigned.
        :param entry_id: The entry id the reduction is bound to; ``None``
            walks the whole book oldest first.
        :param price: The reduction's price; ``0.0`` when it has none.
        :return: The entry ids of the bindings it spent, oldest first (also
            left in :attr:`~BrokerPosition.last_spent_entry_ids`).
        """
        self.last_spent_entry_ids = []
        if qty > 0.0:
            self.book_moves += 1
            self._settle_bindings(entry_id, qty, price)
        return list(self.last_spent_entry_ids)

    def _advance_fill_cursor(self, event: 'OrderEvent', fill_qty: float) -> None:
        """Note how much of the fill's order the book has now taken in.

        Kept under the order's client order id and its venue order id: the
        sum of the fills booked for it, whichever of its ids each one carried.
        Only the slices the book applied count, never the cumulative filled
        quantity the plugin reports on the order: that snapshot can already
        include a slice whose own event is not booked yet, and a cursor
        credited with it would let a restart judge a book that missed the
        slice as complete.

        :param event: The fill being booked.
        :param fill_qty: Its booked quantity.
        """
        order = event.order
        keys = [key for key in (order.client_order_id, order.id) if key]
        booked = max((self._fill_cursors.get(key, 0.0) for key in keys), default=0.0) + fill_qty
        for key in keys:
            self._fill_cursors[key] = booked

    def fill_cursor(self, keys: Sequence[str | None]) -> float:
        """The most the book took in of an order known by any of ``keys``.

        :param keys: The order's client order id and venue order ids.
        :return: ``0.0`` when no fill of it was booked.
        """
        return max((self._fill_cursors.get(key, 0.0) for key in keys if key), default=0.0)

    def seed_fill_cursors(self, cursors: dict[str, float]) -> None:
        """Count the given orders as taken in up to the given quantities.

        For a book built from the journal rather than from fills (a startup
        adoption): it accounts for every quantity the journal shows at the
        time, so only a fill journaled later is one the book has not seen.

        :param cursors: Client / venue order id -> filled quantity.
        """
        for key, qty in cursors.items():
            if qty > self._fill_cursors.get(key, 0.0):
                self._fill_cursors[key] = qty

    def forget_fill_cursors(self) -> None:
        """Drop the fill cursors: a flat book owes no fill of any order."""
        self._fill_cursors.clear()

    def fill_cursor_keys(self) -> frozenset[str]:
        """The order ids (client and venue) the book keeps a fill cursor for."""
        return frozenset(self._fill_cursors)

    def drop_fill_cursors(self, keys: Iterable[str]) -> None:
        """Drop the fill cursors of the given order ids; unknown ids are ignored.

        :param keys: Client / venue order ids.
        """
        for key in keys:
            self._fill_cursors.pop(key, None)

    def entry_book_snapshot(self) -> dict[str, Any]:
        """The whole binding book as a JSON-serializable dict.

        What a restart needs to resume the book exactly
        (:meth:`restore_entry_book`) and to judge whether it saw every fill:
        every open binding with the venue order ids whose further slices grow
        it, the binding sequence counter, the position size the book belongs
        to, whether the book is faithful
        (:attr:`~BrokerPosition.book_faithful`), and the fill cursors
        (:meth:`fill_cursor`).

        :return: ``{'size', 'entry_seq', 'faithful', 'bindings',
            'fill_cursors'}``; each binding is ``{'seq', 'entry_id',
            'init_size', 'bound', 'sign', 'entry_price', 'order_ids'}``,
            oldest first.
        """
        order_ids: dict[int, list[str]] = {}
        for order_id, binding in self._order_bindings.items():
            order_ids.setdefault(binding.seq, []).append(order_id)
        return {
            'size': float(self.size),
            'entry_seq': self._entry_seq,
            'faithful': self.book_faithful,
            'bindings': [
                {'seq': binding.seq, 'entry_id': binding.entry_id,
                 'init_size': binding.init_size, 'bound': binding.bound,
                 'sign': binding.sign, 'entry_price': float(binding.entry_price),
                 'order_ids': order_ids.get(binding.seq, [])}
                for binding in self._entry_book
            ],
            'fill_cursors': dict(self._fill_cursors),
        }

    @staticmethod
    def parse_entry_book_snapshot(payload: dict[str, Any]) -> EntryBookSnapshot | None:
        """Read back a book written by :meth:`entry_book_snapshot`.

        The payload also carries ``written_ms``, the wall-clock time of the
        write. Every number must be finite, every binding's initial size and
        bound quantity positive, and binding sequence numbers and venue order
        ids unique. The bound quantity may exceed the initial size: a spent
        binding's untradable remainder moves onto the other bindings
        (:meth:`_reduce_binding`) without changing their initial sizes.

        :param payload: The stored dict.
        :return: The parsed snapshot, or ``None`` when a field is missing, of
            the wrong type or out of range.
        """
        def number(value: Any) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            value = float(value)
            return value if math.isfinite(value) else None

        def integer(value: Any) -> int | None:
            return None if isinstance(value, bool) or not isinstance(value, int) else value

        size = number(payload.get('size'))
        entry_seq = integer(payload.get('entry_seq'))
        written_ms = integer(payload.get('written_ms'))
        faithful = payload.get('faithful')
        rows = payload.get('bindings')
        raw_cursors = payload.get('fill_cursors')
        if (size is None or entry_seq is None or written_ms is None
                or not isinstance(faithful, bool) or not isinstance(rows, list)
                or not isinstance(raw_cursors, dict)):
            return None
        fill_cursors: dict[str, float] = {}
        for key, raw in raw_cursors.items():
            qty = number(raw)
            if not isinstance(key, str) or qty is None or qty < 0.0:
                return None
            fill_cursors[key] = qty
        bindings: list[_EntryBinding] = []
        order_bindings: dict[str, int] = {}
        seqs: set[int] = set()
        for row in rows:
            if not isinstance(row, dict):
                return None
            seq = integer(row.get('seq'))
            entry_id = row.get('entry_id')
            init_size = number(row.get('init_size'))
            bound = number(row.get('bound'))
            sign = number(row.get('sign'))
            entry_price = number(row.get('entry_price'))
            order_ids = row.get('order_ids')
            if (seq is None or seq in seqs
                    or not (entry_id is None or isinstance(entry_id, str))
                    or init_size is None or init_size <= 0.0
                    or bound is None or bound <= 0.0 or sign not in (1.0, -1.0)
                    or entry_price is None or not isinstance(order_ids, list)
                    or not all(isinstance(order_id, str) and order_id not in order_bindings
                               for order_id in order_ids)
                    or len(set(order_ids)) != len(order_ids)):
                return None
            seqs.add(seq)
            binding = _EntryBinding(seq, entry_id, sign * init_size, entry_price)
            binding.bound = bound
            binding.sign = sign
            for order_id in order_ids:
                order_bindings[order_id] = len(bindings)
            bindings.append(binding)
        return EntryBookSnapshot(
            size=size, entry_seq=entry_seq, faithful=faithful, written_ms=written_ms,
            bindings=tuple(bindings), order_bindings=order_bindings,
            fill_cursors=fill_cursors,
        )

    def restore_entry_book(self, snapshot: EntryBookSnapshot) -> None:
        """Replace the binding book with a stored one, verbatim.

        The bindings, the venue order ids whose further slices grow them, the
        sequence counter, the fill cursors and
        :attr:`~BrokerPosition.book_faithful` come back as they were written;
        the trade rows and the size are left alone. Whether the stored book
        fits the position is the caller's to judge.

        :param snapshot: The book :meth:`parse_entry_book_snapshot` read back.
        """
        self._entry_book = list(snapshot.bindings)
        self._order_bindings = {
            order_id: snapshot.bindings[index]
            for order_id, index in snapshot.order_bindings.items()
        }
        self._entry_seq = max(
            [snapshot.entry_seq, *(binding.seq for binding in snapshot.bindings)])
        self._fill_cursors = dict(snapshot.fill_cursors)
        self.book_faithful = snapshot.faithful

    def record_provisional_close(self, event: 'OrderEvent') -> ProvisionalClose:
        """
        Record a closing fill whose price is only an estimate.

        Booked like :meth:`record_fill`; the returned record lets
        :meth:`reprice_provisional_closes` move it to a real fill's price later.

        :param event: A fee-free closing fill at the estimated price.
        :return: The booked close.
        :raises ValueError: When the event carries a fee.
        """
        if event.fee:
            raise ValueError("a provisional close carries no fee")
        stats_base = copy.deepcopy(self._closed_trade_stats)
        count_base = self.closed_trades_count
        closed_before = len(self.new_closed_trades)
        self.record_fill(event)
        trades = self.new_closed_trades[closed_before:]
        return ProvisionalClose(
            price=event.fill_price or 0.0, trades=trades,
            unpriced=[abs(trade.size) for trade in trades],
            entry_commissions=[trade.commission for trade in trades],
            profit=sum(trade.profit for trade in trades),
            stats_base=stats_base, count_base=count_base,
        )

    def reprice_provisional_closes(
            self, closes: list[ProvisionalClose], qty: float, event: 'OrderEvent',
    ) -> float:
        """
        Move ``qty`` of provisionally booked closes to the price of ``event``'s fill.

        The trades keep their FIFO attribution, which is what the same fill
        reported on time books as well. Walked oldest first, each trade takes
        the fill's price, exit id, exit time and its share of the fill's fee,
        and the P&L, the win / loss counters, the percent sums and the exit
        equity of every later closed trade move with it; the closed-trade
        statistics are rebuilt from the state before the oldest close. A trade
        the fill explains only in part stays one record whose exit price is
        the quantity-weighted blend of the fill's and the estimate.

        :param closes: The provisional closes, oldest first.
        :param qty: The quantity of ``event``'s fill the closes contain.
        :param event: The fill, its ``fill_qty`` the whole fill its fee is for.
        :return: The quantity re-priced (less than ``qty`` when the closes
            hold less still at their estimate).
        """
        fill_qty = event.fill_qty or 0.0
        fill_price = event.fill_price or 0.0
        if qty <= 0.0 or fill_qty <= 0.0 or fill_price <= 0.0 or not closes:
            return 0.0
        fee_rate = event.fee / fill_qty
        remaining = qty
        shifts: dict[int, float] = {}
        for close in closes:
            if remaining <= 1e-12:
                break
            old_profit = close.profit
            for i, trade in enumerate(close.trades):
                if remaining <= 1e-12:
                    break
                unpriced = close.unpriced[i]
                if unpriced <= 1e-12:
                    continue
                part = unpriced if unpriced <= remaining + 1e-12 else remaining
                trade_profit = trade.profit
                base_equity = trade.exit_equity - trade_profit
                self._book_profit_ratio(trade, -1)
                trade.exit_price += (fill_price - close.price) * part / abs(trade.size)
                trade.exit_id = event.pine_id or ""
                trade.exit_time = int(event.timestamp * 1000.0)
                trade.commission += fee_rate * part
                trade.profit = (trade.exit_price - trade.entry_price) * trade.size - trade.commission
                entry_cost = abs(trade.size) * trade.entry_price + close.entry_commissions[i]
                trade.profit_percent = (trade.profit / entry_cost * 100.0) if entry_cost else 0.0
                self._book_profit_ratio(trade, 1)
                trade.exit_equity = base_equity + trade.profit
                shifts[id(trade)] = trade.profit - trade_profit
                close.profit += trade.profit - trade_profit
                close.unpriced[i] = round(unpriced - part, 12)
                remaining = round(remaining - part, 12)
            if close.profit != old_profit:
                self._count_close_profit(old_profit, -1)
                self._count_close_profit(close.profit, 1)
        if shifts:
            self._replay_closed_since(closes[0], shifts)
        return round(qty - max(remaining, 0.0), 12)

    def _replay_closed_since(self, close: ProvisionalClose, shifts: dict[int, float]) -> None:
        """Carry re-priced profits into the trades closed since ``close`` and the statistics.

        :param close: The oldest provisional close.
        :param shifts: The profit change of each re-priced trade, by ``id``.
        """
        since = self.closed_trades_count - close.count_base
        tail = list(self.closed_trades)[-since:] if since > 0 else []
        stats: ClosedTradeStats | None = copy.deepcopy(close.stats_base)
        if since > len(tail):
            _blog_error(
                "re-priced closed trades: %s trades closed since the estimate, more than "
                "the %s kept — the cumulative closed-trade statistics keep the estimate",
                since, len(tail),
            )
            stats = None
        shift = 0.0
        for trade in tail:
            trade.exit_equity += shift
            shift += shifts.get(id(trade), 0.0)
            if stats is not None:
                stats.add(trade)
        if stats is not None:
            self._closed_trade_stats = stats

    def _count_close_profit(self, profit: float, sign: int) -> None:
        """Book (``sign`` 1) or unbook (-1) a closing fill's realized profit and its counter."""
        self.netprofit += sign * profit
        if profit > 0.0:
            self.grossprofit += sign * profit
            self.wintrades += sign
        elif profit < 0.0:
            self.grossloss += sign * profit
            self.losstrades += sign
        else:
            self.eventrades += sign

    def _book_profit_ratio(self, trade: Trade, sign: int) -> None:
        """Book (``sign`` 1) or unbook (-1) a closed trade in the percent sums."""
        ratio = trade.profit_percent / 100.0
        self.sum_profit_ratio += sign * ratio
        if trade.profit > 0.0:
            self.sum_win_profit_ratio += sign * ratio
        elif trade.profit < 0.0:
            self.sum_loss_profit_ratio += sign * ratio

    def flatten_nontradable_dust(
            self, fill_price: float, event: 'OrderEvent',
    ) -> list[str]:
        """Book every remaining open trade closed as non-tradable dust.

        Called by the sync engine when a closing fill leaves the aggregate
        below the venue's minimum tradable quantity: order quantities are
        grid-quantized, so no venue order can ever close the residual.
        Keeping the slice open would also keep its entry out of the
        closed-entry cleanup, leaving the entry's full-size protective
        legs resting against inventory the account no longer holds.

        The bindings still open are spent with it and added to
        :attr:`~BrokerPosition.last_spent_entry_ids`, after those of the closing
        fill.

        :param fill_price: Exit price the dust slices are booked at (the
            closing fill's own price).
        :param event: The closing :class:`OrderEvent` whose fill left the
            sub-step residual.
        :return: The distinct entry ids the flatten consumed, oldest first.
        """
        self.clear_entry_book()
        consumed: list[str] = []
        for trade in list(self.open_trades):
            self._close_trade(trade, fill_price, event, fee_share=0.0)
            profit = trade.profit
            self.netprofit += profit
            if profit > 0.0:
                self.grossprofit += profit
                self.wintrades += 1
            elif profit < 0.0:
                self.grossloss += profit
                self.losstrades += 1
            else:
                self.eventrades += 1
            if trade.entry_id is not None and trade.entry_id not in consumed:
                consumed.append(trade.entry_id)
        self.size = 0.0
        self.sign = 0.0
        self.avg_price = na_float
        self.openprofit = 0.0
        self.open_commission = 0.0
        return consumed

    def update_unrealized_pnl(self, current_price: float) -> None:
        """Mark open trades and :attr:`openprofit` at the given price.

        Per-trade returns include the exchange-reported entry fees;
        :attr:`openprofit` tracks the aggregate gross unrealized profit.

        Also rolls the :attr:`~BrokerPosition.max_equity` peak forward and the
        :attr:`max_drawdown` running maximum off the live price — these feed
        :meth:`_peak_equity` and :meth:`_is_max_drawdown_breached` so the
        broker risk gates see real-time equity, not just realized P&L.
        """
        self._current_price = current_price
        if not self.open_trades or current_price <= 0.0:
            self.openprofit = 0.0
        else:
            total = 0.0
            for trade in self.open_trades:
                gross_profit = (current_price - trade.entry_price) * trade.size
                trade.profit = gross_profit - trade.commission
                entry_cost = abs(trade.size) * trade.entry_price + trade.commission
                trade.profit_percent = trade.profit / entry_cost * 100.0 if entry_cost else 0.0
                total += gross_profit
            self.openprofit = total
        eq = float(self.equity)
        if eq > self.max_equity:
            self.max_equity = eq
        if eq < self.min_equity:
            self.min_equity = eq
        # Drawdown is measured from the running peak, run-up from the running
        # trough — the same metrics the sim fields track, just sourced from
        # mark-to-market: a live account has no bar path to walk. Each percent
        # divides by its excursion's HIGHER endpoint, like the sim's.
        if self.max_equity > -float("inf"):
            dd = self.max_equity - eq
            if dd > self.max_drawdown:
                self.max_drawdown = dd
            if self.max_equity > 0.0:
                dd_percent = dd / self.max_equity * 100.0
                if dd_percent > self.max_drawdown_percent:
                    self.max_drawdown_percent = dd_percent
        if self.min_equity < float("inf"):
            ru = eq - self.min_equity
            if ru > self.max_runup:
                self.max_runup = ru
            if eq > 0.0:
                ru_percent = ru / eq * 100.0
                if ru_percent > self.max_runup_percent:
                    self.max_runup_percent = ru_percent

    def _peak_equity(self) -> float:
        """Reference equity for ``max_drawdown(percent_of_equity)``.

        Falls back to initial capital before the first
        :meth:`update_unrealized_pnl` (or fill) primes ``max_equity``.
        """
        # noinspection PyProtectedMember
        initial = float(lib._script.initial_capital)
        if self.max_equity == -float("inf"):
            return initial
        return max(initial, float(self.max_equity))

    # === Risk management hooks (called by the runner once per bar) =========

    def _handle_bar_open_risk(self) -> None:
        """Day-rollover bookkeeping at the start of each bar.

        Mirrors the rollover block in
        :meth:`SimPosition._process_at_bar_open`: on a new trading day,
        update ``risk_cons_loss_days`` from the prior-day equity delta,
        reset the intraday anchors, and immediately halt if the
        consecutive-loss-day cap is breached so queued entries cannot fill
        at this bar's open. Distinct name from the sim hook because the
        sim variant takes an ``ohlc`` argument and runs additional
        simulator-specific work — the broker version only does
        risk-related rollover.
        """
        try:
            # Statically a value (module_property), at runtime still the function
            current_trading_day = int(lib.time_tradingday())
        except (AttributeError, TypeError, ValueError):
            return
        # A ``max_intraday_loss`` halt expires with its day — release it before
        # the permanent-halt gate below, or the run would never trade again.
        self._release_intraday_halt(current_trading_day)
        if self.risk_halt_trading:
            return
        if current_trading_day == self.risk_last_trading_day:
            return
        current_equity = float(self.equity)
        # On the very first bar there is no prior day to compare against —
        # initialise the trailing-equity anchor without touching the counter.
        if self.risk_last_trading_day != -1:
            if current_equity < self.risk_last_day_equity:
                self.risk_cons_loss_days += 1
            else:
                self.risk_cons_loss_days = 0
        self.risk_last_day_equity = current_equity
        self.risk_intraday_start_equity = current_equity
        self.risk_last_trading_day = current_trading_day
        self.risk_intraday_filled_orders = 0
        if self._is_max_cons_loss_days_breached():
            self._trigger_risk_halt("Max consecutive loss days reached")

    def _enforce_post_bar_risk(self) -> None:
        """Run the post-bar ``strategy.risk.*`` checks.

        Called by the runner after the script executes and before the next
        :meth:`OrderSyncEngine.sync` so the queued risk-close goes out in
        the same dispatch cycle. The first triggered rule wins.
        """
        if self.risk_halt_trading:
            return
        if self._is_max_drawdown_breached():
            self._trigger_risk_halt("Max drawdown reached")
            return
        if self._is_max_intraday_loss_breached():
            self._trigger_risk_halt("Max intraday loss reached", until_day_end=True)
            return
        if self._is_max_cons_loss_days_breached():
            self._trigger_risk_halt("Max consecutive loss days reached")

    def _trigger_risk_halt(self, reason: str, *, until_day_end: bool = False) -> None:
        """Cancel pending orders, queue a market close, set the halt flag.

        Differs from :meth:`SimPosition._trigger_risk_halt` in two ways: no
        synthetic fill (the exchange owns fills, not the position), and no
        OHLC arguments (the close is a plain market order, the broker
        decides the fill price). The next :meth:`OrderSyncEngine.sync`
        observes the cleared books plus the queued close and dispatches
        accordingly.

        :param reason: Rule name embedded in the queued close's comment.
        :param until_day_end: Stamp the halt with the current trading day so
            :meth:`_release_intraday_halt` lifts it at the next rollover —
            ``max_intraday_loss`` blocks the rest of the day, not the run.
        """
        # noinspection PyProtectedMember
        from pynecore.lib.strategy import Order, _order_type_close
        self.entry_orders.clear()
        self.exit_orders.clear()
        if self.size != 0.0:
            close_order = Order(
                None, -self.size,
                exit_id='Risk management close',
                order_type=_order_type_close,
                comment=f"Close Position ({reason})",
            )
            self._add_order(close_order)
        self.risk_halt_trading = True
        self.risk_halt_day = self.risk_last_trading_day if until_day_end else -1

    def record_liquidation(self, event: 'OrderEvent') -> None:
        """Record an exchange-initiated liquidation — close all open trades and bindings."""
        if not self.open_trades:
            return
        self.last_spent_entry_ids = []
        self.clear_entry_book()
        fill_price = event.fill_price or 0.0
        for trade in list(self.open_trades):
            self._close_trade(trade, fill_price, event, fee_share=event.fee / max(len(self.open_trades), 1))
            self.netprofit += trade.profit
            if trade.profit > 0.0:
                self.grossprofit += trade.profit
                self.wintrades += 1
            elif trade.profit < 0.0:
                self.grossloss += trade.profit
                self.losstrades += 1
            else:
                self.eventrades += 1
        self.size = 0.0
        self.sign = 0.0
        self.avg_price = na_float
        self.openprofit = 0.0
        self.open_commission = 0.0

    # === Internals ===

    def _close_trade(self, trade: Trade, fill_price: float,
                     event: 'OrderEvent', fee_share: float) -> None:
        """Move a (possibly split) Trade from open_trades to closed_trades."""
        trade.exit_id = event.pine_id or ""
        trade.exit_bar_index = int(getattr(lib, 'bar_index', 0))
        trade.exit_time = int(event.timestamp * 1000.0)
        trade.exit_price = fill_price
        trade.exit_comment = ''
        entry_commission = trade.commission
        trade.commission += fee_share
        trade.profit = (fill_price - trade.entry_price) * trade.size - trade.commission
        trade.exit_equity = self.equity + trade.profit
        # Total entry cost, position value plus the fee paid to open — the
        # denominator TradingView divides by (see SimPosition's fill loop).
        entry_cost = abs(trade.size) * trade.entry_price + entry_commission
        profit_ratio = (trade.profit / entry_cost) if entry_cost else 0.0
        trade.profit_percent = profit_ratio * 100.0
        # The win/loss counters are booked by the three callers, but the percent
        # sums behind ``strategy.avg_*_trade_percent`` are booked here: this is
        # the single funnel every closing path goes through, so no caller can
        # forget one. The predicate is the callers' own — classify on the
        # after-fee profit.
        self.sum_profit_ratio += profit_ratio
        if trade.profit > 0.0:
            self.sum_win_profit_ratio += profit_ratio
        elif trade.profit < 0.0:
            self.sum_loss_profit_ratio += profit_ratio
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        self.new_closed_trades.append(trade)
        self.closed_trades_count += 1
        self._closed_trade_stats.add(trade)
