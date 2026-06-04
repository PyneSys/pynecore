"""
Position tracking for live broker trading.

:class:`BrokerPosition` extends :class:`~pynecore.lib.strategy.PositionBase`
with no simulation logic — the exchange is the source of truth for fills,
prices, fees, and margin state.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from pynecore import lib
from pynecore.lib.log import broker_warning as _blog_warning
from pynecore.lib.strategy import PositionBase, Trade
from pynecore.types.na import na_float

if TYPE_CHECKING:
    from pynecore.lib.strategy import direction
    from pynecore.lib.strategy import Order
    from pynecore.types.strategy import QtyType
    from pynecore.core.broker.models import OrderEvent

__all__ = ['BrokerPosition']


class BrokerPosition(PositionBase):
    """
    Position state tracker for live broker trading.

    The exchange determines fills, prices, fees, and margin state;
    :meth:`record_fill` consumes :class:`OrderEvent` objects emitted by a
    :class:`~pynecore.core.plugin.broker.BrokerPlugin` and updates the
    local view of the position accordingly.

    Trades are tracked FIFO: the first entry filled is the first closed
    when the position is reduced, matching TradingView default semantics.

    Note: margin, liquidation price, and fee currency conversion are all
    handled by the exchange. This class only records what the exchange
    tells it.
    """

    __slots__ = (
        'size', 'sign', 'avg_price',
        'netprofit', 'openprofit', 'grossprofit', 'grossloss',
        'open_commission',
        'eventrades', 'wintrades', 'losstrades',
        'closed_trades_count',
        'max_drawdown', 'max_runup', 'max_equity',
        'open_trades', 'closed_trades', 'new_closed_trades',
        'entry_orders', 'exit_orders',
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
        'risk_cons_loss_days', 'risk_last_day_index', 'risk_last_day_equity',
        'risk_intraday_filled_orders', 'risk_intraday_start_equity',
        'risk_halt_trading',
        '_current_price',
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
        self.max_drawdown: float = 0.0
        self.max_runup: float = 0.0
        # Mark-to-market peak equity, used by ``_peak_equity`` for the
        # ``max_drawdown(percent_of_equity)`` threshold. Updated on every
        # :meth:`update_unrealized_pnl` and :meth:`record_fill` call.
        self.max_equity: float = -float("inf")

        self.open_trades: list[Trade] = []
        self.closed_trades: deque[Trade] = deque(maxlen=9000)
        self.new_closed_trades: list[Trade] = []

        self.entry_orders: dict[str | None, 'Order'] = {}
        # Composite key ``(exit_id, from_entry)`` mirrors
        # :class:`~pynecore.lib.strategy.SimPosition.exit_orders`. Single-field
        # keys collide on partial-TP fan-out (multiple exits for one entry)
        # and on ``from_entry=na`` fan-out (one exit_id, many per-entry rows).
        self.exit_orders: dict[tuple[str | None, str | None], 'Order'] = {}

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
        self.risk_last_day_index: int = -1
        self.risk_last_day_equity: float = 0.0
        self.risk_intraday_filled_orders: int = 0
        self.risk_intraday_start_equity: float = 0.0
        self.risk_halt_trading: bool = False

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
            _order_type_close, _order_type_entry, _order_type_normal,
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
            self.exit_orders[(order.exit_id, order.order_id)] = order
        else:
            self.entry_orders[order.order_id] = order

    def _remove_order(self, order: 'Order') -> None:
        """Cancel an order locally."""
        order.cancelled = True
        # noinspection PyProtectedMember
        from pynecore.lib.strategy import _order_type_close
        if order.order_type == _order_type_close:
            self.exit_orders.pop((order.exit_id, order.order_id), None)
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

    # === Exchange-side state updates ===

    def record_fill(self, event: 'OrderEvent') -> bool:
        """
        Record an exchange fill.

        :param event: An :class:`OrderEvent` with ``fill_qty`` and
            ``fill_price`` populated, plus Pine identity fields
            (``pine_id``, ``from_entry``, ``leg_type``) filled by the plugin.
        :return: ``True`` if the position side changed as a result of this fill.
        """
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
            if old_abs == 0.0 or self.avg_price is na_float:
                self.avg_price = fill_price
            else:
                self.avg_price = (self.avg_price * old_abs + fill_price * fill_qty) / new_abs
            self.size = new_size
            self.sign = 1.0 if new_size > 0.0 else (-1.0 if new_size < 0.0 else 0.0)

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
            self.open_commission += fee
            return False

        # Reducing or flipping — FIFO close of existing trades
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
                closed_piece = Trade(
                    size=trade.sign * remaining,
                    entry_id=trade.entry_id,
                    entry_bar_index=trade.entry_bar_index,
                    entry_time=trade.entry_time,
                    entry_price=trade.entry_price,
                    commission=trade.commission * (remaining / trade_abs),
                    entry_comment=trade.entry_comment,
                    entry_equity=trade.entry_equity,
                )
                self._close_trade(closed_piece, fill_price, event, fee_share=fee)
                closed_profit += closed_piece.profit
                # Shrink the remaining open trade
                trade.size -= closed_piece.size
                trade.commission -= closed_piece.commission
                remaining = 0.0

        self.size += signed_delta
        # Clamp tiny residuals to zero
        if abs(self.size) < 1e-12:
            self.size = 0.0
            self.sign = 0.0
            self.avg_price = na_float
        else:
            self.sign = 1.0 if self.size > 0.0 else -1.0

        # If there is leftover qty after closing all open_trades → side flip
        if remaining > 0.0:
            new_size = self.sign * remaining if self.sign != 0.0 else signed_delta
            self.size = new_size
            self.sign = 1.0 if new_size > 0.0 else (-1.0 if new_size < 0.0 else 0.0)
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

        # Update running stats
        self.netprofit += closed_profit
        if closed_profit > 0.0:
            self.grossprofit += closed_profit
            self.wintrades += 1
        elif closed_profit < 0.0:
            self.grossloss += closed_profit
            self.losstrades += 1
        else:
            self.eventrades += 1

        self.open_commission = float(sum(t.commission for t in self.open_trades))

        return self.sign != old_sign

    def update_unrealized_pnl(self, current_price: float) -> None:
        """Mark-to-market: recompute :attr:`openprofit` at the given price.

        Also rolls the :attr:`max_equity` peak forward and the
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
                total += (current_price - trade.entry_price) * trade.size
            self.openprofit = total
        eq = float(self.equity)
        if eq > self.max_equity:
            self.max_equity = eq
        # Drawdown is measured from the running peak — same metric the sim
        # ``max_drawdown`` field tracks, just sourced from mark-to-market.
        if self.max_equity > -float("inf"):
            dd = self.max_equity - eq
            if dd > self.max_drawdown:
                self.max_drawdown = dd

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
        if self.risk_halt_trading:
            return
        try:
            current_day = int(lib.dayofmonth())
        except (AttributeError, TypeError, ValueError):
            return
        if current_day == self.risk_last_day_index:
            return
        current_equity = float(self.equity)
        # On the very first bar there is no prior day to compare against —
        # initialise the trailing-equity anchor without touching the counter.
        if self.risk_last_day_index != -1:
            if current_equity < self.risk_last_day_equity:
                self.risk_cons_loss_days += 1
            else:
                self.risk_cons_loss_days = 0
        self.risk_last_day_equity = current_equity
        self.risk_intraday_start_equity = current_equity
        self.risk_last_day_index = current_day
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
            self._trigger_risk_halt("Max intraday loss reached")
            return
        if self._is_max_cons_loss_days_breached():
            self._trigger_risk_halt("Max consecutive loss days reached")

    def _trigger_risk_halt(self, reason: str) -> None:
        """Cancel pending orders, queue a market close, set the halt flag.

        Differs from :meth:`SimPosition._trigger_risk_halt` in two ways: no
        synthetic fill (the exchange owns fills, not the position), and no
        OHLC arguments (the close is a plain market order, the broker
        decides the fill price). The next :meth:`OrderSyncEngine.sync`
        observes the cleared books plus the queued close and dispatches
        accordingly.
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

    def record_liquidation(self, event: 'OrderEvent') -> None:
        """Record an exchange-initiated liquidation — close all open trades."""
        if not self.open_trades:
            return
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
        trade.commission += fee_share
        trade.profit = (fill_price - trade.entry_price) * trade.size - trade.commission
        trade.exit_equity = self.equity + trade.profit
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        self.new_closed_trades.append(trade)
        self.closed_trades_count += 1
