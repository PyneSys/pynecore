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
from pynecore.lib.strategy import PositionBase, Trade
from pynecore.types.na import na_float

if TYPE_CHECKING:
    from pynecore.lib.strategy import Order
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
        'max_drawdown', 'max_runup',
        'open_trades', 'closed_trades',
        'entry_orders', 'exit_orders',
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
        self.max_drawdown: float = 0.0
        self.max_runup: float = 0.0

        self.open_trades: list[Trade] = []
        self.closed_trades: deque[Trade] = deque(maxlen=9000)

        self.entry_orders: dict[str | None, 'Order'] = {}
        self.exit_orders: dict[str | None, 'Order'] = {}

        self._current_price: float = 0.0

    # === Pine-side order book ===

    def _add_order(self, order: 'Order') -> None:
        """Register an order locally (the sync engine forwards it to the exchange)."""
        order.bar_index = int(lib.bar_index)
        from pynecore.lib.strategy import _order_type_close  # local import avoids cycle
        if order.order_type == _order_type_close:
            self.exit_orders[order.order_id] = order
        else:
            self.entry_orders[order.order_id] = order

    def _remove_order(self, order: 'Order') -> None:
        """Cancel an order locally."""
        order.cancelled = True
        from pynecore.lib.strategy import _order_type_close
        if order.order_type == _order_type_close:
            self.exit_orders.pop(order.order_id, None)
        else:
            self.entry_orders.pop(order.order_id, None)

    def _remove_order_by_id(self, order_id: str) -> None:
        order = self.exit_orders.get(order_id) or self.entry_orders.get(order_id)
        if order is not None:
            self._remove_order(order)

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
            return False

        signed_delta = fill_qty if event.order.side == "buy" else -fill_qty
        old_sign = self.sign
        new_size = self.size + signed_delta

        # Commission bookkeeping — realized fee becomes part of net P&L at close
        fee = event.fee

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

        self.open_commission = sum(t.commission for t in self.open_trades)

        return self.sign != old_sign

    def update_unrealized_pnl(self, current_price: float) -> None:
        """Mark-to-market: recompute :attr:`openprofit` at the given price."""
        self._current_price = current_price
        if not self.open_trades or current_price <= 0.0:
            self.openprofit = 0.0
            return
        total = 0.0
        for trade in self.open_trades:
            total += (current_price - trade.entry_price) * trade.size
        self.openprofit = total

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
