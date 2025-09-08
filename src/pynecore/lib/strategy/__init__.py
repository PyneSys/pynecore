from typing import cast, TYPE_CHECKING

from datetime import datetime, UTC
from collections import deque
from copy import copy

from ...core.module_property import module_property
from ... import lib
from .. import syminfo

from ...types.strategy import QtyType
from ...types.base import IntEnum
from ...types.na import NA

from . import direction as direction
from . import commission as _commission
from . import oca as _oca

from . import closedtrades, opentrades

__all__ = [
    "fixed", "cash", "percent_of_equity",
    "long", "short", 'direction',

    'Trade', 'Order', 'Position',
    "cancel", "cancel_all", "close", "close_all", "entry", "exit", "order",

    "closedtrades", "opentrades",
]

#
# Callable modules
#

if TYPE_CHECKING:
    from closedtrades import closedtrades
    from opentrades import opentrades


#
# Types
#

class _OrderType(IntEnum):
    """ Order type """


#
# Constants
#

fixed = QtyType("fixed")
cash = QtyType("cash")
percent_of_equity = QtyType("percent_of_equity")

long = direction.long
short = direction.short

# Possible order types
_order_type_normal = _OrderType()
_order_type_entry = _OrderType()
_order_type_close = _OrderType()

#
# Imports after constants
#

if True:
    # We need to import this here to avoid circular imports
    from . import risk


#
# Classes
#

class Order:
    """
    Represents an order
    """

    __slots__ = (
        "order_id", "size", "sign", "order_type", "limit", "stop", "exit_id", "oca_name", "oca_type",
        "comment", "alert_message",
        "trail_price", "trail_offset",
        "trail_triggered",
        "profit_ticks", "loss_ticks", "trail_points_ticks",  # Store tick values for later calculation
        "is_market_order",  # Flag to check if this is a market order
        "cancelled",  # Flag to mark order as cancelled by OCA
    )

    def __init__(
            self,
            order_id: str | None,
            size: float,
            *,
            order_type: _OrderType = _order_type_normal,
            exit_id: str | None = None,
            limit: float | None = None,
            stop: float | None = None,
            oca_name: str | None = None,
            oca_type: _oca.Oca = _oca.none,
            comment: str | None = None,
            alert_message: str | None = None,
            trail_price: float | None = None,
            trail_offset: float | None = None,
            profit_ticks: float | None = None,
            loss_ticks: float | None = None,
            trail_points_ticks: float | None = None
    ):
        self.order_id = order_id
        self.size = size
        self.sign = 0.0 if size == 0.0 else 1.0 if size > 0.0 else -1.0
        self.limit = limit
        self.stop = stop
        self.order_type = order_type

        self.exit_id = exit_id

        self.oca_name = oca_name
        self.oca_type = oca_type if oca_type is not None else _oca.none

        self.comment = comment
        self.alert_message = alert_message

        self.trail_price = trail_price
        self.trail_offset = trail_offset or 0  # in ticks
        self.trail_triggered = False

        self.profit_ticks = profit_ticks
        self.loss_ticks = loss_ticks
        self.trail_points_ticks = trail_points_ticks

        # Check if this is a market order (no limit, stop, or trail price)
        self.is_market_order = self.limit is None and self.stop is None

        # Initialize cancelled flag
        self.cancelled = False

    def __repr__(self):
        return f"Order(order_id={self.order_id}; exit_id={self.exit_id}; size={self.size}; type: {self.order_type}; " \
               f"limit={self.limit}; stop={self.stop}; " \
               f"trail_price={self.trail_price}; trail_offset={self.trail_offset}; " \
               f"oca_name={self.oca_name}; comment={self.comment})"


class Trade:
    """
    Represents a trade
    """

    __slots__ = (
        "size", "sign", "entry_id", "entry_bar_index", "entry_time", "entry_price", "entry_comment", "entry_equity",
        "exit_id", "exit_bar_index", "exit_time", "exit_price", "exit_comment", "exit_equity",
        "commission", "max_drawdown", "max_drawdown_percent", "max_runup", "max_runup_percent",
        "profit", "profit_percent", "cum_profit", "cum_profit_percent",
        "cum_max_drawdown", "cum_max_runup",
    )

    # noinspection PyShadowingNames
    def __init__(self, *, size: float, entry_id: str, entry_bar_index: int, entry_time: int, entry_price: float,
                 commission: float, entry_comment: str, entry_equity: float):
        self.size: float = size
        self.sign = 0.0 if size == 0.0 else 1.0 if size > 0.0 else -1.0

        self.entry_id: str = entry_id
        self.entry_bar_index: int = entry_bar_index
        self.entry_time: int = entry_time
        self.entry_price: float = entry_price
        self.entry_equity: float = entry_equity
        self.entry_comment: str = entry_comment

        self.exit_id: str = ""
        self.exit_bar_index: int = -1
        self.exit_time: int = -1
        self.exit_price: float = 0.0
        self.exit_comment: str = ''
        self.exit_equity: float | NA = NA(float)

        self.commission = commission

        self.max_drawdown: float | NA[float] = 0.0
        self.max_drawdown_percent: float | NA[float] = 0.0
        self.max_runup: float | NA[float] = 0.0
        self.max_runup_percent: float | NA[float] = 0.0
        self.profit: float | NA[float] = 0.0
        self.profit_percent: float | NA[float] = 0.0

        self.cum_profit: float | NA[float] = 0.0
        self.cum_profit_percent: float | NA[float] = 0.0
        self.cum_max_drawdown: float | NA[float] = 0.0
        self.cum_max_runup: float | NA[float] = 0.0

    def __repr__(self):
        return f"Trade(entry_id={self.entry_id}; size={self.size}; entry_bar_index: {self.entry_bar_index}; " \
               f"entry_price={self.entry_price}; exit_price={self.exit_price}; commission={self.commission}; " \
               f"entry_equity={self.entry_equity}; exit_equity={self.exit_equity}"

    #
    # Support csv.DictWriter
    #

    def keys(self):
        return self.__dict__.keys()

    def get(self, key: str, default=None):
        v = getattr(self, key, default)
        assert v is not None
        if key in ('entry_time', 'exit_time'):
            v = datetime.fromtimestamp(v / 1000.0, tz=UTC)
        elif isinstance(v, float):
            v = round(v, 10)
        return v


# noinspection PyProtectedMember,PyShadowingNames
class Position:
    """
    This holds data about positions and trades

    This is the main class for strategies
    """
    netprofit: float | NA[float] = 0.0
    openprofit: float | NA[float] = 0.0
    grossprofit: float | NA[float] = 0.0
    grossloss: float | NA[float] = 0.0

    entry_orders: dict[str, Order]
    exit_orders: dict[str, Order]

    open_trades: list[Trade]

    closed_trades: deque[Trade]
    new_closed_trades: list[Trade]
    closed_trades_count: int

    wintrades: int
    eventrades: int
    losstrades: int

    size: float = 0.0
    sign: float = 0.0
    avg_price: float = 0.0

    cum_profit: float | NA[float] = 0.0

    had_reversal_in_this_bar: bool = False

    # Risk management settings
    risk_allowed_direction: direction.Direction | None = None
    risk_max_cons_loss_days: int | None = None
    risk_max_cons_loss_days_alert: str | None = None
    risk_max_drawdown_value: float | None = None
    risk_max_drawdown_type: QtyType | None = None
    risk_max_drawdown_alert: str | None = None
    risk_max_intraday_filled_orders: int | None = None
    risk_max_intraday_filled_orders_alert: str | None = None
    risk_max_intraday_loss_value: float | None = None
    risk_max_intraday_loss_type: QtyType | None = None
    risk_max_intraday_loss_alert: str | None = None
    risk_max_position_size: float | None = None

    # Risk management state tracking
    risk_cons_loss_days: int = 0
    risk_last_day_index: int = -1
    risk_last_day_equity: float = 0.0
    risk_intraday_filled_orders: int = 0
    risk_intraday_start_equity: float = 0.0
    risk_halt_trading: bool = False

    def __init__(self):
        self.entry_orders = {}  # Entry orders from strategy.entry()
        self.exit_orders = {}  # Exit orders from strategy.exit(), strategy.close(), etc.

        self.open_trades = []
        self.closed_trades = deque(maxlen=9000)  # 9000 is the limit of TV
        self.closed_trades_count = 0
        self.new_closed_trades = []

        self.entry_equity = 0.0
        self.max_equity = -float("inf")
        self.min_equity = float("inf")
        self.drawdown_summ = 0.0
        self.runup_summ = 0.0
        self.max_drawdown = 0.0
        self.max_runup = 0.0

        self.wintrades = 0
        self.eventrades = 0
        self.losstrades = 0

        self.entry_summ = 0.0
        self.open_commission = 0.0

        self.o = self.h = self.l = self.c = 0.0

    def reset(self):
        """ Reset position variables """
        self.entry_orders.clear()
        self.exit_orders.clear()
        self.open_trades.clear()
        self.closed_trades.clear()
        self.closed_trades_count = 0
        self.new_closed_trades.clear()
        self.entry_equity = 0.0
        self.max_equity = -float("inf")
        self.min_equity = float("inf")
        self.drawdown_summ = 0.0
        self.runup_summ = 0.0
        self.max_drawdown = 0.0
        self.max_runup = 0.0
        self.wintrades = 0
        self.eventrades = 0
        self.losstrades = 0
        self.entry_summ = 0.0
        self.open_commission = 0.0
        self.size = 0.0
        self.sign = 0.0
        self.avg_price = 0.0
        self.netprofit = 0.0
        self.openprofit = 0.0
        self.grossprofit = 0.0
        self.grossloss = 0.0
        self.cum_profit = 0.0
        self.had_reversal_in_this_bar = False

    @property
    def equity(self) -> float | NA[float]:
        """ The current equity """
        assert lib._script is not None
        return lib._script.initial_capital + self.netprofit + self.openprofit

    def _remove_order(self, order: Order):
        """ Remove an order from the strategy """
        if order.order_type == _order_type_close:
            self.exit_orders.pop(order.order_id, None)
        else:
            # Both entry and normal orders are stored in entry_orders dict
            self.entry_orders.pop(order.order_id, None)

    def _cancel_oca_group(self, oca_name: str, executed_order: Order):
        """Cancel all orders in the same OCA group except the executed one"""
        # Cancel entry orders in the same OCA group
        for order in self.entry_orders.values():
            if order.oca_name == oca_name and order != executed_order:
                order.cancelled = True

        # Cancel exit orders in the same OCA group
        for order in self.exit_orders.values():
            if order.oca_name == oca_name and order != executed_order:
                order.cancelled = True

    def _reduce_oca_group(self, oca_name: str, filled_size: float):
        """Reduce the size of all orders in the same OCA group"""
        reduction = abs(filled_size)

        # Reduce entry orders
        for order in self.entry_orders.values():
            if order.oca_name == oca_name and not order.cancelled:
                new_size = abs(order.size) - reduction
                if new_size <= 0:
                    # Mark order as cancelled if size would be 0 or negative
                    order.cancelled = True
                else:
                    # Keep original sign
                    order.size = new_size * order.sign

        # Reduce exit orders
        for order in self.exit_orders.values():
            if order.oca_name == oca_name and not order.cancelled:
                new_size = abs(order.size) - reduction
                if new_size <= 0:
                    order.cancelled = True
                else:
                    order.size = new_size * order.sign

    def _fill_order(self, order: Order, price: float, h: float, l: float):
        """
        Fill an order (actually)

        :param order: The order to fill
        :param price: The price to fill at
        :param h: The high price
        :param l: The low price
        """
        # Save the original order size before any modifications
        filled_size = abs(order.size)

        script = lib._script
        assert script is not None
        commission_type = script.commission_type
        commission_value = script.commission_value

        new_closed_trades = []
        closed_trade_size = 0.0

        # Close order - if it is an exit order or a normal order
        if self.size and order.order_type != _order_type_entry and order.sign != self.sign:
            delete = False

            # Check list of open trades
            new_open_trades = []
            for trade in self.open_trades:
                # Only use if its order id is the same
                if order.size != 0.0 and (trade.entry_id == order.order_id
                                          or order.order_type == _order_type_normal
                                          or order.order_id is None):
                    delete = True

                    size = order.size if abs(order.size) <= abs(trade.size) else -trade.size
                    pnl = -size * (price - trade.entry_price)

                    # Copy and modify actual trade, because it can be partially filled
                    closed_trade = copy(trade)

                    size_ratio = 1 + size / closed_trade.size
                    if closed_trade.size != -size:
                        # Modify commission
                        trade.commission *= size_ratio
                        closed_trade.commission *= (1 - size_ratio)
                        # Modify drawdown and runup
                        trade.max_drawdown *= size_ratio
                        trade.max_runup *= size_ratio
                        closed_trade.max_drawdown *= (1 - size_ratio)
                        closed_trade.max_runup *= (1 - size_ratio)

                    # P/L from high/low to calculate drawdown and runup
                    hprofit = (-size * (h - closed_trade.entry_price) - closed_trade.commission)
                    lprofit = (-size * (l - closed_trade.entry_price) - closed_trade.commission)

                    # Drawdown and runup
                    drawdown = -min(hprofit, lprofit, 0.0)
                    runup = max(hprofit, lprofit, 0.0)
                    # Drawdown summ runup summ
                    self.drawdown_summ += drawdown
                    self.runup_summ += runup

                    closed_trade.size = -size
                    closed_trade.exit_id = order.exit_id if order.exit_id is not None else order.order_id
                    closed_trade.exit_bar_index = int(lib.bar_index)
                    closed_trade.exit_time = lib._time
                    closed_trade.exit_price = price
                    closed_trade.profit = pnl

                    # Add to closed trade
                    new_closed_trades.append(closed_trade)
                    self.closed_trades.append(closed_trade)
                    self.closed_trades_count += 1

                    if order.comment:
                        # TODO: implement comment_profit, comment_loss, comment_trailing...
                        closed_trade.exit_comment = order.comment

                    # Commission summ
                    self.open_commission -= closed_trade.commission

                    # We realize later if it is cash per order or cash per contract
                    if (commission_type == _commission.cash_per_contract or
                            commission_type == _commission.cash_per_order):
                        closed_trade_size += abs(size)
                    else:
                        commission = abs(size) * commission_value

                        if commission_type == _commission.percent:
                            commission *= 0.01
                        closed_trade.commission += commission
                        # Realize commission
                        self.netprofit -= commission
                        closed_trade.profit -= closed_trade.commission

                    # Profit percent
                    entry_value = abs(closed_trade.size) * closed_trade.entry_price
                    try:
                        closed_trade.profit_percent = (pnl / entry_value) * 100.0
                    except ZeroDivisionError:
                        closed_trade.profit_percent = 0.0

                    # Realize profit or loss
                    self.netprofit += pnl

                    # Modify sizes
                    self.size += size
                    # Handle too small sizes because of floating point inaccuracy and rounding
                    if _size_round(self.size) == 0.0:
                        size -= self.size
                        self.size = 0.0
                    self.sign = 0.0 if self.size == 0.0 else 1.0 if self.size > 0.0 else -1.0
                    trade.size += size
                    order.size -= size

                    # Gross P/L and counters
                    if closed_trade.profit == 0.0:
                        self.eventrades += 1
                    elif closed_trade.profit > 0.0:
                        self.wintrades += 1
                        self.grossprofit += closed_trade.profit
                    else:
                        self.losstrades += 1
                        self.grossloss -= closed_trade.profit

                    # Average entry price
                    if self.size:
                        self.entry_summ -= closed_trade.entry_price * abs(closed_trade.size)
                        self.avg_price = self.entry_summ / abs(self.size)

                        # Unrealized P&L
                        self.openprofit = self.size * (self.c - self.avg_price)
                    else:
                        # If position has just closed
                        self.avg_price = 0.0
                        self.openprofit = 0.0

                    # Exit equity
                    closed_trade.exit_equity = self.equity

                    # Remove from open trades if it is fully filled
                    if trade.size == 0.0:
                        continue

                    if pnl > 0.0:
                        # Modify summs and entry equity with commission
                        self.runup_summ -= closed_trade.commission
                        self.drawdown_summ += closed_trade.commission / 2
                        self.entry_equity += closed_trade.commission / 2

                new_open_trades.append(trade)

            self.open_trades = new_open_trades
            if delete:
                # Remove from exit_orders dict
                self.exit_orders.pop(order.exit_id, None)

                if commission_type == _commission.cash_per_order:
                    # Realize commission
                    self.netprofit -= commission_value
                    for trade in new_closed_trades:
                        commission = (commission_value * abs(trade.size)) / closed_trade_size
                        trade.commission += commission

            self.new_closed_trades.extend(new_closed_trades)

        # New trade
        elif order.order_type != _order_type_close:
            # Calculate commission
            if commission_value:
                if commission_type == _commission.cash_per_order:
                    commission = commission_value
                elif commission_type == _commission.percent:
                    commission = abs(order.size) * commission_value * 0.01
                elif commission_type == _commission.cash_per_contract:
                    commission = abs(order.size) * commission_value
                else:  # Should not be here!
                    assert False, 'Wrong commission type: ' + str(commission_type)
            else:
                commission = 0.0

            before_equity = self.equity

            # Realize commission
            self.netprofit -= commission

            entry_equity = self.equity
            if not self.open_trades:
                # Set max and min equity
                self.max_equity = max(self.max_equity, entry_equity)
                self.min_equity = min(self.min_equity, entry_equity)
                # Entry equity
                self.entry_equity = entry_equity

            assert order.order_id is not None

            trade = Trade(
                size=order.size,
                entry_id=order.order_id, entry_bar_index=cast(int, lib.bar_index),
                entry_time=lib._time, entry_price=price,
                commission=commission, entry_comment=order.comment,  # type: ignore
                entry_equity=before_equity
            )
            self.open_trades.append(trade)
            self.size += trade.size
            self.sign = 0.0 if self.size == 0.0 else 1.0 if self.size > 0.0 else -1.0

            # Average entry price
            self.entry_summ += price * abs(order.size)
            try:
                self.avg_price = self.entry_summ / abs(self.size)
            except ZeroDivisionError:
                self.avg_price = 0.0
            # Unrealized P&L
            self.openprofit = self.size * (self.c - self.avg_price)
            # Commission summ
            self.open_commission += commission

            # Remove the order from the appropriate dict
            if order.exit_id:
                self.exit_orders.pop(order.exit_id, None)
            else:
                self.entry_orders.pop(order.order_id, None)

        # If position has just closed
        if not self.open_trades:
            # Reset position variables
            self.entry_summ = 0.0
            self.avg_price = 0.0
            self.openprofit = 0.0
            self.open_commission = 0.0

        # Handle OCA groups after order execution
        # This is done here to avoid code duplication in fill_order()
        if order.oca_name and order.oca_type:
            if order.oca_type == _oca.cancel:
                self._cancel_oca_group(order.oca_name, order)
            elif order.oca_type == _oca.reduce:
                # Use the saved original filled_size from the beginning of this method
                self._reduce_oca_group(order.oca_name, filled_size)

    def fill_order(self, order: Order, price: float, h: float, l: float) -> bool:
        """
        Fill an order

        :param order: The order to fill
        :param price: The price to fill at
        :param h: The high price
        :param l: The low price
        :return: True if the side of the position has changed
        """
        size_addition = 0.0
        # Apply risk management only to entry orders, not normal orders from strategy.order()
        if order.order_type == _order_type_entry or order.order_type == _order_type_normal:
            # Risk management: Check max intraday filled orders
            if self.risk_max_intraday_filled_orders is not None:
                if self.risk_intraday_filled_orders >= self.risk_max_intraday_filled_orders:
                    # Max intraday filled orders reached - don't fill the entry order
                    self._remove_order(order)
                    return False

            # Risk management: Check max position size
            if self.risk_max_position_size is not None:
                new_position_size = abs(self.size + order.size)
                if new_position_size > self.risk_max_position_size:
                    # Adjust order size to not exceed max position size
                    max_allowed_size = self.risk_max_position_size - abs(self.size)
                    if max_allowed_size <= 0:
                        # Can't add to position - remove order
                        self._remove_order(order)
                        return False
                    # Adjust the order size
                    order.size = max_allowed_size * order.sign

            # Check risk allowed direction for new positions (when no current position)
            if self.size == 0.0 and self.risk_allowed_direction is not None:
                if (order.sign > 0 and self.risk_allowed_direction != long) or \
                        (order.sign < 0 and self.risk_allowed_direction != short):
                    # Direction not allowed - don't fill the entry order
                    self._remove_order(order)
                    return False

            # If we have an existing position
            if self.size != 0.0 and order.order_type == _order_type_entry:
                # Check if the order has the same direction
                if self.sign == order.sign:
                    # Check pyramiding limit for entry orders adding to existing position
                    if lib._script.pyramiding <= len(self.open_trades):
                        # Pyramiding limit reached - don't fill the entry order
                        self._remove_order(order)
                        return False

                # TradingView changes direction of the position
                else:
                    # We need to change direction
                    size_addition = -self.size

                    # It is TradingView's behavior to allow only 1 reversal per bar
                    if self.had_reversal_in_this_bar:
                        return False
                    self.had_reversal_in_this_bar = True

        # For normal orders (_order_type_normal), no special risk management or pyramiding limits apply
        # They simply add to or subtract from the position as requested

        # If position direction is about to change, we split it into two separate orders
        # This is necessary to create a new average entry price
        new_size = self.size + order.size + size_addition
        if _size_round(new_size) == 0.0:
            new_size = 0.0
        new_sign = 0.0 if new_size == 0.0 else 1.0 if new_size > 0.0 else -1.0
        if self.size != 0.0 and new_sign != self.sign and new_size != 0.0:
            # Exit orders should never reverse position direction
            # Only entry orders can open new positions or reverse direction
            if order.order_type == _order_type_close:
                # Limit the exit order size to just close the position
                order.size = -self.size
                self._fill_order(order, price, h, l)
                return False

            # Create a copy for closing existing position
            order1 = copy(order)
            order1.order_type = _order_type_close
            order1.size = -self.size
            # Set order_id to None so it will close any open trades
            order1.order_id = None
            # The exit_id will be the order_id of the original order
            order1.exit_id = order.order_id
            # Fill the closing order first
            self._fill_order(order1, price, h, l)

            # Check if new direction is allowed by risk management
            # According to Pine Script docs: "long exit trades will be made instead of reverse trades"
            new_direction_sign = 1.0 if new_size > 0.0 else -1.0
            if self.risk_allowed_direction is not None:
                if (new_direction_sign > 0 and self.risk_allowed_direction != long) or \
                        (new_direction_sign < 0 and self.risk_allowed_direction != short):
                    # Direction not allowed - convert entry to exit only
                    # Don't open new position in restricted direction
                    self._remove_order(order)
                    return False

            # Modify the original order to open a position in the new direction
            order.size = new_size
            # Fill the entry order
            self._fill_order(order, price, h, l)
            return True

        # If position direction is not about to change, we can fill the order directly
        else:
            self._fill_order(order, price, h, l)
            return False

    def _check_high_stop(self, order: Order):
        """ Check high stop and trailing trigger """
        if order.stop is None:
            return
        # Long stop order (size > 0) triggers when price rises to stop level
        if order.size > 0 and order.stop <= self.h:
            p = max(order.stop, self.o)
            self.fill_order(order, p, p, self.l)

    def _check_high(self, order: Order):
        """ Check high limit """
        if order.limit is not None:
            # Short limit order (size < 0) triggers when price rises to limit level
            if order.size < 0 and order.limit <= self.h:
                p = max(order.limit, self.o)
                self.fill_order(order, p, p, self.l)

        # Update trailing stop
        if order.trail_price is not None and order.sign < 0:
            # Check if trailing price has been triggered
            if not order.trail_triggered and self.h > order.trail_price:
                order.trail_triggered = True
            # Update stop if trailing price has been triggered
            if order.trail_triggered:
                offset_price = syminfo.mintick * order.trail_offset
                order.stop = max(lib.math.round_to_mintick(self.h - offset_price), order.stop)  # type: ignore

    def _check_low_stop(self, order: Order):
        """ Check low stop """
        if order.stop is None:
            return
        # Short stop order (size < 0) triggers when price falls to stop level
        if order.size < 0 and order.stop >= self.l:
            p = min(self.o, order.stop)
            self.fill_order(order, p, self.h, p)

    def _check_low(self, order: Order):
        """ Check low limit """
        if order.limit is not None:
            # Long limit order (size > 0) triggers when price falls to limit level
            if order.size > 0 and order.limit >= self.l:
                p = min(self.o, order.limit)
                self.fill_order(order, p, self.h, p)

        # Update trailing stop
        if order.trail_price is not None and order.sign > 0:
            # Check if trailing price has been triggered
            if not order.trail_triggered and self.l < order.trail_price:
                order.trail_triggered = True
            # Update stop if trailing price has been triggered
            if order.trail_triggered:
                offset_price = syminfo.mintick * order.trail_offset
                order.stop = min(lib.math.round_to_mintick(self.l + offset_price), order.stop)  # type: ignore

    def _check_close(self, order: Order, ohlcv: bool):
        """ Check close price if trailing stop is triggered """
        # open → high → low → close
        if ohlcv and order.stop <= self.c:
            self.fill_order(order, order.stop, order.stop, self.l)

        # open → low → high → close
        elif order.stop >= self.c:
            self.fill_order(order, order.stop, self.h, order.stop)

    def process_orders(self):
        """ Process orders """
        # We need to round to the nearest tick to get the same results as in TradingView
        round_to_mintick = lib.math.round_to_mintick
        self.o = round_to_mintick(lib.open)
        self.h = round_to_mintick(lib.high)
        self.l = round_to_mintick(lib.low)
        self.c = round_to_mintick(lib.close)

        # Get script reference for slippage
        script = lib._script

        # If the order is open → high → low → close or open → low → high → close
        ohlc = abs(self.h - self.o) < abs(self.l - self.o)

        self.drawdown_summ = self.runup_summ = 0.0
        self.new_closed_trades.clear()

        self.had_reversal_in_this_bar = False

        # Process all orders: entry orders first, then exit orders (guaranteed order)
        # Create a list of orders to process before iteration
        # This prevents issues when orders are removed during processing
        orders_to_process = list(self.entry_orders.values()) + list(self.exit_orders.values())

        for order in orders_to_process:
            # Skip cancelled orders (cancelled by OCA)
            if order.cancelled:
                continue
            # For exit orders, calculate limit/stop from entry price if ticks are specified
            if order.order_type == _order_type_close and order.order_id:
                # Try to find the trade with matching entry_id
                entry_price = None
                for trade in self.open_trades:
                    if trade.entry_id == order.order_id:
                        entry_price = trade.entry_price
                        break

                # If we found the entry price and have tick values, calculate the actual prices
                if entry_price is not None:
                    # Determine direction from the order
                    direction = 1.0 if order.size < 0 else -1.0  # Exit order size is negative of position

                    # Calculate limit from profit_ticks if specified
                    if order.profit_ticks is not None and order.limit is None:
                        order.limit = entry_price + direction * syminfo.mintick * order.profit_ticks
                        order.limit = _price_round(order.limit, direction)

                    # Calculate stop from loss_ticks if specified
                    if order.loss_ticks is not None and order.stop is None:
                        order.stop = entry_price - direction * syminfo.mintick * order.loss_ticks
                        order.stop = _price_round(order.stop, -direction)

                    # Calculate trail_price from trail_points_ticks if specified
                    if order.trail_points_ticks is not None and order.trail_price is None:
                        order.trail_price = entry_price + direction * syminfo.mintick * order.trail_points_ticks
                        order.trail_price = _price_round(order.trail_price, direction)

            # Market orders
            if order.is_market_order:
                # Apply slippage to market orders
                fill_price = self.o
                if script.slippage > 0:
                    # Slippage is in ticks, always adverse to trade direction
                    # For long orders (buying), slippage increases the price
                    # For short orders (selling), slippage decreases the price
                    slippage_amount = syminfo.mintick * script.slippage * order.sign
                    fill_price = self.o + slippage_amount

                # open → high → low → close
                if ohlc:
                    self.fill_order(order, fill_price, self.o, self.l)
                # open → low → high → close
                else:
                    self.fill_order(order, fill_price, self.l, self.o)

            # Limit/Stop orders
            else:
                # open → high → low → close
                if ohlc:
                    self._check_high_stop(order)
                    self._check_high(order)
                # open → low → high → close
                else:
                    self._check_low_stop(order)
                    self._check_low(order)

        # 2nd round of process open orders
        # Create a new list for the second round (orders might have been removed in first round)
        orders_to_process = list(self.entry_orders.values()) + list(self.exit_orders.values())

        for order in orders_to_process:
            # Skip cancelled orders (cancelled by OCA)
            if order.cancelled:
                continue
            # For exit orders, calculate limit/stop from entry price if not already done
            if order.order_type == _order_type_close and order.order_id:
                # Only recalculate if not already set in first round
                if ((order.profit_ticks is not None or order.loss_ticks is not None
                     or order.trail_points_ticks is not None) and order.limit is None and order.stop is None):
                    # Try to find the trade with matching entry_id
                    entry_price = None
                    for trade in self.open_trades:
                        if trade.entry_id == order.order_id:
                            entry_price = trade.entry_price
                            break

                    # If we found the entry price and have tick values, calculate the actual prices
                    if entry_price is not None:
                        # Determine direction from the order
                        direction = 1.0 if order.size < 0 else -1.0  # Exit order size is negative of position

                        # Calculate limit from profit_ticks if specified
                        if order.profit_ticks is not None and order.limit is None:
                            order.limit = entry_price + direction * syminfo.mintick * order.profit_ticks
                            order.limit = _price_round(order.limit, direction)

                        # Calculate stop from loss_ticks if specified
                        if order.loss_ticks is not None and order.stop is None:
                            order.stop = entry_price - direction * syminfo.mintick * order.loss_ticks
                            order.stop = _price_round(order.stop, -direction)

                        # Calculate trail_price from trail_points_ticks if specified
                        if order.trail_points_ticks is not None and order.trail_price is None:
                            order.trail_price = entry_price + direction * syminfo.mintick * order.trail_points_ticks
                            order.trail_price = _price_round(order.trail_price, direction)

            # Here all market orders should be gone
            # open → high → low → close
            if ohlc:
                self._check_low_stop(order)
                self._check_low(order)
            # open → low → high → close
            else:
                self._check_high_stop(order)
                self._check_high(order)

            if order.trail_triggered and order.stop is not None:
                self._check_close(order, ohlc)

        # Calculate average entry price, unrealized P&L, drawdown and runup...
        if self.open_trades:
            # Unrealized P&L
            self.openprofit = self.size * (self.c - self.avg_price)

            # Calculate open drawdowns and runups
            for trade in self.open_trades:
                # Profit of trade
                trade.profit = trade.size * (self.c - trade.entry_price) - 2 * trade.commission

                # P/L from high/low to calculate drawdown and runup
                hprofit = trade.size * (self.h - self.avg_price) - trade.commission
                lprofit = trade.size * (self.l - self.avg_price) - trade.commission
                # Drawdown
                drawdown = -min(hprofit, lprofit, 0.0)
                trade.max_drawdown = max(drawdown, trade.max_drawdown)
                # Runup
                runup = max(hprofit, lprofit, 0.0)
                trade.max_runup = max(runup, trade.max_runup)

                # Calculate percentage values for drawdown and runup
                # This part is missing in the original code
                trade_value = abs(trade.size) * trade.entry_price
                if trade_value > 0:
                    # Calculate drawdown percentage
                    trade.max_drawdown_percent = max(
                        (drawdown / trade_value) * 100.0 if drawdown > 0 else 0.0,
                        trade.max_drawdown_percent
                    )

                    # Calculate runup percentage
                    trade.max_runup_percent = max(
                        (runup / trade_value) * 100.0 if runup > 0 else 0.0,
                        trade.max_runup_percent
                    )

                # Drawdown summ runup summ
                self.drawdown_summ += drawdown
                self.runup_summ += runup

        # Calculate max drawdown and runup
        if self.drawdown_summ or self.runup_summ:
            self.max_drawdown = max(self.max_drawdown, self.max_equity - self.entry_equity + self.drawdown_summ)
            self.max_runup = max(self.max_runup, self.entry_equity - self.min_equity + self.runup_summ)

        # Cumulative stats
        if self.new_closed_trades:
            assert lib._script is not None
            initial_capital = lib._script.initial_capital
            for closed_trade in self.new_closed_trades:
                previous_cum_profit = self.cum_profit - closed_trade.profit

                self.cum_profit = self.equity - lib._script.initial_capital - self.openprofit
                closed_trade.cum_profit = self.cum_profit
                closed_trade.cum_max_drawdown = self.max_drawdown
                closed_trade.cum_max_runup = self.max_runup

                # Cumulative profit percent
                denominator = initial_capital + previous_cum_profit
                try:
                    closed_trade.cum_profit_percent = (closed_trade.profit / denominator) * 100.0
                except ZeroDivisionError:
                    closed_trade.cum_profit_percent = 0.0

                # Modify entry equity, for max drawdown and runup
                self.entry_equity += closed_trade.profit


#
# Functions
#

# noinspection PyProtectedMember
def _size_round(qty: float) -> float:
    """
    Round size to the nearest possible value

    :param qty: The quantity to round
    :return: The rounded quantity
    """
    rfactor = syminfo._size_round_factor  # noqa
    qrf = int(abs(qty) * rfactor * 10.0) * 0.1  # We need to floor to one decimal place
    sign = 1 if qty > 0 else -1
    return sign * int(qrf) / rfactor


# noinspection PyShadowingNames
def _price_round(price: float | NA[float], direction: int | float) -> float | NA[float]:
    """
    Round price to the nearest tick

    :param price: The price to round
    :param direction: The direction of the price
    :return:
    """
    if isinstance(price, NA):
        return NA(float)
    mintick = syminfo.mintick
    ppmt = round(cast(float, price / mintick), 5)
    ppmt_int = int(ppmt)

    if direction < 0:
        # Round down
        return ppmt_int * mintick
    else:
        # Round up only if ppmt is not already an integer
        if ppmt == ppmt_int:
            # Already an integer, no rounding needed
            return ppmt_int * mintick
        else:
            # Not an integer, round up
            return (ppmt_int + 1) * mintick


# noinspection PyShadowingBuiltins,PyProtectedMember
def cancel(id: str):
    """
    Cancels a pending or unfilled order with a specific identifier

    :param id: The identifier of the order to cancel
    """
    if lib._lib_semaphore:
        return

    assert lib._script is not None and lib._script.position is not None
    # Try to cancel both entry and exit orders with the given ID
    # since we don't know which type it is
    # noinspection PyProtectedMember
    try:
        del lib._script.position.entry_orders[id]
    except KeyError:
        pass
    # noinspection PyProtectedMember
    try:
        del lib._script.position.exit_orders[id]
    except KeyError:
        pass


# noinspection PyProtectedMember
def cancel_all():
    """
    Cancels all pending or unfilled orders
    """
    if lib._lib_semaphore:
        return

    assert lib._script is not None and lib._script.position is not None
    lib._script.position.entry_orders.clear()
    lib._script.position.exit_orders.clear()


# noinspection PyProtectedMember,PyShadowingBuiltins,PyShadowingNames
def close(id: str, comment: str | NA[str] = NA(str), qty: float | NA[float] = NA(float),
          qty_percent: float | NA[float] = NA(float), alert_message: str | NA[str] = NA(str),
          immediately: bool = False):
    """
    Creates an order to exit from the part of a position opened by entry orders with a specific identifier.

    :param id: The identifier of the entry order to close
    :param comment: Additional notes on the filled order
    :param qty: The number of contracts/lots/shares/units to close when an exit order fills
    :param qty_percent: A value between 0 and 100 representing the percentage of the open trade
                        quantity to close when an exit order fills
    :param alert_message: Custom text for the alert that fires when an order fills.
    :param immediately: If true, the closing order executes on the same tick when the strategy places it
    """
    if lib._lib_semaphore:
        return

    assert lib._script is not None and lib._script.position is not None
    position = lib._script.position

    if qty <= 0.0:
        return

    if position.size == 0.0:
        return

    if isinstance(qty, NA):
        size = -position.size * (qty_percent * 0.01) if not isinstance(qty_percent, NA) \
            else -position.size
    else:
        size = -position.sign * qty

    size = _size_round(size)
    if size == 0.0:
        return

    exit_id = f"Close entry(s) order {id}"
    order = Order(id, size, exit_id=exit_id, order_type=_order_type_close,
                  comment=None if isinstance(comment, NA) else comment,
                  alert_message=None if isinstance(alert_message, NA) else alert_message)

    # Store in exit_orders dict
    position.exit_orders[exit_id] = order
    if immediately:
        position.fill_order(order, position.c, position.h, position.l)


# noinspection PyProtectedMember,PyShadowingNames
def close_all(comment: str | NA[str] = NA(str), alert_message: str | NA[str] = NA(str), immediately: bool = False):
    """
    Creates an order to close an open position completely, regardless of the identifiers of the entry
    orders that opened or added to it.

    :param comment: Additional notes on the filled order
    :param alert_message: Custom text for the alert that fires when an order fills
    :param immediately: If true, the closing order executes on the same tick when the strategy places it
    """
    if lib._lib_semaphore:
        return

    assert lib._script is not None and lib._script.position is not None
    position = lib._script.position
    if position.size == 0.0:
        return

    exit_id = 'Close position order'
    order = Order(None, -position.size, exit_id=exit_id, order_type=_order_type_close,
                  comment=comment, alert_message=alert_message)

    # Store in exit_orders dict
    position.exit_orders[exit_id] = order
    if immediately:
        position.fill_order(order, position.c, position.h, position.l)


# noinspection PyProtectedMember,PyShadowingNames,PyShadowingBuiltins
def entry(id: str, direction: direction.Direction, qty: int | float | NA[float] = NA(float),
          limit: int | float | None = None, stop: int | float | None = None,
          oca_name: str | None = None, oca_type: _oca.Oca | None = None,
          comment: str | None = None, alert_message: str | None = None):
    """
    Creates a new order to open or add to a position. If an order with the same id already exists
    and is unfilled, this command will modify that order.

    :param id: The identifier of the order
    :param direction: The direction of the order (long or short)
    :param qty: The number of contracts/lots/shares/units to buy or sell
    :param limit: The price at which the order is filled
    :param stop: The price at which the order is filled
    :param oca_name: The name of the order cancel/replace group
    :param oca_type: The type of the order cancel/replace group
    :param comment: Additional notes on the filled order
    :param alert_message: Custom text for the alert that fires when an order fills
    """
    if lib._lib_semaphore:
        return

    script = lib._script
    position = script.position

    # Risk management: Check if trading is halted
    if position.risk_halt_trading:
        return

    # Get default qty by script parameters if no qty is specified
    if isinstance(qty, NA):
        default_qty_type = script.default_qty_type
        if default_qty_type == fixed:
            qty = script.default_qty_value

        elif default_qty_type == percent_of_equity:
            default_qty_value = script.default_qty_value
            # TradingView calculates position size so that the total investment
            # (position value + commission) equals the specified percentage of equity
            #
            # For percent commission: total_cost = qty * price * (1 + commission_rate)
            # For cash per contract: total_cost = qty * price + qty * commission_value
            #
            # We want: total_cost = equity * percent
            # So: qty = (equity * percent) / (price * (1 + commission_factor))

            equity_percent = default_qty_value * 0.01
            target_investment = script.position.equity * equity_percent

            # Calculate the commission factor based on commission type
            if script.commission_type == _commission.percent:
                # For percentage commission: qty * price * (1 + commission%)
                commission_multiplier = 1.0 + script.commission_value * 0.01
                qty = target_investment / (position.c * syminfo.pointvalue * commission_multiplier)

            elif script.commission_type == _commission.cash_per_contract:
                # For cash per contract: qty * price + qty * commission_value
                # qty * (price + commission_value) = target_investment
                price_plus_commission = position.c * syminfo.pointvalue + script.commission_value
                qty = target_investment / price_plus_commission

            elif script.commission_type == _commission.cash_per_order:
                # For cash per order: qty * price + commission_value = target_investment
                # qty = (target_investment - commission_value) / price
                qty = (target_investment - script.commission_value) / (position.c * syminfo.pointvalue)
                qty = max(0.0, qty)  # Ensure non-negative

            else:
                # No commission
                qty = target_investment / (position.c * syminfo.pointvalue)

        elif default_qty_type == cash:
            default_qty_value = script.default_qty_value
            qty = default_qty_value / (position.c * syminfo.pointvalue)

        else:
            raise ValueError("Unknown default qty type: ", default_qty_type)

    # qty must be greater than 0
    if qty <= 0.0:
        return

    # We need a signed size instead of qty, the sign is the direction
    direction_sign: float = (-1.0 if direction == short else 1.0)
    size = qty * direction_sign

    size = _size_round(size)
    if size == 0.0:
        return

    if isinstance(limit, NA):
        limit = None
    elif limit is not None:
        limit = _price_round(limit, direction_sign)
    if isinstance(stop, NA):
        stop = None
    elif stop is not None:
        stop = _price_round(stop, -direction_sign)

        # If it has an already filled stop, it must be a market order
        if direction_sign == 1.0 and stop <= position.c:
            stop = None
        elif direction_sign == -1.0 and stop >= position.c:
            stop = None

    order = Order(id, size, order_type=_order_type_entry, limit=limit, stop=stop, oca_name=oca_name,
                  oca_type=oca_type, comment=comment, alert_message=alert_message)
    # Store in entry_orders dict
    script.position.entry_orders[id] = order


# noinspection PyShadowingBuiltins,PyProtectedMember,PyShadowingNames,PyUnusedLocal
def exit(id: str, from_entry: str = "",
         qty: float | NA[float] = NA(float), qty_percent: float | NA[float] = NA(float),
         profit: float | NA[float] = NA(float), limit: float | NA[float] = NA(float),
         loss: float | NA[float] = NA(float), stop: float | NA[float] = NA(float),
         trail_price: float | NA[float] = NA(float), trail_points: float | NA[float] = NA(float),
         trail_offset: float | NA[float] = NA(float),
         oca_name: str | NA[str] = NA(str), oca_type: _oca.Oca | None = None,
         comment: str | NA[str] = NA(str), comment_profit: str | NA[str] = NA(str),
         comment_loss: str | NA[str] = NA(str), comment_trailing: str | NA[str] = NA(str),
         alert_message: str | NA[str] = NA(str), alert_profit: str | NA[str] = NA(str),
         alert_loss: str | NA[str] = NA(str), alert_trailing: str | NA[str] = NA(str),
         disable_alert: bool = False):
    """
    Creates an order to exit from a position. If an order with the same id already exists and is unfilled,

    :param id: The identifier of the order
    :param from_entry: The identifier of the entry order to close
    :param qty: The number of contracts/lots/shares/units to close when an exit order fills
    :param qty_percent: A value between 0 and 100 representing the percentage of the open trade quantity to close
    :param profit: The take-profit distance, expressed in ticks
    :param limit: The take-profit price
    :param loss: The stop-loss distance, expressed in ticks
    :param stop: The stop-loss price
    :param trail_price: The price of the trailing stop activation level
    :param trail_points: The trailing stop activation distance, expressed in ticks
    :param trail_offset: The trailing stop offset
    :param oca_name: The name of the order cancel/replace group
    :param oca_type: The type of the order cancel/replace group
    :param comment: Additional notes on the filled order
    :param comment_profit: Additional notes on the filled order
    :param comment_loss: Additional notes on the filled order
    :param comment_trailing: Additional notes on the filled order
    :param alert_message: Custom text for the alert that fires when an order fills
    :param alert_profit: Custom text for the alert that fires when an order fills
    :param alert_loss: Custom text for the alert that fires when an order fills
    :param alert_trailing: Custom text for the alert that fires when an order fills
    :param disable_alert: If true, the alert will not fire when the order fills
    """
    if lib._lib_semaphore:
        return

    script = lib._script
    position = script.position

    if qty < 0.0:
        return

    direction = 0
    size = 0.0

    def _exit():
        nonlocal limit, stop, trail_price, from_entry, direction, size, oca_name, oca_type

        if isinstance(qty, NA):
            size = -size * (qty_percent * 0.01) if not isinstance(qty_percent, NA) else -size
        else:
            size = -direction * qty

        size = _size_round(size)
        if size == 0.0:
            return

        # Store tick values for later calculation when entry price is known
        profit_ticks = None if isinstance(profit, NA) else profit
        loss_ticks = None if isinstance(loss, NA) else loss
        trail_points_ticks = None if isinstance(trail_points, NA) else trail_points

        # We need to have limit, stop or both
        if isinstance(limit, NA) and isinstance(stop, NA) and not isinstance(trail_price, NA):
            return

        if isinstance(limit, NA):
            limit = None
        elif not isinstance(limit, NA):
            limit = _price_round(limit, direction)
        if isinstance(stop, NA):
            stop = None
        elif not isinstance(stop, NA):
            stop = _price_round(stop, -direction)
        if isinstance(trail_price, NA):
            trail_price = None
        elif not isinstance(trail_price, NA):
            trail_price = _price_round(trail_price, direction)

        # Default OCA settings for strategy.exit() - matches TradingView behavior
        # If no oca_name is specified, create a default OCA reduce group
        if isinstance(oca_name, NA):
            # Use a unique name based on the exit id and from_entry
            oca_name = f"__exit_{id}_{from_entry}_oca__"
            # Default to reduce type (TradingView behavior)
            oca_type = _oca.reduce
        else:
            # If oca_name is provided but no type, default to reduce
            if oca_type is None:
                oca_type = _oca.reduce

        # Store in exit_orders dict
        position.exit_orders[id] = Order(
            from_entry, size, exit_id=id, order_type=_order_type_close,
            limit=limit, stop=stop,
            trail_price=trail_price, trail_offset=trail_offset,
            profit_ticks=profit_ticks, loss_ticks=loss_ticks, trail_points_ticks=trail_points_ticks,
            oca_name=oca_name, oca_type=oca_type, comment=comment, alert_message=alert_message
        )

    # Find direction and size
    if from_entry:
        # Get from entry_orders dict
        entry_order = position.entry_orders.get(from_entry, None)

        # Find open trade if no entry order found
        if not entry_order:
            for trade in position.open_trades:
                if trade.entry_id == from_entry:
                    direction = trade.sign
                    size = trade.size
                    _exit()
        else:
            direction = entry_order.sign
            size = entry_order.size
            _exit()

    else:
        # If still no entry order found, we should exit all open trades and open orders
        if not direction:
            for order in list(position.entry_orders.values()):
                direction = order.sign
                size = order.size
                from_entry = order.order_id
                _exit()

            if not direction:
                for trade in position.open_trades:
                    direction = trade.sign
                    size = trade.size
                    from_entry = trade.entry_id
                    _exit()


# noinspection PyProtectedMember,PyShadowingNames,PyShadowingBuiltins,PyUnusedLocal
def order(id: str, direction: direction.Direction, qty: int | float | NA[float] = NA(float),
          limit: int | float | None = None, stop: int | float | None = None,
          oca_name: str | None = None, oca_type: _oca.Oca | None = None,
          comment: str | None = None, alert_message: str | None = None,
          disable_alert: bool = False):
    """
    Creates a new order to open, add to, or exit from a position. If an unfilled order with
    the same id exists, a call to this command modifies that order.

    Unlike strategy.entry, orders from this command are not affected by the pyramiding parameter
    of the strategy declaration. Strategies can open any number of trades in the same direction
    with calls to this function.

    This command does not automatically reverse open positions. For example, if there is an open
    long position of five shares, an order from this command with a qty of 5 and a direction
    of strategy.short triggers the sale of five shares, which closes the position.

    :param id: The identifier of the order
    :param direction: The direction of the trade (strategy.long or strategy.short)
    :param qty: The number of contracts/shares/lots/units to trade when the order fills
    :param limit: The limit price of the order (creates limit or stop-limit order)
    :param stop: The stop price of the order (creates stop or stop-limit order)
    :param oca_name: The name of the One-Cancels-All (OCA) group
    :param oca_type: Specifies how an unfilled order behaves when another order in the same OCA group executes
    :param comment: Additional notes on the filled order
    :param alert_message: Custom text for the alert that fires when an order fills
    :param disable_alert: If true, the strategy does not trigger an alert when the order fills
    """
    if lib._lib_semaphore:
        return

    script = lib._script
    position = script.position

    # Risk management: Check if trading is halted
    # TODO: investigate if it should be checked here
    if position.risk_halt_trading:
        return

    # Get default qty by script parameters if no qty is specified
    if isinstance(qty, NA):
        default_qty_type = script.default_qty_type
        if default_qty_type == fixed:
            qty = script.default_qty_value

        elif default_qty_type == percent_of_equity:
            default_qty_value = script.default_qty_value
            equity_percent = default_qty_value * 0.01
            target_investment = script.position.equity * equity_percent

            # Calculate the commission factor based on commission type
            if script.commission_type == _commission.percent:
                commission_multiplier = 1.0 + script.commission_value * 0.01
                qty = target_investment / (lib.close * syminfo.pointvalue * commission_multiplier)

            elif script.commission_type == _commission.cash_per_contract:
                price_plus_commission = lib.close * syminfo.pointvalue + script.commission_value
                qty = target_investment / price_plus_commission

            elif script.commission_type == _commission.cash_per_order:
                qty = (target_investment - script.commission_value) / (lib.close * syminfo.pointvalue)
                qty = max(0.0, qty)  # Ensure non-negative

            else:
                # No commission
                qty = target_investment / (lib.close * syminfo.pointvalue)

        elif default_qty_type == cash:
            default_qty_value = script.default_qty_value
            qty = default_qty_value / (lib.close * syminfo.pointvalue)

        else:
            raise ValueError("Unknown default qty type: ", default_qty_type)

    # qty must be greater than 0
    if qty <= 0.0:
        return

    # We need a signed size instead of qty, the sign is the direction
    direction_sign: float = (-1.0 if direction == short else 1.0)
    size = qty * direction_sign

    # NOTE: Unlike strategy.entry, strategy.order is NOT affected by pyramiding limit
    # This is a key difference - strategy.order can open unlimited trades in the same direction
    # It uses _order_type_normal to distinguish it from entry/exit orders

    size = _size_round(size)
    if size == 0.0:
        return

    if isinstance(limit, NA):
        limit = None
    elif limit is not None:
        limit = _price_round(limit, direction_sign)
    if isinstance(stop, NA):
        stop = None
    elif stop is not None:
        stop = _price_round(stop, -direction_sign)

    # Create the order with _order_type_normal
    # This is a "normal" order that simply adds to or subtracts from position
    # It doesn't follow entry/exit rules and can freely modify positions
    order = Order(id, size, order_type=_order_type_normal, limit=limit, stop=stop,
                  oca_name=oca_name, oca_type=oca_type, comment=comment,
                  alert_message=alert_message)
    position.entry_orders[id] = order


#
# Properties
#

# noinspection PyProtectedMember
@module_property
def equity() -> float | NA[float]:
    if lib._script is None or lib._script.position is None:
        return 0.0
    return lib._script.position.equity


# noinspection PyProtectedMember
@module_property
def eventrades() -> int | NA[int]:
    if lib._script is None or lib._script.position is None:
        return 0
    return lib._script.position.eventrades


# noinspection PyProtectedMember
@module_property
def initial_capital() -> float:
    if lib._script is None or lib._script.initial_capital is None:
        return 0.0
    return lib._script.initial_capital


# noinspection PyProtectedMember
@module_property
def grossloss() -> float | NA[float]:
    if lib._script is None or lib._script.position is None or lib._script.position.open_commission is None:
        return 0.0
    return lib._script.position.grossloss + lib._script.position.open_commission


# noinspection PyProtectedMember
@module_property
def grossprofit() -> float | NA[float]:
    if lib._script is None or lib._script.position is None:
        return 0.0
    return lib._script.position.grossprofit


# noinspection PyProtectedMember
@module_property
def losstrades() -> int:
    if lib._script is None or lib._script.position is None:
        return 0
    return lib._script.position.losstrades


# noinspection PyProtectedMember
@module_property
def max_drawdown() -> float | NA[float]:
    if lib._script is None or lib._script.position is None:
        return 0.0
    return lib._script.position.max_drawdown


# noinspection PyProtectedMember
@module_property
def max_runup() -> float | NA[float]:
    if lib._script is None or lib._script.position is None:
        return 0.0
    return lib._script.position.max_runup


# noinspection PyProtectedMember
@module_property
def netprofit() -> float | NA[float]:
    if lib._script is None or lib._script.position is None:
        return 0.0
    return lib._script.position.netprofit


# noinspection PyProtectedMember
@module_property
def openprofit() -> float | NA[float]:
    if lib._script is None or lib._script.position is None:
        return 0.0
    return lib._script.position.openprofit


# noinspection PyProtectedMember
@module_property
def position_size() -> float | NA[float]:
    if lib._script is None or lib._script.position is None:
        return 0.0
    return lib._script.position.size


# noinspection PyProtectedMember
@module_property
def position_avg_price() -> float | NA[float]:
    if lib._script is None or lib._script.position is None:
        return 0.0
    return lib._script.position.avg_price


# noinspection PyProtectedMember
@module_property
def wintrades() -> int | NA[int]:
    if lib._script is None or lib._script.position is None:
        return 0
    return lib._script.position.wintrades
