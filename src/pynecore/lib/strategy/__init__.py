from typing import cast, TYPE_CHECKING

import math
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
    "cancel", "cancel_all", "close", "close_all", "entry", "exit",

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
            oca_type: _oca.Oca | None = None,
            comment: str | None = None,
            alert_message: str | None = None,
            trail_price: float | None = None,
            trail_offset: float | None = None
    ):
        self.order_id = order_id
        self.size = size
        self.sign = 0.0 if size == 0.0 else 1.0 if size > 0.0 else -1.0
        self.limit = limit
        self.stop = stop
        self.order_type = order_type

        self.exit_id = exit_id

        self.oca_name = oca_name
        self.oca_type = oca_type

        self.comment = comment
        self.alert_message = alert_message

        self.trail_price = trail_price
        self.trail_offset = trail_offset or 0  # in ticks
        self.trail_triggered = False

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

    orders: dict[str, Order]

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
        self.orders = {}

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
        self.orders.clear()
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

    @property
    def equity(self) -> float | NA[float]:
        """ The current equity """
        assert lib._script is not None
        return lib._script.initial_capital + self.netprofit + self.openprofit

    def _fill_order(self, order: Order, price: float, h: float, l: float):
        """
        Fill an order (actually)

        :param order: The order to fill
        :param price: The price to fill at
        :param h: The high price
        :param l: The low price
        """
        script = lib._script
        assert script is not None
        commission_type = script.commission_type
        commission_value = script.commission_value

        new_closed_trades = []
        closed_trade_size = 0.0

        # Close order
        if self.size and order.order_type != _order_type_entry and order.sign != self.sign:
            delete = False

            # Check list of open trades
            open_trades = []
            for trade in self.open_trades:
                # Only use if its order id is the same
                if order.size != 0.0 and (trade.entry_id == order.order_id or order.order_id is None):
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

                    assert order.exit_id is not None

                    closed_trade.size = -size
                    closed_trade.exit_id = order.exit_id
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

                    # We realize later if it is cash per order
                    if commission_type == _commission.cash_per_contract:
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
                    if math.isclose(self.size, 0.0, abs_tol=1 / syminfo._size_round_factor):
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

                open_trades.append(trade)

            self.open_trades = open_trades
            if delete:
                if order.exit_id is not None:
                    self.orders.pop(order.exit_id, None)

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

            del self.orders[order.order_id]

        # If position has just closed
        if not self.open_trades:
            # Reset position variables
            self.entry_summ = 0.0
            self.avg_price = 0.0
            self.openprofit = 0.0
            self.open_commission = 0.0

    def fill_order(self, order: Order, price: float, h: float, l: float) -> bool:
        """
        Fill an order

        :param order: The order to fill
        :param price: The price to fill at
        :param h: The high price
        :param l: The low price
        :return: True if the side of the position has changed
        """
        # If position direction is about to change, we split it into two separate orders
        # This is necessary to create a new average entry price
        new_size = self.size + order.size
        if new_size != 0.0 and not math.isclose(new_size, 0.0,
                                                abs_tol=1 / syminfo._size_round_factor):  # Check for rounding errors
            new_size = 0.0
        new_sign = 0.0 if new_size == 0.0 else 1.0 if new_size > 0.0 else -1.0
        if self.size != 0.0 and new_sign != self.sign and new_size != 0.0:
            # Create a copy for closing existing position
            order1 = copy(order)
            order1.order_type = _order_type_close
            order1.size = -self.size
            # Set order_id to None so it will close any open trades
            order1.order_id = None
            # Don't need to remove this order, because it is not in the orders list
            order1.exit_id = None
            # Fill the closing order first
            self._fill_order(order1, price, h, l)
            # Modify the original order to open a position in the new direction
            order.size = new_size
            assert order.order_id is not None
            self.orders[order.order_id] = order
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
        if ((order.order_type == _order_type_close and order.size > 0) or (
                order.order_type == _order_type_entry and order.size > 0)) and order.stop <= self.h:
            p = max(order.stop, self.o)
            self.fill_order(order, p, p, self.l)

    def _check_high(self, order: Order):
        """ Check high limit """
        if order.limit is not None:
            if ((order.order_type == _order_type_close and order.size < 0) or (
                    order.order_type == _order_type_entry and order.size < 0)) and order.limit <= self.h:
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
        if ((order.order_type == _order_type_close and order.size < 0) or (
                order.order_type == _order_type_entry and order.size < 0)) and order.stop >= self.l:
            p = min(self.o, order.stop)
            self.fill_order(order, p, self.h, p)

    def _check_low(self, order: Order):
        """ Check low limit """
        if order.limit is not None:
            if ((order.order_type == _order_type_close and order.size > 0) or (
                    order.order_type == _order_type_entry and order.size > 0)) and order.limit >= self.l:
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

        # If the order is open → high → low → close or open → low → high → close
        ohlc = abs(self.h - self.o) < abs(self.l - self.o)

        self.drawdown_summ = self.runup_summ = 0.0
        self.new_closed_trades.clear()

        # Process open orders
        for order in list(self.orders.values()):
            # Market orders
            if order.limit is None and order.stop is None and order.trail_price is None:
                # open → high → low → close
                if ohlc:
                    self.fill_order(order, self.o, self.o, self.l)
                # open → low → high → close
                else:
                    self.fill_order(order, self.o, self.l, self.o)

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
        for order in list(self.orders.values()):
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

    try:
        assert lib._script is not None and lib._script.position is not None
        # noinspection PyProtectedMember
        del lib._script.position.orders[id]
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
    lib._script.position.orders.clear()


# noinspection PyProtectedMember,PyShadowingBuiltins
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

    position.orders[exit_id] = order
    if immediately:
        round_to_mintick = lib.math.round_to_mintick
        position.fill_order(order,
                            round_to_mintick(lib.close),
                            round_to_mintick(lib.high),
                            round_to_mintick(lib.low))


# noinspection PyProtectedMember
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

    position.orders[exit_id] = order
    if immediately:
        round_to_mintick = lib.math.round_to_mintick
        position.fill_order(order,
                            round_to_mintick(lib.close),
                            round_to_mintick(lib.high),
                            round_to_mintick(lib.low))


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
    assert script is not None and script.position is not None
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
                qty = target_investment / (lib.close * syminfo.pointvalue * commission_multiplier)

            elif script.commission_type == _commission.cash_per_contract:
                # For cash per contract: qty * price + qty * commission_value
                # qty * (price + commission_value) = target_investment
                price_plus_commission = lib.close * syminfo.pointvalue + script.commission_value
                qty = target_investment / price_plus_commission

            elif script.commission_type == _commission.cash_per_order:
                # For cash per order: qty * price + commission_value = target_investment
                # qty = (target_investment - commission_value) / price
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
    sign = 0.0 if size == 0.0 else 1.0 if size > 0.0 else -1.0

    # Change direction
    is_direction_change = False
    if position.size:
        if position.sign != sign:
            is_direction_change = True
        else:
            # Handle pyramiding
            if script.pyramiding <= len(script.position.open_trades):
                return

    # Risk management: Check allowed direction (only for new positions, not direction changes)
    if position.risk_allowed_direction is not None:
        if (sign > 0 and position.risk_allowed_direction != long) or \
                (sign < 0 and position.risk_allowed_direction != short):
            if not is_direction_change:
                return
            else:
                is_direction_change = False

    # We need to adjust the size if we are changing direction
    if is_direction_change:
        size -= position.size

    # Risk management: Check max position size
    if position.risk_max_position_size is not None:
        new_position_size = abs(position.size + size)
        if new_position_size > position.risk_max_position_size:
            # Adjust size to not exceed max position size
            max_allowed_size = position.risk_max_position_size - abs(position.size)
            if max_allowed_size <= 0:
                return
            size = max_allowed_size * sign

    # Risk management: Check max intraday filled orders
    if position.risk_max_intraday_filled_orders is not None:
        if position.risk_intraday_filled_orders >= position.risk_max_intraday_filled_orders:
            return

    size = _size_round(size)
    if size == 0.0:
        return

    if limit is not None:
        limit = _price_round(limit, direction_sign)
    if stop is not None:
        stop = _price_round(stop, -direction_sign)

    order = Order(id, size, order_type=_order_type_entry, limit=limit, stop=stop, oca_name=oca_name,
                  oca_type=oca_type, comment=comment, alert_message=alert_message)
    script.position.orders[id] = order


# noinspection PyShadowingBuiltins,PyProtectedMember,PyShadowingNames,PyUnusedLocal
def exit(id: str, from_entry: str = "",
         qty: float | NA[float] = NA(float), qty_percent: float | NA[float] = NA(float),
         profit: float | NA[float] = NA(float), limit: float | NA[float] = NA(float),
         loss: float | NA[float] = NA(float), stop: float | NA[float] = NA(float),
         trail_price: float | NA[float] = NA(float), trail_points: float | NA[float] = NA(float),
         trail_offset: float | NA[float] = NA(float),
         oca_name: str | NA[str] = NA(str),
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
    assert script is not None and script.position is not None
    position = script.position

    if qty < 0.0:
        return

    direction = 0
    size = 0.0

    def _exit():
        nonlocal limit, stop, trail_price, from_entry, direction, size

        if isinstance(qty, NA):
            size = -size * (qty_percent * 0.01) if not isinstance(qty_percent, NA) else -size
        else:
            size = -direction * qty

        size = _size_round(size)
        if size == 0.0:
            return

        if isinstance(limit, NA) and not isinstance(profit, NA):
            limit = lib.close + direction * syminfo.mintick * profit
        if isinstance(stop, NA) and not isinstance(loss, NA):
            stop = lib.close - direction * syminfo.mintick * loss
        if isinstance(trail_price, NA) and not isinstance(trail_points, NA):
            trail_price = lib.close + direction * syminfo.mintick * trail_points

        # We need to have limit, stop or both
        if isinstance(limit, NA) and isinstance(stop, NA) and not isinstance(trail_price, NA):
            return

        if not isinstance(limit, NA):
            limit = _price_round(limit, direction)
        if not isinstance(stop, NA):
            stop = _price_round(stop, -direction)
        if not isinstance(trail_price, NA):
            trail_price = _price_round(trail_price, direction)

        position.orders[id] = Order(from_entry, size, exit_id=id, order_type=_order_type_close,
                                    limit=limit, stop=stop,
                                    trail_price=trail_price, trail_offset=trail_offset,
                                    oca_name=oca_name, comment=comment, alert_message=alert_message)

    # Find direction and size
    if from_entry:
        entry_order = position.orders.get(from_entry, None)

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
            for order in list(position.orders.values()):
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
