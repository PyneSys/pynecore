from typing import TYPE_CHECKING, Literal, overload

import math
from abc import ABC, abstractmethod
from datetime import datetime, UTC
from collections import deque, defaultdict
from copy import copy
from bisect import insort, bisect_left

from ...core.module_property import module_property
from ... import lib
from .. import syminfo

from ...types.strategy import QtyType
from ...types.base import IntEnum
from ...types.na import NA, na_float, na_str
from ...types import PyneFloat, PyneInt, PyneStr

from . import direction as direction
from . import commission as _commission
from . import oca as _oca

from . import closedtrades, opentrades

__all__ = [
    "fixed", "cash", "percent_of_equity",
    "long", "short", 'direction',

    'Trade', 'Order', 'PositionBase', 'SimPosition',
    "cancel", "cancel_all", "close", "close_all", "entry", "exit", "order",

    "closedtrades", "opentrades",
]

#
# Callable modules
#

from ...types.ohlcv import OHLCV

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
# Helpers
#

@overload
def _na_to_none(value: PyneFloat) -> float | None: ...


@overload
def _na_to_none(value: PyneStr) -> str | None: ...


def _na_to_none(value):  # type: ignore[misc]
    """Convert NA to None, pass through everything else."""
    return None if isinstance(value, NA) else value


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
        "comment_profit", "comment_loss", "comment_trailing",
        "alert_profit", "alert_loss", "alert_trailing",
        "trail_price", "trail_offset",
        "trail_triggered", "trail_stop",
        "profit_ticks", "loss_ticks", "trail_points_ticks",  # Store tick values for later calculation
        "is_market_order",  # Flag to check if this is a market order
        "cancelled",  # Flag to mark order as cancelled by OCA
        "bar_index",  # Bar index when the order was placed
        "filled_by_type",  # Type of execution: 'profit', 'loss', 'trailing', or None
        "from_entry_na",  # True if exit was created without explicit from_entry (applies to any position)
        "reserved_size",  # Exit-leg slice of the entry's original size (frozen at creation)
        "consumed",  # True once an exit leg fired its slice while its entry is still open
    )

    def __init__(
            self,
            order_id: str | None,
            size: PyneFloat,
            *,
            order_type: _OrderType = _order_type_normal,
            exit_id: str | None = None,
            limit: float | None = None,
            stop: float | None = None,
            oca_name: str | None = None,
            oca_type: _oca.Oca | None = _oca.none,
            comment: PyneStr | None = None,
            alert_message: PyneStr | None = None,
            comment_profit: str | None = None,
            comment_loss: str | None = None,
            comment_trailing: str | None = None,
            alert_profit: str | None = None,
            alert_loss: str | None = None,
            alert_trailing: str | None = None,
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
        self.comment_profit = comment_profit
        self.comment_loss = comment_loss
        self.comment_trailing = comment_trailing
        self.alert_profit = alert_profit
        self.alert_loss = alert_loss
        self.alert_trailing = alert_trailing

        self.trail_price = trail_price
        self.trail_offset = trail_offset or 0  # in ticks
        self.trail_triggered = False
        self.trail_stop: float | None = None  # active trailing-stop level once triggered

        self.profit_ticks = profit_ticks
        self.loss_ticks = loss_ticks
        self.trail_points_ticks = trail_points_ticks

        # Check if this is a market order (no limit, stop, trail, or tick-based prices)
        self.is_market_order = (self.limit is None and self.stop is None
                                and self.trail_price is None
                                and self.profit_ticks is None
                                and self.loss_ticks is None
                                and self.trail_points_ticks is None)

        self.cancelled = False
        self.bar_index = -1  # Will be set when order is added to position
        self.filled_by_type: Literal['profit', 'loss', 'trailing'] | None = None  # Will be set when order fills
        self.from_entry_na = False
        self.reserved_size = abs(size)
        self.consumed = False

    def __repr__(self):
        return f"Order(order_id={self.order_id}; exit_id={self.exit_id}; size={self.size}; type: {self.order_type}; " \
               f"limit={self.limit}; stop={self.stop}; " \
               f"trail_price={self.trail_price}; trail_offset={self.trail_offset}; " \
               f"oca_name={self.oca_name}; comment={self.comment}; bar_index={self.bar_index})"


class Trade:
    """
    Represents a trade
    """

    __slots__ = (
        "size", "init_size", "sign", "entry_id", "entry_bar_index", "entry_time", "entry_price", "entry_comment", "entry_equity",
        "exit_id", "exit_bar_index", "exit_time", "exit_price", "exit_comment", "exit_equity",
        "commission", "max_drawdown", "max_drawdown_percent", "max_runup", "max_runup_percent",
        "profit", "profit_percent", "cum_profit", "cum_profit_percent",
        "cum_max_drawdown", "cum_max_runup"
    )

    # noinspection PyShadowingNames
    def __init__(self, *, size: PyneFloat, entry_id: str | None, entry_bar_index: int, entry_time: int,
                 entry_price: PyneFloat,
                 commission: PyneFloat, entry_comment: PyneStr | None = None,
                 entry_equity: PyneFloat = 0.0):
        self.size: PyneFloat = size
        # Original entry quantity, frozen — partial exits shrink ``size`` but
        # qty_percent / no-qty "rest" exit legs reserve off this value.
        self.init_size: PyneFloat = size
        self.sign = 0.0 if size == 0.0 else 1.0 if size > 0.0 else -1.0

        self.entry_id: str | None = entry_id
        self.entry_bar_index: int = entry_bar_index
        self.entry_time: int = entry_time
        self.entry_price: PyneFloat = entry_price
        self.entry_equity: PyneFloat = entry_equity
        self.entry_comment: PyneStr | None = entry_comment

        self.exit_id: str | None = ""
        self.exit_bar_index: int = -1
        self.exit_time: int = -1
        self.exit_price: PyneFloat = 0.0
        self.exit_comment: PyneStr = ''
        self.exit_equity: PyneFloat = na_float

        self.commission = commission

        self.max_drawdown: PyneFloat = 0.0
        self.max_drawdown_percent: PyneFloat = 0.0
        self.max_runup: PyneFloat = 0.0
        self.max_runup_percent: PyneFloat = 0.0
        self.profit: PyneFloat = 0.0
        self.profit_percent: PyneFloat = 0.0

        self.cum_profit: PyneFloat = 0.0
        self.cum_profit_percent: PyneFloat = 0.0
        self.cum_max_drawdown: PyneFloat = 0.0
        self.cum_max_runup: PyneFloat = 0.0

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
        if key in ('entry_time', 'exit_time') and isinstance(v, (int, float)):
            v = datetime.fromtimestamp(v / 1000.0, tz=UTC)
        elif isinstance(v, float):
            v = round(v, 10)
        return v


# noinspection PyShadowingNames,DuplicatedCode
class PriceOrderBook:
    """
    Price-based sorted order storage.
    An order can appear multiple times at different prices.
    """

    __slots__ = ('price_levels', 'orders_at_price', 'order_prices')

    def __init__(self):
        self.price_levels = []  # Sorted list of prices
        self.orders_at_price = defaultdict(list)  # price -> [Order]
        self.order_prices = defaultdict(set)  # Order -> {prices}

    def add_order(self, order: Order):
        """Add order to all its relevant price levels.

        Idempotent per (order, price): callers that re-invoke after materializing
        an additional side (e.g. close-pass / `_process_at_bar_open` resolving
        `loss_ticks` on an exit that already had an explicit `limit`) won't
        double-index the side that was already in the book. `remove_order`
        only removes one occurrence per price level, so a duplicate could
        otherwise survive past `_remove_order` and re-fill on the next bar.
        """
        existing = self.order_prices[order]

        # Add to stop price if exists
        if order.stop is not None and order.stop not in existing:
            price = order.stop
            if price not in self.orders_at_price:
                insort(self.price_levels, price)
            self.orders_at_price[price].append(order)
            existing.add(price)

        # Add to limit price if exists
        if order.limit is not None and order.limit not in existing:
            price = order.limit
            if price not in self.orders_at_price:
                insort(self.price_levels, price)
            self.orders_at_price[price].append(order)
            existing.add(price)

        # Add to trail price if exists
        if order.trail_price is not None and order.trail_price not in existing:
            price = order.trail_price
            if price not in self.orders_at_price:
                insort(self.price_levels, price)
            self.orders_at_price[price].append(order)
            existing.add(price)

    def remove_order(self, order: Order):
        """Remove order from all price levels"""
        for price in list(self.order_prices[order]):
            self.orders_at_price[price].remove(order)
            if not self.orders_at_price[price]:
                idx = bisect_left(self.price_levels, price)
                if idx < len(self.price_levels) and self.price_levels[idx] == price:
                    del self.price_levels[idx]
                del self.orders_at_price[price]
        del self.order_prices[order]

    def iter_orders(self, *, desc=False, min_price: float | None = None, max_price: float | None = None):
        """
        Iterate over orders within price range.

        Examples:
            iter_orders()  # All orders, ascending
            iter_orders(desc=True)  # All orders, descending
            iter_orders(min_price=50.0)  # 50, 51, 52, ... (ascending)
            iter_orders(max_price=60.0)  # 60, 59, 58, ... (descending)
            iter_orders(min_price=50.0, max_price=60.0)  # 50, 51, ..., 60 (ascending)

        :param desc: If True, iterate in descending order, only if no min_price or max_price is set
        :param min_price: If set, iterate from this price upward (ascending)
        :param max_price: If set, iterate from this price downward (descending)
        :return: Generator yielding Order objects
        """
        if min_price is not None and max_price is not None:
            # Range query - ascending from min to max (or descending when desc=True,
            # e.g. the open->low price walk, where the level nearest the open is
            # reached first in time). Price levels reverse; within a level the
            # insertion order is preserved so same-price ties keep their sequence.
            min_idx = bisect_left(self.price_levels, min_price)
            max_idx = bisect_left(self.price_levels, max_price)
            # Include max_price if it matches exactly
            if max_idx < len(self.price_levels) and self.price_levels[max_idx] == max_price:
                max_idx += 1
            # Create a copy of price levels to avoid iteration issues when levels are removed
            levels = list(self.price_levels[min_idx:max_idx])
            if desc:
                levels.reverse()
            for p in levels:
                # Create a copy to avoid iteration issues when orders are removed during iteration
                yield from list(self.orders_at_price[p])

        elif min_price is not None:
            # Ascending from min_price
            min_idx = bisect_left(self.price_levels, min_price)
            # Create a copy of price levels to avoid iteration issues when levels are removed
            for p in list(self.price_levels[min_idx:]):
                # Create a copy to avoid iteration issues when orders are removed during iteration
                yield from list(self.orders_at_price[p])

        elif max_price is not None:
            # Descending from max_price
            max_idx = bisect_left(self.price_levels, max_price)
            # Include max_price if it matches exactly
            if max_idx < len(self.price_levels) and self.price_levels[max_idx] == max_price:
                max_idx += 1
            # Iterate in reverse order (high to low prices)
            # Create a copy of price levels to avoid iteration issues when levels are removed
            # Note: reversed() already creates an iterator over a copy of the slice
            for p in reversed(list(self.price_levels[:max_idx])):
                # Create a copy to avoid iteration issues when orders are removed during iteration
                yield from list(self.orders_at_price[p])

        elif desc:
            # All orders, descending
            # Create a copy of price levels to avoid iteration issues when levels are removed
            for p in reversed(list(self.price_levels)):
                # Create a copy to avoid iteration issues when orders are removed during iteration
                yield from list(self.orders_at_price[p])
        else:
            # All orders, ascending
            # Create a copy of price levels to avoid iteration issues when levels are removed
            for p in list(self.price_levels):
                # Create a copy to avoid iteration issues when orders are removed during iteration
                yield from list(self.orders_at_price[p])

    def clear(self):
        """Clear all orders"""
        self.price_levels.clear()
        self.orders_at_price.clear()
        self.order_prices.clear()


# noinspection PyProtectedMember,PyShadowingNames
class PositionBase(ABC):
    """
    Abstract base class for position tracking.

    Both backtest simulation (:class:`SimPosition`) and live broker trading
    (:class:`pynecore.core.broker.position.BrokerPosition`) subclass this.
    The Pine Script API surface — ``strategy.position_size``,
    ``strategy.opentrades``, ``strategy.netprofit``, ``strategy.equity``,
    etc. — reads the attributes declared here, so concrete subclasses MUST
    initialize all of them in ``__init__``.
    """
    __slots__ = ()

    # Attribute surface (declared for documentation and type-checking only —
    # concrete subclasses declare these in ``__slots__`` and initialize them).
    size: float
    sign: float
    avg_price: PyneFloat
    netprofit: PyneFloat
    openprofit: PyneFloat
    grossprofit: PyneFloat
    grossloss: PyneFloat
    open_commission: float
    # Current-bar OHLC the order-fill checks read off the position
    # (sim tracks them as slots; broker serves them from the live feed).
    c: float
    h: float
    l: float
    eventrades: int
    wintrades: int
    losstrades: int
    max_drawdown: float
    max_runup: float
    open_trades: list['Trade']
    closed_trades: 'deque[Trade]'
    new_closed_trades: list['Trade']
    entry_orders: dict[str | None, 'Order']
    exit_orders: dict[tuple[str | None, str | None], 'Order']
    risk_halt_trading: bool

    # Risk management state shared by Sim and Broker positions. Setters
    # in :mod:`pynecore.lib.strategy.risk` populate the ``risk_max_*`` fields;
    # the ``risk_*`` runtime counters are updated by the concrete subclass.
    risk_allowed_direction: 'direction.Direction | None'
    risk_max_drawdown_value: float | None
    risk_max_drawdown_type: 'QtyType | None'
    risk_max_drawdown_alert: str | None
    risk_max_intraday_loss_value: float | None
    risk_max_intraday_loss_type: 'QtyType | None'
    risk_max_intraday_loss_alert: str | None
    risk_max_cons_loss_days: int | None
    risk_max_cons_loss_days_alert: str | None
    risk_max_intraday_filled_orders: int | None
    risk_max_intraday_filled_orders_alert: str | None
    risk_max_position_size: float | None
    risk_intraday_start_equity: float
    risk_intraday_filled_orders: int
    risk_cons_loss_days: int

    @property
    def equity(self) -> PyneFloat:
        """The current equity (initial capital + realized + unrealized P&L)."""
        return lib._script.initial_capital + self.netprofit + self.openprofit

    # === Risk-rule predicates (shared by Sim and Broker positions) ===

    def _peak_equity(self) -> float:
        """Reference equity for ``max_drawdown(..., percent_of_equity)``.

        TradingView measures drawdown from the running peak equity, so the
        percent threshold scales with the high-water mark — a strategy that
        grows from $10k to $20k and is configured with ``max_drawdown(30%)``
        tolerates a $6k drawdown from $20k, not $3k from initial capital.

        Subclasses that track a peak (``SimPosition.max_equity``) override
        this; the base falls back to initial capital, which matches the
        first-bar value before any equity history exists.
        """
        return float(lib._script.initial_capital)

    def _is_max_drawdown_breached(self) -> bool:
        if self.risk_max_drawdown_value is None:
            return False
        if self.risk_max_drawdown_type == percent_of_equity:
            threshold = self._peak_equity() * self.risk_max_drawdown_value * 0.01
        else:
            threshold = float(self.risk_max_drawdown_value)
        return self.max_drawdown >= threshold > 0.0

    def _is_max_intraday_loss_breached(self) -> bool:
        if self.risk_max_intraday_loss_value is None:
            return False
        # Per TV docs: percent_of_equity for max_intraday_loss is measured
        # against the start-of-day equity (the same anchor used for the loss
        # delta), so the threshold scales with the day's opening capital
        # rather than the initial-bar capital.
        if self.risk_max_intraday_loss_type == percent_of_equity:
            threshold = self.risk_intraday_start_equity * self.risk_max_intraday_loss_value * 0.01
        else:
            threshold = float(self.risk_max_intraday_loss_value)
        intraday_loss = self.risk_intraday_start_equity - float(self.equity)
        return intraday_loss >= threshold > 0.0

    def _is_max_cons_loss_days_breached(self) -> bool:
        if self.risk_max_cons_loss_days is None:
            return False
        return self.risk_cons_loss_days >= self.risk_max_cons_loss_days > 0

    # === Pre-fill / pre-submit gates (shared by sim fill loop and broker submit) ===
    # These mirror the inline checks in :meth:`SimPosition.fill_order` so that
    # :class:`~pynecore.core.broker.position.BrokerPosition` can enforce the
    # same policy at its pre-submit boundary (``_add_order``) without
    # duplicating the logic. Sim and broker hit the same predicate at
    # different points in the order lifecycle — sim at fill time, broker at
    # submit time — but the rule body is identical.

    def _is_intraday_filled_cap_reached(self) -> bool:
        """``risk_max_intraday_filled_orders`` already at/above the cap.

        Caller rejects the new entry/normal order when this returns True.
        Mirrors the sim ``is not None`` check; a stored cap of ``0`` is
        treated as "all orders blocked" by both sites — the
        :mod:`~pynecore.lib.strategy.risk` setter is responsible for
        normalizing the no-limit sentinel.
        """
        cap = self.risk_max_intraday_filled_orders
        if cap is None:
            return False
        return self.risk_intraday_filled_orders >= cap

    def _adjust_for_max_position_size(
            self, intent_size: float, intent_sign: float,
    ) -> float | None:
        """Honor ``risk_max_position_size``; trim the order or reject it.

        :param intent_size: Signed order size requested by the caller.
        :param intent_sign: ``+1.0`` for buy intents, ``-1.0`` for sell.
        :return: Possibly trimmed signed size (caller proceeds with this),
                 the original ``intent_size`` if no cap is set or no trim
                 needed, or ``None`` if the cap is already met and the order
                 must be rejected.
        """
        cap = self.risk_max_position_size
        if cap is None:
            return intent_size
        new_position_size = abs(self.size + intent_size)
        if new_position_size <= cap:
            return intent_size
        max_allowed_size = cap - abs(self.size)
        if max_allowed_size <= 0:
            return None
        return max_allowed_size * intent_sign

    def _is_direction_allowed(self, intent_sign: float) -> bool:
        """``risk_allowed_direction`` permits an entry/flip in this direction.

        The caller decides *when* to consult this (sim only checks on
        ``size == 0``; broker checks at every submit). The helper itself is
        stateless w.r.t. current position size — it only inspects the
        configured allowed direction.
        """
        allowed = self.risk_allowed_direction
        if allowed is None:
            return True
        if intent_sign > 0:
            return allowed == long
        if intent_sign < 0:
            return allowed == short
        return True

    def _seed_trail_at_issue(self, order: 'Order') -> None:
        """Sim-only hook: fold the issue bar's extreme into a freshly issued
        trailing exit's water mark.

        This is a backtest price-walk concern. The live broker path tracks the
        trailing stop through the exchange / order-sync engine, so the base
        implementation is a no-op; :class:`SimPosition` overrides it with the
        backtest behaviour.
        """
        return None

    @abstractmethod
    def _add_order(self, order: 'Order') -> None:
        """Register an order with this position."""

    @abstractmethod
    def _remove_order(self, order: 'Order') -> None:
        """Cancel/remove an order from this position."""

    @abstractmethod
    def _remove_order_by_id(self, order_id: str) -> None:
        """Remove an order by its id (searches both exit and entry books)."""

    @abstractmethod
    def _cancel_all_orders(self) -> None:
        """Cancel every pending entry/exit order tracked by this position."""


# noinspection PyProtectedMember,PyShadowingNames,DuplicatedCode
class SimPosition(PositionBase):
    """
    Backtest simulation of position and trade state.

    Reproduces TradingView's strategy simulator faithfully: OHLC-based fill
    detection, synthetic slippage, margin-call emulation, gap-through logic,
    OCA reduce/cancel handling, trailing-stop tracking, etc.

    Live broker trading uses :class:`BrokerPosition` instead — exchange fills
    override all of the simulator logic below.
    """

    __slots__ = (
        'h', 'l', 'c', 'o',
        'netprofit', 'openprofit', 'grossprofit', 'grossloss',
        'entry_orders', 'exit_orders', 'market_orders', 'orderbook',
        'open_trades', 'closed_trades', 'new_closed_trades',
        'closed_trades_count', 'wintrades', 'eventrades', 'losstrades',
        'size', 'sign', 'avg_price', 'cum_profit',
        'entry_equity', 'max_equity', 'min_equity',
        'drawdown_summ', 'runup_summ', 'max_drawdown', 'max_runup',
        'entry_summ', 'open_commission',
        'risk_allowed_direction', 'risk_max_cons_loss_days', 'risk_max_cons_loss_days_alert',
        'risk_max_drawdown_value', 'risk_max_drawdown_type', 'risk_max_drawdown_alert',
        'risk_max_intraday_filled_orders', 'risk_max_intraday_filled_orders_alert',
        'risk_max_intraday_loss_value', 'risk_max_intraday_loss_type', 'risk_max_intraday_loss_alert',
        'risk_max_position_size',
        'risk_cons_loss_days', 'risk_last_trading_day', 'risk_last_day_equity',
        'risk_intraday_filled_orders', 'risk_intraday_start_equity', 'risk_halt_trading',
        '_deferred_margin_call', '_fill_counter'
    )

    def __init__(self):
        # OHLC values
        self.h: float = 0.0
        self.l: float = 0.0
        self.c: float = 0.0
        self.o: float = 0.0

        # Profit/loss tracking
        self.netprofit: PyneFloat = 0.0
        self.openprofit: PyneFloat = 0.0
        self.grossprofit: PyneFloat = 0.0
        self.grossloss: PyneFloat = 0.0

        # Order books
        self.market_orders: dict[tuple[_OrderType, str | None, str | None], Order] = {}  # Market orders from strategy.market()
        self.entry_orders: dict[str | None, Order] = {}  # Entry orders from strategy.entry()
        # Exit orders from strategy.exit(), strategy.close(), etc.
        # Key is (exit_id, from_entry) — both partial-TP fan-out (same from_entry,
        # different ids) and from_entry_na fan-out (same id, different from_entry)
        # must coexist; only repeated calls with both fields equal modify-in-place.
        self.exit_orders: dict[tuple[str | None, str | None], Order] = {}
        self.orderbook = PriceOrderBook()

        # Trades
        self.open_trades: list[Trade] = []
        self.closed_trades: deque[Trade] = deque(maxlen=9000)  # 9000 is the limit of TV
        self.new_closed_trades: list[Trade] = []

        # Trade statistics
        self.closed_trades_count: int = 0
        self.wintrades: int = 0
        self.eventrades: int = 0
        self.losstrades: int = 0
        self.size: float = 0.0
        self.sign: float = 0.0
        self.avg_price: PyneFloat = na_float
        self.cum_profit: PyneFloat = 0.0
        self.entry_equity: PyneFloat = 0.0
        self.max_equity: PyneFloat = -float("inf")
        self.min_equity: PyneFloat = float("inf")
        self.drawdown_summ: float = 0.0
        self.runup_summ: float = 0.0
        self.max_drawdown: float = 0.0
        self.max_runup: float = 0.0
        self.entry_summ: PyneFloat = 0.0
        self.open_commission: float = 0.0

        # Risk management settings
        self.risk_allowed_direction: direction.Direction | None = None
        self.risk_max_cons_loss_days: int | None = None
        self.risk_max_cons_loss_days_alert: str | None = None
        self.risk_max_drawdown_value: float | None = None
        self.risk_max_drawdown_type: QtyType | None = None
        self.risk_max_drawdown_alert: str | None = None
        self.risk_max_intraday_filled_orders: int | None = None
        self.risk_max_intraday_filled_orders_alert: str | None = None
        self.risk_max_intraday_loss_value: float | None = None
        self.risk_max_intraday_loss_type: QtyType | None = None
        self.risk_max_intraday_loss_alert: str | None = None
        self.risk_max_position_size: float | None = None

        # Risk management state tracking
        self.risk_cons_loss_days: int = 0
        self.risk_last_trading_day: int = -1
        self.risk_last_day_equity: float = 0.0
        self.risk_intraday_filled_orders: int = 0
        self.risk_intraday_start_equity: float = 0.0
        self.risk_halt_trading: bool = False

        # Deferred margin call (mc_size==1 and AF@C<0: fire after script runs)
        self._deferred_margin_call: tuple[float, bool] | None = None
        self._fill_counter: int = 0

    def _add_order(self, order: Order):
        """ Add an order to the strategy """
        # Set the bar_index when the order is placed
        order.bar_index = int(lib.bar_index)

        # Add market order to market orders dict. Key on exit_id too: two
        # brackets sharing the same from_entry (order_id) would otherwise
        # collide on the same key, so a second gap-through exit would evict
        # the first and only one of them would fill on the gap bar.
        if order.is_market_order:
            self.market_orders[(order.order_type, order.order_id, order.exit_id)] = order

        # Check if an order with this ID already exists and remove it first
        if order.order_type == _order_type_close:
            exit_key = (order.exit_id, order.order_id)
            existing_order = self.exit_orders.get(exit_key)
            self.exit_orders[exit_key] = order
        else:
            # Both entry and normal orders are stored in entry_orders dict
            existing_order = self.entry_orders.get(order.order_id)
            self.entry_orders[order.order_id] = order

        # Remove existing order from order book before adding new one
        if existing_order is not None:
            self.orderbook.remove_order(existing_order)

        # Add order to order book (automatically adds to all relevant prices)
        self.orderbook.add_order(order)

    def _remove_order(self, order: Order):
        """ Remove an order from the strategy """
        order.cancelled = True
        if order.order_type == _order_type_close:
            self.exit_orders.pop((order.exit_id, order.order_id), None)
        else:
            # Both entry and normal orders are stored in entry_orders dict
            self.entry_orders.pop(order.order_id, None)
        # Remove market order from market orders dict
        if order.is_market_order:
            self.market_orders.pop((order.order_type, order.order_id, order.exit_id), None)
        # Remove order from order book
        self.orderbook.remove_order(order)

    def _remove_order_by_id(self, order_id: str):
        """ Remove order by id """
        # TV-verified semantics (FX:EURUSD 60min, 2026-05-04): cancel matches an exit
        # by its exit_id only, and an entry by its entry id. NO cross-matching —
        # cancel(entry_id) does not cascade to exits that referenced it via from_entry.
        for exit_order in list(self.exit_orders.values()):
            if exit_order.exit_id == order_id:
                self._remove_order(exit_order)

        order = self.entry_orders.get(order_id)
        if order:
            self._remove_order(order)

    def _cancel_all_orders(self) -> None:
        self.entry_orders.clear()
        self.exit_orders.clear()
        self.orderbook.clear()

    def _cancel_oca_group(self, oca_name: str, executed_order: Order):
        """Cancel all orders in the same OCA group except the executed one"""
        # Cancel entry orders in the same OCA group
        for order in list(self.entry_orders.values()):
            if order.oca_name == oca_name and order != executed_order:
                self._remove_order(order)

        # Cancel exit orders in the same OCA group (consumed tombstones are
        # retired — they keep their reservation until the entry fully closes)
        for order in list(self.exit_orders.values()):
            if order.oca_name == oca_name and order != executed_order and not order.consumed:
                self._remove_order(order)

    def _reduce_oca_group(self, oca_name: str, filled_size: PyneFloat):
        """Reduce the size of all orders in the same OCA group"""
        reduction = abs(filled_size)

        # Reduce entry orders
        for order in list(self.entry_orders.values()):
            if order.oca_name == oca_name and not order.cancelled:
                new_size = abs(order.size) - reduction
                if new_size <= 0:
                    # Mark order as cancelled if size would be 0 or negative
                    self._remove_order(order)
                else:
                    # Keep original sign
                    order.size = new_size * order.sign

        # Reduce exit orders (skip consumed tombstones: a leg that fired its
        # slice is retired and keeps its reservation until the entry closes)
        for order in list(self.exit_orders.values()):
            if order.oca_name == oca_name and not order.cancelled and not order.consumed:
                new_size = abs(order.size) - reduction
                if new_size <= 0:
                    self._remove_order(order)
                else:
                    order.size = new_size * order.sign

    def _fill_order(self, order: Order, price: PyneFloat, h: PyneFloat, l: PyneFloat,
                    counts_as_filled_order: bool = True):
        """
        Fill an order (actually)

        :param order: The order to fill
        :param price: The price to fill at
        :param h: The high price
        :param l: The low price
        :param counts_as_filled_order: Whether this fill increments the
                                       ``max_intraday_filled_orders`` counter.
                                       ``False`` for the open half of a
                                       position-reversing order, whose close
                                       half already counted it once — TV treats
                                       a reversal as a single filled order.
        """
        # Close orders cannot fill when no position exists
        if order.order_type == _order_type_close and self.size == 0.0:
            return

        self._fill_counter += 1

        # Save the original order size before any modifications
        filled_size = abs(order.size)

        script = lib._script
        commission_type = script.commission_type
        commission_value = script.commission_value
        # USD value per 1.0-point move per 1 contract — futures-aware PnL conversion factor
        pv = syminfo.pointvalue

        new_closed_trades = []
        closed_trade_size = 0.0

        # Close order - if it is an exit order or a normal order
        if self.size and order.sign != self.sign:
            delete = False

            # Check list of open trades
            new_open_trades = []
            for trade in self.open_trades:
                # Only use if its order id is the same
                if order.size != 0.0 and ((trade.entry_id == order.order_id and order.order_type == _order_type_close)
                                          or order.order_type != _order_type_close
                                          or order.order_id is None):
                    delete = True

                    size = order.size if abs(order.size) <= abs(trade.size) else -trade.size
                    pnl = -size * (price - trade.entry_price) * pv

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
                    hprofit = (-size * (h - closed_trade.entry_price) * pv - closed_trade.commission)
                    lprofit = (-size * (l - closed_trade.entry_price) * pv - closed_trade.commission)

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

                    # Select appropriate comment based on filled_by_type
                    if order.filled_by_type == 'profit' and order.comment_profit:
                        closed_trade.exit_comment = order.comment_profit
                    elif order.filled_by_type == 'loss' and order.comment_loss:
                        closed_trade.exit_comment = order.comment_loss
                    elif order.filled_by_type == 'trailing' and order.comment_trailing:
                        closed_trade.exit_comment = order.comment_trailing
                    elif order.comment:
                        closed_trade.exit_comment = order.comment

                    # Commission summ
                    self.open_commission -= closed_trade.commission

                    # cash_per_order is a flat fee per order: defer realization
                    # until the order is removed so it can be split across all
                    # closed trades it actually filled (see delete block below).
                    if commission_type == _commission.cash_per_order:
                        closed_trade_size += abs(size)
                    else:
                        # Calculate exit commission based on commission type
                        if commission_type == _commission.percent:
                            # For percentage commission, multiply by exit price
                            commission = abs(size) * price * pv * commission_value * 0.01
                        else:
                            # cash_per_contract: size-proportional, charged per leg
                            commission = abs(size) * commission_value

                        closed_trade.commission += commission
                        # Realize commission
                        self.netprofit -= commission
                        closed_trade.profit -= closed_trade.commission

                    # Profit percent — both profit and entry_value are in USD
                    entry_value = abs(closed_trade.size) * closed_trade.entry_price * pv
                    try:
                        # Use closed_trade.profit which includes commission, not pnl which doesn't
                        closed_trade.profit_percent = (closed_trade.profit / entry_value) * 100.0
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

                    # Cancel exit orders for closed trades (TradingView behavior)
                    # When a trade is fully closed, remove its associated exit orders
                    if trade.size == 0.0:
                        # Remove exit orders that have from_entry matching this trade's entry_id
                        exit_orders_to_remove = []
                        for exit_order_id, exit_order in self.exit_orders.items():
                            if exit_order.order_id == trade.entry_id:
                                exit_orders_to_remove.append(exit_order_id)
                        for exit_order_id in exit_orders_to_remove:
                            self._remove_order(self.exit_orders[exit_order_id])

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
                        self.openprofit = self.size * (self.c - self.avg_price) * pv
                    else:
                        # If position has just closed
                        self.avg_price = na_float
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
                # A partial-exit leg that fired its whole slice while its entry's
                # position is still open becomes a tombstone: kept in exit_orders
                # (so its reservation still counts against sibling "rest" legs and
                # a per-bar strategy.exit() re-call cannot resurrect it) and only
                # pulled from the order book. It is purged when the entry fully
                # closes (the trade.size == 0.0 block above).
                if (order.order_type == _order_type_close and order.order_id is not None
                        and _size_round(order.size) == 0.0
                        and any(t.entry_id == order.order_id for t in self.open_trades)):
                    order.consumed = True
                    self.orderbook.remove_order(order)
                else:
                    self._remove_order(order)

                if commission_type == _commission.cash_per_order:
                    # Realize commission
                    self.netprofit -= commission_value
                    for trade in new_closed_trades:
                        commission = (commission_value * abs(trade.size)) / closed_trade_size
                        trade.commission += commission

            self.new_closed_trades.extend(new_closed_trades)

            # close_all overshoot: when deferred MC reduced position, close_all
            # captures original size and overshoots → create opposite position
            if (order.order_id is None and order.size != 0.0 and
                    order.order_type == _order_type_close):
                entry_id = order.exit_id
                overshoot_trade = Trade(
                    size=order.size,
                    entry_id=entry_id, entry_bar_index=int(lib.bar_index),
                    entry_time=lib._time, entry_price=price,
                    commission=0.0, entry_comment=order.comment,
                    entry_equity=self.equity
                )
                self.open_trades.append(overshoot_trade)
                self.size += overshoot_trade.size
                self.sign = 1.0 if self.size > 0.0 else -1.0 if self.size < 0.0 else 0.0
                self.entry_summ = price * abs(overshoot_trade.size)
                self.avg_price = price
                self.openprofit = self.size * (self.c - self.avg_price) * pv
                if not new_closed_trades:
                    self.entry_equity = self.equity
                    self.max_equity = max(self.max_equity, self.equity)
                    self.min_equity = min(self.min_equity, self.equity)

        # New trade
        elif order.order_type != _order_type_close:
            # Calculate commission
            if commission_value:
                if commission_type == _commission.cash_per_order:
                    commission = commission_value
                elif commission_type == _commission.percent:
                    commission = abs(order.size) * price * pv * commission_value * 0.01
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

            # For close_all overshoot, use exit_id as entry_id
            entry_id = order.order_id if order.order_id is not None else order.exit_id

            trade = Trade(
                size=order.size,
                entry_id=entry_id, entry_bar_index=int(lib.bar_index),
                entry_time=lib._time, entry_price=price,
                commission=commission, entry_comment=order.comment,
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
                self.avg_price = na_float
            # Unrealized P&L
            self.openprofit = self.size * (self.c - self.avg_price) * pv
            # Commission summ
            self.open_commission += commission

            # Remove order
            self._remove_order(order)

        # If position has just closed
        if not self.open_trades:
            # Reset position variables
            self.entry_summ = 0.0
            self.avg_price = na_float
            self.openprofit = 0.0
            self.open_commission = 0.0

            # Cancel all exit orders when position is closed (TradingView behavior)
            # Skip exits that have a pending entry (needed during position flips)
            exit_orders_to_remove = list(self.exit_orders.values())
            for exit_order in exit_orders_to_remove:
                if exit_order.order_id in self.entry_orders:
                    continue
                self._remove_order(exit_order)

        # Count this fill toward strategy.risk.max_intraday_filled_orders.
        # TradingView counts every filled order (entry, exit, normal) toward the
        # limit, but a position-reversing order is a SINGLE filled order even
        # though the sim executes it as a close followed by an open — the open
        # half passes counts_as_filled_order=False so the reversal counts once.
        if counts_as_filled_order:
            self.risk_intraday_filled_orders += 1

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
        close_only = False
        # Apply risk management only to entry orders, not normal orders from strategy.order()
        if order.order_type == _order_type_entry or order.order_type == _order_type_normal:
            # Pre-fill risk gates — shared with BrokerPosition pre-submit so
            # the same policy applies regardless of execution mode.
            if self._is_intraday_filled_cap_reached():
                self._remove_order(order)
                return False
            adjusted = self._adjust_for_max_position_size(float(order.size), order.sign)
            if adjusted is None:
                self._remove_order(order)
                return False
            order.size = adjusted
            if self.size == 0.0 and not self._is_direction_allowed(order.sign):
                self._remove_order(order)
                return False

            if order.order_type == _order_type_entry:
                # If we have an existing position
                if self.size != 0.0:
                    # Check if the order has the same direction
                    if self.sign == order.sign:
                        # Check pyramiding limit for entry orders adding to existing position
                        if lib._script.pyramiding <= len(self.open_trades):
                            # Pyramiding limit reached - don't fill the entry order
                            self._remove_order(order)
                            return False

        # For normal orders (_order_type_normal), no special risk management or pyramiding limits apply
        # They simply add to or subtract from the position as requested

        # If position direction is about to change, we split it into two separate orders
        # This is necessary to create a new average entry price
        # Note: The flip quantity is already calculated in entry() for entry orders
        new_size = self.size + order.size
        if _size_round(new_size) == 0.0:
            new_size = 0.0
        new_sign = 0.0 if new_size == 0.0 else 1.0 if new_size > 0.0 else -1.0
        if self.size != 0.0 and new_sign != self.sign and new_size != 0.0:
            # Exit orders should never reverse position direction
            # Only entry orders can open new positions or reverse direction
            if (order.order_type == _order_type_close or close_only) and order.order_id is not None:
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
            if not self._is_direction_allowed(new_direction_sign):
                # Direction not allowed - convert entry to exit only
                # Don't open new position in restricted direction
                self._remove_order(order)
                return False

            # Modify the original order to open a position in the new direction
            order.size = new_size
            # close_all overshoot: change type to allow opening new trade
            if order.order_type == _order_type_close:
                order.order_type = _order_type_normal
            # Fill the entry order. The close half above already counted this
            # reversal toward the intraday filled-orders cap, so the open half
            # must not count it a second time.
            self._fill_order(order, price, h, l, counts_as_filled_order=False)
            # A reversal that hits the cap is flattened too — same as the
            # non-flip path. Without this the cap-close never fires for a
            # position-reversing strategy, the common TradingView idiom.
            if self._is_intraday_filled_cap_reached() and self.size != 0.0:
                self._close_position_at_intraday_cap(order, price)
            return True

        # If position direction is not about to change, we can fill the order directly
        else:
            self._fill_order(order, price, h, l)

            # After filling, close the position if this fill hit the intraday cap
            # (TradingView flattens for the rest of the day; the counter blocks
            # new entries until it resets next day).
            if self._is_intraday_filled_cap_reached() and self.size != 0.0:
                self._close_position_at_intraday_cap(order, price)

            return False

    def _peak_equity(self) -> float:
        """Running high-water mark equity for percent-based drawdown threshold.

        Falls back to initial capital before any fill — ``max_equity`` is
        ``-inf`` until the first ``_fill_order`` updates it.
        """
        initial = float(lib._script.initial_capital)
        if self.max_equity == -float("inf"):
            return initial
        return max(initial, float(self.max_equity))

    def _trigger_risk_halt(self, reason: str, price: float, h: float, l: float) -> None:
        """Cancel pending orders, close any open position at ``price``, halt trading.

        ``reason`` is embedded in the synthetic close order's comment so the
        backtest log identifies which ``strategy.risk.*`` rule fired. Once
        :attr:`risk_halt_trading` is set, ``strategy.entry`` / ``strategy.order``
        early-return, ``process_orders`` short-circuits, and the strategy stays
        flat until the script completes.
        """
        self.entry_orders.clear()
        self.exit_orders.clear()
        self.orderbook.clear()
        if self.size != 0.0:
            close_order = Order(
                None, -self.size,
                exit_id='Risk management close',
                order_type=_order_type_close,
                comment=f"Close Position ({reason})",
            )
            self._fill_order(close_order, price, h, l)
        self.risk_halt_trading = True

    def _close_position_at_intraday_cap(self, order: Order, price: float) -> None:
        """Flatten the position when ``max_intraday_filled_orders`` is reached.

        TradingView closes the open position the moment the daily filled-orders
        cap is hit, tagging the exit ``Close Position (Max number of filled
        orders in one day)``. Unlike :meth:`_trigger_risk_halt` this does NOT
        set :attr:`risk_halt_trading`: the cap is a per-day limit, and the
        intraday counter (already at the cap) blocks any further entry fills
        until it resets at the next day rollover, so trading resumes by itself
        the following day. The forced close is not itself a strategy order, so
        it does not count toward the cap.

        The exit price mirrors TradingView's broker emulation. When the
        cap-triggering fill is a market/stop *entry* that fired intra-bar — past
        the bar open on the favorable side (a long stop above the open, a short
        stop below it) — TV traces the bar path to that extreme and closes there
        (bar high for a long, bar low for a short), not at the entry trigger
        price. Fills that landed at the open (gaps, plain market entries) and
        non-entry fills close at the triggering fill price.
        """
        self.entry_orders.clear()
        self.exit_orders.clear()
        self.orderbook.clear()
        if self.size != 0.0:
            # ``self.h`` / ``self.l`` are the full current-bar extremes; the ``h`` / ``l``
            # arguments threaded through :meth:`fill_order` are truncated to the stop
            # trigger as the intra-bar path is walked, so they cannot stand in for the
            # bar's reached extreme here.
            cap_close_price = price
            if order.order_type == _order_type_entry or order.order_type == _order_type_normal:
                if self.size > 0.0 and price > self.o:
                    cap_close_price = self.h
                elif self.size < 0.0 and price < self.o:
                    cap_close_price = self.l
            close_order = Order(
                None, -self.size,
                exit_id='Risk management close',
                order_type=_order_type_close,
                comment="Close Position (Max number of filled orders in one day)",
            )
            self._fill_order(close_order, cap_close_price, self.h, self.l, counts_as_filled_order=False)

    def _enforce_post_bar_risk(self) -> None:
        """Run the post-bar ``strategy.risk.*`` checks that depend on bar-end P&L.

        ``max_intraday_filled_orders`` is enforced inline in :meth:`fill_order`
        because it is fill-count driven; the rules below need the finalised
        bar P&L (``max_drawdown``) or daily realised equity
        (``max_intraday_loss``, ``max_cons_loss_days``) and therefore run after
        :meth:`_finalize_bar_pnl`. The first triggered rule wins — subsequent
        checks are skipped, since a halt closes all positions and clears
        pending orders.
        """
        if self.risk_halt_trading:
            return
        # Use the bar-close price for the synthetic close — the bar is over.
        price, h, l = self.c, self.h, self.l
        if self._is_max_drawdown_breached():
            self._trigger_risk_halt("Max drawdown reached", price, h, l)
            return
        if self._is_max_intraday_loss_breached():
            self._trigger_risk_halt("Max intraday loss reached", price, h, l)
            return
        if self._is_max_cons_loss_days_breached():
            self._trigger_risk_halt("Max consecutive loss days reached", price, h, l)

    def _check_already_filled(self, order: Order) -> bool:
        """
        Check if a stop or limit order would be immediately fillable due to a gap.
        This is called during process_orders when we have the current bar's OHLC values.

        When there's a gap, orders that would normally wait for price movement
        should execute immediately at the open price.

        :param order: The order to check
        :return: True if the order should be filled immediately at open price
        """
        # if not self.open_trades:
        #     return False

        # Check stop orders with gaps
        if order.stop is not None:
            # Long stop order (size > 0): triggers if open gaps above stop level
            if order.size > 0 and self.o >= order.stop:
                return True
            # Short stop order (size < 0): triggers if open gaps below stop level
            if order.size < 0 and self.o <= order.stop:
                return True

        # Check limit orders with gaps
        if order.limit is not None:
            # Long limit order (size > 0): triggers if open gaps below limit level
            if order.size > 0 and self.o <= order.limit:
                return True
            # Short limit order (size < 0): triggers if open gaps above limit level
            if order.size < 0 and self.o >= order.limit:
                return True

        return False

    def _check_high_stop(self, order: Order) -> bool:
        """ Check high stop and trailing trigger """
        if order.stop is None:
            return False
        # Stop order (size > 0) triggers when price rises to stop level
        if order.size > 0 and order.stop <= self.h:
            p = max(order.stop, self.o)
            slippage = lib._script.slippage
            if slippage > 0:
                p += syminfo.mintick * slippage
            order.filled_by_type = 'loss'
            self.fill_order(order, p, p, self.l)
            return True
        return False

    def _check_high(self, order: Order) -> bool:
        """ Check high limit """
        if order.limit is not None:
            # Short limit order (size < 0) triggers when price rises to limit level
            if order.size < 0 and order.limit <= self.h:
                p = max(order.limit, self.o)
                order.filled_by_type = 'profit'
                self.fill_order(order, p, p, self.l)
                return True
        return False

    def _process_trailing_stop(self, order: Order, ohlc: bool) -> bool:
        """Process a trailing-stop exit for the current bar (TradingView model).

        The activation level is ``order.trail_price`` (``entry ± trail_points``).
        Once the bar's extreme reaches it, the stop anchors at the activation
        level offset by ``trail_offset`` and fills on the SAME bar if the bar
        retraces to it. The current bar's extreme only advances the trail for the
        NEXT bar — TradingView evaluates a trailing stop from the prior
        high/low-water mark, so with ``trail_offset == 0`` the fill lands exactly
        at the activation level on the activation bar instead of riding to the
        bar's extreme.

        A trail that activates on THIS bar arms at the favorable extreme (the high
        for a long, the low for a short). When that extreme is the bar's SECOND
        intra-bar leg, a hard ``stop=`` reached on the FIRST leg fills earlier in
        intra-bar time and must win, so the trail defers to the price walk in that
        case instead of pre-empting it. A trail carried from a prior bar is live
        from the open and keeps priority. When such a bar opens at its own low
        (long) or high (short) -- no wick on the trailing side -- and that open
        gaps past the prior water mark, the open is itself the new water mark and
        the bar never trades back to it, so the fill lands at the open.
        ``ohlc`` selects the leg order:
        ``True`` => open->high->low, ``False`` => open->low->high.

        :param order: The exit order carrying ``trail_price``.
        :param ohlc: The bar's intra-bar leg order (see :meth:`process_orders`).
        :return: True if the order filled this bar.
        """
        if order.trail_price is None:
            return False
        round_to_mintick = lib.math.round_to_mintick
        offset_price = syminfo.mintick * order.trail_offset
        slippage = lib._script.slippage

        if order.sign < 0:
            # Long position: trailing sell-stop riding under the high-water mark.
            just_activated = False
            if not order.trail_triggered:
                if self.h < order.trail_price:
                    return False
                order.trail_triggered = True
                order.trail_stop = round_to_mintick(order.trail_price - offset_price)
                just_activated = True
            stop = order.trail_stop
            if stop is None:
                return False
            # A carried trail on a bar that opens at its low (no lower wick) and
            # gaps above the prior water mark fills at the open: TradingView folds
            # the new bar's open into the high-water mark, so the offset-0 stop
            # sits at the open and the bar -- which never trades lower -- touches
            # it there.
            if not just_activated and self.l == self.o:
                open_stop = round_to_mintick(self.o - offset_price)
                if open_stop > stop:
                    p = open_stop
                    if slippage > 0:
                        p -= syminfo.mintick * slippage
                    order.filled_by_type = 'trailing'
                    self.fill_order(order, p, self.h, p)
                    return True
            # Activated on the high (second leg, open->low->high): a hard stop hit
            # on the first (low) leg fills before the trail arms, so defer to it.
            stop_first = (just_activated and not ohlc
                          and order.stop is not None and order.stop >= self.l)
            if self.l <= stop and not stop_first:
                # A stop carried from a prior bar fills at the open when the bar
                # gaps below it; on the activation bar the stop was just placed
                # intra-bar at the activation level, so the fill is at the stop.
                p = self.o if (not just_activated and self.o <= stop) else stop
                if slippage > 0:
                    p -= syminfo.mintick * slippage
                order.filled_by_type = 'trailing'
                self.fill_order(order, p, self.h, p)
                return True
            # No fill: ratchet the trail up with this bar's high for the next bar.
            new_stop = round_to_mintick(self.h - offset_price)
            if new_stop > stop:
                order.trail_stop = new_stop
            return False

        if order.sign > 0:
            # Short position: trailing buy-stop riding above the low-water mark.
            just_activated = False
            if not order.trail_triggered:
                if self.l > order.trail_price:
                    return False
                order.trail_triggered = True
                order.trail_stop = round_to_mintick(order.trail_price + offset_price)
                just_activated = True
            stop = order.trail_stop
            if stop is None:
                return False
            # A carried trail on a bar that opens at its high (no upper wick) and
            # gaps below the prior water mark fills at the open: TradingView folds
            # the new bar's open into the low-water mark, so the offset-0 stop sits
            # at the open and the bar -- which never trades higher -- touches it
            # there.
            if not just_activated and self.h == self.o:
                open_stop = round_to_mintick(self.o + offset_price)
                if open_stop < stop:
                    p = open_stop
                    if slippage > 0:
                        p += syminfo.mintick * slippage
                    order.filled_by_type = 'trailing'
                    self.fill_order(order, p, p, self.l)
                    return True
            # Activated on the low (second leg, open->high->low): a hard stop hit
            # on the first (high) leg fills before the trail arms, so defer to it.
            stop_first = (just_activated and ohlc
                          and order.stop is not None and order.stop <= self.h)
            if self.h >= stop and not stop_first:
                p = self.o if (not just_activated and self.o >= stop) else stop
                if slippage > 0:
                    p += syminfo.mintick * slippage
                order.filled_by_type = 'trailing'
                self.fill_order(order, p, p, self.l)
                return True
            # No fill: ratchet the trail down with this bar's low for the next bar.
            new_stop = round_to_mintick(self.l + offset_price)
            if new_stop < stop:
                order.trail_stop = new_stop
            return False

        return False

    def _seed_trail_at_issue(self, order: Order) -> None:
        """Fold the issue bar's extreme into a trailing exit's high/low-water mark.

        ``process_orders`` runs before the script body, so an exit issued in the
        script on bar N -- e.g. one gated on ``strategy.position_size``, which is
        only known once the entry has filled -- is first evaluated on bar N+1.
        The entry-fill bar's own extreme would then never seed the trail, leaving
        PyneCore's water mark one bar behind TradingView's, which keeps the
        trailing stop alive from the bar the position is already open. Advance the
        water mark here at issue time (activation + ratchet only -- the fill still
        happens in the next ``process_orders``).

        Exits placed on the entry SIGNAL bar (entry still pending, so no bound
        trade is open yet) are skipped: ``process_orders`` seeds those on their
        fill bar exactly as before, so the single-issue path is unchanged.

        :param order: The freshly (re-)issued trailing exit order.
        """
        if order.trail_points_ticks is None and order.trail_price is None:
            return
        entry_price: float | None = None
        for trade in self.open_trades:
            if trade.entry_id == order.order_id:
                entry_price = trade.entry_price
                break
        if entry_price is None:
            return  # entry still pending -- seeded later on the fill bar

        direction = 1.0 if order.size < 0 else -1.0
        trail_price = order.trail_price
        if trail_price is None and order.trail_points_ticks is not None:
            trail_price = _price_round(
                entry_price + direction * syminfo.mintick * order.trail_points_ticks, direction)
        if trail_price is None:
            return

        round_to_mintick = lib.math.round_to_mintick
        offset_price = syminfo.mintick * order.trail_offset
        # Arming on the issue (entry-fill) bar is gated on the bar CLOSE, not its
        # intrabar extreme: TradingView only carries a trailing stop out of the
        # entry-fill bar when that bar closes past the activation level. A bar
        # whose extreme pierces the activation level but closes back inside it does
        # NOT arm here -- it arms later, intrabar, in the normal price walk (which
        # also performs the same-bar fill that is suppressed on the entry-fill
        # bar). On every later bar a close past the level implies the high already
        # pierced it, so process_orders has already armed the carried order and
        # this gate never fires there.
        if order.sign < 0:
            # Long position: trailing sell-stop riding under the high-water mark.
            if not order.trail_triggered:
                if self.c <= trail_price:
                    return
                order.trail_triggered = True
                order.trail_stop = round_to_mintick(trail_price - offset_price)
            new_stop = round_to_mintick(self.h - offset_price)
            if order.trail_stop is None or new_stop > order.trail_stop:
                order.trail_stop = new_stop
        elif order.sign > 0:
            # Short position: trailing buy-stop riding above the low-water mark.
            if not order.trail_triggered:
                if self.c >= trail_price:
                    return
                order.trail_triggered = True
                order.trail_stop = round_to_mintick(trail_price + offset_price)
            new_stop = round_to_mintick(self.l + offset_price)
            if order.trail_stop is None or new_stop < order.trail_stop:
                order.trail_stop = new_stop

    def _check_margin_call(self, check_price: float, *, for_short: bool,
                           at_open: bool = False,
                           can_defer: bool = True) -> bool:
        """
        Check and execute margin call using TradingView's 10-step algorithm.

        TradingView's 3-branch margin call logic:
        1. AF@O < 0: fire immediately at open price (at_open=True)
        2. mc_size > 1: fire immediately at worst-case price (H for shorts, L for longs)
        3. mc_size == 1 AND can_defer AND AF@C < 0: defer MC to post-script at close price
        4. mc_size == 1 AND (not can_defer OR AF@C >= 0): fire immediately at worst-case

        Deferral is only allowed at the first OHLC extremum (where recovery is still
        possible at the opposite extremum). At the second extremum only close remains,
        so TV fires immediately.

        :param check_price: The price to check margin at
        :param for_short: If True, check short positions. If False, check long positions.
        :param at_open: If True, this is an open check — always fire immediately, never defer.
        :param can_defer: If False, MC fires immediately even when mc_size==1 and AF@C<0.
        :return: True if MC was deferred (caller should stop OHLC processing)
        """
        if not self.open_trades:
            return False

        if for_short and self.sign >= 0:
            return False
        if not for_short and self.sign <= 0:
            return False

        script = lib._script
        margin_percent = script.margin_short if for_short else script.margin_long

        if margin_percent <= 0:
            return False

        quantity = abs(self.size)
        # Convert price * quantity to account-currency for margin/equity comparisons.
        pv = syminfo.pointvalue

        money_spent = quantity * self.avg_price * pv
        mvs = quantity * check_price * pv

        open_profit = mvs - money_spent
        if self.sign < 0:
            open_profit = -open_profit

        equity = script.initial_capital + self.netprofit + open_profit
        margin_ratio = margin_percent / 100.0
        margin = mvs * margin_ratio
        available_funds = equity - margin

        if available_funds >= 0:
            return False

        loss = available_funds / margin_ratio
        # One contract is worth `check_price * pv` in account currency.
        cover_amount = int(loss / (check_price * pv))
        margin_call_size = max(1, abs(cover_amount) * 4)

        if margin_call_size > quantity:
            margin_call_size = quantity

        # Deferral check: mc_size==1 at first OHLC extremum, check if AF@C<0
        # Skip deferral when check_price == close: no recovery possible at same price
        if not at_open and can_defer and margin_call_size == 1 and check_price != self.c:
            c_mvs = quantity * self.c * pv
            c_open_profit = c_mvs - money_spent
            if self.sign < 0:
                c_open_profit = -c_open_profit
            c_equity = script.initial_capital + self.netprofit + c_open_profit
            c_margin = c_mvs * margin_ratio
            c_af = c_equity - c_margin
            if c_af < 0:
                self._deferred_margin_call = (self.c, for_short)
                return True

        fill_price = check_price
        if script.slippage > 0:
            slippage_amount = syminfo.mintick * script.slippage
            if for_short:
                fill_price = check_price + slippage_amount
            else:
                fill_price = check_price - slippage_amount

        margin_call_order = Order(
            None,
            -self.sign * margin_call_size,
            order_type=_order_type_close,
            comment='Margin call'
        )
        margin_call_order.is_market_order = False
        margin_call_order.bar_index = int(lib.bar_index)

        self._fill_order(margin_call_order, fill_price, fill_price, fill_price)
        return False

    def process_deferred_margin_call(self):
        """
        Execute a deferred margin call (after the user script has run).
        Called from script_runner after the user script's main() completes.
        """
        if self._deferred_margin_call is None:
            return

        check_price, for_short = self._deferred_margin_call
        self._deferred_margin_call = None

        prev_count = len(self.new_closed_trades)
        self._check_margin_call(check_price, for_short=for_short, at_open=True)

        initial_capital = lib._script.initial_capital
        for closed_trade in self.new_closed_trades[prev_count:]:
            self.cum_profit += closed_trade.profit
            closed_trade.cum_profit = self.cum_profit
            try:
                closed_trade.cum_profit_percent = (
                                                          closed_trade.cum_profit / initial_capital) * 100.0
            except ZeroDivisionError:
                closed_trade.cum_profit_percent = 0.0
            self.entry_equity += closed_trade.profit

    def _check_low_stop(self, order: Order) -> bool:
        """ Check low stop """
        if order.stop is None:
            return False
        # Stop order (size < 0) triggers when price falls to stop level
        if order.size < 0 and order.stop >= self.l:
            p = min(self.o, order.stop)
            slippage = lib._script.slippage
            if slippage > 0:
                p -= syminfo.mintick * slippage
            order.filled_by_type = 'loss'
            self.fill_order(order, p, self.h, p)
            return True
        return False

    def _check_low(self, order: Order) -> bool:
        """ Check low limit """
        if order.limit is not None:
            # Long limit order (size > 0) triggers when price falls to limit level
            if order.size > 0 and order.limit >= self.l:
                p = min(self.o, order.limit)
                order.filled_by_type = 'profit'
                self.fill_order(order, p, self.h, p)
                return True
        return False

    def process_orders(self):
        """ Process orders """
        # We need to round to the nearest tick to get the same results as in TradingView
        round_to_mintick = lib.math.round_to_mintick
        self.o = round_to_mintick(lib.open)
        self.h = round_to_mintick(lib.high)
        self.l = round_to_mintick(lib.low)
        self.c = round_to_mintick(lib.close)

        # If the order is open → high → low → close or open → low → high → close
        ohlc = self.h - self.o < self.o - self.l

        self.drawdown_summ = self.runup_summ = 0.0
        self.new_closed_trades.clear()

        self._process_at_bar_open(ohlc)
        self._process_limit_stop_orders(ohlc)
        self._finalize_bar_pnl()
        self._enforce_post_bar_risk()
        self._finalize_new_closed_trades()

    def _process_at_bar_open(self, ohlc: bool):
        """Phase 1: Process orders at bar open — gap detection, market fills, margin."""
        # Check if we're in a new trading day for intraday risk management.
        # ``time_tradingday`` is session-aware: for overnight sessions (forex,
        # futures) the day rolls at the session open (e.g. 17:00 ET), not at
        # calendar midnight — matching TradingView's intraday risk reset. For
        # 24/7 crypto and intraday stock sessions it collapses to the calendar
        # day in the exchange timezone, so those symbols are unaffected.
        current_trading_day = int(lib.time_tradingday())
        if current_trading_day != self.risk_last_trading_day:
            current_equity = float(self.equity)
            # Roll over consecutive-loss-day count for ``strategy.risk.max_cons_loss_days``.
            # On the very first bar we have no prior day to compare against — initialise
            # the trailing-equity anchor without touching the loss-day counter.
            if self.risk_last_trading_day != -1:
                if current_equity < self.risk_last_day_equity:
                    self.risk_cons_loss_days += 1
                else:
                    self.risk_cons_loss_days = 0
            self.risk_last_day_equity = current_equity
            # Anchor for ``strategy.risk.max_intraday_loss`` — captured at the
            # start of every trading day, not just the first one.
            self.risk_intraday_start_equity = current_equity
            self.risk_last_trading_day = current_trading_day
            self.risk_intraday_filled_orders = 0
            # ``max_cons_loss_days`` becomes known the moment the day rolls
            # over — halt now rather than at bar end so the new day's queued
            # entries cannot fill at this bar's open.
            if self._is_max_cons_loss_days_breached() and not self.risk_halt_trading:
                self._trigger_risk_halt(
                    "Max consecutive loss days reached", self.o, self.h, self.l,
                )
                return

        # Get script reference for slippage
        script = lib._script

        # Skip market exit order processing if there's no open position (TradingView behavior)
        if not self.open_trades:
            # Remove orphan exit orders when position is flat. An exit is orphan
            # when its ``order_id`` (the ``from_entry`` it was bound to) no longer
            # has a pending entry — the entry was cancelled, margin-rejected, or
            # never existed. Pending entries (limit/stop/market) keep their exits
            # alive so the stop/limit fires once the entry fills.
            for order in list(self.exit_orders.values()):
                if not order.is_market_order:
                    if order.order_id in self.entry_orders:
                        continue
                    if order.from_entry_na:
                        continue
                    self._remove_order(order)

        # For exit orders, calculate limit/stop from entry price if ticks are specified
        for order in self.exit_orders.values():
            # Try to find the trade with matching entry_id
            entry_price: float | None = None
            for trade in self.open_trades:
                if trade.entry_id == order.order_id:
                    entry_price = trade.entry_price
                    break

            # If we found the entry price and have tick values, calculate the actual prices
            if entry_price is not None:
                # Determine direction from the order
                direction = 1.0 if order.size < 0 else -1.0  # Exit order size is negative of position
                changed = False

                # Calculate limit from profit_ticks if specified
                if order.profit_ticks is not None and order.limit is None:
                    order.limit = entry_price + direction * syminfo.mintick * order.profit_ticks
                    order.limit = _price_round(order.limit, direction)
                    changed = True

                # Calculate stop from loss_ticks if specified
                if order.loss_ticks is not None and order.stop is None:
                    order.stop = entry_price - direction * syminfo.mintick * order.loss_ticks
                    order.stop = _price_round(order.stop, -direction)
                    changed = True

                # Calculate trail_price from trail_points_ticks if specified
                if order.trail_points_ticks is not None and order.trail_price is None:
                    order.trail_price = entry_price + direction * syminfo.mintick * order.trail_points_ticks
                    order.trail_price = _price_round(order.trail_price, direction)
                    changed = True

                # Update orderbook only when prices were actually calculated
                if changed:
                    self.orderbook.add_order(order)

        # Check for stop/limit orders that should be converted to market orders
        for order in self.orderbook.iter_orders():
            # Check if the order would be filled immediately (e.g. due to a gap)
            if self._check_already_filled(order):
                if order.exit_id is not None:
                    # Exit order gaps through — check if it's for an open position
                    has_open_trade = any(
                        t.entry_id == order.order_id for t in self.open_trades
                    )
                    if not has_open_trade:
                        associated_entry = self.entry_orders.get(order.order_id)
                        if associated_entry is not None:
                            # Pending entry exists — defer exit, will fill after entry
                            continue
                        # Keep from_entry_na exits — they persist until filled or replaced
                        if order.from_entry_na:
                            continue
                        self._remove_order(order)
                        continue

                # Convert to market order
                order.is_market_order = True
                # Add to market orders dict
                self.market_orders[(order.order_type, order.order_id, order.exit_id)] = order

        # Process Market orders
        for order in list(self.market_orders.values()):
            if order.order_type == _order_type_entry:
                if order.limit is None and order.stop is None:
                    # We need to check pyramiding and flip quantity here for market orders :-/
                    # Check pyramiding limit for entry orders adding to existing position
                    if self.sign == order.sign:
                        if lib._script.pyramiding <= len(self.open_trades):
                            # Pyramiding limit reached - don't add the order
                            self._remove_order(order)
                            continue
                    elif self.size != 0.0:
                        # TradingView calculates the flip quantity 1st order processing
                        # then open a new one in the opposite direction.
                        order.size -= self.size  # Subtract because position.size has opposite sign

            # Apply slippage to market orders
            fill_price = self.o
            if script.slippage > 0:
                # Slippage is in ticks, always adverse to trade direction
                # For long orders (buying), slippage increases the price
                # For short orders (selling), slippage decreases the price
                slippage_amount = syminfo.mintick * script.slippage * order.sign
                fill_price = self.o + slippage_amount

            # Pre-fill margin check for entry orders (TradingView behavior)
            # TV rejects entry orders BEFORE filling if the position would exceed margin
            if order.order_type == _order_type_entry:
                margin_percent = (script.margin_short if order.sign < 0
                                  else script.margin_long)
                if margin_percent > 0:
                    margin_ratio = margin_percent / 100.0
                    # Margin/equity live in account currency; convert price * qty into
                    # account-currency units via the futures pointvalue.
                    pv = syminfo.pointvalue
                    if self.size == 0.0:
                        equity = script.initial_capital + self.netprofit
                        margin_needed = abs(order.size) * fill_price * pv * margin_ratio
                        if margin_needed > equity:
                            self._remove_order(order)
                            continue
                    elif self.sign == order.sign:
                        new_qty = abs(self.size) + abs(order.size)
                        money_spent = (abs(self.size) * self.avg_price
                                       + abs(order.size) * fill_price) * pv
                        mvs = new_qty * fill_price * pv
                        open_profit = ((mvs - money_spent) if self.sign > 0
                                       else (money_spent - mvs))
                        equity = script.initial_capital + self.netprofit + open_profit
                        margin_needed = mvs * margin_ratio
                        if margin_needed > equity:
                            self._remove_order(order)
                            continue

            # open → high → low → close
            if ohlc:
                self.fill_order(order, fill_price, self.o, self.l)
            # open → low → high → close
            else:
                self.fill_order(order, fill_price, self.l, self.o)

        # Convert tick-based exit prices for entries that just filled this bar
        for order in self.exit_orders.values():
            entry_price = None
            for trade in self.open_trades:
                if trade.entry_id == order.order_id:
                    entry_price = trade.entry_price
                    break
            if entry_price is not None:
                direction = 1.0 if order.size < 0 else -1.0
                changed = False
                if order.profit_ticks is not None and order.limit is None:
                    order.limit = entry_price + direction * syminfo.mintick * order.profit_ticks
                    order.limit = _price_round(order.limit, direction)
                    changed = True
                if order.loss_ticks is not None and order.stop is None:
                    order.stop = entry_price - direction * syminfo.mintick * order.loss_ticks
                    order.stop = _price_round(order.stop, -direction)
                    changed = True
                if order.trail_points_ticks is not None and order.trail_price is None:
                    order.trail_price = entry_price + direction * syminfo.mintick * order.trail_points_ticks
                    order.trail_price = _price_round(order.trail_price, direction)
                    changed = True
                if changed:
                    self.orderbook.add_order(order)

        # Adapt orphaned exits from rejected entries to new position (TradingView behavior)
        # When strategy.exit() is called without from_entry, TV keeps the exit even after
        # its entry is rejected by margin. The exit adapts to close any new position that opens.
        if self.open_trades:
            for order in list(self.exit_orders.values()):
                if order.is_market_order:
                    continue
                # Skip exits that match an open trade (they belong to the current position)
                if any(t.entry_id == order.order_id for t in self.open_trades):
                    continue
                # Skip exits whose entry is still pending
                if order.order_id in self.entry_orders:
                    continue
                new_sign = -self.sign
                self._remove_order(order)
                adapted = Order(
                    None, -self.size, exit_id=order.exit_id,
                    order_type=_order_type_close,
                    limit=order.limit, stop=order.stop,
                    comment=order.comment,
                    comment_profit=order.comment_profit,
                    comment_loss=order.comment_loss,
                    comment_trailing=order.comment_trailing,
                    alert_message=order.alert_message,
                    alert_profit=order.alert_profit,
                    alert_loss=order.alert_loss,
                    alert_trailing=order.alert_trailing,
                )
                adapted.bar_index = order.bar_index
                # Check gap-through with the flipped direction
                stop_gap = (adapted.stop is not None
                            and ((new_sign > 0 and self.o >= adapted.stop)
                                 or (new_sign < 0 and self.o <= adapted.stop)))
                limit_gap = (adapted.limit is not None
                             and ((new_sign > 0 and self.o <= adapted.limit)
                                  or (new_sign < 0 and self.o >= adapted.limit)))
                filled = False
                if stop_gap:
                    fill_price = self.o
                    if script.slippage > 0:
                        fill_price += syminfo.mintick * script.slippage * new_sign
                    adapted.filled_by_type = 'loss'
                    if ohlc:
                        self.fill_order(adapted, fill_price, fill_price, self.l)
                    else:
                        self.fill_order(adapted, fill_price, self.l, fill_price)
                    filled = True
                elif limit_gap:
                    adapted.filled_by_type = 'profit'
                    if ohlc:
                        self.fill_order(adapted, self.o, self.o, self.l)
                    else:
                        self.fill_order(adapted, self.o, self.l, self.o)
                    filled = True
                else:
                    self._add_order(adapted)
                # If the adapted exit closed the position, clean up remaining orphan exits
                if filled and not self.open_trades:
                    for remaining in list(self.exit_orders.values()):
                        if not remaining.is_market_order:
                            has_entry = remaining.order_id in self.entry_orders
                            if not has_entry:
                                self._remove_order(remaining)
                    break

        # Fill gap-through exits whose entries just filled
        for order in list(self.exit_orders.values()):
            if order.is_market_order:
                continue
            has_open_trade = any(
                t.entry_id == order.order_id for t in self.open_trades
            )
            if not has_open_trade:
                continue
            # Check limit gap-through
            if order.limit is not None:
                limit_gap = ((order.size > 0 and self.o <= order.limit)
                             or (order.size < 0 and self.o >= order.limit))
                if limit_gap:
                    order.filled_by_type = 'profit'
                    if ohlc:
                        self.fill_order(order, self.o, self.o, self.l)
                    else:
                        self.fill_order(order, self.o, self.l, self.o)
                    continue
            # Check stop gap-through
            if order.stop is not None:
                stop_gap = ((order.size > 0 and self.o >= order.stop)
                            or (order.size < 0 and self.o <= order.stop))
                if stop_gap:
                    fill_price = self.o
                    if script.slippage > 0:
                        fill_price += syminfo.mintick * script.slippage * order.sign
                    order.filled_by_type = 'loss'
                    if ohlc:
                        self.fill_order(order, fill_price, fill_price, self.l)
                    else:
                        self.fill_order(order, fill_price, self.l, fill_price)
                    continue

        # Margin call check at OPEN
        self._check_margin_call(self.o, for_short=True, at_open=True)
        self._check_margin_call(self.o, for_short=False, at_open=True)

    def _process_limit_stop_orders(self, ohlc: bool):
        """Phase 2: Process limit/stop/trailing orders with margin checks at H/L."""
        # Trailing stops are evaluated from the prior high/low-water mark, so they
        # are processed once per bar here rather than inside the price walk — the
        # walk would ride the stop to the current bar's extreme and fill there.
        # Iterate a snapshot since fills mutate the order book; an order indexed at
        # several price levels is yielded once per level, so dedupe by identity.
        seen: set[Order] = set()
        for order in list(self.orderbook.iter_orders()):
            if order in seen or order.cancelled or order.trail_price is None:
                continue
            seen.add(order)
            self._process_trailing_stop(order, ohlc)

        # Process orders: open → high → low → close
        if ohlc:
            # open -> high
            for order in self.orderbook.iter_orders(min_price=self.o, max_price=self.h):
                if self._check_high_stop(order):
                    continue
                if self._check_high(order):
                    continue

            if not self._check_margin_call(self.h, for_short=True):
                # open -> low (descending: the level nearest the open fills first)
                for order in self.orderbook.iter_orders(max_price=self.o, min_price=self.l, desc=True):
                    if self._check_low_stop(order):
                        continue
                    if self._check_low(order):
                        continue

                self._check_margin_call(self.l, for_short=False, can_defer=False)

        # Process orders: open → low → high → close
        else:
            # open -> low (descending: the level nearest the open fills first)
            for order in self.orderbook.iter_orders(max_price=self.o, min_price=self.l, desc=True):
                if self._check_low_stop(order):
                    continue
                if self._check_low(order):
                    continue

            if not self._check_margin_call(self.l, for_short=False):
                # open -> high
                for order in self.orderbook.iter_orders(min_price=self.o, max_price=self.h):
                    if self._check_high_stop(order):
                        continue
                    if self._check_high(order):
                        continue

                self._check_margin_call(self.h, for_short=True, can_defer=False)

    def _finalize_bar_pnl(self):
        """Phase 3: Calculate P&L, drawdown, runup, and cumulative stats."""
        # Calculate average entry price, unrealized P&L, drawdown and runup...
        if self.open_trades:
            # USD value per 1.0-point move per 1 contract — futures-aware PnL conversion factor
            pv = syminfo.pointvalue

            # Unrealized P&L
            self.openprofit = self.size * (self.c - self.avg_price) * pv

            # Calculate open drawdowns and runups
            for trade in self.open_trades:
                # Profit of trade
                trade.profit = trade.size * (self.c - trade.entry_price) * pv - 2 * trade.commission

                # P/L from high/low to calculate drawdown and runup
                hprofit = trade.size * (self.h - self.avg_price) * pv - trade.commission
                lprofit = trade.size * (self.l - self.avg_price) * pv - trade.commission
                # Drawdown
                drawdown = -min(hprofit, lprofit, 0.0)
                trade.max_drawdown = max(drawdown, trade.max_drawdown)
                # Runup
                runup = max(hprofit, lprofit, 0.0)
                trade.max_runup = max(runup, trade.max_runup)

                # Calculate percentage values for drawdown and runup — both in USD
                trade_value = abs(trade.size) * trade.entry_price * pv
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

    def _finalize_new_closed_trades(self) -> None:
        """Apply cumulative stats to every trade closed on this bar.

        Split out from :meth:`_finalize_bar_pnl` so it runs **after**
        :meth:`_enforce_post_bar_risk` — otherwise a synthetic close
        emitted by a risk-rule halt would be appended to
        ``new_closed_trades`` after this loop has finished, ship out with
        default ``cum_profit`` / ``cum_max_drawdown`` / ``cum_max_runup``
        / ``cum_profit_percent`` values, and never be revisited.
        """
        if not self.new_closed_trades:
            return
        initial_capital = lib._script.initial_capital
        for closed_trade in self.new_closed_trades:
            # Incrementally add each trade's profit to cumulative total
            self.cum_profit += closed_trade.profit
            closed_trade.cum_profit = self.cum_profit
            closed_trade.cum_max_drawdown = self.max_drawdown
            closed_trade.cum_max_runup = self.max_runup

            # Cumulative profit percent
            try:
                closed_trade.cum_profit_percent = (closed_trade.cum_profit / initial_capital) * 100.0
            except ZeroDivisionError:
                closed_trade.cum_profit_percent = 0.0

            # Modify entry equity, for max drawdown and runup
            self.entry_equity += closed_trade.profit

    def process_orders_at_close(self):
        """
        Optional post-script pass that fills current-bar-submitted orders at the bar's
        CLOSE — enabled by `script.process_orders_on_close=True`.

        Pine semantics: when the flag is set, orders placed during the strategy's bar
        calculation get an additional fill attempt at the bar close, instead of waiting
        for the next bar's open. This covers BOTH:
          - Market orders: trivially executable at close.
          - Limit/stop orders: executable when the close has reached/crossed the trigger
            price. (Non-current-bar limit/stop orders already had their fair shake in
            `_process_limit_stop_orders` during the H/L walk.)
        Tick-based exit orders submitted on the current bar (`strategy.exit(profit=...,
        loss=...)`) only carry `profit_ticks` / `loss_ticks` until the next bar's
        `_process_at_bar_open` resolves them against the entry price. The close-pass
        materializes those into `limit` / `stop` first so the trigger check sees them.

        Fill price in every case is `self.c` (Pine fills price-based orders "when their
        limit or stop price is hit on the close" — no trigger-price snap on the close
        pass). Slippage matches the rest of the engine: applied to market and
        stop-triggered fills, NOT to limit-triggered fills (Pine guarantees limit
        orders fill at the limit price or better). `filled_by_type` is set on the
        triggering order so `_fill_order` can attach the right exit comment.

        Bookkeeping note: `_finalize_bar_pnl()` already ran in `process_orders()` for the
        same bar. Re-running it here would double-count `cum_profit` / `entry_equity` for
        already-settled `new_closed_trades` and dupe the `drawdown_summ` / `runup_summ`
        contribution of open trades. Instead, we only settle cumulative stats for trades
        that close DURING this pass (`_settle_close_pass_trades`). For positions opened
        right at the close, the bar has no remaining H/L range — their per-trade
        `profit` / `max_drawdown_percent` are intentionally left for the next bar's
        `_finalize_bar_pnl()` to compute, when there will actually be a range to attribute.
        """
        script = lib._script
        current_bar = int(lib.bar_index)
        close = self.c

        # Collect current-bar candidates: market orders (trivially eligible) and
        # limit/stop orders whose trigger condition is already met by the close.
        # Each entry carries the trigger kind so slippage / `filled_by_type` mirror
        # the regular fill paths (`_check_high_stop` etc.).
        # Use id() as the dedup key — order objects may live in multiple dicts.
        candidates: list[tuple[Order, str]] = []
        seen: set[int] = set()

        def _materialize_tick_exit(order: Order) -> None:
            """Resolve profit_ticks/loss_ticks against the matching open trade.

            Mirrors `_process_at_bar_open`: exits submitted during this bar's main()
            still carry the raw tick offsets — the close-pass trigger check needs
            them as concrete limit/stop prices.
            """
            if order.profit_ticks is None and order.loss_ticks is None:
                return
            if order.limit is not None and order.stop is not None:
                return
            entry_price: float | None = None
            for trade in self.open_trades:
                if trade.entry_id == order.order_id:
                    entry_price = trade.entry_price
                    break
            if entry_price is None:
                return
            direction = 1.0 if order.size < 0 else -1.0
            changed = False
            if order.profit_ticks is not None and order.limit is None:
                order.limit = _price_round(
                    entry_price + direction * syminfo.mintick * order.profit_ticks,
                    direction,
                )
                changed = True
            if order.loss_ticks is not None and order.stop is None:
                order.stop = _price_round(
                    entry_price - direction * syminfo.mintick * order.loss_ticks,
                    -direction,
                )
                changed = True
            # If we just resolved the order's price levels, index it in the
            # orderbook (mirrors `_process_at_bar_open`). Without this, an order
            # that fails the close-pass trigger check would persist with
            # `limit`/`stop` set but absent from `PriceOrderBook`, so next bar's
            # H/L walk would never see it (next bar's tick conversion is skipped
            # because `limit`/`stop` are already non-None).
            if changed:
                self.orderbook.add_order(order)

        def _add_market(order: Order):
            oid = id(order)
            if oid in seen or order.cancelled or order.bar_index != current_bar:
                return
            seen.add(oid)
            candidates.append((order, 'market'))

        def _add_trigger(order: Order):
            oid = id(order)
            if oid in seen or order.cancelled or order.bar_index != current_bar:
                return
            if order.is_market_order:
                return
            if order.order_type == _order_type_close:
                _materialize_tick_exit(order)
            trigger: str | None = None
            if order.stop is not None:
                if order.sign > 0 and close >= order.stop:
                    trigger = 'stop'
                elif order.sign < 0 and close <= order.stop:
                    trigger = 'stop'
            if trigger is None and order.limit is not None:
                if order.sign > 0 and close <= order.limit:
                    trigger = 'limit'
                elif order.sign < 0 and close >= order.limit:
                    trigger = 'limit'
            if trigger is not None:
                seen.add(oid)
                candidates.append((order, trigger))

        for order in list(self.market_orders.values()):
            _add_market(order)
        for order in list(self.entry_orders.values()):
            _add_trigger(order)
        for order in list(self.exit_orders.values()):
            _add_trigger(order)

        # Bar is closed; no further H/L range can occur after the fill. Use close for both
        # so any close-pass exit attributes 0 extra drawdown/runup to itself this bar.
        h_after = close
        l_after = close

        closed_before = len(self.new_closed_trades)
        # Snapshot drawdown / runup accumulators: `_finalize_bar_pnl()` in
        # `process_orders()` already booked the open-trade contribution for the full
        # bar H/L. `_fill_order` would add the close-pass exit PnL to the same summs,
        # double-counting the bar for any position that was already open at bar start.
        # We restore the snapshot after the fill loop, before the close-pass settle.
        drawdown_summ_before = self.drawdown_summ
        runup_summ_before = self.runup_summ

        def _apply_fill(order: Order, trigger: str) -> None:
            """Run the per-candidate fill, mirroring `_process_at_bar_open`."""
            if order.cancelled:
                return
            if order.order_type == _order_type_entry:
                if order.limit is None and order.stop is None:
                    # Pyramiding and flip-quantity handling — mirror `_process_at_bar_open`.
                    if self.sign == order.sign:
                        if script.pyramiding <= len(self.open_trades):
                            self._remove_order(order)
                            return
                    elif self.size != 0.0:
                        order.size -= self.size

            # Slippage: market + stop fills get slipped against the order direction,
            # limit fills do not (Pine guarantees limit price or better — matches
            # `_check_high` / `_check_low`).
            fill_price = close
            if trigger != 'limit' and script.slippage > 0:
                fill_price = close + syminfo.mintick * script.slippage * order.sign

            # Pass trigger reason through to `_fill_order` so close-pass exits get the
            # same `exit_comment` as their intrabar counterparts.
            if trigger == 'stop':
                order.filled_by_type = 'loss'
            elif trigger == 'limit':
                order.filled_by_type = 'profit'

            if order.order_type == _order_type_entry:
                margin_percent = (script.margin_short if order.sign < 0
                                  else script.margin_long)
                if margin_percent > 0:
                    margin_ratio = margin_percent / 100.0
                    # Same pointvalue conversion as _process_at_bar_open — margin and
                    # equity live in account currency, not price units.
                    pv = syminfo.pointvalue
                    if self.size == 0.0:
                        equity = script.initial_capital + self.netprofit
                        margin_needed = abs(order.size) * fill_price * pv * margin_ratio
                        if margin_needed > equity:
                            self._remove_order(order)
                            return
                    elif self.sign == order.sign:
                        new_qty = abs(self.size) + abs(order.size)
                        money_spent = (abs(self.size) * self.avg_price
                                       + abs(order.size) * fill_price) * pv
                        mvs = new_qty * fill_price * pv
                        open_profit = ((mvs - money_spent) if self.sign > 0
                                       else (money_spent - mvs))
                        equity = script.initial_capital + self.netprofit + open_profit
                        margin_needed = mvs * margin_ratio
                        if margin_needed > equity:
                            self._remove_order(order)
                            return

            self.fill_order(order, fill_price, h_after, l_after)

        # Phase 1: fill the initial candidates (market entries, previously-open
        # tick exits, current-bar limit/stop orders already executable at close).
        for order, trigger in candidates:
            _apply_fill(order, trigger)

        # Phase 2: a current-bar entry may have just filled in Phase 1, opening a
        # trade whose `entry_price` lets us resolve a same-bar `strategy.exit(...,
        # profit=..., loss=...)` order whose ticks were unresolved before Phase 1.
        # Mirror `_process_at_bar_open` line 1467-1490 — re-scan exit_orders for
        # current-bar tick exits, materialize, and fill any newly executable.
        for order in list(self.exit_orders.values()):
            oid = id(order)
            if oid in seen or order.cancelled or order.bar_index != current_bar:
                continue
            if order.is_market_order:
                continue
            if order.profit_ticks is None and order.loss_ticks is None:
                continue
            _materialize_tick_exit(order)
            trigger2: str | None = None
            if order.stop is not None:
                if order.sign > 0 and close >= order.stop:
                    trigger2 = 'stop'
                elif order.sign < 0 and close <= order.stop:
                    trigger2 = 'stop'
            if trigger2 is None and order.limit is not None:
                if order.sign > 0 and close <= order.limit:
                    trigger2 = 'limit'
                elif order.sign < 0 and close >= order.limit:
                    trigger2 = 'limit'
            if trigger2 is not None:
                seen.add(oid)
                _apply_fill(order, trigger2)

        # Discard the close-pass `_fill_order` contributions to drawdown_summ / runup_summ:
        # the same bar's H/L range is already booked for these trades by the earlier
        # `_finalize_bar_pnl()` call. The drop-on-the-floor edge case is a brand-new
        # trade that opens AND closes within the same close pass — extremely unlikely
        # and its H/L would be 0 anyway since the bar has no remaining range.
        self.drawdown_summ = drawdown_summ_before
        self.runup_summ = runup_summ_before

        # Incrementally settle only the trades that closed during the close pass;
        # everything settled by `process_orders()` earlier in this bar stays untouched.
        if len(self.new_closed_trades) > closed_before:
            self._settle_close_pass_trades(closed_before)

    def _settle_close_pass_trades(self, closed_before: int):
        """
        Apply cumulative bookkeeping for trades that closed during `process_orders_at_close`.

        Mirrors the per-closed-trade cum_profit / entry_equity update tail of
        `_finalize_bar_pnl()`, but only for new_closed_trades appended after the close
        pass started — the earlier entries were already settled when `process_orders()`
        ran for this same bar. Position-level max_drawdown / max_runup is intentionally
        NOT re-rolled here: the bar's H/L drawdown_summ / runup_summ contribution was
        already booked by `_finalize_bar_pnl()` against the open trades (which include
        the trades that close here, since they were opened on this same bar), and the
        close-pass `_fill_order` additions to those summs were discarded above. Re-
        applying the snapshot would inflate `max_drawdown` whenever `entry_equity` had
        already advanced (e.g. a losing regular-pass close shrank `entry_equity`).
        """
        initial_capital = lib._script.initial_capital
        for closed_trade in self.new_closed_trades[closed_before:]:
            self.cum_profit += closed_trade.profit
            closed_trade.cum_profit = self.cum_profit
            closed_trade.cum_max_drawdown = self.max_drawdown
            closed_trade.cum_max_runup = self.max_runup
            try:
                closed_trade.cum_profit_percent = (closed_trade.cum_profit / initial_capital) * 100.0
            except ZeroDivisionError:
                closed_trade.cum_profit_percent = 0.0
            # Entry equity must roll AFTER the max_drawdown/runup snapshot above —
            # same ordering as `_finalize_bar_pnl()`.
            self.entry_equity += closed_trade.profit

    def process_orders_magnified(self, sub_bars: list[OHLCV], aggregated: OHLCV):
        """
        Process orders using bar magnifier — check fills against each sub-bar's OHLC.

        Phase 1 (at-open) runs once using first sub-bar.
        Phase 2 (limit/stop) runs on each sub-bar sequentially.
        Phase 3 (P&L) runs once using aggregated bar values.
        """
        round_to_mintick = lib.math.round_to_mintick
        # Setup from first sub-bar (= chart bar open)
        first = sub_bars[0]
        self.o = round_to_mintick(first.open)
        self.h = round_to_mintick(first.high)
        self.l = round_to_mintick(first.low)
        # Use aggregated close for margin deferral checks
        self.c = round_to_mintick(aggregated.close)
        self.drawdown_summ = self.runup_summ = 0.0
        self.new_closed_trades.clear()

        # Phase 1: at-open processing (gap detection, market orders, margin at open)
        ohlc = self.h - self.o < self.o - self.l
        self._process_at_bar_open(ohlc)

        # Phase 2: process limit/stop orders on each sub-bar
        for sub_bar in sub_bars:
            self.o = round_to_mintick(sub_bar.open)
            self.h = round_to_mintick(sub_bar.high)
            self.l = round_to_mintick(sub_bar.low)
            self.c = round_to_mintick(sub_bar.close)
            ohlc = self.h - self.o < self.o - self.l
            self._process_limit_stop_orders(ohlc)

        # Phase 3: P&L update using aggregated bar values
        self.h = round_to_mintick(aggregated.high)
        self.l = round_to_mintick(aggregated.low)
        self.c = round_to_mintick(aggregated.close)
        self._finalize_bar_pnl()
        self._enforce_post_bar_risk()
        self._finalize_new_closed_trades()


#
# Functions
#

# noinspection PyProtectedMember
def _size_round(qty: PyneFloat) -> PyneFloat:
    """
    Round a size down to the nearest tradable lot (``1 / _size_round_factor``).

    :param qty: The quantity to round
    :return: The rounded quantity
    """
    if isinstance(qty, NA):
        return na_float
    rfactor = syminfo._size_round_factor  # noqa
    # Floor to the lot step (1 / rfactor). The float64 product can land an exact
    # lot multiple a hair below the integer (e.g. 173.432 * 1e4 ->
    # 1734319.9999999998); snap values within a few ULPs of an integer up before
    # the floor so an exact multiple is not truncated a whole lot down.
    scaled = abs(qty) * rfactor
    nearest = round(scaled)
    lots = nearest if abs(scaled - nearest) <= scaled * 1e-12 + 1e-9 else int(scaled)
    sign = 1 if qty > 0 else -1
    return sign * lots / rfactor


def _margin_call_round(qty: float) -> float:
    """
    Ceil rounding for margin call liquidation (minimum 1 unit)

    :param qty: Quantity to round (can be negative for short)
    :return: Rounded quantity (minimum 1 in absolute value)
    """
    rfactor = syminfo._size_round_factor  # noqa
    qrf = math.ceil(abs(qty) * rfactor * 10.0) * 0.1
    sign = 1 if qty > 0 else -1
    return sign * max(1, int(qrf)) / rfactor


# noinspection PyShadowingNames
@overload
def _price_round(price: float, direction: int | float) -> float: ...


# noinspection PyShadowingNames
@overload
def _price_round(price: PyneFloat, direction: int | float) -> PyneFloat: ...


# noinspection PyShadowingNames
def _price_round(price: PyneFloat, direction: int | float) -> PyneFloat:
    """
    Round price to the nearest tick (floor if direction < 0, ceil otherwise)

    Uses `minmove / pricescale` (matches `lib.math.round_to_mintick`), so symbols
    with `minmove != 1` (e.g. QM1!: pricescale=1000, minmove=25, tick=0.025) snap
    to the actual tick grid instead of `1 / pricescale`.

    :param price: The price to round
    :param direction: The direction of the price
    :return: The rounded price
    """
    if isinstance(price, NA):
        return na_float
    pricescale = syminfo.pricescale
    minmove = syminfo.minmove
    tick_count = round(price * pricescale / minmove, 7)
    if direction < 0:
        return int(tick_count) * minmove / pricescale
    return math.ceil(tick_count) * minmove / pricescale


# noinspection PyShadowingBuiltins,PyProtectedMember
def cancel(id: str):
    """
    Cancels a pending or unfilled order with a specific identifier

    :param id: The identifier of the order to cancel
    """
    if lib._lib_semaphore or lib._strategy_suppressed:
        return

    position = lib._script.position
    position._remove_order_by_id(id)


# noinspection PyProtectedMember
def cancel_all():
    """
    Cancels all pending or unfilled orders
    """
    if lib._lib_semaphore or lib._strategy_suppressed:
        return
    lib._script.position._cancel_all_orders()


# noinspection PyProtectedMember,PyShadowingBuiltins,PyShadowingNames
def close(id: str, comment: PyneStr = na_str, qty: PyneFloat = na_float,
          qty_percent: PyneFloat = na_float, alert_message: PyneStr = na_str,
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
    if lib._lib_semaphore or lib._strategy_suppressed:
        return

    position = lib._script.position

    if not isinstance(qty, NA) and qty <= 0.0:
        return

    if position.size == 0.0:
        return

    if isinstance(qty, NA):
        if not isinstance(qty_percent, NA):
            size = _size_round(-position.size * (qty_percent * 0.01))
        else:
            size = -position.size
    else:
        size = _size_round(-position.sign * qty)

    if size == 0.0:
        return

    exit_id = f"Close entry(s) order {id}"
    order = Order(id, size, exit_id=exit_id, order_type=_order_type_close,
                  comment=None if isinstance(comment, NA) else comment,
                  alert_message=None if isinstance(alert_message, NA) else alert_message)

    # Add order to position (this will handle orderbook and exit_orders)
    position._add_order(order)
    # Same-tick fill is a backtest concept; in broker mode the order is already
    # enqueued by ``_add_order`` and the sync engine forwards it to the exchange.
    if immediately and isinstance(position, SimPosition):
        position.fill_order(order, position.c, position.h, position.l)


# noinspection PyProtectedMember,PyShadowingNames
def close_all(comment: PyneStr = na_str, alert_message: PyneStr = na_str, immediately: bool = False):
    """
    Creates an order to close an open position completely, regardless of the identifiers of the entry
    orders that opened or added to it.

    :param comment: Additional notes on the filled order
    :param alert_message: Custom text for the alert that fires when an order fills
    :param immediately: If true, the closing order executes on the same tick when the strategy places it
    """
    if lib._lib_semaphore or lib._strategy_suppressed:
        return

    position = lib._script.position
    if position.size == 0.0:
        return

    exit_id = 'Close position order'
    order = Order(None, -position.size, exit_id=exit_id, order_type=_order_type_close,
                  comment=comment, alert_message=alert_message)

    # Add order to position (this will handle orderbook and exit_orders)
    position._add_order(order)
    # Same-tick fill is a backtest concept; in broker mode the order is already
    # enqueued by ``_add_order`` and the sync engine forwards it to the exchange.
    if immediately and isinstance(position, SimPosition):
        position.fill_order(order, position.c, position.h, position.l)


# noinspection PyProtectedMember,PyShadowingNames,PyShadowingBuiltins,DuplicatedCode
def entry(id: str, direction: direction.Direction, qty: int | PyneFloat = na_float,
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
    if lib._lib_semaphore or lib._strategy_suppressed:
        return

    script = lib._script
    position = script.position

    # Risk management: Check if trading is halted
    if position.risk_halt_trading:
        return

    # Intraday-cap freeze gate: once ``strategy.risk.max_intraday_filled_orders``
    # is reached for the current day, TradingView blocks all subsequent entry
    # placements until the next trading day. Dropping only the fill is not
    # enough — an entry placed on a latched bar would survive the day rollover
    # and fire a phantom entry at the new day's open, where the counter has
    # already reset. Block the placement itself, matching TV's broker emulator.
    if position._is_intraday_filled_cap_reached():
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
        # We need negative direction for entry limit orders - NOTE: it is tested
        limit = _price_round(limit, -direction_sign)
    if isinstance(stop, NA):
        stop = None
    elif stop is not None:
        stop = _price_round(stop, direction_sign)

    # Creation-time margin check for market entry orders (TradingView backtest behavior).
    # Skip in broker mode: the exchange enforces margin authoritatively, and the script's
    # equity view can drift from the exchange (funding, fees, transfers) — making the
    # local check a source of silent false positives rather than a safety net.
    if limit is None and stop is None and isinstance(position, SimPosition):
        margin_percent = (script.margin_short if direction_sign < 0
                          else script.margin_long)
        if margin_percent > 0:
            margin_ratio = margin_percent / 100.0
            slippage_amount = script.slippage * syminfo.mintick
            expected_price = position.c + slippage_amount * direction_sign
            equity = script.initial_capital + position.netprofit + position.openprofit
            # Margin/equity are in account currency — convert via pointvalue.
            margin_needed = abs(size) * expected_price * syminfo.pointvalue * margin_ratio
            if margin_needed > equity:
                return

    # If it is not a market order, we should check pyramiding and flip conditions here
    # Market orders are checked at the order processing time
    if limit is not None or stop is not None:
        # Check if the order has the same direction
        if position.sign == direction_sign:
            # Check pyramiding limit for entry orders adding to existing position
            if lib._script.pyramiding <= len(position.open_trades):
                # Pyramiding limit reached - don't add the order
                return

        elif position.size != 0.0:
            # TradingView calculates the flip quantity at order creation time,
            # not at execution time. If we have an opposite direction position,
            # we need to add the position size to the order size to flip it.
            # This means the order will first close the existing position,
            # then open a new one in the opposite direction.
            size -= position.size  # Subtract because position.size has opposite sign

    order = Order(id, size, order_type=_order_type_entry, limit=limit, stop=stop, oca_name=oca_name,
                  oca_type=oca_type, comment=comment, alert_message=alert_message)
    # Store in entry_orders dict
    position._add_order(order)


# noinspection PyShadowingBuiltins,PyProtectedMember,PyShadowingNames,PyUnusedLocal
def exit(id: str, from_entry: str = "",
         qty: PyneFloat = na_float, qty_percent: PyneFloat = na_float,
         profit: PyneFloat = na_float, limit: PyneFloat = na_float,
         loss: PyneFloat = na_float, stop: PyneFloat = na_float,
         trail_price: PyneFloat = na_float, trail_points: PyneFloat = na_float,
         trail_offset: PyneFloat = na_float,
         oca_name: PyneStr = na_str, oca_type: _oca.Oca | None = None,
         comment: PyneStr = na_str, comment_profit: PyneStr = na_str,
         comment_loss: PyneStr = na_str, comment_trailing: PyneStr = na_str,
         alert_message: PyneStr = na_str, alert_profit: PyneStr = na_str,
         alert_loss: PyneStr = na_str, alert_trailing: PyneStr = na_str,
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
    if lib._lib_semaphore or lib._strategy_suppressed:
        return

    script = lib._script
    position = script.position

    if qty < 0.0:
        return

    direction = 0
    size = 0.0
    init_size = 0.0

    # noinspection PyProtectedMember,PyShadowingNames
    def _exit(entry_pending: bool = False):
        nonlocal limit, stop, trail_price, from_entry, direction, size, oca_name, oca_type

        # Sticky bracket (TV semantics): a leg is identified by (id, from_entry).
        # Re-issuing it every bar updates its prices, but a leg that already fired
        # its slice must not be resurrected, and the slice is reserved off the
        # entry's ORIGINAL size, not the shrinking remainder.
        #
        # The freeze only applies once the bound entry has FILLED. While the entry
        # is still a PENDING order, re-issuing ``strategy.entry`` can change its
        # qty bar-to-bar (e.g. cash-based sizing re-evaluated each bar), and the
        # exit must track the entry's CURRENT pending size — otherwise it locks the
        # first bar's size and under-closes the eventual fill, stranding a sliver
        # that some other rule (a per-bar ``close_all``) then mops up as a phantom
        # second trade.
        exit_key = (id, from_entry)
        existing = position.exit_orders.get(exit_key)
        if existing is not None and existing.consumed:
            return

        if existing is not None and not entry_pending:
            # Live leg on an already-filled position: keep its reserved slice,
            # only update prices.
            reserved = existing.reserved_size
        elif not isinstance(qty, NA):
            reserved = abs(qty)
        elif not isinstance(qty_percent, NA):
            reserved = abs(init_size) * (qty_percent * 0.01)
        else:
            # No-qty "rest" leg: the entry size minus the slices reserved by
            # sibling legs (consumed siblings keep their reservation until the
            # entry fully closes), so it never over-closes the position.
            sibling = sum(o.reserved_size for o in position.exit_orders.values()
                          if o.order_id == from_entry and o is not existing)
            reserved = abs(init_size) - sibling

        reserved = _size_round(reserved)
        if reserved <= 0.0:
            return
        size = -direction * reserved

        # Store tick values for later calculation when entry price is known
        profit_ticks: float | None = _na_to_none(profit)
        loss_ticks: float | None = _na_to_none(loss)
        trail_points_ticks: float | None = _na_to_none(trail_points)

        # An exit must arm at least one trigger. TradingView treats a call whose
        # price/tick args ALL resolve to na as a no-op -- e.g. brackets computed
        # from a flat position_avg_price (na) on a bar before the entry fills --
        # not a level-less market close that fires at the next open.
        if (isinstance(limit, NA) and isinstance(stop, NA) and isinstance(profit, NA)
                and isinstance(loss, NA) and isinstance(trail_price, NA)
                and isinstance(trail_points, NA)):
            return

        # We need to have limit, stop or both
        if isinstance(limit, NA) and isinstance(stop, NA) and not isinstance(trail_price, NA):
            return

        _limit = _na_to_none(limit)
        if _limit is not None:
            _limit = _price_round(_limit, direction)
        _stop = _na_to_none(stop)
        if _stop is not None:
            _stop = _price_round(_stop, -direction)
        _trail_price = _na_to_none(trail_price)
        if _trail_price is not None:
            _trail_price = _price_round(_trail_price, -direction)

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

        # Add order
        order = Order(
            from_entry, size, exit_id=id, order_type=_order_type_close,
            limit=_limit, stop=_stop,
            trail_price=_trail_price, trail_offset=_na_to_none(trail_offset),
            profit_ticks=profit_ticks, loss_ticks=loss_ticks, trail_points_ticks=trail_points_ticks,
            oca_name=_na_to_none(oca_name), oca_type=oca_type,
            comment=_na_to_none(comment),
            alert_message=_na_to_none(alert_message),
            comment_profit=_na_to_none(comment_profit),
            comment_loss=_na_to_none(comment_loss),
            comment_trailing=_na_to_none(comment_trailing),
            alert_profit=_na_to_none(alert_profit),
            alert_loss=_na_to_none(alert_loss),
            alert_trailing=_na_to_none(alert_trailing)
        )

        # Sticky bracket (TV semantics): a re-issued live trailing leg keeps its
        # activated high/low-water mark. TradingView carries ONE logical trailing
        # stop across re-issues -- only the activation level tracks the
        # (entry-anchored) trail_points -- so a fresh Order must inherit the
        # ratcheted ``trail_stop`` instead of re-arming at the bare activation
        # level every bar, which would leave the stop permanently one or more bars
        # behind the carried water mark.
        if existing is not None and not entry_pending and existing.trail_triggered:
            order.trail_triggered = True
            order.trail_stop = existing.trail_stop

        position._add_order(order)
        position._seed_trail_at_issue(order)

    # Find direction and size
    if from_entry:
        # Get from entry_orders dict
        entry_order: Order | None = position.entry_orders.get(from_entry, None)

        # Find open trade if no entry order found
        if not entry_order:
            for trade in position.open_trades:
                if trade.entry_id == from_entry:
                    direction = trade.sign
                    init_size = trade.init_size
                    _exit()

            # The position should be opened, or an entry order should exist
            if not entry_order:
                return
        else:
            direction = entry_order.sign
            init_size = entry_order.size
            _exit(entry_pending=True)

    else:
        # If still no entry order found, we should exit all open trades and open orders
        if not direction:
            for order in list(position.entry_orders.values()):
                direction = order.sign
                init_size = order.size
                from_entry = order.order_id or ""
                # Only mark as from_entry_na on first creation (not replacement)
                exit_key = (id, from_entry)
                had_existing_exit = exit_key in position.exit_orders
                _exit(entry_pending=True)
                if not had_existing_exit:
                    exit_order = position.exit_orders.get(exit_key)
                    if exit_order is not None:
                        exit_order.from_entry_na = True

            if not direction:
                for trade in position.open_trades:
                    direction = trade.sign
                    init_size = trade.init_size
                    from_entry = trade.entry_id or ""
                    _exit()


# noinspection PyProtectedMember,PyShadowingNames,PyShadowingBuiltins,PyUnusedLocal,DuplicatedCode
def order(id: str, direction: direction.Direction, qty: int | PyneFloat = na_float,
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
    :param limit: The limit price of the order. With ``stop`` set too, the order becomes two OCA legs (a limit and a stop), not a single stop-limit order
    :param stop: The stop price of the order. With ``limit`` set too, the order becomes two OCA legs (a limit and a stop), not a single stop-limit order
    :param oca_name: The name of the One-Cancels-All (OCA) group
    :param oca_type: Specifies how an unfilled order behaves when another order in the same OCA group executes
    :param comment: Additional notes on the filled order
    :param alert_message: Custom text for the alert that fires when an order fills
    :param disable_alert: If true, the strategy does not trigger an alert when the order fills
    """
    if lib._lib_semaphore or lib._strategy_suppressed:
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
        limit = _price_round(limit, direction_sign)  # TODO: test this if the direction here is correct
    if isinstance(stop, NA):
        stop = None
    elif stop is not None:
        stop = _price_round(stop, -direction_sign)  # TODO: test this if the direction here is correct

    # Create the order with _order_type_normal
    # This is a "normal" order that simply adds to or subtracts from position
    # It doesn't follow entry/exit rules and can freely modify positions
    order = Order(id, size, order_type=_order_type_normal, limit=limit, stop=stop,
                  oca_name=oca_name, oca_type=oca_type, comment=comment,
                  alert_message=alert_message)
    position._add_order(order)


#
# Properties
#

# Strategy state accessors below return inert defaults when invoked in a
# security child process: there `lib._script` is None because no
# ScriptRunner.run_iter() ever ran. Pine itself rejects strategy.* state
# reads inside any request.*() argument at compile time (CE10059), so the
# values are never consumed by the chart anyway — this only prevents the
# child from crashing when the chart-context body references them.

# noinspection PyProtectedMember
@module_property
def equity() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.equity


# noinspection PyProtectedMember
@module_property
def eventrades() -> PyneInt:
    if lib._script is None:
        return 0
    return lib._script.position.eventrades


# noinspection PyProtectedMember
@module_property
def initial_capital() -> float:
    if lib._script is None:
        return 0.0
    return lib._script.initial_capital


# noinspection PyProtectedMember
@module_property
def grossloss() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.grossloss + lib._script.position.open_commission


# noinspection PyProtectedMember
@module_property
def grossprofit() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.grossprofit


# noinspection PyProtectedMember
@module_property
def losstrades() -> int:
    if lib._script is None:
        return 0
    return lib._script.position.losstrades


# noinspection PyProtectedMember
@module_property
def max_drawdown() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.max_drawdown


# noinspection PyProtectedMember
@module_property
def max_runup() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.max_runup


# noinspection PyProtectedMember
@module_property
def netprofit() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.netprofit


# noinspection PyProtectedMember
@module_property
def openprofit() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.openprofit


# noinspection PyProtectedMember
@module_property
def position_size() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.size


# noinspection PyProtectedMember
@module_property
def position_avg_price() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.avg_price


# noinspection PyProtectedMember
@module_property
def wintrades() -> PyneInt:
    if lib._script is None:
        return 0
    return lib._script.position.wintrades
