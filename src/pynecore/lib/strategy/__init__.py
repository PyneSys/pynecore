from typing import TYPE_CHECKING, Literal, overload
from typing import TypeAlias as _TypeAlias  # underscore-aliased: kept out of the module-property registry

import math
import struct
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
# Function-and-namespace modules — the IDE-facing rebinding; at runtime the AST
# transformer routes bare reads and calls to the module's self-named function
#

from ...types.ohlcv import OHLCV

if TYPE_CHECKING:
    from .closedtrades import closedtrades
    from .opentrades import opentrades


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

# Trailing-stop walk results (see ``SimPosition._process_trailing_stop``)
_trail_filled = 0
_trail_deferred = 1
_trail_pending = 2

# Order-book dict key shapes. A close placed by ``strategy.close()`` /
# ``strategy.close_all()`` in BACKTEST carries a unique ``book_seq`` stamp so
# that multiple same-bar partial closes on one entry STACK instead of colliding
# on a shared key; that stamp becomes the optional last tuple element. Sticky
# ``strategy.exit`` brackets, risk/defensive closes and the live broker path
# leave ``book_seq`` None and keep the bare 2-/3-tuple key unchanged.
_ExitOrderKey: _TypeAlias = tuple[str | None, str | None] | tuple[str | None, str | None, int]
_MarketOrderKey: _TypeAlias = (tuple[_OrderType, str | None, str | None]
                               | tuple[_OrderType, str | None, str | None, int])

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


def _exit_order_key(order_: 'Order') -> '_ExitOrderKey':
    """Order-book key for an exit/close order.

    A backtest partial close stamped with a ``book_seq`` (see
    :meth:`PositionBase._next_close_seq`) appends it as a 3rd element so several
    same-bar closes on one entry get distinct keys and STACK; every other order
    (sticky ``strategy.exit``, risk/defensive close, live broker close) keeps the
    bare ``(exit_id, order_id)`` key, leaving their dedup-by-id semantics intact.
    Insert and pop sites MUST both route through this helper so they never drift.
    """
    if order_.book_seq is None:
        return order_.exit_id, order_.order_id
    return order_.exit_id, order_.order_id, order_.book_seq


def _market_order_key(order_: 'Order') -> '_MarketOrderKey':
    """Market-orders key, mirroring :func:`_exit_order_key`'s ``book_seq`` rule."""
    if order_.book_seq is None:
        return order_.order_type, order_.order_id, order_.exit_id
    return order_.order_type, order_.order_id, order_.exit_id, order_.book_seq


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
        "deferred_qty",  # Default-sized entry: quantity re-resolves at the actual fill price
        "flip_extra",  # Reversal flip magnitude frozen at creation (added back on deferred re-size)
        "bar_index",  # Bar index when the order was placed
        "filled_by_type",  # Type of execution: 'profit', 'loss', 'trailing', or None
        "from_entry_na",  # True if exit was created without explicit from_entry (applies to any position)
        "reserved_size",  # Exit-leg slice of the entry's original size (frozen at creation)
        "rest_leg",  # Exit leg with no explicit qty/qty_percent: closes the WHOLE bound entry
        "consumed",  # True once an exit leg fired its slice while its entry is still open
        "book_seq",  # Monotonic stamp for same-bar strategy.close()/close_all() partial closes
                     # (backtest only); None for non-stacking sticky-exit / risk / live orders
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
        self.deferred_qty = False
        self.flip_extra = 0.0
        self.bar_index = -1  # Will be set when order is added to position
        self.filled_by_type: Literal['profit', 'loss', 'trailing'] | None = None  # Will be set when order fills
        self.from_entry_na = False
        self.reserved_size = abs(size)
        self.rest_leg = False
        self.consumed = False
        # Stamped only by strategy.close()/close_all() in backtest (see _next_close_seq);
        # left None everywhere else so the order-book key keeps its bare shape.
        self.book_seq: int | None = None

    def __repr__(self):
        return f"Order(order_id={self.order_id}; exit_id={self.exit_id}; size={self.size}; type: {self.order_type}; " \
               f"limit={self.limit}; stop={self.stop}; " \
               f"trail_price={self.trail_price}; trail_offset={self.trail_offset}; " \
               f"oca_name={self.oca_name}; comment={self.comment}; book_seq={self.book_seq}; " \
               f"bar_index={self.bar_index})"


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
    __slots__ = ('_close_seq_counter',)

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
    closed_trades_count: int
    max_drawdown: float
    max_runup: float
    open_trades: list['Trade']
    closed_trades: 'deque[Trade]'
    new_closed_trades: list['Trade']
    entry_orders: dict[str | None, 'Order']
    exit_orders: dict['_ExitOrderKey', 'Order']
    risk_halt_trading: bool
    # Monotonic counter feeding _next_close_seq(); initialized by each subclass.
    _close_seq_counter: int

    def _next_close_seq(self) -> int:
        """Return a fresh monotonic stamp for a same-bar partial close.

        ``strategy.close()`` / ``strategy.close_all()`` use this so that several
        partial closes issued on one bar against the same entry id get DISTINCT
        order-book keys and therefore STACK (all fill) instead of the later call
        silently evicting the earlier one. Backtest only — the live broker path
        leaves ``Order.book_seq`` None (handled in a separate change).
        """
        self._close_seq_counter += 1
        return self._close_seq_counter

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

    def _seed_trail_at_issue(self, order: 'Order', *, fold_extreme: bool = True) -> None:
        """Sim-only hook: fold the issue bar into a freshly issued trailing
        exit's water mark.

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
        '_deferred_margin_call', '_fill_counter', '_partial_close_bar',
        '_entry_open_ledger'
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
        self.market_orders: dict[_MarketOrderKey, Order] = {}  # Market orders from strategy.market()
        self.entry_orders: dict[str | None, Order] = {}  # Entry orders from strategy.entry()
        # Exit orders from strategy.exit(), strategy.close(), etc.
        # Key is (exit_id, from_entry) — both partial-TP fan-out (same from_entry,
        # different ids) and from_entry_na fan-out (same id, different from_entry)
        # must coexist; only repeated calls with both fields equal modify-in-place.
        # A backtest strategy.close()/close_all() order additionally carries a
        # book_seq stamp appended as a 3rd key element, so same-bar partial closes
        # on one entry stack instead of evicting each other (see _add_order).
        self.exit_orders: dict[_ExitOrderKey, Order] = {}
        self.orderbook = PriceOrderBook()

        # Trades
        self.open_trades: list[Trade] = []
        self.closed_trades: deque[Trade] = deque(maxlen=9000)  # 9000 is the limit of TV
        self.new_closed_trades: list[Trade] = []
        # Per-entry bound open quantity — drives the exit-order lifecycle (see
        # _reduce_entry_ledger); the trade rows themselves are attributed FIFO.
        self._entry_open_ledger: dict[str, float] = {}

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
        # Monotonic stamp source for same-bar stacking of partial closes.
        self._close_seq_counter: int = 0
        # bar_index of the most recent filled partial strategy.close() (a stamped
        # close with an entry id); lets a same-bar close_all clamp to flat instead
        # of overshooting when the partial already shed part of the position.
        self._partial_close_bar: int = -1

    def _add_order(self, order: Order):
        """ Add an order to the strategy """
        # Set the bar_index when the order is placed
        order.bar_index = int(lib.bar_index)

        # Add market order to market orders dict. Key on exit_id too: two
        # brackets sharing the same from_entry (order_id) would otherwise
        # collide on the same key, so a second gap-through exit would evict
        # the first and only one of them would fill on the gap bar. A stacked
        # partial close additionally keys on book_seq (see _market_order_key).
        if order.is_market_order:
            self.market_orders[_market_order_key(order)] = order

        # Check if an order with this ID already exists and remove it first
        if order.order_type == _order_type_close:
            exit_key = _exit_order_key(order)
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
            self.exit_orders.pop(_exit_order_key(order), None)
        else:
            # Both entry and normal orders are stored in entry_orders dict
            self.entry_orders.pop(order.order_id, None)
        # Remove market order from market orders dict
        if order.is_market_order:
            self.market_orders.pop(_market_order_key(order), None)
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

    def _reduce_entry_ledger(self, entry_id: str | None, qty: float) -> None:
        """Settle a closing fill against the entry it was bound to.

        TradingView keeps two ledgers: closed TRADES are attributed FIFO
        across the whole position, but each exit/close order still settles
        against its own ``from_entry``. Only when that bound quantity is
        exhausted are the entry's remaining exit legs cancelled — a bracket
        survives its entry's trade rows being consumed FIFO by another
        entry's close, and conversely dies once its entry's quantity is
        spent even while those rows still sit open under other entries.
        """
        if entry_id is None:
            return
        left = self._entry_open_ledger.get(entry_id)
        if left is None:
            return
        left -= qty
        if _size_round(left) <= 0.0:
            del self._entry_open_ledger[entry_id]
            for exit_order in list(self.exit_orders.values()):
                if exit_order.order_id == entry_id:
                    self._remove_order(exit_order)
        else:
            self._entry_open_ledger[entry_id] = left

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

        # Record same-bar partial strategy.close() fills (stamped close carrying an
        # entry id) so a later same-bar close_all clamps to flat instead of
        # overshooting on the size it captured before this partial shed part of it.
        # Only a fill that actually sheds size arms the marker: a consumed/zero-size
        # tombstone (a fired partial-exit leg kept alive while its entry stays open)
        # is re-filled as a no-op every bar and must NOT re-arm it, or it would
        # wrongly clamp an unrelated deferred-margin-call close_all overshoot.
        if (order.order_type == _order_type_close and order.order_id is not None
                and order.book_seq is not None
                and not order.consumed and _size_round(order.size) != 0.0):
            self._partial_close_bar = int(lib.bar_index)

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

            # Check list of open trades.
            # close_entries_rule='ANY': an entry-bound close consumes only its
            # own entry's trades. FIFO (TV default): a closing fill consumes
            # open trades oldest-first regardless of the from_entry binding —
            # the binding only sizes the order and gates its activation.
            close_any = (order.order_type == _order_type_close and order.order_id is not None
                         and script.close_entries_rule == 'ANY')
            new_open_trades = []
            for trade in self.open_trades:
                if order.size != 0.0 and (not close_any or trade.entry_id == order.order_id):
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
                    position_flat = _size_round(self.size) == 0.0
                    if position_flat:
                        size -= self.size
                        self.size = 0.0
                    self.sign = 0.0 if self.size == 0.0 else 1.0 if self.size > 0.0 else -1.0
                    trade.size += size
                    if position_flat:
                        # `size` already absorbed the position residual above, so the
                        # trade that flattened the position is fully closed. Snap off
                        # float epsilon so it is removed from open_trades instead of
                        # lingering as a ~0-size ghost trade — a stale leg would force
                        # avg_price to NA and poison equity (and every subsequent
                        # percent-of-equity sizing) on later bars.
                        trade.size = 0.0
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

            # Settle the closed quantity against the entry ledger. A close
            # bound to a from_entry settles that entry regardless of which
            # trades the FIFO fill consumed; an unbound close (close_all,
            # reversal, margin call) settles entries oldest-first, mirroring
            # the trade rows.
            closed_qty = filled_size - abs(order.size)
            if closed_qty > 0.0:
                if order.order_type == _order_type_close and order.order_id is not None:
                    self._reduce_entry_ledger(order.order_id, closed_qty)
                else:
                    for eid in list(self._entry_open_ledger):
                        if closed_qty <= 0.0:
                            break
                        take = min(self._entry_open_ledger[eid], closed_qty)
                        self._reduce_entry_ledger(eid, take)
                        closed_qty -= take

            if delete:
                # A partial-exit leg that fired its whole slice while its entry's
                # bound quantity is still open becomes a tombstone: kept in
                # exit_orders (so its reservation still counts against sibling
                # "rest" legs and a per-bar strategy.exit() re-call cannot
                # resurrect it) and only pulled from the order book. It is purged
                # when the entry's bound quantity is exhausted (_reduce_entry_ledger).
                if (order.order_type == _order_type_close and order.order_id is not None
                        and _size_round(order.size) == 0.0
                        and self._entry_open_ledger.get(order.order_id, 0.0) > 0.0):
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
                if entry_id is not None:
                    self._entry_open_ledger[entry_id] = (
                        self._entry_open_ledger.get(entry_id, 0.0) + abs(overshoot_trade.size))
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
            if entry_id is not None:
                self._entry_open_ledger[entry_id] = (
                    self._entry_open_ledger.get(entry_id, 0.0) + abs(order.size))
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
            self._entry_open_ledger.clear()

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
            # A default-sized order settles its quantity at the actual fill price
            if order.deferred_qty:
                self._resolve_deferred_qty(order, price)
                if order.size == 0.0:
                    self._remove_order(order)
                    return False
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
            # Exit orders should never reverse position direction; only entry
            # orders open or reverse. A close_all (order_id None) is normally
            # allowed to overshoot — a deferred margin call can shrink the
            # position after close_all captured its size, and TV opens the
            # overshoot as an opposite trade. But when the shrink came from a
            # same-bar partial strategy.close() (which stamps book_seq), TV
            # closes only what remains, so clamp to flat instead of reversing.
            if (order.order_type == _order_type_close or close_only) and (
                    order.order_id is not None
                    or self._partial_close_bar == int(lib.bar_index)):
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

    def _exit_awaits_entry(self, order: Order) -> bool:
        """True while an exit leg bound to a ``from_entry`` has no open trade to act on.

        TradingView activates a ``strategy.exit`` bracket only after its bound
        entry fills. Until then (entry pending, cancelled or rejected) the leg
        must not trigger: a fill would cancel its sibling OCA legs and count
        toward the filled-order caps even though there is nothing it can close.
        """
        if order.order_type != _order_type_close or order.order_id is None or order.from_entry_na:
            return False
        return order.order_id not in self._entry_open_ledger

    def _check_high_stop(self, order: Order) -> bool:
        """ Check high stop and trailing trigger """
        if order.stop is None:
            return False
        if self._exit_awaits_entry(order):
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
            if self._exit_awaits_entry(order):
                return False
            # Short limit order (size < 0) triggers when price rises to limit level
            if order.size < 0 and order.limit <= self.h:
                p = max(order.limit, self.o)
                order.filled_by_type = 'profit'
                self.fill_order(order, p, p, self.l)
                return True
        return False

    def _check_close_leg_up(self, order: Order) -> bool:
        """Fill on the closing ascent (low -> close) of the intrabar walk.

        Only an order that became active mid-bar can still be pending here — an
        exit whose entry filled on an earlier leg. The segment starts at the
        bar's low, so fills land exactly at the trigger price (no open-gap
        clamp like :meth:`_check_high` applies).
        """
        if self._exit_awaits_entry(order):
            return False
        # Short limit (sell back) triggers when price rises to the limit level
        if order.limit is not None and order.size < 0 and order.limit <= self.c:
            order.filled_by_type = 'profit'
            self.fill_order(order, order.limit, order.limit, self.l)
            return True
        # Buy stop triggers when price rises to the stop level
        if order.stop is not None and order.size > 0 and order.stop <= self.c:
            p = order.stop
            slippage = lib._script.slippage
            if slippage > 0:
                p += syminfo.mintick * slippage
            order.filled_by_type = 'loss'
            self.fill_order(order, p, p, self.l)
            return True
        return False

    def _process_trailing_stop(self, order: Order, ohlc: bool, close_leg: bool = False) -> int:
        """Process a trailing-stop exit for the current bar (TradingView model).

        TradingView's broker emulator moves the market price along the assumed
        intrabar path (``open -> high -> low -> close`` or
        ``open -> low -> high -> close``, see :meth:`process_orders`) and the
        trailing stop follows it tick by tick: the high/low-water mark advances
        on every favorable segment of the path — including the current bar's own
        extreme — and the stop sits ``trail_offset`` ticks behind it. The trail
        arms when the path touches ``order.trail_price`` (``entry ±
        trail_points``) and can fill on the SAME bar once the path retraces
        ``trail_offset`` ticks from the watermark reached after arming: a bar
        that pierces the activation level, runs on to its extreme and pulls back
        fills at ``extreme -/+ offset``, not at the activation level. With
        ``trail_offset == 0`` the stop sits on the watermark itself, so the fill
        lands at the activation tick (or at the open of a bar opening beyond the
        carried watermark).

        A bar that opens beyond a CARRIED stop (inter-bar gap) fills at the
        open; within the bar the path is assumed gapless, so fills land exactly
        at the trailed stop level. When the same order also carries a hard
        ``stop=`` leg that the path reaches earlier in intrabar time — before
        the trail arms, or at a less favorable level on the same falling
        segment — the trail defers to the price walk so the hard stop wins.
        Likewise a take-profit ``limit=`` leg reached on a favorable segment
        fires before any trailing fill on a later retrace, so the trail defers
        to the price walk there too (verified against TradingView references
        on BINANCE:ETHUSDT.P — TV fills the limit at its level, not the
        trailing stop at ``watermark -/+ offset``); only an offset-0 arming
        fill at a not-stricter activation level precedes the limit on the same
        segment.

        The walk is two-phase so it interleaves with the intrabar margin-call
        checkpoints in :meth:`_process_limit_stop_orders`: the default call
        handles the open tick and the legs up to the second extreme, persists
        the armed/water-mark state on the order and reports ``_trail_pending``;
        a ``close_leg=True`` call resumes from the second extreme and walks the
        final (extreme -> close) segment. A fill on that closing leg happens
        chronologically after a margin call at the adverse extreme, which may
        have already trimmed the position by then.

        :param order: The exit order carrying ``trail_price``.
        :param ohlc: The bar's intra-bar leg order (see :meth:`process_orders`).
        :param close_leg: If True, walk only the closing (second extreme ->
            close) segment, resuming the state a prior default call persisted.
        :return: ``_trail_filled`` if the order filled, ``_trail_deferred`` if
            the walk defers to the price walk (or cannot act this bar),
            ``_trail_pending`` if the closing leg is still outstanding.
        """
        if order.trail_price is None:
            return _trail_deferred
        if self._exit_awaits_entry(order):
            return _trail_deferred
        round_to_mintick = lib.math.round_to_mintick
        offset_price = syminfo.mintick * order.trail_offset
        slippage = lib._script.slippage

        if order.sign < 0:
            # Long position: trailing sell-stop riding under the high-water mark.
            armed = order.trail_triggered
            stop = order.trail_stop if armed else None

            if not close_leg and armed and stop is not None:
                # A carried stop gapped through between bars fills at the open.
                if self.o <= stop:
                    p = self.o
                    if slippage > 0:
                        p -= syminfo.mintick * slippage
                    order.filled_by_type = 'trailing'
                    self.fill_order(order, p, self.h, p)
                    return _trail_filled
                # The open tick advances the water mark; with trail_offset == 0
                # the stop lands on the open itself and fills there.
                new_stop = round_to_mintick(self.o - offset_price)
                if new_stop > stop:
                    stop = new_stop
                    if self.o <= stop:
                        p = stop
                        if slippage > 0:
                            p -= syminfo.mintick * slippage
                        order.filled_by_type = 'trailing'
                        self.fill_order(order, p, self.h, p)
                        return _trail_filled
            elif not close_leg and not armed and self.o >= order.trail_price:
                # The bar opens beyond the activation level: the trail arms on
                # the first tick with the open as its water mark.
                armed = True
                stop = round_to_mintick(self.o - offset_price)
                if self.o <= stop:
                    p = stop
                    if slippage > 0:
                        p -= syminfo.mintick * slippage
                    order.filled_by_type = 'trailing'
                    self.fill_order(order, p, self.h, p)
                    return _trail_filled

            # Walk the assumed intrabar path: rising segments arm the trail and
            # ratchet the water mark, a falling segment fills at the trailed
            # stop when it reaches it.
            if close_leg:
                prev = self.l if ohlc else self.h
                path: tuple[float, ...] = (self.c,)
            else:
                prev = self.o
                path = (self.h, self.l) if ohlc else (self.l, self.h)
            for nxt in path:
                if nxt > prev:
                    if order.limit is not None and nxt >= order.limit and not (
                            not armed and offset_price <= 0
                            and order.trail_price <= order.limit
                            and nxt >= order.trail_price):
                        # The take-profit limit leg is reached on this rising
                        # segment, earlier in intrabar time than any trailing
                        # fill on a later retrace: defer to the price walk so
                        # the limit wins, carrying the trail state ratcheted
                        # so far. Only an offset-0 arming fill at a not-higher
                        # activation level precedes it.
                        order.trail_triggered = armed
                        if armed:
                            order.trail_stop = stop
                        return _trail_deferred
                    if not armed and nxt >= order.trail_price:
                        armed = True
                        stop = round_to_mintick(order.trail_price - offset_price)
                        if order.trail_price <= stop:
                            # trail_offset == 0: the stop sits on the activation
                            # level and the arming tick itself fills it.
                            p = stop
                            if slippage > 0:
                                p -= syminfo.mintick * slippage
                            order.filled_by_type = 'trailing'
                            self.fill_order(order, p, self.h, p)
                            return _trail_filled
                    if armed:
                        new_stop = round_to_mintick(nxt - offset_price)
                        if stop is None or new_stop > stop:
                            stop = new_stop
                else:
                    if order.limit is not None and prev >= order.limit:
                        # The take-profit limit became marketable earlier on
                        # the path (at the open tick or on a prior rising
                        # segment): defer to the price walk so the limit wins.
                        order.trail_triggered = armed
                        if armed:
                            order.trail_stop = stop
                        return _trail_deferred
                    if order.stop is not None and nxt <= order.stop and (
                            not armed or stop is None or order.stop >= stop):
                        # The hard stop leg is reached earlier in intrabar time:
                        # defer to the price walk, carrying the trail state
                        # ratcheted so far.
                        order.trail_triggered = armed
                        if armed:
                            order.trail_stop = stop
                        return _trail_deferred
                    if armed and stop is not None and nxt <= stop:
                        p = stop
                        if slippage > 0:
                            p -= syminfo.mintick * slippage
                        order.filled_by_type = 'trailing'
                        self.fill_order(order, p, self.h, p)
                        return _trail_filled
                prev = nxt

            # No fill: persist the ratcheted state — the default call hands it
            # to the closing-leg call, which in turn carries it into the next bar.
            if armed:
                order.trail_triggered = True
                order.trail_stop = stop
            return _trail_pending

        if order.sign > 0:
            # Short position: trailing buy-stop riding above the low-water mark.
            armed = order.trail_triggered
            stop = order.trail_stop if armed else None

            if not close_leg and armed and stop is not None:
                # A carried stop gapped through between bars fills at the open.
                if self.o >= stop:
                    p = self.o
                    if slippage > 0:
                        p += syminfo.mintick * slippage
                    order.filled_by_type = 'trailing'
                    self.fill_order(order, p, p, self.l)
                    return _trail_filled
                # The open tick advances the water mark; with trail_offset == 0
                # the stop lands on the open itself and fills there.
                new_stop = round_to_mintick(self.o + offset_price)
                if new_stop < stop:
                    stop = new_stop
                    if self.o >= stop:
                        p = stop
                        if slippage > 0:
                            p += syminfo.mintick * slippage
                        order.filled_by_type = 'trailing'
                        self.fill_order(order, p, p, self.l)
                        return _trail_filled
            elif not close_leg and not armed and self.o <= order.trail_price:
                # The bar opens beyond the activation level: the trail arms on
                # the first tick with the open as its water mark.
                armed = True
                stop = round_to_mintick(self.o + offset_price)
                if self.o >= stop:
                    p = stop
                    if slippage > 0:
                        p += syminfo.mintick * slippage
                    order.filled_by_type = 'trailing'
                    self.fill_order(order, p, p, self.l)
                    return _trail_filled

            # Walk the assumed intrabar path: falling segments arm the trail and
            # ratchet the water mark, a rising segment fills at the trailed stop
            # when it reaches it.
            if close_leg:
                prev = self.l if ohlc else self.h
                path = (self.c,)
            else:
                prev = self.o
                path = (self.h, self.l) if ohlc else (self.l, self.h)
            for nxt in path:
                if nxt < prev:
                    if order.limit is not None and nxt <= order.limit and not (
                            not armed and offset_price <= 0
                            and order.trail_price >= order.limit
                            and nxt <= order.trail_price):
                        # The take-profit limit leg is reached on this falling
                        # segment, earlier in intrabar time than any trailing
                        # fill on a later rebound: defer to the price walk so
                        # the limit wins, carrying the trail state ratcheted
                        # so far. Only an offset-0 arming fill at a not-lower
                        # activation level precedes it.
                        order.trail_triggered = armed
                        if armed:
                            order.trail_stop = stop
                        return _trail_deferred
                    if not armed and nxt <= order.trail_price:
                        armed = True
                        stop = round_to_mintick(order.trail_price + offset_price)
                        if order.trail_price >= stop:
                            # trail_offset == 0: the stop sits on the activation
                            # level and the arming tick itself fills it.
                            p = stop
                            if slippage > 0:
                                p += syminfo.mintick * slippage
                            order.filled_by_type = 'trailing'
                            self.fill_order(order, p, p, self.l)
                            return _trail_filled
                    if armed:
                        new_stop = round_to_mintick(nxt + offset_price)
                        if stop is None or new_stop < stop:
                            stop = new_stop
                else:
                    if order.limit is not None and prev <= order.limit:
                        # The take-profit limit became marketable earlier on
                        # the path (at the open tick or on a prior falling
                        # segment): defer to the price walk so the limit wins.
                        order.trail_triggered = armed
                        if armed:
                            order.trail_stop = stop
                        return _trail_deferred
                    if order.stop is not None and nxt >= order.stop and (
                            not armed or stop is None or order.stop <= stop):
                        # The hard stop leg is reached earlier in intrabar time:
                        # defer to the price walk, carrying the trail state
                        # ratcheted so far.
                        order.trail_triggered = armed
                        if armed:
                            order.trail_stop = stop
                        return _trail_deferred
                    if armed and stop is not None and nxt >= stop:
                        p = stop
                        if slippage > 0:
                            p += syminfo.mintick * slippage
                        order.filled_by_type = 'trailing'
                        self.fill_order(order, p, p, self.l)
                        return _trail_filled
                prev = nxt

            # No fill: persist the ratcheted state — the default call hands it
            # to the closing-leg call, which in turn carries it into the next bar.
            if armed:
                order.trail_triggered = True
                order.trail_stop = stop
            return _trail_pending

        return _trail_deferred

    def _seed_trail_at_issue(self, order: Order, *, fold_extreme: bool = True) -> None:
        """Fold the issue bar into a trailing exit's high/low-water mark.

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

        With ``fold_extreme=False`` (a changed-params re-issue) the water mark
        anchors to the issue bar's CLOSE tick instead of its extreme: the
        replaced leg sees only the current price, so it arms there when the
        activation is already met, and the next bar's open advances the stop
        only when favorable. TV-verified both ways on BINANCE:BTCUSDT 30m
        (per-bar ``atr*mult`` trail): a long re-issue filled at
        ``next open - offset`` (open above close, mark advanced) and a short
        re-issue filled at ``close + offset`` (open above close, mark kept).

        :param order: The freshly (re-)issued trailing exit order.
        :param fold_extreme: If True, ratchet the issue bar's H/L extreme into
            the water mark; if False, anchor the water mark to the bar close.
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
            new_stop = round_to_mintick((self.h if fold_extreme else self.c) - offset_price)
            if order.trail_stop is None or new_stop > order.trail_stop:
                order.trail_stop = new_stop
        elif order.sign > 0:
            # Short position: trailing buy-stop riding above the low-water mark.
            if not order.trail_triggered:
                if self.c >= trail_price:
                    return
                order.trail_triggered = True
                order.trail_stop = round_to_mintick(trail_price + offset_price)
            new_stop = round_to_mintick((self.l if fold_extreme else self.c) + offset_price)
            if order.trail_stop is None or new_stop < order.trail_stop:
                order.trail_stop = new_stop

    def _check_margin_call(self, check_price: float, *, for_short: bool,
                           at_open: bool = False,
                           can_defer: bool = True,
                           whole_contracts: bool = False) -> bool:
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
        :param whole_contracts: If True, size the liquidation in whole contracts even on
            fractional-lot symbols. TV's bar-open margin call (the one that fires right
            after entry fills at the open price) liquidates whole contracts, while its
            intrabar (H/L) and deferred margin calls work in lot units.
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

        # From 1e7 account-currency units of equity upward the margin-call
        # trigger is an integer-tick comparison on the STRICT side: it fires
        # once the truncated equity tick-count no longer covers the required
        # margin rounded half-up to a tick, even while the float difference
        # is still a positive surplus. Measured on BINANCE:BTCUSDT 30m,
        # Hybrid 2025-10-02 16:00: available funds +0.0047 USD at every bar
        # price, yet TV liquidated one whole contract at H=120300 — exactly
        # the first walk point where this comparison fails (open and low
        # both pass it). From 1e10 margin ticks upward the margin rounds to
        # the nearest multiple of 10 ticks instead (Hybrid 2026-02-28 20:30:
        # available funds +0.0132 USD at the bar low, yet TV liquidated one
        # whole contract — the margin rounded up to the next multiple of 10
        # ticks while the equity truncated 4 ticks below it; the open and
        # high of the same bar stayed on grid and passed).
        mintick = syminfo.mintick
        big_equity = equity >= 1e7 and mintick and mintick > 0
        big_margin = False
        equity_ticks = 0.0
        margin_ticks = 0.0
        if big_equity:
            equity_ticks = math.floor(equity / mintick)
            margin_ticks = margin / mintick
            big_margin = margin_ticks >= 1e10
            if big_margin:
                margin_ticks = 10.0 * round(margin_ticks / 10.0)
            else:
                margin_ticks = math.floor(margin_ticks + 0.5)
            if equity_ticks >= margin_ticks:
                return False
        elif available_funds >= 0:
            return False

        # One contract is worth `check_price * pv` in account currency. Work in
        # lot units (1 / _size_round_factor): whole-lot symbols (stocks) keep
        # TV's integer-contract truncation, while fractional-lot symbols
        # (crypto) liquidate fractional amounts the way TV does instead of
        # force-closing a minimum of one whole contract.
        rfactor = 1 if whole_contracts else syminfo._size_round_factor  # noqa
        if big_margin:
            # Above 1e10 margin ticks the cover comes from the same tick-shadow
            # shortfall as the trigger, then a plain truncation with no float
            # snap (Hybrid 2026-02-16 15:30 and 2026-02-20 13:30 both round the
            # margin up to an odd tick-count that a half-up or half-to-even
            # rounding would keep down).
            shortfall = (margin_ticks - equity_ticks) * mintick
            loss = shortfall / margin_ratio
            cover_lots = int(loss / (check_price * pv) * rfactor)
            if cover_lots < 0:
                cover_lots = 0
        else:
            loss = available_funds / margin_ratio
            raw_cover_lots = abs(loss) / (check_price * pv) * rfactor
            # TV truncates the fractional cover amount, but snaps a raw value
            # that lands within ~2^-26 (relative) of an integer to that
            # integer. Measured on BINANCE:BTCUSDT 30m corpus margin calls:
            # 21840.99976 (rel dist 1.10e-8) covered 21841 lots on TV, while
            # 26510.99945 (rel dist 2.08e-8) truncated to 26510; 2^-26 =
            # 1.49e-8 lies between them.
            nearest_cover = round(raw_cover_lots)
            if abs(raw_cover_lots - nearest_cover) <= raw_cover_lots * 2.0 ** -26 + 1e-9:
                cover_lots = nearest_cover
            else:
                cover_lots = int(raw_cover_lots)
        if cover_lots == 0 and rfactor > 1:
            # Fractional-lot symbol with a sub-lot shortfall: TradingView closes
            # one whole contract, capped by the current position size.
            mc_lots = 0
            margin_call_size = min(1.0, quantity)
        else:
            mc_lots = max(1, cover_lots * 4)
            margin_call_size = mc_lots / rfactor

        if margin_call_size > quantity:
            margin_call_size = quantity

        # Deferral check: mc_size==1 lot at first OHLC extremum, check if AF@C<0
        # Skip deferral when check_price == close: no recovery possible at same price
        if not at_open and can_defer and mc_lots == 1 and check_price != self.c:
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
        Execute a deferred margin call (after the user script has run), then
        re-check margin at the bar close the way TradingView does.
        Called from script_runner after the user script's main() completes.

        TV evaluates margin at every bar close and books the liquidation on
        that bar at the close price; without this check the same liquidation
        only fires at the next bar's open — one bar late, and at the open
        price on gapped data (Hybrid 2026-05-07 02:00: TV trims 1.0 contract
        at C=80898.0 on the 02:00 bar while the O/H/L walk points all pass
        the margin comparison). Sized like the bar-open check in whole
        contracts; every observed instance trimmed exactly 1.0 contract, so
        the whole-contract choice is untested beyond that.
        """
        prev_count = len(self.new_closed_trades)

        if self._deferred_margin_call is not None:
            check_price, for_short = self._deferred_margin_call
            self._deferred_margin_call = None
            self._check_margin_call(check_price, for_short=for_short, at_open=True)

        if self.open_trades:
            self._check_margin_call(self.c, for_short=self.sign < 0, at_open=True,
                                    whole_contracts=True)

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

    def _resolve_deferred_qty(self, order: Order, fill_price: float) -> None:
        """Finalize a default-sized entry's quantity at its actual fill price.

        TradingView resolves percent_of_equity / cash default sizing when the
        order EXECUTES: the investment target is divided by the real fill
        price, with equity measured at that moment. The placement-time size
        was only the margin-check estimate — a marketable limit filling at the
        open, or a gapped market order, re-sizes here. The reversal flip
        component stays frozen from creation (TV computes the flip quantity at
        order creation time).
        """
        order.deferred_qty = False
        old_abs = abs(order.size)
        qty = _default_entry_qty(float(fill_price))
        if qty <= 0.0:
            order.size = 0.0
            return
        size = _size_round((qty + order.flip_extra) * order.sign)
        if size != 0.0:
            # The big-money sizing judgment applies to the money-sized part of
            # the order only; the reversal flip component is the old position,
            # already an exact lot multiple.
            flip = order.flip_extra * order.sign
            size = _judge_money_entry(size - flip, float(fill_price)) + flip
        order.size = size
        # A default-sized entry that resolves LARGER than its placement estimate
        # would strand a sliver: the bracket's no-qty "rest" leg reserved off the
        # smaller estimate and would under-close the fill. Grow those legs by the
        # extra so they still cover the whole entry, matching TradingView (which
        # sizes the entry at fill and closes all of it). A smaller resolution
        # never strands — the over-reservation is clamped by the FIFO close.
        extra = abs(order.size) - old_abs
        if extra > 0.0 and order.order_id is not None:
            self._grow_rest_exit_legs(order.order_id, extra)

    def _grow_rest_exit_legs(self, entry_id: str, extra: float) -> None:
        """Extend an entry's full-close bracket legs by ``extra`` contracts.

        Only ``rest_leg`` exits (no explicit qty / qty_percent — the "close the
        whole entry" leg) grow; an absolute-qty or qty_percent leg keeps the
        slice it was given. A grown reservation is clamped by the FIFO close to
        the actually open size, so over-reserving is safe.
        """
        for o in self.exit_orders.values():
            if (o.rest_leg and o.order_id == entry_id
                    and not o.consumed and o.book_seq is None and o.size != 0.0):
                grown = _size_round(o.reserved_size + extra)
                o.reserved_size = grown
                o.size = math.copysign(grown, o.size)

    def _cancel_unaffordable_entries(self) -> None:
        """
        Cancel pending price-based entry orders the account can no longer margin.

        TradingView re-evaluates an unfilled entry order's required margin at the
        CURRENT price (the "LastPrice" of its margin formula), cancelling the order
        once the requirement exceeds equity. The sweep runs after the bar's fill
        phases: a marketable order fills at the open before any check can touch it,
        and a resting order gets this bar's fill window first. At 100%
        percent_of_equity sizing this kills every resting buy limit below the
        market (required = equity * price / limit > equity) while a resting sell
        limit above the market survives (required < equity) -- exactly the
        asymmetry TradingView's exported trade lists show.
        """
        if not self.entry_orders:
            return
        script = lib._script
        pv = syminfo.pointvalue
        for order in list(self.entry_orders.values()):
            if order.order_type != _order_type_entry:
                continue
            if order.limit is None and order.stop is None:
                continue
            margin_percent = script.margin_short if order.sign < 0 else script.margin_long
            if margin_percent <= 0:
                continue
            resulting_qty = abs(self.size + order.size)
            margin_needed = resulting_qty * self.c * pv * (margin_percent / 100.0)
            if margin_needed > self.equity:
                self._remove_order(order)

    def _entry_exceeds_margin_after_fill(self, order: Order, fill_price: float) -> bool:
        """
        Check whether an entry's resulting position is affordable at its fill price.

        TV rejects the entry before filling when the position that would remain after
        the fill cannot be margined. Once an entry has filled, later open/high/low
        margin breaches are handled by the margin-call path.
        """
        script = lib._script
        margin_percent = script.margin_short if order.sign < 0 else script.margin_long
        if margin_percent <= 0:
            return False

        pv = syminfo.pointvalue
        margin_ratio = margin_percent / 100.0

        new_qty = abs(self.size + order.size)
        if new_qty == 0.0:
            return False

        equity = self.equity
        margin_needed = new_qty * fill_price * pv * margin_ratio
        # From 1e7 account-currency units of equity upward TV decides the fill
        # with an integer-tick comparison on the PERMISSIVE side: the entry
        # fills while the equity rounded half-up to a tick still covers the
        # truncated tick-count of the required margin — a sub-tick shortfall
        # fills and the bar-open margin-call path then trims the position (a
        # sub-lot shortfall liquidates one whole contract). Measured on
        # BINANCE:BTCUSDT 30m: Hybrid 2025-06-12 22:30 (shortfall 0.0076 USD,
        # 0.76 tick) FILLED + 1-contract MC at the open, while one-shot
        # initial_capital replicas 1.00 and 1.81 ticks short both REJECTED.
        # Below the 1e7 gate the legacy relative tolerance stands (fitted on
        # corpus rejects at 1.75e-9..1.06e-7 relative shortfall).
        mintick = syminfo.mintick
        if equity >= 1e7 and mintick and mintick > 0:
            return math.floor(equity / mintick + 0.5) < math.floor(margin_needed / mintick)
        return margin_needed - equity > abs(equity) * 7.5e-10

    def _cancel_same_bar_reversal_closes(self, entry_order: Order) -> None:
        """
        Cancel market closes made redundant by a same-bar opposite entry.

        A reversing ``strategy.entry`` is itself the close request for the current
        position. If that entry is rejected at its fill, TV does not then fill a
        same-bar ``strategy.close`` for the old position as a fallback.
        """
        if self.size == 0.0 or self.sign == entry_order.sign:
            return

        open_entry_ids = {trade.entry_id for trade in self.open_trades}
        for close_order in list(self.market_orders.values()):
            if close_order.order_type != _order_type_close:
                continue
            if close_order.bar_index != entry_order.bar_index:
                continue
            if close_order.sign != entry_order.sign:
                continue
            if close_order.order_id is None or close_order.order_id in open_entry_ids:
                self._remove_order(close_order)

    def _check_low_stop(self, order: Order) -> bool:
        """ Check low stop """
        if order.stop is None:
            return False
        if self._exit_awaits_entry(order):
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
            if self._exit_awaits_entry(order):
                return False
            # Long limit order (size > 0) triggers when price falls to limit level
            if order.size > 0 and order.limit >= self.l:
                p = min(self.o, order.limit)
                order.filled_by_type = 'profit'
                self.fill_order(order, p, self.h, p)
                return True
        return False

    def _check_close_leg_down(self, order: Order) -> bool:
        """Fill on the closing descent (high -> close) of the intrabar walk.

        Only an order that became active mid-bar can still be pending here — an
        exit whose entry filled on an earlier leg. The segment starts at the
        bar's high, so fills land exactly at the trigger price (no open-gap
        clamp like :meth:`_check_low` applies).
        """
        if self._exit_awaits_entry(order):
            return False
        # Long limit (buy back) triggers when price falls to the limit level
        if order.limit is not None and order.size > 0 and order.limit >= self.c:
            order.filled_by_type = 'profit'
            self.fill_order(order, order.limit, self.h, order.limit)
            return True
        # Sell stop triggers when price falls to the stop level
        if order.stop is not None and order.size < 0 and order.stop >= self.c:
            p = order.stop
            slippage = lib._script.slippage
            if slippage > 0:
                p -= syminfo.mintick * slippage
            order.filled_by_type = 'loss'
            self.fill_order(order, p, self.h, p)
            return True
        return False

    def process_orders(self):
        """ Process orders """
        # We need to round to the nearest tick to get the same results as in TradingView.
        # ``lib.math.round_to_mintick`` is inlined here (this preamble runs every bar):
        # OHLC are always plain floats at this point, so its NA branch is dead code.
        # The expression shape must stay ``int(x / mintick + 0.5) * minmove / pricescale``
        # (left to right) — see the bit-parity note in ``lib/math.py``.
        mintick = syminfo.mintick
        minmove = syminfo.minmove
        pricescale = syminfo.pricescale
        self.o = int(lib.open / mintick + 0.5) * minmove / pricescale
        self.h = int(lib.high / mintick + 0.5) * minmove / pricescale
        self.l = int(lib.low / mintick + 0.5) * minmove / pricescale
        self.c = int(lib.close / mintick + 0.5) * minmove / pricescale

        self.drawdown_summ = self.runup_summ = 0.0
        self.new_closed_trades.clear()

        # Idle fast path: with no open position and no pending orders every phase
        # below is a provable no-op (each loop iterates an empty container, every
        # ``_check_margin_call`` early-returns on ``not open_trades``) except the
        # trading-day rollover and the post-bar risk rules — run just those two.
        if (not self.open_trades and not self.entry_orders and not self.exit_orders
                and not self.market_orders and not self.orderbook.price_levels):
            if self._roll_trading_day():
                return
            if (self.risk_max_drawdown_value is not None
                    or self.risk_max_intraday_loss_value is not None
                    or self.risk_max_cons_loss_days is not None):
                self._enforce_post_bar_risk()
            return

        # If the order is open → high → low → close or open → low → high → close
        ohlc = self.h - self.o < self.o - self.l

        self._process_at_bar_open(ohlc)
        self._process_limit_stop_orders(ohlc)
        self._cancel_unaffordable_entries()
        self._finalize_bar_pnl()
        if (self.risk_max_drawdown_value is not None
                or self.risk_max_intraday_loss_value is not None
                or self.risk_max_cons_loss_days is not None):
            self._enforce_post_bar_risk()
        self._finalize_new_closed_trades()

    def _roll_trading_day(self) -> bool:
        """Roll the intraday risk anchors when the bar enters a new trading day.

        ``time_tradingday`` is session-aware: for overnight sessions (forex,
        futures) the day rolls at the session open (e.g. 17:00 ET), not at
        calendar midnight — matching TradingView's intraday risk reset. For
        24/7 crypto and intraday stock sessions it collapses to the calendar
        day in the exchange timezone, so those symbols are unaffected.

        :return: True when the ``max_cons_loss_days`` halt fired — the caller
            must stop processing the bar's orders.
        """
        current_trading_day = int(lib.time_tradingday())
        if current_trading_day == self.risk_last_trading_day:
            return False
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
            return True
        return False

    def _process_at_bar_open(self, ohlc: bool):
        """Phase 1: Process orders at bar open — gap detection, market fills, margin."""
        if self._roll_trading_day():
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
                    # Exit order gaps through — check if its bound entry still
                    # has open quantity on the ledger (the FIFO fill may have
                    # consumed its trade rows while the binding stays live)
                    has_open_trade = order.order_id in self._entry_open_ledger
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
                self.market_orders[_market_order_key(order)] = order

        # Reversal context for the pre-fill margin reject below. A genuine fresh entry
        # that cannot be margined at its fill price is rejected outright (TV-verified).
        # But the new leg of a reversal — an opposite-direction entry processed after a
        # same-bar close has already flattened the previous position — is NOT rejected:
        # TV fills it and lets the bar-open margin call trim the over-margin excess to a
        # viable remainder. Track the bar-start position sign and whether a same-bar close
        # has filled, so the reject can distinguish the two cases.
        reversal_pre_sign = self.sign
        reversal_close_filled = False

        # Process Market orders
        for order in list(self.market_orders.values()):
            if order.cancelled:
                continue
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
                        if order.deferred_qty:
                            order.flip_extra = abs(self.size)
                    if order.size == 0.0:
                        # Closing-leg-only reversal marker whose opposite position
                        # is already gone: nothing left to close.
                        self._remove_order(order)
                        continue

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
                # Settle a default-sized order's quantity at its fill price first,
                # so the margin check judges the real fill, not the estimate
                if order.deferred_qty:
                    self._resolve_deferred_qty(order, fill_price)
                    if order.size == 0.0:
                        self._remove_order(order)
                        continue
                if self._entry_exceeds_margin_after_fill(order, fill_price):
                    # The reversal's new leg (opposite the bar-start position, with a
                    # same-bar close already filled) is allowed to fill and is trimmed by
                    # the bar-open margin call below; only a fresh entry is hard-rejected.
                    is_reversal_leg = (reversal_close_filled
                                       and reversal_pre_sign != 0.0
                                       and order.sign == -reversal_pre_sign)
                    if not is_reversal_leg:
                        self._cancel_same_bar_reversal_closes(order)
                        self._remove_order(order)
                        continue

            # open → high → low → close
            if ohlc:
                self.fill_order(order, fill_price, self.o, self.l)
            # open → low → high → close
            else:
                self.fill_order(order, fill_price, self.l, self.o)

            # A same-bar close that reduced the bar-start position arms the reversal-leg
            # bypass for a subsequent opposite over-margin entry on this bar.
            if order.order_type == _order_type_close and reversal_pre_sign != 0.0:
                reversal_close_filled = True

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
                # Skip exits whose bound entry still has open quantity on the
                # ledger (they belong to the current position)
                if order.order_id in self._entry_open_ledger:
                    continue
                # Skip exits whose entry is still pending
                if order.order_id in self.entry_orders:
                    continue
                # Only a from_entry-less exit adapts to the surviving position
                # (TV keeps such an exit alive across a rejected entry). A leg
                # bound to an explicit from_entry can only ever close trades
                # from that entry — when the entry is gone it stays dormant.
                if not order.from_entry_na:
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
            if order.order_id not in self._entry_open_ledger:
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

        # Margin call check at OPEN — TV liquidates whole contracts here even on
        # fractional-lot symbols (verified against a TV strategy export where an
        # entry overshooting margin filled and was immediately trimmed by exactly
        # 1.0 contract at the fill price). The sign gates mirror the callee's own
        # direction guards (a liquidation never reverses the position, so the
        # second direction stays a no-op after the first fires).
        if self.sign < 0:
            self._check_margin_call(self.o, for_short=True, at_open=True, whole_contracts=True)
        elif self.sign > 0:
            self._check_margin_call(self.o, for_short=False, at_open=True, whole_contracts=True)

    def _process_limit_stop_orders(self, ohlc: bool):
        """Phase 2: Process limit/stop/trailing orders with margin checks at H/L."""
        # The order-book walks are gated on ``price_levels`` at each walk site
        # (re-checked, not hoisted — margin fills and trailing stops mutate the
        # book between walks); an empty book makes every walk yield nothing, so
        # skipping the generator is exactly behaviour-preserving. The margin
        # checks are gated on the position sign, mirroring the callee's own
        # direction guards — a mismatched direction is a guaranteed ``False``.
        # Trailing stops walk the assumed intrabar path themselves (arming,
        # water-mark ratchet and fill in chronological order), so they are
        # processed here rather than inside the level-indexed walk — but only
        # up to the second extreme. A fill on the walk's closing leg happens
        # chronologically AFTER the intrabar margin-call checkpoints at the
        # extremes, so orders still pending after the first two legs are
        # collected and resumed at the closing-leg site below; walking them
        # to completion here would flatten the position before a margin call
        # TV fires at the adverse extreme (verified against a TV export where
        # a partial 'Margin call' at the high preceded the trailing exit
        # filling near the low of the same bar).
        # Iterate a snapshot since fills mutate the order book; an order indexed at
        # several price levels is yielded once per level, so dedupe by identity.
        trail_close_leg: list[Order] = []
        if self.orderbook.price_levels:
            seen: set[Order] = set()
            for order in list(self.orderbook.iter_orders()):
                if order in seen or order.cancelled or order.trail_price is None:
                    continue
                seen.add(order)
                if self._process_trailing_stop(order, ohlc) == _trail_pending:
                    trail_close_leg.append(order)

        # Process orders: open → high → low → close
        if ohlc:
            # open -> high
            if self.orderbook.price_levels:
                for order in self.orderbook.iter_orders(min_price=self.o, max_price=self.h):
                    if self._check_high_stop(order):
                        continue
                    if self._check_high(order):
                        continue

            mc_deferred = self.sign < 0 and self._check_margin_call(self.h, for_short=True)
            if not mc_deferred:
                # The checkpoint at the position's FAVORABLE extreme runs
                # before this leg's fills. Under the float trigger it is a
                # no-op (available funds only improve toward the favorable
                # side at margin <= 100%), but the >=1e7 integer-tick trigger
                # can trip there: TV liquidated one contract of a LONG at
                # H=120300 (Hybrid 2025-10-02 16:00) before the exit limit at
                # 120290.7 — lower on the same leg — filled the rest.
                if self.sign < 0:
                    self._check_margin_call(self.l, for_short=True, can_defer=False)

                # open -> low (descending: the level nearest the open fills first)
                if self.orderbook.price_levels:
                    for order in self.orderbook.iter_orders(max_price=self.o, min_price=self.l, desc=True):
                        if self._check_low_stop(order):
                            continue
                        if self._check_low(order):
                            continue

                if self.sign > 0:
                    self._check_margin_call(self.l, for_short=False, can_defer=False)

            # Trailing fills on the closing leg — chronologically after both
            # margin-call checkpoints, so a partial liquidation at the extreme
            # trims the position the trailing exit then closes. A deferred
            # margin call stops the level walks but not the trail: its fill
            # precedes the close-price liquidation.
            for order in trail_close_leg:
                if order.cancelled or order.filled_by_type is not None:
                    continue
                self._process_trailing_stop(order, ohlc, close_leg=True)

            if not mc_deferred:
                # low -> close (ascending): the walk's closing leg. Orders that
                # became active mid-bar — an exit whose entry filled on an
                # earlier leg — get the path's final segment, like TV does.
                if self.orderbook.price_levels:
                    for order in self.orderbook.iter_orders(min_price=self.l, max_price=self.c):
                        if self._check_close_leg_up(order):
                            continue

        # Process orders: open → low → high → close
        else:
            # open -> low (descending: the level nearest the open fills first)
            if self.orderbook.price_levels:
                for order in self.orderbook.iter_orders(max_price=self.o, min_price=self.l, desc=True):
                    if self._check_low_stop(order):
                        continue
                    if self._check_low(order):
                        continue

            mc_deferred = self.sign > 0 and self._check_margin_call(self.l, for_short=False)
            if not mc_deferred:
                # Favorable-extreme checkpoint before this leg's fills — see
                # the mirrored comment in the OHLC branch (TV-verified on the
                # Hybrid 2025-10-02 16:00 long margin call at the high).
                if self.sign > 0:
                    self._check_margin_call(self.h, for_short=False, can_defer=False)

                # open -> high
                if self.orderbook.price_levels:
                    for order in self.orderbook.iter_orders(min_price=self.o, max_price=self.h):
                        if self._check_high_stop(order):
                            continue
                        if self._check_high(order):
                            continue

                if self.sign < 0:
                    self._check_margin_call(self.h, for_short=True, can_defer=False)

            # Trailing fills on the closing leg — chronologically after both
            # margin-call checkpoints, so a partial liquidation at the extreme
            # trims the position the trailing exit then closes. A deferred
            # margin call stops the level walks but not the trail: its fill
            # precedes the close-price liquidation.
            for order in trail_close_leg:
                if order.cancelled or order.filled_by_type is not None:
                    continue
                self._process_trailing_stop(order, ohlc, close_leg=True)

            if not mc_deferred:
                # high -> close (descending): the walk's closing leg. Orders that
                # became active mid-bar — an exit whose entry filled on an
                # earlier leg — get the path's final segment, like TV does.
                if self.orderbook.price_levels:
                    for order in self.orderbook.iter_orders(max_price=self.h, min_price=self.c, desc=True):
                        if self._check_close_leg_down(order):
                            continue

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
                if self._exit_awaits_entry(order):
                    return
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
                if self._entry_exceeds_margin_after_fill(order, fill_price):
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
        # ``lib.math.round_to_mintick`` inlined — sub-bar OHLC are plain floats, and
        # this runs per sub-bar. Expression shape must stay left-to-right (see the
        # bit-parity note in ``lib/math.py``).
        mintick = syminfo.mintick
        minmove = syminfo.minmove
        pricescale = syminfo.pricescale
        # Setup from first sub-bar (= chart bar open)
        first = sub_bars[0]
        self.o = int(first.open / mintick + 0.5) * minmove / pricescale
        self.h = int(first.high / mintick + 0.5) * minmove / pricescale
        self.l = int(first.low / mintick + 0.5) * minmove / pricescale
        # Use aggregated close for margin deferral checks
        self.c = int(aggregated.close / mintick + 0.5) * minmove / pricescale
        self.drawdown_summ = self.runup_summ = 0.0
        self.new_closed_trades.clear()

        # Phase 1: at-open processing (gap detection, market orders, margin at open)
        ohlc = self.h - self.o < self.o - self.l
        self._process_at_bar_open(ohlc)

        # Phase 2: process limit/stop orders on each sub-bar
        for sub_bar in sub_bars:
            self.o = int(sub_bar.open / mintick + 0.5) * minmove / pricescale
            self.h = int(sub_bar.high / mintick + 0.5) * minmove / pricescale
            self.l = int(sub_bar.low / mintick + 0.5) * minmove / pricescale
            self.c = int(sub_bar.close / mintick + 0.5) * minmove / pricescale
            ohlc = self.h - self.o < self.o - self.l
            self._process_limit_stop_orders(ohlc)

        # Phase 3: P&L update using aggregated bar values
        self.h = int(aggregated.high / mintick + 0.5) * minmove / pricescale
        self.l = int(aggregated.low / mintick + 0.5) * minmove / pricescale
        self.c = int(aggregated.close / mintick + 0.5) * minmove / pricescale
        self._finalize_bar_pnl()
        if (self.risk_max_drawdown_value is not None
                or self.risk_max_intraday_loss_value is not None
                or self.risk_max_cons_loss_days is not None):
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

    # TV closes only the part of the position opened by entries with this id.
    # Under the default FIFO close_entries_rule the FILL may consume older
    # trades first, but the amount closed is still the bound entry's open size
    # — sizing off the whole position would flatten unrelated entries.
    if isinstance(position, SimPosition):
        # noinspection PyProtectedMember
        bound_size = position.sign * position._entry_open_ledger.get(id, 0.0)
    else:
        bound_size = 0.0
        for trade in position.open_trades:
            if trade.entry_id == id:
                bound_size += trade.size

    if isinstance(qty, NA):
        if not isinstance(qty_percent, NA):
            size = _size_round(-bound_size * (qty_percent * 0.01))
        else:
            size = -bound_size
    else:
        size = _size_round(-position.sign * min(qty, abs(bound_size)))

    if size == 0.0:
        return

    exit_id = f"Close entry(s) order {id}"
    order = Order(id, size, exit_id=exit_id, order_type=_order_type_close,
                  comment=None if isinstance(comment, NA) else comment,
                  alert_message=None if isinstance(alert_message, NA) else alert_message)

    # Stamp a unique book_seq so several same-bar partial closes on this entry
    # stack instead of colliding on a shared exit-order key. Backtest only —
    # the live broker close-dispatch path is handled separately and stays None.
    if isinstance(position, SimPosition):
        order.book_seq = position._next_close_seq()

    # Add order to position (this will handle orderbook and exit_orders)
    position._add_order(order)
    # Same-tick fill is a backtest concept; in broker mode the order is already
    # enqueued by ``_add_order`` and the sync engine forwards it to the exchange.
    if immediately and isinstance(position, SimPosition):
        closed_before = len(position.new_closed_trades)
        position.fill_order(order, position.c, position.h, position.l)
        # The bar's regular settle already ran in process_orders(); a same-tick
        # close lands in new_closed_trades after it, and the next bar's clear()
        # would drop it with cum_profit never booked — settle it now, the same
        # mirror the close pass uses
        position._settle_close_pass_trades(closed_before)


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

    # Stamp book_seq so a close_all stacked behind a same-bar partial close fills
    # too (backtest only; live close-dispatch handled separately, stays None).
    if isinstance(position, SimPosition):
        order.book_seq = position._next_close_seq()

    # Add order to position (this will handle orderbook and exit_orders)
    position._add_order(order)
    # Same-tick fill is a backtest concept; in broker mode the order is already
    # enqueued by ``_add_order`` and the sync engine forwards it to the exchange.
    if immediately and isinstance(position, SimPosition):
        closed_before = len(position.new_closed_trades)
        position.fill_order(order, position.c, position.h, position.l)
        # The bar's regular settle already ran in process_orders(); a same-tick
        # close lands in new_closed_trades after it, and the next bar's clear()
        # would drop it with cum_profit never booked — settle it now, the same
        # mirror the close pass uses
        position._settle_close_pass_trades(closed_before)


# noinspection PyProtectedMember
def _default_entry_budget(price: float) -> tuple[float, float] | None:
    """Money amount and per-unit cost of a default-sized entry at ``price``.

    Returns ``(money, unit_cost)`` so that the raw quantity is
    ``money / unit_cost``, or None for fixed sizing (not money-based).
    """
    script = lib._script
    default_qty_type = script.default_qty_type
    if default_qty_type == fixed:
        return None

    if default_qty_type == percent_of_equity:
        target_investment = script.position.equity * script.default_qty_value * 0.01
        if script.commission_type == _commission.percent:
            commission_multiplier = 1.0 + script.commission_value * 0.01
            return target_investment, price * syminfo.pointvalue * commission_multiplier
        if script.commission_type == _commission.cash_per_contract:
            return target_investment, price * syminfo.pointvalue + script.commission_value
        if script.commission_type == _commission.cash_per_order:
            return (max(0.0, target_investment - script.commission_value),
                    price * syminfo.pointvalue)
        # No commission
        return target_investment, price * syminfo.pointvalue

    if default_qty_type == cash:
        return script.default_qty_value, price * syminfo.pointvalue

    raise ValueError("Unknown default qty type: ", default_qty_type)


# noinspection PyProtectedMember
def _default_entry_qty(price: float) -> float:
    """Contracts a default-sized (no explicit ``qty``) entry buys at ``price``.

    TradingView calculates the position size so that the total investment
    (position value + commission) equals the specified percentage of equity:

    - percent commission: ``total_cost = qty * price * (1 + commission_rate)``
    - cash per contract: ``total_cost = qty * price + qty * commission_value``

    We want ``total_cost = equity * percent``, so
    ``qty = (equity * percent) / (price * (1 + commission_factor))``.

    The price-based types (percent_of_equity, cash) resolve when the order
    EXECUTES — the caller passes the actual fill price at fill time, and only
    an executable-price estimate at placement (for margin checks).
    """
    budget = _default_entry_budget(price)
    if budget is None:
        return lib._script.default_qty_value
    money, unit_cost = budget
    return money / unit_cost


# Distance threshold (in ticks) of the big-money gate's down-step: an
# inflated threshold landing on an even grid multiple steps down one grid
# unit only when it cleared the inflated cost by more than this. Bracketed
# in (0.0783, 0.1034) ticks on TV probes; 3/32 is the binary-exact candidate.
_GATE_DOWN_STEP_DELTA = 0.09375


def _ceil_to_grid(value: float, grid: float) -> tuple[int, float]:
    """Exact smallest multiple of ``grid`` that is >= ``value``.

    ``value / grid`` alone can round across an integer near a grid point; the
    correction loops re-check with ``k * grid`` products, which are exact for
    the tick grids (0.5, 5) and magnitudes (< 2^53) involved.

    :param value: The value to quantize upward
    :param grid: The grid step
    :return: ``(k, k * grid)`` where ``k * grid`` is the quantized value
    """
    k = math.ceil(value / grid)
    while (k - 1) * grid >= value:
        k -= 1
    while k * grid < value:
        k += 1
    return k, k * grid


def _price_has_odd_f32_offset(price: float) -> bool:
    """Whether ``price`` sits an odd number of float32-ULP/25 quanta above
    its float32 lower neighbour, within seven quanta.

    TV's big-money gate inflates its cost threshold only on bars whose close
    has this float32 relationship (measured 38/38 on BINANCE:BTCUSDT 30m; in
    the [2^16, 2^17) binade the quantum is 1/32 tick). A close exactly
    representable in float32 (offset 0) does not inflate.

    :param price: The bar close driving the gate
    :return: True when the odd-offset relationship holds
    """
    if price <= 0.0 or not math.isfinite(price):
        return False
    f32 = struct.unpack('<f', struct.pack('<f', price))[0]
    bits = struct.unpack('<I', struct.pack('<f', f32))[0]
    if f32 > price:
        bits -= 1
        f32 = struct.unpack('<f', struct.pack('<I', bits))[0]
    ulp = struct.unpack('<f', struct.pack('<I', bits + 1))[0] - f32
    if ulp <= 0.0 or not math.isfinite(ulp):
        return False
    quanta = (price - f32) * 25.0 / ulp
    k = round(quanta)
    return k % 2 == 1 and k <= 7 and abs(quanta - k) < 0.25


def _gate_entry_lots(equity_ticks: float, lots: int, rfactor: float,
                     unit_cost: float, mintick: float, price: float) -> int | None:
    """Judge an entry of ``lots`` lots against TV's big-money margin gate.

    From 1e9 cost ticks upward TV quantizes the order cost onto a tick grid
    (0.5 tick, 5 ticks from 1e10 cost ticks) and compares the raw equity tick
    count against the quantized threshold:

    - equity >= threshold: the entry fills as sized;
    - equity below threshold but at least the plain grid ceiling of the cost
      (possible only when the threshold was inflated): the entry is rejected;
    - equity below the plain grid ceiling: the parity of the grid multiple
      decides — even rejects, odd fills one lot less.

    On odd-float32-offset bars (see :func:`_price_has_odd_f32_offset`) with
    price >= 1e5 the threshold is the grid ceiling of the cost inflated by
    2^-31 relative; an inflated threshold landing on an EVEN grid multiple
    steps one grid unit down when it cleared the inflated cost by more than
    ``_GATE_DOWN_STEP_DELTA`` (never below the plain ceiling, and not when
    the cost sits exactly on the grid). Reverse-engineered on BINANCE:BTCUSDT
    30m one-shot probes: 19,613 of 19,614 measurements reproduced, boundary
    decade 21/22 (below C 1e5 rare inflated bars exist whose slope selector
    is unmapped; they are treated as uninflated here).

    :param equity_ticks: Raw equity tick count (equity / mintick)
    :param lots: Entry size in lot units
    :param rfactor: Lots per contract (``syminfo._size_round_factor``)
    :param unit_cost: Account-currency cost of one contract
    :param mintick: Tick size
    :param price: The bar close driving the gate (inflation selector)
    :return: Granted lot count (``lots`` or ``lots - 1``) or None when the
        entry is rejected
    """
    cost = lots / rfactor * unit_cost / mintick
    grid = 5.0 if cost >= 1e10 else 0.5
    k0, m0 = _ceil_to_grid(cost, grid)
    m_eff = m0
    if price >= 1e5 and _price_has_odd_f32_offset(price):
        inflated = cost * (1.0 + 2.0 ** -31)
        k_eff, m_eff = _ceil_to_grid(inflated, grid)
        if k_eff % 2 == 0 and m_eff - inflated > _GATE_DOWN_STEP_DELTA:
            down = m_eff - grid
            if not (down == m0 == cost):
                m_eff = max(m0, down)
    if equity_ticks >= m_eff:
        return lots
    if equity_ticks >= m0:
        return None
    if k0 % 2 == 0:
        return None
    return lots - 1


# noinspection PyProtectedMember
def _judge_money_entry(size: float, price: float) -> float:
    """Apply TV's big-money sizing and margin gate to a money-sized entry.

    From 1e7 account-currency units of order money upward (equivalently 1e9
    ticks at mintick 0.01; the gate is bracketed in (9.0e6, 1.01e7] and is
    indistinguishable between the two at mintick 0.01) TV re-judges the
    floor-sized quantity: when the truncated money tick count reaches one
    grid unit below the NEXT lot's quantized cost, the gate is evaluated at
    that larger size (which its own cost then always exceeds, so the outcome
    is the parity branch: reject or fill the floor size); otherwise the gate
    runs at the floor size directly. See :func:`_gate_entry_lots` for the
    gate itself and the measurement provenance.

    :param size: Signed floor-sized quantity in contracts
    :param price: The sizing/gate price (placement close for market entries,
        fill price for price-based orders resolving at execution)
    :return: The granted signed quantity, or 0.0 when the entry is rejected
    """
    budget = _default_entry_budget(price)
    if budget is None:
        return size
    money, unit_cost = budget
    if money < 1e7:
        return size
    mintick = syminfo.mintick
    if not mintick or mintick <= 0:
        return size
    rfactor = syminfo._size_round_factor  # noqa
    lots = round(abs(size) * rfactor)
    if lots <= 0:
        return size
    money_ticks = money / mintick
    next_cost = (lots + 1) / rfactor * unit_cost / mintick
    next_grid = 5.0 if next_cost >= 1e10 else 0.5
    _, next_m0 = _ceil_to_grid(next_cost, next_grid)
    if math.floor(money_ticks) >= next_m0 - next_grid:
        lots += 1
    granted = _gate_entry_lots(money_ticks, lots, rfactor, unit_cost, mintick, price)
    if granted is None or granted <= 0:
        return 0.0
    sign = 1.0 if size > 0 else -1.0
    return sign * granted / rfactor


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

    # We need a signed size instead of qty, the sign is the direction
    direction_sign: float = (-1.0 if direction == short else 1.0)

    if isinstance(limit, NA):
        limit = None
    elif limit is not None:
        # We need negative direction for entry limit orders - NOTE: it is tested
        limit = _price_round(limit, -direction_sign)
    if isinstance(stop, NA):
        stop = None
    elif stop is not None:
        stop = _price_round(stop, direction_sign)

    # A default-sized order (no explicit qty) resolves its quantity at the
    # actual fill price (TradingView sizes percent_of_equity / cash when the
    # order executes). The size computed here is the placement estimate used
    # for the margin check and order bookkeeping: taken at the price the order
    # would execute at NOW — the current price when immediately executable,
    # the limit/stop price while it rests.
    deferred_default = isinstance(qty, NA)
    market_sizing_price: float | None = None
    if deferred_default:
        exec_price = position.c
        if limit is not None:
            exec_price = min(limit, exec_price) if direction_sign > 0 else max(limit, exec_price)
        elif stop is not None:
            exec_price = max(stop, exec_price) if direction_sign > 0 else min(stop, exec_price)
        else:
            market_sizing_price = float(exec_price)
        qty = _default_entry_qty(exec_price)

    # qty must be greater than 0
    if qty <= 0.0:
        return

    size = qty * direction_sign

    size = _size_round(size)
    if size == 0.0:
        return

    # Market entries keep their placement-close sizing (price-based orders
    # re-resolve at fill), so the big-money sizing gate is judged here.
    if market_sizing_price is not None:
        size = _judge_money_entry(float(size), market_sizing_price)
        if size == 0.0:
            if position.size == 0.0 or position.sign == direction_sign:
                return
            # A nofill judgment does not cancel a reversal outright: TV keeps
            # the order alive as its closing leg, so the opposite position
            # still closes at the next open while the opening leg stays
            # suppressed (Hybrid 2026-05-14 15:00: the short closes at the
            # bar open, the long only fills a bar later from the re-issued,
            # re-judged entry). The zero size nets to a pure close through
            # the reversal flip at order processing.
            order = Order(id, 0.0, order_type=_order_type_entry, oca_name=oca_name,
                          oca_type=oca_type, comment=comment, alert_message=alert_message)
            order.sign = direction_sign
            position._add_order(order)
            return

    # Creation-time margin check for entry orders (TradingView backtest behavior).
    # TV cancels an entry order it cannot open: required margin is evaluated at
    # the CURRENT price (the "LastPrice" of its margin formula), with the order
    # sized at the price it would execute at now. A resting buy limit below the
    # market at 100% percent_of_equity sizing therefore never opens (required =
    # equity * price / limit > equity), while a resting sell limit above the
    # market and any immediately executable order fit within equity.
    # Skip in broker mode: the exchange enforces margin authoritatively, and the script's
    # equity view can drift from the exchange (funding, fees, transfers) — making the
    # local check a source of silent false positives rather than a safety net.
    if isinstance(position, SimPosition):
        margin_percent = (script.margin_short if direction_sign < 0
                          else script.margin_long)
        if margin_percent > 0:
            margin_ratio = margin_percent / 100.0
            if limit is None and stop is None:
                slippage_amount = script.slippage * syminfo.mintick
                check_price = position.c + slippage_amount * direction_sign
            else:
                check_price = position.c
            equity = script.initial_capital + position.netprofit + position.openprofit
            # Margin/equity are in account currency — convert via pointvalue.
            margin_needed = abs(size) * check_price * syminfo.pointvalue * margin_ratio
            # From 1e7 account-currency units of equity upward TV runs this
            # creation-time check as the quantized big-money gate (see
            # _gate_entry_lots): the order is cancelled unless the equity tick
            # count reaches the grid threshold of the required margin. A
            # money-sized market entry already passed _judge_money_entry, and
            # its granted cost always clears this equity-side gate at 100%
            # margin; explicit-qty and resting orders are judged here (the
            # placement estimate — price-based orders re-size at fill).
            # Measured on BINANCE:BTCUSDT 30m: Hybrid 2025-08-25 05:00
            # (equity 18.58M, surplus 0.19 tick) was rejected on TV and
            # refilled one bar later; MAB corpus entries at 1.06M/1.32M
            # equity fill despite the same tick geometry, bracketing the gate
            # below 1e7 together with the sizing-law gate in (9.0e6, 1.01e7].
            mintick = syminfo.mintick
            if equity >= 1e7 and mintick and mintick > 0:
                rfactor = syminfo._size_round_factor  # noqa
                lots = round(abs(size) * rfactor)
                unit_margin = check_price * syminfo.pointvalue * margin_ratio
                if lots > 0:
                    granted = _gate_entry_lots(equity / mintick, lots, rfactor,
                                               unit_margin, mintick, check_price)
                    if granted != lots:
                        return
            elif margin_needed > equity:
                return

    # If it is not a market order, we should check pyramiding and flip conditions here
    # Market orders are checked at the order processing time
    flip_extra = 0.0
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
            flip_extra = abs(position.size)

    order = Order(id, size, order_type=_order_type_entry, limit=limit, stop=stop, oca_name=oca_name,
                  oca_type=oca_type, comment=comment, alert_message=alert_message)
    # Only price-based orders re-size at execution; a market entry keeps its
    # placement-time (signal close) quantity — TV rejects it at the next open
    # when that quantity can no longer be margined, rather than re-sizing.
    if deferred_default and (limit is not None or stop is not None):
        order.deferred_qty = True
        order.flip_extra = flip_extra
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
    def _exit():
        nonlocal limit, stop, trail_price, from_entry, direction, size, oca_name, oca_type

        # Sticky bracket (TV semantics): a leg is identified by (id, from_entry).
        # Re-issuing it every bar updates its prices, but a leg that already fired
        # its slice must not be resurrected (the ``consumed`` tombstone). The
        # reservation is recomputed from ``init_size`` on every issue: that is the
        # ORIGINAL size of everything bound to ``from_entry`` — open pyramid adds
        # at their entry size plus a still-pending entry order at its CURRENT
        # size — so a pyramid add grows the slice, margin-call shrinkage does not
        # erode it, and a pending entry re-sized bar-to-bar keeps being tracked
        # (locking the first bar's size would under-close the eventual fill and
        # strand a sliver).
        exit_key = (id, from_entry)
        existing = position.exit_orders.get(exit_key)
        if existing is not None and existing.consumed:
            return

        is_rest_leg = isinstance(qty, NA) and isinstance(qty_percent, NA)
        if not isinstance(qty, NA):
            reserved = abs(qty)
        elif not isinstance(qty_percent, NA):
            reserved = abs(init_size) * (qty_percent * 0.01)
        else:
            # No-qty "rest" leg: the entry size minus the slices reserved by
            # sibling legs (consumed siblings keep their reservation until the
            # entry fully closes), so it never over-closes the position.
            # Only sticky exit legs (book_seq is None) count as siblings; a
            # stacked strategy.close()/close_all() partial (book_seq set) is an
            # immediate market close, not a reservation against this rest leg.
            sibling = sum(o.reserved_size for o in position.exit_orders.values()
                          if o.order_id == from_entry and o is not existing
                          and o.book_seq is None)
            reserved = abs(init_size) - sibling

        reserved = _size_round(reserved)
        if reserved <= 0.0:
            return
        size = -direction * reserved

        # Store tick values for later calculation when entry price is known
        profit_ticks: float | None = _na_to_none(profit)
        loss_ticks: float | None = _na_to_none(loss)
        trail_points_ticks: float | None = _na_to_none(trail_points)
        # TradingView truncates a fractional ``trail_offset`` tick count to
        # whole ticks (like its qty precision). Verified against a TV
        # reference (BINANCE:BTCUSDT 30m, ``trail_points=trail_offset=
        # atr*mult``): TV's trailing fills land at ``water mark -/+
        # floor(offset_ticks) * mintick``, while fractional ticks would round
        # half the fills one tick further. ``trail_points`` stays fractional:
        # the activation price resolves with directional tick-rounding
        # (bracket trail probe 91, ``trail_points=atr``, matches TV that way).
        _trail_offset = _na_to_none(trail_offset)
        if _trail_offset is not None:
            _trail_offset = float(int(_trail_offset))
        _trail_price = _na_to_none(trail_price)

        # A missing ``trail_offset`` does NOT disable the trailing leg. TradingView's
        # compile rule only requires the offset when the trailing pair is the
        # exit's SOLE trigger; alongside ``stop``/``limit`` the call compiles, and the
        # TV reference exports (pynecomp bracket trail probes 88-91) prove the trailing
        # stop arms with an offset of 0 ticks. The offset-0 default is applied at
        # ``Order`` construction.

        # An exit must arm at least one trigger. TradingView treats a call whose
        # price/tick args ALL resolve to na as a no-op -- e.g. brackets computed
        # from a flat position_avg_price (na) on a bar before the entry fills --
        # not a level-less market close that fires at the next open.
        if (isinstance(limit, NA) and isinstance(stop, NA) and isinstance(profit, NA)
                and isinstance(loss, NA) and _trail_price is None
                and trail_points_ticks is None):
            return

        _limit = _na_to_none(limit)
        if _limit is not None:
            _limit = _price_round(_limit, direction)
        _stop = _na_to_none(stop)
        if _stop is not None:
            _stop = _price_round(_stop, -direction)
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
            trail_price=_trail_price, trail_offset=_trail_offset,
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
        # activated high/low-water mark ONLY when the trailing parameters are
        # unchanged. TradingView carries ONE logical trailing stop across
        # identical re-issues -- a fresh Order must inherit the ratcheted
        # ``trail_stop`` instead of re-arming at the bare activation level every
        # bar, which would leave the stop permanently one or more bars behind
        # the carried water mark. A re-issue with CHANGED trailing parameters
        # (a per-bar recomputed atr-based trail, a stricter activation rebased
        # on a pyramid add, ...) is a cancel+replace: the armed state and the
        # carried water mark are dropped and the replaced leg re-arms from the
        # issue bar's CLOSE tick (see ``_seed_trail_at_issue``); the prior
        # bars' extremes stay out of its water mark. Verified against a TV
        # reference (BINANCE:BTCUSDT 30m, per-bar ``trail_points=atr*mult``):
        # TV's re-armed stop anchored to the issue bar's close instead of
        # carrying the prior high-water mark. The activation is compared in
        # the form it was given -- ``existing.trail_price`` may hold a
        # points-resolved value, so the entry-anchored ``trail_points`` form
        # compares tick counts.
        had_trail = False
        trail_unchanged = False
        if existing is not None and (
                existing.trail_price is not None or existing.trail_points_ticks is not None):
            had_trail = True
            trail_unchanged = (
                existing.trail_offset == order.trail_offset
                and ((order.trail_points_ticks is not None
                      and existing.trail_points_ticks == order.trail_points_ticks)
                     or (order.trail_points_ticks is None
                         and existing.trail_points_ticks is None
                         and existing.trail_price == order.trail_price)))
            if trail_unchanged and existing.trail_triggered:
                order.trail_triggered = True
                order.trail_stop = existing.trail_stop

        order.rest_leg = is_rest_leg
        position._add_order(order)
        # A brand-new trailing leg (first issue, or trailing added to a live
        # bracket) and an identical re-issue fold the issue bar's extreme into
        # the water mark; a changed-params re-issue re-arms anchored to the
        # issue bar's close only (see above).
        position._seed_trail_at_issue(order, fold_extreme=not had_trail or trail_unchanged)

    def _bound_size(entry_id: str) -> tuple[float, float]:
        """Combined sign and ORIGINAL size of everything bound to an entry id:
        open pyramid adds at their entry size plus a still-pending entry order at
        its current size. TradingView's exit covers each of them, so the leg is
        reserved off the combined size and the FIFO fill allocation then closes
        the bound trades the way TV's per-entry exit brackets do."""
        sign = 0.0
        total = 0.0
        pending = position.entry_orders.get(entry_id)
        if pending is not None:
            sign = pending.sign
            total += abs(pending.size)
        for open_trade in position.open_trades:
            if open_trade.entry_id == entry_id:
                sign = open_trade.sign
                total += abs(open_trade.init_size)
        return sign, total

    # Find direction and size
    if from_entry:
        direction, init_size = _bound_size(from_entry)
        # The position should be open, or an entry order should exist
        if not direction:
            return
        _exit()

    else:
        # If still no entry order found, we should exit all open trades and open orders
        if not direction:
            for order in list(position.entry_orders.values()):
                from_entry = order.order_id or ""
                direction, init_size = _bound_size(from_entry)
                # Only mark as from_entry_na on first creation (not replacement)
                exit_key = (id, from_entry)
                had_existing_exit = exit_key in position.exit_orders
                _exit()
                if not had_existing_exit:
                    exit_order = position.exit_orders.get(exit_key)
                    if exit_order is not None:
                        exit_order.from_entry_na = True

            if not direction:
                seen_ids: set[str] = set()
                for trade in position.open_trades:
                    from_entry = trade.entry_id or ""
                    if from_entry in seen_ids:
                        continue
                    seen_ids.add(from_entry)
                    direction, init_size = _bound_size(from_entry)
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

    # We need a signed size instead of qty, the sign is the direction
    direction_sign: float = (-1.0 if direction == short else 1.0)

    if isinstance(limit, NA):
        limit = None
    elif limit is not None:
        limit = _price_round(limit, direction_sign)  # TODO: test this if the direction here is correct
    if isinstance(stop, NA):
        stop = None
    elif stop is not None:
        stop = _price_round(stop, -direction_sign)  # TODO: test this if the direction here is correct

    # A default-sized order resolves its quantity at the actual fill price
    # (TradingView sizes percent_of_equity / cash when the order executes).
    # The size computed here is the placement estimate, taken at the price the
    # order would execute at NOW — the current price when immediately
    # executable, the limit/stop price while it rests.
    deferred_default = isinstance(qty, NA)
    market_sizing_price: float | None = None
    if deferred_default:
        exec_price = float(lib.close)
        if limit is not None:
            exec_price = min(limit, exec_price) if direction_sign > 0 else max(limit, exec_price)
        elif stop is not None:
            exec_price = max(stop, exec_price) if direction_sign > 0 else min(stop, exec_price)
        else:
            market_sizing_price = exec_price
        qty = _default_entry_qty(exec_price)

    # qty must be greater than 0
    if qty <= 0.0:
        return

    size = qty * direction_sign

    # NOTE: Unlike strategy.entry, strategy.order is NOT affected by pyramiding limit
    # This is a key difference - strategy.order can open unlimited trades in the same direction
    # It uses _order_type_normal to distinguish it from entry/exit orders

    size = _size_round(size)
    if size == 0.0:
        return

    # Market orders keep their placement-close sizing (price-based orders
    # re-resolve at fill), so the big-money sizing gate is judged here.
    if market_sizing_price is not None:
        size = _judge_money_entry(float(size), market_sizing_price)
        if size == 0.0:
            return

    # Create the order with _order_type_normal
    # This is a "normal" order that simply adds to or subtracts from position
    # It doesn't follow entry/exit rules and can freely modify positions
    order = Order(id, size, order_type=_order_type_normal, limit=limit, stop=stop,
                  oca_name=oca_name, oca_type=oca_type, comment=comment,
                  alert_message=alert_message)
    # Only price-based orders re-size at execution (see strategy.entry)
    if deferred_default and (limit is not None or stop is not None):
        order.deferred_qty = True
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
def avg_losing_trade() -> PyneFloat:
    if lib._script is None:
        return 0.0
    position = lib._script.position
    if position.losstrades == 0:
        return NA(float)
    return position.grossloss / position.losstrades


# noinspection PyProtectedMember
@module_property
def avg_trade() -> PyneFloat:
    if lib._script is None:
        return 0.0
    position = lib._script.position
    if position.closed_trades_count == 0:
        return NA(float)
    return position.netprofit / position.closed_trades_count


# noinspection PyProtectedMember
@module_property
def avg_winning_trade() -> PyneFloat:
    if lib._script is None:
        return 0.0
    position = lib._script.position
    if position.wintrades == 0:
        return NA(float)
    return position.grossprofit / position.wintrades


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
def max_drawdown_percent() -> PyneFloat:
    if lib._script is None:
        return 0.0
    initial = lib._script.initial_capital
    if initial == 0.0:
        return 0.0
    return lib._script.position.max_drawdown / initial * 100.0


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
def netprofit_percent() -> PyneFloat:
    if lib._script is None:
        return 0.0
    initial = lib._script.initial_capital
    if initial == 0.0:
        return 0.0
    return lib._script.position.netprofit / initial * 100.0


# noinspection PyProtectedMember
@module_property
def openprofit() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.openprofit


# noinspection PyProtectedMember
@module_property
def openprofit_percent() -> PyneFloat:
    if lib._script is None:
        return 0.0
    initial = lib._script.initial_capital
    if initial == 0.0:
        return 0.0
    return lib._script.position.openprofit / initial * 100.0


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
