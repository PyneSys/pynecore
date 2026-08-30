from typing import TYPE_CHECKING, Literal, overload
from typing import TypeAlias as _TypeAlias  # underscore-aliased: kept out of the module-property registry

import logging as _logging  # underscore-aliased: kept out of the module-property registry
import sys as _sys  # underscore-aliased: kept out of the module-property registry
import math
import struct
from abc import ABC, abstractmethod
from datetime import datetime, UTC
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_HALF_UP
from collections import deque, defaultdict
from copy import copy
from bisect import insort, bisect_left

from ...core.module_property import module_property
from ... import lib
from .. import request, syminfo

from ...types.strategy import QtyType, ADOPTED_STARTUP_ENTRY_ID
from ...types.base import IntEnum
from ...types.na import NA, na_float, na_str
from ...types import PyneFloat, PyneInt, PyneStr

from . import direction as direction
from . import commission as _commission
from . import oca as _oca

from . import closedtrades, opentrades

# Deliberately not the Pine ``log.*`` stream (``pyne_core_logger``): that one carries
# script output and is compared against TradingView logs.
_logger = _logging.getLogger(__name__)

__all__ = [
    "fixed", "cash", "percent_of_equity",
    "long", "short", 'direction',

    'Trade', 'Order', 'PositionBase', 'SimPosition',
    "cancel", "cancel_all", "close", "close_all", "convert_to_account", "convert_to_symbol",
    "default_entry_qty", "entry", "exit", "order",

    "closedtrades", "opentrades",
]

#
# Function-and-namespace modules — the IDE-facing rebinding; at runtime the AST
# transformer routes bare reads and calls to the module's self-named function
#

from ...types.ohlcv import OHLCV

if TYPE_CHECKING:
    from ...core import script as _core_script
    from .closedtrades import closedtrades
    from .opentrades import opentrades
    # Static-only public aliases: at runtime the submodule import above already
    # sets these attributes on the package; the underscore aliases keep them out
    # of the module-property registry.
    from . import commission as commission, oca as oca


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

# Order-book dict key shapes. A sticky ``strategy.exit`` leg carries the
# ``entry_seq`` of the single filled entry it is bound to, so two pyramid adds
# sharing a ``from_entry`` id get one leg each. A close placed by
# ``strategy.close()`` / ``strategy.close_all()`` in BACKTEST carries a unique
# call-site stamp in ``book_seq`` so that same-bar partial closes on one entry
# STACK per statement instead of colliding on a shared key. Both elements are None
# for the orders that use neither, which keeps their dedup-by-id semantics intact.
_ExitOrderKey: _TypeAlias = (tuple[str | None, str | None]
                             | tuple[str | None, str | None, int]
                             | tuple[str | None, str | None, int | None, int])
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
def _na_to_none(value: PyneFloat | NA[float]) -> float | None: ...


@overload
def _na_to_none(value: PyneStr | NA[str]) -> str | None: ...


def _na_to_none(value):  # type: ignore[misc]
    """Convert na (NA object or native nan float) to None, pass through everything else."""
    if not (value == value):  # is_na_arg
        return None
    return value


# Call sites of ``strategy.close()`` / ``strategy.close_all()``, numbered in
# first-seen order. The code object is part of the key, so the dict keeps it
# alive and an id can never be recycled onto a different statement.
_close_call_sites: dict[tuple[object, int], int] = {}


def _close_call_site(frame) -> int:
    """Identify the ``strategy.close()``/``close_all()`` STATEMENT that is running.

    MEASURED — TradingView gives every close statement its own order-book slot:
    two statements placed on one bar both fill, whether they sit side by side,
    live in one function body, or in two different functions. But ONE statement
    executed twice on a bar (the same helper called from two sites, or a loop)
    only MODIFIES its pending order — the last call wins, exactly like a repeated
    ``strategy.entry`` under one id. The statement is the call instruction inside
    the frame that reached us, so the code object plus its offset is the identity;
    the caller's own call site is deliberately NOT part of it.

    :param frame: The calling frame, i.e. ``sys._getframe(1)`` at the call site.
    :return: A small stable integer for this statement, for ``Order.book_seq``.
    """
    key = (frame.f_code, frame.f_lasti)
    seq = _close_call_sites.get(key)
    if seq is None:
        seq = len(_close_call_sites) + 1
        _close_call_sites[key] = seq
    return seq


def _exit_key(exit_id: str | None, order_id: str | None,
              entry_seq: int | None = None, book_seq: int | None = None) -> '_ExitOrderKey':
    """Build an exit/close order-book key. THE single construction rule.

    Trailing Nones are dropped, so an order that uses neither discriminator keeps
    the bare ``(exit_id, order_id)`` key and its dedup-by-id semantics.

    ``entry_seq`` is the individual filled entry a sticky ``strategy.exit`` leg is
    bound to: TradingView issues ONE leg per entry, not per ``from_entry`` id, so
    two pyramid adds sharing an id each get their own leg and each is consumed on
    its own. It stays None for a leg still waiting on a pending entry order, for
    every id-bound close, and on the live broker path, which has no binding book.

    ``book_seq`` (see :func:`_close_call_site`) is the call site of a backtest
    partial close, so same-bar closes from different statements get distinct keys
    and STACK while a repeat of one statement evicts its own earlier order.
    """
    if book_seq is not None:
        return exit_id, order_id, entry_seq, book_seq
    if entry_seq is not None:
        return exit_id, order_id, entry_seq
    return exit_id, order_id


def _exit_order_key(order_: 'Order') -> '_ExitOrderKey':
    """Order-book key of an exit/close order (see :func:`_exit_key`).

    Insert and pop sites MUST both route through this helper so they never drift.
    """
    return _exit_key(order_.exit_id, order_.order_id, order_.entry_seq, order_.book_seq)


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
        "ticks_resolved",  # Tick offsets already turned into levels off the entry fill price
        "is_market_order",  # Flag to check if this is a market order
        "cancelled",  # Flag to mark order as cancelled by OCA
        "gap_committed",  # Exit leg locked into the current bar-open gap batch
        "deferred_qty",  # Default-sized entry: quantity re-resolves at the actual fill price
        "budget_money",  # Money budget of a default-sized entry frozen at (last) placement
        "budget_pv",  # Account point value (quote->account rate) frozen with it
        "filled_qty",  # Live: quantity of this entry order already reflected in open_trades
        "flip_extra",  # Reversal flip magnitude frozen at creation (added back on deferred re-size)
        "skip_flip",  # Entry re-placed on the same bar: it keeps its raw qty, no flip augmentation
        "bar_index",  # Bar index when the order was placed
        "filled_by_type",  # Type of execution: 'profit', 'loss', 'trailing', or None
        "from_entry_na",  # True if exit was created without explicit from_entry (applies to any position)
        "reserved_size",  # Exit-leg slice of the entry's original size (frozen at creation)
        "bound_size",  # Size of everything bound to the entry when this leg reserved its slice
        "rest_leg",  # Exit leg with no explicit qty/qty_percent: closes the WHOLE bound entry
        "consumed",  # True once an exit leg fired its slice while its entry is still open
        "entry_seq",  # The single filled entry this sticky exit leg is bound to (_EntryBinding)
        "book_seq",  # Call site of the strategy.close()/close_all() statement that placed this
                     # (backtest only); None for non-stacking sticky-exit / risk / live orders
        "comm_booking",  # Commission pool shared by the two legs of a reversal (see _fill_order)
        "reversal_leg",  # Closing leg an entry order was split into when it flipped the position
        "placed_fill_seq",  # Fills processed before this order was booked (see _fill_order)
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

        self.reversal_leg = False
        self.placed_fill_seq = -1

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
        self.ticks_resolved = False

        # Check if this is a market order (no limit, stop, trail, or tick-based prices)
        self.is_market_order = (self.limit is None and self.stop is None
                                and self.trail_price is None
                                and self.profit_ticks is None
                                and self.loss_ticks is None
                                and self.trail_points_ticks is None)

        self.cancelled = False
        # True while this exit leg sits in the bar-open gap batch. MEASURED on
        # TradingView (BINANCE:BTCUSDT 30m, 6/6 events): when a stop entry that
        # reverses the position and a strategy.exit stop BOTH gap through the
        # same open, both fill there -- the exit is not cancelled by the reversal
        # that filled a moment earlier. It sells its own quantity a second time
        # and opens a fresh position under its own exit id. Only the gap batch
        # behaves this way: an exit level first reached inside the bar (18/18
        # events) and one outlived by a MARKET reversal (6/6) are cancelled unfilled.
        self.gap_committed = False
        self.deferred_qty = False
        self.budget_money: float | None = None
        self.budget_pv: float | None = None
        # Live-only fill accounting: how much of this retained entry order has
        # already been recorded as an open trade. The simulator removes a
        # market entry order on fill, so it stays 0.0 there; the live broker
        # keeps the entry Order in ``entry_orders`` for intent stability, so the
        # bound-size reservation must not double-count the filled slice.
        self.filled_qty: float = 0.0
        self.flip_extra = 0.0
        self.skip_flip = False
        self.bar_index = -1  # Will be set when order is added to position
        self.filled_by_type: Literal['profit', 'loss', 'trailing'] | None = None  # Will be set when order fills
        self.from_entry_na = False
        self.reserved_size = abs(size)
        self.bound_size = 0.0
        self.rest_leg = False
        self.consumed = False
        # The single filled entry a sticky strategy.exit leg is bound to (see
        # _EntryBinding); None while the leg still waits on a pending entry order
        # and for every id-bound close.
        self.entry_seq: int | None = None
        # Stamped only by strategy.close()/close_all() in backtest (see _close_call_site);
        # left None everywhere else so the order-book key keeps its bare shape.
        self.book_seq: int | None = None
        # ``[Decimal qty_total, booked, leg_qtys, order_qty]`` commission pool, set only
        # where one TradingView order is executed as two PyneCore fills (the
        # reversal split in _process_order).
        self.comm_booking: list | None = None

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
        "size", "init_size", "sign", "entry_id", "entry_bar_index", "entry_time", "entry_price", "entry_comment",
        "entry_equity", "entry_fill_node", "entry_fill_seq",
        "exit_id", "exit_bar_index", "exit_time", "exit_price", "exit_comment", "exit_equity",
        "commission", "max_drawdown", "max_drawdown_percent", "max_runup", "max_runup_percent",
        "profit", "profit_percent", "cum_profit", "cum_profit_percent",
        "cum_max_drawdown", "cum_max_runup"
    )

    # noinspection PyShadowingNames
    def __init__(self, *, size: PyneFloat, entry_id: str | None, entry_bar_index: int, entry_time: int,
                 entry_price: PyneFloat,
                 commission: PyneFloat, entry_comment: PyneStr | None = None,
                 entry_equity: PyneFloat = 0.0, entry_fill_node: int = -1,
                 entry_fill_seq: int = -1):
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
        # Which point of the intrabar walk this trade's entry filled at. Two fills
        # sharing a bar AND a node happened at the same moment, which is what the
        # same-instant reversal booking below keys on.
        self.entry_fill_node: int = entry_fill_node
        self.entry_fill_seq: int = entry_fill_seq

        self.exit_id: str | None = ""
        self.exit_bar_index: int = -1
        self.exit_time: int = -1
        self.exit_price: PyneFloat = 0.0
        self.exit_comment: PyneStr = ''
        self.exit_equity: PyneFloat = na_float

        self.commission: PyneFloat = commission

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


class _EntryBinding:
    """One filled entry, in the book the pyramiding limit counts.

    TradingView keeps TWO views of an open position and they drift apart. The
    trade rows -- and with them ``strategy.opentrades``, ``opentrades.size()``
    and ``position_avg_price`` -- settle FIFO across the whole position, oldest
    entry first, no matter which entry the filling order belonged to. Every
    exit/close order, meanwhile, settles against the entry it was BOUND to, and
    an entry keeps its pyramiding slot until its OWN bound quantity is spent --
    even after the FIFO view already consumed its rows under another entry's
    fill.

    Measured on BINANCE:BTCUSDT 30m (359 cycles per variant): with A(1u, id 'A')
    and B(3u, id 'B') open under ``pyramiding=2``, shedding exactly 1 unit lands
    the reported book on ``opentrades=1`` / ``position_size=3`` whichever way it
    is shed -- yet a third entry is REJECTED when the unit went through an order
    bound to B (``strategy.exit('X','B',qty=1)`` or ``strategy.close('B',qty=1)``,
    which FIFO-consume A's row) and ACCEPTED when it went through an order bound
    to A, or through an unbound reduction (``strategy.order``) that settles both
    books FIFO.

    ``init_size`` is frozen at the fill: a qty_percent exit leg reserves off the
    entry's ORIGINAL quantity, not off what is left of it.
    """

    __slots__ = ('seq', 'entry_id', 'init_size', 'bound', 'sign', 'entry_price', 'exit_opened')

    def __init__(self, seq: int, entry_id: str | None, size: PyneFloat, entry_price: PyneFloat,
                 exit_opened: bool = False):
        self.seq: int = seq
        self.entry_id: str | None = entry_id
        self.init_size: float = abs(size)
        self.bound: float = abs(size)
        self.sign: float = 0.0 if size == 0.0 else 1.0 if size > 0.0 else -1.0
        self.entry_price: PyneFloat = entry_price
        # Opened by a gap-committed exit leg rather than by an entry order, which
        # moves it to the FRONT of the leg order a later strategy.exit issues
        # (see _filled_targets).
        self.exit_opened: bool = exit_opened

    def __repr__(self):
        return (f"_EntryBinding(seq={self.seq}; entry_id={self.entry_id}; "
                f"init_size={self.init_size}; bound={self.bound})")


# noinspection PyShadowingNames,DuplicatedCode
class PriceOrderBook:
    """
    Price-based sorted order storage.
    An order can appear multiple times at different prices.
    """

    __slots__ = ('price_levels', 'orders_at_price', 'order_prices')

    def __init__(self):
        self.price_levels: list[float] = []  # Sorted list of prices
        # Plain dict, NOT defaultdict: a stray read must never auto-create an
        # empty bucket. ``price_levels`` (what the intrabar walk iterates) and
        # the keys of ``orders_at_price`` must stay in lock-step; an orphan
        # empty key would make ``add_order`` skip registering a level, silently
        # dropping that leg from the walk. Reads use ``.get(price, ())``.
        self.orders_at_price: dict[float, list[Order]] = {}  # price -> [Order]
        self.order_prices: defaultdict[Order, set[float]] = defaultdict(set)  # Order -> {prices}

    def _index_price(self, order: Order, price: float, existing: set) -> None:
        """Register ``order`` at ``price`` in both the level list and the bucket.

        The level-list insertion is gated on ``price_levels`` itself (the
        structure the walk reads), not on ``orders_at_price``, so the two can
        never desync into a dropped level.
        """
        if price in existing:
            return
        if price not in self.price_levels:
            insort(self.price_levels, price)
        self.orders_at_price.setdefault(price, []).append(order)
        existing.add(price)

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
        if order.stop is not None:
            self._index_price(order, order.stop, existing)
        if order.limit is not None:
            self._index_price(order, order.limit, existing)
        if order.trail_price is not None:
            self._index_price(order, order.trail_price, existing)

    def remove_order(self, order: Order):
        """Remove order from all price levels"""
        for price in list(self.order_prices[order]):
            bucket = self.orders_at_price.get(price)
            if bucket is not None:
                if order in bucket:
                    bucket.remove(order)
                if not bucket:
                    idx = bisect_left(self.price_levels, price)
                    if idx < len(self.price_levels) and self.price_levels[idx] == price:
                        del self.price_levels[idx]
                    del self.orders_at_price[price]
        del self.order_prices[order]

    def _range_levels(self, desc: bool, min_price: float | None, max_price: float | None) -> list[float]:
        """Price levels a walk covers, in walk order.

        Always a copy of the level list: fills remove levels from the book while
        the walk is in flight.
        """
        if min_price is not None and max_price is not None:
            # Range query - ascending from min to max (or descending when desc=True,
            # e.g. the open->low price walk, where the level nearest the open is
            # reached first in time).
            min_idx = bisect_left(self.price_levels, min_price)
            max_idx = bisect_left(self.price_levels, max_price)
            # Include max_price if it matches exactly
            if max_idx < len(self.price_levels) and self.price_levels[max_idx] == max_price:
                max_idx += 1
            levels = self.price_levels[min_idx:max_idx]
            if desc:
                levels.reverse()
            return levels

        if min_price is not None:
            # Ascending from min_price
            min_idx = bisect_left(self.price_levels, min_price)
            return self.price_levels[min_idx:]

        if max_price is not None:
            # Descending from max_price
            max_idx = bisect_left(self.price_levels, max_price)
            # Include max_price if it matches exactly
            if max_idx < len(self.price_levels) and self.price_levels[max_idx] == max_price:
                max_idx += 1
            levels = self.price_levels[:max_idx]
            levels.reverse()
            return levels

        levels = list(self.price_levels)
        if desc:
            levels.reverse()
        return levels

    def iter_orders(self, *, desc=False, min_price: float | None = None, max_price: float | None = None):
        """
        Iterate over orders within price range.

        Examples:
            iter_orders()  # All orders, ascending
            iter_orders(desc=True)  # All orders, descending
            iter_orders(min_price=50.0)  # 50, 51, 52, ... (ascending)
            iter_orders(max_price=60.0)  # 60, 59, 58, ... (descending)
            iter_orders(min_price=50.0, max_price=60.0)  # 50, 51, ..., 60 (ascending)

        Within a level the insertion order is preserved, so same-price ties keep
        their sequence.

        :param desc: If True, iterate in descending order, only if no min_price or max_price is set
        :param min_price: If set, iterate from this price upward (ascending)
        :param max_price: If set, iterate from this price downward (descending)
        :return: Generator yielding Order objects
        """
        for p in self._range_levels(desc, min_price, max_price):
            # Create a copy to avoid iteration issues when orders are removed during iteration
            yield from list(self.orders_at_price.get(p, ()))

    def iter_levels(self, *, desc=False, min_price: float | None = None, max_price: float | None = None):
        """Walk the same range as :meth:`iter_orders`, one price level at a time.

        The intrabar price walk needs the level it currently stands on: a fill
        can materialize new levels (a tick-based bracket whose entry just
        filled), and the walk has to resume at that level to keep the new ones
        in chronological order.

        :param desc: If True, iterate in descending order, only if no min_price or max_price is set
        :param min_price: If set, iterate from this price upward (ascending)
        :param max_price: If set, iterate from this price downward (descending)
        :return: Generator yielding ``(price, orders_at_that_price)`` pairs
        """
        for p in self._range_levels(desc, min_price, max_price):
            # Create a copy to avoid iteration issues when orders are removed during iteration
            yield p, list(self.orders_at_price.get(p, ()))

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
    The ``strategy.*`` API surface — ``strategy.position_size``,
    ``strategy.opentrades``, ``strategy.netprofit``, ``strategy.equity``,
    etc. — reads the attributes declared here, so concrete subclasses MUST
    initialize all of them in ``__init__``.
    """
    # ``_release_intraday_halt`` writes the halt pair from here, so the base owns
    # those two slots; everything else below is declared for documentation and
    # type-checking only and slotted by the concrete subclasses.
    __slots__ = ('risk_halt_trading', 'risk_halt_day')

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
    max_drawdown_percent: float
    max_runup: float
    max_runup_percent: float
    # Running peaks of the POSITION size per direction — what
    # ``strategy.max_contracts_held_*`` reports (see there).
    max_contracts_held_long: float
    max_contracts_held_short: float
    # Running sums of the closed trades' own profit RATIOS (profit divided by
    # total entry cost, without the percent scaling), in close order. The
    # ``strategy.avg_*_trade_percent`` means scale by 100 only after dividing —
    # that is where TradingView applies it, and summing pre-scaled percents
    # lands an ULP away (measured, see ``avg_trade_percent``).
    sum_profit_ratio: PyneFloat
    sum_win_profit_ratio: PyneFloat
    sum_loss_profit_ratio: PyneFloat
    open_trades: list['Trade']
    closed_trades: 'deque[Trade]'
    new_closed_trades: list['Trade']
    entry_orders: dict[str | None, 'Order']
    exit_orders: dict['_ExitOrderKey', 'Order']
    risk_halt_trading: bool
    risk_halt_day: int
    def _pyramid_count(self) -> int:
        """Number of entries counted against the ``pyramiding`` limit.

        The live broker knows only the reported FIFO view, so it counts that.
        :class:`SimPosition` overrides this with its binding book, which is what
        TradingView actually gates on (see :class:`_EntryBinding`).
        """
        return len(self.open_trades)

    def begin_evaluation(self) -> None:
        """Hook fired once per script evaluation; overridden in broker mode.

        :class:`~pynecore.core.broker.position.BrokerPosition` uses it to reset
        its per-evaluation close-netting scope so two same-bar ``strategy.close``
        calls net into one live order. The simulator dispatches nothing live and
        needs no reset, so the base implementation is a no-op.
        """

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

    def _is_max_intraday_loss_breached(self, equity: float | None = None) -> bool:
        """``max_intraday_loss`` breach test against the day's opening equity.

        :param equity: Mark-to-market equity to test; defaults to the current
            bar-close equity. The intra-bar walk passes the equity at a path
            price so the rule can fire — and close — at the same node
            TradingView does.
        """
        if self.risk_max_intraday_loss_value is None:
            return False
        anchor = self.risk_intraday_start_equity
        # MEASURED on TradingView: the rule is enforced only while the day's
        # OPENING equity is positive. A strategy whose equity has gone negative
        # keeps every position open through arbitrarily large intraday losses
        # and the rule re-arms on the first day that opens back above zero —
        # measured over 109 trades of a wild strategy plus a 226-trade probe
        # (BINANCE:BTCUSDT), where the split is exact: every trade entered on a
        # positive-anchor day closes on the rule, every trade entered on a
        # negative-anchor day survives until a later day opens positive.
        if anchor <= 0.0:
            return False
        # Per TV docs: percent_of_equity for max_intraday_loss is measured
        # against the start-of-day equity (the same anchor used for the loss
        # delta), so the threshold scales with the day's opening capital
        # rather than the initial-bar capital.
        if self.risk_max_intraday_loss_type == percent_of_equity:
            threshold = anchor * self.risk_max_intraday_loss_value * 0.01
        else:
            threshold = float(self.risk_max_intraday_loss_value)
        if equity is None:
            equity = float(self.equity)
        return anchor - equity >= threshold > 0.0

    def _release_intraday_halt(self, current_trading_day: int) -> None:
        """Lift a ``max_intraday_loss`` halt once its trading day is over.

        ``max_intraday_loss`` prohibits trading only for the REST OF THE DAY
        (TV docs; measured — a wild strategy fires the rule on 98 separate
        days of one run), unlike ``max_drawdown`` / ``max_cons_loss_days``,
        which stop the strategy for good. Those leave :attr:`risk_halt_day`
        at ``-1`` and are never released here.

        :param current_trading_day: The trading day the caller just rolled into.
        """
        if self.risk_halt_day < 0 or current_trading_day == self.risk_halt_day:
            return
        self.risk_halt_day = -1
        self.risk_halt_trading = False

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
        # Tested against the OPPOSITE direction: the third value, ``strategy.direction.all``,
        # is neither ``long`` nor ``short``, so an equality test against the intended side
        # rejects every entry of a strategy that passes it (the default of every
        # direction-filter input).
        if intent_sign > 0:
            return allowed != short
        if intent_sign < 0:
            return allowed != long
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

    def _size_flippable_by_entry(self) -> float:
        """Signed position size a new price-based entry would flip.

        The base implementation is the position itself; :class:`SimPosition`
        nets the market closes already placed on this bar (see its override).
        """
        return self.size

    def _pyramid_sign_for_entry(self) -> float:
        """Direction the pyramiding gate sees when a new entry is placed.

        The base implementation is the position's own sign; :class:`SimPosition`
        applies the market orders already placed on this bar (see its override).
        """
        return self.sign

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

    Covers OHLC-based fill detection, synthetic slippage, margin-call emulation,
    gap-through logic, OCA reduce/cancel handling, trailing-stop tracking, etc.

    Live broker trading uses :class:`BrokerPosition` instead — exchange fills
    override all of the simulator logic below.
    """

    __slots__ = (
        'h', 'l', 'c', 'o',
        'openprofit', 'grossprofit', 'grossloss',
        'entry_orders', 'exit_orders', 'market_orders', 'orderbook',
        'open_trades', 'closed_trades', 'new_closed_trades',
        'closed_trades_count', 'wintrades', 'eventrades', 'losstrades',
        'size', 'sign', 'avg_price', 'cum_profit',
        'min_equity', 'max_realized_equity',
        'drawdown_summ', 'runup_summ', 'max_drawdown', 'max_drawdown_percent',
        'max_runup', 'max_runup_percent',
        'max_contracts_held_long', 'max_contracts_held_short',
        'sum_profit_ratio', 'sum_win_profit_ratio', 'sum_loss_profit_ratio',
        'open_commission',
        'risk_allowed_direction', 'risk_max_cons_loss_days', 'risk_max_cons_loss_days_alert',
        'risk_max_drawdown_value', 'risk_max_drawdown_type', 'risk_max_drawdown_alert',
        'risk_max_intraday_filled_orders', 'risk_max_intraday_filled_orders_alert',
        'risk_max_intraday_loss_value', 'risk_max_intraday_loss_type', 'risk_max_intraday_loss_alert',
        'risk_max_position_size',
        'risk_cons_loss_days', 'risk_last_trading_day', 'risk_last_day_equity',
        'risk_intraday_filled_orders', 'risk_intraday_start_equity',
        '_deferred_margin_call', '_mc_stage2', '_fill_counter', '_last_fill_price', '_partial_close_bar',
        '_entry_book', '_entry_seq', '_deferred_immediate_closes', '_coof_cursor', '_market_fill_price',
        '_walk_node', '_path_node'
    )

    def __init__(self):
        # OHLC values
        self.h: float = 0.0
        self.l: float = 0.0
        self.c: float = 0.0
        self.o: float = 0.0

        # Profit/loss tracking
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
        # The binding book: one live entry per FILL, oldest first. It drives the
        # exit-order lifecycle and the pyramiding limit, while ``open_trades``
        # above is the FIFO view TradingView reports (see _EntryBinding).
        self._entry_book: list[_EntryBinding] = []
        self._entry_seq: int = 0

        # Trade statistics
        self.closed_trades_count: int = 0
        self.wintrades: int = 0
        self.eventrades: int = 0
        self.losstrades: int = 0
        self.size: float = 0.0
        self.sign: float = 0.0
        self.avg_price: PyneFloat = na_float
        self.cum_profit: PyneFloat = 0.0
        self.min_equity: PyneFloat = float("inf")
        self.max_realized_equity: PyneFloat = -float("inf")
        self.drawdown_summ: float = 0.0
        self.runup_summ: float = 0.0
        self.max_drawdown: float = 0.0
        self.max_drawdown_percent: float = 0.0
        self.max_runup: float = 0.0
        self.max_runup_percent: float = 0.0
        self.max_contracts_held_long: float = 0.0
        self.max_contracts_held_short: float = 0.0
        self.sum_profit_ratio: PyneFloat = 0.0
        self.sum_win_profit_ratio: PyneFloat = 0.0
        self.sum_loss_profit_ratio: PyneFloat = 0.0
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
        self.risk_halt_day: int = -1

        # Deferred margin call (mc_size==1 and AF@C<0: fire after script runs)
        self._deferred_margin_call: tuple[float, bool] | None = None
        # Second margin-call stage — armed by a fired liquidation, consumed at the
        # bar's next margin checkpoint (see _margin_call_stage2). Holds
        # (bar_index, deficit, stage-1 check price, for_short, ((qty, entry_price), ...))
        # with the legs snapshotted FIFO BEFORE the stage-1 fill.
        self._mc_stage2: tuple[int, float, float, bool,
                               tuple[tuple[float, float], ...]] | None = None
        self._fill_counter: int = 0
        # Price of the most recent fill — the broker emulator's "current price"
        # for a calc_on_order_fills body run (see _mark_to_last_fill).
        self._last_fill_price: float = 0.0
        # Node of the bar's assumed path a calc_on_order_fills re-execution stands
        # at, -1 outside one. Set by the runner's COOF loop; picks both the price
        # the body is marked at and the one its market orders fill at.
        self._coof_cursor: int = -1
        # Price a market order fills at in this pass — the bar open outside a COOF
        # re-execution, a later point of the bar's assumed path inside one.
        self._market_fill_price: float = 0.0
        # Step of the tick source the walk currently stands at, and the step the
        # most recent fill happened at. On the assumed intrabar path a step is a
        # node of it: 0 the open, 1 the extreme nearest it, 2 the other extreme,
        # 3 the close (see _path_price). Under the bar magnifier the sub-bars are
        # the tick source, so a step is a sub-bar index instead.
        self._walk_node: int = 0
        self._path_node: int = 0
        # Monotonic stamp source for same-bar stacking of partial closes.
        # bar_index of the most recent filled partial strategy.close() (a stamped
        # close with an entry id); lets a same-bar close_all clamp to flat instead
        # of overshooting when the partial already shed part of the position.
        self._partial_close_bar: int = -1
        # FIFO buffer of strategy.close/close_all(immediately=True) orders enqueued
        # during the body; drained by settle_immediate_closes() right after the body
        # so position series stay constant for the rest of the bar (TV semantics).
        self._deferred_immediate_closes: list[Order] = []

    @property
    def netprofit(self) -> PyneFloat:
        """Realized net profit, derived from the gross P&L totals."""
        # TV keeps NO running netprofit accumulator: it derives netprofit from
        # the gross totals, with an open position's entry commissions still
        # pending in ``open_commission`` (the public ``grossloss()`` getter adds
        # them back, so netprofit = grossprofit - grossloss on the published
        # numbers). Measured on BINANCE:SHIBUSDT 1h (npacc probe, 1119 closed
        # trades): the derived form reproduces every plotted netprofit bit-exact
        # (23144/23144), while a running ``+= profit`` accumulator drifts by ULPs
        # after the 6th close and matches only 1211/23144.
        # The subtraction is over the PUBLISHED grossloss -- one subtraction, not
        # two: measured on the wild "Built-in Kelly ratio" (currency=USD on
        # BINANCE:BTCUSDT 30m), ``netprofit == grossprofit - grossloss`` holds
        # bit-exact on all 28782 plotted bars, while peeling the two terms off
        # one at a time misses 10461 of them by an ULP.
        return self.grossprofit - (self.grossloss + self.open_commission)

    def _recalc_avg_price(self):
        """
        Re-derive the average entry price from the open trades.
        """
        # The cost basis is summed over the open trades oldest-first on every
        # fill; the divisor is the running position size, NOT the sum of the
        # leg sizes (the two differ by ULPs once a leg is partially closed).
        # Measured on BINANCE:BTCUSDT 30m (pyramiding 2, TP cascade): this form
        # reproduces position_avg_price on all 19581 in-position bars, while a
        # cost accumulator carried across fills (cost -= closed_price *
        # closed_size) only matches the 14888 where no leg is a partial
        # remainder, and dividing by the summed leg sizes matches 17739.
        size = abs(self.size)
        if size == 0.0:
            self.avg_price = na_float
            return
        summ = 0.0
        for trade in self.open_trades:
            summ += trade.entry_price * abs(trade.size)
        self.avg_price = summ / size

    def _size_flippable_by_entry(self) -> float:
        """Signed position size a new price-based entry would flip.

        A price-based ``strategy.entry`` freezes its reversal augmentation when
        the script places it, and keeps that quantity however the position moves
        before the fill (MEASURED: an order created against a 0.04 long still
        opened 0.03 short after a 50% close, and 0.03 short after the position
        was flattened entirely). But the freeze reads the position as the market
        closes ALREADY PLACED on this bar leave it: with
        ``strategy.close(qty_percent=100)`` called before the entry the flip is
        zero (0.01 short), with the same close called after it the flip is the
        full position (0.05 short) — the two probes differ only in script order.
        """
        size = self.size
        if size == 0.0 or not self.market_orders:
            return size
        pending = 0.0
        for order in self.market_orders.values():
            if order.order_type is _order_type_close and not order.cancelled:
                pending += order.size
        if pending == 0.0:
            return size
        remaining = size + pending
        # A close only ever shrinks the position; an over-close lands flat.
        return remaining if remaining * size > 0.0 else 0.0

    def _pyramid_sign_for_entry(self) -> float:
        """Direction the pyramiding gate sees when a new entry is placed.

        The gate reads the position as the market orders ALREADY PLACED on this
        bar leave it, walked in placement order: a close hands its slot over (a
        ``close_all`` ahead of a same-id re-entry is what lets that entry through
        under ``pyramiding=1``), and a pending reversal entry makes the next
        opposite-direction entry a reversal of its own instead of a pyramid add
        (MEASURED on the "Pivot Extension Strategy" reference, whose long and
        short entries are placed on the same bar, in that order).
        """
        if not self.market_orders:
            return self.sign
        size = self.size
        for order in self.market_orders.values():
            if order.cancelled:
                continue
            if order.order_type is _order_type_close:
                remaining = size + order.size
                # A close only ever shrinks the position; an over-close lands flat.
                size = remaining if remaining * size > 0.0 else 0.0
            elif order.order_type is _order_type_entry:
                # A reversal entry carries only its OPENING size until processing
                # adds the flip, so its own sign is where the position ends up.
                size = size + order.size if size * order.size >= 0.0 else order.size
        return 0.0 if size == 0.0 else 1.0 if size > 0.0 else -1.0

    def _add_order(self, order: Order):
        """ Add an order to the strategy """
        # Set the bar_index when the order is placed
        order.bar_index = int(lib.bar_index)
        # How many fills the run had processed when this order reached the book.
        # It separates an order that waited in the book ALONGSIDE another from one
        # a fill itself produced (a calc_on_order_fills re-execution) -- see the
        # same-instant reversal booking in ``_fill_order``.
        order.placed_fill_seq = self._fill_counter

        # Add market order to market orders dict. Key on exit_id too: two
        # brackets sharing the same from_entry (order_id) would otherwise
        # collide on the same key, so a second gap-through exit would evict
        # the first and only one of them would fill on the gap bar. A stacked
        # partial close additionally keys on book_seq (see _market_order_key).
        if order.is_market_order:
            market_key = _market_order_key(order)
            previous = self.market_orders.get(market_key)
            if previous is not None and previous.bar_index != order.bar_index:
                # The key is reused (same close call site, same entry id) but the
                # order sitting on it is a leftover from an earlier bar — often a
                # zero-size tombstone re-filled as a no-op. Plain assignment would
                # hand the fresh order that stale INSERTION SLOT, and the market
                # book fills in insertion order, so this bar's close would jump
                # ahead of orders the script placed before it. Drop the key first
                # to queue the new order where it was actually placed.
                del self.market_orders[market_key]
            self.market_orders[market_key] = order

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
        # Market orders live in their own dict too, so clearing the books alone
        # would leave a same-bar market entry to fill at the next bar's open.
        for book in (self.entry_orders, self.exit_orders):
            for order in book.values():
                order.cancelled = True
                if order.is_market_order:
                    self.market_orders.pop(_market_order_key(order), None)
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

    def _bind_entry(self, entry_id: str | None, size: PyneFloat,
                    entry_price: PyneFloat, exit_opened: bool = False) -> None:
        """Open a binding for a just-filled entry and hand it any waiting exit legs.

        A leg issued while the entry order was still pending carries no
        ``entry_seq`` yet; it belongs to the fill that order just produced, so it
        is re-keyed onto the new binding. Without the hand-over the script's next
        ``strategy.exit`` call would find the binding uncovered and add a SECOND
        leg beside the pending-bound one, and the pair would over-close.
        """
        self._entry_seq += 1
        binding = _EntryBinding(self._entry_seq, entry_id, size, entry_price, exit_opened)
        self._entry_book.append(binding)
        if entry_id is None:
            return
        for exit_order in list(self.exit_orders.values()):
            if (exit_order.entry_seq is None and exit_order.book_seq is None
                    and exit_order.order_id == entry_id):
                self.exit_orders.pop(_exit_order_key(exit_order), None)
                exit_order.entry_seq = binding.seq
                self.exit_orders[_exit_order_key(exit_order)] = exit_order

    def _binding(self, seq: int | None) -> '_EntryBinding | None':
        """The live binding with this sequence number, or None once it is spent."""
        if seq is None:
            return None
        for binding in self._entry_book:
            if binding.seq == seq:
                return binding
        return None

    def _bound_qty(self, entry_id: str | None) -> float:
        """Open bound quantity across every live binding of an entry id."""
        return sum(b.bound for b in self._entry_book if b.entry_id == entry_id)

    def _has_bound(self, entry_id: str | None) -> bool:
        """True while any binding of this entry id still holds quantity."""
        for binding in self._entry_book:
            if binding.entry_id == entry_id:
                return True
        return False

    def _pyramid_count(self) -> int:
        """Entries counted against ``pyramiding`` — the binding book, not the FIFO one."""
        return len(self._entry_book)

    def _drop_binding(self, binding: '_EntryBinding', carry_pending: bool = False) -> None:
        """Retire a spent binding and cancel the exit legs that were bound to it.

        :param carry_pending: The binding was spent by a closer that is not one of
            its own exit legs, so a leg whose id still has a LONG order waiting to
            fill survives (see the block below). A leg settling its own binding
            fired, and dies with it however the id is re-entered.
        """
        try:
            self._entry_book.remove(binding)
        except ValueError:
            return
        for exit_order in list(self.exit_orders.values()):
            if exit_order.gap_committed:
                continue
            bound_here = exit_order.entry_seq == binding.seq
            if not (bound_here
                    or (exit_order.entry_seq is None
                        and exit_order.order_id == binding.entry_id
                        and not self._has_bound(binding.entry_id))):
                continue
            # A leg whose ``from_entry`` id has a LONG order still waiting to fill
            # carries over to that order's fill instead of dying with the binding
            # ``close_all`` just spent, and is walked on the entry bar itself. It is
            # re-keyed off the retired binding so ``_bind_entry`` hands it to the new
            # fill, and ``_exit_awaits_entry`` keeps it inert until then.
            #
            # The long-only restriction is TradingView's, not a simplification.
            # Measured on `Gap Filling Strategy` (NASDAQ:AAPL 30m, 906 trades), where
            # a session bar flattens with ``close_all`` and re-enters the same id in
            # the same body: prev long -> new long carried 15/15, prev short -> new
            # short carried 0/19. Reproduced on synthetic BINANCE:BTCUSDT 30m probes
            # for both limit legs (long 338/338, short 0/359) and stop legs (long
            # 192/192, short 0/725). A leg belonging to an id that was NOT open
            # before is a fresh leg, not a carried one, and fills on either side.
            # The asymmetry has the shape of a TradingView bookkeeping BUG -- a
            # ``position_size > 0`` guard where ``!= 0`` was meant -- but a bug in
            # the reference engine is still the reference: what TradingView prints
            # is what a backtest has to reproduce.
            pending = self.entry_orders.get(exit_order.order_id) if carry_pending else None
            if pending is not None and pending.sign > 0.0:
                if bound_here:
                    self.exit_orders.pop(_exit_order_key(exit_order), None)
                    exit_order.entry_seq = None
                    self.exit_orders[_exit_order_key(exit_order)] = exit_order
                continue
            self._remove_order(exit_order)

    def _reduce_binding(self, binding: '_EntryBinding', qty: float,
                        carry_pending: bool = False) -> float:
        """Take up to ``qty`` off one binding; return what it could not absorb."""
        take = min(binding.bound, qty)
        binding.bound -= take
        if _size_round(binding.bound) <= 0.0:
            self._drop_binding(binding, carry_pending)
        return qty - take

    def _settle_entry_book(self, order: Order, qty: float) -> None:
        """Settle a closing fill against the entries it was bound to.

        A sticky ``strategy.exit`` leg is bound to ONE fill and settles exactly
        that one. An id-bound ``strategy.close`` settles that id's entries
        oldest-first. Everything unbound (``close_all``, a reversal, a margin
        call, ``strategy.order``) walks the whole book oldest-first, mirroring the
        trade rows. Only when a binding's quantity is exhausted are its remaining
        exit legs cancelled — a bracket survives its own trade rows being consumed
        FIFO by another entry's close, and conversely dies once its entry's
        quantity is spent even while those rows still sit open under other entries.
        """
        if qty <= 0.0:
            return
        binding = self._binding(order.entry_seq)
        if binding is not None:
            self._reduce_binding(binding, qty)
            return
        bound_id = order.order_id if order.order_type == _order_type_close else None
        for candidate in list(self._entry_book):
            if qty <= 0.0:
                break
            if bound_id is not None and candidate.entry_id != bound_id:
                continue
            qty = self._reduce_binding(candidate, qty, carry_pending=True)

    def _fill_order(self, order: Order, price: PyneFloat,
                    counts_as_filled_order: bool = True):
        """
        Fill an order (actually)

        :param order: The order to fill
        :param price: The price to fill at
        :param counts_as_filled_order: Whether this fill increments the
                                       ``max_intraday_filled_orders`` counter.
                                       ``False`` for the open half of a
                                       position-reversing order, whose close
                                       half already counted it once — TV treats
                                       a reversal as a single filled order.
        """
        # Every booked fill sits on the symbol's tick grid (see _tick_snap): the
        # trigger price computed by the intrabar walk is the level, the price the
        # broker records is that level snapped to a tick.
        price = _tick_snap(price)

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
        self._last_fill_price = price
        self._path_node = self._walk_node

        # Save the original order size before any modifications
        filled_size = abs(order.size)

        script = lib._script
        commission_type = script.commission_type
        commission_value = script.commission_value
        # TradingView books ONE commission per order, rounded once — not one per
        # closed leg. Measured on Acrypto - Weighted Strategy (BINANCE:BTCUSDT 30m,
        # currency.USD): its 2025-01-02 17:00 reversal charges
        # round10(0.02058 * 97229.76 * rate * 0.00075) = 1.497574781, where the two
        # separately rounded legs give 1.4975747813 and put netprofit 3e-10 off for
        # the rest of the run. The pool accumulates the leg quantities and books the
        # DIFFERENCE between the rounded running total and what was booked already,
        # so a single-leg order is bit-identical to rounding it on its own. A
        # reversal executes as two fills here but is one TradingView order, so its
        # pool is created at the split and shared by both legs.
        comm_booking = order.comm_booking
        if comm_booking is None:
            comm_booking = [Decimal(0), 0.0, [], 0.0]
        # Account-currency value of a 1.0-point move on 1 contract: the futures point
        # value scaled by this bar's symbol-to-account rate. Every money amount below
        # rides it, so each is booked at the rate of the bar it is booked on.
        pv = _account_point_value()

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

                    # MEASURED (BINANCE:BTCUSDT 30m, five probe shapes) — when two
                    # opposite ENTRY orders WAITED IN THE BOOK TOGETHER and then fill
                    # at the SAME moment, the lots they annihilate are booked LONG
                    # side first, whichever order came first: the long order is the
                    # record's entry and the short one its exit. `strategy.entry`
                    # placed short-then-long, long-then-short, three orders in one bar
                    # and unequal sizes all report the long leg as the entry, and
                    # `strategy.order` behaves the same.
                    # Two things break the tie back into chronological order:
                    # * a different fill MOMENT — a reversal whose legs land on
                    #   different bars or different points of the bar walk keeps its
                    #   order even when both legs price identically (a buy-limit
                    #   sitting exactly on the short's entry price stays a short round
                    #   trip);
                    # * a reversal order the FILL ITSELF produced — under
                    #   `calc_on_order_fills` the re-execution that a short's fill
                    #   runs can place the opposing entry, and TradingView then books
                    #   a SHORT round trip (probe "coof reversal labeling probe": the
                    #   coof-issued leg reports size -1, the same-bar pair +1). Only
                    #   an order already in the book when the trade opened ties.
                    # Prices are equal here by construction, so only the labels and
                    # the direction move — the P&L is invariant.
                    if (order.reversal_leg and trade.size < 0.0
                            and trade.entry_bar_index == closed_trade.exit_bar_index
                            and trade.entry_fill_node == self._walk_node
                            and order.placed_fill_seq < trade.entry_fill_seq
                            and trade.entry_price == price):
                        closed_trade.entry_id, closed_trade.exit_id = \
                            closed_trade.exit_id, closed_trade.entry_id
                        closed_trade.entry_comment, closed_trade.exit_comment = \
                            closed_trade.exit_comment, closed_trade.entry_comment or ''
                        closed_trade.size = -closed_trade.size
                        closed_trade.sign = -closed_trade.sign

                    # Commission summ
                    self.open_commission -= closed_trade.commission

                    # cash_per_order is a flat fee per order: defer realization
                    # until the order is removed so it can be split across all
                    # closed trades it actually filled (see delete block below).
                    # The entry leg's own fee, before the exit leg is added on
                    # top — the profit percent divides by it (see below).
                    entry_commission = closed_trade.commission

                    if commission_type == _commission.cash_per_order:
                        closed_trade_size += abs(size)
                    else:
                        # Calculate exit commission based on commission type
                        if commission_type == _commission.percent:
                            commission = _book_commission(comm_booking, abs(size), price,
                                                          commission_value, True)
                        else:
                            # cash_per_contract: size-proportional, charged per leg
                            commission = _book_commission(comm_booking, abs(size), price,
                                                          commission_value, False)

                        # TV takes the two commission legs off the gross P&L ONE
                        # AT A TIME; folding them into a single sum first lands a
                        # few ULP away once the two legs convert at different
                        # account rates. Measured on the wild "Built-in Kelly
                        # ratio" (currency=USD on BINANCE:BTCUSDT 30m, trade #1
                        # entered 2025-01-01 and closed 2025-01-03, so the legs
                        # carry different daily USDT/USD rates): TV's plotted
                        # grossprofit is 240.37221009390822, which is exactly
                        # ``gross - entry_comm - exit_comm``, while
                        # ``gross - (entry_comm + exit_comm)`` gives
                        # ...825. Trades whose legs share one rate agree either
                        # way, which is why this only surfaces on a converted
                        # account.
                        closed_trade.profit -= closed_trade.commission
                        closed_trade.commission += commission
                        closed_trade.profit -= commission

                    # Profit percent — both profit and the entry cost are in USD.
                    # TradingView divides by the trade's TOTAL ENTRY COST: the
                    # position value PLUS the fee paid to open it, not the bare
                    # position value. Measured on BINANCE:BTCUSDT 30m against
                    # closedtrades.profit_percent in all three commission modes,
                    # bit-exact in each: percent 0.05% (1 BTC entered at
                    # 93761.90000000001 reports 0.03838998826686757, which is
                    # 36.01317999999651 / (93761.90000000001 + 46.88095)),
                    # cash_per_contract 7.5 (denominator +7.5) and
                    # cash_per_order 25 (denominator +25, the whole flat entry
                    # fee). Dividing by the bare value overstates every percent
                    # by the fee ratio — 5e-4 relative at 0.05% commission.
                    entry_cost = abs(closed_trade.size) * closed_trade.entry_price * pv + entry_commission
                    try:
                        # Use closed_trade.profit which includes commission, not pnl which doesn't
                        profit_ratio = closed_trade.profit / entry_cost
                    except ZeroDivisionError:
                        profit_ratio = 0.0
                    closed_trade.profit_percent = profit_ratio * 100.0

                    # Modify sizes
                    self.size = _size_add(self.size, size)
                    # Handle too small sizes because of floating point inaccuracy and rounding
                    position_flat = _size_round(self.size) == 0.0
                    if position_flat:
                        size -= self.size
                        self.size = 0.0
                    self.sign = 0.0 if self.size == 0.0 else 1.0 if self.size > 0.0 else -1.0
                    # Keep the residual open-trade size on the unit grid with
                    # the position: a snapped position with a dirty trade
                    # residue would leave ±1e-18 dust open after the final close
                    # and export a ghost 0-qty trade.
                    trade.size = _size_add(trade.size, size)
                    if position_flat:
                        # `size` already absorbed the position residual above, so the
                        # trade that flattened the position is fully closed. Snap off
                        # float epsilon so it is removed from open_trades instead of
                        # lingering as a ~0-size ghost trade — a stale leg would force
                        # avg_price to NA and poison equity (and every subsequent
                        # percent-of-equity sizing) on later bars.
                        trade.size = 0.0
                    # The unfilled remainder feeds the next FIFO leg's closed-trade
                    # size, so it must stay on the long lot ledger like the
                    # position itself (see ``_size_add``).
                    order.size = _size_add(order.size, -size)

                    # Gross P/L and counters. Every commission mode classifies
                    # win/loss on the after-fee profit: cash_per_order fees are
                    # only apportioned in the delete block below, so its trades
                    # are classified there once their final profit is known.
                    if commission_type != _commission.cash_per_order:
                        self.sum_profit_ratio += profit_ratio
                        if closed_trade.profit == 0.0:
                            self.eventrades += 1
                        elif closed_trade.profit > 0.0:
                            self.wintrades += 1
                            self.grossprofit += closed_trade.profit
                            self.sum_win_profit_ratio += profit_ratio
                        else:
                            self.losstrades += 1
                            self.grossloss -= closed_trade.profit
                            self.sum_loss_profit_ratio += profit_ratio

                    # Average entry price
                    self._recalc_avg_price()
                    if self.size:
                        # Unrealized P&L
                        self.openprofit = self.size * (_tick_snap(self.c) - self.avg_price) * pv
                    else:
                        # If position has just closed
                        self.openprofit = 0.0

                    # Exit equity
                    closed_trade.exit_equity = self.equity

                    # Remove from open trades if it is fully filled
                    if trade.size == 0.0:
                        continue

                new_open_trades.append(trade)

            self.open_trades = new_open_trades

            # Settle the closed quantity against the binding book, which the FIFO
            # trade rows above have no say over (see _settle_entry_book).
            closed_qty = filled_size - abs(order.size)
            if closed_qty > 0.0:
                self._settle_entry_book(order, closed_qty)

            if delete:
                # A partial-exit leg that fired its whole slice while its entry's
                # bound quantity is still open becomes a tombstone: kept in
                # exit_orders (so its reservation still counts against sibling
                # "rest" legs and a per-bar strategy.exit() re-call cannot
                # resurrect it) and only pulled from the order book. It is purged
                # when the entry's bound quantity is exhausted (_drop_binding).
                if (order.order_type == _order_type_close and order.order_id is not None
                        and _size_round(order.size) == 0.0
                        and (self._binding(order.entry_seq) is not None
                             if order.entry_seq is not None
                             else self._bound_qty(order.order_id) > 0.0)):
                    order.consumed = True
                    self.orderbook.remove_order(order)
                else:
                    self._remove_order(order)

                if commission_type == _commission.cash_per_order:
                    # This leg's share of the order's single flat fee: a reversal
                    # is one TradingView order, so its closing leg only carries
                    # the part proportional to the quantity it closed.
                    leg_commission = _book_flat_commission(comm_booking, closed_trade_size,
                                                           commission_value)
                    for trade in new_closed_trades:
                        commission = (leg_commission * abs(trade.size)) / closed_trade_size
                        entry_commission = trade.commission
                        trade.commission += commission
                        # The percent/cash_per_contract path subtracts the trade's
                        # total commission (entry leg carried on the open trade +
                        # exit leg) from its profit before profit_percent is
                        # computed; the deferred cash_per_order split must do the
                        # same, otherwise the trade list reports raw P&L while
                        # netprofit already includes the fees.
                        trade.profit -= trade.commission
                        # Same total-entry-cost denominator as the fill-loop path
                        # above; for a flat per-order fee the entry leg carries
                        # the whole 25 of a 25-per-order setting.
                        entry_cost = abs(trade.size) * trade.entry_price * pv + entry_commission
                        try:
                            profit_ratio = trade.profit / entry_cost
                        except ZeroDivisionError:
                            profit_ratio = 0.0
                        trade.profit_percent = profit_ratio * 100.0
                        # Deferred Gross P/L and counters (skipped in the fill
                        # loop above): classify on the after-fee profit, exactly
                        # like the percent/cash_per_contract path does.
                        self.sum_profit_ratio += profit_ratio
                        if trade.profit == 0.0:
                            self.eventrades += 1
                        elif trade.profit > 0.0:
                            self.wintrades += 1
                            self.grossprofit += trade.profit
                            self.sum_win_profit_ratio += profit_ratio
                        else:
                            self.losstrades += 1
                            self.grossloss -= trade.profit
                            self.sum_loss_profit_ratio += profit_ratio

            self.new_closed_trades.extend(new_closed_trades)

            # close_all overshoot: when deferred MC reduced position, close_all
            # captures original size and overshoots → create opposite position.
            # The leftover must be a real lot: a reversal's closing leg ends on a
            # sub-lot float residue (the position is snapped to the grid before it
            # can be absorbed into the last fill), and opening that as a trade
            # leaves a ~1e-18 ghost holding a pyramiding slot the next entry needs.
            if (order.order_id is None and _size_round(order.size) != 0.0 and
                    order.order_type == _order_type_close):
                entry_id = order.exit_id
                overshoot_trade = Trade(
                    size=order.size,
                    entry_id=entry_id, entry_bar_index=int(lib.bar_index),
                    entry_time=lib._time, entry_price=price,
                    commission=0.0, entry_comment=order.comment,
                    entry_equity=self.equity, entry_fill_node=self._walk_node,
                    entry_fill_seq=self._fill_counter
                )
                self.open_trades.append(overshoot_trade)
                self._bind_entry(entry_id, overshoot_trade.size, price)
                self.size = _size_add(self.size, overshoot_trade.size)
                self.sign = 1.0 if self.size > 0.0 else -1.0 if self.size < 0.0 else 0.0
                if self.size > self.max_contracts_held_long:
                    self.max_contracts_held_long = self.size
                elif -self.size > self.max_contracts_held_short:
                    self.max_contracts_held_short = -self.size
                self._recalc_avg_price()
                self.openprofit = self.size * (_tick_snap(self.c) - self.avg_price) * pv
                # The flip's opening leg re-anchors the run-up at the realized
                # equity: the position was flat the instant before it filled.
                entry_mark = float(lib._script.initial_capital) + self.netprofit
                if entry_mark < self.min_equity:
                    self.min_equity = entry_mark

        # New trade
        elif order.order_type != _order_type_close or order.gap_committed:
            # Calculate commission
            if commission_value:
                if commission_type == _commission.cash_per_order:
                    commission = _book_flat_commission(comm_booking, abs(order.size),
                                                       commission_value)
                elif commission_type == _commission.percent:
                    commission = _book_commission(comm_booking, abs(order.size), price,
                                                  commission_value, True)
                elif commission_type == _commission.cash_per_contract:
                    commission = _book_commission(comm_booking, abs(order.size), price,
                                                  commission_value, False)
                else:  # Should not be here!
                    assert False, 'Wrong commission type: ' + str(commission_type)
            else:
                commission = 0.0

            before_equity = self.equity

            # Realize commission: the entry fee sits in open_commission until the
            # trade closes, and the derived netprofit subtracts it from there.
            self.open_commission += commission

            # Low-water mark behind ``strategy.max_runup``: the equity AT the
            # fill -- the position already held, marked at the fill price, with
            # this entry's fee already taken. Measured on BINANCE:BTCUSDT 30m: a
            # second long leg added at 94802.99 on top of one leg from 94711.84
            # anchors the run-up at 994705.330255, the 91.15 the first leg was
            # ahead by, and not at the realized equity below it.
            entry_mark = float(lib._script.initial_capital) + self.netprofit
            if self.size:
                open_mark = self.size * (price - self.avg_price) * pv
                if open_mark > 0.0:
                    entry_mark += open_mark
            if entry_mark < self.min_equity:
                self.min_equity = entry_mark

            # A close_all overshoot and a gap-committed exit leg that outlived
            # the position it was bound to both open under their own exit id.
            entry_id = (order.exit_id
                        if order.order_type == _order_type_close or order.order_id is None
                        else order.order_id)

            trade = Trade(
                size=order.size,
                entry_id=entry_id, entry_bar_index=int(lib.bar_index),
                entry_time=lib._time, entry_price=price,
                commission=commission, entry_comment=order.comment,
                entry_equity=before_equity, entry_fill_node=self._walk_node,
                entry_fill_seq=self._fill_counter
            )

            self.open_trades.append(trade)
            self._bind_entry(entry_id, order.size, price, order.gap_committed)
            self.size = _size_add(self.size, trade.size)
            self.sign = 0.0 if self.size == 0.0 else 1.0 if self.size > 0.0 else -1.0
            # ``max_contracts_held_*`` peaks on the POSITION, not on a single
            # trade: measured on BINANCE:BTCUSDT 30m with pyramiding 3, two long
            # legs of 1 and 2 report 3 (not 2) and two short legs of 4 and 3
            # report 7 (not 4), on all 28837 bars. Only a fill can raise it, so
            # the reduce path never needs to look.
            if self.size > self.max_contracts_held_long:
                self.max_contracts_held_long = self.size
            elif -self.size > self.max_contracts_held_short:
                self.max_contracts_held_short = -self.size

            # Average entry price (see _recalc_avg_price). Measured on
            # BINANCE:BTCUSDT 30m (pyramiding 3, 22720 in-position bars): an
            # incremental re-weighting of the previous average -- algebraically
            # the same -- only reproduces the 13440 single-leg bars.
            self._recalc_avg_price()
            # Unrealized P&L
            self.openprofit = self.size * (_tick_snap(self.c) - self.avg_price) * pv

            # Remove order
            self._remove_order(order)

        # If position has just closed
        if not self.open_trades:
            # Reset position variables
            self.avg_price = na_float
            self.openprofit = 0.0
            self.open_commission = 0.0
            self._entry_book.clear()

            # Cancel all exit orders when position is closed (TradingView behavior)
            # Skip exits that have a pending entry (needed during position flips)
            exit_orders_to_remove = list(self.exit_orders.values())
            for exit_order in exit_orders_to_remove:
                if exit_order.order_id in self.entry_orders:
                    continue
                # A leg already committed to this bar-open gap batch survives the
                # flat moment between a reversal's two legs (see gap_committed).
                if exit_order.gap_committed:
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

    def fill_order(self, order: Order, price: float) -> bool:
        """
        Fill an order

        :param order: The order to fill
        :param price: The price to fill at
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
                        if lib._script.pyramiding <= self._pyramid_count():
                            # Pyramiding limit reached - don't fill the entry order
                            self._remove_order(order)
                            return False

        # For normal orders (_order_type_normal), no special risk management or pyramiding limits apply
        # They simply add to or subtract from the position as requested

        # If position direction is about to change, we split it into two separate orders
        # This is necessary to create a new average entry price
        # Note: The flip quantity is already calculated in entry() for entry orders
        # A reversal order carries the old position as its flip component, so what
        # is left after the closing leg is the entry quantity itself -- an exact lot
        # multiple. Deriving it by subtraction loses an ULP (-0.01616 + 0.03206 =
        # 0.015899999999999997), and `strategy.close(qty = position_size/100*rate)`
        # decimal-floors that one lot low for the whole TP cascade that follows.
        # The tolerant floor returns the exact lot value, so this is a re-derivation,
        # not a correction.
        new_size = _size_round(self.size + order.size)
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
                self._fill_order(order, price)
                return False

            # Check if new direction is allowed by risk management
            # According to Pine Script docs: "long exit trades will be made instead of reverse trades"
            new_direction_sign = 1.0 if new_size > 0.0 else -1.0
            direction_allowed = self._is_direction_allowed(new_direction_sign)

            # Both legs are one TradingView order, so they share one commission
            # booking and its single rounding (see _book_commission). The whole
            # order quantity goes in too: a flat cash_per_order fee is size
            # independent, so its closing leg can only size its share against it.
            # When risk management suppresses the opening leg the order executes
            # as a plain close, so the closing quantity is the whole order and it
            # carries the entire flat fee.
            order_qty = _size_add(abs(self.size), abs(new_size) if direction_allowed else 0.0)
            order.comm_booking = [Decimal(0), 0.0, [], order_qty]

            # Create a copy for closing existing position
            order1 = copy(order)
            order1.order_type = _order_type_close
            order1.reversal_leg = True
            order1.size = -self.size
            # Set order_id to None so it will close any open trades
            order1.order_id = None
            # The exit_id will be the order_id of the original order
            order1.exit_id = order.order_id
            # Fill the closing order first
            self._fill_order(order1, price)

            if not direction_allowed:
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
            self._fill_order(order, price, counts_as_filled_order=False)
            order.comm_booking = None
            # A reversal that hits the cap is flattened too — same as the
            # non-flip path. Without this the cap-close never fires for a
            # position-reversing strategy, the common TradingView idiom.
            if self._is_intraday_filled_cap_reached() and self.size != 0.0:
                self._close_position_at_intraday_cap(order, price)
            return True

        # If position direction is not about to change, we can fill the order directly
        else:
            self._fill_order(order, price)

            # After filling, close the position if this fill hit the intraday cap
            # (TradingView flattens for the rest of the day; the counter blocks
            # new entries until it resets next day).
            if self._is_intraday_filled_cap_reached() and self.size != 0.0:
                self._close_position_at_intraday_cap(order, price)

            return False

    def _max_drawdown_reference(self) -> float:
        """Peak equity for ``max_drawdown``: the realized (closed-equity) high-water mark.

        TradingView's ``strategy.risk.max_drawdown`` measures the drop from the
        largest value among the initial capital and the equity recorded at each
        closed trade — a *realized* peak that open-position paper profit never
        lifts. ``max_realized_equity`` tracks exactly that; before the first bar
        priming it, the initial capital is the reference.
        """
        initial = float(lib._script.initial_capital)
        if self.max_realized_equity == -float("inf"):
            return initial
        return max(initial, float(self.max_realized_equity))

    def _is_max_drawdown_breached(self) -> bool:
        """Sim ``max_drawdown`` breach test: realized-peak reference, mark-to-market drop.

        The drawdown is ``realized_peak - mark_to_market_equity`` — the numerator
        includes the open position's floating loss, but the peak is realized-only
        (see :meth:`_max_drawdown_reference`). Enforcement is gated on close fills
        in :meth:`_enforce_post_bar_risk`, so this predicate only runs on a bar
        that booked realized P&L. Measured against TV on ``BINANCE:BTCUSDT`` 30m:
        a single long carried to a 50% floating drawdown never halts (no close
        order), while the same position force-closes the instant any reducing
        order fills while the drop is past the threshold.
        """
        if self.risk_max_drawdown_value is None:
            return False
        peak = self._max_drawdown_reference()
        if self.risk_max_drawdown_type == percent_of_equity:
            threshold = peak * self.risk_max_drawdown_value * 0.01
        else:
            threshold = float(self.risk_max_drawdown_value)
        drawdown = peak - float(self.equity)
        return drawdown >= threshold > 0.0

    def _trigger_risk_halt(self, reason: str, price: float, *,
                           until_day_end: bool = False) -> None:
        """Cancel pending orders, close any open position at ``price``, halt trading.

        ``reason`` is embedded in the synthetic close order's comment so the
        backtest log identifies which ``strategy.risk.*`` rule fired. Once
        :attr:`risk_halt_trading` is set, ``strategy.entry`` / ``strategy.order``
        early-return, ``process_orders`` short-circuits, and the strategy stays
        flat.

        :param reason: Rule name embedded in the synthetic close's comment.
        :param price: Fill price of the synthetic close.
        :param until_day_end: Stamp the halt with the current trading day so
            :meth:`_release_intraday_halt` lifts it at the next rollover —
            ``max_intraday_loss`` blocks the rest of the day, not the run.
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
            self._fill_order(close_order, price)
        self.risk_halt_trading = True
        self.risk_halt_day = self.risk_last_trading_day if until_day_end else -1

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
            self._fill_order(close_order, cap_close_price, counts_as_filled_order=False)

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
        # ``max_drawdown`` is evaluated only on a bar that booked realized P&L
        # (a close/reduce fill). TV never halts on open-position paper drawdown
        # alone: a long carried to a 50% floating drawdown with no closing order
        # keeps riding, and force-closes the instant a reducing order fills while
        # past the threshold — see ``_is_max_drawdown_breached``.
        if self.new_closed_trades and self._is_max_drawdown_breached():
            self._trigger_risk_halt("Max drawdown reached", price)
            return
        if self._is_max_intraday_loss_breached():
            self._trigger_risk_halt("Max intraday loss reached", price, until_day_end=True)
            return
        if self._is_max_cons_loss_days_breached():
            self._trigger_risk_halt("Max consecutive loss days reached", price)

    def _check_already_filled(self, order: Order) -> Literal['stop', 'limit'] | None:
        """
        Check if a stop or limit order would be immediately fillable due to a gap.
        This is called during process_orders when we have the current bar's OHLC values.

        When there's a gap, orders that would normally wait for price movement
        should execute immediately at the open price.

        :param order: The order to check
        :return: The leg the gap triggered, or None if the order stays pending
        """
        # Check stop orders with gaps
        if order.stop is not None:
            # Long stop order (size > 0): triggers if open gaps above stop level
            if order.size > 0 and self.o >= order.stop:
                return 'stop'
            # Short stop order (size < 0): triggers if open gaps below stop level
            if order.size < 0 and self.o <= order.stop:
                return 'stop'

        # Check limit orders with gaps
        if order.limit is not None:
            # Long limit order (size > 0): triggers if open gaps below limit level
            if order.size > 0 and self.o <= order.limit:
                return 'limit'
            # Short limit order (size < 0): triggers if open gaps above limit level
            if order.size < 0 and self.o >= order.limit:
                return 'limit'

        return None

    def _exit_awaits_entry(self, order: Order) -> bool:
        """True while an exit leg bound to a ``from_entry`` has no open trade to act on.

        TradingView activates a ``strategy.exit`` bracket only after its bound
        entry fills. Until then (entry pending, cancelled or rejected) the leg
        must not trigger: a fill would cancel its sibling OCA legs and count
        toward the filled-order caps even though there is nothing it can close.
        """
        if order.order_type != _order_type_close or order.order_id is None or order.from_entry_na:
            return False
        return not self._has_bound(order.order_id)

    def _entry_fill_price(self, entry_id: str | None,
                          entry_seq: int | None = None) -> float | None:
        """Fill price the entry an exit leg is bound to was opened at.

        A leg carrying an ``entry_seq`` prices off THAT entry, so two pyramid adds
        sharing a ``from_entry`` id resolve their tick offsets from their own fill
        instead of both from the oldest one. A leg issued against a still-PENDING
        entry order has no price yet: it must wait for THAT order to fill, even when
        an older trade shares the id — pricing it off the older entry freezes the
        level a bar early (``ticks_resolved`` is one-shot) and the bracket then sits
        at the wrong distance for the whole life of the new entry. MEASURED on the
        wild `How to use Leverage and Margin in PineScript` reference
        (BINANCE:BTCUSDT 30m, 2025-01-27 08:00): TV fills `tp_long` at
        `fill + 100 ticks` of the entry that just opened, on the very bar it opened.
        Only a leg with neither a binding nor a pending entry order falls back to
        the first open trade under the id.

        :param entry_id: The ``from_entry`` an exit leg is bound to
        :param entry_seq: The bound entry, when the leg already has one
        :return: The entry price, or None while the entry has no open trade
        """
        binding = self._binding(entry_seq)
        if binding is not None:
            return binding.entry_price
        if entry_seq is None and entry_id is not None and entry_id in self.entry_orders:
            return None
        for trade in self.open_trades:
            if trade.entry_id == entry_id:
                return trade.entry_price
        return None

    def _resolve_tick_exit(self, order: Order, entry_price: float) -> bool:
        """Turn an exit's tick offsets into concrete price levels.

        ``strategy.exit(profit=/loss=/trail_points=)`` states its levels as tick
        distances from the entry price, so they cannot exist before the entry
        fills. Resolving them also indexes the order in ``PriceOrderBook``:
        without a level the order sits in no price bucket at all, and no leg of
        the intrabar walk can ever yield it.

        A tick offset and its absolute counterpart (``profit``/``limit``,
        ``loss``/``stop``, ``trail_points``/``trail_price``) are BOTH live and
        the one the price path reaches first wins. MEASURED (probes ``ep6_C``
        to ``ep6_M``, BINANCE:BTCUSDT 30m, 28915 bars): with the absolute level
        farther out the trades are bit-identical to the same script written
        without it at all, and with the absolute level nearer the exits move
        onto it -- 30/140 for ``limit``, 36/141 for ``stop``, the rest fired
        from the tick offset because the absolute argument evaluated to ``na``
        on the bar whose order was live when the entry filled. Both levels of a
        pair sit on the same side of the entry, so "reached first" is simply the
        lower of the two on the way up and the higher on the way down.

        Pine made this change in v6; a v4/v5 source means the ABSOLUTE level
        alone, and PyneComp writes that meaning out (``converter/semantics.py``)
        rather than the runtime carrying two rules.

        :param order: The exit order carrying the tick offsets
        :param entry_price: Fill price of the entry the exit is bound to
        :return: True when a level was created or moved — False once resolved
        """
        if order.ticks_resolved:
            return False
        order.ticks_resolved = True
        direction = 1.0 if order.size < 0 else -1.0  # Exit order size is negative of position
        changed = False
        # A tick offset that undercuts an absolute level leaves the old, farther
        # level indexed: ``add_order`` only ever adds, so the book is rebuilt.
        moved = False

        if order.profit_ticks is not None:
            level = _price_round(
                entry_price + direction * syminfo.mintick * order.profit_ticks, direction)
            if order.limit is None:
                order.limit = level
                changed = True
            elif (level < order.limit) if direction > 0 else (level > order.limit):
                order.limit = level
                changed = moved = True

        if order.loss_ticks is not None:
            level = _price_round(
                entry_price - direction * syminfo.mintick * order.loss_ticks, -direction)
            if order.stop is None:
                order.stop = level
                changed = True
            elif (level > order.stop) if direction > 0 else (level < order.stop):
                order.stop = level
                changed = moved = True

        if order.trail_points_ticks is not None:
            level = _price_round(
                entry_price + direction * syminfo.mintick * order.trail_points_ticks, direction)
            if order.trail_price is None:
                order.trail_price = level
                changed = True
            elif (level < order.trail_price) if direction > 0 else (level > order.trail_price):
                order.trail_price = level
                changed = moved = True

        # Update orderbook only when prices were actually calculated
        if changed:
            if moved:
                self.orderbook.remove_order(order)
            self.orderbook.add_order(order)
        return changed

    def _resolve_filled_entry_exits(self) -> bool:
        """Resolve every tick-based exit whose bound entry already has an open trade.

        :return: True when at least one exit gained a price level it did not have
        """
        if not self.exit_orders:
            return False
        resolved = False
        for order in self.exit_orders.values():
            entry_price = self._entry_fill_price(order.order_id, order.entry_seq)
            if entry_price is not None and self._resolve_tick_exit(order, entry_price):
                resolved = True
        return resolved

    def _activate_trails_on_fill(self, entry_id: str | None, ohlc: bool, rising: bool,
                                 awaiting: set[Order], close_leg_queue: list[Order]) -> None:
        """Start the trailing legs an entry fill just activated, at the fill price.

        A trailing leg bound to a ``from_entry`` cannot act before that entry
        fills, so the trailing pre-walk in :meth:`_process_limit_stop_orders`
        passes over it. An entry filling INTRABAR activates it partway through
        the bar, and TradingView runs it from there over the rest of the assumed
        path — including the same-bar fill that usually follows.

        :param entry_id: ``from_entry`` id of the entry that just filled
        :param ohlc: The bar's intra-bar leg order (see :meth:`process_orders`)
        :param rising: True when the entry filled on an open -> high leg
        :param awaiting: Trailing legs still waiting for their entry; one started
            here is dropped, so a later fill of the same entry cannot restart it
        :param close_leg_queue: Legs left pending join the closing-leg pass
        """
        if not awaiting:
            return
        entry_price = self._entry_fill_price(entry_id)
        if entry_price is None:
            return
        for order in [o for o in awaiting if o.order_id == entry_id]:
            awaiting.discard(order)
            if order.cancelled or order.filled_by_type is not None or order.trail_price is None:
                continue
            if self._process_trailing_stop(
                    order, ohlc, start=entry_price, rising=rising) == _trail_pending:
                close_leg_queue.append(order)

    def _walk_leg(self, start: float, leg_end: float, rising: bool, ohlc: bool,
                  trail_awaiting: set[Order], trail_close_leg: list[Order]) -> None:
        """Walk one leg of the assumed intrabar path, level by level.

        A tick-based bracket has no price level until its entry fills, so an
        entry filling mid-leg materializes levels the walk in flight can never
        yield: :meth:`PriceOrderBook.iter_levels` snapshots the level list
        before its first yield. When that happens the walk resumes at the level
        it stopped on, so the fresh bracket takes its chronological place among
        the levels still ahead of the price instead of being appended after the
        whole leg — a later entry level between the two would otherwise be
        reached while the bracket's own trade is still open.

        Orders already offered at the resume level are skipped, so each
        (order, level) pair gets exactly one shot per leg, as it does without a
        resume. A bracket stated as an explicit ``limit=`` / ``stop=`` price
        needs none of this: it was indexed before the walk started.

        :param start: Price the leg starts at (the bar open)
        :param leg_end: The extreme this leg runs to (the bar's high or low)
        :param rising: True for an open -> high leg, False for an open -> low leg
        :param ohlc: The bar's intra-bar leg order (see :meth:`process_orders`)
        :param trail_awaiting: Trailing legs whose entry has not filled yet
        :param trail_close_leg: Collects trailing legs left for the closing-leg pass
        """
        book = self.orderbook
        resume = start
        # Orders of the resume level already offered before the walk stopped there
        offered: tuple[Order, ...] = ()
        while True:
            resumed = False
            if rising:
                levels = book.iter_levels(min_price=resume, max_price=leg_end)
            else:
                levels = book.iter_levels(max_price=resume, min_price=leg_end, desc=True)
            for price, orders in levels:
                for i, order in enumerate(orders):
                    if order in offered:
                        continue
                    if rising:
                        filled = self._check_high_stop(order) or self._check_high(order)
                    else:
                        filled = self._check_low_stop(order) or self._check_low(order)
                    # An entry fill can activate a bracket bound to it: resolving
                    # indexes it at a level this generator cannot reach, and its
                    # trailing leg starts here, at the fill price. A closing fill
                    # never opens a trade, so it can activate nothing.
                    if filled and order.order_type != _order_type_close:
                        materialized = self._resolve_filled_entry_exits()
                        self._activate_trails_on_fill(order.order_id, ohlc, rising,
                                                      trail_awaiting, trail_close_leg)
                        if materialized:
                            resume, offered = price, tuple(orders[:i + 1])
                            resumed = True
                            break
                if resumed:
                    break
            # Every resume consumes at least one still-unresolved tick offset, so
            # the loop is bounded by the number of pending bracket legs.
            if not resumed:
                return

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
            self.fill_order(order, p)
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
                self.fill_order(order, p)
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
            self.fill_order(order, order.limit)
            return True
        # Buy stop triggers when price rises to the stop level
        if order.stop is not None and order.size > 0 and order.stop <= self.c:
            p = order.stop
            slippage = lib._script.slippage
            if slippage > 0:
                p += syminfo.mintick * slippage
            order.filled_by_type = 'loss'
            self.fill_order(order, p)
            return True
        return False

    def _process_trailing_stop(self, order: Order, ohlc: bool, close_leg: bool = False,
                               start: float | None = None, rising: bool = False) -> int:
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

        A bracket its own entry activates partway through the bar starts at
        ``start`` -- the entry's fill price -- instead of the bar open, and only
        the segments still ahead of that point are walked. The water mark starts
        there too: an extreme the bar reached BEFORE the fill belongs to a
        stretch the trail was not live for.

        :param order: The exit order carrying ``trail_price``.
        :param ohlc: The bar's intra-bar leg order (see :meth:`process_orders`).
        :param close_leg: If True, walk only the closing (second extreme ->
            close) segment, resuming the state a prior default call persisted.
        :param start: Price the walk begins at when the order was activated
            mid-bar; None starts it at the bar open.
        :param rising: With ``start``, True when the entry filled on an
            open -> high leg -- it selects the segments still ahead.
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
        # The walk's first tick: the bar open, or the fill price that activated
        # this bracket mid-bar.
        start_tick = self.o if start is None else start

        if order.sign < 0:
            # Long position: trailing sell-stop riding under the high-water mark.
            armed = order.trail_triggered
            stop = order.trail_stop if armed else None

            if not close_leg and armed and stop is not None:
                # A carried stop already passed at the first tick fills there --
                # an inter-bar gap through the stop, or an entry filling past it.
                if start_tick <= stop:
                    p = start_tick
                    if slippage > 0:
                        p -= syminfo.mintick * slippage
                    order.filled_by_type = 'trailing'
                    self.fill_order(order, p)
                    return _trail_filled
                # The first tick advances the water mark; with trail_offset == 0
                # the stop lands on that tick itself and fills there.
                new_stop = round_to_mintick(start_tick - offset_price)
                if new_stop > stop:
                    stop = new_stop
                    if start_tick <= stop:
                        p = stop
                        if slippage > 0:
                            p -= syminfo.mintick * slippage
                        order.filled_by_type = 'trailing'
                        self.fill_order(order, p)
                        return _trail_filled
            elif not close_leg and not armed and start_tick >= order.trail_price:
                # The walk starts beyond the activation level: the trail arms on
                # its first tick with that tick as its water mark.
                armed = True
                stop = round_to_mintick(start_tick - offset_price)
                if start_tick <= stop:
                    p = stop
                    if slippage > 0:
                        p -= syminfo.mintick * slippage
                    order.filled_by_type = 'trailing'
                    self.fill_order(order, p)
                    return _trail_filled

            # Walk the assumed intrabar path: rising segments arm the trail and
            # ratchet the water mark, a falling segment fills at the trailed
            # stop when it reaches it.
            if close_leg:
                prev = self.l if ohlc else self.h
                path: tuple[float, ...] = (self.c,)
            else:
                prev = start_tick
                path = (self.h, self.l) if ohlc else (self.l, self.h)
                if start is not None and rising != ohlc:
                    # Activated on the path's SECOND leg: that leg's own extreme is
                    # behind the fill already, only the other one is still ahead.
                    # Measured on CAPITALCOM:EURUSD 30m with a buy-limit entry (the
                    # fill lands on a descending leg, so the high can precede it):
                    # on open -> high -> low -> close bars TV closed 0 of 2063 trades
                    # on the entry bar -- the pre-fill high never enters the water
                    # mark -- while on open -> low -> high -> close bars it closed
                    # 1497 of 1526, every one at exactly `high - trail_offset`.
                    path = path[1:]
            for nxt in path:
                self._walk_node = 3 if close_leg else (1 if (nxt == self.h) == ohlc else 2)
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
                            self.fill_order(order, p)
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
                        self.fill_order(order, p)
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
                # A carried stop already passed at the first tick fills there --
                # an inter-bar gap through the stop, or an entry filling past it.
                if start_tick >= stop:
                    p = start_tick
                    if slippage > 0:
                        p += syminfo.mintick * slippage
                    order.filled_by_type = 'trailing'
                    self.fill_order(order, p)
                    return _trail_filled
                # The first tick advances the water mark; with trail_offset == 0
                # the stop lands on that tick itself and fills there.
                new_stop = round_to_mintick(start_tick + offset_price)
                if new_stop < stop:
                    stop = new_stop
                    if start_tick >= stop:
                        p = stop
                        if slippage > 0:
                            p += syminfo.mintick * slippage
                        order.filled_by_type = 'trailing'
                        self.fill_order(order, p)
                        return _trail_filled
            elif not close_leg and not armed and start_tick <= order.trail_price:
                # The walk starts beyond the activation level: the trail arms on
                # its first tick with that tick as its water mark.
                armed = True
                stop = round_to_mintick(start_tick + offset_price)
                if start_tick >= stop:
                    p = stop
                    if slippage > 0:
                        p += syminfo.mintick * slippage
                    order.filled_by_type = 'trailing'
                    self.fill_order(order, p)
                    return _trail_filled

            # Walk the assumed intrabar path: falling segments arm the trail and
            # ratchet the water mark, a rising segment fills at the trailed stop
            # when it reaches it.
            if close_leg:
                prev = self.l if ohlc else self.h
                path = (self.c,)
            else:
                prev = start_tick
                path = (self.h, self.l) if ohlc else (self.l, self.h)
                if start is not None and rising != ohlc:
                    # Second-leg activation drops the extreme already behind the
                    # fill -- see the mirrored comment in the long branch.
                    path = path[1:]
            for nxt in path:
                self._walk_node = 3 if close_leg else (1 if (nxt == self.h) == ohlc else 2)
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
                            self.fill_order(order, p)
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
                        self.fill_order(order, p)
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
        """Seed a freshly (re-)issued trailing exit's high/low-water mark.

        ``process_orders`` runs before the script body, so an exit issued in the
        script on bar N -- e.g. one gated on ``strategy.position_size``, which is
        only known once the entry has filled -- is first evaluated on bar N+1.
        The water mark still has to exist by then: TradingView arms such a leg on
        its issue bar and carries the armed stop into the next bar's walk.

        MEASURED (BINANCE:BTCUSDT 30m, 222 trades of a probe entering every 97th
        bar and issuing ``strategy.exit(trail_points=50, trail_offset=10)`` from
        inside ``if strategy.position_size > 0``): the mark anchors to the issue
        bar's CLOSE tick, NOT to its extreme. Folding the entry-fill bar's H/L in
        puts the stop above the next bar's open, which then gaps through and fills
        at that open -- 106 of the 222 exits land on the wrong price that way,
        zero with the close anchor. The same anchor was already verified for a
        changed-params re-issue (per-bar ``atr*mult`` trail): a long re-issue
        filled at ``next open - offset`` (open above close, mark advanced) and a
        short re-issue at ``close + offset`` (open above close, mark kept).

        Exits placed on the entry SIGNAL bar (entry still pending, so no bound
        trade is open yet) are skipped: ``process_orders`` seeds those on their
        fill bar from the fill price onward (``_activate_trails_on_fill``).

        ``fold_extreme`` is True only for an IDENTICAL re-issue, whose carried
        mark ``process_orders`` has already ratcheted through this bar's extreme
        -- re-folding it there is a no-op that keeps the carried leg's state
        explicit.

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
        # Same activation level ``_resolve_tick_exit`` settles, computed without
        # its side effects: whichever of the pair the path reaches first.
        trail_price = order.trail_price
        if order.trail_points_ticks is not None:
            level = _price_round(
                entry_price + direction * syminfo.mintick * order.trail_points_ticks, direction)
            if trail_price is None or ((level < trail_price) if direction > 0
                                       else (level > trail_price)):
                trail_price = level
        if trail_price is None:
            return

        round_to_mintick = lib.math.round_to_mintick
        offset_price = syminfo.mintick * order.trail_offset
        # Arming on the issue (entry-fill) bar is gated on the bar CLOSE, not its
        # intrabar extreme: TradingView only carries a trailing stop out of the
        # entry-fill bar when that bar closes past the activation level. A bar
        # whose extreme pierces the activation level but closes back inside it does
        # NOT arm here -- it arms later, intrabar, in the normal price walk, which
        # fills on its arming bar too. Only a leg first issued AFTER the fill gets
        # here in that state: one that already existed during the entry-fill bar's
        # walk armed (and possibly filled) inside it, from the fill price onward --
        # see ``_activate_trails_on_fill`` -- so it reaches this gate already
        # triggered. On every later bar a close past the level implies the high
        # already pierced it, so process_orders has already armed the carried order
        # and this gate never fires there.
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

    def _check_intraday_loss(self, check_price: float) -> bool:
        """Enforce ``max_intraday_loss`` at one price node of the bar walk.

        MEASURED on TradingView: the rule is an INTRA-BAR check, not a bar-end
        one. The emulator closes the position at the price node where the
        equity first drops ``value`` below the day's opening equity — for an
        open position that node is the bar's UNFAVORABLE extreme (the low for a
        long, the high for a short), so the exit prints there, slippage
        included, exactly like a margin call. Probed on BINANCE:BTCUSDT: 191 of
        226 probe trades and 98 of 109 wild-strategy trades close on their own
        entry bar at that extreme, and a position carried across a day boundary
        closes on the FIRST bar of the next day once the re-anchored equity
        breaks — even while the trade itself is deep in profit.

        :param check_price: Path price to mark the open position to.
        :return: True when the rule fired and flattened the position.
        """
        if (self.risk_max_intraday_loss_value is None or self.risk_halt_trading
                or not self.open_trades):
            return False
        # Account-currency value of a 1.0-point move on 1 contract — same
        # identity latch the margin-call checkpoint uses.
        pv = (syminfo.pointvalue if _conv_identity_script is lib._script
              else _account_point_value())
        quantity = abs(self.size)
        open_profit = (check_price - self.avg_price) * quantity * pv
        if self.sign < 0:
            open_profit = -open_profit
        equity = lib._script.initial_capital + self.netprofit + open_profit
        if not self._is_max_intraday_loss_breached(equity):
            return False
        # The breach is judged at the raw path price, but the forced close is a
        # market order and prints slipped AGAINST the position, like a margin
        # call: measured on the wild strategy above, all 98 rule closes land on
        # the extreme -/+ ``slippage`` ticks (long low - slip, short high + slip).
        fill_price = check_price
        script = lib._script
        if script.slippage > 0:
            slippage_amount = syminfo.mintick * script.slippage
            fill_price = (check_price + slippage_amount if self.sign < 0
                          else check_price - slippage_amount)
        self._trigger_risk_halt("Max intraday loss reached", fill_price, until_day_end=True)
        return True

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

        # A liquidation earlier this bar armed the second margin-call stage: the
        # bar's next checkpoint runs TV's stale-credit walk instead of the real
        # account state (which the stage-1 proceeds usually push back above
        # water). If the walk fires, this checkpoint is done; if it credits the
        # whole deficit (k == 0), fall through to the normal check.
        st2 = self._mc_stage2
        if st2 is not None:
            self._mc_stage2 = None
            if st2[0] == int(lib.bar_index) and st2[3] == for_short:
                if self._margin_call_stage2(st2, check_price, for_short):
                    return False

        quantity = abs(self.size)
        # Convert price * quantity to account-currency for margin/equity comparisons.
        # Identity latch of ``_account_point_value`` inlined: this runs up to four
        # times a bar on an open position, and a non-converting run is the norm.
        pv = (syminfo.pointvalue if _conv_identity_script is lib._script
              else _account_point_value())

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
            # one whole contract, capped by the current position size. This holds
            # at the open AND intrabar, regardless of position size
            # (BINANCE:BTCUSDT 30m Gaussian Channel corpus: 43 margin calls that
            # trim exactly 1.0 from 8+ contract positions — longs at the
            # entry-fill open price, shorts at a high one tick above the open).
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

        # Arm the second stage for the bar's next checkpoint. TV does not credit
        # the liquidation proceeds there: it walks the pre-liquidation trades
        # FIFO, crediting each trade's entry cost immediately but its realized
        # gain (net of the exit commission) one trade LATE, and stops before the
        # credit would cover the deficit — see _margin_call_stage2. The snapshot
        # must be taken before the fill mutates the trade list.
        deficit = (margin_ticks - equity_ticks) * mintick if big_margin else -available_funds
        self._mc_stage2 = (int(lib.bar_index), deficit, check_price, for_short,
                           tuple((abs(t.size), t.entry_price) for t in self.open_trades))

        self._fill_order(margin_call_order, fill_price)
        return False

    def _margin_call_stage2(self, st2: tuple[int, float, float, bool,
                                             tuple[tuple[float, float], ...]],
                            check_price: float, for_short: bool) -> bool:
        """
        Execute the second stage of an intrabar margin call at the bar's next
        checkpoint, reproducing TV's stale settlement of the first stage.

        MEASURED on BINANCE:BTCUSDT 30m (Rocket Grid Algorithm wild corpus run +
        rg_sc550/rg_nc/rg_eq input-sweep probes + mc4/mc5 synthetic probes,
        848/848 margin-call bars exact): after a liquidation at checkpoint price
        C1 with deficit D, TV's available funds at the NEXT checkpoint do not
        reflect the liquidation proceeds. Instead TV walks the pre-liquidation
        trades FIFO, crediting trade j's entry cost at step j but its realized
        gain — net of the exit commission only — at step j+1, and stops with the
        last step k whose running credit still fits inside D:

            k  = max j: cum_cost_j + net_gain_{j-1}(C1) <= D
            D2 = D - cum_cost_k - net_gain_{k-1}

        The remaining shortfall D2 is covered 4x like the first stage, but sized
        at the FIRST stage's checkpoint price C1 (not this checkpoint's price),
        capped by the remaining position, and filled here with slippage. With
        k == 0 (the first trade's cost alone exceeds D — the common single-trade
        case) no second stage fires: 754/754 single-stage probe bars confirm the
        deficit is then fully credited. No third stage was ever observed.

        :param st2: The armed (bar_index, deficit, C1, for_short, legs) state
        :param check_price: This checkpoint's price (the stage-2 fill price base)
        :param for_short: True when liquidating a short position
        :return: True if the second stage fired (the checkpoint is consumed)
        """
        _, deficit, c1, _, legs = st2
        if not self.open_trades:
            return False
        script = lib._script
        margin_percent = script.margin_short if for_short else script.margin_long
        margin_ratio = margin_percent / 100.0
        pv = (syminfo.pointvalue if _conv_identity_script is lib._script
              else _account_point_value())
        comm_type = script.commission_type
        comm_val = script.commission_value

        cum = 0.0
        gain_cur = 0.0
        k = 0
        d2 = deficit
        for qty, entry_price in legs:
            cost = qty * entry_price * pv
            if deficit - (cum + cost) - gain_cur < 0.0:
                break
            cum += cost
            gain_prev = gain_cur
            gain = qty * (entry_price - c1 if for_short else c1 - entry_price) * pv
            if comm_val:
                if comm_type == _commission.percent:
                    gain -= qty * c1 * pv * comm_val / 100.0
                elif comm_type == _commission.cash_per_contract:
                    gain -= qty * comm_val
                else:
                    gain -= comm_val
            gain_cur += gain
            k += 1
            d2 = deficit - cum - gain_prev
        if k == 0 or d2 <= 0.0:
            return False

        loss = d2 / margin_ratio
        rfactor = syminfo._size_round_factor  # noqa
        raw_cover_lots = loss / (c1 * pv) * rfactor
        # Same truncation-with-float-snap as the first stage.
        nearest_cover = round(raw_cover_lots)
        if abs(raw_cover_lots - nearest_cover) <= raw_cover_lots * 2.0 ** -26 + 1e-9:
            cover_lots = int(nearest_cover)
        else:
            cover_lots = int(raw_cover_lots)
        if cover_lots <= 0:
            return False
        margin_call_size = min(cover_lots * 4 / rfactor, abs(self.size))

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

        self._fill_order(margin_call_order, fill_price)
        return True

    def process_deferred_margin_call(self):
        """
        Execute a deferred margin call (after the user script has run), then
        re-check margin at the bar close.

        Called from script_runner after the user script's main() completes.
        Liquidation is booked on the current bar at the close price, in whole
        contracts.
        """
        # Margin is evaluated at every bar close: without this check the same
        # liquidation only fires at the next bar's open — one bar late, and at the
        # open price on gapped data (Hybrid 2026-05-07 02:00: TV trims 1.0 contract
        # at C=80898.0 on the 02:00 bar while the O/H/L walk points all pass the
        # margin comparison). Sized like the bar-open check in whole contracts;
        # every observed instance trimmed exactly 1.0 contract, so the
        # whole-contract choice is untested beyond that.
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

    def _resolve_deferred_qty(self, order: Order, fill_price: float) -> None:
        """Finalize a default-sized entry's quantity at its actual fill price.

        TradingView resolves percent_of_equity / cash default sizing of
        price-based (limit/stop) orders when the order EXECUTES, dividing the
        order's money budget by the per-unit cost at the real fill price. The
        budget itself — the money AND the quote-to-account rate that turns the
        fill price into an account-currency unit cost — is FROZEN at the close of
        the bar where the order was (last) placed or modified: an order resting
        for many bars keeps the placement-close equity, it is not re-marked at
        fill (measured on the
        Trendoscope corpus fork: a buy stop placed 12 bars before its fill
        sized off the placement bar's equity, reproduced one-shot by probe;
        an order re-placed every bar degenerates to prev-close equity, which
        is what earlier fill-time measurements saw). For those the
        placement-time size was only the margin-check estimate — a marketable
        limit filling at the open re-sizes here. Market entries never defer:
        they keep the placement-close size computed in ``entry``
        (TV-probe-verified). The reversal flip component stays frozen from
        creation (TV computes the flip quantity at order creation time).

        MEASURED on the wild-corpus strategy "Breakout Trend Follower"
        (BINANCE:BTCUSDT 30m, a USD account on a USDT-quoted symbol, so the daily
        COINBASE:USDTUSD rate steps at every 00:00 bar): 12 of 580 entries sized
        one lot off while the FILL bar's rate converted the unit cost, ten of them
        on a 00:00 bar. Converting with the PLACEMENT bar's rate instead makes all
        580 exact — the rate rides with the frozen money, not with the fill.
        """
        order.deferred_qty = False
        old_abs = abs(order.size)
        budget = _default_entry_budget(float(fill_price), order.budget_pv)
        if budget is None:
            qty = lib._script.default_qty_value
            money = None
        else:
            money = order.budget_money if order.budget_money is not None else budget[0]
            qty = _sig10_money(money) / budget[1]
        if not (0.0 < qty < math.inf):  # unsizable_qty
            order.size = 0.0
            return
        size = _size_floor((qty + order.flip_extra) * order.sign)
        if size != 0.0:
            # The big-money sizing judgment applies to the money-sized part of
            # the order only; the reversal flip component is the old position,
            # already an exact lot multiple.
            flip = order.flip_extra * order.sign
            size = _judge_money_entry(size - flip, float(fill_price), money=money) + flip
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
                o.bound_size = _size_round(o.bound_size + extra)
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

        The requirement is the ORDER's own margin, not that of the position it
        would leave behind. The two coincide from flat, which is where the
        asymmetry above was measured, but they part company under pyramiding:
        Rocket Grid Algorithm rests ten stacked stop entries whose tenth alone
        fits the account while the resulting eleven-fold position does not, and
        TradingView fills that tenth on every one of its 16 grids (then margin
        calls) instead of cancelling it.
        """
        if not self.entry_orders:
            return
        script = lib._script
        pv = _account_point_value()
        for order in list(self.entry_orders.values()):
            if order.order_type != _order_type_entry:
                continue
            if order.limit is None and order.stop is None:
                continue
            margin_percent = script.margin_short if order.sign < 0 else script.margin_long
            if margin_percent <= 0:
                continue
            margin_needed = abs(order.size) * self.c * pv * (margin_percent / 100.0)
            if margin_needed > self.equity:
                self._remove_order(order)

    def _entry_exceeds_margin_after_fill(self, order: Order, fill_price: float,
                                         base_size: float | None = None,
                                         base_equity: float | None = None) -> bool:
        """
        Check whether an entry's resulting position is affordable at its fill price.

        TV rejects the entry before filling when the position that would remain after
        the fill cannot be margined. Once an entry has filled, later open/high/low
        margin breaches are handled by the margin-call path.

        ``base_size``/``base_equity`` override the position size the fill adds to and
        the equity it is margined against (both default to the current values).
        Passing the bar-start size AND equity tests whether the order would have been
        affordable on its own — an over-margin caused only by a prior same-bar fill
        (which also shifts ``self.equity`` via its open P&L) is handled by the
        margin-call path, not a hard reject.
        """
        script = lib._script
        margin_percent = script.margin_short if order.sign < 0 else script.margin_long
        if margin_percent <= 0:
            return False

        pv = _account_point_value()
        margin_ratio = margin_percent / 100.0

        if base_size is None:
            base_size = self.size
        new_qty = abs(base_size + order.size)
        if new_qty == 0.0:
            return False

        equity = self.equity if base_equity is None else base_equity
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
        # Below the 1e7 gate TV rejects on a strict "margin exceeds equity":
        # a percent_of_equity entry sized at the signal close fills at the next
        # open, so a positive shortfall means the fill price rose above the
        # sizing price and the position no longer fits — TV rejects it (there is
        # no legitimate positive-shortfall fill; only the fill-price move can
        # create one). The tolerance is float noise only. Measured on
        # BINANCE:BTCUSDT 30m: Master Trend 2025-04-17 05:00 rejected at a
        # +0.00045 USD / 4.7e-10 relative shortfall (a 1-tick fill-open move
        # eating the mincontract rounding buffer), tighter than the earlier
        # corpus rejects at 1.75e-9..1.06e-7; the accumulated netprofit float
        # error over a full run stays ~1e-13 relative, so 1e-11 separates real
        # overages from noise.
        mintick = syminfo.mintick
        if equity >= 1e7 and mintick and mintick > 0:
            return math.floor(equity / mintick + 0.5) < math.floor(margin_needed / mintick)
        return margin_needed - equity > abs(equity) * 1e-11

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
            self.fill_order(order, p)
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
                self.fill_order(order, p)
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
            self.fill_order(order, order.limit)
            return True
        # Sell stop triggers when price falls to the stop level
        if order.stop is not None and order.size < 0 and order.stop >= self.c:
            p = order.stop
            slippage = lib._script.slippage
            if slippage > 0:
                p -= syminfo.mintick * slippage
            order.filled_by_type = 'loss'
            self.fill_order(order, p)
            return True
        return False

    def _path_price(self, node: int) -> float:
        """Price of a node of the bar's assumed path.

        The emulator walks open -> the extreme nearest it -> the other extreme ->
        close, and a calc_on_order_fills re-execution sees it standing on one of
        those nodes: the one :attr:`_coof_cursor` names.

        Measured 2026-08-15 on BINANCE:BTCUSDT 30m with a body whose only way
        into a position is a stop placed 0.3% off the close — so its fill is
        provably never at the open. All 1087 entries, both sides, no exception:
        the pass the fill triggers marks the open position at the extreme that
        ends the leg the stop filled on (the high for a buy stop, the low for a
        sell stop), NOT at the bar open and NOT at the stop price itself. Where
        the cursor goes from there is decided by the runner's COOF loop.
        """
        if node <= 0:
            return self.o
        if node >= 3:
            return self.c
        near, far = (self.h, self.l) if self.h - self.o < self.o - self.l else (self.l, self.h)
        return near if node == 1 else far

    def _mark_to_last_fill(self, price: float | None = None) -> None:
        """Reprice the emulator for a calc_on_order_fills re-execution.

        A COOF body execution sees the broker emulator AT THE POINT OF THE BAR
        IT HAS REACHED, not at the bar close: a default-sized market entry it
        places is sized — and the equity it reads is marked — there. Measured
        2026-08-13 on SUPERTREND ATR WITH TRAILING STOP LOSS (BINANCE:BTCUSDT
        30m), whose 2026-06-24 11:30 COOF re-entry sizes at 62382.48 where the
        bar close 62921.19 gives one lot step less; and 2026-08-15 on Hull
        Moving Average Swing Trader, where the entry a pass places is sized at
        the very price that pass fills it at. Which point of the bar that is,
        is decided by :attr:`_coof_cursor` and :meth:`_path_price`.
        ``process_orders`` re-anchors ``c`` and the open P&L to the bar close at
        the start of the next pass, so nothing needs undoing.

        On the magnified path real sub-bars replace the assumed one and there is
        no cursor: a fill-triggered pass marks at the last fill, while a
        ``calc_on_every_history_tick`` pass — which owes nothing to a fill —
        passes the sub-bar it stands at the end of as ``price``.

        :param price: Explicit mark price, overriding both the cursor's path
            node and the last fill.
        """
        self.c = (price if price is not None
                  else self._path_price(self._coof_cursor) if self._coof_cursor >= 0
                  else self._last_fill_price)
        if self.size != 0.0:
            self.openprofit = self.size * (_tick_snap(self.c) - self.avg_price) * _account_point_value()

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

        # The path walk restarts with the bar; the COOF passes of one bar share it.
        if self._coof_cursor < 0:
            self._path_node = 0

        self.drawdown_summ = self.runup_summ = 0.0
        self.new_closed_trades.clear()
        # Undo any immediate close a COOF trial body run enqueued (position-side
        # analog of the restored ``var`` state); no-op in the common case.
        self._discard_deferred_immediate_closes()

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

        # A calc_on_order_fills re-execution does not get the bar open again: the
        # emulator has already walked part of the bar, so the market order it
        # placed fills at the node the walk has reached — the same node the body
        # is marked at, see ``_path_price``. Measured on the wild script
        # `Hull Moving Average Swing Trader` (BINANCE:BTCUSDT 30m), whose body
        # closes and re-enters on every pass, over all 590 of its four-entry bars
        # with no exception: the four in-bar fill moments price at
        # ``open, open, near, far`` — where ``near`` is the extreme the path
        # visits first, the same one ``ohlc`` above selects. (The orders the
        # definitive execution places fill at the NEXT bar's open, which is what
        # already happens.)
        self._walk_node = self._coof_cursor if self._coof_cursor > 0 else 0
        self._market_fill_price = (self._path_price(self._coof_cursor)
                                   if self._coof_cursor >= 0 else self.o)

        self._process_at_bar_open(ohlc)
        self._process_limit_stop_orders(ohlc, self._coof_cursor)
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
        # Statically a value (module_property), at runtime still the function
        current_trading_day = int(lib.time_tradingday())
        self._release_intraday_halt(current_trading_day)
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
                "Max consecutive loss days reached", self.o,
            )
            return True
        return False

    def _process_at_bar_open(self, ohlc: bool):
        """Phase 1: Process orders at bar open — gap detection, market fills, margin."""
        if self._roll_trading_day():
            return

        # Get script reference for slippage
        script = lib._script

        # Drop an exit leg left behind by an entry order that never opened. A leg
        # issued against a PENDING entry carries no ``entry_seq``; once that order
        # is gone -- cancelled, or margin-rejected -- it has nothing to close, and
        # TradingView does not let it reach the position an OLDER entry of the same
        # id still holds. MEASURED on the wild `How to use Leverage and Margin in
        # PineScript` reference (BINANCE:BTCUSDT 30m, 2025-04-02 08:30): TV closes
        # the single open trade in ONE fill while the orphan leg split it in two.
        # A ``from_entry``-less leg is exempt -- TV deliberately re-aims that one at
        # whatever position opens next (see the adoption pass after the open fills).
        if self.exit_orders:
            for order in list(self.exit_orders.values()):
                if (order.is_market_order or order.from_entry_na
                        or order.entry_seq is not None
                        or order.order_id in self.entry_orders):
                    continue
                self._remove_order(order)

        # Skip market exit order processing if there's no open position (TradingView behavior)
        if not self.open_trades and self.exit_orders:
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
        self._resolve_filled_entry_exits()

        # Check for stop/limit orders that should be converted to market orders.
        # The leg that the gap triggered decides how the fill is priced below, so
        # it is carried over to the market loop instead of being re-derived there.
        gap_triggers: dict[_MarketOrderKey, Literal['stop', 'limit']] = {}
        gap_batch: list[Order] = []
        # Everything already in the dict was queued on an EARLIER bar; the scan
        # below appends this bar's gap batch behind it (see the arming point in
        # the market loop).
        queued_before_bar = len(self.market_orders)
        # An empty book yields nothing, so the generator is skipped rather than
        # created and immediately exhausted — every bar of an open position with
        # no resting order passes here.
        for order in (self.orderbook.iter_orders() if self.orderbook.price_levels else ()):
            # Check if the order would be filled immediately (e.g. due to a gap)
            gap_trigger = self._check_already_filled(order)
            if gap_trigger is not None:
                if order.exit_id is not None:
                    # Exit order gaps through — check if its bound entry still
                    # has open quantity on the ledger (the FIFO fill may have
                    # consumed its trade rows while the binding stays live)
                    has_open_trade = self._has_bound(order.order_id)
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

                gap_triggers[_market_order_key(order)] = gap_trigger
                gap_batch.append(order)

        # The batch fills entries first, exits after -- MEASURED on TradingView
        # (BINANCE:BTCUSDT 30m, 12/12 events): a stop entry that reverses the
        # position closes the old one whatever the two levels are, above the exit
        # stop or below it, and whichever of the two the script placed first. The
        # sort is stable, so the order book's own price walk still orders each
        # group internally.
        gap_batch.sort(key=lambda o: o.order_type == _order_type_close)
        for order in gap_batch:
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
        # Position size AND equity before any market order fills this bar. A
        # same-direction entry that only over-margins because a PRIOR same-bar
        # entry already filled (a pyramid stack) is affordable against this base —
        # TV fills it and the bar-open margin call trims the aggregate, so it is
        # not rejected. The first fill also shifts self.equity via its open P&L,
        # so the standalone affordability test must use the bar-start equity too.
        bar_start_size = self.size
        bar_start_equity = float(self.equity)
        # Sign of the position as established by an entry filled earlier in THIS
        # bar-open cycle. A later opposite entry that would reverse such a
        # same-bar position is margin-gated on BOTH legs at once (see the
        # same-bar reversal check below); a prior-bar position never set this,
        # so it keeps the normal net-margin reversal.
        same_bar_entry_sign = 0.0

        # Process Market orders. The snapshot exists because fills mutate the dict;
        # an empty one has nothing to mutate, so it is not copied at all.
        for index, order in enumerate(list(self.market_orders.values())
                                      if self.market_orders else ()):
            if index == queued_before_bar:
                # The gap batch starts here, so it is armed only now: a MARKET
                # order queued on an earlier bar still cancels an exit leg
                # outright (MEASURED, 6/6 events -- a market reversal leaves the
                # gapped exit unfilled), while a fill from inside the batch no
                # longer does (see Order.gap_committed).
                for leg in gap_batch:
                    if leg.exit_id is not None:
                        leg.gap_committed = True
            if order.cancelled:
                continue
            if order.order_type == _order_type_entry:
                if order.limit is None and order.stop is None:
                    # We need to check pyramiding and flip quantity here for market orders :-/
                    # Check pyramiding limit for entry orders adding to existing position
                    if self.sign == order.sign:
                        if lib._script.pyramiding <= self._pyramid_count():
                            # Pyramiding limit reached - don't add the order
                            self._remove_order(order)
                            continue
                    elif self.size != 0.0 and not order.skip_flip:
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

            # Genuine market fills and gap-triggered stops are slipped against the
            # order direction; a gap-triggered limit is not. Being filled here as a
            # market order does not cost a limit its price guarantee: it fills at its
            # own level or better, clamped to the open, exactly like the intrabar
            # `_check_high` / `_check_low` walk it never reached.
            # Measured on CME_MINI:ES1! 30m over 30575 bars with slippage 0 vs 1:
            # all 3057 gapped exits and 3058 gapped entries landed on the bar open
            # and the setting moved none of them.
            gap_trigger = gap_triggers.get(_market_order_key(order))
            limit = order.limit
            # Outside a COOF re-execution this IS the bar open; a gap is a bar-open
            # notion, so that branch keeps reading the open itself.
            fill_price = self._market_fill_price
            if gap_trigger == 'limit' and limit is not None:
                fill_price = max(limit, self.o) if order.size < 0 else min(limit, self.o)
            elif script.slippage > 0:
                # Slippage is in ticks, always adverse to trade direction
                # For long orders (buying), slippage increases the price
                # For short orders (selling), slippage decreases the price
                slippage_amount = syminfo.mintick * script.slippage * order.sign
                fill_price = self._market_fill_price + slippage_amount

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
                # Same-bar opposite entry reversing a position OPENED earlier in
                # this same bar-open cycle: TV margins BOTH legs at once (the
                # closing leg's margin is not freed before the opening leg is
                # gated), so the reversing entry is rejected — the first entry's
                # position is kept — when old + new margin exceeds equity.
                # Verified with a live TradingView probe on BINANCE:BTCUSDT: a
                # same-bar 0.9 BTC pair (~55% equity each leg) rejects the flip,
                # while a PRIOR-bar reversal at the same size fills (its close
                # frees margin first — the normal net check below handles that).
                if (same_bar_entry_sign != 0.0 and self.size != 0.0
                        and self.sign == same_bar_entry_sign
                        and order.sign == -same_bar_entry_sign):
                    pv = _account_point_value()
                    ratio_old = (script.margin_short if self.sign < 0
                                 else script.margin_long) / 100.0
                    ratio_new = (script.margin_short if order.sign < 0
                                 else script.margin_long) / 100.0
                    old_margin = abs(self.size) * fill_price * pv * ratio_old
                    new_margin = abs(self.size + order.size) * fill_price * pv * ratio_new
                    if (old_margin + new_margin) - self.equity > abs(self.equity) * 1e-11:
                        self._cancel_same_bar_reversal_closes(order)
                        self._remove_order(order)
                        continue
                if self._entry_exceeds_margin_after_fill(order, fill_price):
                    # The reversal's new leg (opposite the bar-start position, with a
                    # same-bar close already filled) is allowed to fill and is trimmed by
                    # the bar-open margin call below; only a fresh entry is hard-rejected.
                    is_reversal_leg = (reversal_close_filled
                                       and reversal_pre_sign != 0.0
                                       and order.sign == -reversal_pre_sign)
                    # A same-direction entry that fits against the bar-start position
                    # and only over-margins because a prior same-bar entry already
                    # filled (a pyramid stack) is likewise filled + margin-call trimmed,
                    # not hard-rejected: it cleared its placement-time margin check.
                    stacks_on_same_bar_fill = (
                        self.size != bar_start_size
                        and not self._entry_exceeds_margin_after_fill(
                            order, fill_price, base_size=bar_start_size,
                            base_equity=bar_start_equity))
                    if not is_reversal_leg and not stacks_on_same_bar_fill:
                        self._cancel_same_bar_reversal_closes(order)
                        self._remove_order(order)
                        continue

            # open → high → low → close
            if ohlc:
                self.fill_order(order, fill_price)
            # open → low → high → close
            else:
                self.fill_order(order, fill_price)

            # A same-bar close that reduced the bar-start position arms the reversal-leg
            # bypass for a subsequent opposite over-margin entry on this bar.
            if order.order_type == _order_type_close and reversal_pre_sign != 0.0:
                reversal_close_filled = True
            # A filled market entry establishes the same-bar direction that a
            # later opposite entry must both-legs-margin against (guard above).
            elif (order.order_type == _order_type_entry
                  and order.limit is None and order.stop is None):
                same_bar_entry_sign = order.sign

        # The batch is over: a leg that outlived it (its fill was a no-op) is
        # cancellable again from here on.
        for order in (self.exit_orders.values() if self.exit_orders else ()):
            order.gap_committed = False

        # Convert tick-based exit prices for entries that just filled this bar
        self._resolve_filled_entry_exits()

        # Adapt orphaned exits from rejected entries to new position (TradingView behavior)
        # When strategy.exit() is called without from_entry, TV keeps the exit even after
        # its entry is rejected by margin. The exit adapts to close any new position that opens.
        if self.open_trades and self.exit_orders:
            for order in list(self.exit_orders.values()):
                if order.is_market_order:
                    continue
                # Skip exits whose bound entry still has open quantity on the
                # ledger (they belong to the current position)
                if self._has_bound(order.order_id):
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
                        self.fill_order(adapted, fill_price)
                    else:
                        self.fill_order(adapted, fill_price)
                    filled = True
                elif limit_gap:
                    adapted.filled_by_type = 'profit'
                    if ohlc:
                        self.fill_order(adapted, self.o)
                    else:
                        self.fill_order(adapted, self.o)
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
        for order in (list(self.exit_orders.values()) if self.exit_orders else ()):
            if order.is_market_order:
                continue
            if not self._has_bound(order.order_id):
                continue
            # Check limit gap-through
            if order.limit is not None:
                limit_gap = ((order.size > 0 and self.o <= order.limit)
                             or (order.size < 0 and self.o >= order.limit))
                if limit_gap:
                    order.filled_by_type = 'profit'
                    if ohlc:
                        self.fill_order(order, self.o)
                    else:
                        self.fill_order(order, self.o)
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
                        self.fill_order(order, fill_price)
                    else:
                        self.fill_order(order, fill_price)
                    continue

        # Margin call check at OPEN — sized exactly like the intrabar (H/L)
        # liquidations: 4x the shortfall in lot units, and only when the
        # shortfall truncates below one lot does it fall back to closing a
        # single whole contract (the ``cover_lots == 0`` branch in the callee).
        # A sub-lot open overshoot (fill price a tick above the sizing price)
        # therefore still trims exactly 1.0 contract, while a multi-lot
        # overshoot trims the fractional cover TV's exported trades show
        # (BINANCE:BTCUSDT 30m RCI Strategy: a 90-lot open shortfall trims
        # 0.0038 BTC, not a whole contract). The sign gates mirror the callee's
        # own direction guards (a liquidation never reverses the position, so
        # the second direction stays a no-op after the first fires).
        if self.sign < 0:
            self._check_margin_call(self.o, for_short=True, at_open=True)
        elif self.sign > 0:
            self._check_margin_call(self.o, for_short=False, at_open=True)

    def _process_limit_stop_orders(self, ohlc: bool, walked: int = -1):
        """Phase 2: Process limit/stop/trailing orders with margin checks at H/L.

        :param ohlc: True when the emulator walks open -> high -> low -> close.
        :param walked: Path node the emulator already stands at, for a
            ``calc_on_order_fills`` re-execution. The legs BEHIND it are history:
            an order the re-run body placed cannot fill on a price the bar
            reached before the fill that triggered the re-run. Measured on
            `Donchian Breakout Strategy` (BINANCE:BTCUSDT 30m, 2025-01-24 14:30,
            an open -> low -> high -> close bar): the stop entry fills on the
            rising leg, the re-run re-issues its ``strategy.exit`` stop at the
            completed bar's lower Donchian band, and TradingView KEEPS the
            position because the low is already past. Only the cursor's OWN leg
            is walked again — the emulator is still inside it. ``-1`` (the
            default, and what the bar-magnifier walk passes, since real sub-bars
            carry that bookkeeping in ``process_orders_magnified``'s ``start``)
            walks the whole path.
        """
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
        trail_close_leg: list[Order] = []
        # Trailing legs still waiting for their entry cannot be walked here: a
        # tick-based one has no ``trail_price`` to be indexed at yet, and an
        # explicit one is in the book but inactive. An entry filling intrabar
        # activates them mid-walk instead (see ``_activate_trails_on_fill``).
        trail_awaiting: set[Order] = set()
        for order in (self.exit_orders.values() if self.exit_orders else ()):
            if ((order.trail_price is not None or order.trail_points_ticks is not None)
                    and not order.cancelled and self._exit_awaits_entry(order)):
                trail_awaiting.add(order)
        # Iterate a snapshot since fills mutate the order book; an order indexed at
        # several price levels is yielded once per level, so dedupe by identity.
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
            self._walk_node = 1
            if self.orderbook.price_levels and walked <= 1:
                self._walk_leg(self.o, self.h, rising=True, ohlc=ohlc,
                               trail_awaiting=trail_awaiting, trail_close_leg=trail_close_leg)

            mc_deferred = self.sign < 0 and self._check_margin_call(self.h, for_short=True)
            if not mc_deferred:
                # A short's unfavorable extreme is the high the rising leg just
                # reached — the node ``max_intraday_loss`` closes at.
                if self.sign < 0 and self._check_intraday_loss(self.h):
                    return
                if self.sign > 0:
                    # The favorable extreme is checked AFTER this leg's fills
                    # too: pyramid stop entries filling on the way up can push
                    # the long past its margin at H — the mirror of the
                    # down-walk's post-leg check below. Measured against TV on
                    # Rocket Grid Algorithm 2026-08-25 02:30 (an up-walk bar):
                    # TV liquidates at H-slip right after the rising leg's
                    # fills, then runs the second stage at the low.
                    self._check_margin_call(self.h, for_short=False, can_defer=False)
                # The checkpoint at the position's FAVORABLE extreme runs
                # before this leg's fills. Under the float trigger it is a
                # no-op (available funds only improve toward the favorable
                # side at margin <= 100%), but the >=1e7 integer-tick trigger
                # can trip there: TV liquidated one contract of a LONG at
                # H=120300 (Hybrid 2025-10-02 16:00) before the exit limit at
                # 120290.7 — lower on the same leg — filled the rest.
                self._walk_node = 2
                if self.sign < 0:
                    self._check_margin_call(self.l, for_short=True, can_defer=False)

                # open -> low (descending: the level nearest the open fills first)
                if self.orderbook.price_levels and walked <= 2:
                    self._walk_leg(self.o, self.l, rising=False, ohlc=ohlc,
                                   trail_awaiting=trail_awaiting, trail_close_leg=trail_close_leg)

                if self.sign > 0:
                    self._check_margin_call(self.l, for_short=False, can_defer=False)
                    # A long's unfavorable extreme, reached by the descending leg.
                    if self._check_intraday_loss(self.l):
                        return

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
                self._walk_node = 3
                if self.orderbook.price_levels:
                    for order in self.orderbook.iter_orders(min_price=self.l, max_price=self.c):
                        if self._check_close_leg_up(order):
                            continue

        # Process orders: open → low → high → close
        else:
            # open -> low (descending: the level nearest the open fills first)
            self._walk_node = 1
            if self.orderbook.price_levels and walked <= 1:
                self._walk_leg(self.o, self.l, rising=False, ohlc=ohlc,
                               trail_awaiting=trail_awaiting, trail_close_leg=trail_close_leg)

            mc_deferred = self.sign > 0 and self._check_margin_call(self.l, for_short=False)
            if not mc_deferred:
                # A long's unfavorable extreme is the low the descending leg just
                # reached — the node ``max_intraday_loss`` closes at.
                if self.sign > 0 and self._check_intraday_loss(self.l):
                    return
                # Favorable-extreme checkpoint before this leg's fills — see
                # the mirrored comment in the OHLC branch (TV-verified on the
                # Hybrid 2025-10-02 16:00 long margin call at the high).
                self._walk_node = 2
                if self.sign > 0:
                    self._check_margin_call(self.h, for_short=False, can_defer=False)

                # open -> high
                if self.orderbook.price_levels and walked <= 2:
                    self._walk_leg(self.o, self.h, rising=True, ohlc=ohlc,
                                   trail_awaiting=trail_awaiting, trail_close_leg=trail_close_leg)

                if self.sign < 0:
                    self._check_margin_call(self.h, for_short=True, can_defer=False)
                elif self.sign > 0:
                    # The favorable extreme is checked a SECOND time, now with
                    # what the leg filled: an entry that fills on the way up
                    # raises the long's margin requirement above what the
                    # pre-leg checkpoint could see. Measured against TV on
                    # Rocket Grid Algorithm 2025-11-11 01:00, where two pyramid
                    # stop entries fill on the rising leg and TV then liquidates
                    # 0.51464 of the grown 0.79946-contract position at H-slip
                    # (107499.95) -- more than the whole pre-leg position.
                    self._check_margin_call(self.h, for_short=False, can_defer=False)
                # A short's unfavorable extreme, reached by the rising leg.
                if self.sign < 0 and self._check_intraday_loss(self.h):
                    return

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
                self._walk_node = 3
                if self.orderbook.price_levels:
                    for order in self.orderbook.iter_orders(max_price=self.h, min_price=self.c, desc=True):
                        if self._check_close_leg_down(order):
                            continue

    def _finalize_bar_pnl(self):
        """Phase 3: Calculate P&L, drawdown, runup, and cumulative stats."""
        # Calculate average entry price, unrealized P&L, drawdown and runup...
        if self.open_trades:
            # Account-currency value of a 1.0-point move on 1 contract. Re-read every bar,
            # so an open position's unrealized P&L and its run-up/draw-down extremes are
            # marked at the rate of the bar they occur on. Identity latch inlined as in
            # ``_check_margin_call`` — every bar with an open position lands here.
            pv = (syminfo.pointvalue if _conv_identity_script is lib._script
                  else _account_point_value())

            # Unrealized P&L. The mark rides the FILL tick grid (``_tick_snap``),
            # not the OHLC quantization ``self.c`` carries: measured on the wild
            # "Built-in Kelly ratio" (currency=USD on BINANCE:BTCUSDT 30m), the
            # snapped close reproduces all 10892 in-position bars of
            # ``strategy.openprofit`` bit-exact, while ``self.c`` itself misses
            # 1378 of them. The two forms are algebraically equal and differ by
            # one ULP of the price (94208.43 against 94208.43000000001), which
            # the position size scales into a ~1.5e-12 offset that never washes
            # out -- ``strategy.equity`` is built on it.
            self.openprofit = self.size * (_tick_snap(self.c) - self.avg_price) * pv

            # Calculate open drawdowns and runups
            for trade in self.open_trades:
                # Profit of trade
                trade.profit = trade.size * (self.c - trade.entry_price) * pv - 2 * trade.commission

                # P/L from high/low to calculate drawdown and runup. The
                # POSITION-level summation below measures every open leg against
                # the position average — that is what reproduces
                # ``strategy.max_drawdown``.
                hprofit = trade.size * (self.h - self.avg_price) * pv
                lprofit = trade.size * (self.l - self.avg_price) * pv
                drawdown = -min(hprofit, lprofit, 0.0)
                runup = max(hprofit, lprofit, 0.0)

                # The PER-TRADE excursion is a different quantity: measured from
                # the trade's OWN entry price, with the entry fee counted into
                # it. Measured on BINANCE:BTCUSDT 30m (pyramiding 3, 0.05%
                # commission): trade 0 (long 1 at 93761.90000000001, worst low
                # 93500.0, best high 94509.42) reports max_drawdown 308.78095 =
                # (93761.9 - 93500.0) + 46.88095 and max_runup 700.63905 =
                # (94509.42 - 93761.9) - 46.88095; trade 1 (long 2 at
                # 94098.90000000001) reports 1291.8989 = 2*(94098.9 - 93500.0) +
                # 94.0989 and a max_runup of 0 (it never traded above its own
                # entry). Against the position average both go wrong the moment
                # a second leg shifts it.
                t_hprofit = trade.size * (self.h - trade.entry_price) * pv - trade.commission
                t_lprofit = trade.size * (self.l - trade.entry_price) * pv - trade.commission
                t_drawdown = -min(t_hprofit, t_lprofit, 0.0)
                if t_drawdown > trade.max_drawdown:
                    trade.max_drawdown = t_drawdown
                t_runup = max(t_hprofit, t_lprofit, 0.0)
                if t_runup > trade.max_runup:
                    trade.max_runup = t_runup

                # The percentages divide by the trade's TOTAL ENTRY COST --
                # position value plus the fee paid to open it -- exactly like
                # ``profit_percent`` (see the fill loop). ``trade.commission``
                # still holds the entry leg alone while the trade is open.
                entry_cost = abs(trade.size) * trade.entry_price * pv + trade.commission
                if entry_cost > 0:
                    trade.max_drawdown_percent = trade.max_drawdown / entry_cost * 100.0
                    trade.max_runup_percent = trade.max_runup / entry_cost * 100.0

                # Drawdown summ runup summ
                self.drawdown_summ += drawdown
                self.runup_summ += runup

        # Max drawdown and runup. Both excursions run between a REALIZED equity
        # endpoint and a mark-to-market one: the drop is measured from the
        # realized high-water mark down to the worst mark of the path, the rise
        # from the run-up anchor (see the entry-fill branch) up to the best
        # mark. Open paper profit never lifts the high-water mark -- measured on
        # BINANCE:BTCUSDT 30m (pyramiding 3, 0.05% commission), a bar whose
        # intrabar mark peaked at 1000700.63905 still reports the next drawdown
        # against 1000000.0. The summs carry the GROSS mark, commission-free:
        # every fee is already inside netprofit through open_commission, so
        # charging it per leg would book it twice.
        initial = float(lib._script.initial_capital)
        realized_equity = initial + self.netprofit
        # The high-water mark takes this bar's own realized equity first: the
        # closes that moved it happened DURING the bar, so the rest of the path
        # is measured against the level they left behind.
        if realized_equity > self.max_realized_equity:
            self.max_realized_equity = realized_equity

        peak = self._max_drawdown_reference()
        equity_drawdown = peak - realized_equity + self.drawdown_summ
        if equity_drawdown > self.max_drawdown:
            self.max_drawdown = equity_drawdown
        trough = float(self.min_equity)
        equity_runup = realized_equity + self.runup_summ - trough
        if equity_runup > self.max_runup:
            self.max_runup = equity_runup
        # The percentages are NOT the currency maxima over the initial capital:
        # each excursion is expressed against its HIGHER endpoint -- the peak it
        # fell from, the top it rose to -- so a percent maximum can be set on a
        # different bar than the currency one (measured: a 634150.6465686325
        # drawdown reports 60.7912228982302%, not 63.415%).
        if peak > 0.0:
            drawdown_percent = equity_drawdown / peak * 100.0
            if drawdown_percent > self.max_drawdown_percent:
                self.max_drawdown_percent = drawdown_percent
        top = trough + equity_runup
        if top > 0.0:
            runup_percent = equity_runup / top * 100.0
            if runup_percent > self.max_runup_percent:
                self.max_runup_percent = runup_percent

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

    def process_orders_at_close(self):
        """
        Optional post-script pass that fills current-bar-submitted orders at the bar's
        CLOSE — enabled by `script.process_orders_on_close=True`.

        When the flag is set, orders placed during the strategy's bar calculation get
        an additional fill attempt at the bar close, instead of waiting for the next
        bar's open. This covers BOTH:
          - Market orders: trivially executable at close.
          - Limit/stop orders: executable when the close has reached/crossed the trigger
            price. (Non-current-bar limit/stop orders already had their fair shake in
            `_process_limit_stop_orders` during the H/L walk.)
        Tick-based exit orders submitted on the current bar (`strategy.exit(profit=...,
        loss=...)`) still carry raw `profit_ticks` / `loss_ticks` when their entry has
        not filled yet, so the close pass resolves them through `_resolve_tick_exit`
        first — the trigger check needs concrete `limit` / `stop` prices.

        Fill price in every case is `self.c` — price-based orders fill where their
        limit or stop price is hit on the close, with no trigger-price snap on the
        close pass. Slippage matches the rest of the engine: applied to market and
        stop-triggered fills, NOT to limit-triggered fills, which are guaranteed to
        fill at the limit price or better. `filled_by_type` is set on the
        triggering order so `_fill_order` can attach the right exit comment.

        Bookkeeping note: `_finalize_bar_pnl()` already ran in `process_orders()` for the
        same bar. Re-running it here would double-count `cum_profit` for already-settled
        `new_closed_trades`. Instead, we only settle cumulative stats for trades that
        close DURING this pass (`_settle_close_pass_trades`). For positions opened right
        at the close, the bar has no remaining H/L range — their per-trade `profit` /
        `max_drawdown_percent` are intentionally left for the next bar's
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
                # Exits submitted during this bar's main() still carry raw tick
                # offsets — the trigger check below needs concrete price levels.
                entry_price = self._entry_fill_price(order.order_id, order.entry_seq)
                if entry_price is not None:
                    self._resolve_tick_exit(order, entry_price)
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

        closed_before = len(self.new_closed_trades)

        def _apply_fill(order: Order, trigger: str) -> None:
            """Run the per-candidate fill, mirroring `_process_at_bar_open`."""
            if order.cancelled:
                return
            if order.order_type == _order_type_entry:
                if order.limit is None and order.stop is None:
                    # Pyramiding and flip-quantity handling — mirror `_process_at_bar_open`.
                    if self.sign == order.sign:
                        if script.pyramiding <= self._pyramid_count():
                            self._remove_order(order)
                            return
                    elif self.size != 0.0 and not order.skip_flip:
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

            self.fill_order(order, fill_price)

        # Phase 1: fill the initial candidates (market entries, previously-open
        # tick exits, current-bar limit/stop orders already executable at close).
        for order, trigger in candidates:
            _apply_fill(order, trigger)

        # Phase 2: a current-bar entry may have just filled in Phase 1, opening a
        # trade whose `entry_price` lets us resolve a same-bar `strategy.exit(...,
        # profit=..., loss=...)` order whose ticks were unresolved before Phase 1.
        # Mirror `_resolve_filled_entry_exits` — re-scan exit_orders for
        # current-bar tick exits, materialize, and fill any newly executable.
        for order in list(self.exit_orders.values()):
            oid = id(order)
            if oid in seen or order.cancelled or order.bar_index != current_bar:
                continue
            if order.is_market_order:
                continue
            if order.profit_ticks is None and order.loss_ticks is None:
                continue
            entry_price = self._entry_fill_price(order.order_id, order.entry_seq)
            if entry_price is not None:
                self._resolve_tick_exit(order, entry_price)
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

        # Incrementally settle only the trades that closed during the close pass;
        # everything settled by `process_orders()` earlier in this bar stays untouched.
        if len(self.new_closed_trades) > closed_before:
            self._settle_close_pass_trades(closed_before)

    def _settle_close_pass_trades(self, closed_before: int):
        """
        Apply cumulative bookkeeping for trades that closed during `process_orders_at_close`.

        Mirrors the per-closed-trade cum_profit update tail of `_finalize_bar_pnl()`,
        but only for new_closed_trades appended after the close pass started — the
        earlier entries were already settled when `process_orders()` ran for this same
        bar. Position-level max_drawdown / max_runup is intentionally NOT re-rolled
        here: the bar's H/L drawdown_summ / runup_summ contribution was already booked
        by `_finalize_bar_pnl()` against the open trades, which include the trades that
        close here since they were opened on this same bar.
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

    def settle_immediate_closes(self):
        """
        Fill the strategy.close/close_all(immediately=True) orders enqueued during
        this bar's body, at the bar close.

        Runs right AFTER the body (before the bar's output/equity bookkeeping), so
        the whole position stays coherent — fully open — for the rest of the bar and
        every ``strategy.*`` series (``position_size``, ``position_avg_price``,
        ``netprofit``, ``equity``, ``opentrades`` …) reads its pre-close value —
        the same as in broker mode, where an immediate close does not take effect
        until after the script.

        The fill is a market order at the bar close, so the synthetic slippage
        applies against the position and the SLIPPED price is what the trade
        carries — TradingView's trade list exports it too, and its P&L is exactly
        that price's arithmetic (measured on the Triple CCI and Bollinger Bands
        Enhanced references, BINANCE:BTCUSDT 240, slippage 5: every immediate
        close exports ``close - 5 * mintick`` for a long).
        """
        orders = self._deferred_immediate_closes
        if not orders:
            return
        self._deferred_immediate_closes = []  # drain-once / re-entrancy guard
        for order in orders:
            if self.size == 0.0:
                # An earlier buffered close already flattened. TV treats a close
                # against a zero position as a no-op; drop the order so it cannot
                # zombie-fill on a later bar — ``_fill_order`` early-returns on a
                # zero-size close WITHOUT removing it from the order books.
                self._remove_order(order)
                continue
            closed_before = len(self.new_closed_trades)
            price = self.c
            slippage = lib._script.slippage
            if slippage > 0:
                # Closing a long sells (worse = lower), closing a short buys
                # (worse = higher).
                price += -syminfo.mintick * slippage if self.sign > 0 \
                    else syminfo.mintick * slippage
            self.fill_order(order, price)
            self._settle_close_pass_trades(closed_before)

    def _discard_deferred_immediate_closes(self):
        """
        Cancel immediate closes left buffered by a throwaway COOF trial body run.

        Called at the top of ``process_orders``/``process_orders_magnified``. In
        steady state the buffer is already empty (``settle_immediate_closes``
        drained it after the previous body); this only fires between
        ``calc_on_order_fills`` re-executions, where a trial run's enqueued close
        must be undone — the position-side analog of the restored ``var`` state —
        before the next order-processing pass could wrongly fill it at the bar open.
        """
        if not self._deferred_immediate_closes:
            return
        for order in self._deferred_immediate_closes:
            self._remove_order(order)
        self._deferred_immediate_closes = []

    def process_orders_magnified(self, sub_bars: list[OHLCV], aggregated: OHLCV,
                                 start: int = 0):
        """
        Process orders using bar magnifier — check fills against each sub-bar's OHLC.

        Phase 1 (at-open) runs once using first sub-bar.
        Phase 2 (limit/stop) runs on each sub-bar sequentially.
        Phase 3 (P&L) runs once using aggregated bar values.

        :param sub_bars: The chart bar's lower-timeframe bars, in time order.
        :param aggregated: The chart bar itself, for the P&L phase.
        :param start: Sub-bar to resume the walk at. A calc_on_order_fills
            re-execution passes the sub-bar its triggering fill happened in, so
            the orders it places are offered the rest of the bar and not the
            sub-bars that are already behind them. :attr:`_path_node` carries
            that index — the same bookkeeping the assumed path uses for its own
            nodes, over sub-bar indexes here.
        """
        # ``lib.math.round_to_mintick`` inlined — sub-bar OHLC are plain floats, and
        # this runs per sub-bar. Expression shape must stay left-to-right (see the
        # bit-parity note in ``lib/math.py``).
        mintick = syminfo.mintick
        minmove = syminfo.minmove
        pricescale = syminfo.pricescale
        self.drawdown_summ = self.runup_summ = 0.0
        self.new_closed_trades.clear()
        # Undo any immediate close a COOF trial body run enqueued (position-side
        # analog of the restored ``var`` state); no-op in the common case.
        self._discard_deferred_immediate_closes()

        if start <= 0:
            self._path_node = 0
            # Setup from first sub-bar (= chart bar open)
            first = sub_bars[0]
            self.o = int(first.open / mintick + 0.5) * minmove / pricescale
            self.h = int(first.high / mintick + 0.5) * minmove / pricescale
            self.l = int(first.low / mintick + 0.5) * minmove / pricescale
            # Use aggregated close for margin deferral checks
            self.c = int(aggregated.close / mintick + 0.5) * minmove / pricescale

            # Phase 1: at-open processing (gap detection, market orders, margin at open)
            ohlc = self.h - self.o < self.o - self.l
            # Real sub-bars replace the assumed intrabar path, so a COOF pass claims
            # no path point here — a market order fills at the sub-bar open like any
            # other. The chart bar's open is behind a resuming pass, which is why
            # this whole phase belongs to the first one only.
            self._market_fill_price = self.o
            self._walk_node = 0
            self._process_at_bar_open(ohlc)

        # Phase 2: process limit/stop orders on each sub-bar
        for idx in range(max(start, 0), len(sub_bars)):
            sub_bar = sub_bars[idx]
            self._walk_node = idx
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

# Decimal context of TV's cash rounding. Ten significant digits, half-up,
# applied to the double's shortest repr — see _round_cash.
_CASH_CTX = Context(prec=10, rounding=ROUND_HALF_UP)
# Exact multiply for the split shares: two shortest doubles are at most 17
# digits each, so 40 digits keeps the product unrounded (the thread context's
# 28 would clip it).
_SPLIT_CTX = Context(prec=40)
_DEC_CENT = Decimal('0.01')


# noinspection PyProtectedMember
def _round_cash(amount: float) -> float:
    """
    Round a booked cash flow to ten significant digits.

    :param amount: The raw cash amount
    :return: The amount rounded half-up to 10 significant digits
    """
    # TV books each commission cash flow rounded HALF-UP to TEN SIGNIFICANT
    # DIGITS; realized gross P&L stays raw. Measured on SuperTrend STRATEGY
    # d6fba11d (BINANCE:BTCUSDT 30m): netprofit after the first entry is
    # exactly -1499.774913 (raw commission 1499.7749132048700), all 186
    # entry/exit netprofit steps reproduce to sub-ULP, and the 2025-08-06 step
    # lands exactly one unit below the half-even result, pinning half-up.
    # The digit count — not a fixed 1e-6 grid — is what Acrypto - Weighted
    # Strategy pins: with currency.USD on a USDT-quoted symbol its first
    # commission is 0.750695772375 USDT * 0.99789 = 0.7491118042952888, and TV
    # books exactly 0.7491118043. A 1e-6 grid would give 0.749112 and a grid in
    # the symbol's own currency 0.74911203144; both are wrong, and at the
    # SuperTrend magnitude ten digits coincide with the 1e-6 grid.
    # The rounding operates on the double's shortest decimal repr, BigDecimal
    # style: a 2830/2830 bit-exact single-order probe run (CommProbe3,
    # BINANCE:BTCUSDT 30m) pins that exact-half raws resolve by the repr tail,
    # which a float floor(x*scale + 0.5) misrounds on either side.
    if amount == 0.0 or not math.isfinite(amount):
        return amount
    return float(_CASH_CTX.create_decimal(repr(amount)))


# noinspection PyProtectedMember
def _book_commission(booking: list, qty: float, price: float,
                     commission_value: float, is_percent: bool) -> float:
    """
    Add a fill leg to an order's commission pool and return the amount to realize now.

    :param booking: The order's ``[Decimal qty_total, booked_total, leg_qtys]`` pool
    :param qty: The absolute quantity of this leg
    :param price: The order's fill price
    :param commission_value: The strategy's commission value
    :param is_percent: True for percent commission, False for cash per contract
    :return: The incremental amount to book, so the pool's total stays rounded
    """
    # TV charges ONE commission per order over the order's total quantity — a
    # reversal executes as two PyneCore fills but is one TV order, and an exit
    # covering several pyramided trades is likewise one order. Probe-measured
    # on CommProbe1/3/4 (BINANCE:BTCUSDT 30m, 2830 single orders bit-exact,
    # 2818/2829 reversals): the percent rate is the fill price times the rate
    # in EXACT DECIMAL arithmetic, converted to a double, and the order total
    # is round_cash(rate * qty_total) with qty_total the DECIMAL sum of the leg
    # quantities (float sums lose the lot grid: 0.01333 + 0.0137 != 0.02703).
    # The account-currency conversion multiplies the double product afterwards
    # (measured net-positive, zero regressions, on Acrypto - Weighted Strategy).
    # The rounded total is then SPLIT back over the legs in qty proportion and
    # each share is re-rounded on its own scale; what netprofit sees is the sum
    # of the shares, which sits one grid step off the rounded total whenever
    # the share roundings do not cancel (CommProbe4 pins the split on 2819/2829
    # reversals via the per-trade commission fields). The pool books the
    # DIFFERENCE between that running target and what was booked already, so a
    # single-leg order rounds exactly on its own.
    # The order's own quantity is the split base whenever it is known ahead of
    # the fills (booking[3], stamped for a reversal). Splitting over the running
    # leg sum instead prices the first leg as if it were the whole order and
    # dumps the difference on the last one: measured on the wild "Built-in Kelly
    # ratio" (currency=USD on BINANCE:BTCUSDT 30m), the 2025-01-05 22:00
    # reversal books its closing leg round10(rate * 0.00723) = 0.7121963641
    # where TV's share of the full 0.02038 order is 0.7121963643, and the 2e-10
    # lands on the opening leg -- which is trade #6's entry fee, so TV's plotted
    # grossprofit runs 2.0003e-10 ahead from that trade on. The order total and
    # the sum of the shares are the same either way, so only the per-leg
    # attribution -- what each closed trade reports as its own fee -- moves.
    booking[0] += Decimal(repr(qty))
    booking[2].append(qty)
    qty_total = booking[3] or float(booking[0])
    if is_percent:
        # The account-currency conversion happens on the PRICE, as a plain
        # double multiply, and the cash rounding lands on the double of the
        # decimal price*pct product (CommProbe5, currency.USD on
        # BINANCE:BTCUSDT: 2829/2829 reversal + 2830/2830 single orders
        # reproduce both per-trade commission fields bit-exactly). Rate-level
        # decimal fx placements fail on the exact repr ties the float price
        # conversion breaks the other way (4 orders in the probe corpus).
        # Without conversion the raw double rate is the measured law, so the
        # extra rounding is skipped.
        # The caller sampled pv on this bar, but a non-converting run takes the
        # identity latch in _account_point_value and never refreshes _conv_safe --
        # in a process running several scripts that global can still hold the rate
        # another, converting script sampled. The latch itself is the authority:
        # when it holds this script, this run does not convert and fx IS 1.0.
        fx = 1.0 if _conv_identity_script is lib._script else _conv_safe
        base = price * fx if fx != 1.0 else price
        rate = float(Decimal(repr(base)) * Decimal(repr(commission_value)) * _DEC_CENT)
        pointvalue = syminfo.pointvalue
        if pointvalue != 1.0:
            rate *= pointvalue
        if fx != 1.0:
            rate = _round_cash(rate)
        total = _round_cash(rate * qty_total)
    else:
        total = _round_cash(commission_value * qty_total)
    return _apply_booking(booking, total, qty_total)


def _book_flat_commission(booking: list, qty: float, commission_value: float) -> float:
    """
    Add a fill leg to a ``cash_per_order`` commission pool and return the amount to realize now.

    :param booking: The order's ``[Decimal qty_total, booked_total, leg_qtys, order_qty]`` pool
    :param qty: The absolute quantity of this leg
    :param commission_value: The strategy's flat per-order fee
    :return: The incremental amount to book, so the pool's total stays rounded
    """
    # A flat fee is charged ONCE per TradingView order, independent of the
    # quantity — so a reversal, which is one TV order executed here as two
    # fills, pays it once and splits it over the two legs in qty proportion.
    # Probe-measured on CommCashOrderProbe (BINANCE:BTCUSDT 30m, fee 10,
    # 1-contract long -> short -> long -> close_all): netprofit steps by
    # -10 per reversal (total commission 40 over 4 orders), and the closed
    # trades report 15 / 10 / 15 — the two reversals each split their single
    # fee 5/5 between the closing and the opening leg.
    # Unlike the size-proportional modes, the total does not grow with the legs,
    # so the closing leg cannot derive its share from its own quantity: the pool
    # carries the whole order's quantity (booking[3]), stamped at the split.
    booking[0] += Decimal(repr(qty))
    booking[2].append(qty)
    qty_total = booking[3] or float(booking[0])
    return _apply_booking(booking, _round_cash(commission_value), qty_total)


def _split_lots(leg_qtys: list, qty_total: float) -> tuple[list, Decimal]:
    """
    Express an order's leg quantities and its total as exact lot counts.

    :param leg_qtys: The booked leg quantities in contracts
    :param qty_total: The order's total quantity in contracts
    :return: ``(leg lot counts, total lot count)``, or the shortest reprs when
             the counts are not recoverable
    """
    # The split ratio is a ratio of LOT COUNTS, not of the materialized doubles:
    # TV carries every quantity as an integer count of 1e-8 units (see
    # ``_size_units``), so a reversal that closes and opens the same size splits
    # its fee exactly in half however the doubles landed. Measured on the wild
    # "Built-in Kelly ratio" (currency=USD on BINANCE:BTCUSDT 30m): its
    # 2025-02-16 07:30 reversal is 14000 units against 14000, whose exact half
    # 0.013529148375 rounds UP to 0.01352914838 per leg -- while the contract
    # doubles give 0.00014000000000000001 / 0.00028000000000000003, a quotient a
    # hair under one half that rounds the tie down and leaves grossprofit 1e-11
    # short for the rest of the run.
    rfactor = syminfo._size_round_factor  # noqa
    if rfactor <= 1e8:
        units = [q / 1e-8 for q in leg_qtys]
        total_units = qty_total / 1e-8
        if (all(abs(u) < 9.223372036854776e18 for u in units)
                and abs(total_units) < 9.223372036854776e18):
            return [Decimal(round(u)) for u in units], Decimal(round(total_units))
    return [Decimal(repr(q)) for q in leg_qtys], Decimal(repr(qty_total))


def _apply_booking(booking: list, total: float, qty_total: float) -> float:
    """
    Split an order's rounded commission over its legs and return what is still unbooked.

    :param booking: The order's ``[Decimal qty_total, booked_total, leg_qtys, order_qty]`` pool
    :param total: The order's rounded total commission
    :param qty_total: The quantity the total is spread over
    :return: The incremental amount to book
    """
    if len(booking[2]) == 1 and booking[2][0] == qty_total:
        target = total
    else:
        # The split is decimal division rounded straight to the cash grid, over
        # the legs' exact lot counts (see _split_lots): a double quotient falls a
        # hair short of a split-half tie and misrounds it down (Acrypto -
        # Weighted Strategy 2026-06-30, a perfect half-half reversal; no probe
        # order distinguishes the two).
        total_dec = Decimal(repr(total))
        legs_dec, qty_dec = _split_lots(booking[2], qty_total)
        target = 0.0
        for leg_dec in legs_dec:
            share = _SPLIT_CTX.multiply(total_dec, leg_dec)
            target += float(_CASH_CTX.divide(share, qty_dec))
    amount = target - booking[1]
    booking[1] = target
    return amount


def _tick_snap(price: float) -> float:
    """
    Snap a fill price onto the symbol's tick grid.

    :param price: The raw fill price
    :return: The price rounded half-up to the nearest tick
    """
    # TV books a fill as a tick count scaled back into price units, so the
    # stored price can sit an ULP away from the raw trigger price. Measured on
    # BINANCE:BTCUSDT 30m (mintick 0.01): 3408 of 3408 opentrades.entry_price
    # and 2840 of 2840 closedtrades.exit_price values reproduce with
    # `round(price / mintick) * mintick`, and closedtrades.profit is exactly
    # `size * (exit - entry)` on those snapped prices. The algebraically equal
    # `ticks * minmove / pricescale` form (what round_to_mintick uses) misses
    # 438 of the 3408.
    if not (price == price):  # is_na_arg
        return price
    mintick = syminfo.mintick
    return math.floor(price / mintick + 0.5) * mintick


_explicit_qty_grid: tuple[float, Decimal] | None = None


def _size_units(qty: PyneFloat) -> PyneFloat:
    """
    Materialize a quantity on TradingView's 1e-8 unit grid.

    :param qty: The quantity in contracts
    :return: The quantity as an exact multiple of 1e-8
    """
    # TV carries every quantity as an integer count of 1e-8 units and hands out
    # ``units * 1e-8``. Measured on BINANCE:BTCUSDT 30m (pyramiding 2, full TP
    # cascade): all 19581 in-position bars report position_size and every
    # opentrades.size as an exact unit multiple -- including the ones that
    # differ from the shortest decimal (104 lots of 1e-5 is
    # 0.0010400000000000001, not the 0.00104 that 104 / 1e5 yields). The run's
    # 91 distinct lot counts pin the materialization to this grid alone: no
    # ``n / 10**k`` or ``n * 10**-k`` pairing reproduces the whole table, and
    # the grid also explains the position/open-trade sums, which are integer
    # unit arithmetic. It is not a crypto quirk: CAPITALCOM:EURUSD (lot step
    # 0.1), CAPITALCOM:US500 (0.1) and OANDA:XAUUSD (0.01) report the same grid
    # on every one of their 141940 in-position quantities, 13-15% of which the
    # lot grid gets wrong (TV says 5294.900000000001, not 5294.9). On a
    # whole-contract symbol every quantity is an integer, which is always an
    # exact unit multiple, so this is a no-op there.
    #
    # The trip to units is a DIVISION by 1e-8, not a multiplication by 1e8: the
    # two differ because ``fl(1e-8)`` is not the reciprocal of ``fl(1e8)``.
    # Below 2**53 units both forms agree, but on BINANCE:SHIBUSDT 1h (lot step
    # 1e-3, explicit quantities from 4.5e7 to 2.3e10, so 4.5e15..2.3e18 units)
    # only the division reproduces TV: it is exact on all 1158 fresh legs and
    # all 70366 reported quantities, while ``* 1e8`` drifts by +-1 ULP on 12% of
    # them. Above 2**53 the round() is a no-op -- the double already is an
    # integer -- so the grid keeps working all the way to the long range TV's
    # own unit counter is bounded by; past that the quantity is left alone.
    rfactor = syminfo._size_round_factor  # noqa
    if rfactor > 1e8:
        return qty
    units = qty / 1e-8
    if not (abs(units) < 9.223372036854776e18):  # NaN or beyond TV's long range
        return qty
    return round(units) * 1e-8


def _size_add(qty_a: PyneFloat, qty_b: PyneFloat) -> PyneFloat:
    """
    Combine two on-grid quantities the way TradingView's long lot ledger does.

    :param qty_a: First quantity in contracts (an exact lot multiple)
    :param qty_b: Second quantity in contracts (an exact lot multiple)
    :return: The sum as an exact multiple of the lot step
    """
    # TV books every position change as a LONG count of lots and materializes
    # ``lots * mincontract``. Measured on BINANCE:SHIBUSDT 1h (explicit
    # quantities up to 2.3e10 contracts, so far beyond 2**53 units): the
    # partial-close remainders, the second-generation remainders and every
    # position_size reproduce 7899/7899 ONLY from the exact lot difference --
    # the double subtraction of the materialized sizes drifts +-1 ULP on ~40%
    # of them, and summing the materialized per-trade doubles gets
    # position_size wrong on ~45% (TV materializes the TOTAL lot count
    # instead). Below 2**53 units the double sum snapped by ``_size_units``
    # lands on the same grid point, so this is inert there. Both inputs are
    # exact lot multiples (every fill goes through ``_size_round`` /
    # ``_explicit_qty_round``), which makes the lot counts recoverable from
    # the doubles as long as they fit the 2**53 integer range; past that the
    # recovery itself would lose lots, so fall back to the snapped double sum.
    rfactor = syminfo._size_round_factor  # noqa
    if rfactor > 1e8:
        return qty_a + qty_b
    lots_a = qty_a * rfactor
    lots_b = qty_b * rfactor
    if not (abs(lots_a) < 9007199254740992.0 and abs(lots_b) < 9007199254740992.0):
        return _size_units(qty_a + qty_b)  # NaN or beyond exact lot recovery
    return _size_units((round(lots_a) + round(lots_b)) * (1.0 / rfactor))


def _size_floor(qty: PyneFloat) -> PyneFloat:
    """
    Plain lot floor for a money-derived (default-sized) quantity.

    :param qty: The quantity in contracts, fresh from the budget division
    :return: The quantity floored to the lot grid
    """
    # A default-sized entry floors the budget division with NO near-integer
    # snap: TV divides the sig10-quantized budget by the unit cost in double
    # and truncates whatever comes out, so a quotient one ULP below a whole
    # lot count loses that lot (BINANCE:SHIBUSDT bar 14041: 10328.61155 /
    # 1.285e-05 / 0.001 = 803782999999.9999, TV sizes 803782999999 lots).
    # ``_size_round``'s snap exists for internally derived sizes that are
    # already lot-exact and only dirtied by the scaling multiply; the budget
    # quotient is not one of those.
    if not (qty == qty):  # is_na_arg
        return na_float
    rfactor = syminfo._size_round_factor  # noqa
    lots = int(abs(qty) * rfactor)
    if lots == 0:
        return 0.0
    sign = 1 if qty > 0 else -1
    return _size_units(sign * lots * (1.0 / rfactor))


def _explicit_qty_round(qty: PyneFloat) -> PyneFloat:
    """
    Quantize a user-supplied order quantity down to the mincontract grid.

    :param qty: The requested quantity in contracts (positive, finite)
    :return: The quantity floored to the lot grid
    """
    # TV parses an explicit qty argument through its shortest round-trip
    # decimal (Double.toString) and floors that decimal on the mincontract
    # grid — no float-space snap. Measured on Trend Trader-Remastered
    # (BINANCE:BTCUSDT 30m, 1150 TP/RE partial closes, 2026-08-13):
    # (0.0109/100)*10 = 0.0010899999999999998 closes 108 lots (float
    # rounding gives 109), 150.9 closes 150, while a value whose shortest
    # decimal lands on or above a lot multiple (0.00121, or a dirty
    # position's 0.0011600000000000002) keeps the full amount. The float
    # snap in ``_size_round`` stays for internally derived sizes, where the
    # value is already lot-exact and only the scaling multiply dirties it.
    # Known gap: TV's compiler folds constant qty expressions in exact
    # decimal ((0.0109/100)*10 passed literally arrives clean), PyneCore
    # evaluates them at runtime — only const-expression qty args differ.
    global _explicit_qty_grid
    rfactor = syminfo._size_round_factor  # noqa
    mincontract = 1.0 / rfactor
    cached_grid = _explicit_qty_grid
    if cached_grid is None or cached_grid[0] != mincontract:
        cached_grid = (mincontract, Decimal(repr(mincontract)))
        _explicit_qty_grid = cached_grid
    grid = cached_grid[1]
    lots = int((Decimal(repr(abs(qty))) / grid).to_integral_value(rounding=ROUND_FLOOR))
    sign = 1 if qty > 0 else -1
    return _size_units(sign * lots * mincontract)


def _size_round(qty: PyneFloat) -> PyneFloat:
    """
    Round a size down to the nearest tradable lot (``1 / _size_round_factor``).

    :param qty: The quantity to round
    :return: The rounded quantity
    """
    if not (qty == qty):  # is_na_arg
        return na_float
    rfactor = syminfo._size_round_factor  # noqa
    # Floor to the lot step (1 / rfactor). The float64 product can land an exact
    # lot multiple a hair below the integer (e.g. 173.432 * 1e4 ->
    # 1734319.9999999998); snap values within a few ULPs of an integer up before
    # the floor so an exact multiple is not truncated a whole lot down.
    # Do NOT widen this tolerance to chase a single TV fill: the hair-below
    # razor ties (~2e-4 of boundary entries; the Gaussian Channel extra trade
    # is one) are NOT reachable by any snap width. One-shot TV probes with
    # injected equity proved the up-vs-floor outcome is a deterministic
    # function of (equity, close) following a money-tick grid law (snap up
    # iff floor(money_ticks/G) >= floor(cost_ticks(N0+1)/G), G scale-
    # dependent: 0.05 ticks near 1e6 money, 0.002 near 5e5; 615/618 probe
    # razors reproduced). The law belongs in the money-sizing path, not in
    # this generic lot floor — implementing it here as a tolerance breaks
    # ordinary fills.
    scaled = abs(qty) * rfactor
    nearest = round(scaled)
    # The snap width is a few ULPs of the scaled value: the float dirt it
    # guards against is 1-2 ULPs, while a fixed relative width (the old
    # 1e-12) grows to a third of a LOT at SHIB-scale counts (3.2e11 lots)
    # and swallows the real 0.77-lot fraction of a qty_percent close that
    # TradingView floors (BINANCE:SHIBUSDT probe, bar 9: TV closes
    # ...248.554 lots from 867052023.121 * 0.37, the wide snap closed .555).
    lots = nearest if abs(scaled - nearest) <= 8.0 * math.ulp(scaled) + 1e-9 else int(scaled)
    if lots == 0:
        return 0.0
    sign = 1 if qty > 0 else -1
    # The lot count is scaled UP by the lot step, not divided by the factor: the
    # two land on different unit-grid points once the quantity passes 2**53 units
    # (BINANCE:SHIBUSDT, lot step 1e-3). See ``_size_units``.
    return _size_units(sign * lots * (1.0 / rfactor))


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
    if not (price == price):  # is_na_arg
        return na_float
    pricescale = syminfo.pricescale
    minmove = syminfo.minmove
    tick_count = price * pricescale / minmove
    # A level within 1e-4 of a tick counts as ON that tick. MEASURED on
    # TradingView (BINANCE:BTCUSDT 30m, 66 short-stop and 55 long-stop events,
    # plus BINANCE:ADAUSDT at mintick 1e-4): a short's stop 9.63e-5 ticks above
    # the grid fills AT the grid point while 1.005e-4 above fills one tick
    # higher, and a long's stop 9e-5 ticks below fills at the grid point while
    # 1.5e-4 below fills one tick lower. The width is in TICKS -- not in price
    # units and not in ULPs: the same 1e-4 separates the two outcomes whether
    # the tick count is 5e3 or 8.7e6.
    if direction < 0:
        return math.floor(tick_count + 1e-4) * minmove / pricescale
    return math.ceil(tick_count - 1e-4) * minmove / pricescale


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


# noinspection PyProtectedMember,PyShadowingBuiltins,PyShadowingNames,PyUnusedLocal
def close(id: str, comment: PyneStr = na_str, qty: PyneFloat = na_float,
          qty_percent: PyneFloat = na_float, alert_message: PyneStr = na_str,
          immediately: bool = False, disable_alert: bool = False):
    """
    Creates an order to exit from the part of a position opened by entry orders with a specific identifier.

    :param id: The identifier of the entry order to close
    :param comment: Additional notes on the filled order
    :param qty: The number of contracts/lots/shares/units to close when an exit order fills
    :param qty_percent: A value between 0 and 100 representing the percentage of the open trade
                        quantity to close when an exit order fills
    :param alert_message: Custom text for the alert that fires when an order fills.
    :param immediately: If true, the closing order executes on the same tick when the strategy places it
    :param disable_alert: If true, the strategy does not trigger an alert when the order fills
    """
    if lib._lib_semaphore or lib._strategy_suppressed:
        return

    position = lib._script.position

    if qty == qty and qty <= 0.0:
        return

    if position.size == 0.0:
        return

    # TV closes only the part of the position opened by entries with this id.
    # Under the default FIFO close_entries_rule the FILL may consume older
    # trades first, but the amount closed is still the bound entry's open size
    # — sizing off the whole position would flatten unrelated entries.
    if isinstance(position, SimPosition):
        # noinspection PyProtectedMember
        bound_size = position.sign * position._bound_qty(id)
    else:
        bound_size = 0.0
        adopted_size = 0.0
        for trade in position.open_trades:
            if trade.entry_id == id:
                bound_size += trade.size
            elif trade.entry_id is None or trade.entry_id == ADOPTED_STARTUP_ENTRY_ID:
                adopted_size += trade.size
        if bound_size == 0.0:
            # Startup adoption seeds the open position under a synthetic (or
            # ``None``) parent id because the real ``strategy.entry`` ids from the
            # prior process are unknown, so a keyed ``close(id)`` matches no open
            # trade. Bind it to the adopted exposure instead of dropping the close
            # (early ``size == 0.0`` return) — otherwise the script could never
            # flatten an adopted position by entry id. ``_clamp_close_intents``
            # caps this to the residual position size before dispatch.
            bound_size = adopted_size

    if not (qty == qty):  # is_na_arg
        if qty_percent == qty_percent:
            size = -bound_size * (qty_percent * 0.01)
        else:
            size = -bound_size
    else:
        size = -position.sign * min(qty, abs(bound_size))

    # The Pine-side lot floor is a backtest-only quantization, as with
    # strategy.entry/order. Broker positions can be smaller than syminfo.mincontract
    # after venue-domain conversion (notably inverse contracts), so preserve the raw
    # close size and let the plugin quantize it onto the venue grid.
    if isinstance(position, SimPosition):
        if qty == qty and qty < abs(bound_size):
            size = -position.sign * _explicit_qty_round(qty)
        else:
            size = _size_round(size)

    if size == 0.0:
        return

    exit_id = f"Close entry(s) order {id}"
    order = Order(id, size, exit_id=exit_id, order_type=_order_type_close,
                  comment=None if isinstance(comment, NA) else comment,
                  alert_message=None if isinstance(alert_message, NA) else alert_message)

    # Stamp the call site so several same-bar partial closes on this entry stack
    # when they come from DIFFERENT statements and collapse when they come from
    # the same one (see _close_call_site). Backtest only — the live broker
    # close-dispatch path is handled separately and stays None.
    if isinstance(position, SimPosition):
        order.book_seq = _close_call_site(_sys._getframe(1))

    # Add order to position (this will handle orderbook and exit_orders)
    position._add_order(order)
    # Same-tick fill is a backtest concept; in broker mode the order is already
    # enqueued by ``_add_order`` and the sync engine forwards it to the exchange.
    if immediately and isinstance(position, SimPosition):
        # Deferred immediate settle: fill after the body (settle_immediate_closes)
        # so position series stay at their pre-close values for the rest of the
        # bar — matching TradingView and PyneCore's broker mode.
        position._deferred_immediate_closes.append(order)


# noinspection PyProtectedMember,PyShadowingNames,PyUnusedLocal
def close_all(comment: PyneStr = na_str, alert_message: PyneStr = na_str, immediately: bool = False,
              disable_alert: bool = False):
    """
    Creates an order to close an open position completely, regardless of the identifiers of the entry
    orders that opened or added to it.

    :param comment: Additional notes on the filled order
    :param alert_message: Custom text for the alert that fires when an order fills
    :param immediately: If true, the closing order executes on the same tick when the strategy places it
    :param disable_alert: If true, the strategy does not trigger an alert when the order fills
    """
    if lib._lib_semaphore or lib._strategy_suppressed:
        return

    position = lib._script.position
    if position.size == 0.0:
        return

    exit_id = 'Close position order'
    order = Order(None, -position.size, exit_id=exit_id, order_type=_order_type_close,
                  comment=comment, alert_message=alert_message)

    # Stamp the call site so a close_all stacked behind a same-bar partial close
    # fills too (backtest only; live close-dispatch handled separately, stays None).
    if isinstance(position, SimPosition):
        order.book_seq = _close_call_site(_sys._getframe(1))

    # Add order to position (this will handle orderbook and exit_orders)
    position._add_order(order)
    # Same-tick fill is a backtest concept; in broker mode the order is already
    # enqueued by ``_add_order`` and the sync engine forwards it to the exchange.
    if immediately and isinstance(position, SimPosition):
        # Deferred immediate settle: fill after the body (settle_immediate_closes)
        # so position series stay at their pre-close values for the rest of the
        # bar — matching TradingView and PyneCore's broker mode.
        position._deferred_immediate_closes.append(order)


#
# Account-currency conversion
#
# TradingView converts a strategy's money at the CASH-FLOW level: every amount is
# multiplied by the rate of the day it is booked on, and nothing re-marks afterwards.
# Measured on BINANCE:BTCUSDT against currency.JPY (18.4% rate amplitude): a closed
# trade's profit is gross * rate(exit) - percent_commission_entry * rate(entry) -
# percent_commission_exit * rate(exit), 274/274 trades, worst 1.1e-7 relative.
#
# Every money quantity in this engine is `<something> * syminfo.pointvalue`, so folding
# the rate into the point value reproduces that automatically -- the entry commission is
# computed on the entry bar and the exit legs on the exit bar, each with its own rate,
# with no per-trade rate stored anywhere. It also keeps the percent metrics right: they
# divide two converted amounts, so the rate cancels, exactly as TradingView reports.
#
# The two exceptions are deliberate and measured:
# * cash_per_contract / cash_per_order amounts are already in the account currency and
#   are booked verbatim (584/584 each) -- they carry no point value, so they are
#   untouched by construction.
# * initial_capital is declared in the account currency, so it is never converted.

# The rate is a daily series but every money expression reads it, so it is sampled once
# per bar. Three forms are kept: the raw value the Pine builtins must return (na when
# there is no rate data), the safe multiplier the ledger uses (an unusable rate degrades
# to 1.0, which leaves an unconverted run bit-identical), and the scaled point value.
_conv_script: '_core_script.Script | None' = None
_conv_bar: int = -1
_conv_rate: float = 1.0
_conv_safe: float = 1.0
_conv_pv: float = 0.0
_conv_warned: bool = False
# The script whose account currency was found to be the symbol's own. Whether a run
# converts at all is fixed by the script and the symbol, never by the bar, so the
# first sample latches it here and every later point value is one identity test away
# from ``syminfo.pointvalue`` — no bar memo, no resample. Holding the script object
# rather than a flag keeps the multi-script guard below in the same single test.
_conv_identity_script: '_core_script.Script | None' = None


def _reset_currency_state() -> None:
    """Drop the per-bar account-rate memo between script runs."""
    global _conv_script, _conv_bar, _conv_rate, _conv_safe, _conv_pv, _conv_warned
    global _conv_identity_script
    _conv_script = None
    _conv_bar = -1
    _conv_rate = 1.0
    _conv_safe = 1.0
    _conv_pv = 0.0
    _conv_warned = False
    _conv_identity_script = None


# noinspection PyProtectedMember
def _sample_account_currency() -> float:
    """
    Sample the symbol-to-account rate for the current bar and refresh the memo.

    :return: The point value scaled into the account currency
    """
    global _conv_script, _conv_bar, _conv_rate, _conv_safe, _conv_pv, _conv_warned
    global _conv_identity_script

    script = lib._script
    symbol_cur = syminfo.currency
    # With no script the account is the symbol's own currency, so nothing converts --
    # and so does an undeclared ``strategy(currency=...)``, which Pine spells NONE.
    # Folding both into the symbol's own currency here leaves ONE comparison to decide
    # whether this run converts at all.
    account = (symbol_cur if script is None or script.currency == 'NONE'
               else script.currency)
    rate = 1.0
    if account != symbol_cur:
        rate = request.currency_rate(symbol_cur, account)
    else:
        _conv_identity_script = script

    # A zero or negative rate can only come from a damaged feed, and a sign flip would
    # invert the whole ledger, so it degrades the same way na does.
    safe = rate if (rate == rate and rate > 0.0) else 1.0
    if safe != rate and not _conv_warned:
        _conv_warned = True
        _logger.warning(
            "strategy(currency=%s) needs a %s to %s rate, but no rate data is available; "
            "profits stay in %s. Supply the pair as an OHLCV file with a sibling TOML "
            "declaring its basecurrency and currency.",
            account, symbol_cur, account, symbol_cur,
        )

    _conv_script = script
    _conv_bar = lib.bar_index
    _conv_rate = rate
    _conv_safe = safe
    _conv_pv = syminfo.pointvalue * safe
    return _conv_pv


# noinspection PyProtectedMember
def _account_point_value() -> float:
    """
    Point value of one contract for the current bar, in the account currency.

    :return: ``syminfo.pointvalue`` scaled by the symbol-to-account rate
    """
    # A run whose account currency IS the symbol's own has no rate to sample on any
    # bar, so it skips the memo and the resample entirely. The test doubles as the
    # multi-script guard below: the latch holds the script it was decided for.
    if _conv_identity_script is lib._script:
        return syminfo.pointvalue
    # The script identity is part of the key, not just the bar: PyneAPI runs several
    # scripts in one process and re-applies syminfo every bar, so a bar_index-only memo
    # could hand one script the rate sampled for another. A realtime bar skips the memo
    # because the chart-pair rate source reads lib.close, which moves within the bar.
    if (_conv_script is lib._script and _conv_bar == lib.bar_index
            and not lib.barstate.isrealtime):
        return _conv_pv
    return _sample_account_currency()


def _account_rate() -> float:
    """
    Exchange rate from the symbol's quote currency to the strategy's account currency.

    :return: The rate, 1.0 when no conversion applies, na when no rate data is available
    """
    _account_point_value()  # refreshes the memo when it is stale
    return _conv_rate


def convert_to_account(value: PyneFloat) -> PyneFloat:
    """
    Converts the value from the currency of the chart symbol to the currency of the strategy account.

    :param value: The value to convert, in the symbol's quote currency
    :return: The value expressed in the account currency
    """
    if not (value == value):  # is_na_arg
        return na_float
    rate = _account_rate()
    if not (rate == rate):  # is_na_arg
        return na_float
    return value * rate


def convert_to_symbol(value: PyneFloat) -> PyneFloat:
    """
    Converts the value from the currency of the strategy account to the currency of the chart symbol.

    :param value: The value to convert, in the account currency
    :return: The value expressed in the symbol's quote currency
    """
    # Measured on TradingView (FX:EURUSD 1D, currency=currency.EUR):
    # convert_to_symbol(1.0) = 1.161710037175 is exactly 1 / 0.8608, the reciprocal of
    # convert_to_account(1.0) — TV keeps ONE symbol->account rate and divides by it here,
    # so the two directions stay exactly reciprocal.
    if not (value == value):  # is_na_arg
        return na_float
    rate = _account_rate()
    if not (rate == rate) or rate == 0.0:  # is_na_arg
        return na_float
    return value / rate


# noinspection PyProtectedMember
def _default_entry_budget(price: float, pv: float | None = None) -> tuple[float, float] | None:
    """Money amount and per-unit cost of a default-sized entry at ``price``.

    Returns ``(money, unit_cost)`` so that the raw quantity is
    ``money / unit_cost``, or None for fixed sizing (not money-based).

    Both sides are in the account currency: the budget because equity and
    ``default_qty_value`` are, the unit cost because the point value carries the rate.
    Measured on TradingView with a JPY account on BINANCE:BTCUSDT --
    ``strategy.cash`` sizes 584/584 at ``(cash / rate) / price`` and
    ``percent_of_equity`` 581/584 at ``floor_mc((equity / rate) / price)``, which is the
    same thing as dividing an account-currency budget by an account-currency unit cost.

    ``pv`` overrides the current bar's account point value with the one frozen at
    a resting order's placement: the quote->account rate is part of the frozen
    budget, not of the fill (see :meth:`SimPosition._resolve_deferred_qty`).
    """
    script = lib._script
    default_qty_type = script.default_qty_type
    if default_qty_type == fixed:
        return None

    if pv is None:
        pv = _account_point_value()

    if default_qty_type == percent_of_equity:
        target_investment = script.position.equity * script.default_qty_value * 0.01
        if script.commission_type == _commission.percent:
            commission_multiplier = 1.0 + script.commission_value * 0.01
            return target_investment, price * pv * commission_multiplier
        if script.commission_type == _commission.cash_per_contract:
            # The cash fee is already in the account currency, so it is added unscaled.
            return target_investment, price * pv + script.commission_value
        if script.commission_type == _commission.cash_per_order:
            return max(0.0, target_investment - script.commission_value), price * pv
        # No commission
        return target_investment, price * pv

    if default_qty_type == cash:
        return script.default_qty_value, price * pv

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
    if unit_cost == 0.0:
        # A price that snaps to zero leaves no unit cost to divide the budget by.
        # An unsizable order is dropped by the callers' finite-qty gate.
        return na_float
    return _sig10_money(money) / unit_cost


def default_entry_qty(fill_price: PyneFloat) -> PyneFloat:
    """
    Calculates the default quantity, in units, of an entry order from ``strategy.entry`` or
    ``strategy.order`` if it were to fill at the given price.

    :param fill_price: The fill price to evaluate
    :return: The default order quantity in contracts, 0 when it cannot be sized
    """
    # Measured on TradingView (BINANCE:BTCUSDT 30m, mintick 0.01, lot step 1e-5).
    # The price argument is SNAPPED to the tick grid before sizing: 100.007 and
    # 100.01 both return 23.0707 while 100.003 and 100.0 both return 23.07301,
    # and 0.005 sizes off 0.01 where 0.004 snaps to zero. The quantity is then
    # FLOORED on the lot grid, never rounded -- 7777.77 / 93825.86 =
    # 0.08289580 returns 0.08289, and a fixed default_qty_value of 3.1234567
    # returns 3.12345. ``strategy.fixed`` ignores the price entirely (na, zero
    # and a negative price all return the same floored value), while a
    # money-based size with nothing to divide by (na price, or a price snapping
    # to zero) returns 0. An open position is not considered: the value is the
    # order's own quantity, not the reversal amount.
    qty = _default_entry_qty(_tick_snap(fill_price))
    if not (-math.inf < qty < math.inf):  # is_na_arg or infinite
        return 0.0
    return _size_floor(qty)


def _sig10_money(money: float) -> float:
    """Quantize a sizing money budget to 10 significant decimal digits.

    :param money: The money budget in account currency
    :return: The budget rounded half-up to 10 significant decimals
    """
    # TV rounds the money side of default sizing to 10 significant decimal
    # digits (half-up) before dividing by the unit cost. Measured on
    # BINANCE:SHIBUSDT 1h flat-cycle probes at five percent levels (9, 22.5,
    # 45, 67.5, 90) whose 1e-3 lots on ~1e9 contracts resolve ~1e-8 of the
    # budget: 5777/5785 sizing events reproduce exactly (the rest are 1-lot
    # razor ties at exact lot boundaries), while the raw budget scores under
    # 30/5785. The grid follows the budget's decimal exponent (1e-6 near 9e3,
    # 1e-5 near 3e4..9e4, 1e-4 above 1e5), which also explains every snap-up
    # band measured earlier on BINANCE:BTCUSDT — the 0.05/0.005 tick edges at
    # money 1e6/1.2e5 and the 0.5/5/50 tick grids of the >=1e7 gate are all
    # exactly half of this grid expressed in ticks, including the previously
    # unmapped ON point at cost 1.25e8 ticks. Applied at every magnitude: the
    # >=1e7 MASTER-X gate reads the same quantized budget, which is what puts
    # the four 5e7-budget sizing events of `Fractal Breakout Strategy [KL]`
    # on TradingView's side (see :func:`_judge_money_entry`).
    if not (money > 0.0):
        return money
    dec = Decimal(repr(money))
    return float(dec.quantize(Decimal(1).scaleb(dec.adjusted() - 9), rounding=ROUND_HALF_UP))


def _money_ticks(money: float, mintick: float) -> float:
    """Express a quantized sizing budget on the tick grid.

    :param money: The money budget in account currency
    :param mintick: The instrument's tick size
    :return: The quantized budget in ticks
    """
    # The division stays in decimal because the budget is already on
    # ``_sig10_money``'s decimal grid and a mintick that is not binary-exact
    # drags the quotient off it: 150435449.7 / 0.01 comes out as
    # 15043544969.999998, a whole tick below the value it stands for, and the
    # gate reads the count through ``math.floor``.
    return float(Decimal(repr(_sig10_money(money))) / Decimal(repr(mintick)))


# Distance threshold (in ticks) of the big-money gate's down-step in the
# price >= 1e5 regime: an inflated threshold landing on an even grid
# multiple steps down one grid unit only when it cleared the inflated cost
# by more than this. Bracketed in (0.0783, 0.1034) ticks on TV probes;
# 3/32 is the binary-exact candidate. Below price 1e5 the down-step has
# no such guard (a probe filled 0.056 ticks below the even cell).
_GATE_DOWN_STEP_DELTA = 0.09375


def _ceil_to_grid(value: float, grid: float) -> tuple[int, float]:
    """Exact smallest multiple of ``grid`` that is >= ``value``.

    ``value / grid`` alone can round across an integer near a grid point; the
    correction loops re-check with ``k * grid`` products, which are exact for
    the tick grids (0.5, 5, 50) and magnitudes (< 2^53) involved.

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


def _price_f32_offset_k(price: float) -> int:
    """The odd number of float32-ULP/25 quanta ``price`` sits above its
    float32 lower neighbour, or 0 when the relationship does not hold.

    TV's big-money gate inflates its cost threshold only on bars whose close
    has this float32 relationship within seven quanta (measured 38/38 on
    BINANCE:BTCUSDT 30m; in the [2^16, 2^17) binade the quantum is 1/32
    tick). A close exactly representable in float32 (offset 0) does not
    inflate. The quantum count feeds the membership predicate in
    :func:`_gate_entry_lots` (only offsets small relative to the price
    inflate), so the count itself is returned.

    :param price: The bar close driving the gate
    :return: The odd quantum count (1/3/5/7), or 0 when not an odd-offset bar
    """
    if price <= 0.0 or not math.isfinite(price):
        return 0
    f32 = struct.unpack('<f', struct.pack('<f', price))[0]
    bits = struct.unpack('<I', struct.pack('<f', f32))[0]
    if f32 > price:
        bits -= 1
        f32 = struct.unpack('<f', struct.pack('<I', bits))[0]
    ulp = struct.unpack('<f', struct.pack('<I', bits + 1))[0] - f32
    if ulp <= 0.0 or not math.isfinite(ulp):
        return 0
    quanta = (price - f32) * 25.0 / ulp
    k = round(quanta)
    if k % 2 == 1 and k <= 7 and abs(quanta - k) < 0.25:
        return k
    return 0


def _gate_entry_lots(equity_ticks: float, lots: int, rfactor: float,
                     unit_cost: float, mintick: float, price: float) -> int | None:
    """Judge an entry of ``lots`` lots against TV's big-money margin gate.

    From 1e9 cost ticks upward TV quantizes the order cost onto a tick grid
    (0.5 tick, 5 ticks from 1e10 cost ticks, 50 ticks from 1e11 — decimal
    decade steps; the 1e11 switch bracketed in (8.79e10, 1.2945e11] with the
    binary 2^37 candidate refuted at 1.2945e11) and compares the raw equity
    tick count against the quantized threshold:

    - equity >= threshold: the entry fills as sized;
    - equity below threshold but at least the plain grid ceiling of the cost
      (possible only when the threshold was inflated): the entry is rejected;
    - equity below the plain grid ceiling: the parity of the grid multiple
      decides — even rejects, odd fills one lot less.

    On odd-float32-offset bars (see :func:`_price_f32_offset_k`) whose
    offset is small relative to the price (offset / price < 3 * 2^-27,
    i.e. 8 * k below 75 times the price's binary fraction) the threshold
    is the grid ceiling of an inflated cost: the cost times (1 + 2^-31)
    for closes >= 1e5, or the cost plus an absolute 5e-6 account currency
    per contract for closes below 1e5. An inflated threshold landing on
    an EVEN grid multiple steps one grid unit down (never below the plain
    ceiling, and not when the cost sits exactly on the grid); in the
    >= 1e5 regime the step additionally requires clearing the inflated
    cost by more than ``_GATE_DOWN_STEP_DELTA``. Reverse-engineered on
    BINANCE:BTCUSDT 30m one-shot probes: 19,613 of 19,614 measurements
    reproduced at >= 1e5 (boundary decade 21/22), 378 of 378 in the
    2026-07-30/31 sub-1e5 census (membership band edges bracketed at
    k=5 close 69,826..69,980 and k=7 close 97,849..97,912; per-contract
    inflation windows intersect in (4.9975e-6, 5.0555e-6]).

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
    grid = 50.0 if cost >= 1e11 else 5.0 if cost >= 1e10 else 0.5
    k0, m0 = _ceil_to_grid(cost, grid)
    m_eff = m0
    k_off = _price_f32_offset_k(price)
    # Membership: the threshold inflates only when the close's float32
    # offset is small relative to the price — offset / price < 3 * 2^-27
    # with offset = k * ulp32 / 25 reduces to the binade-free predicate
    # 8 * k < 75 * frac, where price = frac * 2^exp (frac in [0.5, 1)).
    # Magnitude: 2^-31 relative for closes >= 1e5; below 1e5 an absolute
    # 5e-6 account currency per contract instead, and the down-step loses
    # its delta guard (probe census 2026-07-30/31, 378/378 reproduced).
    if k_off and 8.0 * k_off < 75.0 * math.frexp(price)[0]:
        if price >= 1e5:
            inflated = cost * (1.0 + 2.0 ** -31)
            delta = _GATE_DOWN_STEP_DELTA
        else:
            inflated = cost + lots / rfactor * (5e-6 / mintick)
            delta = 0.0
        k_eff, m_eff = _ceil_to_grid(inflated, grid)
        if k_eff % 2 == 0 and m_eff - inflated > delta:
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
def _judge_money_entry(size: float, price: float, market: bool = False,
                       money: float | None = None) -> float:
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

    The budget itself is quantized at the sizing source for every magnitude:
    ``_default_entry_qty`` floors from ``_sig10_money(money)``, whose
    10-significant-digit decimal grid reproduces every snap edge measured
    below 1e7 by one-shot equity-injection sweeps on BINANCE:BTCUSDT 30m (the
    0.05-tick edges of the 2026-07-08 cost cluster at [1e8, 1.16e8] ticks
    including the 1.25e8 ON point the band model could not map, and the
    0.005-tick Fabio Pro Scalper bisection edge at money ticks
    12451249.295 on 2026-07-10) as half of that decimal grid in ticks, and it
    carries the four big-money sizing events of the wild corpus script
    `Fractal Breakout Strategy [KL]` (5% of a 1e9 account, so 5e7 of budget)
    that the raw quotient misses -- three one-lot snaps on 2025-03-11 05:30,
    2026-01-31 08:00 and 2026-03-29 10:00, plus the 2026-08-25 18:30 entry
    whose raw budget lands 0.05 tick under its own cost ceiling and which
    TradingView fills at the full 632.93356 contracts.

    A quantized size then faces the ordinary
    creation-time margin check at the placement close: at 100%
    percent_of_equity sizing the snapped cost always exceeds equity, so the
    entry cancels at placement even when the fill open would fit (measured:
    the Gaussian Channel razor cancel and the 2025-01-02 19:30 flat100 probe
    cancel, where the open HAD gapped down far enough) — which is how the
    Gaussian Channel corpus divergence resolves. Above 1e7 that check is the
    big-money gate itself, and a reversal it rejects keeps its closing leg
    (see :func:`_suppress_opening_leg`).

    :param size: Signed floor-sized quantity in contracts
    :param price: The sizing/gate price (placement close for market entries,
        fill price for price-based orders resolving at execution)
    :param market: True when judging a market entry at placement (enables
        the sub-1e7 last-lot drop; price-based fills keep the plain floor)
    :param money: Placement-frozen money budget of a deferred fill; None
        derives the budget from the current equity (market entries)
    :return: The granted signed quantity, or 0.0 when the entry is rejected
    """
    budget = _default_entry_budget(price)
    if budget is None:
        return size
    if money is None:
        money = budget[0]
    unit_cost = budget[1]
    mintick = syminfo.mintick
    if not mintick or mintick <= 0:
        return size
    rfactor = syminfo._size_round_factor  # noqa
    lots = round(abs(size) * rfactor)
    if lots <= 0:
        return size
    if money < 1e7:
        if not market:
            return size
        # Sub-1e7 last-lot drop: on a bar whose sizing price is an ODD number
        # of ticks TV inflates the floored size's cost by 2^-33 and drops one
        # lot when the money no longer covers it (resize, not cancel) — the
        # sub-1e7 mirror of the >=1e7 MASTER-X s-slope inflation. Measured by
        # 20 one-shot equity-injection probes on BINANCE:BTCUSDT 30m
        # (2026-08-13, SuperTrend d6fba11d fork): on the 2026-03-23 11:00 bar
        # (C=70702.57, 7070257 ticks, odd) the drop edge is bracketed in
        # relative headroom (1.10e-10, 1.17e-10] around 2^-33 = 1.164e-10,
        # while the 2026-07-14 12:30 bar (C=63960.00, even ticks) fills even
        # at 0.8e-10 headroom. Slope selector below C<1e5 beyond these two
        # bars is unmeasured; even-tick bars are treated as s=0.
        # Both measured bars sit near 7e6 ticks; BINANCE:SHIBUSDT refutes the
        # drop far below that (odd closes of 967..1038 ticks fill the full
        # floored size even at 5e-12 relative headroom), so the mechanism is
        # scoped to the measured magnitude until more bars are mapped.
        if lots > 1 and round(price / mintick) % 2 == 1 and price / mintick >= 1e6:
            floor_cost = lots / rfactor * unit_cost
            # The drop was measured in the regime where the floored size's
            # cost sits at or below the raw budget; a size floored from the
            # sig10-rounded-UP budget has floor_cost above the raw money and
            # is NOT dropped.
            if floor_cost <= money < floor_cost * (1.0 + 2.0 ** -33):
                lots -= 1
                sign = 1.0 if size > 0 else -1.0
                size = sign * lots / rfactor
        # The sub-1e7 snap-up to the next lot is the half-grid image of the
        # 10-significant-digit money quantization and now happens at the
        # source: ``_default_entry_qty`` sizes from ``_sig10_money(money)``,
        # so a budget within half a decimal grid unit below the next lot's
        # cost already floors to the larger size (and one just above a lot
        # boundary correctly rounds DOWN, which the one-way snap never
        # could). The 0.05/0.005 tick edges measured on BINANCE:BTCUSDT
        # (2026-07-08/10: the 1.0200e8/1.1575e8 cost cluster, the Fabio Pro
        # Scalper 12451249.295 bisection, the flat100 and Gaussian Channel
        # razor cancels) are exactly that half grid in ticks.
        return size
    money_ticks = _money_ticks(money, mintick)
    next_cost = (lots + 1) / rfactor * unit_cost / mintick
    next_grid = 50.0 if next_cost >= 1e11 else 5.0 if next_cost >= 1e10 else 0.5
    _, next_m0 = _ceil_to_grid(next_cost, next_grid)
    if math.floor(money_ticks) >= next_m0 - next_grid:
        lots += 1
    granted = _gate_entry_lots(money_ticks, lots, rfactor, unit_cost, mintick, price)
    if granted is None or granted <= 0:
        return 0.0
    sign = 1.0 if size > 0 else -1.0
    return sign * granted / rfactor


# noinspection PyProtectedMember,PyShadowingNames,PyShadowingBuiltins,PyUnusedLocal,DuplicatedCode
def entry(id: str, direction: direction.Direction, qty: int | PyneFloat = na_float,
          limit: int | float | None = None, stop: int | float | None = None,
          oca_name: str | None = None, oca_type: _oca.Oca | None = None,
          comment: str | None = None, alert_message: str | None = None,
          disable_alert: bool = False):
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
    :param disable_alert: If true, the strategy does not trigger an alert when the order fills
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

    if not (limit == limit):  # is_na_arg
        limit = None
    elif limit is not None:
        # We need negative direction for entry limit orders - NOTE: it is tested
        limit = _price_round(limit, -direction_sign)
    if not (stop == stop):  # is_na_arg
        stop = None
    elif stop is not None:
        stop = _price_round(stop, direction_sign)

    # A default-sized (no explicit qty) price-based order resolves its
    # quantity at the actual fill price; for those the size computed here is
    # only the placement estimate used for the margin check and order
    # bookkeeping. A MARKET entry keeps this size: TV sizes it from the
    # mark-to-market equity and close of the placement bar (probe-verified on
    # BINANCE:BTCUSDT 30m: flat entries 13281/13281, reversal flips
    # 26560/26561 exact). The sizing price is the price the order would
    # execute at NOW — the current price when immediately executable, the
    # limit/stop price while it rests. "Would execute at" includes slippage:
    # a market entry is sized at the placement close pushed the adverse way
    # for its direction (measured 2026-08-13, Triple CCI BINANCE:BTCUSDT
    # 240m, slippage=5: close+signed-slip fits 47/47 sized entries, raw
    # close only 44/47 — the misses are all one lot step high).
    deferred_default = not (qty == qty)  # is_na_arg
    market_sizing_price: float | None = None
    exec_price = 0.0  # only meaningful when deferred_default
    if deferred_default:
        exec_price = position.c
        if limit is not None:
            exec_price = min(limit, exec_price) if direction_sign > 0 else max(limit, exec_price)
        elif stop is not None:
            exec_price = max(stop, exec_price) if direction_sign > 0 else min(stop, exec_price)
        else:
            slippage = lib._script.slippage
            if slippage > 0 and isinstance(position, SimPosition):
                exec_price = float(exec_price) + direction_sign * syminfo.mintick * slippage
            market_sizing_price = float(exec_price)
        qty = _default_entry_qty(exec_price)

    # qty must be a positive FINITE number. The range test also drops NaN and
    # infinity, which a plain ``qty <= 0.0`` test lets through: default sizing
    # hands back NaN when it has nothing to compute from (an na
    # ``default_qty_value``, or an equity gone NaN), and both non-finite values
    # then blow up the integer conversion inside ``_size_round`` /
    # ``_judge_money_entry``, halting the whole script. An unsizable order is
    # dropped like a zero-sized one below. This is PyneCore robustness, not a
    # TradingView rule: ``default_qty_value`` is a const float that rejects na at
    # compile time (CE10034 on a TV probe), so a compiled script cannot reach
    # this state -- hand-written Pyne code can.
    if not (0.0 < qty < math.inf):  # unsizable_qty
        return

    size = qty * direction_sign

    # Re-placing an entry under an id that already has an unfilled order from this
    # same bar MODIFIES that order, and the modification carries the raw qty only:
    # the reversal flip the replaced order was built with is not computed again.
    # Measured on TradingView (BINANCE:BTCUSDT 30m): with a 10-contract long open,
    # one strategy.entry(short, 4) reverses to 4 short, while the SAME call issued
    # twice on one bar sells only 4 and leaves 6 long.
    existing_entry = position.entry_orders.get(id)
    skip_flip = (existing_entry is not None and not existing_entry.cancelled
                 and existing_entry.bar_index == int(lib.bar_index))

    # The Pine-side lot floor is a backtest-only quantization: TV silently
    # snaps a sub-lot size to zero and drops the order. In broker mode the
    # exchange owns the quantity grid — the plugin quantizes onto the venue
    # step and emits an explicit below-minimum skip. Flooring here would
    # instead drop a below-grid signal silently, before the order is ever
    # built, hiding an invalid live signal from the operator. Keep the raw
    # requested qty in broker mode so the sync engine dispatches it and the
    # plugin's quantity preflight reports the skip.
    if isinstance(position, SimPosition):
        size = _size_floor(size) if deferred_default \
            else direction_sign * _explicit_qty_round(float(qty))
        # A quantity that snaps to zero lots still REVERSES: TV closes the
        # opposite position and opens nothing. Measured on BINANCE:BTCUSDT 30m
        # (mincontract 1e-5): a 9e-6 short against a 0.001 long books the closed
        # trade and leaves the position flat, while the same 1.9e-5 short from
        # flat opens exactly one lot. The zero size nets to a pure close through
        # the reversal flip below for a price-based order, and through the
        # zero-size closing leg the market path already builds; an order with
        # nothing to close is dropped outright.
        if size == 0.0 and (skip_flip or position.size == 0.0
                            or position.sign == direction_sign):
            return

    # Market entries keep their placement-close sizing (price-based orders
    # re-resolve at fill), so the big-money sizing gate is judged here. A
    # sub-1e7 snapped-up size deliberately falls through to the creation-time
    # margin check below: TV cancels a snapped entry whose snapped cost can
    # no longer be margined at the placement close even when the fill open
    # would permit it (measured: the Gaussian Channel razor cancel at
    # 100% sizing, and the 2025-01-02 19:30 flat100 probe cancel where the
    # open HAD gapped down far enough to fit).
    if market_sizing_price is not None:
        size = _judge_money_entry(float(size), market_sizing_price, market=True)
        if size == 0.0:
            _suppress_opening_leg(position, id, direction_sign, oca_name, oca_type,
                                  comment, alert_message)
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
            pv = _account_point_value()
            margin_needed = abs(size) * check_price * pv * margin_ratio
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
            # The tick grid below divides an account-currency equity by the SYMBOL's
            # mintick. That was measured on a chart whose quote currency is the account
            # currency, so the mismatch never showed; with an account rate != 1 the grid's
            # unit is undefined. The gate only engages above 1e7 equity, so this is left
            # as measured rather than guessed at.
            mintick = syminfo.mintick
            if equity >= 1e7 and mintick and mintick > 0:
                rfactor = syminfo._size_round_factor  # noqa
                lots = round(abs(size) * rfactor)
                unit_margin = check_price * pv * margin_ratio
                if lots > 0:
                    granted = _gate_entry_lots(equity / mintick, lots, rfactor,
                                               unit_margin, mintick, check_price)
                    if granted != lots:
                        # The gate refuses to OPEN, which for a reversal is not
                        # the same as cancelling: the closing leg lives on.
                        _suppress_opening_leg(position, id, direction_sign,
                                              oca_name, oca_type, comment,
                                              alert_message)
                        return
            elif margin_needed > equity:
                return

    # The pyramiding limit is judged when the command RUNS, against the position
    # the body sees — a rejected entry is gone for good, it does not wait for the
    # next open to be reconsidered. MEASURED on BINANCE:BTCUSDT 240 (pyramiding=1,
    # probe "PYR probe 1/2"): a second entry placed while the position is open
    # never fills, whether that position is flattened by a same-body
    # ``strategy.close(immediately=true)`` or by an ordinary close order filling at
    # the very open the entry would have used, and regardless of the entry id.
    # Market orders are re-checked at processing time as well, for the entries
    # queued together in one body that only fill one after another.
    # What the gate reads is the position as the market orders already placed on
    # this bar leave it (see _pyramid_sign_for_entry), not the raw position.
    # A call that MODIFIES this bar's unfilled order under the same id (see
    # skip_flip) adds no entry, so the limit has nothing to judge -- rejecting it
    # would leave the order it was meant to rewrite standing. MEASURED on the wild
    # `Strategy for UT Bot Alerts indicator` reference (BINANCE:BTCUSDT 30m), whose
    # two near-identical blocks issue the same entry twice per signal bar: TV takes
    # the second call's flip-free quantity, reversing a 28.20314 long into a
    # 2.10875 short rather than the 30.31189 the first call's flip would open.
    # The flip quantity stays price-order-only: a market order gets it at
    # processing time.
    flip_extra = 0.0
    if not skip_flip and position._pyramid_sign_for_entry() == direction_sign:
        # Check pyramiding limit for entry orders adding to existing position
        if lib._script.pyramiding <= position._pyramid_count():
            # Pyramiding limit reached - don't add the order
            return

    elif limit is not None or stop is not None:
        if position.size != 0.0 and not skip_flip:
            # TradingView calculates the flip quantity at order creation time,
            # not at execution time. If we have an opposite direction position,
            # we need to add the position size to the order size to flip it.
            # This means the order will first close the existing position,
            # then open a new one in the opposite direction.
            # The position it reads is the one the market closes already placed
            # on this bar leave behind (see _size_flippable_by_entry).
            flippable = position._size_flippable_by_entry()
            size -= flippable  # Subtract because the position has the opposite sign
            flip_extra = abs(flippable)

    order = Order(id, size, order_type=_order_type_entry, limit=limit, stop=stop, oca_name=oca_name,
                  oca_type=oca_type, comment=comment, alert_message=alert_message)
    order.skip_flip = skip_flip
    order.flip_extra = flip_extra
    # Only price-based orders re-size at execution; a market entry keeps its
    # placement-time (signal close) quantity — TV rejects it at the next open
    # when that quantity can no longer be margined, rather than re-sizing.
    # The money budget is frozen now, at placement — the fill only supplies
    # the per-unit cost (see _resolve_deferred_qty).
    if deferred_default and (limit is not None or stop is not None):
        order.deferred_qty = True
        budget = _default_entry_budget(float(exec_price))
        if budget is not None:
            order.budget_money = budget[0]
            order.budget_pv = _account_point_value()
    # Store in entry_orders dict
    position._add_order(order)


# noinspection PyShadowingBuiltins,PyProtectedMember
def _suppress_opening_leg(position: PositionBase, id: str, direction_sign: float,
                          oca_name: str | None, oca_type: _oca.Oca | None,
                          comment: str | None, alert_message: str | None) -> None:
    """Keep an entry TradingView refuses to open alive as a reversal's closing leg.

    :param position: The simulated position the order belongs to
    :param id: The entry's id
    :param direction_sign: +1 for a long entry, -1 for a short one
    :param oca_name: OCA group name of the original order
    :param oca_type: OCA type of the original order
    :param comment: Comment of the original order
    :param alert_message: Alert message of the original order
    """
    # A refusal to open is not a cancellation when the entry also reverses: TV
    # closes the opposite position at the next open and suppresses the opening
    # leg only. Measured on the wild corpus script
    # `Hybrid: RSI + Breakout + Dashboard` (BINANCE:BTCUSDT 30m, 100%
    # percent_of_equity) at 2026-05-14 15:00, where the short closes at the bar
    # open and the long first fills a bar later, from the re-issued and
    # re-judged entry. The zero size nets to a pure close through the reversal
    # flip at order processing.
    if position.size == 0.0 or position.sign == direction_sign:
        return
    closing_leg = Order(id, 0.0, order_type=_order_type_entry, oca_name=oca_name,
                        oca_type=oca_type, comment=comment,
                        alert_message=alert_message)
    closing_leg.sign = direction_sign
    position._add_order(closing_leg)


# noinspection PyShadowingBuiltins,PyProtectedMember,PyShadowingNames,PyUnusedLocal
def exit(id: str, from_entry: str = "",
         qty: PyneFloat = na_float, qty_percent: PyneFloat = na_float,
         profit: PyneFloat = na_float, limit: PyneFloat = na_float,
         loss: PyneFloat = na_float, stop: PyneFloat = na_float,
         trail_price: PyneFloat = na_float, trail_points: PyneFloat = na_float,
         trail_offset: PyneFloat = na_float,
         oca_name: PyneStr = na_str,
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
    entry_seq: int | None = None

    # noinspection PyProtectedMember,PyShadowingNames
    def _exit():
        nonlocal limit, stop, trail_price, from_entry, direction, size

        # Sticky bracket (TV semantics): a leg is identified by (id, from_entry,
        # entry_seq) — TradingView issues one leg per FILLED ENTRY, so two pyramid
        # adds sharing a from_entry id get a leg each and each is consumed on its
        # own. ``entry_seq`` is None only while the leg still waits on a pending
        # entry order; :meth:`SimPosition._bind_entry` hands it over on the fill.
        # Re-issuing it every bar updates its prices, but a leg that already fired
        # its slice must not be resurrected (the ``consumed`` tombstone). The
        # reservation is recomputed from ``init_size`` on every issue: that is the
        # ORIGINAL size of the entry it is bound to — frozen at the fill, so
        # margin-call shrinkage does not erode it — or, for a leg still waiting on
        # an entry order, that order's CURRENT size, so a pending entry re-sized
        # bar-to-bar keeps being tracked (locking the first bar's size would
        # under-close the eventual fill and strand a sliver).
        exit_key = _exit_key(id, from_entry, entry_seq)
        existing = position.exit_orders.get(exit_key)
        if existing is not None and existing.consumed:
            return

        is_rest_leg = not (qty == qty) and not (qty_percent == qty_percent)  # is_na_arg
        # Sibling legs reserve slices of the SAME entry first-come-first-served
        # (consumed siblings keep their reservation until the entry fully
        # closes). Only sticky exit legs (book_seq is None) count as siblings;
        # a stacked strategy.close()/close_all() partial (book_seq set) is an
        # immediate market close, not a reservation against this leg.
        # A sibling that reserved its slice against a DIFFERENT bound size holds a
        # stale share: the entry order it waits on was re-placed at a smaller size.
        # It re-derives its own slice the next time the script issues it, so it must
        # not block this leg in the meantime -- otherwise a shrunk bracket stays
        # frozen at its first size and stops tracking the stop level the script
        # keeps moving.
        # A leg restored by ``BrokerPosition.reconstruct_exit_order`` after a restart
        # carries no basis at all (``bound_size`` stays 0.0, which no issued leg can
        # have -- a zero bound reserves nothing and returns above). It still holds a
        # real live broker reservation, so it counts until the script re-issues it
        # and stamps its own basis; skipping it would let the reissued sibling
        # reserve the whole entry and protect more exposure than the script allocated.
        bound = abs(init_size)
        sibling = sum(o.reserved_size for o in position.exit_orders.values()
                      if o.entry_seq == entry_seq and o.order_id == from_entry
                      and o is not existing
                      and o.book_seq is None
                      and (o.bound_size == bound or o.bound_size == 0.0))
        unreserved = bound - sibling
        # A qty/qty_percent leg is capped at the unreserved remainder --
        # TradingView never lets a later exit call take a slice a pre-existing
        # leg already holds. Verified on live TV (BINANCE:BTCUSDT 30m probes):
        # a late qty_percent=50 or qty=1 leg issued while a no-qty stop leg
        # holds 100% never creates an order (553/553 cycles), and against a
        # qty_percent=75 stop leg the same call is reduced to the remaining
        # 25% instead of being dropped.
        if qty == qty:
            reserved = min(abs(qty), unreserved)
        elif qty_percent == qty_percent:
            reserved = min(abs(init_size) * (qty_percent * 0.01), unreserved)
        else:
            # No-qty "rest" leg: the whole unreserved remainder, so it never
            # over-closes the position.
            reserved = unreserved

        # The Pine-side lot floor is a backtest-only quantization. Broker
        # positions can be smaller than syminfo.mincontract after venue-domain
        # conversion, so preserve the raw reservation for plugin quantization.
        if isinstance(position, SimPosition):
            if qty == qty and abs(qty) < unreserved:
                reserved = _explicit_qty_round(abs(qty))
            else:
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
        if (not (limit == limit or stop == stop or profit == profit or loss == loss)
                and _trail_price is None and trail_points_ticks is None):
            return

        _limit = _na_to_none(limit)
        if _limit is not None:
            _limit = _price_round(_limit, direction)
        _stop = _na_to_none(stop)
        if _stop is not None:
            _stop = _price_round(_stop, -direction)
        if _trail_price is not None:
            _trail_price = _price_round(_trail_price, -direction)

        # Default OCA settings for strategy.exit() - matches TradingView behavior.
        # Pine's strategy.exit() has no oca_type parameter: its legs always form a
        # reduce group. If no oca_name is specified, create a default one. It is
        # per ENTRY as well as per exit id: the legs TradingView issues for two
        # pyramid adds are independent, so one add's fill must not reduce the
        # other's leg. The name is built into a local -- assigning the caller's
        # ``oca_name`` would leak the first entry's group onto every later leg.
        leg_oca = oca_name
        if isinstance(leg_oca, NA):
            # Use a unique name based on the exit id and from_entry
            leg_oca = f"__exit_{id}_{from_entry}_{entry_seq}_oca__"

        # Add order
        order = Order(
            from_entry, size, exit_id=id, order_type=_order_type_close,
            limit=_limit, stop=_stop,
            trail_price=_trail_price, trail_offset=_trail_offset,
            profit_ticks=profit_ticks, loss_ticks=loss_ticks, trail_points_ticks=trail_points_ticks,
            oca_name=_na_to_none(leg_oca), oca_type=_oca.reduce,
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
        order.bound_size = bound
        order.entry_seq = entry_seq
        position._add_order(order)
        # Only an identical re-issue folds the issue bar's extreme into the water
        # mark -- and there it is a no-op, since the carried leg was already
        # walked through this bar. A brand-new leg (first issue, or trailing added
        # to a live bracket) and a changed-params re-issue both anchor to the
        # issue bar's CLOSE tick (see ``_seed_trail_at_issue``).
        position._seed_trail_at_issue(order, fold_extreme=had_trail and trail_unchanged)

    # noinspection PyProtectedMember
    def _filled_targets(entry_id: str | None) -> list[tuple[int | None, str, float, float]]:
        """``(entry_seq, entry_id, sign, ORIGINAL size)`` per entry an exit binds to.

        TradingView issues ONE leg per FILLED ENTRY, so two pyramid adds sharing a
        ``from_entry`` id each get their own leg reserved off their own entry size
        — and each is consumed on its own, which is why a bracket can fire again
        for a later add after it already fired for the first. The live broker has
        no binding book, so it keeps the pre-fan-out shape: one target per entry
        id, reserved off that id's combined open size.

        :param entry_id: The ``from_entry`` to bind to, or None for all entries
        """
        if isinstance(position, SimPosition):
            book = [b for b in position._entry_book
                    if entry_id is None or b.entry_id == entry_id]
            # A position a gap-committed exit leg opened (see Order.gap_committed)
            # takes its leg BEFORE the entries that were already open. MEASURED on
            # TradingView (BINANCE:BTCUSDT 30m): closing long 1 / short 5 + exit-leg
            # 1 with one from_entry-less strategy.exit reports the closed rows as
            # 1, 4, 1 -- the exit-opened leg's single unit is taken off the OLDEST
            # trade first, splitting it -- while the same exit over two plain
            # pyramid adds (1 and 2) reports a clean 1, 2.
            book.sort(key=lambda b: not b.exit_opened)
            return [(b.seq, b.entry_id or "", b.sign, b.init_size) for b in book]
        grouped: dict[str, list[float]] = {}
        for open_trade in position.open_trades:
            trade_id = open_trade.entry_id or ""
            if entry_id is not None and open_trade.entry_id != entry_id:
                continue
            slot = grouped.setdefault(trade_id, [0.0, 0.0])
            slot[0] = open_trade.sign
            slot[1] += abs(open_trade.init_size)
        return [(None, trade_id, slot[0], slot[1]) for trade_id, slot in grouped.items()]

    # noinspection PyProtectedMember
    def _pending_size(entry_id: str) -> tuple[float, float]:
        """Sign and bindable size of a still-pending entry order, at its CURRENT size.

        A leg issued against it carries no ``entry_seq`` until the order fills;
        :meth:`SimPosition._bind_entry` then hands it to the entry it produced.
        """
        sign = 0.0
        total = 0.0
        pending = position.entry_orders.get(entry_id)
        if pending is not None:
            sign = pending.sign
            # Only the not-yet-filled remainder of the entry order counts. The
            # backtest simulator removes a market entry order on fill, so
            # ``filled_qty`` stays 0.0 and this is simply ``abs(pending.size)``.
            # The live broker keeps the entry Order in ``entry_orders`` for
            # intent stability while ``record_fill`` moves the filled slice into
            # ``open_trades``; counting the full order size there would
            # double-count the fill and over-reserve the exit (issue BYBIT-001).
            # A market entry that adds to a same-direction position is re-checked
            # against the pyramiding limit when it is processed and dropped there
            # without ever reaching the position, so it must not enlarge the slice
            # this leg reserves. The inflated reservation would also be sticky: on
            # the next bar the sibling leg finds nothing unreserved left and keeps
            # its own oversized share, so a qty_percent leg goes on closing the
            # whole position. TradingView keeps such an exit at half the position
            # that actually exists -- measured on the CAPITALCOM:EURUSD 30m
            # reference of the "TradingView Alerts to MT4 MT5" strategy, whose
            # ``GoShort`` fires again while the short is already open.
            rejected_pyramid = (pending.limit is None and pending.stop is None
                                and position.sign == pending.sign
                                and lib._script.pyramiding <= position._pyramid_count())
            unfilled = 0.0 if rejected_pyramid else abs(pending.size) - pending.filled_qty
            # Only the part of a reversal order that actually OPENS is bindable: the
            # rest closes the opposite position, which carries its own bracket. A
            # market order whose flip is still added at processing already carries
            # the openable size; an order that has the flip baked in (a price-based
            # entry, or one whose flip was consumed by the order it replaced) does
            # not. Counting the closing leg here reserved half of the ORDER instead
            # of half of the position, so a sibling leg then found nothing left and
            # the whole sticky bracket stopped re-issuing its updated stop.
            flip_pending = (pending.limit is None and pending.stop is None
                            and not pending.skip_flip)
            if not flip_pending and position.size != 0.0 and position.sign != pending.sign:
                # A price-based entry FROZE its augmentation at placement time
                # (``_size_flippable_by_entry``), so only that much of the order
                # closes the opposite position — reading the live position instead
                # subtracts a close the order never carries. It reads zero whenever
                # the script flattens with its own ``strategy.close`` before placing
                # the entry, and the whole sticky bracket then bound nothing and
                # never armed. A market order that skipped its flip keeps carrying
                # the live position.
                priced = pending.limit is not None or pending.stop is not None
                unfilled -= pending.flip_extra if priced else abs(position.size)
            if unfilled > 0.0:
                total += unfilled
        return sign, total

    # Find direction and size
    if from_entry:
        # One leg per filled entry, plus one for a still-pending entry order.
        # The position should be open, or an entry order should exist.
        for entry_seq, _target_id, direction, init_size in _filled_targets(from_entry):
            _exit()
        direction, init_size = _pending_size(from_entry)
        if direction:
            entry_seq = None
            _exit()

    else:
        for entry_seq, from_entry, direction, init_size in _filled_targets(None):
            _exit()
        in_position = bool(direction)

        # A still-pending entry order is a target too -- but only one that OPENS
        # from flat or ADDS in the position's own direction, so its bracket is
        # already live on the bar it fills. The opposite leg of a REVERSAL is not:
        # that exit covers the position being reversed out of, and the position
        # the reversal opens gets its own bracket from the NEXT bar's script run,
        # so its stop cannot fire on the bar the reversal filled.
        # MEASURED on TradingView (CAPITALCOM:EURUSD 60, "Technical Ratings
        # Strategy", 580 trades). Of the entries whose stop level was breached on
        # their own fill bar, all 5 that were opened from FLAT exited on that bar,
        # and the 1 opened by a reversal held -- a clean 6/6 split.
        # Binding pending entries first also left the position being reversed out
        # of with no bracket at all for the length of that bar.
        # The same-direction ADD is measured on BINANCE:BTCUSDT 30m (the "MACD
        # Strategy with trailing ATR stop" reference): its from_entry-less
        # qty_percent leg fires on the pyramid add's OWN fill bar, exactly like
        # the from_entry-bound leg beside it.
        entry_seq = None
        for order in list(position.entry_orders.values()):
            if in_position and order.sign != position.sign:
                continue
            from_entry = order.order_id or ""
            direction, init_size = _pending_size(from_entry)
            if not direction:
                continue
            # Only mark as from_entry_na on first creation (not replacement)
            exit_key = _exit_key(id, from_entry)
            had_existing_exit = exit_key in position.exit_orders
            _exit()
            if not had_existing_exit:
                exit_order = position.exit_orders.get(exit_key)
                if exit_order is not None:
                    exit_order.from_entry_na = True


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

    if not (limit == limit):  # is_na_arg
        limit = None
    elif limit is not None:
        limit = _price_round(limit, direction_sign)  # TODO: test this if the direction here is correct
    if not (stop == stop):  # is_na_arg
        stop = None
    elif stop is not None:
        stop = _price_round(stop, -direction_sign)  # TODO: test this if the direction here is correct

    # A default-sized order resolves its quantity at the actual fill price
    # (TradingView sizes percent_of_equity / cash when the order executes).
    # The size computed here is the placement estimate, taken at the price the
    # order would execute at NOW — the current price when immediately
    # executable (including slippage, see strategy.entry), the limit/stop
    # price while it rests.
    deferred_default = not (qty == qty)  # is_na_arg
    market_sizing_price: float | None = None
    exec_price = 0.0  # only meaningful when deferred_default
    if deferred_default:
        exec_price = float(lib.close)
        if limit is not None:
            exec_price = min(limit, exec_price) if direction_sign > 0 else max(limit, exec_price)
        elif stop is not None:
            exec_price = max(stop, exec_price) if direction_sign > 0 else min(stop, exec_price)
        else:
            slippage = lib._script.slippage
            if slippage > 0 and isinstance(position, SimPosition):
                exec_price += direction_sign * syminfo.mintick * slippage
            market_sizing_price = exec_price
        qty = _default_entry_qty(exec_price)

    # qty must be a positive finite number; an unsizable NaN or infinite qty is
    # dropped with the non-positive ones (see strategy.entry)
    if not (0.0 < qty < math.inf):  # unsizable_qty
        return

    size = qty * direction_sign

    # NOTE: Unlike strategy.entry, strategy.order is NOT affected by pyramiding limit
    # This is a key difference - strategy.order can open unlimited trades in the same direction
    # It uses _order_type_normal to distinguish it from entry/exit orders

    # The Pine-side lot floor is a backtest-only quantization (see strategy.entry):
    # in broker mode the venue owns the quantity grid, so keep the raw requested
    # qty and let the plugin's preflight report a below-minimum skip instead of
    # silently dropping a live signal here.
    if isinstance(position, SimPosition):
        size = _size_floor(size) if deferred_default \
            else direction_sign * _explicit_qty_round(float(qty))
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
    # Only price-based orders re-size at execution (see strategy.entry);
    # the money budget is frozen at placement (see _resolve_deferred_qty)
    if deferred_default and (limit is not None or stop is not None):
        order.deferred_qty = True
        budget = _default_entry_budget(float(exec_price))
        if budget is not None:
            order.budget_money = budget[0]
            order.budget_pv = _account_point_value()
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
def account_currency() -> PyneStr:
    """
    The currency of the strategy account, in which the monetary strategy values are expressed.

    :return: The account currency code (e.g. "USD")
    """
    # Measured on TradingView (FX:EURUSD 1D): with the default currency=currency.NONE the
    # account currency is the symbol's quote currency ("USD" on EURUSD); with
    # currency=currency.EUR it is "EUR".
    if lib._script is None:
        return na_str
    cur = str(lib._script.currency)
    if cur == 'NONE':
        return syminfo.currency
    return cur


# noinspection PyProtectedMember
@module_property
def avg_losing_trade() -> PyneFloat:
    # Divides the PUBLISHED gross loss -- open commission included, like the
    # ``grossloss`` getter -- and reports it POSITIVE, even though
    # ``avg_losing_trade_percent`` is negative. Measured on BINANCE:BTCUSDT 30m:
    # with one losing trade of 602.310640000007 booked and a short leg open at
    # 186.8205 commission, TradingView reports 789.13114; the raw field alone
    # answers 602.31064 and drifts on 14410 of 28840 bars.
    if lib._script is None:
        return 0.0
    position = lib._script.position
    if position.losstrades == 0:
        return na_float
    return (position.grossloss + position.open_commission) / position.losstrades


# noinspection PyProtectedMember
@module_property
def avg_losing_trade_percent() -> PyneFloat:
    if lib._script is None:
        return 0.0
    position = lib._script.position
    if position.losstrades == 0:
        return na_float
    return position.sum_loss_profit_ratio / position.losstrades * 100.0


# noinspection PyProtectedMember
@module_property
def avg_trade() -> PyneFloat:
    if lib._script is None:
        return 0.0
    position = lib._script.position
    if position.closed_trades_count == 0:
        return na_float
    return position.netprofit / position.closed_trades_count


# noinspection PyProtectedMember
@module_property
def avg_trade_percent() -> PyneFloat:
    # NOT ``netprofit_percent / closedtrades``: TradingView averages the closed
    # trades' OWN percentages, each measured against that trade's entry cost.
    # Measured on BINANCE:BTCUSDT 30m (initial capital 1e6, ~1 BTC legs, so the
    # two forms differ by an order of magnitude): after 4 trades TV reports
    # -0.41167387332462313 where the capital-relative form gives -0.13109, and
    # ``avg_trade_percent * closedtrades`` equals
    # ``avg_winning_trade_percent * wintrades + avg_losing_trade_percent *
    # losstrades`` on all 28837 bars.
    if lib._script is None:
        return 0.0
    position = lib._script.position
    if position.closed_trades_count == 0:
        return na_float
    return position.sum_profit_ratio / position.closed_trades_count * 100.0


# noinspection PyProtectedMember
@module_property
def avg_winning_trade() -> PyneFloat:
    if lib._script is None:
        return 0.0
    position = lib._script.position
    if position.wintrades == 0:
        return na_float
    return position.grossprofit / position.wintrades


# noinspection PyProtectedMember
@module_property
def avg_winning_trade_percent() -> PyneFloat:
    if lib._script is None:
        return 0.0
    position = lib._script.position
    if position.wintrades == 0:
        return na_float
    return position.sum_win_profit_ratio / position.wintrades * 100.0


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
def grossloss_percent() -> PyneFloat:
    # Percent of the INITIAL CAPITAL, over the same grossloss the plain
    # property reports -- open commission included, so a position that is
    # still open already shows a loss percent (measured: one open 1 BTC leg at
    # 0.05% commission reports 0.004688095 against a 1e6 capital).
    if lib._script is None:
        return 0.0
    initial = lib._script.initial_capital
    if initial == 0.0:
        return 0.0
    position = lib._script.position
    return (position.grossloss + position.open_commission) / initial * 100.0


# noinspection PyProtectedMember
@module_property
def grossprofit() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.grossprofit


# noinspection PyProtectedMember
@module_property
def grossprofit_percent() -> PyneFloat:
    if lib._script is None:
        return 0.0
    initial = lib._script.initial_capital
    if initial == 0.0:
        return 0.0
    return lib._script.position.grossprofit / initial * 100.0


# noinspection PyProtectedMember
@module_property
def losstrades() -> int:
    if lib._script is None:
        return 0
    return lib._script.position.losstrades


# noinspection PyProtectedMember
@module_property
def margin_liquidation_price() -> PyneFloat:
    """
    The price at which the margin call of the open position occurs.

    :return: The margin call price, na while flat, when the position's side requires no
             margin, or in live broker mode (the exchange owns the margin state)
    """
    # Measured on TradingView (BINANCE:BTCUSDT 1D and FX:EURUSD 1D, margin_long=25,
    # margin_short=30): the value solves ``equity(P) = margin(P)`` with
    # ``equity(P) = initial_capital + netprofit + sign * qty * pv * (P - avg_price)`` and
    # ``margin(P) = margin% * qty * pv * P`` — the same balance ``_check_margin_call``
    # compares — then snaps to the tick grid DIRECTIONALLY: a long position floors toward
    # -inf (4933.3866 -> 4933.38 and -255066.6133 -> -255066.62, refuting both
    # nearest-rounding and trunc-toward-zero), a short position ceils
    # (160805.9538 -> 160805.96, 1.1935384 -> 1.19354). TV happily reports a negative
    # price for an unreachable long liquidation. Flat bars and a zero margin percent on
    # the position's side give na (margin_long=0 with an open long is na even while
    # margin_short=30). The pointvalue factor follows the engine's other monetary
    # values; the probe symbols have pointvalue 1.
    script = lib._script
    if script is None:
        return 0.0
    position = script.position
    if not isinstance(position, SimPosition):
        # Live broker mode: the exchange owns collateral, leverage and
        # maintenance-margin tiers — strategy.margin_long/short and
        # initial_capital + netprofit below do not describe them, so the
        # backtest formula would fabricate an unrelated price.
        return na_float
    sign = position.sign
    if sign == 0:
        return na_float
    margin_percent = script.margin_short if sign < 0 else script.margin_long
    if margin_percent <= 0:
        return na_float
    margin_ratio = margin_percent / 100.0
    qpv = abs(position.size) * _account_point_value()
    capital = script.initial_capital + position.netprofit
    if sign > 0:
        denom = qpv * (1.0 - margin_ratio)
        if denom == 0.0:
            return na_float
        price = (qpv * position.avg_price - capital) / denom
    else:
        denom = qpv * (1.0 + margin_ratio)
        price = (capital + qpv * position.avg_price) / denom
    # Directional tick snap on the minmove/pricescale grid (same grid as _price_round,
    # but with a true floor for longs — _price_round truncates toward zero instead).
    pricescale = syminfo.pricescale
    minmove = syminfo.minmove
    tick_count = round(price * pricescale / minmove, 7)
    if sign > 0:
        return math.floor(tick_count) * minmove / pricescale
    return math.ceil(tick_count) * minmove / pricescale


# noinspection PyProtectedMember
@module_property
def max_contracts_held_all() -> PyneFloat:
    if lib._script is None:
        return 0.0
    position = lib._script.position
    long_peak = position.max_contracts_held_long
    short_peak = position.max_contracts_held_short
    return long_peak if long_peak > short_peak else short_peak


# noinspection PyProtectedMember
@module_property
def max_contracts_held_long() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.max_contracts_held_long


# noinspection PyProtectedMember
@module_property
def max_contracts_held_short() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.max_contracts_held_short


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
    return lib._script.position.max_drawdown_percent


# noinspection PyProtectedMember
@module_property
def max_runup() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.max_runup


# noinspection PyProtectedMember
@module_property
def max_runup_percent() -> PyneFloat:
    if lib._script is None:
        return 0.0
    return lib._script.position.max_runup_percent


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
def position_entry_name() -> PyneStr:
    if lib._script is None:
        return na_str
    open_trades = lib._script.position.open_trades
    if not open_trades:
        return na_str
    # MEASURED (probe m600, BINANCE:BTCUSDT 30m): the id of the OLDEST still-open
    # entry, not the latest one -- with A and B pyramided it stays "A", and only
    # after `strategy.close("A")` does it become "B". A reversal leaves the
    # flipping entry as the sole open trade, so it names that one. Flat is `na`.
    entry_id_ = open_trades[0].entry_id
    return entry_id_ if entry_id_ is not None else na_str


# noinspection PyProtectedMember
@module_property
def wintrades() -> PyneInt:
    if lib._script is None:
        return 0
    return lib._script.position.wintrades
