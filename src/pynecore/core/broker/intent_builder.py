"""
Translate the Pine ``strategy.*`` order book into broker intent objects.

The :class:`~pynecore.lib.strategy.SimPosition` maintains two dictionaries
of pending Pine orders created by ``strategy.entry``, ``strategy.order``,
``strategy.exit``, ``strategy.close`` and ``strategy.close_all``:

- ``position.entry_orders[pine_id]`` — entries and strategy-order adds
- ``position.exit_orders[from_entry]`` — exits, closes and close-all

For live trading the :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine`
consumes these via :func:`build_intents`, which returns a flat list of
:class:`EntryIntent`, :class:`ExitIntent` and :class:`CloseIntent`
dataclasses the broker plugin can act on. The translation is **pure**:
no access to ``lib._script`` or ``syminfo``; the caller supplies the
symbol and keeps tick/mintick resolution inside the sync engine.
"""
from __future__ import annotations

from typing import Iterable

from pynecore.core.broker.models import (
    EntryIntent,
    ExitIntent,
    CloseIntent,
    OrderType,
)
from pynecore.lib.strategy import (
    Order,
    _order_type_normal,
    _order_type_close,
)
from pynecore.types.na import NA

__all__ = [
    'build_intents',
    'build_entry_intent',
    'build_exit_intent',
    'build_close_intent',
]

# Prefixes set by ``strategy.close`` / ``strategy.close_all`` in
# ``lib/strategy/__init__.py``. Used to distinguish a CloseIntent from an
# ExitIntent when both share ``_order_type_close``.
_CLOSE_EXIT_ID_PREFIX = "Close entry(s) order "
_CLOSE_ALL_EXIT_ID = "Close position order"


def _side_from_size(size: float) -> str:
    """Signed Pine size → ``"buy"``/``"sell"``.

    Pine uses signed sizes: positive for long, negative for short. Exit/close
    orders always carry the **opposite** sign of the position, so a long close
    has a negative size → ``"sell"``, matching the exchange-side semantics.
    """
    return "buy" if size > 0 else "sell"


def _infer_order_type(limit: float | None, stop: float | None) -> OrderType:
    if limit is not None and stop is not None:
        return OrderType.STOP_LIMIT
    if limit is not None:
        return OrderType.LIMIT
    if stop is not None:
        return OrderType.STOP
    return OrderType.MARKET


def _na_to_none(value):
    """Strip Pine :class:`NA` markers; leave concrete values alone."""
    return None if isinstance(value, NA) else value


def _coerce_oca(order: Order) -> tuple[str | None, str | None]:
    """Return a protocol-friendly ``(oca_name, oca_type)`` pair.

    ``Order`` always stores a non-None ``oca_type`` (defaults to :data:`oca.none`)
    even when no OCA participation is requested. The intent layer uses
    ``None`` to mean "not in an OCA group" — so only emit ``oca_type`` when
    the order actually names a group.
    """
    if order.oca_name is None:
        return None, None
    return order.oca_name, str(order.oca_type) if order.oca_type is not None else None


def build_entry_intent(order: Order, symbol: str) -> EntryIntent:
    """Translate a ``strategy.entry`` / ``strategy.order`` Pine order."""
    oca_name, oca_type_str = _coerce_oca(order)
    return EntryIntent(
        pine_id=order.order_id or "",
        symbol=symbol,
        side=_side_from_size(order.size),
        qty=abs(order.size),
        order_type=_infer_order_type(order.limit, order.stop),
        limit=order.limit,
        stop=order.stop,
        oca_name=oca_name,
        oca_type=oca_type_str,
        comment=_na_to_none(order.comment),
        alert_message=_na_to_none(order.alert_message),
        is_strategy_order=(order.order_type == _order_type_normal),
    )


def build_exit_intent(order: Order, symbol: str) -> ExitIntent:
    """Translate a ``strategy.exit`` Pine order.

    Tick-based exits (``profit=``/``loss=``/``trail_points=``) carry unresolved
    distances: the plugin cannot place absolute TP/SL prices until the entry
    fill price is known. The intent preserves the tick values
    **alongside** empty ``tp_price``/``sl_price`` fields; the sync engine
    resolves them on the corresponding entry fill event.
    """
    oca_name, oca_type_str = _coerce_oca(order)
    # Tick values take priority over explicit prices — mirrors Pine's
    # exit() fill-time logic where profit_ticks overwrites order.limit.
    tp_price = order.limit if order.profit_ticks is None else None
    sl_price = order.stop if order.loss_ticks is None else None
    trail_price = order.trail_price if order.trail_points_ticks is None else None
    has_trail = (
        order.trail_price is not None or order.trail_points_ticks is not None
    )
    trail_offset = order.trail_offset if has_trail else None
    return ExitIntent(
        pine_id=order.exit_id or "",
        from_entry=order.order_id or "",
        symbol=symbol,
        side=_side_from_size(order.size),
        qty=abs(order.size),
        tp_price=tp_price,
        sl_price=sl_price,
        trail_price=trail_price,
        trail_offset=trail_offset,
        profit_ticks=order.profit_ticks,
        loss_ticks=order.loss_ticks,
        trail_points_ticks=order.trail_points_ticks,
        oca_name=oca_name,
        oca_type=oca_type_str,
        comment=_na_to_none(order.comment),
        comment_profit=_na_to_none(order.comment_profit),
        comment_loss=_na_to_none(order.comment_loss),
        comment_trailing=_na_to_none(order.comment_trailing),
        alert_message=_na_to_none(order.alert_message),
    )


def build_close_intent(order: Order, symbol: str, *, is_close_all: bool) -> CloseIntent:
    """Translate ``strategy.close(id)`` or ``strategy.close_all()``.

    ``strategy.close_all()`` uses an empty ``pine_id`` — the Pine order itself
    has ``order_id=None`` because the close targets the whole position rather
    than a specific entry identifier.
    """
    pine_id = "" if is_close_all else (order.order_id or "")
    return CloseIntent(
        pine_id=pine_id,
        symbol=symbol,
        side=_side_from_size(order.size),
        qty=abs(order.size),
        immediately=False,
        comment=_na_to_none(order.comment),
        alert_message=_na_to_none(order.alert_message),
    )


def _classify_exit_side(order: Order) -> str:
    """``'close_all' | 'close' | 'exit'`` — the exit_orders dict is polymorphic."""
    if order.exit_id == _CLOSE_ALL_EXIT_ID:
        return 'close_all'
    if order.exit_id and order.exit_id.startswith(_CLOSE_EXIT_ID_PREFIX):
        return 'close'
    return 'exit'


def build_intents(
    entry_orders: dict,
    exit_orders: dict,
    symbol: str,
) -> list[EntryIntent | ExitIntent | CloseIntent]:
    """Flatten a position's pending orders into intent objects.

    ``entry_orders`` and ``exit_orders`` are the ``dict``s that
    :class:`~pynecore.lib.strategy.SimPosition` exposes. Orders already
    marked ``cancelled`` (e.g. via OCA or :func:`strategy.cancel`) are
    filtered out — the caller treats their absence as an implicit cancel.
    """
    intents: list[EntryIntent | ExitIntent | CloseIntent] = []

    for order in _active(entry_orders.values()):
        intents.append(build_entry_intent(order, symbol))

    for order in _active(exit_orders.values()):
        kind = _classify_exit_side(order)
        if kind == 'close_all':
            intents.append(build_close_intent(order, symbol, is_close_all=True))
        elif kind == 'close':
            intents.append(build_close_intent(order, symbol, is_close_all=False))
        else:
            intents.append(build_exit_intent(order, symbol))

    return intents


def _active(orders: Iterable[Order]) -> Iterable[Order]:
    for order in orders:
        if not order.cancelled:
            yield order
