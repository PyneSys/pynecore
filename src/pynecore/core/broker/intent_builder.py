"""
Translate the Pine ``strategy.*`` order book into broker intent objects.

The :class:`~pynecore.lib.strategy.SimPosition` maintains two dictionaries
of pending Pine orders created by ``strategy.entry``, ``strategy.order``,
``strategy.exit``, ``strategy.close`` and ``strategy.close_all``:

- ``position.entry_orders[pine_id]`` — entries and strategy-order adds
- ``position.exit_orders[(exit_id, from_entry)]`` — exits, closes and
  close-all (composite key disambiguates multiple exits per entry)

For live trading the :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine`
consumes these via :func:`build_intents`, which returns a flat list of
:class:`EntryIntent`, :class:`ExitIntent` and :class:`CloseIntent`
dataclasses the broker plugin can act on. The translation is **pure**:
no access to ``lib._script`` or ``syminfo``; the caller supplies the
symbol and keeps tick/mintick resolution inside the sync engine.
"""
from typing import Iterable

from pynecore.core.broker.models import (
    EntryIntent,
    ExitIntent,
    CloseIntent,
    OcaType,
    OrderType,
)
# noinspection PyProtectedMember
from pynecore.lib.strategy import (
    Order,
    Trade,
    _order_type_normal,
)
from pynecore.types.na import NA

__all__ = [
    'build_intents',
    'build_entry_intent',
    'build_exit_intent',
    'build_close_intent',
    'CLOSE_EXIT_ID_PREFIX',
    'CLOSE_ALL_EXIT_ID',
]

# Prefixes set by ``strategy.close`` / ``strategy.close_all`` in
# ``lib/strategy/__init__.py``. Used to distinguish a CloseIntent from an
# ExitIntent when both share ``_order_type_close``.
CLOSE_EXIT_ID_PREFIX = "Close entry(s) order "
CLOSE_ALL_EXIT_ID = "Close position order"


def _side_from_size(size: float) -> str:
    """Signed Pine size → ``"buy"``/``"sell"``.

    Pine uses signed sizes: positive for long, negative for short. Exit/close
    orders always carry the **opposite** sign of the position, so a long close
    has a negative size → ``"sell"``, matching the exchange-side semantics.
    """
    return "buy" if size > 0 else "sell"


def _infer_order_type(limit: float | None, stop: float | None) -> OrderType:
    """Map a Pine entry's ``limit``/``stop`` prices to a broker order type.

    Pine has no "stop-limit" entry. When BOTH prices are set the order is two
    OCO legs: a LIMIT below the open (guaranteed-price pullback) and a STOP
    above (market-on-rise). The sync engine realises the both-set case as a
    native resting LIMIT plus a software price-watch on the stop side, so the
    inferred type is LIMIT — both-set and limit-only share the same resting
    leg. The ``EntryIntent`` still carries the ``stop`` price for the engine
    to arm its watch.
    """
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

    Unknown ``oca_type`` strings are rejected here rather than silently passed
    through to the sync engine, where a typo would disable cascade cancel
    without any diagnostic. The accepted values are exactly the members of
    :class:`OcaType`.
    """
    if order.oca_name is None:
        return None, None
    if order.oca_type is None:
        return order.oca_name, None
    oca_type_str = str(order.oca_type)
    try:
        OcaType(oca_type_str)
    except ValueError as exc:
        raise ValueError(
            f"unknown oca_type {oca_type_str!r}; expected one of "
            f"{[m.value for m in OcaType]}",
        ) from exc
    return order.oca_name, oca_type_str


def build_entry_intent(order: Order, symbol: str) -> EntryIntent:
    """Translate a ``strategy.entry`` / ``strategy.order`` Pine order.

    The intent carries the RAW script quantity — deliberately. Pine keeps
    a consumed market entry's ``Order`` in ``entry_orders`` (it is never
    popped), so ``build_intents`` re-emits the same intent every bar and
    the sync engine's diff relies on it being byte-stable to recognise an
    already-dispatched entry. The MARKET-reversal stop-and-reverse fold
    (TV: combined = qty + |opposite position|) therefore happens at
    DISPATCH time in :meth:`OrderSyncEngine._dispatch_new`, which runs
    exactly once per fresh intent — folding here would make a stale
    retained order's qty flap with the net position and re-trigger the
    diff (measured on the Capital.com demo E2E, 2026-07-10).
    """
    oca_name, oca_type_str = _coerce_oca(order)
    return EntryIntent(
        pine_id=order.order_id or "",
        symbol=symbol,
        side=_side_from_size(float(order.size)),
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


def build_exit_intent(
        order: Order,
        symbol: str,
        *,
        parent_total_qty: float = 0.0,
) -> ExitIntent:
    """Translate a ``strategy.exit`` Pine order.

    Tick-based exits (``profit=``/``loss=``/``trail_points=``) carry unresolved
    distances: the plugin cannot place absolute TP/SL prices until the entry
    fill price is known. The intent preserves the tick values
    **alongside** empty ``tp_price``/``sl_price`` fields; the sync engine
    resolves them on the corresponding entry fill event.

    :param order: The Pine ``strategy.exit`` :class:`Order` to translate.
    :param symbol: Exchange symbol the resulting :class:`ExitIntent` targets.
    :param parent_total_qty: total open qty currently associated with the
        ``from_entry`` Pine id — used to set
        :attr:`ExitIntent.is_partial_qty_bracket` when the script asks for a
        bracket on a fraction of the parent. ``0.0`` (the default) leaves
        the flag ``False``; ``build_intents`` supplies the live value from
        ``position.open_trades`` so the order sync engine can dispatch on
        ``caps.partial_qty_bracket_exit``.
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
    has_bracket_leg = (
        tp_price is not None
        or sl_price is not None
        or trail_price is not None
        or trail_offset is not None
        or order.profit_ticks is not None
        or order.loss_ticks is not None
        or order.trail_points_ticks is not None
    )
    exit_qty = abs(order.size)
    # A bracket on the whole row stays on the full-row dispatch path even
    # when the capability advertises a partial-aware mechanism — Pine
    # semantics for ``qty == total`` are identical to the simple bracket.
    is_partial = (
        has_bracket_leg
        and parent_total_qty > 0.0
        and exit_qty + 1e-12 < parent_total_qty
    )
    return ExitIntent(
        pine_id=order.exit_id or "",
        from_entry=order.order_id or "",
        symbol=symbol,
        side=_side_from_size(float(order.size)),
        qty=exit_qty,
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
        is_partial_qty_bracket=is_partial,
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
        side=_side_from_size(float(order.size)),
        qty=abs(order.size),
        immediately=False,
        comment=_na_to_none(order.comment),
        alert_message=_na_to_none(order.alert_message),
    )


def _classify_exit_side(order: Order) -> str:
    """``'close_all' | 'close' | 'exit'`` — the exit_orders dict is polymorphic."""
    if order.exit_id == CLOSE_ALL_EXIT_ID:
        return 'close_all'
    if order.exit_id and order.exit_id.startswith(CLOSE_EXIT_ID_PREFIX):
        return 'close'
    return 'exit'


def build_intents(
    entry_orders: dict,
    exit_orders: dict,
    symbol: str,
    open_trades: Iterable[Trade] | None = None,
) -> list[EntryIntent | ExitIntent | CloseIntent]:
    """Flatten a position's pending orders into intent objects.

    ``entry_orders`` and ``exit_orders`` are the ``dict``s that
    :class:`~pynecore.lib.strategy.SimPosition` exposes. Orders already
    marked ``cancelled`` (e.g. via OCA or :func:`strategy.cancel`) are
    filtered out — the caller treats their absence as an implicit cancel.

    ``open_trades`` is ``position.open_trades`` — used solely to compute the
    total open qty per ``from_entry`` for the partial-qty bracket flag on
    :class:`ExitIntent`. Optional; when omitted the flag stays ``False``
    and the exit falls back to the full-row dispatch path (the safe
    default for unit tests and simulator-only callers that don't carry a
    live trade ledger).
    """
    intents: list[EntryIntent | ExitIntent | CloseIntent] = []

    qty_by_entry: dict[str, float] = {}
    count_by_entry: dict[str, int] = {}
    if open_trades is not None:
        for trade in open_trades:
            entry_id = trade.entry_id
            if entry_id is None:
                continue
            qty_by_entry[entry_id] = (
                qty_by_entry.get(entry_id, 0.0) + abs(trade.size)
            )
            count_by_entry[entry_id] = count_by_entry.get(entry_id, 0) + 1

    for order in _active(entry_orders.values()):
        intents.append(build_entry_intent(order, symbol))

    for order in _active(exit_orders.values()):
        kind = _classify_exit_side(order)
        if kind == 'close_all':
            intents.append(build_close_intent(order, symbol, is_close_all=True))
        elif kind == 'close':
            intents.append(build_close_intent(order, symbol, is_close_all=False))
        else:
            from_entry = order.order_id or ""
            # ``parent_total_qty`` drives ``ExitIntent.is_partial_qty_bracket``
            # (the partial-vs-whole-row dispatch). It is sourced by the open
            # row count under ``from_entry``. This relies on a simulator
            # invariant: ``build_intents`` reads ``SimPosition.open_trades``,
            # where each Pine entry event is exactly one ``Trade`` row (the
            # simulator has no split fills — an order fills in full when its
            # condition is met).
            #
            #  * >1 row — pyramiding or repeated ``strategy.order`` under one
            #    Pine id. ``entry_orders[from_entry]`` keeps only the LATEST
            #    same-id row (``_add_order`` overwrites the dict slot), so the
            #    declared size understates the parent and would misclassify a
            #    partial exit as whole-row, mis-hedging the older legs. The
            #    summed open qty is the only faithful total here — and it is
            #    exactly what Pine's ``strategy.exit(qty=N, from_entry=...)``
            #    measures N against.
            #  * ==1 row — single-row: use the script-declared size. It is the
            #    stable "what the script asked for" signal and survives the
            #    fill (``record_fill`` only touches ``open_trades``, never pops
            #    the entry Order), so a broker overfill/partial-fill cannot flip
            #    the flag mid-lifecycle and trigger a spurious native<->partial
            #    bracket cancel/re-arm.
            #  * no declared entry left (cancelled / cleared) — fall back to the
            #    actual open qty.
            declared_entry: Order | None = entry_orders.get(from_entry)
            if count_by_entry.get(from_entry, 0) > 1:
                parent_total_qty = qty_by_entry[from_entry]
            elif declared_entry is not None:
                parent_total_qty = abs(float(declared_entry.size))
            else:
                parent_total_qty = qty_by_entry.get(from_entry, 0.0)
            intents.append(build_exit_intent(
                order, symbol, parent_total_qty=parent_total_qty,
            ))

    return intents


def _active(orders: Iterable[Order]) -> Iterable[Order]:
    for order in orders:
        if not order.cancelled:
            yield order
