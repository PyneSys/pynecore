"""One-way position emulation over a hedging- or netting-mode broker.

A hedging cTrader account can hold several simultaneous open positions ("legs")
for one symbol — each with its own broker position id. Pine Script strategies,
however, see a single *one-way* position per symbol: at most one direction at a
time, opposite orders reduce and then flip it. This module is the core bridge
between the two representations, so a broker plugin only has to provide thin
transport primitives (``fetch_raw_positions`` / ``close_leg`` / ``place_leg``)
and carries none of the netting logic itself.

The two responsibilities live here as pure functions:

* :func:`aggregate_positions` collapses the raw legs into the single
  :class:`~pynecore.core.broker.models.ExchangePosition` the sync engine reads.
* :func:`select_legs_for_close` and :func:`plan_reversal` decide *which* legs to
  close, and by how much, for a reduce / close / reversal — always oldest-first
  (FIFO), matching Pine's trade accounting and giving replay-stable plans.

Aggregation tolerates transient opposite-direction legs (e.g. a pre-existing
manual hedge, or the instant between a reversal's close and open): the minority
side is netted virtually-FIFO against the oldest majority legs, and the
surviving side's volume-weighted price becomes the one-way entry price.
"""

from collections.abc import Callable
from dataclasses import dataclass

from pynecore.core.broker.models import ExchangePosition, PositionLeg

__all__ = [
    'LegClose',
    'ReversalPlan',
    'net_signed_qty',
    'aggregate_positions',
    'select_legs_for_close',
    'plan_reversal',
    'plan_leg_close_volumes',
    'legs_on_position_side',
]

# Quantities are in Pine units (fractional lots possible). Anything below this
# is treated as flat / fully consumed — coarse enough to absorb broker rounding,
# fine enough never to swallow a real partial lot.
_QTY_EPS = 1e-9


@dataclass(frozen=True)
class LegClose:
    """A single planned leg reduction: close ``qty`` of broker leg ``leg_id``."""
    leg_id: str
    qty: float


@dataclass(frozen=True)
class ReversalPlan:
    """Decomposition of a one-way order against the current legs.

    Produced by :func:`plan_reversal` for an order whose direction may oppose
    the open position. The order is split into a FIFO close of the opposing
    legs plus a residual open in the order's own direction.

    :ivar closes: Opposing legs to close (oldest first) and by how much. Empty
        when the order only adds to the existing direction.
    :ivar open_qty: Residual size to open in the order's direction after the
        closes. Zero when the order exactly flattens the opposite exposure.
    :ivar open_side: The order's direction (``"buy"`` / ``"sell"``); the side
        of any residual open.
    """
    closes: tuple[LegClose, ...]
    open_qty: float
    open_side: str


def net_signed_qty(legs: list[PositionLeg]) -> float:
    """Signed net size across legs: long (``buy``) positive, short negative."""
    return sum(leg.qty if leg.side == 'buy' else -leg.qty for leg in legs)


def _surviving_weighted_avg(legs: list[PositionLeg], keep_side: str) -> float:
    """Volume-weighted entry price of ``keep_side`` after virtual-FIFO netting.

    The opposite-side legs virtually close the oldest ``keep_side`` legs first
    (FIFO), exactly as a real reducing fill would; the price returned is the
    weighted average of whatever ``keep_side`` volume survives. Returns ``0.0``
    when nothing survives (fully netted).
    """
    survivors = sorted(
        (leg for leg in legs if leg.side == keep_side),
        key=lambda leg: (leg.open_time, leg.leg_id),
    )
    offset = sum(leg.qty for leg in legs if leg.side != keep_side)
    qty_acc = 0.0
    notional = 0.0
    for leg in survivors:
        remaining = leg.qty
        if offset > 0.0:
            consumed = min(remaining, offset)
            offset -= consumed
            remaining -= consumed
        if remaining > _QTY_EPS:
            qty_acc += remaining
            notional += remaining * leg.entry_price
    return notional / qty_acc if qty_acc > _QTY_EPS else 0.0


def aggregate_positions(
        symbol: str,
        legs: list[PositionLeg],
        *,
        leverage: float = 1.0,
        margin_mode: str = 'cross',
) -> ExchangePosition | None:
    """Collapse raw broker legs into one net one-way position snapshot.

    The net signed size decides the side; the surviving side's virtual-FIFO
    weighted-average price is the entry price; unrealized P&L is the plain sum
    of every leg's mark-to-market (correct regardless of leg direction).

    :param symbol: Pine symbol the legs belong to.
    :param legs: All open legs for ``symbol`` (any direction). May be empty.
    :param leverage: Carried onto the snapshot; legs do not report it per-leg.
    :param margin_mode: Carried onto the snapshot.
    :return: The aggregated :class:`ExchangePosition`, or ``None`` when there
        are no legs at all (a genuinely flat symbol — distinct from a netted
        ``flat`` snapshot, which is only produced when offsetting legs remain
        open but sum to zero).
    """
    if not legs:
        return None
    net = net_signed_qty(legs)
    unrealized = sum(leg.unrealized_pnl for leg in legs)
    if abs(net) <= _QTY_EPS:
        # Offsetting legs still open but net flat — surface it as flat with the
        # residual gross P&L so a caller can spot a stuck hedge rather than a
        # clean flat.
        return ExchangePosition(
            symbol=symbol, side='flat', size=0.0, entry_price=0.0,
            unrealized_pnl=unrealized, liquidation_price=None,
            leverage=leverage, margin_mode=margin_mode,
        )
    keep_side = 'buy' if net > 0.0 else 'sell'
    return ExchangePosition(
        symbol=symbol,
        side='long' if net > 0.0 else 'short',
        size=abs(net),
        entry_price=_surviving_weighted_avg(legs, keep_side),
        unrealized_pnl=unrealized,
        liquidation_price=None,
        leverage=leverage,
        margin_mode=margin_mode,
    )


def select_legs_for_close(
        qty: float, legs: list[PositionLeg], side: str,
) -> tuple[tuple[LegClose, ...], float]:
    """Pick which ``side`` legs to close for a ``qty`` reduction, oldest first.

    FIFO by :attr:`~pynecore.core.broker.models.PositionLeg.open_time` so the
    plan matches Pine's oldest-trade-first close order and is deterministic
    across restarts (a replay sees the same legs in the same order).

    :param qty: Total size to close (positive).
    :param legs: All open legs for the symbol; only those on ``side`` are
        considered.
    :param side: Direction of the legs to reduce (``"buy"`` for a long
        position, ``"sell"`` for a short).
    :return: ``(closes, shortfall)`` — the per-leg close plan and any quantity
        that could not be covered because the open legs summed to less than
        ``qty`` (``0.0`` in the normal case). A non-zero shortfall is the
        caller's signal that the broker holds less than Pine believes.
    """
    candidates = sorted(
        (leg for leg in legs if leg.side == side),
        # ``leg_id`` is the stable secondary key: two legs sharing an
        # ``open_time`` must close in the same order across a restart even if
        # the broker returns them in a different sequence, so the plan stays
        # replay-deterministic as the FIFO docstring promises.
        key=lambda leg: (leg.open_time, leg.leg_id),
    )
    closes: list[LegClose] = []
    remaining = qty
    for leg in candidates:
        if remaining <= _QTY_EPS:
            break
        take = min(leg.qty, remaining)
        if take > _QTY_EPS:
            closes.append(LegClose(leg_id=leg.leg_id, qty=take))
            remaining -= take
    # A residual at or below the tolerance is fully covered as far as the rest
    # of the module is concerned (the loop already stops consuming it), so it
    # must not leak out as a spurious positive shortfall for broker-rounded
    # quantities such as qty=1.0 against an open leg of 0.9999999995.
    return tuple(closes), remaining if remaining > _QTY_EPS else 0.0


def plan_reversal(
        side: str, qty: float, legs: list[PositionLeg],
) -> ReversalPlan:
    """Split a one-way order into opposite-leg closes plus a residual open.

    Pine folds a reversal into one combined order whose size is
    ``target + |opposite_net|`` (it assumes a netting auto-flip). On a hedging
    account that single order would instead open a separate opposing leg and
    bloat gross exposure, so the order is decomposed: close the opposing legs
    FIFO (up to the order size), then open whatever target size remains in the
    order's own direction.

    Works uniformly for the three shapes a single order can take:

    * **pure add** — no opposing legs: ``closes`` empty, ``open_qty == qty``;
    * **reversal** — opposing legs smaller than ``qty``: close them all, open
      the remainder (``qty - opposite_total``) in ``side``;
    * **partial reduce via order** — ``qty`` ≤ opposing exposure: close
      ``qty`` worth of opposing legs FIFO, ``open_qty == 0`` (no flip).

    :param side: The order's direction (``"buy"`` / ``"sell"``).
    :param qty: The combined order size as Pine dispatched it.
    :param legs: Current open legs for the symbol.
    :return: A :class:`ReversalPlan`.
    """
    opposite_side = 'sell' if side == 'buy' else 'buy'
    closes, _shortfall = select_legs_for_close(qty, legs, opposite_side)
    closed_total = sum(close.qty for close in closes)
    open_qty = qty - closed_total
    if open_qty < _QTY_EPS:
        open_qty = 0.0
    return ReversalPlan(closes=closes, open_qty=open_qty, open_side=side)


def plan_leg_close_volumes(
        closes: tuple[LegClose, ...],
        quantize: Callable[[float], int],
) -> list[tuple[str, int]]:
    """Fan a FIFO close plan out to per-leg broker-grid close volumes.

    Quantizing each FIFO slice independently changes the total close size on the
    broker's volume grid: a sub-grid slice rounds to ``0`` (an invalid empty
    close that under-reduces the position) and a fractional-grid slice rounds UP
    past what the leg holds. Snapping the *total* once is not enough either: if a
    slice that carries part of the owed total quantizes to ``0`` on its own —
    e.g. two 10-unit legs on a 1000-centi grid closing 15 units, where the
    second 5-unit slice snaps to ``0`` — per-slice rounded volumes leave the
    snapped total under-dispatched.

    Instead a single running total is snapped via ``quantize``: each leg receives
    the delta between the snapped cumulative close-through-this-leg and the
    snapped cumulative through the previous leg, capped at the snapped grand
    total. The sub-grid remainder a slice would have dropped is carried into the
    next leg, so the dispatched volumes always sum to the same grand total the
    single-position close path would use, every volume sits on the broker grid,
    and no leg gets more than its own slice rounded up to the next grid step. Any
    leg whose delta is zero is dropped (no zero-volume request is sent).

    :param closes: FIFO close plan (oldest first); each ``qty`` is the slice to
        take from that leg, in Pine units, never above the leg's open size.
    :param quantize: Maps a Pine-unit quantity to the broker's integer volume
        grid (e.g. cTrader centi-units snapped to ``stepVolume``). Owns the
        broker-specific unit; the FIFO carry logic here stays unit-agnostic.
    :return: ``(leg_id, volume)`` pairs to dispatch in FIFO order, each with the
        broker-grid integer ``volume`` the per-leg close expects. ``leg_id``
        stays the :class:`PositionLeg` string id; the transport casts it.
    """
    grand_total = quantize(sum(close.qty for close in closes))
    out: list[tuple[str, int]] = []
    cumulative_units = 0.0
    dispatched = 0
    for close in closes:
        if dispatched >= grand_total:
            break
        cumulative_units += close.qty
        snapped = min(quantize(cumulative_units), grand_total)
        volume = snapped - dispatched
        if volume <= 0:
            continue
        out.append((close.leg_id, volume))
        dispatched = snapped
    return out


def legs_on_position_side(
        legs: list[PositionLeg],
) -> tuple[str, list[PositionLeg]]:
    """Return the aggregate one-way side and the legs that make it up, FIFO.

    The net signed size decides the side (``'buy'`` for a net-long one-way
    position, ``'sell'`` for net-short); the returned legs are those on that
    side, oldest first (FIFO by ``(open_time, leg_id)``) so bracket replication
    and the fail-safe SL fan-out are replay-stable. A net-flat book (offsetting
    legs summing to zero, or no legs at all) returns ``('flat', [])``.

    Replaces the ``'buy' if side == 'long' else 'sell'`` + filter that bracket
    replication, modify, clear, and fail-safe each used to inline.
    """
    net = net_signed_qty(legs)
    if abs(net) <= _QTY_EPS:
        return 'flat', []
    side = 'buy' if net > 0.0 else 'sell'
    survivors = sorted(
        (leg for leg in legs if leg.side == side),
        key=lambda leg: (leg.open_time, leg.leg_id),
    )
    return side, survivors
