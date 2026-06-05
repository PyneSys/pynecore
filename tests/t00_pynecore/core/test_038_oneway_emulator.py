"""
Unit tests for the one-way position emulator pure functions.

Covers leg aggregation (single leg, same-direction weighted average,
opposite-leg virtual-FIFO netting, net-flat, unrealized P&L sum), FIFO
close-leg selection (ordering, partial split, exact cover, shortfall), and the
reversal decomposition (pure add, full reversal, partial reduce, exact
flatten).
"""
from pynecore.core.broker.emulator import (
    LegClose,
    aggregate_positions,
    legs_on_position_side,
    net_signed_qty,
    net_survivor_legs,
    plan_leg_close_volumes,
    plan_reversal,
    select_legs_for_close,
)
from pynecore.core.broker.models import PositionLeg


def _leg(leg_id: str, side: str, qty: float, price: float,
         open_time: float, upnl: float = 0.0) -> PositionLeg:
    return PositionLeg(
        leg_id=leg_id, symbol="EURUSD", side=side, qty=qty,
        entry_price=price, open_time=open_time, unrealized_pnl=upnl,
    )


# --- net_signed_qty ------------------------------------------------------

def __test_net_signed_qty_mixed__():
    """Net signed quantity of a mixed buy/sell book is the directional sum."""
    legs = [_leg("1", "buy", 2.0, 1.10, 0.0),
            _leg("2", "sell", 0.5, 1.11, 1.0)]
    assert net_signed_qty(legs) == 1.5


# --- aggregate_positions -------------------------------------------------

def __test_aggregate_empty_is_none__():
    """Aggregating an empty leg list yields ``None`` (no position)."""
    assert aggregate_positions("EURUSD", []) is None


def __test_aggregate_single_long_leg__():
    """A single buy leg aggregates to a long position with its size, price and P&L."""
    pos = aggregate_positions("EURUSD", [_leg("1", "buy", 1.0, 1.2345, 0.0, upnl=3.0)])
    assert pos is not None
    assert pos.side == "long"
    assert pos.size == 1.0
    assert pos.entry_price == 1.2345
    assert pos.unrealized_pnl == 3.0


def __test_aggregate_single_short_leg__():
    """A single sell leg aggregates to a short position with its size and price."""
    pos = aggregate_positions("EURUSD", [_leg("1", "sell", 2.0, 1.5, 0.0)])
    assert pos is not None
    assert pos.side == "short"
    assert pos.size == 2.0
    assert pos.entry_price == 1.5


def __test_aggregate_same_direction_weighted_average__():
    """Same-direction legs aggregate to a size-weighted average entry price."""
    legs = [_leg("1", "buy", 1.0, 1.00, 0.0),
            _leg("2", "buy", 3.0, 2.00, 1.0)]
    pos = aggregate_positions("EURUSD", legs)
    assert pos is not None
    assert pos.side == "long"
    assert pos.size == 4.0
    # (1*1.00 + 3*2.00) / 4 = 1.75
    assert abs(pos.entry_price - 1.75) < 1e-12


def __test_aggregate_opposite_legs_virtual_fifo_net_long__():
    """An opposing short virtually FIFO-offsets the oldest long, leaving the survivor's price."""
    # Oldest long 1.0 @1.00, newer long 2.0 @2.00, a short 1.0 offsets the
    # OLDEST long (FIFO) → surviving 2.0 long @2.00.
    legs = [_leg("1", "buy", 1.0, 1.00, 0.0),
            _leg("2", "buy", 2.0, 2.00, 1.0),
            _leg("3", "sell", 1.0, 1.50, 2.0)]
    pos = aggregate_positions("EURUSD", legs)
    assert pos is not None
    assert pos.side == "long"
    assert pos.size == 2.0
    assert abs(pos.entry_price - 2.00) < 1e-12


def __test_aggregate_net_flat_with_open_legs__():
    """Offsetting legs aggregate to flat (zero size/price) but still surface residual P&L."""
    legs = [_leg("1", "buy", 1.0, 1.00, 0.0, upnl=2.0),
            _leg("2", "sell", 1.0, 1.10, 1.0, upnl=-1.0)]
    pos = aggregate_positions("EURUSD", legs)
    assert pos is not None
    assert pos.side == "flat"
    assert pos.size == 0.0
    assert pos.entry_price == 0.0
    # Residual gross P&L surfaced even though net is flat.
    assert pos.unrealized_pnl == 1.0


def __test_aggregate_unrealized_pnl_is_sum__():
    """Aggregated unrealized P&L is the sum across all legs."""
    legs = [_leg("1", "buy", 1.0, 1.00, 0.0, upnl=5.0),
            _leg("2", "buy", 1.0, 1.20, 1.0, upnl=-2.0)]
    pos = aggregate_positions("EURUSD", legs)
    assert pos is not None
    assert pos.unrealized_pnl == 3.0


# --- select_legs_for_close -----------------------------------------------

def __test_select_close_fifo_order__():
    """Close selection consumes the oldest leg first, regardless of input order."""
    legs = [_leg("new", "buy", 5.0, 1.0, 10.0),
            _leg("old", "buy", 5.0, 1.0, 1.0)]
    closes, shortfall = select_legs_for_close(3.0, legs, "buy")
    assert shortfall == 0.0
    # Oldest first regardless of input order.
    assert closes == (LegClose(leg_id="old", qty=3.0),)


def __test_select_close_partial_split_across_legs__():
    """A close larger than the oldest leg spills the remainder onto the next leg."""
    legs = [_leg("a", "buy", 2.0, 1.0, 0.0),
            _leg("b", "buy", 2.0, 1.0, 1.0)]
    closes, shortfall = select_legs_for_close(3.0, legs, "buy")
    assert shortfall == 0.0
    assert closes == (LegClose(leg_id="a", qty=2.0), LegClose(leg_id="b", qty=1.0))


def __test_select_close_exact_cover__():
    """A close matching the leg size fully consumes it with no shortfall."""
    legs = [_leg("a", "buy", 2.0, 1.0, 0.0)]
    closes, shortfall = select_legs_for_close(2.0, legs, "buy")
    assert shortfall == 0.0
    assert closes == (LegClose(leg_id="a", qty=2.0),)


def __test_select_close_shortfall_when_underfunded__():
    """A close exceeding available legs closes all of them and reports the shortfall."""
    legs = [_leg("a", "buy", 1.0, 1.0, 0.0)]
    closes, shortfall = select_legs_for_close(3.0, legs, "buy")
    assert closes == (LegClose(leg_id="a", qty=1.0),)
    assert abs(shortfall - 2.0) < 1e-12


def __test_select_close_ignores_other_side__():
    """Close selection only considers legs on the requested side, ignoring the other."""
    legs = [_leg("long", "buy", 5.0, 1.0, 0.0),
            _leg("short", "sell", 5.0, 1.0, 1.0)]
    closes, shortfall = select_legs_for_close(2.0, legs, "sell")
    assert closes == (LegClose(leg_id="short", qty=2.0),)
    assert shortfall == 0.0


# --- plan_reversal -------------------------------------------------------

def __test_plan_reversal_pure_add__():
    """A same-side order adds to the position with no closes."""
    legs = [_leg("1", "buy", 1.0, 1.0, 0.0)]
    plan = plan_reversal("buy", 2.0, legs)
    assert plan.closes == ()
    assert plan.open_qty == 2.0
    assert plan.open_side == "buy"


def __test_plan_reversal_full_flip__():
    """An opposite order larger than the position closes it and opens the residual short."""
    # Long 2.0 open; Pine reverses to short 1.0 → combined sell qty 3.0.
    legs = [_leg("a", "buy", 2.0, 1.0, 0.0)]
    plan = plan_reversal("sell", 3.0, legs)
    assert plan.closes == (LegClose(leg_id="a", qty=2.0),)
    assert plan.open_qty == 1.0
    assert plan.open_side == "sell"


def __test_plan_reversal_partial_reduce_via_order__():
    """An opposite order smaller than the position only reduces it, opening nothing."""
    # Long 5.0 open; an opposite order of 2.0 only reduces, never flips.
    legs = [_leg("a", "buy", 5.0, 1.0, 0.0)]
    plan = plan_reversal("sell", 2.0, legs)
    assert plan.closes == (LegClose(leg_id="a", qty=2.0),)
    assert plan.open_qty == 0.0


def __test_plan_reversal_exact_flatten__():
    """An opposite order equal to the position flattens it with nothing opened."""
    legs = [_leg("a", "buy", 2.0, 1.0, 0.0)]
    plan = plan_reversal("sell", 2.0, legs)
    assert plan.closes == (LegClose(leg_id="a", qty=2.0),)
    assert plan.open_qty == 0.0


def __test_plan_reversal_flip_across_multiple_legs_fifo__():
    """A flip closes multiple legs in FIFO order before opening the residual."""
    legs = [_leg("old", "buy", 1.0, 1.0, 1.0),
            _leg("new", "buy", 1.0, 1.0, 2.0)]
    plan = plan_reversal("sell", 3.0, legs)
    assert plan.closes == (LegClose(leg_id="old", qty=1.0),
                           LegClose(leg_id="new", qty=1.0))
    assert plan.open_qty == 1.0


# --- plan_leg_close_volumes ----------------------------------------------

def __test_plan_leg_close_volumes_basic_fifo__():
    """Whole-unit leg closes map to per-leg integer volumes in FIFO order."""
    closes = (LegClose("a", 2.0), LegClose("b", 1.0))
    assert plan_leg_close_volumes(closes, lambda u: int(u)) == [("a", 2), ("b", 1)]


def __test_plan_leg_close_volumes_carries_sub_grid_remainder__():
    """Sub-grid slices accumulate so the owed volume lands on the leg that crosses the grid."""
    # Each 0.6-unit slice rounds to 0 on its own, but the running total reaches
    # the grid on the second leg, so the owed volume lands there (not dropped).
    closes = (LegClose("a", 0.6), LegClose("b", 0.6))
    assert plan_leg_close_volumes(closes, lambda u: int(u)) == [("b", 1)]


def __test_plan_leg_close_volumes_empty_below_grid__():
    """A single below-grid slice rounds to zero volume, yielding no leg closes."""
    closes = (LegClose("a", 0.3),)
    assert plan_leg_close_volumes(closes, lambda u: int(u)) == []


# --- legs_on_position_side -----------------------------------------------

def __test_legs_on_position_side_net_long_fifo__():
    """A net-long book returns the buy side with its buy legs in FIFO order."""
    legs = [_leg("1", "buy", 2.0, 1.10, 0.0),
            _leg("2", "sell", 0.5, 1.11, 1.0),
            _leg("3", "buy", 1.0, 1.12, 2.0)]
    side, on_side = legs_on_position_side(legs)
    assert side == "buy"
    assert [leg.leg_id for leg in on_side] == ["1", "3"]


def __test_legs_on_position_side_net_flat_is_empty__():
    """A net-flat book reports the flat side with no on-side legs."""
    legs = [_leg("1", "buy", 1.0, 1.10, 0.0),
            _leg("2", "sell", 1.0, 1.11, 1.0)]
    assert legs_on_position_side(legs) == ("flat", [])


def __test_legs_on_position_side_tie_break_by_leg_id__():
    """Legs with equal open time are ordered deterministically by leg id."""
    # Equal open_time -> deterministic order by leg_id (replay-stable).
    legs = [_leg("b", "buy", 1.0, 1.10, 5.0),
            _leg("a", "buy", 1.0, 1.11, 5.0)]
    _side, on_side = legs_on_position_side(legs)
    assert [leg.leg_id for leg in on_side] == ["a", "b"]


# --- net_survivor_legs ---------------------------------------------------

def __test_net_survivor_legs_drops_fully_consumed_majority_leg__():
    """Survivor legs drop a majority leg fully consumed by an opposing leg, unlike gross."""
    # Mixed book net long 2: the opposing 1.0 sell virtually FIFO-closes the
    # oldest buy leg, so only the two newest survive. legs_on_position_side
    # would return all three gross legs (the F4 over-protection).
    legs = [_leg("1", "buy", 1.0, 1.10, 0.0),
            _leg("2", "buy", 1.0, 1.11, 1.0),
            _leg("3", "buy", 1.0, 1.12, 2.0),
            _leg("4", "sell", 1.0, 1.13, 3.0)]
    side, survivors = net_survivor_legs(legs)
    assert side == "buy"
    assert [leg.leg_id for leg in survivors] == ["2", "3"]
    # Contrast: the gross helper protects the consumed leg too.
    assert [leg.leg_id for leg in legs_on_position_side(legs)[1]] == ["1", "2", "3"]


def __test_net_survivor_legs_clean_book_matches_position_side__():
    """With no opposing legs, survivor legs equal ``legs_on_position_side``."""
    # No opposing legs -> identical to legs_on_position_side (the common case).
    legs = [_leg("1", "buy", 2.0, 1.10, 0.0),
            _leg("2", "buy", 1.0, 1.12, 1.0)]
    assert net_survivor_legs(legs) == legs_on_position_side(legs)


def __test_net_survivor_legs_partial_boundary_leg_survives__():
    """A partially consumed boundary leg keeps its remainder and survives the netting."""
    # The opposing 0.5 consumes only part of the oldest buy (2.0) -> it keeps
    # 1.5 and survives; the bracket protects the net exposure including it.
    legs = [_leg("1", "buy", 2.0, 1.10, 0.0),
            _leg("2", "sell", 0.5, 1.11, 1.0),
            _leg("3", "buy", 1.0, 1.12, 2.0)]
    side, survivors = net_survivor_legs(legs)
    assert side == "buy"
    assert [leg.leg_id for leg in survivors] == ["1", "3"]


def __test_net_survivor_legs_net_flat_is_empty__():
    """A net-flat book has no survivor legs and reports the flat side."""
    legs = [_leg("1", "buy", 1.0, 1.10, 0.0),
            _leg("2", "sell", 1.0, 1.11, 1.0)]
    assert net_survivor_legs(legs) == ("flat", [])
