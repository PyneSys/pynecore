"""
Unit tests for :class:`BrokerPosition`.

Covers entry accounting (open, add), exit accounting (full close, partial
close, FIFO close across multiple entries), side flip in a single fill,
mark-to-market, and liquidation.
"""
import logging
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.models import (
    ExchangeOrder,
    OrderEvent,
    OrderStatus,
    OrderType,
    LegType,
)
from pynecore.lib.strategy import Order, _order_type_close


def _close_order(order_id, size, *, exit_id, comment=None, alert_message=None):
    """Build a Pine close Order as ``strategy.close`` / ``close_all`` would."""
    return Order(order_id, size, order_type=_order_type_close, exit_id=exit_id,
                 comment=comment, alert_message=alert_message)


@pytest.fixture(autouse=True)
def _stub_script():
    """Give :attr:`lib._script.initial_capital` a stable value for equity."""
    prev = lib._script
    lib._script = SimpleNamespace(initial_capital=1_000_000.0)
    try:
        yield
    finally:
        lib._script = prev


def _fill(side: str, qty: float, price: float, *,
          pine_id: str = "Long", leg: LegType = LegType.ENTRY,
          fee: float = 0.0) -> OrderEvent:
    """Build an OrderEvent as a plugin would emit it."""
    order = ExchangeOrder(
        id=f"xchg-{pine_id}-{side}-{qty}",
        symbol="BTCUSDT",
        side=side,
        order_type=OrderType.MARKET,
        qty=qty,
        filled_qty=qty,
        remaining_qty=0.0,
        price=None,
        stop_price=None,
        average_fill_price=price,
        status=OrderStatus.FILLED,
        timestamp=0.0,
        fee=fee,
        fee_currency="USDT",
    )
    return OrderEvent(
        order=order,
        event_type="filled",
        fill_price=price,
        fill_qty=qty,
        timestamp=0.0,
        pine_id=pine_id,
        from_entry=None,
        leg_type=leg,
        fee=fee,
        fee_currency="USDT",
    )


def __test_record_fill_opens_long__():
    """A buy fill opens a long with matching size, ``avg_price`` and one open trade."""
    p = BrokerPosition()
    assert not p.record_fill(_fill("buy", 2.0, 50_000.0, fee=1.0))
    assert p.size == 2.0
    assert p.sign == 1.0
    assert p.avg_price == 50_000.0
    assert len(p.open_trades) == 1
    assert p.open_trades[0].entry_price == 50_000.0
    assert p.open_commission == 1.0


def __test_record_fill_adds_to_long_updates_avg_price__():
    """Adding to a long recomputes ``avg_price`` as the size-weighted average."""
    p = BrokerPosition()
    p.record_fill(_fill("buy", 1.0, 40_000.0))
    p.record_fill(_fill("buy", 3.0, 48_000.0))
    assert p.size == 4.0
    # weighted average: (1*40000 + 3*48000) / 4 = 46000
    assert p.avg_price == pytest.approx(46_000.0)
    assert len(p.open_trades) == 2


def __test_record_fill_full_close_realizes_profit__():
    """Fully closing a long realizes profit into ``netprofit`` and counts a win."""
    p = BrokerPosition()
    p.record_fill(_fill("buy", 1.0, 40_000.0))
    flipped = p.record_fill(_fill("sell", 1.0, 42_000.0, pine_id="TP",
                                   leg=LegType.TAKE_PROFIT))
    assert p.size == 0.0
    assert flipped is True
    assert len(p.open_trades) == 0
    assert len(p.closed_trades) == 1
    # profit = (42000 - 40000) * 1 - 0 commission
    assert p.netprofit == pytest.approx(2_000.0)
    assert p.wintrades == 1


def __test_record_fill_partial_close_splits_trade__():
    """Closing half a long-only trade must split it — remaining stays open."""
    p = BrokerPosition()
    p.record_fill(_fill("buy", 4.0, 50_000.0))
    p.record_fill(_fill("sell", 1.0, 52_000.0, pine_id="TP",
                        leg=LegType.TAKE_PROFIT))
    assert p.size == 3.0
    assert len(p.open_trades) == 1
    assert p.open_trades[0].size == 3.0
    assert len(p.closed_trades) == 1
    assert p.closed_trades[0].size == 1.0
    assert p.netprofit == pytest.approx(2_000.0)


def __test_record_fill_fifo_closes_oldest_first__():
    """With two entries, a partial close consumes the oldest first (FIFO)."""
    p = BrokerPosition()
    p.record_fill(_fill("buy", 1.0, 40_000.0, pine_id="E1"))
    p.record_fill(_fill("buy", 1.0, 50_000.0, pine_id="E2"))
    p.record_fill(_fill("sell", 1.0, 60_000.0, pine_id="TP",
                        leg=LegType.TAKE_PROFIT))
    assert p.size == 1.0
    assert len(p.open_trades) == 1
    assert p.open_trades[0].entry_price == 50_000.0
    assert p.closed_trades[0].entry_price == 40_000.0
    assert p.netprofit == pytest.approx(20_000.0)


def __test_record_fill_fifo_partial_close_splits_exit_fee__():
    """A multi-trade FIFO close must split the exit fee proportionally.

    Two pyramided entries closed by one exit fill that closes the first
    trade fully and the second partially: the exit fee is shared by filled
    quantity, so the fully-closed and partial pieces together book exactly
    the fill fee — the partial piece must NOT be charged the whole fee on
    top of the proportional shares.
    """
    p = BrokerPosition()
    p.record_fill(_fill("buy", 1.0, 40_000.0, pine_id="E1"))
    p.record_fill(_fill("buy", 1.0, 50_000.0, pine_id="E2"))
    # Exit 1.5 of 2.0: closes E1 (1.0) fully + E2 (0.5) partially, fee 3.0.
    p.record_fill(_fill("sell", 1.5, 60_000.0, pine_id="TP",
                        leg=LegType.TAKE_PROFIT, fee=3.0))
    assert p.size == pytest.approx(0.5)
    assert len(p.closed_trades) == 2
    # fee shares: 3.0*(1.0/1.5)=2.0 and 3.0*(0.5/1.5)=1.0 → total 3.0.
    total_exit_fee = sum(t.commission for t in p.closed_trades)
    assert total_exit_fee == pytest.approx(3.0)
    # netprofit = (20000 - 2.0) + (5000 - 1.0)
    assert p.netprofit == pytest.approx(24_997.0)


# Three short pyramid legs of 200 USD contracts on an inverse market, as the
# venue booked them in base units (bybit-inverse cycle 340): the same order
# quantity converts to a different base quantity at each fill's price.
_INVERSE_LEGS = (
    ("S1", 0.0019906400369125304),
    ("S2", 0.0019896564219125737),
    ("S3", 0.0019892553861395007),
)


def _inverse_pyramid(least_tradable_qty=None) -> BrokerPosition:
    """Open the three inverse legs, each with a 0.01 entry fee."""
    p = BrokerPosition()
    p.least_tradable_qty = least_tradable_qty
    for pine_id, qty in _INVERSE_LEGS:
        p.record_fill(_fill("sell", qty, 200.0 / qty, pine_id=pine_id, fee=0.01))
    return p


def _one_contract(price: float) -> float:
    """One whole USD contract in base units at ``price``."""
    return 1.0 / price


def __test_untradable_fifo_remainder_is_spread_over_the_open_trades__(caplog):
    """A FIFO close leaving less than one contract on a trade closes that trade.

    S3's exit fill FIFO-closes S1 instead, and S1 keeps 1.4e-6 BTC, under a
    hundredth of a contract: no exit could ever be resized to it. The
    remainder moves pro rata onto S2 and S3, so S1 closes in full, the trades
    still add up to the venue position and the entry fees stay booked.
    """
    p = _inverse_pyramid(_one_contract)
    held = sum(qty for _, qty in _INVERSE_LEGS)
    fill_qty = _INVERSE_LEGS[2][1]

    with caplog.at_level(logging.INFO):
        p.record_fill(_fill("buy", fill_qty, 100_500.0, pine_id="S3-X",
                            leg=LegType.TAKE_PROFIT, fee=0.02))

    assert [t.entry_id for t in p.open_trades] == ["S2", "S3"]
    assert p.size == pytest.approx(-(held - fill_qty), rel=1e-12)
    assert sum(t.size for t in p.open_trades) == pytest.approx(p.size, rel=1e-12)
    residue = _INVERSE_LEGS[0][1] - fill_qty
    for trade, (_, qty) in zip(p.open_trades, _INVERSE_LEGS[1:]):
        assert abs(trade.size) - qty == pytest.approx(residue / 2.0, rel=1e-3)
    closed = p.closed_trades[-1]
    assert closed.entry_id == "S1"
    assert closed.size == pytest.approx(-fill_qty, rel=1e-12)
    assert closed.commission == pytest.approx(0.01 + 0.02)
    assert p.open_commission == pytest.approx(0.02)
    spread = [rec for rec in caplog.records
              if "spread over the open trades" in rec.getMessage()]
    assert [rec.levelno for rec in spread] == [logging.INFO]


def __test_spread_remainder_keeps_its_cost_basis__():
    """The trades taking a remainder also take its entry cost.

    E1 keeps 0.5 under a least tradable quantity of 1.0 and E2 takes it: the
    open trade's entry price becomes the volume-weighted price of both, so the
    two closes book exactly the P&L of the fills (+50 - 25 = +25).
    """
    p = BrokerPosition()
    p.least_tradable_qty = lambda price: 1.0
    p.record_fill(_fill("buy", 1.5, 100.0, pine_id="E1"))
    p.record_fill(_fill("buy", 1.0, 200.0, pine_id="E2"))
    p.record_fill(_fill("sell", 1.0, 150.0, pine_id="X1", leg=LegType.CLOSE))

    assert [t.entry_id for t in p.open_trades] == ["E2"]
    assert p.open_trades[0].size == pytest.approx(1.5)
    assert p.open_trades[0].entry_price == pytest.approx((0.5 * 100.0 + 1.0 * 200.0) / 1.5)

    p.record_fill(_fill("sell", 1.5, 150.0, pine_id="X2", leg=LegType.CLOSE))
    assert p.size == 0.0
    assert p.netprofit == pytest.approx(25.0)


def __test_tradable_fifo_remainder_stays_on_its_trade__():
    """A remainder of at least one contract keeps its trade open."""
    p = _inverse_pyramid(lambda price: 1e-7)
    fill_qty = _INVERSE_LEGS[2][1]
    p.record_fill(_fill("buy", fill_qty, 100_500.0, pine_id="S3-X",
                        leg=LegType.TAKE_PROFIT))

    assert [t.entry_id for t in p.open_trades] == ["S1", "S2", "S3"]
    assert abs(p.open_trades[0].size) == pytest.approx(
        _INVERSE_LEGS[0][1] - fill_qty, rel=1e-9)
    assert [abs(t.size) for t in p.open_trades[1:]] == [qty for _, qty in _INVERSE_LEGS[1:]]


def __test_fifo_remainder_stays_without_a_quantity_grid__():
    """Without the venue's least tradable quantity no remainder moves."""
    p = _inverse_pyramid()
    fill_qty = _INVERSE_LEGS[2][1]
    p.record_fill(_fill("buy", fill_qty, 100_500.0, pine_id="S3-X",
                        leg=LegType.TAKE_PROFIT))

    assert [t.entry_id for t in p.open_trades] == ["S1", "S2", "S3"]


def __test_untradable_remainder_of_the_last_trade_stays__():
    """The last open trade keeps an untradable remainder: no trade can take it."""
    p = BrokerPosition()
    p.least_tradable_qty = _one_contract
    p.record_fill(_fill("sell", 0.002, 100_000.0, pine_id="S1"))
    p.record_fill(_fill("buy", 0.002 - 2e-6, 100_000.0, pine_id="S1-X",
                        leg=LegType.TAKE_PROFIT))

    assert [t.entry_id for t in p.open_trades] == ["S1"]
    assert p.open_trades[0].size == pytest.approx(-2e-6, rel=1e-6)


def __test_record_fill_side_flip_in_single_event__():
    """Selling more than the open long flips the position to short."""
    p = BrokerPosition()
    p.record_fill(_fill("buy", 1.0, 50_000.0))
    flipped = p.record_fill(_fill("sell", 3.0, 52_000.0, pine_id="Flip",
                                  leg=LegType.ENTRY))
    assert flipped is True
    assert p.size == pytest.approx(-2.0)
    assert p.sign == -1.0
    assert p.avg_price == 52_000.0
    assert len(p.open_trades) == 1
    assert p.closed_trades[0].exit_price == 52_000.0


@pytest.mark.parametrize("leg", [
    LegType.CLOSE, LegType.TAKE_PROFIT, LegType.STOP_LOSS, LegType.TRAILING_STOP,
])
def __test_exit_fill_while_flat_does_not_open__(leg):
    """A late exit-leg fill on a flat book leaves the position flat."""
    p = BrokerPosition()
    assert not p.record_fill(_fill("sell", 1.0, 49_000.0, pine_id="Exit", leg=leg))
    assert p.size == 0.0
    assert p.sign == 0.0
    assert p.open_trades == []
    assert len(p.closed_trades) == 0
    assert p.risk_intraday_filled_orders == 0


@pytest.mark.parametrize("leg", [
    LegType.CLOSE, LegType.TAKE_PROFIT, LegType.STOP_LOSS, LegType.TRAILING_STOP,
])
def __test_exit_fill_on_the_position_side_does_not_add__(leg):
    """An exit-leg fill in the direction of the open position does not grow it."""
    p = BrokerPosition()
    p.record_fill(_fill("sell", 1.0, 50_000.0, pine_id="Short"))
    assert not p.record_fill(_fill("sell", 1.0, 49_000.0, pine_id="Exit", leg=leg))
    assert p.size == -1.0
    assert p.avg_price == 50_000.0
    assert [(t.entry_id, t.size) for t in p.open_trades] == [("Short", -1.0)]


def __test_entry_fill_while_flat_still_opens__():
    """The flat-book guard applies to exit legs only; an entry fill still opens."""
    p = BrokerPosition()
    p.record_fill(_fill("sell", 1.0, 50_000.0, pine_id="Short"))
    assert p.size == -1.0
    assert [(t.entry_id, t.size) for t in p.open_trades] == [("Short", -1.0)]


def __test_update_unrealized_pnl_marks_to_market__():
    """Marking to market sets ``openprofit`` and rolls it into ``equity``."""
    p = BrokerPosition()
    p.record_fill(_fill("buy", 2.0, 40_000.0))
    p.update_unrealized_pnl(45_000.0)
    # (45000 - 40000) * 2 = 10000
    assert p.openprofit == pytest.approx(10_000.0)
    # equity = initial + net + open
    assert p.equity == pytest.approx(1_010_000.0)


@pytest.mark.parametrize(
    ("side", "entries", "marks"),
    [
        ("buy", [(2.0, 100.0, 4.0), (1.0, 120.0, 3.0)], [130.0, 90.0]),
        ("sell", [(2.0, 100.0, 4.0), (1.0, 80.0, 3.0)], [70.0, 110.0]),
    ],
)
def __test_update_unrealized_pnl_updates_each_open_trade__(side, entries, marks):
    """Each long/short trade exposes fee-aware profit and percent at every mark."""
    p = BrokerPosition()
    for index, (qty, entry_price, fee) in enumerate(entries):
        p.record_fill(_fill(side, qty, entry_price, pine_id=f"E{index}", fee=fee))

    for mark in marks:
        p.update_unrealized_pnl(mark)
        expected_openprofit = 0.0
        for trade in p.open_trades:
            gross_profit = (mark - trade.entry_price) * trade.size
            expected_profit = gross_profit - trade.commission
            entry_cost = abs(trade.size) * trade.entry_price + trade.commission
            assert trade.profit == pytest.approx(expected_profit)
            assert trade.profit_percent == pytest.approx(expected_profit / entry_cost * 100.0)
            expected_openprofit += gross_profit
        assert p.openprofit == pytest.approx(expected_openprofit)


def __test_update_unrealized_pnl_zero_entry_cost_clears_percent__():
    """A zero entry cost writes zero instead of retaining the preceding mark's percent."""
    p = BrokerPosition()
    p.record_fill(_fill("buy", 1.0, 100.0))
    p.update_unrealized_pnl(110.0)
    assert p.open_trades[0].profit_percent == pytest.approx(10.0)

    p.open_trades[0].entry_price = 0.0
    p.update_unrealized_pnl(120.0)
    assert p.open_trades[0].profit_percent == 0.0


def __test_partial_close_retains_remaining_entry_fee_share__():
    """A partial close removes only its entry-fee share from the remaining trade."""
    p = BrokerPosition()
    p.record_fill(_fill("buy", 4.0, 100.0, fee=8.0))
    p.record_fill(_fill(
        "sell", 1.0, 110.0, pine_id="Partial",
        leg=LegType.TAKE_PROFIT, fee=3.0,
    ))

    assert len(p.open_trades) == 1
    assert p.open_trades[0].size == pytest.approx(3.0)
    assert p.open_trades[0].commission == pytest.approx(6.0)
    assert p.closed_trades[0].commission == pytest.approx(5.0)

    p.update_unrealized_pnl(120.0)
    assert p.open_trades[0].profit == pytest.approx(54.0)
    assert p.open_trades[0].profit_percent == pytest.approx(54.0 / 306.0 * 100.0)


def __test_record_liquidation_closes_everything__():
    """Liquidation flattens the position and books the loss as ``netprofit``."""
    p = BrokerPosition()
    p.record_fill(_fill("buy", 2.0, 50_000.0))
    liq = _fill("sell", 2.0, 45_000.0, pine_id="LIQ", leg=LegType.CLOSE)
    p.record_liquidation(liq)
    assert p.size == 0.0
    assert p.sign == 0.0
    assert p.openprofit == 0.0
    assert len(p.open_trades) == 0
    # Liquidated at loss → netprofit negative
    assert p.netprofit == pytest.approx(-10_000.0)
    assert p.losstrades == 1


# === Same-bar close netting (live close-dispatch fix) ===


def __test_same_eval_closes_net_into_one_order__():
    """Two same-key ``strategy.close`` calls in one evaluation net into one order.

    The live broker fix replaces the old dict-overwrite (which lost the first
    slice) with summing: two 3-unit closes of "L" become one 6-unit close that
    keeps a single bare exit-order key.
    """
    p = BrokerPosition()
    p.begin_evaluation()
    p._add_order(_close_order("L", -3.0, exit_id="Close entry(s) order L"))
    p._add_order(_close_order("L", -3.0, exit_id="Close entry(s) order L"))
    assert len(p.exit_orders) == 1
    o = next(iter(p.exit_orders.values()))
    assert o.size == -6.0
    assert o.reserved_size == 6.0


def __test_next_eval_close_replaces_not_doubles__():
    """A close re-issued on the next evaluation replaces, never doubles.

    Guards the ``calc_on_every_tick`` case: the same two closes re-firing on the
    next tick must stay at 6, not accumulate to 12, because ``begin_evaluation``
    reset the per-evaluation close scope.
    """
    p = BrokerPosition()
    p.begin_evaluation()
    p._add_order(_close_order("L", -3.0, exit_id="Close entry(s) order L"))
    p._add_order(_close_order("L", -3.0, exit_id="Close entry(s) order L"))
    assert next(iter(p.exit_orders.values())).size == -6.0
    # Next tick: same two closes re-emit → replaced, not accumulated.
    p.begin_evaluation()
    p._add_order(_close_order("L", -3.0, exit_id="Close entry(s) order L"))
    p._add_order(_close_order("L", -3.0, exit_id="Close entry(s) order L"))
    assert len(p.exit_orders) == 1
    assert next(iter(p.exit_orders.values())).size == -6.0


def __test_netted_close_metadata_last_wins__():
    """Netting keeps the last slice's comment/alert — matches prior overwrite."""
    p = BrokerPosition()
    p.begin_evaluation()
    p._add_order(_close_order("L", -3.0, exit_id="Close entry(s) order L",
                              comment="TP1", alert_message="a1"))
    p._add_order(_close_order("L", -3.0, exit_id="Close entry(s) order L",
                              comment="TP2", alert_message="a2"))
    o = next(iter(p.exit_orders.values()))
    assert o.size == -6.0
    assert o.comment == "TP2"
    assert o.alert_message == "a2"


def __test_close_and_close_all_stay_separate__():
    """close(id) and close_all() use different keys and are not netted together."""
    p = BrokerPosition()
    p.begin_evaluation()
    p._add_order(_close_order("L", -3.0, exit_id="Close entry(s) order L"))
    p._add_order(_close_order(None, -10.0, exit_id="Close position order"))
    assert len(p.exit_orders) == 2


def __test_same_eval_exit_bracket_replaces_not_nets__():
    """A re-issued ``strategy.exit`` leg in one evaluation REPLACES, never nets.

    ``strategy.exit`` orders share ``_order_type_close`` with ``strategy.close``
    but use the bare user exit-id (no ``"Close entry(s) order "`` prefix). Only
    market closes net; a same-evaluation re-issue of a sticky bracket leg must
    overwrite so its size stays the leg size and the latest limit/stop levels
    win instead of the first call's stale levels surviving.
    """
    p = BrokerPosition()
    p.begin_evaluation()
    first = Order("L", -6.0, order_type=_order_type_close, exit_id="tp",
                  limit=100.0, stop=90.0)
    second = Order("L", -6.0, order_type=_order_type_close, exit_id="tp",
                   limit=105.0, stop=95.0)
    p._add_order(first)
    p._add_order(second)
    assert len(p.exit_orders) == 1
    o = next(iter(p.exit_orders.values()))
    # Replaced, not summed: size stays the leg size, not -12.0.
    assert o.size == -6.0
    # Latest levels win — the first call's stale 100/90 must not survive.
    assert o.limit == 105.0
    assert o.stop == 95.0


# === Binding book ===

def _bfill(side: str, qty: float, price: float, *, pine_id: str | None,
           order_id: str, leg: LegType = LegType.ENTRY,
           from_entry: str | None = None) -> OrderEvent:
    """Build a fill event with an explicit venue order id and ``from_entry``."""
    order = ExchangeOrder(
        id=order_id, symbol="BTCUSDT", side=side, order_type=OrderType.MARKET,
        qty=qty, filled_qty=qty, remaining_qty=0.0, price=None, stop_price=None,
        average_fill_price=price, status=OrderStatus.FILLED, timestamp=0.0,
        fee=0.0, fee_currency="USDT",
    )
    return OrderEvent(
        order=order, event_type="filled", fill_price=price, fill_qty=qty,
        timestamp=0.0, pine_id=pine_id, from_entry=from_entry, leg_type=leg,
    )


def _book(p: BrokerPosition) -> list[tuple[str | None, float]]:
    """The binding book as ``(entry_id, bound)`` pairs, oldest first."""
    return [(b.entry_id, b.bound) for b in p._entry_book]


def _assert_book_covers_position(p: BrokerPosition) -> None:
    """The bound quantities add up to the position size."""
    assert sum(b.bound for b in p._entry_book) == pytest.approx(abs(p.size), abs=1e-12)


def _pyramid(*legs: tuple[str, float]) -> BrokerPosition:
    """Open one long entry order per ``(entry id, qty)`` at 100, each its own venue order."""
    p = BrokerPosition()
    for i, (entry_id, qty) in enumerate(legs):
        p.record_fill(_bfill("buy", qty, 100.0, pine_id=entry_id, order_id=f"e{i}"))
    return p


def __test_exit_legs_settle_their_own_bindings_while_rows_drain_fifo__():
    """Exit fills of L3 then L2 consume the oldest trade rows, but L1's binding survives.

    The c432 sequence (bybit-spot ``pyramid``): three entries open, ``L3-X``
    fills, then ``L2-X``. The trade rows settle FIFO, so L1's and L2's rows go
    and L3's row is the one left; the bindings settle by origin, so L3's and
    L2's bindings are spent and L1 still holds the open unit.
    """
    p = _pyramid(("L1", 1.0), ("L2", 1.0), ("L3", 1.0))
    assert _book(p) == [("L1", 1.0), ("L2", 1.0), ("L3", 1.0)]

    p.record_fill(_bfill("sell", 1.0, 110.0, pine_id="L3-X", order_id="x3",
                         leg=LegType.TAKE_PROFIT, from_entry="L3"))
    assert [t.entry_id for t in p.open_trades] == ["L2", "L3"]
    assert _book(p) == [("L1", 1.0), ("L2", 1.0)]
    assert p.last_spent_entry_ids == ["L3"]

    p.record_fill(_bfill("sell", 1.0, 110.0, pine_id="L2-X", order_id="x2",
                         leg=LegType.TAKE_PROFIT, from_entry="L2"))
    assert [t.entry_id for t in p.open_trades] == ["L3"]
    assert _book(p) == [("L1", 1.0)]
    assert p.last_spent_entry_ids == ["L2"]
    assert p._bound_qty("L1") == 1.0
    assert p._bound_qty("L3") == 0.0
    _assert_book_covers_position(p)


@pytest.mark.parametrize("leg", [LegType.TAKE_PROFIT, LegType.STOP_LOSS, LegType.TRAILING_STOP])
def __test_partial_exit_leg_reduces_only_its_binding__(leg):
    """A partial exit-leg fill takes its quantity off its own entry's binding only."""
    p = _pyramid(("L1", 1.0), ("L2", 2.0))
    p.record_fill(_bfill("sell", 0.5, 110.0, pine_id="L2-X", order_id="x",
                         leg=leg, from_entry="L2"))
    assert _book(p) == [("L1", 1.0), ("L2", 1.5)]
    assert p.last_spent_entry_ids == []
    _assert_book_covers_position(p)


def __test_strategy_close_id_settles_that_ids_bindings_oldest_first__():
    """A ``strategy.close(id)`` fill spends that id's bindings, oldest first, and no other."""
    p = _pyramid(("L1", 1.0), ("L2", 2.0), ("L1", 1.0))
    p.record_fill(_bfill("sell", 1.5, 105.0, pine_id="L1", order_id="c",
                         leg=LegType.CLOSE), close_entry_id="L1")

    assert [(t.entry_id, t.size) for t in p.open_trades] == [("L2", 1.5), ("L1", 1.0)]
    assert _book(p) == [("L2", 2.0), ("L1", 0.5)]
    assert p.last_spent_entry_ids == ["L1"]
    assert p._bound_qty("L1") == 0.5
    _assert_book_covers_position(p)


def __test_close_leg_without_an_attributed_id_is_unbound__():
    """A close fill the caller does not attribute walks the whole book oldest first.

    ``event.pine_id`` names the closed entry on some venues, but the event
    alone cannot tell a ``strategy.close(id)`` from an engine-made close, so
    only ``close_entry_id`` binds a close.
    """
    p = _pyramid(("L1", 1.0), ("L2", 2.0))
    p.record_fill(_bfill("sell", 1.5, 105.0, pine_id="L2", order_id="c",
                         leg=LegType.CLOSE, from_entry="L2"))
    assert _book(p) == [("L2", 1.5)]
    assert p.last_spent_entry_ids == ["L1"]
    _assert_book_covers_position(p)


def __test_unbound_reduction_walks_the_whole_book_oldest_first__():
    """An opposite ``strategy.order`` fill settles the bindings oldest first, like the rows."""
    p = _pyramid(("L1", 1.0), ("L2", 2.0), ("L3", 1.0))
    p.record_fill(_bfill("sell", 2.5, 105.0, pine_id="O", order_id="o"))
    assert _book(p) == [("L2", 0.5), ("L3", 1.0)]
    assert [(t.entry_id, t.size) for t in p.open_trades] == [("L2", 0.5), ("L3", 1.0)]
    assert p.last_spent_entry_ids == ["L1"]
    _assert_book_covers_position(p)


def __test_external_reduction_is_settled_unbound__():
    """A venue reduction booked as a provisional close settles the bindings oldest first."""
    p = _pyramid(("L1", 1.0), ("L2", 2.0))
    p.record_provisional_close(_bfill("sell", 1.5, 105.0, pine_id=None, order_id="ext",
                                      leg=LegType.CLOSE))
    assert _book(p) == [("L2", 1.5)]
    assert p.last_spent_entry_ids == ["L1"]
    _assert_book_covers_position(p)


def __test_exit_fill_beyond_its_bindings_settles_the_rest_of_the_book__():
    """An exit fill larger than its entry's bindings takes the excess off the oldest others."""
    p = _pyramid(("L1", 1.0), ("L2", 1.0), ("L3", 1.0))
    p.record_fill(_bfill("sell", 1.5, 110.0, pine_id="L3-X", order_id="x",
                         leg=LegType.STOP_LOSS, from_entry="L3"))
    assert _book(p) == [("L1", 0.5), ("L2", 1.0)]
    assert p.last_spent_entry_ids == ["L3"]
    _assert_book_covers_position(p)

    p.record_fill(_bfill("sell", 0.5, 110.0, pine_id="Z-X", order_id="z",
                         leg=LegType.TAKE_PROFIT, from_entry="Z"))
    assert _book(p) == [("L2", 1.0)]
    assert p.last_spent_entry_ids == ["L1"]


def __test_slices_of_one_entry_order_grow_one_binding__():
    """Partial fills of the same entry order make one binding; another order of the id makes a second."""
    p = BrokerPosition()
    p.record_fill(_bfill("buy", 1.0, 100.0, pine_id="L", order_id="e1"))
    p.record_fill(_bfill("buy", 3.0, 104.0, pine_id="L", order_id="e1"))
    assert len(p.open_trades) == 2
    assert len(p._entry_book) == 1
    binding = p._entry_book[0]
    assert (binding.init_size, binding.bound) == (4.0, 4.0)
    assert binding.entry_price == pytest.approx(103.0)

    p.record_fill(_bfill("buy", 1.0, 100.0, pine_id="L", order_id="e2"))
    assert _book(p) == [("L", 4.0), ("L", 1.0)]
    _assert_book_covers_position(p)


def __test_late_slice_grows_a_partly_settled_binding__():
    """A slice arriving after an exit took part of its binding grows what is left of it."""
    p = BrokerPosition()
    p.record_fill(_bfill("buy", 1.0, 100.0, pine_id="L", order_id="e1"))
    p.record_fill(_bfill("sell", 0.5, 110.0, pine_id="L-X", order_id="x",
                         leg=LegType.TAKE_PROFIT, from_entry="L"))
    p.record_fill(_bfill("buy", 1.0, 100.0, pine_id="L", order_id="e1"))
    assert len(p._entry_book) == 1
    assert (p._entry_book[0].init_size, p._entry_book[0].bound) == (2.0, 1.5)
    _assert_book_covers_position(p)


def __test_slice_after_its_binding_is_spent_opens_a_new_one__():
    """A slice of an entry order whose binding is already spent opens a fresh binding."""
    p = _pyramid(("L1", 1.0))
    p.record_fill(_bfill("buy", 1.0, 100.0, pine_id="L2", order_id="e2"))
    p.record_fill(_bfill("sell", 1.0, 110.0, pine_id="L2-X", order_id="x",
                         leg=LegType.TAKE_PROFIT, from_entry="L2"))
    p.record_fill(_bfill("buy", 0.5, 100.0, pine_id="L2", order_id="e2"))
    assert [(b.entry_id, b.init_size, b.bound) for b in p._entry_book] == [
        ("L1", 1.0, 1.0), ("L2", 0.5, 0.5)]
    _assert_book_covers_position(p)


def __test_untradable_binding_remainder_is_spread_over_the_other_bindings__():
    """A binding left below the least tradable quantity is spent and its crumb spread pro rata.

    L1's exit leaves 0.2 on L1, under a least tradable quantity of 0.5: L1 is
    spent, and L2 (1.0) and L3 (2.0) take a third and two thirds of the
    crumb, so the book still adds up to the position.
    """
    p = _pyramid(("L1", 1.0), ("L2", 1.0), ("L3", 2.0))
    p.least_tradable_qty = lambda price: 0.5
    p.record_fill(_bfill("sell", 0.8, 110.0, pine_id="L1-X", order_id="x",
                         leg=LegType.TAKE_PROFIT, from_entry="L1"))

    assert [b.entry_id for b in p._entry_book] == ["L2", "L3"]
    assert p._entry_book[0].bound == pytest.approx(1.0 + 0.2 / 3.0)
    assert p._entry_book[1].bound == pytest.approx(2.0 + 0.4 / 3.0)
    assert [b.init_size for b in p._entry_book] == [1.0, 2.0]
    assert p.last_spent_entry_ids == ["L1"]
    _assert_book_covers_position(p)


def __test_rows_and_bindings_spread_their_crumbs_independently__():
    """L2's exit leaves its crumb on L2's binding but on L1's FIFO row, and each moves on its own.

    The binding crumb lands on L1's binding; the row crumb, left on L1's row
    by the FIFO walk, lands on L2's row.
    """
    p = _pyramid(("L1", 1.0), ("L2", 1.0))
    p.least_tradable_qty = lambda price: 0.5
    p.record_fill(_bfill("sell", 0.8, 110.0, pine_id="L2-X", order_id="x",
                         leg=LegType.TAKE_PROFIT, from_entry="L2"))
    assert _book(p) == [("L1", pytest.approx(1.2))]
    assert p.last_spent_entry_ids == ["L2"]
    assert [(t.entry_id, pytest.approx(t.size)) for t in p.open_trades] == [("L2", 1.2)]
    _assert_book_covers_position(p)


def __test_tradable_binding_remainder_stays__():
    """A binding remainder at or above the least tradable quantity keeps its binding."""
    p = _pyramid(("L1", 1.0), ("L2", 1.0))
    p.least_tradable_qty = lambda price: 0.5
    p.record_fill(_bfill("sell", 0.5, 110.0, pine_id="L1-X", order_id="x",
                         leg=LegType.TAKE_PROFIT, from_entry="L1"))
    assert _book(p) == [("L1", 0.5), ("L2", 1.0)]


def __test_untradable_remainder_of_the_last_binding_stays__():
    """The last binding keeps an untradable remainder: no other binding can take it."""
    p = _pyramid(("L1", 1.0))
    p.least_tradable_qty = lambda price: 0.5
    p.record_fill(_bfill("sell", 0.8, 110.0, pine_id="L1-X", order_id="x",
                         leg=LegType.TAKE_PROFIT, from_entry="L1"))
    assert _book(p) == [("L1", pytest.approx(0.2))]
    assert p.last_spent_entry_ids == []


def __test_going_flat_spends_every_binding__():
    """A fill that flattens the position spends the whole book."""
    p = _pyramid(("L1", 1.0), ("L2", 2.0))
    p.record_fill(_bfill("sell", 3.0, 105.0, pine_id="L2-X", order_id="x",
                         leg=LegType.TAKE_PROFIT, from_entry="L2"))
    assert p.size == 0.0
    assert p._entry_book == []
    assert p.last_spent_entry_ids == ["L2", "L1"]


def __test_exit_overshoot_clamped_to_flat_spends_every_binding__():
    """An exit fill beyond the booked position clamps to flat and leaves no binding."""
    p = _pyramid(("L1", 1.0))
    p.record_fill(_bfill("sell", 1.5, 105.0, pine_id="L1-X", order_id="x",
                         leg=LegType.STOP_LOSS, from_entry="L1"))
    assert p.size == 0.0
    assert p._entry_book == []
    assert p.last_spent_entry_ids == ["L1"]


def __test_reversal_spends_the_old_book_and_binds_the_new_side__():
    """A reversing entry fill spends every old binding and opens one for its flipped quantity."""
    p = _pyramid(("L1", 1.0), ("L2", 1.0))
    p.record_fill(_bfill("sell", 5.0, 105.0, pine_id="S", order_id="s"))
    assert p.size == pytest.approx(-3.0)
    assert [(b.entry_id, b.bound, b.sign) for b in p._entry_book] == [("S", 3.0, -1.0)]
    assert p.last_spent_entry_ids == ["L1", "L2"]

    p.record_fill(_bfill("sell", 1.0, 104.0, pine_id="S", order_id="s"))
    assert [(b.entry_id, b.init_size, b.bound) for b in p._entry_book] == [("S", 4.0, 4.0)]
    _assert_book_covers_position(p)


def __test_record_fill_resets_the_spent_entry_ids__():
    """Each booked fill reports only the bindings it spent itself."""
    p = _pyramid(("L1", 1.0), ("L2", 1.0))
    p.record_fill(_bfill("sell", 1.0, 110.0, pine_id="L1-X", order_id="x",
                         leg=LegType.TAKE_PROFIT, from_entry="L1"))
    assert p.last_spent_entry_ids == ["L1"]
    p.record_fill(_bfill("buy", 1.0, 100.0, pine_id="L3", order_id="e3"))
    assert p.last_spent_entry_ids == []


def __test_dust_flatten_spends_the_remaining_bindings__():
    """Booking the dust residual flat adds the still-open bindings to the spent ids."""
    p = _pyramid(("L1", 1.0), ("L2", 1.0))
    event = _bfill("sell", 1.0, 110.0, pine_id="L1-X", order_id="x",
                   leg=LegType.TAKE_PROFIT, from_entry="L1")
    p.record_fill(event)
    p.flatten_nontradable_dust(110.0, event)
    assert p._entry_book == []
    assert p.last_spent_entry_ids == ["L1", "L2"]


def __test_liquidation_spends_every_binding__():
    """An exchange liquidation leaves no binding."""
    p = _pyramid(("L1", 1.0), ("L2", 1.0))
    p.record_liquidation(_bfill("sell", 2.0, 90.0, pine_id=None, order_id="liq"))
    assert p._entry_book == []
    assert p.last_spent_entry_ids == ["L1", "L2"]


def __test_adopted_parent_gets_its_binding__():
    """Re-seeding an adopted parent's trade also seeds its binding, once."""
    p = BrokerPosition()
    p.size = -2.0
    p.sign = -1.0
    p.reconstruct_parent_trade(entry_id="P", size=-2.0, entry_price=100.0)
    p.reconstruct_parent_trade(entry_id="P", size=-2.0, entry_price=100.0)
    assert [(b.entry_id, b.bound, b.sign) for b in p._entry_book] == [("P", 2.0, -1.0)]
    _assert_book_covers_position(p)


def __test_a_bound_leg_naming_no_binding_warns_and_settles_the_oldest__(caplog):
    """A take-profit whose entry holds no binding is reported, not passed over in silence.

    The fill still reduces the venue position, so it settles the oldest
    bindings; the warning names the fill's ids. A leg whose entry merely holds
    less than the fill does not warn.
    """
    p = _pyramid(("L1", 1.0), ("L2", 1.0))
    with caplog.at_level(logging.WARNING):
        p.record_fill(_bfill("sell", 0.5, 110.0, pine_id="L2-X", order_id="x2",
                             leg=LegType.TAKE_PROFIT, from_entry="L2"))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    with caplog.at_level(logging.WARNING):
        p.record_fill(_bfill("sell", 1.0, 110.0, pine_id="L9-X", order_id="x9",
                             leg=LegType.TAKE_PROFIT, from_entry="L9"))
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "'L9-X'" in warnings[0] and "'L9'" in warnings[0]
    assert _book(p) == [("L2", 0.5)]
    _assert_book_covers_position(p)


def __test_a_book_seeded_without_fills_is_not_faithful_until_flat__(caplog):
    """An adopted parent or a rebuild from the rows leaves the book FIFO-shaped.

    Only a flat book is faithful again: its bindings then come from fills.
    """
    p = BrokerPosition()
    assert p.book_faithful
    p.size = 2.0
    p.sign = 1.0
    p.reconstruct_parent_trade(entry_id="L1", size=2.0, entry_price=100.0)
    assert not p.book_faithful
    p.record_fill(_bfill("sell", 2.0, 110.0, pine_id="", order_id="c1",
                         leg=LegType.CLOSE))
    assert p.size == 0.0 and p.book_faithful

    rebuilt = _pyramid(("L1", 1.0))
    rebuilt._entry_book.clear()
    with caplog.at_level(logging.WARNING):
        rebuilt.record_fill(_bfill("buy", 1.0, 100.0, pine_id="L2", order_id="e9"))
    assert any("invariant breached" in r.getMessage() for r in caplog.records)
    assert not rebuilt.book_faithful
    assert _book(rebuilt) == [("L1", 1.0), ("L2", 1.0)]


def __test_a_book_with_a_moved_sub_lot_remainder_survives_a_restart_round_trip__():
    """A binding that took another's untradable remainder binds more than it opened.

    The stored book must still read back: rejecting it would drop the
    attribution at a routine restart and adopt the position FIFO.
    """
    p = _pyramid(("L1", 1.0), ("L2", 2.0))
    p.least_tradable_qty = lambda price: 0.5
    p.record_fill(_bfill("sell", 0.8, 110.0, pine_id="", order_id="c1",
                         leg=LegType.CLOSE))
    assert _book(p) == [("L2", pytest.approx(2.2))]
    assert p._entry_book[0].init_size == 2.0
    snapshot = BrokerPosition.parse_entry_book_snapshot(
        {**p.entry_book_snapshot(), 'written_ms': 0})
    assert snapshot is not None
    restored = BrokerPosition()
    restored.restore_entry_book(snapshot)
    assert _book(restored) == [("L2", pytest.approx(2.2))]


def __test_the_fill_cursor_counts_only_the_booked_slices__():
    """An order snapshot already showing a later slice does not advance the cursor past the booked ones.

    A restart compares the cursor with the journal's filled quantity; a cursor
    credited with the unbooked slice would accept a book that missed it.
    """
    p = BrokerPosition()
    event = _bfill("buy", 1.0, 100.0, pine_id="L1", order_id="e1")
    event.order.client_order_id = "coid-e1"
    event.order.qty = 2.0
    event.order.filled_qty = 2.0  # the venue already filled the second slice
    p.record_fill(event)
    assert p.fill_cursor(["coid-e1", "e1"]) == 1.0
    # A slice reported under the venue id only still adds to the order's cursor.
    second = _bfill("buy", 1.0, 100.0, pine_id="L1", order_id="e1")
    p.record_fill(second)
    assert p.fill_cursor(["coid-e1", "e1"]) == 2.0
