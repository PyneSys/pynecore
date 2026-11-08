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
