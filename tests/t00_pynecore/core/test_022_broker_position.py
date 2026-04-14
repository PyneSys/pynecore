"""
Unit tests for :class:`BrokerPosition`.

Covers entry accounting (open, add), exit accounting (full close, partial
close, FIFO close across multiple entries), side flip in a single fill,
mark-to-market, and liquidation.
"""
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
    p = BrokerPosition()
    assert not p.record_fill(_fill("buy", 2.0, 50_000.0, fee=1.0))
    assert p.size == 2.0
    assert p.sign == 1.0
    assert p.avg_price == 50_000.0
    assert len(p.open_trades) == 1
    assert p.open_trades[0].entry_price == 50_000.0
    assert p.open_commission == 1.0


def __test_record_fill_adds_to_long_updates_avg_price__():
    p = BrokerPosition()
    p.record_fill(_fill("buy", 1.0, 40_000.0))
    p.record_fill(_fill("buy", 3.0, 48_000.0))
    assert p.size == 4.0
    # weighted average: (1*40000 + 3*48000) / 4 = 46000
    assert p.avg_price == pytest.approx(46_000.0)
    assert len(p.open_trades) == 2


def __test_record_fill_full_close_realizes_profit__():
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


def __test_update_unrealized_pnl_marks_to_market__():
    p = BrokerPosition()
    p.record_fill(_fill("buy", 2.0, 40_000.0))
    p.update_unrealized_pnl(45_000.0)
    # (45000 - 40000) * 2 = 10000
    assert p.openprofit == pytest.approx(10_000.0)
    # equity = initial + net + open
    assert p.equity == pytest.approx(1_010_000.0)


def __test_record_liquidation_closes_everything__():
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
