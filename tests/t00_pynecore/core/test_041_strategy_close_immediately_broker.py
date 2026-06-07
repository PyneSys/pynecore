"""
Regression: ``strategy.close(immediately=True)`` / ``close_all(immediately=True)``
must not crash in broker (live) mode.

``immediately`` requests a same-tick fill, which is a backtest-only concept
implemented by :meth:`SimPosition.fill_order`. :class:`BrokerPosition` has no
``fill_order`` -- in live mode ``_add_order`` enqueues the order and the sync
engine forwards it to the exchange, which fills asynchronously. Before the
guard, ``immediately=True`` reached ``position.fill_order(...)`` on a
:class:`BrokerPosition` and raised ``AttributeError``, halting the live bot.
The fix runs the explicit sim fill only when ``isinstance(position, SimPosition)``;
in broker mode the close order is left enqueued for the sync engine.
"""
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.lib import strategy
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.models import (
    ExchangeOrder,
    OrderEvent,
    OrderStatus,
    OrderType,
    LegType,
)


def _buy_fill(qty: float, price: float, *, pine_id: str = "Long") -> OrderEvent:
    """Build a filled buy :class:`OrderEvent` as a broker plugin would emit it."""
    order = ExchangeOrder(
        id=f"xchg-{pine_id}-buy-{qty}",
        symbol="BTCUSDT",
        side="buy",
        order_type=OrderType.MARKET,
        qty=qty,
        filled_qty=qty,
        remaining_qty=0.0,
        price=None,
        stop_price=None,
        average_fill_price=price,
        status=OrderStatus.FILLED,
        timestamp=0.0,
        fee=0.0,
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
        leg_type=LegType.ENTRY,
        fee=0.0,
        fee_currency="USDT",
    )


@pytest.fixture
def broker_position():
    """A long :class:`BrokerPosition` wired to ``lib._script`` as the runner does."""
    prev_script = lib._script
    prev_sem = lib._lib_semaphore
    prev_supp = lib._strategy_suppressed
    prev_bar = getattr(lib, "bar_index", 0)

    pos = BrokerPosition()
    lib._script = SimpleNamespace(initial_capital=1_000_000.0, position=pos)
    lib._lib_semaphore = False
    lib._strategy_suppressed = False
    lib.bar_index = 5

    pos.record_fill(_buy_fill(2.0, 50_000.0))
    assert pos.size == 2.0, "fixture failed to open the long"

    try:
        yield pos
    finally:
        lib._script = prev_script
        lib._lib_semaphore = prev_sem
        lib._strategy_suppressed = prev_supp
        lib.bar_index = prev_bar


def __test_broker_position_has_no_fill_order__():
    """Root cause: the sim-only fill API is absent on :class:`BrokerPosition`."""
    assert not hasattr(BrokerPosition(), "fill_order")


def __test_close_immediately_enqueues_without_same_bar_fill__(broker_position):
    """``strategy.close(immediately=True)`` enqueues the exit; no AttributeError."""
    strategy.close("Long", immediately=True)

    assert ("Close entry(s) order Long", "Long") in broker_position.exit_orders
    # Not filled on the spot -- the sync engine forwards it to the exchange.
    assert broker_position.size == 2.0


def __test_close_all_immediately_enqueues_without_same_bar_fill__(broker_position):
    """``strategy.close_all(immediately=True)`` enqueues the full-close order."""
    strategy.close_all(immediately=True)

    assert ("Close position order", None) in broker_position.exit_orders
    assert broker_position.size == 2.0
