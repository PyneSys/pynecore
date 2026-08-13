"""
Regression: ``strategy.order`` must keep its raw quantity in broker (live) mode.

The Pine-side lot floor is a backtest quantization: TradingView snaps a sub-lot
size to zero and silently drops the order. In live mode the venue owns the
quantity grid -- the plugin quantizes onto the exchange step and reports an
explicit below-minimum skip to the operator. ``strategy.entry`` already limits
the floor to :class:`SimPosition`; ``strategy.order`` applied it unconditionally,
so a positive sub-lot live signal disappeared before an intent was ever built.
"""
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.lib import strategy
from pynecore.core.broker.models import ExchangeOrder, LegType, OrderEvent, OrderStatus, OrderType
from pynecore.core.broker.position import BrokerPosition


def _buy_fill(qty: float, price: float, *, pine_id: str) -> OrderEvent:
    order = ExchangeOrder(
        id=f"xchg-{pine_id}",
        symbol="BTCUSD",
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
        fee_currency="BTC",
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
        fee_currency="BTC",
    )


@pytest.fixture
def broker_position():
    """A flat :class:`BrokerPosition` wired to ``lib._script`` as the runner does."""
    prev_script = lib._script
    prev_sem = lib._lib_semaphore
    prev_supp = lib._strategy_suppressed
    prev_bar = getattr(lib, "bar_index", 0)
    prev_close = lib.close
    prev_pricescale = lib.syminfo.pricescale
    prev_minmove = lib.syminfo.minmove
    prev_mintick = lib.syminfo.mintick
    prev_rfactor = getattr(lib.syminfo, "_size_round_factor", None)

    pos = BrokerPosition()
    lib._script = SimpleNamespace(initial_capital=1_000_000.0, position=pos)
    lib._lib_semaphore = False
    lib._strategy_suppressed = False
    lib.bar_index = 5
    lib.close = 50_000.0
    lib.syminfo.pricescale = 10
    lib.syminfo.minmove = 5
    lib.syminfo.mintick = 0.5
    # The lot grid the sim path would floor onto (mincontract 1e-4)
    lib.syminfo._size_round_factor = 10_000.0

    try:
        yield pos
    finally:
        lib._script = prev_script
        lib._lib_semaphore = prev_sem
        lib._strategy_suppressed = prev_supp
        lib.bar_index = prev_bar
        lib.close = prev_close
        lib.syminfo.pricescale = prev_pricescale
        lib.syminfo.minmove = prev_minmove
        lib.syminfo.mintick = prev_mintick
        if prev_rfactor is None:
            del lib.syminfo._size_round_factor
        else:
            lib.syminfo._size_round_factor = prev_rfactor


def __test_sub_lot_order_reaches_the_broker_unrounded__(broker_position):
    """A below-lot quantity is enqueued as requested, not floored away."""
    strategy.order("Tiny", strategy.long, qty=0.000005)

    order = broker_position.entry_orders.get("Tiny")
    assert order is not None, "the sub-lot order was dropped before dispatch"
    assert order.size == 0.000005


def __test_sub_lot_partial_close_reaches_the_broker_unrounded__(broker_position):
    """A broker close preserves venue-converted exposure below the Pine lot grid."""
    broker_position.record_fill(_buy_fill(0.00015868, 63_019.4, pine_id="Inverse"))

    strategy.close("Inverse", qty_percent=50, immediately=True)

    order = broker_position.exit_orders.get(("Close entry(s) order Inverse", "Inverse"))
    assert order is not None, "the inverse partial close was dropped before dispatch"
    assert order.size == pytest.approx(-0.00007934)


def __test_sub_lot_exit_reaches_the_broker_unrounded__(broker_position):
    """A protective broker exit preserves venue-converted exposure below the Pine lot grid."""
    broker_position.record_fill(_buy_fill(0.00015868, 63_019.4, pine_id="Inverse"))

    strategy.exit("Protection", from_entry="Inverse", limit=64_000.0, stop=62_000.0)

    order = broker_position.exit_orders.get(("Protection", "Inverse"))
    assert order is not None, "the inverse protection was dropped before dispatch"
    assert order.size == pytest.approx(-0.00015868)


def __test_unsizable_order_is_still_dropped__(broker_position):
    """An infinite quantity is unsizable in broker mode as well."""
    strategy.order("Inf", strategy.long, qty=float('inf'))

    assert broker_position.entry_orders == {}
